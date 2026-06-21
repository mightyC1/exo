# Copyright © 2026
#
# minimax_m3.py — MLX implementation of MiniMax-M3 (`minimax_m3_vl`) TEXT BACKBONE
# with MiniMax Sparse Attention (MSA), arXiv:2606.13392.
#
# v4: rewritten against the REAL released config.json
# (MiniMaxAI/MiniMax-M3 @ 3a41b31). Verified facts from that config:
#   hidden 6144 · 60 layers · 64 q-heads / 4 kv-heads (G=16) · head_dim 128
#   rope_theta 5e6, rotary_dim 64 (partial 0.5) · vocab 200064 · ctx 1,048,576
#   HYBRID stack: layers 0-2 = full attention + dense MLP (12288);
#                 layers 3-59 = MSA + MoE(128 experts, top-4, sigmoid+bias,
#                 routed_scaling_factor 2.0) + 1 shared expert (3072)
#   activation "swigluoai" (alpha 1.702, limit 7.0 — gpt-oss style)
#   use_gemma_norm: true  → ALL RMSNorms are (1+w)·x̂
#   use_qk_norm: true, qk_norm_type "per_head"
#   MSA: index_dim 128, num_index_heads 4 (== n_kv_heads → one per GQA group ✓),
#        topk_blocks 16, block_size 128, score_type "max" ✓, local_block 1,
#        init_block 0, sparse layers = layers 3-59 (sparse_attention_freq)
#   MTP present (num_mtp_modules 7) → dropped in sanitize.
#
# In-tree precedents reused:
#   minimax.py (M2.7 scaffolding) · deepseek_v32.py (indexer-cache idiom:
#   CacheList(main, indexer), put_along_axis mask / take_along_axis gather,
#   mx.depends) · deepseek_v3.py (shared expert + routed_scaling_factor) ·
#   gpt_oss.py (swigluoai) · gemma.py ((1+w) norm) · mimo_v2_flash.py
#   (heterogeneous per-layer make_cache, MTP drop).
#
# ── REMAINING UNKNOWNS (checkpoint-level, not config-level) ──────────────────
#   ✗ exact weight names: indexer projections, shared-expert path, MTP prefix,
#     whether LM weights sit under "language_model." (VL checkpoint) — handled
#     by the remap hooks in sanitize(); CONFIRM against model.safetensors.index.json.
#   ✗ per-head qk-norm weight shape: assumed (head_dim,) shared across heads
#     (Qwen3/Gemma2 convention). If it is (n_heads*head_dim,) per-layer joint —
#     switch use_per_head_qk_norm=False (falls back to M2.7-style joint norm).
#   ✗ norm_topk_prob not present in config — M2-family normalized top-k scores,
#     we keep normalize=True (flag provided).
#   ✗ the vision tower/projector — out of scope (mlx-vlm); vision weights dropped.

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

try:  # in-tree (file dropped into mlx_lm/models/)
    from .base import BaseModelArgs, scaled_dot_product_attention
    from .cache import CacheList, KVCache
    from .switch_layers import SwitchGLU
except ImportError:  # standalone next to an installed mlx-lm
    from mlx_lm.models.base import BaseModelArgs, scaled_dot_product_attention
    from mlx_lm.models.cache import CacheList, KVCache
    from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "minimax_m3_vl"
    hidden_size: int = 6144
    num_hidden_layers: int = 60
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 200064
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5e6
    rotary_dim: int = 64
    tie_word_embeddings: bool = False
    use_gemma_norm: bool = True
    use_qk_norm: bool = True
    qk_norm_type: str = "per_head"           # "per_head" | "per_layer" (M2.7 joint)
    attention_output_gate: bool = False       # config says false; gate unsupported here
    # FFN / MoE
    hidden_act: str = "swigluoai"
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    intermediate_size: int = 3072             # MoE expert width
    dense_intermediate_size: int = 12288      # dense layers (0-2)
    shared_intermediate_size: int = 3072      # shared expert width
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    scoring_func: str = "sigmoid"
    use_routing_bias: bool = True
    routed_scaling_factor: float = 2.0
    norm_topk_prob: bool = True               # not in config; M2-family default
    moe_layer_freq: List[int] = field(default_factory=lambda: [0] * 3 + [1] * 57)
    # MSA (flattened from config["text_config"]["sparse_attention_config"])
    use_sparse_attention: bool = True
    sparse_index_dim: int = 128
    sparse_num_index_heads: int = 4            # must equal num_key_value_heads
    sparse_topk_blocks: int = 16
    sparse_block_size: int = 128
    sparse_score_type: str = "max"
    sparse_init_block: int = 0                 # forced initial blocks (count)
    sparse_local_block: int = 1                # forced local blocks (count)
    sparse_attention_freq: List[int] = field(default_factory=lambda: [0] * 3 + [1] * 57)
    msa_prefill_chunk: int = 512  # query-chunk for block-sparse prefill (perf only)

    @classmethod
    def from_dict(cls, params):
        # The released config nests everything under text_config and
        # sparse_attention_config — flatten both, then defer to the base filter.
        params = dict(params)
        if "text_config" in params and isinstance(params["text_config"], dict):
            merged = dict(params["text_config"])
            merged["model_type"] = params.get("model_type", "minimax_m3_vl")
            params = merged
        sac = params.pop("sparse_attention_config", None)
        if isinstance(sac, dict):
            params.update(sac)
        return super().from_dict(params)

    def __post_init__(self):
        assert self.sparse_num_index_heads == self.num_key_value_heads, (
            "MSA assumes one index head per GQA group "
            f"(got {self.sparse_num_index_heads} index heads, {self.num_key_value_heads} kv heads)"
        )
        assert self.sparse_score_type == "max", "only max block-pooling implemented"
        assert not self.attention_output_gate, "attention_output_gate not implemented"


# ─────────────────────────────────────────────────────────────────────────────
# Norms & activation
# ─────────────────────────────────────────────────────────────────────────────
class GemmaRMSNorm(nn.Module):
    """(1 + w) · x̂  — config: use_gemma_norm=true (mlx_lm/models/gemma.py idiom)."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))  # gemma convention: stored weight ≈ 0, applied as 1+w
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


def make_norm(dims: int, args: "ModelArgs"):
    if args.use_gemma_norm:
        return GemmaRMSNorm(dims, eps=args.rms_norm_eps)
    return nn.RMSNorm(dims, eps=args.rms_norm_eps)


class SwiGLUOAI(nn.Module):
    """gpt-oss style clamped swiglu (mlx_lm/models/gpt_oss.py):
    out = clip(g, max=L) · σ(α·clip(g, max=L)) · (clip(u, ±L) + 1).
    Call signature (x_up, x_gate) matches SwitchGLU's activation contract."""

    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def __call__(self, x_up, x_gate):
        g = mx.clip(x_gate, a_min=None, a_max=self.limit)
        u = mx.clip(x_up, a_min=-self.limit, a_max=self.limit)
        return (g * mx.sigmoid(self.alpha * g)) * (u + 1)


class DenseMLP(nn.Module):
    """Dense FFN for layers with moe_layer_freq == 0 (and the shared expert)."""

    def __init__(self, hidden: int, intermediate: int, args: "ModelArgs"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act = SwiGLUOAI(args.swiglu_alpha, args.swiglu_limit)

    def __call__(self, x):
        return self.down_proj(self.act(self.up_proj(x), self.gate_proj(x)))


# ─────────────────────────────────────────────────────────────────────────────
# MSA Index Branch (eq. 5-7; config-confirmed: max-pool, k=16, B=128, local=1)
# ─────────────────────────────────────────────────────────────────────────────

def _offset_info(cache, B, L_kv):
    """Return (offset_vec[B], left_pad[B]) for both KVCache (scalar offset) and
    BatchKVCache (array offset + left_padding). offset_vec[b] = position of the
    FIRST real (non-pad) key for sample b is 0; the absolute position of physical
    column j for sample b is (j - left_pad[b]). For plain KVCache left_pad=0."""
    off = cache.offset if cache is not None else 0
    lp = getattr(cache, "left_padding", None)
    if lp is None:
        # scalar offset (KVCache); no left padding. offset here = tokens BEFORE this
        # forward's queries already in cache (== L_kv - L for the current call path).
        lp_vec = mx.zeros((B,), dtype=mx.int32)
    else:
        lp_vec = lp.astype(mx.int32)
    return lp_vec


class MSAIndexer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_groups = args.num_key_value_heads
        self.head_dim = args.sparse_index_dim
        self.block_size = args.sparse_block_size
        self.top_k = args.sparse_topk_blocks
        self.n_init = args.sparse_init_block
        self.n_local = args.sparse_local_block
        self.scale = self.head_dim**-0.5

        self.q_idx_proj = nn.Linear(
            args.hidden_size, self.n_groups * self.head_dim, bias=False
        )
        self.k_idx_proj = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        # Index Branch has its OWN qk-norm (checkpoint: index_q_norm / index_k_norm,
        # both shape [d_idx]). Confirmed only from real weights — not in the paper.
        # use_gemma_norm=true ⇒ (1+w)·x̂, applied per index-head over d_idx, pre-score.
        self.q_norm = make_norm(self.head_dim, args)
        self.k_norm = make_norm(self.head_dim, args)
        # Official M3 applies RoPE to index q/k (partial, rotary_dim of head_dim);
        # cos/sin are rotary_dim-wide so the index head_dim slice is a no-op clamp.
        self.rope = nn.RoPE(args.rotary_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x: mx.array, cache: Optional[Any] = None):
        """Batch + left-padding aware MSA index branch — faithful to official HF M3.

        Matches MiniMaxM3VLIndexer: RoPE on index q/k (partial rotary_dim; cos/sin
        are rotary_dim-wide so the head_dim slice is a no-op), fp32 scores, block
        max-pool, then GLOBAL max over index-heads -> ONE block set [B, L, K] shared
        by every attention head. Block grid is content-relative ((j-left_pad)//Bk) so
        non-block-aligned left padding can't shift boundaries (official is slot-relative
        with a known TODO; identical to it on the no-padding case used for parity).

        Returns (topk_idx[B, L, k_eff] LOGICAL block ids, left_pad[B]);
        topk_idx is None when every block is visible.
        """
        B, L, _ = x.shape

        # Cache UNROPED index keys, then RoPE the full stack at concrete LOGICAL
        # positions on read. Rationale: BatchKVCache.offset is lazily tied to the
        # mutable _idx — reading it after update_and_fetch yields the POST-update
        # value even via a pre-captured reference and even after mx.eval. Deriving
        # positions from shapes (L_kv int, left_pad stable) sidesteps that. Roping
        # the full stack at (j - left_pad) is bit-identical to official's
        # rope-at-add-time caching (verified Δ=0).
        k = self.k_idx_proj(x).reshape(B, 1, L, self.head_dim)
        k = self.k_norm(k)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros([B, 1, L, 0], dtype=k.dtype))
        L_kv = k.shape[2]

        lp = getattr(cache, "left_padding", None)
        left_pad = mx.zeros((B,), dtype=mx.int32) if lp is None else lp.astype(mx.int32)
        # logical position of THIS call's first query, per sample (qoff[b]+t).
        qoff = (L_kv - L) - left_pad                                  # (B,)

        # RoPE: physical key column j -> logical position (j - left_pad[b]);
        # new queries -> logical positions qoff[b]+t. Both offsets are concrete.
        k = self.rope(k, offset=(-left_pad))
        q = self.q_idx_proj(x).reshape(B, L, self.n_groups, self.head_dim)
        q = self.q_norm(q).swapaxes(1, 2)                  # (B, H_idx, L, d_idx)
        q = self.rope(q, offset=qoff)

        # fp32 scores, NO scale (top-k is scale-invariant; matches official matmul).
        scores = q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)  # (B,H_idx,L,L_kv)

        # causal + pad visibility in PHYSICAL columns (mlx create_causal_mask contract):
        # query t occupies physical col qphys=(L_kv-L)+t; key j visible iff
        # j <= qphys AND j >= left_pad. left_pad does NOT shift the query.
        neg = mx.array(float("-inf"), mx.float32)
        j = mx.arange(L_kv)[None, None, None, :]
        qphys = ((L_kv - L) + mx.arange(L))[None, None, :, None]
        lpb = left_pad[:, None, None, None]
        scores = mx.where((j <= qphys) & (j >= lpb), scores, neg)

        # block max-pool in the content-relative LOGICAL grid via GATHER — O(L_kv),
        # NOT the O(L_kv·n_blocks) membership broadcast (which at 1M ctx materializes
        # a [B,H_idx,L,L_kv,n_blocks] tensor → OOM). Logical block b, token t lives at
        # physical col left_pad[b] + b·Bk + t; gather those cols, reshape, max over the
        # block, then GLOBAL max over index-heads. Bit-identical to the membership pool.
        Bk = self.block_size
        H_idx = scores.shape[1]
        n_blocks = (L_kv + Bk - 1) // Bk
        phys = mx.arange(n_blocks * Bk)                              # (n_blocks*Bk,) logical pos
        gidx = left_pad[:, None] + phys[None, :]                     # (B, n_blocks*Bk) physical col
        valid_g = gidx < L_kv                                        # in-buffer real cols
        gidx_c = mx.minimum(gidx, L_kv - 1)                          # clamp for safe gather
        gidx_e = mx.broadcast_to(gidx_c[:, None, None, :], (B, H_idx, L, gidx_c.shape[1]))
        gathered = mx.take_along_axis(scores, gidx_e, axis=3)        # (B,H_idx,L,n_blocks*Bk)
        gathered = mx.where(valid_g[:, None, None, :], gathered, neg)
        m_blk = gathered.reshape(B, H_idx, L, n_blocks, Bk).max(axis=-1)  # (B,H_idx,L,n_blocks)
        m_blk = m_blk.max(axis=1)                                    # (B,L,n_blocks) GLOBAL over heads

        big = mx.array(float("inf"), mx.float32)
        if self.n_local > 0:
            qlog = qoff[:, None] + mx.arange(L)[None, :]             # (B,L) logical query pos
            local_blk = (qlog // Bk)[:, :, None]                     # (B,L,1)
            bid = mx.arange(n_blocks)[None, None, :]
            is_local = (bid >= local_blk - (self.n_local - 1)) & (bid <= local_blk)
            m_blk = mx.where(is_local, big, m_blk)
        if self.n_init > 0:
            is_init = mx.arange(n_blocks)[None, None, :] < self.n_init
            m_blk = mx.where(is_init, big, m_blk)

        k_eff = min(self.top_k, n_blocks)
        if k_eff >= n_blocks:
            return None, left_pad
        topk_idx = mx.argpartition(-m_blk, kth=k_eff - 1, axis=-1)[..., :k_eff]
        return topk_idx, left_pad  # (B, L, k_eff) LOGICAL block ids, shared across heads


# ─────────────────────────────────────────────────────────────────────────────
# Attention — one class, two modes: MSA (sparse layers) / full causal (layers 0-2)
# ─────────────────────────────────────────────────────────────────────────────
class MiniMaxM3Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_sparse: bool):
        super().__init__()
        self.is_sparse = is_sparse
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5
        self.block_size = args.sparse_block_size
        self.prefill_chunk = args.msa_prefill_chunk

        self.q_proj = nn.Linear(args.hidden_size, self.num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * head_dim, args.hidden_size, bias=False)

        if is_sparse:
            self.indexer = MSAIndexer(args)

        self.use_qk_norm = args.use_qk_norm
        self.per_head_qk_norm = args.qk_norm_type == "per_head"
        if self.use_qk_norm:
            if self.per_head_qk_norm:
                # (head_dim,) weight shared across heads, applied per head, BEFORE RoPE.
                self.q_norm = make_norm(head_dim, args)
                self.k_norm = make_norm(head_dim, args)
            else:  # M2.7-style joint norm over the concatenated heads
                self.q_norm = make_norm(head_dim * self.num_attention_heads, args)
                self.k_norm = make_norm(head_dim * self.num_key_value_heads, args)

        # Partial RoPE: rotary_dim=64 of head_dim=128 (config: partial_rotary_factor 0.5)
        self.rope = nn.RoPE(args.rotary_dim, traditional=False, base=args.rope_theta)

    def _blocks_to_token_mask(self, topk_idx, L_q, L_kv, qoff, left_pad):
        """Selected (LOGICAL) block → physical-token mask. topk_idx is [B, L_q, K]
        (ONE block set shared by all heads). Physical col j has logical block
        (j - left_pad[b]) // Bk; kept iff its block is selected AND causal AND non-pad.
        Returns [B, 1, L_q, L_kv] (head axis = 1, broadcasts to all heads)."""
        Bk = self.block_size
        blk_of_col = (mx.arange(L_kv)[None, :] - left_pad[:, None]) // Bk       # (B,L_kv)
        sel = (topk_idx[:, :, None, :] == blk_of_col[:, None, :, None]).any(axis=-1)  # (B,L_q,L_kv)
        j = mx.arange(L_kv)[None, None, :]
        lp = left_pad[:, None, None]
        qpos = (qoff[:, None] + mx.arange(L_q)[None, :])[:, :, None]            # (B,L_q,1) logical
        causal_pad = (j <= (qpos + lp)) & (j >= lp)
        return (sel & causal_pad)[:, None, :, :]                               # (B,1,L_q,L_kv)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.use_qk_norm and not self.per_head_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if self.use_qk_norm and self.per_head_qk_norm:
            queries = self.q_norm(queries)  # norm over last dim (= per head)
            keys = self.k_norm(keys)

        if not self.is_sparse:
            # Full-attention layer (0-2): standard GQA path, mask from the model.
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
            out = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
            out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(out)

        # ── MSA layer ──
        if cache is not None:
            main_cache, idx_cache = cache[0], cache[1]
        else:
            main_cache = idx_cache = None
        offset = main_cache.offset if main_cache is not None else 0

        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if main_cache is not None:
            keys, values = main_cache.update_and_fetch(keys, values)
        L_kv = keys.shape[2]

        topk_idx, left_pad = self.indexer(x, cache=idx_cache)

        if main_cache is not None and idx_cache is not None:
            main_cache.keys = mx.depends(
                main_cache.keys, (idx_cache.keys, idx_cache.values)
            )

        G = self.num_attention_heads // self.num_key_value_heads
        # qoff = logical query position (for block ids / local block);
        # qphys = physical query column (for causal visibility, mlx contract).
        qoff = (L_kv - L) - left_pad                                  # (B,) logical

        if topk_idx is None:
            # Everything visible → dense; causal in physical cols + pad blanking.
            j = mx.arange(L_kv)[None, None, None, :]
            lp = left_pad[:, None, None, None]
            qphys = ((L_kv - L) + mx.arange(L))[None, None, :, None]   # (1,1,L,1)
            dense_mask = (j <= qphys) & (j >= lp)                      # (B,1,L,L_kv)
            dense_mask = mx.repeat(dense_mask, self.num_attention_heads, axis=1)
            out = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=dense_mask
            )
        elif L == 1:
            # Decode: ONE selected block set [B,K] (shared by all heads) → physical ids.
            tk = topk_idx[:, 0, :]                                         # (B,K)
            log_tok = (tk[..., None] * self.block_size
                       + mx.arange(self.block_size)).reshape(B, -1)        # (B,k*Bk) logical
            lp_b = left_pad[:, None]                                       # (B,1)
            tok_idx = log_tok + lp_b                                       # (B,k*Bk) physical
            qlog = qoff[:, None]                                           # (B,1) logical query pos
            valid = (log_tok >= 0) & (tok_idx < L_kv) & (tok_idx >= lp_b) & (log_tok <= qlog)
            tok_idx_c = mx.minimum(mx.maximum(tok_idx, 0), L_kv - 1)       # (B,k*Bk)
            # gather the SAME physical columns across all KV heads
            idx_g = mx.broadcast_to(
                tok_idx_c[:, None, :, None],
                (B, self.num_key_value_heads, tok_idx_c.shape[1], keys.shape[-1]),
            )
            k_sel = mx.take_along_axis(keys, idx_g, axis=2)               # (B,H_kv,k*Bk,D)
            v_sel = mx.take_along_axis(values, idx_g, axis=2)
            k_sel = mx.repeat(k_sel, G, axis=1)                           # (B,H_q,k*Bk,D)
            v_sel = mx.repeat(v_sel, G, axis=1)
            smask = mx.broadcast_to(
                valid[:, None, None, :],
                (B, self.num_attention_heads, 1, valid.shape[1]),
            )
            out = mx.fast.scaled_dot_product_attention(
                queries, k_sel, v_sel, scale=self.scale, mask=smask
            )
        else:
            out = self._msa_prefill(queries, keys, values, topk_idx, qoff, left_pad, G)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)

    # ── chunked block-sparse prefill (perf path; ≡ _prefill_mask numerically) ──
    def _msa_prefill(self, queries, keys, values, topk_idx, qoff, left_pad, G):
        """Chunked block-sparse prefill. topk_idx: [B, L, K] (ONE block set shared by
        all heads). Per chunk slice K/V to a tail bound and mask to the selected
        blocks + causal + pad. Cost O(C × kv_hi)."""
        B, H_q, L, Dh = queries.shape
        L_kv = keys.shape[2]
        Bk = self.block_size
        chunk = self.prefill_chunk
        base_phys = L_kv - L                       # physical self-index of query 0
        outs = []
        for s in range(0, L, chunk):
            e = min(s + chunk, L)
            last_self = base_phys + (e - 1)
            kv_hi = min(((last_self // Bk) + 1) * Bk, L_kv)
            tk = topk_idx[:, s:e, :]               # (B,chunk,K)
            qc = queries[:, :, s:e, :]
            kc = keys[:, :, :kv_hi, :]
            vc = values[:, :, :kv_hi, :]
            qoff_chunk = qoff + s                  # (B,)
            keep = self._blocks_to_token_mask(tk, e - s, kv_hi, qoff_chunk, left_pad)  # (B,1,chunk,kv_hi)
            keep = mx.repeat(keep, self.num_attention_heads, axis=1)
            kc = mx.repeat(kc, G, axis=1)
            vc = mx.repeat(vc, G, axis=1)
            outs.append(mx.fast.scaled_dot_product_attention(
                qc, kc, vc, scale=self.scale, mask=keep))
        return mx.concatenate(outs, axis=2)

    def _prefill_mask(self, queries, keys, values, topk_idx, qoff, left_pad, G):
        """Reference: full L_q×L_kv block→token mask + dense SDPA (head-shared blocks)."""
        L = queries.shape[2]
        L_kv = keys.shape[2]
        keep = self._blocks_to_token_mask(topk_idx, L, L_kv, qoff, left_pad)   # (B,1,L,L_kv)
        keep = mx.repeat(keep, self.num_attention_heads, axis=1)
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=keep)


# ─────────────────────────────────────────────────────────────────────────────
# MoE: sigmoid + routing bias + top-4 → normalize → ×routed_scaling_factor,
# plus one always-on shared expert (deepseek_v3.py idiom).
# ─────────────────────────────────────────────────────────────────────────────
class MiniMaxM3SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.routed_scaling_factor
        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.intermediate_size,
            args.num_local_experts,
            activation=SwiGLUOAI(args.swiglu_alpha, args.swiglu_limit),
        )
        if args.use_routing_bias:
            self.e_score_correction_bias = mx.zeros((args.num_local_experts,))
        else:
            self.e_score_correction_bias = None
        if args.n_shared_experts and args.n_shared_experts > 0:
            self.shared_experts = DenseMLP(
                args.hidden_size,
                args.shared_intermediate_size * args.n_shared_experts,
                args,
            )
        else:
            self.shared_experts = None
        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        gates = self.gate(x.astype(mx.float32))
        scores = mx.sigmoid(gates)
        orig_scores = scores
        if self.e_score_correction_bias is not None:
            scores = scores + self.e_score_correction_bias
        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        scores = scores * self.routed_scaling_factor
        scores = scores.astype(x.dtype)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        if self.sharding_group is not None:
            # fp32 reduce: bf16 all_sum of MoE outputs (top-4 × routed_scaling 2.0 +
            # shared expert) drifts the hidden state on massive-activation channels —
            # same failure mode MiMo's F2 fixed. Reduce in fp32, cast back.
            y = mx.distributed.all_sum(
                y.astype(mx.float32), group=self.sharding_group
            ).astype(x.dtype)
        return y


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_sparse_attn = bool(args.sparse_attention_freq[layer_idx]) and args.use_sparse_attention
        self.is_moe = bool(args.moe_layer_freq[layer_idx])
        self.self_attn = MiniMaxM3Attention(args, is_sparse=self.is_sparse_attn)
        if self.is_moe:
            self.block_sparse_moe = MiniMaxM3SparseMoeBlock(args)
        else:
            self.mlp = DenseMLP(args.hidden_size, args.dense_intermediate_size, args)
        self.input_layernorm = make_norm(args.hidden_size, args)
        self.post_attention_layernorm = make_norm(args.hidden_size, args)

    def __call__(self, x, mask=None, cache=None):
        r = x + self.self_attn(self.input_layernorm(x), mask, cache)
        ffn = self.block_sparse_moe if self.is_moe else self.mlp
        return r + ffn(self.post_attention_layernorm(r))


class MiniMaxM3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MiniMaxM3DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = make_norm(args.hidden_size, args)
        # Index of the first full-attention layer (for the shared causal mask).
        self._full_idx = next(
            (i for i, l in enumerate(self.layers) if not l.is_sparse_attn), None
        )

    def __call__(self, inputs, mask=None, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        full_mask = None
        if self._full_idx is not None:
            try:
                from .base import create_attention_mask
            except ImportError:
                from mlx_lm.models.base import create_attention_mask
            full_mask = create_attention_mask(h, cache[self._full_idx])
        for layer, c in zip(self.layers, cache):
            h = layer(h, full_mask if not layer.is_sparse_attn else None, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniMaxM3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, mask=None, cache=None):
        out = self.model(inputs, mask=mask, cache=cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def make_cache(self):
        # Heterogeneous (mimo_v2_flash idiom):
        #   full-attn layers → plain KVCache
        #   MSA layers       → CacheList(main KV, indexer keys) — exo handles
        #                      CacheList natively (copies/trims it as a unit).
        # NEVER substitute RotatingKVCache anywhere: MSA selects blocks at
        # arbitrary distance and layers 0-2 are full attention.
        caches = []
        for l in self.model.layers:
            if l.is_sparse_attn:
                caches.append(CacheList(KVCache(), KVCache()))
            else:
                caches.append(KVCache())
        return caches

    # ── weight loading ───────────────────────────────────────────────────────
    def sanitize(self, weights):
        def dequant(weight, scale_inv):
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            bs = 128
            m, n = weight.shape
            pb, ps = (-m) % bs, (-n) % bs
            weight = mx.pad(weight, ((0, pb), (0, ps)))
            weight = weight.reshape(((m + pb) // bs, bs, (n + ps) // bs, bs))
            weight = (weight * scale_inv[:, None, :, None]).reshape(m + pb, n + ps)
            return weight[:m, :n].astype(mx.bfloat16)

        out = {}
        for k, v in weights.items():
            # VL checkpoint: language weights sit under "language_model.".
            if k.startswith("language_model."):
                k = k[len("language_model."):]
            # Text backbone only: keep just model.* / lm_head.* (drops the whole
            # vision tower + projector, whatever their exact prefixes).
            if not (k.startswith("model.") or k.startswith("lm_head")):
                continue
            # Belt-and-suspenders for multimodal bits that hide under model.*
            if any(s in k for s in ("vision", "multi_modal", "image_newline",
                                    "mm_projector", "visual")):
                continue
            # Drop MTP (next-token-prediction) modules and any layer index beyond
            # num_hidden_layers (MTP appended as extra layers).
            if k.startswith("model.mtp") or ".mtp" in k:
                continue
            mm = re.match(r"model\.layers\.(\d+)\.", k)
            if mm and int(mm.group(1)) >= self.args.num_hidden_layers:
                continue
            out[k] = v
        weights = out

        # FP8 block-dequant (family convention)
        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                wk = k.replace("_scale_inv", "")
                new_weights[wk] = dequant(weights[wk], v)
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        # ── Indexer / shared-expert weight-name remap hook ──────────────────
        # Fill in once model.safetensors.index.json is inspected. Matching is
        # by suffix; left side = checkpoint name tail, right side = ours.
        _RENAMES = {
            # confirmed against model.safetensors.index.json @ 3a41b31:
            "self_attn.index_q_proj.weight": "self_attn.indexer.q_idx_proj.weight",
            "self_attn.index_k_proj.weight": "self_attn.indexer.k_idx_proj.weight",
            "self_attn.index_q_norm.weight": "self_attn.indexer.q_norm.weight",
            "self_attn.index_k_norm.weight": "self_attn.indexer.k_norm.weight",
        }
        if _RENAMES:
            remapped = {}
            for k, v in weights.items():
                for old, new in _RENAMES.items():
                    if k.endswith(old):
                        k = k[: -len(old)] + new
                        break
                remapped[k] = v
            weights = remapped

        # ── transformers-integration (v5.12.0) FUSED gate_up_proj format ─────────
        # Official MiniMax-M3 stores gate+up FUSED and names the MoE block `mlp`
        # (verified vs transformers v5.12.0: DenseMLP.gate_up_proj=Linear[2I,H];
        # Experts.gate_up_proj=Parameter[E,2I,H]; SparseMoeBlock.{gate,experts,
        # shared_experts}). My model uses SPLIT gate/up and names the MoE block
        # block_sparse_moe + switch_mlp, so remap+split here. Detect by a fused key.
        # Split is on UNQUANTIZED (bf16) weights only — at convert-time, BEFORE
        # nn.quantize. A pre-quantized fused ckpt can't be split (groups span 2I) →
        # reject loudly rather than silently mis-split.
        if any(k.endswith("gate_up_proj") or k.endswith("gate_up_proj.weight") for k in weights):
            for k in weights:
                if "gate_up_proj" in k and (k.endswith(".scales") or k.endswith(".biases")):
                    raise ValueError(
                        f"pre-quantized fused gate_up_proj ({k}) cannot be split safely "
                        "(quant groups span the 2*intermediate axis). Convert from bf16 "
                        "(split runs before quantization) or pre-split the checkpoint.")
            freq = list(self.args.moe_layer_freq)
            fused = {}
            for l in range(self.args.num_hidden_layers):
                p = f"model.layers.{l}"
                is_moe = l < len(freq) and bool(freq[l])
                if not is_moe:
                    gu = weights.pop(f"{p}.mlp.gate_up_proj.weight", None)
                    if gu is not None:
                        g, u = mx.split(gu, 2, axis=0)          # [2I,H] -> two [I,H]
                        fused[f"{p}.mlp.gate_proj.weight"] = g
                        fused[f"{p}.mlp.up_proj.weight"] = u
                    continue
                # MoE layer: rename mlp.* -> block_sparse_moe.*, split experts/shared
                bsm = f"{p}.block_sparse_moe"
                ge = weights.pop(f"{p}.mlp.experts.gate_up_proj", None)
                if ge is not None:
                    g, u = mx.split(ge, 2, axis=1)              # [E,2I,H] -> two [E,I,H]
                    fused[f"{bsm}.switch_mlp.gate_proj.weight"] = g
                    fused[f"{bsm}.switch_mlp.up_proj.weight"] = u
                de = weights.pop(f"{p}.mlp.experts.down_proj", None)
                if de is not None:
                    fused[f"{bsm}.switch_mlp.down_proj.weight"] = de
                gs = weights.pop(f"{p}.mlp.shared_experts.gate_up_proj.weight", None)
                if gs is not None:
                    g, u = mx.split(gs, 2, axis=0)
                    fused[f"{bsm}.shared_experts.gate_proj.weight"] = g
                    fused[f"{bsm}.shared_experts.up_proj.weight"] = u
                ds = weights.pop(f"{p}.mlp.shared_experts.down_proj.weight", None)
                if ds is not None:
                    fused[f"{bsm}.shared_experts.down_proj.weight"] = ds
                # router correction bias: HF registers it as a buffer ON the router
                # (gate.e_score_correction_bias), but my model holds it at BLOCK level
                # (block_sparse_moe.e_score_correction_bias). Pull it out BEFORE the
                # generic mlp.gate.* remap, else it lands at block_sparse_moe.gate.*.
                eb = weights.pop(f"{p}.mlp.gate.e_score_correction_bias", None)
                if eb is not None:
                    fused[f"{bsm}.e_score_correction_bias"] = eb
                # router weight (+ any remaining gate params): mlp.gate.* -> block_sparse_moe.gate.*
                for rk in [kk for kk in list(weights) if kk.startswith(f"{p}.mlp.gate.")]:
                    fused[rk.replace(f"{p}.mlp.gate.", f"{bsm}.gate.", 1)] = weights.pop(rk)
            weights.update(fused)

        # Stack experts; also stack quantized parts so pre-quantized ckpts load.
        if any(".block_sparse_moe.experts.0." in k for k in weights):
            for l in range(self.args.num_hidden_layers):
                prefix = f"model.layers.{l}"
                for orig, new in {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}.items():
                    for part in ("weight", "scales", "biases"):
                        key0 = f"{prefix}.block_sparse_moe.experts.0.{orig}.{part}"
                        if key0 in weights:
                            to_join = [
                                weights.pop(f"{prefix}.block_sparse_moe.experts.{e}.{orig}.{part}")
                                for e in range(self.args.num_local_experts)
                            ]
                            weights[f"{prefix}.block_sparse_moe.switch_mlp.{new}.{part}"] = mx.stack(to_join)
                    # HF-style expert names (gate/up/down) — alternative convention
                    for orig in ("gate_proj", "up_proj", "down_proj"):
                        for part in ("weight", "scales", "biases"):
                            key0 = f"{prefix}.block_sparse_moe.experts.0.{orig}.{part}"
                            if key0 in weights:
                                to_join = [
                                    weights.pop(f"{prefix}.block_sparse_moe.experts.{e}.{orig}.{part}")
                                    for e in range(self.args.num_local_experts)
                                ]
                                weights[f"{prefix}.block_sparse_moe.switch_mlp.{orig}.{part}"] = mx.stack(to_join)
        return weights

    # ── tensor parallel ──────────────────────────────────────────────────────
    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        assert self.args.num_key_value_heads % N == 0, (
            f"TP degree {N} must divide num_key_value_heads={self.args.num_key_value_heads}"
        )
        for layer in self.model.layers:
            sa = layer.self_attn
            sa.q_proj = shard_linear(sa.q_proj, "all-to-sharded", group=group)
            sa.k_proj = shard_linear(sa.k_proj, "all-to-sharded", group=group)
            sa.v_proj = shard_linear(sa.v_proj, "all-to-sharded", group=group)
            sa.o_proj = shard_linear(sa.o_proj, "sharded-to-all", group=group)
            if sa.is_sparse:
                # MSA indexer FULLY REPLICATED. Official block selection does a
                # GLOBAL amax over ALL index-heads -> one [B,L,K] set shared by every
                # attention head; each rank needs all index-heads to reproduce it.
                # Inputs are replicated across ranks (per-layer all-reduce) so every
                # rank computes the identical selection, then gathers tokens for its
                # own kv-head shard. q_idx_proj/k_idx_proj/q_norm/k_norm untouched.
                # (Indexer is tiny vs the 428B MoE; redundant per-rank compute is fine.)
                pass
            # per-head qk-norm weight is (head_dim,) and head-local → replicate (no-op).
            # (per_layer/joint variant would need ShardedRMSNorm — not used by M3.)
            if sa.use_qk_norm and not sa.per_head_qk_norm:
                raise NotImplementedError(
                    "TP for joint (per_layer) qk_norm not wired; M3 uses per_head"
                )
            sa.num_attention_heads //= N
            sa.num_key_value_heads //= N

            if layer.is_moe:
                moe = layer.block_sparse_moe
                shard_inplace(moe.switch_mlp.gate_proj, "all-to-sharded", group=group)
                shard_inplace(moe.switch_mlp.down_proj, "sharded-to-all", group=group)
                shard_inplace(moe.switch_mlp.up_proj, "all-to-sharded", group=group)
                if moe.shared_experts is not None:
                    # Rides the MoE block's single all_sum (deepseek_v3 idiom).
                    shard_inplace(moe.shared_experts.gate_proj, "all-to-sharded", group=group)
                    shard_inplace(moe.shared_experts.up_proj, "all-to-sharded", group=group)
                    shard_inplace(moe.shared_experts.down_proj, "sharded-to-all", group=group)
                moe.sharding_group = group
            else:
                mlp = layer.mlp
                mlp.gate_proj = shard_linear(mlp.gate_proj, "all-to-sharded", group=group)
                mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
                mlp.down_proj = shard_linear(mlp.down_proj, "sharded-to-all", group=group)

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        return lambda k: "e_score_correction_bias" not in k

    @property
    def quant_predicate(self):
        """Auto-picked-up by mlx_lm convert (utils.quantize_model:
        `quant_predicate or getattr(model, "quant_predicate", None)`).

        Quality recipe: BOTH routing paths stay in original bf16 —
          • MoE router gate (60 × 128 × 6144 ≈ 94 MB total): expert routing
            is the most quant-sensitive part of a MoE.
          • MSA indexer q/k projections (57 × ~3.9 M ≈ 0.45 GB total): block
            selection IS a routing decision; quant noise here corrupts which
            KV the model even sees.
        Combined cost ≈ 0.55 GB on a ~450 GB model — free insurance.
        Everything else (incl. embed/lm_head) → standard affine 8-bit, which
        is effectively lossless vs bf16."""

        def predicate(path, _):
            if path.endswith("block_sparse_moe.gate"):
                return False  # keep bf16
            if "indexer" in path:
                return False  # keep bf16
            return True       # quantize with the CLI-level bits/group_size

        return predicate
