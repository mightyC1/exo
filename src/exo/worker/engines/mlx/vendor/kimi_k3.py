# Kimi K3 (moonshotai/Kimi-K3, text-only) — MLX implementation for EXO.
#
# Vendored (vendor+register pattern, как minimax_m3_vl / mimo_v2): регистрируется
# в sys.modules["mlx_lm.models.kimi_k3"] через custom_models. Пин mlx-lm не трогаем.
#
# ИСТОЧНИК СЕМАНТИКИ: официальный modeling_kimi_linear.py из репозитория
# moonshotai/Kimi-K3 (auto_map) + шейпы реального чекпоинта (A0/A0.5, 2026-07-29).
# База реализации: mlx_lm.models.kimi_linear из пина (та же KDA-семья, parity-
# проверенный порт) + K3-дельты:
#   * KDA: full-rank выходной гейт g_proj (вместо g_a/g_b);
#     decay = exp(lower_bound * sigmoid(exp(A_log) * (f(x) + dt_bias)))
#     (safe_gate, lower_bound=-5.0) вместо softplus-формы kimi_linear;
#     o_norm = RMSNorm(head_dim) с ВЕСОМ * sigmoid(g_proj) (FusedRMSNormGated).
#   * MLA: q-LoRA (q_a/q_a_ln/q_b), 64-мерный MQA-хвост БЕЗ ротации
#     (rotary_emb=None в официальном коде; mla_use_nope=True), scale=192^-0.5,
#     выходной гейт sigmoid(g_proj) перед o_proj; absorbed-кэш: latent 512 + хвост 64.
#   * LatentMoE: down(7168->3584) -> 896 экспертов (SwitchGLU в латенте, SiTU)
#     -> weighted sum -> [all_sum] -> RMSNorm(3584) -> up(3584->7168); shared
#     (2x3072=6144) на полном hidden. Роутер: fp32 matmul, sigmoid, выбор по
#     scores+e_score_correction_bias, ВЕСА из сырых scores, renorm(+1e-20).
#   * AttnRes: softmax-аттеншен по снапшотам резидуал-стрима. Снапшоты на
#     0-based слоях i%12==0 (стрим РЕБЕЙЗИТСЯ); на каждом слое два применения
#     (перед attention при наличии снапшотов и перед MLP), финальное — перед
#     model.norm. score = <v/rms(v), norm.w*proj.w>, softmax, mix сырых v (fp32).
#   * SiTU: b*tanh(gate/b)*sigmoid(gate) * lb*tanh(up/lb), fp32 (b=4.0, lb=25.0).
#
# ИНВАРИАНТЫ:
#   INV#2  full-rank g_proj (g_a/g_b отсутствуют) — подтверждено A0.
#   INV#3/#11  ровно один routed all_sum ДО routed RMSNorm.
#   INV#12 fp32 вокруг distributed all_sum (fused concat: latent+shared).
#   INV#13 router matmul fp32.
#   INV#14 SiTU и AttnRes-score в fp32.
#
# P0-008b (семантика A_log): чекпоинт хранит [128] при num_heads=96.
# Официальный код объявляет Parameter([num_heads]) => per-head; llama.cpp
# (A_log[:96]) прошёл logit-parity 6.7e-05 vs Moonshot. ДЕФОЛТ: per_head со
# срезом [:96] в remap. EXO_K3_ALOG_SEMANTICS=per_channel оставлен для
# контрольного parity-эксперимента (A3).

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_ops
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.switch_layers import SwitchGLU

ALOG_SEMANTICS = os.environ.get("EXO_K3_ALOG_SEMANTICS", "per_head")  # per_head|per_channel
DTBIAS_IN_GATE = os.environ.get("EXO_K3_DTBIAS_IN_GATE", "1") == "1"

_logged = False


def _log_once() -> None:
    global _logged
    if not _logged:
        print(
            f"[kimi_k3] A_log semantics = {ALOG_SEMANTICS} "
            f"(per_head = официальный код + llama.cpp parity 6.7e-05), "
            f"dt_bias_in_gate = {DTBIAS_IN_GATE}"
        )
    _logged = True


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "kimi_k3"
    vocab_size: int = 163840
    hidden_size: int = 7168
    num_hidden_layers: int = 93
    num_attention_heads: int = 96
    num_key_value_heads: int = 96
    intermediate_size: int = 33792
    hidden_act: str = "situ"
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 1048576

    linear_attn_config: Dict[str, Any] = field(default_factory=dict)

    # MLA
    q_lora_rank: Optional[int] = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64  # МХА-хвост БЕЗ ротации (mla_use_nope)
    v_head_dim: int = 128
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True

    # MoE
    num_experts: int = 896
    num_experts_per_token: int = 16
    num_shared_experts: int = 2
    moe_intermediate_size: int = 3072
    routed_expert_hidden_size: Optional[int] = 3584
    latent_moe_use_norm: bool = True
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    num_expert_group: int = 1
    topk_group: int = 1
    use_grouped_topk: bool = True
    topk_method: str = "noaux_tc"

    # SiTU
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0

    # AttnRes
    attn_res_block_size: Optional[int] = 12

    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        # Официальный конфиг — мультимодальная обёртка: текст в text_config.
        if "text_config" in params and isinstance(params["text_config"], dict):
            merged = dict(params["text_config"])
            for k in ("quantization", "quantization_config"):
                if k in params and k not in merged:
                    merged[k] = params[k]
            params = merged
        params = dict(params)
        params["model_type"] = "kimi_k3"  # text_config несёт "kimi_linear"
        return super().from_dict(params)

    # KDA-параметры из nested linear_attn_config
    @property
    def kda_layers(self) -> List[int]:
        return list(self.linear_attn_config.get("kda_layers", []))

    @property
    def gate_lower_bound(self) -> Optional[float]:
        v = self.linear_attn_config.get("gate_lower_bound", None)
        return None if v is None else float(v)

    @property
    def kda_num_heads(self) -> int:
        return int(self.linear_attn_config.get("num_heads", self.num_attention_heads))

    @property
    def kda_head_dim(self) -> int:
        return int(self.linear_attn_config.get("head_dim", 128))

    @property
    def conv_kernel(self) -> int:
        return int(self.linear_attn_config.get("short_conv_kernel_size", 4))

    @property
    def use_full_rank_gate(self) -> bool:
        return bool(self.linear_attn_config.get("use_full_rank_gate", True))


# ---------------------------------------------------------------------------
# SiTU (fp32) — точная форма из официального SituAndMul
# ---------------------------------------------------------------------------

def situ_mul(gate: mx.array, up: mx.array, beta: float, linear_beta: Optional[float]) -> mx.array:
    dt = up.dtype
    g = gate.astype(mx.float32)
    u = up.astype(mx.float32)
    a = beta * mx.tanh(g / beta) * mx.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * mx.tanh(u / linear_beta)
    return (a * u).astype(dt)


class SituGLUAct:
    """Активация для SwitchGLU пина: вызывается как activation(x_up, x_gate)."""

    def __init__(self, beta: float, linear_beta: Optional[float]):
        self.beta = beta
        self.linear_beta = linear_beta

    def __call__(self, x_up: mx.array, x_gate: mx.array) -> mx.array:
        return situ_mul(x_gate, x_up, self.beta, self.linear_beta)


class KimiK3MLP(nn.Module):
    """Dense MLP / shared experts (gate/up/down + SiTU)."""

    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        inter = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, args.hidden_size, bias=False)
        self._beta = args.activation_situ_beta
        self._lbeta = args.activation_situ_linear_beta

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(situ_mul(self.gate_proj(x), self.up_proj(x), self._beta, self._lbeta))


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

class KimiK3MoEGate(nn.Module):
    """noaux_tc: fp32-логиты, sigmoid; выбор по scores+bias, веса из сырых scores."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        if args.num_expert_group != 1 or args.topk_group != 1:
            raise NotImplementedError(
                "kimi_k3: grouped top-k (num_expert_group>1) не реализован — "
                "K3-конфиг использует 1/1"
            )
        self.args = args
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.num_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        a = self.args
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T  # INV#13
        if a.moe_router_activation_func == "sigmoid":
            scores = mx.sigmoid(logits)
        else:
            scores = mx.softmax(logits, axis=-1)
        choice = scores + self.e_score_correction_bias
        k = a.num_experts_per_token
        inds = mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if k > 1 and a.moe_renormalize:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = weights * a.routed_scaling_factor
        return inds, weights


class KimiK3LatentMoE(nn.Module):
    """Латентный MoE. При TP: эксперты и shared шардируются, роутер/латентные
    down/norm/up реплицируются; ОДИН fused fp32 all_sum (latent‖shared)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        latent = args.routed_expert_hidden_size or args.hidden_size
        self.latent = latent
        self.gate = KimiK3MoEGate(args)
        self.switch_mlp = SwitchGLU(
            latent,
            args.moe_intermediate_size,
            args.num_experts,
            activation=SituGLUAct(args.activation_situ_beta, args.activation_situ_linear_beta),
        )
        self.routed_expert_down_proj = nn.Linear(args.hidden_size, latent, bias=False)
        self.routed_expert_up_proj = nn.Linear(latent, args.hidden_size, bias=False)
        if args.latent_moe_use_norm:
            self.routed_expert_norm = nn.RMSNorm(latent, eps=args.rms_norm_eps)
        if args.num_shared_experts:
            self.shared_experts = KimiK3MLP(
                args, intermediate_size=args.moe_intermediate_size * args.num_shared_experts
            )
        else:
            self.shared_experts = None
        self.sharding_group: Optional[mx.distributed.Group] = None

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        inds, weights = self.gate(x)
        lat = self.routed_expert_down_proj(x)
        y = self.switch_mlp(lat, inds)
        y = (y * weights[..., None].astype(y.dtype)).sum(axis=-2)  # [B,T,latent], partial при TP
        shared = self.shared_experts(identity) if self.shared_experts is not None else None

        if self.sharding_group is not None:
            # INV#12: fp32; один коллектив на слой — конкат latent‖shared.
            if shared is not None:
                fused = mx.concatenate(
                    [y.astype(mx.float32), shared.astype(mx.float32)], axis=-1
                )
                fused = mx.distributed.all_sum(fused, group=self.sharding_group)
                y = fused[..., : self.latent].astype(y.dtype)
                shared = fused[..., self.latent:].astype(identity.dtype)
            else:
                y = mx.distributed.all_sum(
                    y.astype(mx.float32), group=self.sharding_group
                ).astype(y.dtype)

        if self.args.latent_moe_use_norm:
            y = self.routed_expert_norm(y)  # INV#3/#11: строго ПОСЛЕ полного суммирования
        y = self.routed_expert_up_proj(y)
        if shared is not None:
            y = y + shared
        return y


# ---------------------------------------------------------------------------
# KDA (delta attention) — база: пин kimi_linear; дельты K3 см. шапку
# ---------------------------------------------------------------------------

class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            bias=False,
            groups=channels,
            padding=0,
        )

    def __call__(
        self,
        x: mx.array,
        state: Optional[mx.array],
        mask: Optional[mx.array],
        lengths: Optional[mx.array],
    ) -> Tuple[mx.array, mx.array]:
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)
        if state is None:
            state = mx.zeros((x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        out = nn.silu(self.conv(conv_input))
        n_keep = self.kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_state = mx.contiguous(conv_input[:, -n_keep:, :])
        return out, new_state


def compute_decay_k3(
    a_logits: mx.array,      # [B,T,H,D] = f_b(f_a(x))
    A_log: mx.array,         # per_head: [H]; per_channel: [D]
    dt_bias: mx.array,       # [H*D]
    lower_bound: float,
    num_heads: int,
    head_dim: int,
) -> mx.array:
    """K3 safe_gate: log_g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias)),
    decay = exp(log_g) in (exp(lb), 1). Всё в fp32."""
    x = a_logits.astype(mx.float32)
    if DTBIAS_IN_GATE:
        x = x + dt_bias.astype(mx.float32).reshape(num_heads, head_dim)
    if ALOG_SEMANTICS == "per_head":
        A = mx.exp(A_log.astype(mx.float32)).reshape(num_heads, 1)
    else:  # per_channel (контрольный эксперимент P0-008b)
        A = mx.exp(A_log.astype(mx.float32)).reshape(1, head_dim)
    log_g = lower_bound * mx.sigmoid(A * x)
    return mx.exp(log_g)


class KimiK3DeltaAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = args.kda_num_heads
        self.head_dim = args.kda_head_dim
        self.conv_kernel = args.conv_kernel
        self.lower_bound = args.gate_lower_bound
        if self.lower_bound is None:
            raise ValueError("kimi_k3: ожидается gate_lower_bound в linear_attn_config")
        if not args.use_full_rank_gate:
            raise ValueError("kimi_k3: use_full_rank_gate=False не поддержан (INV#2)")

        self.projection_dim = self.num_heads * self.head_dim
        hidden = args.hidden_size
        self.scale = float(self.head_dim) ** -0.5

        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)  # INV#2

        if ALOG_SEMANTICS == "per_head":
            self.A_log = mx.zeros((self.num_heads,), dtype=mx.float32)
        else:
            self.A_log = mx.zeros((self.head_dim,), dtype=mx.float32)
        self.dt_bias = mx.zeros((self.projection_dim,), dtype=mx.float32)

        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        if cache is not None:
            q_state, k_state, v_state, ssm_state = cache
            lengths = cache.lengths
        else:
            q_state = k_state = v_state = ssm_state = None
            lengths = None
        if q_state is None:
            s = mx.zeros((B, self.conv_kernel - 1, self.projection_dim), dtype=x.dtype)
            q_state, k_state, v_state = s, s, s

        q_conv, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k_conv, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v_conv, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)
        if cache is not None:
            cache[0], cache[1], cache[2] = q_state, k_state, v_state

        q = q_conv.reshape(B, T, H, D)
        k = k_conv.reshape(B, T, H, D)
        v = v_conv.reshape(B, T, H, D)

        # qk L2-norm + масштабирование — ровно как в parity-порте kimi_linear
        # (use_qk_l2norm_in_kernel в официальном вызове).
        q = (self.scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = self.scale * mx.fast.rms_norm(k, None, 1e-6)

        a_logits = self.f_b_proj(self.f_a_proj(x)).reshape(B, T, H, D)
        decay = compute_decay_k3(
            a_logits, self.A_log, self.dt_bias, self.lower_bound, H, D
        )
        beta = mx.sigmoid(self.b_proj(x).astype(mx.float32))  # sigmoid в ядре у Moonshot

        out, ssm_state = gated_delta_ops(q, k, v, decay, beta, ssm_state, mask)
        if cache is not None:
            cache[3] = ssm_state
            if hasattr(cache, "advance"):
                cache.advance(T)

        gate = self.g_proj(x).reshape(B, T, H, D)
        out = (self.o_norm(out.reshape(B, T, H, D)) * mx.sigmoid(gate)).reshape(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# MLA (q-LoRA + безротационный MQA-хвост + выходной гейт), absorbed-кэш
# ---------------------------------------------------------------------------

class KimiK3MLAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.mla_use_nope, "kimi_k3: официальный код assert'ит mla_use_nope"
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim  # хвост, ротация НЕ применяется
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = self.q_head_dim ** -0.5  # 192^-0.5

        hidden = args.hidden_size
        self.q_a_proj = nn.Linear(hidden, args.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(args.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(self.kv_lora_rank, self.v_head_dim, self.num_heads)

        self.use_output_gate = args.mla_use_output_gate
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden, self.num_heads * self.v_head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = mx.expand_dims(self.kv_a_layernorm(compressed_kv), axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        # Хвост участвует в скорах БЕЗ ротации (официально rotary_emb=None).
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.use_output_gate:
            output = output * mx.sigmoid(self.g_proj(x))
        return self.o_proj(output)


# ---------------------------------------------------------------------------
# AttnRes: softmax-микс по снапшотам резидуал-стрима (fp32)
# ---------------------------------------------------------------------------

def _attn_res_mix(
    prefix: mx.array,               # [B,T,Hid]
    snapshots: List[mx.array],      # список [B,T,Hid]
    proj_weight: mx.array,          # [1,Hid]
    norm_weight: mx.array,          # [Hid]
    eps: float,
) -> mx.array:
    v = mx.stack(snapshots + [prefix], axis=-2).astype(mx.float32)  # [B,T,S,Hid]
    k = v * mx.rsqrt(mx.mean(v * v, axis=-1, keepdims=True) + eps)
    sw = (norm_weight.astype(mx.float32) * proj_weight.reshape(-1).astype(mx.float32))
    scores = (k * sw).sum(axis=-1)                     # [B,T,S]  (INV#14: fp32)
    probs = mx.softmax(scores, axis=-1)
    out = (probs[..., None] * v).sum(axis=-2)          # микс СЫРЫХ снапшотов
    return out.astype(prefix.dtype)


# ---------------------------------------------------------------------------
# decoder / model
# ---------------------------------------------------------------------------

class KimiK3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear = (layer_idx + 1) in args.kda_layers  # 1-based список

        if self.is_linear:
            self.self_attn = KimiK3DeltaAttention(args, layer_idx)
        else:
            self.self_attn = KimiK3MLAAttention(args)

        self.is_moe = (
            args.num_experts is not None
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        )
        if self.is_moe:
            self.block_sparse_moe = KimiK3LatentMoE(args)
        else:
            self.mlp = KimiK3MLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.use_attn_residuals = args.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = args.attn_res_block_size
            self.self_attention_res_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.mlp_res_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
        self._eps = args.rms_norm_eps

    def _ffn(self, h: mx.array) -> mx.array:
        return self.block_sparse_moe(h) if self.is_moe else self.mlp(h)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        snapshots: Optional[List[mx.array]] = None,
    ) -> mx.array:
        if not self.use_attn_residuals:
            h = x + self.self_attn(self.input_layernorm(x), mask, cache)
            return h + self._ffn(self.post_attention_layernorm(h))

        # Официальный датафлоу _forward_attn_residual (ребейз стрима на снапшотах).
        prefix = x
        h = x
        if snapshots:
            h = _attn_res_mix(
                prefix, snapshots,
                self.self_attention_res_proj.weight,
                self.self_attention_res_norm.weight, self._eps,
            )
        rebase = (self.layer_idx % self.attn_res_block_size) == 0
        if rebase:
            snapshots.append(prefix)
            prefix = None

        a = self.self_attn(self.input_layernorm(h), mask, cache)
        prefix = a if prefix is None else prefix + a

        h2 = _attn_res_mix(
            prefix, snapshots,
            self.mlp_res_proj.weight, self.mlp_res_norm.weight, self._eps,
        )
        m = self._ffn(self.post_attention_layernorm(h2))
        return prefix + m


class KimiK3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [KimiK3DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.use_attn_residuals = args.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.output_attn_res_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.output_attn_res_proj = nn.Linear(args.hidden_size, 1, bias=False)

        kda = args.kda_layers
        self.ssm_idx = kda[0] - 1 if kda else 0
        self.attn_idx = 0
        for i in range(args.num_hidden_layers):
            if (i + 1) not in kda:
                self.attn_idx = i
                break

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        attn_mask = create_attention_mask(h, cache[self.attn_idx], return_array=True)

        snapshots: Optional[List[mx.array]] = [] if self.use_attn_residuals else None
        for layer, layer_cache in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else attn_mask
            h = layer(h, mask=mask, cache=layer_cache, snapshots=snapshots)

        if self.use_attn_residuals:
            h = _attn_res_mix(
                h, snapshots,
                self.output_attn_res_proj.weight,
                self.output_attn_res_norm.weight, self.args.rms_norm_eps,
            )
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        _log_once()
        self.args = args
        self.model_type = args.model_type
        self.model = KimiK3Model(args)
        self.lm_head = None if args.tie_word_embeddings else nn.Linear(
            args.hidden_size, args.vocab_size, bias=False
        )

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        out = self.model(inputs, cache)
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            ArraysCache(size=4) if layer.is_linear else KVCache()
            for layer in self.layers
        ]

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return remap_checkpoint(weights, self.args, self.layers)

    @property
    def cast_predicate(self):
        keep_fp32 = ("A_log", "dt_bias", "e_score_correction_bias")

        def predicate(path: str) -> bool:
            return not any(path.endswith(s) or s in path for s in keep_fp32)

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path: str, _module) -> bool:
            # MXFP4 только у routed experts (R3-скоуп чекпоинта).
            return "switch_mlp" in path

        return predicate


# ---------------------------------------------------------------------------
# remap: ЕДИНСТВЕННЫЙ источник соответствия имён чекпоинта модулю
# ---------------------------------------------------------------------------

_WRAPPER_PREFIX = "language_model."
_DROP_MARKERS = ("vision_tower", "mm_projector", "multi_modal")
_CONV_RENAMES = (
    (".q_conv1d.weight", ".q_conv.conv.weight"),
    (".k_conv1d.weight", ".k_conv.conv.weight"),
    (".v_conv1d.weight", ".v_conv.conv.weight"),
)


def remap_checkpoint(
    weights: Dict[str, mx.array], args: ModelArgs, layers: List[KimiK3DecoderLayer]
) -> Dict[str, mx.array]:
    out: Dict[str, mx.array] = {}
    for k, v in weights.items():
        if k.startswith(_WRAPPER_PREFIX):
            k = k[len(_WRAPPER_PREFIX):]
        if any(m in k for m in _DROP_MARKERS) or ".mtp" in k or k.startswith("model.mtp"):
            continue
        if k.endswith((".weight_packed", ".weight_scale")):
            raise RuntimeError(
                "kimi_k3: сырой compressed-tensors MXFP4 чекпоинт. Прогони offline-"
                "конвертер (B2) — пин mlx-lm интерпретирует compressed-tensors как "
                "affine и это числовой мусор."
            )
        for src, dst in _CONV_RENAMES:
            if k.endswith(src):
                k = k[: -len(src)] + dst
                if v.ndim == 3:
                    v = v.moveaxis(2, 1)  # torch [C,1,K] -> mlx [C,K,1]
                break
        if k.endswith("A_log"):
            n = v.reshape(-1).shape[0]
            H, D = args.kda_num_heads, args.kda_head_dim
            if ALOG_SEMANTICS == "per_head":
                if n == D and n != H:
                    # Чекпоинт хранит [128] при 96 головах; официальный код
                    # объявляет Parameter([num_heads]) => срез [:H]
                    # (llama.cpp-parity 6.7e-05). P0-008b: контроль per_channel.
                    v = v.reshape(-1)[:H]
                elif n != H:
                    raise ValueError(f"kimi_k3: неожиданная форма A_log {v.shape}")
                else:
                    v = v.reshape(-1)
            else:
                if n < D:
                    raise ValueError(f"kimi_k3: A_log {v.shape} < head_dim при per_channel")
                v = v.reshape(-1)[:D]
        if k.endswith("dt_bias") and v.ndim > 1:
            v = v.reshape(-1)
        out[k] = v

    # Стекинг сырых bf16-экспертов (parity/тесты; у конвертированного репо
    # уже стековые имена switch_mlp.*)
    for li, layer in enumerate(layers):
        if not getattr(layer, "is_moe", False):
            continue
        moe_p = f"model.layers.{li}.block_sparse_moe"
        for src, dst in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
            k0 = f"{moe_p}.experts.0.{src}.weight"
            if k0 in out:
                stacked = [
                    out.pop(f"{moe_p}.experts.{i}.{src}.weight")
                    for i in range(args.num_experts)
                ]
                out[f"{moe_p}.switch_mlp.{dst}.weight"] = mx.stack(stacked)

    # kv_b absorption -> embed_q / unembed_out (bf16 в K3: self_attn вне квант-скоупа)
    for li, layer in enumerate(layers):
        ap = f"model.layers.{li}.self_attn"
        kv_b_key = f"{ap}.kv_b_proj.weight"
        if kv_b_key not in out:
            continue
        nope, vh = args.qk_nope_head_dim, args.v_head_dim
        H = args.num_attention_heads
        w = out.pop(kv_b_key).reshape(H, nope + vh, -1)
        out[f"{ap}.embed_q.weight"] = mx.contiguous(w[:, :nope, :].swapaxes(-1, -2))
        out[f"{ap}.unembed_out.weight"] = mx.contiguous(w[:, nope:, :])

    return out
