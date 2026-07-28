# Kimi K3 (moonshotai/Kimi-K3) — text-only MLX implementation for EXO.
#
# Vendored into EXO (vendor+register pattern, как minimax_m3_vl / mimo_v2):
# регистрируется в sys.modules["mlx_lm.models.kimi_k3"] через
# custom_models.register_custom_models(). mlx-lm пин НЕ трогаем.
#
# Архитектура (план kimi-k3-exo-implementation-baseline v2, §2.1/§7.2/§8):
#   93 слоя = 1 dense-MLP (layer 1, 1-based) + LatentMoE (896 exp, top-16,
#   2 shared, latent 3584, expert intermediate 3072);
#   attention: 69 KDA (linear attention, delta rule) + 24 Gated MLA (NoPE!)
#   на 1-based позициях 4,8,...,92,93; hidden 7168; 96 heads x 128;
#   AttnRes (softmax-mixture блочных резидуалов, block 12, K_max=8, + финальное
#   применение на выходе); SiTU-GLU активация; vocab 163840; MXFP4 QAT gs32
#   (только routed experts; всё остальное bf16).
#
# ИНВАРИАНТЫ (обязательные к соблюдению, см. план §2.2):
#   INV#1  A_log грузится как [head_dim]=[128]; НИКОГДА не режется по головам
#          при шардинге. СЕМАНТИКА ПОТРЕБЛЕНИЯ ОСПОРЕНА (P0-008b):
#            per_channel  (pipenetwork: broadcast по каналам, shared heads)
#            per_head     (llama.cpp PR#26185: -exp(A_log[:num_heads]),
#                          прошёл logit-parity vs Moonshot 6.7e-05)
#          Переключатель: EXO_K3_ALOG_SEMANTICS = per_head|per_channel,
#          default per_head (единственная интерпретация с torch-parity).
#   INV#2  Full-rank KDA gate = единый g_proj (hidden -> 96*128).
#          g_a_proj/g_b_proj НЕ существуют.
#   INV#3/#11  LatentMoE: ровно ОДИН routed all_sum, ДО routed RMSNorm.
#   INV#12 fp32-cast вокруг всех distributed all_sum (паттерн F2).
#   INV#13 Router matmul в fp32 (паттерн F6). KDA-гейт: форма ЗАВИСИТ от
#          gate_lower_bound — это НЕ clamp, а смена активации:
#            unset      -> log_g = -exp(A_log) * softplus(x)      (kimi_linear)
#            K3 (=-5.0) -> log_g = lower_bound * sigmoid(exp(A_log) * x)
#   INV#14 SiTU-GLU и AttnRes score-путь считаются в fp32.
#
# AUDIT-МАРКЕРЫ (закрываются на A0/A2 по реальному чекпоинту и
# официальному modeling_kimi_k3.py; все имена тензоров централизованы в
# remap_checkpoint() — правится ОДНО место):
#   AUDIT-SITU    точная формула SiTU (betas 4.0/25.0) — реконструкция ниже.
#   AUDIT-DTBIAS  место dt_bias в sigmoid-форме гейта (default: внутри).
#   AUDIT-ATTNRES имена тензоров AttnRes + точка финального применения.
#   AUDIT-QKNORM  наличие весов q/k-норм KDA (default: weightless, как в
#                 kimi_linear).
#   AUDIT-NAMES   имена MLA (kv_a_proj_with_mqa vs kv_a_proj), latent
#                 down/up/norm, shared intermediate.

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


# ---------------------------------------------------------------------------
# env-переключатели неразрешённых семантик (см. AUDIT выше)
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


ALOG_SEMANTICS = _env("EXO_K3_ALOG_SEMANTICS", "per_head")  # per_head|per_channel
DTBIAS_IN_GATE = _env("EXO_K3_DTBIAS_IN_GATE", "1") == "1"
SITU_FORM = _env("EXO_K3_SITU_FORM", "softcap_swish")  # softcap_swish|swish
FINAL_ATTN_RES = _env("EXO_K3_FINAL_ATTN_RES", "1") == "1"

_logged_semantics = False


def _log_semantics_once() -> None:
    global _logged_semantics
    if _logged_semantics:
        return
    _logged_semantics = True
    print(
        f"[kimi_k3] A_log semantics = {ALOG_SEMANTICS} "
        f"(P0-008b unresolved; per_head = llama.cpp-parity evidence), "
        f"dt_bias_in_gate = {DTBIAS_IN_GATE}, situ_form = {SITU_FORM}"
    )


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
    head_dim: int = 128
    intermediate_size: int = 33792  # dense MLP (layer 1, 1-based)
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 1048576

    # KDA / linear attention. kda_layers — 1-based список (как в kimi_linear).
    linear_attn_config: Dict[str, Any] = field(default_factory=dict)

    # MLA (NoPE: qk_rope_head_dim == 0, RoPE ветви НЕТ — ревью R1)
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 0
    v_head_dim: int = 128

    # LatentMoE
    num_experts: int = 896
    num_experts_per_token: int = 16
    num_shared_experts: int = 2
    moe_intermediate_size: int = 3072
    moe_latent_dim: int = 3584
    shared_expert_intermediate_size: Optional[int] = None  # AUDIT-NAMES; default = moe_intermediate_size * num_shared_experts
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    num_expert_group: int = 1
    topk_group: int = 1

    # SiTU-GLU
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0

    # AttnRes
    attn_res_block_size: int = 12

    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        # Официальный конфиг nested: top-level kimi_k3 -> text_config.
        # Принимаем и nested, и плоский.
        if "text_config" in params:
            merged = dict(params["text_config"])
            merged.setdefault("model_type", "kimi_k3")
            params = merged
        return super().from_dict(params)

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
        return int(self.linear_attn_config.get("head_dim", self.head_dim))

    @property
    def conv_kernel(self) -> int:
        return int(
            self.linear_attn_config.get(
                "short_conv_kernel_size",
                self.linear_attn_config.get("short_conv_kernel", 4),
            )
        )


# ---------------------------------------------------------------------------
# SiTU-GLU (AUDIT-SITU: формула — реконструкция; tiny-parity фальсифицирует)
# ---------------------------------------------------------------------------

def situ(x: mx.array, beta: float, linear_beta: float) -> mx.array:
    """SiTU activation, считается в fp32 (INV#14).

    softcap_swish (default): linear_beta * tanh(x * sigmoid(beta*x) / linear_beta)
      - около нуля ~= swish(beta*x); большие |x| мягко капятся на +-linear_beta
        (tanh-сатурация, видимая в bf16 — наблюдение pipenetwork).
    swish (fallback): x * sigmoid(beta * x).
    """
    x32 = x.astype(mx.float32)
    sw = x32 * mx.sigmoid(beta * x32)
    if SITU_FORM == "swish":
        return sw.astype(x.dtype)
    out = linear_beta * mx.tanh(sw / linear_beta)
    return out.astype(x.dtype)


class SiTUGLUAct:
    """activation-объект для SwitchGLU: act(gate, up) -> situ(gate) * up."""

    def __init__(self, beta: float, linear_beta: float):
        self.beta = beta
        self.linear_beta = linear_beta

    def __call__(self, gate: mx.array, up: mx.array) -> mx.array:
        return situ(gate, self.beta, self.linear_beta) * up


class KimiK3MLP(nn.Module):
    """Dense MLP (layer 1) и shared experts: gate/up/down + SiTU-GLU."""

    def __init__(self, args: ModelArgs, dim: Optional[int] = None, hidden: Optional[int] = None):
        super().__init__()
        d = dim or args.hidden_size
        h = hidden or args.intermediate_size
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)
        self._beta = args.situ_beta
        self._linear_beta = args.situ_linear_beta

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            situ(self.gate_proj(x), self._beta, self._linear_beta) * self.up_proj(x)
        )


# ---------------------------------------------------------------------------
# short conv (копия паттерна kimi_linear.ShortConv1d — локально, т.к. TP режет
# каналы и conv-state)
# ---------------------------------------------------------------------------

class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, bias=False
        )

    def __call__(
        self,
        x: mx.array,
        state: Optional[mx.array],
        mask: Optional[mx.array],
        lengths: Optional[mx.array],
    ) -> Tuple[mx.array, mx.array]:
        B, T, C = x.shape
        if state is None:
            state = mx.zeros((B, self.kernel_size - 1, C), dtype=x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        out = nn.silu(self.conv(conv_input))
        n_keep = self.kernel_size - 1
        if lengths is not None:
            positions = lengths[:, None, None] + mx.arange(n_keep)[None, :, None]
            positions = mx.broadcast_to(positions, (B, n_keep, C))
            new_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_state = mx.contiguous(conv_input[:, -n_keep:, :])
        return out, new_state


# ---------------------------------------------------------------------------
# KDA — Kimi Delta Attention (69 слоёв)
# ---------------------------------------------------------------------------

def compute_g_k3(
    A_log: mx.array,
    a: mx.array,
    dt_bias: mx.array,
    lower_bound: float,
    num_heads: int,
    head_dim: int,
) -> mx.array:
    """K3-форма decay-гейта (INV#13): g = exp(lower_bound * sigmoid(exp(A)*x)).

    a: [B, T, H, D] (выход f_b_proj, решейпнутый); dt_bias: [H, D].
    A_log приходит [head_dim] (как в шардах). Broadcast по семантике:
      per_channel: A_log -> [1,1,1,D]  (shared across heads)
      per_head:    A_log[:num_heads] -> [1,1,H,1]
    Всё в fp32; возвращает g [B,T,H,D] fp32 (vectorized gating для
    gated_delta_ops).
    """
    a32 = a.astype(mx.float32)
    if DTBIAS_IN_GATE:  # AUDIT-DTBIAS
        a32 = a32 + dt_bias.astype(mx.float32)
    A = A_log.astype(mx.float32)
    if ALOG_SEMANTICS == "per_channel":
        A = A.reshape(1, 1, 1, head_dim)
    else:  # per_head
        A = A[:num_heads].reshape(1, 1, num_heads, 1)
    log_g = lower_bound * mx.sigmoid(mx.exp(A) * a32)
    return mx.exp(log_g)


def compute_g_softplus(
    A_log: mx.array,
    a: mx.array,
    dt_bias: mx.array,
    num_heads: int,
    head_dim: int,
) -> mx.array:
    """kimi_linear-форма (gate_lower_bound unset): g = exp(-exp(A)*softplus(x+dt))."""
    a32 = a.astype(mx.float32) + dt_bias.astype(mx.float32)
    A = A_log.astype(mx.float32)
    if ALOG_SEMANTICS == "per_channel":
        A = A.reshape(1, 1, 1, head_dim)
    else:
        A = A[:num_heads].reshape(1, 1, num_heads, 1)
    return mx.exp(-mx.exp(A) * nn.softplus(a32))


class KimiK3DeltaAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = args.kda_num_heads
        self.head_dim = args.kda_head_dim
        self.conv_kernel = args.conv_kernel
        self.projection_dim = self.num_heads * self.head_dim
        self.gate_lower_bound = args.gate_lower_bound
        hidden = args.hidden_size
        self.scale = float(self.head_dim) ** -0.5

        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)

        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)

        # low-rank вход decay-гейта (f) — как в kimi_linear
        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        # beta для delta rule
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        # INV#2: full-rank output gate — ЕДИНЫЙ g_proj, никаких g_a/g_b
        self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)

        # A_log грузится [head_dim]=[128] (см. INV#1); dt_bias [projection_dim]
        self.A_log = mx.zeros((self.head_dim,))
        self.dt_bias = mx.zeros((self.projection_dim,))

        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        _log_semantics_once()
        B, T, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, ssm_state = cache
            lengths = cache.lengths
        else:
            q_state = k_state = v_state = ssm_state = None
            lengths = None

        if q_state is None:
            s = mx.zeros((B, self.conv_kernel - 1, self.projection_dim), dtype=dtype)
            q_state, k_state, v_state = s, s, s

        q_conv, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k_conv, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v_conv, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)

        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        H, D = self.num_heads, self.head_dim
        q = q_conv.reshape(B, T, H, D)
        k = k_conv.reshape(B, T, H, D)
        v = v_conv.reshape(B, T, H, D)

        # q/k нормализация — как в kimi_linear (weightless). AUDIT-QKNORM.
        inv_scale = self.scale
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        a_logits = self.f_b_proj(self.f_a_proj(x)).reshape(B, T, H, D)
        b_logits = self.b_proj(x).reshape(B, T, H)

        dt = self.dt_bias.reshape(H, D)
        if self.gate_lower_bound is not None:
            g = compute_g_k3(self.A_log, a_logits, dt, self.gate_lower_bound, H, D)
        else:
            g = compute_g_softplus(self.A_log, a_logits, dt, H, D)
        beta = mx.sigmoid(b_logits.astype(mx.float32))

        # gated_delta_ops: pure-ops путь с ПРЕДВЫЧИСЛЕННЫМ vectorized g —
        # fused-кернел gated_delta_kernel НЕ используется: он хардкодит
        # softplus-форму гейта (неверную для K3). Кернел-адаптация = P1.
        out, ssm_state = gated_delta_ops(q, k, v, g, beta, state=ssm_state, mask=mask)

        if cache is not None:
            cache[3] = ssm_state
            cache.advance(T)

        gate = self.g_proj(x).reshape(B, T, H, D)  # INV#2 full-rank
        out = (
            self.o_norm(out.astype(dtype).reshape(B, T, H, D)) * mx.sigmoid(gate)
        ).reshape(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Gated MLA (NoPE) — 24 слоя
# ---------------------------------------------------------------------------

class KimiK3MLAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.qk_rope_head_dim == 0, (
            "K3 MLA = NoPE (ревью R1): qk_rope_head_dim должен быть 0; "
            f"получено {args.qk_rope_head_dim} — проверь конфиг/A0"
        )
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = float(self.qk_nope_head_dim) ** -0.5
        hidden = args.hidden_size

        # q-LoRA
        self.q_a_proj = nn.Linear(hidden, args.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False
        )

        # kv-LoRA: 512, БЕЗ rope-хвоста. Имя модуля = имя в чекпоинте
        # (kv_a_proj_with_mqa — deepseek-наследие; AUDIT-NAMES, remap примет оба).
        self.kv_a_proj_with_mqa = nn.Linear(hidden, self.kv_lora_rank, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)

        # absorption (из kv_b_proj в remap): embed_q / unembed_out
        self.embed_q = MultiLinear(self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(self.kv_lora_rank, self.v_head_dim, self.num_heads)

        # выходной гейт: out * sigmoid(g_proj(x)) -> o_proj
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
        q_nope = q.reshape(B, L, self.num_heads, self.qk_nope_head_dim).transpose(0, 2, 1, 3)

        kv_latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        kv_latent = mx.expand_dims(kv_latent, axis=1)  # [B, 1, L, 512]

        if cache is not None:
            # KVCache ждёт пару (k, v); latent играет обе роли — храним один
            # тензор, второй слот нулевой длины не нужен: кладём latent как k
            # и как v (references, память одна и та же до copy-on-write).
            kv_latent, _ = cache.update_and_fetch(kv_latent, kv_latent)

        if L == 1:
            # decode: absorb q в latent-пространство
            q_abs = self.embed_q(q_nope)                 # [B, H, 1, 512]
            k = v = kv_latent                            # [B, 1, S, 512]
            out = scaled_dot_product_attention(
                q_abs, k, v, cache=cache, scale=self.scale, mask=mask
            )
            out = self.unembed_out(out)                  # [B, H, 1, v_dim]
        else:
            # prefill: материализуем k/v per-head из latent
            k = self.embed_q(kv_latent, transpose=False)  # [B, H, S, nope]
            v = self.unembed_out(kv_latent)               # [B, H, S, v_dim]
            out = scaled_dot_product_attention(
                q_nope, k, v, cache=cache, scale=self.scale, mask=mask
            )

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        gate = self.g_proj(x)
        out = out * mx.sigmoid(gate)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# router select (копия семантики kimi_linear._group_expert_select, без compile
# для читаемости; fp32 по INV#13 обеспечивается кастом на входе)
# ---------------------------------------------------------------------------

def group_expert_select(
    gates: mx.array,
    bias: Optional[mx.array],
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
    score_function: str,
) -> Tuple[mx.array, mx.array]:
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates)
    elif score_function == "softmax":
        scores = mx.softmax(gates, axis=-1, precise=True)
    else:
        raise ValueError(f"Unsupported MoE router activation '{score_function}'")

    orig_scores = scores
    if bias is not None:
        scores = scores + bias.astype(scores.dtype)

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(0.0, dtype=scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and renormalize:
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)

    return inds, scores * routed_scaling_factor


# ---------------------------------------------------------------------------
# LatentMoE (INV#3/#11/#12/#13)
# ---------------------------------------------------------------------------

class KimiK3LatentMoE(nn.Module):
    """Stable LatentMoE.

    x(7168) --down--> latent(3584) --SwitchGLU(3584->3072->3584, SiTU)-->
    weighted top-16 sum --> [fp32 ALL_SUM при TP, ДО нормы] -->
    RMSNorm(3584) --up--> 7168; плюс shared experts от исходного x.

    TP-контракт (KimiK3ShardingStrategy):
      - router/gate + e_score_correction_bias: replicated;
      - routed_expert_down_proj / routed_expert_norm / routed_expert_up_proj:
        replicated (v1);
      - switch_mlp gate/up: in-place shard по intermediate (3072 -> 3072/N);
      - switch_mlp down: in-place shard (partial latent, БЕЗ collective);
      - shared_experts gate/up in-place a2s, down in-place s2a (partial);
      - strategy выставляет self.sharding_group; ВСЕ collectives живут ЗДЕСЬ.
    Никакого generic ShardedMoE поверх (INV#3).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        latent = args.moe_latent_dim
        experts = args.num_experts

        self.gate = nn.Linear(hidden, experts, bias=False)  # router; matmul fp32
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)

        self.routed_expert_down_proj = nn.Linear(hidden, latent, bias=False)
        self.switch_mlp = SwitchGLU(
            latent,
            args.moe_intermediate_size,
            experts,
            activation=SiTUGLUAct(args.situ_beta, args.situ_linear_beta),
        )
        self.routed_expert_norm = nn.RMSNorm(latent, eps=args.rms_norm_eps)
        self.routed_expert_up_proj = nn.Linear(latent, hidden, bias=False)

        shared_hidden = args.shared_expert_intermediate_size or (
            args.moe_intermediate_size * args.num_shared_experts
        )
        if args.num_shared_experts:
            self.shared_experts = KimiK3MLP(args, dim=hidden, hidden=shared_hidden)
        else:
            self.shared_experts = None

        self.sharding_group: Optional[mx.distributed.Group] = None

    def _all_sum_fp32(self, t: mx.array) -> mx.array:
        # INV#12: bf16 distributed reduce = corruption-класс бага (паттерн F2)
        g = self.sharding_group
        if g is None:
            return t
        dt = t.dtype
        return mx.distributed.all_sum(t.astype(mx.float32), group=g).astype(dt)

    def __call__(self, x: mx.array) -> mx.array:
        # INV#13 (F6): router matmul в fp32 независимо от dtype весов
        router_scores = x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T
        inds, weights = group_expert_select(
            router_scores,
            self.e_score_correction_bias,
            self.args.num_experts_per_token,
            self.args.num_expert_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.moe_renormalize,
            self.args.moe_router_activation_func,
        )

        latent_in = self.routed_expert_down_proj(x)
        expert_out = self.switch_mlp(latent_in, inds)          # [..., top_k, latent]
        routed = (expert_out * weights[..., None].astype(expert_out.dtype)).sum(axis=-2)

        # INV#3/#11: ровно ОДИН routed reduction, ДО RMSNorm
        routed = self._all_sum_fp32(routed)
        routed = self.routed_expert_up_proj(self.routed_expert_norm(routed))

        if self.shared_experts is not None:
            shared = self.shared_experts(x)      # partial при TP (down = in-place s2a)
            shared = self._all_sum_fp32(shared)  # второй collectив слоя; P1: фьюз
            routed = routed + shared
        return routed


# ---------------------------------------------------------------------------
# AttnRes (AUDIT-ATTNRES)
# ---------------------------------------------------------------------------

class AttnRes(nn.Module):
    """Softmax-mixture блочных резидуалов.

    На границах блоков (layer_idx % block_size == 0: слои 0,12,...,84 при 93/12)
    и один раз на выходе модели: candidates = stack(residuals + [h]) (N,K+1,H),
    score_k = <candidate_k, w_slot> в fp32 (w — fused norm x proj вектор из
    чекпоинта), softmax по K+1, h' = сумма. h' добавляется в стек резидуалов.

    Параметр: proj (n_slots, hidden) fp32; n_slots = n_boundaries (+1 финальный
    при FINAL_ATTN_RES). Слот i = i-я граница; последний слот = финальное
    применение. Точные имена тензоров чекпоинта -> remap_checkpoint().
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        n_boundaries = (
            args.num_hidden_layers + args.attn_res_block_size - 1
        ) // args.attn_res_block_size
        self.n_boundaries = n_boundaries
        n_slots = n_boundaries + (1 if FINAL_ATTN_RES else 0)
        self.proj = mx.zeros((n_slots, args.hidden_size))

    def mix(self, h: mx.array, residuals: List[mx.array], slot: int) -> mx.array:
        w = self.proj[slot].astype(mx.float32)              # (H,)
        cands = residuals + [h]
        stack = mx.stack([c.astype(mx.float32) for c in cands], axis=-2)  # (..., K+1, H)
        scores = (stack * w).sum(axis=-1)                   # (..., K+1) fp32 (INV#14)
        p = mx.softmax(scores, axis=-1, precise=True)
        mixed = (p[..., None] * stack).sum(axis=-2)
        return mixed.astype(h.dtype)


# ---------------------------------------------------------------------------
# decoder layer / model
# ---------------------------------------------------------------------------

class KimiK3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # 1-based списки в конфиге (план §2.1)
        self.is_linear = (layer_idx + 1) in args.kda_layers

        if self.is_linear:
            self.self_attn = KimiK3DeltaAttention(args, layer_idx)
        else:
            self.self_attn = KimiK3MLAAttention(args)

        if (
            args.num_experts > 0
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        ):
            self.block_sparse_moe = KimiK3LatentMoE(args)
            self.mlp = None
            self.is_moe = True
        else:
            self.block_sparse_moe = None
            self.mlp = KimiK3MLP(args)
            self.is_moe = False

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        y = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + y
        ffn = self.block_sparse_moe if self.is_moe else self.mlp
        z = ffn(self.post_attention_layernorm(h))
        return h + z


class KimiK3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            KimiK3DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn_res = AttnRes(args)
        self.block_size = args.attn_res_block_size

        kda_layers = args.kda_layers
        self.ssm_idx = kda_layers[0] - 1 if kda_layers else 0
        self.attn_idx = 0
        for i in range(len(self.layers)):
            if (i + 1) not in kda_layers:
                self.attn_idx = i
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        attn_mask = create_attention_mask(h, cache[self.attn_idx], return_array=True)

        # AttnRes state — per-forward workspace (между chunk'ами НЕ персистится:
        # cross-token информация течёт только через KDA state / MLA KV; ревью R13)
        residuals: List[mx.array] = []
        slot = 0

        for i, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            if i % self.block_size == 0:
                h = self.attn_res.mix(h, residuals, slot)
                residuals.append(h)
                slot += 1
            mask = ssm_mask if layer.is_linear else attn_mask
            h = layer(h, mask=mask, cache=layer_cache)

        if FINAL_ATTN_RES:
            # AUDIT-ATTNRES: финальное применение — на выходе последнего слоя,
            # ДО финальной нормы (default; официальный код может отличаться)
            h = self.attn_res.mix(h, residuals, slot)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        # layout: Model.model (get_inner_model в exo проверяет `model` первым)
        self.model = KimiK3Model(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            caches.append(ArraysCache(size=4) if layer.is_linear else KVCache())
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return remap_checkpoint(weights, self.args, self.layers)

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            if "attn_res" in path:  # INV#14: score-путь fp32
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path: str, _):
            # роутер, A_log/dt_bias, нормы, AttnRes — никогда не квантуются
            if path.endswith(".gate") or "attn_res" in path:
                return False
            return True

        return predicate


# ---------------------------------------------------------------------------
# remap_checkpoint — ЕДИНСТВЕННЫЙ источник правды по именам тензоров
# (используется Model.sanitize и (позже) streaming-конвертером; правки имён
# после A0 — только здесь)
# ---------------------------------------------------------------------------

def remap_checkpoint(
    weights: Dict[str, mx.array],
    args: ModelArgs,
    layers: List[KimiK3DecoderLayer],
) -> Dict[str, mx.array]:
    # 0) официальный multimodal-wrapper: text tower под language_model.*
    if any(k.startswith("language_model.") for k in weights):
        weights = {
            (k[len("language_model."):] if k.startswith("language_model.") else k): v
            for k, v in weights.items()
        }

    # 1) выкинуть vision / mm_projector / mtp
    drop_prefixes = ("vision_tower", "mm_projector", "model.mtp", "mtp")
    weights = {
        k: v for k, v in weights.items() if not k.startswith(drop_prefixes)
    }

    if args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    num_experts = args.num_experts

    for layer_idx, layer in enumerate(layers):
        prefix = f"model.layers.{layer_idx}"
        attn_prefix = f"{prefix}.self_attn"

        # --- MoE: expert stacking + латентные проекции -----------------------
        if layer.is_moe:
            src_prefix = f"{prefix}.block_sparse_moe"
            # 1a) routed experts w1/w2/w3 -> switch_mlp.{gate,down,up}
            #     (для MXFP4-чекпоинта здесь также приходят .weight_packed /
            #      .weight_scale — стекуются той же логикой по суффиксам)
            for src, dst in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
                for suffix in ("weight", "weight_packed", "weight_scale", "scales", "biases"):
                    key0 = f"{src_prefix}.experts.0.{src}.{suffix}"
                    if key0 in weights:
                        stacked = [
                            weights.pop(f"{src_prefix}.experts.{i}.{src}.{suffix}")
                            for i in range(num_experts)
                        ]
                        weights[f"{src_prefix}.switch_mlp.{dst}.{suffix}"] = mx.stack(stacked)

            # 1b) латентные проекции / норма — AUDIT-NAMES (варианты имён)
            latent_alias = {
                "routed_expert_down_proj": ("routed_expert_down_proj", "down_proj", "latent_down_proj"),
                "routed_expert_up_proj": ("routed_expert_up_proj", "up_proj", "latent_up_proj"),
                "routed_expert_norm": ("routed_expert_norm", "latent_norm", "expert_norm"),
            }
            for dst, srcs in latent_alias.items():
                for s in srcs:
                    key = f"{src_prefix}.{s}.weight"
                    if key in weights and s != dst:
                        weights[f"{src_prefix}.{dst}.weight"] = weights.pop(key)
                        break

            # 1c) router bias name
            for bias_src in ("gate.e_score_correction_bias", "e_score_correction_bias"):
                key = f"{src_prefix}.{bias_src}"
                if key in weights and bias_src != "e_score_correction_bias":
                    weights[f"{src_prefix}.e_score_correction_bias"] = weights.pop(key)

        # --- KDA -------------------------------------------------------------
        if layer.is_linear:
            for src_name, dst_name in (
                ("q_conv1d", "q_conv"), ("k_conv1d", "k_conv"), ("v_conv1d", "v_conv"),
            ):
                src_key = f"{attn_prefix}.{src_name}.weight"
                if src_key in weights:
                    w = weights.pop(src_key)
                    if w.ndim == 3:
                        w = w.moveaxis(2, 1)
                    weights[f"{attn_prefix}.{dst_name}.conv.weight"] = w
            dt_key = f"{attn_prefix}.dt_bias"
            if dt_key in weights and weights[dt_key].ndim > 1:
                weights[dt_key] = mx.reshape(weights[dt_key], (-1,))
            # INV#1: A_log приходит [128]; форму НЕ трогаем, семантика — runtime
            a_key = f"{attn_prefix}.A_log"
            if a_key in weights and weights[a_key].ndim > 1:
                weights[a_key] = mx.reshape(weights[a_key], (-1,))

        # --- MLA: имена kv_a + kv_b absorption -------------------------------
        else:
            alt = f"{attn_prefix}.kv_a_proj.weight"
            if alt in weights:  # AUDIT-NAMES: NoPE-имя без _with_mqa
                weights[f"{attn_prefix}.kv_a_proj_with_mqa.weight"] = weights.pop(alt)

            kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
            if kv_b_key in weights:
                qk_nope = args.qk_nope_head_dim
                v_head = args.v_head_dim
                head_dim = qk_nope + v_head
                num_heads = args.num_attention_heads

                quantized = f"{attn_prefix}.kv_b_proj.scales" in weights
                v = weights.pop(kv_b_key)
                if quantized:
                    dims = args.kv_lora_rank
                    scales = weights.pop(f"{attn_prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{attn_prefix}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
                wv = mx.contiguous(v[:, qk_nope:, :])
                if quantized:
                    wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                    wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                    weights[f"{attn_prefix}.embed_q.scales"] = wk_s
                    weights[f"{attn_prefix}.embed_q.biases"] = wk_b
                    weights[f"{attn_prefix}.unembed_out.scales"] = wv_s
                    weights[f"{attn_prefix}.unembed_out.biases"] = wv_b
                weights[f"{attn_prefix}.embed_q.weight"] = wk
                weights[f"{attn_prefix}.unembed_out.weight"] = wv

    # --- AttnRes: собрать per-boundary векторы в model.attn_res.proj ---------
    # AUDIT-ATTNRES: имена в чекпоинте неизвестны до A0. Принимаем варианты:
    #   model.attn_res.proj                       (уже стек)
    #   model.attn_res_proj.{i}.weight            (по слотам)
    #   model.layers.{L}.attn_res_proj.weight     (на boundary-слоях, L=0,12,..)
    #   model.attn_res_final.weight / final_attn_res.weight (финальный слот)
    dst_key = "model.attn_res.proj"
    if dst_key not in weights:
        n_boundaries = (args.num_hidden_layers + args.attn_res_block_size - 1) // args.attn_res_block_size
        slots: List[Optional[mx.array]] = [None] * (n_boundaries + (1 if FINAL_ATTN_RES else 0))
        found = False
        for i in range(n_boundaries):
            for cand in (
                f"model.attn_res_proj.{i}.weight",
                f"model.attn_res.{i}.weight",
                f"model.layers.{i * args.attn_res_block_size}.attn_res_proj.weight",
            ):
                if cand in weights:
                    slots[i] = mx.reshape(weights.pop(cand), (-1,))
                    found = True
                    break
        if FINAL_ATTN_RES:
            for cand in ("model.attn_res_final.weight", "model.final_attn_res.weight",
                         "model.attn_res_proj.final.weight"):
                if cand in weights:
                    slots[-1] = mx.reshape(weights.pop(cand), (-1,))
                    found = True
                    break
        if found:
            H = args.hidden_size
            weights[dst_key] = mx.stack(
                [s if s is not None else mx.zeros((H,)) for s in slots]
            )
        # если не found — загрузка упадёт на missing key model.attn_res.proj:
        # это ЖЕЛАЕМОЕ поведение (лучше громкий fail, чем нулевой AttnRes)

    return weights
