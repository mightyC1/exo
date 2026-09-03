"""Runtime GLM-5.2 IndexShare support for mlx-lm's DeepSeekV3.2 path.

GLM-5.2 uses the ``glm_moe_dsa`` architecture plus an ``indexer_types`` list.
Most sparse-attention layers are ``"shared"`` and should reuse the top-k token
indices computed by the nearest previous ``"full"`` indexer layer. Some MLX
conversions materialize local indexers on shared layers; running those local
indexers is not equivalent and can corrupt output once DSA becomes active after
``index_topk`` cumulative tokens.

This module monkey-patches the loaded mlx-lm classes at runtime. Keeping it in
Exo lets a patched Exo tree be tested without publishing a separate mlx-lm fork.
"""

from __future__ import annotations

import contextvars
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx

from exo.worker.runner.bootstrap import logger as default_logger

from .glm52_prefill import (
    GLM52PrefillConfig,
    PrefillProfileRecorder,
    UnsupportedPrefillMask,
    cache_requires_dense_prefill,
    profiled_monolithic_indexer_call,
    read_prefill_config,
    set_cache_growth,
    shared_cache_placeholder_width,
    sparse_prefill_attention,
    tiled_indexer_call,
)

_INDEXSHARE_CTX: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "exo_glm52_indexshare_ctx",
    default=None,
)
_MISSING = object()
_CLASSES_PATCHED = False
_DEFAULT_PREFILL_CFG = GLM52PrefillConfig()

# Query lengths up to this ride the sparse gather path whenever the indexer
# produced top-k, bypassing the sparse_min_kv prefill-economics threshold —
# but only inside an explicit MTP verify context (see mtp_verify_context):
# the W4 gate for ordinary prefill/decode is untouched (ТЗ invariant).
_MICRO_DECODE_L = 4
_MTP_VERIFY_CTX: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "exo_glm52_mtp_verify", default=False
)


_MOE_SORT_MIN: contextvars.ContextVar[int] = contextvars.ContextVar("exo_glm52_moe_sort_min", default=64)
_MOE_SORT_PATCHED = False


def install_moe_sort_override(sort_min: int, logger: Any = default_logger) -> bool:
    """Experiment (audit #3 §14): the pinned SwitchGLU sorts token-expert pairs
    only when there are >= 64 of them, so an L=2..4 verify (16-32 pairs)
    gathers experts unsorted. Inside mtp_verify_context() the threshold
    becomes `sort_min`; everything else keeps the pin's 64."""
    global _MOE_SORT_PATCHED
    if _MOE_SORT_PATCHED:
        return True
    try:
        from mlx_lm.models import switch_layers as sl
    except Exception:
        logger.warning("[EXO][GLM-5.2] switch_layers unavailable; MoE sort override not installed")
        return False
    orig = sl.SwitchGLU.__call__

    def _call(self, x, indices):
        thr = _MOE_SORT_MIN.get()
        if thr == 64 or not _MTP_VERIFY_CTX.get():
            return orig(self, x, indices)
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= thr
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = sl._gather_sort(x, indices)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        if do_sort:
            x = sl._scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

    sl.SwitchGLU.__call__ = _call
    _MOE_SORT_MIN.set(int(sort_min))
    _MOE_SORT_PATCHED = True
    logger.info(f"[EXO][GLM-5.2] MoE sort override installed: sort_min={sort_min} (verify context only)")
    return True


class mtp_verify_context:
    """Marks the enclosed model call as an MTP verify forward (B==1, L<=4)."""

    def __enter__(self):
        self._token = _MTP_VERIFY_CTX.set(True)
        return self

    def __exit__(self, *exc):
        _MTP_VERIFY_CTX.reset(self._token)
        return False


def _env_raw(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return None


def _env_bool(*names: str, default: bool = False) -> bool:
    value = _env_raw(*names)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on", "auto"}:
        return True
    if value in {"0", "false", "no", "n", "off", "none", "disabled"}:
        return False
    return default


def _env_choice(*names: str, default: str, allowed: set[str]) -> str:
    value = _env_raw(*names)
    if value is None:
        return default
    value = value.strip().lower()
    return value if value in allowed else default


def _read_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        default_logger.warning(
            f"[EXO][GLM-5.2][IndexShare] failed to read {config_path}: {exc!r}"
        )
        return {}
    return data if isinstance(data, dict) else {}




def _expected_full_indexer_layers(indexer_types: Sequence[str]) -> set[int]:
    return {i for i, typ in enumerate(indexer_types) if typ in {"full", "sparse"}}


def _checkpoint_indexer_layers(model_path: Path) -> set[int] | None:
    """Return layers that have materialized self_attn.indexer weights, if discoverable.

    MLX/HF checkpoints commonly have a model.safetensors.index.json with a
    weight_map. Some local conversions may have one or more .safetensors files
    without an index; in that case return None and let runtime object checks
    below provide the useful signal.
    """

    index_paths = [
        model_path / "model.safetensors.index.json",
        model_path / "model.fp8.safetensors.index.json",
    ]
    index_paths.extend(sorted(model_path.glob("*.safetensors.index.json")))

    seen_paths: set[Path] = set()
    for path in index_paths:
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            default_logger.warning(
                f"[EXO][GLM-5.2][IndexShare] failed to read safetensors index {path}: {exc!r}"
            )
            continue

        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        if not isinstance(weight_map, dict):
            continue

        layers: set[int] = set()
        for key in weight_map:
            parts = str(key).split(".")
            for i, part in enumerate(parts[:-2]):
                if part != "layers" or i + 2 >= len(parts):
                    continue
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    continue
                rest = ".".join(parts[i + 2 :])
                if rest.startswith("self_attn.indexer."):
                    layers.add(layer_idx)
        return layers

    return None


def _runtime_indexer_layers(layers: Sequence[Any]) -> set[int]:
    present: set[int] = set()
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and getattr(attn, "indexer", None) is not None:
            present.add(i)
    return present


def _validate_indexshare_materialization(
    *,
    model_path: Path,
    layers: Sequence[Any],
    indexer_types: Sequence[str],
    logger: Any,
) -> bool:
    """Check that config IndexShare layout matches the loaded/materialized checkpoint.

    Returns True when safe to apply the patch. By default, missing full indexers
    disable the runtime patch rather than silently corrupting shared-layer reuse.
    Use EXO_GLM52_INDEXSHARE_VALIDATE=0 only for experiments.
    """

    if not _env_bool(
        "EXO_GLM52_INDEXSHARE_VALIDATE",
        "EXO_GLM_INDEXSHARE_VALIDATE",
        default=True,
    ):
        return True

    expected = _expected_full_indexer_layers(indexer_types)
    runtime_present = _runtime_indexer_layers(layers)

    # Before we drop shared indexers, common broken conversions have indexers on
    # all layers. That is safe to patch because the config tells us which ones
    # are shared. What is unsafe is missing a full indexer.
    missing_full_runtime = expected - runtime_present
    if missing_full_runtime:
        logger.warning(
            "[EXO][GLM-5.2][IndexShare] checkpoint/runtime is missing full "
            f"indexers required by config: {sorted(missing_full_runtime)[:16]}"
            + ("..." if len(missing_full_runtime) > 16 else "")
            + "; patch not applied"
        )
        return False

    checkpoint_layers = _checkpoint_indexer_layers(model_path)
    if checkpoint_layers is not None:
        strict_checkpoint = _env_bool(
            "EXO_GLM52_INDEXSHARE_STRICT_CHECKPOINT",
            "EXO_GLM_INDEXSHARE_STRICT_CHECKPOINT",
            default=False,
        )
        if strict_checkpoint and checkpoint_layers != expected:
            logger.warning(
                "[EXO][GLM-5.2][IndexShare] strict checkpoint/config mismatch: "
                f"expected_full={sorted(expected)[:32]}"
                + ("..." if len(expected) > 32 else "")
                + f" checkpoint_indexer_layers={sorted(checkpoint_layers)[:32]}"
                + ("..." if len(checkpoint_layers) > 32 else "")
                + "; patch not applied"
            )
            return False

        missing_full_ckpt = expected - checkpoint_layers
        if missing_full_ckpt:
            logger.warning(
                "[EXO][GLM-5.2][IndexShare] safetensors index is missing full "
                f"indexers required by config: {sorted(missing_full_ckpt)[:16]}"
                + ("..." if len(missing_full_ckpt) > 16 else "")
                + "; patch not applied"
            )
            return False

        extra_shared_ckpt = checkpoint_layers - expected
        if extra_shared_ckpt:
            logger.info(
                "[EXO][GLM-5.2][IndexShare] safetensors has materialized shared "
                f"indexers on {len(extra_shared_ckpt)} layers; they will be ignored/dropped"
            )

    return True

def _inner_model(model: Any) -> Any:
    return getattr(model, "model", model)


def _layers(model: Any) -> list[Any]:
    inner = _inner_model(model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return []
    try:
        return list(layers)
    except TypeError:
        return []


def _nearest_previous_full_sources(indexer_types: Sequence[str]) -> list[int | None]:
    """Map every layer to the previous full/sparse indexer layer."""

    sources: list[int | None] = []
    last_full: int | None = None
    for i, typ in enumerate(indexer_types):
        if typ in {"full", "sparse"}:
            last_full = i
            sources.append(i)
        elif typ == "shared":
            sources.append(last_full)
        else:
            sources.append(None)
    return sources


def _patch_mlx_lm_classes_once() -> None:
    global _CLASSES_PATCHED
    if _CLASSES_PATCHED:
        return

    from mlx_lm.models import deepseek_v32
    from mlx_lm.models.base import scaled_dot_product_attention

    DeepseekV32Model = deepseek_v32.DeepseekV32Model
    DeepseekV32Attention = deepseek_v32.DeepseekV32Attention

    if not hasattr(DeepseekV32Model, "_exo_original_call"):
        DeepseekV32Model._exo_original_call = DeepseekV32Model.__call__  # type: ignore[attr-defined]

        def _exo_model_call(self: Any, x: mx.array, cache: Any = None) -> mx.array:
            if not getattr(self, "_exo_glm52_indexshare_enabled", False):
                return self._exo_original_call(x, cache)  # type: ignore[attr-defined]

            token = _INDEXSHARE_CTX.set(
                {
                    "topk_by_layer": {},
                    "debug": _env_bool(
                        "EXO_GLM52_INDEXSHARE_DEBUG",
                        "EXO_GLM_INDEXSHARE_DEBUG",
                        default=False,
                    ),
                    "missing": _env_choice(
                        "EXO_GLM52_INDEXSHARE_MISSING",
                        "EXO_GLM_INDEXSHARE_MISSING",
                        default="dense",
                        allowed={"dense", "error"},
                    ),
                    "missing_warned": False,
                    "mask_fallback_warned": False,
                    "padded_batch_fallback_warned": False,
                }
            )
            try:
                return self._exo_original_call(x, cache)  # type: ignore[attr-defined]
            finally:
                _INDEXSHARE_CTX.reset(token)

        DeepseekV32Model.__call__ = _exo_model_call

    if not hasattr(DeepseekV32Attention, "_exo_original_call"):
        DeepseekV32Attention._exo_original_call = DeepseekV32Attention.__call__  # type: ignore[attr-defined]

        def _exo_attention_call(
            self: Any,
            x: mx.array,
            mask: mx.array | None = None,
            cache: Any = None,
        ) -> mx.array:
            # Fallback to upstream behavior unless this particular attention layer
            # was annotated by apply_glm52_indexshare_patch().
            if not getattr(self, "_exo_glm52_indexshare_enabled", False):
                return self._exo_original_call(x, mask, cache)  # type: ignore[attr-defined]

            cfg: GLM52PrefillConfig = getattr(
                self,
                "_exo_glm52_prefill_cfg",
                _DEFAULT_PREFILL_CFG,
            )
            set_cache_growth(cache, cfg)

            # This body mirrors the pinned mlx-lm DeepseekV32Attention. The
            # decode branch below remains the old IndexShare implementation;
            # W3/W4 can only be selected when L > 1.
            B, L, _ = x.shape
            offset = cache[0].offset if cache is not None else 0
            try:
                expected_kv_len = int(offset) + int(L)
            except (TypeError, ValueError):
                expected_kv_len = int(L)
            profiler = (
                PrefillProfileRecorder.for_attention(
                    self,
                    cfg,
                    offset=offset,
                    qlen=int(L),
                    kvlen=expected_kv_len,
                    logger=default_logger,
                )
                if cfg.profile
                else None
            )

            qr = self.q_a_layernorm(self.q_a_proj(x))
            q = self.q_b_proj(qr)
            q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
            q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

            compressed_kv = self.kv_a_proj_with_mqa(x)
            compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
            k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
            kv_latent = self.kv_a_layernorm(compressed_kv)

            q_pe = self.rope(q_pe, offset)
            k_pe = self.rope(k_pe, offset)
            if profiler is not None:
                profiler.stage("attn_qkv_proj_rope", q_nope, q_pe, kv_latent, k_pe)

            kv_latent = mx.expand_dims(kv_latent, axis=1)
            if cache is not None:
                kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
            else:
                cache = [None] * 2
            if profiler is not None:
                profiler.stage("attn_cache0_grow_update", kv_latent, k_pe)

            ctx = _INDEXSHARE_CTX.get()
            layer_idx = getattr(self, "_exo_glm52_layer_idx", None)
            indexer_type = getattr(self, "_exo_glm52_indexer_type", "full")
            source_layer = getattr(self, "_exo_glm52_indexshare_source", None)
            topk_indices: mx.array | None = None
            indexer_was_called = False

            if ctx is not None and indexer_type == "shared":
                topk_by_layer = ctx["topk_by_layer"]
                value = topk_by_layer.get(source_layer, _MISSING)
                if value is _MISSING:
                    missing_mode = ctx.get("missing", "dense")
                    if missing_mode == "error":
                        raise RuntimeError(
                            "GLM-5.2 IndexShare source top-k is missing for "
                            f"layer {layer_idx}; source={source_layer}. This can "
                            "happen with unsupported pipeline splits."
                        )
                    # A local shared indexer is not mathematically valid for
                    # IndexShare, so both accepted fallback values are dense.
                    topk_indices = None
                    if not ctx.get("missing_warned", False):
                        default_logger.warning(
                            "[EXO][GLM-5.2][IndexShare] missing source top-k "
                            f"for shared layer {layer_idx} source={source_layer}; "
                            f"fallback={missing_mode}"
                        )
                        ctx["missing_warned"] = True
                else:
                    # None is meaningful before cumulative length crosses top-k.
                    topk_indices = value
                    if ctx.get("debug", False):
                        default_logger.info(
                            "[EXO][GLM-5.2][IndexShare] layer "
                            f"{layer_idx} reused top-k from layer {source_layer}"
                        )
            else:
                if int(L) > 1 and cfg.indexer_enabled:
                    if (
                        _MTP_VERIFY_CTX.get()
                        and 1 < int(L) <= _MICRO_DECODE_L
                        and int(B) == 1
                    ):
                        # MTP verify (B==1, L<=4): the streaming/tiled indexer
                        # is prefill machinery — its k-chunk loop (9 chunks at
                        # kv=147k) is pure per-token overhead here, while the
                        # full score block for L<=4 is a few MB. Monolithic
                        # pinned call, no fallback: a failure surfaces like
                        # every other MTP-path failure (rollback + no MTP).
                        topk_indices = self.indexer(x, qr, mask, cache=cache[1])
                    else:
                        topk_indices = tiled_indexer_call(
                            self.indexer,
                            x,
                            qr,
                            mask,
                            cache[1],
                            cfg,
                            profiler=profiler,
                        )
                elif profiler is not None:
                    topk_indices = profiled_monolithic_indexer_call(
                        self.indexer,
                        x,
                        qr,
                        mask,
                        cache[1],
                        profiler=profiler,
                    )
                else:
                    topk_indices = self.indexer(x, qr, mask, cache=cache[1])
                indexer_was_called = True
                if ctx is not None and layer_idx is not None:
                    ctx["topk_by_layer"][layer_idx] = topk_indices

            # Shared IndexShare layers do not execute an indexer. Populate their
            # cache[1] so CacheList.state/trim/filter remain valid. "compact"
            # stores one unused channel instead of index_head_dim channels.
            if not indexer_was_called and cache is not None and cache[1] is not None:
                placeholder_width = shared_cache_placeholder_width(self, cfg)
                cache[1].update_and_fetch(
                    mx.zeros((B, 1, L, placeholder_width), dtype=kv_latent.dtype),
                    mx.zeros((B, 1, L, 0), dtype=kv_latent.dtype),
                )
                if profiler is not None:
                    profiler.stage(
                        "shared_cache1_zero_grow_update",
                        cache[1].keys,
                        cache[1].values,
                    )

            # Keep the indexer graph bounded only on layers that ran it. This
            # must happen before the sparse early return as well as dense SDPA.
            if indexer_was_called and cache is not None and cache[0] is not None:
                cache[0].keys = mx.depends(
                    cache[0].keys,
                    (cache[1].keys, cache[1].values),
                )

            kv_length = int(kv_latent.shape[2])
            # Micro multi-token decode (the MTP verify forward, L=2): the
            # sparse_min_kv crossover is prefill economics (L~1024 chunks);
            # for L<=_MICRO_DECODE_L gather-topk is strictly cheaper than
            # dense whenever the indexer produced indices, and — decisive —
            # it keeps the verify in the same topk math family as the L==1
            # decode branch, so t1/h match what MTP=off would compute.
            micro_decode = (
                topk_indices is not None
                and 1 < int(L) <= _MICRO_DECODE_L
                and int(B) == 1
                and _MTP_VERIFY_CTX.get()
            )
            sparse_requested = (
                topk_indices is not None
                and int(L) > 1
                and (
                    micro_decode
                    or cfg.sparse_enabled_for(
                        query_length=int(L),
                        kv_length=kv_length,
                    )
                )
            )
            padded_batch_fallback = bool(
                sparse_requested
                and cache is not None
                and cache[0] is not None
                and cache_requires_dense_prefill(cache[0])
            )
            use_sparse_prefill = sparse_requested and not padded_batch_fallback
            if padded_batch_fallback and (
                ctx is None or not ctx.get("padded_batch_fallback_warned", False)
            ):
                default_logger.warning(
                    "[EXO][GLM-5.2][Prefill] padded batched cache can contain "
                    "all-masked query rows; using exact masked-dense fallback"
                )
                if ctx is not None:
                    ctx["padded_batch_fallback_warned"] = True

            if use_sparse_prefill:
                try:
                    output = sparse_prefill_attention(
                        self,
                        q_nope=q_nope,
                        q_pe=q_pe,
                        kv_latent=kv_latent,
                        k_pe=k_pe,
                        topk_indices=topk_indices,
                        mask=mask,
                        scaled_dot_product_attention=scaled_dot_product_attention,
                        cfg=cfg,
                        profiler=profiler,
                    )
                except UnsupportedPrefillMask as exc:
                    # Preserve exact upstream semantics instead of guessing at
                    # additive/string/custom masks.
                    if ctx is None or not ctx.get("mask_fallback_warned", False):
                        default_logger.warning(
                            "[EXO][GLM-5.2][Prefill] sparse mask unsupported; "
                            f"using masked-dense fallback: {exc}"
                        )
                        if ctx is not None:
                            ctx["mask_fallback_warned"] = True
                else:
                    if profiler is not None:
                        profiler.finish(output, path="sparse")
                    return output

            # Original decode and masked-dense prefill path. Decode is preserved
            # byte-for-byte in its gather/mask behavior.
            if topk_indices is not None:
                if L == 1:
                    idx = topk_indices[:, :, 0, :, None]
                    kv_latent = mx.take_along_axis(
                        kv_latent,
                        mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                        axis=2,
                    )
                    k_pe = mx.take_along_axis(
                        k_pe,
                        mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                        axis=2,
                    )
                    if mask is not None and hasattr(cache[0], "left_padding"):
                        gathered_idx = topk_indices[:, :, 0, :]
                        left_pad = cache[0].left_padding[:, None, None]
                        mask = (gathered_idx >= left_pad)[:, :, None, :]
                    else:
                        mask = None
                else:
                    shape = list(topk_indices.shape)
                    shape[-1] = kv_latent.shape[2]
                    sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                    sparse_mask = mx.put_along_axis(
                        sparse_mask,
                        topk_indices,
                        mx.array(True),
                        axis=-1,
                    )
                    if mask is not None:
                        sparse_mask = sparse_mask & mask
                    mask = sparse_mask
                    if profiler is not None:
                        profiler.stage("sparse_mask_build", mask)

            pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
            if mask is not None:
                pe_scores = mx.where(
                    mask,
                    pe_scores,
                    mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
                )
            if profiler is not None:
                profiler.stage("pe_scores_full", pe_scores)

            if L == 1:
                q_nope = self.embed_q(q_nope)
                k = v = kv_latent
            else:
                k = self.embed_q(kv_latent, transpose=False)
                v = self.unembed_out(kv_latent)
                if profiler is not None:
                    profiler.stage("kv_materialize", k, v)

            output = scaled_dot_product_attention(
                q_nope,
                k,
                v,
                cache=cache,
                scale=self.scale,
                mask=pe_scores,
            )
            if profiler is not None:
                profiler.stage("sdpa", output)
            if L == 1:
                output = self.unembed_out(output)
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            output = self.o_proj(output)
            if profiler is not None:
                profiler.stage("unembed_o_proj_collective", output)
                profiler.finish(output, path="decode" if L == 1 else "masked_dense")
            return output

        DeepseekV32Attention.__call__ = _exo_attention_call

    _CLASSES_PATCHED = True

def _fix_indexer_rope_noninterleaved(
    attn: Any,
    *,
    qk_rope_head_dim: int,
    rope_theta: float,
    logger: Any = default_logger,
) -> bool:
    """Rebuild a full layer's indexer RoPE as non-interleaved (half-split).

    GLM-5.2's indexer uses ``apply_rotary_pos_emb`` (non-interleaved / half-split),
    explicitly *unlike* the main MLA attention, which uses
    ``apply_rotary_pos_emb_interleave``. mlx-lm's deepseek_v32 builds the indexer
    rope with ``traditional=True`` (interleaved) for both — correct for
    DeepSeek-V3.2 but wrong for GLM-5.2, which corrupts top-k selection once DSA
    becomes active. Only the plain (rope_type=default) case is handled; other rope
    types are left untouched and reported so they can be checked manually.
    """
    indexer = getattr(attn, "indexer", None)
    if indexer is None:
        return False
    old = getattr(indexer, "rope", None)
    if old is None:
        return False
    if old.__class__.__name__ != "RoPE":
        logger.warning(
            "[EXO][GLM-5.2][IndexShare] indexer rope is "
            f"{old.__class__.__name__}, not plain RoPE; non-interleaved fix "
            "skipped (verify the indexer rope convention manually)"
        )
        return False
    if not getattr(old, "traditional", False):
        return False  # already non-interleaved
    try:
        import mlx.nn as _mlxnn

        indexer.rope = _mlxnn.RoPE(
            qk_rope_head_dim,
            traditional=False,
            base=rope_theta,
            scale=getattr(old, "scale", 1.0),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            f"[EXO][GLM-5.2][IndexShare] could not rebuild indexer rope: {exc}"
        )
        return False
    return True


def indexer_rope_fix_for_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """The same half-split RoPE decision apply_glm52_indexshare_patch makes for
    the main layers, exported so the MTP block's own indexer gets the identical
    treatment (audit P1.1). Returns the rebuild kwargs or None (no fix)."""
    fix = _env_bool(
        "EXO_GLM52_INDEXER_HALF_SPLIT_ROPE",
        "EXO_GLM_INDEXER_HALF_SPLIT_ROPE",
        default=(config.get("indexer_rope_interleave") is False),
    )
    rope_params = config.get("rope_parameters") or {}
    rope_type = rope_params.get("rope_type") or rope_params.get("type") or "default"
    rope_theta = rope_params.get("rope_theta") or config.get("rope_theta")
    qk_rope = config.get("qk_rope_head_dim")
    if not (fix and str(rope_type) in {"default", ""} and rope_theta and qk_rope):
        return None
    return {"qk_rope_head_dim": int(qk_rope), "rope_theta": float(rope_theta)}


def annotate_mtp_attention(attn: Any, layers: list[Any], layer_idx: int) -> bool:
    """Give the MTP block's attention the same IndexShare annotation as a
    main 'full' layer (own indexer, no source, unique layer index) so it
    takes the fork's optimized decode/prefill branches instead of the raw
    pinned path — over a prompt-sized MTP cache the raw path costs ~60ms per
    draft. No ctx side effects: drafts/ingest run outside the model-forward
    context, so topk_by_layer is never written by this layer."""
    template = None
    for layer in layers:
        a = getattr(layer, "self_attn", None)
        if a is not None and getattr(a, "_exo_glm52_indexshare_enabled", False):
            template = a
            break
    if template is None:
        return False
    setattr(attn, "_exo_glm52_indexshare_enabled", True)
    setattr(attn, "_exo_glm52_layer_idx", layer_idx)
    setattr(attn, "_exo_glm52_indexer_type", "full")
    setattr(attn, "_exo_glm52_indexshare_source", None)
    setattr(attn, "_exo_glm52_index_head_dim", getattr(template, "_exo_glm52_index_head_dim"))
    setattr(attn, "_exo_glm52_prefill_cfg", getattr(template, "_exo_glm52_prefill_cfg"))
    setattr(attn, "_exo_glm52_profile_selected", False)
    return True


def apply_mtp_indexer_rope_fix(attn: Any, config: dict[str, Any], *, logger: Any) -> bool:
    params = indexer_rope_fix_for_config(config)
    if params is None:
        return False
    return _fix_indexer_rope_noninterleaved(attn, logger=logger, **params)


def apply_glm52_indexshare_patch(
    model: Any,
    model_path: Path,
    *,
    logger: Any = default_logger,
) -> Any:
    """Enable GLM-5.2 IndexShare on a loaded mlx-lm model if applicable.

    Always returns the original model object so callers can use it inline in
    load paths without special-casing unsupported models.
    """

    if not _env_bool(
        "EXO_GLM52_INDEXSHARE",
        "EXO_GLM_INDEXSHARE",
        default=True,
    ):
        logger.info("[EXO][GLM-5.2][IndexShare] disabled by environment")
        return model

    config = _read_config(model_path)
    if config.get("model_type") != "glm_moe_dsa":
        return model

    raw_indexer_types = config.get("indexer_types")
    if not isinstance(raw_indexer_types, list) or not raw_indexer_types:
        return model

    indexer_types = [str(x) for x in raw_indexer_types]
    if "shared" not in indexer_types:
        return model

    layers = _layers(model)
    if not layers:
        logger.warning("[EXO][GLM-5.2][IndexShare] no layers found; patch not applied")
        return model

    if len(indexer_types) != len(layers):
        logger.warning(
            "[EXO][GLM-5.2][IndexShare] indexer_types length does not match "
            f"layers: {len(indexer_types)} vs {len(layers)}; patch not applied"
        )
        return model

    sources = _nearest_previous_full_sources(indexer_types)
    if any(t == "shared" and sources[i] is None for i, t in enumerate(indexer_types)):
        logger.warning(
            "[EXO][GLM-5.2][IndexShare] shared layer without previous full layer; "
            "patch not applied"
        )
        return model

    if not _validate_indexshare_materialization(
        model_path=model_path,
        layers=layers,
        indexer_types=indexer_types,
        logger=logger,
    ):
        return model

    # Resolve and hash-check the pinned mlx-lm hot paths before monkey-patching
    # their classes. A mismatch disables W3/W4 but leaves existing IndexShare
    # behavior available.
    from mlx_lm.models import deepseek_v32

    prefill_cfg = read_prefill_config(deepseek_v32, logger=logger)
    _patch_mlx_lm_classes_once()

    inner = _inner_model(model)
    setattr(inner, "_exo_glm52_indexshare_enabled", True)
    setattr(inner, "_exo_glm52_indexer_types", indexer_types)
    setattr(inner, "_exo_glm52_indexshare_sources", sources)
    setattr(inner, "_exo_glm52_prefill_cfg", prefill_cfg)

    full_count = 0
    shared_count = 0
    dropped_shared_indexers = 0
    drop_shared_indexers = _env_bool(
        "EXO_GLM52_INDEXSHARE_DROP_SHARED_INDEXERS",
        "EXO_GLM52_DROP_SHARED_INDEXERS",
        "EXO_GLM_INDEXSHARE_DROP_SHARED_INDEXERS",
        default=True,
    )
    diag = _env_bool(
        "EXO_GLM_INDEXSHARE_DIAG",
        "EXO_GLM52_INDEXSHARE_DIAG",
        "EXO_GLM_DIAG",
        default=False,
    )

    # Shared layers run no indexer, so their per-layer indexer KV cache must be
    # kept populated with a correctly-shaped zero update (see _exo_attention_call)
    # to avoid an mlx cache-state crash during prefill. Prefer the config value;
    # fall back to a surviving full-layer indexer's head_dim.
    index_head_dim = config.get("index_head_dim")
    if not index_head_dim:
        for _layer in layers:
            _a = getattr(_layer, "self_attn", None)
            _ix = getattr(_a, "indexer", None) if _a is not None else None
            if _ix is not None and getattr(_ix, "head_dim", None):
                index_head_dim = _ix.head_dim
                break
    try:
        index_head_dim = int(index_head_dim or 0)
    except (TypeError, ValueError):
        logger.warning(
            "[EXO][GLM-5.2][IndexShare] invalid index_head_dim="
            f"{index_head_dim!r}; falling back to zero-width shared indexer cache"
        )
        index_head_dim = 0
    if not index_head_dim:
        logger.warning(
            "[EXO][GLM-5.2][IndexShare] could not determine index_head_dim; "
            "using zero-width shared-layer indexer cache placeholder"
        )

    # GLM-5.2's indexer needs non-interleaved (half-split) RoPE, unlike mlx-lm's
    # deepseek_v32 default (traditional=True). Rebuild full-layer indexer ropes
    # for the plain rope_type=default case (GLM-5.2). Toggleable for A/B.
    fix_indexer_rope = _env_bool(
        "EXO_GLM52_INDEXER_HALF_SPLIT_ROPE",
        "EXO_GLM_INDEXER_HALF_SPLIT_ROPE",
        default=(config.get("indexer_rope_interleave") is False),
    )
    _rope_params = config.get("rope_parameters") or {}
    _rope_type = _rope_params.get("rope_type") or _rope_params.get("type") or "default"
    _rope_theta = _rope_params.get("rope_theta") or config.get("rope_theta")
    _qk_rope = config.get("qk_rope_head_dim")
    _rope_fixable = (
        fix_indexer_rope
        and str(_rope_type) in {"default", ""}
        and bool(_rope_theta)
        and bool(_qk_rope)
    )
    if fix_indexer_rope and not _rope_fixable:
        logger.warning(
            "[EXO][GLM-5.2][IndexShare] indexer non-interleaved rope fix requested "
            f"but not applicable (rope_type={_rope_type!r}, rope_theta={_rope_theta}, "
            f"qk_rope_head_dim={_qk_rope}); skipping"
        )
    indexer_rope_fixed = 0

    if prefill_cfg.profile_layers:
        profile_layers = set(prefill_cfg.profile_layers)
    elif prefill_cfg.profile:
        # Default to one full and one shared layer so attribution catches both
        # indexer compute and shared-cache/IndexShare reuse.
        profile_layers: set[int] = set()
        first_full = next(
            (i for i, typ in enumerate(indexer_types) if typ in {"full", "sparse"}),
            None,
        )
        first_shared = next(
            (i for i, typ in enumerate(indexer_types) if typ == "shared"),
            None,
        )
        if first_full is not None:
            profile_layers.add(first_full)
        if first_shared is not None:
            profile_layers.add(first_shared)
    else:
        profile_layers = set()

    for i, layer in enumerate(layers):
        if layer is None:
            continue
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        typ = indexer_types[i]
        if typ in {"full", "sparse"}:
            full_count += 1
            if _rope_fixable and _fix_indexer_rope_noninterleaved(
                attn,
                qk_rope_head_dim=_qk_rope,
                rope_theta=_rope_theta,
                logger=logger,
            ):
                indexer_rope_fixed += 1
        elif typ == "shared":
            shared_count += 1
            if drop_shared_indexers and getattr(attn, "indexer", None) is not None:
                setattr(attn, "_exo_glm52_dropped_indexer_class", attn.indexer.__class__.__name__)
                attn.indexer = None
                dropped_shared_indexers += 1
        setattr(attn, "_exo_glm52_indexshare_enabled", True)
        setattr(attn, "_exo_glm52_layer_idx", i)
        setattr(attn, "_exo_glm52_indexer_type", typ)
        setattr(attn, "_exo_glm52_indexshare_source", sources[i])
        setattr(attn, "_exo_glm52_index_head_dim", index_head_dim)
        setattr(attn, "_exo_glm52_prefill_cfg", prefill_cfg)
        setattr(attn, "_exo_glm52_profile_selected", i in profile_layers)

    logger.info(
        "[EXO][GLM-5.2][IndexShare] enabled: "
        f"layers={len(layers)}, full={full_count}, shared={shared_count}, "
        f"index_topk={config.get('index_topk')}, "
        f"index_topk_freq={config.get('index_topk_freq')}, "
        f"index_skip_topk_offset={config.get('index_skip_topk_offset')}, "
        f"dropped_shared_indexers={dropped_shared_indexers}, "
        f"indexer_rope_fixed={indexer_rope_fixed}, "
        f"validate={_env_bool('EXO_GLM52_INDEXSHARE_VALIDATE', 'EXO_GLM_INDEXSHARE_VALIDATE', default=True)}, "
        f"strict_checkpoint={_env_bool('EXO_GLM52_INDEXSHARE_STRICT_CHECKPOINT', 'EXO_GLM_INDEXSHARE_STRICT_CHECKPOINT', default=False)}, "
        f"missing_fallback={_env_choice('EXO_GLM52_INDEXSHARE_MISSING', 'EXO_GLM_INDEXSHARE_MISSING', default='dense', allowed={'dense', 'error'})}, "
        f"hotpath_compatible={prefill_cfg.hotpath_compatible}, "
        f"sparse_mode={prefill_cfg.sparse_mode}, "
        f"sparse_q_chunk={prefill_cfg.sparse_q_chunk}, "
        f"sparse_min_kv={prefill_cfg.sparse_min_kv}, "
        f"indexer_mode={prefill_cfg.indexer_mode}, "
        f"indexer_q_chunk={prefill_cfg.indexer_q_chunk}, "
        f"indexer_k_chunk={prefill_cfg.indexer_k_chunk}, "
        f"cache_growth={prefill_cfg.cache_growth or 'off'}, "
        f"shared_index_cache={prefill_cfg.shared_index_cache}, "
        f"profile_layers={sorted(profile_layers)}, "
        f"upstream_indexer_sha256={prefill_cfg.upstream_indexer_sha256}, "
        f"upstream_attention_sha256={prefill_cfg.upstream_attention_sha256}"
    )
    if not prefill_cfg.hotpath_compatible:
        logger.warning(
            "[EXO][GLM-5.2][Prefill] compatibility guard: "
            f"{prefill_cfg.compatibility_reason}"
        )
    if diag:
        examples: list[str] = []
        for i, typ in enumerate(indexer_types):
            if typ == "shared":
                examples.append(f"{i}->{sources[i]}")
            if len(examples) >= 16:
                break
        if examples:
            logger.info("[EXO][GLM-5.2][IndexShare] shared source examples: " + ", ".join(examples))
    return model
