"""EXO-local GLM-5.2 DSA prefill hot paths.

This module contains the performance-sensitive parts kept separate from
``glm52_indexshare.py``:

* fail-closed compatibility checks for the pinned mlx-lm DeepSeek-V3.2 path;
* a 2D-tiled DSA indexer (reference and streaming variants);
* query-tiled sparse prefill using the existing absorbed MLA/SDPA path;
* scoped cache-growth controls and low-overhead profiling helpers.

Every behavior-changing feature is environment-gated and defaults to off.
The existing decode path is intentionally not implemented here and therefore
cannot be selected by these helpers.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

import mlx.core as mx

from exo.worker.runner.bootstrap import logger as default_logger


class UnsupportedPrefillMask(RuntimeError):
    """Raised before graph construction when a mask cannot be preserved exactly."""


@dataclass(frozen=True)
class GLM52PrefillConfig:
    """Resolved process-wide settings attached to annotated GLM-5.2 layers."""

    sparse_mode: str = "off"
    sparse_q_chunk: int = 256
    sparse_min_kv: int = 0
    sparse_eval_blocks: bool = False

    indexer_mode: str = "off"
    indexer_q_chunk: int = 256
    indexer_k_chunk: int = 16384
    indexer_eval_chunks: bool = False

    cache_growth: int = 0
    shared_index_cache: str = "zero"

    profile: bool = False
    profile_sync: bool = True
    profile_every: int = 8
    profile_layers: tuple[int, ...] = ()

    hotpath_compatible: bool = False
    compatibility_reason: str = "not checked"
    upstream_indexer_sha256: str = "unknown"
    upstream_attention_sha256: str = "unknown"

    @property
    def indexer_enabled(self) -> bool:
        return self.hotpath_compatible and self.indexer_mode != "off"

    def sparse_enabled_for(self, *, query_length: int, kv_length: int) -> bool:
        if not self.hotpath_compatible or query_length <= 1:
            return False
        if self.sparse_mode == "on":
            return True
        if self.sparse_mode == "auto":
            return self.sparse_min_kv > 0 and kv_length >= self.sparse_min_kv
        return False


def _env_raw(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return None


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "none", "disabled"}:
        return False
    return default


def _parse_choice(
    value: str | None,
    *,
    default: str,
    allowed: set[str],
    name: str,
    logger: Any,
) -> str:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in allowed:
        return normalized
    logger.warning(
        f"[EXO][GLM-5.2][Prefill] invalid {name}={value!r}; using {default!r}"
    )
    return default


def _normalize_toggle_choice(
    value: str | None,
    *,
    default: str,
    enabled: str,
    allowed: set[str],
    name: str,
    logger: Any,
) -> str:
    """Accept both explicit modes and shell-friendly 0/1 boolean values."""

    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "enabled"}:
        return enabled
    if normalized in {"0", "false", "no", "n", "off", "none", "disabled"}:
        return "off"
    if normalized in allowed:
        return normalized
    logger.warning(
        f"[EXO][GLM-5.2][Prefill] invalid {name}={value!r}; using {default!r}"
    )
    return default


def _parse_int(
    value: str | None,
    *,
    default: int,
    minimum: int,
    maximum: int,
    name: str,
    logger: Any,
) -> int:
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except (TypeError, ValueError):
        logger.warning(
            f"[EXO][GLM-5.2][Prefill] invalid {name}={value!r}; using {default}"
        )
        return default
    clamped = max(minimum, min(maximum, parsed))
    if clamped != parsed:
        logger.warning(
            f"[EXO][GLM-5.2][Prefill] clamped {name}={parsed} to {clamped}"
        )
    return clamped


def _parse_profile_layers(value: str | None, *, logger: Any) -> tuple[int, ...]:
    if value is None or not value.strip():
        return ()
    layers: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            layer = int(raw)
        except ValueError:
            logger.warning(
                "[EXO][GLM-5.2][Prefill] invalid "
                f"EXO_GLM52_PREFILL_PROFILE_LAYER component {raw!r}; ignored"
            )
            continue
        if layer < 0:
            logger.warning(
                "[EXO][GLM-5.2][Prefill] negative profile layer "
                f"{layer}; ignored"
            )
            continue
        layers.append(layer)
    return tuple(sorted(set(layers)))


def _parse_cache_growth(value: str | None, *, logger: Any) -> int:
    if value is None or value.strip().lower() in {"", "0", "off", "none", "disabled"}:
        return 0
    if value.strip().lower() == "geometric":
        logger.warning(
            "[EXO][GLM-5.2][Prefill] EXO_GLM52_CACHE_GROWTH=geometric is "
            "not enabled by this revision; using the upstream growth policy"
        )
        return 0
    parsed = _parse_int(
        value,
        default=0,
        minimum=256,
        maximum=131072,
        name="EXO_GLM52_CACHE_GROWTH",
        logger=logger,
    )
    if parsed == 0:
        return 0
    # KVCache capacities are conventionally aligned to its 256-token base step.
    return ((parsed + 255) // 256) * 256


def _function_sha256(fn: Any) -> str:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return "unavailable"
    normalized = "\n".join(line.rstrip() for line in source.splitlines()).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def check_deepseek_v32_compatibility(deepseek_v32: Any) -> tuple[bool, str, str, str]:
    """Verify the exact structural assumptions used by the EXO hot paths.

    The check deliberately fails closed. IndexShare itself can remain enabled,
    but W3/W4 stay off when mlx-lm changes the expected call signatures or core
    score construction.
    """

    indexer_cls = getattr(deepseek_v32, "Indexer", None)
    attention_cls = getattr(deepseek_v32, "DeepseekV32Attention", None)
    indexer_call = getattr(
        indexer_cls,
        "_exo_original_call",
        getattr(indexer_cls, "__call__", None),
    )
    attention_call = getattr(
        attention_cls,
        "_exo_original_call",
        getattr(attention_cls, "__call__", None),
    )
    if indexer_call is None or attention_call is None:
        return False, "Indexer/DeepseekV32Attention is missing", "missing", "missing"

    indexer_sha = _function_sha256(indexer_call)
    attention_sha = _function_sha256(attention_call)

    try:
        indexer_params = tuple(inspect.signature(indexer_call).parameters)
        attention_params = tuple(inspect.signature(attention_call).parameters)
    except (TypeError, ValueError) as exc:
        return (
            False,
            f"could not inspect call signatures: {exc}",
            indexer_sha,
            attention_sha,
        )

    if indexer_params != ("self", "x", "qr", "mask", "cache"):
        return (
            False,
            f"unexpected Indexer.__call__ signature {indexer_params}",
            indexer_sha,
            attention_sha,
        )
    if attention_params != ("self", "x", "mask", "cache"):
        return (
            False,
            f"unexpected DeepseekV32Attention.__call__ signature {attention_params}",
            indexer_sha,
            attention_sha,
        )

    try:
        indexer_source = inspect.getsource(indexer_call)
        attention_source = inspect.getsource(attention_call)
    except (OSError, TypeError) as exc:
        return (
            False,
            f"could not inspect upstream source: {exc}",
            indexer_sha,
            attention_sha,
        )

    indexer_sentinels = (
        "scores = q @ k.swapaxes(-1, -2)",
        "scores = mx.maximum(scores, 0)",
        "self.weights_proj(x)",
        "mx.argpartition",
        "cache.update_and_fetch",
    )
    attention_sentinels = (
        "pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)",
        "self.embed_q(kv_latent, transpose=False)",
        "self.unembed_out(kv_latent)",
        "scaled_dot_product_attention",
        "if L == 1",
    )
    missing = [s for s in indexer_sentinels if s not in indexer_source]
    missing += [s for s in attention_sentinels if s not in attention_source]
    if missing:
        return (
            False,
            "upstream source sentinels missing: " + ", ".join(repr(x) for x in missing),
            indexer_sha,
            attention_sha,
        )

    return True, "compatible pinned DeepSeek-V3.2 structure", indexer_sha, attention_sha


def read_prefill_config(deepseek_v32: Any, *, logger: Any = default_logger) -> GLM52PrefillConfig:
    compatible, reason, indexer_sha, attention_sha = check_deepseek_v32_compatibility(
        deepseek_v32
    )

    sparse_mode = _normalize_toggle_choice(
        _env_raw("EXO_GLM52_SPARSE_PREFILL"),
        # Site default (this cluster): validated 2026-09-01 on 4x M3 Ultra,
        # GLM-5.3-8bit-idxbf16. Upstream PR: revert to "off".
        default="auto",
        enabled="on",
        allowed={"off", "on", "auto"},
        name="EXO_GLM52_SPARSE_PREFILL",
        logger=logger,
    )

    legacy_indexer_chunk = _env_raw("EXO_GLM52_INDEXER_CHUNK")
    indexer_mode_raw = _env_raw("EXO_GLM52_INDEXER_MODE")
    if indexer_mode_raw is None and legacy_indexer_chunk is not None:
        try:
            indexer_mode_raw = (
                "reference" if int(legacy_indexer_chunk.strip()) > 0 else "off"
            )
        except (TypeError, ValueError):
            indexer_mode_raw = legacy_indexer_chunk
    indexer_mode = _normalize_toggle_choice(
        indexer_mode_raw,
        # Site default: streaming won the A/B on this cluster (-5.9% wall,
        # -11GB peak on 157k cold prefill). Upstream PR: revert to "off".
        default="streaming",
        enabled="reference",
        allowed={"off", "reference", "streaming"},
        name="EXO_GLM52_INDEXER_MODE",
        logger=logger,
    )
    shared_index_cache = _parse_choice(
        _env_raw("EXO_GLM52_SHARED_INDEX_CACHE"),
        default="zero",
        allowed={"zero", "compact"},
        name="EXO_GLM52_SHARED_INDEX_CACHE",
        logger=logger,
    )

    cfg = GLM52PrefillConfig(
        sparse_mode=sparse_mode,
        sparse_q_chunk=_parse_int(
            _env_raw("EXO_GLM52_SPARSE_Q_CHUNK"),
            default=256,
            minimum=1,
            maximum=16384,
            name="EXO_GLM52_SPARSE_Q_CHUNK",
            logger=logger,
        ),
        sparse_min_kv=_parse_int(
            _env_raw("EXO_GLM52_SPARSE_MIN_KV"),
            # Site default: measured dense/sparse crossover between 8k and 16k
            # (GLM-5.3-8bit-idxbf16, step=1024, 2026-09-01) -> midpoint.
            # Upstream PR: revert to 0.
            default=12288,
            minimum=0,
            maximum=4_194_304,
            name="EXO_GLM52_SPARSE_MIN_KV",
            logger=logger,
        ),
        sparse_eval_blocks=_parse_bool(
            _env_raw("EXO_GLM52_SPARSE_EVAL_BLOCKS"), default=False
        ),
        indexer_mode=indexer_mode,
        indexer_q_chunk=_parse_int(
            _env_raw("EXO_GLM52_INDEXER_Q_CHUNK"),
            default=256,
            minimum=1,
            maximum=16384,
            name="EXO_GLM52_INDEXER_Q_CHUNK",
            logger=logger,
        ),
        indexer_k_chunk=_parse_int(
            _env_raw("EXO_GLM52_INDEXER_K_CHUNK", "EXO_GLM52_INDEXER_CHUNK"),
            default=16384,
            minimum=256,
            maximum=1_048_576,
            name="EXO_GLM52_INDEXER_K_CHUNK",
            logger=logger,
        ),
        indexer_eval_chunks=_parse_bool(
            _env_raw("EXO_GLM52_INDEXER_EVAL_CHUNKS"), default=False
        ),
        cache_growth=_parse_cache_growth(
            _env_raw("EXO_GLM52_CACHE_GROWTH"), logger=logger
        ),
        shared_index_cache=shared_index_cache,
        profile=_parse_bool(
            _env_raw("EXO_GLM52_PREFILL_PROFILE"), default=False
        ),
        profile_sync=_parse_bool(
            _env_raw("EXO_GLM52_PREFILL_PROFILE_SYNC"), default=True
        ),
        profile_every=_parse_int(
            _env_raw("EXO_GLM52_PREFILL_PROFILE_EVERY"),
            default=8,
            minimum=1,
            maximum=1_000_000,
            name="EXO_GLM52_PREFILL_PROFILE_EVERY",
            logger=logger,
        ),
        profile_layers=_parse_profile_layers(
            _env_raw("EXO_GLM52_PREFILL_PROFILE_LAYER"), logger=logger
        ),
        hotpath_compatible=compatible,
        compatibility_reason=reason,
        upstream_indexer_sha256=indexer_sha,
        upstream_attention_sha256=attention_sha,
    )

    if not compatible and (cfg.sparse_mode != "off" or cfg.indexer_mode != "off"):
        logger.warning(
            "[EXO][GLM-5.2][Prefill] W3/W4 requested but compatibility guard "
            f"failed: {reason}; hot paths remain disabled"
        )
    if cfg.sparse_mode == "auto" and cfg.sparse_min_kv == 0:
        logger.warning(
            "[EXO][GLM-5.2][Prefill] sparse auto mode has no measured crossover "
            "(EXO_GLM52_SPARSE_MIN_KV=0); sparse prefill remains disabled"
        )

    return cfg


def mlx_memory_stats() -> dict[str, int]:
    """Return MLX allocator counters without assuming a particular MLX build."""

    result: dict[str, int] = {}
    for key, function_name in (
        ("active_bytes", "get_active_memory"),
        ("peak_bytes", "get_peak_memory"),
        ("cache_bytes", "get_cache_memory"),
    ):
        fn = getattr(mx, function_name, None)
        try:
            result[key] = int(fn()) if callable(fn) else -1
        except Exception:  # pragma: no cover - diagnostics must never break inference
            result[key] = -1
    return result


def _flatten_eval_values(values: Iterable[Any]) -> list[mx.array]:
    flattened: list[mx.array] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            flattened.extend(_flatten_eval_values(value))
        else:
            flattened.append(value)
    return flattened


class PrefillProfileRecorder:
    """Optional, intrusive stage timer; the disabled path has near-zero overhead."""

    __slots__ = (
        "enabled",
        "sync",
        "layer",
        "layer_type",
        "offset",
        "qlen",
        "kvlen",
        "rank",
        "warmup",
        "_start",
        "_last",
        "_stages",
        "_logger",
    )

    def __init__(
        self,
        *,
        enabled: bool,
        sync: bool,
        layer: int,
        layer_type: str,
        offset: int,
        qlen: int,
        kvlen: int,
        rank: int,
        warmup: bool,
        logger: Any,
    ) -> None:
        self.enabled = enabled
        self.sync = sync
        self.layer = layer
        self.layer_type = layer_type
        self.offset = offset
        self.qlen = qlen
        self.kvlen = kvlen
        self.rank = rank
        self.warmup = warmup
        self._start = time.perf_counter()
        self._last = self._start
        self._stages: list[tuple[str, float, dict[str, int]]] = []
        self._logger = logger

    @classmethod
    def for_attention(
        cls,
        attn: Any,
        cfg: GLM52PrefillConfig,
        *,
        offset: int,
        qlen: int,
        kvlen: int,
        logger: Any = default_logger,
    ) -> "PrefillProfileRecorder | None":
        # No object allocation and no per-layer counters on the normal path.
        if not cfg.profile or qlen <= 1:
            return None
        layer = int(getattr(attn, "_exo_glm52_layer_idx", -1))
        if not bool(getattr(attn, "_exo_glm52_profile_selected", False)):
            return None
        calls = int(getattr(attn, "_exo_glm52_profile_calls", 0)) + 1
        setattr(attn, "_exo_glm52_profile_calls", calls)
        if calls % cfg.profile_every != 0:
            return None
        sample = int(getattr(attn, "_exo_glm52_profile_samples", 0)) + 1
        setattr(attn, "_exo_glm52_profile_samples", sample)
        return cls(
            enabled=True,
            sync=cfg.profile_sync,
            layer=layer,
            layer_type=str(getattr(attn, "_exo_glm52_indexer_type", "unknown")),
            offset=_safe_int(offset, default=-1),
            qlen=qlen,
            kvlen=kvlen,
            rank=_best_effort_rank(),
            warmup=sample <= 2,
            logger=logger,
        )

    def stage(self, name: str, *values: Any) -> None:
        if not self.enabled:
            return
        if self.sync:
            arrays = _flatten_eval_values(values)
            if arrays:
                mx.eval(*arrays)
        now = time.perf_counter()
        self._stages.append((name, (now - self._last) * 1000.0, mlx_memory_stats()))
        self._last = now

    def finish(self, *values: Any, path: str) -> None:
        if not self.enabled:
            return
        if self.sync:
            arrays = _flatten_eval_values(values)
            if arrays:
                mx.eval(*arrays)
        now = time.perf_counter()
        memory = mlx_memory_stats()
        stages = " ".join(
            f"{name}_ms={elapsed:.3f} "
            f"{name}_active={stats['active_bytes']} "
            f"{name}_peak={stats['peak_bytes']} "
            f"{name}_cache={stats['cache_bytes']}"
            for name, elapsed, stats in self._stages
        )
        self._logger.info(
            "[PROFILE] "
            f"rank={self.rank} layer={self.layer} layer_type={self.layer_type} "
            f"path={path} warmup={int(self.warmup)} sync={int(self.sync)} "
            f"chunk_offset={self.offset} qlen={self.qlen} kvlen={self.kvlen} "
            f"{stages} attention_total_ms={(now - self._start) * 1000.0:.3f} "
            f"active_bytes={memory['active_bytes']} peak_bytes={memory['peak_bytes']} "
            f"cache_bytes={memory['cache_bytes']}"
        )


def _best_effort_rank() -> int:
    for name in ("EXO_RANK", "RANK", "JACCL_RANK", "OMPI_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return -1


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def set_cache_growth(cache_list: Any, cfg: GLM52PrefillConfig) -> None:
    """Apply a per-instance growth quantum without touching mlx-lm globally."""

    if cfg.cache_growth <= 0 or cache_list is None:
        return
    try:
        caches = list(cache_list)
    except TypeError:
        return
    for cache in caches:
        if cache is None or not hasattr(cache, "step"):
            continue
        if not getattr(cache, "_exo_glm52_growth_configured", False):
            try:
                cache.step = cfg.cache_growth
                cache._exo_glm52_growth_configured = True
            except Exception:  # pragma: no cover - defensive, fallback is upstream policy
                continue


def shared_cache_placeholder_width(attn: Any, cfg: GLM52PrefillConfig) -> int:
    if cfg.shared_index_cache == "compact":
        # Width one preserves ordinary KVCache state/filter/trim semantics while
        # reducing unused shared-indexer history by index_head_dim×. A true
        # OffsetOnly cache needs changes at cache construction and is intentionally
        # not impersonated here.
        return 1
    return max(0, _safe_int(getattr(attn, "_exo_glm52_index_head_dim", 0), default=0))


def cache_has_active_right_padding(cache: Any) -> bool:
    """Return whether a pinned mlx-lm batched cache is in right-padded prefill."""

    if cache is None:
        return False
    for name in ("right_padding", "_right_padding", "_lengths"):
        try:
            if getattr(cache, name, None) is not None:
                return True
        except Exception:  # pragma: no cover - hostile proxy objects are fail-safe
            continue
    return False


def cache_requires_dense_prefill(cache: Any) -> bool:
    """Fail closed for batched padding whose all-masked rows are not sparse-equivalent.

    The pinned masked-dense path applies a finite additive mask across the
    full key axis.  A completely masked row is therefore still normalized over
    full ``N`` (kernel/dtype details decide whether the content contribution is
    numerically retained), while a top-k-only path normalizes over ``K``.  It
    cannot guarantee the same output without full-N attention.  BatchKVCache and
    BatchRotatingKVCache expose ``left_padding``; active right padding is also
    checked explicitly for forward-compatible cache wrappers.
    """

    if cache is None:
        return False
    try:
        if hasattr(cache, "left_padding"):
            return True
    except Exception:  # pragma: no cover - hostile proxy objects are fail-safe
        return True
    return cache_has_active_right_padding(cache)


def mask_is_supported(mask: Any, *, batch: int, query_length: int, kv_length: int) -> bool:
    if mask is None:
        return True
    try:
        _slice_boolean_mask(
            mask,
            batch=batch,
            q_start=0,
            q_end=query_length,
            k_start=0,
            k_end=kv_length,
        )
    except UnsupportedPrefillMask:
        return False
    return True


def _is_boolean_dtype(value: Any) -> bool:
    try:
        return value.dtype == mx.bool_
    except Exception:
        return False


def _slice_boolean_mask(
    mask: Any,
    *,
    batch: int,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
) -> mx.array | None:
    if mask is None:
        return None
    if isinstance(mask, str):
        raise UnsupportedPrefillMask(f"string mask {mask!r} is not an exact boolean array")
    if not _is_boolean_dtype(mask):
        raise UnsupportedPrefillMask(f"non-boolean mask dtype {getattr(mask, 'dtype', None)!r}")

    ndim = int(getattr(mask, "ndim", -1))
    shape = tuple(int(x) for x in getattr(mask, "shape", ()))
    if ndim == 2:
        if shape[0] < q_end or shape[1] < k_end:
            raise UnsupportedPrefillMask(
                f"2D mask shape {shape} is smaller than q={q_end}, kv={k_end}"
            )
        view = mask[q_start:q_end, k_start:k_end][None, None, :, :]
    elif ndim == 3:
        if shape[0] not in (1, batch) or shape[1] < q_end or shape[2] < k_end:
            raise UnsupportedPrefillMask(
                f"3D mask shape {shape} is incompatible with batch={batch}, q={q_end}, kv={k_end}"
            )
        view = mask[:, q_start:q_end, k_start:k_end][:, None, :, :]
    elif ndim == 4:
        if (
            shape[0] not in (1, batch)
            or shape[1] != 1
            or shape[2] < q_end
            or shape[3] < k_end
        ):
            raise UnsupportedPrefillMask(
                f"4D mask shape {shape} is incompatible with batch={batch}, q={q_end}, kv={k_end}"
            )
        view = mask[:, :, q_start:q_end, k_start:k_end]
    else:
        raise UnsupportedPrefillMask(f"unsupported mask rank {ndim} shape={shape}")

    if int(view.shape[0]) == 1 and batch > 1:
        view = mx.broadcast_to(
            view,
            (batch, 1, q_end - q_start, k_end - k_start),
        )
    return view


def batched_take_sequence(x: mx.array, idx: mx.array) -> mx.array:
    """Gather per-batch sequence positions without a D-wide index tensor.

    Args:
        x: ``[B, 1, N, D]``.
        idx: ``[B, 1, Q, K]``.

    Returns:
        ``[B, 1, Q, K, D]``.
    """

    if x.ndim != 4 or idx.ndim != 4:
        raise ValueError(f"expected rank-4 x/idx, got {x.shape=} {idx.shape=}")
    batch, singleton, kv_length, width = (int(v) for v in x.shape)
    idx_batch, idx_singleton, query_length, topk = (int(v) for v in idx.shape)
    if singleton != 1 or idx_singleton != 1 or idx_batch != batch:
        raise ValueError(f"invalid gather shapes: x={x.shape}, idx={idx.shape}")

    x_flat = x[:, 0].reshape(batch * kv_length, width)
    # Contexts are bounded far below int32 max. Normalize argpartition's
    # backend-dependent signed/unsigned index dtype before adding B*N offsets,
    # otherwise a narrow unsigned dtype could wrap beyond 64k positions.
    idx_flat = idx[:, 0].reshape(batch, query_length * topk).astype(mx.int32)
    offsets = (mx.arange(batch).astype(mx.int32) * kv_length)[:, None]
    absolute = (idx_flat + offsets).reshape(-1)
    gathered = mx.take(x_flat, absolute, axis=0)
    return gathered.reshape(batch, 1, query_length, topk, width)


def gather_selected_mask(
    mask: Any,
    idx: mx.array,
    *,
    batch: int,
    q_start: int,
    q_end: int,
    kv_length: int,
) -> mx.array | None:
    if mask is None:
        return None
    full_row_mask = _slice_boolean_mask(
        mask,
        batch=batch,
        q_start=q_start,
        q_end=q_end,
        k_start=0,
        k_end=kv_length,
    )
    if full_row_mask is None:
        return None
    return mx.take_along_axis(full_row_mask, idx, axis=-1)


def _reduced_index_scores(
    q: mx.array,
    k: mx.array,
    weights: mx.array,
) -> mx.array:
    scores = q @ k.swapaxes(-1, -2)
    scores = mx.maximum(scores, 0)
    scores = scores * weights
    return scores.sum(axis=1, keepdims=True)


def _apply_index_mask(
    scores: mx.array,
    mask: Any,
    *,
    batch: int,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
) -> mx.array:
    selected = _slice_boolean_mask(
        mask,
        batch=batch,
        q_start=q_start,
        q_end=q_end,
        k_start=k_start,
        k_end=k_end,
    )
    if selected is None:
        return scores
    # Match the pinned mlx-lm Indexer exactly: it uses negative infinity here.
    return mx.where(selected, scores, -float("inf"))


def profiled_monolithic_indexer_call(
    indexer: Any,
    x: mx.array,
    qr: mx.array,
    mask: Any,
    cache: Any,
    *,
    profiler: PrefillProfileRecorder,
) -> mx.array | None:
    """Pinned Indexer.__call__ with intrusive barriers around each W1 stage."""

    batch, query_length, _ = (int(v) for v in x.shape)
    offset = cache.offset if cache is not None else 0
    k = indexer.wk(x)
    k = indexer.k_norm(k)
    k = mx.reshape(k, (batch, 1, query_length, indexer.head_dim))
    k = indexer.rope(k, offset=offset)
    if cache is not None:
        k, _ = cache.update_and_fetch(
            k,
            mx.zeros((batch, 1, query_length, 0)),
        )
    profiler.stage("idx_k", k)

    if int(k.shape[2]) <= int(indexer.index_topk):
        return None

    q = indexer.wq_b(qr)
    q = q.reshape(
        batch, query_length, indexer.n_heads, indexer.head_dim
    ).swapaxes(1, 2)
    q = indexer.rope(q, offset=offset)
    scores = q @ k.swapaxes(-1, -2)
    scores = mx.maximum(scores, 0)
    weights = indexer.weights_proj(x) * (
        indexer.n_heads**-0.5 * indexer.softmax_scale
    )
    weights = weights.swapaxes(-1, -2)[..., None]
    scores = (scores * weights).sum(axis=1, keepdims=True)
    profiler.stage("idx_score", scores)

    if mask is not None:
        scores = mx.where(mask, scores, -float("inf"))
    profiler.stage("idx_mask", scores)

    selected = mx.argpartition(
        scores, kth=-int(indexer.index_topk), axis=-1
    )[..., -int(indexer.index_topk) :]
    profiler.stage("idx_argpart", selected)
    return selected



def tiled_indexer_call(
    indexer: Any,
    x: mx.array,
    qr: mx.array,
    mask: Any,
    cache: Any,
    cfg: GLM52PrefillConfig,
    *,
    profiler: PrefillProfileRecorder | None = None,
) -> mx.array | None:
    """Pinned Indexer.__call__ semantics with 2D query/key tiling."""

    if not cfg.indexer_enabled:
        return indexer(x, qr, mask, cache=cache)

    batch, query_length, _ = (int(v) for v in x.shape)
    offset = cache.offset if cache is not None else 0
    if mask is not None and not isinstance(mask, str) and getattr(mask, "shape", None):
        expected_kv = int(mask.shape[-1])
    else:
        expected_kv = _safe_int(offset, default=0) + query_length
    if not mask_is_supported(
        mask,
        batch=batch,
        query_length=query_length,
        kv_length=expected_kv,
    ):
        default_logger.warning(
            "[EXO][GLM-5.2][Prefill] tiled indexer encountered an unsupported "
            f"mask; falling back to pinned monolithic Indexer for shape={getattr(mask, 'shape', None)}"
        )
        return indexer(x, qr, mask, cache=cache)

    k = indexer.wk(x)
    k = indexer.k_norm(k)
    k = mx.reshape(k, (batch, 1, query_length, indexer.head_dim))
    k = indexer.rope(k, offset=offset)
    if profiler is not None:
        profiler.stage("idx_k_proj_norm_rope", k)

    if cache is not None:
        k, _ = cache.update_and_fetch(
            k,
            mx.zeros((batch, 1, query_length, 0)),
        )
    if profiler is not None:
        profiler.stage("idx_k_cache_update", k)

    kv_length = int(k.shape[2])
    if kv_length <= int(indexer.index_topk):
        return None

    q = indexer.wq_b(qr)
    q = q.reshape(batch, query_length, indexer.n_heads, indexer.head_dim).swapaxes(1, 2)
    q = indexer.rope(q, offset=offset)
    weights = indexer.weights_proj(x) * (
        indexer.n_heads**-0.5 * indexer.softmax_scale
    )
    weights = weights.swapaxes(-1, -2)[..., None]
    if profiler is not None:
        profiler.stage("idx_q_proj_rope", q, weights)

    topk = int(indexer.index_topk)
    query_outputs: list[mx.array] = []
    for q_start in range(0, query_length, cfg.indexer_q_chunk):
        q_end = min(query_length, q_start + cfg.indexer_q_chunk)
        q_block = q[:, :, q_start:q_end, :]
        w_block = weights[:, :, q_start:q_end, :]

        if cfg.indexer_mode == "reference":
            reduced_slices: list[mx.array] = []
            for k_start in range(0, kv_length, cfg.indexer_k_chunk):
                k_end = min(kv_length, k_start + cfg.indexer_k_chunk)
                reduced = _reduced_index_scores(
                    q_block,
                    k[:, :, k_start:k_end, :],
                    w_block,
                )
                reduced = _apply_index_mask(
                    reduced,
                    mask,
                    batch=batch,
                    q_start=q_start,
                    q_end=q_end,
                    k_start=k_start,
                    k_end=k_end,
                )
                if cfg.indexer_eval_chunks:
                    mx.eval(reduced)
                reduced_slices.append(reduced)
            all_scores = (
                reduced_slices[0]
                if len(reduced_slices) == 1
                else mx.concatenate(reduced_slices, axis=-1)
            )
            if profiler is not None:
                profiler.stage("idx_score_tiled", all_scores)
            selected = mx.argpartition(all_scores, kth=-topk, axis=-1)[..., -topk:]
            if profiler is not None:
                profiler.stage("idx_argpart", selected)
            query_outputs.append(selected)
            continue

        # Streaming candidate: never materialize [B,1,Q,N]. Ordering at ties is
        # intentionally unspecified, exactly as with argpartition itself.
        best_scores: mx.array | None = None
        best_indices: mx.array | None = None
        for k_start in range(0, kv_length, cfg.indexer_k_chunk):
            k_end = min(kv_length, k_start + cfg.indexer_k_chunk)
            reduced = _reduced_index_scores(
                q_block,
                k[:, :, k_start:k_end, :],
                w_block,
            )
            reduced = _apply_index_mask(
                reduced,
                mask,
                batch=batch,
                q_start=q_start,
                q_end=q_end,
                k_start=k_start,
                k_end=k_end,
            )
            positions = mx.arange(k_start, k_end).astype(mx.int32)[None, None, None, :]
            positions = mx.broadcast_to(positions, reduced.shape)
            if best_scores is not None and best_indices is not None:
                reduced = mx.concatenate((best_scores, reduced), axis=-1)
                positions = mx.concatenate((best_indices, positions), axis=-1)

            keep_count = min(topk, int(reduced.shape[-1]))
            if int(reduced.shape[-1]) > keep_count:
                keep = mx.argpartition(reduced, kth=-keep_count, axis=-1)[
                    ..., -keep_count:
                ]
                best_scores = mx.take_along_axis(reduced, keep, axis=-1)
                best_indices = mx.take_along_axis(positions, keep, axis=-1)
            else:
                best_scores = reduced
                best_indices = positions
            if cfg.indexer_eval_chunks:
                mx.eval(best_scores, best_indices)

        if best_indices is None:
            raise RuntimeError("streaming indexer produced no candidates")
        if profiler is not None:
            profiler.stage(
                "idx_score_streaming",
                best_indices,
            )
        query_outputs.append(best_indices)

    return query_outputs[0] if len(query_outputs) == 1 else mx.concatenate(query_outputs, axis=2)


def sparse_prefill_attention(
    attn: Any,
    *,
    q_nope: mx.array,
    q_pe: mx.array,
    kv_latent: mx.array,
    k_pe: mx.array,
    topk_indices: mx.array,
    mask: Any,
    scaled_dot_product_attention: Any,
    cfg: GLM52PrefillConfig,
    profiler: PrefillProfileRecorder | None = None,
) -> mx.array:
    """True sparse GLM-5.2 prefill using per-query gathered latent/PE keys."""

    batch, heads, query_length, _ = (int(v) for v in q_nope.shape)
    kv_length = int(kv_latent.shape[2])
    if tuple(int(v) for v in topk_indices.shape[:3]) != (batch, 1, query_length):
        raise UnsupportedPrefillMask(
            "top-k shape does not match sparse prefill query: "
            f"topk={topk_indices.shape}, expected=({batch}, 1, {query_length}, K)"
        )
    if query_length <= 1:
        raise ValueError("sparse_prefill_attention is prefill-only; query_length must exceed one")
    if not mask_is_supported(
        mask,
        batch=batch,
        query_length=query_length,
        kv_length=kv_length,
    ):
        raise UnsupportedPrefillMask(
            f"cannot preserve mask semantics for shape={getattr(mask, 'shape', None)}"
        )

    q_latent = attn.embed_q(q_nope)
    if profiler is not None:
        profiler.stage("sparse_q_absorb", q_latent)

    output_blocks: list[mx.array] = []
    for q_start in range(0, query_length, cfg.sparse_q_chunk):
        q_end = min(query_length, q_start + cfg.sparse_q_chunk)
        block_q = q_end - q_start
        idx = topk_indices[:, :, q_start:q_end, :]
        topk = int(idx.shape[-1])

        latent_gathered = batched_take_sequence(kv_latent, idx)
        pe_gathered = batched_take_sequence(k_pe, idx)
        selected_mask = gather_selected_mask(
            mask,
            idx,
            batch=batch,
            q_start=q_start,
            q_end=q_end,
            kv_length=kv_length,
        )
        if profiler is not None:
            profiler.stage("sparse_gather", latent_gathered, pe_gathered, selected_mask)

        q_latent_flat = (
            q_latent[:, :, q_start:q_end, :]
            .transpose(0, 2, 1, 3)
            .reshape(batch * block_q, heads, 1, int(q_latent.shape[-1]))
        )
        latent_flat = latent_gathered[:, 0].reshape(
            batch * block_q, topk, int(latent_gathered.shape[-1])
        )[:, None, :, :]

        q_pe_flat = (
            q_pe[:, :, q_start:q_end, :]
            .transpose(0, 2, 1, 3)
            .reshape(batch * block_q, heads, 1, int(q_pe.shape[-1]))
        )
        pe_flat = pe_gathered[:, 0].reshape(
            batch * block_q, topk, int(pe_gathered.shape[-1])
        )[:, None, :, :]
        pe_scores = (q_pe_flat * attn.scale) @ pe_flat.swapaxes(-1, -2)

        if selected_mask is not None:
            selected_mask = selected_mask.reshape(batch * block_q, 1, 1, topk)
            pe_scores = mx.where(
                selected_mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )
        if profiler is not None:
            profiler.stage("sparse_pe_scores", pe_scores)

        # The existing wrapper keeps the content scaling and fp32 SDPA softmax
        # identical to the absorbed decode path. PE is already scaled above.
        block_output = scaled_dot_product_attention(
            q_latent_flat,
            latent_flat,
            latent_flat,
            cache=None,
            scale=attn.scale,
            mask=pe_scores,
        )
        block_output = (
            block_output.reshape(batch, block_q, heads, int(block_output.shape[-1]))
            .transpose(0, 2, 1, 3)
        )
        if cfg.sparse_eval_blocks:
            mx.eval(block_output)
        if profiler is not None:
            profiler.stage("sparse_sdpa", block_output)
        output_blocks.append(block_output)

    output_latent = (
        output_blocks[0]
        if len(output_blocks) == 1
        else mx.concatenate(output_blocks, axis=2)
    )
    output = attn.unembed_out(output_latent)
    output = output.transpose(0, 2, 1, 3).reshape(batch, query_length, -1)
    output = attn.o_proj(output)
    if profiler is not None:
        profiler.stage("unembed_o_proj_collective", output)
    return output
