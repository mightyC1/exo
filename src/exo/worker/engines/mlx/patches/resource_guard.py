"""Metal resource-handle guard for long decode runs.

Problem
-------
MLX's BufferCache evicts by BYTES (``cache_limit``). Decode-step allocations
are tiny, so during a long generation the byte total never crosses the
eviction threshold while the *number* of live MTLBuffer handles grows
monotonically with each step — until it hits the device ``resource_limit``
(499000 on M3 Ultra) and Metal refuses to allocate:

    RuntimeError: [metal::malloc] Resource limit (499000) exceeded.

This is a handle-count exhaustion, not an out-of-memory: it fires with
single-digit GiB resident.

Fix
---
Periodically drop the buffer cache. ``clear_cache()`` releases only buffers
that are already free (nothing live is touched), it is local to the process
allocator, and it needs no collective barrier across ranks — so it is safe to
call from every rank at the same point in the step loop.

Knobs
-----
EXO_MLX_CACHE_DRAIN_EVERY   steps between drains. Default 512. 0 disables.
                            Lower it (128) under heavy concurrent batching:
                            handle growth scales with batch width.
EXO_MLX_MEM_LOG_EVERY       steps between memory log lines. Default 0 (off).
                            Use it to tell a cache leak (cache_bytes grows)
                            from a real leak (active_bytes grows).
"""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return value if value >= 0 else default


_DRAIN_EVERY = _env_int("EXO_MLX_CACHE_DRAIN_EVERY", 512)
_LOG_EVERY = _env_int("EXO_MLX_MEM_LOG_EVERY", 0)

_steps = 0
_drains = 0
_resolved = False
_clear_cache = None
_get_active = None
_get_cache = None
_get_peak = None


def _log(message: str) -> None:
    try:
        from loguru import logger

        logger.info(message)
    except Exception:
        print(message, flush=True)


def _resolve() -> None:
    """Bind MLX memory helpers.

    They live at ``mx.*`` in current MLX and at ``mx.metal.*`` in older
    builds; resolve both rather than assuming either.
    """
    global _resolved, _clear_cache, _get_active, _get_cache, _get_peak

    import mlx.core as mx

    metal = getattr(mx, "metal", None)

    def pick(name: str):
        fn = getattr(mx, name, None)
        if fn is None and metal is not None:
            fn = getattr(metal, name, None)
        return fn

    _clear_cache = pick("clear_cache")
    _get_active = pick("get_active_memory")
    _get_cache = pick("get_cache_memory")
    _get_peak = pick("get_peak_memory")
    _resolved = True

    if _clear_cache is None:
        _log("[RESGUARD] mx.clear_cache() not found — guard is a no-op")
    else:
        _log(
            f"[RESGUARD] armed: drain_every={_DRAIN_EVERY} log_every={_LOG_EVERY}"
        )


def tick() -> None:
    """Call once per decode step, after the step's tokens are materialised."""
    global _steps, _drains

    _steps += 1

    if not (_DRAIN_EVERY or _LOG_EVERY):
        return

    if not _resolved:
        try:
            _resolve()
        except Exception as exc:  # never take the runner down over telemetry
            _log(f"[RESGUARD] disabled, resolve failed: {exc!r}")
            globals()["_DRAIN_EVERY"] = 0
            globals()["_LOG_EVERY"] = 0
            globals()["_resolved"] = True
            return

    if _LOG_EVERY and _steps % _LOG_EVERY == 0:
        try:
            active = _get_active() if _get_active else -1
            cached = _get_cache() if _get_cache else -1
            peak = _get_peak() if _get_peak else -1
            _log(
                f"[RESGUARD] step={_steps} drains={_drains} "
                f"active_bytes={active} cache_bytes={cached} peak_bytes={peak}"
            )
        except Exception:
            pass

    if _DRAIN_EVERY and _clear_cache and _steps % _DRAIN_EVERY == 0:
        try:
            _clear_cache()
            _drains += 1
        except Exception as exc:
            _log(f"[RESGUARD] clear_cache failed: {exc!r}")


def reset() -> None:
    """Reset the step counter (call between independent generations)."""
    global _steps
    _steps = 0


def stats() -> dict:
    return {
        "steps": _steps,
        "drains": _drains,
        "drain_every": _DRAIN_EVERY,
        "log_every": _LOG_EVERY,
        "armed": _clear_cache is not None,
    }
