"""Snapshot policy for SSM/recurrent prefill checkpoints.

SSM caches (ArraysCache: KDA/GDN state) cannot be trimmed backwards, so the
prefix cache can only resume from an explicit snapshot. Snapshots are taken
during prefill; each one is a full per-layer state copy (~220MB/rank on K3),
which dominates long-context memory growth.

EXO_SSM_SNAPSHOT_EVERY:
    "1" (default)  snapshot every prefill progress tick (legacy behaviour)
    "K" (int > 1)  snapshot every K-th tick PLUS always the final tick, so
                   exact-continuation reuse is byte-identical to legacy
    "0"/"off"      only the mandatory rollback pair (last two ticks); callers skip PARKING
                   SSM entries (without a snapshot they can never be reused,
                   see KVPrefixCache._get_snapshot -> (0, None) -> fresh cache)
"""

from __future__ import annotations

import os

_OFF = {"0", "off", "none", "disable", "disabled"}


def snapshot_every() -> int:
    """0 = disabled, N >= 1 = every N-th tick (read per call: cheap, testable)."""
    raw = os.environ.get("EXO_SSM_SNAPSHOT_EVERY", "1").strip().lower()
    if raw in _OFF:
        return 0
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def snapshots_enabled() -> bool:
    return snapshot_every() > 0


def should_snapshot(tick_index: int, processed: int, total: int) -> bool:
    """tick_index is 1-based.

    The last TWO ticks (processed >= total - 1) are snapshotted UNCONDITIONALLY,
    in every mode including off: prefill's "+2 rollback" restores SSM state
    from snapshots[-2] (state at total-1) because recurrent caches cannot be
    trimmed backwards (see generate.py prefill / is_non_trimmable_cache_entry).
    These two are a correctness requirement, not a reuse optimization.
    """
    if processed >= total - 1:
        return True
    every = snapshot_every()
    if every == 0:
        return False
    if every == 1:
        return True
    return tick_index % every == 0
