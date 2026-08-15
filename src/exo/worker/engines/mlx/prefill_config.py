"""Shared prefill configuration and telemetry for EXO's MLX generator."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


_DEFAULT_PREFILL_STEP = 4096
_MIN_PREFILL_STEP = 512
_MAX_PREFILL_STEP = 16384


def resolve_prefill_step(logger: Any) -> int:
    """Read EXO_PREFILL_STEP once per prefill and clamp it safely."""

    raw = os.environ.get("EXO_PREFILL_STEP")
    if raw is None:
        return _DEFAULT_PREFILL_STEP
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        logger.warning(
            f"[PREFILL_CONFIG] invalid EXO_PREFILL_STEP={raw!r}; "
            f"using {_DEFAULT_PREFILL_STEP}"
        )
        return _DEFAULT_PREFILL_STEP
    clamped = max(_MIN_PREFILL_STEP, min(_MAX_PREFILL_STEP, value))
    if clamped != value:
        logger.warning(
            f"[PREFILL_CONFIG] clamped EXO_PREFILL_STEP={value} to {clamped}"
        )
    return clamped


def _relevant_prefill_environment(prefill_step: int) -> dict[str, str]:
    values = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("EXO_GLM52_")
        or key.startswith("EXO_GLM_")
        or key.startswith("EXO_PREFILL_")
    }
    values["EXO_PREFILL_STEP_RESOLVED"] = str(prefill_step)
    return dict(sorted(values.items()))


def prefill_config_fingerprint(prefill_step: int) -> tuple[str, dict[str, str]]:
    values = _relevant_prefill_environment(prefill_step)
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), values


def assert_distributed_prefill_config(
    group: mx.distributed.Group | None,
    prefill_step: int,
    logger: Any,
) -> str:
    """Fail before model execution if ranks resolved different prefill settings."""

    fingerprint, values = prefill_config_fingerprint(prefill_step)
    digest = bytes.fromhex(fingerprint)
    # Use positive int32 words: JACCL/MLX collectives support int32 broadly,
    # while uint32 support has varied across backend revisions.
    fingerprint_words = [
        int.from_bytes(digest[i : i + 2], "big") for i in range(0, len(digest), 2)
    ]

    if group is not None and group.size() > 1:
        local = mx.array(fingerprint_words, dtype=mx.int32)
        gathered = mx.distributed.all_gather(local, group=group)
        mx.eval(gathered)
        rank_words = gathered.reshape(group.size(), len(fingerprint_words)).tolist()
        rank_fingerprints = [
            "".join(f"{int(word):04x}" for word in words) for words in rank_words
        ]
        if any(value != fingerprint for value in rank_fingerprints):
            raise RuntimeError(
                "distributed prefill configuration mismatch before model call: "
                f"local={fingerprint[:16]} "
                f"gathered={[value[:16] for value in rank_fingerprints]} "
                f"resolved={values}"
            )

    rank = group.rank() if group is not None else 0
    size = group.size() if group is not None else 1
    logger.info(
        "[PREFILL_CONFIG] "
        f"rank={rank}/{size} fingerprint={fingerprint[:16]} "
        f"prefill_step={prefill_step} env={json.dumps(values, sort_keys=True)}"
    )
    return fingerprint


def mlx_memory_stats() -> dict[str, int]:
    result: dict[str, int] = {}
    for key, function_name in (
        ("active_bytes", "get_active_memory"),
        ("peak_bytes", "get_peak_memory"),
        ("cache_bytes", "get_cache_memory"),
    ):
        fn = getattr(mx, function_name, None)
        try:
            result[key] = int(fn()) if callable(fn) else -1
        except Exception:  # pragma: no cover - telemetry must not break inference
            result[key] = -1
    return result


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


@dataclass
class PrefillTelemetry:
    """Structured segment/cumulative timing emitted by the existing callback."""

    logger: Any
    expected_segment_tokens: int
    start_time: float = field(default_factory=time.perf_counter)
    previous_processed: int = 0
    previous_time: float | None = None
    full_segment_tps: list[float] = field(default_factory=list)
    final_logged: bool = False

    def __post_init__(self) -> None:
        if self.previous_time is None:
            self.previous_time = self.start_time
        self.expected_segment_tokens = max(1, int(self.expected_segment_tokens))

    def update(self, processed: int, total: int) -> None:
        now = time.perf_counter()
        previous_time = self.previous_time if self.previous_time is not None else self.start_time
        elapsed = max(0.0, now - self.start_time)
        delta_seconds = max(0.0, now - previous_time)
        delta_tokens = max(0, int(processed) - int(self.previous_processed))
        segment_tps = delta_tokens / delta_seconds if delta_seconds > 0 and delta_tokens else 0.0
        cumulative_tps = processed / elapsed if elapsed > 0 else 0.0
        memory = mlx_memory_stats()

        self.logger.info(
            "[PREFILL] "
            f"processed={processed} total={total} delta_tokens={delta_tokens} "
            f"delta_ms={delta_seconds * 1000.0:.3f} "
            f"segment_tps={segment_tps:.3f} cumulative_tps={cumulative_tps:.3f} "
            f"elapsed_ms={elapsed * 1000.0:.3f} "
            f"peak_bytes={memory['peak_bytes']} active_bytes={memory['active_bytes']} "
            f"cache_bytes={memory['cache_bytes']}"
        )

        # Exclude the final one/two-token cache bookkeeping callback. Allow an
        # incomplete last real chunk, but only if it is at least half a full one.
        if delta_tokens >= max(1, self.expected_segment_tokens // 2):
            self.full_segment_tps.append(segment_tps)

        if processed >= total and not self.final_logged:
            tail = self.full_segment_tps[-3:]
            tail_median = _median(tail) if tail else 0.0
            self.logger.info(
                "[PREFILL_SUMMARY] "
                f"processed={processed} total={total} elapsed_ms={elapsed * 1000.0:.3f} "
                f"cumulative_tps={cumulative_tps:.3f} "
                f"tail_segment_tps_median={tail_median:.3f} "
                f"tail_segments={json.dumps([round(v, 3) for v in tail])}"
            )
            self.final_logged = True

        self.previous_processed = max(self.previous_processed, int(processed))
        self.previous_time = now
