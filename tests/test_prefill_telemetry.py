from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# The telemetry math itself is platform-neutral. Stub the MLX module on Linux so
# W0 remains covered in CI; distributed/allocator calls are monkey-patched below.
try:
    mlx_available = importlib.util.find_spec("mlx.core") is not None
except ModuleNotFoundError:
    mlx_available = False
if not mlx_available:
    mlx_package = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_package.core = mlx_core
    sys.modules.setdefault("mlx", mlx_package)
    sys.modules.setdefault("mlx.core", mlx_core)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from exo.worker.engines.mlx import prefill_config  # noqa: E402


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, value: str) -> None:
        self.lines.append(value)

    def warning(self, value: str) -> None:
        self.lines.append(value)


def test_segment_and_cumulative_metrics_are_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([1.0, 3.0, 4.0])
    monkeypatch.setattr(prefill_config.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(
        prefill_config,
        "mlx_memory_stats",
        lambda: {"active_bytes": 1, "peak_bytes": 2, "cache_bytes": 3},
    )
    logger = _Logger()
    telemetry = prefill_config.PrefillTelemetry(
        logger=logger,
        expected_segment_tokens=100,
        start_time=0.0,
        previous_time=0.0,
    )
    telemetry.update(100, 300)  # 100 t/s cumulative/segment
    telemetry.update(200, 300)  # 50 t/s segment, 66.7 cumulative
    telemetry.update(300, 300)  # 100 t/s segment, 75 cumulative

    prefill_lines = [line for line in logger.lines if line.startswith("[PREFILL] ")]
    assert "segment_tps=50.000" in prefill_lines[1]
    assert "cumulative_tps=66.667" in prefill_lines[1]
    summary = next(line for line in logger.lines if line.startswith("[PREFILL_SUMMARY]"))
    assert "tail_segment_tps_median=100.000" in summary


def test_prefill_step_validation_and_clamping(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _Logger()
    monkeypatch.setenv("EXO_PREFILL_STEP", "bad")
    assert prefill_config.resolve_prefill_step(logger) == 4096
    monkeypatch.setenv("EXO_PREFILL_STEP", "999999")
    assert prefill_config.resolve_prefill_step(logger) == 16384
    monkeypatch.setenv("EXO_PREFILL_STEP", "256")
    assert prefill_config.resolve_prefill_step(logger) == 512
