from __future__ import annotations

import importlib.util
from pathlib import Path


BUNDLE_FILES = Path(__file__).resolve().parents[1]
REPORT_PATH = BUNDLE_FILES / "tools" / "glm52_prefill_report.py"
SPEC = importlib.util.spec_from_file_location("glm52_prefill_report", REPORT_PATH)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_report_parser_separates_prefill_and_profile(tmp_path: Path) -> None:
    log = tmp_path / "exo.log"
    log.write_text(
        "\n".join(
            [
                '[PREFILL_CONFIG] rank=0/4 fingerprint=abc prefill_step=4096 env={"EXO_PREFILL_STEP_RESOLVED": "4096"}',
                "[PREFILL] processed=4096 total=8192 delta_tokens=4096 delta_ms=1000.0 segment_tps=4096.0 cumulative_tps=4096.0 elapsed_ms=1000.0 peak_bytes=20 active_bytes=10 cache_bytes=5",
                "[PREFILL_SUMMARY] processed=8192 total=8192 elapsed_ms=2500.0 cumulative_tps=3276.8 tail_segment_tps_median=3500.0 tail_segments=[4096.0, 2904.0]",
                "[PROFILE] rank=0 layer=0 layer_type=full path=sparse warmup=0 sync=1 chunk_offset=4096 qlen=4096 kvlen=8192 sparse_gather_ms=12.5 attention_total_ms=30.0 active_bytes=10 peak_bytes=20 cache_bytes=5",
            ]
        ),
        encoding="utf-8",
    )
    parsed = report.parse_logs([str(log)])
    assert len(parsed["config"]) == 1
    assert parsed["config"][0]["prefill_step"] == 4096
    assert len(parsed["prefill"]) == 1
    assert parsed["prefill"][0]["segment_tps"] == 4096.0
    assert parsed["summary"][0]["tail_segments"] == [4096.0, 2904.0]
    assert parsed["profile"][0]["sparse_gather_ms"] == 12.5
    markdown = report.render_markdown(parsed)
    assert "Prefill summaries" in markdown
    assert "sparse_gather:12.50" in markdown
