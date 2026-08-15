#!/usr/bin/env python3
"""Parse EXO GLM-5.2 structured prefill/profile log lines.

Examples:
    python tools/glm52_prefill_report.py /tmp/exo-node*.log
    python tools/glm52_prefill_report.py /tmp/exo.log --csv-dir /tmp/report
    log stream --style syslog --predicate 'process == "exo"' | \
        python tools/glm52_prefill_report.py -
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, TextIO

_MARKERS = ("[PREFILL_CONFIG]", "[PREFILL]", "[PREFILL_SUMMARY]", "[PROFILE]")
_TOKEN_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>\[[^\]]*\]|\{.*\}|\S+)")


def _coerce(value: str) -> Any:
    value = value.rstrip(",")
    if value in {"True", "False"}:
        return value == "True"
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_fields(text: str) -> dict[str, Any]:
    return {match.group("key"): _coerce(match.group("value")) for match in _TOKEN_RE.finditer(text)}


def _open_inputs(paths: list[str]) -> Iterable[tuple[str, TextIO]]:
    if not paths:
        paths = ["-"]
    for raw in paths:
        if raw == "-":
            yield "stdin", sys.stdin
            continue
        path = Path(raw).expanduser()
        handle = path.open("r", encoding="utf-8", errors="replace")
        try:
            yield str(path), handle
        finally:
            handle.close()


def parse_logs(paths: list[str]) -> dict[str, list[dict[str, Any]]]:
    parsed: dict[str, list[dict[str, Any]]] = {
        "config": [],
        "prefill": [],
        "summary": [],
        "profile": [],
    }
    marker_to_key = {
        "[PREFILL_CONFIG]": "config",
        "[PREFILL]": "prefill",
        "[PREFILL_SUMMARY]": "summary",
        "[PROFILE]": "profile",
    }
    for source, handle in _open_inputs(paths):
        for line_number, line in enumerate(handle, 1):
            marker = next((candidate for candidate in _MARKERS if candidate in line), None)
            if marker is None:
                continue
            payload = line.split(marker, 1)[1].strip()
            row = _parse_fields(payload)
            row["source"] = source
            row["line"] = line_number
            parsed[marker_to_key[marker]].append(row)
    return parsed


def _fmt_number(value: Any, digits: int = 2) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return "-"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def render_markdown(parsed: dict[str, list[dict[str, Any]]]) -> str:
    sections: list[str] = ["# GLM-5.2 prefill report"]

    summaries = parsed["summary"]
    summary_rows: list[list[str]] = []
    for row in summaries:
        elapsed_ms = row.get("elapsed_ms")
        summary_rows.append(
            [
                Path(str(row.get("source", ""))).name,
                str(row.get("line", "-")),
                str(row.get("processed", "-")),
                _fmt_number(float(elapsed_ms) / 60000.0 if isinstance(elapsed_ms, (int, float)) else None, 3),
                _fmt_number(row.get("cumulative_tps"), 3),
                _fmt_number(row.get("tail_segment_tps_median"), 3),
                json.dumps(row.get("tail_segments", []), ensure_ascii=False),
            ]
        )
    sections.extend(
        [
            "\n## Prefill summaries",
            _markdown_table(
                ["source", "line", "tokens", "elapsed min", "cumulative t/s", "tail median t/s", "tail segments"],
                summary_rows,
            ),
        ]
    )

    # Aggregate intrusive profile stages by path/layer/type. Warmup rows are shown
    # separately so compile effects are not silently mixed into steady state.
    profile_groups: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in parsed["profile"]:
        key = (
            Path(str(row.get("source", ""))).name,
            row.get("layer"),
            row.get("layer_type"),
            row.get("path"),
            row.get("warmup", 0),
        )
        for name, value in row.items():
            if name.endswith("_ms") and name != "attention_total_ms" and isinstance(value, (int, float)):
                profile_groups[key][name].append(float(value))
        if isinstance(row.get("attention_total_ms"), (int, float)):
            profile_groups[key]["attention_total_ms"].append(float(row["attention_total_ms"]))

    profile_rows: list[list[str]] = []
    for key, stages in sorted(profile_groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        source, layer, layer_type, path, warmup = key
        total_values = stages.get("attention_total_ms", [])
        stage_medians = sorted(
            (
                (name.removesuffix("_ms"), statistics.median(values))
                for name, values in stages.items()
                if name != "attention_total_ms" and values
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        profile_rows.append(
            [
                str(source),
                str(layer),
                str(layer_type),
                str(path),
                str(warmup),
                str(len(total_values)),
                _fmt_number(statistics.median(total_values) if total_values else None, 3),
                ", ".join(f"{name}:{value:.2f}" for name, value in stage_medians[:8]),
            ]
        )
    sections.extend(
        [
            "\n## Intrusive profile medians",
            _markdown_table(
                ["source", "layer", "type", "path", "warmup", "samples", "attention ms", "largest stages (ms)"],
                profile_rows,
            ),
        ]
    )

    if parsed["prefill"]:
        segment_values = [
            float(row["segment_tps"])
            for row in parsed["prefill"]
            if isinstance(row.get("segment_tps"), (int, float)) and int(row.get("delta_tokens", 0)) > 1
        ]
        if segment_values:
            sections.extend(
                [
                    "\n## Segment distribution",
                    f"Samples: **{len(segment_values)}**; median: **{statistics.median(segment_values):.3f} t/s**; "
                    f"min/max: **{min(segment_values):.3f}/{max(segment_values):.3f} t/s**.",
                ]
            )

    return "\n".join(sections) + "\n"


def write_csv(directory: Path, parsed: dict[str, list[dict[str, Any]]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in parsed.items():
        if not rows:
            continue
        columns = sorted({key for row in rows for key in row})
        with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                serializable = {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
                writer.writerow(serializable)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", help="EXO log files; '-' or no arguments reads stdin")
    parser.add_argument("--csv-dir", type=Path, help="also write raw parsed CSV tables")
    parser.add_argument("--json", action="store_true", help="emit parsed JSON instead of Markdown")
    args = parser.parse_args()

    parsed = parse_logs(args.logs)
    if args.csv_dir:
        write_csv(args.csv_dir.expanduser(), parsed)
    if args.json:
        json.dump(parsed, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_markdown(parsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
