#!/usr/bin/env python3
"""Run repeatable cold-prefill requests against EXO's OpenAI-compatible API.

Prompt files should already contain the desired synthetic text. The script records
actual ``usage.prompt_tokens`` returned by EXO; it does not pretend that character
count equals tokenizer count.

Example:
    python tools/glm52_prefill_bench.py \
      --model local/GLM-5.2-8bit-idxbf16 \
      --case 16k=/tmp/prompt-16k.txt \
      --case 48k=/tmp/prompt-48k.txt \
      --case 96k=/tmp/prompt-96k.txt \
      --case 135k=/tmp/prompt-135k.txt \
      --runs 2 --output /tmp/glm52-bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Result:
    case: str
    run: int
    prompt_file: str
    prompt_bytes: int
    started_unix: float
    status: str
    http_status: int | None
    ttft_seconds: float | None
    total_seconds: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None


def parse_case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be LABEL=/path/to/prompt.txt")
    label, raw_path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("case label is empty")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"prompt file not found: {path}")
    return label.strip(), path


def _extract_usage(payload: Any) -> tuple[int | None, int | None]:
    if not isinstance(payload, dict):
        return None, None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    return (
        int(prompt) if isinstance(prompt, (int, float)) else None,
        int(completion) if isinstance(completion, (int, float)) else None,
    )


def _contains_generated_token(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    for choice in payload.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or choice.get("message") or {}
        if not isinstance(delta, dict):
            continue
        for key in ("content", "reasoning_content", "reasoning", "text"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                return True
    # Responses-style event compatibility.
    event_type = payload.get("type")
    if isinstance(event_type, str) and event_type.endswith((".delta", ".done")):
        for key in ("delta", "text"):
            if isinstance(payload.get(key), str) and payload[key]:
                return True
    return False


def _iter_sse(response: Any) -> Iterable[dict[str, Any]]:
    data_lines: list[str] = []
    while True:
        raw = response.readline()
        if not raw:
            if data_lines:
                joined = "\n".join(data_lines)
                if joined != "[DONE]":
                    yield json.loads(joined)
            return
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                joined = "\n".join(data_lines)
                data_lines.clear()
                if joined == "[DONE]":
                    return
                yield json.loads(joined)
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())


def run_request(
    *,
    url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    timeout: float,
    extra: dict[str, Any],
) -> tuple[int, float | None, float, int | None, int | None]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        # EXO uses these fields to avoid prefix-cache contamination in benchmarks.
        "bench": True,
        "use_prefix_cache": False,
    }
    body.update(extra)
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=encoded, headers=headers, method="POST")

    started = time.perf_counter()
    first_token: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = int(getattr(response, "status", 200))
        content_type = response.headers.get_content_type()
        if content_type == "text/event-stream" or "event-stream" in response.headers.get("Content-Type", ""):
            for payload in _iter_sse(response):
                now = time.perf_counter()
                if first_token is None and _contains_generated_token(payload):
                    first_token = now - started
                p, c = _extract_usage(payload)
                prompt_tokens = p if p is not None else prompt_tokens
                completion_tokens = c if c is not None else completion_tokens
        else:
            payload = json.loads(response.read().decode("utf-8"))
            first_token = time.perf_counter() - started
            prompt_tokens, completion_tokens = _extract_usage(payload)
    return status, first_token, time.perf_counter() - started, prompt_tokens, completion_tokens


def load_extra(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    candidate = Path(raw).expanduser()
    text = candidate.read_text(encoding="utf-8") if candidate.is_file() else raw
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("--extra-json must decode to an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:52415/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--extra-json", help="inline JSON object or path; overrides defaults")
    parser.add_argument("--output", type=Path, required=True, help="append-only JSONL result file")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    extra = load_extra(args.extra_json)
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("a", encoding="utf-8") as handle:
        for label, path in args.case:
            prompt = path.read_text(encoding="utf-8")
            prompt_bytes = len(prompt.encode("utf-8"))
            for run in range(1, args.runs + 1):
                result = Result(
                    case=label,
                    run=run,
                    prompt_file=str(path.resolve()),
                    prompt_bytes=prompt_bytes,
                    started_unix=time.time(),
                    status="error",
                    http_status=None,
                    ttft_seconds=None,
                    total_seconds=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    error=None,
                )
                try:
                    status, ttft, total, prompt_tokens, completion_tokens = run_request(
                        url=args.url,
                        api_key=args.api_key,
                        model=args.model,
                        prompt=prompt,
                        timeout=args.timeout,
                        extra=extra,
                    )
                    result.status = "ok"
                    result.http_status = status
                    result.ttft_seconds = ttft
                    result.total_seconds = total
                    result.prompt_tokens = prompt_tokens
                    result.completion_tokens = completion_tokens
                except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                    result.error = f"{type(exc).__name__}: {exc}"
                line = json.dumps(asdict(result), ensure_ascii=False, sort_keys=True)
                handle.write(line + "\n")
                handle.flush()
                print(line, flush=True)
                if result.status != "ok" and not args.continue_on_error:
                    return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
