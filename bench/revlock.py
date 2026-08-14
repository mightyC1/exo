# type: ignore
#!/usr/bin/env python3
"""R0 revision lock for GLM-5.2 prefill benchmarks.

Prints a JSON fingerprint of everything that affects benchmark validity on
this node: git revision, hashes of the MLX patch files, mlx / mlx-lm
versions, relevant env, model config hashes, running exo PIDs/RSS and (best
effort) the local EXO API instance list.

Run on every node with the node's venv python, so mlx versions are the ones
EXO actually uses:

    .venv/bin/python bench/revlock.py \
        --model-dir /path/to/model --out /tmp/revlock-$(hostname -s).json

Compare nodes before a benchmark (exit 1 on hard mismatch):

    python bench/revlock.py --diff s1.json s2.json s3.json s4.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PATCHES_DIR = REPO / "src/exo/worker/engines/mlx/patches"

# Must be identical across nodes for a valid distributed benchmark.
HARD_KEYS = [
    "git.head",
    "git.branch",
    "git.dirty",
    "patches",
    "mlx.version",
    "mlx_lm.version",
    "env",
    "model_files",
]
# Divergence here is suspicious but not automatically fatal.
WARN_KEYS = ["python", "macos"]

MODEL_FILE_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
]


def _sh(*cmd: str) -> str:
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=15)
        return r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR:{e}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:  # noqa: BLE001
        return f"ERROR:{e}"


def _mlx_version() -> dict[str, Any]:
    try:
        import mlx.core as mx

        return {"version": str(mx.__version__), "file": str(getattr(mx, "__file__", ""))}
    except Exception as e:  # noqa: BLE001
        return {"version": f"ERROR:{e}", "file": ""}


def _module_version(name: str) -> dict[str, Any]:
    try:
        mod = __import__(name)
        return {
            "version": str(getattr(mod, "__version__", "unknown")),
            "file": str(getattr(mod, "__file__", "")),
        }
    except Exception as e:  # noqa: BLE001
        return {"version": f"ERROR:{e}", "file": ""}


def _exo_processes() -> list[str]:
    out = _sh("ps", "-axo", "pid=,rss=,command=")
    rows: list[str] = []
    for line in out.splitlines():
        if "exo" in line and "revlock" not in line and "grep" not in line:
            rows.append(line.strip()[:200])
    return rows


def _api_instances(port: int) -> Any:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/state", timeout=3) as r:
            state = json.loads(r.read().decode())
        if isinstance(state, dict):
            for key in ("instances", "instancePreviews", "instance_previews"):
                if key in state:
                    return state[key]
            return sorted(state.keys())
        return "unparsed"
    except Exception as e:  # noqa: BLE001
        return f"ERROR:{e}"


def fingerprint(model_dir: str | None, api_port: int, with_api: bool) -> dict[str, Any]:
    patches = {p.name: _sha256(p) for p in sorted(PATCHES_DIR.glob("*.py"))}
    env = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(("EXO_", "JACCL", "MLX_", "RDMA"))
    }
    fp: dict[str, Any] = {
        "host": platform.node(),
        "git": {
            "head": _sh("git", "rev-parse", "HEAD"),
            "branch": _sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "describe": _sh("git", "describe", "--tags", "--always"),
            "dirty": bool(_sh("git", "status", "--porcelain")),
        },
        "patches": patches,
        "mlx": _mlx_version(),
        "mlx_lm": _module_version("mlx_lm"),
        "python": sys.version.split()[0],
        "macos": platform.mac_ver()[0] or platform.platform(),
        "env": env,
        "model_files": {},
        "processes": _exo_processes(),
    }
    if model_dir:
        d = Path(model_dir).expanduser()
        files: dict[str, str] = {}
        for pat in MODEL_FILE_PATTERNS:
            p = d / pat
            if p.exists():
                files[pat] = _sha256(p)
        for p in sorted(d.glob("*.safetensors.index.json")):
            files[p.name] = _sha256(p)
        fp["model_files"] = files
        fp["model_dir"] = str(d)
    if with_api:
        fp["api_instances"] = _api_instances(api_port)
    return fp


def _get(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def diff(paths: list[str]) -> int:
    fps: list[tuple[str, dict[str, Any]]] = []
    for p in paths:
        with open(p) as f:
            fps.append((p, json.load(f)))
    ref_path, ref = fps[0]
    rc = 0
    for key in HARD_KEYS:
        ref_val = _get(ref, key)
        for p, fp in fps[1:]:
            val = _get(fp, key)
            if val != ref_val:
                print(f"HARD MISMATCH {key}: {ref_path} != {p}")
                print(f"  {ref_path}: {json.dumps(ref_val, sort_keys=True)[:400]}")
                print(f"  {p}: {json.dumps(val, sort_keys=True)[:400]}")
                rc = 1
    for p, fp in fps:
        if _get(fp, "git.dirty"):
            print(f"HARD: dirty working tree on {p}")
            rc = 1
    for key in WARN_KEYS:
        ref_val = _get(ref, key)
        for p, fp in fps[1:]:
            val = _get(fp, key)
            if val != ref_val:
                print(f"warn: {key} differs: {ref_path}={ref_val} vs {p}={val}")
    if rc == 0:
        head = str(_get(ref, "git.head"))[:12]
        print(f"OK: {len(fps)} fingerprints consistent (head={head})")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(prog="revlock")
    ap.add_argument("--model-dir", default=None, help="Model dir to hash config/tokenizer/index files.")
    ap.add_argument("--port", type=int, default=52415, help="Local EXO API port for instance capture.")
    ap.add_argument("--no-api", action="store_true", help="Skip the local EXO API query.")
    ap.add_argument("--out", default=None, help="Also write the JSON to this path.")
    ap.add_argument("--diff", nargs="+", default=None, help="Compare fingerprint JSON files instead of capturing.")
    args = ap.parse_args()

    if args.diff:
        return diff(args.diff)

    fp = fingerprint(args.model_dir, args.port, not args.no_api)
    text = json.dumps(fp, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
