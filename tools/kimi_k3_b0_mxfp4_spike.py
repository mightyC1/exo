#!/usr/bin/env python3
"""B0-spike: mxfp4-путь на пиновом mlx — single-node, Kimi K3 (план §B0, P0-039-prep).

Проверяет РОВНО то, что нужно K3-конвейеру (routed experts = MXFP4 gs32,
остальное bf16), на текущем пине mlx/mlx-lm, БЕЗ загрузки весов:

  1. core roundtrip: mx.quantize/dequantize(mode="mxfp4", group_size=32)
     - scales обязаны быть uint8 (e8m0) — на этом стоит byte-preserving
       repack в конвертере (B2);
  2. mx.quantized_matmul(mode="mxfp4") vs matmul по явно деквантованным
     весам — согласие обязано быть плотным (одна и та же математика);
  3. gather_qmm-путь: SwitchGLU -> nn.quantize(mode="mxfp4", только
     SwitchLinear) -> forward vs форвард свежего bf16-SwitchGLU с
     деквантованными весами. Кейс A: 896 экспертов (реальное число K3,
     маленькие dims — проверка индексации). Кейс B: 8 экспертов на
     реальных dims 3584->3072 (проверка большого кернела).
  4. информационно (БЕЗ assert): расхождение mxfp4 vs исходный bf16 —
     это и есть цена 4 бит, вердикта о качестве здесь нет.

НЕ проверяет: распределённое поведение MX-форматов в EXO (это ТОЛЬКО
P0-039 на меше, отдельным прогоном), скорость (никаких t/s-вердиктов).

Запуск на ноде:
  cd ~/Desktop/ai/exo && .venv/bin/python tools/kimi_k3_b0_mxfp4_spike.py [--json]
--small — санити самого скрипта в средах с малой памятью (CPU/CI): режет
экспертов кейса A и dims кейса B. Полный профиль гонять на ноде (Metal).
"""

from __future__ import annotations

import argparse
import functools
import json
import sys

import mlx.core as mx
import mlx.nn as nn

GS = 32
BITS = 4
MODE = "mxfp4"


def _rel(a: mx.array, b: mx.array) -> tuple[float, float]:
    """(mean-rel, max-abs-rel-to-linf) в fp32."""
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    denom = float(mx.abs(b32).mean()) or 1.0
    linf = float(mx.abs(b32).max()) or 1.0
    return (
        float(mx.abs(a32 - b32).mean()) / denom,
        float(mx.abs(a32 - b32).max()) / linf,
    )


def check_core_roundtrip(out: dict) -> bool:
    w = mx.random.normal((256, 512)).astype(mx.bfloat16)
    wq, scales = mx.quantize(w, group_size=GS, bits=BITS, mode=MODE)
    ok = True

    # e8m0-скейлы = uint8; packed = uint32. Конвертер B2 на это опирается.
    out["scales_dtype"] = str(scales.dtype)
    out["packed_dtype"] = str(wq.dtype)
    if scales.dtype != mx.uint8:
        out["FAIL"] = f"scales dtype {scales.dtype} != uint8 (e8m0)"
        ok = False

    wd = mx.dequantize(wq, scales, group_size=GS, bits=BITS, mode=MODE)
    mean_rel, _ = _rel(wd, w)
    out["dequant_vs_bf16_mean_rel"] = round(mean_rel, 4)  # информационно (4 бита)
    out["dequant_finite"] = bool(mx.isfinite(wd).all())
    ok = ok and out["dequant_finite"]

    # qmm vs dequant-референс — та же математика, расхождение только bf16-раундинг.
    x = mx.random.normal((8, 512)).astype(mx.bfloat16)
    y_q = mx.quantized_matmul(
        x, wq, scales, transpose=True, group_size=GS, bits=BITS, mode=MODE
    )
    y_ref = x.astype(mx.float32) @ wd.astype(mx.float32).T
    m, lin = _rel(y_q, y_ref)
    out["qmm_vs_dequant_mean_rel"] = round(m, 6)
    out["qmm_vs_dequant_linf_rel"] = round(lin, 6)
    out["qmm_finite"] = bool(mx.isfinite(y_q).all())
    if not out["qmm_finite"] or m > 2e-2:
        out["FAIL"] = out.get("FAIL", "") + f" | qmm mismatch mean_rel={m}"
        ok = False
    return ok


def _dequant_clone(mq: nn.Module, din: int, dh: int, experts: int) -> nn.Module:
    """Свежий bf16-SwitchGLU с деквантованными весами из mq (generic по путям)."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

    ref = SwitchGLU(din, dh, experts)
    for path, mod in mq.named_modules():
        if not isinstance(mod, QuantizedSwitchLinear):
            continue
        wd = mx.dequantize(
            mod.weight, mod.scales,
            getattr(mod, "biases", None) if MODE == "affine" else None,
            group_size=GS, bits=BITS, mode=MODE,
        ).astype(mx.float32)
        target = functools.reduce(getattr, path.split("."), ref)
        target.weight = wd
        b = getattr(mod, "bias", None)
        if b is not None and hasattr(target, "bias"):
            target.bias = b.astype(mx.float32)
    return ref


def check_switchglu(out: dict, experts: int, din: int, dh: int, label: str) -> bool:
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU, SwitchLinear

    mx.random.seed(0)
    m = SwitchGLU(din, dh, experts)  # веса fp32 (дефолт init)
    x32 = mx.random.normal((2, 5, din))            # fp32 для реф-форвардов:
    x = x32.astype(mx.bfloat16)                    # CPU gather_mm (некв.)умеет только fp32
    inds = mx.random.randint(0, experts, (2, 5, 2))
    y_orig = m(x32, inds)
    mx.eval(y_orig)

    nn.quantize(
        m, group_size=GS, bits=BITS, mode=MODE,
        class_predicate=lambda _p, mod: isinstance(mod, SwitchLinear),
    )
    n_q = sum(isinstance(mod, QuantizedSwitchLinear) for _, mod in m.named_modules())
    modes = {mod.mode for _, mod in m.named_modules() if isinstance(mod, QuantizedSwitchLinear)}
    out[f"{label}_quantized_layers"] = n_q
    out[f"{label}_modes"] = sorted(modes)
    ok = n_q == 3 and modes == {MODE}
    if not ok:
        out["FAIL"] = out.get("FAIL", "") + f" | {label}: layers={n_q} modes={modes}"

    y_q = m(x, inds)
    out[f"{label}_finite"] = bool(mx.isfinite(y_q).all())
    ok = ok and out[f"{label}_finite"]

    y_ref = _dequant_clone(m, din, dh, experts)(x32, inds)
    mx.eval(y_ref)
    mean_rel, linf = _rel(y_q, y_ref)
    out[f"{label}_gatherqmm_vs_dequant_mean_rel"] = round(mean_rel, 6)
    out[f"{label}_gatherqmm_vs_dequant_linf_rel"] = round(linf, 6)
    if mean_rel > 2e-2:
        out["FAIL"] = out.get("FAIL", "") + f" | {label}: gather_qmm mean_rel={mean_rel}"
        ok = False

    info_rel, _ = _rel(y_q, y_orig)
    out[f"{label}_vs_fp32src_mean_rel_info"] = round(info_rel, 4)  # цена 4 бит, без assert
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--small", action="store_true", help="уменьшить кейс B (CPU-санити)")
    args = ap.parse_args()

    out: dict = {
        "mlx_version": mx.__version__,
        "device": str(mx.default_device()),
        "mode": MODE, "group_size": GS, "bits": BITS,
    }
    print("[1/3] core roundtrip (quantize/dequantize/qmm)...", file=sys.stderr, flush=True)
    ok = check_core_roundtrip(out)
    # Кейс A: реальное число экспертов K3 (896), минимальные dims — проверяется
    # ИНДЕКСАЦИЯ gather_qmm по 896, ширина тут не нужна. --small режет и экспертов
    # (CPU-санити самого скрипта; nn.quantize на CPU прожорлив по transient-памяти).
    n_exp = 128 if args.small else 896
    print(f"[2/3] gather_qmm: {n_exp} experts (индексация)...", file=sys.stderr, flush=True)
    ok &= check_switchglu(out, experts=n_exp, din=64, dh=64, label=f"A_{n_exp}exp")
    # Кейс B: реальные dims латентных экспертов K3 (3584->3072), 8 экспертов.
    din, dh = (512, 384) if args.small else (3584, 3072)
    print(f"[3/3] gather_qmm: полные dims {din}->{dh}, 8 experts...", file=sys.stderr, flush=True)
    ok &= check_switchglu(out, experts=8, din=din, dh=dh, label="B_fulldim")

    out["verdict"] = "PASS" if ok else "FAIL"
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        w = max(len(k) for k in out)
        for k, v in out.items():
            print(f"{k:<{w}}  {v}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
