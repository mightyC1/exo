#!/usr/bin/env python3
"""Аудит загрузки Kimi K3: ловит МОЛЧАЛИВЫЕ расхождения имён/форм.

Зачем: EXO грузит веса через load_model(..., strict=False) — параметр, которого
нет в чекпоинте под ожидаемым именем, ОСТАЁТСЯ СЛУЧАЙНОЙ ИНИЦИАЛИЗАЦИЕЙ без
единой строки в логе. Внешне: модель грузится, генерирует мусор.

Режим A (дефолт, метаданные, секунды, память ~0):
  строит скелет модели по config, собирает {имя: zeros(shape)} из ЗАГОЛОВКОВ
  safetensors, прогоняет model.sanitize(), применяет nn.quantize тем же
  предикатом, что и mlx_lm.utils.load_model, и сравнивает множества:
     MISSING  — модуль ждёт, чекпоинт не даёт  -> СЛУЧАЙНЫЕ ВЕСА (фатально)
     EXTRA    — чекпоинт даёт, модуль не ждёт  -> молча выброшено
     SHAPE    — имя есть, форма другая
  Ленивые mx-массивы не материализуются (никаких mx.eval).

Режим B (--probe-layer N): грузит РЕАЛЬНЫЕ веса ОДНОГО слоя, прогоняет модуль
  на случайном входе, печатает статистику (finite/min/max/rms) — локализует,
  на каком типе слоя выход разъезжается/NaN'ится.

Запуск:
  python3 tools/kimi_k3_load_audit.py ~/.exo/models/mightyC1--Kimi-K3-MLX-mxfp4
  python3 tools/kimi_k3_load_audit.py <repo> --probe-layer 0 --probe-layer 4
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mu

_DT = {
    "BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32,
    "U8": mx.uint8, "I8": mx.int8, "U32": mx.uint32, "I32": mx.int32,
    "F8_E8M0": mx.uint8, "U16": mx.uint16, "I64": mx.int64,
}


def headers(repo: Path) -> Dict[str, dict]:
    idx = json.loads((repo / "model.safetensors.index.json").read_text())
    out: Dict[str, dict] = {}
    for shard in sorted(set(idx["weight_map"].values())):
        p = repo / shard
        with p.open("rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(n))
        for name, e in hdr.items():
            if name == "__metadata__":
                continue
            out[name] = {"dtype": e["dtype"], "shape": e["shape"], "shard": shard}
    return out


def build_model(repo: Path):
    from exo.worker.engines.mlx.vendor.custom_models import register_custom_models
    register_custom_models()
    import exo.worker.engines.mlx.vendor.kimi_k3 as k3

    cfg = json.loads((repo / "config.json").read_text())
    args = k3.ModelArgs.from_dict(cfg)
    return k3, cfg, args, k3.Model(args)


def audit(repo: Path) -> int:
    k3, cfg, args, model = build_model(repo)
    hdr = headers(repo)
    print(f"[audit] чекпоинт: {len(hdr)} тензоров; model_type={cfg.get('model_type')}; "
          f"quantization={cfg.get('quantization')}")
    if "quantization_config" in cfg:
        print("[audit] !! config содержит quantization_config — пин уйдёт в "
              "legacy-ветку (возможен режим affine). Удалить.")

    fake = {
        n: mx.zeros(tuple(e["shape"]), dtype=_DT.get(e["dtype"], mx.float32))
        for n, e in hdr.items()
    }
    weights = model.sanitize(fake)

    quant = cfg.get("quantization")
    if quant:
        def class_predicate(p, m):
            if p in quant:
                return quant[p]
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights
        nn.quantize(
            model,
            group_size=quant["group_size"], bits=quant["bits"],
            mode=quant.get("mode", "affine"), class_predicate=class_predicate,
        )

    expected = {p: v.shape for p, v in mu.tree_flatten(model.parameters())}
    got = {k: v.shape for k, v in weights.items()}

    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    shape_bad = sorted(
        k for k in set(expected) & set(got) if tuple(expected[k]) != tuple(got[k])
    )

    def head(xs: List[str], n: int = 12) -> str:
        return "\n    " + "\n    ".join(xs[:n]) + (f"\n    ... ещё {len(xs)-n}" if len(xs) > n else "")

    print(f"[audit] ожидает модуль: {len(expected)} | даёт чекпоинт (после sanitize): {len(got)}")
    ok = True
    if missing:
        ok = False
        print(f"[audit] MISSING {len(missing)} — ЭТИ ВЕСА ОСТАНУТСЯ СЛУЧАЙНЫМИ "
              f"(strict=False!):{head(missing)}")
    if shape_bad:
        ok = False
        print(f"[audit] SHAPE MISMATCH {len(shape_bad)}:" + "".join(
            f"\n    {k}: модуль {tuple(expected[k])} vs чекпоинт {tuple(got[k])}"
            for k in shape_bad[:12]))
    if extra:
        print(f"[audit] EXTRA {len(extra)} — молча выброшено:{head(extra)}")
    print("[audit] ВЕРДИКТ:", "OK — имена и формы сходятся" if ok else "РАСХОЖДЕНИЕ (см. выше)")
    return 0 if ok else 2


def probe(repo: Path, layer_ids: List[int]) -> int:
    k3, cfg, args, model = build_model(repo)
    hdr = headers(repo)
    idx = json.loads((repo / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]

    for li in layer_ids:
        # имена шардов: конвертированные ("model.layers.") ИЛИ сырые официальные
        # ("language_model.model.layers.")
        src_prefixes = (f"model.layers.{li}.", f"language_model.model.layers.{li}.")
        names = [n for n in hdr if n.startswith(src_prefixes)]
        if not names:
            print(f"[probe] слой {li}: тензоров не найдено")
            continue
        need_shards = sorted({wmap[n] for n in names})
        raw: Dict[str, mx.array] = {}
        for s in need_shards:
            d = mx.load(str(repo / s))
            for n in names:
                if n in d:
                    raw[n] = d[n]
            del d
        # sanitize понимает подмножество; их sanitize добавляет префикс language_model.
        weights = model.sanitize(raw)
        layer = model.model.layers[li]
        kind = "KDA" if layer.is_linear else "MLA"
        ffn = type(layer.mlp).__name__

        lp = f"language_model.model.layers.{li}."
        quant = cfg.get("quantization")
        if quant:
            def cp(p, m):
                return hasattr(m, "to_quantized") and f"{lp}{p}.scales" in weights
            nn.quantize(layer, group_size=quant["group_size"], bits=quant["bits"],
                        mode=quant.get("mode", "affine"), class_predicate=cp)

        sub = {k[len(lp):]: v for k, v in weights.items() if k.startswith(lp)}
        exp = {p for p, _ in mu.tree_flatten(layer.parameters())}
        miss = sorted(exp - set(sub))
        if miss:
            print(f"[probe] слой {li} ({kind}/{ffn}): MISSING {len(miss)}: {miss[:6]}")
        layer.load_weights(list(sub.items()), strict=False)
        mx.eval(layer.parameters())

        mx.random.seed(0)
        x = (mx.random.normal((1, 16, args.text_config.hidden_size
                               if hasattr(args, "text_config") else args.hidden_size))
             * 0.02).astype(mx.bfloat16)
        blocks = k3.ResidualBlocks(1e-5)
        blocks.append(x)
        y = layer(x, mask=None, cache=None, blocks=blocks)
        if isinstance(y, tuple):
            y = y[0]
        mx.eval(y)
        f = y.astype(mx.float32)
        print(f"[probe] слой {li} ({kind}/{ffn}): finite={bool(mx.isfinite(f).all())} "
              f"rms={float(mx.sqrt(mx.mean(f*f))):.4f} "
              f"min={float(f.min()):.4f} max={float(f.max()):.4f}")
        del raw, weights, sub
        mx.clear_cache()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("--probe-layer", type=int, action="append", default=[])
    a = ap.parse_args()
    if a.probe_layer:
        return probe(a.repo, a.probe_layer)
    return audit(a.repo)


if __name__ == "__main__":
    sys.exit(main())
