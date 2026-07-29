#!/usr/bin/env python3
"""B2: offline-конвертер moonshotai/Kimi-K3 (compressed-tensors MXFP4)
-> MLX-нативный репозиторий под vendor/kimi_k3.py (EXO).

Принципы:
  * BYTE-PRESERVING: fp4-нибблы и e8m0-скейлы экспертов копируются как байты
    (репак uint8-пар -> uint32-слова, БЕЗ арифметики). Конвенция источника:
    lo-first (элемент 2i в младшем нибле) — по спеке compressed-tensors
    pack-quantized; независимо подтверждена pipenetwork end-to-end zero-error
    на этой же паре форматов. Override: --nibble-order hi. ВАЖНО: из самих
    байтов конвенцию вывести нельзя (любая самосверка тавтологична);
    финальный судья — A3-parity логитов. Периодическая сверка (--verify-every)
    проверяет КОНСИСТЕНТНОСТЬ репака (оси/шейпы/view), не конвенцию.
  * Всё вне routed experts — bf16/f32 pass-through (скоуп R3: quant только
    эксперты; подтверждено A0: weight_packed вне экспертов = 0).
  * Имена: strip "language_model.", drop vision/mm; эксперты стекуются в
    switch_mlp.{gate,up,down}_proj.{weight,scales} [E,...]; остальное — имена
    источника (conv1d/A_log/kv_b и т.д. обрабатывает runtime remap vendor'а).
  * config: плоский text_config + model_type="kimi_k3" + mlx-блок
    {"quantization": {group_size:32, bits:4, mode:"mxfp4"}};
    quantization_config УДАЛЯЕТСЯ (иначе пин уйдёт в affine-ветку).

Запуск (на ноде с чекпоинтом; ~1.5TB чтения + ~1.4TB записи):
  python3 tools/kimi_k3_convert_mxfp4.py \
      ~/.exo/models/moonshotai--Kimi-K3 \
      /Volumes/Models/mightyC1--Kimi-K3-MLX-mxfp4 \
      [--shard-gb 12] [--verify-every 64] [--self-check-only]
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

WRAPPER = "language_model."
DROP = ("vision_tower", "mm_projector", "multi_modal")
SRC2DST = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
GS = 32

# e2m1 (fp4) LUT: код 0..7 -> величина, бит 3 = знак
_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def _ref_decode(packed_u8: np.ndarray, scales_u8: np.ndarray, lo_first: bool) -> np.ndarray:
    """Референс-декод compressed-tensors mxfp4: [out, in/2]x[out, in/32] -> [out, in]."""
    lo = packed_u8 & 0x0F
    hi = packed_u8 >> 4
    a, b = (lo, hi) if lo_first else (hi, lo)
    nib = np.stack([a, b], axis=-1).reshape(packed_u8.shape[0], -1)  # [out, in]
    mag = _E2M1[nib & 0x7]
    val = np.where(nib & 0x8, -mag, mag)
    scale = np.exp2(scales_u8.astype(np.float64) - 127.0)            # e8m0
    scale = np.repeat(scale, GS, axis=-1)
    return val * scale


def _repack_u8_to_u32(packed_u8: mx.array) -> mx.array:
    """[out, in/2] uint8 -> [out, in/8] uint32 (little-endian байты слова)."""
    a = np.ascontiguousarray(np.array(packed_u8))
    assert a.dtype == np.uint8 and a.shape[-1] % 4 == 0
    return mx.array(a.view("<u4"))


def _swap_nibbles(packed_u8: mx.array) -> mx.array:
    a = np.array(packed_u8)
    return mx.array(((a & 0x0F) << 4) | (a >> 4))


class NibbleOrder:
    """Репак с проверкой КОНСИСТЕНТНОСТИ (не конвенции — она задаётся снаружи).

    lo_first=True (дефолт): спека compressed-tensors + pipenetwork zero-error.
    Сверка: mx.dequantize(наших uint32) == numpy-референс тех же байтов при
    той же конвенции. Ловит ошибки осей/view/шейпов и несовпадение
    внутреннего порядка ниблов MLX; конвенцию источника доказать не может
    (тавтология) — её судит A3-parity."""

    def __init__(self, lo_first: bool = True) -> None:
        self.lo_first = lo_first
        self.checked = 0
        self._announced = False

    def repack(self, packed: mx.array, scales: Optional[mx.array], verify: bool) -> mx.array:
        if not self._announced:
            print(f"[convert] nibble order: {'lo-first' if self.lo_first else 'hi-first'} "
                  f"(задано; спека compressed-tensors = lo-first)")
            self._announced = True
        p = packed if self.lo_first else _swap_nibbles(packed)
        w32 = _repack_u8_to_u32(p)
        if verify and scales is not None:
            self._verify(w32, packed, scales)
        return w32

    def _verify(self, w32: mx.array, packed: mx.array, scales: mx.array) -> None:
        rows = min(4, packed.shape[0])
        got = np.array(
            mx.dequantize(
                w32[:rows], scales[:rows], group_size=GS, bits=4, mode="mxfp4"
            ).astype(mx.float32)
        ).astype(np.float64)
        ref = _ref_decode(np.array(packed[:rows]), np.array(scales[:rows]), self.lo_first)
        if not np.array_equal(got, ref):
            raise RuntimeError(
                "convert: сверка консистентности репака ПРОВАЛЕНА (оси/раскладка/"
                "порядок ниблов MLX) — стоп"
            )
        self.checked += 1


def read_header(path: Path) -> Tuple[dict, int]:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


class ShardWriter:
    def __init__(self, out_dir: Path, shard_gb: float):
        self.out_dir = out_dir
        self.limit = int(shard_gb * (1 << 30))
        self.buf: Dict[str, mx.array] = {}
        self.buf_bytes = 0
        self.files: List[Tuple[str, List[str]]] = []
        self.total = 0

    def add(self, name: str, arr: mx.array) -> None:
        self.buf[name] = arr
        nb = arr.size * arr.dtype.size
        self.buf_bytes += nb
        self.total += nb
        if self.buf_bytes >= self.limit:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        idx = len(self.files) + 1
        fname = f"model-{idx:05d}.safetensors"
        mx.save_safetensors(str(self.out_dir / fname), self.buf)
        self.files.append((fname, list(self.buf)))
        print(f"[convert]   wrote {fname}: {len(self.buf)} tensors, "
              f"{self.buf_bytes / (1 << 30):.2f} GiB")
        self.buf = {}
        self.buf_bytes = 0
        mx.clear_cache()

    def finalize(self) -> dict:
        self.flush()
        total_shards = len(self.files)
        wmap: Dict[str, str] = {}
        renames = []
        for i, (fname, names) in enumerate(self.files, start=1):
            final = f"model-{i:05d}-of-{total_shards:06d}.safetensors"
            renames.append((fname, final))
            for n in names:
                wmap[n] = final
        for src, dst in renames:
            (self.out_dir / src).rename(self.out_dir / dst)
        return {"metadata": {"total_size": self.total}, "weight_map": wmap}


def rewrite_config(src_dir: Path, out_dir: Path) -> None:
    cfg = json.loads((src_dir / "config.json").read_text())
    text = dict(cfg.get("text_config") or cfg)
    text.pop("quantization_config", None)
    new = text
    new["model_type"] = "kimi_k3"
    new["architectures"] = ["KimiK3ForCausalLM"]
    for k in ("eos_token_id", "bos_token_id", "tie_word_embeddings"):
        if k not in new and k in cfg:
            new[k] = cfg[k]
    new["quantization"] = {"group_size": GS, "bits": 4, "mode": "mxfp4"}
    new.pop("quantization_config", None)  # НИКОГДА не оставлять — affine-ветка пина
    (out_dir / "config.json").write_text(json.dumps(new, indent=2, ensure_ascii=False))


def copy_aux(src_dir: Path, out_dir: Path) -> None:
    keep_globs = [
        "generation_config.json", "tokenizer_config.json", "special_tokens_map.json",
        "tokenization_kimi.py", "encoding_k3.py", "chat_template*", "*.tiktoken",
        "tiktoken*", "vocab*", "merges*",
    ]
    for g in keep_globs:
        for p in src_dir.glob(g):
            if p.is_file():
                shutil.copy2(p, out_dir / p.name)


def convert(src_dir: Path, out_dir: Path, shard_gb: float, verify_every: int,
            self_check_only: bool, nibble_lo_first: bool = True) -> int:
    index = json.loads((src_dir / "model.safetensors.index.json").read_text())
    wmap: Dict[str, str] = index["weight_map"]
    shard_names = sorted(set(wmap.values()))
    print(f"[convert] источник: {len(wmap)} тензоров, {len(shard_names)} шардов")

    order = NibbleOrder(lo_first=nibble_lo_first)
    if self_check_only:
        # найти первый экспертный тензор и прогнать детект/сверку
        for shard in shard_names:
            hdr, _ = read_header(src_dir / shard)
            names = [n for n in hdr if n != "__metadata__" and ".weight_packed" in n]
            if not names:
                continue
            data = mx.load(str(src_dir / shard))
            n = sorted(names)[0]
            order.repack(data[n], data[n.replace("weight_packed", "weight_scale")], True)
            print("[convert] self-check OK: репак консистентен (оси/view/раскладка MLX). "
                  "Конвенция источника = lo-first по спеке; финальная валидация — A3-parity.")
            return 0
        print("[convert] weight_packed не найден?!")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(out_dir, shard_gb)
    # pending[(layer, dst, kind)] -> {expert_idx: array}
    pending: Dict[Tuple[str, str, str], Dict[int, mx.array]] = {}
    n_experts_cfg = (json.loads((src_dir / "config.json").read_text())
                     .get("text_config", {}) or {}).get("num_experts", 896)
    seen_expert_tensors = 0
    t0 = time.time()

    for si, shard in enumerate(shard_names, 1):
        data = mx.load(str(src_dir / shard))
        for name in sorted(data):
            v = data[name]
            k = name[len(WRAPPER):] if name.startswith(WRAPPER) else name
            if any(m in k for m in DROP) or ".mtp" in k:
                continue
            if ".block_sparse_moe.experts." in k and (
                k.endswith(".weight_packed") or k.endswith(".weight_scale")
            ):
                # model.layers.N.block_sparse_moe.experts.E.wX.{weight_packed|weight_scale}
                parts = k.split(".")
                layer = parts[2]
                e_idx = int(parts[5])
                dst = SRC2DST[parts[6]]
                kind = "weight" if k.endswith("weight_packed") else "scales"
                if kind == "weight":
                    scales_name = name.replace("weight_packed", "weight_scale")
                    scales = data.get(scales_name)
                    if scales is None:
                        # скейлы в другом шарде — редко; репак без verify, порядок
                        # уже задетекчен на первом полном паре
                        v = order.repack(v, None, False) if order.lo_first is not None \
                            else (_ for _ in ()).throw(RuntimeError(
                                "первый weight_packed без scales в том же шарде"))
                    else:
                        do_verify = (seen_expert_tensors % max(1, verify_every) == 0)
                        v = order.repack(v, scales, do_verify)
                    seen_expert_tensors += 1
                bucket = pending.setdefault((layer, dst, kind), {})
                bucket[e_idx] = v
                if len(bucket) == n_experts_cfg:
                    stacked = mx.stack([bucket[i] for i in range(n_experts_cfg)])
                    writer.add(
                        f"model.layers.{layer}.block_sparse_moe.switch_mlp.{dst}.{kind}",
                        stacked,
                    )
                    del pending[(layer, dst, kind)]
                continue
            if k.endswith(".weight_packed") or k.endswith(".weight_scale"):
                raise RuntimeError(f"MXFP4 вне routed experts: {k} (нарушение R3)")
            writer.add(k, v)
        del data
        mx.clear_cache()
        el = time.time() - t0
        print(f"[convert] shard {si}/{len(shard_names)} за {el:.0f}s "
              f"(pending groups: {len(pending)})")

    if pending:
        missing = {key: n_experts_cfg - len(b) for key, b in pending.items()}
        raise RuntimeError(f"незакрытые экспертные группы: {missing}")

    idx = writer.finalize()
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))
    rewrite_config(src_dir, out_dir)
    copy_aux(src_dir, out_dir)
    print(f"[convert] DONE: {writer.total / (1 << 40):.3f} TiB, "
          f"{len(idx['weight_map'])} тензоров, периодических сверок: {order.checked}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--shard-gb", type=float, default=12.0)
    ap.add_argument("--verify-every", type=int, default=64,
                    help="каждый N-й экспертный тензор сверять с референс-декодом")
    ap.add_argument("--self-check-only", action="store_true",
                    help="только сверка консистентности репака, без записи")
    ap.add_argument("--nibble-order", choices=["lo", "hi"], default="lo",
                    help="конвенция упаковки источника (дефолт lo = спека "
                         "compressed-tensors)")
    a = ap.parse_args()
    if not a.self_check_only and a.out is None:
        ap.error("нужен out (или --self-check-only)")
    return convert(a.src, a.out, a.shard_gb, a.verify_every, a.self_check_only,
                   nibble_lo_first=(a.nibble_order == "lo"))


if __name__ == "__main__":
    sys.exit(main())
