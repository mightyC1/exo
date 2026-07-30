#!/usr/bin/env python3
"""A0.5-проба v2: шейпы/dtypes решающих тензоров Kimi K3 БЕЗ чтения весов.

v2: официальный чекпоинт — мультимодальная обёртка
(KimiK3ForConditionalGeneration): текстовый конфиг вложен в `text_config`,
имена тензоров несут префикс обёртки (например `language_model.`).
Проба спускается в text_config и матчит тензоры по ПОДСТРОКАМ, плюс всегда
печатает перепись префиксов (census) — фактическую структуру имён.

Читает только 8-байтовые префиксы + JSON-заголовки шардов. Секунды на 1.5TB.

Запуск:
  python3 kimi_k3_shape_probe.py ~/.exo/models/moonshotai--Kimi-K3 --json ~/k3-shapes.json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path

# Подстроки (регистр важен, точки фиксируют границы индексов слоёв).
SUBSTRINGS = [
    "model.layers.0.",                        # KDA-слой + dense MLP
    "model.layers.3.self_attn.",              # MLA-слой
    "model.layers.4.block_sparse_moe.gate",   # роутер
    "model.layers.4.block_sparse_moe.routed_expert",
    "model.layers.4.block_sparse_moe.experts.0.",
    "model.layers.4.block_sparse_moe.shared_experts.",
    "embed_tokens.",
    "output_attn_res",
    "lm_head.",
]
SKIP_MARKERS = ("vision", "mm_projector", "multi_modal", "audio")

TEXT_KEYS = [
    "model_type", "dtype", "mla_use_nope", "mla_use_output_gate",
    "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
    "num_attention_heads", "num_key_value_heads", "num_hidden_layers",
    "hidden_size", "q_lora_rank", "kv_lora_rank",
    "routed_expert_hidden_size", "latent_moe_use_norm",
    "moe_intermediate_size", "num_experts", "num_experts_per_token",
    "num_shared_experts", "moe_router_activation_func", "moe_renormalize",
    "routed_scaling_factor", "num_expert_group", "topk_group",
    "use_grouped_topk", "topk_method", "moe_layer_freq",
    "first_k_dense_replace", "intermediate_size", "hidden_act",
    "activation_situ_beta", "activation_situ_linear_beta",
    "attn_res_block_size", "rms_norm_eps", "max_position_embeddings",
    "eos_token_id", "bos_token_id", "tie_word_embeddings",
]
TOP_KEYS = ["model_type", "architectures", "dtype", "eos_token_id",
            "bos_token_id", "tie_word_embeddings"]


def read_st_header(path: Path) -> dict:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    root: Path = args.root

    out: dict = {"config_top": {}, "config_text": {}, "generation_config": {},
                 "tokenizer": {}, "census": {}, "raw_name_samples": [],
                 "tensors": {}, "hints": []}

    cfg = json.loads((root / "config.json").read_text())
    for k in TOP_KEYS:
        if k in cfg:
            out["config_top"][k] = cfg[k]
    text = cfg.get("text_config") or cfg
    out["config_top"]["has_text_config"] = "text_config" in cfg
    for k in TEXT_KEYS:
        if k in text:
            out["config_text"][k] = text[k]
    out["config_text"]["linear_attn_config"] = text.get("linear_attn_config")

    gc = root / "generation_config.json"
    if gc.exists():
        g = json.loads(gc.read_text())
        out["generation_config"] = {k: g[k] for k in
            ("eos_token_id", "bos_token_id", "pad_token_id", "temperature",
             "top_p", "do_sample") if k in g}
    tc = root / "tokenizer_config.json"
    if tc.exists():
        t = json.loads(tc.read_text())
        out["tokenizer"] = {k: t[k] for k in ("eos_token", "bos_token") if k in t}

    index = json.loads((root / "model.safetensors.index.json").read_text())
    wmap: dict[str, str] = index["weight_map"]

    # census: первая и первые-две компоненты имени
    c1, c2 = Counter(), Counter()
    for n in wmap:
        parts = n.split(".")
        c1[parts[0]] += 1
        c2[".".join(parts[:2])] += 1
    out["census"] = {"level1": dict(c1.most_common(8)),
                     "level2": dict(c2.most_common(12)),
                     "total": len(wmap)}
    sample = sorted(wmap)[:3]
    for probe_sub in ("block_sparse_moe.experts.0.", "output_attn_res", "self_attn.A_log"):
        hit = next((n for n in wmap if probe_sub in n), None)
        if hit:
            sample.append(hit)
    out["raw_name_samples"] = sample

    wanted = sorted(
        n for n in wmap
        if any(s in n for s in SUBSTRINGS) or n.endswith("model.norm.weight")
    )
    wanted = [n for n in wanted if not any(m in n.lower() for m in SKIP_MARKERS)]

    by_shard: dict[str, list[str]] = {}
    for n in wanted:
        by_shard.setdefault(wmap[n], []).append(n)
    for shard, names in sorted(by_shard.items()):
        p = root / shard
        if not p.exists():
            out["hints"].append(f"WARN шард отсутствует: {shard}")
            continue
        hdr = read_st_header(p)
        for n in names:
            e = hdr.get(n)
            if e is None:
                out["hints"].append(f"WARN {n}: в index есть, в {shard} нет")
            else:
                out["tensors"][n] = {"dtype": e["dtype"], "shape": e["shape"]}

    T = out["tensors"]

    def find(sub: str):
        for n, e in T.items():
            if sub in n:
                return n, e["shape"]
        return None, None

    kv_lora = out["config_text"].get("kv_lora_rank")
    rope = int(out["config_text"].get("qk_rope_head_dim") or 0)
    _, s = find("kv_a_proj_with_mqa.weight")
    if s and kv_lora:
        if s[0] == kv_lora:
            out["hints"].append(f"HINT kv_a out={s[0]} == kv_lora -> ЧИСТЫЙ NoPE")
        elif s[0] == kv_lora + rope:
            out["hints"].append(f"HINT kv_a out={s[0]} == kv_lora+rope({rope}) -> rope-хвост ЕСТЬ")
        else:
            out["hints"].append(f"HINT kv_a out={s[0]} — нестандарт, разбирать")

    heads = out["config_text"].get("num_attention_heads")
    _, qb = find("layers.3.self_attn.q_b_proj.weight")
    if qb and heads:
        nope = int(out["config_text"].get("qk_nope_head_dim") or 0)
        out["hints"].append(
            f"HINT q_b out={qb[0]} -> {qb[0]/heads:g}/голову (nope={nope}, nope+rope={nope+rope})")

    for n in list(T):
        if "layers.0.self_attn" in n and (n.endswith("A_log") or n.endswith("dt_bias")):
            out["hints"].append(f"HINT {n.split('self_attn.')[-1]} shape={T[n]['shape']}")

    _, rd = find("routed_expert_down_proj.weight")
    if rd:
        out["hints"].append(f"HINT routed_expert_down_proj {rd} -> latent={rd[0]}? (или [latent,hidden] иная ось)")
    _, ar = find("output_attn_res_proj.weight")
    if ar:
        out["hints"].append(f"HINT output_attn_res_proj {ar}")
    for suf in ("weight_packed", "weight_scale"):
        n, sh = find(f"experts.0.w1.{suf}")
        if n:
            out["hints"].append(f"HINT w1.{suf}: dtype={T[n]['dtype']} shape={sh}")

    if not T:
        out["hints"].append("WARN 0 тензоров сматчено — смотри census/samples и правь SUBSTRINGS")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"JSON -> {args.json}", file=sys.stderr)

    print("=" * 72)
    print("## config TOP:", json.dumps(out["config_top"], ensure_ascii=False))
    print("## config TEXT (выборочно)")
    for k, v in out["config_text"].items():
        print(f"  {k} = {v}")
    print("## generation_config:", out["generation_config"])
    print("## tokenizer:", out["tokenizer"])
    print("## census L1:", out["census"]["level1"])
    print("## census L2:", out["census"]["level2"])
    print("## samples:", *out["raw_name_samples"], sep="\n   ")
    print("=" * 72)
    print(f"## тензоры ({len(T)})")
    w = max((len(n) for n in T), default=10)
    for n in sorted(T):
        print(f"  {n:<{w}}  {T[n]['dtype']:<9} {T[n]['shape']}")
    print("=" * 72)
    for h in out["hints"]:
        print(h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
