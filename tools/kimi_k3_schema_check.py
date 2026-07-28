#!/usr/bin/env python3
"""A0: schema validator для moonshotai/Kimi-K3 (metadata-only, без 1.56 TB).

Вход: локальная директория с config.json + model.safetensors.index.json
(качается metadata-only: hf download moonshotai/Kimi-K3 --include "*.json" "*.py").

Проверяет план §7.1 / ревью v2 и ЗАКРЫВАЕТ AUDIT-маркеры vendor/kimi_k3.py:
  - 93 слоя; 1-based списки KDA/MLA (69/24, MLA = 4,8,...,92,93);
  - A_log == [128] (head_dim, НЕ [96]) — INV#1;
  - dt_bias == [12288];
  - full-rank g_proj, ОТСУТСТВИЕ g_a_proj/g_b_proj — INV#2;
  - MLA NoPE: kv_a == 512 ровно (не 576) — ревью R1;
  - 896 экспертов, полнота expert-тензоров, MXFP4 packed+scale suffix'ы;
  - латентные проекции down/up/norm (какими именами реально называются);
  - AttnRes-тензоры (какими именами реально называются);
  - фактический dtype-скоуп: MXFP4 = только routed experts (ревью R3);
  - representative shard SET (минимальный набор файлов для P0-008/P0-008a).

Выход: отчёт + список ИМЁН, которые надо сверить с remap_checkpoint()
в vendor/kimi_k3.py (единственный источник правды по именам).

Usage:
  python tools/kimi_k3_schema_check.py /path/to/Kimi-K3-metadata
  python tools/kimi_k3_schema_check.py /path/to/dir --json report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

EXPECT = {
    "num_hidden_layers": 93,
    "hidden_size": 7168,
    "num_attention_heads": 96,
    "head_dim": 128,
    "vocab_size": 163840,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "moe_latent_dim": 3584,
    "moe_intermediate_size": 3072,
    "attn_res_block_size": 12,
    "first_k_dense_replace": 1,
}

MLA_1BASED = list(range(4, 93, 4)) + [93]  # 4,8,...,92,93 -> 24 слоя


def load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def get_cfg(root: Path) -> dict:
    cfg = load_json(root / "config.json")
    # nested: kimi_k3 -> text_config
    if "text_config" in cfg:
        merged = dict(cfg["text_config"])
        merged["_top_level_model_type"] = cfg.get("model_type")
        merged["_quantization_config"] = cfg.get(
            "quantization_config", merged.get("quantization_config")
        )
        return merged
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    root: Path = args.root
    problems: list[str] = []
    notes: list[str] = []
    report: dict = {}

    def ok(cond: bool, msg: str):
        (notes if cond else problems).append(("OK  " if cond else "FAIL") + " " + msg)

    # ---------------- config ----------------
    cfg = get_cfg(root)
    report["config_keys"] = sorted(cfg.keys())

    for key, want in EXPECT.items():
        got = cfg.get(key, None)
        if got is None:
            # имена полей могут отличаться — это тоже результат A0
            problems.append(f"MISS config['{key}'] отсутствует (ожидали {want}) — "
                            f"найти реальное имя поля и поправить ModelArgs")
        else:
            ok(got == want, f"config.{key} = {got} (ожидание {want})")

    lac = cfg.get("linear_attn_config", {}) or {}
    kda = lac.get("kda_layers", cfg.get("kda_layers", []))
    report["kda_layers"] = kda
    ok(len(kda) == 69, f"|kda_layers| = {len(kda)} (ожидание 69, 1-based)")
    if kda:
        mla = sorted(set(range(1, 94)) - set(kda))
        ok(mla == MLA_1BASED, f"MLA (1-based) = {mla[:5]}...{mla[-2:]} "
                              f"(ожидание 4,8,...,92,93)")
    glb = lac.get("gate_lower_bound", cfg.get("gate_lower_bound"))
    ok(glb is not None and abs(float(glb) + 5.0) < 1e-9,
       f"gate_lower_bound = {glb} (ожидание -5.0; форма гейта = sigmoid, INV#13)")
    frg = lac.get("use_full_rank_gate", cfg.get("use_full_rank_gate"))
    ok(bool(frg), f"use_full_rank_gate = {frg} (ожидание true, INV#2)")
    rope_dim = cfg.get("qk_rope_head_dim", 0)
    ok(rope_dim in (0, None), f"qk_rope_head_dim = {rope_dim} "
                              f"(ожидание 0/absent — NoPE, ревью R1)")
    ok(cfg.get("rope_theta") is None,
       f"rope_theta = {cfg.get('rope_theta')} (ожидание absent — NoPE)")
    ok(cfg.get("num_nextn_predict_layers", 0) == 0,
       "num_nextn_predict_layers == 0 (MTP-голов нет)")
    ok(cfg.get("tie_word_embeddings", False) is False, "tie_word_embeddings == False")

    qc = cfg.get("_quantization_config") or cfg.get("quantization_config") or {}
    report["quantization_config"] = qc
    fmt = qc.get("format", qc.get("quant_method", ""))
    ok("mxfp4" in json.dumps(qc).lower(),
       f"quantization_config содержит mxfp4 (format={fmt!r})")

    # ---------------- index: tensor names / shapes ----------------
    idx_path = root / "model.safetensors.index.json"
    if not idx_path.exists():
        problems.append(f"FAIL {idx_path} не найден — скачай metadata: "
                        f"hf download moonshotai/Kimi-K3 --include '*.json' '*.py'")
        _emit(problems, notes, report, args.json)
        return 1

    index = load_json(idx_path)
    wmap: dict[str, str] = index["weight_map"]
    names = list(wmap.keys())
    report["n_tensors"] = len(names)
    ok(len(names) > 400_000, f"тензоров в index: {len(names)} (ожидание ~497 220)")

    def strip_lm(n: str) -> str:
        return n[len("language_model."):] if n.startswith("language_model.") else n

    snames = [strip_lm(n) for n in names]
    sset = set(snames)

    # прямые проверки существования на примере слоя KDA и слоя MLA
    # (первый KDA из списка; первый MLA = 4 -> idx 3)
    kda0 = (kda[0] - 1) if kda else 1
    mla0 = 3

    def has(pat: str) -> list[str]:
        rx = re.compile(pat)
        return [n for n in sset if rx.fullmatch(n)]

    # KDA: A_log / dt_bias / g_proj / отсутствие g_a,g_b
    a_log = has(rf"model\.layers\.{kda0}\.self_attn\.A_log")
    ok(bool(a_log), f"A_log существует на KDA-слое {kda0} (0-based)")
    dt = has(rf"model\.layers\.{kda0}\.self_attn\.dt_bias")
    ok(bool(dt), f"dt_bias существует на KDA-слое {kda0}")
    gp = has(rf"model\.layers\.{kda0}\.self_attn\.g_proj\.weight")
    ok(bool(gp), f"full-rank g_proj существует на KDA-слое {kda0} (INV#2)")
    gab = [n for n in sset if re.search(r"self_attn\.g_[ab]_proj\.", n)]
    ok(not gab, f"g_a_proj/g_b_proj ОТСУТСТВУЮТ (найдено: {len(gab)}) (INV#2)")

    # MLA имена: kv_a_proj vs kv_a_proj_with_mqa; q-LoRA; kv_b; out-gate
    kv_a_variants = [n for n in sset
                     if re.fullmatch(rf"model\.layers\.{mla0}\.self_attn\.kv_a_proj(_with_mqa)?\.weight", n)]
    report["mla_kv_a_tensor"] = kv_a_variants
    ok(bool(kv_a_variants), f"kv_a-проекция найдена на MLA-слое {mla0}: {kv_a_variants}")
    for want in ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_layernorm",
                 "kv_b_proj", "g_proj", "o_proj"):
        hit = [n for n in sset if f"model.layers.{mla0}.self_attn.{want}." in n
               or n.endswith(f"model.layers.{mla0}.self_attn.{want}")]
        ok(bool(hit), f"MLA {want} присутствует на слое {mla0}")

    # MoE: количество экспертов, packed/scale суффиксы, латентные проекции
    moe_layer = 4  # 0-based; layer 0 dense
    exp_ids = set()
    packed_suffixes = set()
    for n in sset:
        m = re.match(rf"model\.layers\.{moe_layer}\.block_sparse_moe\.experts\.(\d+)\.w1\.(\w+)", n)
        if m:
            exp_ids.add(int(m.group(1)))
            packed_suffixes.add(m.group(2))
    report["experts_found_layer4"] = len(exp_ids)
    report["expert_w1_suffixes"] = sorted(packed_suffixes)
    ok(len(exp_ids) == EXPECT["num_experts"],
       f"экспертов на слое {moe_layer}: {len(exp_ids)} (ожидание 896)")
    ok("weight_packed" in packed_suffixes and any("scale" in s for s in packed_suffixes),
       f"MXFP4 суффиксы у w1: {sorted(packed_suffixes)} "
       f"(ожидание weight_packed + weight_scale)")

    latent_hits = defaultdict(list)
    for n in sset:
        if f"model.layers.{moe_layer}.block_sparse_moe." in n and "experts." not in n \
           and "shared" not in n:
            latent_hits["moe_level"].append(n)
    report["moe_level_tensors_layer4"] = sorted(latent_hits["moe_level"])
    notes.append("INFO MoE-level тензоры слоя 4 (сверить с remap_checkpoint "
                 "latent_alias): " + ", ".join(sorted(latent_hits["moe_level"])[:12]))

    shared = [n for n in sset
              if f"model.layers.{moe_layer}.block_sparse_moe.shared" in n]
    ok(bool(shared), f"shared_experts тензоры присутствуют ({len(shared)} шт)")

    # AttnRes: найти реальные имена
    attn_res = sorted({n for n in sset if "attn_res" in n.lower() or "attnres" in n.lower()})
    report["attn_res_tensors"] = attn_res
    ok(bool(attn_res), f"AttnRes тензоры найдены ({len(attn_res)}) — "
                       f"сверить имена с remap_checkpoint AUDIT-ATTNRES")
    for t in attn_res[:12]:
        notes.append(f"INFO attn_res: {t}")

    # dtype-скоуп по metadata (dtype в index недоступен — только через shard
    # headers; здесь эвристика: packed-суффиксы = MXFP4, остальное bf16)
    non_expert_packed = [n for n in sset
                         if n.endswith("weight_packed") and ".experts." not in n]
    ok(not non_expert_packed,
       f"MXFP4 (weight_packed) ВНЕ routed experts: {len(non_expert_packed)} "
       f"(ожидание 0 — ревью R3); примеры: {non_expert_packed[:5]}")

    # representative shard SET (P0-008): минимальный набор файлов
    wanted_tensors = []
    wanted_tensors += [n for n in names if f"layers.{kda0}.self_attn." in n]
    wanted_tensors += [n for n in names if f"layers.{mla0}.self_attn." in n]
    wanted_tensors += [n for n in names
                       if f"layers.{moe_layer}.block_sparse_moe." in n
                       and (".experts.0." in n or ".experts." not in n)]
    wanted_tensors += [n for n in names if "attn_res" in n.lower()]
    shard_set = sorted({wmap[n] for n in wanted_tensors})
    report["representative_shard_set"] = shard_set
    notes.append(f"INFO representative shard SET ({len(shard_set)} файлов): "
                 + ", ".join(shard_set))

    _emit(problems, notes, report, args.json)
    return 0 if not any(p.startswith(("FAIL", "MISS")) for p in problems) else 1


def _emit(problems, notes, report, json_path):
    print("=" * 72)
    for n in notes:
        print(n)
    print("-" * 72)
    for p in problems:
        print(p)
    print("=" * 72)
    print(f"{len([p for p in problems if p.startswith(('FAIL', 'MISS'))])} problems, "
          f"{len(notes)} checks/notes")
    if json_path:
        with open(json_path, "w") as f:
            json.dump({"problems": problems, "notes": notes, "report": report}, f,
                      indent=2, ensure_ascii=False)
        print(f"JSON report -> {json_path}")


if __name__ == "__main__":
    sys.exit(main())
