#!/usr/bin/env python3
"""GLM-5.2/5.3 MTP side-car extractor (phase 0 gate 0-A tooling).

Extracts ``model.layers.78.*`` (the nextn/MTP layer that Model.sanitize drops)
from the vendor source checkpoint tails and converts them with the SAME policy
as the main idxbf16 converts:

  * fp8 (e4m3, block-128 scale_inv) -> bf16 via the exact dequant math from the
    pinned ``mlx_lm.models.deepseek_v32.Model.sanitize`` (rltakashige @ 6a3df6cd).
    GLM-5.2 source is plain bf16 (no scale_inv) -> passthrough.
  * expert stack: ``mlp.experts.{e}.{gate,up,down}_proj`` -> ``mlp.switch_mlp.*``
    stacked [E, out, in] (mirror of sanitize).
  * absorbed MLA split: ``kv_b_proj`` -> ``embed_q`` / ``unembed_out``
    (mirror of sanitize, incl. int8 re-quantization).
  * int8 affine group_size=64 for the heavy linears (matches idxbf16 quant cfg).
  * kept bf16 (idxbf16 lesson + vendor modules_to_not_convert): the whole
    indexer (wq_b / wk / weights_proj / k_norm), eh_proj, mlp.gate.weight,
    all norms (enorm / hnorm / shared_head.norm / *_layernorm).
  * kept f32: mlp.gate.e_score_correction_bias (cast_predicate of the pin).

Output: ``<dest>/mtp.safetensors`` + ``<dest>/mtp.manifest.json``.
The runtime side-loader (patches/glm52_mtp.py, phase 1) is presence-driven:
a module is quantized iff ``<name>.scales`` exists in the file — config flags
are not consulted (5.2/5.3 idxbf16 configs differ cosmetically here).

Run with the EXO venv python (needs mlx + the pinned mlx-lm for --self-test):

  .venv/bin/python tools/glm52_mtp_extract_sidecar.py --self-test
  .venv/bin/python tools/glm52_mtp_extract_sidecar.py \
      --src /Volumes/Models2/fp8-mtp-tails/GLM-5.3 \
      --dest /Volumes/Models2/local--GLM-5.3-8bit-idxbf16
"""

from __future__ import annotations

import argparse


def _base_provenance(src):
    """Bind the side-car to the exact base checkpoint it was extracted from
    (audit 0061): SHA-256 of the source config.json + model_type. Missing
    config -> no binding keys (loader warns about a legacy manifest)."""
    import hashlib as _hl
    import json as _js
    cfg = src / "config.json"
    if not cfg.is_file():
        return {}
    out = {"base_config_sha256": _hl.sha256(cfg.read_bytes()).hexdigest()}
    try:
        out["base_model_type"] = str(_js.loads(cfg.read_text()).get("model_type", ""))
    except Exception:
        pass
    return out
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

MTP_LAYER = 78
PREFIX = f"model.layers.{MTP_LAYER}."
GROUP_SIZE = 64
BITS = 8
FP8_BLOCK = 128

# Final tensor names (post-transform) that stay bf16 by policy.
_BF16_KEEP_SUBSTR = (
    ".self_attn.indexer.",   # wq_b / wk / weights_proj / k_norm (idxbf16 lesson)
    ".eh_proj.",             # vendor keeps bf16 even in fp8 release
    ".mlp.gate.weight",      # routing-sensitive; vendor keeps bf16; ~3MB
    "norm",                  # enorm/hnorm/shared_head.norm/*_layernorm
)

# Final names that must be int8-quantized (module prefixes, relative to PREFIX).
_QUANT_MODULES = (
    "self_attn.q_a_proj",
    "self_attn.q_b_proj",
    "self_attn.kv_a_proj_with_mqa",
    "self_attn.o_proj",
    "mlp.switch_mlp.gate_proj",
    "mlp.switch_mlp.up_proj",
    "mlp.switch_mlp.down_proj",
    "mlp.shared_experts.gate_proj",
    "mlp.shared_experts.up_proj",
    "mlp.shared_experts.down_proj",
    # embed_q / unembed_out handled by the kv_b split path explicitly
)


def dequant_block_fp8(weight: mx.array, scale_inv: mx.array) -> mx.array:
    """Verbatim math of the pinned ds32 sanitize dequant (block-128)."""
    weight = mx.from_fp8(weight, dtype=mx.bfloat16)
    bs = FP8_BLOCK
    m, n = weight.shape
    pad_bottom = (-m) % bs
    pad_side = (-n) % bs
    weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
    weight = weight.reshape((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
    weight = (weight * scale_inv[:, None, :, None]).reshape(
        m + pad_bottom, n + pad_side
    )
    return weight[:m, :n].astype(mx.bfloat16)


def split_kv_b(
    kv_b: mx.array, *, num_heads: int, qk_nope: int, v_head: int
) -> tuple[mx.array, mx.array]:
    """Mirror of the pinned sanitize kv_b_proj -> embed_q / unembed_out split."""
    head_dim = qk_nope + v_head
    v = kv_b.reshape(num_heads, head_dim, -1)
    wk = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
    wv = mx.contiguous(v[:, qk_nope:, :])
    return wk, wv


def _bf16_keep(final_name: str) -> bool:
    return any(s in final_name for s in _BF16_KEEP_SUBSTR)


def _load_layer_tensors(src: Path) -> dict[str, mx.array]:
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    wanted = sorted(k for k in wmap if k.startswith(PREFIX))
    if not wanted:
        raise SystemExit(f"[MTP-SIDE-CAR] FAIL: no {PREFIX}* in {src}/index")
    by_shard: dict[str, list[str]] = {}
    for k in wanted:
        by_shard.setdefault(wmap[k], []).append(k)
    out: dict[str, mx.array] = {}
    for shard, keys in sorted(by_shard.items()):
        loaded = mx.load(str(src / shard))
        for k in keys:
            out[k] = loaded[k]
        mx.eval([out[k] for k in keys])
        mx.clear_cache()
        print(f"  loaded {len(keys):4d} tensors from {shard}")
    return out


def _dequant_all(raw: dict[str, mx.array]) -> dict[str, mx.array]:
    """fp8+scale_inv -> bf16; bf16 passthrough. Drops scale_inv keys."""
    out: dict[str, mx.array] = {}
    n_fp8 = 0
    for k, v in raw.items():
        if k.endswith("weight_scale_inv"):
            continue
        sk = k[: -len(".weight")] + ".weight_scale_inv" if k.endswith(".weight") else None
        if sk is not None and sk in raw:
            out[k] = dequant_block_fp8(v, raw[sk])
            n_fp8 += 1
        else:
            out[k] = v
        mx.eval(out[k])
    print(f"  dequant: {n_fp8} fp8 tensors -> bf16, "
          f"{len(out) - n_fp8} passthrough")
    return out


def _transform(weights: dict[str, mx.array], cfg: dict) -> dict[str, mx.array]:
    """Expert stack + kv_b split, mirroring the pinned sanitize for layer 78."""
    n_experts = int(cfg["n_routed_experts"])
    num_heads = int(cfg["num_attention_heads"])
    qk_nope = int(cfg["qk_nope_head_dim"])
    v_head = int(cfg["v_head_dim"])
    kv_lora = int(cfg["kv_lora_rank"])

    out = dict(weights)

    # Expert stack -> switch_mlp
    for m in ("gate_proj", "down_proj", "up_proj"):
        first = f"{PREFIX}mlp.experts.0.{m}.weight"
        if first not in out:
            raise SystemExit(f"[MTP-SIDE-CAR] FAIL: missing {first}")
        to_join = [
            out.pop(f"{PREFIX}mlp.experts.{e}.{m}.weight") for e in range(n_experts)
        ]
        out[f"{PREFIX}mlp.switch_mlp.{m}.weight"] = mx.stack(to_join)
        mx.eval(out[f"{PREFIX}mlp.switch_mlp.{m}.weight"])
        mx.clear_cache()

    # kv_b_proj -> embed_q / unembed_out
    kv_b_key = f"{PREFIX}self_attn.kv_b_proj.weight"
    kv_b = out.pop(kv_b_key)
    if kv_b.shape != (num_heads * (qk_nope + v_head), kv_lora):
        raise SystemExit(
            f"[MTP-SIDE-CAR] FAIL: kv_b_proj shape {kv_b.shape} != "
            f"({num_heads * (qk_nope + v_head)}, {kv_lora}) from config"
        )
    wk, wv = split_kv_b(kv_b, num_heads=num_heads, qk_nope=qk_nope, v_head=v_head)
    out[f"{PREFIX}self_attn.embed_q.weight"] = wk
    out[f"{PREFIX}self_attn.unembed_out.weight"] = wv
    mx.eval(wk, wv)
    return out


def _quantize(weights: dict[str, mx.array]) -> tuple[dict[str, mx.array], int, int]:
    out: dict[str, mx.array] = {}
    n_q = n_bf16 = 0
    quant_full = tuple(f"{PREFIX}{m}.weight" for m in _QUANT_MODULES) + (
        f"{PREFIX}self_attn.embed_q.weight",
        f"{PREFIX}self_attn.unembed_out.weight",
    )
    for k, v in sorted(weights.items()):
        if k in quant_full:
            if _bf16_keep(k):
                raise SystemExit(f"[MTP-SIDE-CAR] FAIL: policy conflict on {k}")
            if v.shape[-1] % GROUP_SIZE:
                raise SystemExit(
                    f"[MTP-SIDE-CAR] FAIL: {k} last dim {v.shape[-1]} "
                    f"not divisible by {GROUP_SIZE}"
                )
            wq, scales, biases = mx.quantize(
                v.astype(mx.bfloat16), group_size=GROUP_SIZE, bits=BITS, mode="affine"
            )
            base = k[: -len(".weight")]
            out[f"{base}.weight"] = wq
            out[f"{base}.scales"] = scales
            out[f"{base}.biases"] = biases
            mx.eval(wq, scales, biases)
            n_q += 1
        else:
            # keep dtype as delivered (bf16 for weights/norms, f32 for
            # e_score_correction_bias per the pin's cast_predicate)
            out[k] = v
            n_bf16 += 1
        mx.clear_cache()
    return out, n_q, n_bf16


def extract(src: Path, dest: Path) -> int:
    t0 = time.perf_counter()
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("model_type") != "glm_moe_dsa":
        raise SystemExit(f"[MTP-SIDE-CAR] FAIL: model_type={cfg.get('model_type')!r}")
    if int(cfg.get("num_nextn_predict_layers", 0)) != 1:
        raise SystemExit("[MTP-SIDE-CAR] FAIL: num_nextn_predict_layers != 1")
    if int(cfg.get("num_hidden_layers", 0)) != MTP_LAYER:
        raise SystemExit(
            f"[MTP-SIDE-CAR] FAIL: num_hidden_layers="
            f"{cfg.get('num_hidden_layers')} (expected {MTP_LAYER})"
        )

    print(f"[MTP-SIDE-CAR] src={src}")
    raw = _load_layer_tensors(src)
    weights = _dequant_all(raw)
    del raw
    mx.clear_cache()
    weights = _transform(weights, cfg)
    final, n_q, n_bf16 = _quantize(weights)
    del weights
    mx.clear_cache()

    # Structural sanity: every non-expert tensor family we expect is present.
    must_have = [
        "eh_proj.weight", "enorm.weight", "hnorm.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "shared_head.norm.weight",
        "mlp.gate.weight", "mlp.gate.e_score_correction_bias",
        "self_attn.indexer.wq_b.weight", "self_attn.indexer.wk.weight",
        "self_attn.indexer.weights_proj.weight",
        "self_attn.indexer.k_norm.weight", "self_attn.indexer.k_norm.bias",
        "self_attn.q_a_layernorm.weight", "self_attn.kv_a_layernorm.weight",
        "self_attn.embed_q.scales", "self_attn.unembed_out.scales",
        "mlp.switch_mlp.gate_proj.scales",
    ]
    missing = [m for m in must_have if f"{PREFIX}{m}" not in final]
    if missing:
        raise SystemExit(f"[MTP-SIDE-CAR] FAIL: missing after convert: {missing}")
    leftovers = [k for k in final if "scale_inv" in k or ".experts." in k
                 or k.endswith("kv_b_proj.weight")]
    if leftovers:
        raise SystemExit(f"[MTP-SIDE-CAR] FAIL: untransformed keys: {leftovers[:5]}")

    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "mtp.safetensors"
    tmp_path = dest / ".mtp.partial.safetensors"
    meta = {
        "format": "mlx-mtp-sidecar-v1",
        "source": str(src),
        "layer": str(MTP_LAYER),
        "quant": json.dumps({"group_size": GROUP_SIZE, "bits": BITS,
                             "mode": "affine"}),
    }
    # Atomic publish: write+fsync a temp file, then os.replace, so an
    # interrupted run never leaves a partial side-car under the final name.
    mx.save_safetensors(str(tmp_path), final, metadata=meta)
    with open(tmp_path, "rb+") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp_path, out_path)

    sha = hashlib.sha256()
    with open(out_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            sha.update(chunk)
    manifest = {
        "file": out_path.name,
        "layer": MTP_LAYER,
        "sha256": sha.hexdigest(),
        **_base_provenance(src),
        "bytes": out_path.stat().st_size,
        "tensors": len(final),
        "quantized_modules": n_q,
        "kept_unquantized": n_bf16,
        "policy": {
            "quant": {"group_size": GROUP_SIZE, "bits": BITS, "mode": "affine"},
            "bf16_keep": list(_BF16_KEEP_SUBSTR),
            "fp8_dequant": "ds32.sanitize block-128 (pin 6a3df6cd)",
        },
        "source": str(src),
    }
    man_tmp = dest / "mtp.manifest.json.tmp"
    with open(man_tmp, "w") as fh:
        fh.write(json.dumps(manifest, indent=2))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(man_tmp, dest / "mtp.manifest.json")
    print(
        f"[MTP-SIDE-CAR] OK dest={out_path} tensors={len(final)} "
        f"quantized={n_q} bf16/f32={n_bf16} "
        f"bytes={out_path.stat().st_size:,} sha256={sha.hexdigest()[:16]}… "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    return 0


# ---------------------------------------------------------------------------
# --self-test: operator-level parity vs the pinned ds32 sanitize
# ---------------------------------------------------------------------------

def self_test() -> int:
    """Byte-exact parity of dequant / expert-stack / kv_b-split vs the pin."""
    from mlx_lm.models import deepseek_v32 as ds

    mx.random.seed(0)
    args = ds.ModelArgs(
        model_type="deepseek_v32", vocab_size=64, hidden_size=32,
        index_head_dim=16, index_n_heads=2, index_topk=4,
        intermediate_size=64, moe_intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        n_shared_experts=1, n_routed_experts=4, routed_scaling_factor=1.0,
        kv_lora_rank=64, q_lora_rank=32, qk_rope_head_dim=16, v_head_dim=64,
        qk_nope_head_dim=64, topk_method="noaux_tc", scoring_func="sigmoid",
        norm_topk_prob=True, n_group=1, topk_group=1, num_experts_per_tok=2,
        moe_layer_freq=1, first_k_dense_replace=0, max_position_embeddings=256,
        rms_norm_eps=1e-6, rope_theta=10000.0,
    )
    model = ds.Model(args)
    H, nope, vh, lora = 4, 64, 64, 64

    # fp8 dequant parity: feed layer-0 tensor through real sanitize
    w_u8 = mx.random.randint(0, 256, (200, 300)).astype(mx.uint8)
    s_inv = mx.random.uniform(0.5, 2.0, (2, 3)).astype(mx.float32)
    ref = model.sanitize({
        "model.layers.0.self_attn.o_proj.weight": w_u8,
        "model.layers.0.self_attn.o_proj.weight_scale_inv": s_inv,
    })["model.layers.0.self_attn.o_proj.weight"]
    mine = dequant_block_fp8(w_u8, s_inv)
    assert mx.array_equal(ref, mine).item(), "dequant parity FAILED"

    # expert stack + kv_b split parity on layer 0 (unquantized path)
    weights = {}
    for e in range(4):
        for m in ("gate_proj", "down_proj", "up_proj"):
            weights[f"model.layers.0.mlp.experts.{e}.{m}.weight"] = (
                mx.random.normal((8, 16)).astype(mx.bfloat16))
    kv_b = mx.random.normal((H * (nope + vh), lora)).astype(mx.bfloat16)
    weights["model.layers.0.self_attn.kv_b_proj.weight"] = kv_b
    ref = model.sanitize(dict(weights))

    renamed = {k.replace("layers.0.", f"layers.{MTP_LAYER}."): v
               for k, v in weights.items()}
    cfg = {"n_routed_experts": 4, "num_attention_heads": H,
           "qk_nope_head_dim": nope, "v_head_dim": vh, "kv_lora_rank": lora}
    mine = _transform(renamed, cfg)

    for m in ("gate_proj", "down_proj", "up_proj"):
        a = ref[f"model.layers.0.mlp.switch_mlp.{m}.weight"]
        b = mine[f"{PREFIX}mlp.switch_mlp.{m}.weight"]
        assert mx.array_equal(a, b).item(), f"switch_mlp.{m} parity FAILED"
    for m in ("embed_q", "unembed_out"):
        a = ref[f"model.layers.0.self_attn.{m}.weight"]
        b = mine[f"{PREFIX}self_attn.{m}.weight"]
        assert mx.array_equal(a, b).item(), f"{m} parity FAILED"

    # quantize round-trip smoke: shapes + finite error
    wq, sc, bi = mx.quantize(kv_b, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    deq = mx.dequantize(wq, sc, bi, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    err = mx.abs(deq - kv_b.astype(deq.dtype)).max().item()
    assert err < 0.05, f"int8 roundtrip err too high: {err}"

    print("[MTP-SIDE-CAR] self-test OK "
          "(dequant / switch_mlp / embed_q+unembed_out parity vs pin, int8 rt)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, help="dir with source shards+index+config")
    ap.add_argument("--dest", type=Path, help="model dir to place mtp.safetensors")
    ap.add_argument("--self-test", action="store_true")
    ns = ap.parse_args()
    if ns.self_test:
        return self_test()
    if not ns.src or not ns.dest:
        ap.error("--src and --dest are required (or use --self-test)")
    return extract(ns.src, ns.dest)


if __name__ == "__main__":
    sys.exit(main())
