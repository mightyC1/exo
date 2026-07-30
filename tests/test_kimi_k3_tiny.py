# Smoke для vendored апстримного kimi_k3 (mlx-lm PR #1626).
# Нода: .venv/bin/python -m pytest tests/test_kimi_k3_tiny.py -x -q (или из /tmp)
from __future__ import annotations
import pytest
mx = pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

TINY = {
    "model_type": "kimi_k3", "vocab_size": 256, "hidden_size": 128,
    "num_hidden_layers": 5, "num_attention_heads": 4, "num_key_value_heads": 4,
    "intermediate_size": 192, "hidden_act": "situ", "rms_norm_eps": 1e-5,
    "max_position_embeddings": 4096, "q_lora_rank": 64, "kv_lora_rank": 32,
    "qk_nope_head_dim": 32, "qk_rope_head_dim": 16, "v_head_dim": 32,
    "mla_use_nope": True, "mla_use_output_gate": True,
    "num_experts": 8, "num_experts_per_token": 2, "num_shared_experts": 1,
    "moe_intermediate_size": 64, "routed_expert_hidden_size": 96,
    "latent_moe_use_norm": True, "moe_router_activation_func": "sigmoid",
    "moe_renormalize": True, "routed_scaling_factor": 1.0,
    "first_k_dense_replace": 1, "moe_layer_freq": 1, "num_expert_group": 1,
    "topk_group": 1, "topk_method": "noaux_tc",
    "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
    "attn_res_block_size": 4, "tie_word_embeddings": False,
    "linear_attn_config": {
        "kda_layers": [1, 2, 3], "full_attn_layers": [4, 5],
        "head_dim": 32, "num_heads": 4, "gate_lower_bound": -5.0,
        "short_conv_kernel_size": 4, "use_full_rank_gate": True,
    },
}


def _model():
    import exo.worker.engines.mlx.vendor.kimi_k3 as k3
    m = k3.Model(k3.ModelArgs.from_dict(dict(TINY)))
    mx.eval(m.parameters())
    return k3, m


def test_forward_cache_and_chunking():
    k3, m = _model()
    toks = mx.array([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]])
    ref = m(toks)
    assert ref.shape == (1, 12, TINY["vocab_size"]) and bool(mx.isfinite(ref).all())
    cache = m.make_cache()
    for s in range(0, 12, 4):
        out = m(toks[:, s : s + 4], cache=cache)
    rel = float(mx.abs(out[:, -1] - ref[:, -1]).max()) / (float(mx.abs(ref[:, -1]).max()) or 1.0)
    assert rel < 5e-2, rel
    for t in range(3):
        out = m(mx.array([[7 + t]]), cache=cache)
        assert bool(mx.isfinite(out).all())


def test_sanitize_official_names_strict_load():
    """Официальные имена (language_model.*, conv1d [C,1,K], per-expert w1/w2/w3
    U8-packed, fused kv_b, A_log [head_dim], vision-мусор) -> их sanitize ->
    quantize(mode=mxfp4 по scales) -> strict load -> forward."""
    import numpy as np
    import mlx.utils as mu
    k3, ref = _model()
    P = "language_model."
    args = ref.args.text_config if hasattr(ref, "args") else None
    ckpt = {"vision_tower.enc.w": mx.zeros((3, 3))}
    params = dict(mu.tree_flatten(ref.parameters()))
    heads, nope, vh = TINY["num_attention_heads"], TINY["qk_nope_head_dim"], TINY["v_head_dim"]
    H, D = 4, 32
    for path, w in params.items():
        assert path.startswith(P)
        base = path[len(P):]
        if ".switch_mlp." in base:
            li = base.split(".")[2]
            src = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}[base.split(".")[-2]]
            for e in range(TINY["num_experts"]):
                wq, sc = mx.quantize(w[e].astype(mx.bfloat16), group_size=32, bits=4, mode="mxfp4")
                u8 = mx.array(np.ascontiguousarray(np.array(wq)).view(np.uint8))
                b = f"{P}model.layers.{li}.block_sparse_moe.experts.{e}.{src}"
                ckpt[b + ".weight_packed"] = u8
                ckpt[b + ".weight_scale"] = sc
            continue
        if base.endswith(("embed_q.weight", "unembed_out.weight")):
            continue
        if ".qkv_proj." in base or ".qkv_conv." in base:
            continue  # соберём раздельными q/k/v ниже
        if base.endswith("A_log"):
            ckpt[path] = mx.concatenate([w.reshape(-1), mx.zeros((D - H,), dtype=w.dtype)])
            continue
        # их атрибут mlp у MoE-слоёв <- имя чекпоинта block_sparse_moe
        name = path
        for li in range(TINY["num_hidden_layers"]):
            if f".layers.{li}.mlp." in name and li >= TINY["first_k_dense_replace"]:
                name = name.replace(f".layers.{li}.mlp.", f".layers.{li}.block_sparse_moe.")
        ckpt[name] = w
    # раздельные q/k/v + conv1d из fused
    for li in [i - 1 for i in TINY["linear_attn_config"]["kda_layers"]]:
        ap = f"{P}model.layers.{li}.self_attn"
        qkv = params[f"{ap}.qkv_proj.weight"]
        Pd = H * D
        for j, n in enumerate("qkv"):
            ckpt[f"{ap}.{n}_proj.weight"] = qkv[j * Pd : (j + 1) * Pd]
        cw = params[f"{ap}.qkv_conv.conv.weight"]  # [3P, K, 1]
        for j, n in enumerate("qkv"):
            ckpt[f"{ap}.{n}_conv1d.weight"] = cw[j * Pd : (j + 1) * Pd].moveaxis(1, 2)
    # fused kv_b
    for li in [i - 1 for i in TINY["linear_attn_config"]["full_attn_layers"]]:
        ap = f"{P}model.layers.{li}.self_attn"
        wk, wv = params[f"{ap}.embed_q.weight"], params[f"{ap}.unembed_out.weight"]
        fused = mx.concatenate([wk.swapaxes(-1, -2), wv], axis=1)
        ckpt[f"{ap}.kv_b_proj.weight"] = fused.reshape(heads * (nope + vh), -1)

    m = k3.Model(k3.ModelArgs.from_dict(dict(TINY)))
    w = m.sanitize(ckpt)
    nn.quantize(m, group_size=32, bits=4, mode="mxfp4",
                class_predicate=lambda p, mod: hasattr(mod, "to_quantized") and f"{p}.scales" in w)
    m.load_weights(list(w.items()), strict=True)
    out = m(mx.array([[1, 2, 3, 4]]))
    assert bool(mx.isfinite(out).all())


def test_alog_sliced_to_heads():
    k3, m = _model()
    import mlx.utils as mu
    for p, w in mu.tree_flatten(m.parameters()):
        if p.endswith("A_log"):
            assert w.shape == (TINY["linear_attn_config"]["num_heads"],)
