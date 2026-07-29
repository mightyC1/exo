# Tiny smoke для vendor/kimi_k3.py (K3-семантика по официальному modeling_kimi_linear).
# Запуск на ноде: .venv/bin/python -m pytest tests/test_kimi_k3_tiny.py -x -q
# (в контейнере/CI: из /tmp, мимо conftest exo_tools)

from __future__ import annotations

import importlib
import os

import pytest

mx = pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
import mlx.utils as mu  # noqa: E402


TINY_TEXT = {
    "model_type": "kimi_linear",
    "vocab_size": 256,
    "hidden_size": 128,
    "num_hidden_layers": 13,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 192,
    "hidden_act": "situ",
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 4096,
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 16,
    "v_head_dim": 32,
    "mla_use_nope": True,
    "mla_use_output_gate": True,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "num_shared_experts": 1,
    "moe_intermediate_size": 64,
    "routed_expert_hidden_size": 96,
    "latent_moe_use_norm": True,
    "moe_router_activation_func": "sigmoid",
    "moe_renormalize": True,
    "routed_scaling_factor": 1.0,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "num_expert_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
    "attn_res_block_size": 4,
    "tie_word_embeddings": False,
    "linear_attn_config": {
        "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11],
        "full_attn_layers": [4, 8, 12, 13],
        "head_dim": 32,
        "num_heads": 4,
        "gate_lower_bound": -5.0,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
    },
}
TINY_CFG = {"model_type": "kimi_k3", "text_config": TINY_TEXT}


def _load_module():
    import exo.worker.engines.mlx.vendor.kimi_k3 as k3
    return importlib.reload(k3)


def _build(k3):
    args = k3.ModelArgs.from_dict(TINY_CFG)
    model = k3.Model(args)
    mx.eval(model.parameters())
    return args, model


def test_forward_and_cache():
    k3 = _load_module()
    args, model = _build(k3)
    x = mx.array([[1, 2, 3, 4, 5, 6, 7]])
    out = model(x)
    assert out.shape == (1, 7, TINY_TEXT["vocab_size"])
    assert bool(mx.isfinite(out).all())

    cache = model.make_cache()
    out = model(x, cache=cache)
    assert bool(mx.isfinite(out).all())
    for step in range(4):
        out = model(mx.array([[9 + step]]), cache=cache)
        assert out.shape[1] == 1
        assert bool(mx.isfinite(out).all())


def test_chunked_prefill_matches_single_shot():
    k3 = _load_module()
    args, model = _build(k3)
    toks = mx.array([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]])

    ref = model(toks)[:, -1]

    cache = model.make_cache()
    for s in range(0, toks.shape[1], 4):
        out = model(toks[:, s : s + 4], cache=cache)
    got = out[:, -1]

    denom = float(mx.abs(ref).max()) or 1.0
    rel = float(mx.abs(got - ref).max()) / denom
    assert rel < 5e-2, f"chunked vs single rel={rel}"


@pytest.mark.parametrize("semantics", ["per_head", "per_channel"])
def test_alog_semantics_paths_run(semantics, monkeypatch):
    monkeypatch.setenv("EXO_K3_ALOG_SEMANTICS", semantics)
    k3 = _load_module()
    assert k3.ALOG_SEMANTICS == semantics
    args, model = _build(k3)
    out = model(mx.array([[1, 2, 3]]))
    assert bool(mx.isfinite(out).all())


def test_remap_roundtrip_from_hf_names(monkeypatch):
    """Синтетический чекпоинт с ОФИЦИАЛЬНЫМИ именами (language_model.*,
    conv1d [C,1,K], A_log [head_dim] как в реальном K3, пер-экспертные
    w1/w2/w3, fused kv_b, vision-мусор) -> remap -> strict load -> forward."""
    monkeypatch.delenv("EXO_K3_ALOG_SEMANTICS", raising=False)  # auto
    k3 = _load_module()
    args, model = _build(k3)

    H = args.kda_num_heads
    D = args.kda_head_dim
    nope, vh = args.qk_nope_head_dim, args.v_head_dim
    heads = args.num_attention_heads
    lat = args.routed_expert_hidden_size

    ckpt = {"vision_tower.encoder.layers.0.fc.weight": mx.zeros((4, 4))}
    for path, w in mu.tree_flatten(model.parameters()):
        name = "language_model." + path
        if ".switch_mlp." in path:
            li = path.split(".")[2]
            src = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}[
                path.split(".")[-2]
            ]
            for e in range(args.num_experts):
                ckpt[
                    f"language_model.model.layers.{li}.block_sparse_moe.experts.{e}.{src}.weight"
                ] = w[e]
            continue
        if path.endswith("embed_q.weight") or path.endswith("unembed_out.weight"):
            continue  # соберём fused kv_b ниже
        if ".conv.weight" in path:
            name = "language_model." + path.replace(
                "q_conv.conv.weight", "q_conv1d.weight"
            ).replace("k_conv.conv.weight", "k_conv1d.weight").replace(
                "v_conv.conv.weight", "v_conv1d.weight"
            )
            ckpt[name] = w.moveaxis(1, 2)  # mlx [C,K,1] -> torch [C,1,K]
            continue
        if path.endswith("A_log"):
            assert w.size == D, "auto-режим: A_log должен строиться как [head_dim]"
            ckpt[name] = w.reshape(-1)  # [head_dim], как в реальном K3
            continue
        ckpt[name] = w

    # fused kv_b из embed_q/unembed_out
    params = dict(mu.tree_flatten(model.parameters()))
    for li, layer in enumerate(model.layers):
        if layer.is_linear:
            continue
        ap = f"model.layers.{li}.self_attn"
        wk = params[f"{ap}.embed_q.weight"]      # [H, lat, nope]
        wv = params[f"{ap}.unembed_out.weight"]  # [H, vh, lat]
        fused = mx.concatenate([wk.swapaxes(-1, -2), wv], axis=1)  # [H, nope+vh, lat]
        ckpt["language_model." + ap + ".kv_b_proj.weight"] = fused.reshape(
            heads * (nope + vh), -1
        )

    remapped = model.sanitize(ckpt)
    assert not any("vision_tower" in k for k in remapped)
    model.load_weights(list(remapped.items()), strict=True)
    out = model(mx.array([[1, 2, 3, 4]]))
    assert bool(mx.isfinite(out).all())


def test_router_deterministic():
    k3 = _load_module()
    args, model = _build(k3)
    moe = None
    for layer in model.layers:
        if getattr(layer, "is_moe", False):
            moe = layer.block_sparse_moe
            break
    assert moe is not None
    x = mx.random.normal((2, 5, args.hidden_size)).astype(mx.bfloat16)
    i1, w1 = moe.gate(x)
    i2, w2 = moe.gate(x)
    assert bool((i1 == i2).all())
    assert float(mx.abs(w1 - w2).max()) == 0.0


def test_alog_shape_mismatch_is_loud(monkeypatch):
    """Форма A_log, не равная ни [num_heads], ни [head_dim] — громкий отказ,
    молчаливый срез запрещён (P0-008b)."""
    monkeypatch.delenv("EXO_K3_ALOG_SEMANTICS", raising=False)
    k3 = _load_module()
    args, model = _build(k3)
    with pytest.raises(ValueError):
        model.sanitize({"model.layers.0.self_attn.A_log": mx.zeros((7,))})


def test_kda_gate_matches_canonical_formula(monkeypatch):
    """decay = exp(max(-exp(A_log)*softplus(a+dt_bias), lower_bound)):
    сверка с независимо посчитанной формулой + границы (exp(lb), 1]."""
    monkeypatch.delenv("EXO_K3_ALOG_SEMANTICS", raising=False)
    k3 = _load_module()
    H, D, lb = 4, 32, -5.0
    a = mx.random.normal((2, 3, H, D)) * 3.0
    A_log = mx.random.normal((D,)) * 0.5
    dt = mx.random.normal((H * D,)) * 0.5

    got = k3.compute_decay_k3(a, A_log, dt, lb, H, D)

    x = a.astype(mx.float32) + dt.astype(mx.float32).reshape(H, D)
    sp = mx.log(1.0 + mx.exp(x))
    ref = mx.exp(mx.maximum(-mx.exp(A_log.astype(mx.float32)).reshape(1, D) * sp, lb))
    assert float(mx.abs(got - ref).max()) < 1e-5
    assert float(got.max()) <= 1.0 + 1e-6
    assert float(got.min()) >= float(mx.exp(mx.array(lb))) - 1e-6

    # без lower_bound форма сводится к kimi_linear compute_g
    got2 = k3.compute_decay_k3(a, A_log, dt, None, H, D)
    ref2 = mx.exp(-mx.exp(A_log.astype(mx.float32)).reshape(1, D) * sp)
    assert float(mx.abs(got2 - ref2).max()) < 1e-5
