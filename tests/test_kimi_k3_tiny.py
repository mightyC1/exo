"""Tiny Kimi K3 smoke (A3-подготовка, план §7.3).

Запускается на ноде с mlx (НЕ в CI без Metal):
  cd ~/Desktop/ai/exo && .venv/bin/python -m pytest tests/test_kimi_k3_tiny.py -x -q

Проверяет на tiny-конфиге (13 слоёв, hidden 256, 4 головы — кратно TP4):
  - модель строится из nested config (text_config), forward без NaN;
  - prefill+decode с hybrid cache (ArraysCache для KDA / KVCache для MLA);
  - chunked prefill == single-shot (INV#6/A4, tolerances bf16);
  - AttnRes boundary на слое 12 (block 0 = 0..11, block 1 = 12) не ломает shapes;
  - обе A_log-семантики выполняются (численный вердикт — P0-008b, не здесь);
  - SwitchGLU top-2 роутинг детерминирован (fp32 router).

Это smoke, НЕ parity: parity-референс (pure-torch KDA / llama.cpp CPU) — A3.
"""

from __future__ import annotations

import os

import pytest

mx = pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402


TINY_CFG = {
    "model_type": "kimi_k3",
    "text_config": {
        "model_type": "kimi_k3",
        "vocab_size": 512,
        "hidden_size": 256,
        "num_hidden_layers": 13,
        "num_attention_heads": 4,
        "head_dim": 64,
        "intermediate_size": 512,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        # KDA на 1-based слоях: все, кроме MLA {4, 8, 12, 13}
        "linear_attn_config": {
            "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11],
            "num_heads": 4,
            "head_dim": 64,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        "q_lora_rank": 64,
        "kv_lora_rank": 64,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 0,
        "v_head_dim": 64,
        "num_experts": 8,
        "num_experts_per_token": 2,
        "num_shared_experts": 1,
        "moe_intermediate_size": 64,
        "moe_latent_dim": 128,
        "first_k_dense_replace": 1,
        "attn_res_block_size": 12,
        "situ_beta": 4.0,
        "situ_linear_beta": 25.0,
    },
}


def _build():
    from exo.worker.engines.mlx.vendor.custom_models import register_custom_models

    register_custom_models()
    from exo.worker.engines.mlx.vendor.kimi_k3 import Model, ModelArgs

    args = ModelArgs.from_dict(TINY_CFG)
    model = Model(args)
    # random init всех параметров детерминированно
    mx.random.seed(0)

    def init(p):
        return mx.random.normal(p.shape).astype(p.dtype) * 0.02

    model.update(model.parameters())  # materialize tree
    import mlx.utils as mu

    flat = dict(mu.tree_flatten(model.parameters()))
    model.update(mu.tree_unflatten([(k, init(v)) for k, v in flat.items()]))
    mx.eval(model.parameters())
    return model, args


def test_forward_and_cache():
    model, args = _build()
    B, T = 1, 33
    tokens = mx.random.randint(0, args.vocab_size, (B, T))

    # single-shot prefill
    logits = model(tokens)
    mx.eval(logits)
    assert logits.shape == (B, T, args.vocab_size)
    assert bool(mx.all(mx.isfinite(logits.astype(mx.float32))))

    # prefill + 4 decode-шага c hybrid cache
    cache = model.make_cache()
    out = model(tokens, cache=cache)
    mx.eval(out)
    step = tokens[:, -1:]
    for _ in range(4):
        out = model(step, cache=cache)
        mx.eval(out)
        assert out.shape == (B, 1, args.vocab_size)
        assert bool(mx.all(mx.isfinite(out.astype(mx.float32))))
        step = mx.argmax(out[:, -1:, :], axis=-1)


def test_chunked_prefill_matches_single_shot():
    model, args = _build()
    B, T, chunk = 1, 32, 8
    tokens = mx.random.randint(0, args.vocab_size, (B, T))

    single = model(tokens)
    mx.eval(single)
    last_single = single[:, -1, :].astype(mx.float32)

    cache = model.make_cache()
    out = None
    for s in range(0, T, chunk):
        out = model(tokens[:, s : s + chunk], cache=cache)
        mx.eval(out)
    last_chunked = out[:, -1, :].astype(mx.float32)

    diff = mx.abs(last_single - last_chunked)
    max_abs = float(mx.max(diff))
    denom = float(mx.max(mx.abs(last_single))) + 1e-9
    rel = max_abs / denom
    # bf16-накопление: порог утверждается на A3/A4; здесь smoke-допуск
    assert rel < 5e-2, f"chunked != single-shot: max_abs={max_abs}, rel={rel}"


@pytest.mark.parametrize("semantics", ["per_head", "per_channel"])
def test_alog_semantics_both_run(semantics, monkeypatch):
    # обе интерпретации ДОЛЖНЫ выполняться; какая верна — решает P0-008b
    import exo.worker.engines.mlx.vendor.kimi_k3 as k3

    monkeypatch.setattr(k3, "ALOG_SEMANTICS", semantics)
    model, args = _build()
    logits = model(mx.random.randint(0, args.vocab_size, (1, 9)))
    mx.eval(logits)
    assert bool(mx.all(mx.isfinite(logits.astype(mx.float32))))


def test_router_fp32_deterministic():
    model, args = _build()
    x = mx.random.randint(0, args.vocab_size, (1, 17))
    a = model(x)
    b = model(x)
    mx.eval(a, b)
    assert bool(mx.all(a == b)), "router/top-k недетерминирован при одинаковом входе"
