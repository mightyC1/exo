"""Unit tests for the GLM-5.2/5.3 MTP block, side-load and shadow bookkeeping.

Runs on CPU mlx + the pinned mlx-lm. ``exo.worker.runner.bootstrap`` is
stubbed before importing the patch modules so the tests do not depend on the
full exo runtime (same trick works in a bare container and in the node venv).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def _stub_pkg(name: str, path: Path | None = None) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    if path is not None:
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
    return mod


class _StubLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, msg: str, *a, **k) -> None:
        self.lines.append(str(msg))

    def warning(self, msg: str, *a, **k) -> None:
        self.lines.append(str(msg))

    def opt(self, *a, **k) -> "_StubLogger":
        return self


sys.path.insert(0, str(SRC))
_stub_pkg("exo", SRC / "exo")  # dodge version("exo") in the real __init__
_stub_pkg("exo.worker", SRC / "exo" / "worker")
_stub_pkg("exo.worker.runner", SRC / "exo" / "worker" / "runner")
_boot = _stub_pkg("exo.worker.runner.bootstrap")
_boot.logger = _StubLogger()  # type: ignore[attr-defined]

from exo.worker.engines.mlx.patches.glm52_mtp import (  # noqa: E402
    GLM52MTPModule,
    _MTPState,
    _shadow_step,
    apply_glm52_mtp_patch,
    load_mtp_module,
)

LAYER = 78
HID, HEADS, NOPE, VH, ROPE, LORA, QLORA, EXPERTS = 64, 4, 64, 64, 16, 64, 64, 4


def _extractor():
    spec = importlib.util.spec_from_file_location(
        "glm52_mtp_extract_sidecar",
        REPO_ROOT / "tools" / "glm52_mtp_extract_sidecar.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _tiny_args():
    from mlx_lm.models import deepseek_v32 as ds

    return ds.ModelArgs(
        model_type="deepseek_v32", vocab_size=97, hidden_size=HID,
        index_head_dim=16, index_n_heads=2, index_topk=64,
        intermediate_size=64, moe_intermediate_size=64,
        num_hidden_layers=LAYER, num_attention_heads=HEADS,
        num_key_value_heads=HEADS, n_shared_experts=1,
        n_routed_experts=EXPERTS, routed_scaling_factor=1.0,
        kv_lora_rank=LORA, q_lora_rank=QLORA, qk_rope_head_dim=ROPE,
        v_head_dim=VH, qk_nope_head_dim=NOPE, topk_method="noaux_tc",
        scoring_func="sigmoid", norm_topk_prob=True, n_group=1, topk_group=1,
        num_experts_per_tok=2, moe_layer_freq=1, first_k_dense_replace=0,
        max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=10000.0,
    )


def _make_synth_src(d: Path) -> None:
    """Vendor-shaped bf16 layer-78 tail (5.2-style source)."""
    mx.random.seed(7)
    P = f"model.layers.{LAYER}."
    w: dict[str, mx.array] = {}

    def lin(name: str, o: int, i: int) -> None:
        w[P + name + ".weight"] = mx.random.normal((o, i)).astype(mx.bfloat16)

    for e in range(EXPERTS):
        lin(f"mlp.experts.{e}.gate_proj", 64, HID)
        lin(f"mlp.experts.{e}.up_proj", 64, HID)
        lin(f"mlp.experts.{e}.down_proj", HID, 64)
    lin("mlp.shared_experts.gate_proj", 64, HID)
    lin("mlp.shared_experts.up_proj", 64, HID)
    lin("mlp.shared_experts.down_proj", HID, 64)
    lin("self_attn.q_a_proj", QLORA, HID)
    lin("self_attn.q_b_proj", HEADS * (NOPE + ROPE), QLORA)
    lin("self_attn.kv_a_proj_with_mqa", LORA + ROPE, HID)
    lin("self_attn.kv_b_proj", HEADS * (NOPE + VH), LORA)
    lin("self_attn.o_proj", HID, HEADS * VH)
    lin("self_attn.indexer.wq_b", 2 * 16, QLORA)
    lin("self_attn.indexer.wk", 16, HID)
    lin("self_attn.indexer.weights_proj", 2, HID)
    lin("eh_proj", HID, 2 * HID)
    lin("mlp.gate", EXPERTS, HID)
    for n, dim in [
        ("enorm", HID), ("hnorm", HID), ("input_layernorm", HID),
        ("post_attention_layernorm", HID), ("shared_head.norm", HID),
        ("self_attn.q_a_layernorm", QLORA), ("self_attn.kv_a_layernorm", LORA),
    ]:
        w[P + n + ".weight"] = mx.ones((dim,)).astype(mx.bfloat16)
    w[P + "self_attn.indexer.k_norm.weight"] = mx.ones((16,)).astype(mx.bfloat16)
    w[P + "self_attn.indexer.k_norm.bias"] = mx.zeros((16,)).astype(mx.bfloat16)
    w[P + "mlp.gate.e_score_correction_bias"] = mx.zeros((EXPERTS,)).astype(
        mx.float32
    )
    mx.save_safetensors(str(d / "model-00001-of-00001.safetensors"), w)
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00001-of-00001.safetensors" for k in w}}
    ))
    (d / "config.json").write_text(json.dumps({
        "model_type": "glm_moe_dsa", "num_nextn_predict_layers": 1,
        "num_hidden_layers": LAYER, "n_routed_experts": EXPERTS,
        "num_attention_heads": HEADS, "qk_nope_head_dim": NOPE,
        "v_head_dim": VH, "kv_lora_rank": LORA,
    }))


@pytest.fixture(scope="module")
def sidecar(tmp_path_factory) -> Path:
    src = tmp_path_factory.mktemp("mtp-src")
    dest = tmp_path_factory.mktemp("mtp-dest")
    _make_synth_src(src)
    assert _extractor().extract(src, dest) == 0
    return dest / "mtp.safetensors"


def _load(sidecar: Path) -> GLM52MTPModule:
    return load_mtp_module(
        _tiny_args(), sidecar, layer_idx=LAYER,
        quant={"group_size": 64, "bits": 8, "mode": "affine"},
        logger=_StubLogger(),
    )


def test_sideload_presence_driven_quantization(sidecar: Path) -> None:
    m = _load(sidecar)
    # int8 where the side-car carries scales…
    assert isinstance(m.block.self_attn.q_a_proj, nn.QuantizedLinear)
    assert hasattr(m.block.self_attn.embed_q, "scales")
    assert hasattr(m.block.mlp.switch_mlp.gate_proj, "scales")
    # …plain bf16 where it does not (indexer / eh_proj / gate).
    assert isinstance(m.block.self_attn.indexer.wq_b, nn.Linear)
    assert m.block.self_attn.indexer.wq_b.weight.dtype == mx.bfloat16
    assert isinstance(m.eh_proj, nn.Linear)
    assert m.block.mlp.gate.weight.dtype == mx.bfloat16
    assert m.block.mlp.gate.e_score_correction_bias.dtype == mx.float32


@pytest.mark.parametrize("concat", ["eh", "he"])
def test_draft_forward_advances_cache(sidecar: Path, concat: str) -> None:
    args = _tiny_args()
    m = _load(sidecar)
    embed = nn.Embedding(args.vocab_size, HID)
    lm_head = nn.Linear(HID, args.vocab_size, bias=False)
    mx.eval(embed.parameters(), lm_head.parameters())
    state = _MTPState(
        mode="shadow", concat=concat, mtp=m, embed=embed, lm_head=lm_head,
        store={}, logger=_StubLogger(),
    )
    state.start_request(uid=1)
    h = mx.random.normal((1, 1, HID)).astype(mx.bfloat16)
    for step in range(1, 5):
        d = state.draft(h, mx.array([step % 7], dtype=mx.uint32))
        mx.eval(d, *state.mtp_cache_arrays())
        assert d.shape == (1,)
        offs = (state.mtp_cache[0].offset, state.mtp_cache[1].offset)
        assert offs == (step, step), offs
    assert 0 <= int(d.item()) < args.vocab_size


def test_shadow_delayed_accounting(sidecar: Path) -> None:
    """One-step-delayed match accounting over a scripted token stream."""
    m = _load(sidecar)
    log = _StubLogger()
    state = _MTPState(
        mode="shadow", concat="eh", mtp=m,
        embed=nn.Embedding(97, HID), lm_head=nn.Linear(HID, 97, bias=False),
        store={}, logger=log,
    )
    # tokens the fake main model "samples" per step; drafts scripted so that
    # draft@k predicts sample@k+1: hits at k=1,3 (0-based), misses at 0,2.
    samples = [10, 20, 30, 40, 50, 60]
    drafts = [99, 30, 99, 50, 99, 99]  # draft@k predicts samples[k+1]
    expected_hits = [False, True, False, True]  # comparisons for k=0..3

    class FakeBatch:
        uids = [7]
        model = types.SimpleNamespace()
        _next_tokens = None

    calls = {"k": -1}

    def prev_step(batch: FakeBatch):
        calls["k"] += 1
        batch._next_tokens = mx.array([samples[calls["k"]]], dtype=mx.uint32)
        state.store["h"] = mx.zeros((1, 1, HID)).astype(mx.bfloat16)
        return [123], [mx.zeros((1,))]

    def scripted_draft(h_last, next_tok):
        return mx.array([drafts[calls["k"]]])

    state.draft = scripted_draft  # type: ignore[method-assign]
    batch = FakeBatch()
    n = len(samples)
    for _ in range(n):
        out = _shadow_step(state, prev_step, batch)
        assert out[0] == [123]  # prev_step result passed through untouched
    # after n steps, comparisons harvested = n-2 (one-step scheduling delay)
    assert state.steps == n - 2
    assert state.matches == sum(expected_hits[: n - 2])
    assert state.win == expected_hits[: n - 2]


def test_apply_fail_closed(tmp_path: Path, monkeypatch) -> None:
    log = _StubLogger()
    model = types.SimpleNamespace()

    monkeypatch.delenv("EXO_GLM52_MTP", raising=False)
    assert apply_glm52_mtp_patch(model, tmp_path, logger=log) is model
    assert not hasattr(model, "_exo_glm52_mtp_state")

    monkeypatch.setenv("EXO_GLM52_MTP", "shadow")
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "glm_moe_dsa", "num_nextn_predict_layers": 1,
        "num_hidden_layers": LAYER,
    }))
    # model object lacks the expected structure -> warn + unchanged
    assert apply_glm52_mtp_patch(model, tmp_path, logger=log) is model
    assert not hasattr(model, "_exo_glm52_mtp_state")
    assert any("patch not applied" in line for line in log.lines)


# ---------------------------------------------------------------------------
# Phase 2: battle loop (greedy M1)
# ---------------------------------------------------------------------------

from exo.worker.engines.mlx.patches.glm52_mtp import (  # noqa: E402
    _MTPState as _State2,
    _PreNormCapture,
    _flush_buffered,
    _install_hooks,
)


def _tiny_model():
    from mlx_lm.models import deepseek_v32 as ds

    mx.random.seed(11)
    m = ds.Model(_tiny_args())
    mx.eval(m.parameters())
    return m


def _run_bg(model, prompt, max_tokens):
    """Real pin path: BatchGenerator.insert -> prompt batch -> graduation ->
    GenerationBatch over BatchKVCache (what prod decodes on)."""
    from mlx_lm.generate import BatchGenerator

    bg = BatchGenerator(
        model, stop_tokens=None, prefill_step_size=64,
        completion_batch_size=4, prefill_batch_size=2,
    )
    bg.insert([list(prompt)], max_tokens=[max_tokens])
    toks, last = [], None
    for _ in range(max_tokens * 3 + 32):
        out = bg.next()
        responses = out[1] if isinstance(out, tuple) else out
        for r in responses or []:
            toks.append(r.token)
            last = r
        if last is not None and last.finish_reason is not None:
            break
    return bg, toks, last


def _battle_state(model, sidecar, draft_override=None):
    m = _load(sidecar)
    st = _State2(
        mode="on", concat="eh", mtp=m,
        embed=model.model.embed_tokens, lm_head=model.lm_head,
        store={}, logger=_StubLogger(),
    )
    if draft_override is not None:
        st.draft_multi = draft_override  # type: ignore[method-assign]
    return st


def _attach(model, state):
    if not isinstance(model.model.norm, _PreNormCapture):
        model.model.norm = _PreNormCapture(model.model.norm, state.store)
    else:
        state.store = model.model.norm._store
    model._exo_glm52_mtp_state = state


def _detach(model):
    if hasattr(model, "_exo_glm52_mtp_state"):
        delattr(model, "_exo_glm52_mtp_state")


def _oracle_draft(state_holder):
    """Peek the true next token via the live generation cache, then trim."""
    def draft(pairs):
        gb = state_holder[0].cur_batch
        y = pairs[-1][1].reshape(1, 1).astype(mx.uint32)
        lg = gb.model(y, cache=gb.prompt_cache)
        t = mx.argmax(lg[:, -1, :], axis=-1).reshape(1)
        mx.eval(t)
        for cl in gb.prompt_cache:
            cl.trim(1)
        return t
    return draft


PROMPT = [5, 17, 42, 9, 88, 3, 61, 24]
N_GEN = 12


@pytest.fixture(scope="module")
def off_stream():
    model = _tiny_model()
    _install_hooks(_StubLogger())
    _detach(model)
    _, toks, last = _run_bg(model, PROMPT, N_GEN)
    assert last is not None and last.finish_reason == "length"
    assert len(toks) >= N_GEN
    return model, toks


def test_battle_byte_equal_all_accept(off_stream, sidecar):
    model, ref = off_stream
    holder = []
    state = _battle_state(model, sidecar, draft_override=_oracle_draft(holder))
    holder.append(state)
    _attach(model, state)
    try:
        from mlx_lm.generate import BatchGenerator

        bg = BatchGenerator(
            model, stop_tokens=None, prefill_step_size=64,
            completion_batch_size=4, prefill_batch_size=2,
        )
        bg.insert([list(PROMPT)], max_tokens=[N_GEN])
        toks, last = [], None
        for _ in range(N_GEN * 3 + 32):
            out = bg.next()
            responses = out[1] if isinstance(out, tuple) else out
            for r in responses or []:
                toks.append(r.token)
                last = r
            if last is not None and last.finish_reason is not None:
                break
    finally:
        _detach(model)
    assert state.accepted > 0, "oracle draft must produce accepts"
    assert toks == ref, f"battle(all-accept) diverged: {toks} vs {ref}"


def test_battle_byte_equal_all_reject(off_stream, sidecar):
    model, ref = off_stream
    state = _battle_state(
        model, sidecar, draft_override=lambda pairs: mx.array([3])
    )
    _attach(model, state)
    try:
        _, toks, last = _run_bg(model, PROMPT, N_GEN)
    finally:
        _detach(model)
    assert state.cycles > 0
    assert toks == ref, f"battle(all-reject) diverged: {toks} vs {ref}"
    assert last.finish_reason == "length"


def test_battle_real_draft_byte_equal(off_stream, sidecar):
    """Random-weight MTP head: drafts are essentially arbitrary; the stream
    must still be byte-identical (mixed accept/reject exercise)."""
    model, ref = off_stream
    state = _battle_state(model, sidecar)
    _attach(model, state)
    try:
        _, toks, _ = _run_bg(model, PROMPT, N_GEN)
    finally:
        _detach(model)
    assert state.cycles > 0
    assert toks == ref, f"battle(real draft) diverged: {toks} vs {ref}"


def test_battle_extract_trims_buffered_surplus(off_stream, sidecar):
    model, _ = off_stream
    for n in (5, 6):  # parity: finish on cycle vs on buffered emission
        holder = []
        state = _battle_state(
            model, sidecar, draft_override=_oracle_draft(holder)
        )
        holder.append(state)
        _attach(model, state)
        try:
            from mlx_lm.generate import BatchGenerator

            bg = BatchGenerator(
                model, stop_tokens=None, prefill_step_size=64,
                completion_batch_size=4, prefill_batch_size=2,
            )
            bg.insert([list(PROMPT)], max_tokens=[n])
            last = None
            for _ in range(n * 3 + 32):
                out = bg.next()
                responses = out[1] if isinstance(out, tuple) else out
                for r in responses or []:
                    last = r
                if last is not None and last.finish_reason is not None:
                    break
        finally:
            _detach(model)
        assert last is not None and last.finish_reason == "length"
        cache = last.prompt_cache
        assert cache is not None
        assert cache[0][0].offset == len(last.all_tokens), n
        assert cache[0][1].offset == cache[0][0].offset


def test_flush_buffered_rolls_back_to_reject_state(off_stream, sidecar):
    model, ref = off_stream
    holder = []
    state = _battle_state(model, sidecar, draft_override=_oracle_draft(holder))
    holder.append(state)
    _attach(model, state)
    try:
        from mlx_lm.generate import BatchGenerator

        bg = BatchGenerator(
            model, stop_tokens=None, prefill_step_size=64,
            completion_batch_size=4, prefill_batch_size=2,
        )
        bg.insert([list(PROMPT)], max_tokens=[N_GEN])
        emitted = []
        for _ in range(64):
            out = bg.next()
            responses = out[1] if isinstance(out, tuple) else out
            emitted.extend(r.token for r in responses or [])
            if state.buffer:
                break
        assert state.buffer and state.flush_pack is not None
        gb = bg._generation_batch
        t1_expected = int(state.flush_pack[0].reshape(-1)[0].item())
        _flush_buffered(state, gb)
        assert state.buffer == [] and state.flush_pack is None
        assert _batch_off(gb) == len(gb.tokens[0])
        assert int(gb._next_tokens.reshape(-1)[0].item()) == t1_expected
        # the rolled-back request must continue byte-identically to plain decode
        toks_after, last = [], None
        for _ in range(N_GEN * 3 + 32):
            out = bg.next()
            responses = out[1] if isinstance(out, tuple) else out
            for r in responses or []:
                toks_after.append(r.token)
                last = r
            if last is not None and last.finish_reason is not None:
                break
    finally:
        _detach(model)
    assert (emitted + toks_after) == ref, (emitted, toks_after, ref)


def _batch_off(gb):
    o = gb.prompt_cache[0][0].offset
    return int(o.reshape(-1)[0].item()) if isinstance(o, mx.array) else int(o)


def test_dense_prefill_predicate_by_actual_padding():
    """B==1 unpadded BatchKVCache (the MTP verify case) must keep the sparse
    path; real padding still forces the exact masked-dense fallback."""
    from mlx_lm.models.cache import BatchKVCache
    from exo.worker.engines.mlx.patches.glm52_prefill import (
        cache_requires_dense_prefill,
    )

    c = BatchKVCache([0])
    assert cache_requires_dense_prefill(c) is False
    assert cache_requires_dense_prefill(c) is False  # memo hit

    p = BatchKVCache([2, 0])
    assert cache_requires_dense_prefill(p) is True

    # identity-based memo invalidation: repadding via reassignment re-reads
    c.left_padding = mx.array([3])
    assert cache_requires_dense_prefill(c) is True
    c.left_padding = mx.array([0])
    assert cache_requires_dense_prefill(c) is False
