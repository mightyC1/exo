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


class _GreedyPolicySampler:
    """argmax sampler carrying the explicit policy the MTP loop keys on."""

    def __init__(self, greedy=True, logprobs=False, raise_on_call=False):
        self.greedy, self.logprobs, self.raise_on_call = greedy, logprobs, raise_on_call
        self.calls = 0

    def __call__(self, lp):
        self.calls += 1
        if self.raise_on_call and self.calls > 1:
            # exactly one call is allowed per request: the seed step (P1.2)
            raise AssertionError("sampler must not be called on the battle path")
        return mx.argmax(lp, axis=-1)


def _run_bg(model, prompt, max_tokens, sampler=None, processors=None):
    """Real pin path: BatchGenerator.insert -> prompt batch -> graduation ->
    GenerationBatch over BatchKVCache (what prod decodes on), with the fork's
    _patched_step as the base step."""
    from mlx_lm.generate import BatchGenerator

    bg = BatchGenerator(
        model, stop_tokens=None, prefill_step_size=64,
        completion_batch_size=4, prefill_batch_size=2,
    )
    bg.insert(
        [list(prompt)], max_tokens=[max_tokens],
        samplers=[sampler or _GreedyPolicySampler()],
        logits_processors=[list(processors or [])],
    )
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


def _battle_state(model, sidecar, draft_override=None, k=1):
    m = _load(sidecar)
    st = _State2(
        mode="on", concat="eh", mtp=m,
        embed=model.model.embed_tokens, lm_head=model.lm_head,
        store={}, logger=_StubLogger(), draft_k=k,
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
        return t, pairs[-1][0]
    return draft


def _oracle_draft2(state_holder):
    """k=2 oracle: step-1 peeks [y]; step-2 peeks [y, d1] (calls alternate)."""
    mem = {"n": 0, "y": None}

    def draft(pairs):
        gb = state_holder[0].cur_batch
        tok = pairs[-1][1].reshape(1, 1).astype(mx.uint32)
        if mem["n"] % 2 == 0:
            mem["y"] = tok
            seq = tok
        else:
            seq = mx.concatenate([mem["y"], tok], axis=1)
        mem["n"] += 1
        lg = gb.model(seq, cache=gb.prompt_cache)
        t = mx.argmax(lg[:, -1, :], axis=-1).reshape(1)
        mx.eval(t)
        for cl in gb.prompt_cache:
            cl.trim(seq.shape[1])
        return t, pairs[-1][0]
    return draft


PROMPT = [5, 17, 42, 9, 88, 3, 61, 24]
N_GEN = 12


@pytest.fixture(scope="module")
def off_stream():
    from exo.worker.engines.mlx.patches.opt_batch_gen import apply_batch_gen_patch

    apply_batch_gen_patch()  # prod base step (history semantics, top-k buffer)
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
        bg.insert([list(PROMPT)], max_tokens=[N_GEN], samplers=[_GreedyPolicySampler()])
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
        model, sidecar, draft_override=lambda pairs: (mx.array([3]), pairs[-1][0])
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
            bg.insert([list(PROMPT)], max_tokens=[n], samplers=[_GreedyPolicySampler()])
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
        bg.insert([list(PROMPT)], max_tokens=[N_GEN], samplers=[_GreedyPolicySampler()])
        emitted = []
        for _ in range(64):
            out = bg.next()
            responses = out[1] if isinstance(out, tuple) else out
            emitted.extend(r.token for r in responses or [])
            if state.buffer:
                break
        assert state.buffer and state.cycle_pack is not None
        gb = bg._generation_batch
        t1_expected = int(state.cycle_pack[0][:, 0].reshape(-1)[0].item())
        _flush_buffered(state, gb)
        assert state.buffer == [] and state.cycle_pack is None
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



# ---------------------------------------------------------------------------
# Phase 4: k=2 chained drafts
# ---------------------------------------------------------------------------


def _run_on(model, sidecar, n, draft_override, k, prompt=PROMPT,
            sampler=None, processors=None):
    holder = []
    state = _battle_state(model, sidecar, draft_override=None, k=k)
    if draft_override is not None:
        state.draft_multi = draft_override(holder)  # type: ignore[method-assign]
    holder.append(state)
    _attach(model, state)
    try:
        _, toks, last = _run_bg(model, prompt, n, sampler=sampler, processors=processors)
    finally:
        _detach(model)
    return state, toks, last


def test_k2_byte_equal_all_accept(off_stream, sidecar):
    model, ref = off_stream
    state, toks, _ = _run_on(model, sidecar, N_GEN, _oracle_draft2, k=2)
    assert state.acc_pos[1] > 0, "oracle2 must produce m=2 cycles"
    assert toks == ref, f"k2(all-accept) diverged: {toks} vs {ref}"


def test_k2_byte_equal_all_reject(off_stream, sidecar):
    model, ref = off_stream
    state, toks, _ = _run_on(
        model, sidecar, N_GEN, lambda h: (lambda pairs: (mx.array([3]), pairs[-1][0])), k=2
    )
    assert state.cycles > 0 and state.accepted == 0
    assert toks == ref, f"k2(all-reject) diverged: {toks} vs {ref}"


def test_k2_byte_equal_half(off_stream, sidecar):
    """step-1 correct, step-2 wrong -> m=1 every cycle (trim 1, backlog 1)."""
    model, ref = off_stream

    def half(holder):
        oracle = _oracle_draft2(holder)
        mem = {"n": 0}

        def draft(pairs):
            t, h = oracle(pairs)
            mem["n"] += 1
            if mem["n"] % 2 == 0:  # step-2: sabotage
                return mx.array([3]), h
            return t, h
        return draft

    state, toks, _ = _run_on(model, sidecar, N_GEN, half, k=2)
    assert state.acc_pos[0] > 0 and state.acc_pos[1] == 0
    assert toks == ref, f"k2(half) diverged: {toks} vs {ref}"


def test_k2_byte_equal_real_draft(off_stream, sidecar):
    model, ref = off_stream
    state, toks, _ = _run_on(model, sidecar, 40, None, k=2)
    _, ref40, _ = ( _detach(model), None, None) and (None, None, None)
    _detach(model)
    _, ref40, _ = _run_bg(model, PROMPT, 40)
    assert state.cycles > 0
    assert toks == ref40, f"k2(real) diverged: {toks} vs {ref40}"


def test_k2_extract_trims_buffered_surplus(off_stream, sidecar):
    """finish landing on cycle / buffered-d1 / buffered-d2 emissions."""
    model, _ = off_stream
    for n in (5, 6, 7):
        state, _, last = _run_on(model, sidecar, n, _oracle_draft2, k=2)
        assert last is not None and last.finish_reason == "length"
        cache = last.prompt_cache
        assert cache[0][0].offset == len(last.all_tokens), n
        assert cache[0][1].offset == cache[0][0].offset


def test_k2_flush_rolls_back_r2_and_r1(off_stream, sidecar):
    model, ref = off_stream
    for pops in (0, 1):  # r=2 (nothing popped) and r=1 (one draft emitted)
        holder = []
        state = _battle_state(model, sidecar, k=2)
        state.draft_multi = _oracle_draft2(holder)  # type: ignore[method-assign]
        holder.append(state)
        _attach(model, state)
        try:
            from mlx_lm.generate import BatchGenerator

            bg = BatchGenerator(
                model, stop_tokens=None, prefill_step_size=64,
                completion_batch_size=4, prefill_batch_size=2,
            )
            bg.insert([list(PROMPT)], max_tokens=[N_GEN], samplers=[_GreedyPolicySampler()])
            emitted = []
            for _ in range(64):
                out = bg.next()
                responses = out[1] if isinstance(out, tuple) else out
                emitted.extend(r.token for r in responses or [])
                if len(state.buffer) == 2 - pops and state.cycle_pack is not None \
                        and (pops == 0 or state.accepted >= 2):
                    break
            assert state.buffer, "expected buffered drafts"
            gb = bg._generation_batch
            _flush_buffered(state, gb)
            assert state.buffer == [] and state.cycle_pack is None
            assert _batch_off(gb) == len(gb.tokens[0])
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
        assert (emitted + toks_after) == ref, (pops, emitted, toks_after, ref)



def _oracle_draftk(state_holder, k):
    """k-step oracle: step j peeks [y, d1..d_{j-1}] (calls cycle mod k)."""
    mem = {"n": 0, "seq": None}

    def draft(pairs):
        gb = state_holder[0].cur_batch
        tok = pairs[-1][1].reshape(1, 1).astype(mx.uint32)
        if mem["n"] % k == 0:
            mem["seq"] = tok
        else:
            mem["seq"] = mx.concatenate([mem["seq"], tok], axis=1)
        mem["n"] += 1
        seq = mem["seq"]
        lg = gb.model(seq, cache=gb.prompt_cache)
        t = mx.argmax(lg[:, -1, :], axis=-1).reshape(1)
        mx.eval(t)
        for cl in gb.prompt_cache:
            cl.trim(seq.shape[1])
        return t, pairs[-1][0]
    return draft


def test_k3_byte_equal_all_accept(off_stream, sidecar):
    model, ref = off_stream
    state, toks, _ = _run_on(model, sidecar, N_GEN, lambda h: _oracle_draftk(h, 3), k=3)
    assert state.acc_pos[2] > 0, "oracle3 must produce m=3 cycles"
    assert toks == ref, f"k3(all-accept) diverged: {toks} vs {ref}"


def test_k3_byte_equal_all_reject_and_real(off_stream, sidecar):
    model, ref = off_stream
    state, toks, _ = _run_on(
        model, sidecar, N_GEN, lambda h: (lambda pairs: (mx.array([3]), pairs[-1][0])), k=3
    )
    assert state.accepted == 0 and toks == ref
    state, toks, _ = _run_on(model, sidecar, N_GEN, None, k=3)
    assert state.cycles > 0 and toks == ref


def test_k3_extract_parities(off_stream, sidecar):
    model, _ = off_stream
    for n in (5, 6, 7, 8):
        _, _, last = _run_on(model, sidecar, n, lambda h: _oracle_draftk(h, 3), k=3)
        assert last.finish_reason == "length"
        assert last.prompt_cache[0][0].offset == len(last.all_tokens), n


def test_norm_capture_modes():
    from exo.worker.engines.mlx.patches.glm52_mtp import _PreNormCapture

    norm = nn.RMSNorm(8, eps=1e-6)
    norm.weight = mx.arange(1, 9).astype(mx.float32)  # non-uniform: pre != post
    x = mx.random.normal((1, 3, 8))
    post_store, pre_store = {}, {}
    _PreNormCapture(norm, post_store, mode="post")(x)
    _PreNormCapture(norm, pre_store, mode="pre")(x)
    assert mx.array_equal(post_store["h"], norm(x)).item()
    assert mx.array_equal(pre_store["h"], x).item()
    assert not mx.array_equal(post_store["h"], pre_store["h"]).item()


def test_chain_recycles_post_norm(sidecar):
    """The hidden handed to the next chained step must be head_norm(block out)."""
    m = _load(sidecar)
    embed = nn.Embedding(97, HID)
    lm_head = nn.Linear(HID, 97, bias=False)
    st = _State2(mode="on", concat="eh", mtp=m, embed=embed, lm_head=lm_head,
                 store={}, logger=_StubLogger(), draft_k=2)
    st.start_request(uid=1)
    h = mx.random.normal((1, 1, HID)).astype(mx.bfloat16)
    _, h_rec = st.draft_multi([(h, mx.array([5], dtype=mx.uint32))])
    # re-run the block computation manually on a fresh state to compare
    st2 = _State2(mode="on", concat="eh", mtp=m, embed=embed, lm_head=lm_head,
                  store={}, logger=_StubLogger(), draft_k=2)
    st2.start_request(uid=2)
    e = m.enorm(embed(mx.array([[5]], dtype=mx.uint32)))
    x = m.eh_proj(mx.concatenate([e, m.hnorm(h.astype(e.dtype))], axis=-1))
    y = m.block(x, mask=None, cache=st2.mtp_cache)
    assert mx.array_equal(h_rec, m.head_norm(y[:, -1:, :])).item()



# ---------------------------------------------------------------------------
# Audit P0.1 / P0.2 / P0.4: processors parity, explicit policy, eligibility
# ---------------------------------------------------------------------------


def _ban(tid):
    def proc(_hist, logits):
        logits[..., tid] = -1e9
        return logits
    return proc


def _penalize_last(hist_dep):
    """history-dependent processor: penalize the last token in history."""
    def proc(hist, logits):
        if hist.size == 0:
            return logits
        last = hist[-1]
        return logits - mx.where(
            mx.arange(logits.shape[-1]) == last, mx.array(6.0), mx.array(0.0)
        )
    return proc


@pytest.mark.parametrize("k", [1, 2, 3])
def test_processors_parity_eos_ban_like(off_stream, sidecar, k):
    model, ref_plain = off_stream
    banned = ref_plain[1]  # a token the plain run emits early
    _detach(model)
    _, ref, _ = _run_bg(model, PROMPT, N_GEN, processors=[_ban(banned)])
    assert ref != ref_plain and banned not in ref
    for name, reg in (("oracle", lambda h: _oracle_draftk(h, k)), ("real", None)):
        state, toks, _ = _run_on(model, sidecar, N_GEN, reg, k=k, processors=[_ban(banned)])
        assert state.cycles > 0
        assert toks == ref, f"k={k} {name}: {toks} vs {ref}"
        assert banned not in toks


@pytest.mark.parametrize("k", [1, 2])
def test_processors_parity_history_dependent(off_stream, sidecar, k):
    model, _ = off_stream
    _detach(model)
    _, ref, _ = _run_bg(model, PROMPT, N_GEN, processors=[_penalize_last(True)])
    for name, reg in (("oracle", lambda h: _oracle_draftk(h, k)), ("real", None)):
        state, toks, _ = _run_on(
            model, sidecar, N_GEN, reg, k=k, processors=[_penalize_last(True)]
        )
        assert state.cycles > 0
        assert toks == ref, f"k={k} {name}: {toks} vs {ref}"


def test_policy_unknown_sampler_fails_closed(off_stream, sidecar):
    model, ref = off_stream
    plain = lambda lp: mx.argmax(lp, axis=-1)  # no policy attributes
    state, toks, _ = _run_on(model, sidecar, N_GEN, None, k=1, sampler=plain)
    assert state.last_battle is False and state.cycles == 0
    assert toks == ref


def test_policy_temperature_not_greedy(off_stream, sidecar):
    model, ref = off_stream
    samp = _GreedyPolicySampler(greedy=False)  # argmax fn, but declared sampling
    state, toks, _ = _run_on(model, sidecar, N_GEN, None, k=1, sampler=samp)
    assert state.last_battle is False and state.cycles == 0
    assert toks == ref


def test_policy_logprobs_excluded(off_stream, sidecar):
    model, ref = off_stream
    samp = _GreedyPolicySampler(greedy=True, logprobs=True)
    state, toks, _ = _run_on(model, sidecar, N_GEN, None, k=1, sampler=samp)
    assert state.last_battle is False and state.cycles == 0 and toks == ref


def test_battle_never_calls_sampler(off_stream, sidecar):
    model, ref = off_stream
    samp = _GreedyPolicySampler(greedy=True, raise_on_call=True)
    state, toks, _ = _run_on(model, sidecar, N_GEN, None, k=1, sampler=samp)
    assert state.last_battle is True and state.cycles > 0
    assert samp.calls == 1 and toks == ref  # seed step only


# ---------------------------------------------------------------------------
# Audit P0.3: exactly-once real step, transactional rollback, VALIDATE not masked
# ---------------------------------------------------------------------------


def _all_offsets(gb):
    return [[_cur_off(c) for c in cl.caches] for cl in gb.prompt_cache]


def _cur_off(c):
    idx = getattr(c, "_idx", None)
    return int(idx) if idx is not None else int(c.offset)


@pytest.mark.parametrize("fail_after_mutation", [False, True])
def test_fault_in_draft_rolls_back_and_continues(off_stream, sidecar, fail_after_mutation):
    """An exception inside the cycle (before or after the MTP cache mutated)
    must roll every cache back to the cycle entry and continue the request
    with plain decode — byte-identical to MTP=off, real step run once."""
    model, ref = off_stream
    holder = []
    state = _battle_state(model, sidecar, k=1)
    real = state.draft_multi
    hits = {"n": 0}

    def faulty(pairs):
        hits["n"] += 1
        if hits["n"] == 3:
            if fail_after_mutation:
                real(pairs)            # MTP cache grows, then we blow up
            raise RuntimeError("injected draft failure")
        return real(pairs)

    state.draft_multi = faulty  # type: ignore[method-assign]
    holder.append(state)
    _attach(model, state)
    try:
        from mlx_lm.generate import BatchGenerator

        bg = BatchGenerator(model, stop_tokens=None, prefill_step_size=64,
                            completion_batch_size=4, prefill_batch_size=2)
        bg.insert([list(PROMPT)], max_tokens=[N_GEN], samplers=[_GreedyPolicySampler()])
        toks, last = [], None
        for _ in range(N_GEN * 3 + 32):
            out = bg.next()
            rs = out[1] if isinstance(out, tuple) else out
            for r in rs or []:
                toks.append(r.token); last = r
            if last is not None and last.finish_reason is not None:
                break
            gb = bg._generation_batch
            if len(gb):
                # invariant after every call: main cache == committed tokens (+ buffered)
                assert _all_offsets(gb)[0][0] == len(gb.tokens[0]) + len(state.buffer)
    finally:
        _detach(model)
    assert hits["n"] >= 3 and state.last_battle is True
    assert toks == ref, (toks, ref)
    assert last.prompt_cache[0][0].offset == len(last.all_tokens)


def test_validate_error_is_not_swallowed(off_stream, sidecar, monkeypatch):
    from exo.worker.engines.mlx.patches import glm52_mtp as gm

    model, _ = off_stream
    state = _battle_state(model, sidecar, k=1)
    state.validate = True

    def boom(state_, batch, m):
        raise gm._MTPValidateError("injected validate failure")

    monkeypatch.setattr(gm, "_validate_cycle", boom)
    _attach(model, state)
    try:
        with pytest.raises(gm._MTPValidateError):
            _run_bg(model, PROMPT, N_GEN)
    finally:
        _detach(model)


def test_norm_wrapper_keeps_parameter_tree(sidecar):
    from mlx.utils import tree_flatten
    from exo.worker.engines.mlx.patches import glm52_mtp as gm

    model = _tiny_model()
    before = sorted(k for k, _ in tree_flatten(model.parameters()))
    x = mx.random.normal((1, 3, HID)).astype(mx.bfloat16)
    ref = model.model.norm(x)
    model.model.norm = gm._PreNormCapture(model.model.norm, {}, mode="post")
    after = sorted(k for k, _ in tree_flatten(model.parameters()))
    assert before == after
    assert mx.array_equal(model.model.norm(x), ref).item()
    mx.eval(model.parameters())
