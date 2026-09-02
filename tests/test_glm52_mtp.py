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
    """Sampler carrying the explicit policy the MTP loop keys on. greedy=True
    -> argmax; greedy=False -> the pin's make_sampler chain (temperature,
    top_p, min_p, top_k), exactly what exo's ExoSampler wraps in prod."""

    def __init__(self, greedy=True, logprobs=False, raise_on_call=False,
                 temperature=0.0, top_p=1.0, min_p=0.0, top_k=0):
        from mlx_lm.sample_utils import make_sampler

        self.greedy, self.logprobs, self.raise_on_call = greedy, logprobs, raise_on_call
        self.temperature = 0.0 if greedy else float(temperature)
        self.top_p, self.min_p, self.top_k = float(top_p), float(min_p), int(top_k)
        self.fn = (None if greedy else
                   make_sampler(temp=self.temperature, top_p=top_p, min_p=min_p, top_k=top_k))
        self.calls = 0

    def __call__(self, lp):
        self.calls += 1
        if self.raise_on_call and self.calls > 1:
            # exactly one call is allowed per request: the seed step (P1.2)
            raise AssertionError("sampler must not be called on the battle path")
        return mx.argmax(lp, axis=-1) if self.greedy else self.fn(lp)


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
    assert mx.array_equal(h_rec, m.head_norm(y[:, -1:, :])).item()  # default recycle=post



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


def test_policy_temperature_routes_to_rs(off_stream, sidecar):
    model, _ = off_stream
    samp = _GreedyPolicySampler(greedy=False, temperature=0.7, min_p=0.05)
    mx.random.seed(3)
    state, toks, last = _run_on(model, sidecar, N_GEN, None, k=1, sampler=samp)
    assert state.last_battle is True and state.cycles > 0
    assert last.finish_reason == "length"


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

    def boom(state_, batch, m, base=None):
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


# ---------------------------------------------------------------------------
# Audit P0.6: side-car content validation (fail-closed)
# ---------------------------------------------------------------------------


def _model_dir_for(tmp_path, sidecar: Path, mutate=None) -> Path:
    import shutil

    d = tmp_path / "model"
    d.mkdir()
    shutil.copy(sidecar, d / "mtp.safetensors")
    shutil.copy(sidecar.with_name("mtp.manifest.json"), d / "mtp.manifest.json")
    (d / "config.json").write_text(json.dumps({
        "model_type": "glm_moe_dsa", "num_nextn_predict_layers": 1,
        "num_hidden_layers": LAYER,
        "quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
    }))
    if mutate:
        mutate(d)
    return d


def _apply_on_tiny(model, d, monkeypatch):
    from exo.worker.engines.mlx.patches import glm52_mtp as gm

    monkeypatch.setenv("EXO_GLM52_MTP", "on")
    log = _StubLogger()
    _detach(model)
    out = gm.apply_glm52_mtp_patch(model, d, logger=log)
    applied = hasattr(model, "_exo_glm52_mtp_state")
    _detach(model)
    return applied, log


def test_sidecar_valid_applies(sidecar, tmp_path, monkeypatch):
    model = _tiny_model()
    applied, log = _apply_on_tiny(model, _model_dir_for(tmp_path, sidecar), monkeypatch)
    assert applied, log.lines
    assert any("sha256 verified" in line for line in log.lines)


def test_sidecar_corrupted_bytes_rejected(sidecar, tmp_path, monkeypatch):
    def corrupt(d):
        pth = d / "mtp.safetensors"
        b = bytearray(pth.read_bytes()); b[-1] ^= 0xFF; pth.write_bytes(bytes(b))
    model = _tiny_model()
    applied, log = _apply_on_tiny(model, _model_dir_for(tmp_path, sidecar, corrupt), monkeypatch)
    assert not applied and any("SHA-256" in line for line in log.lines)


def test_sidecar_missing_manifest_rejected(sidecar, tmp_path, monkeypatch):
    model = _tiny_model()
    applied, log = _apply_on_tiny(
        model, _model_dir_for(tmp_path, sidecar, lambda d: (d / "mtp.manifest.json").unlink()), monkeypatch
    )
    assert not applied and any("manifest" in line for line in log.lines)


def test_sidecar_policy_mismatch_rejected(sidecar, tmp_path, monkeypatch):
    def bad_policy(d):
        m = json.loads((d / "mtp.manifest.json").read_text())
        m["policy"]["quant"]["bits"] = 4
        (d / "mtp.manifest.json").write_text(json.dumps(m))
    model = _tiny_model()
    applied, log = _apply_on_tiny(model, _model_dir_for(tmp_path, sidecar, bad_policy), monkeypatch)
    assert not applied and any("policy" in line for line in log.lines)


def test_mtp_indexer_rope_follows_config(sidecar, tmp_path, monkeypatch):
    """Audit P1.1: the MTP block's indexer gets the same half-split RoPE
    decision as the main full-indexer layers."""
    from exo.worker.engines.mlx.patches import glm52_mtp as gm

    monkeypatch.setenv("EXO_GLM52_MTP", "on")
    for interleave, expect_fixed in ((True, False), (False, True)):
        def mut(d, interleave=interleave):
            c = json.loads((d / "config.json").read_text())
            c.update({"indexer_rope_interleave": interleave, "rope_theta": 10000.0,
                      "qk_rope_head_dim": ROPE})
            (d / "config.json").write_text(json.dumps(c))
        d = _model_dir_for(tmp_path / f"m{int(interleave)}", sidecar, mut) if False else None
        sub = tmp_path / f"m{int(interleave)}"; sub.mkdir()
        d = _model_dir_for(sub, sidecar, mut)
        model = _tiny_model()
        log = _StubLogger()
        gm.apply_glm52_mtp_patch(model, d, logger=log)
        st = getattr(model, "_exo_glm52_mtp_state", None)
        assert st is not None, log.lines
        rope = st.mtp.block.self_attn.indexer.rope
        assert bool(getattr(rope, "traditional", True)) is (not expect_fixed)
        assert any(f"mtp_indexer_rope_fixed={int(expect_fixed)}" in line for line in log.lines)
        _detach(model)


def test_extractor_scale_key_and_atomic_publish(tmp_path):
    """Audit P1.7: scale-key derivation must only touch the suffix; the
    side-car and manifest are published atomically (no temp leftovers)."""
    src = tmp_path / "src"; dest = tmp_path / "dest"; src.mkdir(); dest.mkdir()
    _make_synth_src(src)
    assert _extractor().extract(src, dest) == 0
    names = sorted(p.name for p in dest.iterdir())
    assert names == ["mtp.manifest.json", "mtp.safetensors"], names
    man = json.loads((dest / "mtp.manifest.json").read_text())
    assert man["layer"] == LAYER and man["bytes"] == (dest / "mtp.safetensors").stat().st_size
    # scale-key derivation on a name with 'weight' twice must stay suffix-only
    k = "model.layers.78.self_attn.indexer.weights_proj.weight"
    assert k[: -len(".weight")] + ".weight_scale_inv" == \
        "model.layers.78.self_attn.indexer.weights_proj.weight_scale_inv"


def test_corpus_sizer_exact_and_empty():
    """Audit P1.8: exact token sizing, arithmetic tiling, empty corpus rejected."""
    import importlib.util as ilu
    # exo_tools is the cluster harness (not in the runtime venv): stub it.
    for name in ("exo_tools", "exo_tools.client", "exo_tools.harness"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    for attr in ("ExoClient", "ExoHttpError"):
        setattr(sys.modules["exo_tools.client"], attr, type(attr, (), {}))
    h = sys.modules["exo_tools.harness"]
    src_text = (REPO_ROOT / "bench" / "exo_bench.py").read_text()
    block = src_text.split("from exo_tools.harness import (", 1)[1].split(")", 1)[0]
    for attr in [x.strip().rstrip(",") for x in block.split("\n") if x.strip()]:
        if attr and not hasattr(h, attr):
            setattr(h, attr, lambda *a, **k: None)
    spec = ilu.spec_from_file_location("exo_bench", REPO_ROOT / "bench" / "exo_bench.py")
    mod = ilu.module_from_spec(spec); spec.loader.exec_module(mod)

    class Tok:  # whitespace tokenizer stand-in with a 2-token template
        def apply_chat_template(self, msgs, tokenize=True, **kw):
            text = " ".join(m["content"] for m in msgs)
            ids = ["<s>"] + text.split() + ["</s>"]
            return ids if tokenize else " ".join(ids)
        def encode(self, text, **kw):
            return text.split()
    sizer = mod.PromptSizer(Tok(), corpus="alpha beta gamma delta")
    content, tok = sizer.build(37)
    assert tok == 37
    with pytest.raises(ValueError):
        mod.PromptSizer(Tok(), corpus="   \n").build(20)



# ---------------------------------------------------------------------------
# M2: speculative sampling for temperature > 0
# ---------------------------------------------------------------------------


def test_rs_kernel_matches_target_distribution():
    """Greedy draft + accept w.p. p(d) + residual on reject must reproduce the
    exact target distribution (Leviathan et al., q = one-hot)."""
    from exo.worker.engines.mlx.patches.glm52_mtp import _rs_residual_logits, _target_logits

    mx.random.seed(0)
    V, N = 97, 20000
    row = mx.log(mx.softmax(mx.random.normal((1, V)) * 2.5, axis=-1))
    pol = {"temperature": 0.7, "top_p": 1.0, "min_p": 0.05, "top_k": 0}
    z = _target_logits(row, pol)
    p = mx.softmax(z, axis=-1).reshape(-1)
    order = mx.argsort(-p)                                   # draft = 2nd most likely (p(d) > 0)
    d = order[1:2].astype(mx.int32)
    pd = p[d.item()]
    us = mx.random.uniform(shape=(N,))
    accept = us < pd
    res = mx.random.categorical(mx.broadcast_to(_rs_residual_logits(z, d), (N, V)))
    toks = mx.where(accept, d.reshape(1).astype(res.dtype), res)
    counts = mx.zeros((V,)).at[toks].add(mx.ones((N,)))
    emp = counts / N
    tv = 0.5 * float(mx.sum(mx.abs(emp - p)).item())
    assert tv < 0.02, tv
    assert 0.0 < float(mx.mean(accept).item()) < 1.0


def _run_rs(model, sidecar, n, k, seed, draft_override=None, temperature=0.5, min_p=0.0):
    samp = _GreedyPolicySampler(greedy=False, temperature=temperature, min_p=min_p)
    mx.random.seed(seed)
    return _run_on(model, sidecar, n, draft_override, k=k, sampler=samp)


@pytest.mark.parametrize("k", [1, 2])
def test_rs_pipeline_deterministic_and_consistent(off_stream, sidecar, k):
    model, _ = off_stream
    st1, toks1, last1 = _run_rs(model, sidecar, 40, k, seed=11,
                                draft_override=lambda h: _oracle_draftk(h, k))
    st2, toks2, last2 = _run_rs(model, sidecar, 40, k, seed=11,
                                draft_override=lambda h: _oracle_draftk(h, k))
    assert st1.cycles > 0 and st1.accepted > 0, (st1.cycles, st1.accepted)
    assert toks1 == toks2, "same seed -> identical stream (replicated RNG)"
    assert last1.finish_reason == "length"
    assert last1.prompt_cache[0][0].offset == len(last1.all_tokens)
    assert last1.prompt_cache[0][1].offset == last1.prompt_cache[0][0].offset


def test_rs_second_token_matches_exact_target(off_stream, sidecar):
    """Pipeline-level distribution gate: the second generated token (first
    produced by the RS cycle) must follow the exact target distribution
    conditioned on the first (seed-step) token."""
    from exo.worker.engines.mlx.patches.glm52_mtp import _target_logits

    model, _ = off_stream
    # low temperature keeps both positions peaked so the empirical estimate
    # converges with a few hundred samples (the tiny random model is flat).
    pol = {"temperature": 0.15, "top_p": 1.0, "min_p": 0.0, "top_k": 0}
    pairs = []
    for seed in range(300):
        _, toks, _ = _run_rs(model, sidecar, 2, 1, seed=seed, temperature=0.15)
        pairs.append((toks[0], toks[1]))
    from collections import Counter
    first = Counter(t1 for t1, _ in pairs).most_common(1)[0][0]
    second = [t2 for t1, t2 in pairs if t1 == first]
    assert len(second) >= 40, len(second)
    # exact target for the second token given the first
    _detach(model)
    cache = model.make_cache()
    lg = model(mx.array([PROMPT + [first]], dtype=mx.uint32), cache=cache)
    row = lg[:, -1, :] - mx.logsumexp(lg[:, -1, :], axis=-1, keepdims=True)
    p = mx.softmax(_target_logits(row, pol), axis=-1).reshape(-1)
    counts = Counter(second)
    emp = mx.zeros((p.shape[0],))
    for t, c in counts.items():
        emp = emp.at[t].add(c / len(second))
    tv = 0.5 * float(mx.sum(mx.abs(emp - p)).item())
    assert tv < 0.2, (tv, len(second))


# ---------------------------------------------------------------------------
# Audit #2: merge-flush keeps the accepted sample, exact trims, rank-safe
# VALIDATE ordering, terminal finalize
# ---------------------------------------------------------------------------


def test_rs_flush_preserves_accepted_draft_and_consumes_no_rng(off_stream, sidecar, monkeypatch):
    model, _ = off_stream
    holder = []
    state = _battle_state(model, sidecar, k=2)
    state.draft_multi = _oracle_draftk(holder, 2)  # type: ignore[method-assign]
    holder.append(state)
    samp = _GreedyPolicySampler(greedy=False, temperature=0.5)
    _attach(model, state)
    try:
        from mlx_lm.generate import BatchGenerator

        mx.random.seed(21)
        bg = BatchGenerator(model, stop_tokens=None, prefill_step_size=64,
                            completion_batch_size=4, prefill_batch_size=2)
        bg.insert([list(PROMPT)], max_tokens=[80], samplers=[samp], logits_processors=[[]])
        for _ in range(300):
            bg.next()
            if len(state.buffer) == 2:
                break
        assert len(state.buffer) == 2 and state.cycle_pack is not None, "no m=2 cycle observed"
        gb = bg._generation_batch
        t_all, lp2, hv, drafts, m = state.cycle_pack
        expected = int(drafts[0].reshape(-1)[0].item())

        def boom(*a, **k):
            raise AssertionError("merge flush must not draw from the RNG")

        monkeypatch.setattr(mx.random, "categorical", boom)
        monkeypatch.setattr(mx.random, "uniform", boom)
        _flush_buffered(state, gb)
        monkeypatch.undo()
        assert int(gb._next_tokens.reshape(-1)[0].item()) == expected
        assert state.buffer == [] and _batch_off(gb) == len(gb.tokens[0])
        # the request keeps decoding normally afterwards
        last = None
        for _ in range(N_GEN * 3 + 32):
            out = bg.next()
            rs = out[1] if isinstance(out, tuple) else out
            for r in rs or []:
                last = r
            if last is not None and last.finish_reason is not None:
                break
        assert last is not None and last.finish_reason == "length"
        assert last.prompt_cache[0][0].offset == len(last.all_tokens)
    finally:
        _detach(model)


def test_trim_to_exact_is_strict():
    from exo.worker.engines.mlx.patches.glm52_mtp import _trim_to_exact

    class Fake:
        def __init__(self, idx, broken=False):
            self._idx, self.broken = idx, broken
        def trim(self, n):
            if not self.broken:
                self._idx -= n
            return n
    c = Fake(10); _trim_to_exact(c, 7); assert c._idx == 7
    with pytest.raises(RuntimeError):
        _trim_to_exact(Fake(5), 7)              # behind target
    with pytest.raises(RuntimeError):
        _trim_to_exact(Fake(10, broken=True), 7)  # trim did nothing


def test_validate_cycle_reports_local_error_after_consensus(off_stream, sidecar):
    from exo.worker.engines.mlx.patches import glm52_mtp as gm

    model, _ = off_stream
    state = _battle_state(model, sidecar, k=1)
    state.group = None  # single process: consensus degenerates to local
    from mlx_lm.generate import BatchGenerator

    _attach(model, state)
    try:
        bg = BatchGenerator(model, stop_tokens=None, prefill_step_size=64,
                            completion_batch_size=4, prefill_batch_size=2)
        bg.insert([list(PROMPT)], max_tokens=[N_GEN], samplers=[_GreedyPolicySampler()])
        for _ in range(3):
            bg.next()
        gb = bg._generation_batch
        gb.prompt_cache[37].caches[1].trim(1)  # skew one slot of layer 37
        with pytest.raises(gm._MTPValidateError) as ei:
            gm._validate_cycle(state, gb, 0)
        assert "local check failed" in str(ei.value)
    finally:
        _detach(model)


def test_finalize_request_resets_state_and_counters(off_stream, sidecar):
    from exo.worker.engines.mlx.patches import glm52_mtp as gm

    model, _ = off_stream
    state = _battle_state(model, sidecar, k=1)
    state.start_request(uid=5)
    state.cycles, state.accepted, state.out_tokens = 4, 3, 7
    state.buffer = [(mx.array([1]), mx.zeros((1, 97)))]
    _attach(model, state)
    try:
        gm.finalize_glm52_mtp_request(model, 4, reason="other")   # other uid: untouched
        assert state.uid == 5
        gm.finalize_glm52_mtp_request(model, 5, reason="text-stop")
        assert state.uid is None and state.buffer == []
    finally:
        _detach(model)


def test_lp_row_hard_fails_on_unexpected_structure():
    from exo.worker.engines.mlx.patches.glm52_mtp import _lp_row

    assert _lp_row(mx.zeros((1, 4))).shape == (4,)
    assert _lp_row([]).shape == (1,)
    with pytest.raises(RuntimeError):
        _lp_row({"bad": 1})


# ---------------------------------------------------------------------------
# 0029: history buffer + vectorized RS
# ---------------------------------------------------------------------------


def test_history_buffer_tracks_tokens_list(off_stream, sidecar):
    from exo.worker.engines.mlx.patches.glm52_mtp import _history

    model, _ = off_stream
    state = _battle_state(model, sidecar, k=1)
    state.start_request(uid=1)

    class B:
        tokens = [[5, 17, 42]]
    b = B()
    assert _history(state, b).tolist() == [5, 17, 42]
    b.tokens[0].extend([9, 88])
    assert _history(state, b).tolist() == [5, 17, 42, 9, 88]         # O(delta) append
    del b.tokens[0][3:]
    assert _history(state, b).tolist() == [5, 17, 42]                # shrink -> rebuild
    b.tokens[0].extend(list(range(9000)))                            # growth beyond capacity
    assert _history(state, b).tolist() == b.tokens[0]


def test_vectorized_target_and_accept_match_reference():
    from exo.worker.engines.mlx.patches.glm52_mtp import (
        _rs_accepts, _rs_accepts_vec, _target_logits,
    )

    mx.random.seed(2)
    k, V = 3, 97
    rows = mx.log(mx.softmax(mx.random.normal((k + 1, V)) * 2.0, axis=-1))
    pol = {"temperature": 0.7, "top_p": 0.9, "min_p": 0.05, "top_k": 40}
    z_rows = _target_logits(rows, pol)
    per_row = mx.concatenate([_target_logits(rows[j:j + 1], pol) for j in range(k + 1)], axis=0)
    assert mx.array_equal(z_rows, per_row).item()
    drafts = [mx.array([int(mx.argsort(-rows[j])[i % 3].item())]) for i, j in enumerate(range(k))]
    for trial in range(20):
        u = mx.random.uniform(shape=(k,))
        ref = _rs_accepts([z_rows[j:j + 1] for j in range(k + 1)], drafts, [u[j:j + 1] for j in range(k)])
        vec = _rs_accepts_vec(z_rows, drafts, u)
        assert [bool(a.item()) for a in ref] == [bool(a.item()) for a in vec], trial


# ---------------------------------------------------------------------------
# D2: prompt ingestion into the MTP cache
# ---------------------------------------------------------------------------


def _capture_prompt_hidden(model, prompt_prefix):
    """Forward the prompt prefix through the model so the capture store holds
    its hiddens (what exo's prefill does chunk by chunk)."""
    cache = model.make_cache()
    model(mx.array([prompt_prefix], dtype=mx.uint32), cache=cache)


def test_prompt_ingest_chunked_equals_single(off_stream, sidecar):
    from exo.worker.engines.mlx.patches import glm52_mtp as gm

    model, _ = off_stream
    prompt = [5, 17, 42, 9, 88, 3, 61, 24, 12, 7, 33]
    prefix = prompt[:-1]
    # reference: one-shot ingest of all (h_i, t_{i+1}) pairs
    st_a = _battle_state(model, sidecar, k=1); _attach(model, st_a)
    try:
        _capture_prompt_hidden(model, prefix)
        h = st_a.store.pop("h")
        st_a.start_request(uid=1)
        st_a.ingest(st_a.mtp_cache, h[:, :-1, :], mx.array([prefix[1:]], dtype=mx.uint32))
        ref_off = st_a.mtp_cache[0].offset
        ref_k = st_a.mtp_cache[0].keys[..., :ref_off, :]
        mx.eval(ref_k)
        # chunked path with carry: chunks of 4, 3, 3 tokens
        st_b = _battle_state(model, sidecar, k=1); _attach(model, st_b)
        gm.mtp_prefill_begin(model, 0, len(prefix))
        for a, b in ((0, 4), (4, 7), (7, 10)):
            _capture_prompt_hidden(model, prefix)            # store gets full-prefix h
            full = st_b.store.pop("h")
            st_b.store["h"] = full[:, a:b, :]                # emulate the chunk's capture
            gm.mtp_prefill_chunk(model, prefix[a:b])
        pend = st_b.pending
        assert pend is not None and pend["ingested"] == len(prefix) - 1 and pend["toks"] == prefix
        got_off = pend["cache"][0].offset
        got_k = pend["cache"][0].keys[..., :got_off, :]
        assert got_off == ref_off
        assert mx.allclose(got_k, ref_k, atol=1e-6, rtol=1e-5).item()
    finally:
        _detach(model)


def test_prompt_ingest_end_to_end_byte_equal(off_stream, sidecar):
    from exo.worker.engines.mlx.patches import glm52_mtp as gm
    from mlx_lm.generate import BatchGenerator

    model, ref = off_stream
    prefix = PROMPT[:-1]
    state = _battle_state(model, sidecar, k=1)
    _attach(model, state)
    try:
        gm.mtp_prefill_begin(model, 0, len(prefix))
        _capture_prompt_hidden(model, prefix)
        gm.mtp_prefill_chunk(model, prefix)
        assert state.pending is not None
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
    finally:
        _detach(model)
    assert state.pending is None
    assert any("prompt ingested" in line for line in state.logger.lines), state.logger.lines[-3:]
    assert state.cycles > 0
    assert toks == ref, (toks, ref)        # mechanics untouched by the warm head


def test_prompt_ingest_skipped_on_prefix_hit_or_mismatch(off_stream, sidecar):
    from exo.worker.engines.mlx.patches import glm52_mtp as gm

    model, _ = off_stream
    state = _battle_state(model, sidecar, k=1); _attach(model, state)
    try:
        gm.mtp_prefill_begin(model, 13, 100)      # prefix-cache hit -> not armed
        assert state.pending is None
        gm.mtp_prefill_begin(model, 0, 8)
        _capture_prompt_hidden(model, PROMPT[:-1])
        gm.mtp_prefill_chunk(model, PROMPT[:-1])
        assert state.pending is not None

        class B:
            tokens = [[1, 2, 3, 4, 5, 6, 7, 8]]   # a different prompt
        state.start_request(uid=9)
        gm._adopt_prefill(state, B())
        assert state.pending is None and state.mtp_cache[0].offset == 0
        assert any("prompt mismatch" in line for line in state.logger.lines)
    finally:
        _detach(model)


def test_prompt_ingest_adopts_with_prod_two_token_insert(off_stream, sidecar):
    """exo inserts only the last two prompt tokens after its own prefill:
    tokens[0] is a one-token tail at the seed step; adoption must still
    match structurally (suffix + main cache position + pending token)."""
    from exo.worker.engines.mlx.patches import glm52_mtp as gm
    from mlx_lm.models.cache import CacheList, KVCache

    model, _ = off_stream
    prompt = list(PROMPT)
    prefix = prompt[:-1]
    state = _battle_state(model, sidecar, k=1); _attach(model, state)
    try:
        gm.mtp_prefill_begin(model, 0, len(prefix))
        _capture_prompt_hidden(model, prefix)
        gm.mtp_prefill_chunk(model, prefix)
        n = state.pending["n"]
        main = CacheList(KVCache(), KVCache())
        main.caches[0].offset = n
        main.caches[1].offset = n

        class B:
            tokens = [[prompt[-2]]]                            # exo's two-token insert, one committed
            _next_tokens = mx.array([prompt[-1]], dtype=mx.uint32)
            prompt_cache = [main]
        state.start_request(uid=3)
        gm._adopt_prefill(state, B())
        assert any("prompt ingested" in line for line in state.logger.lines), state.logger.lines[-2:]
        assert state.mtp_cache[0].offset == len(prompt) - 1
    finally:
        _detach(model)


def test_prompt_ingest_window_keeps_last_pairs(off_stream, sidecar):
    from exo.worker.engines.mlx.patches import glm52_mtp as gm

    model, _ = off_stream
    mx.random.seed(9)
    prompt = [int(x) for x in mx.random.randint(0, 97, (30,)).tolist()]
    prefix = prompt[:-1]                          # 29 tokens -> pairs 0..27, carry h_28
    state = _battle_state(model, sidecar, k=1); _attach(model, state)
    try:
        # reference: full ingest, then compare the tail of its cache with a windowed ingest
        state.prefill_window = 0
        gm.mtp_prefill_begin(model, 0, len(prefix))
        _capture_prompt_hidden(model, prefix)
        h = state.store.pop("h"); state.store["h"] = h
        gm.mtp_prefill_chunk(model, prefix)
        assert state.pending["ingested"] == 28
        st2 = _battle_state(model, sidecar, k=1); _attach(model, st2)
        st2.prefill_window = 5
        gm.mtp_prefill_begin(model, 0, len(prefix))
        for a, b in ((0, 10), (10, 20), (20, 29)):     # chunked, window starts inside chunk 3
            st2.store["h"] = h[:, a:b, :]
            gm.mtp_prefill_chunk(model, prefix[a:b])
        assert st2.pending["ingested"] == 5 and st2.pending["start"] == 29 - 1 - 5
        assert st2.pending["cache"][0].offset == 5
        # the windowed pairs are exactly pairs 23..27: verify via a fresh sequential ingest
        st3 = _battle_state(model, sidecar, k=1); st3.start_request(uid=7)
        st3.ingest(st3.mtp_cache, h[:, 23:28, :], mx.array([prefix[24:29]], dtype=mx.uint32))
        assert mx.allclose(st2.pending["cache"][0].keys[..., :5, :], st3.mtp_cache[0].keys[..., :5, :], atol=1e-5).item()
    finally:
        _detach(model)


def test_prompt_context_dropped_after_n_cycles(off_stream, sidecar):
    from exo.worker.engines.mlx.patches import glm52_mtp as gm
    from mlx_lm.generate import BatchGenerator

    model, ref = off_stream
    prefix = PROMPT[:-1]
    state = _battle_state(model, sidecar, k=1)
    state.prefill_cycles = 4
    _attach(model, state)
    try:
        gm.mtp_prefill_begin(model, 0, len(prefix))
        _capture_prompt_hidden(model, prefix)
        gm.mtp_prefill_chunk(model, prefix)
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
    finally:
        _detach(model)
    assert any("prompt ingested" in l for l in state.logger.lines)
    assert any("prompt context dropped after 4 cycles" in l for l in state.logger.lines), state.logger.lines
    assert not state.prompt_ctx
    line = next(l for l in state.logger.lines if "prompt context dropped" in l)
    assert "re-ingested 4 generated pairs" in line, line   # one committed pair per cycle (k=1)
    assert toks == ref                                    # mechanics untouched
