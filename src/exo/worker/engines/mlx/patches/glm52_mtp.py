"""GLM-5.2/5.3 MTP speculative decoding — phases 1+2: shadow and greedy battle.

Env (default **off**, zero prod impact until enabled):

  EXO_GLM52_MTP=off|shadow|on
      shadow: draft in parallel, compare, never alter output ([MTP_SHADOW]).
      on:     M1 battle loop for *greedy* requests (verify L=2, accept m∈{0,1}
              + bonus, trim rollback). Non-greedy (temp>0) requests fall back
              to shadow until M2 lands — behaviour, not an error.
  EXO_GLM52_MTP_WEIGHTS=auto|/path/to/mtp.safetensors
  EXO_GLM52_MTP_CONCAT=eh|he      eh = [enorm(embed), hnorm(hidden)] — vendor
                                  order per vLLM deepseek_mtp (default).
  EXO_GLM52_MTP_VALIDATE=0|1      canary: per-cycle cross-rank all_sum asserts
                                  on (m, offsets, buffer) + slot0/slot1 checks.
  EXO_GLM52_MTP_DRAFT_K=1         parsed and clamped; >1 is phase 4.

Battle-step mapping onto the pinned ``GenerationBatch._step`` pipelining
(entry: ``_next_tokens`` = pending token y; exit: emit y, set new pending):

  cycle:   d = MTP(backlog + (h_last, y));  verify = model([y, d], cache)
           t1 = argmax(verify[0]); accept ⇔ d == t1
           accept: emit y, buffer d (emitted next call, no forward),
                   pending = argmax(verify[1]), backlog = [(h_y, d)]
           reject: trim(1) on every CacheList (both slots), emit y,
                   pending = t1
  The MTP cache lags the main cache by the backlog and ingests it inside the
  next draft (one L=len forward — bandwidth-bound, same weight read as L=1).

Consistency hooks (installed with the step wrapper):
  * ``GenerationBatch.extract_cache`` — on finish with a buffered token the
    cache holds one uncommitted position; trim it so KVPrefixCache never
    stores a cache longer than ``all_tokens`` (recon D4).
  * ``GenerationBatch.extend`` — continuous-batching merge while a token is
    buffered would strand it; pre-merge we roll the request back to the
    reject-equivalent state (trim, pending = t1) while B is still 1.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from exo.worker.runner.bootstrap import logger as default_logger

_ENV_MODE = "EXO_GLM52_MTP"
_ENV_WEIGHTS = "EXO_GLM52_MTP_WEIGHTS"
_ENV_CONCAT = "EXO_GLM52_MTP_CONCAT"
_ENV_VALIDATE = "EXO_GLM52_MTP_VALIDATE"
_ENV_DRAFT_K = "EXO_GLM52_MTP_DRAFT_K"
_ENV_TRACE = "EXO_GLM52_MTP_TRACE"
_ENV_PROF = "EXO_GLM52_MTP_PROF"
_ENV_HIDDEN = "EXO_GLM52_MTP_HIDDEN"

_LOG_EVERY = 64
_WIN = 256

_HOOKS_INSTALLED = False
_WARNED: set[str] = set()


def _warn_once(logger: Any, key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(msg)


def _env_choice(name: str, default: str, allowed: set[str], logger: Any) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value not in allowed:
        _warn_once(
            logger, f"env:{name}",
            f"[MTP] invalid {name}={raw!r}; using {default!r} (allowed {sorted(allowed)})",
        )
        return default
    return value


def _env_int(name: str, default: int, lo: int, hi: int, logger: Any) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        v = int(raw.strip())
    except (TypeError, ValueError):
        _warn_once(logger, f"env:{name}", f"[MTP] invalid {name}={raw!r}; using {default}")
        return default
    if not lo <= v <= hi:
        _warn_once(logger, f"env:{name}", f"[MTP] {name}={v} clamped to [{lo},{hi}]")
        return max(lo, min(hi, v))
    return v


# ---------------------------------------------------------------------------
# MTP block
# ---------------------------------------------------------------------------


class GLM52MTPModule(nn.Module):
    """The nextn/MTP layer: enorm/hnorm/eh_proj + full decoder block + norm.

    embed_tokens / lm_head are *not* owned here — they are tied with the main
    model and passed at call sites (keeps this module's parameters exactly the
    mtp.safetensors contents; strict load_weights guards the mapping).
    """

    def __init__(self, args: Any, layer_idx: int) -> None:
        super().__init__()
        from mlx_lm.models.deepseek_v32 import DeepseekV32DecoderLayer

        hidden = int(args.hidden_size)
        eps = float(args.rms_norm_eps)
        self.enorm = nn.RMSNorm(hidden, eps=eps)
        self.hnorm = nn.RMSNorm(hidden, eps=eps)
        self.eh_proj = nn.Linear(2 * hidden, hidden, bias=False)
        self.block = DeepseekV32DecoderLayer(args, layer_idx)
        self.head_norm = nn.RMSNorm(hidden, eps=eps)


def _rename_sidecar_key(key: str, layer_idx: int) -> str | None:
    prefix = f"model.layers.{layer_idx}."
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix):]
    if rest.startswith(("enorm.", "hnorm.", "eh_proj.")):
        return rest
    if rest.startswith("shared_head.norm."):
        return "head_norm." + rest[len("shared_head.norm."):]
    return "block." + rest


def load_mtp_module(
    args: Any,
    weights_path: Path,
    *,
    layer_idx: int,
    quant: dict[str, Any] | None,
    logger: Any = default_logger,
) -> GLM52MTPModule:
    """Build the block and side-load mtp.safetensors (presence-driven quant)."""
    raw = mx.load(str(weights_path))
    weights: dict[str, mx.array] = {}
    for k, v in raw.items():
        nk = _rename_sidecar_key(k, layer_idx)
        if nk is None:
            raise RuntimeError(f"[MTP] unexpected tensor {k!r} in {weights_path}")
        weights[nk] = v

    module = GLM52MTPModule(args, layer_idx)

    q = quant or {}
    group_size = int(q.get("group_size", 64))
    bits = int(q.get("bits", 8))
    mode = str(q.get("mode", "affine"))

    def class_predicate(p: str, m: nn.Module) -> bool:
        return hasattr(m, "to_quantized") and f"{p}.scales" in weights

    nn.quantize(
        module, group_size=group_size, bits=bits, mode=mode,
        class_predicate=class_predicate,
    )
    module.load_weights(list(weights.items()), strict=True)
    mx.eval(module.parameters())

    n_bytes = sum(v.nbytes for _, v in tree_flatten(module.parameters()))
    logger.info(
        f"[MTP] side-loaded {weights_path.name}: {len(weights)} tensors, "
        f"{n_bytes / 1e9:.2f} GB (int{bits} g{group_size} {mode}, "
        f"presence-driven), layer_idx={layer_idx}"
    )
    return module


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


class _PreNormCapture(nn.RMSNorm):
    """Drop-in replacement for the model's final RMSNorm that also stores the
    hidden it produces (mode="post", vLLM convention: DeepseekV2Model.forward
    returns the normed hidden and the MTP proposer consumes it) or consumes
    (mode="pre", donor convention, kept for A/B).

    It *is* an nn.RMSNorm owning the same weight array, so the parameter
    tree (model.norm.weight), children() and traversal stay intact — a
    plain callable in that slot would drop the norm from the module tree.
    """

    def __init__(self, orig: nn.RMSNorm, store: dict[str, Any], mode: str = "post") -> None:
        nn.Module.__init__(self)
        self.weight = orig.weight
        self.eps = orig.eps
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_mode", mode)

    def __call__(self, x: mx.array) -> mx.array:
        out = mx.fast.rms_norm(x, self.weight, self.eps)
        self._store["h"] = out if self._mode == "post" else x
        return out


class _MTPState:
    def __init__(
        self,
        *,
        mode: str,
        concat: str,
        mtp: GLM52MTPModule,
        embed: Any,
        lm_head: Any,
        store: dict[str, Any],
        logger: Any,
        validate: bool = False,
        trace_n: int = 0,
        prof: bool = False,
        draft_k: int = 1,
    ) -> None:
        from mlx_lm.models.cache import CacheList, KVCache

        self.mode = mode
        self.concat = concat
        self.mtp = mtp
        self.embed = embed
        self.lm_head = lm_head
        self.store = store
        self.logger = logger
        self.validate = validate
        self.trace_n = trace_n
        self.prof = prof
        self.draft_k = draft_k
        self.cycle_pack: tuple | None = None
        self.acc_pos = [0, 0, 0]  # accepted at draft positions 1..3
        self.p_build = 0.0
        self.p_resolve = 0.0
        self.p_post = 0.0
        self._cache_cls = lambda: CacheList(KVCache(), KVCache())

        self.uid: int | None = None
        self.cur_batch: Any = None
        self.mtp_cache: Any = None
        # shadow fields
        self.pending_draft: mx.array | None = None
        self.lazy_match: mx.array | None = None
        self.steps = 0
        self.matches = 0
        self.win: list[bool] = []
        # battle fields
        self.request_battle: bool | None = None
        self.shadow_disabled = False
        self.prev_ran = False   # exactly-once guard for the real step
        self.last_battle: bool | None = None  # last per-request decision (diagnostics)
        self.buffer: list[tuple[mx.array, mx.array]] = []
        self.h_last: mx.array | None = None
        self.mtp_backlog: list[tuple[mx.array, mx.array]] = []
        self.flush_pack: tuple[mx.array, mx.array, mx.array] | None = None
        self.cycles = 0
        self.proposed = 0
        self.accepted = 0
        self.out_tokens = 0
        self.t0 = 0.0

    # -- request lifecycle --------------------------------------------------

    def start_request(self, uid: int) -> None:
        if self.uid is not None and (self.steps or self.cycles):
            self.finish()
        self.uid = uid
        self.mtp_cache = self._cache_cls()
        self.pending_draft = None
        self.lazy_match = None
        self.steps = 0
        self.matches = 0
        self.win = []
        self.request_battle = None
        self.shadow_disabled = False
        self.buffer = []
        self.h_last = None
        self.mtp_backlog = []
        self.flush_pack = None
        self.cycle_pack = None
        self.acc_pos = [0, 0, 0]
        self.cycles = 0
        self.proposed = 0
        self.accepted = 0
        self.out_tokens = 0
        self.t0 = time.perf_counter()

    def reset(self) -> None:
        self.uid = None
        self.cur_batch = None
        self.mtp_cache = None
        self.pending_draft = None
        self.lazy_match = None
        self.request_battle = None
        self.buffer = []
        self.h_last = None
        self.mtp_backlog = []
        self.flush_pack = None
        self.cycle_pack = None

    def finish(self) -> None:
        if self.lazy_match is not None:  # materialize the last shadow comparison
            try:
                self.account(bool(self.lazy_match.item()))
            except Exception:
                pass
            self.lazy_match = None
        if self.uid is not None and (self.steps or self.cycles):
            self._summary("finish")
        self.reset()

    # -- shadow accounting ---------------------------------------------------

    def account(self, hit: bool) -> None:
        self.steps += 1
        self.matches += int(hit)
        self.win.append(hit)
        if len(self.win) > _WIN:
            self.win.pop(0)
        if self.steps % _LOG_EVERY == 0:
            wr = (sum(self.win) / len(self.win)) if self.win else 0.0
            self.logger.info(
                f"[MTP_SHADOW] uid={self.uid} steps={self.steps} "
                f"match={self.matches} rate={self.matches / self.steps:.3f} "
                f"win{len(self.win)}={wr:.3f}"
            )

    def account_cycle(self, m: int) -> None:
        self.cycles += 1
        self.proposed += self.draft_k
        self.accepted += m
        self.out_tokens += m + 1
        for i in range(min(m, 3)):
            self.acc_pos[i] += 1
        if self.cycles % _LOG_EVERY == 0:
            a1 = self.acc_pos[0] / max(self.cycles, 1)
            a2 = self.acc_pos[1] / max(self.acc_pos[0], 1)
            a3 = self.acc_pos[2] / max(self.acc_pos[1], 1)
            self.logger.info(
                f"[MTP] uid={self.uid} cycles={self.cycles} k={self.draft_k} "
                f"proposed={self.proposed} accepted={self.accepted} "
                f"accept_rate={self.accepted / self.proposed:.3f} "
                f"a1={a1:.3f} a2={a2:.3f} a3={a3:.3f} out={self.out_tokens} "
                f"eff_tokens_per_step={self.out_tokens / self.cycles:.3f}"
            )

    def _summary(self, why: str) -> None:
        elapsed = max(time.perf_counter() - self.t0, 1e-9)
        if self.cycles:
            a1 = self.acc_pos[0] / max(self.cycles, 1)
            a2 = self.acc_pos[1] / max(self.acc_pos[0], 1)
            a3 = self.acc_pos[2] / max(self.acc_pos[1], 1)
            self.logger.info(
                f"[MTP_SUMMARY] ({why}) uid={self.uid} req_cycles={self.cycles} "
                f"k={self.draft_k} proposed={self.proposed} accepted={self.accepted} "
                f"accept_rate={self.accepted / max(self.proposed, 1):.3f} "
                f"a1={a1:.3f} a2={a2:.3f} a3={a3:.3f} out={self.out_tokens} "
                f"eff_tokens_per_step={self.out_tokens / self.cycles:.3f} "
                f"gen_tps={self.out_tokens / elapsed:.2f}"
            )
        if self.steps:
            wr = (sum(self.win) / len(self.win)) if self.win else 0.0
            self.logger.info(
                f"[MTP_SHADOW] summary({why}) uid={self.uid} steps={self.steps} "
                f"match={self.matches} rate={self.matches / self.steps:.3f} "
                f"win{len(self.win)}={wr:.3f}"
            )

    # -- math ---------------------------------------------------------------

    def draft_multi(self, pairs: list[tuple[mx.array, mx.array]]) -> mx.array:
        """MTP forward over (h, token) pairs; returns greedy draft for the
        last position. Ingests every pair into the MTP cache (backlog+1)."""
        from mlx_lm.models.base import create_attention_mask

        h_in = mx.concatenate([h for h, _ in pairs], axis=1)
        t_in = mx.concatenate([t.reshape(1, 1) for _, t in pairs], axis=1)
        e = self.mtp.enorm(self.embed(t_in))
        h = self.mtp.hnorm(h_in.astype(e.dtype))
        if self.concat == "eh":
            x = mx.concatenate([e, h], axis=-1)
        else:
            x = mx.concatenate([h, e], axis=-1)
        x = self.mtp.eh_proj(x)
        mask = None
        if x.shape[1] > 1:
            mask = create_attention_mask(x, self.mtp_cache[0], return_array=True)
        y = self.mtp.block(x, mask=mask, cache=self.mtp_cache)
        # Recycle the POST-final-norm hidden into the next chained step
        # (vLLM PR #47448: pre-norm recycle mismatches hnorm, 3.6 -> 4.4
        # accepted length on GLM-5.2). Same tensor feeds the LM head.
        h_post = self.mtp.head_norm(y[:, -1:, :])
        logits = self.lm_head(h_post)
        return mx.argmax(logits[..., -1, :], axis=-1).reshape(1), h_post

    def draft(self, h_last: mx.array, next_tok: mx.array) -> mx.array:
        return self.draft_multi([(h_last, next_tok)])[0]

    def mtp_cache_arrays(self) -> list[mx.array]:
        if self.mtp_cache is None:
            return []
        out: list[mx.array] = []
        for c in self.mtp_cache.caches:
            if getattr(c, "keys", None) is not None:
                out.append(c.keys)
                out.append(c.values)
        return out


def _request_policy(batch: Any) -> dict[str, bool] | None:
    """Explicit sampling policy carried by the request's sampler (ExoSampler
    in exo's submit path). Never probe: with min_p filtering a stochastic
    sampler is indistinguishable from greedy on a peaked probe, and probing
    consumes RNG. Unknown sampler -> None (fail closed: no battle)."""
    sampler = None
    if getattr(batch, "samplers", None) and batch.samplers[0] is not None:
        sampler = batch.samplers[0]
    else:
        sampler = getattr(batch, "fallback_sampler", None)
    greedy = getattr(sampler, "greedy", None)
    if greedy is None:
        return None
    return {"greedy": bool(greedy), "logprobs": bool(getattr(sampler, "logprobs", False))}


def _dist_group() -> Any | None:
    try:
        g = mx.distributed.init()
        return g if g is not None and g.size() > 1 else None
    except Exception:
        return None


def _off(c: Any) -> int:
    o = c.offset
    if isinstance(o, mx.array):
        return int(o.reshape(-1)[0].item())
    return int(o)


def _validate_cycle(state: _MTPState, batch: Any, m: int) -> None:
    cl0 = batch.prompt_cache[0]
    cl_last = batch.prompt_cache[-1]
    off00, off01 = _off(cl0[0]), _off(cl0[1])
    offl0, offl1 = _off(cl_last[0]), _off(cl_last[1])
    if not (off00 == off01 == offl0 == offl1):
        raise _MTPValidateError(
            f"[MTP][VALIDATE] slot/layer offset skew: "
            f"L0=({off00},{off01}) Ln=({offl0},{offl1})"
        )
    grp = _dist_group()
    if grp is None:
        return
    mtp_off = _off(state.mtp_cache[0]) if state.mtp_cache is not None else 0
    vec = mx.array([m, off00, mtp_off, len(state.buffer)], dtype=mx.int32)
    total = mx.distributed.all_sum(vec, group=grp)
    mx.eval(total)
    expected = vec * grp.size()
    if not mx.array_equal(total, expected).item():
        raise _MTPValidateError(
            f"[MTP][VALIDATE] cross-rank divergence: local={vec.tolist()} "
            f"sum={total.tolist()} size={grp.size()}"
        )


# ---------------------------------------------------------------------------
# Shadow step
# ---------------------------------------------------------------------------


def _shadow_pre(state: _MTPState, batch: Any) -> bool:
    if len(batch.uids) != 1:
        state.reset()
        return False
    uid = batch.uids[0]
    if state.uid != uid:
        state.start_request(uid)
    if state.lazy_match is not None:
        state.account(bool(state.lazy_match.item()))
        state.lazy_match = None
    return True


def _shadow_step(state: _MTPState, prev_step: Any, batch: Any):
    """Shadow = the real step plus a side comparison. The real step runs
    exactly once and never inside a try: a failure in the pre phase skips
    shadow for this step, a failure in the post phase disables shadow for
    the request; neither re-runs or masks the decode."""
    try:
        active = _shadow_pre(state, batch)
    except Exception:
        _warn_once(state.logger, "shadow-pre", "[MTP_SHADOW] pre-phase failed; idle")
        active = False
    result = prev_step(batch)
    if not active or state.shadow_disabled:
        return result
    try:
        _shadow_post(state, batch)
    except Exception:
        state.logger.opt(exception=True).warning(
            "[MTP_SHADOW] post-phase failed; shadow disabled for this request"
        )
        state.shadow_disabled = True
    return result


def _shadow_post(state: _MTPState, batch: Any) -> None:
    h = state.store.pop("h", None)
    next_tok = batch._next_tokens
    if h is None or next_tok is None:
        _warn_once(
            state.logger, "no-h",
            "[MTP_SHADOW] hidden not captured; shadow idle this step",
        )
        return

    if state.pending_draft is not None:
        state.lazy_match = mx.equal(
            state.pending_draft, next_tok.reshape(-1)[0:1].astype(mx.int64)
        )
        mx.async_eval(state.lazy_match)

    d = state.draft(h[:, -1:, :], next_tok.reshape(-1)[0:1])
    state.pending_draft = d.astype(mx.int64)
    mx.async_eval(state.pending_draft, *state.mtp_cache_arrays())


# ---------------------------------------------------------------------------
# Battle step (M1, greedy)
# ---------------------------------------------------------------------------


def _lp_row(lp: Any) -> mx.array:
    if isinstance(lp, mx.array):
        return lp[0] if lp.ndim == 2 else lp
    if isinstance(lp, list) and lp:
        first = lp[0]
        return first[0] if isinstance(first, mx.array) and first.ndim == 2 else first
    return mx.zeros((1,))


class _MTPBail(RuntimeError):
    """Abort the current battle cycle; the guard rolls back and falls back."""


class _MTPValidateError(RuntimeError):
    """VALIDATE assertion: never swallowed, never converted into a fallback."""


def _cur(c: Any) -> int:
    idx = getattr(c, "_idx", None)
    if idx is not None:
        return int(idx)
    return _off(c)


def _snapshot(state: _MTPState, batch: Any) -> dict[str, Any]:
    return {
        "main": [[_cur(c) for c in cl.caches] for cl in batch.prompt_cache],
        "mtp": [_cur(c) for c in state.mtp_cache.caches] if state.mtp_cache is not None else None,
        "next_tokens": batch._next_tokens,
        "next_logprobs": batch._next_logprobs,
        "tokens_len": len(batch.tokens[0]) if batch.tokens else 0,
        "h_last": state.h_last,
        "backlog": list(state.mtp_backlog),
        "buffer": list(state.buffer),
        "cycle_pack": state.cycle_pack,
    }


def _rollback(state: _MTPState, batch: Any, snap: dict[str, Any]) -> None:
    for cl, offs in zip(batch.prompt_cache, snap["main"]):
        for c, before in zip(cl.caches, offs):
            n = _cur(c) - before
            if n > 0:
                c.trim(n)
    if snap["mtp"] is not None and state.mtp_cache is not None:
        for c, before in zip(state.mtp_cache.caches, snap["mtp"]):
            n = _cur(c) - before
            if n > 0:
                c.trim(n)
    batch._next_tokens = snap["next_tokens"]
    batch._next_logprobs = snap["next_logprobs"]
    if batch.tokens:
        del batch.tokens[0][snap["tokens_len"]:]
    state.h_last = snap["h_last"]
    state.mtp_backlog = snap["backlog"]
    state.buffer = snap["buffer"]
    state.cycle_pack = snap["cycle_pack"]


def _apply_processors_rows(batch: Any, logits: mx.array, fed: list[mx.array]) -> mx.array:
    """Apply the request's logits processors to each verify row exactly as
    the serial decode step would: row j sees the history the step after
    feeding fed[0..j-1] would see. Mirrors opt_batch_gen._patched_step, whose
    history is ``mx.array(self.tokens[e])`` — the committed tokens *before*
    the current input. Without processors the logits pass through untouched."""
    procs = getattr(batch, "logits_processors", None)
    procs = procs[0] if procs and procs[0] else None
    if not procs:
        return logits
    hist = mx.array(batch.tokens[0], dtype=mx.int32)
    rows = []
    for j in range(logits.shape[1]):
        row = logits[:, j, :]
        for pr in procs:
            row = pr(hist, row)
        rows.append(row)
        if j + 1 < logits.shape[1]:
            hist = mx.concatenate([hist, fed[j].reshape(-1)[0:1].astype(mx.int32)])
    return mx.stack(rows, axis=1)


def _battle_step(state: _MTPState, prev_step: Any, batch: Any):
    from exo.worker.engines.mlx.patches.opt_batch_gen import (
        _advance_chain_arrays,
        _drain_prompt_cache_if_needed,
    )

    if len(batch.uids) != 1:
        state.reset()
        return prev_step(batch)
    uid = batch.uids[0]
    if state.uid != uid:
        state.start_request(uid)

    # ---- buffered emission first: a verified-but-unemitted draft must never
    # be skipped by a dynamic gate (cache already holds its position) ------
    if state.buffer:
        tok, lp = state.buffer.pop(0)
        ti = int(tok.reshape(-1)[0].item())
        batch.tokens[0].append(ti)
        if not state.buffer:
            state.cycle_pack = None
        return [ti], [_lp_row(lp)]

    # ---- request-level eligibility, decided once from explicit policy -----
    if state.request_battle is None:
        pol = _request_policy(batch)
        state.request_battle = bool(pol and pol["greedy"] and not pol["logprobs"])
        state.last_battle = state.request_battle
        if not state.request_battle:
            why = (
                "unknown sampler policy" if pol is None
                else "logprobs requested" if pol["logprobs"]
                else "non-greedy request"
            )
            _warn_once(
                state.logger, f"nobattle:{why}",
                f"[MTP] {why} under mode=on: running shadow for it "
                "(M2 rejection sampling lands in phase 3)",
            )
    if not state.request_battle:
        return _shadow_step(state, prev_step, batch)

    tb = getattr(batch, "_topk_buffer", None)
    if tb is not None and getattr(tb, "needs_topk", False):
        # policy already excludes logprobs requests; defensive only (cycle
        # boundary: no buffered/outstanding state here).
        _warn_once(state.logger, "topk", "[MTP] top-k requested mid-request; normal step")
        return prev_step(batch)

    state.cur_batch = batch

    # ---- speculation cycle -------------------------------------------------
    y = batch._next_tokens
    y_lp = batch._next_logprobs

    _t0 = time.perf_counter() if state.prof else 0.0
    if state.h_last is None:
        # Seed step: one normal decode step for this request. Its forward
        # writes the hidden of *this* batch into the capture store, so the
        # first draft never relies on a stale global capture (audit P1.2).
        result = prev_step(batch)
        h = state.store.pop("h", None)
        if h is not None and h.shape[0] == 1:
            state.h_last = h[:, -1:, :]
        return result

    k = state.draft_k
    pairs = state.mtp_backlog + [(state.h_last, y.reshape(-1)[0:1])]
    state.mtp_backlog = []
    d1, h_mtp = state.draft_multi(pairs)
    drafts = [d1]
    for _ in range(1, k):
        # chained proposal: the MTP block's own post-norm output stands in
        # for the (not yet computed) main-model hidden of the previous draft.
        # GLM-5 trains the head with 3 parameter-shared steps, so the chain
        # is in-distribution up to k=3.
        d_next, h_mtp = state.draft_multi([(h_mtp, drafts[-1])])
        drafts.append(d_next)

    verify_in = mx.concatenate(
        [y.reshape(-1)[0:1]] + [x.astype(y.dtype).reshape(-1)[0:1] for x in drafts]
    ).reshape(1, k + 1)
    logits2 = batch.model(verify_in, cache=batch.prompt_cache)
    hv = state.store.pop("h", None)
    logits2 = _apply_processors_rows(batch, logits2, [y] + drafts)
    lp2 = logits2 - mx.logsumexp(logits2, axis=-1, keepdims=True)
    t_all = mx.argmax(lp2, axis=-1)                       # (1, k+1)
    t1 = t_all[:, 0]
    accs = []
    prev = None
    for i in range(k):
        hit = mx.equal(drafts[i].astype(mx.int64), t_all[:, i].astype(mx.int64))
        prev = hit if prev is None else mx.logical_and(prev, hit)
        accs.append(prev)
    acc1 = accs[0]

    if state.prof:
        _t1 = time.perf_counter()
    mx.eval(*accs, y)
    m = 0
    for a in accs:
        if int(a.reshape(-1)[0].item()):
            m += 1
        else:
            break
    if state.prof:
        _t2 = time.perf_counter()
    d = d1

    if state.cycles < state.trace_n:
        row = lp2[0, 0, :].astype(mx.float32)
        top1 = mx.max(row)
        t1i = int(t1.reshape(-1)[0].item())
        second = mx.max(
            mx.where(
                mx.arange(row.shape[0]) == t1i, mx.array(-3.4e38), row
            )
        )
        margin = float((top1 - second).item())
        extra = ""
        if state.cycles == 0:
            h0 = state.h_last if state.h_last is not None else hv[:, 0:1, :]
            extra = f" h0sum={float(mx.sum(h0.astype(mx.float32)).item()):.6f}"
        state.logger.info(
            f"[MTP_TRACE] uid={state.uid} c={state.cycles} "
            f"y={int(y.reshape(-1)[0].item())} "
            f"d={[int(x.reshape(-1)[0].item()) for x in drafts]} "
            f"t1={t1i} m={m} margin={margin:.6g}{extra}"
        )

    if hv is None or hv.shape[1] < k + 1:
        # capture broke mid-flight: the guard rolls every cache back to the
        # cycle-entry snapshot and runs the normal step exactly once.
        raise _MTPBail("verify hidden not captured")

    # The chained MTP entries (steps 2..k) were computed from the block's own
    # hidden; drop them — accepted positions re-enter through the backlog
    # with the true main-model hidden from this verify.
    if k >= 2 and state.mtp_cache is not None:
        for c in state.mtp_cache.caches:
            c.trim(k - 1)
    if k - m:
        for c in batch.prompt_cache:
            c.trim(k - m)

    batch._next_tokens = t_all[:, m].astype(y.dtype)
    batch._next_logprobs = lp2[:, m]
    state.h_last = hv[:, m:m + 1, :]
    state.mtp_backlog = [(hv[:, i:i + 1, :], drafts[i]) for i in range(m)]
    state.buffer = [(drafts[i], lp2[:, i]) for i in range(m)]
    state.cycle_pack = (t_all, lp2, hv, drafts, m) if m else None
    state.flush_pack = None
    mx.async_eval(
        batch._next_tokens, batch._next_logprobs, *drafts,
        *_advance_chain_arrays(batch.prompt_cache),
        *state.mtp_cache_arrays(),
    )

    batch._current_tokens = y
    batch._current_logprobs = y_lp
    _drain_prompt_cache_if_needed(batch)

    ti = int(y.reshape(-1)[0].item())
    batch.tokens[0].append(ti)
    if state.prof:
        state.p_build += _t1 - _t0
        state.p_resolve += _t2 - _t1
        state.p_post += time.perf_counter() - _t2
        if state.cycles % _LOG_EVERY == _LOG_EVERY - 1:
            n = _LOG_EVERY
            state.logger.info(
                f"[MTP_PROF] uid={state.uid} win={n} "
                f"build_ms={1e3 * state.p_build / n:.2f} "
                f"resolve_ms={1e3 * state.p_resolve / n:.2f} "
                f"post_ms={1e3 * state.p_post / n:.2f}"
            )
            state.p_build = state.p_resolve = state.p_post = 0.0
    state.account_cycle(m)
    if state.validate:
        _validate_cycle(state, batch, m)
    return [ti], [_lp_row(y_lp)]


def _flush_buffered(state: _MTPState, batch: Any) -> None:
    """Roll buffered (verified, unemitted) drafts back so the request ends
    exactly where its emitted stream ends (B==1 only)."""
    if not state.buffer or state.cycle_pack is None:
        state.buffer = []
        return
    t_all, lp2, hv, drafts, m = state.cycle_pack
    r = len(state.buffer)          # still-buffered drafts to drop
    kept = m - r                   # drafts already emitted stay committed
    for c in batch.prompt_cache:
        c.trim(r)
    batch._next_tokens = t_all[:, kept].astype(batch._next_tokens.dtype)
    batch._next_logprobs = lp2[:, kept]
    state.h_last = hv[:, kept:kept + 1, :]
    state.mtp_backlog = [(hv[:, i:i + 1, :], drafts[i]) for i in range(kept)]
    state.buffer = []
    state.cycle_pack = None
    state.flush_pack = None
    state.out_tokens = max(state.out_tokens - r, 0)
    state.accepted = max(state.accepted - r, 0)
    state.logger.info(
        f"[MTP] uid={state.uid} buffered token flushed (batch merge/teardown); "
        f"rolled back to reject-equivalent state"
    )


# ---------------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------------


def _install_hooks(logger: Any) -> None:
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    from mlx_lm.generate import GenerationBatch

    prev_step = GenerationBatch._step
    if getattr(prev_step, "_exo_glm52_mtp_wrapper", False):
        _HOOKS_INSTALLED = True
        return
    if getattr(prev_step, "__name__", "") not in ("_patched_step", "_step"):
        _warn_once(
            logger, "base-step",
            f"[MTP] unexpected base GenerationBatch._step "
            f"({getattr(prev_step, '__name__', '?')}); wrapping anyway",
        )

    def _prev_once(state: _MTPState, batch: Any):
        state.prev_ran = True
        return prev_step(batch)

    def _mtp_step(self: Any):
        state = getattr(self.model, "_exo_glm52_mtp_state", None)
        if state is None or state.mode == "off":
            return prev_step(self)
        if state.mode != "on":
            return _shadow_step(state, prev_step, self)

        state.prev_ran = False
        snap = None
        if len(self.uids) == 1 and state.uid == self.uids[0] and state.h_last is not None:
            snap = _snapshot(state, self)
        try:
            return _battle_step(state, lambda b: _prev_once(state, b), self)
        except _MTPValidateError:
            raise
        except Exception as exc:
            if state.prev_ran:
                # the real step already executed for this call: never run it
                # twice; surface the failure instead of masking it.
                raise
            if snap is not None:
                _rollback(state, self, snap)
            if isinstance(exc, _MTPBail):
                _warn_once(logger, f"bail:{exc}", f"[MTP] {exc}; request continues without MTP")
            else:
                logger.opt(exception=True).warning(
                    "[MTP] exception in battle cycle; rolled back, request continues without MTP"
                )
            state.request_battle = False
            state.buffer = []
            state.cycle_pack = None
            return _prev_once(state, self)

    _mtp_step._exo_glm52_mtp_wrapper = True  # type: ignore[attr-defined]
    GenerationBatch._step = _mtp_step

    prev_extract = GenerationBatch.extract_cache

    def _mtp_extract_cache(self: Any, idx: int):
        state = getattr(self.model, "_exo_glm52_mtp_state", None)
        if state is not None and state.buffer and state.uid in self.uids:
            # finish landed while a verified position sat in the buffer: the
            # cache is one position ahead of the emitted stream — drop it so
            # KVPrefixCache stores tokens/cache in lockstep (recon D4).
            for c in self.prompt_cache:
                c.trim(len(state.buffer))
            state.buffer = []
            state.flush_pack = None
            state.cycle_pack = None
        if state is not None and self.uids and state.uid == self.uids[idx]:
            state.finish()
        return prev_extract(self, idx)

    GenerationBatch.extract_cache = _mtp_extract_cache

    prev_extend = GenerationBatch.extend

    def _mtp_extend(self: Any, other: Any):
        state = getattr(self.model, "_exo_glm52_mtp_state", None)
        if state is not None and state.buffer and len(self.uids) == 1 \
                and state.uid == self.uids[0]:
            _flush_buffered(state, self)
        return prev_extend(self, other)

    GenerationBatch.extend = _mtp_extend

    _HOOKS_INSTALLED = True
    logger.info("[MTP] GenerationBatch hooks installed (_step/extract_cache/extend)")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

_REQUIRED_FAMILIES = (
    "enorm.weight", "hnorm.weight", "eh_proj.weight", "head_norm.weight",
    "block.input_layernorm.weight", "block.post_attention_layernorm.weight",
    "block.self_attn.q_a_proj.scales", "block.self_attn.embed_q.scales",
    "block.self_attn.indexer.wq_b.weight",
    "block.mlp.switch_mlp.gate_proj.scales", "block.mlp.gate.weight",
)


def _validate_sidecar(weights_path: Path, layer_idx: int) -> tuple[bool, str]:
    if not weights_path.is_file():
        return False, f"{weights_path} not found"
    try:
        keys = set(mx.load(str(weights_path)).keys())
    except Exception as e:  # noqa: BLE001 — fail-closed with reason
        return False, f"cannot read {weights_path}: {e!r}"
    renamed = {_rename_sidecar_key(k, layer_idx) for k in keys}
    missing = [f for f in _REQUIRED_FAMILIES if f not in renamed]
    if missing:
        return False, f"sidecar missing families: {missing[:4]}"
    return True, "ok"


def apply_glm52_mtp_patch(
    model: Any,
    model_path: Path,
    *,
    logger: Any = default_logger,
) -> Any:
    """Enable GLM-5.2/5.3 MTP (shadow or battle) on a loaded glm_moe_dsa model.

    Fail-closed: any precondition miss logs one line and returns the model
    unchanged. Always returns the original model object.
    """
    mode = _env_choice(_ENV_MODE, "off", {"off", "shadow", "on"}, logger)
    if mode == "off":
        return model

    try:
        config = json.loads((model_path / "config.json").read_text())
    except Exception:
        logger.warning("[MTP] cannot read config.json; patch not applied")
        return model
    if config.get("model_type") != "glm_moe_dsa":
        return model
    if int(config.get("num_nextn_predict_layers", 0)) != 1:
        logger.warning("[MTP] num_nextn_predict_layers != 1; patch not applied")
        return model

    args = getattr(model, "args", None)
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    lm_head = getattr(model, "lm_head", None)
    embed = getattr(inner, "embed_tokens", None)
    norm = getattr(inner, "norm", None)
    if args is None or not layers or lm_head is None or embed is None or norm is None:
        logger.warning("[MTP] model structure unexpected; patch not applied")
        return model
    layer_idx = int(config["num_hidden_layers"])
    if len(layers) != layer_idx:
        logger.warning(
            f"[MTP] {len(layers)} layers != num_hidden_layers={layer_idx}; "
            "patch not applied"
        )
        return model

    raw_weights = os.environ.get(_ENV_WEIGHTS, "auto").strip()
    weights_path = (
        model_path / "mtp.safetensors" if raw_weights in ("", "auto")
        else Path(raw_weights).expanduser()
    )
    ok, reason = _validate_sidecar(weights_path, layer_idx)
    if not ok:
        logger.warning(f"[MTP] {reason}; patch not applied")
        return model

    concat = _env_choice(_ENV_CONCAT, "eh", {"eh", "he"}, logger)
    validate = _env_int(_ENV_VALIDATE, 0, 0, 1, logger) == 1
    trace_n = _env_int(_ENV_TRACE, 0, 0, 4096, logger)
    prof = _env_int(_ENV_PROF, 0, 0, 1, logger) == 1
    draft_k = _env_int(_ENV_DRAFT_K, 1, 1, 3, logger)
    hidden_mode = _env_choice(_ENV_HIDDEN, "post", {"post", "pre"}, logger)

    try:
        mtp = load_mtp_module(
            args, weights_path, layer_idx=layer_idx,
            quant=config.get("quantization"), logger=logger,
        )
    except Exception:
        logger.opt(exception=True).warning(
            "[MTP] side-load failed; patch not applied"
        )
        return model

    store: dict[str, Any] = {}
    if isinstance(norm, _PreNormCapture):  # re-apply after a reload path
        store = norm._store
        object.__setattr__(norm, "_mode", hidden_mode)
    elif isinstance(norm, nn.RMSNorm):
        inner.norm = _PreNormCapture(norm, store, mode=hidden_mode)
    else:
        logger.warning("[MTP] final norm is not RMSNorm; patch not applied")
        return model

    state = _MTPState(
        mode=mode, concat=concat, mtp=mtp, embed=embed, lm_head=lm_head,
        store=store, logger=logger, validate=validate, trace_n=trace_n,
        prof=prof, draft_k=draft_k,
    )
    model._exo_glm52_mtp_state = state
    _install_hooks(logger)
    logger.info(
        f"[MTP] enabled mode={mode} k={draft_k} hidden={hidden_mode} concat={concat} "
        f"validate={int(validate)} "
        f"weights={weights_path.name} layer_idx={layer_idx}"
    )
    return model
