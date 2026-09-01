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


class _PreNormCapture:
    """Passthrough wrapper over the model's final norm; stores its input."""

    def __init__(self, orig: Any, store: dict[str, Any]) -> None:
        self._orig = orig
        self._store = store

    def __call__(self, x: mx.array) -> mx.array:
        self._store["h"] = x
        return self._orig(x)

    @property
    def weight(self) -> Any:  # keep donors of .weight working
        return self._orig.weight

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


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
        self.request_greedy: bool | None = None
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
            self._summary("switch")
        self.uid = uid
        self.mtp_cache = self._cache_cls()
        self.pending_draft = None
        self.lazy_match = None
        self.steps = 0
        self.matches = 0
        self.win = []
        self.request_greedy = None
        self.buffer = []
        self.h_last = None
        self.mtp_backlog = []
        self.flush_pack = None
        self.cycles = 0
        self.proposed = 0
        self.accepted = 0
        self.out_tokens = 0
        self.t0 = time.perf_counter()

    def reset(self) -> None:
        if self.uid is not None and (self.steps or self.cycles):
            self._summary("reset")
        self.uid = None
        self.cur_batch = None
        self.mtp_cache = None
        self.pending_draft = None
        self.lazy_match = None
        self.request_greedy = None
        self.buffer = []
        self.h_last = None
        self.mtp_backlog = []
        self.flush_pack = None

    def finish(self) -> None:
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
        self.proposed += 1
        self.accepted += m
        self.out_tokens += m + 1
        if self.cycles % _LOG_EVERY == 0:
            self.logger.info(
                f"[MTP] uid={self.uid} cycles={self.cycles} "
                f"proposed={self.proposed} accepted={self.accepted} "
                f"accept_rate={self.accepted / self.proposed:.3f} "
                f"out={self.out_tokens} "
                f"eff_tokens_per_step={self.out_tokens / self.cycles:.3f}"
            )

    def _summary(self, why: str) -> None:
        elapsed = max(time.perf_counter() - self.t0, 1e-9)
        if self.cycles:
            self.logger.info(
                f"[MTP_SUMMARY] ({why}) uid={self.uid} req_cycles={self.cycles} "
                f"proposed={self.proposed} accepted={self.accepted} "
                f"accept_rate={self.accepted / max(self.proposed, 1):.3f} "
                f"out={self.out_tokens} "
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
        logits = self.lm_head(self.mtp.head_norm(y[:, -1:, :]))
        return mx.argmax(logits[..., -1, :], axis=-1).reshape(1)

    def draft(self, h_last: mx.array, next_tok: mx.array) -> mx.array:
        return self.draft_multi([(h_last, next_tok)])

    def mtp_cache_arrays(self) -> list[mx.array]:
        if self.mtp_cache is None:
            return []
        out: list[mx.array] = []
        for c in self.mtp_cache.caches:
            if getattr(c, "keys", None) is not None:
                out.append(c.keys)
                out.append(c.values)
        return out


def _request_is_greedy(batch: Any) -> bool:
    sampler = None
    if getattr(batch, "samplers", None) and batch.samplers[0] is not None:
        sampler = batch.samplers[0]
    else:
        sampler = batch.fallback_sampler
    probe = mx.log(mx.array([[0.02, 0.96, 0.02]]))
    try:
        a = int(sampler(probe).reshape(-1)[0].item())
        b = int(sampler(probe).reshape(-1)[0].item())
    except Exception:
        return False
    return a == 1 and b == 1


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
        raise RuntimeError(
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
        raise RuntimeError(
            f"[MTP][VALIDATE] cross-rank divergence: local={vec.tolist()} "
            f"sum={total.tolist()} size={grp.size()}"
        )


# ---------------------------------------------------------------------------
# Shadow step
# ---------------------------------------------------------------------------


def _shadow_step(state: _MTPState, prev_step: Any, batch: Any):
    if len(batch.uids) != 1:
        state.reset()
        return prev_step(batch)
    uid = batch.uids[0]
    if state.uid != uid:
        state.start_request(uid)

    if state.lazy_match is not None:
        state.account(bool(state.lazy_match.item()))
        state.lazy_match = None

    result = prev_step(batch)

    h = state.store.pop("h", None)
    next_tok = batch._next_tokens
    if h is None or next_tok is None:
        _warn_once(
            state.logger, "no-h",
            "[MTP_SHADOW] pre-norm hidden not captured; shadow idle this step",
        )
        return result

    if state.pending_draft is not None:
        state.lazy_match = mx.equal(
            state.pending_draft, next_tok.reshape(-1)[0:1].astype(mx.int64)
        )
        mx.async_eval(state.lazy_match)

    d = state.draft(h[:, -1:, :], next_tok.reshape(-1)[0:1])
    state.pending_draft = d.astype(mx.int64)
    mx.async_eval(state.pending_draft, *state.mtp_cache_arrays())
    return result


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

    tb = getattr(batch, "_topk_buffer", None)
    if tb is not None and getattr(tb, "needs_topk", False):
        _warn_once(
            state.logger, "topk",
            "[MTP] logprobs requested; battle loop falls back to normal decode "
            "for such requests in M1",
        )
        return prev_step(batch)

    if state.request_greedy is None:
        state.request_greedy = _request_is_greedy(batch)
        if not state.request_greedy:
            _warn_once(
                state.logger, "nongreedy",
                "[MTP] non-greedy request under mode=on: running shadow for it "
                "(M2 rejection sampling lands in phase 3)",
            )
    if not state.request_greedy:
        return _shadow_step(state, prev_step, batch)

    state.cur_batch = batch

    # ---- buffered emission (accepted draft from the previous cycle) -------
    if state.buffer:
        tok, lp = state.buffer.pop(0)
        ti = int(tok.reshape(-1)[0].item())
        batch.tokens[0].append(ti)
        state.flush_pack = None
        return [ti], [_lp_row(lp)]

    # ---- speculation cycle -------------------------------------------------
    y = batch._next_tokens
    y_lp = batch._next_logprobs
    h_entry = state.store.pop("h", None)
    if state.h_last is None:
        if h_entry is None:
            _warn_once(
                state.logger, "no-h-battle",
                "[MTP] pre-norm hidden unavailable; normal step (will seed next)",
            )
            return prev_step(batch)
        state.h_last = h_entry[:, -1:, :]

    pairs = state.mtp_backlog + [(state.h_last, y.reshape(-1)[0:1])]
    state.mtp_backlog = []
    d = state.draft_multi(pairs)

    verify_in = mx.concatenate(
        [y.reshape(-1)[0:1], d.astype(y.dtype).reshape(-1)[0:1]]
    ).reshape(1, 2)
    logits2 = batch.model(verify_in, cache=batch.prompt_cache)
    hv = state.store.pop("h", None)
    lp2 = logits2 - mx.logsumexp(logits2, axis=-1, keepdims=True)
    t1 = mx.argmax(lp2[:, 0, :], axis=-1)
    acc = mx.equal(d.astype(mx.int64), t1.astype(mx.int64))

    mx.eval(acc, y)
    m = int(acc.reshape(-1)[0].item())

    if hv is None or hv.shape[1] < 2:
        # capture broke mid-flight: undo the verify and decay to normal decode
        for c in batch.prompt_cache:
            c.trim(2)
        _warn_once(
            state.logger, "no-hv",
            "[MTP] verify hidden not captured; battle disabled for request",
        )
        state.request_greedy = False
        return prev_step(batch)

    if m:
        b_tok = mx.argmax(lp2[:, 1, :], axis=-1)
        batch._next_tokens = b_tok.astype(y.dtype)
        batch._next_logprobs = lp2[:, 1]
        state.h_last = hv[:, 1:2, :]
        state.mtp_backlog = [(hv[:, 0:1, :], d)]
        state.buffer.append((d, lp2[:, 0]))
        state.flush_pack = (t1.astype(y.dtype), lp2[:, 0], hv[:, 0:1, :])
        mx.async_eval(
            batch._next_tokens, batch._next_logprobs, d,
            *_advance_chain_arrays(batch.prompt_cache),
            *state.mtp_cache_arrays(),
        )
    else:
        for c in batch.prompt_cache:
            c.trim(1)
        batch._next_tokens = t1.astype(y.dtype)
        batch._next_logprobs = lp2[:, 0]
        state.h_last = hv[:, 0:1, :]
        state.flush_pack = None
        mx.async_eval(
            batch._next_tokens, batch._next_logprobs,
            *_advance_chain_arrays(batch.prompt_cache),
            *state.mtp_cache_arrays(),
        )

    batch._current_tokens = y
    batch._current_logprobs = y_lp
    _drain_prompt_cache_if_needed(batch)

    ti = int(y.reshape(-1)[0].item())
    batch.tokens[0].append(ti)
    state.account_cycle(m)
    if state.validate:
        _validate_cycle(state, batch, m)
    return [ti], [_lp_row(y_lp)]


def _flush_buffered(state: _MTPState, batch: Any) -> None:
    """Roll an accept-cycle back to its reject-equivalent (B==1 only)."""
    if not state.buffer or state.flush_pack is None:
        state.buffer = []
        return
    t1, t1_lp, h0 = state.flush_pack
    for c in batch.prompt_cache:
        c.trim(len(state.buffer))
    batch._next_tokens = t1
    batch._next_logprobs = t1_lp
    state.h_last = h0
    state.mtp_backlog = []
    state.buffer = []
    state.flush_pack = None
    state.out_tokens = max(state.out_tokens - 1, 0)
    state.accepted = max(state.accepted - 1, 0)
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

    def _mtp_step(self: Any):
        state = getattr(self.model, "_exo_glm52_mtp_state", None)
        if state is None or state.mode == "off":
            return prev_step(self)
        try:
            if state.mode == "on":
                return _battle_step(state, prev_step, self)
            return _shadow_step(state, prev_step, self)
        except Exception:
            _warn_once(
                logger, "mtp-crash",
                "[MTP] exception in MTP path; disabling for this model",
            )
            logger.opt(exception=True).warning("[MTP] traceback")
            state.mode = "off"
            return prev_step(self)

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
    draft_k = _env_int(_ENV_DRAFT_K, 1, 1, 3, logger)
    if draft_k > 1:
        _warn_once(logger, "draft-k", "[MTP] DRAFT_K>1 is phase 4; using k=1")

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
    if not isinstance(norm, _PreNormCapture):
        inner.norm = _PreNormCapture(norm, store)
    else:  # re-apply after a reload path: reuse the wrapper's store
        store = norm._store

    state = _MTPState(
        mode=mode, concat=concat, mtp=mtp, embed=embed, lm_head=lm_head,
        store=store, logger=logger, validate=validate,
    )
    model._exo_glm52_mtp_state = state
    _install_hooks(logger)
    logger.info(
        f"[MTP] enabled mode={mode} concat={concat} validate={int(validate)} "
        f"weights={weights_path.name} layer_idx={layer_idx}"
    )
    return model
