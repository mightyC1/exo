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

import hashlib
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
_ENV_RECYCLE = "EXO_GLM52_MTP_RECYCLE"
_ENV_PREFILL = "EXO_GLM52_MTP_PREFILL"
_ENV_PREFILL_WINDOW = "EXO_GLM52_MTP_PREFILL_WINDOW"
_ENV_PREFILL_CYCLES = "EXO_GLM52_MTP_PREFILL_CYCLES"
_ENV_PROPOSAL = "EXO_GLM52_MTP_PROPOSAL"
_ENV_CF = "EXO_GLM52_MTP_CF"
_ENV_SPEC_DRAFT = "EXO_GLM52_MTP_SPEC_DRAFT"
_ENV_VERIFY_PAD = "EXO_GLM52_MTP_VERIFY_PAD"
_ENV_Q_TEMP = "EXO_GLM52_MTP_Q_TEMPERATURE"
_PREFILL_SUBCHUNK = 256

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
    raw: dict[str, mx.array] | None = None,
) -> GLM52MTPModule:
    """Build the block and side-load mtp.safetensors (presence-driven quant)."""
    if raw is None:
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

    def __init__(self, orig: nn.RMSNorm, store: dict[str, Any], mode: str = "pre") -> None:
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
        self.recycle = "post"
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
        self.request_sampling = False
        self.policy: dict[str, Any] | None = None
        self.hist_buf: mx.array | None = None   # committed tokens (int32), amortized growth
        self.hist_len = 0
        self.prefill_enabled = True
        self.prefill_window = 2048           # last W prompt pairs (0 = whole prompt)
        self.prefill_cycles = 64             # drop prompt context after N cycles (0 = keep)
        self.prompt_ctx = False              # MTP cache currently holds prompt context
        self.gen_pairs: list[tuple[mx.array, mx.array]] = []
        self.retired: list[Any] = []         # caches swapped out mid-request; freed at finish
        self.proposal = "sample"             # draft proposal under sampling: sample (measured +0.04) | argmax
        self.spec_draft = False              # M1.5-lite: measured net-zero/negative on MLX; opt-in
        self.verify_pad = 0                  # measurement only: extra dummy rows in the verify
        self.q_temperature: float | None = None  # proposal temperature override (None = target's)
        self.spec_next: tuple | None = None  # (d1, h_mtp, zq) valid for the next cycle
        self.cycle_spec_entries = 0          # MTP entries this cycle's spec left in the cache
        self.cf = False                      # counterfactual telemetry (alpha one-hot vs full-q)
        self.cf_acc: mx.array | None = None  # lazy (2,) accumulator [alpha_onehot, alpha_fullq]
        self.cf_n = 0
        self.pending: dict[str, Any] | None = None  # package currently being filled by prefill
        self.ready: list[dict[str, Any]] = []       # completed packages awaiting their request (≤4)
        self.shadow_disabled = False
        self.prev_ran = False   # exactly-once guard for the real step
        self.in_resolve = False # inside the target verify eval / collectives
        self.group: Any = None  # distributed group, resolved once at apply
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
        self.request_sampling = False
        self.policy = None
        self.hist_buf = None
        self.hist_len = 0
        self.prompt_ctx = False
        self.gen_pairs = []
        self.retired = []                    # nothing in flight between requests: safe to free
        self.cf_acc = None
        self.cf_n = 0
        self.spec_next = None
        self.cycle_spec_entries = 0
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

    def finish(self, why: str = "finish") -> None:
        if self.lazy_match is not None:  # materialize the last shadow comparison
            try:
                self.account(bool(self.lazy_match.item()))
            except Exception:
                pass
            self.lazy_match = None
        if self.uid is not None and (self.steps or self.cycles):
            self._summary(why)
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
                f"mode={'rs' if self.request_sampling else 'greedy'} "
                f"{self._cf_str()}"
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
                f"mode={'rs' if self.request_sampling else 'greedy'} proposal={self.proposal} "
                f"{self._cf_str()}"
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
        # Hidden recycled into the next chained step: "post" = after
        # shared_head.norm (vLLM PR #47448, measured with a post-norm target
        # hidden); "pre" = the block's raw residual output (consistent with
        # our measured-better pre-norm *target* hidden). A/B via env.
        h_raw = y[:, -1:, :]
        h_post = self.mtp.head_norm(h_raw)
        logits = self.lm_head(h_post)[..., -1, :]                  # (1, V)
        h_rec = h_post if self.recycle == "post" else h_raw
        if self.proposal == "sample" and self.request_sampling and self.policy is not None:
            # Full-q proposal: draft ~ q = the head's own distribution under the
            # request's sampling transform (optionally its own temperature —
            # any q is valid, acceptance is 1 - TV(p, q)).
            pol_q = self.policy if self.q_temperature is None else dict(self.policy, temperature=self.q_temperature)
            zq = _target_logits(logits - mx.logsumexp(logits, axis=-1, keepdims=True), pol_q)
            return mx.random.categorical(zq).reshape(1), h_rec, zq
        return mx.argmax(logits, axis=-1).reshape(1), h_rec, logits

    def _cf_str(self) -> str:
        if not self.cf or self.cf_acc is None or self.cf_n == 0:
            return ""
        try:
            a, b, c, d, q07, q085, q12 = (float(x) / self.cf_n for x in self.cf_acc.tolist())
            return (f"cf_onehot={a:.3f} cf_fullq={b:.3f} cf_top2={c:.3f} cf_top4={d:.3f} "
                    f"cf_q0.7={q07:.3f} cf_q0.85={q085:.3f} cf_q1.2={q12:.3f} ")
        except Exception:
            return ""

    def draft(self, h_last: mx.array, next_tok: mx.array) -> mx.array:
        return self.draft_multi([(h_last, next_tok)])[0]

    def cf_update(self, zp: mx.array, zq_raw: mx.array | None, d: mx.array) -> None:
        """Counterfactual telemetry: alpha_onehot = p(argmax q) (what the
        one-hot proposal accepts) and alpha_fullq = sum(min(p, q)) (what the
        full-q proposal would). zp: target logits (1, V); zq_raw: head logits
        (1, V) in whichever form draft_multi returned (raw or transformed)."""
        if zq_raw is None or self.policy is None:
            return
        p = mx.softmax(zp, axis=-1)
        zq = zq_raw if self.proposal == "sample" else _target_logits(
            zq_raw - mx.logsumexp(zq_raw, axis=-1, keepdims=True), self.policy
        )
        q = mx.softmax(zq, axis=-1)
        a_onehot = mx.take_along_axis(p, mx.argmax(q, axis=-1, keepdims=True), axis=-1).reshape(1)
        a_fullq = mx.sum(mx.minimum(p, q), axis=-1).reshape(1)
        # breadth ceiling: target mass on the head's top-2 / top-4 candidates
        # (= greedy tree acceptance at position 1 with 2 / 4 branches)
        top4 = mx.argpartition(-q, kth=3, axis=-1)[..., :4]
        p_top4 = mx.take_along_axis(p, top4, axis=-1)
        q_top4 = mx.take_along_axis(q, top4, axis=-1)
        order = mx.argsort(-q_top4, axis=-1)
        p_sorted = mx.take_along_axis(p_top4, order, axis=-1)
        a_top2 = mx.sum(p_sorted[..., :2], axis=-1).reshape(1)
        a_top4 = mx.sum(p_sorted, axis=-1).reshape(1)
        # proposal-temperature calibration sweep: overlap with q at T = policy_T * f
        sweep = []
        base_logq = zq_raw - mx.logsumexp(zq_raw, axis=-1, keepdims=True)
        for f in (0.7, 0.85, 1.2):
            if self.proposal == "sample":
                # zq_raw is already transformed at T; undo by rescaling logits
                zq_f = zq_raw * (1.0 / f)
            else:
                zq_f = _target_logits(base_logq, dict(self.policy, temperature=self.policy["temperature"] * f))
            sweep.append(mx.sum(mx.minimum(p, mx.softmax(zq_f, axis=-1)), axis=-1).reshape(1))
        upd = mx.concatenate([a_onehot, a_fullq, a_top2, a_top4] + sweep).astype(mx.float32)
        self.cf_acc = upd if self.cf_acc is None else self.cf_acc + upd
        self.cf_n += 1

    def ingest(self, cache: Any, h_seq: mx.array, tok_seq: mx.array) -> None:
        """Feed (h_i, t_{i+1}) pairs through the MTP block into ``cache`` with
        no LM head — used to give the head the prompt context before the
        first draft. h_seq: (1, L, H) pre-norm hiddens, tok_seq: (1, L)."""
        from mlx_lm.models.base import create_attention_mask

        e = self.mtp.enorm(self.embed(tok_seq))
        h = self.mtp.hnorm(h_seq.astype(e.dtype))
        x = mx.concatenate([e, h] if self.concat == "eh" else [h, e], axis=-1)
        x = self.mtp.eh_proj(x)
        mask = create_attention_mask(x, cache[0], return_array=True) if x.shape[1] > 1 else None
        self.mtp.block(x, mask=mask, cache=cache)

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
    return {
        "greedy": bool(greedy),
        "logprobs": bool(getattr(sampler, "logprobs", False)),
        "temperature": float(getattr(sampler, "temperature", 0.0)),
        "top_p": float(getattr(sampler, "top_p", 1.0)),
        "min_p": float(getattr(sampler, "min_p", 0.0)),
        "top_k": int(getattr(sampler, "top_k", 0)),
    }


def _target_logits(row_logprobs: mx.array, pol: dict[str, Any]) -> mx.array:
    """Tempered, filtered logits whose softmax is exactly the distribution the
    normal step samples from: make_sampler applies top_p -> min_p -> top_k to
    the (untempered) logprobs and then categorical(logits / temp). Works on
    any leading batch shape (the pin's filters act along the last axis), so
    all k+1 verify rows are filtered in one pass."""
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

    # float32 throughout: p and q are compared and subtracted; bf16 loses the
    # significant bits exactly where it matters (p ≈ q).
    x = row_logprobs.astype(mx.float32)
    top_p, min_p, top_k, temp = pol["top_p"], pol["min_p"], pol["top_k"], pol["temperature"]
    if 0.0 < top_p < 1.0:
        x = apply_top_p(x, top_p)
    if min_p != 0.0:
        x = apply_min_p(x, min_p, 1)
    if top_k > 0:
        x = apply_top_k(x, top_k)
    return x * (1.0 / temp)


def _rs_accepts(zs: list[mx.array], drafts: list[mx.array], us: list[mx.array]) -> list[mx.array]:
    """Speculative sampling with a deterministic (greedy) draft: position j is
    accepted with probability p_j(d_j) (Leviathan et al. with q = one-hot);
    acceptance is a prefix. Returns the lazy prefix-accept flags."""
    accs: list[mx.array] = []
    prev = None
    for j, d in enumerate(drafts):
        pj = mx.softmax(zs[j], axis=-1)
        pd = mx.take_along_axis(pj, d.reshape(1, 1).astype(mx.int32), axis=-1).reshape(1)
        hit = us[j] < pd
        prev = hit if prev is None else mx.logical_and(prev, hit)
        accs.append(prev)
    return accs


def _rs_accepts_vec(z_rows: mx.array, drafts: list[mx.array], u: mx.array) -> list[mx.array]:
    """Vectorized form of _rs_accepts: z_rows is (k+1, V) tempered+filtered
    logits, u is (k,) uniforms. log p_j(d_j) = z[j, d_j] - logsumexp(z[j]) —
    no full-vocab softmax; the prefix-and is a cumulative product."""
    k = len(drafts)
    zk = z_rows[:k]
    d = mx.concatenate([x.reshape(1).astype(mx.int32) for x in drafts]).reshape(k, 1)
    log_pd = mx.take_along_axis(zk, d, axis=-1).reshape(k) - mx.logsumexp(zk, axis=-1)
    hits = (mx.log(u) < log_pd).astype(mx.int32)
    prefix = mx.cumprod(hits)
    return [prefix[j:j + 1].astype(mx.bool_) for j in range(k)]


def _rs_accepts_q(z_rows: mx.array, zq_rows: list[mx.array], drafts: list[mx.array], u: mx.array) -> list[mx.array]:
    """Speculative sampling with proposal q: accept d_j w.p. min(1, p_j(d_j)/q_j(d_j))."""
    k = len(drafts)
    d = mx.concatenate([x.reshape(1).astype(mx.int32) for x in drafts]).reshape(k, 1)
    zk = z_rows[:k]
    log_p = mx.take_along_axis(zk, d, axis=-1).reshape(k) - mx.logsumexp(zk, axis=-1)
    zq = mx.concatenate([z.reshape(1, -1) for z in zq_rows[:k]], axis=0)
    log_q = mx.take_along_axis(zq, d, axis=-1).reshape(k) - mx.logsumexp(zq, axis=-1)
    thresh = mx.minimum(log_p - log_q, mx.array(0.0, dtype=log_p.dtype))
    hits = (mx.log(u) < thresh).astype(mx.int32)
    prefix = mx.cumprod(hits)
    return [prefix[j:j + 1].astype(mx.bool_) for j in range(k)]


def _rs_residual_logits_q(zp: mx.array, zq: mx.array) -> mx.array:
    """Logits of norm(max(p - q, 0)) — the distribution to sample from after
    rejecting a draft proposed from q. Computed in log space (float32):
    log r = log p + log1p(-exp(log q - log p)) where p > q, -inf elsewhere.
    If numerically nothing survives (p == q up to rounding), fall back to the
    target itself — a valid sample of p and never an all-(-inf) categorical."""
    zp = zp.astype(mx.float32)
    zq = zq.reshape(zp.shape).astype(mx.float32)
    logp = zp - mx.logsumexp(zp, axis=-1, keepdims=True)
    logq = zq - mx.logsumexp(zq, axis=-1, keepdims=True)
    delta = mx.minimum(logq - logp, mx.array(-1e-7, dtype=mx.float32))
    logr = mx.where(logp > logq, logp + mx.log1p(-mx.exp(delta)), mx.array(-float("inf"), dtype=mx.float32))
    alive = mx.any(logr > -float("inf"), axis=-1, keepdims=True)
    return mx.where(alive, logr, logp)


def _rs_residual_logits(z: mx.array, d: mx.array) -> mx.array:
    """Distribution to sample from after rejecting greedy draft d at this
    position: the target with d removed and renormalized (max(0, p - q) for a
    one-hot q), expressed as logits for categorical()."""
    vocab = z.shape[-1]
    z = z.astype(mx.float32)
    mask = mx.arange(vocab) == d.reshape(-1)[0:1].astype(mx.int32)
    out = mx.where(mask, mx.array(-float("inf"), dtype=mx.float32), z)
    alive = mx.any(out > -float("inf"), axis=-1, keepdims=True)
    return mx.where(alive, out, z)   # single-survivor target: d was that token, keep p


def _dist_group() -> Any | None:
    """Resolve the distributed group. Any failure is loud: a silently
    absent group would turn every cross-rank check into a no-op."""
    g = mx.distributed.init()
    return g if g is not None and g.size() > 1 else None


def _rank_consensus(grp: Any, local: list[int]) -> tuple[bool, list[int]]:
    """Every rank ALWAYS enters this collective with a fixed-size vector;
    returns (all ranks agree on every element, the summed vector)."""
    vec = mx.array(local, dtype=mx.int32)
    if grp is None:
        return True, local
    total = mx.distributed.all_sum(vec, group=grp)
    mx.eval(total)
    agree = bool(mx.array_equal(total, vec * grp.size()).item())
    return agree, [int(x) for x in total.tolist()]


def _off(c: Any) -> int:
    o = c.offset
    if isinstance(o, mx.array):
        return int(o.reshape(-1)[0].item())
    return int(o)


def _validate_pre_verify(state: _MTPState, y: mx.array, drafts: list[mx.array]) -> None:
    """Cross-rank agreement on the tokens about to enter the TP verify."""
    grp = state.group
    if grp is None:
        return
    toks = [y.reshape(-1)[0:1].astype(mx.int32)] + [d.reshape(-1)[0:1].astype(mx.int32) for d in drafts]
    vec = mx.concatenate(toks)
    total = mx.distributed.all_sum(vec, group=grp)
    mx.eval(total)
    if not mx.array_equal(total, vec * grp.size()).item():
        raise _MTPValidateError(
            f"[MTP][VALIDATE] pre-verify token divergence: local={vec.tolist()} "
            f"sum={total.tolist()} size={grp.size()}"
        )


def _validate_cycle(state: _MTPState, batch: Any, m: int, base: int | None = None) -> None:
    """Local checks produce an error code; every rank then joins ONE collective
    carrying [err, m, offsets, buffer, pending]; the decision (raise or not) is
    made from the summed vector, so all ranks raise together — a rank that
    raised before the collective would leave the others waiting forever."""
    offs = [[_cur(c) for c in cl.caches] for cl in batch.prompt_cache]
    err = 0
    detail = ""
    flat = {o for row in offs for o in row}
    if len(flat) != 1:
        bad = [(i, row) for i, row in enumerate(offs) if len(set(row)) != 1 or row[0] != offs[0][0]]
        err, detail = 1, f"slot/layer offset skew at layers {bad[:3]}"
    elif base is not None and offs[0][0] != base + 1 + m:
        err, detail = 2, f"committed delta {offs[0][0] - base} != 1+m={1 + m}"
    elif state.mtp_cache is not None:
        mo = [_cur(c) for c in state.mtp_cache.caches]
        if len(set(mo)) != 1:
            err, detail = 3, f"MTP slot skew {mo}"
    off00 = offs[0][0]
    mtp_off = _cur(state.mtp_cache[0]) if state.mtp_cache is not None else 0
    pend = int(batch._next_tokens.reshape(-1)[0].item()) if batch._next_tokens is not None else -1
    local = [err, m, off00, mtp_off, len(state.buffer), pend]
    agree, total = _rank_consensus(state.group, local)
    if total[0] != 0:
        raise _MTPValidateError(
            f"[MTP][VALIDATE] local check failed on some rank (err sum={total[0]}): {detail or 'remote'}"
        )
    if not agree:
        size = state.group.size() if state.group is not None else 1
        raise _MTPValidateError(
            f"[MTP][VALIDATE] cross-rank divergence: local={local} sum={total} size={size}"
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
        if isinstance(first, mx.array):
            return first[0] if first.ndim == 2 else first
    if isinstance(lp, list) and not lp:
        return mx.zeros((1,))  # constructor step: no logprobs yet (pin semantics)
    raise RuntimeError(f"[MTP] unexpected logprobs structure: {type(lp).__name__}")


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
        "spec_next": state.spec_next,
        "cycle_spec_entries": state.cycle_spec_entries,
    }


def _trim_to_exact(cache: Any, target: int) -> None:
    """Trim a cache to an exact position; any inconsistency is fatal."""
    current = _cur(cache)
    if current < target:
        raise RuntimeError(f"[MTP] cache behind target: current={current} target={target}")
    delta = current - target
    if delta:
        cache.trim(delta)
    restored = _cur(cache)
    if restored != target:
        raise RuntimeError(f"[MTP] cache restore mismatch: got={restored} expected={target}")


def _rollback(state: _MTPState, batch: Any, snap: dict[str, Any]) -> None:
    for cl, offs in zip(batch.prompt_cache, snap["main"]):
        for c, before in zip(cl.caches, offs):
            _trim_to_exact(c, before)
    if snap["mtp"] is not None and state.mtp_cache is not None:
        for c, before in zip(state.mtp_cache.caches, snap["mtp"]):
            _trim_to_exact(c, before)
    batch._next_tokens = snap["next_tokens"]
    batch._next_logprobs = snap["next_logprobs"]
    if batch.tokens:
        del batch.tokens[0][snap["tokens_len"]:]
    state.h_last = snap["h_last"]
    state.mtp_backlog = snap["backlog"]
    state.buffer = snap["buffer"]
    state.cycle_pack = snap["cycle_pack"]
    state.spec_next = snap["spec_next"]
    state.cycle_spec_entries = snap["cycle_spec_entries"]


def _history(state: _MTPState, batch: Any) -> mx.array:
    """The committed-token history as an int32 array, kept in lockstep with
    ``batch.tokens[0]`` by appending only the delta each cycle (the serial
    step rebuilds ``mx.array(self.tokens[e])`` from the Python list every
    token — O(n) per token, 150k ints at deep context; here it is O(k))."""
    toks = batch.tokens[0]
    n = len(toks)
    if state.hist_buf is None or state.hist_len > n:
        cap = max(n + 4096, 8192)
        buf = mx.zeros((cap,), dtype=mx.int32)
        if n:
            buf[:n] = mx.array(toks, dtype=mx.int32)
        state.hist_buf, state.hist_len = buf, n
        return buf[:n]
    if state.hist_len < n:
        if n > state.hist_buf.shape[0]:
            grown = mx.zeros((max(2 * state.hist_buf.shape[0], n + 4096),), dtype=mx.int32)
            grown[: state.hist_len] = state.hist_buf[: state.hist_len]
            state.hist_buf = grown
        state.hist_buf[state.hist_len:n] = mx.array(toks[state.hist_len:], dtype=mx.int32)
        state.hist_len = n
    return state.hist_buf[:n]


def _apply_processors_rows(
    batch: Any, logits: mx.array, fed: list[mx.array], state: _MTPState | None = None
) -> mx.array:
    """Apply the request's logits processors to each verify row exactly as
    the serial decode step would: row j sees the history the step after
    feeding fed[0..j-1] would see. Mirrors opt_batch_gen._patched_step, whose
    history is ``mx.array(self.tokens[e])`` — the committed tokens *before*
    the current input. Without processors the logits pass through untouched."""
    procs = getattr(batch, "logits_processors", None)
    procs = procs[0] if procs and procs[0] else None
    if not procs:
        return logits
    hist = _history(state, batch) if state is not None else mx.array(batch.tokens[0], dtype=mx.int32)
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
        state.finish(why="batch-merge")   # MTP is B==1 only; the request continues in the batch
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
        state.request_battle = bool(pol and not pol["logprobs"])
        state.request_sampling = bool(pol and not pol["greedy"])
        state.policy = pol
        state.last_battle = state.request_battle
        if not state.request_battle:
            why = "unknown sampler policy" if pol is None else "logprobs requested"
            _warn_once(
                state.logger, f"nobattle:{why}",
                f"[MTP] {why} under mode=on: running shadow for it",
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
        _adopt_prefill(state, batch)
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
    if state.prompt_ctx:
        state.gen_pairs.extend(pairs)          # committed pairs, in cache order
    if state.spec_next is not None and not state.mtp_backlog and not pairs[:-1]:
        # M1.5-lite hit: the previous cycle's verify train already drafted this
        # cycle's step 1 (and ingested its pairs); the pending token is that
        # cycle's pre-drawn bonus, so the speculative pairs are exactly right.
        d1, h_mtp, zq1 = state.spec_next
        state.spec_next = None
    else:
        state.spec_next = None
        res = state.draft_multi(pairs)
        d1, h_mtp = res[0], res[1]
        zq1 = res[2] if len(res) > 2 else None
    zq_rows = [zq1]
    drafts = [d1]
    chain_before = _cur(state.mtp_cache[0]) if state.mtp_cache is not None else 0
    for _ in range(1, k):
        # chained proposal: the MTP block's own post-norm output stands in
        # for the (not yet computed) main-model hidden of the previous draft.
        # GLM-5 trains the head with 3 parameter-shared steps, so the chain
        # is in-distribution up to k=3.
        res = state.draft_multi([(h_mtp, drafts[-1])])
        d_next, h_mtp = res[0], res[1]
        zq_rows.append(res[2] if len(res) > 2 else None)
        drafts.append(d_next)

    # The chained MTP entries (steps 2..k) were computed from the block's own
    # hidden; drop them now (host-side bookkeeping only — the already-built
    # chain graph keeps its own array versions) so the speculative next-cycle
    # pairs land at the right positions.
    if k >= 2 and state.mtp_cache is not None:
        chain_entries = _cur(state.mtp_cache[0]) - chain_before
        for c in state.mtp_cache.caches:
            _trim_to_exact(c, _cur(c) - chain_entries)
    base_off = _cur(batch.prompt_cache[0][0]) if state.validate else None
    if state.validate:
        _validate_pre_verify(state, y, drafts)
    pad = state.verify_pad
    verify_in = mx.concatenate(
        [y.reshape(-1)[0:1]] + [x.astype(y.dtype).reshape(-1)[0:1] for x in drafts]
        + [drafts[-1].astype(y.dtype).reshape(-1)[0:1]] * pad
    ).reshape(1, k + 1 + pad)
    from exo.worker.engines.mlx.patches.glm52_indexshare import mtp_verify_context

    with mtp_verify_context():
        logits2 = batch.model(verify_in, cache=batch.prompt_cache)
    hv = state.store.pop("h", None)
    if pad:
        # measurement rows: drop them from logits/hidden, and from every cache
        logits2 = logits2[:, : k + 1, :]
        if hv is not None:
            hv = hv[:, : k + 1, :]
        for c in batch.prompt_cache:
            c.trim(pad)
    logits2 = _apply_processors_rows(batch, logits2, [y] + drafts, state)
    logits2 = logits2.astype(mx.float32)       # normalize in float32, not bf16
    lp2 = logits2 - mx.logsumexp(logits2, axis=-1, keepdims=True)
    t_all = mx.argmax(lp2, axis=-1)                       # (1, k+1)
    t1 = t_all[:, 0]
    zs = None
    if state.request_sampling:
        # M2: speculative sampling — accept d_j w.p. p_j(d_j) under the exact
        # target distribution (temperature/top_p/min_p/top_k of the request);
        # the uniforms are drawn in program order from the replicated RNG.
        z_rows = _target_logits(lp2[0], state.policy)          # (k+1, V) in one pass
        zs = [z_rows[j:j + 1] for j in range(k + 1)]
        u = mx.random.uniform(shape=(k,))
        full_q = state.proposal == "sample" and all(z is not None for z in zq_rows)
        if full_q:
            accs = _rs_accepts_q(z_rows, zq_rows, drafts, u)
        else:
            accs = _rs_accepts_vec(z_rows, drafts, u)
        if state.cf:
            state.cf_update(zs[0], zq_rows[0], drafts[0])
    else:
        accs = []
        prev = None
        for i in range(k):
            hit = mx.equal(drafts[i].astype(mx.int64), t_all[:, i].astype(mx.int64))
            prev = hit if prev is None else mx.logical_and(prev, hit)
            accs.append(prev)
    acc1 = accs[0]

    # ---- M1.5-lite: draft the NEXT cycle's step 1 inside this verify's train,
    # assuming full acceptance. Its pairs are (hv_i, d_i) for i<k plus
    # (hv_k, bonus candidate); the bonus is drawn now (lazily) so both branches
    # use the same program-order RNG. On a reject the extra MTP entries are
    # trimmed and the draft is recomputed as before. The verify is never
    # speculated. ---------------------------------------------------------
    spec = None
    spec_ok = (
        state.spec_draft and hv is not None and hv.shape[1] >= k + 1
        and state.mtp_cache is not None
    )
    if spec_ok:
        bonus_c = (mx.random.categorical(zs[k]).reshape(1) if zs is not None
                   else t_all[:, k].reshape(1)).astype(y.dtype)
        pairs_spec = [(hv[:, i:i + 1, :], drafts[i]) for i in range(k)] + [(hv[:, k:k + 1, :], bonus_c)]
        spec_before = _cur(state.mtp_cache[0])
        res_s = state.draft_multi(pairs_spec)
        spec_entries = _cur(state.mtp_cache[0]) - spec_before   # what actually landed
        spec = (res_s[0], res_s[1], res_s[2] if len(res_s) > 2 else None, bonus_c, spec_entries)

    if state.prof:
        _t1 = time.perf_counter()
    state.in_resolve = True
    try:
        if spec is not None:
            # Launch verify + speculative draft together: the host traversal
            # of the draft graph then overlaps the GPU verify instead of
            # landing in the post phase; the wait below returns as soon as
            # the accept flags are ready while the draft keeps running.
            mx.async_eval(
                *accs, y, spec[0], spec[1], *([spec[2]] if spec[2] is not None else []),
                *state.mtp_cache_arrays(),
            )
        mx.eval(*accs, y)
    finally:
        state.in_resolve = False
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

    if k - m:
        for c in batch.prompt_cache:
            c.trim(k - m)

    spec_arrays: list[mx.array] = []
    if spec is not None:
        d1n, h_mtpn, zqn, bonus_c, spec_entries = spec
        if m == k:
            # full accept: the speculative pairs are the committed ones, the
            # pending token is the pre-drawn bonus, the next draft is ready.
            batch._next_tokens = bonus_c
            state.spec_next = (d1n, h_mtpn, zqn)
            state.cycle_spec_entries = spec_entries
            state.mtp_backlog = []
            if state.prompt_ctx:
                state.gen_pairs.extend(pairs_spec)
            spec_arrays = [d1n, h_mtpn] + ([zqn] if zqn is not None else [])
        else:
            # reject at m: keep the m accepted pairs (already in the cache),
            # drop the rest of the speculative entries and the draft.
            # entries are in pair order: the first m are the accepted pairs
            keep = min(m, spec_entries)
            for c in state.mtp_cache.caches:
                _trim_to_exact(c, _cur(c) - (spec_entries - keep))
            state.spec_next = None
            state.cycle_spec_entries = keep
            # pairs that did not land in the cache (stubbed draft) go through the backlog
            state.mtp_backlog = [] if spec_entries else [(hv[:, i:i + 1, :], drafts[i]) for i in range(m)]
            if state.prompt_ctx:
                state.gen_pairs.extend(pairs_spec[:m])
            if zs is not None:
                pend = (mx.random.categorical(_rs_residual_logits_q(zs[m], zq_rows[m])) if full_q
                        else mx.random.categorical(_rs_residual_logits(zs[m], drafts[m])))
                batch._next_tokens = pend.reshape(1).astype(y.dtype)
            else:
                batch._next_tokens = t_all[:, m].astype(y.dtype)
    else:
        state.spec_next = None
        state.cycle_spec_entries = 0
        if zs is not None:
            # pending token: bonus from the target at position m when every
            # draft was accepted, else from the residual of the rejecting position.
            if m == k:
                pend = mx.random.categorical(zs[k])
            elif full_q:
                pend = mx.random.categorical(_rs_residual_logits_q(zs[m], zq_rows[m]))
            else:
                pend = mx.random.categorical(_rs_residual_logits(zs[m], drafts[m]))
            batch._next_tokens = pend.reshape(1).astype(y.dtype)
        else:
            batch._next_tokens = t_all[:, m].astype(y.dtype)
        state.mtp_backlog = [(hv[:, i:i + 1, :], drafts[i]) for i in range(m)]
    batch._next_logprobs = lp2[:, m]
    state.h_last = hv[:, m:m + 1, :]
    state.buffer = [(drafts[i], lp2[:, i]) for i in range(m)]
    state.cycle_pack = (t_all, lp2, hv, drafts, m) if m else None
    state.flush_pack = None
    mx.async_eval(
        batch._next_tokens, batch._next_logprobs, *drafts, *spec_arrays,
        *_advance_chain_arrays(batch.prompt_cache),
        *state.mtp_cache_arrays(),
        *([state.hist_buf] if state.hist_buf is not None else []),
        *([state.cf_acc] if state.cf_acc is not None else []),
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
    if state.prompt_ctx and state.prefill_cycles > 0 and state.cycles >= state.prefill_cycles:
        _drop_prompt_context(state)
    if state.validate:
        _validate_cycle(state, batch, m, base=base_off)
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
    for cl in batch.prompt_cache:
        for c in cl.caches:
            _trim_to_exact(c, _cur(c) - r)
    # The buffered drafts were already accepted — under sampling they are
    # valid samples of the target. The first of them simply becomes the
    # pending token; no new draw (a residual re-sample would exclude it and
    # skew the distribution whenever a merge lands after an accept).
    batch._next_tokens = drafts[kept].reshape(1).astype(batch._next_tokens.dtype)
    if state.cycle_spec_entries:
        # speculative MTP entries beyond the kept pairs are dropped; the kept
        # pairs stay in the cache (no backlog replay for them)
        if state.mtp_cache is not None:
            for c in state.mtp_cache.caches:
                _trim_to_exact(c, _cur(c) - max(state.cycle_spec_entries - kept, 0))
        state.spec_next = None
        state.cycle_spec_entries = 0
        spec_kept = True
    else:
        spec_kept = False
    batch._next_logprobs = lp2[:, kept]
    state.h_last = hv[:, kept:kept + 1, :]
    # with the speculative draft the kept pairs are already in the MTP cache
    state.mtp_backlog = [] if spec_kept else [(hv[:, i:i + 1, :], drafts[i]) for i in range(kept)]
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


def _install_hooks(logger: Any) -> bool:
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return True
    from mlx_lm.generate import GenerationBatch

    prev_step = GenerationBatch._step
    if getattr(prev_step, "_exo_glm52_mtp_wrapper", False):
        _HOOKS_INSTALLED = True
        return True
    if getattr(prev_step, "__name__", "") not in ("_patched_step", "_step"):
        logger.warning(
            f"[MTP] unexpected base GenerationBatch._step "
            f"({getattr(prev_step, '__name__', '?')}); MTP not installed"
        )
        return False

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
            if state.prev_ran or state.in_resolve:
                # the real step already executed, or the failure came from the
                # target verify / a collective: never run a second target
                # forward on top of that — surface the failure.
                state.in_resolve = False
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
            for cl in self.prompt_cache:
                for c in cl.caches:
                    _trim_to_exact(c, _cur(c) - len(state.buffer))
            state.out_tokens = max(state.out_tokens - len(state.buffer), 0)
            state.accepted = max(state.accepted - len(state.buffer), 0)
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
    return True


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_sidecar(
    weights_path: Path, layer_idx: int, *, quant: dict[str, Any] | None, logger: Any,
    grp: Any = None,
) -> tuple[dict[str, mx.array] | None, str]:
    """Rank-safe wrapper: run the local checks, then ALWAYS join one collective
    with [ok, digest words]; a rank that failed locally still participates (with
    zeros), so no rank can be left waiting in all_sum. Every rank then makes the
    same decision."""
    raw, reason, digest = _validate_sidecar_local(weights_path, layer_idx, quant=quant, logger=logger)
    ok = raw is not None
    words = [int(digest[i:i + 8], 16) & 0x7FFFFFFF for i in range(0, 64, 8)] if ok else [0] * 8
    agree, total = _rank_consensus(grp, [1 if ok else 0] + words)
    size = grp.size() if grp is not None else 1
    if total[0] != size:
        return None, (reason if not ok else f"side-car rejected on {size - total[0]} rank(s)")
    if not agree:
        return None, "side-car digest differs across ranks"
    return raw, "ok"


def _validate_sidecar_local(
    weights_path: Path, layer_idx: int, *, quant: dict[str, Any] | None, logger: Any
) -> tuple[dict[str, mx.array] | None, str, str]:
    """Fail-closed side-car validation: manifest present, byte size and
    SHA-256 match, policy fields agree with the model, required tensor
    families present, and — under TP — the digest identical on every rank
    (different side-cars would feed different draft tokens into the TP
    verify before VALIDATE could see it). Returns the loaded tensors."""
    if not weights_path.is_file():
        return None, f"{weights_path} not found", ""
    manifest_path = weights_path.with_name("mtp.manifest.json")
    if not manifest_path.is_file():
        return None, f"{manifest_path.name} missing next to side-car", ""
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:  # noqa: BLE001
        return None, f"cannot read manifest: {e!r}", ""
    size = weights_path.stat().st_size
    if int(manifest.get("bytes", -1)) != size:
        return None, f"side-car size {size} != manifest {manifest.get('bytes')}", ""
    if int(manifest.get("layer", layer_idx)) != layer_idx:
        return None, f"manifest layer {manifest.get('layer')} != {layer_idx}", ""
    q = quant or {}
    pol = ((manifest.get("policy", {}) or {}).get("quant", {}) or {})
    for key, want in (("bits", int(q.get("bits", 8))), ("group_size", int(q.get("group_size", 64)))):
        have = pol.get(key)
        if have is not None and int(have) != want:
            return None, f"manifest policy {key}={have} != model {want}", ""
    t0 = time.perf_counter()
    digest = _sha256_file(weights_path)
    if digest != str(manifest.get("sha256", "")):
        return None, "side-car SHA-256 does not match manifest", ""
    logger.info(f"[MTP] side-car sha256 verified ({digest[:16]}…) in {time.perf_counter() - t0:.1f}s")
    try:
        raw = mx.load(str(weights_path))
    except Exception as e:  # noqa: BLE001 — fail-closed with reason
        return None, f"cannot read {weights_path}: {e!r}", ""
    renamed = {_rename_sidecar_key(k, layer_idx) for k in raw.keys()}
    missing = [f for f in _REQUIRED_FAMILIES if f not in renamed]
    if missing:
        return None, f"sidecar missing families: {missing[:4]}", ""
    return raw, "ok", digest


def mtp_prefill_begin(model: Any, start_offset: int, n_tokens: int) -> None:
    """Called by exo's prefill() before the chunk loop. Arms ingestion of the
    tokens this prefill will actually compute: the whole prompt for a cold
    request, or the uncached suffix after a prefix-cache hit (multi-turn
    chat: the latest turn). The MTP cache is position-relative, so a suffix
    starting at `start_offset` is ingested from MTP position 0."""
    state = getattr(model, "_exo_glm52_mtp_state", None)
    if state is None or state.mode != "on" or not state.prefill_enabled:
        return
    if n_tokens < 2:
        state.pending = None
        return
    from mlx_lm.models.cache import CacheList, KVCache

    # pair i = (h_i, t_{i+1}); the last prompt pair is i = n_tokens-2 (the
    # carried one is completed at adoption). A window keeps only the last W.
    start = 0 if state.prefill_window <= 0 else max(0, n_tokens - 1 - state.prefill_window)
    _park_pending(state)
    state.pending = {
        "cache": CacheList(KVCache(), KVCache()), "carry_h": None,
        "n": 0, "ingested": 0, "toks": [], "t0": time.perf_counter(),
        "start": start, "base": int(start_offset),
    }


def _park_pending(state: _MTPState) -> None:
    """Prefills can interleave (a second request arrives while the first is
    still being prefilled, or before it starts generating). Completed packages
    are parked and matched to their request at the seed step instead of one
    package overwriting the other."""
    pend = state.pending
    state.pending = None
    if pend is not None and pend["carry_h"] is not None and pend["n"] > 0:
        state.ready.append(pend)
        if len(state.ready) > 4:
            state.ready.pop(0)


def mtp_prefill_chunk(model: Any, chunk_tokens: list[int]) -> None:
    """Called after each prompt chunk forward: the capture store holds the
    chunk's hiddens. Ingests (h_i, t_{i+1}) pairs, carrying the last hidden
    to pair with the next chunk's first token."""
    state = getattr(model, "_exo_glm52_mtp_state", None)
    if state is None or state.pending is None or not chunk_tokens:
        return
    pend = state.pending
    try:
        h = state.store.pop("h", None)
        L = len(chunk_tokens)
        if h is None or h.shape[0] != 1 or h.shape[1] != L:
            raise RuntimeError(f"prefill capture mismatch: h={None if h is None else h.shape} L={L}")
        toks = mx.array(chunk_tokens, dtype=mx.uint32)
        s0 = pend["n"]                                   # position of chunk_tokens[0]
        if pend["carry_h"] is not None:
            h_seq = mx.concatenate([pend["carry_h"], h[:, :-1, :]], axis=1)   # h_{s-1}..h_{e-2}
            tok_seq = toks.reshape(1, L)                                        # t_s..t_{e-1}
            first_pair = s0 - 1
        else:
            h_seq = h[:, :-1, :]                                                # h_s..h_{e-2}
            tok_seq = toks[1:].reshape(1, L - 1)                                # t_{s+1}..t_{e-1}
            first_pair = s0
        # window: drop pairs below the start index (they are before the last W)
        skip = max(0, pend["start"] - first_pair)
        if skip:
            h_seq, tok_seq = h_seq[:, skip:, :], tok_seq[:, skip:]
        n_pairs = tok_seq.shape[1]
        for a in range(0, n_pairs, _PREFILL_SUBCHUNK):
            b = min(a + _PREFILL_SUBCHUNK, n_pairs)
            state.ingest(pend["cache"], h_seq[:, a:b, :], tok_seq[:, a:b])
            # Keep the graph shallow without a host-side sync inside exo's
            # prefill loop (the chunk's TP all-reduce is in flight on the
            # comm stream under --fast-sync); materialization happens at the
            # seed step.
            mx.async_eval(*[arr for c in pend["cache"].caches for arr in (c.keys, c.values) if arr is not None])
        pend["carry_h"] = h[:, -1:, :]
        pend["n"] += L
        pend["ingested"] += n_pairs
        pend["toks"].extend(int(t) for t in chunk_tokens)
    except Exception:
        state.logger.opt(exception=True).warning("[MTP] prompt ingest failed; head starts cold")
        state.pending = None


def _drop_prompt_context(state: _MTPState) -> None:
    """The head drafts better once it has its own context (measured: prompt
    context lifts the first window but lowers steady state). Rebuild the MTP
    cache from the committed generated pairs only. Never raises: on failure
    the prompt-backed cache simply stays."""
    from mlx_lm.models.cache import CacheList, KVCache

    try:
        pairs = state.gen_pairs
        fresh = CacheList(KVCache(), KVCache())
        if pairs:
            h_seq = mx.concatenate([h for h, _ in pairs], axis=1)
            t_seq = mx.concatenate([t.reshape(1, 1).astype(mx.uint32) for _, t in pairs], axis=1)
            for a in range(0, len(pairs), _PREFILL_SUBCHUNK):
                b = min(a + _PREFILL_SUBCHUNK, len(pairs))
                state.ingest(fresh, h_seq[:, a:b, :], t_seq[:, a:b])
        mx.async_eval(*[arr for c in fresh.caches for arr in (c.keys, c.values) if arr is not None])
        # Do not free the prompt-backed cache mid-request: an async train and
        # the TP collectives are in flight; keep it referenced until finish().
        if state.mtp_cache is not None:
            state.retired.append(state.mtp_cache)
        state.mtp_cache = fresh
        state.logger.info(
            f"[MTP] uid={state.uid} prompt context dropped after {state.cycles} cycles "
            f"(re-ingested {len(pairs)} generated pairs)"
        )
    except Exception:
        state.logger.opt(exception=True).warning("[MTP] prompt context drop failed; keeping it")
    finally:
        state.prompt_ctx = False
        state.gen_pairs = []


def _adopt_prefill(state: _MTPState, batch: Any) -> None:
    """At request start: attach the prompt-prefilled MTP cache if it belongs
    to this request's prompt, ingesting the final carried pair."""
    _park_pending(state)
    t = list(batch.tokens[0])
    try:
        main_pos = _cur(batch.prompt_cache[0][0])
    except Exception:
        main_pos = None
    # pick the package whose (base + n) matches the main cache position and
    # whose ingested tokens end with the committed tail
    pend = None
    for cand in reversed(state.ready):
        base_n = cand.get("base", 0) + cand["n"]
        if main_pos is not None and base_n != main_pos:
            continue
        toks_c, n_c = cand["toks"], cand["n"]
        if (len(t) == n_c + 1 and t[:n_c] == toks_c) or (0 < len(t) <= n_c and toks_c[n_c - len(t):] == t):
            pend = cand
            break
    if pend is None:
        if state.ready:
            state.logger.info(
                f"[MTP] prompt ingest discarded: no matching package "
                f"(tokens={len(t)} main_pos={main_pos} parked={[(c.get('base', 0), c['n']) for c in state.ready]})"
            )
        return
    state.ready.remove(pend)
    n = pend["n"]
    toks = pend["toks"]
    # Structural match. exo prefills prompt[:-1] itself and inserts only the
    # last two tokens into the generator, so at the seed step tokens[0] is a
    # short tail (prod) or the full prefix (pin harness); either way it must
    # be a suffix of what we ingested, the main cache must sit at position n
    # (prompt[:n] committed), and the last prompt token is either listed or
    # pending as _next_tokens.
    try:
        main_pos = _cur(batch.prompt_cache[0][0])
    except Exception:
        main_pos = None
    if len(t) == n + 1:
        tail_ok, last_tok = (t[:n] == toks), int(t[n])
    elif 0 < len(t) <= n and batch._next_tokens is not None:
        tail_ok = toks[n - len(t):] == t
        last_tok = int(batch._next_tokens.reshape(-1)[0].item())
    else:
        tail_ok, last_tok = False, None
    expect_pos = pend.get("base", 0) + n
    if last_tok is None or not tail_ok or (main_pos is not None and main_pos != expect_pos):
        state.logger.info(
            f"[MTP] prompt ingest discarded: prompt mismatch "
            f"(tokens={len(t)} n={n} base={pend.get('base', 0)} main_pos={main_pos})"
        )
        return
    try:
        state.ingest(
            pend["cache"], pend["carry_h"],
            mx.array([[last_tok]], dtype=mx.uint32),
        )
        state.mtp_cache = pend["cache"]
        state.prompt_ctx = True
        state.gen_pairs = []
        state.logger.info(
            f"[MTP] uid={state.uid} prompt ingested: {pend['ingested'] + 1} pairs "
            f"(base={pend.get('base', 0)} mtp_offset={_cur(state.mtp_cache[0])}) "
            f"in {time.perf_counter() - pend['t0']:.1f}s"
        )
    except Exception:
        state.logger.opt(exception=True).warning("[MTP] prompt ingest adopt failed; head starts cold")


def _reset_terminal(state: _MTPState) -> None:
    """close(): drop everything, including parked/in-flight prompt packages
    and retired caches (reset() keeps parked packages for their requests)."""
    state.finish(why="close")
    state.pending = None
    state.ready = []
    state.retired = []
    state.gen_pairs = []
    state.spec_next = None


def finalize_glm52_mtp_request(model: Any, uid: int | None, *, reason: str) -> None:
    """Terminal teardown of the MTP state for a request that exo removes from
    the generator (custom text stop, cancel, close). The generator drops the
    request's cache wholesale, so only the state needs finalizing; uid=None
    finalizes whatever request is active."""
    state = getattr(model, "_exo_glm52_mtp_state", None)
    if state is None:
        return
    if uid is None:               # close(): terminal
        _reset_terminal(state)
        return
    if state.uid is None:
        return
    if state.uid == uid:
        if state.buffer:
            state.out_tokens = max(state.out_tokens - len(state.buffer), 0)
            state.accepted = max(state.accepted - len(state.buffer), 0)
        state.logger.info(f"[MTP] uid={state.uid} finalized ({reason})")
        state.finish()


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
    try:
        grp = _dist_group()
    except Exception:
        logger.opt(exception=True).warning("[MTP] distributed group unavailable; patch not applied")
        return model
    raw, reason = _validate_sidecar(
        weights_path, layer_idx, quant=config.get("quantization"), logger=logger, grp=grp
    )
    if raw is None:
        logger.warning(f"[MTP] {reason}; patch not applied")
        return model

    concat = _env_choice(_ENV_CONCAT, "eh", {"eh", "he"}, logger)
    validate = _env_int(_ENV_VALIDATE, 0, 0, 1, logger) == 1
    trace_n = _env_int(_ENV_TRACE, 0, 0, 4096, logger)
    prof = _env_int(_ENV_PROF, 0, 0, 1, logger) == 1
    draft_k = _env_int(_ENV_DRAFT_K, 1, 1, 3, logger)
    # Measured on GLM-5.3-8bit-idxbf16 (kv=49k, code+docs corpus, k=1, warm
    # slots): target hidden PRE-final-norm a1=0.816 vs POST 0.741. vLLM's
    # post-norm convention loses on this head/quant; keep 'post' as the A/B
    # knob. (The chained-step recycle stays post-norm — measured separately.)
    hidden_mode = _env_choice(_ENV_HIDDEN, "pre", {"post", "pre"}, logger)
    recycle_mode = _env_choice(_ENV_RECYCLE, "post", {"post", "pre"}, logger)
    prefill_enabled = _env_int(_ENV_PREFILL, 1, 0, 1, logger) == 1
    prefill_window = _env_int(_ENV_PREFILL_WINDOW, 2048, 0, 1 << 20, logger)
    prefill_cycles = _env_int(_ENV_PREFILL_CYCLES, 64, 0, 1 << 20, logger)
    proposal = _env_choice(_ENV_PROPOSAL, "sample", {"argmax", "sample"}, logger)
    cf = _env_int(_ENV_CF, 0, 0, 1, logger) == 1
    spec_draft = _env_int(_ENV_SPEC_DRAFT, 0, 0, 1, logger) == 1
    verify_pad = _env_int(_ENV_VERIFY_PAD, 0, 0, 2, logger)
    q_temp_raw = os.environ.get(_ENV_Q_TEMP, "").strip()
    q_temperature = float(q_temp_raw) if q_temp_raw else None

    mtp = None
    try:
        mtp = load_mtp_module(
            args, weights_path, layer_idx=layer_idx,
            quant=config.get("quantization"), logger=logger, raw=raw,
        )
        rope_fixed = None
    except Exception:
        logger.opt(exception=True).warning("[MTP] side-load failed")
    # Second rank-safe collective: MTP is installed only if EVERY rank built
    # the module — a rank decoding L=1 while the others verify L=k+1 would
    # break the TP collectives.
    n_params = sum(v.size for _, v in tree_flatten(mtp.parameters())) if mtp is not None else 0
    agree, total = _rank_consensus(grp, [1 if mtp is not None else 0, int(n_params & 0x7FFFFFFF)])
    size = grp.size() if grp is not None else 1
    if total[0] != size or not agree:
        logger.warning(f"[MTP] module not ready on all ranks (ready={total[0]}/{size}); patch not applied")
        return model

    # The MTP block carries its own DSA indexer; give it the same half-split
    # RoPE decision the main full-indexer layers get (matters once the MTP
    # cache exceeds index_topk, i.e. after ~2048 generated tokens). Pure
    # config-driven host work — identical on every rank, no collective needed.
    from exo.worker.engines.mlx.patches.glm52_indexshare import (
        annotate_mtp_attention,
        apply_mtp_indexer_rope_fix,
    )

    rope_fixed = apply_mtp_indexer_rope_fix(mtp.block.self_attn, config, logger=logger)
    fast_attn = annotate_mtp_attention(mtp.block.self_attn, list(layers), layer_idx)

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
    state.recycle = recycle_mode
    state.prefill_enabled = prefill_enabled
    state.prefill_window = prefill_window
    state.prefill_cycles = prefill_cycles
    state.proposal = proposal
    state.cf = cf
    state.spec_draft = spec_draft
    state.verify_pad = verify_pad
    state.q_temperature = q_temperature
    state.group = grp
    if not _install_hooks(logger):
        return model
    model._exo_glm52_mtp_state = state
    logger.info(
        f"[MTP] enabled mode={mode} k={draft_k} hidden={hidden_mode} recycle={recycle_mode} "
        f"prompt_prefill={int(prefill_enabled)} prefill_window={prefill_window} "
        f"prefill_cycles={prefill_cycles} proposal={proposal} cf={int(cf)} spec_draft={int(spec_draft)} "
        f"verify_pad={verify_pad} q_temp={q_temperature if q_temperature is not None else 'target'} "
        f"fast_attn={int(fast_attn)} concat={concat} "
        f"validate={int(validate)} mtp_indexer_rope_fixed={int(rope_fixed)} "
        f"weights={weights_path.name} layer_idx={layer_idx}"
    )
    return model
