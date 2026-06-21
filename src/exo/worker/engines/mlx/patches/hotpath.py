"""EXO hot-path decode helpers for the common MLX decode path.

This module installs an alternative ``GenerationBatch._step`` only when one of
``EXO_HOTPATH*`` flags is enabled. With all flags OFF, the stock
``opt_batch_gen.apply_batch_gen_patch`` step remains active. The always-on PR0
pieces (card schema, guarded imports, GLM drain helper) live in the shell
patches, not in this module.

Implemented here:
  * profiling: wall-clock JSONL plus optional invasive attribution fences;
  * WS-B: split logsumexp requirements:
      - need_normalized_logits: sampler/top-p or API logprobs require logprobs;
      - need_response_logprobs: response/API path requires per-row logprobs;
  * WS-C1: group-by sampler signature, one sampler call per identical group;
  * WS-D: safe placeholder. The flag warns once and remains eager until D0.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, cast

import mlx.core as mx
from mlx_lm.generate import GenerationBatch

from exo.worker.engines.mlx.patches.opt_batch_gen import (
    _PRECOMPUTE_TOP_K,
    BatchTopKLogprobs,
    _drain_prompt_cache_if_needed,
    _get_buffer,
)

try:
    from exo.worker.runner.bootstrap import logger as _logger
except Exception:  # pragma: no cover - logging is best-effort
    import logging

    _logger = logging.getLogger("exo.hotpath")

_TRUTHY = {"1", "true", "yes", "y", "on", "wall", "jsonl", "attribution"}
_PROFILE_LOCK = threading.Lock()
_COMPILE_WARNED = False


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _profile_mode() -> str | None:
    raw = os.environ.get("EXO_HOTPATH_PROFILE", "").strip().lower()
    if raw in {"", "0", "false", "no", "n", "off"}:
        return None
    mode = os.environ.get("EXO_HOTPATH_PROFILE_MODE", "").strip().lower()
    if raw == "attribution" or mode == "attribution":
        return "attribution"
    return "wall"


def hotpath_enabled() -> bool:
    """True when any runtime hot-path flag requires replacing _step."""
    return any(
        _flag(name)
        for name in (
            "EXO_HOTPATH",
            "EXO_HOTPATH_PROFILE",
            "EXO_HOTPATH_GATE_LOGSUMEXP",
            "EXO_HOTPATH_GROUP_SAMPLER",
            "EXO_HOTPATH_COMPILE",
        )
    )


@dataclass(frozen=True)
class SamplerSignature:
    """Machine-readable sampler parameters for WS-C1 grouping.

    This covers only parameters consumed by ``mlx_lm.sample_utils.make_sampler``.
    Repetition/presence/frequency penalties are logits processors and are
    profiled separately; they intentionally do not belong in this signature.
    """

    temperature: float
    top_p: float
    top_k: int
    min_p: float


def attach_sampler_signature(sampler: Any, task_params: Any) -> None:
    """Attach a grouping signature to a ``make_sampler`` closure.

    ``make_sampler`` returns a bare Python closure. Two closures with identical
    params are distinct objects, so grouping by identity is wrong. Python
    function objects accept attributes, but this is defensive in case a future
    sampler object does not.
    """
    signature = SamplerSignature(
        temperature=float(task_params.temperature if task_params.temperature is not None else 0.7),
        top_p=float(task_params.top_p if task_params.top_p is not None else 1.0),
        top_k=int(task_params.top_k if task_params.top_k is not None else 0),
        min_p=float(task_params.min_p if task_params.min_p is not None else 0.05),
    )
    try:
        sampler._exo_signature = signature
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        pass


def _now() -> float:
    return time.perf_counter()


def _ms(start: float, end: float | None = None) -> float:
    return ((end if end is not None else _now()) - start) * 1000.0


def _profile_write(row: dict[str, Any]) -> None:
    path = os.environ.get("EXO_HOTPATH_PROFILE_PATH", "/tmp/exo_hotpath_profile.jsonl")
    try:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with _PROFILE_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:  # pragma: no cover - profiling must never break decode
        pass


def _grouped_sample(
    samplers: list[Callable[[mx.array], mx.array]] | None,
    fallback_sampler: Callable[[mx.array], mx.array],
    sampler_input: mx.array,
    n: int,
) -> tuple[mx.array | None, bool, int]:
    """Sample one group at a time for rows with identical signatures.

    Returns ``(sampled, grouped_used, call_count)``. ``sampled is None`` means
    there is no grouping benefit, so caller should use the stock per-row path.

    RNG caveat: one categorical call on ``[G, V]`` consumes random state
    differently from ``G`` calls on ``[1, V]``. Greedy parity is exact;
    stochastic fixed-seed token parity is not an acceptance criterion.
    """
    if n <= 1:
        return None, False, 0

    groups: dict[Any, tuple[list[int], Callable[[mx.array], mx.array]]] = {}
    order: list[Any] = []
    for row in range(n):
        sampler = samplers[row] if samplers is not None and row < len(samplers) else None
        if sampler is None:
            key: Any = ("__fallback__",)
            fn = fallback_sampler
        else:
            signature = getattr(sampler, "_exo_signature", None)
            if signature is None:
                # Unknown sampler: do not group across closures accidentally.
                key = ("__id__", id(sampler))
            else:
                key = signature
            fn = sampler
        if key not in groups:
            groups[key] = ([], fn)
            order.append(key)
        groups[key][0].append(row)

    if all(len(rows) == 1 for rows, _ in groups.values()):
        return None, False, 0

    sampled_rows: list[mx.array | None] = [None] * n
    calls = 0
    for key in order:
        rows, fn = groups[key]
        if len(rows) == 1:
            sub = sampler_input[rows[0] : rows[0] + 1]
        else:
            # mx.take also works, but concatenate preserves the simple [1,V]
            # slicing path and is easy to validate against stock per-row calls.
            sub = mx.concatenate([sampler_input[row : row + 1] for row in rows], axis=0)
        sampled_sub = fn(sub)
        calls += 1
        for j, row in enumerate(rows):
            sampled_rows[row] = sampled_sub[j : j + 1]

    return mx.concatenate(cast(list[mx.array], sampled_rows), axis=0), True, calls


def _maybe_compiled_forward(self: GenerationBatch, inputs: mx.array) -> mx.array:
    """WS-D placeholder: warn once and use eager forward.

    A correct implementation must thread KV cache state and Python metadata
    (offset/write_index/quant fields) through ``mx.compile`` inputs/outputs and
    must prove JACCL collective correctness. Closure-capturing
    ``self.prompt_cache`` would be unsafe.
    """
    global _COMPILE_WARNED
    if not _COMPILE_WARNED:
        _logger.warning(
            "[EXO][hotpath] EXO_HOTPATH_COMPILE is set, but WS-D is only a "
            "D0 spike placeholder. Falling through to eager forward."
        )
        _COMPILE_WARNED = True
    return self.model(inputs[:, None], cache=self.prompt_cache)


def _sample_stock_per_row(
    self: GenerationBatch,
    sampler_input: mx.array,
) -> tuple[mx.array, int]:
    if self.samplers is not None and any(self.samplers):
        all_samples: list[mx.array] = []
        for row in range(len(self.uids)):
            sample_sampler = self.samplers[row] or self.fallback_sampler
            all_samples.append(sample_sampler(sampler_input[row : row + 1]))
        return mx.concatenate(all_samples, axis=0), len(all_samples)
    return self.fallback_sampler(sampler_input), 1


def _hotpath_step(self: GenerationBatch) -> tuple[list[int], list[mx.array]]:
    profile_mode = _profile_mode()
    attribution = profile_mode == "attribution"
    gate_on = _flag("EXO_HOTPATH_GATE_LOGSUMEXP")
    group_on = _flag("EXO_HOTPATH_GROUP_SAMPLER")
    compile_on = _flag("EXO_HOTPATH_COMPILE")
    timings: dict[str, float] = {}
    step_start = _now()

    self._current_tokens = self._next_tokens
    self._current_logprobs = self._next_logprobs
    inputs = self._current_tokens

    buf = _get_buffer(self)
    buf.ready = buf.pending
    buf.pending = BatchTopKLogprobs()

    n = len(self.uids)

    # With WS-B off, keep stock behaviour: always normalize and retain response
    # logprobs. With WS-B on, consume split requirements computed by the wrapper.
    need_normalized_logits = (
        bool(getattr(buf, "need_normalized_logits", True)) if gate_on else True
    )
    need_response_logprobs = (
        bool(getattr(buf, "need_response_logprobs", True)) if gate_on else True
    )

    t = _now()
    logits = _maybe_compiled_forward(self, inputs) if compile_on else self.model(
        inputs[:, None], cache=self.prompt_cache
    )
    logits = logits[:, -1, :]
    if attribution:
        mx.eval(logits)
    timings["forward_ms"] = _ms(t)

    t = _now()
    if self.logits_processors is not None and any(self.logits_processors):
        processed_logits: list[mx.array] = []
        for row in range(n):
            sample_logits = logits[row : row + 1]
            for processor in self.logits_processors[row]:
                sample_logits = processor(mx.array(self.tokens[row]), sample_logits)
            processed_logits.append(sample_logits)
        logits = mx.concatenate(processed_logits, axis=0)
    if attribution:
        mx.eval(logits)
    timings["processors_ms"] = _ms(t)

    t = _now()
    if need_normalized_logits:
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampler_input = logprobs
    else:
        logprobs = None
        sampler_input = logits
    if attribution:
        mx.eval(sampler_input)
    timings["logsumexp_ms"] = _ms(t)

    t = _now()
    grouped_sampler_used = False
    sampler_call_count = 0
    sampled: mx.array | None = None
    if group_on and self.samplers is not None and any(self.samplers):
        sampled, grouped_sampler_used, sampler_call_count = _grouped_sample(
            self.samplers,
            self.fallback_sampler,
            sampler_input,
            n,
        )
    if sampled is None:
        sampled, sampler_call_count = _sample_stock_per_row(self, sampler_input)
    if attribution:
        mx.eval(sampled)
    timings["sampler_ms"] = _ms(t)

    self._next_tokens = sampled

    t = _now()
    if need_response_logprobs:
        if logprobs is None:
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        self._next_logprobs = list(logprobs)
    else:
        dummy = mx.array([], dtype=logits.dtype)
        self._next_logprobs = [dummy] * n
    timings["response_logprobs_ms"] = _ms(t)

    t = _now()
    if buf.needs_topk:
        if logprobs is None:
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        k = min(_PRECOMPUTE_TOP_K, logprobs.shape[1])
        pending_indices = mx.argpartition(-logprobs, k, axis=1)[:, :k]
        pending_values = mx.take_along_axis(logprobs, pending_indices, axis=1)
        sort_order = mx.argsort(-pending_values, axis=1)
        pending_indices = mx.take_along_axis(pending_indices, sort_order, axis=1)
        pending_values = mx.take_along_axis(pending_values, sort_order, axis=1)
        pending_selected = logprobs[mx.arange(n), sampled]
        buf.pending = BatchTopKLogprobs(
            uids=list(self.uids),
            indices=pending_indices,
            values=pending_values,
            selected=pending_selected,
        )
        mx.async_eval(
            self._next_tokens,
            *self._next_logprobs,
            pending_indices,
            pending_values,
            pending_selected,
        )
    else:
        mx.async_eval(self._next_tokens, *self._next_logprobs)
    timings["topk_async_eval_ms"] = _ms(t)

    t = _now()
    mx.eval(inputs, *self._current_logprobs)
    timings["eval_current_ms"] = _ms(t)

    t = _now()
    _drain_prompt_cache_if_needed(self)
    timings["drain_ms"] = _ms(t)

    t = _now()
    token_list = cast(list[int], inputs.tolist())
    for stream_tokens, token in zip(self.tokens, token_list, strict=True):
        stream_tokens.append(token)
    timings["tolist_update_ms"] = _ms(t)

    if profile_mode is not None:
        context = getattr(buf, "profile_context", {}) or {}
        row = {
            "event": "hotpath_decode_step",
            "profile_mode": profile_mode,
            "ts": step_start,
            "step_total_ms": _ms(step_start),
            "batch_size": n,
            **context,
            "need_normalized_logits": need_normalized_logits,
            "need_response_logprobs": need_response_logprobs,
            "needs_topk": bool(buf.needs_topk),
            "grouped_sampler_used": grouped_sampler_used,
            "sampler_call_count": sampler_call_count,
            "drain_last_ok": getattr(self, "_exo_cache_drain_last_ok", None),
            "drain_last_count": getattr(self, "_exo_cache_drain_last_count", None),
        }
        if attribution:
            row.update(timings)
        _profile_write(row)

    return token_list, self._current_logprobs


def apply_hotpath_patch() -> None:
    """Install hot-path step only when runtime flags require it."""
    if not hotpath_enabled():
        return
    GenerationBatch._step = _hotpath_step  # pyright: ignore[reportAttributeAccessIssue]
    _logger.info(
        "[EXO][hotpath] installed "
        f"profile={_profile_mode()} gate={_flag('EXO_HOTPATH_GATE_LOGSUMEXP')} "
        f"group={_flag('EXO_HOTPATH_GROUP_SAMPLER')} "
        f"compile={_flag('EXO_HOTPATH_COMPILE')}"
    )
