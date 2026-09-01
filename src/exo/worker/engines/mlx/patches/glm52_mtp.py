"""GLM-5.2/5.3 MTP speculative decoding — phase 1: block + side-load + shadow.

Env-gated, default **off** (zero prod impact until enabled):

  EXO_GLM52_MTP=off|shadow|on   "on" is the phase-2 battle loop; in phase 1 it
                                warns once and behaves as shadow (output is
                                never altered by this module).
  EXO_GLM52_MTP_WEIGHTS=auto|/path/to/mtp.safetensors  (auto: next to model)
  EXO_GLM52_MTP_CONCAT=eh|he    eh_proj input order: eh = [enorm(embed),
                                hnorm(hidden)] (DeepSeek-V3 reference order,
                                default); he = swapped. Shadow A/B knob in
                                case accept-rate points at the order.

Shadow mode: on every decode step (B==1 only) the MTP head drafts the next
token in parallel from (pre-final-norm hidden, sampled token) and the draft is
compared against the token the main model actually samples one step later.
Model output is bit-identical to MTP=off; the only artifacts are
``[MTP_SHADOW]`` telemetry lines and one extra small forward per step.

Design notes (docs/glm52-mtp-recon.md D1–D7):
  * block = pinned ``DeepseekV32DecoderLayer(args, layer_idx=num_hidden_layers)``
    (MoE branch + own full indexer come from the pin) + enorm/hnorm/eh_proj +
    shared_head.norm; embed_tokens / lm_head are shared with the main model
    (tied in the checkpoint), replicated on every rank → zero new collectives.
  * side-load is presence-driven: a submodule is quantized iff
    ``<name>.scales`` exists in mtp.safetensors (config flags not consulted).
  * hidden capture: the model's final ``norm`` is wrapped by a passthrough
    object storing its input (pre-norm h). TP-safe: h is replicated.
  * decode-step hook wraps whatever ``GenerationBatch._step`` is currently
    installed (normally opt_batch_gen's ``_patched_step``); everything about
    the wrapped step (topk buffer, cache drain, async chains, pipelining) is
    preserved because shadow only *reads* ``_next_tokens`` lazily.
  * accounting is one step delayed so no extra GPU syncs are introduced:
    draft/eq arrays ride the existing async_eval train.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from exo.worker.runner.bootstrap import logger as default_logger

_ENV_MODE = "EXO_GLM52_MTP"
_ENV_WEIGHTS = "EXO_GLM52_MTP_WEIGHTS"
_ENV_CONCAT = "EXO_GLM52_MTP_CONCAT"

_LOG_EVERY = 64
_WIN = 256

_STEP_WRAPPED = False
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

    n_bytes = sum(
        v.nbytes for _, v in tree_flatten(module.parameters())
    )
    logger.info(
        f"[MTP] side-loaded {weights_path.name}: {len(weights)} tensors, "
        f"{n_bytes / 1e9:.2f} GB (int{bits} g{group_size} {mode}, "
        f"presence-driven), layer_idx={layer_idx}"
    )
    return module


# ---------------------------------------------------------------------------
# Shadow state + step hook
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
    ) -> None:
        from mlx_lm.models.cache import CacheList, KVCache

        self.mode = mode
        self.concat = concat
        self.mtp = mtp
        self.embed = embed
        self.lm_head = lm_head
        self.store = store
        self.logger = logger
        self._cache_cls = lambda: CacheList(KVCache(), KVCache())

        self.uid: int | None = None
        self.mtp_cache: Any = None
        self.pending_draft: mx.array | None = None
        self.lazy_match: mx.array | None = None
        self.steps = 0
        self.matches = 0
        self.win: list[bool] = []

    # -- request lifecycle --------------------------------------------------

    def start_request(self, uid: int) -> None:
        if self.uid is not None and self.steps:
            self._summary("switch")
        self.uid = uid
        self.mtp_cache = self._cache_cls()
        self.pending_draft = None
        self.lazy_match = None
        self.steps = 0
        self.matches = 0
        self.win = []

    def reset(self) -> None:
        if self.uid is not None and self.steps:
            self._summary("reset")
        self.uid = None
        self.mtp_cache = None
        self.pending_draft = None
        self.lazy_match = None

    def account(self, hit: bool) -> None:
        self.steps += 1
        self.matches += int(hit)
        self.win.append(hit)
        if len(self.win) > _WIN:
            self.win.pop(0)
        if self.steps % _LOG_EVERY == 0:
            self._line("")

    def _rate(self) -> float:
        return self.matches / self.steps if self.steps else 0.0

    def _line(self, tag: str) -> None:
        wr = (sum(self.win) / len(self.win)) if self.win else 0.0
        self.logger.info(
            f"[MTP_SHADOW]{tag} uid={self.uid} steps={self.steps} "
            f"match={self.matches} rate={self._rate():.3f} "
            f"win{len(self.win)}={wr:.3f}"
        )

    def _summary(self, why: str) -> None:
        self._line(f" summary({why})")

    # -- math ---------------------------------------------------------------

    def draft(self, h_last: mx.array, next_tok: mx.array) -> mx.array:
        """One MTP forward: (pre-norm h at pos t, token t+1) -> argmax t+2."""
        tok = next_tok.reshape(1, 1)
        e = self.enorm_embed(tok)
        h = self.mtp.hnorm(h_last)
        if self.concat == "eh":
            x = mx.concatenate([e, h], axis=-1)
        else:
            x = mx.concatenate([h, e], axis=-1)
        x = self.mtp.eh_proj(x)
        y = self.mtp.block(x, mask=None, cache=self.mtp_cache)
        logits = self.lm_head(self.mtp.head_norm(y))
        return mx.argmax(logits[..., -1, :], axis=-1).reshape(1)

    def enorm_embed(self, tok: mx.array) -> mx.array:
        return self.mtp.enorm(self.embed(tok))

    def mtp_cache_arrays(self) -> list[mx.array]:
        if self.mtp_cache is None:
            return []
        out: list[mx.array] = []
        for c in self.mtp_cache.caches:
            if getattr(c, "keys", None) is not None:
                out.append(c.keys)
                out.append(c.values)
        return out


def _shadow_step(state: _MTPState, prev_step: Any, batch: Any):
    if len(batch.uids) != 1:
        state.reset()
        return prev_step(batch)
    uid = batch.uids[0]
    if state.uid != uid:
        state.start_request(uid)

    # 1) harvest the comparison scheduled two calls ago (fully materialized
    #    by now — its inputs rode the previous steps' async_eval trains).
    if state.lazy_match is not None:
        state.account(bool(state.lazy_match.item()))
        state.lazy_match = None

    # 2) the real step (forward + sampling + emission). The norm wrapper
    #    stores pre-norm h of this forward into state.store.
    result = prev_step(batch)

    h = state.store.pop("h", None)
    next_tok = batch._next_tokens  # lazy: token sampled by this step
    if h is None or next_tok is None:
        _warn_once(
            state.logger, "no-h",
            "[MTP_SHADOW] pre-norm hidden not captured; shadow idle this step",
        )
        return result

    # 3) schedule comparison of the previous draft vs this step's sample.
    if state.pending_draft is not None:
        state.lazy_match = mx.equal(
            state.pending_draft, next_tok.reshape(-1)[0:1].astype(mx.int64)
        )
        mx.async_eval(state.lazy_match)

    # 4) draft the next token; ride the async train, no syncs.
    d = state.draft(h[:, -1:, :], next_tok.reshape(-1)[0:1])
    state.pending_draft = d.astype(mx.int64)
    mx.async_eval(state.pending_draft, *state.mtp_cache_arrays())
    return result


def _install_step_wrapper(logger: Any) -> None:
    global _STEP_WRAPPED
    if _STEP_WRAPPED:
        return
    from mlx_lm.generate import GenerationBatch

    prev_step = GenerationBatch._step
    if getattr(prev_step, "_exo_glm52_mtp_wrapper", False):
        _STEP_WRAPPED = True
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
            return _shadow_step(state, prev_step, self)
        except Exception:
            _warn_once(
                logger, "shadow-crash",
                "[MTP_SHADOW] exception in shadow path; disabling for this "
                "model (output unaffected)",
            )
            logger.opt(exception=True).warning("[MTP_SHADOW] traceback")
            state.mode = "off"
            return prev_step(self)

    _mtp_step._exo_glm52_mtp_wrapper = True  # type: ignore[attr-defined]
    GenerationBatch._step = _mtp_step
    _STEP_WRAPPED = True
    logger.info("[MTP] GenerationBatch._step wrapped (shadow-capable)")


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
    """Enable GLM-5.2/5.3 MTP shadow on a loaded glm_moe_dsa model.

    Fail-closed: any precondition miss logs one line and returns the model
    unchanged. Always returns the original model object.
    """
    mode = _env_choice(_ENV_MODE, "off", {"off", "shadow", "on"}, logger)
    if mode == "off":
        return model
    if mode == "on":
        _warn_once(
            logger, "mode-on",
            "[MTP] EXO_GLM52_MTP=on: battle loop lands in phase 2; "
            "running SHADOW (output unchanged)",
        )
        mode = "shadow"

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
        store=store, logger=logger,
    )
    model._exo_glm52_mtp_state = state
    _install_step_wrapper(logger)
    logger.info(
        f"[MTP] enabled mode={mode} concat={concat} weights={weights_path.name} "
        f"layer_idx={layer_idx} (shadow: output byte-identical to MTP=off)"
    )
    return model
