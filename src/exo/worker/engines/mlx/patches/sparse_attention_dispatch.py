"""Generic runtime dispatch for sparse-attention patches, keyed by model_type.

Several models EXO loads have sparse-attention behaviour that mlx-lm does not
(yet) handle correctly out of the box. Instead of scattering per-model hooks
through the loader, the loader calls :func:`apply_sparse_attention_patches`
once right after ``load_model``. This inspects the local ``config.json``'s
``model_type`` and routes to the matching patch module.

Implemented
-----------
``glm_moe_dsa``
    GLM-5.2 DSA IndexShare. Most sparse-attention layers are ``"shared"`` and
    must reuse the top-k token indices computed by the nearest previous
    ``"full"`` indexer layer, rather than running a (here randomly-initialised)
    per-layer indexer that corrupts generation past ``index_topk`` cumulative
    tokens. Implemented in :mod:`glm52_indexshare`.

Registered placeholders (currently no-op)
-----------------------------------------
These use *different* sparse mechanisms — the GLM top-k / IndexShare math does
**not** transfer — and they additionally need base architecture support in
mlx-lm/EXO before there is anything to patch. They are listed here so the
dispatch point is the single place to extend.

``minimax*``
    MiniMax M3 MSA / block-sparse attention: a fixed query/key block structure,
    not a learned top-k lightning indexer. Needs MSA support in mlx-lm first.
``mimo*``
    MiMo-V2.5-Pro SWA/GA hybrid (sliding-window + periodic global) over a
    ``RotatingKVCache``: window/global masking, not top-k selection. Needs the
    mlx-lm architecture (PR #1219) first, plus the RotatingKVCache rewind fix.

To add a real implementation: write a module exposing
``apply_<arch>_patch(model, model_path, *, logger) -> model`` that self-gates on
the config (returning ``model`` unchanged when not applicable), then register it
in :data:`_PATCHES` (exact ``model_type`` match) or, for a family of related
model types, in :data:`_PLACEHOLDER_PREFIXES`-style prefix matching.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from exo.worker.runner.bootstrap import logger as default_logger

from exo.worker.engines.mlx.patches.glm52_indexshare import (
    apply_glm52_indexshare_patch,
)

_PatchFn = Callable[..., Any]


def _disabled() -> bool:
    value = os.environ.get("EXO_SPARSE_ATTENTION_PATCHES", "1").strip().lower()
    return value in {"0", "false", "no", "n", "off", "none", "disabled"}


def _read_model_type(model_path: Path) -> str | None:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - diagnostics only
        return None
    if not isinstance(data, dict):
        return None
    model_type = data.get("model_type")
    return str(model_type) if model_type is not None else None


def _not_implemented(arch: str, needs: str) -> _PatchFn:
    def _stub(model: Any, model_path: Path, *, logger: Any = default_logger) -> Any:
        logger.info(
            f"[EXO][SparseAttn] '{arch}' detected but no sparse-attention patch "
            f"is implemented yet. {needs} Falling back to mlx-lm's default path."
        )
        return model

    return _stub


# Exact ``model_type`` -> patch function.
_PATCHES: dict[str, _PatchFn] = {
    "glm_moe_dsa": apply_glm52_indexshare_patch,
}

# Prefix-matched placeholders for architectures whose mechanism differs and
# which are not yet loadable. Kept distinct from ``_PATCHES`` so any future
# exact-match implementation always wins over a placeholder.
_PLACEHOLDER_PREFIXES: dict[str, _PatchFn] = {
    "mimo": _not_implemented(
        "mimo",
        "MiMo-V2.5-Pro uses an SWA/GA sliding-window+global hybrid over a "
        "RotatingKVCache (a different mechanism) and needs mlx-lm arch support "
        "(PR #1219) first.",
    ),
}


def _resolve(model_type: str) -> _PatchFn | None:
    patch = _PATCHES.get(model_type)
    if patch is not None:
        return patch
    for prefix, stub in _PLACEHOLDER_PREFIXES.items():
        if model_type.startswith(prefix):
            return stub
    return None


def apply_sparse_attention_patches(
    model: Any,
    model_path: Path,
    *,
    logger: Any = default_logger,
) -> Any:
    """Apply the sparse-attention patch matching this model's architecture.

    Always returns the (possibly patched) model so callers can use it inline in
    load paths. Dense / fully-supported models pass through unchanged. Disable
    the whole dispatch with ``EXO_SPARSE_ATTENTION_PATCHES=0``.
    """
    if _disabled():
        logger.info(
            "[EXO][SparseAttn] dispatch disabled by EXO_SPARSE_ATTENTION_PATCHES=0"
        )
        return model

    model_type = _read_model_type(model_path)
    if model_type is None:
        return model

    patch = _resolve(model_type)
    if patch is None:
        return model

    return patch(model, model_path, logger=logger)
