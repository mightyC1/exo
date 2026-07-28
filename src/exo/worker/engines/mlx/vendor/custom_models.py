"""Register EXO-vendored custom models + tokenizer fixes into mlx-lm at runtime.

mlx-lm resolves model_type -> mlx_lm.models.<type> via importlib, and infers
reasoning tags from a hardcoded list in tokenizer_utils. MiniMax-M3
(model_type=minimax_m3_vl) ships in neither. Instead of copying files into
site-packages (wiped on every mlx-lm reinstall), register the vendored module
into sys.modules and wrap the thinking detector here, in EXO's own tree.
Idempotent. Run from utils_mlx before any load_model() call.
"""
from __future__ import annotations

import sys

_REGISTERED = False


def register_custom_models() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    # 1) make mlx-lm's load_model resolve `mlx_lm.models.minimax_m3_vl` to our
    #    vendored module -- the SAME object auto_parallel imports, so isinstance
    #    in the sharding-strategy selector lines up. Survives mlx-lm reinstall.
    from exo.worker.engines.mlx.vendor import minimax_m3_vl as _m3
    import mlx_lm.models as _mlx_models

    sys.modules["mlx_lm.models.minimax_m3_vl"] = _m3
    setattr(_mlx_models, "minimax_m3_vl", _m3)

    # MiMo-V2.5-Pro (model_type=mimo_v2): same pattern. The vendored copy has its
    # relative imports rewritten to absolute (mlx_lm.models.*) so it imports
    # cleanly from EXO. auto_parallel's mimo_v2 import is UNGUARDED -- a missing
    # file would crash every runner -- so vendoring removes that risk entirely.
    from exo.worker.engines.mlx.vendor import mimo_v2 as _mimo

    sys.modules["mlx_lm.models.mimo_v2"] = _mimo
    setattr(_mlx_models, "mimo_v2", _mimo)

    # Kimi K3 (model_type=kimi_k3): same vendor+register pattern. Официальный
    # конфиг nested (kimi_k3 -> text_config) — обрабатывается в
    # kimi_k3.ModelArgs.from_dict. auto_parallel импортирует ЭТОТ ЖЕ модуль
    # (isinstance в selector'е совпадает). TP-контракт LatentMoE — в докстринге
    # KimiK3LatentMoE; sharding-strategy: KimiK3ShardingStrategy.
    from exo.worker.engines.mlx.vendor import kimi_k3 as _k3

    sys.modules["mlx_lm.models.kimi_k3"] = _k3
    setattr(_mlx_models, "kimi_k3", _k3)

    # 2) prefer <mm:think> over <think> for M3 (replaces the tokenizer_utils
    #    edit). M3 vocab has both pairs but emits <mm:think>; upstream picks
    #    <think> first. Other models fall through to the original detector.
    import mlx_lm.tokenizer_utils as _tu

    _orig_infer = _tu._infer_thinking

    def _infer_thinking(tokenizer):  # type: ignore[no-redef]
        try:
            vocab = tokenizer.get_vocab()
            if "<mm:think>" in vocab and "</mm:think>" in vocab:
                return (
                    "<mm:think>", "</mm:think>",
                    (vocab["<mm:think>"],), (vocab["</mm:think>"],),
                )
        except Exception:
            pass
        return _orig_infer(tokenizer)

    _tu._infer_thinking = _infer_thinking
    _REGISTERED = True
