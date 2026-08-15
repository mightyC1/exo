from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from exo.worker.engines.mlx.patches.glm52_prefill import (  # noqa: E402
    batched_take_sequence,
    cache_has_active_right_padding,
    cache_requires_dense_prefill,
    gather_selected_mask,
)


def test_active_right_padding_guard_covers_pinned_cache_fields() -> None:
    class Cache:
        pass

    cache = Cache()
    assert not cache_has_active_right_padding(cache)
    cache._right_padding = object()
    assert cache_has_active_right_padding(cache)
    cache._right_padding = None
    cache._lengths = object()
    assert cache_has_active_right_padding(cache)
    cache._lengths = None
    cache.right_padding = object()
    assert cache_has_active_right_padding(cache)

    ordinary = Cache()
    assert not cache_requires_dense_prefill(ordinary)
    ordinary.left_padding = None
    assert cache_requires_dense_prefill(ordinary)


def test_mlx_batched_take_sequence_b2() -> None:
    x_np = np.arange(2 * 7 * 3, dtype=np.float32).reshape(2, 1, 7, 3)
    idx_np = np.array(
        [
            [[[0, 2], [4, 6]]],
            [[[6, 5], [1, 3]]],
        ],
        dtype=np.int32,
    )
    result = batched_take_sequence(mx.array(x_np), mx.array(idx_np))
    mx.eval(result)
    expected = np.stack(
        [
            x_np[0, 0][idx_np[0, 0]],
            x_np[1, 0][idx_np[1, 0]],
        ]
    )[:, None]
    np.testing.assert_array_equal(np.array(result), expected)


def test_mlx_gathers_original_mask_semantics() -> None:
    mask_np = np.array(
        [
            [
                [True, False, True, False, True, False],
                [False, True, False, True, False, True],
            ],
            [
                [False, False, True, True, False, True],
                [True, True, False, False, True, False],
            ],
        ],
        dtype=bool,
    )
    idx_np = np.array(
        [
            [[[0, 2, 5], [1, 3, 4]]],
            [[[2, 3, 5], [0, 1, 4]]],
        ],
        dtype=np.int32,
    )
    result = gather_selected_mask(
        mx.array(mask_np),
        mx.array(idx_np),
        batch=2,
        q_start=0,
        q_end=2,
        kv_length=6,
    )
    assert result is not None
    mx.eval(result)
    expected = np.take_along_axis(mask_np[:, None], idx_np, axis=-1)
    np.testing.assert_array_equal(np.array(result), expected)


class _IdentityAttention:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def embed_q(self, value, transpose=True):  # noqa: ARG002
        return value

    def unembed_out(self, value):
        return value

    def o_proj(self, value):
        return value


def _sdpa(queries, keys, values, cache, scale, mask):  # noqa: ARG001
    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
    )


def test_mlx_sparse_prefill_matches_masked_dense_b2_and_q_tiles() -> None:
    from exo.worker.engines.mlx.patches.glm52_prefill import (
        GLM52PrefillConfig,
        sparse_prefill_attention,
    )

    rng = np.random.default_rng(91)
    batch, heads, query, kv, topk, dim, rope_dim = 2, 3, 7, 17, 6, 8, 4
    offset = kv - query
    q_np = rng.normal(size=(batch, heads, query, dim)).astype(np.float32)
    q_pe_np = rng.normal(size=(batch, heads, query, rope_dim)).astype(np.float32)
    latent_np = rng.normal(size=(batch, 1, kv, dim)).astype(np.float32)
    pe_np = rng.normal(size=(batch, 1, kv, rope_dim)).astype(np.float32)
    mask_np = np.arange(kv)[None, :] <= (offset + np.arange(query))[:, None]
    mask_np = np.broadcast_to(mask_np[None, None], (batch, 1, query, kv)).copy()
    mask_np[0] &= np.arange(kv)[None, None, :] >= 2
    mask_np[1] &= np.arange(kv)[None, None, :] >= 4
    idx_np = np.empty((batch, 1, query, topk), dtype=np.int32)
    for b in range(batch):
        for q in range(query):
            valid = np.flatnonzero(mask_np[b, 0, q])
            invalid = np.flatnonzero(~mask_np[b, 0, q])
            chosen = valid[-min(valid.size, topk - 1) :]
            idx_np[b, 0, q] = np.concatenate((chosen, invalid[: topk - chosen.size]))

    q = mx.array(q_np)
    q_pe = mx.array(q_pe_np)
    latent = mx.array(latent_np)
    pe = mx.array(pe_np)
    mask = mx.array(mask_np)
    idx = mx.array(idx_np)
    scale = float((dim + rope_dim) ** -0.5)
    attn = _IdentityAttention(scale)

    config = GLM52PrefillConfig(
        sparse_mode="on",
        sparse_q_chunk=3,
        hotpath_compatible=True,
    )
    sparse = sparse_prefill_attention(
        attn,
        q_nope=q,
        q_pe=q_pe,
        kv_latent=latent,
        k_pe=pe,
        topk_indices=idx,
        mask=mask,
        scaled_dot_product_attention=_sdpa,
        cfg=config,
    )

    sparse_mask = mx.zeros((batch, 1, query, kv), dtype=mx.bool_)
    sparse_mask = mx.put_along_axis(sparse_mask, idx, mx.array(True), axis=-1)
    sparse_mask = sparse_mask & mask
    pe_scores = (q_pe * scale) @ pe.swapaxes(-1, -2)
    pe_scores = mx.where(
        sparse_mask,
        pe_scores,
        mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
    )
    dense = _sdpa(q, latent, latent, cache=None, scale=scale, mask=pe_scores)
    dense = dense.transpose(0, 2, 1, 3).reshape(batch, query, -1)
    mx.eval(sparse, dense)
    np.testing.assert_allclose(np.array(sparse), np.array(dense), rtol=2e-4, atol=2e-4)


class _Identity:
    def __call__(self, value):
        return value


class _FakeCache:
    def __init__(self, past):
        self.keys = past
        self.values = mx.zeros((*past.shape[:-1], 0), dtype=past.dtype)
        self.offset = int(past.shape[2])

    def update_and_fetch(self, keys, values):
        self.keys = mx.concatenate((self.keys, keys), axis=2)
        self.values = mx.concatenate((self.values, values), axis=2)
        self.offset = int(self.keys.shape[2])
        return self.keys, self.values


class _FakeIndexer:
    def __init__(self, q, current_k, weights, topk):
        self._q = q
        self._current_k = current_k
        self._weights = weights
        self.n_heads = int(q.shape[1])
        self.head_dim = int(q.shape[-1])
        self.index_topk = topk
        self.softmax_scale = self.head_dim**-0.5
        self.wk = lambda x: self._current_k[:, 0]  # noqa: ARG005
        self.k_norm = _Identity()
        self.rope = lambda value, offset=0: value  # noqa: ARG005
        self.wq_b = lambda qr: self._q.swapaxes(1, 2).reshape(  # noqa: ARG005
            self._q.shape[0], self._q.shape[2], -1
        )
        # tiled_indexer_call multiplies this by a fixed scalar; compensate here.
        scalar = self.n_heads**-0.5 * self.softmax_scale
        self.weights_proj = lambda x: (  # noqa: ARG005
            self._weights[..., 0].swapaxes(1, 2) / scalar
        )


def test_mlx_tiled_indexer_reference_and_streaming_match_monolithic_set() -> None:
    from exo.worker.engines.mlx.patches.glm52_prefill import (
        GLM52PrefillConfig,
        tiled_indexer_call,
    )

    rng = np.random.default_rng(101)
    batch, heads, query, past_len, dim, topk = 2, 4, 5, 13, 6, 4
    current_k_np = rng.normal(size=(batch, 1, query, dim)).astype(np.float32)
    past_np = rng.normal(size=(batch, 1, past_len, dim)).astype(np.float32)
    q_np = rng.normal(size=(batch, heads, query, dim)).astype(np.float32)
    weights_np = rng.normal(size=(batch, heads, query, 1)).astype(np.float32)
    total = past_len + query
    mask_np = np.arange(total)[None, :] <= (past_len + np.arange(query))[:, None]
    mask_np = np.broadcast_to(mask_np[None, None], (batch, 1, query, total)).copy()

    q = mx.array(q_np)
    current_k = mx.array(current_k_np)
    weights = mx.array(weights_np)
    mask = mx.array(mask_np)
    x = mx.zeros((batch, query, dim), dtype=mx.float32)
    qr = mx.zeros_like(x)

    combined_np = np.concatenate((past_np, current_k_np), axis=2)
    scores_np = np.einsum("bhqd,bnd->bhqn", q_np, combined_np[:, 0])
    scores_np = np.maximum(scores_np, 0) * weights_np
    scores_np = scores_np.sum(axis=1, keepdims=True)
    scores_np = np.where(mask_np, scores_np, -np.inf)
    expected = np.sort(
        np.argpartition(scores_np, kth=-topk, axis=-1)[..., -topk:], axis=-1
    )

    outputs = []
    for mode in ("reference", "streaming"):
        cache = _FakeCache(mx.array(past_np))
        indexer = _FakeIndexer(q, current_k, weights, topk)
        config = GLM52PrefillConfig(
            indexer_mode=mode,
            indexer_q_chunk=2,
            indexer_k_chunk=7,
            hotpath_compatible=True,
        )
        selected = tiled_indexer_call(indexer, x, qr, mask, cache, config)
        assert selected is not None
        mx.eval(selected)
        outputs.append(np.sort(np.array(selected), axis=-1))

    np.testing.assert_array_equal(outputs[0], expected)
    np.testing.assert_array_equal(outputs[1], expected)
