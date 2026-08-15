"""CPU reference tests for the GLM-5.2 prefill mathematics.

These tests deliberately use NumPy so they can run before deployment to an
Apple/MLX host. The companion MLX test checks the actual gather primitives.
"""

from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _batched_take(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    # x: [B,N,D], idx: [B,Q,K]
    batch = np.arange(x.shape[0])[:, None, None]
    return x[batch, idx]


def _dense_topk_attention(
    q_lat: np.ndarray,
    q_pe: np.ndarray,
    latent: np.ndarray,
    pe: np.ndarray,
    idx: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> np.ndarray:
    content = np.einsum("bhqd,bnd->bhqn", q_lat, latent) * scale
    rope = np.einsum("bhqr,bnr->bhqn", q_pe, pe) * scale

    selected = np.zeros((idx.shape[0], idx.shape[1], latent.shape[1]), dtype=bool)
    np.put_along_axis(selected, idx, True, axis=-1)
    selected &= mask
    # Match the pinned implementation: only the PE additive mask is replaced;
    # SDPA adds the content term afterwards.
    pe_scores = np.where(selected[:, None], rope, np.finfo(rope.dtype).min)
    probs = _softmax(content + pe_scores)
    return np.einsum("bhqn,bnd->bhqd", probs, latent)


def _sparse_attention(
    q_lat: np.ndarray,
    q_pe: np.ndarray,
    latent: np.ndarray,
    pe: np.ndarray,
    idx: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> np.ndarray:
    latent_g = _batched_take(latent, idx)
    pe_g = _batched_take(pe, idx)
    mask_g = np.take_along_axis(mask, idx, axis=-1)
    content = np.einsum("bhqd,bqkd->bhqk", q_lat, latent_g) * scale
    rope = np.einsum("bhqr,bqkr->bhqk", q_pe, pe_g) * scale
    pe_scores = np.where(mask_g[:, None], rope, np.finfo(rope.dtype).min)
    probs = _softmax(content + pe_scores)
    return np.einsum("bhqk,bqkd->bhqd", probs, latent_g)


def _make_mask(batch: int, query: int, kv: int, offset: int) -> np.ndarray:
    positions = offset + np.arange(query)
    mask = np.arange(kv)[None, :] <= positions[:, None]
    return np.broadcast_to(mask[None], (batch, query, kv)).copy()


def test_batched_gather_uses_different_indices_per_row() -> None:
    x = np.arange(2 * 7 * 3).reshape(2, 7, 3)
    idx = np.array(
        [
            [[0, 2], [4, 6]],
            [[6, 5], [1, 3]],
        ]
    )
    gathered = _batched_take(x, idx)
    assert gathered.shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(gathered[0, 0, 1], x[0, 2])
    np.testing.assert_array_equal(gathered[1, 0, 0], x[1, 6])
    np.testing.assert_array_equal(gathered[1, 1, 1], x[1, 3])


def test_sparse_matches_masked_dense_for_causal_padding_and_window() -> None:
    rng = np.random.default_rng(7)
    batch, heads, query, kv, topk, dim, rope_dim = 2, 3, 7, 19, 6, 8, 4
    offset = kv - query
    q_lat = rng.normal(size=(batch, heads, query, dim)).astype(np.float32)
    q_pe = rng.normal(size=(batch, heads, query, rope_dim)).astype(np.float32)
    latent = rng.normal(size=(batch, kv, dim)).astype(np.float32)
    pe = rng.normal(size=(batch, kv, rope_dim)).astype(np.float32)

    mask = _make_mask(batch, query, kv, offset)
    # Unequal left padding and a synthetic sliding window.
    mask[0] &= np.arange(kv)[None, :] >= 2
    mask[1] &= np.arange(kv)[None, :] >= 5
    positions = offset + np.arange(query)
    mask &= np.arange(kv)[None, None, :] >= (positions[None, :, None] - 9)

    # Pick valid positions plus a few invalid candidates to exercise gathered mask.
    idx = np.empty((batch, query, topk), dtype=np.int32)
    for b in range(batch):
        for q in range(query):
            valid = np.flatnonzero(mask[b, q])
            assert valid.size
            chosen_valid = valid[-min(valid.size, topk - 1) :]
            invalid = np.flatnonzero(~mask[b, q])
            need = topk - chosen_valid.size
            assert invalid.size >= need
            idx[b, q] = np.concatenate((chosen_valid, invalid[:need]))

    scale = (dim + rope_dim) ** -0.5
    dense = _dense_topk_attention(q_lat, q_pe, latent, pe, idx, mask, scale)
    sparse = _sparse_attention(q_lat, q_pe, latent, pe, idx, mask, scale)
    np.testing.assert_allclose(sparse, dense, rtol=2e-5, atol=2e-5)


def test_threshold_crossing_and_non_full_last_query_tile() -> None:
    rng = np.random.default_rng(11)
    batch, heads, query, kv, topk, dim, rope_dim = 1, 2, 9, 17, 8, 6, 2
    offset = 8
    mask = _make_mask(batch, query, kv, offset)
    idx = np.tile(np.arange(topk, dtype=np.int32), (batch, query, 1))
    # Later rows select newer keys, while early rows retain masked future entries.
    for q in range(query):
        idx[0, q] = np.arange(max(0, q), max(0, q) + topk) % kv

    q_lat = rng.normal(size=(batch, heads, query, dim)).astype(np.float32)
    q_pe = rng.normal(size=(batch, heads, query, rope_dim)).astype(np.float32)
    latent = rng.normal(size=(batch, kv, dim)).astype(np.float32)
    pe = rng.normal(size=(batch, kv, rope_dim)).astype(np.float32)
    scale = (dim + rope_dim) ** -0.5

    dense = _dense_topk_attention(q_lat, q_pe, latent, pe, idx, mask, scale)
    # Simulate Q chunks 4/4/1 and concatenate.
    sparse_parts = []
    for start in range(0, query, 4):
        end = min(query, start + 4)
        sparse_parts.append(
            _sparse_attention(
                q_lat[:, :, start:end],
                q_pe[:, :, start:end],
                latent,
                pe,
                idx[:, start:end],
                mask[:, start:end],
                scale,
            )
        )
    sparse = np.concatenate(sparse_parts, axis=2)
    np.testing.assert_allclose(sparse, dense, rtol=2e-5, atol=2e-5)


def test_all_masked_row_requires_full_n_dense_fallback() -> None:
    rng = np.random.default_rng(19)
    q_lat = rng.normal(size=(1, 2, 1, 4)).astype(np.float32)
    q_pe = rng.normal(size=(1, 2, 1, 2)).astype(np.float32)
    latent = rng.normal(size=(1, 7, 4)).astype(np.float32)
    pe = rng.normal(size=(1, 7, 2)).astype(np.float32)
    idx = np.array([[[0, 2, 4]]], dtype=np.int32)
    mask = np.zeros((1, 1, 7), dtype=bool)
    sparse = _sparse_attention(q_lat, q_pe, latent, pe, idx, mask, 0.5)
    dense = _dense_topk_attention(q_lat, q_pe, latent, pe, idx, mask, 0.5)
    assert np.isfinite(sparse).all() and np.isfinite(dense).all()
    # The pinned dense path still normalizes this row over full N, while the
    # sparse path can only normalize over K selected values. They are not exact.
    assert not np.allclose(sparse, dense, rtol=2e-5, atol=2e-5)


def _monolithic_index_scores(
    q: np.ndarray, k: np.ndarray, weights: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    scores = np.einsum("bhqd,bhnd->bhqn", q, k)
    scores = np.maximum(scores, 0) * weights
    scores = scores.sum(axis=1, keepdims=True)
    return np.where(mask[:, None], scores, -np.inf)


def _tiled_index_scores(
    q: np.ndarray,
    k: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    q_chunk: int,
    k_chunk: int,
) -> np.ndarray:
    query_parts = []
    for qs in range(0, q.shape[2], q_chunk):
        qe = min(q.shape[2], qs + q_chunk)
        key_parts = []
        for ks in range(0, k.shape[2], k_chunk):
            ke = min(k.shape[2], ks + k_chunk)
            partial = np.einsum("bhqd,bhnd->bhqn", q[:, :, qs:qe], k[:, :, ks:ke])
            partial = np.maximum(partial, 0) * weights[:, :, qs:qe]
            partial = partial.sum(axis=1, keepdims=True)
            key_parts.append(np.where(mask[:, None, qs:qe, ks:ke], partial, -np.inf))
        query_parts.append(np.concatenate(key_parts, axis=-1))
    return np.concatenate(query_parts, axis=2)


def _topk_set(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(scores, kth=-k, axis=-1)[..., -k:]
    return np.sort(idx, axis=-1)


def _streaming_topk(scores: np.ndarray, k: int, key_chunk: int) -> np.ndarray:
    best_scores = None
    best_idx = None
    for start in range(0, scores.shape[-1], key_chunk):
        end = min(scores.shape[-1], start + key_chunk)
        part = scores[..., start:end]
        idx = np.broadcast_to(np.arange(start, end), part.shape)
        if best_scores is not None:
            part = np.concatenate((best_scores, part), axis=-1)
            idx = np.concatenate((best_idx, idx), axis=-1)
        keep = min(k, part.shape[-1])
        positions = np.argpartition(part, kth=-keep, axis=-1)[..., -keep:]
        best_scores = np.take_along_axis(part, positions, axis=-1)
        best_idx = np.take_along_axis(idx, positions, axis=-1)
    assert best_idx is not None
    return np.sort(best_idx, axis=-1)


def test_reference_and_streaming_indexer_valid_sets() -> None:
    rng = np.random.default_rng(23)
    batch, heads, query, kv, dim = 2, 5, 11, 37, 7
    q = rng.normal(size=(batch, heads, query, dim)).astype(np.float32)
    k = rng.normal(size=(batch, 1, kv, dim)).astype(np.float32)
    weights = rng.normal(size=(batch, heads, query, 1)).astype(np.float32)
    mask = _make_mask(batch, query, kv, kv - query)

    mono = _monolithic_index_scores(q, k, weights, mask)
    tiled = _tiled_index_scores(q, k, weights, mask, q_chunk=4, k_chunk=9)
    np.testing.assert_allclose(tiled, mono, rtol=1e-6, atol=1e-6)

    topk = 6
    baseline_set = _topk_set(mono, topk)
    tiled_set = _topk_set(tiled, topk)
    streaming_set = _streaming_topk(tiled, topk, key_chunk=9)
    np.testing.assert_array_equal(tiled_set, baseline_set)
    np.testing.assert_array_equal(streaming_set, baseline_set)


def test_both_content_and_rope_terms_are_scaled() -> None:
    q_lat = np.array([[[[2.0]]]], dtype=np.float32)
    q_pe = np.array([[[[3.0]]]], dtype=np.float32)
    latent = np.array([[[5.0], [7.0]]], dtype=np.float32)
    pe = np.array([[[11.0], [13.0]]], dtype=np.float32)
    idx = np.array([[[0, 1]]], dtype=np.int32)
    mask = np.ones((1, 1, 2), dtype=bool)
    scale = 0.25

    latent_g = _batched_take(latent, idx)
    pe_g = _batched_take(pe, idx)
    actual_scores = (
        np.einsum("bhqd,bqkd->bhqk", q_lat, latent_g)
        + np.einsum("bhqr,bqkr->bhqk", q_pe, pe_g)
    ) * scale
    expected = np.array([[[[(2 * 5 + 3 * 11) * scale, (2 * 7 + 3 * 13) * scale]]]])
    np.testing.assert_allclose(actual_scores, expected.reshape(actual_scores.shape))
