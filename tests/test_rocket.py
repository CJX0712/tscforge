"""Unit tests for the TinyRocket kernel machinery.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

from tscforge.core.seed import get_rng
from tscforge.domain.rocket import sample_kernels, transform, transform_pool

N_KERNELS = 64
T = 64


def _hash_kernels(k):
    return (
        np.ascontiguousarray(k.lengths).tobytes()
        + np.ascontiguousarray(k.dilations).tobytes()
        + np.ascontiguousarray(k.weights).tobytes()
        + np.ascontiguousarray(k.biases).tobytes()
        + np.ascontiguousarray(k.paddings).tobytes()
    )


def test_sample_kernels_deterministic():
    r1 = get_rng("rocket", 123)
    r2 = get_rng("rocket", 123)
    k1 = sample_kernels(r1, N_KERNELS, T)
    k2 = sample_kernels(r2, N_KERNELS, T)
    assert _hash_kernels(k1) == _hash_kernels(k2)


def test_sample_kernels_different_streams():
    r1 = get_rng("rocket", 123)
    r2 = get_rng("rocket_pilot", 123)
    k1 = sample_kernels(r1, N_KERNELS, T)
    k2 = sample_kernels(r2, N_KERNELS, T)
    # distinct streams => distinct kernels (extremely unlikely to be identical)
    assert _hash_kernels(k1) != _hash_kernels(k2)


def test_kernel_lengths_within_bounds():
    rng = get_rng("rocket", 7)
    k = sample_kernels(rng, 256, T)
    assert k.n_kernels == 256
    assert int(k.lengths.min()) >= 2
    assert int(k.lengths.max()) <= int(min(0.4 * T, 100)) + 1
    assert int(k.dilations.min()) >= 1


def test_transform_shape_and_range():
    rng = get_rng("rocket", 9)
    k = sample_kernels(rng, N_KERNELS, T)
    n = 8
    X = rng.standard_normal((n, 1, T))
    feats = transform(X, k)
    assert feats.shape == (n, N_KERNELS * 2)
    # PPV column (even index) must lie in [0, 1]
    assert feats[:, 0::2].min() >= 0.0
    assert feats[:, 0::2].max() <= 1.0
    # Max column (odd index) must be >= 0 (it is a max over ReLU >= 0)
    assert feats[:, 1::2].min() >= 0.0


def test_transform_pool_ablation():
    rng = get_rng("rocket", 9)
    k = sample_kernels(rng, N_KERNELS, T)
    X = rng.standard_normal((4, 1, T))
    full = transform_pool(X, k, multi_pool=True)
    ppv_only = transform_pool(X, k, multi_pool=False)
    assert full.shape == (4, N_KERNELS * 2)
    assert ppv_only.shape == (4, N_KERNELS)
    np.testing.assert_array_equal(ppv_only, full[:, 0::2])


def test_transform_deterministic_for_fixed_input():
    rng = get_rng("rocket", 11)
    k = sample_kernels(rng, N_KERNELS, T)
    X = get_rng("x", 11).standard_normal((5, 1, T))
    a = transform(X, k)
    b = transform(X, k)
    np.testing.assert_array_equal(a, b)
