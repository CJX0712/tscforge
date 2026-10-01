"""Unit tests for the DTW / prototype module.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

from tscforge.domain.dtw import (
    cross_dtw,
    dtw_distance,
    pairwise_dtw,
    prototype_distances,
    select_prototypes,
)


def test_dtw_distance_symmetric_and_zero_diag():
    rng = np.random.default_rng(0)
    a = np.sin(np.linspace(0, 6 * np.pi, 24)) + 0.1 * rng.standard_normal(24)
    b = np.cos(np.linspace(0, 6 * np.pi, 24)) + 0.1 * rng.standard_normal(24)
    d1 = dtw_distance(a, b, band=8)
    d2 = dtw_distance(b, a, band=8)
    assert abs(d1 - d2) < 1e-9
    d0 = dtw_distance(a, a, band=8)
    assert d0 == 0.0


def test_pairwise_dtw_symmetric_positive():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((8, 20))
    D = pairwise_dtw(X, band=6)
    assert D.shape == (8, 8)
    np.testing.assert_array_almost_equal(D, D.T, decimal=9)
    assert np.all(np.diag(D) == 0.0)
    assert np.all(D >= 0.0)


def test_pairwise_dtw_does_not_mutate_input():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((6, 18))
    X_copy = X.copy()
    pairwise_dtw(X, band=5)
    np.testing.assert_array_equal(X, X_copy)


def test_prototype_selection_returns_per_class():
    rng = np.random.default_rng(3)
    n = 12
    X = rng.standard_normal((n, 20))
    y = np.array([0, 1] * (n // 2))
    protos = select_prototypes(X, y, seed=3, band=6)
    assert protos.shape == (2, 20)


def test_prototype_distances_shape():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((10, 20))
    y = np.array([0, 1] * 5)
    protos = select_prototypes(X, y, seed=4, band=6)
    d = prototype_distances(X, protos, band=6)
    assert d.shape == (10, 2)
    assert np.all(d >= 0.0)


def test_cross_dtw_shape():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((3, 22))
    B = rng.standard_normal((5, 22))
    D = cross_dtw(A, B, band=7)
    assert D.shape == (3, 5)
