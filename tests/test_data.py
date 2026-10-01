"""Data generation + splitting tests.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

from tscforge.data.generators import DATASET_REGISTRY
from tscforge.data.split import stratified_split


def test_generators_deterministic():
    for name in DATASET_REGISTRY:
        a = DATASET_REGISTRY[name](n_total=100, length=128, seed=42)
        b = DATASET_REGISTRY[name](n_total=100, length=128, seed=42)
        np.testing.assert_array_equal(a.X, b.X)
        np.testing.assert_array_equal(a.y, b.y)


def test_generators_distinct_seed_changes_data():
    a = DATASET_REGISTRY["order_freq2a"](n_total=100, length=128, seed=1)
    b = DATASET_REGISTRY["order_freq2a"](n_total=100, length=128, seed=2)
    assert not np.array_equal(a.X, b.X)


def test_stratified_split_preserves_classes():
    batch = DATASET_REGISTRY["order_freq3a"](n_total=120, length=128, seed=7)
    sp = stratified_split(batch, 7)
    for arm in ("train", "val", "holdout"):
        # each arm must contain (almost) every class
        assert set(np.unique(sp[arm].y)) == set(np.unique(batch.y))
    # sizes sum to total
    total = sum(sp[arm].n_samples for arm in ("train", "val", "holdout"))
    assert total == batch.n_samples
    # ratio roughly preserved (class counts are not uniform, so use tolerance)
    assert abs(sp["train"].n_samples / batch.n_samples - 0.4) < 0.05
    assert abs(sp["val"].n_samples / batch.n_samples - 0.2) < 0.05


def test_split_does_not_mutate_source():
    batch = DATASET_REGISTRY["order_freq2a"](n_total=90, length=128, seed=3)
    X0 = batch.X.copy()
    stratified_split(batch, 3)
    np.testing.assert_array_equal(batch.X, X0)
