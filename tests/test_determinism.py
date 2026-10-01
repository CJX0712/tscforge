"""Determinism tests: same seed => identical predictions (no hidden randomness).

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

from tscforge.core.config import TscConfig
from tscforge.core.seed import set_global_seed
from tscforge.data.generators import DATASET_REGISTRY
from tscforge.data.split import stratified_split
from tscforge.domain.model import PrismClassifier


def _cell(ds="order_freq2a", n=80, length=128, seed=101):
    set_global_seed(seed)
    batch = DATASET_REGISTRY[ds](n_total=n, length=length, seed=seed)
    sp = stratified_split(batch, seed)
    return sp["train"], sp["val"], sp["holdout"]


def test_prism_deterministic():
    train, val, holdout = _cell()
    cfg = TscConfig(seed=101)
    m1 = PrismClassifier(cfg)
    m1.fit(train, val)
    p1 = np.asarray(m1.predict(holdout))

    train2, val2, holdout2 = _cell()  # rebuild identical data
    m2 = PrismClassifier(TscConfig(seed=101))
    m2.fit(train2, val2)
    p2 = np.asarray(m2.predict(holdout2))
    np.testing.assert_array_equal(p1, p2)


def test_tinyrocket_variant_deterministic():
    train, val, holdout = _cell()
    cfg = TscConfig(seed=202).with_overrides(use_dtw_features=False, use_norm_gate=False)
    m1 = PrismClassifier(cfg)
    m1.fit(train, val)
    p1 = np.asarray(m1.predict(holdout))

    train2, val2, holdout2 = _cell()
    m2 = PrismClassifier(cfg)
    m2.fit(train2, val2)
    p2 = np.asarray(m2.predict(holdout2))
    np.testing.assert_array_equal(p1, p2)


def test_pipeline_run_deterministic():
    from tscforge.pipeline.pipeline import TscPipeline

    pipe = TscPipeline(TscConfig(seed=303))
    r1 = pipe.run("order_freq3b", 303, collect_failures=False)
    pipe2 = TscPipeline(TscConfig(seed=303))
    r2 = pipe2.run("order_freq3b", 303, collect_failures=False)
    for name in r1.accuracies:
        assert abs(r1.accuracies[name] - r2.accuracies[name]) < 1e-12
