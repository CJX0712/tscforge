"""Tests for baseline classifiers and the PRISM flagship.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np
import pytest

from tscforge.core.config import TscConfig
from tscforge.core.seed import set_global_seed
from tscforge.data.generators import DATASET_REGISTRY
from tscforge.data.split import stratified_split
from tscforge.domain.model import PrismClassifier
from tscforge.pipeline.baselines import (
    LogRegFlat,
    OneNNDtw,
    OneNNEd,
    RfHandcrafted,
)

N = 80
LEN = 128


def _cell(ds="order_freq2a", seed=101):
    set_global_seed(seed)
    batch = DATASET_REGISTRY[ds](n_total=N, length=LEN, seed=seed)
    sp = stratified_split(batch, seed)
    return sp["train"], sp["val"], sp["holdout"]


BASELINES = [OneNNEd, OneNNDtw, LogRegFlat, RfHandcrafted]


@pytest.mark.parametrize("baseline_cls", BASELINES)
def test_baseline_predictions_valid(baseline_cls):
    train, val, holdout = _cell()
    model = (
        baseline_cls(seed=101) if baseline_cls in (LogRegFlat, RfHandcrafted) else baseline_cls()
    )
    model.fit(train, val)
    pred = np.asarray(model.predict(holdout))
    assert pred.shape == holdout.y.shape
    assert set(np.unique(pred)).issubset(set(np.unique(train.y)))


def test_prism_predictions_valid():
    train, val, holdout = _cell()
    cfg = TscConfig(seed=101)
    m = PrismClassifier(cfg)
    m.fit(train, val)
    pred = np.asarray(m.predict(holdout))
    assert pred.shape == holdout.y.shape
    assert set(np.unique(pred)).issubset(set(np.unique(train.y)))


def test_prism_fit_does_not_mutate_inputs():
    train, val, holdout = _cell()
    train_X0 = train.X.copy()
    cfg = TscConfig(seed=101)
    m = PrismClassifier(cfg)
    m.fit(train, val)
    np.testing.assert_array_equal(train.X, train_X0)
