"""Tier-0 backend registry tests: graceful availability + degrade.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np
import pytest

from tscforge.core.errors import BackendUnavailableError
from tscforge.core.types import TimeSeriesBatch
from tscforge.domain.backends.registry import available_backends, get_backend


def test_available_backends_returns_dict():
    statuses = available_backends()
    assert set(statuses.keys()) >= {"aeon", "tslearn"}
    for s in statuses.values():
        assert s.available in (True, False)


def test_aeon_probe_consistency():
    statuses = available_backends()
    aeon = statuses["aeon"]
    if aeon.available:
        # get_backend must succeed when probe says available
        clf = get_backend("aeon", n_kernels=50, seed=0)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((12, 1, 20))
        y = np.array([0, 1] * 6)
        batch = TimeSeriesBatch(X=X, y=y)
        clf.fit(batch, None)
        pred = np.asarray(clf.predict(batch))
        assert pred.shape == (12,)
    else:
        with pytest.raises(BackendUnavailableError):
            get_backend("aeon", n_kernels=50, seed=0)


def test_tslearn_degrade_path():
    statuses = available_backends()
    ts = statuses["tslearn"]
    if not ts.available:
        with pytest.raises(BackendUnavailableError):
            get_backend("tslearn", seed=0)
