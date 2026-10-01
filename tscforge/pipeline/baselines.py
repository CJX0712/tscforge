"""Baseline classifiers (gate anchors) and Tier-0 report-only models.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

from tscforge.core.types import TimeSeriesBatch
from tscforge.domain.dtw import cross_dtw, pairwise_dtw
from tscforge.domain.features import extract_handcrafted


class OneNNEd:
    """1-nearest-neighbor with Euclidean distance (pure numpy)."""

    def __init__(self, band: int = 0) -> None:
        self.band = band  # unused; kept for uniform construction
        self._train: TimeSeriesBatch | None = None

    def fit(self, train: TimeSeriesBatch, val: TimeSeriesBatch | None = None) -> None:
        self._train = train

    def predict(self, X: TimeSeriesBatch) -> np.ndarray:
        assert self._train is not None, "fit() required"
        a = X.X.reshape(X.n_samples, -1).astype(np.float64)
        b = self._train.X.reshape(self._train.n_samples, -1).astype(np.float64)
        d2 = np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2.0 * (a @ b.T)
        return np.asarray(self._train.y[np.argmin(d2, axis=1)])


class OneNNDtw:
    """1NN with banded DTW (pure numpy). Optional precomputed train pairwise."""

    def __init__(self, band: int = 32, pairwise_train: np.ndarray | None = None) -> None:
        self.band = band
        self._train: TimeSeriesBatch | None = None
        self._pw = pairwise_train

    def fit(self, train: TimeSeriesBatch, val: TimeSeriesBatch | None = None) -> None:
        self._train = train
        if self._pw is None:
            self._pw = pairwise_dtw(train.X, self.band)

    def predict(self, X: TimeSeriesBatch) -> np.ndarray:
        assert self._train is not None, "fit() required"
        d = cross_dtw(X.X, self._train.X, self.band)
        return np.asarray(self._train.y[np.argmin(d, axis=1)])


class LogRegFlat:
    """Logistic regression on raw flattened series (sklearn, lbfgs)."""

    def __init__(self, seed: int = 0) -> None:
        from sklearn.linear_model import LogisticRegression

        self._clf = LogisticRegression(max_iter=2000, random_state=seed)

    def fit(self, train: TimeSeriesBatch, val: TimeSeriesBatch | None = None) -> None:
        self._clf.fit(train.X.reshape(train.n_samples, -1), train.y)

    def predict(self, X: TimeSeriesBatch) -> np.ndarray:
        return np.asarray(self._clf.predict(X.X.reshape(X.n_samples, -1)))


class RfHandcrafted:
    """RandomForest over catch22-lite handcrafted features."""

    def __init__(self, seed: int = 0) -> None:
        from sklearn.ensemble import RandomForestClassifier

        # n_jobs=1: joblib multiprocessing is unstable in restricted Windows envs
        self._clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)

    def fit(self, train: TimeSeriesBatch, val: TimeSeriesBatch | None = None) -> None:
        self._clf.fit(extract_handcrafted(train.X), train.y)

    def predict(self, X: TimeSeriesBatch) -> np.ndarray:
        return np.asarray(self._clf.predict(extract_handcrafted(X.X)))
