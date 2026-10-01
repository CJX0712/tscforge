"""Tier-0 optional backend registry with graceful degradation.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from tscforge.core.errors import BackendUnavailableError
from tscforge.core.types import TimeSeriesBatch


@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    version: str = ""
    reason: str = ""


def _find(spec: str) -> bool:
    return importlib.util.find_spec(spec) is not None


def _aeon_rocket_kwargs(n_kernels: int, seed: int) -> dict:
    """Build aeon RocketClassifier kwargs across API versions.

    aeon renamed ``num_kernels`` -> ``n_kernels`` in 1.x; probe the signature
    instead of hard-coding one spelling.
    """
    import inspect

    from aeon.classification.convolution_based import RocketClassifier

    params = inspect.signature(RocketClassifier.__init__).parameters
    key = "n_kernels" if "n_kernels" in params else "num_kernels"
    return {key: n_kernels, "random_state": seed}


def _probe_aeon() -> BackendStatus:
    if not _find("aeon"):
        return BackendStatus("aeon", False, reason="module not installed")
    try:
        import aeon
        from aeon.classification.convolution_based import RocketClassifier

        X = np.random.default_rng(0).standard_normal((10, 1, 16))
        y = np.array([0, 1] * 5)
        clf = RocketClassifier(**_aeon_rocket_kwargs(50, 0))
        clf.fit(X, y)
        clf.predict(X)
        return BackendStatus("aeon", True, version=str(aeon.__version__))
    except Exception as exc:  # broken install / API drift
        return BackendStatus("aeon", False, reason=f"probe failed: {exc}")


def _probe_tslearn() -> BackendStatus:
    if not _find("tslearn"):
        return BackendStatus("tslearn", False, reason="module not installed")
    try:
        import tslearn
        from tslearn.neighbors import KNeighborsTimeSeriesClassifier

        X = np.random.default_rng(0).standard_normal((10, 16, 1))
        y = np.array([0, 1] * 5)
        clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
        clf.fit(X, y)
        clf.predict(X)
        return BackendStatus("tslearn", True, version=str(tslearn.__version__))
    except Exception as exc:
        return BackendStatus("tslearn", False, reason=f"probe failed: {exc}")


_PROBES = {"aeon": _probe_aeon, "tslearn": _probe_tslearn}


def available_backends() -> dict[str, BackendStatus]:
    """Probe every optional Tier-0 backend; never raises."""
    return {name: probe() for name, probe in _PROBES.items()}


def get_backend(name: str, n_kernels: int = 1000, seed: int = 42):
    """Return a fitted-ready Tier-0 classifier wrapped to the TscForge protocol.

    Raises BackendUnavailableError if the backend is missing or broken;
    callers must catch and degrade to the pure-numpy path.
    """
    status = available_backends().get(name)
    if status is None or not status.available:
        reason = status.reason if status is not None else f"unknown backend {name!r}"
        raise BackendUnavailableError(f"backend {name!r} unavailable: {reason}")
    if name == "aeon":
        from aeon.classification.convolution_based import RocketClassifier

        return _AeonWrapper(RocketClassifier(**_aeon_rocket_kwargs(n_kernels, seed)))
    if name == "tslearn":
        from tslearn.neighbors import KNeighborsTimeSeriesClassifier

        return _TslearnWrapper(KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw"))
    raise BackendUnavailableError(f"unknown backend {name!r}")


class _AeonWrapper:
    """Adapts aeon [N, C, T] to TscForge fit/predict."""

    def __init__(self, clf) -> None:
        self.clf = clf

    def fit(self, train: TimeSeriesBatch, val: TimeSeriesBatch | None = None) -> None:
        self.clf.fit(np.ascontiguousarray(train.X, dtype=np.float64), train.y)

    def predict(self, X: TimeSeriesBatch) -> np.ndarray:
        return np.asarray(self.clf.predict(np.ascontiguousarray(X.X, dtype=np.float64)))


class _TslearnWrapper:
    """Adapts tslearn [N, T, C] to TscForge fit/predict."""

    def __init__(self, clf) -> None:
        self.clf = clf

    def fit(self, train: TimeSeriesBatch, val: TimeSeriesBatch | None = None) -> None:
        self.clf.fit(train.X.transpose(0, 2, 1), train.y)

    def predict(self, X: TimeSeriesBatch) -> np.ndarray:
        return np.asarray(self.clf.predict(X.X.transpose(0, 2, 1)))
