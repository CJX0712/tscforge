"""Classifier protocol (structural interface).

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from tscforge.core.types import TimeSeriesBatch


@runtime_checkable
class Classifier(Protocol):
    """Every TSC model must implement fit(train, val) + predict(X).

    The signature is leakage-safe by construction: holdout data can only ever
    enter through predict(), never through fit().
    """

    def fit(self, train: TimeSeriesBatch, val: TimeSeriesBatch | None = None) -> None: ...

    def predict(self, X: TimeSeriesBatch) -> np.ndarray: ...
