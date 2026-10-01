"""Data structures.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SplitName = Literal["train", "val", "holdout"]


@dataclass
class TimeSeriesBatch:
    """A batch of time series.

    X: array of shape [N, C, T] (N samples, C channels, T timesteps).
    y: integer label array of shape [N].
    """

    X: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        if self.X.ndim != 3:
            raise ValueError(f"X must be [N, C, T], got shape {self.X.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"y must be [N], got shape {self.y.shape}")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X/y length mismatch: X has {self.X.shape[0]} samples, y has {self.y.shape[0]}"
            )
        if self.X.shape[0] == 0:
            raise ValueError("empty batch (N == 0)")

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def length(self) -> int:
        return int(self.X.shape[2])

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.y)

    def subset(self, idx: np.ndarray) -> TimeSeriesBatch:
        """Return a sub-batch selected by integer index array."""
        return TimeSeriesBatch(X=self.X[idx], y=self.y[idx])
