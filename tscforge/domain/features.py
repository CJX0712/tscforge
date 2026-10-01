"""catch22-lite handcrafted statistical features (pure numpy, ~13 dims).

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

N_FEATURES_PER_CHANNEL = 13


def _channel_features(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    mean = float(x.mean())
    std = float(x.std()) + 1e-12
    z = (x - mean) / std
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4) - 3.0)
    d = np.diff(x)
    mean_abs_change = float(np.mean(np.abs(d)))
    zcr = float(np.mean((x[:-1] * x[1:]) < 0))
    denom = float(np.dot(x, x)) + 1e-12
    ac1 = float(np.dot(x[:-1], x[1:]) / denom)
    ac2 = float(np.dot(x[:-2], x[2:]) / denom)
    q25, q75 = (float(v) for v in np.quantile(x, [0.25, 0.75]))
    iqr = q75 - q25
    slope = float(np.polyfit(np.arange(n, dtype=np.float64), x, 1)[0])
    return np.array(
        [
            mean,
            std,
            skew,
            kurt,
            q25,
            q75,
            iqr,
            mean_abs_change,
            zcr,
            ac1,
            ac2,
            slope,
            float(x.max() - x.min()),
        ],
        dtype=np.float64,
    )


def extract_handcrafted(X: np.ndarray) -> np.ndarray:
    """Extract per-channel statistical features -> [N, C * 13]."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"expected [N, C, T], got shape {X.shape}")
    n, c, _ = X.shape
    out = np.empty((n, c * N_FEATURES_PER_CHANNEL), dtype=np.float64)
    for i in range(n):
        blocks = [_channel_features(X[i, ch]) for ch in range(c)]
        out[i] = np.concatenate(blocks)
    return out
