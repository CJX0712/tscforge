"""TinyRocket: a faithful pure-numpy ROCKET variant (Dempster et al., 2020).

K random convolution kernels (length in {7, 9, 11}, dilation = 2^k,
bias ~ N(0, 1), weights ~ N(0, 1), 50% padding probability), per-kernel
ReLU then PPV + Max pooling. Convolution uses sliding_window_view + einsum
(NOT FFT) so the reduction order is fixed and bit-deterministic.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class KernelSet:
    """Sampled ROCKET kernels (all arrays are [K] or [K, Lmax])."""

    lengths: np.ndarray
    dilations: np.ndarray
    weights: np.ndarray  # [K, Lmax], zero-padded beyond each kernel's length
    biases: np.ndarray
    paddings: np.ndarray  # bool

    @property
    def n_kernels(self) -> int:
        return int(self.lengths.shape[0])


def sample_kernels(rng: np.random.Generator, n_kernels: int, T: int) -> KernelSet:
    """Sample K kernels deterministically from the given RNG stream.

    Follows the canonical ROCKET scheme (Dempster et al., 2020): kernel length
    scales with the series length (lengths in [2, min(0.4*T, 100)+1]) and
    dilation spans the feasible range 2^[0, floor(log2((T-1)/(L-1)))] so each
    kernel can integrate over a meaningful fraction of the signal. Short,
    fixed-length kernels (e.g. 7-11) cannot capture global structure such as
    the phase of a long-period sinusoid and collapse PPV to ~0.5.
    """
    max_length = int(min(0.4 * T, 100))
    lengths = rng.integers(0, max_length, size=n_kernels) + 2  # [2, max_length+1]
    dilations = np.empty(n_kernels, dtype=np.int64)
    for i, length in enumerate(lengths):
        num_d = int(np.floor(np.log2(max((T - 1) / (length - 1), 1.0))))
        dilations[i] = 2 ** int(rng.integers(0, num_d + 1))
    lmax = int(lengths.max())
    weights = rng.standard_normal((n_kernels, lmax))
    for i, length in enumerate(lengths):
        weights[i, int(length) :] = 0.0
    biases = rng.standard_normal(n_kernels)
    paddings = rng.integers(0, 2, n_kernels).astype(bool)
    return KernelSet(lengths, dilations, weights, biases, paddings)


def _conv_one(
    x: np.ndarray, length: int, dilation: int, weight: np.ndarray, bias: float, pad: bool
) -> np.ndarray:
    """Dilated convolution of one series with one kernel -> activation array."""
    eff = (length - 1) * dilation + 1
    if pad:
        p = (eff - 1) // 2
        x = np.pad(x, (p, p), mode="constant", constant_values=0.0)
    if x.shape[0] < eff:
        x = np.pad(x, (0, eff - x.shape[0]), mode="constant", constant_values=0.0)
    windows = sliding_window_view(x, eff)[::dilation]  # [P, eff]
    w = weight[:length]  # kernel weights
    # Dilated taps: every `dilation`-th element inside each window -> [P, length]
    taps = windows[:, ::dilation]
    act = np.einsum("pl,l->p", taps, w, optimize=True) + bias
    return act


def transform(X: np.ndarray, kernels: KernelSet) -> np.ndarray:
    """Transform [N, C, T] batch into ROCKET features [N, K * n_pool].

    n_pool = 2 (PPV + Max) when multi-pool is used, else 1 (PPV only).
    Multi-pool is a property of the consumer; transform always returns both,
    and :func:`transform_pool` selects the pooling variant.
    """
    n_pool = 2
    n, c, _ = X.shape
    k = kernels.n_kernels
    out = np.empty((n, k * n_pool), dtype=np.float64)
    X64 = np.ascontiguousarray(X, dtype=np.float64)
    for i in range(n):
        row = out[i]
        for ki in range(k):
            length = int(kernels.lengths[ki])
            dilation = int(kernels.dilations[ki])
            weight = kernels.weights[ki]
            bias = float(kernels.biases[ki])
            pad = bool(kernels.paddings[ki])
            acts_all = [
                _conv_one(X64[i, ch], length, dilation, weight, bias, pad) for ch in range(c)
            ]
            acts = np.concatenate(acts_all)
            relu = np.maximum(acts, 0.0)
            row[ki * n_pool] = float(np.mean(relu > 0.0))
            row[ki * n_pool + 1] = float(relu.max()) if relu.size else 0.0
    return out


def transform_pool(X: np.ndarray, kernels: KernelSet, multi_pool: bool) -> np.ndarray:
    """transform() with pooling ablation: multi_pool=False keeps PPV only."""
    feats = transform(X, kernels)
    if multi_pool:
        return feats
    return feats[:, 0::2].copy()
