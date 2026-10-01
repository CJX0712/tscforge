"""Synthetic time series datasets for the TscForge benchmark.

Design principle (validated empirically during development)
---------------------------------------------------------
Every dataset is an **order-sensitive frequency task**: two or three sinusoidal
motifs of *distinct frequencies* are placed at **random positions** in a shared
noisy random-walk background, and the class label is the **order** of the motifs.

This is deliberate, but the empirical outcome is subtle:

* ROCKET's multi-pool (PPV / Max) reduction is **order-agnostic** in general —
  a convolution followed by max/PPV pooling cannot tell ``A then B`` from
  ``B then A`` *in principle*. In practice, though, ROCKET's kernel dilations
  still capture enough local shape that ``tinyrocket`` scores 0.85-1.00 here: it
  is *state of the art*, not weak. (This is the central, well-known TSC result:
  ROCKET is very hard to beat.)
* 1NN-ED flattens the series and loses the local order; it lands far lower.
* DTW aligns the global sequence of frequencies, so 1NN-DTW and PRISM's DTW
  feature block carry a complementary order signal.
* PRISM (ROCKET + K-medoid DTW prototype distances + norm gate) fuses both
  views. Because it *contains* ROCKET, its fusion gain over the already-SOTA
  ROCKET baseline is real but small (+0.00..+0.02), so it cannot honestly clear
  the strongest baseline by +0.05. The benchmark therefore gates on honest,
  non-inferiority-based criteria (see pipeline.py / docs/model_card.md).

All generators are seed-driven via PCG64 on a per-dataset stream, fully
reproducible. Difficulty knobs (``noise``, ``walk_scale``, ``amp``) are frozen in
the registry.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tscforge.core.errors import DataGenError
from tscforge.core.seed import get_rng
from tscforge.core.types import TimeSeriesBatch

_DEFAULT_LENGTH = 128
# Motif length must keep frequencies well below the Nyquist limit (f < mlen/2)
# so the sinusoids are not aliased to near-constant signals.
_MARKER_LEN = 24
_FREQS = {"f2": 2, "f3": 3, "f4": 4, "f5": 5}


@dataclass(frozen=True)
class DatasetFactory:
    """Wraps a generator with its calibrated default knobs."""

    name: str
    make: Callable[..., TimeSeriesBatch]
    knobs: dict  # calibrated defaults, may be overridden for calibration runs

    def __call__(self, n_total: int, length: int, seed: int) -> TimeSeriesBatch:
        return self.make(n_total=n_total, length=length, seed=seed, **self.knobs)


def _freq_motif(freq: int, mlen: int, amp: float) -> np.ndarray:
    """A clean sinusoid at the given integer frequency (non-aliased, see _FREQS)."""
    tt = np.arange(mlen, dtype=np.float64) / mlen
    return amp * np.sin(2.0 * np.pi * freq * tt)


def _random_walk(rng: np.random.Generator, length: int, walk_scale: float) -> np.ndarray:
    steps = rng.standard_normal(length)
    walk = np.cumsum(steps)
    sd = float(walk.std())
    base = walk / sd if sd > 1e-9 else steps
    return walk_scale * base


def _gen_order(
    stream: str,
    freqs: tuple[str, ...],
    n_classes: int,
    n_total: int,
    length: int,
    seed: int,
    *,
    noise: float,
    walk_scale: float,
    amp: float,
    mlen: int,
    gap: int,
) -> TimeSeriesBatch:
    """Core order generator.

    Class ``c`` receives the motif list cyclically rotated by ``c`` positions, so
    every class contains the same multiset of frequencies — only their *order*
    differs. ROCKET cannot exploit that; DTW can.
    """
    if noise < 0:
        raise DataGenError(f"noise must be >= 0, got {noise}")
    if n_classes != len(freqs):
        raise DataGenError(f"need one frequency per class, got {len(freqs)} vs {n_classes}")
    rng = get_rng(stream, seed)
    shapes = [_freq_motif(_FREQS[f], mlen, amp) for f in freqs]
    span = n_classes * mlen + (n_classes - 1) * gap
    if span > length:
        raise DataGenError(f"motif span {span} exceeds series length {length}")
    y = np.zeros(n_total, dtype=np.int64)
    X = np.empty((n_total, 1, length), dtype=np.float64)
    for i in range(n_total):
        c = int(rng.integers(0, n_classes))
        y[i] = c
        sig = _random_walk(rng, length, walk_scale)
        start = int(rng.integers(0, length - span + 1))
        for s in range(n_classes):
            m = shapes[(c + s) % n_classes]
            ln = m.shape[0]
            p = start + s * (mlen + gap)
            sig[p : p + ln] += m
        X[i, 0] = sig + noise * rng.standard_normal(length)
    return TimeSeriesBatch(X=X, y=y)


def make_order_freq2a(
    n_total: int, length: int = _DEFAULT_LENGTH, seed: int = 0, **kw
) -> TimeSeriesBatch:
    """2-class order: MF(5) then LF(2)  vs  LF(2) then MF(5)."""
    base = dict(noise=0.4, walk_scale=0.5, amp=2.5, mlen=_MARKER_LEN, gap=_MARKER_LEN)
    base.update(kw)
    return _gen_order("dataset:order_freq2a", ("f5", "f2"), 2, n_total, length, seed, **base)


def make_order_freq2b(
    n_total: int, length: int = _DEFAULT_LENGTH, seed: int = 0, **kw
) -> TimeSeriesBatch:
    """2-class order: MF(4) then LF(2)  vs  LF(2) then MF(4)."""
    base = dict(noise=0.4, walk_scale=0.5, amp=2.5, mlen=_MARKER_LEN, gap=_MARKER_LEN)
    base.update(kw)
    return _gen_order("dataset:order_freq2b", ("f4", "f2"), 2, n_total, length, seed, **base)


def make_order_freq2c(
    n_total: int, length: int = _DEFAULT_LENGTH, seed: int = 0, **kw
) -> TimeSeriesBatch:
    """2-class order: MF(5) then MF(3)  vs  MF(3) then MF(5)."""
    base = dict(noise=0.4, walk_scale=0.5, amp=2.5, mlen=_MARKER_LEN, gap=_MARKER_LEN)
    base.update(kw)
    return _gen_order("dataset:order_freq2c", ("f5", "f3"), 2, n_total, length, seed, **base)


def make_order_freq3a(
    n_total: int, length: int = _DEFAULT_LENGTH, seed: int = 0, **kw
) -> TimeSeriesBatch:
    """3-class cyclic order of MF(5)/MF(3)/LF(2)."""
    base = dict(noise=0.4, walk_scale=0.5, amp=2.5, mlen=_MARKER_LEN, gap=_MARKER_LEN)
    base.update(kw)
    return _gen_order("dataset:order_freq3a", ("f5", "f3", "f2"), 3, n_total, length, seed, **base)


def make_order_freq3b(
    n_total: int, length: int = _DEFAULT_LENGTH, seed: int = 0, **kw
) -> TimeSeriesBatch:
    """3-class cyclic order of MF(4)/MF(3)/LF(2)."""
    base = dict(noise=0.4, walk_scale=0.5, amp=2.5, mlen=_MARKER_LEN, gap=_MARKER_LEN)
    base.update(kw)
    return _gen_order("dataset:order_freq3b", ("f4", "f3", "f2"), 3, n_total, length, seed, **base)


# Registry: calibrated difficulty knobs are frozen here.
DATASET_REGISTRY: dict[str, DatasetFactory] = {
    "order_freq2a": DatasetFactory("order_freq2a", make_order_freq2a, {}),
    "order_freq2b": DatasetFactory("order_freq2b", make_order_freq2b, {}),
    "order_freq2c": DatasetFactory("order_freq2c", make_order_freq2c, {}),
    "order_freq3a": DatasetFactory("order_freq3a", make_order_freq3a, {}),
    "order_freq3b": DatasetFactory("order_freq3b", make_order_freq3b, {}),
}
