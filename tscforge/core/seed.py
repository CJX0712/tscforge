"""Deterministic seed entry points.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import os
import random
import zlib

import numpy as np

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _pin_threads() -> None:
    """Pin BLAS/OpenMP thread counts to 1 for deterministic reduction order."""
    for var in _THREAD_VARS:
        os.environ[var] = "1"


def set_global_seed(seed: int) -> None:
    """Set every global RNG (python random, numpy legacy) plus thread pinning.

    This is the unique entry point for global determinism; component-level
    randomness must use :func:`get_rng` with named streams instead.
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    _pin_threads()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def get_rng(stream: str, seed: int) -> np.random.Generator:
    """Return an isolated PCG64 generator for a named stream.

    Different streams never share RNG state, so adding a new component cannot
    perturb the sampling sequence of existing components.
    """
    stream_tag = zlib.crc32(stream.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng([int(seed), stream_tag])
