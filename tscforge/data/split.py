"""Stratified train/val/holdout split.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

from tscforge.core.seed import get_rng
from tscforge.core.types import SplitName, TimeSeriesBatch


def stratified_split(
    batch: TimeSeriesBatch,
    seed: int,
    ratios: tuple = (0.4, 0.2, 0.4),
) -> dict[SplitName, TimeSeriesBatch]:
    """Per-class shuffled split preserving class ratios across the three arms.

    ratios = (train, val, holdout); each per-class count is split with
    floor-then-remainder allocation so every arm receives samples of every
    class whenever the class count allows it.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    rng = get_rng("split", seed)
    classes = batch.classes
    n_classes = len(classes)
    min_arm = min(int(np.floor(r * batch.n_samples / n_classes)) for r in ratios)
    if min_arm < 1:
        raise ValueError("not enough samples per class for the requested split")

    idx_by_arm: dict[SplitName, list[int]] = {"train": [], "val": [], "holdout": []}
    names: tuple = ("train", "val", "holdout")
    for c in classes:
        cls_idx = np.flatnonzero(batch.y == c)
        perm = rng.permutation(len(cls_idx))
        cls_idx = cls_idx[perm]
        start = 0
        for name, r in zip(names, ratios, strict=False):
            take = int(np.floor(len(cls_idx) * r))
            idx_by_arm[name].extend(cls_idx[start : start + take].tolist())
            start += take
        # remainder (from floor rounding) goes to holdout
        idx_by_arm["holdout"].extend(cls_idx[start:].tolist())

    out: dict[SplitName, TimeSeriesBatch] = {}
    for name in names:
        idx = np.array(sorted(idx_by_arm[name]), dtype=np.int64)
        out[name] = batch.subset(idx)
    return out
