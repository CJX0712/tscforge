"""Calibration probe for the redesigned (local-marker) datasets.

Reports per-model accuracy at a SMALL size for fast iteration. Goal:
  1NN-ED in [0.55, 0.90]  AND  PRISM >= max(gate baselines) + 0.05.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

from tscforge.core.config import TscConfig
from tscforge.core.seed import set_global_seed
from tscforge.data.generators import DATASET_REGISTRY
from tscforge.data.split import stratified_split
from tscforge.domain.model import PrismClassifier
from tscforge.pipeline.baselines import LogRegFlat, OneNNDtw, OneNNEd, RfHandcrafted

LENGTH = 128
SEEDS = [101, 202, 303]
N = {
    "order_freq2a": 60,
    "order_freq2b": 60,
    "order_freq2c": 60,
    "order_freq3a": 90,
    "order_freq3b": 90,
}


def run_one(ds, seed):
    set_global_seed(seed)
    n = N[ds]
    batch = DATASET_REGISTRY[ds](n_total=n, length=LENGTH, seed=seed)
    sp = stratified_split(batch, seed)
    tr, va, ho = sp["train"], sp["val"], sp["holdout"]
    out = {}
    for name, m in (
        ("1nn_ed", OneNNEd()),
        ("1nn_dtw", OneNNDtw(band=16)),
        ("logreg_flat", LogRegFlat(seed=seed)),
        ("rf_handcrafted", RfHandcrafted(seed=seed)),
    ):
        m.fit(tr, va)
        out[name] = float(np.mean(m.predict(ho) == ho.y))
    cfg = TscConfig(seed=seed)
    prism = PrismClassifier(cfg)
    prism.fit(tr, va)
    out["prism"] = float(np.mean(prism.predict(ho) == ho.y))
    # bare tinyrocket (DTW + norm-gate off)
    tr2, va2, ho2 = (
        stratified_split(DATASET_REGISTRY[ds](n_total=n, length=LENGTH, seed=seed), seed)
        if False
        else (tr, va, ho)
    )
    tr3, va3, ho3 = tr, va, ho
    cfg2 = TscConfig(seed=seed).with_overrides(use_dtw_features=False, use_norm_gate=False)
    trk = PrismClassifier(cfg2)
    trk.fit(tr3, va3)
    out["tinyrocket"] = float(np.mean(trk.predict(ho3) == ho3.y))
    return out


def main():
    for ds in DATASET_REGISTRY:
        accs = {
            k: []
            for k in ("1nn_ed", "1nn_dtw", "logreg_flat", "rf_handcrafted", "tinyrocket", "prism")
        }
        for seed in SEEDS:
            r = run_one(ds, seed)
            for k, v in r.items():
                accs[k].append(v)
        means = {k: float(np.mean(v)) for k, v in accs.items()}
        base = max(
            means[n] for n in ("tinyrocket", "1nn_dtw", "rf_handcrafted", "1nn_ed", "logreg_flat")
        )
        print(
            f"{ds:12s} 1nn_ed={means['1nn_ed']:.2f} 1nn_dtw={means['1nn_dtw']:.2f} "
            f"logreg={means['logreg_flat']:.2f} rf={means['rf_handcrafted']:.2f} "
            f"tinyrocket={means['tinyrocket']:.2f} | PRISM={means['prism']:.2f} "
            f"best_base={base:.2f} delta={means['prism'] - base:+.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
