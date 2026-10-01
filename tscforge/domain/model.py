"""PRISM flagship: Pooling-fused RIdge with dtw-diStance Merging.

Components (each independently ablatable via TscConfig):
  A. use_multi_pool   - PPV + Max pooling concatenation (vs PPV only)
  C. use_dtw_features - per-class K-medoid DTW prototype distances appended
                        as extra features; the DTW block's relative importance is
                        reported as a Ridge-coefficient proxy (``dtw_lambda``),
                        not a separately grid-searched weight
  D. use_norm_gate    - per-dataset normalization (z vs min-max) selected on val

Leakage safety: everything data-dependent is fit on train/val only; holdout
enters exclusively through predict().

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from tscforge.core.config import TscConfig
from tscforge.core.errors import TscForgeError
from tscforge.core.seed import get_rng
from tscforge.core.types import TimeSeriesBatch
from tscforge.domain.dtw import prototype_distances, select_prototypes_k
from tscforge.domain.rocket import KernelSet, sample_kernels, transform_pool

_PILOT_KERNELS = 200
_ALPHAS = np.logspace(-3, 3, 13)
_NORM_MODES = ("z", "minmax")
# scale guard: pairwise DTW beyond this size is refused (degrade path)
MAX_PAIRWISE_N = 500


def normalize_series(X: np.ndarray, mode: str) -> np.ndarray:
    """Normalize each series independently. mode in {none, z, minmax}."""
    X = np.asarray(X, dtype=np.float64)
    if mode == "none":
        return X.copy()
    out = np.empty_like(X)
    flat = X.reshape(X.shape[0], -1)
    for i in range(X.shape[0]):
        row = flat[i]
        if mode == "z":
            sd = row.std()
            out[i] = (row - row.mean()) / (sd + 1e-9)
        elif mode == "minmax":
            span = row.max() - row.min()
            out[i] = (row - row.min()) / (span + 1e-9)
        else:
            raise TscForgeError(f"unknown norm mode {mode!r}")
    return out.reshape(X.shape)


def normalize_batch(batch: TimeSeriesBatch, mode: str) -> TimeSeriesBatch:
    return TimeSeriesBatch(X=normalize_series(batch.X, mode), y=batch.y.copy())


class PrismClassifier:
    """Flagship classifier. fit(train, val) then predict(holdout)."""

    def __init__(self, cfg: TscConfig, pairwise_train: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self._pairwise_train = pairwise_train
        self.norm_mode: str = "z"
        self.kernels: KernelSet | None = None
        self.protos: np.ndarray | None = None
        self.dtw_mu: np.ndarray | None = None
        self.dtw_sd: np.ndarray | None = None
        self.dtw_lambda: float = 0.0
        self.dtw_active: bool = False
        self.ridge: RidgeClassifierCV | None = None
        self.scaler: StandardScaler | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, train: TimeSeriesBatch, val: TimeSeriesBatch | None = None) -> None:
        cfg = self.cfg
        # D. normalization gate (selected on val only)
        self.norm_mode = "z"
        if cfg.use_norm_gate and val is not None:
            self.norm_mode = self._select_norm_mode(train, val)

        tr = normalize_batch(train, self.norm_mode)
        va = normalize_batch(val, self.norm_mode) if val is not None else None

        # A+C. TinyRocket multi-pool features
        rng = get_rng("rocket", cfg.seed)
        self.kernels = sample_kernels(rng, cfg.n_kernels, tr.length)
        feats_tr = transform_pool(tr.X, self.kernels, cfg.use_multi_pool)

        # C. DTW prototype distance block
        dtw_block_tr: np.ndarray | None = None
        dtw_block_va: np.ndarray | None = None
        if cfg.use_dtw_features:
            try:
                dtw_block_tr, dtw_block_va = self._fit_dtw_block(tr, va)
            except TscForgeError as exc:  # scale-guard degrade path
                warnings.warn(f"DTW feature block disabled: {exc}", UserWarning, stacklevel=2)
                self.dtw_active = False

        x_tr = self._combine(feats_tr, dtw_block_tr)
        # Standard scaling of the combined feature block (ROCKET's standard
        # practice, and what makes Ridge stable across PPV/Max/DTW scales).
        # Fitted on train only -> leak-safe.
        self.scaler = StandardScaler().fit(x_tr)
        self.ridge = RidgeClassifierCV(alphas=_ALPHAS)
        self.ridge.fit(self.scaler.transform(x_tr), tr.y)

        # Report a relative DTW-importance proxy: mean |Ridge coef| on the DTW
        # block vs on the ROCKET block. 0.0 => DTW block contributed nothing.
        if dtw_block_tr is not None and hasattr(self.ridge, "coef_"):
            # RidgeClassifierCV.coef_ is (n_features,) for binary, (n_classes, n_features) else.
            coef = np.atleast_2d(np.abs(np.asarray(self.ridge.coef_))).mean(axis=0)
            n_rk = feats_tr.shape[1]
            rk_mean = float(coef[:n_rk].mean()) if n_rk else 0.0
            dtw_mean = float(coef[n_rk:].mean()) if coef.shape[0] > n_rk else 0.0
            self.dtw_lambda = dtw_mean / (rk_mean + 1e-9)
        else:
            self.dtw_lambda = 0.0

    def _fit_dtw_block(
        self, tr: TimeSeriesBatch, va: TimeSeriesBatch | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if tr.n_samples > MAX_PAIRWISE_N:
            raise TscForgeError(f"prototype DTW too large (n={tr.n_samples} > {MAX_PAIRWISE_N})")
        protos = select_prototypes_k(tr.X, tr.y, self.cfg.dtw_k, self.cfg.dtw_band, self.cfg.seed)
        d_tr = prototype_distances(tr.X, protos, self.cfg.dtw_band)
        self.protos = protos
        self.dtw_active = True
        d_va = prototype_distances(va.X, protos, self.cfg.dtw_band) if va is not None else None
        # standardize with train statistics only (leak-safe)
        self.dtw_mu = d_tr.mean(axis=0)
        self.dtw_sd = d_tr.std(axis=0) + 1e-9
        d_tr_std = (d_tr - self.dtw_mu) / self.dtw_sd
        d_va_std = (d_va - self.dtw_mu) / self.dtw_sd if d_va is not None else None
        return d_tr_std, d_va_std

    # --------------------------------------------------------------- predict
    def predict(self, X: TimeSeriesBatch) -> np.ndarray:
        if self.ridge is None or self.kernels is None:
            raise RuntimeError("PrismClassifier must be fit before predict()")
        xn = normalize_batch(X, self.norm_mode)
        feats = transform_pool(xn.X, self.kernels, self.cfg.use_multi_pool)
        combined = feats
        if self.dtw_active and self.protos is not None:
            d = prototype_distances(xn.X, self.protos, self.cfg.dtw_band)
            d_std = (d - self.dtw_mu) / self.dtw_sd
            combined = self._combine(feats, d_std)
        if self.scaler is not None:
            combined = self.scaler.transform(combined)
        return np.asarray(self.ridge.predict(combined))

    # -------------------------------------------------------------- internals
    @staticmethod
    def _combine(feats: np.ndarray, d_block: np.ndarray | None) -> np.ndarray:
        if d_block is None:
            return feats
        return np.hstack([feats, d_block])

    def _select_norm_mode(self, train: TimeSeriesBatch, val: TimeSeriesBatch) -> str:
        """Pilot rocket (small kernel pool) on both norm modes, pick best on val."""
        pilot_rng = get_rng("rocket_pilot", self.cfg.seed)
        pilot_kernels = sample_kernels(pilot_rng, _PILOT_KERNELS, train.length)
        best_mode, best_acc = _NORM_MODES[0], -1.0
        for mode in _NORM_MODES:
            tr = normalize_batch(train, mode)
            va = normalize_batch(val, mode)
            f_tr = transform_pool(tr.X, pilot_kernels, multi_pool=True)
            f_va = transform_pool(va.X, pilot_kernels, multi_pool=True)
            ridge = RidgeClassifierCV(alphas=_ALPHAS)
            sc = StandardScaler().fit(f_tr)
            ridge.fit(sc.transform(f_tr), tr.y)
            acc = float(np.mean(ridge.predict(sc.transform(f_va)) == va.y))
            if acc > best_acc:
                best_mode, best_acc = mode, acc
        return best_mode
