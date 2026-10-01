"""Pure numpy DTW with Sakoe-Chiba banding, batched over pairs.

The DP is vectorized: one Python loop over rows (T iterations) x one over
band columns (2*band+1), with all pairs processed as flat numpy arrays.
This keeps reduction order fixed (single-threaded numpy) and runs fast
enough for N~100, T=128 (seconds per full distance matrix).

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

# Unreachable-cell sentinel; real distances must stay far below it.
_BIG = 1e15
_MAX_AMPLITUDE = 1e9  # input guard: T * 2 * amp << _BIG must hold


def _validate_band(T: int, band: int) -> int:
    if band <= 0:
        raise ValueError(f"band must be > 0, got {band}")
    return min(band, T - 1)


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int) -> float:
    """Banded DTW distance between two equal-length univariate series."""
    a = np.ascontiguousarray(a, dtype=np.float64).ravel()
    b = np.ascontiguousarray(b, dtype=np.float64).ravel()
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"length mismatch: {a.shape[0]} vs {b.shape[0]}")
    return float(_banded_dtw_batch(a[None, :], b[None, :], band)[0])


def _banded_dtw_batch(A: np.ndarray, B: np.ndarray, band: int) -> np.ndarray:
    """Batched banded DTW over paired sequences.

    A, B: [P, T] aligned pairs. Returns [P] distances.

    Unreachable cells keep a large FINITE sentinel BIG (never inf: inf would
    propagate through `cost + inf` and destroy reachable paths). Row 0 gets
    an explicit base cell so only (0,0) may start a path.
    """
    P, T = A.shape
    band = _validate_band(T, band)
    w = 2 * band + 1
    big = _BIG

    A64 = np.ascontiguousarray(A, dtype=np.float64)
    B64 = np.ascontiguousarray(B, dtype=np.float64)
    cols = np.arange(w) - band  # j offset relative to row i

    def _row_cost(i: int) -> np.ndarray:
        js = i + cols
        valid = (js >= 0) & (js < T)
        jsc = np.clip(js, 0, T - 1)
        cost = np.abs(A64[:, i : i + 1] - B64[:, jsc])  # [P, w]
        cost[:, ~valid] = big
        return cost

    # ---- row 0: only (0,0) starts a path (k == band); others chain left
    cost0 = _row_cost(0)
    prev = np.full((P, w), big, dtype=np.float64)
    left = np.full(P, big, dtype=np.float64)
    for k in range(w):
        v = cost0[:, k].copy() if k == band else cost0[:, k] + left
        prev[:, k] = v
        left = v

    # ---- rows 1..T-1: standard DP; all-BIG preds keep the cell at BIG.
    # Column index k = j - i + band is row-relative, so predecessor mapping is:
    #   up   (i-1, j  ) -> prev[k+1]
    #   diag (i-1, j-1) -> prev[k]
    #   left (i,   j-1) -> cur[k-1] (running min)
    for i in range(1, T):
        cost = _row_cost(i)
        up = np.full((P, w), big, dtype=np.float64)
        up[:, :-1] = prev[:, 1:]
        cur = np.empty_like(prev)
        left = np.full(P, big, dtype=np.float64)
        for k in range(w):
            m = np.minimum(up[:, k], prev[:, k])
            np.minimum(m, left, out=m)
            v = np.where(m >= big, big, cost[:, k] + m)
            cur[:, k] = v
            left = v
        prev = cur
    return prev[:, band]  # cell (T-1, T-1)


def _as_matrix(X: np.ndarray) -> np.ndarray:
    """Accept [N, T] or [N, 1, T]; return [N, T] float64."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        if X.shape[1] != 1:
            raise ValueError("only univariate (C=1) DTW is supported")
        X = X[:, 0, :]
    if X.ndim != 2:
        raise ValueError(f"expected [N, T] or [N, 1, T], got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("DTW input contains non-finite values")
    if float(np.abs(X).max()) > _MAX_AMPLITUDE:
        raise ValueError(f"input amplitude exceeds guard {_MAX_AMPLITUDE}")
    return np.ascontiguousarray(X)


def cross_dtw(A: np.ndarray, B: np.ndarray, band: int) -> np.ndarray:
    """Full cross distance matrix [len(A), len(B)] between two series sets."""
    Am = _as_matrix(A)
    Bm = _as_matrix(B)
    if Am.shape[1] != Bm.shape[1]:
        raise ValueError("series length mismatch between A and B")
    Ap = np.repeat(Am, Bm.shape[0], axis=0)
    Bp = np.tile(Bm, (Am.shape[0], 1))
    d = _banded_dtw_batch(Ap, Bp, band)
    return d.reshape(Am.shape[0], Bm.shape[0])


def pairwise_dtw(X: np.ndarray, band: int) -> np.ndarray:
    """Symmetric pairwise distance matrix [N, N] (diagonal = 0)."""
    Xm = _as_matrix(X)
    d = cross_dtw(Xm, Xm, band)
    # cross_dtw is deterministic and DTW is symmetric; enforce exact symmetry
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


def select_prototypes(X: np.ndarray, y: np.ndarray, seed: int, band: int) -> np.ndarray:
    """Per-class medoid under DTW (deterministic: argmin of distance sum).

    Returns the selected prototype series, shape [n_classes, T].
    """
    Xm = _as_matrix(X)
    y = np.asarray(y)
    classes = np.unique(y)
    protos = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        sub = Xm[idx]
        d = pairwise_dtw(sub, band)
        medoid_pos = int(np.argmin(d.sum(axis=1)))
        protos.append(sub[medoid_pos])
    return np.stack(protos, axis=0)


def select_prototypes_k(
    X: np.ndarray, y: np.ndarray, k: int, band: int, seed: int, rng_stream: str = "dtw_proto_k"
) -> np.ndarray:
    """K medoids per class under DTW (farthest-point sampling from the medoid).

    A single medoid smears a class whose motif sits at random positions; K
    diverse representatives preserve that structure so the resulting prototype
    distances carry real signal for the Ridge fusion. Shape [n_classes * k, T].
    Deterministic via PCG64 on ``rng_stream``.
    """

    Xm = _as_matrix(X)
    y = np.asarray(y)
    classes = np.unique(y)
    protos = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        sub = Xm[idx]
        d = pairwise_dtw(sub, band)
        n = sub.shape[0]
        kk = min(k, n)
        medoid_pos = int(np.argmin(d.sum(axis=1)))
        chosen = [medoid_pos]
        # farthest-point sampling: maximize min-distance to already-chosen
        mind = d[medoid_pos].copy()
        for _ in range(kk - 1):
            nxt = int(np.argmax(mind))
            chosen.append(nxt)
            mind = np.minimum(mind, d[nxt])
        protos.extend(sub[chosen])
    return np.stack(protos, axis=0)


def prototype_distances(X: np.ndarray, protos: np.ndarray, band: int) -> np.ndarray:
    """Distances of each series in X to each prototype -> [N, n_classes]."""
    return cross_dtw(_as_matrix(X), _as_matrix(protos), band)
