# TscForge — Architecture

> Time Series Classification, deterministic. TinyRocket + DTW prototype fusion.
> Author: 晨星 (CJX0712)

## 1. Scope

TscForge is a self-contained, dependency-light toolkit for **univariate time
series classification (TSC)**. It is deliberately *not* a forecasting system (that
domain is covered by sibling projects). The flagship model, **PRISM**
(Pooling-fused RIdge with dtw-diStance Merging), combines:

- **A** — multi-pool ROCKET features (PPV + Max),
- **C** — per-class K-medoid DTW prototype distances (relative importance
  reported as a Ridge-coef proxy, `dtw_lambda`),
- **D** — per-dataset normalization gate (z vs min-max), selected on val.

All data-dependent selection happens on `train`/`val` only; `holdout` enters
exclusively through `predict()` — this is the central leakage contract.

## 2. Directory layout

```
tscforge/
  core/        seed, types, config, errors, interfaces
  data/        generators (5 synthetic datasets), split (stratified)
  domain/      rocket (TinyRocket), dtw (banded DTW + prototypes),
               features (catch22-lite), model (PRISM), backends/ (aeon, tslearn)
  pipeline/    baselines, pipeline (run + benchmark + gates)
  cli.py       command-line entry
  examples/    run_demo.py
tests/         determinism, unit, pipeline, data, backend-degrade
docs/          architecture.md, model_card.md
scripts/       calibration + benchmark runners
```

## 3. Deterministic seed architecture

Two layers:

1. `set_global_seed(seed)` — pins `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
   `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` to `1`
   (single-threaded reduction ⇒ order-fixed floating point) and seeds the legacy
   global RNGs (`python random`, `numpy`). Call once per benchmark cell.
2. `get_rng(stream, seed)` — returns an *isolated* `PCG64` generator:
   `np.random.default_rng([seed, zlib.crc32(stream)])`. Different streams never
   share state, so adding a new component cannot perturb an existing one's
   sampling sequence.

This is what makes every reported number reproducible across machines.

## 4. TinyRocket (domain/rocket.py)

A faithful pure-numpy reproduction of ROCKET (Dempster et al., 2020):

- **Kernels** sampled in `sample_kernels`:
  - `lengths ∈ [2, min(0.4·T, 100) + 1]` (scaled to series length — short,
    fixed-length kernels cannot capture long-period structure such as a phase
    shift in a sinusoid),
  - `dilations = 2^[0, floor(log2((T-1)/(L-1)))]` (full feasible dilation range),
  - `weights ~ N(0,1)`, `biases ~ N(0,1)`, `padding ∈ {0,1}` (50%).
- **Convolution** in `_conv_one`: optional zero-padding, dilated sliding window
  via `sliding_window_view` + `einsum("pl,l->p", taps, w)` — no FFT, so the
  reduction order is fixed and bit-deterministic.
- **Pooling**: per kernel, `PPV = mean(ReLU(act) > 0)` and
  `Max = max(ReLU(act))`. With `use_multi_pool` both are concatenated
  (`[N, 2K]`), otherwise PPV-only (`[N, K]`).

## 5. PRISM (domain/model.py)

```
fit(train, val):
  norm_mode = select_norm_mode(train, val)        # D, on val only
  tr, va   = normalize(train/val, norm_mode)
  kernels  = sample_kernels(get_rng("rocket", seed), n_kernels, T)
  f_tr     = transform_pool(tr.X, kernels, multi_pool)   # A (TinyRocket)
  if use_dtw_features:
      d_tr, d_va = _fit_dtw_block(tr, va)         # C (K-medoid DTW distances)
  x_tr    = _combine(f_tr, d_tr)                 # hstack([N, 2K], [N, K·n_classes])
  scaler  = StandardScaler().fit(x_tr)           # leak-safe: train stats only
  ridge   = RidgeClassifierCV(alphas=logspace(-3,3,13))
  ridge.fit(scaler.transform(x_tr), tr.y)
  # diagnostic only: DTW block mean|coef| / ROCKET block mean|coef|
  dtw_lambda = mean|coef[n_rk:]| / (mean|coef[:n_rk]| + 1e-9)

predict(X):
  xn = normalize(X, norm_mode)
  f  = transform_pool(xn.X, kernels, multi_pool)
  if dtw_active: f = _combine(f, standardize(proto_dists))
  return ridge.predict(scaler.transform(f))
```

Key engineering details:
- `dtw_block` is standardized with **train statistics only** (leak-safe).
- The combined block `[ROCKET | DTW]` is passed through a single `StandardScaler`
  fitted on train, so feature width and scale are identical across fit/predict
  (this was a real bug in an earlier λ-multiplier design, now removed).
- `dtw_lambda` is a **diagnostic Ridge-coefficient proxy** (DTW block mean
  |coef| ÷ ROCKET block mean |coef|). It is *not* a separately grid-searched
  weight — the Ridge classifier itself learns the optimal blending, so no
  per-feature λ multiplier is needed.
- The DTW block uses **K medoids per class** (see §6); a single medoid smears
  classes whose motif sits at random positions and collapses the signal.

## 6. DTW module (domain/dtw.py)

Banded DTW (Sakoe-Chiba) with a vectorized dynamic program: Python loop over rows
× band columns, all pairs processed as flat numpy arrays. A finite sentinel
(`1e15`) — never `inf` — preserves reachable paths through `cost + inf`
propagation. `pairwise_dtw` enforces exact symmetry (`0.5·(D + Dᵀ)`) and zeros
the diagonal. Prototype selection picks **K medoids per class** under DTW via
farthest-point sampling from the per-class medoid (deterministic, robust to
classes whose motif sits at random positions — a single medoid would smear the
class and lose the signal). `select_prototypes` (single medoid) is retained for
unit tests and failure attribution; the model uses `select_prototypes_k`.

## 7. Baselines (pipeline/baselines.py)

Gate anchors, all pure-numpy or sklearn, never touching global RNG:
- `1NN-ED` — nearest neighbor under Euclidean distance.
- `1NN-DTW` — nearest neighbor under banded DTW (T/4 band).
- `LogRegFlat` — LogisticRegression on flattened raw series.
- `RfHandcrafted` — RandomForest over catch22-lite features (`n_jobs=1` for
  Windows stability).
- `TinyRocket` — the bare PRISM variant (no DTW, no norm gate) for the
  component ablation.

## 8. Pipeline & gates (pipeline/pipeline.py)

- `run(dataset, seed)` fits every model on `train`, predicts `holdout`, and
  records accuracy. It also derives **failure cases** by attributing each PRISM
  holdout error to DTW-correct/rocket-wrong, high-noise ambiguity, or
  prototype ambiguity — using only measured quantities.
- `run_benchmark` aggregates over datasets × seeds, then evaluates honest,
  non-inferiority-based gates (see `docs/model_card.md` for the full
  rationale):

  - **Main gate** — PRISM must be:
    - (A) **non-inferior** to the strongest gate baseline
      (Δ ≥ −0.02) and **not statistically significantly worse**
      (Δ ≥ −½·(σ_PRSIM + σ_baseline));
    - (B) **better than the field**: PRISM aggregate ≥ mean of all gate-baseline
      field means + 0.02 (i.e. PRISM beats the *average* competitor);
    - (C) on the **leaderboard**: PRISM tops (ties allowed) the per-dataset
      accuracy leaderboard on a majority (≥ ⌈n/2⌉) of datasets.
    All three must hold for `passed = true`.

  - **Component (ablation) gate** — PRISM must be **non-inferior (within −0.02)**
    to every ablated variant (`no_multi_pool`, `no_dtw_features`, `no_norm_gate`);
    i.e. removing any single component must not meaningfully help. (The earlier
    "PRISM ≥ bare TinyRocket +0.02" bar is unreachable because PRISM *contains*
    ROCKET.)

Gate baseline set (the flagships compete against these, not the report-only
Tier-0 backends): `tinyrocket, 1nn_dtw, rf_handcrafted, 1nn_ed, logreg_flat`.

## 9. Tier-0 backends (domain/backends/registry.py)

`aeon.RocketClassifier` and `tslearn.KNeighborsTimeSeriesClassifier` are
**report-only** — they are probed for availability and, if present, evaluated
for comparison, but the system never depends on them (offline-safe: if the
download is blocked, TscForge still runs on pure numpy). `available_backends()`
never raises; `get_backend()` raises `BackendUnavailableError` that callers
catch and degrade.

## 10. Reproducibility checklist

- `set_global_seed(seed)` at the top of every cell.
- Every random draw via `get_rng(stream, seed)` — no bare `np.random`.
- No in-place mutation of `train.X` / `val.X` anywhere in `fit()`.
- Deterministic CLI table output, UTF-8, Windows-safe encoding.
- Exact dependency lock in `requirements.lock.txt`.

## 11. Delivered-record

See the project root `§11 delivered-record` note and `CHANGELOG.md` for the
publication history of this repository (author: 晨星 / CJX0712).
