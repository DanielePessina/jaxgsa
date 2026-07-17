# Changelog

## 0.4.0

### Changed

- Replaced root-level command aliases with method namespaces such as
  `gsax.sobol.sample`, `gsax.sobol.analyze`, `gsax.pce.analyze`, and
  `gsax.hdmr.analyze`.
- Moved surrogate prediction and Shapley derivation onto `PCEResult` and
  `HDMRResult` through `predict(...)` and `shapley(...)`.
- Enforced one output contract across methods: `(N,)`, `(N, K)`, or
  `(N, T, K)`. Axis inference and automatic transposition were removed.
- Renamed the main sampling/result types to `SobolSamples`, `SobolResult`, and
  `MorrisSamples`.
- Simplified Sobol persistence to one NPZ file through
  `SobolSamples.save(...)` and `SobolSamples.load(...)`; removed pandas and
  format-specific persistence dependencies.

### Added

- Correlation-aware HDMR Shapley effects with
  `result.shapley(include_correlative=True)`.
- Dense structural `HDMRResult.S2` and `HDMRResult.S3` interaction arrays.
- Bounded-memory batched prediction for PCE and HDMR result objects.
- A 0.3 to 0.4 API migration guide.

## 0.3.0b1

### Added

- **Optimal-transport sensitivity indices**
  (`gsax.optimal_transport`, re-exported as `gsax.analyze_optimal_transport()`):
  the Wasserstein-based distributional indices of Borgonovo, Figalli,
  Plischke & Savaré (2024, Management Science, doi:10.1287/mnsc.2023.01796) —
  the class-averaged squared 2-Wasserstein distance between conditional and
  unconditional output distributions, normalized to [0, 1] by `2 * Var(Y)`,
  with
  - an exact **advective/diffusive decomposition** of every index: the
    advective (location-shift) component equals half the given-data
    first-order Sobol index, the diffusive remainder captures changes in
    spread, tails and shape;
  - three output modes: `"univariate"` (default; per output column via the
    closed-form 1-D sorted-quantile coupling — no iterative solver),
    `"multivariate"` (one index per input over the flattened joint output as a
    point cloud), and `"trajectory"` (one index per input per output
    over each output's whole time course), covering scalar, multi-output,
    and time-series `Y`;
  - a pure-JAX **log-domain Sinkhorn** solver for the point-cloud modes (entropic
    regularization on the max-scaled cost, unregularized `<P, C>` reported,
    single post-hoc convergence warning), with per-output standardization
    on by default;
  - **rank-based conditioning**: distribution-free in X (uniform, Gaussian,
    truncated-Gaussian, or mixed marginals work unchanged) and well-defined
    for correlated inputs (total, correlation-inclusive influence);
  - an opt-in `dummy=True` irrelevance baseline (a synthetic independent
    input scored through the identical pipeline) to threshold against the
    entropic/finite-sample index floor;
  - percentile bootstrap confidence intervals sharing one scanned code path
    with the point estimate, and `OTResult.to_dataset()` xarray export.
  Implemented from the published equations; numerics are validated in
  the test suite against POT (`ot.wasserstein_1d`,
  `ot.sinkhorn2`, `ot.emd2`; new `pot` dev extra), analytic Gaussian closed
  forms (new `gsax.benchmarks.gaussian_linear.ANALYTICAL_OT` constant), the
  published Ishigami anchor, and the `2 * advective == S1` identity.
- Shared `gsax._partition` module: the equal-frequency rank-partition
  helpers (`_class_layout`, `_build_class_indices`) moved out of
  `gsax.borgonovo` so the Borgonovo delta and optimal-transport estimators
  use one implementation (no behavior change).

## 0.2.0

### Added

- **Morris** elementary-effects screening
  (`gsax.morris`, re-exported as `gsax.morris.sample()` /
  `gsax.morris.analyze()`): globalized one-at-a-time screening that reduces
  `r * (D + 1)` model evaluations to mu, mu_star (importance ranking), and
  sigma (nonlinearity/interaction flag), with
  - trajectory (Morris, 1991) and radial (Campolongo et al., 2011,
    scrambled-Sobol' star) designs;
  - uniform and Gaussian marginals — Gaussian coordinates are confined to a
    truncated-quantile grid (`truncation_quantile`, default 0.005) so the
    design probes the 0.5%–99.5% quantile range instead of unbounded tails;
  - unique-row deduplication, so grid collisions across trajectories do not
    cost real model evaluations (same contract as Saltelli `sample()`);
  - scalar, multi-output, and time-series outputs, non-finite trajectory
    cleaning, and bootstrap confidence intervals over trajectories;
  - `MorrisSamples.downsample()` prefix-slicing to fewer trajectories
    without re-simulation, and `MorrisResult.to_physical_units()` /
    `to_dataset()` for derivative-scale measures and labeled xarray export.
- **Shapley effects** (`gsax.analyze_shapley`, `ShapleyResult`) — global
  Shapley-value allocation of output variance across inputs (Owen 2014;
  Song, Nelson & Staum 2016), computed analytically from a fitted surrogate's
  variance decomposition instead of permutation Monte Carlo. Two backends:
  `"pce"` (default; subset variances read off orthonormal polynomial
  coefficients) and `"hdmr"` (RS-HDMR component-function variances); both
  accept scalar, multi-output, and time-series outputs. Indices are
  normalized by the surrogate's total decomposed variance, so `Sh` sums to
  exactly 1 (the Shapley efficiency property); the `explained_variance` field
  reports the fraction of `Var(Y)` the surrogate captured, and the `order`
  field the effective surrogate order used. For the `"pce"` backend `S1`/`ST`
  match `analyze_pce` exactly. A `UserWarning` flags a pathological fit
  (`explained_variance` far from 1). Assumes independent inputs.
- Closed-form analytical Shapley values (`ANALYTICAL_SHAPLEY`,
  `analytical_shapley(...)`) for the Ishigami, linear, and Sobol-G
  benchmarks, used to validate the new method.
- `gsax.borgonovo` subpackage — moment-independent, density-based sensitivity via
  the Plischke, Borgonovo & Smith (2013) given-data estimator of Borgonovo's
  (2007) delta index. `analyze` (top-level alias `analyze_borgonovo`) returns both
  the delta index and the given-data first-order Sobol `S1` from the same
  rank-class partition, with bootstrap bias correction (`2*d_hat - mean(d_boot)`)
  and percentile confidence intervals. Works with any `(X, Y)` pairs and supports
  scalar, multi-output, and time-series outputs.
- `DeltaResult` dataclass holding the delta and `S1` indices with optional
  bootstrap intervals and a `to_dataset()` xarray export.
- `gsax.benchmarks.gaussian_linear` — a Gaussian linear additive benchmark whose
  Gaussian marginals give the Borgonovo delta index a semi-analytic solution
  (`ANALYTICAL_DELTA`), for ground-truth validation of the delta estimator.
- **Uniform output support across all ten methods** — every analyze entry point
  now accepts `(N,)`, `(N, K)`, and `(N, T, K)` outputs. `analyze_pce`,
  `analyze_shapley` (`"pce"` backend), and `dgsm.analyze` gained
  multi-output/time-series support: PCE fits all output slices against one
  shared basis in a single multi-right-hand-side solve, and DGSM accepts
  `(T, K)`-valued functions and precomputed Jacobians. A precomputed `dfdx`
  must mirror `Y`'s layout with one extra trailing `(D,)` axis, and singleton
  promotions are tolerated (`(N,)` pairs with `(N, 1, D)`, `(N, 1)` with
  `(N, D)`); whatever axis moves layout inference applies to a transposed or
  single-labeled `Y` are replayed on `dfdx`.
- **Smart output-layout inference** at every public entry point. `Y` is
  resolved from two signals — the expected sample count identifies the sample
  axis, and `len(problem.output_names)` (when set) identifies the output
  axis — through a strict ladder: exact canonical shapes pass silently;
  unambiguously recoverable layouts (e.g. a transposed `(K, N)` array, or a
  3-D `(N, K, T)` array whose middle axis matches the labels) are fixed with
  a `UserWarning` naming the transformation; ambiguous layouts raise. The
  warnings point at the user's call site (a corrected `stacklevel` across all
  ten methods). Label rule: a 2-D `Y` of shape `(N, M)` with exactly **one**
  entry in `problem.output_names` and `M > 1` is read as `M` timepoints of that
  single output and flows as `(N, M, 1)`; a lone column `(N, 1)` stays canonical
  as `(N, K=1)` (a scalar output, not a 1-timepoint series — pass `(N, 1, 1)`
  for that). Without `output_names`, 2-D `Y` still always means `(N, K)`. A 1-D
  `(N,)` `Y` is accepted as one output regardless of how many names the problem
  lists.
- `time_coords` parameter on `PCEResult.to_dataset()` and
  `DGSMResult.to_dataset()` for labeled time axes on 3-D results
  (dims `(time, output, param)`).

### Changed

- RS-HDMR now returns `NaN` sensitivity indices (with the existing
  zero-variance warning) for constant-output slices, matching the
  package-wide convention used by Sobol and PCE, instead of silent zeros.
- `analyze_pce` now emits a `UserWarning` when the requested polynomial
  `order` is automatically reduced to fit the sample budget, and warns on a
  constant (zero-variance) output.
- Input-validation errors are now uniform across methods: RS-HDMR, PCE, and
  DGSM share the same `X`/`Y` contract checks (and error messages) as the
  given-data methods, and RS-HDMR now also rejects `X`/`Y` row-count
  mismatches up front.
- `PCEResult` field layouts now mirror the output layout: `S1`/`ST` are
  `(..., D)`, `S2` is `(..., D, D)` with a `NaN` diagonal, `coefficients` is
  `(..., n_terms)` with the term axis last, and `loo_rmse` is per output slice
  (`()`, `(K,)`, or `(T, K)`). `order` remains a single int — all slices share
  one basis.
- `emulate_pce` and `emulate_hdmr` mirror the original training-`Y` rank: a 2-D
  `(N, T)` single-labeled training `Y` (fit internally as `(N, T, 1)`) now
  predicts `(N_new, T)` rather than `(N_new, T, 1)`, so `pred - Y_train`
  broadcasts as expected.
- PCE Sobol-index extraction is vectorized end to end: `S1`/`ST`/`S2` are
  masked matmuls over the term axis (`S2` uses an upper-triangle pair mask),
  replacing the previous Python pair loop.

### Internal

- Bootstrap confidence-interval helpers moved to a shared `gsax._bootstrap`
  module (previously private to `gsax.sobol` and cross-imported by Morris);
  PAWN and Borgonovo now reuse the same percentile-CI implementation.
- All ten result classes now build their `to_dataset()` dims/coords through the
  shared `_dims_and_coords` helper (including `HDMRResult`, whose per-term
  arrays swap the trailing `param` dim for `term`), instead of hand-rolling the
  `param`/`output`/`time` schema; the singleton-axis squeeze after analysis is
  likewise a single shared `_squeeze_output_axes` helper.
- `analyze_shapley` now fits through the shared PCE/HDMR cores directly: the
  default `"pce"` backend reuses `analyze_pce`'s fit without recomputing the
  `S2` einsum and LOO diagnostic it discards, and both backends validate `Y`
  and warn about zero-variance slices exactly once.

## 0.1.2

### Added

- `gsax.config.enable_compilation_cache(path, ...)` — opt-in helper that enables JAX's
  persistent, on-disk compilation cache so compiled kernels are reused across
  process restarts (parameter sweeps, CI, HPC batches).
- Configuration guide covering double precision (`jax_enable_x64`) for
  precision-sensitive Sobol/HSIC estimators and the compilation cache.

### Changed

- Raise the minimum JAX/jaxlib version to 0.6 — the oldest release the test
  suite is validated against (previously advertised 0.4 but never tested there).
- Benchmark harness times the SALib HDMR path with the same best-of-N method as
  the other paths and refreshes the published tables (measured on Apple M1 Pro,
  JAX 0.10.2); the one-off XLA compile is documented as excluded from the
  steady-state numbers.
- The test suite now promotes `DeprecationWarning`/`FutureWarning` to errors
  (ignore-listing known third-party ones), so upcoming API removals — JAX API
  changes especially — surface in CI instead of passing silently.

### Fixed

- Benchmark timing helper in `examples/benchmark_all.py` now synchronizes every
  result array (including bootstrap confidence intervals) before stopping the
  timer, so no async device work leaks out of the timed region.

## 0.1.1

### Fixed

- HDMR: widen the `scan`/`vmap` loop-variable annotations to `int | Array`
  so the package type-checks under newer `ty` releases (no runtime change).
- Loosen the HDMR chunk-size regression test tolerance to absorb
  floating-point reduction-order differences across jax versions.

### Changed

- Documentation installs gsax from PyPI (`pip install gsax` / `uv add gsax`).
- Update pinned development/CI dependencies and GitHub Actions; add weekly
  Dependabot updates for Python dependencies.

## 0.1.0

Initial public release.

`gsax` provides global sensitivity analysis in JAX with seven complementary
methods:

- **Sobol** indices via Saltelli sampling — first-, total-, and second-order,
  with JAX-accelerated bootstrap confidence intervals.
- **RS-HDMR** — B-spline surrogate with ANCOVA decomposition and a built-in
  emulator; works with any `(X, Y)` sample pairs.
- **PCE** — Polynomial Chaos Expansion with analytical Sobol indices and
  leave-one-out cross-validation.
- **eFAST** — Extended Fourier Amplitude Sensitivity Test.
- **DGSM** — Derivative-based measures via JAX autodiff, with Poincaré-constant
  bounds on total Sobol indices.
- **HSIC** — Hilbert–Schmidt Independence Criterion (kernel-based dependence,
  R2-HSIC and Total HSIC, permutation p-values).
- **PAWN** — moment-independent, CDF-based sensitivity via Kolmogorov–Smirnov
  distances.

All methods support scalar, multi-output, and time-series outputs and export to
labeled `xarray` Datasets.
