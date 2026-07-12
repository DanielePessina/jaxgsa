# Changelog

## Unreleased

### Added

- **Morris** elementary-effects screening
  (`gsax.morris`, re-exported as `gsax.sample_morris()` /
  `gsax.analyze_morris()`): globalized one-at-a-time screening that reduces
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
  - `MorrisSamplingResult.downsample()` prefix-slicing to fewer trajectories
    without re-simulation, and `MorrisResult.to_physical_units()` /
    `to_dataset()` for derivative-scale measures and labeled xarray export.
- **Shapley effects** (`gsax.analyze_shapley`, `ShapleyResult`) — global
  Shapley-value allocation of output variance across inputs (Owen 2014;
  Song, Nelson & Staum 2016), computed analytically from a fitted surrogate's
  variance decomposition instead of permutation Monte Carlo. Two backends:
  `"pce"` (default; subset variances read off orthonormal polynomial
  coefficients, scalar outputs) and `"hdmr"` (RS-HDMR component-function
  variances, supports scalar, multi-output, and time-series outputs). Indices are
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

### Internal

- Bootstrap confidence-interval helpers moved to a shared `gsax._bootstrap`
  module (previously private to `gsax.sobol` and cross-imported by Morris);
  PAWN and Borgonovo now reuse the same percentile-CI implementation.
- All result classes (`SAResult`, `EFASTResult`, `MorrisResult`,
  `ShapleyResult`) now build their `to_dataset()` dims/coords through the
  shared `_dims_and_coords` helper instead of hand-rolling the
  `param`/`output`/`time` schema.

## 0.1.2

### Added

- `gsax.enable_compilation_cache(path, ...)` — opt-in helper that enables JAX's
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
