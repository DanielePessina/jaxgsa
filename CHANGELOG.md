# Changelog

## 0.2.0

### Added

- **Shapley effects** (`gsax.analyze_shapley`, `ShapleyResult`) — global
  Shapley-value allocation of output variance across inputs (Owen 2014;
  Song, Nelson & Staum 2016), computed analytically from a fitted surrogate's
  variance decomposition instead of permutation Monte Carlo. Two backends:
  `"hdmr"` (default; RS-HDMR component-function variances, supports scalar,
  multi-output, and time-series outputs) and `"pce"` (subset variances read
  off orthonormal polynomial coefficients, scalar outputs). Results carry
  `Sh` alongside `S1`/`ST` from the same surrogate, all normalized by the
  empirical output variance so the sum reports the explained-variance
  fraction. Assumes independent inputs.
- Closed-form analytical Shapley values (`ANALYTICAL_SHAPLEY`,
  `analytical_shapley(...)`) for the Ishigami, linear, and Sobol-G
  benchmarks, used to validate the new method.

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
