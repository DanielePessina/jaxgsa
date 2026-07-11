# Changelog

## Unreleased

### Added

- **Morris** elementary-effects screening as an eighth method
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
