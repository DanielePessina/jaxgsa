# Changelog

## Unreleased (0.6.0)

### Added

- **`Problem.correlation` — declared input dependence via a Gaussian copula.**
  A `Problem` can now carry an optional `(D, D)` correlation matrix. Pass it
  as `correlation=` on the constructor or on `from_dict`. Or use the
  `problem.with_correlation(R)` copy constructor, since problems are frozen.
  The matrix lives on the latent standard-normal scale of a Gaussian copula.
  Each parameter keeps its declared marginal exactly. Only the coupling
  between columns changes. Pass `correlation_kind="spearman"` to declare a
  rank correlation instead. The exact relation `2 sin(pi rho_s / 6)` converts
  it. That route is invertible for non-Gaussian marginals. Validation on
  entry checks shape, symmetry, unit diagonal, and entry range. A
  non-positive-definite matrix usually signals inconsistent pairwise
  correlations. Eigenvalue clipping repairs it. The report is graded by the
  largest change to a single entry, measured on the scale you declared. Below
  `1e-8` the repair is floating-point noise and says nothing. Between `1e-8`
  and `0.05` a `UserWarning` reports the change and the minimum eigenvalue.
  At `0.05` or more a `ValueError` rejects the matrix. Such a matrix is
  structurally inconsistent. Correct it, or fit a valid one from data with
  `jaxgsa.sampling.fit_correlation`. Sampling therefore
  never follows a silently different dependence structure. A matrix declared
  with `correlation_kind="spearman"` is reported on the Spearman scale, in
  the units you wrote, not on the converted latent scale. `fit_correlation`
  never raises for this reason. Inconsistent data is not a user error. A fit
  that had to move an entry by `0.05` or more only warns. The correlation
  round-trips through the JSON problem metadata and the NPZ design files.
  Saved designs do not silently drop it.
- **Correlated sampling in `jaxgsa.sampling`.** `monte_carlo` now honors
  `problem.correlation` transparently. It draws correlated standard normals
  on the latent scale. It pushes them through each marginal's inverse CDF
  (the NORTA construction). Independent problems keep the previous
  pseudo-random path bit-for-bit. Existing seeds reproduce existing samples.
  Two new companions cover the remaining workflows. `correlate(X, problem)`
  retrofits the declared correlation onto an existing sample. It uses
  Iman–Conover rank re-pairing: van der Waerden scores, de-correlated by
  `chol(R) chol(corr(M))^-1`, then matched rank for rank. Each output column
  is an exact permutation of the input column. The marginal values therefore
  stay intact. `fit_correlation(problem, X)` estimates the latent matrix from
  observed data via Spearman ranks. Attach it with
  `problem.with_correlation(fit_correlation(problem, X))`. Tied values get
  average ranks, per the Spearman convention. Heavily discrete data still
  biases the fit toward zero. A polychoric estimator is future work.
  `correlation_from_covariance(cov)` converts a published covariance matrix
  to the correlation form the API accepts. It discards the variances on the
  diagonal in favor of the declared marginals.
- **Hard errors from correlation-naive methods.** Some methods compute
  indices that assume independent inputs. When `problem.correlation`
  declares a non-identity dependence structure, these methods now refuse to
  run. They no longer return silently wrong numbers. This covers the
  structured design samplers: `sobol.sample`, `morris.sample`,
  `efast.sample`. It also covers the given-data analyzers whose theory needs
  independence: `pce.analyze`, `dgsm.analyze`, and `shapley.analyze` with
  the PCE backend. Each error names correlation-tolerant alternatives.
  `optimal_transport`, `borgonovo`, `hdmr`, `hsic`, and `pawn` all accept
  correlated problems. The ANCOVA `Sb` term of `hdmr` is precisely the
  correlation-induced contribution. `shapley.analyze(backend="hdmr",
  include_correlative=True)` allocates the ANCOVA decomposition. Internally,
  a `correlation_ok` capability flag sits on the shared `(X, Y)` validation.
  Future methods must therefore make the decision explicitly.

### Changed

- **`hdmr` now says what `ST` means under correlated inputs.** No number
  changed. `HDMRResult.ST` was labelled a total-order index with no caveat.
  It is the SCSA total: the sum of `S = Sa + Sb` over every term that
  contains the parameter. That is Eq. (8) of Sarazin, Viaud & Cournède
  (2017), and it is the same convention as SALib's HDMR. With independent
  inputs the correlative shares vanish and it reduces to the ordinary Sobol'
  total-order index. With correlated inputs it does not. It can be negative,
  it is not bounded in `[0, 1]`, and it does not measure the expected
  variance reduction from fixing a parameter. So it must not be used as a
  criterion for fixing one. The bias runs toward "cannot be fixed", and a
  parameter the model ignores can outrank one with a negative value. It is
  also not comparable with the `ST` of `jaxgsa.kucherenko` or the `S_TU` of
  `jaxgsa.vkoga`. Use one of those for a conditional-variance total under
  dependence. `HDMRResult.S1` carries the matching caveat: it is the
  structural share `Sa` of the first-order term, not the Sobol' first-order
  index. `hdmr.analyze` emits one `UserWarning` per call on a correlated
  problem to say all of this. Independent problems stay silent. The
  per-term `Sa`, `Sb` and `S` fields keep their ANCOVA meaning and are
  unaffected.

### Fixed

- **`correlation_kind="spearman"` now checks the matrix you declared.** The
  structural checks (shape, symmetry, unit diagonal, entry range) ran after
  the `2 sin(pi rho_s / 6)` conversion. That conversion pins the diagonal to
  1, so a Spearman matrix with a nonsense diagonal was accepted and silently
  rewritten. The checks now run on the declared matrix, before any
  conversion. Both kinds reject the same structural errors.
- **`correlate()` is now the real Iman–Conover method.** It drew a plain
  correlated normal score matrix, which carries its own sampling noise into
  the re-pairing. It now uses van der Waerden scores and the
  `chol(corr(M))^-1` de-correlation step, which is what removes that noise.
  At N = 50 and a target latent correlation of 0.8, the standard deviation of
  the achieved rank correlation falls from 0.065 to 0.024, and a bias of
  -0.006 disappears. The same ratio holds at every sample size. Output for a
  correlated problem therefore changes for a given seed. Each output column
  is still an exact permutation of the input column, and the function is
  still deterministic for a given seed.
- **The positive-definiteness repair is now idempotent.** Clipping the
  eigenvalues and then renormalising the diagonal could push the smallest
  eigenvalue back under the floor. A repaired matrix was therefore repaired
  again on the next call, so a `Problem` moved by about 1e-9 on every
  save-and-load round trip. The repair now repeats until the floor holds, so
  it reaches a fixed point and round trips are stable.

## 0.5.0

### Added

- `SobolSamples.to_morris()` — derive Morris elementary effects from a Sobol
  design you have already evaluated, at no extra model cost. A Saltelli
  design already contains the radial (star) structure Morris needs: within each
  base point, `A` and each `AB_j` differ in exactly one parameter. Writing
  `EE_j = (f(AB_j) - f(A)) / (B_j - A_j)`, Jansen's total-order estimator is
  `E[(delta_j * EE_j)^2] / (2 Var Y)` while Morris reports `mu_star = E|EE_j|`,
  so both methods are moments of the same increments. Pass the returned
  `MorrisSamples` and your existing `Y` to `jaxgsa.morris.analyze` to get
  `mu`/`mu_star`/`sigma`, bootstrap CIs, and multi-output support with no new
  model runs. `n_trajectories` is `base_n` for both design variants: one radial
  block per base point, based at `A`. Second-order designs also contain a block
  based at `B`. It is deliberately unused, because pooling it reduces no
  measured variance (pooled / A-only variance ratio [1.07, 1.00, 1.59] over 150
  seeds at `base_n=128`) and would need a cluster bootstrap over base points to
  keep confidence intervals honest.

  A derived design is a radial design, so it estimates
  `E|f(A with B_j) - f(A)| / |B_j - A_j|`, not the classical fixed-step-delta
  grid quantity. `morris.sample` defaults to `method="trajectory"`, so compare
  a derived result against `morris.sample(..., method="radial")`.
- `Problem.from_dict(..., truncate_gaussians=q)` — opt in to one bounded input
  model. It writes explicit `low` and `high` into every Gaussian spec that does
  not already declare them, at that marginal's own `q` and `1 - q` quantiles.
  Default `None` keeps the previous unbounded behaviour. Sides the spec already
  declares are kept as written.

### Changed

- `morris.sample` now squashes only the open sides of a Gaussian marginal.
  A Gaussian with an explicit `low` and `high` is already bounded, and was
  being truncated a second time into a range `truncation_quantile` narrower on
  each side. A one-sided truncation now squashes only its open side.
- `morris.sample`'s `truncation_quantile` default drops from `5e-3` to `1e-4`.
  Measured, `q=5e-3` discarded 7.5% of the marginal variance and 24% of the
  fourth moment and perturbed rankings (Kendall tau 0.66 against a
  near-untruncated design on Oakley-O'Hagan); `q=1e-4` discards 0.29% and 5.0%.
  Note that `mu_star` on an unbounded marginal has no `q -> 0` limit — the
  design always includes unit levels 0 and 1 exactly — so magnitudes there are
  scale-dependent by construction and only rankings are comparable across
  truncation settings.
- `pce.analyze` no longer forces a wide truncated Gaussian onto Legendre.
  Any truncation used to route the input through its truncated CDF, which the
  low-order Legendre basis approximates badly. A truncation whose every
  declared bound is at least 5 standard deviations out now keeps Hermite. On
  Oakley-O'Hagan at order 3 this restores the unbounded fit exactly, where the
  Legendre route cost about a factor 2 in LOO RMSE. Above order 7 Legendre is
  used even for a wide truncation, because the Hermite Gram defect against the
  truncated measure grows with degree.

### Fixed

- Morris trajectory designs recorded `ee_delta` before the open-side squash
  rescaled the coordinate, so the elementary-effect divisor did not match the
  step actually taken — a systematic error of about 1% at the former default
  `q`. The divisor is now rescaled with the coordinate, per dimension. A model
  linear in the unit coordinate now recovers its coefficients exactly.
- `morris.analyze` now reports a design that lost blocks. The warning follows
  the cause, not the surviving count. A small design that you asked for stays
  silent. A design that lost blocks gives a warning at any count, and the
  message names each cause: blocks that `SobolSamples.to_morris` dropped for
  having no measurable step, and blocks that non-finite cleaning removed. The
  message adds the "statistically unreliable" note when fewer than 10
  trajectories remain. Before, `to_morris` could drop to 4 surviving blocks in
  silence, and a deliberate `r = 8` design gave a warning it did not deserve.
  `MorrisSamples` records the loss in the new `n_blocks_dropped` field.

### Internal

- Added `jaxgsa._core.sampling._inverse_transform_samples`, the float64 inverse
  of `_transform_samples` (physical units back to the unit cube). Needed
  because derived elementary-effect denominators are differences of unit
  coordinates, and the existing float32 JAX helper
  (`_core.transforms.cdf_to_unit_interval`) loses too many digits to divide by.
  Unifying the two remains a follow-up.
- Named the unit-cube clip that bounds every unbounded marginal:
  `jaxgsa._core.sampling.UNIT_CLIP`, still `1e-12`, equal to +/-7.0345 sigma for
  a Gaussian. It replaces four hard-coded literals. No behaviour change.

## 0.4.0

### Changed

- **BREAKING: renamed the package from `gsax` to `jaxgsa`.** The distribution
  and import name are now `jaxgsa` (`pip install jaxgsa`, `import jaxgsa`). The
  old `gsax` project on PyPI is frozen at `0.3.0b1` with no compatibility shim;
  see the [0.3 → 0.4 migration guide](docs/guide/migration-0.4.md).
- Replaced root-level command aliases with method namespaces such as
  `jaxgsa.sobol.sample`, `jaxgsa.sobol.analyze`, `jaxgsa.pce.analyze`, and
  `jaxgsa.hdmr.analyze`.
- Moved surrogate prediction and Shapley derivation onto `PCEResult` and
  `HDMRResult` through `predict(...)` and `shapley(...)`.
- Enforced one output contract across methods: `(N,)`, `(N, K)`, or
  `(N, T, K)`. Axis inference and automatic transposition were removed.
- Renamed the main sampling/result types to `SobolSamples`, `SobolResult`, and
  `MorrisSamples`.
- Simplified Sobol persistence to one NPZ file through
  `SobolSamples.save(...)` and `SobolSamples.load(...)`; removed pandas and
  format-specific persistence dependencies.
- Renamed the design row-count fields on `SobolSamples` and `MorrisSamples`:
  `n_total` is now `n_runs` (unique rows you evaluate, one model run per row)
  and `expanded_n_total` is now `n_expanded` (pre-deduplication design size).
- Retyped the eFAST workflow: `jaxgsa.efast.sample(problem, n_per_curve, *, M=4,
  ...)` (the second parameter was `N`) returns a typed `EFASTSamples` carrying
  `samples`, `n_per_curve`, `M`, `problem`, and an `n_runs` property;
  `jaxgsa.efast.analyze(samples, Y, ...)` is samples-first and no longer takes
  `M` or `problem` — both are threaded from the design object, so a mismatch
  between sampling and analysis is now impossible.
- Tightened the eFAST design bound to `n_per_curve >= 4*M^2*(D-1) + 1` (was
  `n_per_curve > 4*M^2`, independent of `D`). Below the new bound there are
  not enough integer frequencies below `omega_0/(2M)` to give each non-focal
  parameter a distinct one; the old code wrapped them cyclically, so two
  parameters got the same frequency and the same phase and were therefore
  identical along that search curve — a silent bias. Such designs now raise
  `ValueError` from both `efast.sample` and `EFASTSamples`.
- Standardized the batching vocabulary package-wide: `batch_size` always means
  rows of X/Y per batch (`pce.analyze`, `hdmr.analyze`, `dgsm.analyze`,
  `hsic.analyze`, and `result.predict`); output-slice chunking parameters were
  renamed from `chunk_size` to `slice_chunk_size` on `hdmr`, `efast`, `sobol`,
  `borgonovo`, `optimal_transport`, and `pawn` `analyze`.
- Moved shared internals into a private `jaxgsa._core` package and introduced
  private base classes for sample designs and surrogate results (internal
  reorganization; no user-facing API change).

### Added

- Correlation-aware HDMR Shapley effects with
  `result.shapley(include_correlative=True)`.
- Dense structural `HDMRResult.S2` and `HDMRResult.S3` interaction arrays.
- Bounded-memory batched prediction for PCE and HDMR result objects.
- Automatic streaming fits for `jaxgsa.pce.analyze` and `jaxgsa.hdmr.analyze`:
  when the estimated single-pass fit memory exceeds the active budget, the
  fit streams over row batches. The streamed path is mathematically exact —
  it accumulates the same Gram matrices and moments as the in-memory path
  (PCE leave-one-out diagnostics stay exact via a second streamed pass),
  differing only in floating-point summation order. An explicit `batch_size=`
  forces streaming.
- `jaxgsa.config.set_memory_budget(bytes)` / `jaxgsa.config.get_memory_budget()`:
  an opt-in, process-global transient-memory budget (default 512 MiB) that
  sizes every automatic batching decision — surrogate `predict` batches, HDMR
  output-slice chunking, and the streaming fits. Explicit per-call
  `batch_size` / `slice_chunk_size` parameters always take precedence.
- `MorrisSamples.save(path)` and `MorrisSamples.load(path)`, using the same
  single-NPZ format and metadata schema as `SobolSamples`.
- `jaxgsa.shapley.analyze(problem, X, Y, backend="pce"|"hdmr", ...)`: a thin
  convenience wrapper over the canonical result methods
  (`pce.analyze(...).shapley()` / `hdmr.analyze(...).shapley(...)`).
- A 0.3 to 0.4 API migration guide.

### Improved

- Power-of-2 validation errors now name the two nearest valid values, so the
  message states exactly which sample counts would be accepted.

## 0.3.0b1

### Added

- **Optimal-transport sensitivity indices**
  (`jaxgsa.optimal_transport`, re-exported as `jaxgsa.analyze_optimal_transport()`):
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
  forms (new `jaxgsa.benchmarks.gaussian_linear.ANALYTICAL_OT` constant), the
  published Ishigami anchor, and the `2 * advective == S1` identity.
- Shared `jaxgsa._partition` module: the equal-frequency rank-partition
  helpers (`_class_layout`, `_build_class_indices`) moved out of
  `jaxgsa.borgonovo` so the Borgonovo delta and optimal-transport estimators
  use one implementation (no behavior change).

## 0.2.0

### Added

- **Morris** elementary-effects screening
  (`jaxgsa.morris`, re-exported as `jaxgsa.morris.sample()` /
  `jaxgsa.morris.analyze()`): globalized one-at-a-time screening that reduces
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
- **Shapley effects** (`jaxgsa.analyze_shapley`, `ShapleyResult`) — global
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
- `jaxgsa.borgonovo` subpackage — moment-independent, density-based sensitivity via
  the Plischke, Borgonovo & Smith (2013) given-data estimator of Borgonovo's
  (2007) delta index. `analyze` (top-level alias `analyze_borgonovo`) returns both
  the delta index and the given-data first-order Sobol `S1` from the same
  rank-class partition, with bootstrap bias correction (`2*d_hat - mean(d_boot)`)
  and percentile confidence intervals. Works with any `(X, Y)` pairs and supports
  scalar, multi-output, and time-series outputs.
- `DeltaResult` dataclass holding the delta and `S1` indices with optional
  bootstrap intervals and a `to_dataset()` xarray export.
- `jaxgsa.benchmarks.gaussian_linear` — a Gaussian linear additive benchmark whose
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

- Bootstrap confidence-interval helpers moved to a shared `jaxgsa._bootstrap`
  module (previously private to `jaxgsa.sobol` and cross-imported by Morris);
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

- `jaxgsa.config.enable_compilation_cache(path, ...)` — opt-in helper that enables JAX's
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

- Documentation installs jaxgsa from PyPI (`pip install jaxgsa` / `uv add jaxgsa`).
- Update pinned development/CI dependencies and GitHub Actions; add weekly
  Dependabot updates for Python dependencies.

## 0.1.0

Initial public release.

`jaxgsa` provides global sensitivity analysis in JAX with seven complementary
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
