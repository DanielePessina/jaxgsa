# Changelog

## 1.2.0 (unreleased)

Some fixes change index values where the old values were wrong. See Fixed.

### Fixed

- `vkoga.analyze` now raises a `ValueError` when an input column of `X` is
  constant, equal to another column, or a linear combination of earlier
  columns (same check as `pce` and `hdmr`). Before, a duplicated input gave
  `S_TC` 0.86 against a true 0.79, and a constant input gave `S_TC` 0.18
  against a true 0. Duplicate rows are still allowed.
- `vkoga` with independent inputs now sets `S_U = S_TC`, so `S_C` is exactly
  0. Before, the fitted component functions left a spurious `S_C` up to
  8e-3 (Ishigami, `x2`) and an `S_IU` of 0.012 where the exact value is 0.
  `S_TC` and `S_TU` do not change.
- `pce`, and `shapley` with the PCE backend, now raise a `ValueError` when an
  input is a linear combination of other inputs, such as `c = 1 - a`,
  `c = 2 * a` or `c = a + b`. 1.1.2 caught only exact copies. An affine copy
  gave a singular design: float32 raised, but float64 warned and returned
  wrong indices. The check reads the inputs, so it does not depend on the
  dtype or on `ridge`. It raises when the combination holds to within 100
  times the rounding error of the input dtype. A copy with more noise, for
  example 1e-3, still goes to the near-singular warning.
- PCE warnings about a reduced order, a poor fit and an overfit now name the
  function you called. Through `jaxgsa.shapley.analyze` they said
  `jaxgsa.pce` and pointed at a line inside the package. They now say
  `jaxgsa.shapley.analyze` and point at your line. Values do not change.
- The PCE fit now solves the ridge least squares by QR, not by the normal
  equations. The normal equations square the condition number of the design.
  In float32 that lost every digit at high order: a Hermite fit on three
  Gaussian inputs at order 12 (N = 6000) raised "the Gram matrix of the design
  is numerically singular". The same fit now returns the exact indices
  (ST = 0.8062, 0.1938, 0.0310) at orders 12 and 13, with a precision
  warning. Order 14 and above still raise in float32. This also holds for `shapley`
  with the PCE backend. The streamed fit (`batch_size`) uses a tall-skinny QR
  that folds in one batch at a time, so its memory stays within the budget.
- The rank check reads the pivot share from the QR factor. It is finite in
  float32, so a failed Cholesky factor no longer refuses a fit. The limits
  are the same as in 1.1.2. float32 raises below 1e-5 and warns below 1e-4.
  float64 only warns, below 1e-9. Measured on three Gaussian
  inputs, N = 6000: share 1.3e-8 gives a total index off by 2e-3, and share
  1.8e-9 off by 0.05. In the warning band the leave-one-out error `cv_rmse`
  can be wrong by a large factor, because `1 - h` (h is the leverage of a
  row) is mostly rounding noise, while the indices stay exact much longer.
  At share 9.8e-5 float32 gives `cv_rmse` 0.23 against 0.005 in float64. A
  non-finite share raises. Constant, duplicated and linearly dependent inputs
  raise in both dtypes, because they are read off the inputs.
- `pce.analyze` with `n_bootstrap > 0` warns when some resamples fit a
  near-singular design, and gives the count. The rank check ran on the full
  sample only.
- A truncated Gaussian input now gets the orthonormal polynomials of its own
  truncated distribution, at every truncation width and every order. This
  replaces two routes that were not exact. Raw Hermite polynomials lose
  orthogonality against a truncated normal as the degree grows. The Legendre
  route through the truncated CDF has to approximate the steep inverse normal
  CDF. The order-7 switch between the two, and its warning, are gone. On a
  wide truncation (+-8 sigma) at order 8, S1 went from 0.7141 (true 0.7273)
  with `cv_rmse` 0.23, to 0.7273 with `cv_rmse` 5e-6. For a narrow truncation,
  `cv_rmse` of a cubic response fell from 0.052 to 2e-6. The basis is built on
  the host in float64 and evaluated by a three-term recurrence, so it is
  accurate in float32 too. Narrow (0.1 sigma wide) and far one-sided (z >= 3,
  z >= 38) truncations give exact indices. This also holds for `shapley` with
  the PCE backend. Untruncated Gaussian and uniform inputs are bitwise
  unchanged.
- Equal-frequency classes now use exact integer edges `(m * N) // M`. Before,
  the edges were the floor of `linspace(0, N, M + 1)`. Float rounding put
  some edges one row low, so one row went to the wrong class. This affects
  `optimal_transport` and `borgonovo` (continuous inputs). Only `(N, M)`
  pairs where `floor(linspace)` differs from `(m * N) // M` change. For
  example, at `N=60, M=22` one edge moves: Borgonovo delta changes by 0.013
  (4.2% relative) and OT `ot` by 0.021 (4.1% relative). The largest change
  is OT `ot_dummy`, by 0.029 (6.3% relative; this can change `above_dummy`).
  Other `borgonovo` and `optimal_transport` outputs move by a similar size.
  `(N=1000, M=7)` and `(N=997, M=49)` are bitwise unchanged. At the changed
  pairs the results can differ from SALib by one row per class.
- `pawn` bin edges are now the exact values `k / n_classes`. Before, an
  input value exactly on an interior edge (for example `0.3` with
  `n_classes=10`) could go to the bin below. Only inputs that sit exactly on
  an edge change bin.
- OT: with float64 on, the overlap weights of categorical inputs in `mode="univariate"` were rounded to float32. They now use the working dtype.
- `hdmr.analyze` now raises `ValueError` for a constant input, two equal inputs, or an input that is a linear combination of others (for example `c = a`). Before, the fit split the effect between `a` and `c` in any proportion without an error (S1 of `a` 0.000 and of `c` 0.796). It uses the same check as `pce.analyze`.
- `hsic.analyze` and `hsic.indices`: the median bandwidth now uses the median of the non-zero pairwise squared distances. Before, an atom holding more than half of the sample (for example 75% zeros) pulled the median to 0 and the kernel only tested equality. Columns with no tied values are unchanged. The 1.1.2 warning now fires on near-ties instead (median non-zero squared distance below 1e-6 times the variance), because those still collapse. The binary-column exemption is gone: a two-valued column now gets a real bandwidth.
- `pawn.analyze` bootstrap intervals with `ci_method="quantile"` were biased upward. The KS maximum is not differentiable at a zero gap, so the replicates sit above the point estimate (null parameter, N=1000: point 0.072, replicate mean 0.111). The draws are now mean-centred on the point estimate before the quantiles are read. `ci.replicates` keeps the raw draws. The interval is not a confidence interval for the population index.
- `pawn` KS statistics were off by 1 ulp because XLA turns the division into a reciprocal multiply. A constant output gave 6e-8 instead of 0. The gap is now computed from exact integer counts. Cumulative bin counts are int32, so they stay exact above N=2^24.
- `sobol.analyze` with `n_bootstrap` no longer builds a full `(R, N)` index matrix outside the memory budget. At N=2^17, D=2, R=1000 and a 512 MiB budget, peak memory above the baseline falls from about 2800 MiB to about 450 MiB (146 MiB at a 64 MiB budget). The resample width now follows the budget.

### Changed values

- `vkoga` with independent inputs: `S_C` is now exactly 0 and `S_IU` drops
  (Ishigami `x2` `S_IU` 0.0119 to 0.0035). See Fixed.
- `vkoga` with correlated inputs and a non-Gaussian marginal (uniform and
  other Legendre cases) now fits its component functions at degree 10, not 6.
  This removes a truncation bias in `S_U`, `S_C` and `S_IU` (Ishigami with
  rho = 0.5, `x2` `S_C` 0.0437 to 0.0354, reference 0.0359). The fit falls
  back to degree 6 when `n_variance` is below two rows per coefficient, and
  warns when even that is under-determined. Gaussian marginals do not change.
- `vkoga` cross-validation compiles once per data shape, not once per
  dataset. In float32 the chosen `gamma` and `ridge` can differ from 1.1.x on
  many datasets, because the old choice was rounding noise. In float64 they
  do not change.
- The PCE fit now uses QR (see Fixed). Index values change only by float
  rounding, listed below. The solver is also slower (see Performance).
- PCE values change by float rounding in every fit, and so do Shapley values
  from the PCE backend. Ishigami, N = 1000, orders 3, 5 and 7: at most 7e-7
  in float32 and 1.4e-15 in float64, for S1, ST, S2, `cv_rmse` and the
  bootstrap intervals (the Shapley PCE backend reaches 6.9e-7 in float32).
  Fits that 1.1.2 refused in float32 now succeed. Row padding changes the
  order of the sums, which adds float rounding of the same size and no more.
- PCE values change for every truncated Gaussian input, and so do Shapley
  values from the PCE backend. The old S1 error on a polynomial response was
  4e-3 to 5e-2 on the Legendre route (every truncation narrower than 7 sigma,
  and every order above 7). It is now 1e-12 in float64 and below 1e-6 in
  float32. A truncation of 7 sigma or wider at order 7 or lower moves by at
  most the old Hermite defect, 2e-4.
- OT with categorical inputs, `mode="univariate"`, float64 only: `ot` changes by about 1e-11 to 1e-10 absolute (below 1e-9 relative), `diffusive` by up to a few 1e-9 relative, and the `ot` confidence interval endpoints by about 1e-10 absolute. The old values carried float32 rounding. Float32 runs and continuous inputs are bitwise unchanged.
- Bootstrap intervals (`*_conf`) and `ci.replicates` of `hdmr`, `pce` and `shapley` change at rounding level. The row draws are the same. The largest change is 6e-5 in float32 (HDMR) and 4e-14 in float64. PCE and Shapley with the PCE backend are equal to about 1e-16 in float64. Point estimates do not change.
- `hsic` with tied values in an input or output: `R2` and `T` change because the bandwidth changes. Spike output (75% zeros), N=400: `R2` 0.0151 to 0.0066 and `T` 0.0304 to 0.0151 for the second parameter. Two-valued inputs change `T` only. Data with no ties is unchanged.
- `pawn_conf` with `ci_method="quantile"` and `n_bootstrap>0` moves down by the mean bootstrap excess: about 0.03 for a null parameter at N=1000, about 0.006 for a strong one. The width is unchanged. `"gaussian"` intervals do not change.
- `pawn` point values and bootstrap draws change by at most 1 ulp (6e-8 in float32). A constant output now gives exactly 0.
- Bootstrap draws now come from one key per resample (`split(key, R)`) in `sobol`, `kucherenko`, `morris`, `borgonovo`, `dgsm` and `optimal_transport`. For a fixed `key`, every bootstrap replicate and every confidence interval changes. The intervals are statistically equivalent. Point estimates do not change. A draw still does not depend on chunk width or memory budget, and the full index matrix is no longer built.
- `sobol` bootstrap replicates now use count-weighted moments, not a row gather. In float32 the replicates differ from a gather on the same draws by at most about 3e-6, and in x64 by about 4e-15. An input the model never reads now gives a first-order replicate of exactly 0 for `saltelli-jansen` and `mauntz-kucherenko`.

### Performance

- `vkoga` cross-validation no longer recompiles for each new dataset. The
  scorer compiles once, so repeated analyses skip some compile time (about
  0.2 to 0.4 s per new dataset in one small test). Wall-clock gains on small
  cases are within run-to-run noise.
- The PCE fit is slower, because QR does more work than the normal
  equations. Steady-state `analyze` takes 2 to 3 times longer at large sizes:
  about 270 ms to 800 ms for D = 6, order 6, N = 20000 (924 terms; 3 to 3.5
  times), and 152 ms to 312 ms for D = 10, order 3, N = 50000. Small fits
  (D = 3, order 5, N = 1000) go from about 1.9 ms to about 3 ms. The estimate
  of fit memory, which decides when the fit streams, grows from 2 to 4 arrays
  of the design size. The streamed fit is a `lax.scan` over the row batches,
  so it also traces under `lax.map`.
- `pce.analyze`, `pce.indices` and `shapley` with the PCE backend no longer
  recompile the heavy fit programs for every new number of rows. The rows are
  padded to a bucket (eight buckets per octave, so at most 12.5% extra rows),
  and a 0/1 row mask keeps the extra rows out of every sum. Row counts 497,
  498 and 499 now share one compiled fit. The first `analyze` call at a new
  row count in the same bucket took 0.8 s and now takes about 0.4 s (N = 500,
  D = 6). Small programs for the padding and the shared input checks still
  compile for each new row count. Padding adds up to 12.5% more work for a
  row count just above a bucket edge, for example N = 513 runs as 576 rows.
- The new check for linearly dependent inputs runs on the host and costs
  0.73 s at N = 100000, D = 200. It runs once per `analyze` call.
- `borgonovo.analyze` and `optimal_transport.analyze` now build the class
  tensors in blocks of bootstrap replicates, sized by the memory budget
  (`jaxgsa.config.set_memory_budget`). Before, one `(R, D, M, P)` tensor held
  all replicates and ignored the budget. At `N=20000`, `D=10`, 500
  bootstrap replicates, peak memory falls from 1198 MB to 844 MB with the
  default budget, and to 531 MB with a 64 MiB budget. Replicate blocking
  itself changes no value. A smaller budget also shrinks the automatic output-slice chunk,
  which can move values by about 1 ulp (5.6e-17 in x64 for OT univariate).
  No new argument.
- `optimal_transport.analyze` no longer recompiles for every new categorical
  dataset of the same shape. The class width is now rounded up to a
  quarter-octave grid (a multiple of 8, never above `N`). On a repeated call
  with new data, the number of compile events falls from 60 to 0 (8 when
  the width crosses a bucket edge) in the
  `univariate` and `multivariate` modes. The padding is masked, so values
  with categorical inputs change only by float rounding. Measured against
  1.1.2: at most about 1.5e-7 absolute in float32, and about 2e-16 absolute
  in x64. `borgonovo` with categorical inputs shares the change.
- `hdmr.analyze`, `pce.analyze` and `shapley.analyze` run bootstrap replicates in one cached `jax.jit` with a sequential `lax.map`, not a Python loop. The program compiles once and a repeat call compiles nothing. Steady state is faster: about 7 times for HDMR (0.95 s to 0.13 s at N=1024, 50 replicates) and about 1.1 times for PCE. The first call is not faster, because compile time dominates it. Peak memory at N=32768 with 20 HDMR replicates rose from 478 to 576 MB.
- `sobol` bootstrap runs on a count matrix and one matrix product per chunk. Measured on this machine (loaded, so read as ratios): N=2^17, D=2, R=1000 went from 3.0 s to 1.7 s. The design measured 2.8x at D=10 and 6.8x at D=50.

### Docs

- OT: the `tol` docs now give the cost of a trajectory solve. The time is in Sinkhorn iterations, not the loop over outputs. The float64 default `tol=1e-9` is about twice as slow as `1e-5`, and it stays the default.

## 1.1.2

This release fixes bugs and makes several methods faster. Correct index
values do not change, with these exceptions:

- DGSM values for a truncated Gaussian input change where they were wrong,
  that is where a declared bound lies far outside the mass, or where the
  interval is narrower than 0.1 standard deviations (1.1.1 could return a
  negative variance there). Intervals from 0.05 to 0.1 standard deviations
  wide were right in 1.1.1, and their values move in the last digits (1e-12
  to 2e-9 relative). Other truncated Gaussian inputs do not change.
- DGSM bootstrap intervals (`*_conf`) can differ in the last bits for every
  input, because the resampled variance is now computed on shifted outputs.
  In float32 with a large output offset they move more, by a size that grows
  with `offset**2 / Var(Y)`: about 2e-4 relative at an offset of 50 when the
  output standard deviation is 3.3.
- When the problem has 17 or more inputs, PCE values change slightly, and so
  do Shapley values from the PCE backend: by up to about 1e-5 relative in
  float32, and about 4e-14 relative in float64. In float32, HDMR, PAWN, HSIC
  and VKOGA values can differ in the last bit. Below 17 inputs all of them
  are bitwise unchanged.
- `pce`, and `shapley` with the PCE backend, now raise a `ValueError` when an
  input is constant or two inputs are equal in every row. 1.1.1 returned
  indices for some of these designs.
- Two dataset attributes change: `to_dataset()` on an optimal-transport
  result writes `ot_mode`, not `mode`, and boolean attributes such as the
  PCE `streamed` are written as 0 or 1.

`optimal_transport.analyze` and eager `optimal_transport.indices` give
bitwise the same values as 1.1.1, in float32 and float64.

### Fixed

- `jax.grad` and `jax.jacrev` now work through
  `jaxgsa.optimal_transport.indices` in the `"multivariate"` and
  `"trajectory"` modes. Before, they raised `ValueError: Reverse-mode
  differentiation does not work for lax.while_loop`, although the docs said
  that reverse mode worked. The Sinkhorn solver keeps its loop, so every index
  value is bitwise unchanged. The derivative now comes from the implicit
  function theorem at the solver's fixed point. Forward-mode derivatives
  therefore change slightly: they are now the derivative of the converged
  solution, not of the iterate where the loop stopped. GMRES solves the
  linear system of the derivative. Second derivatives match finite
  differences to about 1e-8 for a converged solve.
- `jaxgsa.hsic` no longer returns NaN indices for more than 46,340 samples
  when JAX runs in single precision (`jax_enable_x64` off). The median
  bandwidth counted sample pairs in a 32-bit signed integer, which
  overflowed. It now counts in an unsigned 32-bit integer, which is exact up
  to 65,535 samples. Above that, single-precision HSIC raises a `ValueError`
  that tells you to enable x64. Double precision is unchanged.

#### Gradients

- `jax.grad` through `optimal_transport.indices` no longer returns NaN when
  one output column is constant, in the multivariate and trajectory modes.
  The same guard fixes reverse-mode gradients through the output
  standardisation of `sobol` and `morris`. Values do not change.
- `borgonovo.indices` no longer returns NaN gradients when a conditioning
  class or an output column has zero variance. The gradient of a constant
  output column is now exactly 0.

#### Wrong or missing results

- `vkoga` in float32 no longer returns NaN indices when the training data has
  duplicate rows. The greedy search could pick an exact copy of a centre,
  which made the kernel matrix singular. Bootstrap intervals, which resample
  with replacement, were all NaN for this reason. VKOGA now skips exact
  copies, and retries a singular solve once with a larger jitter. Results on
  data without repeated rows do not change.
- `dgsm` bootstrap intervals were wrong in float32 when the output had a large
  offset. At an offset of 1e4 the interval did not contain the point
  estimate. The resampled variance is now computed on shifted outputs.
- `dgsm` crashed with `LinAlgError` for a truncated Gaussian input whose
  bounds lie far from the mean, such as `GaussianSpec(mean=100, variance=1,
  low=0)`. For a one-sided truncation in the tail it gave a Poincare constant
  below the variance, so the upper bound was invalid. The constant and the
  marginal variance are now computed on the support that holds the mass.
- `dgsm` crashed with `TypeError` for a model that cannot be hashed, such as a
  frozen dataclass or Equinox module with array fields. It now runs without
  the kernel cache.
- `hsic` returned a p-value of 1.0 for an output with zero variance, while R2
  and T were NaN. The p-value is now NaN too.
- `to_dataset().to_netcdf()` failed for PCE, HDMR and Shapley results with
  the netCDF4 engine, and for optimal-transport results with the scipy
  engine. Boolean attributes are now written as 0 or 1, and the
  optimal-transport `mode` attribute is now `ot_mode`.

#### Warnings and errors

- `pce` now raises a `ValueError` when an input is constant, or when two
  inputs are equal in every row. The PCE cannot estimate the indices of such
  inputs from the sample. The message names the inputs. Only inputs that are
  equal value for value count as duplicates: an affine copy, such as
  `c = 1 - a`, gets the warning described below in float64. Before, float64
  fits gave finite but wrong indices with no message: a duplicated input
  shared the index of the true input, and a constant input got a large total
  index.
- `pce` now checks the Gram matrix of the design. For each basis column, it
  finds the fraction of the column's variance that is left after the earlier
  columns explain it. In float64 the fit warns when this fraction is below
  1e-9: dependent or nearly duplicated inputs then make the indices
  unreliable. A very high order on Gaussian inputs can also cause the
  warning, and its indices can still be accurate. In float32 the fit raises
  below 1e-5 or when the factorisation fails, and the message says that
  float32 rounding may be the cause. Between 1e-5 and 1e-4 it warns, because
  float32 rounding cannot tell a singular design from a well-posed one there.
  When `jaxgsa.shapley` runs these checks, the messages name
  `jaxgsa.shapley.analyze`.
- `pce` now warns when the leave-one-out error is not finite on an output
  that is not constant. Before, no warning appeared. Non-finite coefficients
  now raise a `ValueError`, except with `on_invalid="propagate"`.
- `pce` now warns when an order above 7 moves a wide truncated Gaussian input
  from the Hermite to the Legendre basis. The warning names the input.
- `hsic` now warns when more than half of the samples of an input or output
  share one value. The median bandwidth then collapses and the kernel only
  tests for equality. Values do not change.
- `vkoga` now warns when the fitted coefficients or some bootstrap replicates
  are not finite.
- `Problem` now rejects zero parameters, a bare string as categorical
  `labels`, duplicate or non-string `output_names`, a complex correlation
  matrix, and an invalid `correlation_type` when no correlation is given.
- `slice_chunk_size` and `batch_size` in `sobol`, `efast`, `hdmr` and
  `shapley` now reject a float or a bool at once. Before, a float crashed
  later inside the run.
- Zero-variance warnings now point at your own line of code, also through
  `shapley.analyze`, `dgsm.analyze` and `kucherenko.analyze`. `efast` no
  longer warns twice for one constant slice. `kucherenko` no longer invents a
  time label for a 2-D output.
- In the optimal-transport joint modes, the zero-variance warning no longer
  says that the indices will be 0 when only some slices are constant.
- A Sinkhorn solve that ends with a NaN residual now counts as not
  converged. The non-convergence warning also suggests a looser `tol` or x64.
- `shapley.analyze` now names itself in its drop and propagate messages. The
  `dgsm` key error now starts with `jaxgsa.dgsm:`.
- The `sobol.analyze` verbose summary now shows the chunk sizes that the
  bootstrap really used, and the true run count.

### Performance

- The first `hsic.analyze` call compiles much faster: at N = 1024, about 25 s
  becomes under 1 s. Values are bitwise unchanged.
- The PCE design matrix now builds each term from its non-zero factors when
  the problem has 17 or more inputs. At D = 1000 the coefficient step drops
  from 81 s to 0.5 s.
- The input transforms of PCE, HDMR, PAWN, HSIC and VKOGA work on one
  marginal family at a time when the problem has 17 or more inputs. At
  D = 300 the transform drops from 19 s to 1.2 s.
- PCE skips the second-order index step when no term has two inputs.
- The `dgsm` bootstrap no longer slows down as the number of batches grows
  (40 batches: 4.8 s to 0.5 s), and it holds one fewer `(n_bootstrap, N)`
  array in memory.
- The conditional plan of `kucherenko` and `vkoga` no longer builds an unused
  `(D, D-1, D-1)` array. At D = 500 it uses half the memory.

### Docs

- The differentiation guide and the `optimal_transport.indices` docstring
  now describe how the point-cloud modes are differentiated. They also say
  that only `analyze` and eager `indices` are bitwise stable across releases:
  under your own `jax.jit`, float32 values can differ in the last bits.
- The PAWN docs now say that its row bootstrap biases intervals upward. For a
  weak parameter the interval can lie above the point estimate.
- The `rank` docs now say that `ci.replicates` holds the plain bootstrap
  draws, so an interval rebuilt from them can be narrower than `S1_conf`.
- Five examples had code that did not run: inverse Sobol bounds, non-uniform
  inputs, save and load, DGSM, and correlated inputs. They are fixed.
- The bootstrap helper docstrings now name the methods that really call
  `bootstrap_draws` (pce and hdmr). pawn and shapley use their own loops.
- The dependent-inputs skill reference and the Shapley example no longer
  claim "two decimals" of agreement where the gap is up to 0.010 and 0.015.

### Internal

- The test helper for correlated Gaussian-linear indices now accepts any
  covariance matrix, not only a correlation matrix.

## 1.1.1

This release changes no API and no results. It corrects the minimum versions
in the package metadata.

### Fixed

- jaxgsa now requires `jax>=0.8` and `jaxlib>=0.8`. The metadata said 0.6,
  but no CI job tested that version. On jax 0.7, `jaxgsa.hdmr` fails with an
  `UnexpectedTracerError`, and on jax 0.6 most of the test suite fails. If
  your environment has jax 0.6 or 0.7, pip now upgrades it.
- The `dev` extra now requires `SALib>=1.5`. SALib 1.4.0 does not build on a
  current Python.

### Docs and CI

- A new `lowest-versions` CI job installs the lowest version of every
  dependency that `pyproject.toml` allows, on Python 3.12, and runs the
  default test tier. A minimum version that is too low now fails CI.
- `ruff format` no longer formats Markdown files. Ruff 0.16 started to format
  Python code blocks in Markdown, which removed the aligned comments in the
  docs and changed old CHANGELOG entries.

## 1.1.0

jaxgsa 1.0.x had no users. This release therefore removes and renames some API
without a deprecation period. From 1.1.0 on, the API follows semantic
versioning.

### Breaking changes

#### Interface cleanup before the announcement

Each line shows the old form, then the new form.

##### Names and constructors

- `Problem.num_vars` → `Problem.n_params`.
- `Problem(...)`: every argument after `bounds` is now keyword-only.
- `Problem.from_dict(...)`: every argument after `params` is now keyword-only.
- `Problem` raises `TypeError` for a bare string as `names` or `output_names`.
- `Problem.with_sampler(f, dimension=k)` → `with_sampler(f, sampler_dimension=k)`.
- `Problem.has_correlated_inputs` → `problem.dependence`. It returns
  `"independent"`, `"gaussian"` or `"custom"`. An identity correlation matrix
  still gives `"independent"`.
- `KucherenkoResult.is_correlated` and `VKOGAResult.is_correlated` →
  `result.dependence`, with the same three values. The saved dataset attribute
  `is_correlated` is now `dependence`.
- `Theta` is no longer exported from `jaxgsa`. Import it from `jaxgsa.sobol`.
- `SPEC`, `MethodSpec`, `register` and `InvalidUnit` no longer appear in the
  method packages. `jaxgsa.sampling` and `jaxgsa.config` no longer export their
  imports.

##### Sampling and failed runs

- `on_invalid="none"` → `on_invalid="skip"`. `"none"` now raises `ValueError`.
- `sampling.monte_carlo(problem, n, *, seed)` →
  `monte_carlo(problem, n_samples, *, seed=None)`.
- `sampling.correlate(X, problem, *, seed)` → `correlate(problem, X, *, seed=None)`.
- A custom sampler is now checked against the support of each input. Uniform and
  truncated bounds, and integer codes for categorical inputs, are enforced.
  A value that overshoots a bound by float32 rounding is clipped to the bound.
  A value further out raises `ValueError`.
- `sobol.sample(scramble=False, seed=...)` now raises `ValueError`, as
  `kucherenko.sample` already did.
- `config.get_memory_budget()` now returns megabytes by default, like
  `set_memory_budget()`. Pass `unit="b"` for bytes. `set_memory_budget()` no
  longer rejects large unit-less values.

##### Design-based methods

- `sobol.sample(problem, n_samples, ...)` →
  `sobol.sample(problem, n_samples=None, *, base_n=None, ...)`. Give exactly one
  of `n_samples` and `base_n`. Otherwise `ValueError`.
- `kucherenko.sample(problem, n, ...)` →
  `kucherenko.sample(problem, n_samples=None, *, base_n=None, subsets=...)`.
  `n_samples` is now the minimum number of model runs, like Sobol. It was the
  number of base points per block. Use `base_n=` for the old meaning. The same
  `base_n` and `seed` give the same design.
- The first parameter of design-based `analyze()` and `indices()` →
  `samples`. It was `sampling_result`. Passing a `Problem` raises `TypeError`
  that points to the given-data methods.
- `sobol.indices` now always returns three fields. `S2` is `None` for a
  first-order design.
- `indices()` of every method returns a named tuple: `SobolIndices(S1, ST, S2)`,
  `MorrisIndices(mu, mu_star, sigma)`, `EFASTIndices(S1, ST)`, `DGSMIndices`,
  `HDMRIndices(Sa, Sb, S, ST)`, `PCEIndices(S1, ST, S2)`,
  `ShapleyIndices(Sh, S1, ST)`, `HSICIndices(R2, T, p_values, hsic_raw)`,
  `BorgonovoIndices(delta, S1)`, `OTIndices(ot, advective, diffusive, S1)`,
  `PAWNIndices(pawn)` and `RankIndices(S1)`. Plain tuple unpacking still works,
  but the field count of `OTIndices` grew (see below).

##### DGSM, Morris and eFAST

- `dgsm.analyze(problem, fn, X, *, Y, dfdx)` →
  `dgsm.analyze(problem, X, Y=None, *, fn=None, dfdx=None)`. `dgsm.indices`
  changes the same way. Model path: `dgsm.analyze(problem, X, fn=f)`. Precomputed
  path: `dgsm.analyze(problem, X, Y, dfdx=J)`.
- `DGSMResult.sigma` → `DGSMResult.mu`. `sigma_conf` → `mu_conf`. The value is
  the mean signed derivative.
- `DGSMResult.var_y` → `DGSMResult.variance`. `to_dataset()` now includes it.
- `MorrisResult.to_physical_units()` on a result that is already physical now
  returns `self`.
- `EFASTResult.ci` is now always `None`.

##### HDMR, PCE and Shapley

- `hdmr.analyze` and `hdmr.indices`: `maxorder` → `order`, `maxiter` →
  `max_iter`, `lambdax` → `ridge`, `m` → `n_intervals`.
- `shapley(backend="hdmr", order=3)` now works with the new names.
- HDMR with fewer than 300 rows now warns. It raises only below
  `(n_intervals + 3)**min(order, D) + 1` rows. It used to raise below 300.
- `PCEResult.loo_rmse` → `PCEResult.cv_rmse`. `PCEResult.rmse` is new.
- Surrogate fit fields now hold one value per output slice. `rmse` is in-sample.
  `cv_rmse` is out-of-sample. `explained_variance` is `1 - MSE / Var(Y)`.
  `HDMRResult` gains `explained_variance`, and its `cv_rmse` is `None`.
  `VKOGAResult.cv_rmse` was one pooled float. It is now per output slice.
  `VKOGAResult` gains `explained_variance`.
- Error messages from every method start with `jaxgsa.<method>: `.

##### VKOGA and HSIC

- `vkoga.analyze(..., correlation=R)` is removed. The dependence comes from the
  problem only. Use `problem.with_correlation(R)`.
- `VKOGAResult.correlation` is removed. Use `result.problem.correlation`.
- `VKOGAResult.shapley()` is removed.
- `key` is now required and keyword-only on `hsic.analyze` and `vkoga.analyze`.
- `HSICResult.R2_HSIC` → `R2`. `HSICResult.T_HSIC` → `T`.
- `HSICResult.ci` is now always `None`.

##### Borgonovo, OT, PAWN and rank

- `optimal_transport.analyze` and `indices`: `n_partitions` → `n_classes`.
- `borgonovo.analyze(bias_correct=None)` → `bias_correct: bool = False`.
  `False` always gives the plain estimate. `True` gives the bias-corrected
  estimate and needs `n_bootstrap > 0`. Any other value, such as `None`,
  raises `TypeError`. The first-use warning is gone.
- `pawn.analyze` and `indices`: `n_bins` → `n_classes`, like Borgonovo and OT.
  `PAWNResult.n_valid_bins` → `PAWNResult.n_valid_classes`.
- `morris.sample(num_levels=...)` → `n_levels=...`. `MorrisSamples.num_levels` →
  `MorrisSamples.n_levels`. `MorrisSamples.load` does not read files saved
  with `num_levels`.
  It now raises `ValueError`, not `KeyError`, for a file saved by another
  samples class.
- `borgonovo.DeltaResult` → `borgonovo.BorgonovoResult`. There is no alias.
- `OTIndices` has a fourth field, `S1`. Code that unpacks three names breaks.

#### Other breaking changes

- Kucherenko: the default design keeps its blocks, but a given `seed` now gives
  new draws. This holds for every problem, including independent ones. The
  indices are statistically unchanged.
- VKOGA: the same `key` now gives slightly different indices. The outer and
  inner quasi-Monte Carlo draws changed. The differences are Monte Carlo noise,
  about 1e-4 to 2.5e-3. `variance` is bit-identical.
- `Problem.__eq__` and `__hash__` now include the sampler. A custom-sampler
  problem loaded from a file is not equal to the original.
- Messages changed. Exception types are the same, but `pytest.raises(match=...)`
  on the old text fails.
  - The verbose line `correlation: correlated (Gaussian copula)` is now
    `dependence: gaussian copula`. The Kucherenko sample line
    `dependence=copula-conditional` is now `dependence=gaussian`.
  - The `repr` of `KucherenkoSamples` shows `dependence=...` and `subsets=...`
    instead of `correlated=...`.
  - Every independence refusal and dependent-input warning has new text. It
    says "dependence-tolerant" instead of "correlation-tolerant", lists
    `jaxgsa.rank`, and names `problem.sampler`.
  - The HDMR `ST` warning, `HDMRResult.shapley(include_correlative=False)` and
    the Shapley `backend="pce"` refusal now also apply to custom-sampler
    problems.
  - `jaxgsa.vkoga.analyze` no longer warns about correlated training data.

### Added

- `jaxgsa.sampling.transform(problem, U)` maps your own unit-cube design
  through the problem's marginals, copula or custom sampler. Use it with a
  Latin hypercube of your own.
- Every `analyze(key=...)` accepts an `int` and calls `jax.random.key` on it.
  A bad `key` raises `TypeError`. A JAX key as `seed=` on a sampler raises
  `TypeError`. Both messages use jaxgsa wording.
- `Problem.from_dict` accepts `[low, high]` lists as bounds.
- `SobolSamples`, `MorrisSamples` and `EFASTSamples` have a compact `repr`.
- `SobolResult.to_dataset()` has a new `variance` variable.
- `VKOGAResult.S1` and `VKOGAResult.ST` are read-only aliases of `S_TC` and
  `S_TU`.
- `PCEResult.rmse` and the `explained_variance` fields listed above.
- Error messages are clearer. They name the method and the knob that applies.
- `HDMRResult.order` holds the order the fit used, after it is cut down to D.
- `to_dataset()` attrs include `method` and `jaxgsa_version` for every result.
- A saved samples file records its class. Loading it with the wrong class
  raises an error that names the right one. Old files still load.

### New

#### Custom input samplers

A problem can carry a custom sampler. A custom sampler describes any joint
distribution, not only a Gaussian copula.

Declare the sampler on the problem:

- `jaxgsa.Problem(names=..., bounds=..., sampler=f, sampler_dimension=2)`
- `jaxgsa.Problem.from_dict(..., sampler=f, sampler_dimension=2)`
- `problem.with_sampler(f, sampler_dimension=2)`. Use `problem.with_sampler(None)` to
  drop it.

`Problem.sampler` and `Problem.sampler_dimension` hold the values. The type
alias `jaxgsa.problem.Dependence = Literal["independent", "gaussian",
"custom"]` is new. `jaxgsa.sampling.InputSampler` is the public type of the
callback.

The sampler contract:

- jaxgsa calls `sampler(uniform_points, *, conditioning_rows=None,
  fixed_indices=())`.
- The sampler must be a deterministic inverse Rosenblatt map, also called a
  conditional-quantile map. Rejection sampling and MCMC are not valid.
- It must accept any sorted `fixed_indices`. On the default Kucherenko design
  it gets three kinds of call: a joint draw, a draw that keeps `x_i`, and a draw
  that keeps every input except `x_i`. With `subsets="all"` it also gets every
  other subset.
- It must return `(n, D)` finite rows, copy the fixed columns exactly, and give
  values inside the support of each input. A new check raises `ValueError` for
  values outside the support. It clips values within float32 rounding of a
  bound.

A problem has either a Gaussian-copula `correlation` or a `sampler`, never
both. New validation errors:

| Input | Error |
|---|---|
| `correlation=` and `sampler=` together | `ValueError` |
| `sampler_dimension` without a sampler | `ValueError` |
| `with_sampler(3)` | `TypeError`: sampler must be callable |
| `sampler_dimension=0` or `True` | `ValueError`: positive integer needed |
| wrong output shape | `ValueError` |
| non-finite output | `ValueError` |
| fixed columns changed | `ValueError` |

Where a sampler works:

- `jaxgsa.kucherenko.sample(problem, n_samples)` and
  `jaxgsa.vkoga.analyze(problem, X, Y, key=...)` read the sampler from the
  problem.
- `jaxgsa.sampling.monte_carlo(problem, n_samples)` honours the sampler. Use it to
  draw `X` for given-data methods such as PAWN, Borgonovo, optimal transport,
  HSIC, HDMR and rank.
- Sobol, Morris and eFAST designs and the independence-only analyses refuse a
  custom-sampler problem. They refuse it in the same way as a correlated
  problem.
- A saved design or result does not store the sampler function. A loaded
  problem still reports `dependence == "custom"`, so the independence checks
  still refuse it. It raises if you ask it to draw new rows. Attach the
  sampler again with `problem.with_sampler(f, sampler_dimension=k)`.
- Verbose output prints `dependence: independent | gaussian copula | custom
  sampler`.

Kucherenko and VKOGA share one estimator for the Gaussian copula and for a
custom sampler.

#### Exact Shapley effects and PME from Kucherenko

`jaxgsa.kucherenko.sample(problem, base_n=n, subsets="all")` builds a design with one
block for every proper non-empty subset of inputs, plus the joint blocks A and
B. The design has `base_n * 2^D` runs. It needs `2 <= D <= 12`. To build it,
jaxgsa asks the copula or the custom sampler to hold any group of inputs fixed
and redraw the rest. The default `subsets="singletons"` design keeps its
layout.

On the new design, `KucherenkoResult` gains:

- `closed` and `subset_total`: the closed and total index of every subset,
  with `closed_conf` and `subset_total_conf` intervals, and the `subsets`
  membership matrix.
- `result.shapley(cost="total")`: exact conditional-variance Shapley effects
  (Owen 2014; Song, Nelson and Staum 2016). Use `cost="closed"` for the closed
  cost.
- `result.pme()`: proportional marginal effects (Herin et al. 2024). An input
  that the model does not use gets exactly 0.

Both methods return a `ShapleyResult` with `backend="exact"`. Intervals need
`n_bootstrap > 0` and `keep_replicates=True`. `shapley()` on a singletons
design raises `ValueError` and points to `subsets="all"`. `D > 12` raises and
points to `shapley` or `vkoga`.

On a `subsets="all"` design, `S1` and `ST` come from the subset blocks. They
differ slightly from a singletons run.

`KucherenkoSamples` has new attributes `dependence`, `subsets` and `n_blocks`.
Saved designs and results store `subsets` and the metadata keys `dependence`,
`subsets` and `sampler_dimension`. Results store `closed*`, `subset_total*` and
`subset_members`. Older files load as `"gaussian"` or `"independent"` with
`subsets="singletons"`.

#### `jaxgsa.rank`

`jaxgsa.rank.analyze(problem, X, Y)` estimates the first-order Sobol' index
from one existing sample, with the rank statistic of Gamboa, Gremaud, Klein and
Lagnoux (2022). `jaxgsa.rank.indices` is the traceable core. See the
[rank page](https://danielepessina.github.io/jaxgsa/api/rank).

- It accepts independent, Gaussian-copula and custom-sampler problems.
- It supports scalar, multi-output, time-series and irregular `Y`.
- It refuses categorical inputs and needs at least 3 rows.
- `n_bootstrap > 0` needs `key`.
- `S1` is not clipped, so it can be negative.
- The intervals use 80% subsamples without replacement, rescaled to full size.
  The reported interval is the hull of an interval around the estimate and one
  around a bias-corrected estimate. This rule is a heuristic, checked by
  simulation.
- The estimator divides by `n`, as in the paper. R `sensitivity::sobolrank`
  divides by `n - 1`.

#### `result.aggregate()`

`SobolResult`, `KucherenkoResult` and `PCEResult` have
`aggregate(*, weights="variance")`. It collapses the output axes into one index
per input. The weights are the output variances (Gamboa, Janon, Klein and
Lagnoux 2014; Lamboni, Monod and Makowski 2011). Use `weights="equal"` for equal
weights.

- The output axes of the result collapse to size 1. A result with no output
  axes is returned unchanged.
- The single output of an aggregated result is named `"aggregate"`, so
  `to_dataset()` works on a problem with output names.
- With `keep_replicates=True`, each bootstrap draw is aggregated and the
  interval is drawn again.
- With a bootstrap but without `keep_replicates=True`, the intervals are `None`
  and a warning says why.
- An aggregated `PCEResult` holds indices only. `predict()` and `shapley()`
  raise `ValueError`.
- `SobolResult` has a new optional field `variance`, set by `sobol.analyze`.
  Results built without it accept `weights="equal"` only. The Sobol' indices
  are unchanged.

### Changed

- VKOGA training-data guidance. The old guidance said the training design must
  be independent. That was wrong. Both nested estimators draw from conditionals
  of the joint distribution. They never draw an input from its whole marginal.
  A surrogate trained on draws from the analysis distribution works well. In a
  check on the Ishigami function with a Gaussian copula (`rho_ab = 0.8`,
  `rho_bc = -0.4`, 400 training rows, 4 seeds), the index error against
  Kucherenko at 2^16 base points was 0.007 for copula training draws and 0.010
  for independent draws. The new rule: the surrogate must be accurate where the
  analysis distribution puts its mass. An independent design is a safe default
  when the correlation is not fixed yet, or when you reuse one fit under
  several correlations.

- `KucherenkoResult.shapley()` and `pme()` set `explained_variance` and
  `order` to `None`, because no surrogate is fitted.
- New clear errors:
  - `dgsm.analyze` and `dgsm.indices` tell you to use `fn=` when you pass the
    model as the third argument.
  - `sampling.correlate`, `monte_carlo` and `transform` show the correct
    argument order when the first argument is not a `Problem`.
  - `to_dataset(time_coords=...)` checks the length and names both lengths.
  - The error for a wrong number of `Y` rows names the source of the count.
    For `sobol`, `morris`, `kucherenko` and `efast`, it says the design has
    that many rows and points to `samples.samples`. For the other methods, it
    says `X` has that many rows.
  - `dgsm.analyze` checks that `X`, `Y` and `dfdx` have the same number of
    rows.
  - `EFASTSamples.load` and the other `load` methods name the right class
    when you load the wrong file.
- The shared argument checks of every `analyze()` and `indices()` start the
  error with the method, such as `jaxgsa.sobol: `. Before, only five methods
  did this.
- The `scramble` docs say that SciPy uses a linear matrix scramble with a
  digital shift. They no longer say Owen scrambling.

### Performance

Measured on Ishigami (D = 3, N = 1024, float32) on an Apple M1 Pro CPU.

- `rank.analyze` compiled a new program on every call. With 20 resamples a
  call took 770 ms. It now takes 9 ms.
- `pawn.analyze` with a bootstrap had the same fault in its resample keys.
  A one-slice call now takes 3 ms instead of 7 ms. A new test checks that a
  second call of every method compiles nothing.
- The first call of `pce.analyze` and `shapley.analyze` takes 0.4 s instead
  of 1.3 to 1.5 s. Later calls are 5 times faster.
- `hsic.analyze` finds the median bandwidth from the sorted sample, with no
  N x N distance matrix. A one-slice call takes 33 ms instead of 75 ms and
  peaks 130 MiB lower.
- `optimal_transport.analyze` computes the W2 term of a continuous input in
  closed form, with M times less working memory.
- The shared input checks compile once instead of once per operation, which
  shortens the first call of every method.
- `dgsm.analyze` with a bootstrap evaluates the model and its Jacobian once,
  not twice. It still makes a second pass when it drops non-finite rows.
- The Kucherenko Shapley and PME allocation handles a game with a zero-effect
  input in one vectorised step per pattern. Before, it looped in Python over
  every bootstrap column, which was 150 to 200 times slower.
- `pce` builds its multi-index once per dimension and order. Before, every
  bootstrap and Shapley refit built it again (63 ms at D = 50, order 3).

These changes move some float32 results at rounding level, by at most 4e-6.
No HSIC p-value changed. The fitted VKOGA hyperparameters and centres did
not change on any of five test problems at the default settings.

### Fixed

- The dependent-input warnings now also fire for custom samplers.
- `hsic.analyze` no longer gives int64 dtype warnings in float32.

### Studio

The browser app moved from `/jaxgsa/workbench/` to
[`/jaxgsa/studio/`](https://danielepessina.github.io/jaxgsa/studio/). It is
now called jaxgsa studio. The old `/workbench/` address redirects to the new
one and keeps the query and hash.

- Opening step 01 again no longer deletes the design, the outputs and the
  results. A custom problem stays when you move between steps.
- The studio saves the problem, the form settings and the method selection in
  the browser. A reload restores them. "Reset session" clears them. Uploaded
  data and results are not saved.
- PCE and PCE-based Shapley results show the explained variance and the
  leave-one-out error. A warning shows when the explained variance is below
  0.5, as in the Python package.
- The demo uses PCE order 8, so its PCE indices match the analytic Ishigami
  values. The default order for your own data stays 3.
- The PCE order must be a whole number of 1 or more. The result shows the order
  that the fit actually used.
- Uploads reject `Infinity`, `1e999` and hexadecimal values. CSV files can use
  commas, semicolons or tabs, quoted fields and a byte order mark.
- A downloaded design CSV works as X on the "completed model runs" route.
- The upload check mark shows only after the file passes validation.
- The output names field now changes the expected Y columns.
- The analytical reference line shows only for demo outputs.
- PAWN and Borgonovo now run on truncated Gaussian inputs.
- The step bar fits on a phone screen. Method checkboxes and form fields have
  accessible names.
- `@jax-js/jax` is pinned to 0.1.24.

### Docs and CI

- New `api/rank.md` page. The Kucherenko, VKOGA and dependent-inputs pages were
  rewritten. Every API page now has citations. The method count is fourteen.
- SEO: sitemap, canonical and Open Graph tags, JSON-LD, `llms.txt` and
  `robots.txt`. The canonical and sitemap URLs now keep the `/jaxgsa/` path.
- `CITATION.cff` uses the concept DOI 10.5281/zenodo.22099117. It always
  resolves to the latest version. The README has BibTeX.
- CI runs the f32 and f64 numerical tests on every pull request. Tests run with
  `pytest-xdist` (`-n auto`, added to the `dev` extra). A new fast end-to-end
  test suite exists.
- The web CI runs lint, type checks, unit tests, a Playwright suite and the
  build for `web/`, on Node 24.

## 1.0.2

Publish the validated 1.0 code under a fresh version number because PyPI
cannot reuse the filenames from a previously deleted 1.0.0 upload. There
are no Python API or numerical implementation changes from the archived
1.0.0 release. The original GitHub tag and Zenodo record remain available.

Documentation figures now share the paper-style fonts, palette and minor
ticks. A new worked example fits input ranges to target Sobol indices and
checks the result against an independent design and analytical solution.

## 1.0.0

First stable release. The API contract of 0.9.1 is unchanged; 1.0.0 marks it
as supported.

### New: jaxgsa studio

A [WASM/React companion](https://danielepessina.github.io/jaxgsa/studio/)
runs eight ported methods locally
in the browser (float64, no data leaves the page): define a problem, sample a
Sobol/Morris/Kucherenko/eFAST design, bring the model outputs back, and read
the indices — plus the fit-any-data estimators (PCE, Shapley, Borgonovo,
PAWN) on uploaded point clouds. The tolerances are pinned against the Python
package through golden fixtures in `goldens/`.

### Release summary

The Python API retains the call patterns of 0.9.1. Version 1.0.0 is the
API-stability statement: breaking changes now require a major version.

- Faster optimal-transport and PAWN kernels, and reduced Shapley fitting
  overhead, preserve the numerical results of the previous implementations.
- Rectangular nested lists and tuples with two time slices now follow the
  same analysis path as arrays. Irregular channels retain their input
  precision until the analyzer can warn about float32 downcasting.
- Nightly CI runs the ordinary numerical regression suite in both supported
  precision settings. Publication validates every test, including slow tests,
  against the pinned release commit before building and publishing.
- Built-wheel checks cover sampling persistence, analytical indices, NetCDF
  export and surrogate prediction in a fresh installation. Source archives
  explicitly select release files and exclude local build dependencies.
- Manual publication requires an existing release tag and uses the same
  validation gates as publication triggered by a GitHub release.

## 0.9.1

### New: irregularly sampled outputs

`Y` now accepts per-output-channel time grids: each channel lives on its own
(non-uniform) time axis. Pass a list of `(times, values)` pairs — or a dict
keyed by output name — to any of the twelve `analyze` entry points that accept
ragged outputs:

```python
sr = jaxgsa.sobol.sample(problem, n_samples=1024, seed=0)
Y = [(t_conc, y_conc), (t_d43, y_d43)]   # y_conc: (n_runs, 5), y_d43: (n_runs, 4)
result = jaxgsa.sobol.analyze(sr, Y)
```

Each channel runs through the method's regular kernels on its own grid (no
padding, no fabricated values), and the returned
`jaxgsa._core.irregular.IrregularResult` holds one normal result per channel,
with `to_dataset()` emitting one variable per channel on its own `time`
coordinate. Bootstrap and permutation randomness is folded per channel, so
the channels' confidence bands are independent. `dgsm` is the one method
that refuses the ragged form (its pre-computed Jacobian path would need a
per-channel `dfdx`). There is no mask mode: padding onto a common grid would
add work without helping the per-channel estimators.
Performance is the sum of the per-channel analyses plus a host-side coercion
pass; `scripts/benchmark_irregular.py` measures it.

### Release summary

Version 0.9.1 improves performance and the behavior contract. It also adds
the irregular-output form of `Y`, the `on_invalid='none'` policy, and
`Problem.with_output_names()`. The thirteen analysis methods retain their
existing call patterns.

### Performance

- **Sobol.** `S2` is computed as `BA.T @ AB` instead of a materialised
  `(N, D, D)` outer product (no quadratic intermediate, up to ~6x on the
  kernel, lower peak memory); the first-order cross-moments
  (Mauntz-Kucherenko, Janon-Monod, Martinez) are matmuls instead of
  broadcast-multiply-plus-reduce. The bootstrap resampler chunks the
  resample axis from the same transient budget and, with
  `keep_replicates=False`, reduces each chunk to CI endpoints before the
  next runs. True-scale scalar second-order `analyze`
  (`base_n=16384`, `D=30`): **12.25 ms -> 1.77 ms** with
  `on_invalid='none'` (6.65 ms default path).
- **Entry path.** The host `np.bincount` wall over every expanded row is
  gone (per-unit verdicts collapse on device), the zero-variance
  max/min/var mask runs in one jitted scan, and the scalar front half
  (standardize + split + slice flatten) traces as one executable.
- **`on_invalid='none'`.** A new policy value that skips the isfinite and
  constant-slice scans entirely and runs the analysis on the data exactly
  as given. On clean data the indices are bit-identical to the default
  policy.
- **Deferred design expansion.** Under `on_invalid='none'` the expansion
  gather is deferred and fused into the scalar estimator front half for
  sobol and morris; the design map is warmed once at construction, so the
  per-call device transfer disappears from every path.
- **HSIC.** The median-bandwidth selection replaces a 32-step bitwise
  bisection with a 2-level 16-bit bincount radix select:
  **142 ms -> 23 ms**, bit-exact.
- **VKOGA.** The nested CV scorer and the greedy center selection are
  jitted once (module-level hoist, content-keyed memo of the surrogate
  predictor) instead of re-traced per call: **341 ms -> 23 ms** on the
  benchmark case, with no coefficient or CV-rounding drift (the unsafe
  variants that moved numbers were reverted).
- **Kucherenko.** `S1` uses an einsum that removes the `(D, N, S)`
  broadcast temporary.

### Behavior

- New `on_invalid='none'` policy (see above): shape-contract checks still
  run, but no isfinite scan, no drop, and no constant-slice warning. A
  `NaN` reaches the estimator and flows into the indices.
- Sobol and Morris call `analyze(..., n_bootstrap=...)` (the renamed
  `num_resamples=`); HSIC and VKOGA take `key=` instead of `seed=` in the
  benchmark harness.

### Numerics

The kernel rewrites reorder float32 arithmetic, so a few reported values
move by units in the last place. The numerical baseline
(`scripts/baseline_check.py`) still passes at the recorded 406 accepted
moves; all parity tests (SALib, OpenTURNS) still pass.

## 0.9.0

Version 0.9.0 freezes the public interface. It gives eleven methods a pure
`indices()` core that survives `jit`, `vmap` and differentiation. It adds
confidence intervals to six more methods, a printed run summary on every entry
point, one batching contract for the whole library, and one policy for a
non-finite model output. It also fixes defects, and some of them move reported
numbers.

**Read "Breaking" before you upgrade.** It covers keyword renames, one batching
contract for every method, `verbose=True` as the new default, changed defaults,
removals, and the few changes that move reported values.

### Breaking

- **Every entry point now prints a summary by default.** All thirteen
  `analyze()` functions and all four samplers take `verbose: bool = True`.
  With the default, `analyze()` prints three short sections to stdout: the
  problem and the data, the wall-clock timing, and the top parameters by the
  method's headline index. Samplers print one line about the design. No
  `analyze()` printed anything before, so a script that ran silently prints
  now. `sobol.sample` and `morris.sample` already printed and already took
  `verbose`; `efast.sample` and `kucherenko.sample` join them. Your numbers do
  not change. To keep a run silent, pass `verbose=False`:

  ```python
  result = jaxgsa.sobol.analyze(sr, Y, verbose=False)
  ```

- **One batching contract for every method.** Four rules now hold everywhere,
  and tests enforce them.

  - **`batch_size` sizes row blocks, clamped to `N`. Nothing more.** It never
    selects a different algorithm. In PCE, an explicit `batch_size` used to
    force the streamed fit even when it was larger than `N`. Now an explicit
    `batch_size < N` streams; an explicit `batch_size >= N` is one full
    block, which is the single-pass fit — even when the single pass exceeds
    the memory budget, because an explicit width always wins. Only when
    `batch_size` is `None` does the budget pick the path. Migration: to
    force the PCE streamed fit, pass a `batch_size` smaller than your row
    count. A `batch_size` at or above the row count now runs the single-pass
    fit and gives the exact default numbers.
  - **`None` on a batching keyword means "derive the width from the memory
    budget".** Only `pce`, `hdmr` and `vkoga` read the budget before. The
    other methods each had their own rule: `sobol`, `morris`, `efast` and
    `pawn` defaulted to a fixed 2048 that ignored the dtype, `N` and the
    budget, and `borgonovo` and `optimal_transport` read `None` as a
    hardcoded element count. Every one of them now defaults to `None` and
    derives its width from `jaxgsa.config.get_memory_budget()`. DGSM read
    `None` as one batch of every row; it now derives the batch width from
    `jaxgsa.config.get_memory_budget()` with a real bytes model (a few
    Jacobian-sized transients per row, `T*K*D` floats each). This holds on
    both DGSM paths: the autodiff path and a precomputed `dfdx`. At ordinary
    sizes the derived width is one full block, so nothing changes. On a run
    large enough that the budget now splits the sample, only the float32
    summation order moves — the same statement PCE and HDMR make about
    their streamed paths. `dgsm.analyze`, and the new `dgsm.indices`, now
    raise `ValueError` on `batch_size=0` or a negative value. `dgsm.analyze`
    used to read both as "one batch of every row".
  - **An explicit chunk value always wins.** The budget only sizes the `None`
    default. Morris silently capped an explicit `chunk_size` at an internal
    element budget, so a caller who asked for a wide chunk could get a narrow
    one. Under its new name `resample_chunk_size` it honours the caller's
    value, capped only at the axis length. Migration: none needed, unless you
    relied on that cap to shrink a too-large explicit value — pass the width
    you actually want.
  - **`hsic` loses `batch_size`, with no replacement.** The keyword
    row-blocked one kernel build while the resident kernel stacks — about
    `(2D + 1) * N^2` floats — stayed whole. It never bounded peak memory,
    and chunked evaluation was measured useless. A keyword that cannot do
    what its name promises misleads, so it is gone. Migration: delete the
    argument. If the sample does not fit in memory, reduce `N` (HSIC
    converges quickly in `N`) or screen parameters first.

- **One word for one concept across every method.** 1.0 freezes the public
  interface, so the same idea now has the same name, type and position
  everywhere it appears. Every rename below is a clean break with no alias.

  | Was | Now | Where |
  |---|---|---|
  | `num_resamples` | `n_bootstrap` | sobol, morris |
  | `seed: int` | `key: Array` | pawn, borgonovo, optimal_transport, hsic, vkoga |
  | `chunk_size` | `resample_chunk_size` | morris |
  | `samples` | `sampling_result` | efast |

  `key` replaces `seed` because a key can be split and an integer cannot.
  Reseeding from an integer silently correlates repeated or nested draws.
  Pass `jax.random.key(0)` where you passed `seed=0`.

  `sample()` keeps `seed`. Design generation is host-side `scipy.stats.qmc`
  and has no JAX PRNG interface, so the split is real and now documented
  rather than accidental.

- **The last vocabulary renames land: `standardize_outputs` and
  `correlation_type`.** Each rename is a clean break. There is no alias and
  no deprecation period. Change the keyword and the call works as before.

  | Was | Now | Where |
  |---|---|---|
  | `standardize` | `standardize_outputs` | `optimal_transport.analyze` |
  | `correlation_kind` | `correlation_type` | `Problem(...)`, `Problem.from_dict(...)` |
  | `kind` | `correlation_type` | `Problem.with_correlation(...)` |

  One flag standardizes outputs, so it has one name everywhere:
  `standardize_outputs`. One keyword declares the scale of a correlation
  matrix, so it has one name on all three `Problem` surfaces:
  `correlation_type`. Error and warning messages use the new spellings too.

- **`kucherenko.sample` gets the shared seed interface.** The keyword is now
  `seed: int | np.random.Generator | None = None`, the same as `sobol`,
  `morris` and `efast`. It was `seed: int = 0`. Pass `seed=0` to keep the old
  default design bit for bit.

  The sampler also rejects one inert setting: a seed passed with
  `scramble=False` now raises `ValueError`. The seed only feeds the Owen
  scrambling, so without scrambling it does nothing, and the project policy
  is to raise on a setting that cannot do what it says.

- **`borgonovo.analyze` no longer bootstraps by default.** `n_bootstrap` was
  `100`; it is now `0`, matching every other method. Combined with the new
  key requirement, the old default would have made the plainest possible call
  an error. **The reported `delta` moves for a default call**: with
  `bias_correct=True` and 100 resamples, the old default returned the
  Plischke-corrected delta, and the new default returns the plug-in estimate
  with no interval. Pass `n_bootstrap=100` and a key to get the old numbers.

  `bias_correct` becomes tri-state to match. `None` (the default) applies the
  Plischke correction whenever there are replicates and does nothing
  otherwise; `True` asks explicitly and warns when `n_bootstrap` is `0`;
  `False` never applies it. For a corrected delta, pass `n_bootstrap=100` and
  a key. The uncorrected estimate is biased upward, because a KDE separation
  is a distance and sampling noise can only increase it.

  Because `None` resolves by the replicate count, adding bootstrap intervals
  to a call also changes the reported `delta` from the plug-in estimate to
  the corrected one. Borgonovo says so: the first default call per process
  that resolves this way emits one `JaxgsaWarning` naming which delta it
  returned. Pass an explicit `bias_correct=True` or `False` to silence it.
  The docstring spells out the resolution rule.

- **`prenormalize` is gone.** It meant four different things across the
  methods that took it, and on two of them it meant nothing at all.

  | Method | Was | Now |
  |---|---|---|
  | `sobol` | `prenormalize: bool = False` | removed; the standardization always runs |
  | `efast` | `prenormalize: bool = False` | removed; it was a measured no-op (6e-16) |
  | `hdmr` | `prenormalize: bool = False` | removed; a no-op only after the backfit stop rule below became relative — see "Fixed" |
  | `morris` | `prenormalize: bool = False` | renamed `standardize_outputs` |
  | `dgsm` | — | new `standardize_outputs: bool = False` |

  `hsic` also loses the keyword, with no replacement. The median heuristic
  carries the scale of `Y`, so the indices are invariant under
  `Y -> a*Y + b`. Standardizing `Y` first therefore changes nothing, and a
  no-op keyword would mislead.

  `hsic.bandwidth` changes meaning in the same pass. It was
  `float | None = None`: `None` picked the median heuristic and a number was
  the kernel width itself. It is now `float = 1.0`, a multiplier on the
  median heuristic, so the heuristic always runs and every value is relative
  to it. The resolved width is `bandwidth * sqrt(median_sq)`, per parameter.
  A default call is unchanged, because `bandwidth=None` and `bandwidth=1.0`
  both give the plain heuristic. Migration: a caller who passed an absolute
  width has no exact replacement, because the heuristic is per parameter and
  one absolute width was not. Pass a multiplier around 1 instead, or scale
  the inputs.

  `optimal_transport` keeps this behavior (default `True`). It does real
  work there: the method builds distances from `Y` itself, not from a
  ratio. The keyword is now spelled `standardize_outputs`; see the rename
  entry above.

  On `morris` and `dgsm` the keyword earns its place, because those two
  return dimensional quantities. Under `Y -> a*Y + b`, Morris's `mu`,
  `mu_star` and `sigma` all scale by `a`, and DGSM's `sigma` scales by `a`
  and `nu` by `a**2`. `standardize_outputs=True` reports them in units of the
  output standard deviation, so output slices of different magnitude compare
  with each other. DGSM's `upper_bound` and `lower_bound` are ratios and do
  not move; its reported `var_y` becomes 1. DGSM had no way to ask for this
  before.

  On `efast` and `hdmr` the keyword did nothing measurable, because their
  indices are ratios. A no-op keyword on a frozen interface is worse than an
  absent one: a caller sets it and believes it acted.

- **HDMR fits on the caller's output scale.** With `prenormalize` gone, the
  fit no longer standardizes `Y`, so `predict()` and `rmse` need no inverse
  transform. The private fit state loses its `prenormalize`, `y_mean` and
  `y_std` entries. Numbers do not move: `prenormalize` defaulted to `False`.

- **DGSM selects the autodiff mode from the output shape.** `dgsm.analyze`
  and `dgsm.indices` used to run `jax.jacrev` always. They now run
  `jax.jacfwd` when the output slices outnumber the inputs (`T*K > D`) and
  `jax.jacrev` otherwise. There is no keyword: the right mode follows from
  two numbers the library already has. This makes a time-series DGSM call
  cheaper in proportion to `T*K / D`.

  The two modes compute the same Jacobian. Only the order of the float
  arithmetic differs, so where the mode changed (`T*K > D`), `sigma` and
  `lower_bound` can move at float32 precision (about 1e-8). The numerical
  baseline was regenerated once for this; see `scripts/baseline/README.md`.
  The verbose summary now names the mode that ran.

- **`jaxgsa.dgsm.poincare_constant` loses its `grid` keyword.** It took
  `grid: int = 512`, the element count of the finite-element solve for a
  truncated Gaussian marginal. The count is now fixed at that same 512, which
  is what the memoised solve caches on (see "Performance"). Migration: delete
  the argument. Every value the function returns is unchanged.

- **`Problem.input_specs` returns spec dataclasses.** Before, it returned a
  private 6-slot tuple. The tuple used the same two slots for different
  things: `(low, high)` for a uniform marginal, `(mean, variance)` for a
  Gaussian one, and dummy zeros for a categorical one. You had to know the
  slot layout to read it.

  Each entry is now a `jaxgsa.UniformSpec`, a `jaxgsa.GaussianSpec`, or a
  `jaxgsa.CategoricalSpec`. Read the fields by name. Tell the three apart with
  `isinstance`. `jaxgsa.InputSpec` is the union of the three, for annotations.

  ```python
  spec = problem.input_specs[0]
  spec.low, spec.high        # was spec[1], spec[2]
  ```

  `jaxgsa.dgsm.poincare_constant()` takes the dataclass too. It is public, and
  it took the private tuple before, so reading its argument meant knowing the
  slot layout.

  The input side does not change. `Problem.from_dict` still accepts a
  `(low, high)` tuple and the `UniformInputSpec` / `GaussianInputSpec` /
  `CategoricalInputSpec` dicts, exactly as before. It also accepts the new
  dataclasses. A saved `.npz` file keeps its dict form, so files written by
  earlier versions still load.

- **`SobolResult.nan_counts` is removed.** It counted the `NaN` entries in the
  computed indices and threw away which model run produced them, which is the
  one thing you need in order to act. `result.invalid` replaces it and keeps
  the positions. See the `on_invalid` entry under "Added".

- **A non-finite model output now raises by default.** Before, what happened
  depended on the method you called: `sobol`, `morris` and `kucherenko` dropped
  the affected data and warned, `efast` warned and computed anyway, and the
  other nine let the value into the estimator, where it either reached the
  indices unremarked or surfaced later as an error about something else. The
  "Fixed" entries below name those late errors one by one.

  To keep the old behaviour of those first three, pass `on_invalid="drop"`. To
  keep `efast`'s, and that of the nine that let the value through, pass
  `on_invalid="propagate"`.

- **`jaxgsa.config.set_memory_budget` now reads megabytes.** It took bytes
  before, so the same call means a million times more than it used to.

  ```python
  jaxgsa.config.set_memory_budget(512)              # 512 MiB, the default
  jaxgsa.config.set_memory_budget(2, unit="gb")     # 2 GiB
  jaxgsa.config.set_memory_budget(536870912, unit="b")   # the old spelling
  ```

  `unit` accepts `b`, `bytes`, `kb`, `mb`, `gb`, `tb` and the explicit binary
  spellings `kib`, `mib`, `gib`, `tib`. Case and surrounding spaces do not
  matter. The multiples are binary, so `mb` is 1024 squared and
  `set_memory_budget(512)` is exactly the previous default. Values may be
  floats.

  A call that gives no unit and a value of 1048576 or more now raises, because
  such a value is almost certainly bytes and reading it as megabytes would ask
  for more memory than any machine has. The message shows both ways to say what
  you meant. A unit-less value below that is read as megabytes without
  complaint, so a caller who deliberately set a budget under 1 MiB should add
  `unit="b"`.

  Two smaller consequences: the first parameter is now called `budget`, not
  `budget_bytes`, so a keyword call must be renamed; and `unit` is
  keyword-only.

  `get_memory_budget()` is **unchanged**. It still returns an `int` of bytes,
  because a silently changed return value cannot be guarded the way an argument
  can. It gained an optional `unit=`, which returns a float for any unit other
  than bytes.

### Added

- **New guide page, "Scale and limits".** It says how many parameters each of
  the thirteen methods supports, what sets each limit, and which setting moves
  it. The numbers come from a measured sweep over `D` from 10 to 5000 on one
  machine, and the page says so. It also explains why the Saltelli design
  array, and not the analysis, is what limits a high-dimensional Sobol' study:
  `sobol.analyze` adds close to no memory at any `D`, while the design you
  must hold and evaluate grows with the square of `D`.

- **Five fit-quality and usability warnings.** Each one names a state where the
  reported numbers look fine and mean less than they appear to. All are
  `JaxgsaWarning`, so `warnings.filterwarnings("error", category=JaxgsaWarning)`
  turns them into CI failures. No index value changes.

  - `jaxgsa.pce.analyze` now reads its own two fit diagnostics and warns.
    `explained_variance` below 0.5 on any output slice means the expansion does
    not represent the model, and `loo_rmse` above 0.71 times `std(Y)` is the
    same line read out of sample, which is the only place an overfit shows.
    The default `order=3` does not resolve a strong nonlinearity: on Ishigami
    it reports `S1(x1) = 0.662` against a truth of 0.314, and until now it did
    so in silence. `jaxgsa.pce.indices` stays silent, because it traces.
  - `jaxgsa.hdmr.analyze` now checks that the per-term `S` values sum to about
    1, which is the precondition Li et al. (2010, Eq. 24) attach to every index
    the fit reports. The shortfall is variance the decomposition never
    captured, and `Sa`, `Sb`, `S` and `ST` all inherit it.
  - `jaxgsa.optimal_transport.analyze` now warns when `mode="multivariate"` or
    `mode="trajectory"` runs with `dummy=False`. Those modes solve entropic
    transport, whose bias holds an irrelevant parameter's index above zero
    however large `N` is. `dummy=True` measures that floor and reports
    `above_dummy`.
  - `jaxgsa.dgsm.analyze` now warns when every `upper_bound` on an output slice
    exceeds 1. A total Sobol index is at most 1 by definition, so the bound
    then excludes nothing, and the Poincare constant rather than the sample
    size sets the slack, so more samples will not tighten it.
  - `HSICResult` now carries the `bandwidth` and `n_perms` the run used, and
    `to_dataset()` writes both into the dataset attributes. The HSIC index
    moves with the bandwidth, so a stored number without it does not say what
    it measured. This mirrors `SobolResult.estimator`.

- **An agent skill at `skills/jaxgsa/SKILL.md`.** `SKILL.md` carries the
  `Problem`, the four sampling designs, the shape contract, the keywords every
  method shares, and the capability table. Four files under
  `skills/jaxgsa/reference/` carry the per-method call, a worked example against
  a benchmark with the values it prints, and the caveats that decide whether an
  index means anything: `variance-based.md`, `dependent-inputs.md`,
  `screening.md` and `moment-independent.md`. Install it with
  `npx skills add https://github.com/DanielePessina/jaxgsa`.

- **`jaxgsa.kucherenko.sample` warns when the problem declares no
  correlation.** Without a correlation the conditional redraws are plain
  independent draws, so the design reduces to the Saltelli column-swap scheme
  and the indices are the classic Sobol' `S1` and `ST`. You then pay
  `base_n * (2D + 1)` model runs for numbers `jaxgsa.sobol` gives from
  `base_n * (D + 2)`, with `S2` available for one block more. On a
  3-parameter problem at `base_n = 4096` that is 28,672 runs against 20,480,
  or 32,768 with `S2`. The warning quotes all three counts. Nothing raises:
  running Kucherenko on an independent problem to cross-check a correlated
  analysis is a valid use. Silence it with a filter on `jaxgsa.JaxgsaWarning`.

- **`verbose=True` observability on every entry point.** One keyword, the
  same on all seventeen public entry points. On `analyze()` it prints three
  sections. First, the problem and the data: dimensionality, parameter
  names, marginal kinds, correlation status, output layout, and what the
  non-finite check found. Second, the timings: wall-clock time for the
  computation (the first call includes JIT compile time), and the resolved
  batch or chunk widths where batching applies. Third, the results: the top
  `min(D, 5)` parameters by the method's headline index, with confidence
  intervals when you bootstrapped, averaged over the output slices for a
  `(T, K)` output. On `sample()` it prints one line about the design: runs
  generated, and duplicate rows removed where the design deduplicates. The
  output is plain `print()` to stdout, always outside `jit`. The pure
  `indices()` cores stay silent and traceable. Pass `verbose=False` for a
  silent run; a test in `tests/test_vocabulary.py` pins the keyword on every
  entry point.

- **`EFASTSamples` can now save and load.** It was the one design object
  without persistence. `save()` writes one compressed NPZ file with the same
  layout as the other designs: the sample matrix plus a JSON metadata blob
  with the problem, the design parameters and the jaxgsa version. `load()`
  rebuilds the design, and analyzing it gives the same numbers.

- **PAWN reports how many bins carried each index.** A conditioning bin with
  fewer than two samples gives no KS value and was dropped without a word.
  `PAWNResult.n_valid_bins` now counts the contributing bins per parameter
  (same shape as `pawn`, exported by `to_dataset`). When a parameter keeps
  fewer than half of its bins, one warning names it and the fix: use fewer
  bins or more samples.

- **VKOGA checks that the training design is independent.** The estimator
  needs an independent, space-filling training `X` even for a correlated
  analysis, because `S_TU` resamples each `X_i` across its whole marginal. A
  correlated analysis now fits the training sample's own rank correlation
  and warns when an off-diagonal entry is above 0.3. The warning says why
  (the surrogate extrapolates on the conditional draws) and where the
  correlation belongs: in `problem.correlation`, not in the training data.

- **Near-singular conditionals are no longer repaired in silence.** The
  Gaussian-copula conditional plan (kucherenko designs, vkoga indices) used
  to floor collapsing conditional covariances without a word as `|rho|`
  approached 1. It now warns once per call, with the smallest conditional
  variance — or the eigenvalue lift, when the floor really engaged — as the
  severity.

- **`OTResult` gains `S1` and `above_dummy`.** `S1` is the given-data
  first-order Sobol index, the advective component rescaled onto the
  population-variance (ddof=0) convention that `jaxgsa.borgonovo` uses, so
  the two estimates share one definition and no ddof caveat is left.
  `S1_conf` scales `advective_conf` the same exact way. `above_dummy` is
  `max(ot - ot_dummy, 0)`: the part of the total index above the
  irrelevance floor, present when the analysis ran with `dummy=True`.

- **`jaxgsa.__version__`.** The installed package version, from
  `importlib.metadata`.

- **`Theta` is re-exported** at the top level and from `jaxgsa.sobol`, so
  gradient users can type their `transform(theta)` code without importing
  from `_core`.

- **`Problem` rejects bad parameter names at construction.** A non-string
  name or a duplicate now raises a `ValueError` that names the fix, instead
  of failing later in a `Theta` lookup or a dataset export.

- **Eleven methods gain a pure `indices()` core.** `sobol`, `efast`, `pawn`,
  `morris`, `hsic`, `borgonovo`, `optimal_transport`, `dgsm`, `pce`, `hdmr`
  and `shapley` each export one. No method exported an `indices()` before.

  A core takes the design object (or `problem, X, Y`) and returns a bare tuple
  of arrays. No result class, no diagnostics, no host read of any array value,
  and every branch on a shape or a Python scalar. It survives `jit`, `vmap`
  and differentiation, so an index can sit inside a larger JAX computation.

  ```python
  S1, ST, S2 = jaxgsa.sobol.indices(samples, Y)
  grad = jax.jacrev(lambda y: jaxgsa.pce.indices(problem, X, y)[0].sum())(Y)
  ```

  `analyze()` keeps the policy: validation, invalid-row handling, warnings and
  the bootstrap. That split is why `analyze()` cannot be traced and
  `indices()` can — dropping rows by value makes the row count depend on the
  data, and `jit` needs static shapes.

  `kucherenko` and `vkoga` have **no** core, and say so. Both are host NumPy
  and SciPy end to end; kucherenko is the fastest method in the library
  precisely because it never touches the device. The exemption is declared as
  `MethodSpec.pure_core` and checked — a method claiming a core it does not
  have now fails a test.

  Two limits are structural, not incidental. A core refuses **categorical
  inputs**, because a categorical partition pads to `counts.max()`, a shape
  read off the data. And `hdmr.indices` supports `jacfwd` but not `jacrev`,
  because its backfitting stops early through a `lax.while_loop`, which JAX
  will not differentiate in reverse.

- **Six more methods report confidence intervals.** `dgsm`, `kucherenko`,
  `pce`, `hdmr`, `vkoga` and `shapley` now take `n_bootstrap`, alongside
  `conf_level`, `ci_method`, `key` and `keep_replicates`. Eleven of thirteen
  methods now offer an interval.

  `ci_method` now reaches every method that bootstraps. `pawn`, `borgonovo`
  and `optimal_transport` bootstrapped already but were hard-wired to
  percentile endpoints. All three now accept `"quantile"` or `"gaussian"`,
  and `"quantile"` is the default, so their intervals do not move.

  `n_bootstrap` defaults to `0` everywhere, so nothing costs more than before
  unless asked. That default matters most for the four surrogate-backed
  methods, which refit their surrogate on **every** replicate.

  The resampling unit is the one the design allows: rows for the given-data
  methods, and **base points** for kucherenko, each carrying `2D+1`
  conditional rows — resampling individual rows there would leave the
  estimator reading misaligned blocks.

  Not every field gets an interval, and the omissions are deliberate.
  `var_y` and `variance` are the denominators of the indices rather than
  sensitivity measures, and their uncertainty is already inside the index
  intervals. `rmse`, `cv_rmse` and `n_centers` describe the fit that was
  reported, so an interval over other fits would not be about the thing they
  name.

- **`jaxgsa.pce.effective_order(problem, n_samples, *, order, fit_ratio)`**
  answers what order a PCE fit will actually use, with no fit and no side
  effect. PCE may reduce the requested order when the design matrix would be
  underdetermined; that used to be reported only by a warning during the fit,
  which a pure core cannot emit.

- **`estimator=` on `sobol.analyze()` and `sobol.indices()`.** Six named
  estimator pairs, where before the formulas were fixed with no option.

  ```python
  result = jaxgsa.sobol.analyze(samples, Y, estimator="azzini-rosati")
  ```

  | Name | First order | Total order | Design |
  |---|---|---|---|
  | `"saltelli-jansen"` (default) | Sobol' et al. (2007) | Jansen (1999) | `N(D+2)` |
  | `"jansen"` | Jansen (1999) | Jansen (1999) | `N(D+2)` |
  | `"janon-monod"` | Monod (2006), Janon (2014) | same | `N(D+2)` |
  | `"martinez"` | Martinez (2011) | Martinez (2011) | `N(D+2)` |
  | `"mauntz-kucherenko"` | Sobol' et al. (2007) | Sobol' et al. (2007) | `N(D+2)` |
  | `"azzini-rosati"` | Azzini, Mara & Rosati (2021) | same | `N(2D+2)` |

  **No number moves.** The default is the estimator jaxgsa has always used,
  and the numerical baseline reports zero changed values. The default was
  measured before it was kept: on Ishigami and Sobol-G against their
  analytical indices, over 100 seeds, with each estimator given the design it
  needs so the model-run budgets are comparable, `"saltelli-jansen"` is the
  best or joint-best pair for the `N(D+2)` design at every budget tested.

  `"azzini-rosati"` needs a design drawn with `calc_second_order=True`,
  because it reads the `BA` blocks. It is the better choice at a small budget:
  on Sobol-G at 640 model runs its `S1` error is 0.029 against 0.087 for the
  default. It is also the only pair that can never report `S1 > ST`. The
  advantage narrows as the budget grows.

  Second-order indices keep the Saltelli (2002) pairwise formula for every
  estimator. Only the `S1` terms it subtracts follow your choice.

- **`on_invalid` on every `analyze()` function.** It takes `"raise"` (the
  default), `"propagate"` or `"drop"`.

  ```python
  result = jaxgsa.sobol.analyze(samples, Y, on_invalid="drop")

  result.invalid.n_invalid        # how many blocks held a non-finite value
  result.invalid.unit_indices     # which blocks
  result.invalid.bad_row_indices  # the rows that actually failed
  result.invalid.row_indices      # every row those blocks cover
  result.invalid.sources          # whether they were in X, in Y, or both
  ```

  `bad_row_indices` and `row_indices` answer different questions. One failed
  run inside an eFAST search curve gives one entry in the first and 257 in the
  second: the first says which model run to investigate, the second says what
  `"drop"` would remove. Both use the numbering of the array you passed, so a
  Saltelli or Morris design reports the unique rows you evaluated, not the
  expanded rows it analyses internally.

  The default refuses because an index computed from part of a sample is a
  different quantity from the one you asked for, and `analyze()` is cheap to
  run again once you know which runs failed.

  What `"drop"` removes depends on the design. A Saltelli group, a Morris
  trajectory and a Kucherenko base point are each read as one block, so one bad
  value removes the whole block; keeping part of one would leave the estimator
  reading rows that no longer line up, with nothing to report it. For the
  given-data methods one bad value removes one row. A bad input row always
  takes its matching output row with it.

  `jaxgsa.efast.analyze` takes `"raise"` and `"propagate"` only. Its design is
  an ordered sweep read by a Fourier transform, so removing a point does not
  shrink the sample, it changes what the estimator computes. Asking for
  `"drop"` raises and says so.

- **`jaxgsa.InvalidReport` and `jaxgsa.InvalidUnit`**, the two supporting
  types. Every result now carries `result.invalid`, whichever policy ran. A
  report with `n_invalid == 0` means the check ran and found nothing, which is
  not the same as no check having run. Positions always refer to the array as
  you passed it, before anything was removed.

- **`result.ci` records how a confidence interval was made.** Until now a
  `*_conf` array was an interval of unknown level: nothing on the result said
  whether it was a 95% or a 68% interval, which endpoint rule drew it, or how
  many resamples it rested on. Every result that reports intervals now carries
  a `ci` field holding `level`, `method`, `n_bootstrap` and, on request,
  `replicates`. `S1` and `S1_conf` are unchanged plain arrays.

- **`keep_replicates=True` keeps the bootstrap draws.** `analyze` discarded
  them, so recomputing an interval at another level meant re-running the whole
  analysis. Pass `keep_replicates=True` and the draws arrive on
  `result.ci.replicates`, keyed by the estimate they belong to. It is off by
  default because the draws are large: 1000 resamples of a
  `(T=100, K=5, D=20)` index array is 80 MB. Every method that reports an
  interval accepts the keyword, and it is keyword-only and last in each
  signature.

  The stored draws follow the reported convention. Sobol's `S2` replicates are
  symmetric with a NaN diagonal, exactly as the `S2` point estimate and its
  intervals are, so read the upper triangle.

- **`SobolSamples.unit` and `SobolSamples.transform(theta)`.** `unit` holds the
  design in the unit cube, before any input distribution is applied, so it does
  not depend on the distributions at all. `transform` applies a set of
  distribution parameters to it. That allows one design to be reused across
  different assumed input ranges, and, because `transform` is written in JAX,
  it allows a Sobol index to be differentiated with respect to those
  parameters:

  ```python
  def s1(theta):
      return jaxgsa.sobol.indices(samples, model(samples.transform(theta)))[0]

  dS1_dtheta = jax.jacrev(s1)(theta)
  ```

  The chain runs through the model, so the model must be differentiable in
  JAX, and float64 should be enabled. `transform` raises for a categorical
  problem: a categorical inverse CDF is a step function, so it has no useful
  derivative and `unit` and `samples` do not have the same number of rows.

- **Every result class now prints a one-line summary.** `MorrisResult`,
  `DGSMResult`, `DeltaResult`, `PAWNResult` and `OTResult` had no `__repr__`,
  so echoing one in a notebook printed every index array and the whole
  `Problem`. All thirteen now print their field shapes and their provenance,
  in one style.

- **`PCEResult.streamed` and `HDMRResult.streamed`.** They record which fit path
  ran. A fit that took much longer than expected is a real reason to ask whether
  the memory budget engaged, and until now nothing answered that.

- **`jaxgsa.JaxgsaWarning`**, exported from the package root. Every warning the
  package raises now passes this category. Before, all of them defaulted to
  `UserWarning`, so the only way to tell a jaxgsa warning from a NumPy or JAX
  one was the message text, and no filter selected exactly this package's
  warnings. To silence them:

  ```python
  warnings.filterwarnings("ignore", category=jaxgsa.JaxgsaWarning)
  ```

  `JaxgsaWarning` subclasses `UserWarning`, so filters and
  `pytest.warns(UserWarning, ...)` assertions that worked before still work.

### Changed

- The PCE fit-quality warning now comes from `jaxgsa.pce` rather than
  `jaxgsa.shapley`. `jaxgsa.shapley.analyze(backend="pce")` and
  `PCEResult.shapley()` both reach it through the backend's own `analyze`, so
  the warning still fires on the same fits, once, under the namespace that owns
  the number. `HDMRResult.shapley()` likewise defers to `jaxgsa.hdmr.analyze`.

- **The Sobol default-estimator rationale is the real one.** The docstring
  used to justify `estimator="saltelli-jansen"` with historical continuity.
  The recorded reasons are now that Jansen's total-order estimator is a mean
  of squares, so `ST` can never go negative, and that it matches SALib's
  default pairing, so the two libraries agree out of the box. The default
  itself does not change.

- **Hygiene sweep for 1.0.** No number moves and no schema changes. In short:

  - Every warning now starts with `jaxgsa.<method>:`, so you can see which
    method spoke. Before, six prefix spellings were in use: a bare `jaxgsa:` on
    most warnings, `jaxgsa.vkoga:`, `jaxgsa.sobol.sample:`, and the bare
    `PAWN:`, `eFAST:` and `DGSM:`.
  - The vkoga `S_U` clip warning now names the parameters. It used to give
    integer positions.
  - `benchmarks` is importable as `jaxgsa.benchmarks`, like every other
    subpackage. The broken examples in the `benchmarks`, `hsic` and `dgsm`
    package docstrings now run as written.
  - `morris.analyze` refuses a missing bootstrap `key` right after the input
    checks. It used to compute the full point estimate first.
  - An internal dgsm consistency check is now a real error instead of an
    `assert`, so it still fires under `python -O`.
  - Dead code is gone: an unused validator, the unused per-parameter Sobol
    estimator trio and its legacy test module, an unused result-axes
    variant, and two never-used parameters.
  - Stale docstrings and comments now describe the current code, and the
    remaining test modules without per-test tier docstrings carry a
    module-level oracle-tier line instead (T0 closed form, T1 recorded
    external literal, T2 external library at test time, T4 internal
    consistency only).

- **eFAST and HSIC report no bootstrap interval, and the docs now say why.**
  eFAST has one search curve per parameter, so there is nothing to resample —
  removing a point does not shrink the sample, it changes what the estimator
  computes. HSIC already reports permutation `p_values`, which is the
  uncertainty statement for a V-statistic; a row bootstrap would repeat rows
  onto the kernel diagonal, where the kernel is exactly 1, biasing the
  resampled index upward by construction.

- **One vocabulary for the whole interface**, enforced by
  `tests/test_vocabulary.py` reading it back off the method registry. A
  signature that drifts from the specification now fails a test rather than
  shipping. Two rules the code does not satisfy yet are recorded as strict
  xfails, so closing the gap forces the exemption to be deleted.

- **A result declares its fields, and `to_dataset` is derived from that.**
  There were thirteen hand-written `to_dataset` methods and four copies of the
  `param_i`/`param_j` splice. Each result now declares a field schema, and the
  export, the summary and the `*_lower`/`*_upper` split all read it. The
  exported dims, coordinates and variable names are unchanged for all thirteen
  results, which `tests/test_result_schema.py` pins against a snapshot taken
  from the old methods.

- **`SurrogateResult` no longer declares `shapley`.** The abstract method fixed
  no contract: `VKOGAResult` could only satisfy it by raising, and
  `HDMRResult` widened the signature. `PCEResult.shapley` and
  `HDMRResult.shapley` are plain methods now, and `VKOGAResult.shapley` still
  raises `NotImplementedError` with the same message. `predict` stays on the
  base class. This also removes an inverted dependency: `jaxgsa._core` no
  longer imports a method package's private module.

- **NumPy is now a declared dependency**, at `numpy>=2`. NumPy is imported
  directly by 45 of the package's 74 modules, but it reached users only
  through JAX and SciPy, both of which allow NumPy 1.x. An install from PyPI
  could therefore get a NumPy the package does not support.

- **The SciPy floor rises to `scipy>=1.15`**, from `>=1.10`. This makes
  `scipy.stats.chatterjeexi` always available as a verification oracle.

- **One compatibility matrix, and a test that keeps it true.** The
  documentation stated it three times and the three disagreed: one table had a
  short row that rendered a column blank, one credited DGSM with a sampler it
  does not have, and one listed eight of the nine given-data methods. There is
  now one table, "Method capabilities" in `docs/guide/methods.md`. It gains a
  Bootstrap CI column that names the keyword each method uses.
  `docs/index.md` and `docs/api/index.md` link to it instead of restating it.
  `tests/test_docs_matrix.py` checks every cell against the method
  registry, so a fourteenth method, or a changed capability, fails the
  suite rather than leaving the documentation stale.

- **Packaging metadata for a stable release.** The PyPI classifier moves from
  `Development Status :: 4 - Beta` to `5 - Production/Stable`. `CITATION.cff`
  keeps its `version` and `date-released` fields, and the release process now
  sets them from the tag. The planning documents at the repository root are
  gitignored, so a working tree during a release is clean.

- **"Every estimator runs under jit" was not true.** `README.md` and the
  `CITATION.cff` abstract both said it. Eleven methods export a traceable
  core. `kucherenko` and `vkoga` are host NumPy and SciPy by design, as the
  registry has recorded all along. Both texts now say so.

### Removed

- **`SobolSamples.sample_ids`.** It always held `arange(n_runs)`, had no
  reader anywhere in `src/`, and existed only to be saved, loaded and
  documented. Join on row position instead; it was always the same thing.
  `.npz` files written by an older jaxgsa still load: the field is dropped
  from the metadata, not required.

- `_PCEFit.coeffs_flat`, a private field that was written and never read. It
  kept a large array alive inside the module whose purpose is bounding memory.

- The private `validate_correlation` helper. It had no production caller and
  stayed alive only because tests used it. `canonicalize_correlation` runs the
  same code, and its tests now call that instead.

### Performance

- **The PCE streamed fit dispatches one jitted step per row batch.** Each
  batch used to run several eager ops: an unjitted design-matrix build plus
  accumulation matmuls. The per-batch step is now one jitted call, and the
  ragged trailing batch is zero-padded and masked, so each step compiles
  once. Padded rows contribute exact zeros to the normal equations and to
  the leave-one-out sum. This is the pattern the HDMR fit already uses,
  where the same fix was measured at 1.2-1.67x.

- **The truncated-Gaussian Poincare constant is cached.** DGSM's FEM
  eigensolve for a truncated Gaussian marginal used to run on every
  `analyze` call. It is a pure function of five scalars, so it is now
  memoised with `functools.lru_cache` and runs once per distinct marginal
  per process. Zero numeric change: the cached value is the value the solve
  returned.

- **PCE second-order extraction no longer builds an unbudgeted transient.**
  `sobol_from_coefficients` materialized an `(n_terms, D*(D-1)/2)` float
  array for the S2 pair mask — about 3.5 GB at `D=100`, order 3. The pair
  axis now streams in chunks sized by the memory budget. At small `D` the
  budget allows one chunk, which runs the exact old matmul, so shipped
  values are bit-for-bit unchanged.

- **`jaxgsa.hsic.analyze` is about 10x faster on many outputs.** On a 1024-row
  Ishigami problem with 128 output slices the call went from 32.2 s to 3.2 s,
  and the cost of one slice went from 252 ms to 25 ms. Compile time no longer
  grows with the number of slices.

  Two changes did it. The estimator now runs as one kernel over one output
  slice, mapped over every slice inside a single compiled call; before, a
  Python loop over T and K crossed the compile boundary once per slice and
  wrote each answer back from the host. And the median bandwidth heuristic
  now *selects* its two order statistics instead of sorting the whole
  `(N, N)` distance matrix. The sort alone was 92% of the run time.

  The same heuristic also allocates about a third of what it did. It built a
  distance matrix, two index arrays and a copy of the upper triangle, only to
  skip the diagonal zeros. Those zeros are the smallest entries, so their
  effect folds into the quantile position instead. Measured on an Apple M1 Pro
  in float32: peak transient memory falls from `3.57 * N^2` to `1.00 * N^2`,
  which is 228 MiB down to 64 MiB at `N = 4096`.

  No index changed. Every value is bit-for-bit what it was.

- **`jaxgsa.optimal_transport.analyze` is about 2x faster on many outputs.**
  At 128 output slices the call went from 511 ms to 258 ms, and the cost of
  one slice from 4.0 ms to 2.0 ms. The per-class sort inside the 1-D kernel
  was a *stable* sort, which the estimator never needed: it only wants order
  statistics, and equal float32 values are the same bits either way. An
  unstable sort returns the identical array. The rank table is also built
  once for the whole replicate scan instead of once per replicate. No index
  changed.

- **`jaxgsa.borgonovo.analyze` uses about half the memory on many outputs.**
  At 128 output slices peak resident memory went from 402 MiB to 179 MiB.
  The conditional-KDE tensor is now evaluated in tiles across the output
  grid, and the slice chunk is sized from `set_memory_budget` instead of a
  hardcoded element count. Run time is unchanged at 64 slices and below; at
  128 slices it is flat to a few percent slower, which is the trade for the
  memory. No index changed: each grid point keeps its own sum over its own
  class members, and every tile width from 2 upward is bit-identical to the
  untiled result.

- **The Sobol bootstrap is 9-20x faster on many outputs.** On a 1024-row
  Ishigami problem with 128 output slices, `analyze(..., n_bootstrap=...)`
  went from 84 ms to 4-9 ms. The resampler now runs one estimator kernel over
  one output slice, mapped over the resample draws and then over a chunk of
  slices; before, a Python loop dispatched twice per slice. Confidence
  endpoints are one call on the whole grid instead of one per slice.

  `slice_chunk_size` now caps output slices on both the plain and the
  bootstrap path. On the bootstrap path it used to cap resample draws.

  A chunk that holds one slice takes a single-axis path instead. Mapping over
  a length-one slice axis makes every gather batched on two axes, and XLA
  compiles that to a slower gather than a single-axis one, so the outer
  `vmap` is dropped there. On an Apple M1 Pro at N=1024, D=3 and 1000
  resamples, one output slice takes 8.5 ms. Every interval endpoint is
  bit-identical between the two paths, on every estimator. The shortcut also
  applies when many slices exist but the memory budget has narrowed the chunk
  to one.

  This moves 44 sobol values in the last 1 to 4 bits of float32. Widening the
  batch is the speedup, and XLA schedules a float32 reduction differently at
  a different width. `scripts/baseline/README.md` records the review.

- **Row deduplication is about 3.4 times faster.** `_stable_unique_rows` runs
  twice on the Sobol sampling path and once on the Morris path. It built one
  array view and one `bytes` key per row in Python. It now builds every key in
  one C-level call and keeps the dictionary that was already doing the work.
  Measured on an Apple M1 Pro at `N = 2**20`, `D = 20`, float64: 1099 ms
  before, 325 ms after. Output is bit-identical, including row order, which
  matters because the design prefix logic depends on it.

### Fixed

- **PCE and Shapley no longer crash on a problem with about 1000 parameters
  or more.** `build_multi_index` enumerated the basis with one recursion per
  parameter, so its recursion depth was `D`. CPython stops at 1000 frames.
  `pce.analyze` and `shapley.analyze`, which uses the PCE backend by default,
  therefore raised `RecursionError` at `D >= 997`. This happened at every
  order. Even order 1 failed, and its basis is only `D + 1` terms and a few
  megabytes, so the crash had nothing to do with problem size. The good
  "cannot fit even the order-1 expansion" error never got the chance to run.
  The enumerator now works by support: it picks which parameters are
  non-zero, then splits the total degree between them. A term of total degree
  `p` has at most `p` non-zero entries, so the walk no longer scales with
  `D`. The multi-index set it returns is identical, term for term and in the
  same order. A fit at `D = 1000` now completes. At `D = 5000` you get the
  clear sample-budget error instead of a crash.

- **Sobol's `S2` now averages its two triangles instead of keeping one.**
  `S2[j, k]` and `S2[k, j]` come from the same formula with `j` and `k`
  swapped, so both are independent Monte Carlo estimates of the same
  pairwise index. The previous code kept the upper triangle and mirrored it,
  discarding the lower triangle's estimate. The two differ by sampling
  noise, not floating-point drift (mean absolute gap 0.026 on Ishigami at
  `base_n=256`). Averaging them usually lowers the error but does not always:
  measured on Ishigami over 200 seeds and all three parameter pairs, the RMSE
  falls 25% at `base_n=64` and 13% at `base_n=1024`, and rises 9% at
  `base_n=256`, where one pair's lower-triangle estimate happened to be three
  times noisier than its upper one. Averaging is still the better default,
  because a caller cannot tell which triangle is the lucky one. This is a
  deliberate departure from SALib, which reports the upper triangle alone, so
  `S2` no longer matches SALib bit for bit, only up to that same noise.
  `S2_conf` moves too: the bootstrap now averages the two triangles of every
  draw before it reads an endpoint, so the interval describes the same
  averaged quantity the point estimate reports. It used to read its endpoints
  from the raw draws and average afterwards.

- **Sobol standardizes the outputs, always, and this fixes real numbers.**
  The Sobol'-Mauntz first-order estimator and every second-order estimator
  are *uncentred* products, so a non-zero output mean adds an error term
  proportional to that mean. On Ishigami at N=4096 with an output offset of
  1e4, `S1` came back `[6.26, 0.434, 1.71]` against the analytic
  `[0.314, 0.442, 0.000]`. Float64 gave `[6.27, 0.433, 1.72]`, so this was
  estimator bias and not rounding.

  `sobol.analyze` and `sobol.indices` now standardize every output slice to
  mean 0 and unit standard deviation over the sample axis before the
  estimators run, which is what SALib has always done
  (`SALib/analyze/sobol.py`: `Y = (Y - Y.mean()) / Y.std()`). It happens in
  one place that both paths reach, so the traceable core and the checked
  entry point cannot disagree, and the bootstrap resamples an
  already-standardized array.

  Sobol `S1` and `S2` point estimates and intervals move. `ST` moves only in
  the last bits of a float32 result: the Jansen total-order estimator is a
  difference, so it was already shift-invariant. Be clear about the size of
  the win at a *small* output mean: Ishigami's own mean is 3.5, and there the
  change is close to a wash (largest S1 error 0.106 against 0.123 at
  N=1024, 0.0017 against 0.0017 at N=16384). What it removes is an error term
  proportional to the output mean, whose size was otherwise unpredictable: at
  the same N=1024 with an offset of 1e4 the largest S1 error was 50.8, and it
  is now still 0.106.

- **VKOGA derived its per-parameter streams by adding to a seed.** The index
  estimator seeded its quasi-Monte-Carlo draws with `seed + 1 + i` and
  `seed + 7919`. Streams that differ by a constant are not independent, which
  is the reason the public interface moved from `seed` to `key`. The
  estimators are host-side scipy, so they cannot split a key; they now spawn
  one `numpy.random.SeedSequence` child per draw, which is the host-side
  equivalent. Every VKOGA index moves by the size of its own Monte-Carlo
  noise. The fitted surrogate is unchanged: `gamma`, `ridge` and the greedy
  centres are bit-for-bit the same.

- **VKOGA ignored `batch_size` in its index estimator.** The keyword reached
  the surrogate `predict` path only; the estimator's own chunking passed
  `None`, so the caller's value was silently dropped. It is now threaded
  through and is a required argument internally, so it cannot be dropped
  again.

- **HSIC now warns in single precision.** Its V-statistic is a difference of
  three same-magnitude sums, so float32 keeps three or four digits and the
  index moves with row order — measured at 4e-4 relative against 8e-13 in
  float64. VKOGA was previously the only method in the library that checked
  the x64 flag.

- **eFAST derives its slice chunk from the memory budget.** It used a fixed
  2048 that ignored both dtype and `N`, so one chunk was about 4 GiB at
  `n_per_curve = 65536` in float64.

- **`to_dataset()` lost the analysis settings.** A result printed its settings
  in its summary, but it did not always export them. `eFAST`, `HDMR`, `PCE`,
  `Sobol` and `VKOGA` exported none of them. Shapley dropped its `order`.
  Kucherenko exported `correlated` although it printed `is_correlated`. A saved
  dataset therefore did not say which estimator, which order, or which mode
  produced it.

  A result now declares these settings once. The summary and
  `ds.attrs` both read that declaration, so the two cannot disagree. Every
  method exports what it prints, under the same name.

  ```python
  ds = jaxgsa.sobol.analyze(samples, Y).to_dataset()
  ds.attrs["estimator"]  # 'saltelli-jansen'; was absent
  ```

  Two names change. `ds.attrs["correlated"]` is now
  `ds.attrs["is_correlated"]`, on Kucherenko and on VKOGA. The `method` key is
  removed from those two datasets. Only they wrote it, and the result class
  already names the method.

  Every value is a plain string, number or boolean, so `to_netcdf` writes it. A
  setting that does not apply leaves its key out. VKOGA writes no `cv_rmse`
  when no cross-validation ran, because netCDF has no null attribute.

- **A Sobol bootstrap no longer reports indices that differ from the plain
  analysis.** `analyze(sr, Y, n_bootstrap=20).S2` and
  `analyze(sr, Y, n_bootstrap=0).S2` could differ in the last bits for the
  same design, because the two paths ran the estimator at different batch
  widths. An interval was centred on a number the plain analysis never
  reported. Both paths now run the same kernel and agree bit-for-bit.

- **The first-order Sobol estimator is now attributed correctly.** The
  docstrings called `E[B (AB_j - A)] / Var(Y)` "the Saltelli (2010)
  estimator". Saltelli et al. (2010) tabulate and recommend it, but the
  formula is the improved form of Sobol', Tarantola, Gatelli, Kucherenko and
  Mauntz (2007). The Saltelli (2002) estimator is the plain cross-moment,
  which jaxgsa does not use. No number changes.

- **PAWN silently discarded non-finite inputs.** A `NaN` in `X` failed the
  in-range comparison in `_equal_width_bins` and took the `-1` sentinel, so
  that sample vanished from every conditional set with no warning. Genuinely
  out-of-range values still use the sentinel; a non-finite value can no longer
  reach it.

- **Optimal transport reported a failed computation as zero influence.** The
  normalization guarded with `V > 0`, which is false for a `NaN` variance, so a
  non-finite output gave `ot = 0.0` for every parameter — indistinguishable
  from a parameter that does nothing. It now gives `NaN`. A constant output
  still gives exactly `0.0`.

- **DGSM could not see a non-finite derivative.** Its bound-consistency warning
  excludes non-finite entries from the comparison, so a `NaN` Jacobian with a
  finite output passed unremarked. Both call styles now check the derivative as
  well as the output.

- **Borgonovo reported a failed run as a bandwidth problem.** A non-finite
  output surfaced only after the whole kernel-density estimate and bootstrap
  had run, as an out-of-range delta error mentioning `degenerate_bandwidth`.
  It is now named as a bad row up front.

  The out-of-range check still guards genuine estimator failure, and it now
  says how to fix it. The estimate is a half L1 distance, so a value outside
  `[0, 1]` is a failed computation. The error reports whether a conditioning
  class was floored, which output column failed, the bandwidth actually used
  against the real grid step, and the value that would resolve it.

- **VKOGA reported a failed run as a solver failure.** A non-finite output made
  every cross-validation score non-finite, which surfaced as
  `"Every cross-validation score is non-finite; the kernel solves failed"`.
  That guard remains for real solver failure, but is no longer reachable from
  non-finite input.

- **DGSM: passing arguments from both call styles now raises.** `dgsm.analyze`
  accepts either a model and inputs, or precomputed outputs and derivatives.
  The dispatch was first-match, not exclusive, so
  `analyze(problem, X=X, Y=Y, dfdx=J)` took the precomputed branch and
  discarded `X`. The check that validates `X` against the problem's bounds and
  shape runs only in the other branch, so a user who passed inputs and believed
  they were checked had them thrown away unchecked. **Now raises**, naming the
  conflicting arguments.

- **DGSM: a batch model now raises up front.** `dgsm.analyze` differentiates a
  one-sample function, `(D,) -> ...`. Every other method in the package takes
  `(N, D)`, so passing a batch model is an easy mistake, and it used to fail
  with an `IndexError` from deep inside the autodiff machinery. **Now raises**
  with the expected signature and a wrapper snippet. The check costs no model
  evaluations: it traces shapes only. An unrelated failure inside your model is
  reported plainly, without the wrapper advice.

- **PAWN: `slice_chunk_size` now does something.** It was declared, documented
  as "accepted for signature parity", never validated, and never used, so the
  whole `(T*K, D, N, n_bins)` working set was built in one call — on the
  time-series case this project recommends. It now chunks the output columns,
  and `slice_chunk_size=0` **now raises** like it does elsewhere. Its default
  is `None`, derived from the memory budget, so the width adapts to `N`, `D`
  and `n_bins`; the old fixed 2048 was compared against `T*K`, which is
  smaller than 2048 in every realistic case. Only peak memory changes. Every
  index is identical.

- **eFAST computed its frequency plan twice.** The sampler built the design
  from one copy of the formula and the analyzer recomputed it from another. The
  two agreed only because one frequency band happened to contain the other, and
  no test could catch a divergence, because the tests recomputed the formula
  too. Both sides now read one plan, which checks its own invariant, and a new
  test recovers the bands from a synthetic signal instead of from the formula.

- **PCE computed its leave-one-out error twice, two different ways**, with a
  comment asserting the two agreed. One copy built an array of `(n_terms, N)`
  and then transposed it, so a third array scaling with the sample count was
  alive at once. Both paths now take the leverage from a Cholesky factor of the
  Gram matrix, which is `(n_terms, n_terms)`. The Gram matrix is never
  inverted: the default ridge is deliberately small and PCE conditioning
  degrades as the polynomial order rises, so an explicit inverse would make a
  bad condition number worse, and the leave-one-out value feeds back into
  automatic order selection.

  The memory estimate that decides when the streaming fit engages was wrong in
  the same place. It charged three sample-sized arrays for a phase that holds
  two. It now charges the larger of the fit's two phases, which leaves a
  scalar-output fit exactly where it was and only moves the threshold for
  multi-slice fits.

- **HDMR unpacked twelve values positionally** at three call sites, two of them
  with blind placeholders. Reordering them produced silently wrong indices
  rather than an error. They are now named fields. The per-term order map, which
  was written out identically in two files, is now written once.

- `MorrisSamples.downsample` no longer carries a previous design's dropped-block
  count into the smaller design. A `downsample` caller names the trajectory
  count and receives exactly it, so nothing is missing. Carrying the count
  forward made `morris.analyze` warn about a "requested" total the user had
  never asked for.

- **`jax.jit(hdmr.indices)` used to poison every later `hdmr` call at the same
  row count.** `_compute_f_crits` cached its result with `functools.lru_cache`
  and returned a `jnp.array`. Under a `jit` trace that array is a tracer, and
  the cache kept it: the next call from outside that trace raised
  `UnexpectedTracerError`. The cached function now returns a plain tuple, and
  the caller converts it to an array itself. No number changes.

- **eFAST raised `OverflowError` for `n_per_curve >= 46341` in the default
  float32 config.** The kernel divided by the Python integer `N**2` before
  taking the ratio; JAX passes that literal into the jitted call as `int32`,
  and `46341**2` overflows it. It now divides by `N` before squaring, which
  is the same ratio and cannot overflow. `S1` and `ST` move by rounding only
  (float32 reassociation, on the order of 1e-8) at every `n_per_curve`,
  including sizes well under the old crash threshold.

- **VKOGA's `S_TU` reused one inner Monte Carlo block for every outer point.**
  The estimator averages an inner-conditional variance over outer draws, so a
  block shared across outer points is a *bias* that raising `n_outer` cannot
  reduce — the opposite of the advice `analyze`'s docstring gave. Each outer
  point now gets its own stratified inner block. Measured on an exact model
  with no surrogate error, the run-to-run standard deviation of `S_TU` drops
  by one to two orders of magnitude at the default sizes (0.044 → 0.0023 on a
  quadratic term, 0.317 → 0.022 on a cubic one). `_total_correlated`'s shared
  block is untouched: there the shared block is a variance of means, which is
  correct.

- **VKOGA's additive components fit a fixed Legendre basis regardless of the
  marginal.** Legendre in `u = Φ(x)` is not orthonormal for a Gaussian
  marginal in `z`, so `S_C` and `S_IU` read nonzero for models that are
  provably additive or independent, with no warning. Each component now fits
  in a basis orthonormal under its own marginal — probabilists' Hermite in
  `z` for an untruncated Gaussian column, Legendre in `u` otherwise, reusing
  the basis machinery `pce` already has. `S_U`, `S_C` and `S_IU` move for any
  problem with a Gaussian marginal; `S_TC` and `S_TU` do not.

- **VKOGA's cross-validation grid for `ridge` started an order of magnitude
  below the numerical noise floor.** `_RIDGE_GRID` started at `1e-16`, but the
  kernel matrix already carries a `1e-8`-relative jitter (about `2.2e-6` at
  `n = 1024`), so six of the ten grid points were numerically indistinguishable
  and a cross-validation pick among them reported whichever tied first.
  `VKOGAResult.ridge` was therefore not always the regularisation actually in
  force. The grid now starts at the jitter scale. Indices do not move; the
  reported `ridge` can pick a different (still tied) grid point.

- **HDMR's backfitting stop rule used an absolute threshold, so it stopped
  after one sweep on a correlated problem with a small or large output
  scale, or `m >= 4`.** The coefficients it compares scale with `Y` and with
  `m**-3`, so `varmax > 1e-3` is not a fixed bar. The rule is now relative:
  `varmax > 1e-3 * max(var_new)`. Independent-input fits barely move (backfitting
  only does work when dimensions couple); a correlated fit that used to stop
  early can move a lot — the reviewed case moved `x1`'s `S1` from a
  stopped-early 0.014-0.066 to a converged ~0.104, more than 5x. This also
  means the earlier claim that dropping `prenormalize` from `hdmr` was "a
  measured no-op (1e-6)" held only for the independent, `O(1)`-scaled test
  problem it was measured on: with the old absolute stop rule, a correlated
  or differently-scaled problem's backfit could stop at a different sweep
  depending on `Y`'s scale, so `prenormalize` was not provably a no-op there.
  The relative rule removes the scale dependence that made it uncertain:
  re-measured on a correlated additive model at `N = 1500`, every combination
  of `m` in {2, 4, 8} and output scale in {1e-3, 1, 1e3} agrees to within
  2e-4, where before the same sweep spread by up to 0.09.

- **HDMR's `S` did not equal `Sa + Sb`, so `ST` and `shapley()` summed
  different quantities.** `S` was `Cov(f_u, Y) / Var(Y)`, referenced against
  the raw output; `Sa + Sb` is the ANCOVA split referenced against the fitted
  expansion. Li et al. (2010) define `S_u = Cov(f_u, f_hat) / Var(f_hat)`,
  which makes the identity exact by construction. `S` now reads that way.
  `S` and `ST` move: about 6e-3 per term on independent Ishigami, up to 4e-2
  on a correlated one. This is a deliberate departure from SALib's `ancova`,
  which references `Y`; the `S` docstring says so.

- **PCE's zero-variance guard tested `var == 0`, which a constant float32
  output almost never satisfies exactly.** For `Y = full(N, 0.1)`, the mean
  rounds and the measured variance is around 1e-16, not bit-exact zero, so
  `sobol` and `hdmr` returned all-`NaN` silently and `pce.analyze` returned a
  plausible-looking finite `S1` with no warning. The guard now tests
  constancy directly (`flat.max(0) == flat.min(0)`), which is exact and
  traceable. A constant-output slice now reads `NaN` with a warning
  everywhere, including `pce`.

- **PCE accepted an underdetermined fit silently.** `_auto_order` never
  dropped below order 1 (`D + 1` terms), and `fit_ratio > 1` passed
  validation, so `N <= D` rows produced a full-rank-looking answer with
  `explained_variance = 1.0` and no warning. `fit_ratio` is now bounded to
  `<= 1`, and both `analyze` and `indices` raise `ValueError` when the number
  of order-1 terms exceeds `int(fit_ratio * N)`.

- **PCE's truncated-Gaussian gate kept a Hermite basis past the width it was
  calibrated for.** The gate compared a truncation to 5 standard deviations,
  but the orthonormality error of the Hermite basis at that width is already
  1.5e-3 to 2.0e-1 depending on the order — nothing a caller could see, since
  `explained_variance` and `loo_rmse` do not detect a basis mismatch. The gate
  is now 7 standard deviations, where the same error is 4e-8 to 1.6e-4. A
  truncated Gaussian between 5 and 7 sigma now routes to Legendre after CDF
  mapping instead of Hermite; `S1`/`ST`/`S2` move for such a problem only.

- **PCE's single-pass fit built an `(n_terms, N)` intermediate it did not
  need.** It solved `solve(gram, Phi.T)` and then multiplied by `Y`, holding
  a second sample-sized array alive across the leave-one-out step. It now
  solves `solve(gram, Phi.T @ Y)` in one step, which is the same coefficients
  in exact arithmetic and one array fewer in memory. The per-row memory
  estimate that decides when the streaming fit engages drops from three
  arrays to two, matching the streamed path. In float32 the arithmetic
  reassociates, so every PCE `coefficients`, `S1`, `ST`, `S2`, `loo_rmse` and
  `explained_variance` value moves in its last bits — measured on the stored
  baseline at up to 6e-6 absolute on a coefficient and 9e-7 on an index.
  `shapley(backend="pce")` carries the same shift through.

- **`HDMRResult.shapley()` was silent on a correlated problem with
  `include_correlative=False`**, while `jaxgsa.shapley.analyze` warned about
  the same thing. That path allocates the structural ANCOVA share `Sa` only,
  renormalized to sum to 1, and drops the correlative share. The warning now
  lives on `HDMRResult.shapley()`, so both routes raise it, and each raises it
  once.

- **Optimal transport's Sinkhorn `epsilon` was relative to each class's own
  maximum cost**, which is outlier-driven and not comparable across
  parameters, leaving up to a +45% bias in the multivariate index against an
  exact-EMD oracle. `epsilon` is now relative to `V`, the index's own
  `2*tr(Cov)` normaliser. Convergence needs more iterations at the same
  `epsilon`, so the `max_iter` default rises from 1000 to 2000 and the
  `epsilon` default rises from 0.01 to 0.03. Against POT's exact `emd2` on
  Ishigami (N=1000, 10 classes) the pair reads 7.3 per cent high, against 45
  per cent before; 0.01 would read 2.4 per cent high but leaves a third of
  the solves at the iteration cap on an ordinary run, and four times slower.
  Because one `V` scales every cost, the residual bias is close to uniform
  across parameters, so it shifts the indices together and leaves their
  ranking and the `above_dummy` comparison alone. Every
  `multivariate`/`trajectory`-mode `ot`, `advective`, `diffusive`, `ot_dummy`
  and `above_dummy` value moves.

- **The one-dimensional transport cost used a nearest-rank lookup, which is
  only exact when the class size divides the sample count.** A class of `c`
  rows out of `N` needs a coupling that splits weight between two neighbouring
  order statistics whenever `c` does not divide `N`; the old code took the
  midpoint rank instead. The kernel now builds the exact two-gather,
  split-weight coupling. Checked against POT's `wasserstein_1d` at a
  non-dividing size (`N = 1013`): the gap falls from 1.7e-3 to 3.5e-3 relative
  down to 1.4e-7. Every `univariate`-mode `ot`, `advective` and `diffusive`
  value moves by up to about 3.5e-3 relative when the class size does not
  divide `N`, and is unchanged to float32 precision when it does.

- **A large output offset ate the precision of the transport cost in float32.**
  The sort, gather and squared-difference arithmetic ran on the raw output, so
  an output far from zero spent its mantissa on the offset. Each output column
  is now centred before the cost is built. Transport is invariant to a shift of
  both clouds, so this changes no estimand. Measured on an output offset of
  1e5, the spurious drift in `advective` falls from 1.2e-3 to 6.5e-6.

- **`above_dummy` compared a categorical parameter against a continuous
  dummy's floor.** The finite-sample permutation floor scales with the
  inverse of the class size, so a 3-level categorical column was measured
  against a 25-class continuous floor. On the reviewed case (`N = 4000`,
  pure-noise output) that floor read 0.010 to 0.012, while the 3-level
  column's own noise level was 0.00074, about fifteen times lower. A real
  0.25-sigma level shift on that column raised its index to 0.0081, eight
  times its own floor, and `above_dummy` still read 0 for every parameter.
  Each categorical parameter now gets its own permutation dummy built
  from that column's own level counts. `ot_dummy` and `above_dummy` move for
  any analysis with `dummy=True` on a problem with at least one categorical
  parameter; `ot_dummy`'s shape also changes, from one value per output slice
  to one value per parameter per output slice (`FieldSpec` `"slice"` ->
  `"param"`).

- **`Problem` and the marginal spec dataclasses accepted non-finite bounds.**
  `Problem(("x",), ((0, inf),))`, `GaussianSpec(nan, 1)` and similar all
  passed validation, and the resulting non-finite sample surfaced later as an
  unrelated-looking error deep in whichever method ran on it. The three
  `__post_init__` methods now check `math.isfinite` directly, so the error
  points at the parameter that is actually wrong.

- **`shapley(include_correlative=True)` is renamed on the result and in the
  docs to what it computes: an ANCOVA variance allocation, not the
  conditional-variance Shapley effect.** Exact linear-Gaussian Shapley (Owen
  2014; Owen & Prieur 2017) and the ANCOVA allocation disagree — an exact
  [0.339, 0.661] against a reported [0.288, 0.712] on a 2-parameter,
  `rho=0.5` case, re-measured after the HDMR `S` fix below. `ShapleyResult`,
  `HDMRResult.shapley()` and
  `docs/api/shapley.md` now say so. Numbers do not change.
  `analyze(backend="hdmr")` on a correlated problem with
  `include_correlative=False` now warns that the reported shares are
  structural-only, renormalised by `sum(Sa)`.

- **Every result class is now frozen.** `SobolResult`, `MorrisResult`,
  `EFASTResult`, `DGSMResult`, `HDMRResult`, `PCEResult`, `DeltaResult`,
  `PAWNResult`, `HSICResult`, `KucherenkoResult`, `VKOGAResult`,
  `ShapleyResult` and `OTResult` all take `@dataclass(frozen=True)`. Before
  1.0, only `KucherenkoResult` did: `result.S1 = None` silently succeeded on
  every other result, and `result.space = "physical"` followed by
  `to_physical_units()` raised "already in physical units" instead of naming
  the field that was hand-edited. 1.0 freezes the public interface, so this
  is the last chance to close the gap without a breaking change later.

- **`optimal_transport`'s partition grouping always returns `(R, Dg, Mg)`
  counts.** `_replicate_slice`, which rebuilt a replicate axis by hand at the
  one call site that needed it, is gone; the partition builder now
  guarantees the shape every caller wants. The `dm // M` clamp used to read
  `jnp.minimum(dm // M, G - 1)`, a defensive clamp against a group count `G`
  that could only accidentally be smaller than `Dg`; `G` is always `Dg` now,
  so the clamp is gone too. No number moves.

- **`PCEResult.shapley()` returned plausible-looking Shapley effects on a
  constant output slice whose `S1` and `ST` were already NaN.** A constant
  slice's fitted non-constant coefficients are rarely bit-exact zero in
  float32, so `explained_variance` (guarded by `var == 0`) stayed finite and
  the Shapley engine normalized those noise coefficients to sum to 1. On
  `Y = full(N, 0.1)` the result was a confident-looking `[0.56, 0.15, 0.29]`
  on three inputs that had no effect on anything. `explained_variance` now
  uses the same `max == min` constant check that guards `S1`/`ST`, so a
  constant slice yields NaN everywhere in the PCE result and in
  `shapley.analyze`'s PCE backend. Numbers do not change on any slice with
  real variance.

- **EFAST returned finite garbage, not NaN, for a constant output curve.**
  The kernel's `V == 0` guard almost never fires for a constant non-zero
  curve: the FFT's non-DC bins carry rounding noise, so `V` is tiny but not
  zero and `D1/V`, `Dt/V` are arbitrary-but-deterministic numbers (measured
  `S1 = 0.013, ST = 0.72` for `Y = 3.0`, varying with the constant's value),
  contradicting the per-curve warning that promised NaN. The guard now tests
  `max == min` on the curve, so a constant curve yields NaN exactly as
  warned.

- **HSIC reported a spurious `R2_HSIC` for a constant input column.**
  A degenerate column's kernel is all-ones, so `hsic_xx` is zero to rounding
  and `R2` became cancellation noise over the `eps` denominator clamp —
  measured `1.31`, outside the documented `[0, 1]`. A constant input has
  provably zero dependence, so `R2_HSIC` is now exactly 0 for such columns
  (matching PAWN), and `analyze` warns once naming the parameter.

- **Borgonovo reported a spurious positive `delta` for a constant input
  column.** The rank-based classes of a degenerate column are arbitrary
  groups of the sample, and the finite-sample null read `delta = 0.043`
  where the true value is 0. `delta` and `S1` are now exactly 0 for constant
  columns (the true values, matching PAWN), with a warning naming the
  parameter.

- **HDMR turned a near-constant output slice into finite garbage indices.**
  The `V_Y == 0` guard missed a slice whose variance genuinely underflows
  (e.g. `Y ~ 1e-16 * (1 + x)` in float64), and the `1e-30` denominator floor
  then manufactured finite ratios in place of NaN. A slice now counts as
  zero-variance when it is `max == min` constant or when its variance is
  below machine epsilon times its own scale (`max |Y|` squared), so such
  slices report NaN like Sobol does. The extremes are read from the real
  rows only, so padded rows stay unobservable (a padded slot may carry any
  value; the previous scale check could in principle see one).

- **Entry points now reject empty data with a clear error.** `X` or `Y`
  with zero rows used to pass validation and die inside `reshape(n, -1)`
  with an opaque `ZeroDivisionError` several frames away. `_validate_x` and
  `_validate_output` raise `ValueError` naming the array instead.

- **`resolve_batch_size` accepted non-integer widths.** `batch_size = 2.5`
  passed the `>= 1` check and failed later inside `range(...)`. It now
  raises `TypeError` at the argument.

- **`correlate()` dropped the declared correlation without a word when the
  sample was too small for the score covariance to be invertible** (`n <= D`
  or `n < 3`). It now warns that the achieved pairing does not follow the
  declared matrix at that sample size.

- **`theta` field values of `None` silently un-truncated a Gaussian.**
  `theta = {"x": {"low": None}}` passed validation and the field read then
  returned `None`, turning a declared truncated input into an unbounded one
  the user never wrote. `_validate_theta` now raises `ValueError` naming the
  parameter and field.

- **A dict input spec with an unknown key was silently accepted.** A typo
  such as `{"dist": "gaussian", "sdt": 1}` (for `std`) did nothing and the
  spec parsed anyway. Unknown keys now raise `ValueError` listing them.

- **Sobol analysis on a hand-built design whose rows do not divide the
  Saltelli step silently truncated the design.** `indices()` accepts a
  design constructed by hand, and `n_rows % step != 0` discarded the
  trailing rows and analyzed a misaligned A/B/AB/BA split. It now raises
  `ValueError`. Designs from `sample()` are unaffected.

- **Sobol `sample()` could return a degenerate one-point design without a
  word** when `n_samples` was tiny relative to the step. It now warns when
  the resolved base count is below 16 that the design is degenerate and the
  indices will be unreliable.

- **`morris.indices()` did not validate the rank of `Y`.** A 0-D or 4-D
  array died inside the estimator with a cryptic broadcast error instead of
  the clear shape message `analyze` gives. `indices()` now raises the same
  message.

### Documentation

- **`kucherenko` indices are not comparable to `sobol` indices, and the
  module docstring and API page lead with it.** Under a declared correlation
  `kucherenko.S1` is correlation-inclusive and `kucherenko.ST` is
  correlation-exclusive, so `ST >= S1` does not hold and neither number
  belongs in a table beside a `sobol` index. They are a different estimand,
  not the same estimand reached by a different route.
- **The module name is explained where it is used.** Kucherenko calls these
  Sobol' sensitivity indices generalised to dependent inputs; the 2012 paper
  is titled for the general problem, not for its author. The module follows
  the UQLab and UQpy naming instead, because a name reading as `sobol` would
  invite exactly the comparison above.
- **"Four indices under dependence" in the Methods guide.** One table putting
  `kucherenko`, `vkoga`, HDMR's ANCOVA split, and the ANCOVA Shapley
  allocation side by side: what each estimates, what each needs, and when to
  ask for it. They measure different things and disagree on the same data, so
  the section also states which two are conditional-variance indices and which
  two are not.
