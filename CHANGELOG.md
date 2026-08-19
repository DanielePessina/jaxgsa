# Changelog

## Unreleased (0.10.0)

Version 0.10 adds capability.

**Breaking changes:** `SobolResult.nan_counts` is removed, and
`Problem.input_specs` now returns dataclasses. See "Breaking".

### Breaking

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
    budget".** This was already true for most methods. DGSM read `None` as
    one batch of every row; it now derives the batch width from
    `jaxgsa.config.get_memory_budget()` with a real bytes model (a few
    Jacobian-sized transients per row, `T*K*D` floats each). This holds on
    both DGSM paths: the autodiff path and a precomputed `dfdx`. At ordinary
    sizes the derived width is one full block, so nothing changes. On a run
    large enough that the budget now splits the sample, only the float32
    summation order moves — the same statement PCE and HDMR make about
    their streamed paths. `dgsm.indices` also now raises `ValueError` on
    `batch_size=0` or a negative value; it used to read them as "one batch"
    (`dgsm.analyze` already rejected them).
  - **An explicit chunk value always wins.** The budget only sizes the `None`
    default. Sobol's bootstrap path and Morris silently capped an explicit
    `slice_chunk_size` / `resample_chunk_size` at the budget-derived width.
    They now honour the caller's value, capped only at the axis length.
    Migration: none needed, unless you relied on the budget to shrink a
    too-large explicit value — pass the width you actually want.
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
  | `CIInfo.n_resamples` | `CIInfo.n_bootstrap` | every result with intervals |
  | `samples` | `sampling_result` | efast |

  `key` replaces `seed` because a key can be split and an integer cannot.
  Reseeding from an integer silently correlates repeated or nested draws.
  Pass `jax.random.key(0)` where you passed `seed=0`.

  `sample()` keeps `seed`. Design generation is host-side `scipy.stats.qmc`
  and has no JAX PRNG interface, so the split is real and now documented
  rather than accidental.

- **`borgonovo.analyze` no longer bootstraps by default.** `n_bootstrap` was
  `100`; it is now `0`, matching every other method. Combined with the new
  key requirement, the old default would have made the plainest possible call
  an error.

  `bias_correct` becomes tri-state to match. `None` (the default) applies the
  Plischke correction whenever there are replicates and does nothing
  otherwise; `True` asks explicitly and warns when `n_bootstrap` is `0`;
  `False` never applies it. For a corrected delta, pass `n_bootstrap=100` and
  a key. The uncorrected estimate is biased upward, because a KDE separation
  is a distance and sampling noise can only increase it.

- **`ci_method` reaches every method that bootstraps.** pawn, borgonovo and
  optimal_transport were hard-wired to percentile endpoints while recording
  `"quantile"`. All three now accept `"quantile"` or `"gaussian"`.

- **`keep_replicates` is keyword-only and last** in every signature. It sat in
  three different positions.

- **`prenormalize` is gone.** It meant four different things across the
  methods that took it, and on two of them it meant nothing at all.

  | Method | Was | Now |
  |---|---|---|
  | `sobol` | `prenormalize: bool = False` | removed; the standardization always runs |
  | `efast` | `prenormalize: bool = False` | removed; it was a measured no-op (6e-16) |
  | `hdmr` | `prenormalize: bool = False` | removed; it was a measured no-op (1e-6) |
  | `morris` | `prenormalize: bool = False` | renamed `standardize_outputs` |
  | `dgsm` | — | new `standardize_outputs: bool = False` |

  `hsic` also loses the keyword, with no replacement. Its `bandwidth` is now
  a multiplier on the median heuristic. The heuristic carries the scale of
  `Y`, so the indices are invariant under `Y -> a*Y + b`. Standardizing `Y`
  first therefore changes nothing, and a no-op keyword would mislead.

  `optimal_transport` keeps this behavior under the name `standardize`
  (default `True`). It does real work there: the method builds distances
  from `Y` itself, not from a ratio.

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

### Performance

- **The PCE streamed fit dispatches one jitted step per row batch.** Each
  batch used to run several eager ops: an unjitted design-matrix build plus
  accumulation matmuls. The per-batch step is now one jitted call, and the
  ragged trailing batch is zero-padded and masked, so each step compiles
  once. Padded rows contribute exact zeros to the normal equations and to
  the leave-one-out sum. This is the pattern the HDMR fit already uses,
  where the same fix was measured at 1.2-1.67x.

- **PCE second-order extraction no longer builds an unbudgeted transient.**
  `sobol_from_coefficients` materialized an `(n_terms, D*(D-1)/2)` float
  array for the S2 pair mask — about 3.5 GB at `D=100`, order 3. The pair
  axis now streams in chunks sized by the memory budget. At small `D` the
  budget allows one chunk, which runs the exact old matmul, so shipped
  values are bit-for-bit unchanged.

### Fixed

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
  index moves with row order — measured at 6e-4 relative against 2.5e-12 in
  float64. VKOGA was previously the only method in the library that checked
  the x64 flag.

- **eFAST derives its slice chunk from the memory budget.** It used a fixed
  2048 that ignored both dtype and `N`, so one chunk was about 4 GiB at
  `n_per_curve = 65536` in float64.

### Added

- **Eleven methods gain a pure `indices()` core.** `sobol.indices` already
  existed; `efast`, `pawn`, `morris`, `hsic`, `borgonovo`,
  `optimal_transport`, `dgsm`, `pce`, `hdmr` and `shapley` now have one too.

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

### Changed

- **eFAST and HSIC report no bootstrap interval, and the docs now say why.**
  eFAST has one search curve per parameter, so there is nothing to resample —
  removing a point does not shrink the sample, it changes what the estimator
  computes. HSIC already reports permutation `p_values`, which is the
  uncertainty statement for a V-statistic; a row bootstrap would repeat rows
  onto the kernel diagonal, where the kernel is exactly 1, biasing the
  resampled index upward by construction.

- **`CONTEXT.md`** states the vocabulary the interface is frozen against, and
  `tests/test_vocabulary.py` reads it back off the method registry. A
  signature that drifts from the specification now fails a test rather than
  shipping. Two rules the code does not satisfy yet are recorded as strict
  xfails, so closing the gap forces the exemption to be deleted.

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

  `jaxgsa.dgsm.poincare_constant()` and `jaxgsa.dgsm.marginal_variance()` take
  the dataclass too. Before, they took the private tuple, so you could not call
  them from outside the package.

  The input side does not change. `Problem.from_dict` still accepts a
  `(low, high)` tuple and the `UniformInputSpec` / `GaussianInputSpec` /
  `CategoricalInputSpec` dicts, exactly as before. It also accepts the new
  dataclasses. A saved `.npz` file keeps its dict form, so files written by
  earlier versions still load.

- **`SobolResult.nan_counts` is removed.** It counted the `NaN` entries in the
  computed indices and threw away which model run produced them, which is the
  one thing you need in order to act. `result.invalid` replaces it and keeps
  the positions. See "Failed model runs" below.

- **A non-finite model output now raises by default.** Before, what happened
  depended on the method you called: `sobol`, `morris` and `kucherenko` dropped
  the affected data and warned, `efast` warned and computed anyway, and the
  other nine let the value reach the indices with no warning at all.

  To keep the old behaviour of those first three, pass `on_invalid="drop"`. To
  keep `efast`'s, and that of the nine silent ones, pass
  `on_invalid="propagate"`.

### Added

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

### Performance

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
  Ishigami problem with 128 output slices, `analyze(..., num_resamples=...)`
  went from 84 ms to 4-9 ms. The resampler now runs one estimator kernel over
  one output slice, mapped over the resample draws and then over a chunk of
  slices; before, a Python loop dispatched twice per slice. Confidence
  endpoints are one call on the whole grid instead of one per slice.

  `slice_chunk_size` now caps output slices on both the plain and the
  bootstrap path. On the bootstrap path it used to cap resample draws.

  This moves 44 sobol values in the last 1 to 4 bits of float32. Widening the
  batch is the speedup, and XLA schedules a float32 reduction differently at
  a different width. `scripts/baseline/README.md` records the review.

### Fixed

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
  analysis.** `analyze(sr, Y, num_resamples=20).S2` and
  `analyze(sr, Y, num_resamples=0).S2` could differ in the last bits for the
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
  It is now named as a bad row up front. The out-of-range check still guards
  genuine estimator failure.

- **VKOGA reported a failed run as a solver failure.** A non-finite output made
  every cross-validation score non-finite, which surfaced as
  `"Every cross-validation score is non-finite; the kernel solves failed"`.
  That guard remains for real solver failure, but is no longer reachable from
  non-finite input.

## Unreleased (0.9.0)

Version 0.9.0 fixes defects. It adds no method and removes no API.

**One breaking change:** `jaxgsa.config.set_memory_budget` now reads its value
in megabytes, not bytes. See "Breaking" below before you upgrade.

Three further calls that were accepted before now raise. Each one was accepted
while doing something the caller did not ask for, so refusing it is the fix,
not a side effect. They are marked **now raises** below.

### Breaking

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

- **`result.ci` records how a confidence interval was made.** Until now a
  `*_conf` array was an interval of unknown level: nothing on the result said
  whether it was a 95% or a 68% interval, which endpoint rule drew it, or how
  many resamples it rested on. The five results that carry intervals
  (`SobolResult`, `MorrisResult`, `DeltaResult`, `PAWNResult`, `OTResult`) now
  carry a `ci` field holding `level`, `method`, `n_bootstrap` and, on request,
  `replicates`. `S1` and `S1_conf` are unchanged plain arrays.
- **`keep_replicates=True` keeps the bootstrap draws.** `analyze` discarded
  them, so recomputing an interval at another level meant re-running the whole
  analysis. Pass `keep_replicates=True` and the draws arrive on
  `result.ci.replicates`, keyed by the estimate they belong to. It is off by
  default because the draws are large: 1000 resamples of a
  `(T=100, K=5, D=20)` index array is 80 MB.

  All five methods that report intervals accept the keyword: `jaxgsa.sobol`,
  `jaxgsa.morris`, `jaxgsa.pawn`, `jaxgsa.borgonovo` and
  `jaxgsa.optimal_transport`.
- **`jaxgsa.sobol.indices()` computes the indices with nothing around them.**
  `analyze` reads output values on the host to apply its `on_invalid` policy,
  and a policy decision needs a concrete number, so it cannot run inside
  `jax.jit` or `jax.vmap`. `indices` reads nothing, so it can. Both call one
  estimator, so the numbers are the same.
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
  directly by about twenty modules but reached users only through JAX and
  SciPy, both of which allow NumPy 1.x. An install from PyPI could therefore
  get a NumPy the package does not support.
- **The SciPy floor rises to `scipy>=1.15`**, from `>=1.10`. This makes
  `scipy.stats.chatterjeexi` always available as a verification oracle.
- **PAWN's `slice_chunk_size` default is now derived from the memory budget**,
  as `None`, rather than a fixed 2048. The fixed value never engaged: it is
  compared against `T*K`, which is smaller than 2048 in every realistic case,
  so the knob did nothing unless you found and lowered it yourself. The default
  now adapts to `N`, `D` and `n_bins` through
  `jaxgsa.config.set_memory_budget`. Only peak memory changes; every index is
  identical.

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

### Performance

- **HSIC allocates about a third of what it did** when it picks its own kernel
  bandwidth. The median heuristic built a distance matrix, two index arrays and
  a copy of the upper triangle, only to skip the diagonal zeros. Those zeros are
  the smallest entries, so their effect folds into the quantile position
  instead. Measured on an Apple M1 Pro in float32: peak transient memory falls
  from `3.57 * N^2` to `1.00 * N^2`, which is 228 MiB down to 64 MiB at
  `N = 4096`. Every index is unchanged.
- **Row deduplication is about 3.4 times faster.** `_stable_unique_rows` runs
  twice on the Sobol sampling path and once on the Morris path. It built one
  array view and one `bytes` key per row in Python. It now builds every key in
  one C-level call and keeps the dictionary that was already doing the work.
  Measured on an Apple M1 Pro at `N = 2**20`, `D = 20`, float64: 1099 ms
  before, 325 ms after. Output is bit-identical, including row order, which
  matters because the design prefix logic depends on it.

### Fixed

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
  and `slice_chunk_size=0` **now raises** like it does elsewhere. Results are
  unchanged.
- **eFAST computed its frequency plan twice.** The sampler built the design
  from one copy of the formula and the analyzer recomputed it from another. The
  two agreed only because one frequency band happened to contain the other, and
  no test could catch a divergence, because the tests recomputed the formula
  too. Both sides now read one plan, which checks its own invariant, and a new
  test recovers the bands from a synthetic signal instead of from the formula.
- **Borgonovo: a failed delta estimate now says how to fix it.** The estimate
  is a half L1 distance, so a value outside `[0, 1]` is a failed computation.
  The error now reports whether a conditioning class was floored, which output
  column failed, the bandwidth actually used against the real grid step, and
  the value that would resolve it.
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

### Removed

- `_PCEFit.coeffs_flat`, a private field that was written and never read. It
  kept a large array alive inside the module whose purpose is bounding memory.
- The private `validate_correlation` helper. It had no production caller and
  stayed alive only because tests used it. `canonicalize_correlation` runs the
  same code, and its tests now call that instead.
- The `version` and `date-released` fields in `CITATION.cff`. Nothing updated
  them and nothing checked them, so they had drifted three releases behind.
  Both are optional, and GitHub and Zenodo take the version from the release
  tag when they are absent.

## 0.8.0

### Added

- **`jaxgsa.vkoga` — variance-based sensitivity analysis for correlated
  inputs.** The two-stage surrogate-based sensitivity analysis (SSA) of Hilhorst,
  Quicken, van de Vosse & Huberts (2024, *Int. J. Numer. Meth. Biomed. Engng.*
  40(2):e3797): fit a VKOGA kernel surrogate to given `(X, Y)` data, then compute
  the correlated variance-based indices of Li et al. (2010, *J. Phys. Chem. A*
  114:6022-6032) against it under a Gaussian copula. Splitting the work this way
  is what makes the method affordable. The indices need nested conditional
  sampling. That is hopeless against an expensive model, but trivial against a
  kernel expansion.

  `jaxgsa.vkoga.analyze(problem, X, Y, correlation=...)` returns a
  `VKOGAResult` with five indices per parameter, shaped by the usual output
  contract (`(D,)`, `(K, D)`, `(T, K, D)`):
  - `S_TC`, total correlated, `V(E(Y|X_i))/V(Y)` — what `X_i` explains
    through itself and through its correlation with the others. The measure for
    input prioritisation.
  - `S_TU`, total uncorrelated, `E(V(Y|X_-i))/V(Y)` — what only `X_i` can
    explain. The measure for input fixing.
  - `S_U` (the independent contribution alone), `S_C = S_TC - S_U` (the
    correlation-borne part, which can be negative when a correlation opposes
    a direct effect), and `S_IU = S_TU - S_U` (independent interactions).

  Under independent inputs the five collapse to the familiar picture: `S_TC` is
  the first-order Sobol' index `S1`, `S_TU` is the total index `ST`, and `S_C` is
  zero.

  The dependency structure comes from the problem. `analyze` reads
  `problem.correlation` by default. An uncorrelated problem gives independent
  inputs. A `(D, D)` matrix passed as `correlation=` overrides the declaration
  for one call. To fit a matrix from observed data, use
  `jaxgsa.sampling.fit_correlation(problem, X_data)` and attach it with
  `problem.with_correlation(...)`. That is one workflow, and it makes explicit
  which sample the copula comes from. The matrix actually used is returned on
  `result.correlation`. Categorical problems raise: the isotropic RBF needs a
  continuous CDF map per coordinate. All conditioning happens in the latent
  standard-normal space, where the Gaussian conditionals are closed-form. Every
  expectation is a scrambled-Sobol' quasi-Monte-Carlo average against the
  surrogate.

  Stage one is a pure-JAX VKOGA (Wirtz & Haasdonk 2013): Gaussian RBF, centres
  chosen by the P-greedy rule in a nested Newton basis, coefficients from
  RKHS-regularised normal equations, with `gamma` and `ridge` selected by k-fold
  cross validation over a 10x10 grid. Held-out rows are masked out of the greedy
  stopping rule, so cross-validation fits stop exactly like the final fit and
  score the hyperparameters under the same stopping behaviour. Explicit `gamma`
  and `ridge` values must be finite and positive; anything else raises before
  any fitting. Centre selection depends only on `X`, so all output slices share
  one basis and one set of centres. That is the "vectorial" part, and it makes a
  multi-output fit cost barely more than a scalar one. The result keeps the
  surrogate. `result.predict(X_new, batch_size=None)` evaluates it with the
  usual bounded-memory batching. `result.to_dataset()` exports the indices, the
  fit diagnostics (`n_centers`, `gamma`, `ridge`, `rmse`), the copula matrix,
  and the output `variance` under the correlated measure.

  Two things to get right, both documented prominently:
  - **Train on an independent, space-filling design even when the analysis is
    correlated.** The correlated measure concentrates on a ridge. But `S_TU`
    conditions on `X_-i` and then resamples `X_i` across its whole marginal. A
    surrogate trained only on correlated data extrapolates for exactly those
    draws.
  - **Enable float64.** The coefficient step forms `A^T A`, which squares the
    condition number of the cross kernel. float32 cannot carry that for small
    `gamma`. Call `jax.config.update("jax_enable_x64", True)` before fitting;
    `analyze` warns when x64 is off.

  `VKOGAResult.shapley()` raises `NotImplementedError` by design. Shapley effects
  need a variance decomposition indexed by parameter subsets. A kernel expansion
  is a sum over centres, and every centre involves every parameter, so there
  is no membership matrix to allocate from. Use `jaxgsa.hdmr` (which also offers
  `shapley(include_correlative=True)`) or `jaxgsa.pce`.

- **`jaxgsa.kucherenko` — design-based Sobol' indices for dependent inputs.**
  The single-loop estimators of Kucherenko, Tarantola & Annoni (2012, *Comput.
  Phys. Commun.* 183:937-946). `kucherenko.sample(problem, n)` builds a
  conditional-copula design of `n * (2D + 1)` rows: one joint block, one
  conditional block per parameter for `S1`, one per parameter for `ST`. You
  evaluate your actual model on the rows — no surrogate is fitted. This is
  the design-based counterpart to `jaxgsa.vkoga`: the same two quantities,
  `S1 = V(E(Y|X_i))/V(Y)` (correlation-inclusive, VKOGA's `S_TC`) and
  `ST = E(V(Y|X_~i))/V(Y)` (correlation-exclusive, VKOGA's `S_TU`).

  The sampler reads `problem.correlation`. Both conditionals are closed-form
  Gaussians in the latent copula space. It is deliberately exempt from the
  0.6.0 correlated-design error: conditioning on the declared copula is the
  method's purpose. With no declared correlation the design reduces exactly to
  the Saltelli column-swap scheme and the indices to the classic Sobol' `S1`
  and `ST`. Categorical problems raise (the conditional copula needs
  continuous marginals). `KucherenkoSamples` uses the standard one-file NPZ
  `save`/`load`; `KucherenkoResult` has `to_dataset` and follows the usual
  output contract.

  Two numerical safeguards are built in. The `S1` estimator subtracts one
  shared shift (the joint-block mean) from both product factors. The
  estimator is algebraically unchanged, but a large output mean no longer
  destroys the covariance term in rounding (a `+1e8` offset moves `S1` by
  less than `3e-11`; the uncentered form drifted by up to `0.11`). And with
  `scramble=False` the Sobol' sequence skips its all-zeros origin point,
  which the probit would map to a clipped extreme deviate.

  Validation: the linear-Gaussian closed form (max error `7.5e-4` at
  `base_n = 4096`), independent Ishigami against the analytic values and
  `jaxgsa.sobol`, and a dedicated cross-route module
  (`tests/test_correlated_agreement.py`). That module pins analytic, `vkoga`,
  and `kucherenko` to the same reference, and the two routes to each other.

- **Optimal transport is certified valid under correlated inputs.** It was
  already exempt from the correlated-input error; two dedicated tests now
  certify the reading. With `corr(X1, X2) = 0.8` and `Y = X1` only, the unused
  input `X2` gets a clearly non-zero index (0.40 vs 0.94) — the documented
  correlation-inclusive interpretation. And the indices on a correlated
  problem are bit-equal to the indices on the same `(X, Y)` with the
  correlation stripped. The estimator never reads the matrix, which proves
  distribution-freeness in `X`. `methods.md` states the guarantee explicitly.

### Changed

- **Every correlated-problem refusal now names `jaxgsa.vkoga` and
  `jaxgsa.kucherenko` first.** `sobol.sample`, `morris.sample`, `efast.sample`,
  `pce.analyze` and `dgsm.analyze` raise the shared message, and it listed only
  the moment-independent and rank-based routes. It now leads with the two
  variance-based routes this release adds, because they answer the question the
  refused method was asked. The list also names `jaxgsa.shapley` with
  `backend="hdmr"`, which accepts correlated problems. The `backend="pce"`
  refusal in `shapley.analyze` gained the same two names.
- **`S_U` is clipped to `S_TU`, so `S_IU` can no longer go negative.** `S_U`
  measures the output against fitted additive component functions. No additive
  function of `X_i` can represent an interaction. So on a model with
  interactions under a correlated measure the raw `S_U` could exceed `S_TU` and
  drive `S_IU` below zero, a silently negative "interaction" index. The clip
  restores the invariant. A clip wider than 1% of the output variance now
  raises a `UserWarning`. That warning says the additive projection is
  inadequate for that model, so `S_U`, `S_C` and `S_IU` are indicative while
  `S_TC` and `S_TU` stay reliable. `S_C` is not clipped; a negative `S_C` is a
  real reading.
  Additive models are unaffected.
- **`jaxgsa.vkoga` reports its out-of-sample error and warns when the surrogate
  fails.** The pooled cross-validated RMSE was computed and then discarded; only
  the optimistic training RMSE reached the result. It is now on
  `VKOGAResult.cv_rmse` (and in `to_dataset().attrs`), and `analyze` warns when
  it passes half the output standard deviation. Every index is measured against
  the surrogate, so a failed fit gives meaningless indices. On
  `sin(2*pi*12*u1) + 0.5*u2` the reported ranking is inverted, and nothing said
  so. `cv_rmse` is `None` when the caller fixes both `gamma` and `ridge`,
  because no cross-validation runs.
- **`S_U`'s attribution is corrected.** The code cited Li et al. (2010,
  Equation 25), whose structural index is `Var(f_i)/V(Y)`. The implemented
  quantity is `E[Var(f_i|X_-i)]/V(Y)`, the decorrelated first-order index of
  Mara & Tarantola (2012). The docstrings now cite the right result. `S_TC`'s
  docstring also states its formula, so the word "total" cannot be read as
  total-order: `S_TC` is `V(E(Y|X_i))/V(Y)`, a first-order quantity, and
  "total" names the pathways it counts.
- Docstring `Raises:` sections across `pce`, `hsic`, `pawn`, `hdmr`, `dgsm`,
  `shapley`, `sobol.to_morris`, `efast.sample` and `morris.sample` now state
  the categorical and correlated gating the code performs. The correlation tolerance of `hsic`,
  `pawn`, `hdmr` and `borgonovo` moved from code comments into docstring
  prose. The `sobol`,
  `vkoga` and `kucherenko` namespace docstrings state their gating. `docs/api`
  and `docs/examples/categorical-inputs.md` match the refusal lists the code
  actually raises, and the correlation-inclusive caveat in `methods.md` now
  covers HSIC and PAWN as well as optimal transport.
- `docs/examples/vkoga.md` documents three limits with their signals: a failed
  surrogate on oscillatory responses (read `cv_rmse`), the additive projection
  behind `S_U` (read the clip warning), and the roughly -3.5% low bias in
  `result.variance` for Gaussian marginals, which the CDF-space kernel causes by
  under-resolving the tails.
- Pre-existing passages that list the correlated-input options no longer omit
  `jaxgsa.vkoga` and `jaxgsa.kucherenko`. The "I only have existing simulation
  data" menu in `methods.md` now names VKOGA. The correlation caveat in
  `docs/examples/shapley.md` said only that dependent-input Shapley effects are
  future work; it now names the ANCOVA route and the two conditional-variance
  methods. `docs/examples/correlated-inputs.md` splits the correlation-tolerant
  methods into the total, correlation-inclusive group and the group that
  separates the direct from the correlation-borne effect, with VKOGA and
  Kucherenko in the second. `docs/api/sampling.md` records that
  `kucherenko.sample` is a design builder that reads `problem.correlation`
  instead of refusing it, and `README.md` distinguishes the Shapley-style
  allocation from the conditional-variance route.
- The documentation tempers HDMR's `ST` under dependence. It is the SCSA
  convention of Sarazin et al. (2017, Eq. 8), a sum over the terms a parameter
  appears in. It is not a total-effect index: it has no variance-reduction
  reading, it does not answer the input-fixing question, and it can go negative
  because `Sb` can. `methods.md` and `docs/examples/correlated-inputs.md` say
  so and send readers wanting a total index to `kucherenko.ST` or
  `vkoga.S_TU`. HDMR's per-term `Sa` / `Sb` split remains the thing it uniquely
  provides. No code or field names change here.
- The VKOGA and Kucherenko documentation drops rhetorical bold and italics.
  Structural emphasis stays: table headers, term labels that open a
  definition-style bullet, and admonition titles. The change covers
  `docs/examples/vkoga.md`, `docs/examples/kucherenko.md`, `docs/api/vkoga.md`,
  `docs/api/kucherenko.md`, and the lines this release adds to `README.md`,
  `docs/guide/methods.md`, `docs/api/index.md` and
  `docs/examples/correlated-inputs.md`. No index value changes.

### Internal

- The `n_inner` coupling in `jaxgsa.vkoga` is documented. The estimators drop
  the iid inner-noise correction and rely on a large shared QMC inner block, so
  `n_inner` cannot be lowered freely to buy speed.
- The correlated tests gained a D=4 case with six distinct off-diagonals of
  mixed sign, plus a parameter-permutation equivariance probe. The previous
  cases all used one non-zero off-diagonal, which cannot catch a transposed
  parameter axis. Both routes pass: Kucherenko errors are at most 7e-4 (`S1`)
  and 2e-5 (`ST`); VKOGA errors are at most 0.013 (`S_TC`) and 0.006 (`S_TU`).
- Added `jaxgsa._core.legendre`: one orthonormal Legendre recurrence, shared by
  the PCE basis and the VKOGA component-function fit. The recurrence runs in
  the backend default float dtype (float64 under x64), the same promotion the
  Hermite basis applies, so a float32 `X` does not downgrade the PCE design
  matrix and mixed uniform/Gaussian problems get one basis dtype.
- The categorical-design error is situation-aware: when the categorical
  problem also declares a correlation, it no longer recommends
  `jaxgsa.sobol.sample` (which refuses correlated problems). It states that no
  variance-based route exists and names the given-data methods. Both
  design-side gates now route the combined case to that message. Previously
  only `kucherenko.sample` could emit it. `morris.sample`, `efast.sample` and
  `sobol.sample` ran the correlated check first and raised the correlated-only
  text, which recommends methods that then refuse the problem for being
  categorical. `_raise_correlated_design` and
  `_raise_categorical_design` both detect the combination, so the order the two
  checks run in no longer matters. A test asserts the combined message for
  `morris`, `efast`, `sobol` and `kucherenko`.
- The analysis-side gates got the same treatment. `_raise_correlated_analysis`
  and `_raise_categorical_analysis` had no combined message at all, so a
  problem that is both categorical and correlated got a single-fault message
  from whichever check its caller ran first. Both now route the combination to
  the shared text, with an analysis-side wording that does not tell a caller
  who already holds `(X, Y)` to go and draw samples. This covers `dgsm`,
  `pce`, `hdmr`, `hsic`, `shapley` and `vkoga`, whether they gate through
  `_validate_xy_inputs` or call the raisers directly. A parametrised test
  covers all six, and a companion test asserts that the three methods the
  message recommends — `jaxgsa.optimal_transport`, `jaxgsa.borgonovo` and
  `jaxgsa.pawn` — do accept the combined problem.
- **`jaxgsa.pawn` accepts categorical and correlated inputs together.** It
  gained categorical support in 0.7.0 and was already correlation-tolerant,
  so it is the third method that handles the combination. Every categorical
  refusal message now names it alongside optimal transport and Borgonovo
  delta.
- The Gaussian conditional-draw algebra lives in `jaxgsa._core.copula`, next
  to the conditional plan; the VKOGA estimators and the Kucherenko design
  share it.

## 0.7.0

### Added

- **Categorical inputs — first-class unordered discrete marginals.**
  `Problem.from_dict` accepts
  `{"dist": "categorical", "probs": [p0, ..., pL-1], "labels": [...]}`
  (new `CategoricalInputSpec`). A categorical parameter has `L >= 2`
  levels. The probabilities must be positive and sum to 1. A small
  rounding error is renormalized; a clearly wrong sum raises. Samples
  carry the integer level codes `0 .. L-1` as floats — codes, never
  physical values. The optional `labels` (strings or numbers) live on the
  `Problem` for reporting only. `problem.categorical_labels` maps each
  categorical parameter to its label tuple.
  `problem.has_categorical_inputs` reports their presence. Problems with
  categorical marginals round-trip through JSON metadata and NPZ design
  files, labels included.
- **Sampling.** `jaxgsa.sampling.monte_carlo` draws categorical columns
  through a step-function inverse CDF on the unit interval
  (`searchsorted` on the cumulative probabilities). Level frequencies
  follow the declared `probs` exactly in distribution.
- **Optimal transport, Borgonovo delta, and PAWN support categorical
  inputs.** A categorical column conditions on one class per level. For
  PAWN the level code is already a bin index, so the KS kernel is
  unchanged. The kernel compiles at `n_eff = max(n_bins, max_levels)`. The
  unused bins stay empty, return `NaN`, and are dropped by the
  nan-aware median/max/mean over bins. `n_bins` applies to continuous
  columns only. Continuous-only PAWN results are bit-for-bit unchanged.
  For optimal transport and Borgonovo delta, class sizes are the observed
  level counts. They vary per column and per bootstrap
  resample; the shared partition layer builds a per-replicate padded
  layout for them. Continuous columns keep their equal-frequency rank
  classes; `n_partitions` / `n_classes` apply to continuous columns only.
  The indices depend only on the level partition, so relabeling levels
  does not change them (tested). Declared levels with no observed samples
  are dropped from the class average with a `UserWarning`. All OT modes
  (`univariate`, `multivariate`, `trajectory`) and the `dummy` baseline
  work.
- **The Saltelli column-swap scheme works with categorical columns.**
  The Saltelli design only copies coordinate values between rows, so the
  estimators are unaffected. A guard caps the unique-row inflation loop:
  a low-cardinality categorical problem has finitely many distinct rows.
  The sampler stops doubling `base_n` when a doubling adds no new unique
  rows or the known distinct-row bound is reached. A candidate-row cap
  is checked before the next doubling is built, so the warning fires
  before any huge allocation. The design then keeps duplicate rows and
  explains why in a `UserWarning`. Duplicates are valid Saltelli
  samples; deduplication only saves model evaluations.
- **Clear errors from code-order-sensitive methods.** `morris.sample`,
  `efast.sample`, `SobolSamples.to_morris`, `dgsm.analyze`,
  `pce.analyze`, `hdmr.analyze`, `hsic.analyze`, and
  `shapley.analyze` refuse a categorical problem with a `ValueError`
  that names the categorical parameters and the supported alternatives.
  The guard is a `categorical_ok` capability flag in the shared
  validation layer, parallel to the 0.6.0 `correlation_ok` flag.
- **No correlation × categorical.** A `problem.correlation` entry that
  touches a categorical parameter raises at construction (a Gaussian
  copula does not define a coupling for an unordered marginal; polychoric
  coupling is future work). Identity rows and columns are fine. The
  check runs on the declared matrix, before the positive-definiteness
  repair, so repair noise never reads as a coupling; the stored matrix
  carries exact-identity categorical rows and columns.

### Fixed

- **`borgonovo.analyze` could return a delta far outside `[0, 1]` in
  silence.** Delta is a half L1 distance between densities, so it lives in
  `[0, 1]`. The degenerate-class detector only fired below `1e-6` of the
  full-sample bandwidth. A class that was almost a point mass — one
  output value plus a little jitter — passed that test but still sat
  orders of magnitude below the output grid step. Its conditional density
  aliased on the grid and the trapezoid integral exploded. On a
  three-atom model with true delta `2/3` and jitter `1e-5`, `analyze`
  returned delta 121 with no warning. Two changes close this:
  - The degenerate tolerance is now `1e-2`, the scale at which the
    default 100-point grid stops resolving a class. The same sweep now
    returns 0.611 (jitter `1e-5`), 0.624 (`1e-3`) and 0.907 (`1e-2`), all
    in range.
  - A range guard checks every returned delta. An excursion beyond
    `[-0.05, 1.05]` in the point estimate now raises `ValueError`. The
    message names the parameter, gives the observed range, states the
    cause, and names both knobs: `grid_size` (with its current value) and
    `degenerate_bandwidth`. A value outside `[0, 1]` is a failed
    computation, not an estimate, so it must not reach a plot. A
    confidence bound outside the range still raises a `UserWarning` only:
    the point estimate is the contract and the interval is a diagnostic.
    Nothing is clipped, because clipping 121 to 1.0 would turn an obvious
    failure into a plausible wrong answer. The guard covers continuous and
    categorical inputs alike.

  The bug predated categorical support — a continuous step-function model
  hit it too — but categorical inputs make "one level, one output value"
  the normal case.
- **`borgonovo.analyze` under-budgeted memory on imbalanced categorical
  columns.** The default `slice_chunk_size` assumed the padded class
  layout `M * P` was about `N`. That holds for equal-frequency continuous
  classes. A categorical column pads every level up to the largest one, so
  with `probs = [0.91] + [0.01] * 9` at `N = 20000` the layout is `9.1x N`
  and the default chose a chunk whose KDE tensor was 2.29 GiB against a
  256 MiB budget. The default now sizes from the real per-group layout
  `sum_g(Dg * Mg * Pg)`, the same way `optimal_transport.analyze` already
  did. On that case the chunk width goes 33 -> 3 and the peak lands at
  54.6M elements, inside the budget. Continuous problems are unchanged.

### Changed

- **`borgonovo.analyze` refuses a discrete output.** The delta estimator
  compares Gaussian kernel density estimates on a shared output grid. An
  atomic density is a spike no grid resolves, so on a discrete output the
  index reports the grid resolution rather than the model. `analyze` now
  checks the output before any expensive work and raises `ValueError` when
  a column takes at most 20 distinct values and those values are fewer
  than 1% of the samples. Both conditions must hold, so a continuous
  output rounded to two decimals is not refused. The message names the
  offending column and points at `jaxgsa.optimal_transport.analyze`, which
  compares empirical distributions and needs no density. Continuous
  outputs are the supported contract; no atomic estimator is planned.
  A constant column is exempt: it needs no density and its exact answer is
  `delta = S1 = 0`, which is the documented behaviour and stays. A
  rare-event 0/1 indicator has two atoms and is now refused; use
  `jaxgsa.optimal_transport.analyze` or `jaxgsa.pawn.analyze` for it.
  Categorical inputs stay supported. The limit applies to the output only.
- **Borgonovo delta floors the bandwidth of classes the output grid cannot
  resolve.** A conditioning class with zero output variance (a point mass,
  e.g. one categorical level mapping to one output value) used to get
  bandwidth 0. Its density dropped out of the L1 integrand and delta
  biased far low (0.33 instead of 2/3 on a noise-free three-level repro).
  The class now gets a grid-resolvable kernel at its value:
  `max(0.1 * full-sample Silverman bandwidth, grid step)`. The repro
  recovers delta 0.60–0.62. One `UserWarning` reports the floor. Classes
  with genuine spread are untouched, so continuous results are
  bit-identical.

  The grid-step term is the one that binds in practice, so the delta of a
  near-degenerate class is set by `grid_size`, not by the bandwidth
  fraction. That estimate is biased low, and the bias does not vanish as
  `N` grows: on the three-atom repro (true `2/3`) it reads 0.56 at
  `grid_size=50`, 0.61 at 100, and 0.61 at 200 and above. Read delta on an
  atomic conditional as a ranking signal, not a calibrated number. The
  `analyze` docstring and the categorical docs now say so. A consistent
  estimator for atomic conditionals needs a different estimator and is
  future work.
- **`borgonovo.analyze` exposes the degenerate-class settings.** The two
  module constants are now overridable: `degenerate_tol` (when a class
  counts as unresolvable, a fraction of the full-sample bandwidth,
  default `1e-2`) and `degenerate_bandwidth` (the floor, `"auto"` for
  `max(0.1 * h_full, grid_step)` or a fraction of `h_full` applied
  exactly).
- **`SobolSamples.samples` is documented as an evaluation set, not a
  sample.** Deduplication collapses repeated rows, so the empirical
  marginal of a column in `samples` does not match the declared one. With
  `probs = [0.9, 0.1]` the column reads about `[0.84, 0.16]`. Categorical
  dedup rates are high, so the distortion shows. `analyze` is correct: it
  reconstructs the expanded design through `expanded_to_unique`, and that
  design carries `[0.900, 0.100]`. The docstring and the categorical docs
  now warn against reusing `samples` as a standalone Monte Carlo design.
- **`fit_correlation` keeps categorical parameters at identity.** A
  Spearman rank correlation over unordered level codes depends on the
  arbitrary code order (relabeling flips its sign). The fit now excludes
  categorical columns, returns exact-identity rows and columns for them,
  and warns once naming them. The fit is invariant under level
  relabeling. Polychoric estimation is future work.
- **`n_partitions` / `n_classes` are always validated when passed.** An
  all-categorical problem used to accept any value silently. A passed
  value now validates against its range; a valid value that nothing uses
  draws an "ignored" `UserWarning`. `optimal_transport.analyze` defaults
  `n_partitions` to `min(25, N // 2)`, so small samples no longer raise
  over a default the user never passed; an out-of-range value that only
  the `dummy` baseline consumes names the dummy in the error.

## 0.6.0

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
  `1e-6` the repair is floating-point noise and says nothing. Between `1e-6`
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
  contains the parameter. Li et al. (2010) define it in Section 2.2.3, from
  the per-term indices of their Eqs. (19)-(22); Sarazin, Viaud & Cournède
  (2017) restate it as their Eq. (8). It is the same convention as SALib's
  HDMR. The source paper invites the misreading, reusing the symbol `S_Ti`
  for both this term-membership sum and the classical conditional-variance
  total of its own Eq. (4). With independent
  inputs the correlative shares vanish and it reduces to the ordinary Sobol'
  total-order index. With correlated inputs it does not. It can be negative,
  it is not bounded in `[0, 1]`, and it does not measure the expected
  variance reduction from fixing a parameter. So it must not be used as a
  criterion for fixing one. The bias runs toward "cannot be fixed", and a
  parameter the model ignores can outrank one with a negative value. It is
  also not comparable with the `ST` of `jaxgsa.kucherenko` or the `S_TU` of
  `jaxgsa.vkoga`. Use one of those for a conditional-variance total under
  dependence. Li et al. also attach a precondition: the totals are reliable
  only when the per-term `S` values sum to about 1 (their Eq. 24), the
  shortfall being the variance the surrogate leaves unexplained, so check
  `result.S.sum()`. `HDMRResult.S1` carries the matching caveat: it is the
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
  `mu_star` on an unbounded marginal has no `q -> 0` limit, because the
  design always includes unit levels 0 and 1 exactly. Magnitudes there are
  scale-dependent by construction, so only rankings are comparable across
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
