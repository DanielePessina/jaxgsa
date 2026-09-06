# API reference

jaxgsa groups its API by namespace. The package root holds the problem types.
Each method has its own namespace, and that namespace holds the sampling and
analysis commands for the method.

`jaxgsa.__version__` is the installed version string, the same value as
`importlib.metadata.version("jaxgsa")`.

## Foundational types

- `jaxgsa.Problem` — the set of input parameters and their marginal
  distributions.
- `jaxgsa.UniformSpec(low, high)` — a uniform marginal.
- `jaxgsa.GaussianSpec(mean, variance, low=None, high=None)` — a Gaussian
  marginal. `low` and `high` are each optional, so the marginal can be open on
  both sides, on one side, or on neither.
- `jaxgsa.CategoricalSpec(probs, labels)` — an unordered discrete marginal.
  `probs` and `labels` are tuples, so the spec stays hashable.
- `jaxgsa.InputSpec` — the union of the three, for type annotations.

`Problem.input_specs` returns these dataclasses, one per parameter. Tell the
families apart with `isinstance`, and read the fields by name.

Construct problems with `Problem.from_dict(...)`. It accepts four input forms
per parameter: one of the dataclasses above, a bare `(low, high)` tuple
(shorthand for uniform), or the matching plain dict.

- `jaxgsa.UniformInputSpec` — `{"dist": "uniform", "low": ..., "high": ...}`.
- `jaxgsa.GaussianInputSpec` — `{"dist": "gaussian", "mean": ...,
  "variance": ..., "low": ..., "high": ...}`, where `low` and `high` are
  optional.
- `jaxgsa.CategoricalInputSpec` — `{"dist": "categorical", "probs": [...],
  "labels": [...]}`, where `labels` is optional.

The three `...InputSpec` names are `TypedDict`s. They describe the dict form
and give it type checking. A problem written that way is JSON-expressible.

```python
problem = jaxgsa.Problem.from_dict(
    {
        "x1": (0.0, 1.0),
        "x2": jaxgsa.GaussianSpec(mean=0.0, variance=4.0, high=3.0),
        "x3": {"dist": "categorical", "probs": [0.5, 0.5]},
    }
)
problem.input_specs[1].variance  # 4.0
```

`jaxgsa.Theta` is the type of the distribution-parameter mapping that
`SobolSamples.transform(theta)` takes: `Mapping[str, Mapping[str, Any]]`,
keyed by parameter name and then by field name. It is re-exported from
`jaxgsa.sobol` as well, next to the method that consumes it. See
[Sampling](/api/sampling#sobolsamples).

### Warnings

`jaxgsa.JaxgsaWarning` is the category of every warning the library raises. It
subclasses `UserWarning`, so filters you already have keep working, and it
gives you one handle for jaxgsa alone.

```python
import warnings
from jaxgsa import JaxgsaWarning

# Turn every jaxgsa warning into an exception. Good for CI: a degraded run
# fails the build instead of scrolling past.
warnings.filterwarnings("error", category=JaxgsaWarning)

# Or silence jaxgsa only, leaving other libraries' warnings alone.
warnings.filterwarnings("ignore", category=JaxgsaWarning)

# Or capture them and decide per warning.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = jaxgsa.sobol.analyze(samples, Y)
[(type(w.message).__name__, str(w.message)) for w in caught]
```

```
[('JaxgsaWarning',
  'jaxgsa.sobol.analyze: output has zero variance — all indices will be NaN')]
```

Escalating to `"error"` is the setting I would reach for in a pipeline. jaxgsa
warns when a run is degraded but still valid: a repaired correlation matrix, a
zero-variance output slice, a design thinned by deduplication. Every one of
those produces numbers that look fine and mean less than they appear to.

### Printed run summaries

All thirteen `analyze()` functions and all four `sample()` functions take a
keyword-only `verbose: bool = True`. The default prints, so a bare call writes
to stdout:

```python
result = jaxgsa.sobol.analyze(samples, Y)
```

```
jaxgsa.sobol.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=8192 runs, T=1 x K=1 output slice
    invalid: none found in 1024 Saltelli groups (policy 'raise')
  timing:
    estimators (includes compile on the first call): 0.5518 s
    slice_chunk_size: 1 (resolved from the memory budget)
    estimator: saltelli-jansen
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.556
    2. x2  ST=0.4417
    3. x3  ST=0.2413
```

Read it as a receipt. It states the shape jaxgsa inferred from your `Y`, the
resample unit the non-finite check used, whether the run compiled, and the
ranking, so a transposed output array or a silently dropped block shows up
before you plot anything. Pass `verbose=False` in loops, in tests, and
anywhere the print is noise.

### Failed model runs

Every `analyze()` function takes `on_invalid`, which says what to do when the
model output holds `NaN` or `Inf`. It accepts `"raise"` (the default),
`"propagate"` and `"drop"`. See
[Failed model runs](/guide/methods#failed-model-runs) for what each one does
and which unit of data `"drop"` removes.

Two supporting types live at the package root.

#### InvalidReport

`jaxgsa.InvalidReport` — what the non-finite check found, carried by every
result as `result.invalid`. It records `n_invalid`, `n_units`, `n_kept`,
`unit_indices`, `bad_row_indices`, `row_indices`, `sources` and the `policy`
that ran. A report with `n_invalid == 0` means the check ran and found
nothing.

`bad_row_indices` names the rows that actually held a non-finite value;
`row_indices` names every row of the blocks those rows condemn, which is what
`"drop"` removes. For a block design the two differ by a large factor. Both
refer to the array as you passed it, so they name model runs you can find.

#### OnInvalid

`jaxgsa.OnInvalid` — the type of the `on_invalid` argument, the literal
`"raise" | "propagate" | "drop" | "none"`. Exported so that typed code can
name the policy it passes.

#### InvalidUnit

`jaxgsa.InvalidUnit` — the block of data that one bad value invalidates,
reported as `result.invalid.unit`. It is one of `ROW`, `SALTELLI_GROUP`,
`TRAJECTORY`, `BASE_POINT` or `CURVE`. A design that is read in blocks cannot
lose part of one and stay valid, which is why the unit, and not the row, is
what `"drop"` removes.

### Categorical inputs

Declare a categorical parameter as
`{"dist": "categorical", "probs": [p0, ..., pL-1], "labels": [...]}`. Samples
carry the integer level codes `0 .. L-1` as floats. The optional `labels` map
codes to names for reporting.

- `problem.categorical_labels` — the label tuple for each categorical
  parameter, keyed by parameter name.
- `problem.has_categorical_inputs` — whether the problem declares any
  categorical parameter.

Optimal transport, Borgonovo delta, PAWN, and the Saltelli-based Sobol
pipeline accept categorical inputs. The other methods raise a `ValueError`
when a problem declares one. To analyze such a problem, use one of the four
supported methods. The
[method capability table](/guide/methods#method-capabilities) lists every
method's answer. See [Categorical Inputs](/examples/categorical-inputs).

### Dependent inputs

Declare dependence with the optional Gaussian-copula `correlation=` argument,
together with `correlation_type="latent"` (the default) or
`correlation_type="spearman"`. To attach a matrix to an existing problem, call
`problem.with_correlation(R)`. The validated latent matrix is then available
as `problem.correlation`.

Methods whose indices assume independent inputs raise a `ValueError` on a
correlated problem. Use `jaxgsa.kucherenko` or `jaxgsa.vkoga` instead. The
[method capability table](/guide/methods#method-capabilities) lists which
routes accept a correlated problem.

A correlation entry that touches a categorical parameter is also rejected.
Polychoric coupling is future work. See
[Correlated Inputs](/examples/correlated-inputs).

## Shape contract

Every analysis accepts one of three output layouts:

| Output | Shape |
| --- | --- |
| Scalar | `(N,)` |
| Multiple outputs | `(N, K)` |
| Time series with multiple outputs | `(N, T, K)` |

The sample axis is always first and the output axis is always last. jaxgsa does
not infer, transpose, or insert axes.

### How a 2-D Y is read

A 2-D `Y` is always `(N, K)`. There is no heuristic that might read it as
`(N, T)` instead, and no shape jaxgsa will quietly transpose for you. A time
series is `(N, T, K)`, which means a single time-varying output is written
explicitly as `(N, T, 1)`.

```python
result = jaxgsa.sobol.analyze(samples, Y_2d)     # Y_2d is (8192, 5)
result.S1.shape                                  # (5, 3)  ->  (K, D)

result = jaxgsa.sobol.analyze(samples, Y_2d[:, :, None])
result.S1.shape                                  # (5, 1, 3)  ->  (T, K, D)
```

The index arrays are the tell. `(K, D)` means jaxgsa read 5 separate outputs
at one time step. `(T, K, D)` means it read 5 time steps of one output. Check
that shape once, on the first run, and a transposed array cannot survive to
your plots.

`problem.output_names` is the guard rail worth setting. When it is present its
length must equal the trailing axis, and the mismatch is caught before any
array work:

```python
problem = jaxgsa.Problem(("x1", "x2", "x3"), bounds, output_names=("y",))
jaxgsa.sobol.analyze(samples, Y_2d)              # Y_2d is (8192, 5)
```

```
ValueError: output_names length 1 does not match the output axis K=5
```

That is one time series passed as `(N, T)`, caught by the name list. Without
`output_names` the same array is a perfectly valid five-output run and nothing
complains, so declare your outputs whenever `T` and `K` could be confused.

## Sobol

```python
samples = jaxgsa.sobol.sample(problem, n_samples=8192, seed=0)
Y = model(samples.samples)
result = jaxgsa.sobol.analyze(samples, Y)
result.S1   # Array([0.3223, 0.4361, 0.0014], dtype=float32)
result.ST   # Array([0.556 , 0.4417, 0.2413], dtype=float32)
```

Public objects:

- `jaxgsa.sobol.sample`
- `jaxgsa.sobol.analyze`
- `jaxgsa.sobol.indices`
- `jaxgsa.sobol.SobolSamples`
- `jaxgsa.sobol.SobolResult`
- `jaxgsa.sobol.Theta`

`SobolSamples.save(path)` and `SobolSamples.load(path)` use one compressed NPZ
file. `SobolSamples.downsample(...)` returns a prefix-nested smaller design.
`SobolSamples.unit` holds the same design in the unit cube, before the input
distributions are applied, and `SobolSamples.transform(theta)` applies a set
of distribution parameters to it. See
[Sampling](/api/sampling#sobolsamples).

`jaxgsa.sobol.indices(samples, Y)` returns `S1` and `ST` as plain arrays. It
runs the same estimator as `analyze`, but it checks nothing and builds no
result, so it works inside `jax.jit`, `jax.vmap` and `jax.grad`. Pair it with
`transform` to differentiate an index with respect to the input distribution
parameters. See [Sobol](/api/sobol).

Eleven of the thirteen methods expose an `indices()` with the same deal: raw
arrays, no checks, no result object, safe under a JAX transformation. Only
`kucherenko` and `vkoga` do not. `kucherenko` is host NumPy from end to end,
which is what makes it the fastest method here, and `vkoga`'s index stage is
a host quasi-Monte-Carlo loop. Neither has a traceable core to export.

## Confidence intervals

Eleven of the thirteen `analyze()` functions compute bootstrap confidence
intervals. Only `jaxgsa.efast` and `jaxgsa.hsic` do not, and in both cases
that is a decision rather than a gap.

eFAST has nothing to resample. It draws one ordered search curve per
parameter and reads it with a discrete Fourier transform, so dropping a point
does not shrink the sample, it changes what the estimator computes. An eFAST
interval needs replicated designs at different random phase shifts, which
would be a change to `efast.sample`, not a keyword on `analyze`.

HSIC already reports its uncertainty as a permutation `p_value`. Its index is
a V-statistic, a double sum over all `N^2` pairs, so a row bootstrap would
duplicate rows onto the kernel diagonal where the kernel is exactly 1. That
biases the resampled statistic upward by construction. HSIC still requires
`key`, because the permutation test needs randomness.

The keyword is `n_bootstrap` on all eleven, and it defaults to `0`, meaning no
interval. Three more keywords travel with it, everywhere with the same
meaning:

- `conf_level: float = 0.95` — the two-sided confidence level.
- `ci_method: Literal["quantile", "gaussian"] = "quantile"` — how the
  endpoints are formed. `"quantile"` reads them off the empirical bootstrap
  distribution. `"gaussian"` centres them on the point estimate and takes
  `± z · sd` of the draws, which is smoother at small `n_bootstrap` but
  assumes the draws are normal.
- `key: jax.Array | None = None` — a `jax.random` key for the resampling.
  Required as soon as `n_bootstrap > 0`; the call raises without it rather
  than seeding itself, so a reported interval is always reproducible. Pass
  `jax.random.key(0)` if all you have is an integer.

Every interval comes with `result.ci`, a `CIInfo` record that says how it was
made. A bare `*_conf` array does not say whether it is a 95% or a 68%
interval, or how many resamples it rests on. `CIInfo` does:

- `level` — the `conf_level` the analysis ran with.
- `method` — the `ci_method` that formed the endpoints.
- `n_bootstrap` — the number of bootstrap resamples drawn.
- `replicates` — the per-resample values, or `None`.

`result.ci` is `None` when the analysis ran no bootstrap.

### Keeping the bootstrap draws

All eleven take `keep_replicates`. It defaults to `False`, which throws the
draws away once the interval is computed. Pass `keep_replicates=True` to keep
them in `result.ci.replicates`, a mapping from the name of an estimate
(`"S1"`, `"mu_star"`, and so on) to an array whose leading axis has length
`n_bootstrap`.

Keep them to recompute an interval at another level, or to compute a
bias-corrected one, without running the analysis again:

```python
result = jaxgsa.sobol.analyze(
    samples, Y, n_bootstrap=1000, key=jax.random.key(0), keep_replicates=True
)
result.ci.level                      # 0.95
result.ci.n_bootstrap                # 1000
lo, hi = jnp.quantile(result.ci.replicates["S1"], jnp.array([0.05, 0.95]), axis=0)
```

The draws are large. 1000 resamples of a `(T=100, K=5, D=20)` index array is
80 MB, which is more than the rest of the result put together. That is why
they are dropped by default.

## Given-data methods

These nine methods analyze arbitrary aligned `(X, Y)` pairs. They need no
design of their own.

| Namespace | Command | Result |
| --- | --- | --- |
| `jaxgsa.borgonovo` | `analyze` | `DeltaResult` |
| `jaxgsa.dgsm` | `analyze` | `DGSMResult` |
| `jaxgsa.hdmr` | `analyze` | `HDMRResult` |
| `jaxgsa.hsic` | `analyze` | `HSICResult` |
| `jaxgsa.optimal_transport` | `analyze` | `OTResult` |
| `jaxgsa.pawn` | `analyze` | `PAWNResult` |
| `jaxgsa.pce` | `analyze` | `PCEResult` |
| `jaxgsa.shapley` | `analyze` | `ShapleyResult` |
| `jaxgsa.vkoga` | `analyze` | `VKOGAResult` |

Which of them accept a correlated or a categorical problem is in the
[method capability table](/guide/methods#method-capabilities). Shapley is the
one route whose answer depends on an argument:
`jaxgsa.shapley.analyze(backend="pce")`, the default, raises on a correlated
problem, and `backend="hdmr"` accepts one. With `include_correlative=True` the
HDMR backend folds the ANCOVA correlative share into the allocation. The
design builders `sobol.sample`, `morris.sample`, and `efast.sample` raise as
well. When you hit one of those errors, switch to a route the table marks with
a ✓, or to `jaxgsa.kucherenko`, which conditions on the declared correlation
by construction.

### Drawing inputs

Draw plain Monte Carlo inputs with
`jaxgsa.sampling.monte_carlo(problem, n, seed=...)`. It honors
`problem.correlation` transparently when one is declared. The same namespace
provides three more helpers:

- `correlate(X, problem)` — impose the declared correlation on an existing
  sample by rank re-pairing.
- `fit_correlation(problem, X)` — estimate the latent correlation matrix from
  data.
- `correlation_from_covariance(cov)` — rescale a covariance matrix to
  correlation form.

See [Sampling](/api/sampling) for the signatures.

### Fitted surrogates

PCE and HDMR results retain their fitted surrogate:

```python
pce_result = jaxgsa.pce.analyze(problem, X, Y, order=4)
Y_pred = pce_result.predict(X_new, batch_size=2048)
effects = pce_result.shapley()

hdmr_result = jaxgsa.hdmr.analyze(problem, X, Y, maxorder=2)
Y_pred = hdmr_result.predict(X_new, batch_size=2048)
effects = hdmr_result.shapley(include_correlative=True)
```

`HDMRResult.S1`, `S2`, and `S3` expose structural indices in dense vector,
matrix, and tensor layouts. Correlation-aware Shapley effects are available
from HDMR because its ANCOVA decomposition separates structural and
correlative contributions.

`jaxgsa.vkoga` is the third surrogate-carrying namespace. Reach for it when
the inputs are dependent. It fits a VKOGA kernel surrogate, then estimates the
correlated variance-based indices of Li et al. (2010) against it under a
Gaussian copula:

```python
vkoga_result = jaxgsa.vkoga.analyze(problem, X, Y)  # reads problem.correlation
vkoga_result.S_TC          # total correlated — input prioritisation
vkoga_result.S_TU          # total uncorrelated — input fixing
Y_pred = vkoga_result.predict(X_new, batch_size=2048)
```

`VKOGAResult` carries the indices `S_TC`, `S_TU`, `S_U`, `S_C`, and `S_IU`. It
also carries the `correlation` matrix used and the `n_centers`, `gamma`,
`ridge`, and `rmse` fit diagnostics. `VKOGAResult.shapley()` raises
`NotImplementedError`, because a kernel expansion has no term-wise variance
decomposition to allocate from. Use `jaxgsa.hdmr` or `jaxgsa.pce` for Shapley
effects. See the [VKOGA page](/api/vkoga) for the full index reference.

### Batching

Three keywords bound peak memory, and which one a method has depends on which
axis it batches over:

| Keyword | Batches over | On |
| --- | --- | --- |
| `batch_size` | sample rows | `dgsm`, `hdmr`, `pce`, `vkoga`, and the surrogate `predict` methods |
| `slice_chunk_size` | `(T, K)` output slices | `sobol`, `hdmr`, `pawn`, `borgonovo`, `efast`, `optimal_transport` |
| `resample_chunk_size` | bootstrap replicates | `morris` |

All three follow the same contract. The value is clamped to the axis it sizes,
and it never selects a different algorithm, so the answer does not depend on it
beyond float summation order. `None` derives a width from the memory budget; an
explicit value always overrides the budget.

`hsic`, `shapley` and `kucherenko` take none of the three.
`jaxgsa.hsic.analyze(..., batch_size=...)` raises a `TypeError`. HSIC holds
`2D+1` kernel matrices of shape `(N, N)` at once, and no keyword bounds that,
so `N` is the only knob.

## Structured methods

| Namespace | Workflow | Result |
| --- | --- | --- |
| `jaxgsa.efast` | `sample` then `analyze` | `EFASTResult` |
| `jaxgsa.morris` | `sample` then `analyze` | `MorrisResult` |
| `jaxgsa.kucherenko` | `sample` then `analyze` | `KucherenkoResult` |

Morris sampling returns `jaxgsa.morris.MorrisSamples`. It supports the same
single-NPZ `save(path)` and `load(path)` persistence as `SobolSamples`.
`SobolSamples.to_morris()` also returns a `MorrisSamples`. It reinterprets an
already-evaluated Saltelli design as a radial Morris design, so screening
measures cost no extra model runs:

```python
samples = jaxgsa.sobol.sample(problem, 8192, seed=0)
Y = model(samples.samples)
sobol_result = jaxgsa.sobol.analyze(samples, Y)
morris_result = jaxgsa.morris.analyze(samples.to_morris(), Y)
```

eFAST sampling returns `jaxgsa.efast.EFASTSamples`. That object carries the
design metadata `n_per_curve`, `M`, and `problem` into
`jaxgsa.efast.analyze(samples, Y)`, so sampling and analysis can never be
mismatched:

```python
samples = jaxgsa.efast.sample(problem, n_per_curve=1024, seed=0)
Y = model(samples.samples)
result = jaxgsa.efast.analyze(samples, Y)
```

`EFASTSamples` gained `save(path)` and `load(path)` in 1.0, so all four design
classes now persist the same way.

## Kucherenko

`jaxgsa.kucherenko` estimates Sobol' indices for dependent inputs. It
evaluates the actual model on a conditional-copula design and uses no
surrogate. It reads `problem.correlation` and is exempt from the
correlated-design error. With no declared correlation it reduces to the
classic Saltelli column-swap scheme and the classic `S1` and `ST`:

```python
ks = jaxgsa.kucherenko.sample(problem, 4096, seed=0)
Y = model(ks.samples)
result = jaxgsa.kucherenko.analyze(ks, Y)
result.S1   # correlation-inclusive first-order (VKOGA's S_TC)
result.ST   # correlation-exclusive total (VKOGA's S_TU)
```

Public objects: `jaxgsa.kucherenko.sample`, `jaxgsa.kucherenko.analyze`,
`jaxgsa.kucherenko.KucherenkoSamples` (with the standard NPZ `save` and
`load`), and `jaxgsa.kucherenko.KucherenkoResult`. A categorical problem
raises a `ValueError`; see the [Kucherenko page](/api/kucherenko) for details.

## Shapley effects

The `jaxgsa.shapley` namespace exposes `analyze` and `ShapleyResult`. The
canonical form derives Shapley effects from an existing PCE or HDMR result
with `result.shapley(...)`. There is no separate Shapley pipeline.
`jaxgsa.shapley.analyze(problem, X, Y, backend="pce")` is a thin convenience
that fits the chosen surrogate and calls `.shapley()` in one step.

All result objects support `to_dataset(...)` for labeled xarray export.

## Benchmarks

`jaxgsa.benchmarks` holds analytical test functions whose Sobol indices are
known in closed form. Use them to check an estimator, or to measure how fast
it converges. The submodules are `ishigami`, `sobol_g`, `linear`,
`gaussian_linear` and `oakley_ohagan`. Each provides a `PROBLEM`, a JAX
`evaluate(X)`, the precomputed `ANALYTICAL_S1` / `ANALYTICAL_ST` /
`ANALYTICAL_S2`, and an `analytical_indices(...)` for non-default parameters.

## Configuration

`jaxgsa.config.enable_compilation_cache(path)` points JAX's persistent
compilation cache at a directory, so a second run skips the compile.

`jaxgsa.config.set_memory_budget(budget, *, unit=None)` and
`jaxgsa.config.get_memory_budget()` read and write the process-global
transient-memory budget, default 512 MiB. Every automatic `batch_size`,
`slice_chunk_size` and `resample_chunk_size` is derived from it, and an
explicit per-call value always wins.

The [configuration guide](/guide/configuration) has the full list of what
reads the budget, the unit rules, and worked demos.
