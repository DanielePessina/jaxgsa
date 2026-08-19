# API Reference

jaxgsa 0.4 uses a namespace-oriented API. The package root holds the problem
types. Each method has its own namespace, and that namespace holds the
sampling and analysis commands for the method.

## Foundational Types

- `jaxgsa.Problem` — the set of input parameters and their marginal
  distributions.
- `jaxgsa.UniformInputSpec` — a uniform marginal.
- `jaxgsa.GaussianInputSpec` — a Gaussian marginal, optionally truncated.
- `jaxgsa.CategoricalInputSpec` — an unordered discrete marginal.

Construct problems with `Problem.from_dict(...)`. Uniform inputs may use the
short `(low, high)` form. Gaussian inputs use `GaussianInputSpec`.

The package root also holds `jaxgsa.JaxgsaWarning`, the category of every
warning that jaxgsa raises. It is a subclass of `UserWarning`, so existing
`UserWarning` filters keep working. Filter on `JaxgsaWarning` to select the
jaxgsa warnings alone, for example
`warnings.filterwarnings("ignore", category=JaxgsaWarning)`.

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
`"raise" | "propagate" | "drop"`. Exported so that typed code can name the
policy it passes.

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
together with `correlation_kind="latent"` or `correlation_kind="spearman"`.
To attach a matrix to an existing problem, call
`problem.with_correlation(R)`. The validated latent matrix is then available
as `problem.correlation`.

Methods whose indices assume independent inputs raise a `ValueError` on a
correlated problem. Use `jaxgsa.kucherenko` or `jaxgsa.vkoga` instead. The
[method capability table](/guide/methods#method-capabilities) lists which
routes accept a correlated problem.

A correlation entry that touches a categorical parameter is also rejected.
Polychoric coupling is future work. See
[Correlated Inputs](/examples/correlated-inputs).

## Shape Contract

Every analysis accepts one of three output layouts:

| Output | Shape |
| --- | --- |
| Scalar | `(N,)` |
| Multiple outputs | `(N, K)` |
| Time series with multiple outputs | `(N, T, K)` |

The sample axis is always first and the output axis is always last. jaxgsa does
not infer, transpose, or insert axes. When `problem.output_names` is set, its
length must equal `K`. Represent a single time-varying output explicitly as
`(N, T, 1)`.

## Sobol

```python
samples = jaxgsa.sobol.sample(problem, n_samples=4096, seed=42)
Y = model(samples.samples)
result = jaxgsa.sobol.analyze(samples, Y)
```

Public objects:

- `jaxgsa.sobol.sample`
- `jaxgsa.sobol.analyze`
- `jaxgsa.sobol.indices`
- `jaxgsa.sobol.SobolSamples`
- `jaxgsa.sobol.SobolResult`

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
parameters. See [Analyze (Sobol)](/api/analyze).

## Confidence Intervals

Five methods report bootstrap confidence intervals: `sobol`, `morris`,
`pawn`, `borgonovo` and `optimal_transport`. Two keyword spellings are in
use. `sobol.analyze` and `morris.analyze` take `num_resamples`, and
`pawn.analyze`, `borgonovo.analyze` and `optimal_transport.analyze` take
`n_bootstrap`. The
[method capability table](/guide/methods#method-capabilities) records the
spelling for each method. The other eight methods report no intervals.

Every interval comes with `result.ci`, a `CIInfo` record that says how it was
made. A bare `*_conf` array does not say whether it is a 95% or a 68%
interval, or how many resamples it rests on. `CIInfo` does:

- `level` — the two-sided confidence level, the `conf_level` the analysis ran
  with.
- `method` — the endpoint rule used. `"quantile"` takes empirical bootstrap
  quantiles and `"gaussian"` takes a normal approximation. `sobol` and
  `morris` choose between them with `ci_method`. The other three always use
  the percentile interval and record `"quantile"`.
- `n_resamples` — the number of bootstrap resamples drawn.
- `replicates` — the per-resample values, or `None`.

`result.ci` is `None` when the analysis ran no bootstrap.

### Keeping the bootstrap draws

All five `analyze()` functions take `keep_replicates`. It defaults to `False`,
which throws the draws away once the interval is computed. Pass
`keep_replicates=True` to keep them in `result.ci.replicates`, a mapping from
the name of an estimate (`"S1"`, `"mu_star"`, and so on) to an array whose
leading axis has length `n_resamples`.

Keep them to recompute an interval at another level, or to compute a
bias-corrected one, without running the analysis again:

```python
result = jaxgsa.sobol.analyze(samples, Y, num_resamples=1000, keep_replicates=True)
result.ci.level                      # 0.95
result.ci.n_resamples                # 1000
lo, hi = jnp.quantile(result.ci.replicates["S1"], jnp.array([0.05, 0.95]), axis=0)
```

The draws are large. 1000 resamples of a `(T=100, K=5, D=20)` index array is
80 MB, which is more than the rest of the result put together. That is why
they are dropped by default.

## Given-Data Methods

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

## Structured Methods

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
samples = jaxgsa.sobol.sample(problem, 1024)
Y = model(samples.samples)
sobol_result = jaxgsa.sobol.analyze(samples, Y)
morris_result = jaxgsa.morris.analyze(samples.to_morris(), Y)
```

eFAST sampling returns `jaxgsa.efast.EFASTSamples`. That object carries the
design metadata `n_per_curve`, `M`, and `problem` into
`jaxgsa.efast.analyze(samples, Y)`, so sampling and analysis can never be
mismatched:

```python
samples = jaxgsa.efast.sample(problem, n_per_curve=4096, seed=42)
Y = model(samples.samples)
result = jaxgsa.efast.analyze(samples, Y)
```

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

## Shapley Effects

The `jaxgsa.shapley` namespace exposes `analyze` and `ShapleyResult`. The
canonical form derives Shapley effects from an existing PCE or HDMR result
with `result.shapley(...)`. There is no separate Shapley pipeline.
`jaxgsa.shapley.analyze(problem, X, Y, backend="pce")` is a thin convenience
that fits the chosen surrogate and calls `.shapley()` in one step.

All result objects support `to_dataset(...)` for labeled xarray export.

## Configuration

Use `jaxgsa.config.enable_compilation_cache(path)` to configure JAX's persistent
compilation cache.

Use `jaxgsa.config.set_memory_budget(megabytes)` and
`jaxgsa.config.get_memory_budget()` to adjust the global transient-memory
budget. The default is 512 MiB. This budget sizes automatic batching: the
surrogate `predict` batches, the HDMR output-slice chunking, and the PCE
streaming fit. An explicit per-call `batch_size` or `slice_chunk_size` always
takes precedence.

See the [0.3 to 0.4 migration guide](/guide/migration-0.4) for direct API
replacements.
