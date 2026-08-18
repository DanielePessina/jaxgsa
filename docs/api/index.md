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
supported methods. See [Categorical Inputs](/examples/categorical-inputs).

### Dependent inputs

Declare dependence with the optional Gaussian-copula `correlation=` argument,
together with `correlation_kind="latent"` or `correlation_kind="spearman"`.
To attach a matrix to an existing problem, call
`problem.with_correlation(R)`. The validated latent matrix is then available
as `problem.correlation`.

Methods whose indices assume independent inputs raise a `ValueError` on a
correlated problem. Use `jaxgsa.kucherenko` or `jaxgsa.vkoga` instead; the
table under [Given-Data Methods](#given-data-methods) lists which routes
accept a correlated problem.

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
- `jaxgsa.sobol.SobolSamples`
- `jaxgsa.sobol.SobolResult`

`SobolSamples.save(path)` and `SobolSamples.load(path)` use one compressed NPZ
file. `SobolSamples.downsample(...)` returns a prefix-nested smaller design.

## Given-Data Methods

These methods analyze arbitrary aligned `(X, Y)` pairs. The last column says
whether the route accepts a problem that declares a correlation.

| Namespace | Command | Result | Correlated problem |
| --- | --- | --- | --- |
| `jaxgsa.hdmr` | `analyze` | `HDMRResult` | accepted |
| `jaxgsa.pce` | `analyze` | `PCEResult` | raises `ValueError` |
| `jaxgsa.dgsm` | `analyze` | `DGSMResult` | raises `ValueError` |
| `jaxgsa.hsic` | `analyze` | `HSICResult` | accepted |
| `jaxgsa.pawn` | `analyze` | `PAWNResult` | accepted |
| `jaxgsa.borgonovo` | `analyze` | `DeltaResult` | accepted |
| `jaxgsa.optimal_transport` | `analyze` | `OTResult` | accepted |
| `jaxgsa.vkoga` | `analyze` | `VKOGAResult` | accepted |

`jaxgsa.shapley.analyze(backend="hdmr")` also accepts a correlated problem.
With `include_correlative=True` it folds the ANCOVA correlative share into the
allocation. `jaxgsa.shapley.analyze(backend="pce")` raises, as do the design
builders `sobol.sample`, `morris.sample`, and `efast.sample`. When you hit one
of those errors, switch to a route marked accepted above, or to
`jaxgsa.kucherenko`, which conditions on the declared correlation by
construction.

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
