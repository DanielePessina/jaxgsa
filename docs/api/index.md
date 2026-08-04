# API Reference

jaxgsa 0.4 uses a namespace-oriented API. The package root contains the
foundational problem types and method namespaces; sampling and analysis
commands live under the namespace for their method.

## Foundational Types

- `jaxgsa.Problem`
- `jaxgsa.UniformInputSpec`
- `jaxgsa.GaussianInputSpec`
- `jaxgsa.CategoricalInputSpec`

Construct problems with `Problem.from_dict(...)`. Uniform inputs may use the
short `(low, high)` form; Gaussian inputs use `GaussianInputSpec`.

Categorical (unordered discrete) inputs use `CategoricalInputSpec`:
`{"dist": "categorical", "probs": [p0, ..., pL-1], "labels": [...]}`. Samples
carry the integer level codes `0 .. L-1` as floats. Optional `labels` map
codes to names for reporting (`problem.categorical_labels`);
`problem.has_categorical_inputs` reports their presence. Optimal transport,
Borgonovo delta, PAWN, and the Saltelli-based Sobol pipeline support
categorical inputs; the other methods refuse them with a `ValueError`. See
[Categorical Inputs](/examples/categorical-inputs).

Dependent inputs are declared with the optional Gaussian-copula
`correlation=` argument (`correlation_kind="latent"` or `"spearman"`), or
attached to an existing problem with `problem.with_correlation(R)`; the
validated latent matrix is available as `problem.correlation`. Methods whose
indices assume independent inputs refuse a correlated problem with a
`ValueError`. A correlation entry touching a categorical parameter is
rejected (polychoric coupling is future work). See
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

These methods analyze arbitrary aligned `(X, Y)` pairs:

| Namespace | Command | Result |
| --- | --- | --- |
| `jaxgsa.hdmr` | `analyze` | `HDMRResult` |
| `jaxgsa.pce` | `analyze` | `PCEResult` |
| `jaxgsa.dgsm` | `analyze` | `DGSMResult` |
| `jaxgsa.hsic` | `analyze` | `HSICResult` |
| `jaxgsa.pawn` | `analyze` | `PAWNResult` |
| `jaxgsa.borgonovo` | `analyze` | `DeltaResult` |
| `jaxgsa.optimal_transport` | `analyze` | `OTResult` |
| `jaxgsa.vkoga` | `analyze` | `VKOGAResult` |

Draw plain Monte Carlo inputs with
`jaxgsa.sampling.monte_carlo(problem, n, seed=...)`; it honors
`problem.correlation` transparently when one is declared. The same namespace
provides `correlate(X, problem)` (impose the declared correlation on an
existing sample by rank re-pairing), `fit_correlation(problem, X)` (estimate
the latent matrix from data), and `correlation_from_covariance(cov)`. Under a
declared correlation, `optimal_transport`, `borgonovo`, `hdmr`, `hsic`,
`pawn`, and `vkoga` accept the data. `shapley.analyze(backend="hdmr")` also
accepts it, and `include_correlative=True` then folds the ANCOVA correlative
share into the allocation. These routes raise instead: `pce.analyze`,
`dgsm.analyze`, `shapley.analyze(backend="pce")`, and the design builders
`sobol.sample`, `morris.sample`, and `efast.sample`. The design-based
`kucherenko` conditions on the declared correlation by construction.

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

`jaxgsa.vkoga` is the third surrogate-carrying namespace, and the one to reach
for when the inputs are dependent. It fits a VKOGA kernel surrogate and then
estimates the correlated variance-based indices of Li et al. (2010) against it
under a Gaussian copula:

```python
vkoga_result = jaxgsa.vkoga.analyze(problem, X, Y)  # reads problem.correlation
vkoga_result.S_TC          # total correlated — input prioritisation
vkoga_result.S_TU          # total uncorrelated — input fixing
Y_pred = vkoga_result.predict(X_new, batch_size=2048)
```

`VKOGAResult` carries `S_TC`, `S_TU`, `S_U`, `S_C`, and `S_IU`, plus the
`correlation` matrix used and the `n_centers` / `gamma` / `ridge` / `rmse` fit
diagnostics. `VKOGAResult.shapley()` raises `NotImplementedError` — a kernel
expansion has no term-wise variance decomposition to allocate from. See the
[VKOGA page](/api/vkoga) for the full index reference.

## Structured Methods

| Namespace | Workflow | Result |
| --- | --- | --- |
| `jaxgsa.efast` | `sample` then `analyze` | `EFASTResult` |
| `jaxgsa.morris` | `sample` then `analyze` | `MorrisResult` |
| `jaxgsa.kucherenko` | `sample` then `analyze` | `KucherenkoResult` |

Morris sampling returns `jaxgsa.morris.MorrisSamples`, which supports the same
single-NPZ `save(path)` / `load(path)` persistence as `SobolSamples`.
`SobolSamples.to_morris()` also returns a `MorrisSamples`, reinterpreting an
already-evaluated Saltelli design as a radial Morris design so screening
measures cost no extra model runs:

```python
samples = jaxgsa.sobol.sample(problem, 1024)
Y = model(samples.samples)
sobol_result = jaxgsa.sobol.analyze(samples, Y)
morris_result = jaxgsa.morris.analyze(samples.to_morris(), Y)
```

eFAST
sampling returns
`jaxgsa.efast.EFASTSamples`, which carries the design metadata (`n_per_curve`,
`M`, `problem`) into `jaxgsa.efast.analyze(samples, Y)` so they can never be
mismatched:

```python
samples = jaxgsa.efast.sample(problem, n_per_curve=4096, seed=42)
Y = model(samples.samples)
result = jaxgsa.efast.analyze(samples, Y)
```

## Kucherenko

`jaxgsa.kucherenko` estimates Sobol' indices for dependent inputs by
evaluating the actual model on a conditional-copula design (no surrogate).
It reads `problem.correlation` and is exempt from the correlated-design
error; with no declared correlation it reduces to the classic Saltelli
column-swap scheme and the classic `S1` / `ST`:

```python
ks = jaxgsa.kucherenko.sample(problem, 4096, seed=0)
Y = model(ks.samples)
result = jaxgsa.kucherenko.analyze(ks, Y)
result.S1   # correlation-inclusive first-order (VKOGA's S_TC)
result.ST   # correlation-exclusive total (VKOGA's S_TU)
```

Public objects: `jaxgsa.kucherenko.sample`, `jaxgsa.kucherenko.analyze`,
`jaxgsa.kucherenko.KucherenkoSamples` (with the standard NPZ
`save` / `load`), and `jaxgsa.kucherenko.KucherenkoResult`. Categorical
problems raise. See the [Kucherenko page](/api/kucherenko) for details.

## Shapley Effects

The `jaxgsa.shapley` namespace exposes `analyze` and `ShapleyResult`. The
canonical form derives Shapley effects from an existing PCE or HDMR result
(`result.shapley(...)`); `jaxgsa.shapley.analyze(problem, X, Y, backend="pce")`
is a thin convenience that fits the chosen surrogate and calls `.shapley()`
in one step — there is no separate Shapley pipeline.

All result objects support `to_dataset(...)` for labeled xarray export.

## Configuration

Use `jaxgsa.config.enable_compilation_cache(path)` to configure JAX's persistent
compilation cache.

Use `jaxgsa.config.set_memory_budget(bytes)` / `jaxgsa.config.get_memory_budget()`
to adjust the global transient-memory budget (default 512 MiB) that sizes
automatic batching: surrogate `predict` batches, HDMR output-slice chunking,
and the PCE streaming fit. Explicit per-call `batch_size` / `slice_chunk_size`
parameters always take precedence.

See the [0.3 to 0.4 migration guide](/guide/migration-0.4) for direct API
replacements.
