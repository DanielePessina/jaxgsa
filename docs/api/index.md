# API Reference

gsax 0.4 uses a namespace-oriented API. The package root contains the
foundational problem types and method namespaces; sampling and analysis
commands live under the namespace for their method.

## Foundational Types

- `gsax.Problem`
- `gsax.UniformInputSpec`
- `gsax.GaussianInputSpec`

Construct problems with `Problem.from_dict(...)`. Uniform inputs may use the
short `(low, high)` form; Gaussian inputs use `GaussianInputSpec`.

## Shape Contract

Every analysis accepts one of three output layouts:

| Output | Shape |
| --- | --- |
| Scalar | `(N,)` |
| Multiple outputs | `(N, K)` |
| Time series with multiple outputs | `(N, T, K)` |

The sample axis is always first and the output axis is always last. gsax does
not infer, transpose, or insert axes. When `problem.output_names` is set, its
length must equal `K`. Represent a single time-varying output explicitly as
`(N, T, 1)`.

## Sobol

```python
samples = gsax.sobol.sample(problem, n_samples=4096, seed=42)
Y = model(samples.samples)
result = gsax.sobol.analyze(samples, Y)
```

Public objects:

- `gsax.sobol.sample`
- `gsax.sobol.analyze`
- `gsax.sobol.SobolSamples`
- `gsax.sobol.SobolResult`

`SobolSamples.save(path)` and `SobolSamples.load(path)` use one compressed NPZ
file. `SobolSamples.downsample(...)` returns a prefix-nested smaller design.

## Given-Data Methods

These methods analyze arbitrary aligned `(X, Y)` pairs:

| Namespace | Command | Result |
| --- | --- | --- |
| `gsax.hdmr` | `analyze` | `HDMRResult` |
| `gsax.pce` | `analyze` | `PCEResult` |
| `gsax.dgsm` | `analyze` | `DGSMResult` |
| `gsax.hsic` | `analyze` | `HSICResult` |
| `gsax.pawn` | `analyze` | `PAWNResult` |
| `gsax.borgonovo` | `analyze` | `DeltaResult` |
| `gsax.optimal_transport` | `analyze` | `OTResult` |

Draw ordinary independent inputs with
`gsax.sampling.monte_carlo(problem, n, seed=...)`.

PCE and HDMR results retain their fitted surrogate:

```python
pce_result = gsax.pce.analyze(problem, X, Y, order=4)
Y_pred = pce_result.predict(X_new, batch_size=2048)
effects = pce_result.shapley()

hdmr_result = gsax.hdmr.analyze(problem, X, Y, maxorder=2)
Y_pred = hdmr_result.predict(X_new, batch_size=2048)
effects = hdmr_result.shapley(include_correlative=True)
```

`HDMRResult.S1`, `S2`, and `S3` expose structural indices in dense vector,
matrix, and tensor layouts. Correlation-aware Shapley effects are available
from HDMR because its ANCOVA decomposition separates structural and
correlative contributions.

## Structured Methods

| Namespace | Workflow | Result |
| --- | --- | --- |
| `gsax.efast` | `sample` then `analyze` | `EFASTResult` |
| `gsax.morris` | `sample` then `analyze` | `MorrisResult` |

Morris sampling returns `gsax.morris.MorrisSamples`.

## Shapley Results

The `gsax.shapley` namespace exposes `ShapleyResult`. Shapley effects are
derived from an existing PCE or HDMR result rather than fitted through a
standalone analysis function.

All result objects support `to_dataset(...)` for labeled xarray export.

## Configuration

Use `gsax.config.enable_compilation_cache(path)` to configure JAX's persistent
compilation cache.

See the [0.3 to 0.4 migration guide](/guide/migration-0.4) for direct API
replacements.
