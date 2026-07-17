# Migrating from 0.3 to 0.4

gsax 0.4 intentionally breaks the 0.3 API. The new interface keeps commands
inside method namespaces and moves operations on fitted surrogates onto their
result objects.

## Imports and Namespaces

The package root now exports `Problem`, the input specification types, and
method namespaces. Replace root-level shortcuts with namespace calls:

| 0.3 | 0.4 |
| --- | --- |
| `gsax.sample(...)` | `gsax.sobol.sample(...)` |
| `gsax.analyze(...)` | `gsax.sobol.analyze(...)` |
| `gsax.sample_mc(...)` | `gsax.sampling.monte_carlo(...)` |
| `gsax.sample_efast(...)` | `gsax.efast.sample(...)` |
| `gsax.analyze_efast(...)` | `gsax.efast.analyze(...)` |
| `gsax.sample_morris(...)` | `gsax.morris.sample(...)` |
| `gsax.analyze_morris(...)` | `gsax.morris.analyze(...)` |
| `gsax.analyze_dgsm(...)` | `gsax.dgsm.analyze(...)` |
| `gsax.analyze_hsic(...)` | `gsax.hsic.analyze(...)` |
| `gsax.analyze_pawn(...)` | `gsax.pawn.analyze(...)` |
| `gsax.analyze_borgonovo(...)` | `gsax.borgonovo.analyze(...)` |
| `gsax.analyze_optimal_transport(...)` | `gsax.optimal_transport.analyze(...)` |
| `gsax.enable_compilation_cache(...)` | `gsax.config.enable_compilation_cache(...)` |

`monte_carlo` uses `n=...`, not `N=...`.

## Sobol Workflow

Before:

```python
samples = gsax.sample(problem, n_samples=4096, seed=42)
Y = model(samples.samples)
result = gsax.analyze(samples, Y)
```

After:

```python
samples = gsax.sobol.sample(problem, n_samples=4096, seed=42)
Y = model(samples.samples)
result = gsax.sobol.analyze(samples, Y)
```

The sampling result type is now `gsax.sobol.SobolSamples`; the analysis result
is `gsax.sobol.SobolResult`.

## PCE and HDMR

Analysis remains namespace-based, but prediction is now a result method:

```python
pce_result = gsax.pce.analyze(problem, X, Y, order=4)
Y_pred = pce_result.predict(X_new)

hdmr_result = gsax.hdmr.analyze(problem, X, Y, maxorder=2)
Y_pred = hdmr_result.predict(X_new)
```

Replace `emulate_pce(result, X_new)` and `emulate_hdmr(result, X_new)` with
`result.predict(X_new)`. Both methods accept `batch_size=...` for bounded-memory
prediction.

HDMR now exposes structural interaction arrays directly:

```python
hdmr_result.S1  # (..., D)
hdmr_result.S2  # (..., D, D)
hdmr_result.S3  # (..., D, D, D)
```

## Shapley Effects

There is no standalone `analyze_shapley(...)` in 0.4. Fit the surrogate you
want, then derive Shapley effects from that result:

```python
pce_result = gsax.pce.analyze(problem, X, Y, order=4)
effects = pce_result.shapley()

hdmr_result = gsax.hdmr.analyze(problem, X, Y, maxorder=2)
structural = hdmr_result.shapley()
correlation_aware = hdmr_result.shapley(include_correlative=True)
```

This makes the fit reusable for prediction, diagnostics, Sobol-style indices,
and Shapley effects without fitting the same surrogate twice.

## Output Shapes

0.4 accepts only:

- `(N,)` for one scalar output;
- `(N, K)` for multiple outputs;
- `(N, T, K)` for time-varying multiple outputs.

The sample axis must be first and the output axis must be last. Automatic
transpose detection and the single-output-name interpretation of `(N, T)` were
removed. Use `(N, T, 1)` for one time-varying output.

When `problem.output_names` is set, its length must match `K`.

## DGSM Pre-computed Jacobians

The `dfdx` contract of `gsax.dgsm.analyze` narrowed. In 0.3, singleton axes
were paired loosely: `(N,)` outputs were accepted with a `(N, 1, D)` Jacobian,
and `(N, 1)` outputs with a `(N, D)` Jacobian. Both tolerances were removed.
`dfdx.ndim` must now equal `Y.ndim + 1`, with the leading axes matching `Y`
exactly and the trailing axis of length `D`:

- `(N,)` outputs require `(N, D)`;
- `(N, K)` outputs require `(N, K, D)`;
- `(N, T, K)` outputs require `(N, T, K, D)`.

## Sobol Persistence

Sobol designs now use one NPZ file:

```python
samples.save("runs/design")
samples = gsax.sobol.SobolSamples.load("runs/design")
```

The `.npz` suffix is optional. CSV, text, pickle, Excel, and Parquet
persistence were removed, along with the pandas dependency.
