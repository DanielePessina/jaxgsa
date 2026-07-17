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
| `gsax.analyze_shapley(...)` | `gsax.shapley.analyze(...)` or `result.shapley()` |
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

## Design Row Counts

The two row-count fields on sampling results were renamed, on both
`SobolSamples` and `MorrisSamples`:

| 0.3 | 0.4 |
| --- | --- |
| `samples.n_total` | `samples.n_runs` |
| `samples.expanded_n_total` | `samples.n_expanded` |

`n_runs` is the number of unique rows you evaluate (one model run per row);
`n_expanded` is the size of the full design layout before deduplication.

## eFAST Workflow

`efast.sample` renamed its second parameter from `N` to `n_per_curve` and now
returns a typed `EFASTSamples` object instead of a bare array. `efast.analyze`
takes that object first — the `M` and `problem` parameters are gone, because
both travel inside the design object and can no longer be mismatched between
sampling and analysis.

Before:

```python
X = gsax.sample_efast(problem, 4096, M=4, seed=42)
Y = model(X)
result = gsax.analyze_efast(problem, Y, M=4)
```

After:

```python
samples = gsax.efast.sample(problem, n_per_curve=4096, M=4, seed=42)
Y = model(samples.samples)
result = gsax.efast.analyze(samples, Y)
```

`EFASTSamples` carries `samples`, `n_per_curve`, `M`, `problem`, and an
`n_runs` property (`n_per_curve * D`, the package-wide meaning: unique rows
you run the model on).

## Batching Parameters

0.4 uses one vocabulary for the two kinds of batching, package-wide:

- `batch_size` always means **rows of X/Y** processed per batch
  (`pce.analyze`, `hdmr.analyze`, `dgsm.analyze`, `hsic.analyze`, and
  `result.predict`);
- `slice_chunk_size` always means **output slices** (`T * K` columns)
  processed per batch.

The former `chunk_size` parameters were renamed accordingly:

| 0.3 | 0.4 |
| --- | --- |
| `gsax.sobol.analyze(..., chunk_size=...)` | `gsax.sobol.analyze(..., slice_chunk_size=...)` |
| `gsax.efast.analyze(..., chunk_size=...)` | `gsax.efast.analyze(..., slice_chunk_size=...)` |
| `gsax.hdmr.analyze(..., chunk_size=...)` | `gsax.hdmr.analyze(..., slice_chunk_size=...)` |
| `gsax.pawn.analyze(..., chunk_size=...)` | `gsax.pawn.analyze(..., slice_chunk_size=...)` |
| `gsax.borgonovo.analyze(..., chunk_size=...)` | `gsax.borgonovo.analyze(..., slice_chunk_size=...)` |
| `gsax.optimal_transport.analyze(..., chunk_size=...)` | `gsax.optimal_transport.analyze(..., slice_chunk_size=...)` |
| `gsax.dgsm.analyze(..., chunk_size=...)` | `gsax.dgsm.analyze(..., batch_size=...)` |
| `gsax.hsic.analyze(..., chunk_size=...)` | `gsax.hsic.analyze(..., batch_size=...)` |

`gsax.morris.analyze` keeps its `chunk_size` parameter unchanged — there it
bounds bootstrap resamples per batch, which is neither rows nor output slices.

## Streaming Fits and the Memory Budget

`gsax.pce.analyze` and `gsax.hdmr.analyze` gained a `batch_size` parameter and
automatic streaming: when the estimated memory of the single-pass fit exceeds
the active budget, the fit streams over row batches automatically. The
streamed fit is mathematically exact — it accumulates the same Gram matrices
and moments as the in-memory path (and PCE leave-one-out diagnostics stay
exact via a second pass), differing only in floating-point summation order.
Passing an explicit `batch_size=` forces the streamed path.

The budget itself is a new process-global knob (default 512 MiB):

```python
import gsax

gsax.config.set_memory_budget(256 * 1024**2)  # 256 MiB
gsax.config.get_memory_budget()               # 268435456

result = gsax.pce.analyze(problem, X, Y, order=4)           # streams if needed
result = gsax.hdmr.analyze(problem, X, Y, batch_size=8192)  # streaming forced
```

It sizes every automatic batching decision (surrogate `predict`, HDMR
output-slice chunking, and the streaming fits). Explicit per-call
`batch_size` / `slice_chunk_size` parameters always take precedence. See the
[configuration guide](/guide/configuration) for details.

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

There is no standalone Shapley pipeline in 0.4. The canonical form is to fit
the surrogate you want, then derive Shapley effects from that result:

```python
pce_result = gsax.pce.analyze(problem, X, Y, order=4)
effects = pce_result.shapley()

hdmr_result = gsax.hdmr.analyze(problem, X, Y, maxorder=2)
structural = hdmr_result.shapley()
correlation_aware = hdmr_result.shapley(include_correlative=True)
```

This makes the fit reusable for prediction, diagnostics, Sobol-style indices,
and Shapley effects without fitting the same surrogate twice.

When only the Shapley effects are needed, `gsax.shapley.analyze` wraps the
two steps as a thin convenience — it is literally
`gsax.pce.analyze(...).shapley()` (or the HDMR equivalent), with no separate
pipeline behind it:

```python
effects = gsax.shapley.analyze(problem, X, Y, backend="pce", order=4)
effects = gsax.shapley.analyze(
    problem, X, Y, backend="hdmr", include_correlative=True
)
```

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

## Design Persistence

Sobol designs now use one NPZ file:

```python
samples.save("runs/design")
samples = gsax.sobol.SobolSamples.load("runs/design")
```

The `.npz` suffix is optional. CSV, text, pickle, Excel, and Parquet
persistence were removed, along with the pandas dependency.

`MorrisSamples` gained the same `save(path)` / `load(path)` pair, using the
identical single-NPZ format and metadata schema:

```python
samples = gsax.morris.sample(problem, n_trajectories=64, seed=42)
samples.save("runs/morris_design")
samples = gsax.morris.MorrisSamples.load("runs/morris_design")
```
