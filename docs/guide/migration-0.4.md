# Migrating from 0.3 to 0.4

jaxgsa 0.4 intentionally breaks the 0.3 API. The new interface keeps commands
inside method namespaces. It also moves operations on fitted surrogates onto
their result objects.

Work through the steps below in order. Each step names the 0.3 code that
triggers it, then the 0.4 replacement. Every "0.3" snippet uses `import gsax`,
the original name, so the before and after are faithful.

## 1. Install the renamed package

0.4 renames the distribution and the import package from `gsax` to `jaxgsa`.
The old name is frozen at `0.3.0b1` on PyPI and receives no further releases.
There is no compatibility shim.

```sh
pip uninstall gsax      # remove the old package
pip install jaxgsa      # install the new one
```

```python
import gsax             # 0.3
import jaxgsa           # 0.4
```

## 2. Replace root-level shortcuts with namespace calls

The package root now exports `Problem`, the input specification types, and the
method namespaces. If your code calls any function in the left column, replace
it with the call in the right column.

| 0.3 | 0.4 |
| --- | --- |
| `gsax.sample(...)` | `jaxgsa.sobol.sample(...)` |
| `gsax.analyze(...)` | `jaxgsa.sobol.analyze(...)` |
| `gsax.sample_mc(...)` | `jaxgsa.sampling.monte_carlo(...)` |
| `gsax.sample_efast(...)` | `jaxgsa.efast.sample(...)` |
| `gsax.analyze_efast(...)` | `jaxgsa.efast.analyze(...)` |
| `gsax.sample_morris(...)` | `jaxgsa.morris.sample(...)` |
| `gsax.analyze_morris(...)` | `jaxgsa.morris.analyze(...)` |
| `gsax.analyze_dgsm(...)` | `jaxgsa.dgsm.analyze(...)` |
| `gsax.analyze_hsic(...)` | `jaxgsa.hsic.analyze(...)` |
| `gsax.analyze_pawn(...)` | `jaxgsa.pawn.analyze(...)` |
| `gsax.analyze_borgonovo(...)` | `jaxgsa.borgonovo.analyze(...)` |
| `gsax.analyze_optimal_transport(...)` | `jaxgsa.optimal_transport.analyze(...)` |
| `gsax.analyze_shapley(...)` | `jaxgsa.shapley.analyze(...)` or `result.shapley()` |
| `gsax.enable_compilation_cache(...)` | `jaxgsa.config.enable_compilation_cache(...)` |

If you called `sample_mc(N=...)`, rename the argument: `monte_carlo` uses
`n=...`, not `N=...`.

## 3. Update the Sobol workflow

Before:

```python
samples = gsax.sample(problem, n_samples=4096, seed=42)
Y = model(samples.samples)
result = gsax.analyze(samples, Y)
```

After:

```python
samples = jaxgsa.sobol.sample(problem, n_samples=4096, seed=42)
Y = model(samples.samples)
result = jaxgsa.sobol.analyze(samples, Y)
```

If your code names the types, note that the sampling result type is now
`jaxgsa.sobol.SobolSamples` and the analysis result type is
`jaxgsa.sobol.SobolResult`.

## 4. Rename the design row-count fields

The two row-count fields were renamed on both `SobolSamples` and
`MorrisSamples`. If you read either field, rename it.

| 0.3 | 0.4 |
| --- | --- |
| `samples.n_total` | `samples.n_runs` |
| `samples.expanded_n_total` | `samples.n_expanded` |

`n_runs` is the number of unique rows you evaluate, one model run per row.
`n_expanded` is the size of the full design layout before deduplication.

## 5. Update the eFAST workflow

`efast.sample` renamed its second parameter from `N` to `n_per_curve`. It now
returns a typed `EFASTSamples` object instead of a bare array. `efast.analyze`
takes that object first. The `M` and `problem` parameters are gone, because
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
samples = jaxgsa.efast.sample(problem, n_per_curve=4096, M=4, seed=42)
Y = model(samples.samples)
result = jaxgsa.efast.analyze(samples, Y)
```

`EFASTSamples` carries `samples`, `n_per_curve`, `M`, `problem`, and an
`n_runs` property. `n_runs` is `n_per_curve * D`, matching the package-wide
meaning: unique rows you run the model on.

Then check your design size against the stricter bound. 0.3 required only
`n_per_curve > 4*M^2`. 0.4 requires `n_per_curve >= 4*M^2*(D-1) + 1`, which
grows with the number of parameters. Below that bound there are not enough
frequencies to give every non-focal parameter a distinct one. 0.3 wrapped them
cyclically, so two parameters shared a frequency and a phase. That made them
identical along the search curve and silently biased the indices. Such designs
now raise `ValueError`. To fix the error, raise `n_per_curve` or lower `M`.

## 6. Rename the batching parameters

0.4 uses one vocabulary for the two kinds of batching, package-wide:

- `batch_size` always means rows of X/Y processed per batch. It appears on
  `pce.analyze`, `hdmr.analyze`, `dgsm.analyze`, `hsic.analyze`, and
  `result.predict`.
- `slice_chunk_size` always means output slices (`T * K` columns) processed
  per batch.

If you passed `chunk_size` to any of the calls below, rename it as shown.

| 0.3 | 0.4 |
| --- | --- |
| `gsax.analyze(..., chunk_size=...)` | `jaxgsa.sobol.analyze(..., slice_chunk_size=...)` |
| `gsax.analyze_efast(..., chunk_size=...)` | `jaxgsa.efast.analyze(..., slice_chunk_size=...)` |
| `gsax.analyze_hdmr(..., chunk_size=...)` | `jaxgsa.hdmr.analyze(..., slice_chunk_size=...)` |
| `gsax.analyze_pawn(..., chunk_size=...)` | `jaxgsa.pawn.analyze(..., slice_chunk_size=...)` |
| `gsax.analyze_borgonovo(..., chunk_size=...)` | `jaxgsa.borgonovo.analyze(..., slice_chunk_size=...)` |
| `gsax.analyze_optimal_transport(..., chunk_size=...)` | `jaxgsa.optimal_transport.analyze(..., slice_chunk_size=...)` |
| `gsax.analyze_dgsm(..., chunk_size=...)` | `jaxgsa.dgsm.analyze(..., batch_size=...)` |
| `gsax.analyze_hsic(..., chunk_size=...)` | `jaxgsa.hsic.analyze(..., batch_size=...)` |

`jaxgsa.morris.analyze` keeps its `chunk_size` parameter unchanged. There it
bounds bootstrap resamples per batch, which is neither rows nor output slices.

## 7. Set the memory budget if the defaults do not suit you

`jaxgsa.pce.analyze` and `jaxgsa.hdmr.analyze` gained a `batch_size` parameter
and automatic streaming. When the estimated memory of the single-pass fit
exceeds the active budget, the fit streams over row batches automatically. The
streamed fit is mathematically exact. It accumulates the same Gram matrices and
moments as the in-memory path, and PCE leave-one-out diagnostics stay exact
through a second pass. Only the floating-point summation order differs. To
force the streamed path, pass an explicit `batch_size=`.

The budget itself is a new process-global knob. The default is 512 MiB.
Since 0.9 the value is read in megabytes, and `unit=` takes `"b"`, `"kb"`,
`"mb"`, `"gb"` or `"tb"`.

```python
import jaxgsa

jaxgsa.config.set_memory_budget(256)  # 256 MiB
jaxgsa.config.get_memory_budget()     # 268435456, always in bytes

result = jaxgsa.pce.analyze(problem, X, Y, order=4)           # streams if needed
result = jaxgsa.hdmr.analyze(problem, X, Y, batch_size=8192)  # streaming forced
```

The budget sizes every automatic batching decision: surrogate `predict`, HDMR
output-slice chunking, and the streaming fits. An explicit per-call
`batch_size` or `slice_chunk_size` always takes precedence. See the
[configuration guide](/guide/configuration) for details.

## 8. Move PCE and HDMR prediction onto the result

Analysis stays namespace-based, but prediction is now a result method. Replace
`emulate_pce(result, X_new)` and `emulate_hdmr(result, X_new)` with
`result.predict(X_new)`:

```python
pce_result = jaxgsa.pce.analyze(problem, X, Y, order=4)
Y_pred = pce_result.predict(X_new)

hdmr_result = jaxgsa.hdmr.analyze(problem, X, Y, maxorder=2)
Y_pred = hdmr_result.predict(X_new)
```

Both methods accept `batch_size=...` for bounded-memory prediction.

HDMR now exposes structural interaction arrays directly:

```python
hdmr_result.S1  # (..., D)
hdmr_result.S2  # (..., D, D)
hdmr_result.S3  # (..., D, D, D)
```

## 9. Derive Shapley effects from a fitted result

There is no standalone Shapley pipeline in 0.4. The canonical form is to fit
the surrogate you want, then derive Shapley effects from that result:

```python
pce_result = jaxgsa.pce.analyze(problem, X, Y, order=4)
effects = pce_result.shapley()

hdmr_result = jaxgsa.hdmr.analyze(problem, X, Y, maxorder=2)
structural = hdmr_result.shapley()
correlation_aware = hdmr_result.shapley(include_correlative=True)
```

This makes the fit reusable for prediction, diagnostics, Sobol-style indices,
and Shapley effects, without fitting the same surrogate twice.

If you need only the Shapley effects, use `jaxgsa.shapley.analyze`. It wraps
the two steps as a thin convenience. It is literally
`jaxgsa.pce.analyze(...).shapley()`, or the HDMR equivalent, with no separate
pipeline behind it:

```python
effects = jaxgsa.shapley.analyze(problem, X, Y, backend="pce", order=4)
effects = jaxgsa.shapley.analyze(
    problem, X, Y, backend="hdmr", include_correlative=True
)
```

## 10. Reshape your model outputs

0.4 accepts only three output layouts:

- `(N,)` for one scalar output.
- `(N, K)` for multiple outputs.
- `(N, T, K)` for time-varying multiple outputs.

The sample axis must be first and the output axis must be last. Automatic
transpose detection was removed, and so was the single-output-name
interpretation of `(N, T)`. If you have one time-varying output, pass
`(N, T, 1)`.

If you set `problem.output_names`, its length must match `K`.

## 11. Widen pre-computed DGSM Jacobians

The `dfdx` contract of `jaxgsa.dgsm.analyze` narrowed. In 0.3, singleton axes
were paired loosely: `(N,)` outputs were accepted with a `(N, 1, D)` Jacobian,
and `(N, 1)` outputs with a `(N, D)` Jacobian. Both tolerances were removed.
`dfdx.ndim` must now equal `Y.ndim + 1`, with the leading axes matching `Y`
exactly and the trailing axis of length `D`:

- `(N,)` outputs require `(N, D)`.
- `(N, K)` outputs require `(N, K, D)`.
- `(N, T, K)` outputs require `(N, T, K, D)`.

## 12. Move saved designs to the NPZ format

Sobol designs now use one NPZ file:

```python
samples.save("runs/design")
samples = jaxgsa.sobol.SobolSamples.load("runs/design")
```

The `.npz` suffix is optional. CSV, text, pickle, Excel, and Parquet
persistence were removed, along with the pandas dependency. If you relied on
one of those formats, regenerate the design and save it as NPZ.

`MorrisSamples` gained the same `save(path)` and `load(path)` pair, using the
identical single-NPZ format and metadata schema:

```python
samples = jaxgsa.morris.sample(problem, n_trajectories=64, seed=42)
samples.save("runs/morris_design")
samples = jaxgsa.morris.MorrisSamples.load("runs/morris_design")
```
