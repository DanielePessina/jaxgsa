# Configuration

jaxgsa needs no configuration to get started. The defaults are fine for most
workloads, so you can skip this page at first.

jaxgsa runs on JAX and inherits JAX's runtime defaults. Two of those defaults
are worth revisiting for sensitivity-analysis workloads: numerical precision
and compilation caching. jaxgsa adds one setting of its own, the memory budget
that sizes automatic batching. All three are opt-in, because they change
global, process-wide state that the host application may also depend on. This
page explains when to change each one and how.

## Precision (float32 vs float64)

JAX computes in 32-bit precision by default. It also silently downcasts any
`float64` array you pass in to `float32`. Double-precision inputs therefore do
not give you double-precision arithmetic on their own, because the cast happens
before your model or the estimators run.

Single precision is fine for most problems. Variance- and covariance-based
estimators are the exception. They subtract large, nearly-equal quantities: a
conditional variance from a total variance. Rounding error in that difference
propagates directly into the indices. In jaxgsa this matters most
for Sobol' and HSIC. Heavy or ill-conditioned problems there may need double
precision to stay accurate: large sample counts, near-zero indices, or outputs
spanning many orders of magnitude.

To enable it, set the JAX flag before the first array is created, that is,
before you import or call into jaxgsa:

```python
import jax
jax.config.update("jax_enable_x64", True)

import jaxgsa  # arrays created after the flag is set now honour float64
```

The flag has no effect on arrays that JAX already initialised. That is why the
order matters.

Double precision is not free. Enabling x64 roughly doubles memory use. It is
also substantially slower on GPU and TPU, where 32-bit throughput dominates the
hardware. The flag is global and process-wide. It affects every array in the
process, not only jaxgsa's.

For this reason jaxgsa deliberately provides no `jaxgsa.enable_x64()` helper.
The flag must be set before JAX initialises any array, and a library call
cannot guarantee that it runs first. So x64 stays documentation only, and you
set the raw `jax.config.update` flag yourself.

## Persistent compilation cache

JAX compiles your analysis to XLA kernels and caches them in memory for the
lifetime of the process. jaxgsa also memoizes its jitted kernels. Each analysis
therefore compiles once per configuration, and later calls in the same process
reuse the compiled code.

The persistent on-disk cache goes one step further. It reuses compiled kernels
across process restarts. Without it, every fresh process pays the cold XLA
compile again. The on-disk cache therefore helps with parameter sweeps, CI
jobs, and HPC batches that re-run the same analysis shape many times.

jaxgsa exposes an opt-in helper for the on-disk cache. Call it once before your
first `analyze` call (such as `jaxgsa.sobol.analyze`), so the cache is active
when the first compilation happens:

```python
import jaxgsa
jaxgsa.config.enable_compilation_cache("~/.cache/jaxgsa-jax")
```

### Signature

```python
enable_compilation_cache(
    path,
    *,
    min_compile_time_secs=1.0,
    min_entry_size_bytes=0,
)
```

| Argument | Meaning |
| --- | --- |
| `path` | On-disk cache directory. A leading `~` is expanded to the user's home directory; the directory is created lazily by JAX on the first cache write. |
| `min_compile_time_secs` | Only persist kernels whose compilation took at least this many seconds, so trivially cheap kernels are skipped. Default `1.0`. |
| `min_entry_size_bytes` | Minimum serialized executable size, in bytes, to cache. `0` allows a filesystem-specific default. |

It returns the expanded cache-directory path that was configured.

> [!WARNING]
> The cache directory is effectively executable. Anyone who can write to it can
> make this process load and run arbitrary compiled code. Never point it at a
> world-writable or shared, untrusted location. Keep it under a directory that
> only you control, such as `~/.cache/jaxgsa-jax`.

## Memory budget

jaxgsa bounds peak transient memory in several places. It processes the data in
batches, and it sizes each batch against a budget in bytes. The default budget
is 512 MiB. The batched places are:

- surrogate `predict` for PCE and HDMR,
- HDMR output-slice chunking,
- PAWN output-slice chunking, which sizes `slice_chunk_size` against the
  budget when you leave it at `None`,
- the streaming fits of `jaxgsa.pce.analyze` and `jaxgsa.hdmr.analyze`, which
  engage automatically when the single-pass fit would exceed the budget.

`jaxgsa.config.set_memory_budget(...)` adjusts it globally:

```python
import jaxgsa

jaxgsa.config.set_memory_budget(256 * 1024**2)  # 256 MiB
jaxgsa.config.get_memory_budget()               # -> 268435456
```

Lower the budget on memory-constrained devices, which gives more and smaller
batches. Raise it when you have headroom and want fewer, larger batches.

Like the other settings on this page, the budget is opt-in. Nothing changes
until you call `set_memory_budget`, and the new value affects
only later jaxgsa calls. Analyses that are already running keep the budget they
started with. Explicit per-call parameters (`batch_size`, `slice_chunk_size`)
always take precedence over the budget.

The API-level summary lives in the [API overview](/api/#configuration). The
migration notes on the new streaming fits are in the
[0.3 to 0.4 migration guide](/guide/migration-0.4).
