# Configuration

jaxgsa needs no configuration to get started — the defaults are fine for most
workloads, and you can safely skip this page at first. It runs on JAX and
inherits JAX's runtime defaults, two of which are worth revisiting for
sensitivity-analysis workloads: numerical precision and compilation
caching. jaxgsa adds one knob of its own, the memory budget that sizes
automatic batching. All three are opt-in, because they mutate global,
process-wide state that the host application may also depend on. This page
documents when to change each and how.

## Precision (float32 vs float64)

JAX computes in 32-bit precision by default. It also silently
downcasts any `float64` array you pass in to `float32`, so double-precision
inputs do not by themselves buy you double-precision arithmetic — the cast
happens before your model or the estimators ever run.

For most problems single precision is fine. But variance- and covariance-based
estimators are precision-sensitive: they subtract large, nearly-equal quantities
(a conditional variance from a total variance), and rounding error in those
differences propagates directly into the indices. In jaxgsa this matters most for
Sobol' and HSIC, where heavy or ill-conditioned problems — large sample
counts, near-zero indices, or outputs spanning many orders of magnitude — may
need double precision to stay accurate.

To enable it, set the JAX flag before the first array is created, that is,
before you import or call into jaxgsa:

```python
import jax
jax.config.update("jax_enable_x64", True)

import jaxgsa  # arrays created after the flag is set now honour float64
```

Setting the flag after JAX has already initialised arrays has no effect on those
arrays, which is why the order matters.

Double precision is not free. Enabling x64 roughly doubles memory use. It is
also substantially slower on GPU and TPU, where 32-bit throughput dominates the
hardware. The flag is global and process-wide: every array in the
process is affected, not just jaxgsa's.

For this reason there is deliberately no `jaxgsa.enable_x64()` helper. The flag
must be set before JAX initialises any array, and a library call cannot guarantee
it runs first — so this is documentation only, and you set the raw
`jax.config.update` flag yourself.

## Persistent compilation cache

JAX compiles your analysis to XLA kernels and caches the compiled kernels in
memory for the lifetime of the process. jaxgsa additionally memoizes its jitted
kernels, so each analysis compiles once per configuration and subsequent calls in
the same process reuse the compiled code.

The persistent on-disk cache goes one step further: it reuses compiled kernels
across process restarts. Every fresh process otherwise pays the cold XLA
compile again. The on-disk cache therefore helps with parameter sweeps, CI jobs,
and HPC batches that re-run the same analysis shape many times.

jaxgsa exposes an opt-in helper. Call it once, before your first `analyze`
call (such as `jaxgsa.sobol.analyze`), so the cache is active when the first
compilation happens:

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
> The cache directory is effectively executable: anyone who can write to it
> can make this process load and run arbitrary compiled code. Never point it at a
> world-writable or shared, untrusted location — keep it under a directory only
> you control, such as `~/.cache/jaxgsa-jax`.

## Memory budget

jaxgsa bounds peak transient memory in several places by processing data in
batches sized against a bytes budget: surrogate `predict` for PCE and HDMR,
HDMR output-slice chunking, and the streaming fits of `jaxgsa.pce.analyze` and
`jaxgsa.hdmr.analyze` (which engage automatically when the single-pass fit would
exceed the budget). The default budget is 512 MiB.

`jaxgsa.config.set_memory_budget(...)` adjusts it globally:

```python
import jaxgsa

jaxgsa.config.set_memory_budget(256 * 1024**2)  # 256 MiB
jaxgsa.config.get_memory_budget()               # -> 268435456
```

Lower it on memory-constrained devices (more, smaller batches); raise it when
you have headroom and want fewer, larger batches.

Consistent with this page's never-on-import philosophy, nothing changes until
you call it, and it only affects subsequent jaxgsa calls — analyses already
running keep the budget they started with. Explicit per-call parameters
(`batch_size`, `slice_chunk_size`) always take precedence over the budget.

The API-level summary lives in the [API overview](/api/#configuration); the
migration notes on the new streaming fits are in the
[0.3 to 0.4 migration guide](/guide/migration-0.4).
