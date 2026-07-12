# Configuration

gsax needs no configuration to get started — the defaults are fine for most
workloads, and you can safely skip this page at first. It runs on JAX and
inherits JAX's runtime defaults, two of which are worth revisiting for
sensitivity-analysis workloads: **numerical precision** and **compilation
caching**. Both are opt-in and off by default, because they mutate global,
process-wide state that the host application may also depend on. This page
documents when to enable each and how.

## Precision (float32 vs float64)

JAX computes in **32-bit** precision by default. Crucially, it also silently
**downcasts** any `float64` array you pass in to `float32`, so double-precision
inputs do not by themselves buy you double-precision arithmetic — the cast
happens before your model or the estimators ever run.

For most problems single precision is fine. But variance- and covariance-based
estimators are precision-sensitive: they subtract large, nearly-equal quantities
(a conditional variance from a total variance), and rounding error in those
differences propagates directly into the indices. In gsax this matters most for
**Sobol'** and **HSIC**, where heavy or ill-conditioned problems — large sample
counts, near-zero indices, or outputs spanning many orders of magnitude — may
need double precision to stay accurate.

To enable it, set the JAX flag **before the first array is created**, that is,
before you import or call into gsax:

```python
import jax
jax.config.update("jax_enable_x64", True)

import gsax  # arrays created after the flag is set now honour float64
```

Setting the flag after JAX has already initialised arrays has no effect on those
arrays, which is why the order matters.

Double precision is not free. Enabling x64 roughly **doubles memory use** and is
**substantially slower on GPU and TPU**, where 32-bit throughput dominates the
hardware. It is also a **global, process-wide** setting: every array in the
process is affected, not just gsax's.

For this reason there is deliberately **no `gsax.enable_x64()` helper**. The flag
must be set before JAX initialises any array, and a library call cannot guarantee
it runs first — so this is documentation only, and you set the raw
`jax.config.update` flag yourself.

## Persistent compilation cache

JAX compiles your analysis to XLA kernels and caches the compiled kernels **in
memory** for the lifetime of the process. gsax additionally memoizes its jitted
kernels, so each analysis compiles once per configuration and subsequent calls in
the same process reuse the compiled code.

The *persistent* on-disk cache goes one step further: it reuses compiled kernels
**across process restarts**. Every fresh process otherwise pays the cold XLA
compile again, so the on-disk cache is valuable for **parameter sweeps, CI jobs,
and HPC batches** that re-run the same analysis shape many times.

gsax exposes an opt-in helper. Call it **once, before your first
`gsax.analyze*` call**, so the cache is active when the first compilation
happens:

```python
import gsax
gsax.enable_compilation_cache("~/.cache/gsax-jax")
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
> The cache directory is effectively **executable**: anyone who can write to it
> can make this process load and run arbitrary compiled code. Never point it at a
> world-writable or shared, untrusted location — keep it under a directory only
> you control, such as `~/.cache/gsax-jax`.
