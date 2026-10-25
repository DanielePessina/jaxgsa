# Configuration

jaxgsa needs no configuration. Skip this page until something forces you back
to it, which is usually one of three things: an index you do not trust to
enough digits, a device that runs out of memory, or a job that pays the XLA
compile again on every restart.

Three settings answer those. Precision is a JAX flag. The compilation cache and
the memory budget are jaxgsa helpers in `jaxgsa.config`. All three change
process-global state that the host application may also depend on, so none of
them is applied on import. Nothing changes until you make the call.

## Precision (float32 vs float64)

JAX computes in 32-bit by default, and it downcasts any `float64` array you
pass to `float32` on the way to the device. Producing double-precision outputs
from your model therefore buys you nothing on its own. The cast happens before
the estimator runs.

jaxgsa will not let a real loss go by silently. It does not read the dtype on
its own, because NumPy makes `float64` arrays by default and an ordinary
`Y = model(X)` is `float64` without you asking for double precision. Instead it
casts `Y` to `float32` and back. If the values come through unchanged, within
float32's own resolution, nothing is lost that float32 could have held and
there is no warning. If they do not — a value too large for float32, which
becomes `inf`, or too small, which collapses to zero — you get one warning per
analysis:

```
JaxgsaWarning: jaxgsa.sobol.analyze: Y was passed as float64, but JAX is
configured for float32, and some values do not survive the cast: they are
outside the range float32 can hold, so they arrive as inf or collapse to
zero. This is not a matter of lost trailing digits. Turn float64 on with
jax.config.update("jax_enable_x64", True), or the jax.enable_x64()
context manager, before the analysis. Rescaling the affected values into
float32's range also works.
```

The input matrix `X` is deliberately not a candidate for that warning. jaxgsa's
own samplers build every design on the host in `float64`, so warning on `X`
would fire on every analysis made entirely out of jaxgsa parts.

### When single precision is not enough

Variance-based estimators subtract large, nearly equal quantities, a
conditional variance from a total variance. Rounding error in that difference
lands straight in the index. Sobol and HSIC feel it most. Reach for float64
when you have large sample counts, indices near zero, or outputs spanning many
orders of magnitude.

Set the flag before JAX creates its first array, which means before you import
or call into jaxgsa:

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jaxgsa

problem = jaxgsa.Problem.from_dict({
    "x1": (-np.pi, np.pi), "x2": (-np.pi, np.pi), "x3": (-np.pi, np.pi),
})
design = jaxgsa.sobol.sample(problem, n_samples=4096, seed=42, verbose=False)
X = np.asarray(design.samples, dtype=np.float64)
Y = np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])

result = jaxgsa.sobol.analyze(design, Y, verbose=False)
print("S1 dtype", result.S1.dtype, np.asarray(result.S1))
```

```
S1 dtype float64 [0.33872493 0.44205618 0.01550918]
```

The flag has no effect on arrays JAX already made, which is why the order
matters.

### What float64 will and will not fix

It will not make repeated runs bit-identical. Batching an estimator at a
different width changes the order XLA sums the reduction in, and reassociating
a sum changes its last bits. That is arithmetic, not precision. We measured
it: the batch-width discrepancy falls from about `2e-7` to about `2e-16` under
x64, and never reaches zero.

The cost is real. Up to 2.1x memory, and as little as 1/64 of float32
throughput on consumer NVIDIA GPUs. TPUs have no float64 at all.

jaxgsa ships no `jaxgsa.enable_x64()` helper on purpose. `jax.config.update`
and `jax.enable_x64()` already do the job, the flag has to be set before JAX
initialises any array so a library call cannot guarantee it runs first, and a
wrapper would imply jaxgsa-specific behaviour that does not exist.

## Persistent compilation cache

JAX caches compiled XLA kernels in memory for the life of the process, and
jaxgsa memoizes its own jitted kernels, so an analysis compiles once per
configuration and later calls in the same process reuse it. Look at the
`timing:` line of any verbose summary and you are reading the first call's
compile cost folded in.

A fresh process pays that cost again. The on-disk cache is what stops it, which
matters for parameter sweeps, CI jobs, and HPC batches that re-run the same
analysis shape. Call it once, before the first `analyze`:

```python
import jaxgsa
print(jaxgsa.config.enable_compilation_cache("~/.cache/jaxgsa-jax"))
```

```
/Users/you/.cache/jaxgsa-jax
```

It returns the absolute path it configured. A leading `~` is expanded and the
path is made absolute, so the cache does not follow your working directory
around. JAX creates the directory lazily on the first write.

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
| `path` | Cache directory. `~` is expanded, the result is made absolute. |
| `min_compile_time_secs` | Only persist kernels whose compile took at least this long, so trivially cheap kernels are skipped. Default `1.0`. |
| `min_entry_size_bytes` | Minimum serialized executable size, in bytes. `0` lets the filesystem default decide. Coerced to `int`. |

::: warning
The cache directory is effectively executable. Anyone who can write to it can
make this process load and run arbitrary compiled code. Never point it at a
world-writable or shared, untrusted location. Keep it somewhere only you
control, such as `~/.cache/jaxgsa-jax`.
:::

## Memory budget

jaxgsa bounds peak transient memory by processing data in blocks and sizing
each block against a byte budget. The default is 2 GiB. The budget bounds the
transient arrays inside an analysis' forward pass. It does not bound the inputs,
the compile memory or reverse-mode gradients. Reading and writing it:

```python
import jaxgsa

print(jaxgsa.config.get_memory_budget())            # MB, the default
print(jaxgsa.config.get_memory_budget(unit="b"))

jaxgsa.config.set_memory_budget(256)                # 256 MiB
jaxgsa.config.set_memory_budget(1.5, unit="gb")     # 1.5 GiB
jaxgsa.config.set_memory_budget(65536, unit="b")    # 64 KiB
print(jaxgsa.config.get_memory_budget())
```

```
2048.0
2147483648
0.0625
```

Lower the budget on a memory-constrained device and you get more, smaller
blocks. Raise it when you have headroom and want fewer, larger ones. The
answer does not change either way, only the peak.

### What reads the budget

Most of the library, as of 1.0:

- surrogate `predict` for PCE and HDMR,
- the output-slice chunking of Sobol, HDMR, PAWN, Borgonovo, eFAST, and optimal transport,
- the Sobol bootstrap and the Morris resample chunking,
- the DGSM Jacobian batching,
- HSIC, which checks whether it can handle all inputs at once within the
  budget. If not, it handles them one at a time,
- the PCE S2 pair-mask chunking,
- the PCE and HDMR streaming fits, which engage on their own when the
  single-pass design matrix would not fit.

Borgonovo handles its output columns in batches of at most 16. The budget sets
the batch width. Each column in a batch holds about three copies of the class
layout, which is close to `D * N` values. Borgonovo does not tile its output
grid.

### Units

| `unit` | Bytes in one |
| --- | --- |
| `b`, `bytes` | 1 |
| `kb`, `kib` | 1024 |
| `mb`, `mib` | 1 048 576 (1024²) |
| `gb`, `gib` | 1 073 741 824 (1024³) |
| `tb`, `tib` | 1 099 511 627 776 (1024⁴) |

Every unit is binary. `mb` means 1024², not 1000², because memory is counted in
binary units, and because it makes `set_memory_budget(512)` restate the default
exactly. `mib` is an exact synonym of `mb`, and the same holds for each other
pair. Names ignore case and surrounding whitespace, so `"MB"`, `" mb "` and
`"Mb"` are one unit. Anything else raises and lists what it accepts:

```
ValueError: unknown memory unit 'megabytes'; accepted units are: b, bytes, gb, gib, kb, kib, mb, mib, tb, tib
```

The value may be an `int` or a `float`, must be positive and finite, and is
rounded to the nearest whole byte, which must come to at least one.

`get_memory_budget()` reads megabytes by default, the same as the setter, so
`set_memory_budget(get_memory_budget())` restores the current budget. The
return value is a `float`, because the budget need not be a whole number of
megabytes. Pass `unit="b"` to get an exact `int` of bytes.

A new budget applies only to calls made after it. An analysis already running
keeps the budget it started with.

## The batching contract

Four rules, and they hold everywhere the keyword appears.

**`batch_size` sizes row blocks, clamped to N.** It is a cap on how many rows
are resident, not a request for a particular block count. Passing more rows
than you have is legal and means "one shot". Below 1 it raises:
`ValueError: batch_size must be >= 1, got 0`.

**`None` derives the width from the memory budget.** That is the default
everywhere.

**An explicit value always wins over the budget.** No batching keyword ever
selects a different algorithm, so the indices do not depend on it. Only the
peak memory and the wall clock do.

**`hsic.analyze` has no `batch_size` at all.** Its estimator holds `2D + 1`
resident `N x N` kernel matrices, and no row block bounds those. At `D = 10`
and `N = 20000`, that is 21 matrices of 400 million float32 entries, about
34 GB, and no keyword makes it smaller. Passing one gets you `TypeError:
analyze() got an unexpected keyword argument 'batch_size'`, which is the honest
answer. HSIC's memory is set by `N`, so lower `N`.

### Which keyword each method takes

| Keyword | Blocks over | Methods |
| --- | --- | --- |
| `batch_size` | rows of `X` | `pce`, `hdmr`, `dgsm`, `vkoga`, `shapley`, and `SurrogateResult.predict` |
| `slice_chunk_size` | flattened `(T, K)` output slices | `sobol`, `efast`, `borgonovo`, `pawn`, `optimal_transport`, `hdmr`, `kucherenko`, `morris`, `rank`, and `shapley` with `backend="hdmr"` |
| `resample_chunk_size` | bootstrap replicates | `morris` |
| none | | `hsic` |

Each keyword is on `analyze` and, where the method has one, on `indices`
too. `vkoga` and `kucherenko` have no `indices`. `shapley` passes both
keywords to its backend. The PCE backend fits every output slice in one
shared solve, so `slice_chunk_size` with `backend="pce"` raises
`ValueError`.

`slice_chunk_size` and `resample_chunk_size` follow the same four rules over
their own axis, clamped to the number of slices or replicates there are.

### Seeing it work

The verbose summary prints the resolved width, so you never have to guess which
rule applied. Three widths, one DGSM analysis of 4096 rows:

```python
import numpy as np, jax.numpy as jnp, jaxgsa

problem = jaxgsa.Problem.from_dict({
    "x1": (-np.pi, np.pi), "x2": (-np.pi, np.pi), "x3": (-np.pi, np.pi),
})
X = jnp.asarray(np.random.default_rng(0).uniform(-np.pi, np.pi, size=(4096, 3)))

def f(x):  # DGSM differentiates a one-row function
    return jnp.sin(x[0]) + 7.0 * jnp.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * jnp.sin(x[0])

jaxgsa.dgsm.analyze(problem, X, fn=f)                       # None
jaxgsa.dgsm.analyze(problem, X, fn=f, batch_size=256)       # explicit
jaxgsa.dgsm.analyze(problem, X, fn=f, batch_size=10**9)     # clamped to 4096
```

The three summaries differ in one line and agree on every index. Trimmed to the
lines that differ plus the results, because the problem and timing sections are
identical and machine-specific:

```
    batch_size: auto (resolved from the memory budget)
  results: top 3 of 3 parameters by nu
    1. x2  nu=24.49
    2. x3  nu=10.77
    3. x1  nu=7.806

    batch_size: 256 (user-set)
  results: top 3 of 3 parameters by nu
    1. x2  nu=24.49
    2. x3  nu=10.77
    3. x1  nu=7.806

    batch_size: 1000000000 (user-set)
  results: top 3 of 3 parameters by nu
    1. x2  nu=24.49
    2. x3  nu=10.77
    3. x1  nu=7.806
```

Every summary also carries `gradients: reverse-mode autodiff (T*K=1, D=3)`.
DGSM picks its autodiff mode from the shape, `jacfwd` when `T*K > D` and
`jacrev` otherwise. There is no keyword for it.

The summary echoes the value you passed, not the clamp. The effective width is
`min(batch_size, N)`, so the third call ran in one shot over all 4096 rows.

Lowering the budget shows the `None` rule moving. Here is a time-resolved Sobol
analysis, `Y` of shape `(8192, 200, 1)`, so 200 output slices:

```python
design = jaxgsa.sobol.sample(problem, n_samples=8192, seed=42, verbose=False)
X = design.samples
t = jnp.linspace(0.0, 1.0, 200)
Y = (
    jnp.sin(X[:, 0])[:, None] * jnp.exp(-t)[None, :]
    + 7.0 * jnp.sin(X[:, 1])[:, None] ** 2
    + 0.1 * (X[:, 2] ** 4)[:, None] * jnp.sin(X[:, 0])[:, None] * t[None, :]
)[:, :, None]

jaxgsa.sobol.analyze(design, Y)          # default 2 GiB
jaxgsa.config.set_memory_budget(8)       # 8 MiB
jaxgsa.sobol.analyze(design, Y)
```

```
    slice_chunk_size: 200 (resolved from the memory budget)
  results: top 3 of 3 parameters by ST, mean over 200 output slices
    1. x2  ST=0.7319
    2. x1  ST=0.2634
    3. x3  ST=0.1129

    slice_chunk_size: 81 (resolved from the memory budget)
  results: top 3 of 3 parameters by ST, mean over 200 output slices
    1. x2  ST=0.7319
    2. x1  ST=0.2634
    3. x3  ST=0.1129
```

At 2 GiB all 200 slices fit in one device call. At 8 MiB jaxgsa takes them 81
at a time. Same indices to four digits. Note that the ranking line averages over
the 200 slices; the per-slice indices are in `result.ST`, shape `(200, 1, 3)`.

### The one place the budget changes the algorithm

PCE and HDMR fits are the exception to "batching never picks a different
algorithm", and they announce it. When the single-pass design matrix would not
fit the budget, the fit switches to a streaming Gram accumulation instead of
failing. 20000 rows, 5 parameters, order 6:

```python
problem5 = jaxgsa.Problem.from_dict({f"x{i}": (-np.pi, np.pi) for i in range(1, 6)})
X5 = jnp.asarray(np.random.default_rng(0).uniform(-np.pi, np.pi, size=(20000, 5)))
Y5 = jnp.sin(X5[:, 0]) + 7.0 * jnp.sin(X5[:, 1]) ** 2 + 0.1 * X5[:, 2] ** 4 * jnp.sin(X5[:, 0])

jaxgsa.config.set_memory_budget(512)
jaxgsa.pce.analyze(problem5, X5, Y5, order=6)

jaxgsa.config.set_memory_budget(1, unit="mb")
jaxgsa.pce.analyze(problem5, X5, Y5, order=6)
```

```
    order: 6
    fit: single-pass
    batch_size: auto (resolved from the memory budget)

    order: 6
    fit: streamed
    batch_size: auto (resolved from the memory budget)
```

Read the `fit:` line if you care which one ran. Streaming trades passes over
the data for peak memory, and it is what lets a fit that does not fit in memory
finish at all.

The signatures are in the [API overview](/api/#configuration).
