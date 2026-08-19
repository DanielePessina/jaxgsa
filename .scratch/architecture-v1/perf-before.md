# Kernel baseline — "before" measurements

Taken 2026-08-19 with `scripts/benchmark_methods.py` on branch `perf/kernel-baseline`.
Machine-readable copy: `perf-before.json` next to this file.

**These numbers are not portable.** They are one machine, one backend, one JAX
build. A later run is comparable only if the header below matches.

| field | value |
|---|---|
| jaxgsa | 0.8.0 |
| JAX | 0.10.2 |
| NumPy | 2.4.2 |
| Python | 3.12.13 |
| platform | macOS 15.6.1, arm64 (Apple silicon) |
| JAX backend | `cpu` |
| x64 | disabled |
| whole sweep | 494 s |
| cases | 78 (13 methods x 6 output shapes), all completed |

## How to reproduce

```
uv run scripts/benchmark_methods.py --repeats 2 --timeout 900 \
  --out .scratch/architecture-v1/perf-before.json
```

## What was measured

One problem throughout: Ishigami, D = 3, uniform, independent inputs. Only the
shape of `Y` changes between rows. Given-data methods get N = 1024 rows; sobol
gets a 256-base Saltelli design (1280 rows); morris gets 32 trajectories;
eFAST gets 257 points per curve. Only `analyze` is timed — sampling is not the
estimator.

Output shapes, and why these: `(N,)`, `(N, 8)`, `(N, 32)`, `(N, 16, 4)`,
`(N, 8, 16)`, `(N, 16, 8)`. That is 1, 8, 32, 64, 128 and 128 independent
output slices — just over two orders of magnitude, enough to tell a cost that
is flat in the slice count from one that is linear in it. The ceiling is 128
because hsic is linear: at 256 slices it already runs past three minutes per
call, and the shape of its curve is unambiguous well before that. The last two
shapes hold the slice count at 128 and swap the axes, which would catch a
method that treats T and K differently. None does — every pair agrees inside
noise.

## What "peak memory" means here, and what it does not

`jax.devices()[0].memory_stats()` returns **`None`** on this backend. The CPU
client keeps no allocator statistics, so the obvious measurement is simply not
available. `jax.profiler.device_memory_profile()` gives a pprof of *live*
buffers at one instant, which is not a high-water mark. `tracemalloc` only
sees the CPython allocator, and both XLA buffers and NumPy arrays bypass it,
so it would under-report exactly the allocations that matter.

What is left is the OS high-water mark, `getrusage(RUSAGE_SELF).ru_maxrss`.
Because it is monotonic within a process, **each case runs in its own
subprocess**; the worker reads the mark after setup and again at the end, and
reports the difference. As a side effect the compile timings are honest cold
numbers too — no JIT cache is shared between cases.

Honest limits:

- It is whole-process RSS, not device allocation. It captures XLA scratch,
  result arrays, NumPy temporaries and the compiler's own memory together.
- It is a high-water mark, so it cannot fall, and `run extra` (growth after
  the first call) reads `0` whenever compilation already peaked higher. That
  means "the steady-state calls stayed under the compile-time mark", not
  "they allocated nothing".
- Floor is a few MiB of allocator jitter. **Treat differences below ~4 MiB as
  noise.**

Timing spread is reported per row. Two repeats after the first call; where the
spread is 20-80% the absolute time is under 2 ms and the row is dominated by
dispatch overhead, so read those as "too fast to distinguish".

## Full table

| method | shape | slices | compile | run (best) | spread | per slice | peak RSS | run extra |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| borgonovo | (N) | 1 | 776.9 ms | 9.7 ms | 2% | 9.7 ms | 119.5 MiB | 1.3 MiB |
| borgonovo | (N, 8) | 8 | 826.8 ms | 23.8 ms | 5% | 3.0 ms | 115.4 MiB | 0.1 MiB |
| borgonovo | (N, 32) | 32 | 871.6 ms | 145.8 ms | 3% | 4.6 ms | 161.6 MiB | 2.5 MiB |
| borgonovo | (N, 16, 4) | 64 | 860.0 ms | 187.6 ms | 2% | 2.9 ms | 215.2 MiB | 4.8 MiB |
| borgonovo | (N, 8, 16) | 128 | 853.8 ms | 300.4 ms | 5% | 2.3 ms | 348.3 MiB | 33.2 MiB |
| borgonovo | (N, 16, 8) | 128 | 861.3 ms | 301.0 ms | 4% | 2.4 ms | 346.7 MiB | 26.3 MiB |
| dgsm | (N) | 1 | 420.9 ms | 35.0 ms | 3% | 35.0 ms | 64.0 MiB | 6.5 MiB |
| dgsm | (N, 8) | 8 | 488.2 ms | 42.8 ms | 2% | 5.3 ms | 68.5 MiB | 12.4 MiB |
| dgsm | (N, 32) | 32 | 515.0 ms | 45.3 ms | 2% | 1.4 ms | 68.8 MiB | 11.2 MiB |
| dgsm | (N, 16, 4) | 64 | 539.3 ms | 46.8 ms | 7% | 0.7 ms | 70.5 MiB | 12.9 MiB |
| dgsm | (N, 8, 16) | 128 | 559.9 ms | 49.3 ms | 5% | 0.4 ms | 78.5 MiB | 15.9 MiB |
| dgsm | (N, 16, 8) | 128 | 548.5 ms | 47.5 ms | 2% | 0.4 ms | 78.2 MiB | 17.4 MiB |
| efast | (N) | 1 | 272.9 ms | 0.7 ms | 45% | 0.7 ms | 36.9 MiB | <noise |
| efast | (N, 8) | 8 | 352.9 ms | 0.9 ms | 14% | 0.1 ms | 44.2 MiB | 0.0 MiB |
| efast | (N, 32) | 32 | 381.9 ms | 1.0 ms | 28% | 0.0 ms | 37.4 MiB | 0.2 MiB |
| efast | (N, 16, 4) | 64 | 338.5 ms | 1.0 ms | 27% | 0.0 ms | 40.6 MiB | 1.1 MiB |
| efast | (N, 8, 16) | 128 | 368.5 ms | 1.5 ms | 10% | 0.0 ms | 47.5 MiB | 2.1 MiB |
| efast | (N, 16, 8) | 128 | 359.6 ms | 1.7 ms | 2% | 0.0 ms | 43.3 MiB | 1.9 MiB |
| hdmr | (N) | 1 | 1.29 s | 12.4 ms | 6% | 12.4 ms | 194.2 MiB | 0.2 MiB |
| hdmr | (N, 8) | 8 | 1.36 s | 14.2 ms | 9% | 1.8 ms | 179.3 MiB | 1.8 MiB |
| hdmr | (N, 32) | 32 | 1.36 s | 15.5 ms | 3% | 0.5 ms | 188.0 MiB | 3.8 MiB |
| hdmr | (N, 16, 4) | 64 | 1.33 s | 17.9 ms | 4% | 0.3 ms | 180.9 MiB | 8.9 MiB |
| hdmr | (N, 8, 16) | 128 | 1.35 s | 20.0 ms | 8% | 0.2 ms | 202.6 MiB | 9.5 MiB |
| hdmr | (N, 16, 8) | 128 | 1.36 s | 20.6 ms | 4% | 0.2 ms | 193.3 MiB | 0.7 MiB |
| hsic | (N) | 1 | 541.5 ms | 962.6 ms | 7% | 962.6 ms | 309.3 MiB | 36.1 MiB |
| hsic | (N, 8) | 8 | 615.9 ms | 2.73 s | 1% | 340.7 ms | 314.9 MiB | 28.1 MiB |
| hsic | (N, 32) | 32 | 984.6 ms | 8.63 s | 1% | 269.8 ms | 314.3 MiB | 14.0 MiB |
| hsic | (N, 16, 4) | 64 | 1.75 s | 16.40 s | 1% | 256.2 ms | 352.8 MiB | 48.9 MiB |
| hsic | (N, 8, 16) | 128 | 2.03 s | 32.24 s | 0% | 251.9 ms | 349.6 MiB | 34.6 MiB |
| hsic | (N, 16, 8) | 128 | 2.68 s | 32.12 s | 2% | 250.9 ms | 365.3 MiB | 56.6 MiB |
| kucherenko | (N) | 1 | 137.8 ms | 0.5 ms | 62% | 0.5 ms | 21.9 MiB | <noise |
| kucherenko | (N, 8) | 8 | 141.1 ms | 0.8 ms | 22% | 0.1 ms | 16.0 MiB | 0.1 MiB |
| kucherenko | (N, 32) | 32 | 145.4 ms | 0.6 ms | 79% | 0.0 ms | 17.2 MiB | 0.0 MiB |
| kucherenko | (N, 16, 4) | 64 | 131.6 ms | 0.8 ms | 31% | 0.0 ms | 18.3 MiB | 0.8 MiB |
| kucherenko | (N, 8, 16) | 128 | 134.6 ms | 1.3 ms | 17% | 0.0 ms | 23.3 MiB | 1.8 MiB |
| kucherenko | (N, 16, 8) | 128 | 133.0 ms | 1.3 ms | 9% | 0.0 ms | 23.0 MiB | 1.8 MiB |
| morris | (N) | 1 | 495.0 ms | 1.6 ms | 8% | 1.6 ms | 70.3 MiB | <noise |
| morris | (N, 8) | 8 | 551.8 ms | 1.1 ms | 44% | 0.1 ms | 59.8 MiB | 0.0 MiB |
| morris | (N, 32) | 32 | 577.9 ms | 1.6 ms | 5% | 0.1 ms | 64.0 MiB | 0.1 MiB |
| morris | (N, 16, 4) | 64 | 525.7 ms | 1.3 ms | 34% | 0.0 ms | 63.7 MiB | 0.2 MiB |
| morris | (N, 8, 16) | 128 | 573.5 ms | 1.6 ms | 33% | 0.0 ms | 68.2 MiB | 0.2 MiB |
| morris | (N, 16, 8) | 128 | 540.5 ms | 1.5 ms | 34% | 0.0 ms | 60.8 MiB | 0.2 MiB |
| optimal_transport | (N) | 1 | 683.2 ms | 11.5 ms | 5% | 11.5 ms | 92.6 MiB | 0.0 MiB |
| optimal_transport | (N, 8) | 8 | 673.9 ms | 47.2 ms | 0% | 5.9 ms | 87.9 MiB | 0.0 MiB |
| optimal_transport | (N, 32) | 32 | 618.9 ms | 139.8 ms | 0% | 4.4 ms | 99.6 MiB | 7.7 MiB |
| optimal_transport | (N, 16, 4) | 64 | 588.8 ms | 260.3 ms | 0% | 4.1 ms | 113.5 MiB | 7.7 MiB |
| optimal_transport | (N, 8, 16) | 128 | 623.6 ms | 511.0 ms | 10% | 4.0 ms | 111.2 MiB | 3.7 MiB |
| optimal_transport | (N, 16, 8) | 128 | 534.8 ms | 512.8 ms | 1% | 4.0 ms | 107.7 MiB | 1.8 MiB |
| pawn | (N) | 1 | 870.4 ms | 7.7 ms | 3% | 7.7 ms | 113.5 MiB | 0.0 MiB |
| pawn | (N, 8) | 8 | 996.1 ms | 19.7 ms | 4% | 2.5 ms | 116.0 MiB | 4.0 MiB |
| pawn | (N, 32) | 32 | 969.2 ms | 65.8 ms | 2% | 2.1 ms | 149.0 MiB | 7.1 MiB |
| pawn | (N, 16, 4) | 64 | 861.7 ms | 118.5 ms | 1% | 1.9 ms | 114.1 MiB | 0.1 MiB |
| pawn | (N, 8, 16) | 128 | 889.1 ms | 237.4 ms | 0% | 1.9 ms | 154.6 MiB | 0.8 MiB |
| pawn | (N, 16, 8) | 128 | 973.3 ms | 236.7 ms | 21% | 1.8 ms | 158.7 MiB | 27.2 MiB |
| pce | (N) | 1 | 1.42 s | 5.6 ms | 41% | 5.6 ms | 149.4 MiB | 0.4 MiB |
| pce | (N, 8) | 8 | 1.43 s | 4.7 ms | 6% | 0.6 ms | 137.5 MiB | 0.1 MiB |
| pce | (N, 32) | 32 | 1.46 s | 5.3 ms | 3% | 0.2 ms | 141.9 MiB | 0.2 MiB |
| pce | (N, 16, 4) | 64 | 1.45 s | 4.4 ms | 30% | 0.1 ms | 132.8 MiB | 0.4 MiB |
| pce | (N, 8, 16) | 128 | 1.40 s | 5.1 ms | 17% | 0.0 ms | 135.1 MiB | 0.8 MiB |
| pce | (N, 16, 8) | 128 | 1.46 s | 4.3 ms | 28% | 0.0 ms | 140.8 MiB | 2.3 MiB |
| shapley | (N) | 1 | 1.58 s | 5.7 ms | 6% | 5.7 ms | 173.5 MiB | 0.3 MiB |
| shapley | (N, 8) | 8 | 1.70 s | 4.8 ms | 34% | 0.6 ms | 169.6 MiB | 0.2 MiB |
| shapley | (N, 32) | 32 | 1.70 s | 5.8 ms | 5% | 0.2 ms | 162.4 MiB | 0.1 MiB |
| shapley | (N, 16, 4) | 64 | 1.61 s | 4.5 ms | 19% | 0.1 ms | 154.6 MiB | 0.8 MiB |
| shapley | (N, 8, 16) | 128 | 1.66 s | 5.7 ms | 16% | 0.0 ms | 155.6 MiB | 2.5 MiB |
| shapley | (N, 16, 8) | 128 | 1.67 s | 5.2 ms | 25% | 0.0 ms | 154.2 MiB | 1.3 MiB |
| sobol | (N) | 1 | 835.5 ms | 2.2 ms | 11% | 2.2 ms | 93.1 MiB | 0.0 MiB |
| sobol | (N, 8) | 8 | 990.9 ms | 6.7 ms | 6% | 0.8 ms | 90.3 MiB | 0.1 MiB |
| sobol | (N, 32) | 32 | 1.02 s | 21.7 ms | 0% | 0.7 ms | 93.4 MiB | 0.2 MiB |
| sobol | (N, 16, 4) | 64 | 1.00 s | 41.1 ms | 8% | 0.6 ms | 95.5 MiB | 0.2 MiB |
| sobol | (N, 8, 16) | 128 | 1.13 s | 84.8 ms | 3% | 0.7 ms | 112.8 MiB | 0.6 MiB |
| sobol | (N, 16, 8) | 128 | 1.10 s | 80.2 ms | 6% | 0.6 ms | 109.8 MiB | 0.3 MiB |
| vkoga | (N) | 1 | 1.65 s | 362.6 ms | 5% | 362.6 ms | 292.3 MiB | 80.3 MiB |
| vkoga | (N, 8) | 8 | 1.56 s | 334.3 ms | 4% | 41.8 ms | 270.0 MiB | 74.4 MiB |
| vkoga | (N, 32) | 32 | 1.57 s | 360.9 ms | 7% | 11.3 ms | 256.4 MiB | 59.9 MiB |
| vkoga | (N, 16, 4) | 64 | 2.01 s | 523.5 ms | 2% | 8.2 ms | 275.8 MiB | 77.2 MiB |
| vkoga | (N, 8, 16) | 128 | 2.03 s | 637.1 ms | 1% | 5.0 ms | 289.6 MiB | 78.1 MiB |
| vkoga | (N, 16, 8) | 128 | 2.01 s | 463.6 ms | 2% | 3.6 ms | 285.7 MiB | 78.6 MiB |

## Scaling summary

Run time at 1 slice, at 128 slices, and the cost of one slice at 128. The last
column is the one that ranks methods; the ratio column says whether the method
amortises across slices at all.

| method | 1 slice | 128 slices | ratio | ms per slice @128 | peak MiB @1 | peak MiB @128 | compile @1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hsic | 962.6 ms | **32.1 s** | 33.4x | **250.9** | 309 | 365 | 541 ms |
| vkoga | 362.6 ms | 463.6 ms | 1.3x | 3.62 | 292 | 286 | 1645 ms |
| optimal_transport | 11.5 ms | 512.8 ms | 44.7x | 4.01 | 93 | 108 | 683 ms |
| borgonovo | 9.7 ms | 301.0 ms | 31.0x | 2.35 | 120 | **347** | 777 ms |
| pawn | 7.7 ms | 236.7 ms | 30.9x | 1.85 | 113 | 159 | 870 ms |
| sobol | 2.2 ms | 80.2 ms | 36.0x | 0.63 | 93 | 110 | 835 ms |
| dgsm | 35.0 ms | 47.5 ms | 1.4x | 0.37 | 64 | 78 | 421 ms |
| hdmr | 12.4 ms | 20.6 ms | 1.7x | 0.16 | 194 | 193 | 1295 ms |
| shapley | 5.7 ms | 5.2 ms | 0.9x | 0.041 | 174 | 154 | 1582 ms |
| pce | 5.6 ms | 4.3 ms | 0.8x | 0.034 | 149 | 141 | 1419 ms |
| efast | 0.7 ms | 1.7 ms | 2.5x | 0.013 | 37 | 43 | 273 ms |
| morris | 1.6 ms | 1.5 ms | 1.0x | 0.012 | 70 | 61 | 495 ms |
| kucherenko | 0.5 ms | 1.3 ms | 2.5x | 0.010 | 22 | 23 | 138 ms |

Three things stand out before any code is read.

**hsic is in a class of its own.** At 128 slices it takes 32 seconds. The next
slowest method takes 0.51 s. Its cost per slice is 250 ms and stays 250 ms as
slices are added, so nothing is being shared between slices at all — 63x the
per-slice cost of the next worst method.

**A ratio near 1 does not mean "well optimised".** pce, shapley, hdmr, morris,
efast and kucherenko are flat because their real work does not depend on the
slice count (one multi-RHS solve, one set of elementary effects). vkoga and
dgsm are flat because a large fixed cost dominates. Flatness is the *right*
shape; it is only evidence of good kernels when the absolute per-slice number
is also small.

**The predicted "no jit means near-zero compile" contrast does not hold, and
that is itself informative.** pce and shapley have no `jax.jit` anywhere on
their hot path, yet they report the two *highest* first-call costs in the
sweep (1.42 s and 1.58 s). Eager JAX still compiles: every `jnp` primitive
gets its own small HLO module on first use, so op-by-op code pays compilation
per operation instead of once. The genuine no-compile signature belongs to
kucherenko (138 ms), and it earns that by not touching the device at all — its
estimator is host float64 NumPy.

## Where the time goes — attribution

Every claim below points at a line. Paths are relative to `src/jaxgsa/`.

### 1. hsic: a nested Python loop over T and K around the whole estimator

`hsic/_analyze.py:495` `for t in range(T):` and `hsic/_analyze.py:496`
`for k in range(K):` wrap `_compute_slice`, and each iteration writes its
answer back with a host-side scatter at `hsic/_analyze.py:515-518`
(`r2_all = r2_all.at[t, k].set(r2)`), outside any jit. The method *is* jitted
(`hsic/_analyze.py:325`, cached at `:254`), but the jit boundary is inside the
loop, one slice at a time. So T*K separate dispatches, T*K separate
`jax.lax.scan` permutation runs (`hsic/_analyze.py:320`), and T*K host round
trips.

The measurement matches: per-slice cost is 341 ms at 8 slices, 270 ms at 32,
256 ms at 64 and 251 ms at 128 — it settles onto a flat line and stays there.
The only thing amortised is the one-off setup; the estimator itself shares
nothing between slices.

Two costs are being repeated that need not be. The input kernels are built by
`vmap` over the D parameters at `hsic/_analyze.py:298`, `:305`, `:314`, `:488`
— those depend only on `X` and are the same for every output slice, yet they
sit inside the per-slice call path. `_build_one_kernel` at
`hsic/_analyze.py:356` rebuilds the (N, N) output kernel and re-derives the
median-heuristic bandwidth per slice, which genuinely does vary per slice and
must stay.

There is a second, corroborating signal: hsic's *compile* time grows with the
slice count (541 ms at 1 slice, 2.68 s at 128) even though the cached kernel
is keyed only on `n_perms`. Compilation should be constant here. The growth is
consistent with the per-slice host scatter at `:515-518` being traced outside
the jit, and it is the only method in the sweep whose compile time scales with
output size.

Memory is also the highest in the sweep (309-365 MiB), and the D-wise `vmap`s
at `:298`/`:305`/`:314` are unchunked — D * N^2 kernels live at once. At
D = 3, N = 1024 that is fine; it is a wall waiting at larger D and N, and it
is exactly the failure mode the memory budget exists to prevent.

### 2. optimal_transport: chunked correctly, but sequential inside

512.8 ms at 128 slices, 4.0 ms per slice, flat. The slice loop is already the
target shape — `optimal_transport/_analyze.py:747`
`for start in range(0, total, cs):` with `cs` derived at `:742-745`, and a
chunk `vmap` at `:276` inside jitted kernels (`:175`, `:285`). So the outer
structure is right and the per-slice cost is the kernel's own.

What stays sequential is inside: bootstrap replicates go through
`jax.lax.scan` at `optimal_transport/_analyze.py:281` and `:424`, and the
per-class loop uses `jax.lax.map` at `:381`. Both are deliberate
memory-safety choices, and both are serial. This is a real cost but a
defensible one; the win here is smaller and riskier than hsic's.

### 3. borgonovo: fastest-growing memory in the sweep

301 ms and 347 MiB at 128 slices, against 9.7 ms and 120 MiB at one — +227 MiB
of peak for 128 slices, the largest growth in the sweep. `run extra` climbs
to 33 MiB, so the steady-state calls are pushing past the compile-time mark
rather than reusing what compilation already touched. The slice loop is chunked (`borgonovo/_analyze.py:827`, budget at
`:824`), so the growth is the chunk budget doing its job rather than an
unbounded `vmap` — but the KDE kernel behind `borgonovo/_analyze.py:320` is
the heaviest per-chunk allocator in the library, and the bootstrap `lax.scan`
at `:327` serialises on top of it.

### 4. sobol: a deliberate Python loop over slices on the bootstrap path

`sobol/_analyze.py:377-378` `for t in range(T): for k in range(K):` inside
`_analyze_with_bootstrap`. The docstring at `:343-348` says this is a
conscious trade of vectorisation for bounded memory. The measurement agrees
that the trade is being made: 36x more time for 128x more slices, 0.63 ms per
slice, flat. It also shows the trade is currently paying too much — the
non-bootstrap path right next to it *is* chunk-`vmap`ped (`:212`, `:230`,
kernels at `:69-70`) and there is no reason the bootstrap path cannot use the
same chunking instead of a chunk size of one.

Note: another agent is editing `sobol/**` in a separate worktree, so these
line numbers may move.

### 5. pawn: chunked slices, but the bootstrap re-runs everything

236.7 ms at 128 slices, 1.85 ms per slice, flat. Slices are chunked properly
(`pawn/_analyze.py:338`, kernel at `:224-226`). The cost multiplier is
`pawn/_analyze.py:503` `for _ in range(n_bootstrap):`, which re-enters the
entire chunked core once per resample and re-gathers `bin_idx[idx]` and
`Y_3d[idx]` each time. With `n_bootstrap = 10` here, roughly ten times the
necessary gather traffic.

### 6. kucherenko: no device work at all — a capability gap, not a speed one

Fastest method in the table (1.3 ms at 128 slices) and the lowest memory
(23 MiB). It earns none of that from good kernels: `kucherenko/_analyze.py:57
analyze` is un-jitted and its estimator body is host float64 NumPy
(`:130` `F = np.asarray(...)`, `:154-156`, `:160`). No `jit`, no `vmap`,
nothing on the device. At N = 1024 and D = 3 that is simply faster than paying
dispatch. It also means the method cannot be differentiated, cannot be
`vmap`ped by a caller, and materialises a full `(2D+1, N, S)` float64 host
array at `:130` — which is the wall at large N. Optimising it will make the
benchmark *worse* at this size while being the right thing to do.

### 7. pce and shapley: eager, but structurally right

Both are flat and cheap per slice (0.034 and 0.041 ms) because all T*K slices
are solved as RHS columns of a single Gram solve (`pce/_analyze.py:348`,
`:355-358`); shapley just delegates (`shapley/_analyze.py:111`). Nothing is
jitted — `_fit_pce_core` at `pce/_analyze.py:303` and `build_design_matrix` at
`pce/_engine.py:88` are eager — and that costs them the two highest first-call
times in the sweep (1.42 s, 1.58 s) with a steady state of about 5 ms. The
opportunity here is compile time, not run time.

### 8. Ragged final chunks: one avoidable extra compile per run

`efast/_analyze.py:256-258`, `morris/_analyze.py:350-354` and
`dgsm/_analyze.py:399-403` pad the last chunk back to full width, so exactly
one shape is ever traced. `sobol/_analyze.py:212` and `:230`,
`pawn/_analyze.py:338`, `borgonovo/_analyze.py:827`,
`optimal_transport/_analyze.py:747`, `hdmr/_analyze.py:573` and
`_core/batching.py:124` do not, so whenever `total % cs != 0` a second shape
of the same kernel is compiled. Visible but small: sobol's compile goes
835 ms -> 1.13 s and hdmr's 1.29 s -> 1.36 s as slices grow, while efast and
morris stay flat.

### 9. `jax.jit` applied inside a function body — a latent risk, not a measured cost

`dgsm/_analyze.py:364` builds `jax.jit(jax.vmap(jax.jacrev(...)))` inside
`_compute_moments`, and `vkoga/_analyze.py:498` and `vkoga/_engine.py:543` do
the same inside their own bodies. In principle a fresh wrapper per call means
a fresh compilation cache per call.

**The measurement does not show that happening.** dgsm's first call is 456 ms
and its repeats are 35 ms with 3% spread; if it recompiled per call the
repeats would sit near the first. JAX's underlying trace cache is keyed on the
code object, which appears to be saving it. The only residue is dgsm's `run
extra` RSS of 6-17 MiB, which persists across shapes. Report it as a latent
hazard — it will bite the moment a closure variable becomes part of the key —
but do not claim a speed-up for fixing it, because there is none to measure
here.

## Ranked opportunities

Gains are for the 128-slice case on this machine. "Expected" means what the
structure predicts, not what has been demonstrated — that is the point of
taking the before-numbers.

| # | opportunity | file:line | now | expected after | confidence |
|---|---|---|---|---|---|
| 1 | **hsic**: fuse the T,K loop into the jit, `vmap` over a chunk of slices, and hoist the X-kernel build out of the per-slice path | `hsic/_analyze.py:495-496`, `:515-518`, `:298`/`:305`/`:314`, `:356` | 32.1 s | 1-5 s | high |
| 2 | **hsic** (follow-on): chunk the D-wise kernel `vmap`s against the memory budget | `hsic/_analyze.py:298`, `:305`, `:314`, `:488` | 365 MiB, O(D N^2) live | budget-bounded | high |
| 3 | **sobol**: give the bootstrap path the same chunked `vmap` the point-estimate path already has | `sobol/_analyze.py:377-378` vs `:212`/`:230`/`:69-70` | 80 ms | 10-30 ms | high |
| 4 | **pawn**: batch the bootstrap resamples instead of re-entering the core per resample | `pawn/_analyze.py:503` | 237 ms | 60-120 ms | medium |
| 5 | **optimal_transport**: replace the per-replicate `scan` with a budget-sized chunked `vmap` | `optimal_transport/_analyze.py:281`, `:424`, `:381` | 513 ms | 150-300 ms | medium |
| 6 | **borgonovo**: same treatment for the bootstrap `scan`; watch the peak, this is the heaviest allocator in the library | `borgonovo/_analyze.py:327`, `:320`, `:824-827` | 301 ms, 347 MiB | 120-200 ms at equal or lower peak | medium |
| 7 | **pad ragged final chunks** (six sites; three others already do it) | `sobol/_analyze.py:212`, `:230`; `pawn/_analyze.py:338`; `borgonovo/_analyze.py:827`; `optimal_transport/_analyze.py:747`; `hdmr/_analyze.py:573`; `_core/batching.py:124` | 100-300 ms extra compile per run | one compile | high, small |
| 8 | **pce**: jit `_fit_pce_core` and `build_design_matrix` | `pce/_analyze.py:303`, `pce/_engine.py:88` | 1.42 s first call | 0.3-0.6 s first call; run time unchanged | medium |
| 9 | **hdmr**: `_f_ppf` bisects with 100 sequential `betainc` calls, each forcing a host sync | `hdmr/_engine.py:330` | inside a 1.3 s compile | modest | medium |
| 10 | **dgsm / vkoga**: lift the in-body `jax.jit` to module level | `dgsm/_analyze.py:364`, `vkoga/_analyze.py:498`, `vkoga/_engine.py:543` | no measured cost | correctness of the cache, not speed | low gain, do anyway |
| 11 | **kucherenko**: port the host-NumPy estimator to a jitted device kernel | `kucherenko/_analyze.py:57`, `:130`, `:154-160` | 1.3 ms (already fastest) | likely *slower* at N=1024; needed for autodiff and large N | capability, not speed |

Items 1 and 2 are worth more than everything below them combined. hsic alone
accounts for 32.1 s of the 33.8 s that all thirteen methods together spend on
the (N, 16, 8) case — 95% of the total. Nothing else on the list changes the
picture until it is fixed.

## Things found while measuring, not fixed

- No bugs. All 78 cases ran and returned results; no method raised.
- hsic's compile time scaling with output size (541 ms -> 2.68 s) is the one
  result that is not explained by the loop alone and deserves a second look
  when the loop is fused.
- The `(N, 8, 16)` and `(N, 16, 8)` pairs agree within noise for all thirteen
  methods, so no method is currently treating the T axis differently from the
  K axis. Worth re-checking after any change.
