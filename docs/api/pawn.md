# PAWN

```python
jaxgsa.pawn.analyze(
    problem, X, Y, *,
    n_bins=10,
    statistic="median",
    n_bootstrap=0,
    conf_level=0.95,
    ci_method="quantile",
    key=None,
    slice_chunk_size=None,
    on_invalid="raise",
    verbose=True,
    keep_replicates=False,
) -> PAWNResult
```

PAWN asks how much fixing a parameter changes the whole output distribution,
not only its variance. For each parameter the samples are split into `n_bins`
conditioning bins, equal-width on the CDF-transformed unit interval and
therefore equal-probability under that parameter's marginal. Each bin's
conditional output CDF is compared with the unconditional CDF by the
Kolmogorov-Smirnov statistic, the largest vertical gap between the two curves,
a number in `[0, 1]`. The per-bin values are then aggregated into one index per
parameter.

Any `(X, Y)` pair works. PAWN earns its place when the output is skewed or
multimodal, where a variance-based index summarises the uncertainty badly.

## A run

```python
import numpy as np
import jaxgsa
from jaxgsa.sampling import monte_carlo

problem = jaxgsa.Problem(names=("x1", "x2", "x3"), bounds=((-np.pi, np.pi),) * 3)

def ishigami(X):
    return (np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2
            + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0]))

X = monte_carlo(problem, 4000, seed=0)
res = jaxgsa.pawn.analyze(problem, X, ishigami(X))
print(np.asarray(res.pawn))
print(np.asarray(res.n_valid_bins))
```

```
jaxgsa.pawn.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=4000 runs, T=1 x K=1 output slice
    invalid: none found in 4000 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.6993 s
    slice_chunk_size: 1 (resolved from the memory budget)
    statistic: median
    n_bins: 10
  results: top 3 of 3 parameters by PAWN
    1. x2  PAWN=0.4022
    2. x1  PAWN=0.2484
    3. x3  PAWN=0.08682

[0.24840467 0.40216702 0.08681973]
[10 10 10]
```

`verbose=True` is the default and printed the block. Pass `verbose=False` for a
silent run.

## n_valid_bins, and the sparse-bin warning

A bin contributes to the index only when it holds at least two samples. One
sample cannot define a conditional CDF, so the KS kernel returns `NaN` and the
nan-aware aggregation drops that bin. `n_valid_bins` counts what survived, per
parameter.

In the run above all three parameters kept all 10 bins. Push the bin count past
what the sample supports and they do not:

```python
X = monte_carlo(problem, 60, seed=1)
res = jaxgsa.pawn.analyze(problem, X, ishigami(X), n_bins=40, verbose=False)
print(np.asarray(res.n_valid_bins))
```

```
JaxgsaWarning: jaxgsa.pawn: parameters 'x1' (19/40), 'x2' (18/40), 'x3' (15/40)
have fewer than half of their conditioning bins contributing (a bin needs at
least 2 samples to define a conditional CDF; the rest are dropped). The
reported indices rest on those few bins. Use fewer bins (lower n_bins) or more
samples.

[19 18 15]
```

The warning fires when a parameter keeps fewer than half its bins. Half is the
threshold for the warning, not the threshold for a trustworthy index. Read
`n_valid_bins` on every run where `N / n_bins` is small, and treat any count
well below `n_bins` as a reason to lower `n_bins` or collect more samples.

Bin occupancy depends on `X` alone, so the count is constant across the output
`T` and `K` axes. It is broadcast to `pawn`'s shape anyway, so the exported
dataset aligns the two. For a categorical parameter the reference count is its
level count rather than `n_bins`.

## Arguments

| Argument | Default | What it changes |
| --- | --- | --- |
| `n_bins` | `10` | Conditioning bins per continuous parameter, equal-probability under its marginal. More bins condition more tightly but leave about `N / n_bins` samples per bin, which makes each KS value noisier and eventually drops bins entirely. The default suits `N` in the thousands. A categorical parameter ignores it and uses one bin per level. |
| `statistic` | `"median"` | How the per-bin KS values are aggregated. `"median"` shrugs off a few noisy bins. `"max"` is the conservative choice for screening: a parameter is negligible only if no bin shifts the output, so `"max"` will not call a parameter unimportant on the strength of a majority of quiet bins. `"mean"` weights all bins equally. |
| `n_bootstrap` | `0` | Row resamples behind `pawn_conf`. `0` skips them and leaves `pawn_conf` at `None`. This is the cheap kind of bootstrap: nothing is refitted, the estimator just re-reduces numbers it already has, so a few hundred replicates are affordable here in a way they are not for `pce` or `vkoga`. |
| `conf_level` | `0.95` | Confidence level for the intervals. |
| `ci_method` | `"quantile"` | `"quantile"` reads the endpoints off the empirical bootstrap distribution. `"gaussian"` centres them on the point estimate and takes `+/- z * sd` of the draws, which is smoother for a small `n_bootstrap` but assumes the draws are normal. |
| `key` | `None` | A `jax.random` key for the resampling. Required when `n_bootstrap > 0`. Use `jax.random.key(0)` if you have an integer seed. |
| `slice_chunk_size` | `None` | Flattened `T*K` output columns per kernel call. `None` derives one from the memory budget (`jaxgsa.config.set_memory_budget`). Peak memory is dominated by the ECDF tables, roughly `2 * slice_chunk_size * D * N * n_bins` elements, because the inner `vmap` holds a full `(N, n_bins)` table per (column, parameter) pair. Lower it when a time-series output runs the device out of memory. It changes no index: output columns are independent. |
| `on_invalid` | `"raise"` | Policy for non-finite rows. `"drop"` removes the `(X, Y)` pair, `"propagate"` warns and computes anyway. |
| `verbose` | `True` | Prints the summary block shown above. |
| `keep_replicates` | `False` | Keeps the per-resample indices on `result.ci.replicates`, `n_bootstrap` copies of the index array. Turn it on to recompute an interval at another level without re-running the analysis. |

## PAWNResult

| Field | Shape | Meaning |
| --- | --- | --- |
| `pawn` | `(..., D)` | The index, in `[0, 1]`. 0 means fixing the parameter leaves the output distribution unchanged. |
| `pawn_conf` | `(2, ..., D)` | `[lower, upper]`, `None` when `n_bootstrap=0`. |
| `n_valid_bins` | same as `pawn` | Contributing bins per parameter. See above. |
| `problem` | | The problem the analysis ran on. |
| `invalid` | | What the non-finite check found and which policy ran. |
| `ci` | | Confidence level, endpoint rule, resample count, and the draws when `keep_replicates=True`. |

Leading axes follow the shape contract: `(D,)` for `Y` of shape `(N,)`,
`(K, D)` for `(N, K)`, `(T, K, D)` for `(N, T, K)`.

`res.to_dataset(time_coords=None)` gives the labeled xarray view.

## Correlated inputs

Supported. PAWN conditions on bins of one parameter and compares output CDFs, so
a declared `problem.correlation` does not invalidate the indices. Each index
then reports a parameter's total influence, including what it carries through
its correlated partners. A parameter the model ignores can score above 0 when it
correlates with an influential one. That reading is correct.

## What it refuses

`ValueError` for a non-2-D `X`, a column count that disagrees with the problem,
mismatched row counts, a `statistic` outside the three names, `n_bins < 2`,
`conf_level` outside `(0, 1)`, a `slice_chunk_size` that is not a positive
integer, an unknown `on_invalid`, `n_bootstrap < 0`, an unknown `ci_method`,
`n_bootstrap > 0` with no `key`, or a sample the non-finite policy refuses.

`JaxgsaWarning` for a zero-variance output slice, where every conditional
distribution equals the unconditional one so the index is an exact 0 rather
than an answer, and for the sparse-bin case above.

## Traceable core

`jaxgsa.pawn.indices(problem, X, Y, *, n_bins=10, statistic="median",
slice_chunk_size=None)` returns a one-element tuple holding the index array,
with none of the checks, so it composes with `jit`, `vmap` and `jacrev`. It is a
tuple for consistency with the other `indices` functions, which return several
arrays. `n_valid_bins` is not among them, and neither is the sparse-bin warning.

See the [PAWN example](/examples/pawn), [Methods](/guide/methods), and the
[API overview](/api/).
