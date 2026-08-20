# Borgonovo delta

```python
jaxgsa.borgonovo.analyze(
    problem, X, Y, *,
    n_classes=None,
    grid_size=100,
    bandwidth="silverman",
    n_bootstrap=0,
    conf_level=0.95,
    ci_method="quantile",
    bias_correct=None,
    key=None,
    slice_chunk_size=None,
    degenerate_tol=0.01,
    degenerate_bandwidth="auto",
    on_invalid="raise",
    verbose=True,
    keep_replicates=False,
) -> DeltaResult
```

The delta index is moment-independent. It measures how much fixing an input
shifts the whole output density, on a `[0, 1]` scale, so it sees changes in
spread, tails and shape that a variance-based index cannot report. Any
`(X, Y)` pair works and no structured design is needed. `analyze` also returns
the given-data first-order Sobol index `S1` computed from the same class
partition, which costs nothing extra and is useful for comparison.

## A run

```python
import numpy as np
import jaxgsa
from jaxgsa.sampling import monte_carlo

problem = jaxgsa.Problem(names=("x1", "x2", "x3"), bounds=((-np.pi, np.pi),) * 3)
X = monte_carlo(problem, 4000, seed=0)
Y = np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])

res = jaxgsa.borgonovo.analyze(problem, X, Y)
print(np.asarray(res.delta).round(4))
print(np.asarray(res.S1).round(4))
```

```
jaxgsa.borgonovo.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=4000 runs, T=1 x K=1 output slice
    invalid: none found in 4000 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.4616 s
    slice_chunk_size: 1 (resolved from the memory budget)
    grid_size: 100
    bandwidth: silverman
  results: top 3 of 3 parameters by delta
    1. x2  delta=0.3387
    2. x1  delta=0.2149
    3. x3  delta=0.1606

[0.2149 0.3387 0.1606]
[0.3057 0.4208 0.0026]
```

Compare the two rows for `x3`. Its `S1` is 0.003, so fixing it barely moves the
output mean. Its `delta` is 0.161, third of three but far from nothing, because
fixing it does reshape the output density through the `x3^4 * sin(x1)` term.
That gap is the whole reason to run this method.

## bias_correct is a tri-state

`bias_correct` takes `True`, `False`, or `None`, and `None` is the default. What
`None` does depends on `n_bootstrap`.

| Value | With `n_bootstrap > 0` | With `n_bootstrap == 0` |
| --- | --- | --- |
| `None` (default) | Applies the Plischke correction `2*d_hat - mean(d_boot)` and warns once per process. | Does nothing. |
| `True` | Applies the correction, silently. | Cannot deliver it, so it warns. |
| `False` | Never applies it. `delta` is the plug-in estimate. | Never applies it. |

The consequence to internalise: adding `n_bootstrap=100` to a default call does
not only add intervals. It changes the reported `delta` from the plug-in
estimate to the corrected one.

```python
import jax

r = jaxgsa.borgonovo.analyze(problem, X, Y, n_bootstrap=50,
                             key=jax.random.key(0), verbose=False)
print(np.asarray(r.delta).round(4))

r = jaxgsa.borgonovo.analyze(problem, X, Y, n_bootstrap=50,
                             key=jax.random.key(0), bias_correct=False,
                             verbose=False)
print(np.asarray(r.delta).round(4))
```

```
JaxgsaWarning: jaxgsa.borgonovo: bias_correct was left at its default (None)
and n_bootstrap > 0, so the Plischke bias correction IS applied: the delta
reported is 2*d_hat - mean(d_boot), not the plug-in estimate. Pass
bias_correct=True to keep this and silence this warning, or bias_correct=False
for the uncorrected delta. This warning is shown once per process.

[0.2106 0.3347 0.1562]
[0.2149 0.3387 0.1606]
```

The correction is small here, about 0.004 on each parameter, and it moves in the
direction you would expect. The uncorrected delta is positively biased: a KDE
separation is a distance, so sampling noise can only push it up. Use
`n_bootstrap >= 100` when the value matters and not only the ranking.

The tri-state exists because the correction needs replicates and replicates are
opt-in. A plain `bias_correct=True` default next to `n_bootstrap=0` would be a
contradiction that warned on every default call.

`S1` is never bias-corrected, matching SALib.

## Arguments

| Argument | Default | What it changes |
| --- | --- | --- |
| `n_classes` | `None` | Equal-frequency conditioning classes per continuous parameter. `None` uses the Plischke sample-size heuristic, roughly `N**(2/7)` classes saturating at 48, identical to SALib's rule. More classes condition more tightly and leave fewer samples per class, which makes each conditional KDE noisier. A passed value is validated against `[2, N]`. A categorical parameter ignores it and uses one class per level, sized by the observed level counts; declared levels with no samples are dropped with a warning. When every parameter is categorical, a `JaxgsaWarning` says the value is ignored. |
| `grid_size` | `100` | Points on the shared output grid the densities are compared on, spanning `[Y.min(), Y.max()]` per column. It is the knob that moves the answer for near-atomic conditional classes, because `grid_step` sets the floored bandwidth. `grid_size < 2` raises. |
| `bandwidth` | `"silverman"` | KDE bandwidth rule, or a positive float used directly. |
| `bias_correct` | `None` | See above. |
| `key` | `None` | A `jax.random` key for the resampling. Required when `n_bootstrap > 0`. |
| `slice_chunk_size` | `None` | Flattened `T*K` output columns per kernel call. `None` derives it from the memory budget and the real class layout. One column costs the summed padded class layout `sum_g(Dg * Mg * Pg)` per output-grid point, which is about `D * N` for continuous parameters. An imbalanced categorical parameter pads every level up to the largest one, so it can cost many times `D * N`. The grid is then evaluated in tiles, so peak memory follows the tile rather than the whole grid, and narrowing this chunk is rarely the thing that saves memory here. |
| `degenerate_tol` | `0.01` | A conditioning class counts as degenerate when its KDE bandwidth falls below this fraction of the full-sample bandwidth. Degenerate classes get the floor below. Lowering it lets narrower classes keep their own bandwidth. Raising it above the floor fraction biases the result: a class whose own bandwidth sits between the floor and this tolerance is then *narrowed* to the floor, which inflates delta for exactly the classes the higher tolerance said to distrust. Valid range `[0, 1)`. |
| `degenerate_bandwidth` | `"auto"` | Bandwidth floor for a degenerate class. `"auto"` uses `max(0.1 * h_full, grid_step)`, which never goes below what the output grid can integrate. A float is a fraction of `h_full` applied exactly, with no grid-step bound, and a value far below `grid_step / h_full` risks aliasing. Whether it aliases depends on where the narrow class falls relative to the grid, so `analyze` does not refuse the setting up front. It checks the returned delta instead, and the error message then names this argument and the value that would fix it. On data with no degenerate class this argument cannot change the result at any value. |
| `on_invalid` | `"raise"` | Policy for non-finite rows. The check runs before the KDE, so a failed model run is named for what it is instead of turning up later as a bandwidth complaint. |
| `verbose` | `True` | Prints the summary block shown above. |
| `keep_replicates` | `False` | Keeps `n_bootstrap` copies of both index arrays on `result.ci.replicates`. The `delta` draws are the ones the interval was taken from, so they carry the bias correction when it was applied. |

`conf_level` and `ci_method` behave as on every other method.

## DeltaResult

| Field | Shape | Meaning |
| --- | --- | --- |
| `delta` | `(..., D)` | The delta index. 0 means the output distribution does not change with the parameter, 1 means it fully determines it. Bias-corrected when the correction applied. |
| `delta_conf` | `(2, ..., D)` | `[lower, upper]`, `None` when `n_bootstrap=0`. |
| `S1` | `(..., D)` | Given-data first-order Sobol index from the same class partition. |
| `S1_conf` | `(2, ..., D)` | `None` when `n_bootstrap=0`. |
| `problem`, `invalid`, `ci` | | Problem, non-finite report, interval provenance. |

Leading axes follow the shape contract: `(D,)`, `(K, D)`, `(T, K, D)`.
`res.to_dataset(time_coords=None)` gives the labeled xarray view.

The plug-in estimate stays inside `[0, 1]`. The bias-corrected estimate and its
bounds can fall marginally below 0 for weak parameters at small `N`. A
confidence bound outside `[0, 1]` by more than 0.05 raises a `JaxgsaWarning`
naming the parameter and the bound; the point estimate still stands, only the
interval is suspect. A returned `delta` outside `[0, 1]` by more than 0.05 is an
error, not an estimate, and raises.

## Continuous output only

This estimator compares Gaussian kernel density estimates on a shared output
grid, and a discrete output has atoms no grid resolves. `analyze` checks the
output up front and raises `ValueError` when a column takes at most 20 distinct
values and each value repeats at least 5 times on average. Use
[`jaxgsa.optimal_transport`](/api/optimal-transport) for a discrete output: it
compares empirical distributions directly and needs no density.

The check does not refuse a continuous output rounded to a few decimals, and it
does not refuse a constant column, whose exact answer is `delta = S1 = 0` (SALib
raises there instead).

The restriction is on the output. **Categorical parameters are supported.**

## Atomic conditional classes

A conditioning class can be a point mass or nearly one. That is the normal case
for a categorical level that maps to a single output value. For such a class the
delta estimate depends on the grid resolution and is biased low. On a noise-free
three-atom model with true delta `2/3` the estimate is 0.56 at `grid_size=50`,
0.61 at 100, and 0.61 at 200 and above. The bias also does not vanish as `N`
grows, so on atomic conditionals this estimator is not consistent. Treat delta
on such parameters as a ranking signal, not a calibrated number. Parameters with
genuine conditional spread are unaffected.

A class the output grid cannot resolve gets the floored KDE bandwidth and one
`JaxgsaWarning`.

## Traceable core

`jaxgsa.borgonovo.indices(problem, X, Y, *, n_classes=None, grid_size=100,
bandwidth="silverman", slice_chunk_size=None, degenerate_tol=0.01,
degenerate_bandwidth="auto")` returns `(delta, S1)` as bare arrays with none of
the checks, so it composes with `jit`, `vmap` and `jacrev`. There is no bias
correction there, because there is no bootstrap.

See the [Borgonovo delta example](/examples/borgonovo),
[Methods](/guide/methods), and the [API overview](/api/).
