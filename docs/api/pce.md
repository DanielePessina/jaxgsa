# PCE

```python
jaxgsa.pce.analyze(
    problem, X, Y, *,
    order=3,
    ridge=1e-8,
    fit_ratio=0.5,
    batch_size=None,
    n_bootstrap=0,
    conf_level=0.95,
    ci_method="quantile",
    key=None,
    on_invalid="raise",
    verbose=True,
    keep_replicates=False,
) -> PCEResult
```

`analyze` fits an orthogonal polynomial surrogate to arbitrary `(X, Y)` pairs,
then reads the first-, total- and second-order Sobol indices straight off the
expansion coefficients (Sudret, 2008). No structured design, and no extra model
evaluations once the fit is done. On a smooth response it needs far fewer
samples than a Monte-Carlo Sobol estimator.

It only captures what the polynomial can represent, so check `loo_rmse` before
you trust the indices.

## A run

```python
import numpy as np
import jaxgsa
from jaxgsa.sampling import monte_carlo

problem = jaxgsa.Problem(names=("x1", "x2", "x3"), bounds=((-np.pi, np.pi),) * 3)
X = monte_carlo(problem, 2000, seed=0)
Y = np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])

res = jaxgsa.pce.analyze(problem, X, Y, order=8)
print("S1      ", np.asarray(res.S1).round(4))
print("ST      ", np.asarray(res.ST).round(4))
print("loo_rmse", float(res.loo_rmse))
print("streamed", res.streamed)
```

```
jaxgsa.pce.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=2000 runs, T=1 x K=1 output slice
    invalid: none found in 2000 rows (policy 'raise')
  timing:
    fit + estimator (includes compile on the first call): 1.544 s
    order: 8
    fit: single-pass
    batch_size: auto (resolved from the memory budget)
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.5576
    2. x2  ST=0.4425
    3. x3  ST=0.243

S1       [0.3146 0.4424 0.    ]
ST       [0.5576 0.4425 0.243 ]
loo_rmse 0.08115970949711526
streamed False
```

The analytic Ishigami values are `S1 = [0.314, 0.442, 0]` and
`ST = [0.558, 0.442, 0.244]`. A degree-8 expansion on 2000 rows reproduces them
to three decimals, and `loo_rmse = 0.081` against an output standard deviation
of 3.59 says the surrogate is not the limiting factor.

Drop to the default `order=3` on the same data and the answer falls apart. The
cubic basis cannot represent `sin(x2)^2` or `x3^4 * sin(x1)`, and the fit
misattributes the variance to `x1`:

```
S1 [0.6499 0.0585 0.0009]   ST [0.9396 0.0627 0.2885]
loo_rmse 2.709   explained_variance 0.4437
```

Nothing in the index arrays says they are wrong. `loo_rmse` jumping from 0.08 to
2.71, and `explained_variance` at 0.44, are the only signal. Read them first,
every time.

`verbose=True` is the default and printed the block. Pass `verbose=False` for a
silent run.

## Arguments

| Argument | Default | What it changes |
| --- | --- | --- |
| `order` | `3` | Maximum total polynomial degree. Higher orders capture sharper nonlinearity and larger interactions, but the term count `C(D+order, order)` grows fast and needs more rows to fit. It is reduced automatically, with a warning, when the term count would exceed `fit_ratio * N`. |
| `ridge` | `1e-8` | Tikhonov regularization on the least-squares fit. The default is small enough to be a guard against a singular normal matrix and nothing more. Raise it when coefficients look unstable, which happens with noisy `Y` or near-duplicate rows. |
| `fit_ratio` | `0.5` | The terms-to-samples ceiling that triggers the `order` reduction. Lower it to demand more samples per term, which gives a more conservative fit that is harder to overfit. Must be `<= 1`: a value above 1 asks for more terms than one row per term can support. `order` is never reduced below 1 (`D + 1` terms), so a row count too small even for that raises `ValueError` instead of fitting an underdetermined system. |
| `batch_size` | `None` | Rows per batch during the fit. See below. |
| `n_bootstrap` | `0` | Row resamples. See below. |
| `key` | `None` | A `jax.random` key for the resampling. Required when `n_bootstrap > 0`. |
| `on_invalid` | `"raise"` | Policy for non-finite rows. One row is one unit, so `"drop"` removes the `(X, Y)` pair. This check matters more here than for a reduction-only method: one NaN reaches the normal equations and poisons every coefficient, every leave-one-out RMSE and every index. |
| `verbose` | `True` | Prints the summary block shown above. |
| `keep_replicates` | `False` | Retains the per-replicate index arrays on `result.ci`. |

`conf_level` and `ci_method` behave as on every other method.

## The two fit paths

There are two, and they give the same numbers. The single-pass path builds the
full design matrix. The streamed path reads rows in batches, so it holds far
less memory. Both solve the same normal equations and compute the same exact
leave-one-out error. They differ only in the order of the float32 sums.

The streamed path engages when `batch_size` is an integer below `N`, or when the
estimated resident memory of the single-pass fit (design matrix, Gram
factorization, LOO residuals) exceeds the active memory budget, about 512 MiB by
default. Set that budget with `jaxgsa.config.set_memory_budget`.
`batch_size >= N` is one full block, which is the single-pass fit.

`result.streamed` reports which path ran, and the verbose block prints it as
`fit: single-pass` or the streamed equivalent. Read it when a fit takes much
longer than you expect: `True` means the budget engaged.

The single-pass path holds two `(N, n_terms)` arrays at its peak, not three: the
design matrix and the leverage solve inside the leave-one-out step. The
coefficient solve reads `Phi.T @ Y` directly into the normal-equations solve,
so it never separately builds and holds an `(n_terms, N)` product across the
leave-one-out step. This changes the size at which streaming starts, but not
the fitted numbers.

## Confidence intervals

This is the expensive kind of bootstrap. Every replicate refits the whole
expansion, a fresh design matrix and a fresh Gram solve, so twenty replicates
cost about twenty fits. That is an order of magnitude above the row resample of
a method like [`jaxgsa.pawn`](/api/pawn), which only re-reduces numbers it
already has. Hence the `0` default, and hence a small `n_bootstrap` when you do
want intervals.

The interval measures the sampling variability of `(X, Y)` propagated through
the fit. It does not measure truncation error. An expansion too coarse to
represent the model gives a biased index, every replicate inherits the same
bias, and the interval stays narrow and wrong. `loo_rmse` and
`explained_variance` are what report that.

## PCEResult

| Field | Shape | Meaning |
| --- | --- | --- |
| `S1`, `ST` | `(D,)` / `(K, D)` / `(T, K, D)` | First-order and total Sobol indices. |
| `S2` | `(D, D)` / `(K, D, D)` / `(T, K, D, D)` | Second-order indices. |
| `S1_conf`, `ST_conf`, `S2_conf` | leading `(2, ...)` | `None` when `n_bootstrap=0`. |
| `coefficients`, `multi_index` | | The fitted expansion, reused by `predict`. |
| `order` | | The degree the fit actually used, after any reduction. |
| `loo_rmse` | `()` / `(K,)` / `(T, K)` | Exact leave-one-out RMSE per output slice, in the units of `Y`. |
| `explained_variance` | | Share of the sample output variance the fit reproduces: the in-sample R^2, so it stays in `[0, 1]`. |
| `streamed` | | Which fit path ran. |
| `problem`, `invalid`, `ci` | | Problem, non-finite report, interval provenance. |

Operations:

- `result.predict(X_new, batch_size=None)` evaluates the fitted expansion on new
  inputs.
- `result.shapley()` derives Shapley effects from the fit. See
  [Shapley effects](/api/shapley).
- `result.to_dataset(time_coords=None)` gives the labeled xarray view.

All output slices share one basis and are fitted in a single multi-right-hand-
side solve. Indices are computed independently per `(t, k)` slice.

## effective_order

```python
jaxgsa.pce.effective_order(problem, n_samples, *, order=3, fit_ratio=0.5) -> int
```

Reports the degree a fit would actually use, given only the sample size. The
reduction is real information: an expansion fitted at degree 2 does not
represent the cubic effects a caller who asked for degree 3 expects to see.

```python
jaxgsa.pce.effective_order(problem, 100, order=8)
```

```
4
```

`analyze` reports the same number twice, as a warning and as `result.order`.
`indices` reports it neither way, because it is traceable and so has no result
object and no place for a side effect. This function is where an `indices`
caller reads it, and it answers before the model has been run.

## What it refuses

`ValueError` when `X` fails validation against the problem, when `Y`'s layout
cannot be resolved against `X`'s row count, when `batch_size` is given and is
not a positive integer, when `on_invalid` is unknown or refuses the sample, when
`n_bootstrap > 0` without a `key`, when `fit_ratio > 1`, when the row count
(after `on_invalid` has run) cannot fit even the order-1 expansion at the given
`fit_ratio`, and in two structural cases:

- `problem.correlation` declares a dependence. The Wiener-Askey basis is
  orthogonal only under independent inputs, so the coefficient-to-index reading
  would be wrong. Use [`jaxgsa.hdmr`](/api/hdmr),
  [`jaxgsa.vkoga`](/api/vkoga), or [`jaxgsa.kucherenko`](/api/kucherenko).
- Any categorical parameter. A polynomial in an unordered level code has no
  meaning. Use [`jaxgsa.optimal_transport`](/api/optimal-transport),
  [`jaxgsa.borgonovo`](/api/borgonovo), or [`jaxgsa.pawn`](/api/pawn).

## Traceable core

`jaxgsa.pce.indices(problem, X, Y, *, order=3, ridge=1e-8, fit_ratio=0.5,
batch_size=None)` returns `(S1, ST, S2)` as bare arrays with none of the checks,
so it composes with `jit`, `vmap` and `jacrev`. It has no result object, so it
reports neither the reduced order nor `loo_rmse`. Use `effective_order` for the
first.

See the [PCE example](/examples/pce) and the [API overview](/api/).
