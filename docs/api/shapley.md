# Shapley effects

Shapley effects allocate the output variance among the parameters so that the
allocation sums to one and every parameter gets a non-negative share. They come
from a fitted surrogate's variance decomposition, so `jaxgsa.shapley` has no
pipeline of its own.

The canonical form is a result method. Fit a PCE or HDMR surrogate, then ask it
for the effects:

```python
effects = jaxgsa.pce.analyze(problem, X, Y).shapley()
effects = jaxgsa.hdmr.analyze(problem, X, Y).shapley()
effects = jaxgsa.hdmr.analyze(problem, X, Y).shapley(include_correlative=True)
```

## A run

```python
import numpy as np
import jaxgsa
from jaxgsa.sampling import monte_carlo

problem = jaxgsa.Problem(names=("x1", "x2", "x3"), bounds=((-np.pi, np.pi),) * 3)
X = monte_carlo(problem, 2000, seed=0)
Y = np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])

res = jaxgsa.pce.analyze(problem, X, Y, order=8, verbose=False).shapley()
print("Sh ", np.asarray(res.Sh).round(4), "sum", float(np.asarray(res.Sh).sum()))
print("S1 ", np.asarray(res.S1).round(4))
print("ST ", np.asarray(res.ST).round(4))
```

```
Sh  [0.4361 0.4424 0.1215] sum 1.0000000000000007
S1  [0.3146 0.4424 0.    ]
ST  [0.5576 0.4425 0.243 ]
```

This is the reason to reach for Shapley effects. `x3` has `S1 = 0` and
`ST = 0.243`, so the first-order and total indices disagree about it completely
and neither is a share of anything. `Sh = 0.122` splits the `x1`/`x3`
interaction evenly between the two parameters and lands between them. The three
effects sum to 1, so they are directly readable as percentages of output
variance.

`x1` shows the same pattern more mildly: `S1 = 0.315`, `ST = 0.558`,
`Sh = 0.436`.

## The wrapper

```python
jaxgsa.shapley.analyze(
    problem, X, Y, *,
    backend="pce",
    include_correlative=False,
    n_bootstrap=0,
    conf_level=0.95,
    ci_method="quantile",
    key=None,
    on_invalid="raise",
    verbose=True,
    keep_replicates=False,
    **backend_kwargs,
) -> ShapleyResult
```

This is literally `jaxgsa.pce.analyze(problem, X, Y, **kw).shapley()`, or the
HDMR equivalent with `include_correlative=...`. There is no separate Shapley
pipeline behind it.

```python
res = jaxgsa.shapley.analyze(problem, X, Y, backend="pce", order=8)
```

```
jaxgsa.shapley.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=2000 runs, T=1 x K=1 output slice
    invalid: none found in 2000 rows (policy 'raise')
  timing:
    backend fit + Shapley (includes compile on the first call): 0.03328 s
    backend: pce
    order: 8
  results: top 3 of 3 parameters by Sh
    1. x2  Sh=0.4424
    2. x1  Sh=0.4361
    3. x3  Sh=0.1215
```

Use the wrapper when only the Shapley effects are needed. Use the two-step form
when you also want the fitted surrogate for `predict`, for the Sobol-style
indices, or for the fit diagnostics that tell you whether to believe any of it.

The backend's own verbose summary is always suppressed, so one call prints one
summary.

## Arguments

| Argument | Default | What it changes |
| --- | --- | --- |
| `backend` | `"pce"` | Which surrogate supplies the variance decomposition. `"pce"` reads subset variances off orthonormal polynomial coefficients. `"hdmr"` fits B-spline component functions and also separates correlation-induced variance, which makes it the only route to a correlated problem here. |
| `include_correlative` | `False` | HDMR only. Allocates `Sa + Sb` instead of `Sa` alone, which keeps the allocation meaningful under correlated inputs. Passing it with `backend="pce"` raises. |
| `on_invalid` | `"raise"` | Named here rather than left to `backend_kwargs` on purpose: naming it forwards it to exactly one backend `analyze`, which applies the policy exactly once. `ShapleyResult.invalid` is that backend's report. |
| `n_bootstrap` | `0` | Row resamples on `Sh`, `S1` and `ST`. See below. |
| `key` | `None` | A `jax.random` key. Required when `n_bootstrap > 0`. |
| `verbose` | `True` | Prints the summary block shown above. |
| `**backend_kwargs` | | Passed through unchanged to the selected backend's `analyze`: `order`, `ridge`, `fit_ratio` for PCE, `maxorder`, `m`, `lambdax` for HDMR. A keyword the backend does not accept raises `TypeError`. |

Python requires `**kwargs` to come last in a signature, which is why
`keep_replicates` is not the final parameter here as it is on every other
method.

`conf_level` and `ci_method` behave as on every other method.

## Confidence intervals

The most expensive bootstrap of the three surrogate methods. A Shapley effect is
an allocation of *fitted* variances, so every replicate refits the whole
surrogate, and each replicate runs through the same backend the point estimate
used. Hence the `0` default.

The interval measures the sampling variability of `(X, Y)` propagated through
the fit. It says nothing about how well the surrogate represents the model:
every replicate shares the same basis and the same order, so a systematic misfit
sits inside every interval. `explained_variance` is what reports that.

## ShapleyResult

| Field | Shape | Meaning |
| --- | --- | --- |
| `Sh` | `(D,)` / `(K, D)` / `(T, K, D)` | The Shapley effect per parameter. Sums to 1 over the parameter axis. |
| `S1`, `ST` | same | First-order and total indices from the same fit, for comparison. |
| `explained_variance` | | Share of output variance the fit explains. Read it before the effects. |
| `backend` | | `"pce"` or `"hdmr"`. |
| `order` | | The effective order the backend fitted at. |
| `include_correlative` | | Whether `Sb` was folded in. |
| `Sh_conf`, `S1_conf`, `ST_conf` | leading `(2, ...)` | `None` when `n_bootstrap=0`. |
| `problem`, `invalid`, `ci` | | Problem, non-finite report, interval provenance. |

`res.to_dataset(time_coords=None)` gives the labeled xarray view.

## How the allocation works

The HDMR route allocates each fitted ANCOVA term's variance share equally among
the parameters that take part in that term. `include_correlative=True` allocates
`Sa + Sb`; the default allocates `Sa` alone. The PCE route does the same over
polynomial subset variances.

`VKOGAResult.shapley()` raises `NotImplementedError`. A kernel expansion is a
sum over centres and every centre involves every parameter, so there is no
membership matrix to allocate from.

## What it refuses

`ValueError` when `backend` is unknown, when `include_correlative` is requested
with the PCE backend, when `problem.correlation` declares a dependence with the
PCE backend (use `backend="hdmr"` with `include_correlative=True`), when the
problem has any categorical parameter (both backends fit a smooth surrogate over
the inputs, which is undefined for unordered level codes), or when the
underlying `analyze` rejects its inputs.

`TypeError` when `backend_kwargs` holds a keyword the selected backend's
`analyze` does not accept.

## Traceable core

`jaxgsa.shapley.indices(problem, X, Y, *, backend="pce",
include_correlative=False, **backend_kwargs)` returns `(Sh, S1, ST)` as bare
arrays with none of the checks, so it composes with `jit`, `vmap` and `jacrev`.
`explained_variance` lives on the result object only.

See the [Shapley example](/examples/shapley) and the [API overview](/api/).
