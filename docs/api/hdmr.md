# HDMR

```python
jaxgsa.hdmr.analyze(
    problem, X, Y, *,
    maxorder=2,
    maxiter=100,
    m=2,
    lambdax=0.01,
    slice_chunk_size=None,
    batch_size=None,
    n_bootstrap=0,
    conf_level=0.95,
    ci_method="quantile",
    key=None,
    on_invalid="raise",
    verbose=True,
    keep_replicates=False,
) -> HDMRResult
```

RS-HDMR fits B-spline component functions to arbitrary `(X, Y)` pairs: one per
parameter, one per parameter pair, and so on up to `maxorder`. The ANCOVA
sensitivity indices then come from the fitted components. No structured design
is needed, so it suits existing datasets and models too expensive for a Sobol or
eFAST scheme.

Its distinguishing feature is the ANCOVA split. Each term's variance share
divides into a structural part `Sa` and a correlation-induced part `Sb`, and
that split stays meaningful when the inputs are dependent.

`analyze` requires `N >= 300`.

## A run

```python
import numpy as np
import jaxgsa
from jaxgsa.sampling import monte_carlo

problem = jaxgsa.Problem(names=("x1", "x2", "x3"), bounds=((-np.pi, np.pi),) * 3)
X = monte_carlo(problem, 2000, seed=0)
Y = np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])

res = jaxgsa.hdmr.analyze(problem, X, Y)
print("terms ", res.terms)
print("Sa    ", np.asarray(res.Sa).round(4))
print("Sb    ", np.asarray(res.Sb).round(4))
print("select", np.asarray(res.select))
print("S.sum ", float(np.asarray(res.S).sum()))
```

```
jaxgsa.hdmr.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=2000 runs, T=1 x K=1 output slice
    invalid: none found in 2000 rows (policy 'raise')
  timing:
    fit + estimator (includes compile on the first call): 1.559 s
    maxorder: 2
    slice_chunk_size: auto (resolved from the memory budget)
    batch_size: auto (resolved from the memory budget)
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.545
    2. x2  ST=0.3904
    3. x3  ST=0.25

terms  ('x1', 'x2', 'x3', 'x1/x2', 'x1/x3', 'x2/x3')
Sa     [0.3016 0.3772 0.0018 0.0057 0.2304 0.004 ]
Sb     [-0.0072 -0.0026 -0.0002  0.0033  0.0112  0.0028]
select [1. 1. 0. 0. 1. 0.]
S.sum  0.9279446005821228
```

The per-term layout is what makes this method different from PCE. `x1/x3` carries
`Sa = 0.230` while `x3` alone carries 0.002, so the fit puts almost all of `x3`'s
influence in its interaction with `x1`. That is correct for Ishigami. The inputs
are independent here, so every `Sb` sits near zero, as it should.

`select` is the F-test significance count per term. `x3`, `x1/x2` and `x2/x3`
scored 0 out of 1 output slice, so the F-test did not keep them.

`S.sum() = 0.928` is the precondition check, not a decoration. Li et al. attach
their totals to the condition that the per-term `S` values sum to about 1; the
shortfall is unexplained variance. Read it before ranking anything.

`verbose=True` is the default and printed the block. Pass `verbose=False` for a
silent run.

## Arguments

| Argument | Default | What it changes |
| --- | --- | --- |
| `maxorder` | `2` | Largest interaction size modelled, 1, 2 or 3. Order 2 captures pairwise interactions. Order 3 adds triples, and both the term count and the fit cost grow combinatorially. Clamped to `D` with a warning when `D < maxorder`. |
| `maxiter` | `100` | Backfitting iterations for the first-order terms. It rarely needs raising, because iteration stops early once the coefficients stop moving. |
| `m` | `2` | B-spline intervals per dimension, giving a basis of size `m + 3`. A larger `m` resolves sharper features in the component functions. It also multiplies the coefficient count, since a per-term basis grows as `(m+3)^order`, and it needs more samples to avoid overfitting. |
| `lambdax` | `0.01` | Tikhonov regularization strength. Raise it for noisy `Y` or small `N`, which smooths and stabilises the components. Lower it when genuine sharp features are being flattened. |
| `slice_chunk_size` | `None` | Maximum `(T, K)` output slices fitted per vmap. It bounds the per-slice statistics transient, about `batch_size * slice_chunk_size * n_terms`. `None` derives it from the memory budget. |
| `batch_size` | `None` | Rows per batch during the fit. It bounds the B-spline basis tensors, roughly `batch_size * (m+3)^2 * n2` floats at order 2 and `batch_size * (m+3)^3 * n3` at order 3, where `n2` and `n3` count parameter pairs and triples. `None` derives it from the memory budget, which resolves to a single batch whenever the whole fit fits. |
| `n_bootstrap` | `0` | Row resamples on `Sa`, `Sb`, `S` and `ST`. See below. |
| `key` | `None` | A `jax.random` key for the resampling. Required when `n_bootstrap > 0`. |
| `on_invalid` | `"raise"` | Policy for non-finite rows. One row is one unit, so `"drop"` removes the `(X, Y)` pair and fits on the rest. |
| `verbose` | `True` | Prints the summary block shown above. |
| `keep_replicates` | `False` | Retains the per-replicate index arrays on `result.ci`. |

The two size arguments compose: there is one fit path and it honours both. Both
default to the whole axis when the memory budget allows it.

## The two fit paths

Row batching is exact. The fit accumulates the same Gram matrices and moments,
solves the same regressions, runs the same F-test and reports the same indices
either way. The paths differ only in the order of the float32 sums and in peak
memory.

Streaming engages when `batch_size` is an integer below `N`, or when a single
batch would exceed the memory budget (`jaxgsa.config.set_memory_budget`).
`result.streamed` reports which path ran. Read it when a fit takes much longer
than you expect: `True` means the budget engaged.

See [`jaxgsa.pce`](/api/pce) for the equivalent memory estimate there; the two
methods size their single-pass thresholds independently.

## Confidence intervals

Expensive, like PCE's. Every replicate refits the whole expansion: fresh
B-spline bases, a fresh backfitting solve, a fresh F-test. Twenty replicates
cost about twenty fits, an order of magnitude above the row resample of
[`jaxgsa.pawn`](/api/pawn). Hence the `0` default.

The interval measures the sampling variability of `(X, Y)` propagated through
the fit. It does not measure truncation error: every replicate uses the same
basis and the same maximum order, so a systematic misfit sits inside every
interval. `rmse` and `S.sum()` are what report that.

## HDMRResult

Per-term arrays carry a trailing `n_terms` axis; per-parameter arrays carry a
trailing `D` axis.

| Field | Shape | Meaning |
| --- | --- | --- |
| `Sa` | `(..., n_terms)` | Structural variance fraction per term, the part independent of other inputs. |
| `Sb` | `(..., n_terms)` | Correlative contribution per term. Near zero under independence. A non-zero value flags variance shared through input correlation, and it can be negative. |
| `S` | `(..., n_terms)` | `Sa + Sb`, exactly. Measured against the fitted expansion (Li et al. 2010), not against `Y` itself: this is why the identity holds exactly rather than approximately, and it is what `S.sum()` reads for the Eq. (24) reliability check. SALib's `ancova` instead measures against `Y`, which differs by a few percent under correlated inputs. |
| `ST` | `(..., D)` | SCSA total per parameter. See the warning below. |
| `terms` | | Human-readable term labels, e.g. `("x1", "x2", "x1/x2")`. Interaction terms join names with `/`. |
| `select` | `(n_terms,)` | F-test significance count per term, summed over the `T*K` output slices, so the maximum is `T*K`. A low count marks a term the F-test deems insignificant. |
| `rmse` | `()` / `(K,)` / `(T, K)` | Emulator fit RMSE per output slice, in the units of `Y`. |
| `streamed` | | Which fit path ran. |
| `Sa_conf`, `Sb_conf`, `S_conf`, `ST_conf` | leading `(2, ...)` | `None` when `n_bootstrap=0`. |
| `problem`, `invalid`, `ci` | | Problem, non-finite report, interval provenance. |

Three properties reshape the structural (`Sa`) blocks into the conventional
Sobol layouts: `S1` as a `(D,)` vector, `S2` as a symmetric `(D, D)` matrix, and
`S3` as a `(D, D, D)` tensor. They are views on `Sa`, not separate estimates.

Operations:

- `result.predict(X_new, batch_size=None)` evaluates the fitted surrogate.
- `result.shapley(include_correlative=False)` derives Shapley effects. See
  [Shapley effects](/api/shapley).
- `result.to_dataset(time_coords=None)` gives the labeled xarray view.

## ST under correlated inputs

Correlated inputs are supported and a declared `problem.correlation` is welcome.
`Sb` doubles as the correlation diagnostic, and
`result.shapley(include_correlative=True)` folds it into the Shapley allocation.

`ST` is the SCSA total: `ST_i` sums `Sa_u + Sb_u` over every term `u` containing
parameter `i`. That is the convention of Li et al. (2010) Section 2.2.3,
restated as Eq. (8) by Sarazin, Viaud & Cournede (2017), and the same one SALib
and Vrugt's `HDMR_end.m` use. With independent inputs the `Sb` shares vanish and
it reduces to the ordinary Sobol total-order index.

With correlated inputs it does not. It can be negative. It is not bounded in
`[0, 1]`. It does not measure the expected reduction of output variance from
fixing a parameter, so **do not use it as a fixing criterion**: the bias runs
toward "cannot be fixed", and a parameter the model ignores can outrank one with
a negative value. Li et al. reuse the symbol `S_Ti` for two different
quantities, this term-membership sum and the classical conditional-variance
total of their Eq. (4); only the first is computed here.

`ST` is also not comparable with the `ST` of
[`jaxgsa.kucherenko`](/api/kucherenko) or the `S_TU` of
[`jaxgsa.vkoga`](/api/vkoga). Use one of those when you need a genuine
conditional-variance total under dependence.

The `S1` property carries the matching caveat: it is the structural share only,
so under correlation it sits below the Sobol `S1` of the same parameter by
whatever `Sb` holds. The per-term `Sa`, `Sb` and `S` fields keep their ANCOVA
meaning throughout. A correlated problem emits one `JaxgsaWarning` saying all of
this.

## What it refuses

`ValueError` when `X` or `Y` violates the shape contract, when `N < 300`, when
`maxorder` is not 1, 2 or 3, when an explicit `slice_chunk_size < 1` or
`batch_size < 1` is passed, when `on_invalid` is unknown or refuses the sample,
when `n_bootstrap > 0` without a `key`, and when the problem has any categorical
parameter. The B-spline component functions need an orderable axis, which an
unordered level code does not give. Use
[`jaxgsa.optimal_transport`](/api/optimal-transport),
[`jaxgsa.borgonovo`](/api/borgonovo), or [`jaxgsa.pawn`](/api/pawn) instead.

A 2-D `Y` is read as `(N, K)` unless `problem.output_names` has exactly one
entry, in which case the columns are `T` timepoints of that single output.

## Traceable core

`jaxgsa.hdmr.indices(problem, X, Y, *, maxorder=2, maxiter=100, m=2,
lambdax=0.01, slice_chunk_size=None, batch_size=None)` returns
`(Sa, Sb, S, ST)` as bare arrays with none of the checks, so it composes with
`jit`, `vmap` and `jacrev`. The first three carry the `n_terms` axis, `ST`
carries `D`, and there is no `terms` tuple to label them with.

See the [HDMR example](/examples/hdmr) and the [API overview](/api/).
