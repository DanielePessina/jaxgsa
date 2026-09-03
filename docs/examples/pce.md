# PCE (polynomial chaos expansion)

This page fits a polynomial surrogate to model runs you already have, then
reads Sobol indices off the fitted coefficients. You finish with S1, ST and S2
for the Ishigami test function, a cheap stand-in model you can call at new
input points, and the one number that decides whether any of it is worth
reporting.

A surrogate is a cheap function fitted to the model's inputs and outputs and
used in its place. PCE builds that surrogate from orthogonal polynomials
(Sudret, 2008). Orthogonality is the whole trick. Once the basis is
orthonormal, each squared coefficient *is* a partial variance, so the Sobol
indices fall out of the fit by arithmetic. No Saltelli design. No extra model
runs. Sobol indices split the output variance into shares. S1 is the share an
input explains alone, ST the share it explains alone or in any interaction,
and S2 the share owned by one pair acting together.

That arithmetic is exact for the polynomial. It says nothing about the model.
If the polynomial misses the model, the indices describe the polynomial and
you will publish them anyway unless you look at `loo_rmse`. The contrast
further down is the most useful thing on this page.

Reach for PCE when you already have `(X, Y)` pairs and cannot re-run the
model, when you want S1, ST and S2 out of a single fit, or when you need an
emulator for prediction. It pays off best on smooth responses, because that is
where a polynomial of modest degree gets close.

## Import style

The PCE module lives at `jaxgsa.pce`:

```python
from jaxgsa import pce
# pce.analyze(...)
```

## Basic example (Ishigami)

The Ishigami function is a three-input benchmark that ships with jaxgsa. Its
response contains $\sin(x_1)$ and $\sin^2(x_2)$ over $[-\pi, \pi]$, so a
polynomial has to reach degree 8 or so before it tracks the curvature. That
choice of `order=8` is deliberate and the next section shows what happens
without it.

Four steps. Draw 2000 uniform samples inside the bounds, because a
least-squares fit only needs points that cover the input space and any design
will do. Run the model on them once. Fit the surrogate. Read the indices and
`loo_rmse` together, never apart.

```python
import jax
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

# Any design works. PCE does not need Saltelli structure.
key = jax.random.PRNGKey(42)
bounds = jnp.array(PROBLEM.bounds)
X = jax.random.uniform(
    key, (2000, PROBLEM.num_vars),
    minval=bounds[:, 0], maxval=bounds[:, 1],
)
Y = evaluate(X)

result = jaxgsa.pce.analyze(PROBLEM, X, Y, order=8)

print("S1:", result.S1)
print("ST:", result.ST)
print("S2:", result.S2)
print("order:", result.order)
print("LOO RMSE:", result.loo_rmse)
print("std(Y):", jnp.std(Y))
```

```text
jaxgsa.pce.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=2000 runs, T=1 x K=1 output slice
    invalid: none found in 2000 rows (policy 'raise')
  timing:
    fit + estimator (includes compile on the first call): 1.297 s
    order: 8
    fit: single-pass
    batch_size: auto (resolved from the memory budget)
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.5571
    2. x2  ST=0.4429
    3. x3  ST=0.2432
S1: [3.1396714e-01 4.4286707e-01 1.3369902e-06]
ST: [0.557123   0.44291067 0.2431572 ]
S2: [[          nan 8.7611215e-06 2.4312086e-01]
 [8.7611215e-06           nan 8.7074231e-06]
 [2.4312086e-01 8.7074231e-06           nan]]
order: 8
LOO RMSE: 0.07379977
std(Y): 3.7769964
```

The block above `S1:` is the verbose summary. `analyze` prints it by default
in 1.0. It restates the problem it was given, so a wrong `bounds` array or a
misread output shape shows up before you read a single index. Pass
`verbose=False` to silence it.

The arrays. Ishigami's analytical answer is `S1 = [0.3139, 0.4424, 0]` and
`ST = [0.5576, 0.4424, 0.2437]`, and the fit lands on all six to three
decimals. x3's S1 is 1.3e-06, which is zero as far as float32 is concerned.
x3 still has `ST = 0.243`, and `S2[0, 2] = 0.2431` says where that came from:
x3 acts only through its interaction with x1. The S2 diagonal is NaN on
purpose, because an input has no pairwise interaction with itself, and NaN
refuses to be summed by accident where a 0 would slip through.

Now the number that matters. `loo_rmse` is 0.0738 against a `std(Y)` of 3.777,
so the surrogate's out-of-sample error is 2% of the output spread. That is a
faithful stand-in.

## The fit decides everything

`loo_rmse` is a leave-one-out cross-validation error. Each sample is scored by
a fit that excludes it, which measures the surrogate on data it did not see.
jaxgsa gets it from the hat matrix rather than by refitting N times, so it is
free.

Read it as a fraction of `std(Y)`. My working rule: under 5% and the indices
are reportable, 5% to 20% and treat rankings as provisional, over 20% and the
numbers describe a polynomial nobody cares about. Here is that last case. Same
data, same call, `order=3` instead of 8:

```python
low = jaxgsa.pce.analyze(PROBLEM, X, Y, order=3)
print("S1:", low.S1)
print("ST:", low.ST)
print("LOO RMSE:", low.loo_rmse)
```

```text
jaxgsa.pce.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=2000 runs, T=1 x K=1 output slice
    invalid: none found in 2000 rows (policy 'raise')
  timing:
    fit + estimator (includes compile on the first call): 1.434 s
    order: 3
    fit: single-pass
    batch_size: auto (resolved from the memory budget)
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.9375
    2. x3  ST=0.3023
    3. x2  ST=0.06529
S1: [0.63639057 0.05677766 0.0026459 ]
ST: [0.9375337  0.06528562 0.3023086 ]
LOO RMSE: 2.7552462
```

Nothing here errors. Nothing warns. The indices are finite, ordered, and
wrong.

x2's true S1 is 0.4424 and the degree-3 fit reports 0.0568, low by a factor of
eight. The ranking inverts. x2 falls from second to last and x3 climbs past
it. Present this table to a modelling team and they will stop measuring x2.
The only signal that anything is amiss is `loo_rmse = 2.755` against a
`std(Y)` of 3.777, which is 73% of the output spread. A degree-3 polynomial
cannot bend the way $\sin^2(x_2)$ bends over $[-\pi, \pi]$, so it fits x2's
contribution as noise and hands the leftover variance to whoever will take it.

This is the failure mode of every surrogate method, and it is quiet. Check
`loo_rmse` first, then read the indices. If it is high, raise `order` and refit
until it stops falling.

`order` is not free. The term count is $\binom{D + \text{order}}{\text{order}}$,
so degree 8 in 3 inputs is 165 terms and degree 8 in 10 inputs is 43758.
`analyze` will not let the term count pass `fit_ratio * N`, and it reduces
`order` with a warning rather than fitting an underdetermined system:

```text
JaxgsaWarning: jaxgsa.pce: PCE order reduced from 8 to 4 to keep the term
count within the sample budget (fit_ratio=0.5, N=100)
```

That was the same problem with only the first 100 rows. The resulting
`loo_rmse` is 3.627 against that subset's `std(Y)` of 3.934, so 92%. Do not
treat a reduction warning as advice. Treat it
as a demand for more samples. Always compare `result.order` against what you
asked for.

## Batching

`batch_size` sizes row blocks during the fit. It never changes the algorithm.
Both paths solve the same normal equations and compute the same exact
leave-one-out error, so they differ only in float32 summation order.

Leave it at `None` and the memory budget decides. jaxgsa estimates the
resident cost of the design matrix, the Gram factorization and the LOO
residuals; under the budget it runs a single pass, over it streams over
auto-sized blocks. The budget defaults to about 512 MiB and moves with
`jaxgsa.config.set_memory_budget`.

An explicit value always wins over the budget. `batch_size=250` on the N=2000
fit above streams:

```text
  timing:
    fit + estimator (includes compile on the first call): 1.216 s
    order: 8
    fit: streamed
    batch_size: 250 (user-set)
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.5571
    2. x2  ST=0.4429
    3. x3  ST=0.2432
```

`batch_size=2000` is one full block, which is the single-pass fit:

```text
  timing:
    fit + estimator (includes compile on the first call): 1.32 s
    order: 8
    fit: single-pass
    batch_size: 2000 (user-set)
```

`batch_size` is clamped to N, so any value at or above N means one block. The
streamed `loo_rmse` is 0.07380029 against the single-pass 0.07379977. That
gap, 7e-7, is float32 reassociation and nothing more. The `fit:` and
`batch_size:` lines in the verbose block are how you confirm which path ran.

## Expansion details

The fitted expansion stays on the result:

```python
print("coefficients:", result.coefficients.shape)
print("multi_index:", result.multi_index.shape)
print("order:", result.order)
print("loo_rmse:", result.loo_rmse)
```

```text
coefficients: (165,)
multi_index: (165, 3)
order: 8
loo_rmse: 0.07379977
```

165 is the term count for degree 8 in 3 inputs. Each term has one coefficient
and one row of per-input degrees, which is why both shapes share it. Row `t`
of `multi_index` gives the polynomial degree of term `t` in each dimension, so
`[2, 0, 0]` is a pure x1 term and `[1, 0, 1]` is an x1-x3 interaction. Squaring
`coefficients[1:]` and grouping by which columns of `multi_index[1:]` are
non-zero is exactly how the indices above were computed.

The basis follows the marginal. Uniform and truncated-Gaussian inputs get
Legendre polynomials, untruncated Gaussian inputs get Hermite. You do not
choose it.

## Emulation

`predict()` evaluates the fitted polynomial at new inputs. This is the
emulator, and it costs a polynomial evaluation instead of a model run:

```python
X_new = jax.random.uniform(
    jax.random.PRNGKey(99), (100, PROBLEM.num_vars),
    minval=bounds[:, 0], maxval=bounds[:, 1],
)
Y_pred = result.predict(X_new)
print("Y_pred shape:", Y_pred.shape)

Y_true = evaluate(X_new)
print("Max error:", jnp.max(jnp.abs(Y_pred - Y_true)))
```

```text
Y_pred shape: (100,)
Max error: 0.1725297
```

100 predictions for 100 new rows. The 0.17 is a worst case over those rows,
not an average, which is why it sits above `loo_rmse = 0.0738`. Both are small
against `std(Y) = 3.777`. A max error many times `loo_rmse` usually means the
new points reach into a corner of the input space your training design left
thin.

## xarray export

`to_dataset()` returns a labeled `xarray.Dataset`, so you select an input by
name instead of by position:

```python
ds = result.to_dataset()
print(ds)
print(ds.S1.sel(param="x1").values)
```

```text
<xarray.Dataset> Size: 140B
Dimensions:             (param: 3, param_i: 3, param_j: 3)
Coordinates:
  * param               (param) <U2 24B 'x1' 'x2' 'x3'
  * param_i             (param_i) <U2 24B 'x1' 'x2' 'x3'
  * param_j             (param_j) <U2 24B 'x1' 'x2' 'x3'
Data variables:
    S1                  (param) float32 12B 0.314 0.4429 1.337e-06
    ST                  (param) float32 12B 0.5571 0.4429 0.2432
    S2                  (param_i, param_j) float32 36B nan 8.761e-06 ... nan
    loo_rmse            float32 4B 0.0738
    explained_variance  float32 4B 0.9997
Attributes:
    order:     8
    streamed:  False
0.31396714
```

S1 and ST carry one `param` dimension. S2 is a matrix, so it needs `param_i`
for rows and `param_j` for columns. Both diagnostics travel with the indices:
`loo_rmse` and `explained_variance`, the fraction of the sample `Var(Y)` the
fitted expansion reproduces, which is 0.9997 here. That one is the in-sample
R^2, so it cannot pass 1; a fit that memorises the sample reads near 1 while
`loo_rmse` climbs, which is why the two travel together. The
`streamed: False` attribute records which fit path produced the numbers.

## Limitations

`(N, K)` and `(N, T, K)` outputs work, but all slices share one basis and one
effective `order` because they are fitted in a single multi-right-hand-side
solve. A hard slice and an easy slice therefore get the same polynomial
budget.

Correlated inputs raise `ValueError`. PCE's orthogonality argument assumes
independent marginals, so a declared `problem.correlation` breaks the step
from coefficients to partial variances. The error message says so and names
the alternatives: [RS-HDMR](/examples/hdmr), [VKOGA](/examples/vkoga) and
[Kucherenko](/examples/kucherenko). This refusal is deliberate. Silently wrong
indices are worse than no indices.

Everything the polynomial cannot represent is missing from the indices, and
its absence is reported through `loo_rmse` alone.

## See also

- [Basic Example](/examples/basic) for the Sobol workflow when you can afford
  a structured Saltelli design.
- [RS-HDMR](/examples/hdmr) for the other given-data surrogate, which uses
  B-splines and handles correlated inputs.
- [Shapley Effects](/examples/shapley) for a fair allocation read off this
  same fit.
- [DGSM](/examples/dgsm) for derivative-based sensitivity bounds.
- [Methods](/guide/methods) for the theory behind PCE.
- [API Reference](/api/#given-data-methods) for full parameter documentation.
