# VKOGA (Correlated-Input Sensitivity)

VKOGA is the given-data variance-based method in jaxgsa that does **not** assume
independent inputs (its design-based counterpart is
[Kucherenko](/examples/kucherenko)). It is the two-stage surrogate-based
sensitivity analysis of
Hilhorst et al. (2024): fit a greedy kernel surrogate to whatever `(X, Y)` data
you have, then estimate the correlated variance-based indices of Li et al. (2010)
against it under a Gaussian copula. The nested conditional sampling those indices
need would be unaffordable against the real model; against a kernel expansion it
is cheap.

It returns five indices per parameter, of which two do most of the work:

- **`S_TC`** (total correlated) — what an input explains through itself *and*
  through its correlation with the others. The **input prioritisation** measure:
  "which parameter should I measure more accurately?"
- **`S_TU`** (total uncorrelated) — what only that input can explain, correlated
  pathways removed. The **input fixing** measure: "which parameter can I freeze?"

When to use VKOGA:

- Your inputs are genuinely correlated and you still want **variance fractions**,
  not a distributional distance.
- You need to tell "worth measuring" apart from "safe to fix" — under dependence
  these are different questions with different answers.
- You want to state the dependency structure explicitly, or sweep several.
- You have existing (X, Y) pairs and also want a fast surrogate
  (`result.predict`).

## Enable float64 first

The coefficient solve forms `A.T @ A`, squaring the condition number of the cross
kernel; float32 cannot carry that for small `gamma`. Turn on double precision
**before** creating any arrays:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

`jaxgsa.vkoga.analyze()` emits a `UserWarning` if you forget.

## Import style

The VKOGA module lives at `jaxgsa.vkoga`:

```python
import jaxgsa
# jaxgsa.vkoga.analyze(...)
```

As with the other given-data methods, `monte_carlo` is in `jaxgsa.sampling`, not
in `jaxgsa.vkoga`.

## Scalar example (correlated linear model)

A model with a known answer: `Y = 2 x1 + x2 + 0.5 x3` with standard-normal
marginals, analysed under a copula that correlates `x1` and `x2` at `rho = 0.6`
and leaves `x3` independent.

Note that the **training design is independent** even though the analysis is
correlated — see [Practical caveats](#practical-caveats) for why this matters.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import jaxgsa

problem = jaxgsa.Problem.from_dict(
    {
        "x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x2": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x3": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
    }
)

# Independent, space-filling training design — any (X, Y) pairs work
X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 2048, seed=0))
Y = X @ jnp.array([2.0, 1.0, 0.5])

# The dependency structure the indices are computed under. Declare it on the
# problem; analyze() reads problem.correlation by default. A (D, D) matrix
# passed as correlation= overrides the declaration for one call.
R = np.array(
    [
        [1.0, 0.6, 0.0],
        [0.6, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
problem_corr = problem.with_correlation(R)

result = jaxgsa.vkoga.analyze(problem_corr, X, Y)

print("S_TC:", np.round(result.S_TC, 3))  # [0.881 0.627 0.025]
print("S_TU:", np.round(result.S_TU, 3))  # [0.34  0.085 0.034]
print("S_U: ", np.round(result.S_U, 3))   # [0.336 0.083 0.033]
print("S_C: ", np.round(result.S_C, 3))   # [ 0.545  0.544 -0.008]
print("S_IU:", np.round(result.S_IU, 3))  # [0.004 0.002 0.001]
```

The default run cross-validates `gamma` and `ridge` over a 10x10 grid, which
dominates the runtime (tens of seconds here). Pass both explicitly to skip it —
they are reported on the result, so a first exploratory run tells you what to
pin.

## Reading the five indices

- **`x1` and `x2` both look important (`S_TC` 0.88 and 0.63) but neither is
  individually necessary (`S_TU` 0.34 and 0.09).** They share correlated
  variance: each explains a large slice of the output, but most of that slice is
  also reachable through the other. Measuring either one more accurately pays
  off; fixing either one while leaving the other free does not.
- **`S_C` is 0.545 for `x1` and 0.544 for `x2`, and effectively zero
  (-0.008) for `x3`.** That is the correlation-borne part, and it is symmetric
  here precisely because the correlation is what carries it. `x3` is
  uncorrelated, so it has none. `S_C` **can be negative** — that is not a bug,
  it is a correlation working against a direct effect (here it is just estimator
  noise around zero).
- **`S_IU` is ~0 for every parameter**, as it must be: the model is additive, so
  there are no independent interactions to find. `S_IU` is never negative: it is
  `S_TU - S_U`, and `S_U` is clipped to `S_TU` (see below).

## Three things that can go wrong

These are limits of the method, not bugs. Each one has a signal you can read.

### The surrogate can fail, and the ranking can invert

Every index is measured against the surrogate, never against your model. If the
surrogate misses the response, the indices describe the surrogate alone.

A greedy Gaussian kernel cannot resolve a high-frequency or oscillatory
response. On `sin(2*pi*12*u1) + 0.5*u2` with 2048 training points the reported
`S_TC` is `[0.18, 0.75, 0.00]` against a true `[0.96, 0.04, 0.00]` — **the
ranking is inverted**. A step function at the same size is fine.

`analyze` warns when the cross-validated error passes half the output standard
deviation, and it reports the score as `result.cv_rmse`. Check it:

```python
print(result.cv_rmse, np.std(Y))   # honest out-of-sample error vs output scale
```

`result.rmse` is the *training* error, so it is optimistic. Use `cv_rmse` to
judge the fit. `cv_rmse` is `None` when you pass both `gamma` and `ridge`,
because no cross-validation ran — pass at least one as `None` if you want the
diagnostic. When the surrogate cannot be improved by more training points, use
`jaxgsa.kucherenko` instead: it evaluates your actual model on a conditional
design and needs no surrogate.

### `S_U` uses an additive projection

`S_U` compares the output against fitted additive component functions `f_i`. No
additive function of `X_i` can represent an interaction, so on a model with
interactions under a correlated measure the raw `S_U` can exceed `S_TU`.

jaxgsa clips `S_U` to `S_TU`, which keeps `S_IU` non-negative, and warns when
the clip is wider than 1% of the output variance. Treat that warning as a
statement about the model: `S_TC` and `S_TU` are unaffected and stay reliable,
but read `S_U`, `S_C` and `S_IU` as indicative only. `S_C` is never clipped —
a negative `S_C` is a real reading, not an artefact.

### The reported variance runs slightly low

The surrogate works in CDF space, where the tails of a Gaussian marginal are
compressed into a small part of the unit cube. The kernel under-resolves them,
so `result.variance` is biased **low**. On a four-parameter Gaussian case it
reported 15.64 against a true 16.20, about -3.5%. The bias grows with heavier
tails. It affects the variance figure, not the index ratios, because every
index is divided by the same number.

## Ground-truth check

For a linear model under a Gaussian copula on Gaussian marginals both indices are
closed-form: with `a = (2, 1, 0.5)` and covariance `R`,
`S_TC_i = (R a)_i^2 / (R_ii Var Y)` and
`S_TU_i = a_i^2 Var(X_i | X_-i) / Var Y`.

```python
a = np.array([2.0, 1.0, 0.5])
var_y = a @ R @ a                      # 7.65
closed_S_TC = (R @ a) ** 2 / var_y
closed_S_TU = a**2 * np.array([1 - 0.6**2, 1 - 0.6**2, 1.0]) / var_y

print("S_TC estimated:", np.round(result.S_TC, 3))  # [0.881 0.627 0.025]
print("S_TC closed:   ", np.round(closed_S_TC, 3))  # [0.884 0.633 0.033]
print("S_TU estimated:", np.round(result.S_TU, 3))  # [0.34  0.085 0.034]
print("S_TU closed:   ", np.round(closed_S_TU, 3))  # [0.335 0.084 0.033]
```

The residual gap is surrogate error plus Monte-Carlo noise; raise `n_outer` /
`n_inner` / `n_variance` (they only touch the cheap surrogate) or the training
sample size to shrink it.

## Fitting the copula from the data

If your data is observational and already correlated, fit the copula from the
data with `jaxgsa.sampling.fit_correlation` and attach it to the problem. The
fit uses Spearman rank correlation, so it is invariant to the declared
marginals — a skewed parameter cannot distort the dependency structure. One
workflow, one explicit choice of *which* sample the copula comes from.

```python
rng = np.random.default_rng(0)
X_corr = jnp.asarray(rng.standard_normal((2048, 3)) @ np.linalg.cholesky(R).T)
Y_corr = X_corr @ jnp.array([2.0, 1.0, 0.5])

R_fit = jaxgsa.sampling.fit_correlation(problem, X_corr)
emp = jaxgsa.vkoga.analyze(problem.with_correlation(R_fit), X_corr, Y_corr)

print(np.round(emp.correlation, 3))
# [[1.    0.609 0.041]
#  [0.609 1.    0.003]
#  [0.041 0.003 1.   ]]
print("S_TC:", np.round(emp.S_TC, 3))  # [0.888 0.629 0.036]
```

The fitted matrix recovers `rho_12 = 0.6` to sampling accuracy. Read the
uncorrelated indices from a run like this with care — the surrogate was trained
only on the correlated ridge, and `S_TU` queries it off that ridge.

## Independent inputs collapse to S1 / ST

When neither the problem nor the call declares a correlation, the five indices
reduce to the familiar picture: `S_TC` is the first-order Sobol' index, `S_TU`
is the total index, and `S_C` vanishes.

```python
indep = jaxgsa.vkoga.analyze(problem, X, Y)  # problem declares no correlation

print("S_TC:", np.round(indep.S_TC, 3))  # [0.76  0.183 0.04 ]
print("S_TU:", np.round(indep.S_TU, 3))  # [0.77  0.192 0.049]
print("S_C: ", np.round(indep.S_C, 4))   # [-0.0015 -0.0064 -0.0074] — ~0
print("analytical S1 = ST:", np.round(a**2 / (a**2).sum(), 3))  # [0.762 0.19  0.048]
print("is_correlated:", indep.is_correlated)  # False
```

## The fitted surrogate

The result keeps the kernel expansion, so `predict` costs one kernel product
against a few hundred centres. Batching is automatic and bounded by the global
memory budget.

```python
X_new = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 1000, seed=1))
Y_pred = result.predict(X_new)

print("prediction shape:", Y_pred.shape)      # (1000,)
print("n_centers:", result.n_centers)         # 300
print("gamma:", round(result.gamma, 3))       # 7.533
print("ridge:", result.ridge)                 # ~7.7e-06
print("training rmse:", float(result.rmse))   # 0.1595
print("Var(Y) under the copula:", float(result.variance))  # 7.4615
```

`n_centers` is the greedy's stopping point, capped by `max_centers` (default
300). `rmse` is the fit on its own training rows — a diagnostic, not a
generalisation estimate; check held-out points yourself if it matters. `variance`
is the output variance under the **correlated** input measure, which is the
denominator of every index, and differs from `Y.var()` on an independent training
design.

## xarray export

`VKOGAResult.to_dataset()` converts results to a labeled `xarray.Dataset`,
including the copula matrix on its own pair of parameter dimensions.

```python
ds = result.to_dataset()
print(ds)
# <xarray.Dataset>
# Dimensions:      (param: 3, param_i: 3, param_j: 3)
# Data variables:  S_TC, S_TU, S_U, S_C, S_IU, variance, rmse, correlation
# Attributes:      method, n_centers, gamma, ridge, correlated

print(ds.S_TC.sel(param="x1"))
print(ds.correlation)
```

For time-series results, pass `time_coords` to label the time dimension.

## Multi-output

`Y` may be `(N,)`, `(N, K)`, or `(N, T, K)`; all output slices share one greedy
basis and one set of centres, so a multi-output fit costs barely more than a
scalar one.

```python
problem_multi = jaxgsa.Problem.from_dict(
    {
        "x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x2": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x3": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
    },
    output_names=("linear", "quadratic"),
)
Y_multi = jnp.column_stack([Y, jnp.sum(X**2, axis=1)])

multi = jaxgsa.vkoga.analyze(problem_multi, X, Y_multi, correlation=R)

print("S_TC shape:", multi.S_TC.shape)      # (K, D) = (2, 3)
print("variance shape:", multi.variance.shape)  # (K,) = (2,)
```

## No Shapley effects

`VKOGAResult.shapley()` raises `NotImplementedError` on purpose:

```python
try:
    result.shapley()
except NotImplementedError as exc:
    print(exc)
# VKOGAResult has no term-wise variance decomposition, so Shapley effects are
# undefined for it; use jaxgsa.hdmr or jaxgsa.pce instead
```

Shapley effects allocate variance across *parameter subsets*. A kernel expansion
is a sum over **centres**, and every centre involves every parameter, so there is
no membership matrix to allocate from. Use
[`jaxgsa.hdmr`](/examples/hdmr) — whose ANCOVA terms are labelled, and which
supports `shapley(include_correlative=True)` for dependent inputs — or
[`jaxgsa.pce`](/examples/pce).

## Shape rules

| `Y` shape | `S_TC` / `S_TU` / `S_U` / `S_C` / `S_IU` | `variance` / `rmse` |
|---|---|---|
| `(N,)` | `(D,)` | `()` |
| `(N, K)` | `(K, D)` | `(K,)` |
| `(N, T, K)` | `(T, K, D)` | `(T, K)` |

D is always the last axis of the index arrays. `correlation` is a property of the
input model, not of any output slice, so it stays `(D, D)` throughout.

## Practical caveats

- **Train on an independent, space-filling design, even when the analysis is
  correlated.** This is the easiest way to get wrong answers. A correlated
  sample concentrates on a ridge, but `S_TU` conditions on the other parameters
  and then resamples `X_i` across its whole marginal — exactly the off-ridge
  region a correlated training set never visited, so the surrogate extrapolates
  where the estimator queries it hardest.
- **float64 is strongly recommended.** The normal equations square the condition
  number of the cross kernel, which float32 cannot carry for small `gamma`.
  Cross validation partly self-corrects, since the scores are computed in the
  same arithmetic and penalise the blown-up corner of the grid, but the ceiling
  is real.
- **Cost is the hyperparameter search.** A 10x10 `gamma`/`ridge` grid, each point
  refitted `n_folds` times, is the bulk of the runtime. Pass `gamma=` and
  `ridge=` explicitly once you know good values; raising `n_outer`, `n_inner`, or
  `n_variance` is comparatively cheap because those only touch the surrogate.
- **At least two parameters.** `D = 1` raises `ValueError` — conditioning on
  "the other parameters" has no meaning.
- **`S_C` can be negative** when a correlation opposes a direct effect, and small
  negative values around zero are ordinary estimator noise for an uncorrelated
  parameter.
- The kernel is isotropic, so inputs are mapped to `[0, 1]` through their
  marginal CDFs before fitting — the same transform HDMR uses — and `predict`
  applies it too.

## See also

- [RS-HDMR](/examples/hdmr) for the other correlation-aware given-data method,
  which decomposes term by term (structural `Sa` vs correlative `Sb`) and can
  produce Shapley effects.
- [PCE](/examples/pce) for analytical Sobol indices when inputs are independent.
- [HSIC](/examples/hsic), [PAWN](/examples/pawn), and
  [Optimal Transport](/examples/optimal-transport) for correlation-tolerant
  measures that are not variance fractions.
- [Methods](/guide/methods#vkoga-correlated-input-variance-indices) for the
  theory, and for when to choose VKOGA over HDMR's ANCOVA split.
- [API Reference](/api/vkoga) for full parameter documentation.
