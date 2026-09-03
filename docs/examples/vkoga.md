# VKOGA (Correlated-Input Sensitivity)

This page turns a set of input/output pairs you already have into five
variance-based sensitivity indices per input, valid when the inputs are
correlated. It also leaves you with a fast surrogate model you can call on new
points.

Most variance-based methods assume the inputs vary independently of each
other. VKOGA is the given-data method in jaxgsa whose indices are defined
without that assumption. Given-data means it uses input/output pairs you
already have and asks for no new model runs. Its design-based counterpart,
which runs your model on a fresh design instead, is
[Kucherenko](/examples/kucherenko).

VKOGA is the two-stage surrogate-based sensitivity analysis of Hilhorst et
al. (2024). Stage one fits a greedy kernel surrogate to whatever `(X, Y)` data
you have. A greedy kernel surrogate is a sum of kernel bumps placed one at a
time at the training points that reduce the error most. Stage two estimates
the correlated variance-based indices of Li et al. (2010) against that
surrogate, under a Gaussian copula. A Gaussian copula describes the dependence
between the inputs separately from each input's own distribution. Those
indices need nested conditional sampling, which would be unaffordable against
the real model. Against a kernel expansion it is cheap.

## The five indices

VKOGA returns five indices per parameter. Two of them do most of the work:

- `S_TC` (total correlated) is what an input explains through itself, plus what
  it explains through its correlation with the others. This is the input
  prioritisation measure. It answers "which parameter should I measure more
  accurately?"
- `S_TU` (total uncorrelated) is what only that input can explain, with the
  correlated pathways removed. This is the input fixing measure. It answers
  "which parameter can I freeze?"

The distinction matters because under dependence the two questions have
different answers. An input can be worth measuring and still be safe to fix,
if a correlated partner supplies the same information.

The remaining three split those totals further: `S_U` is the uncorrelated
first-order part, `S_C` the correlated part, and `S_IU` the uncorrelated
interaction part.

Use VKOGA when:

- Your inputs are genuinely correlated and you still want variance fractions,
  not a distributional distance.
- You need to tell "worth measuring" apart from "safe to fix".
- You want to state the dependency structure explicitly, or sweep several.
- You have existing `(X, Y)` pairs and also want a fast surrogate
  (`result.predict`).

## Enable float64 first

The coefficient solve forms `A.T @ A`, squaring the condition number of the
cross kernel. float32 cannot carry that for small `gamma`. Turn on double
precision before you create any arrays:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

`jaxgsa.vkoga.analyze()` emits a `JaxgsaWarning` if you forget.

## Scalar example (correlated linear model)

The example model is `Y = 2 x1 + x2 + 0.5 x3` with standard-normal marginals.
It is analysed under a copula that correlates `x1` and `x2` at `rho = 0.6` and
leaves `x3` independent. A linear model is used here because its indices have
a known closed form, so the estimates can be checked.

The steps:

1. Declare the marginal distribution of each input with `Problem.from_dict`.
2. Build the training design with `sampling.monte_carlo` and evaluate the
   model on it. The training design is independent even though the analysis is
   correlated. See
   [the training design must be independent](#the-training-design-must-be-independent)
   for why.
3. Declare the dependency structure with `with_correlation`. `analyze` reads
   `problem.correlation` by default. Passing a `(D, D)` matrix as
   `correlation=` overrides the declaration for one call.
4. Call `vkoga.analyze` with the problem, the inputs, the outputs, and a JAX
   `key`. The key is required: the index estimate is Monte-Carlo against the
   surrogate, so there is randomness to seed.

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

# Independent, space-filling training design. Any (X, Y) pairs work.
X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 2048, seed=0))
Y = X @ jnp.array([2.0, 1.0, 0.5])

# The dependency structure the indices are computed under.
R = np.array(
    [
        [1.0, 0.6, 0.0],
        [0.6, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
problem_corr = problem.with_correlation(R)

result = jaxgsa.vkoga.analyze(problem_corr, X, Y, key=jax.random.key(0))

print("S_TC:", np.round(result.S_TC, 3))
print("S_TU:", np.round(result.S_TU, 3))
print("S_U: ", np.round(result.S_U, 3))
print("S_C: ", np.round(result.S_C, 3))
print("S_IU:", np.round(result.S_IU, 3))
```

```
jaxgsa.vkoga.analyze
  problem: D=3 (x1, x2, x3)
    marginals: gaussian=3
    correlation: correlated (Gaussian copula)
    output: N=2048 runs, T=1 x K=1 output slice
    invalid: none found in 2048 rows (policy 'raise')
  timing:
    fit + compute: 5.96 s
    n_centers: 300
    gamma: 7.533
    ridge: 5.995e-05
    batch_size: auto (resolved from the memory budget)
  results: top 3 of 3 parameters by S_TC
    1. x1  S_TC=0.8824
    2. x2  S_TC=0.6307
    3. x3  S_TC=0.03245
S_TC: [0.882 0.631 0.032]
S_TU: [0.341 0.085 0.034]
S_U:  [0.337 0.083 0.033]
S_C:  [ 0.546  0.548 -0.   ]
S_IU: [0.004 0.002 0.001]
```

Almost all of that time is the hyperparameter search. The default run
cross-validates `gamma` and `ridge` over a 10x10 grid, refitting each point
`n_folds` times. Cross-validation means the fit is scored on points it was not
trained on. Both chosen values are reported in the summary block, so a first
exploratory run tells you what to pin: pass `gamma=7.533, ridge=5.995e-05` on
the next call and the search disappears.

## Reading the five indices

- `x1` and `x2` both look important (`S_TC` 0.88 and 0.63), but neither is
  individually necessary (`S_TU` 0.34 and 0.08). They share correlated
  variance. Each explains a large slice of the output, but most of that slice
  is also reachable through the other. So measuring either one more accurately
  pays off, while fixing either one and leaving the other free does not.
- `S_C` is 0.546 for `x1` and 0.548 for `x2`, and 0.000 for `x3`. That is the
  correlation-borne part. It is symmetric here because the correlation is what
  carries it. `x3` is uncorrelated, so it has none. A negative `S_C` is a valid
  reading, not a bug: it is a correlation working against a direct effect.
- `S_IU` is at most 0.004, near zero for every parameter, as it must be. The
  model is additive, so there are no independent interactions to find. `S_IU`
  is never negative, because it is `S_TU - S_U` and `S_U` is clipped to `S_TU`
  (see below).

Against the closed form for a linear model under a Gaussian copula on Gaussian
marginals, with `a = (2, 1, 0.5)` and covariance `R`:

$$S^{TC}_i = \frac{(R a)_i^2}{R_{ii}\,V(Y)}, \qquad
S^{TU}_i = \frac{a_i^2\,V(X_i \mid X_{\sim i})}{V(Y)}$$

```python
a = np.array([2.0, 1.0, 0.5])
var_y = a @ R @ a  # 7.65
print("S_TC closed:", np.round((R @ a) ** 2 / var_y, 3))
print("S_TU closed:", np.round(a**2 * np.array([0.64, 0.64, 1.0]) / var_y, 3))
```

```
S_TC closed: [0.884 0.633 0.033]
S_TU closed: [0.335 0.084 0.033]
```

Every estimate sits within 0.008 of its closed form and the parameter ordering
matches. The residual gap is surrogate error plus Monte-Carlo noise. Raise
`n_outer`, `n_inner` or `n_variance`, which only touch the cheap surrogate, or
raise the training sample size, to shrink it.

## The training design must be independent

This is the mistake that costs the most, and the reason is worth spelling out.

`S_TU` conditions on the other parameters and then resamples `X_i` across its
whole marginal. Those resampled points are, by construction, off the ridge that
a correlated sample concentrates on. If the surrogate only ever saw correlated
data, it has no information there and is extrapolating at exactly the points
the estimator leans on hardest. Every index then inherits that extrapolation
error.

`analyze` checks the training design and warns:

```python
ridge_problem = jaxgsa.Problem.from_dict(
    {
        "x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x2": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
    }
).with_correlation([[1.0, 0.99], [0.99, 1.0]])
X_ridge = jnp.asarray(jaxgsa.sampling.monte_carlo(ridge_problem, 4096, seed=0))
Y_ridge = (X_ridge[:, 0] - X_ridge[:, 1]) ** 2

bad = jaxgsa.vkoga.analyze(
    ridge_problem, X_ridge, Y_ridge, key=jax.random.key(0), verbose=False
)
print("cv_rmse:", round(float(bad.cv_rmse), 4), " std(Y):", round(float(np.std(Y_ridge)), 4))
print("S_TU:", np.round(bad.S_TU, 3))
```

```
JaxgsaWarning: jaxgsa.vkoga: the training X is itself correlated (fitted latent
correlation between 'x1' and 'x2' is 0.99). The estimator needs an independent,
space-filling training design even for a correlated analysis: S_TU conditions on
the other parameters and then resamples each X_i across its whole marginal, so a
surrogate trained only on correlated data extrapolates for exactly those
conditional draws. Train on an independent design; the dependence structure
belongs in problem.correlation (or the correlation= argument), not in the
training data.

cv_rmse: 0.007  std(Y): 0.0276
S_TU: [1.008 1.001]
```

`S_TU` is a variance fraction, so 1.008 and 1.001 are impossible. That is the
useful part of this example: the failure is loud only because the true answer
happens to sit near the boundary. Note what did *not* catch it. `cv_rmse` is
0.007, well inside the surrogate warning threshold, because the held-out folds
come from the same correlated ridge as the training rows. Cross-validation
scores the surrogate where the data is, and the problem is where the data is
not.

Kucherenko, which runs the model rather than a surrogate, gives the reference:

```python
ks = jaxgsa.kucherenko.sample(ridge_problem, 65536, seed=0, verbose=False)
kr = jaxgsa.kucherenko.analyze(ks, (ks.samples[:, 0] - ks.samples[:, 1]) ** 2, verbose=False)
print("S1:", np.round(kr.S1, 3), " ST:", np.round(kr.ST, 3))
```

```
S1: [0. 0.]  ST: [0.999 1.   ]
```

The fix is to train on an independent design and put the dependence in the
problem, not in the data:

```python
X_indep = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 2048, seed=0))
Y_indep = X_indep @ jnp.array([2.0, 1.0, 0.5])
good = jaxgsa.vkoga.analyze(problem_corr, X_indep, Y_indep, key=jax.random.key(0), verbose=False)
print("S_TU:", np.round(good.S_TU, 3))
```

```
S_TU: [0.341 0.085 0.034]
```

That is the scalar example from the top of this page, and it matches the closed
form. If your data is observational and you cannot re-run the model, VKOGA is
being asked a question it cannot answer from that sample. Use
[Kucherenko](/examples/kucherenko) instead, or read only `S_TC`, which is
evaluated on the correlated measure the training data already covers.

## Two more things that can go wrong

These are limits of the method, not bugs. Each one has a signal you can read.

### The surrogate can fail, and the ranking can invert

Every index is measured against the surrogate, never against your model. If
the surrogate misses the response, the indices describe the surrogate alone.

A greedy Gaussian kernel cannot resolve a high-frequency response:

```python
osc = jaxgsa.Problem.from_dict({"u1": (0.0, 1.0), "u2": (0.0, 1.0), "u3": (0.0, 1.0)})
Xo = jnp.asarray(jaxgsa.sampling.monte_carlo(osc, 2048, seed=0))
Yo = jnp.sin(2 * jnp.pi * 12 * Xo[:, 0]) + 0.5 * Xo[:, 1]

r = jaxgsa.vkoga.analyze(osc, Xo, Yo, key=jax.random.key(0), verbose=False)
print("S_TC:", np.round(r.S_TC, 3))
print("cv_rmse:", round(float(r.cv_rmse), 3), " std(Y):", round(float(np.std(Yo)), 3))
```

```
JaxgsaWarning: jaxgsa.vkoga: the cross-validated surrogate error is 0.97 of the
output standard deviation, so the surrogate misses most of the output variation.
Every index is computed against the surrogate, so the reported values — including
the ranking — are not trustworthy. This happens on high-frequency or oscillatory
responses, which a greedy Gaussian kernel cannot resolve. Add training points, or
use a method that does not need a surrogate (jaxgsa.kucherenko on a conditional
design).

S_TC: [0.189 0.712 0.037]
cv_rmse: 0.703  std(Y): 0.725
```

True `S1` here is about `[0.96, 0.04, 0.00]`: `u1` carries almost everything.
VKOGA reports `u2` as the leader. The ranking is inverted, not merely
imprecise. The 12-cycle sine is beyond the kernel, so the fit keeps the one
part it can represent, the ramp in `u2`, and calls that the model. A step
function at the same training size is fine, because a step is not
high-frequency; it is one sharp feature.

`result.rmse` is the training error, so it is optimistic. Use `cv_rmse` to
judge the fit. `cv_rmse` is `None` when you pass both `gamma` and `ridge`,
because no cross-validation ran. Pass at least one as `None` if you want the
diagnostic. When more training points cannot improve the surrogate, use
`jaxgsa.kucherenko` instead.

### `S_U` uses an additive projection

`S_U` compares the output against fitted additive component functions `f_i`.
No additive function of `X_i` can represent an interaction. So on a model with
interactions under a correlated measure the raw `S_U` can exceed `S_TU`.

jaxgsa clips `S_U` to `S_TU`, which keeps `S_IU` non-negative. It warns when
the clip is wider than 1% of the output variance. Treat that warning as a
statement about the model. `S_TC` and `S_TU` are unaffected and stay reliable,
but read `S_U`, `S_C` and `S_IU` as indicative only. `S_C` is never clipped, so
a negative `S_C` is a real reading, not an artefact.

### The reported variance runs slightly low

The surrogate works in CDF space, where the tails of a Gaussian marginal are
compressed into a small part of the unit cube. The kernel under-resolves them,
so `result.variance` is biased low. In the scalar example above it reports
7.462 against the closed-form 7.65, about -2.5%. The bias grows with heavier
tails. It affects the variance figure, not the index ratios, because every
index is divided by the same number.

## Fitting the copula from the data

If your data is observational and already correlated, fit the copula from the
data with `jaxgsa.sampling.fit_correlation` and attach it to the problem. The
fit uses Spearman rank correlation, which compares ranks rather than values.
It is therefore invariant to the declared marginals, so a skewed parameter
cannot distort the dependency structure.

```python
rng = np.random.default_rng(0)
X_corr = jnp.asarray(rng.standard_normal((2048, 3)) @ np.linalg.cholesky(R).T)
Y_corr = X_corr @ jnp.array([2.0, 1.0, 0.5])

R_fit = jaxgsa.sampling.fit_correlation(problem, X_corr)
emp = jaxgsa.vkoga.analyze(
    problem.with_correlation(R_fit), X_corr, Y_corr, key=jax.random.key(0), verbose=False
)
print(np.round(emp.correlation, 3))
print("S_TC:", np.round(emp.S_TC, 3))
print("S_TU:", np.round(emp.S_TU, 3))
```

```
JaxgsaWarning: jaxgsa.vkoga: the training X is itself correlated (fitted latent
correlation between 'x1' and 'x2' is 0.61). ...

[[1.    0.609 0.041]
 [0.609 1.    0.003]
 [0.041 0.003 1.   ]]
S_TC: [0.889 0.632 0.044]
S_TU: [0.328 0.082 0.033]
```

The fit recovers `rho_12 = 0.6` as 0.609, and the two entries that should be
zero come out at 0.041 and 0.003. That is sampling scatter at N=2048.

This run trips the training-design warning, and it should: the training sample
and the copula are the same correlated data. `S_TC` is still meaningful,
because it is evaluated on the measure the sample covers. `S_TU` lands within
0.007 of the closed form here, which is luck you should not count on: the model
is linear, and a linear response is the one case where extrapolating off the
ridge costs nothing. Treat `S_TU`, `S_U` and `S_IU` from a run like this as
unverified, or move to [Kucherenko](/examples/kucherenko).

## Independent inputs collapse to S1 / ST

When neither the problem nor the call declares a correlation, the five indices
reduce to the familiar picture. `S_TC` is the first-order Sobol' index, `S_TU`
is the total index, and `S_C` vanishes.

```python
indep = jaxgsa.vkoga.analyze(problem, X, Y, key=jax.random.key(0), verbose=False)

print("S_TC:", np.round(indep.S_TC, 3))
print("S_TU:", np.round(indep.S_TU, 3))
print("S_C: ", np.round(indep.S_C, 4))
print("analytical S1 = ST:", np.round(a**2 / (a**2).sum(), 3))
print("is_correlated:", indep.is_correlated)
```

```
S_TC: [0.762 0.189 0.047]
S_TU: [0.77  0.192 0.049]
S_C:  [0.0008 0.0003 0.    ]
analytical S1 = ST: [0.762 0.19  0.048]
is_correlated: False
```

`S_TC` and `S_TU` now agree with each other and with the analytic answer to
about 0.008, and `S_C` sits within 0.0008 of zero. The gap between the
prioritisation and fixing measures has closed, because the correlation that
opened it is gone.

## The fitted surrogate

The result keeps the kernel expansion, so `predict` costs one kernel product
against a few hundred centres. A centre is a training point the greedy fit
selected to place a kernel bump on. Batching is automatic and bounded by the
global memory budget.

```python
X_new = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 1000, seed=1))
Y_pred = result.predict(X_new)

print("prediction shape:", Y_pred.shape)
print("n_centers:", result.n_centers)
print("gamma:", round(result.gamma, 3))
print("ridge:", result.ridge)
print("training rmse:", float(result.rmse))
print("cv rmse:", float(result.cv_rmse))
print("Var(Y) under the copula:", float(result.variance))
```

```
prediction shape: (1000,)
n_centers: 300
gamma: 7.533
ridge: 5.994842503189409e-05
training rmse: 0.15957695444377903
cv rmse: 0.2133223694760349
Var(Y) under the copula: 7.461814001203544
```

`n_centers` is the greedy's stopping point, capped by `max_centers`, which
defaults to 300. Reaching exactly 300 means the cap bound the fit rather than
the error criterion, so raising `max_centers` may still help here. `rmse`
(0.160) is the fit on its own training rows and `cv_rmse` (0.213) is the
held-out error. Both are small against `std(Y) = 2.297`, which is why the
indices above are trustworthy. `variance` is the output variance under the
correlated input measure, and it differs from `Y.var()` on an independent
training design.

## xarray export

`VKOGAResult.to_dataset()` converts results to a labeled `xarray.Dataset`,
including the copula matrix on its own pair of parameter dimensions.

```python
print(result.to_dataset())
```

```
<xarray.Dataset> Size: 280B
Dimensions:      (param: 3, param_i: 3, param_j: 3)
Coordinates:
  * param        (param) <U2 24B 'x1' 'x2' 'x3'
  * param_i      (param_i) <U2 24B 'x1' 'x2' 'x3'
  * param_j      (param_j) <U2 24B 'x1' 'x2' 'x3'
Data variables:
    S_TC         (param) float64 24B 0.8824 0.6307 0.03245
    S_TU         (param) float64 24B 0.3406 0.08459 0.03393
    S_U          (param) float64 24B 0.3368 0.08298 0.03268
    S_C          (param) float64 24B 0.5456 0.5477 -0.0002319
    S_IU         (param) float64 24B 0.00376 0.001608 0.001243
    variance     float64 8B 7.462
    rmse         float64 8B 0.1596
    correlation  (param_i, param_j) float64 72B 1.0 0.6 0.0 0.6 ... 0.0 0.0 1.0
Attributes:
    n_centers:      300
    gamma:          7.533150951473334
    ridge:          5.994842503189409e-05
    is_correlated:  True
    cv_rmse:        0.2133223694760349
```

The indices sit on the `param` dimension, so you can select by parameter name
instead of by position (`ds.S_TC.sel(param="x1")`). The correlation matrix
needs two parameter axes, which is why `param_i` and `param_j` appear alongside
`param`. For time-series results, pass `time_coords` to label the time
dimension.

## Multi-output

`Y` may be `(N,)`, `(N, K)`, or `(N, T, K)`. All output slices share one greedy
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
    correlation=R,
)
Y_multi = jnp.column_stack([Y, jnp.sum(X**2, axis=1)])

multi = jaxgsa.vkoga.analyze(problem_multi, X, Y_multi, key=jax.random.key(0))
print("S_TC:", np.round(multi.S_TC, 3))
```

```
jaxgsa.vkoga.analyze
  problem: D=3 (x1, x2, x3)
    marginals: gaussian=3
    correlation: correlated (Gaussian copula)
    output: N=2048 runs, T=1 x K=2 output slices
    invalid: none found in 2048 rows (policy 'raise')
  timing:
    fit + compute: 4.65 s
    n_centers: 300
    gamma: 7.533
    ridge: 0.001292
    batch_size: auto (resolved from the memory budget)
  results: top 3 of 3 parameters by S_TC, mean over 2 output slices
    1. x1  S_TC=0.6766
    2. x2  S_TC=0.5451
    3. x3  S_TC=0.1571
S_TC: [[0.882 0.631 0.032]
 [0.471 0.459 0.282]]
```

Two outputs give a `(2, 3)` index array: one row of three parameter indices per
output. Row one is the linear output and reproduces the scalar run exactly. Row
two is the sum of squares, where the ranking is much flatter, since `x3`
contributes to the sum of squares on equal terms with the others. The summary
block ranks by the mean over output slices, which is a convenience, not an
index. Read the rows.

## No Shapley effects

`VKOGAResult.shapley()` raises `NotImplementedError` on purpose:

```python
result.shapley()
```

```
NotImplementedError: VKOGAResult has no term-wise variance decomposition, so
Shapley effects are undefined for it; use jaxgsa.hdmr or jaxgsa.pce instead
```

Shapley effects allocate variance across parameter subsets. A kernel expansion
is a sum over centres instead, and every centre involves every parameter, so
there is no membership matrix to allocate from. Use
[`jaxgsa.hdmr`](/examples/hdmr), whose ANCOVA terms are labelled and which
supports `shapley(include_correlative=True)` for dependent inputs, or
[`jaxgsa.pce`](/examples/pce).

## Shape rules

| `Y` shape | `S_TC` / `S_TU` / `S_U` / `S_C` / `S_IU` | `variance` / `rmse` |
|---|---|---|
| `(N,)` | `(D,)` | `()` |
| `(N, K)` | `(K, D)` | `(K,)` |
| `(N, T, K)` | `(T, K, D)` | `(T, K)` |

D is always the last axis of the index arrays. `correlation` is a property of
the input model, not of any output slice, so it stays `(D, D)` throughout.

## Practical caveats

- Train on an independent, space-filling design, even when the analysis is
  correlated. See [above](#the-training-design-must-be-independent).
- Use float64. The normal equations square the condition number of the cross
  kernel, which float32 cannot carry for small `gamma`. Cross validation
  partly self-corrects, since the scores are computed in the same arithmetic
  and penalise the blown-up corner of the grid, but the ceiling is real.
- The hyperparameter search dominates the cost. A 10x10 `gamma`/`ridge` grid,
  each point refitted `n_folds` times, is the bulk of the runtime. Pass
  `gamma=` and `ridge=` explicitly once you know good values. Raising
  `n_outer`, `n_inner`, or `n_variance` is comparatively cheap, because those
  only touch the surrogate.
- `key=` is required. `analyze` raises `ValueError: key is required for the
  Monte-Carlo index estimate` without it, rather than seeding itself, so a run
  is always reproducible from what you passed.
- The problem needs at least two parameters. `D = 1` raises `ValueError`,
  because conditioning on "the other parameters" has no meaning.
- `S_C` can be negative when a correlation opposes a direct effect. Small
  negative values around zero are ordinary estimator noise for an uncorrelated
  parameter.
- The kernel is isotropic, so inputs are mapped to `[0, 1]` through their
  marginal CDFs before fitting. This is the same transform HDMR uses, and
  `predict` applies it too.

## See also

- [Kucherenko](/examples/kucherenko) for the same two quantities from model
  runs instead of a surrogate. It is also the check when a VKOGA index leaves
  `[0, 1]`.
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
