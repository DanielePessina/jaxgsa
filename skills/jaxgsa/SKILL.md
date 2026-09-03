---
name: jaxgsa
description: Use when writing, reviewing, or documenting code that uses jaxgsa for global sensitivity analysis in JAX. Covers defining a Problem, drawing a Sobol, Morris, eFAST, or Kucherenko design, drawing plain Monte Carlo samples, and running any of the thirteen analysis methods (sobol, morris, efast, kucherenko, pce, hdmr, shapley, dgsm, hsic, pawn, borgonovo, optimal_transport, vkoga), including correlated and categorical inputs, confidence intervals, batching, and xarray export.
---

# jaxgsa

Global sensitivity analysis in JAX. Every method lives in its own namespace, and
that namespace holds the sampling and analysis commands for the method.

```python
import jaxgsa

jaxgsa.Problem, jaxgsa.UniformSpec, jaxgsa.GaussianSpec, jaxgsa.CategoricalSpec
jaxgsa.sampling            # plain draws and correlation helpers
jaxgsa.sobol, jaxgsa.morris, jaxgsa.efast, jaxgsa.kucherenko   # build own design
jaxgsa.pce, jaxgsa.hdmr, jaxgsa.shapley, jaxgsa.dgsm, jaxgsa.hsic,
jaxgsa.pawn, jaxgsa.borgonovo, jaxgsa.optimal_transport, jaxgsa.vkoga  # given data
jaxgsa.benchmarks, jaxgsa.config
```

## Quick start

```python
import jax.numpy as jnp
import jaxgsa

problem = jaxgsa.Problem.from_dict({
    "x1": (-jnp.pi, jnp.pi),
    "x2": (-jnp.pi, jnp.pi),
    "x3": (-jnp.pi, jnp.pi),
})

design = jaxgsa.sobol.sample(problem, n_samples=4096, seed=42)
Y = model(design.samples)          # (n_runs, D) in, (N,) or (N, K) or (N, T, K) out
result = jaxgsa.sobol.analyze(design, Y)

result.S1, result.ST, result.S2
```

`verbose=True` is the default on all thirteen `analyze()` functions and on the
four design samplers. The printed block is a receipt. It echoes the marginals,
the inferred `(N, T, K)` shape, the invalid-run count, and the top-k ranking, so
a transposed `Y` shows up before you plot anything. Read it on the first run and
pass `verbose=False` in loops and tests. `jaxgsa.sampling.monte_carlo` has no
`verbose` keyword.

## Problem

The direct constructor takes uniform marginals only.

```python
jaxgsa.Problem(
    names: tuple[str, ...],
    bounds: tuple[tuple[float, float], ...],
    output_names: tuple[str, ...] | None = None,
    correlation=None,
    correlation_type: Literal["latent", "spearman"] = "latent",
)
```

`from_dict` takes any mix of marginals, in dict insertion order, which is the
model's column order.

```python
problem = jaxgsa.Problem.from_dict(
    {
        "x1": (0.0, 1.0),                                        # uniform shorthand
        "x2": jaxgsa.GaussianSpec(mean=0.0, variance=4.0, high=3.0),
        "x3": {"dist": "categorical", "probs": [0.5, 0.5], "labels": ["off", "on"]},
    },
    output_names=("y",),
    truncate_gaussians=None,     # e.g. 1e-4 to bound every Gaussian at its own quantiles
    correlation=None,
    correlation_type="latent",
)
```

Each value may be a `(low, high)` tuple, a spec dataclass (`UniformSpec`,
`GaussianSpec`, `CategoricalSpec`), or the matching plain dict
(`UniformInputSpec`, `GaussianInputSpec`, `CategoricalInputSpec`). The dict form
is a TypedDict, so a problem written that way is JSON-expressible.

Properties: `names`, `bounds`, `output_names`, `num_vars`, `input_specs`,
`correlation`, `has_correlated_inputs`, `has_non_uniform_inputs`,
`has_categorical_inputs`, `categorical_labels`.

Keep in mind:

- `problem.bounds` is `None` as soon as any marginal is not uniform. Read
  `input_specs` and branch with `isinstance` instead.
- Names must be unique strings. Both rules are enforced at construction, because
  a bad name only fails much later and much less clearly.
- Set `output_names` whenever `T` and `K` could be confused. Its length must
  equal the trailing axis of `Y`, and the mismatch is caught before array work.
- `Problem` is frozen, so `problem.with_correlation(R)` returns a copy. Pass
  `None` to drop a matrix.
- `correlation_type="spearman"` converts a rank correlation with
  `2 sin(pi rho_s / 6)` before storing. A published Spearman 0.8 is a latent
  0.8135, so declaring it as `"latent"` understates the dependence.
- A slightly non-positive-definite matrix is repaired by clipping negative
  eigenvalues and rescaling the diagonal, with a warning. A matrix whose repair
  would move any entry by 0.05 or more is rejected.
- Correlation and categorical parameters do not mix. Keep the categorical row
  and column at identity.

## Sampling

```python
# Design builders. The method must be able to run your model at its points.
jaxgsa.sobol.sample(problem, n_samples, *, base_n=None, calc_second_order=True,
                    scramble=True, seed=None, verbose=True)      -> SobolSamples
jaxgsa.morris.sample(problem, n_trajectories, *, num_levels=4,
                     method="trajectory", scramble=True, seed=None,
                     truncation_quantile=1e-4, verbose=True)     -> MorrisSamples
jaxgsa.efast.sample(problem, n_per_curve, *, M=4, seed=None,
                    verbose=True)                                -> EFASTSamples
jaxgsa.kucherenko.sample(problem, n_samples, *, scramble=True, seed=None,
                         verbose=True)                           -> KucherenkoSamples

# Plain draws and correlation helpers, for the nine given-data methods.
jaxgsa.sampling.monte_carlo(problem, n, *, seed=None)     -> np.ndarray  (n, D)
jaxgsa.sampling.correlate(X, problem, *, seed=None)       -> np.ndarray  (N, D)
jaxgsa.sampling.fit_correlation(problem, X)               -> np.ndarray  (D, D)
jaxgsa.sampling.correlation_from_covariance(cov)          -> np.ndarray  (D, D)
```

`seed` is keyword-only everywhere. It accepts an int, an existing
`np.random.Generator`, or `None` for fresh OS entropy.

Keep in mind:

- **`sobol.sample` takes the total budget, not the base count.**
  `sample(problem, 8192)` on `D=3` gives 8192 model runs from `base_n=1024`, not
  8192 times 8. `efast.sample` (per curve) and `morris.sample` (per trajectory)
  do multiply. Check `samples.samples.shape[0]` if you are not sure.
- Duplicate rows are removed, so `n_runs` is below `n_expanded`. Evaluate
  exactly the rows in `samples.samples`, in order, and pass the outputs straight
  to `analyze`, which rebuilds the blocks itself.
- The seed feeds the Owen scramble, not the sequence. Keep `scramble=True`.
  Scrambling is what removes the Sobol sequence's structural duplicates, and
  without it a Saltelli design loses real blocks to deduplication.
  `kucherenko.sample` raises on `scramble=False, seed=...` rather than accept a
  seed that would do nothing. `efast.sample` has no `scramble`, because its
  randomness is the phase shift of each search curve.
- `monte_carlo` honors `problem.correlation` through the NORTA construction and
  returns level codes for categorical columns. An independent problem keeps the
  plain uniform path bit for bit, so old seeds reproduce old samples.
- `correlate` re-pairs an existing sample by rank with the Iman-Conover method,
  so the marginal values survive untouched and only the pairing changes. It
  raises on a non-finite `X`, because `np.sort` puts `NaN` last and a bad row
  would be pinned to the extreme rank scores.
- `fit_correlation` estimates the latent matrix from data through the Spearman
  rank correlation. Feed the result to `problem.with_correlation(...)`.

### Design objects

All four carry `save(path)` and `load(path)` as one compressed NPZ file holding
the sample matrix and a JSON metadata blob. `SobolSamples` and `MorrisSamples`
also have `downsample(base_n, Y=None)`, which returns a prefix-nested smaller
design and the matching subset of outputs, so you can plot convergence against
sample size from one evaluated design.

`SobolSamples` fields: `samples`, `n_runs`, `n_expanded`, `expanded_to_unique`,
`base_n`, `n_params`, `calc_second_order`, `problem`, `unit`,
`expanded_to_unit`.

```python
# Reuse one design under different input ranges. unit holds the quasi-random
# points before any distribution is applied, and transform applies a new set.
narrow = samples.transform({"x1": {"low": 0.0, "high": 1.0}, ...})
wide   = samples.transform({"x1": {"low": -1.0, "high": 2.0}, ...})
```

`theta` has type `jaxgsa.Theta`, a mapping keyed by parameter name and then by
the field names of that parameter's distribution. `transform` is written in JAX
and is differentiable with respect to `theta`. It raises for a problem with
categorical parameters, because a categorical inverse CDF is a step function.

```python
# Free Morris screening from an already-evaluated Saltelli design.
morris_result = jaxgsa.morris.analyze(samples.to_morris(), Y)
```

## Shape contract

| Output | Shape |
| --- | --- |
| Scalar | `(N,)` |
| Multiple outputs | `(N, K)` |
| Time series with multiple outputs | `(N, T, K)` |

The sample axis is first and the output axis is last. jaxgsa never infers,
transposes, or inserts axes. **A 2-D `Y` is always `(N, K)`.** A single
time-varying output must be written `(N, T, 1)`. The index arrays follow:
`(D,)` for a scalar, `(K, D)` for multi-output, `(T, K, D)` for a time series.
Check that shape once, on the first run, and a transposed array cannot survive
to your plots.

## Cross-cutting keywords

Every `analyze()` takes `verbose` and `on_invalid`. Eleven of the thirteen take
the bootstrap block. Which batching keyword exists depends on the axis the
method batches over.

### Failed model runs

`on_invalid: "raise" | "propagate" | "drop"`, default `"raise"`. The default
refuses, because an index computed from part of a sample is a different quantity
from the one you asked for, and `analyze()` is cheap to run again.

`"drop"` removes the whole block a bad value sits in: a Saltelli group, a Morris
trajectory, a Kucherenko base point, or one row for a given-data method.
`jaxgsa.efast.analyze` accepts only `"raise"` and `"propagate"`, because its
ordered Fourier sweep changes meaning if a point is removed.

Every result carries `result.invalid`.

```python
result.invalid.n_invalid        # blocks that held a bad value
result.invalid.unit_indices     # which blocks
result.invalid.bad_row_indices  # the rows that actually failed
result.invalid.row_indices      # every row those blocks cover, which "drop" removes
result.invalid.sources          # bad values in X, in Y, or both
result.invalid.unit             # ROW, SALTELLI_GROUP, TRAJECTORY, BASE_POINT, CURVE
```

Both index arrays refer to the array as you passed it, so they name model runs
you can find. For a block design they differ by a large factor: one failed run
inside an eFAST search curve gives one entry in `bad_row_indices` and 257 in
`row_indices`.

### Confidence intervals

```python
result = jaxgsa.sobol.analyze(
    samples, Y,
    n_bootstrap=1000,          # default 0, meaning no interval
    conf_level=0.95,
    ci_method="quantile",      # or "gaussian"
    key=jax.random.key(0),     # REQUIRED once n_bootstrap > 0, raises without it
    keep_replicates=True,      # keeps result.ci.replicates
)
result.S1_conf   # (2, ...), the leading axis holds [lower, upper]
result.ci.level, result.ci.method, result.ci.n_bootstrap, result.ci.replicates
```

`"quantile"` reads the endpoints off the empirical bootstrap distribution.
`"gaussian"` centres them on the point estimate at plus or minus z times the
standard deviation of the draws, which is smoother at small `n_bootstrap` but
assumes the draws are normal.

`result.ci` is `None` when no bootstrap ran. `keep_replicates` defaults to
`False` because the draws are large: 1000 resamples of a `(T=100, K=5, D=20)`
index array is 80 MB, more than the rest of the result put together.

`efast` and `hsic` have no bootstrap, by decision. eFAST has nothing to resample
inside one search curve, since removing a point changes what the estimator
computes rather than shrinking the sample. HSIC reports permutation `p_values`
instead, because a row bootstrap duplicates rows onto the kernel diagonal where
the kernel is exactly 1, which biases the resampled index upward by
construction.

The four surrogate-backed methods (`pce`, `hdmr`, `vkoga`, `shapley`) refit
their surrogate on every replicate, so an interval there costs an order of
magnitude more than a row resample on a direct estimator.

### Batching

| Keyword | Batches over | On |
| --- | --- | --- |
| `batch_size` | sample rows | `dgsm`, `hdmr`, `pce`, `vkoga`, surrogate `predict` |
| `slice_chunk_size` | `(T, K)` output slices | `sobol`, `hdmr`, `pawn`, `borgonovo`, `efast`, `optimal_transport` |
| `resample_chunk_size` | bootstrap replicates | `morris` |

The value is clamped to the axis it sizes and never selects a different
algorithm, so the answer does not depend on it beyond float summation order.
`None` derives a width from the memory budget, and an explicit value always
wins.

`hsic`, `shapley`, and `kucherenko` take none of the three.
`jaxgsa.hsic.analyze(..., batch_size=...)` raises `TypeError`, because HSIC
holds `2D+1` matrices of shape `(N, N)` at once and only `N` bounds that.

### indices(), the traceable core

Eleven namespaces expose `indices(...)`, which returns raw arrays, checks
nothing, builds no result, and is safe under `jax.jit`, `jax.vmap`, and
`jax.grad`. `kucherenko` and `vkoga` do not have one, because Kucherenko is host
NumPy from end to end and VKOGA's index stage is a host quasi-Monte-Carlo loop.
Pair `sobol.indices` with `SobolSamples.transform` to differentiate an index
with respect to the input distribution parameters.

### Warnings

```python
import warnings
from jaxgsa import JaxgsaWarning

warnings.filterwarnings("error", category=JaxgsaWarning)   # good default for CI
```

Every jaxgsa warning is a `JaxgsaWarning`, a subclass of `UserWarning`. jaxgsa
warns when a run is degraded but still valid: a repaired correlation matrix, a
zero-variance output slice, a design thinned by deduplication. Each of those
produces numbers that look fine and mean less than they appear to, so escalate
to `"error"` in a pipeline.

### Configuration

```python
import jax
jax.config.update("jax_enable_x64", True)     # BEFORE any analysis

jaxgsa.config.enable_compilation_cache("~/.cache/jaxgsa-jax")
jaxgsa.config.set_memory_budget(512)          # MiB by default
jaxgsa.config.set_memory_budget(1.5, unit="gb")
jaxgsa.config.get_memory_budget()
```

JAX computes in 32-bit by default and downcasts a `float64` `Y` you pass it.
jaxgsa warns when it sees this. HSIC and VKOGA need float64 outright, for the
reasons in their sections. `set_memory_budget` reads megabytes unless `unit=`
says otherwise, and an old byte-count call raises rather than silently meaning
512 TB.

## Method capabilities

| Method | Reports | Own design | Correlated | Categorical | Bootstrap CI |
|---|---|:--:|:--:|:--:|---|
| `borgonovo` | delta, S1 | no | yes, section | yes | `n_bootstrap` |
| `dgsm` | bounds on ST | no | no | no | `n_bootstrap` |
| `efast` | S1, ST | yes | no | no | none |
| `hdmr` | Sa, Sb, S per term, surrogate | no | yes, caveat | no | `n_bootstrap` |
| `hsic` | dependence measure | no | yes, section | no | none |
| `kucherenko` | S1, ST under dependence | yes | yes | no | `n_bootstrap` |
| `morris` | mu\*, sigma | yes | no | no | `n_bootstrap` |
| `optimal_transport` | W2 squared index, advective and diffusive | no | yes, section | yes | `n_bootstrap` |
| `pawn` | KS distance | no | yes, section | yes | `n_bootstrap` |
| `pce` | S1, S2, ST, surrogate | no | no | no | `n_bootstrap` |
| `shapley` | allocation summing to 1 | no | pce no, hdmr yes | no | `n_bootstrap` |
| `sobol` | S1, S2, ST | yes | no | yes | `n_bootstrap` |
| `vkoga` | S_TC, S_TU, S_U, S_C, S_IU, surrogate | no | yes | no | `n_bootstrap` |

A "no" in the Correlated or Categorical column is a refusal with a `ValueError`
that names the parameters and the alternatives, never a silent approximation.

The four methods marked "yes, section" are correlation-inclusive. A parameter
that does not enter the model, but that correlates with one that does, scores
above zero. That is the correct reading of those indices, not an estimation
error. Use HDMR's Sa and Sb split, VKOGA, or Kucherenko when you must separate
the structural effect from the correlation-induced one.

HDMR's caveat is that its `ST` under dependence is the SCSA convention and not a
total-effect index. Read its section before ranking anything from a correlated
fit.

Choosing a method:

- Can run the model, independent inputs, want the reference answer: `sobol`.
- Expensive model with many parameters: screen with `morris` at r(D+1) runs, or
  with `dgsm` if the model is JAX-differentiable, fix the inert parameters, then
  spend the rest of the budget on `sobol` for the survivors.
- Only existing (X, Y) data: any given-data method.
- Correlated inputs: `kucherenko` if you can still run the model, `vkoga` or
  `hdmr` if you cannot, or a moment-independent method.
- Skewed or heavy-tailed output, where variance is the wrong summary: `pawn`,
  `borgonovo`, `optimal_transport`.
- Want to know whether a parameter shifts the output or reshapes it:
  `optimal_transport`.
- Want one fair number per parameter that sums to 1: Shapley effects.
- Want a reusable surrogate too: `pce`, `hdmr`, `vkoga`.

## Methods

### sobol, variance decomposition on a Saltelli design

```python
samples = jaxgsa.sobol.sample(problem, 8192, seed=0, calc_second_order=True)
Y = model(samples.samples)
result = jaxgsa.sobol.analyze(
    samples, Y,
    estimator="saltelli-jansen",   # default; also "jansen", "janon-monod",
                                   # "martinez", "mauntz-kucherenko", "azzini-rosati"
    slice_chunk_size=None, on_invalid="raise", verbose=True,
)
result.S1, result.ST, result.S2, result.estimator
```

Cost is `N(D + 2)` model runs, or `N(2D + 2)` with `calc_second_order=True`,
which is the default.

Keep in mind:

- `analyze` standardizes each output slice over the sample axis before it
  estimates, because the S1 and S2 estimators are uncentred products and a
  non-zero output mean biases them. SALib standardizes the same way.
- `ST >= S1` holds for the true population values, not for every estimator's
  finite-sample output. Only `"azzini-rosati"` enforces it sample-wise, and the
  others can print `S1 > ST` on noisy data. jaxgsa never clips.
- `S2` is symmetric with a `NaN` diagonal, and `S2[j, k]` and `S2[k, j]` are
  estimated independently then averaged. SALib reports the upper triangle alone.
- Below about `100 * D` runs the estimates carry more sampling noise than
  signal. Screen with Morris first and come back.
- `calc_second_order=False` nearly halves the bill when a ranking is all you
  need.

### morris, elementary-effects screening

```python
samples = jaxgsa.morris.sample(problem, n_trajectories=50, num_levels=4,
                               method="trajectory", seed=0)
Y = model(samples.samples)
result = jaxgsa.morris.analyze(samples, Y, n_bootstrap=0,
                               resample_chunk_size=None)
result.mu, result.mu_star, result.sigma
result.to_physical_units()      # uniform marginals only, raises on a Gaussian one
```

Cost is r(D + 1) model runs, deduplicated.

Keep in mind:

- mu\* is a mean absolute slope in unit-cube coordinates, not a variance share,
  and it does not sum to anything. It is a proxy for the ST ranking, not a
  substitute, and it misranks: on Ishigami mu\* puts x2 first while ST puts x1
  first. Use Morris to decide what to drop, which is what it is good at, and let
  Sobol rank what is left. Drop only what sits near the origin of the mu\* and
  sigma plot.
- Large sigma relative to mu\* means nonlinearity or interactions, and it is not
  attributable to a pair.
- With unbounded Gaussian marginals mu\* has no fixed magnitude, because how far
  the design reaches into the tail sets it. Only rankings survive a change of
  `truncation_quantile`. Declare `truncate_gaussians=` on the `Problem` if
  magnitudes have to mean anything.
- `to_morris()` on an evaluated Saltelli design gives the same measures for zero
  extra model runs, because a Saltelli design already contains a radial Morris
  design. It produces a radial design, so compare against
  `morris.sample(..., method="radial")`, never against the `"trajectory"`
  default. On Ishigami the two differ by a factor of 1.9 on x2.
- The derived mu\* and the Sobol ST come from the same model outputs, so their
  agreement is not an independent check of either.

### efast, Fourier amplitude sensitivity test

```python
samples = jaxgsa.efast.sample(problem, n_per_curve=2048, M=4, seed=0)
Y = model(samples.samples)      # (n_per_curve * D, D) rows, evaluated in order
result = jaxgsa.efast.analyze(samples, Y, slice_chunk_size=None,
                              on_invalid="raise")
result.S1, result.ST
```

`M`, `n_per_curve`, and the problem travel inside `EFASTSamples`, so sampling
and analysis can never be mismatched.

Keep in mind: no `S2`, no bootstrap, no `on_invalid="drop"`, and a known upward
bias on `ST` from harmonic interference. Pick eFAST when the run budget is the
binding constraint and you have measured that `N * D` beats `N'(D+2)` at the
accuracy you need. Otherwise run `sobol`, which gives you more.

### kucherenko, Sobol indices under declared dependence

```python
ks = jaxgsa.kucherenko.sample(problem, 4096, seed=0)   # reads problem.correlation
Y = model(ks.samples)
result = jaxgsa.kucherenko.analyze(ks, Y, n_bootstrap=0)
result.S1   # correlation-inclusive first order, VKOGA's S_TC
result.ST   # correlation-exclusive total, VKOGA's S_TU
result.variance
```

Cost is `N(2D + 1)` model runs. It evaluates the real model on a
conditional-copula design and uses no surrogate.

Keep in mind:

- `ST >= S1` no longer holds in general under dependence.
- The design is built from the declared copula, so a wrong matrix gives clean
  estimates of the wrong quantity. Conditioning is closed-form only in latent
  normal space, which means tail-dependent or non-monotone dependence is not
  representable here.
- With no declared correlation it reduces to the Saltelli column-swap scheme,
  but uses the Homma-Saltelli S1 estimator, so it agrees with `sobol` only up to
  Monte Carlo noise. Run `sobol` in that case, which also gives you `S2`.
- No `S2`, no surrogate, and a categorical parameter raises.

### pce, polynomial chaos expansion

Given data, and it keeps a surrogate.

```python
X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, n=4000, seed=0))
Y = model(X)
result = jaxgsa.pce.analyze(problem, X, Y, order=3, ridge=1e-8,
                            fit_ratio=0.5, batch_size=None)
result.S1, result.S2, result.ST, result.coefficients
result.loo_rmse, result.explained_variance, result.order
Y_pred = result.predict(X_new, batch_size=2048)
effects = result.shapley()

jaxgsa.pce.effective_order(problem, n_samples, order=10, fit_ratio=0.5)
```

Keep in mind:

- **Check the fit before you read the indices.** They are exact within the
  fitted polynomial, so a wrong polynomial gives exactly wrong indices. `order`
  defaults to 3, which is not enough for a strong nonlinearity. On Ishigami,
  `order=3` reports `S1(x1) = 0.662` against a truth of 0.314 and captures 46%
  of the variance.
- `explained_variance` should sit near 1, and `loo_rmse` should be small next to
  `Y.std()`. Raise `order` until `loo_rmse` stops falling, and watch for it
  turning back up, which is overfitting. `explained_variance` is in-sample, so
  it climbs as you add terms and cannot detect an overfit on its own.
  `pce.analyze` does not warn about it, so check both numbers yourself.
- A total-degree basis at order p in D parameters has C(D+p, p) terms: 286 at
  D=3, p=10, but 3003 at D=10, p=5. `effective_order` returns the order the data
  supports, capped at the `order` you pass, and `analyze` drops to it and
  reports the drop as `result.order`. If it comes back at 1 or 2, PCE is not the
  method for this dataset.
- Polynomials ring around a step, and no `order` fixes it. For a discontinuity,
  threshold, or hard saturation use HDMR's B-splines or a non-surrogate method.
- Correlated inputs are refused, because the orthogonality that makes the
  coefficients readable as variances is orthogonality under the independent
  product measure.

### hdmr, RS-HDMR with the ANCOVA split

Given data, and it keeps a surrogate.

```python
result = jaxgsa.hdmr.analyze(problem, X, Y, maxorder=2, maxiter=100, m=2,
                             lambdax=0.01, slice_chunk_size=None,
                             batch_size=None)
result.Sa, result.Sb, result.S, result.ST, result.terms, result.rmse
result.S1, result.S2, result.S3          # dense vector, matrix, tensor views
Y_pred = result.predict(X_new, batch_size=2048)
effects = result.shapley(include_correlative=True)
```

`Sa` is the structural share of a term's variance, `Sb` the correlative share,
and `S` is their sum.

Keep in mind:

- **Under correlated inputs, HDMR's `ST` is not a total-effect index.** It is
  the SCSA term-membership sum of Li et al. (2010, section 2.2.3): the sum of
  `Sa + Sb` over every term containing the parameter. It can be negative, is not
  bounded in [0, 1], and does not answer the parameter-fixing question. The bias
  runs toward "cannot be fixed" and can be an order of magnitude. On
  `Y = X1 + X2 + X3` with standard normal marginals and `corr(X1, X2) = 0.95`,
  HDMR reports `ST = [0.398, 0.397, 0.207]` where the true conditional-variance
  totals are `[0.020, 0.020, 0.204]`. `analyze` emits one warning saying so. Use
  `kucherenko.ST` or `vkoga.S_TU` for a conditional-variance total. `S1` carries
  the matching caveat: it is the structural share `Sa` of the first-order term,
  not the Sobol first-order index.
- **Check `result.S.sum()` is near 1 before ranking anything.** The shortfall is
  variance the surrogate left unexplained, and every index inherits it. Raise
  `maxorder` or `m`, or accept that this model does not decompose into low-order
  terms.
- Only the first-order components are backfitted, up to `maxiter` sweeps with an
  early stop once the coefficients settle. Every higher-order component is a
  single ridge solve.
- The surrogate is trained on the outputs you supply, so `predict` and `rmse`
  are on that same scale, with no inverse transform.
- Reach for HDMR over PCE when you want the per-term Sa and Sb split, or when
  the response has kinks a polynomial cannot follow. Otherwise PCE fits in one
  linear solve and gives the same numbers with fewer knobs to get wrong.

### shapley, a fair allocation summing to 1

```python
# Canonical form: derive from a fitted surrogate.
result = jaxgsa.pce.analyze(problem, X, Y, order=10, verbose=False).shapley()
result = jaxgsa.hdmr.analyze(problem, X, Y, maxorder=2).shapley(
    include_correlative=True)

# Convenience wrapper: fit the chosen surrogate and allocate in one step.
result = jaxgsa.shapley.analyze(problem, X, Y, backend="pce",
                                include_correlative=False, **backend_kwargs)
result.Sh, result.S1, result.ST, result.explained_variance, result.backend
```

`Sh` sums to exactly 1 by Shapley efficiency, and `S1 <= Sh <= ST` is visible at
a glance because all three come from the same surrogate.

Keep in mind:

- **Check `explained_variance` first, every time.** The effects are exact within
  the surrogate, and a bad surrogate gives exact nonsense. At PCE `order=3`
  Ishigami gives `Sh = [0.803, 0.055, 0.141]`, against `[0.436, 0.442, 0.122]`
  at `order=10`, which matches the analytical values. A warning fires when the
  fit is too poor to trust. For `backend="pce"` the number is a coefficient of
  determination in [0, 1]; for `backend="hdmr"` it is the decomposed fraction
  and can exceed 1.
- Shapley is not the fixing measure. Splitting an interaction gives a share to
  both participants, so a parameter can be safe to fix and still carry a visible
  `Sh`. Use `ST` for that question.
- Shapley dissolves interactions into the participants by design. For the
  interaction itself, read `S2` from `sobol` or `pce`, or the `ST - S1` gap.
- Backend keyword arguments are forwarded unchecked, so `backend="pce",
  maxorder=3` raises a plain `TypeError` from `pce.analyze` rather than a
  jaxgsa-specific message.
- `VKOGAResult.shapley()` raises `NotImplementedError` on purpose, because a
  kernel expansion has no term-wise variance decomposition to allocate from.
- With `backend="hdmr", include_correlative=True` the correlative shares can go
  negative. That is a defensible reading of the ANCOVA split, and it no longer
  has the fair-split interpretation that makes the method attractive.
- The surrogate-free Song et al. (2016) conditional-variance estimator is not
  implemented.

### dgsm, derivative-based screening

```python
X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, n=1024, seed=0))

# fn takes ONE sample row, shape (D,), and returns a scalar, (K,), or (T, K).
result = jaxgsa.dgsm.analyze(problem, lambda x: model(x[None, :])[0], X,
                             batch_size=None)

# Or pass a Jacobian computed elsewhere and skip fn entirely.
result = jaxgsa.dgsm.analyze(problem, Y=Y, dfdx=dfdx)

result.nu, result.sigma, result.upper_bound, result.lower_bound, result.var_y
```

Cost is `N` Jacobians, each about `min(D, T*K)` evaluations. jaxgsa picks
forward or reverse mode from the shapes.

Keep in mind:

- Passing a batch model unwrapped raises a `ValueError` that spells out the
  `lambda x: model(x[None, :])[0]` fix, so you will not be left guessing.
- **The bounds can be far too loose to rank with.** On Ishigami at N=1024 the
  Poincare upper bound is `[2.35, 7.38, 3.11]`, above 1 on every parameter, and
  it ranks x2 above x3 above x1 while the true ST ranks x1 above x2 above x3.
  More samples do not fix it. DGSM is a fast way to find parameters that do
  nothing at all. It is not a reliable ranking of the ones that do.
- The Kucherenko-Song lower bound is a proven bound only for an untruncated
  Gaussian marginal. Otherwise it is an estimate that can exceed the true ST
  when the response is curved, and `analyze` warns and names the marginals. It
  collapses to zero for any response that is not monotone in the parameter,
  because the mean derivative cancels.
- At `T*K` much larger than `D` the Jacobian costs D forward passes and the cost
  argument for DGSM evaporates. Run Sobol.
- A truncated Gaussian needs its Poincare constant from a finite-element
  spectral solve rather than a closed form.

### hsic, kernel dependence measure

```python
import jax
jax.config.update("jax_enable_x64", True)      # before the analysis

result = jaxgsa.hsic.analyze(problem, X, Y, n_perms=200,
                             key=jax.random.key(0), bandwidth=1.0)
result.R2_HSIC, result.T_HSIC, result.p_values, result.hsic_raw
```

`key` is required, because the permutation test always runs and there is no
`n_bootstrap=0` equivalent that skips it.

Keep in mind:

- **Use float64.** The V-statistic cancels three large sums against each other,
  so float32 leaves three or four correct digits and the index changes with the
  order of the sample rows. `analyze` warns. Small indices and close rankings
  are not reliable without it.
- **`bandwidth` is a real choice, not a tuning detail.** It multiplies the
  median-heuristic length scale. On Ishigami at 2000 samples, `bandwidth=0.25`
  ranks x2 above x1 above x3, agreeing with S1, while the default 1.0 ranks x1
  above x3 above x2 and puts x2 at 0.008 despite it owning 44% of the output
  variance. A wide kernel smooths the twice-oscillating x2 term into a
  near-constant. Sweep it, and report the value you used.
- Time and memory are O(N squared): about `2D+1` matrices of shape `(N, N)`.
  Nothing chunks it, and `batch_size` raises `TypeError`. Above N of about 20000
  it is impractical, since N=20000 with D=5 in float64 is roughly 35 GB.
- R2-HSIC has no units, does not sum to 1, and moves with the bandwidth. What it
  answers well is whether a parameter is doing anything at all, through the
  p-value. For a magnitude on a fixed [0, 1] scale use `optimal_transport` or
  `borgonovo`.
- Outputs of extreme magnitude can overflow float32 in the squared distances.
  Rescale by hand with `(Y - Y.mean(0)) / Y.std(0)`, which changes nothing else.
- The `T_HSIC` minus `R2_HSIC` gap says interactions exist, not which pairs.

### pawn, CDF-based sensitivity

```python
result = jaxgsa.pawn.analyze(problem, X, Y, n_bins=10, statistic="median",
                             n_bootstrap=0, slice_chunk_size=None)
result.pawn, result.n_valid_bins
```

`statistic` is `"median"`, `"max"`, or `"mean"` over the per-bin KS distances.

Keep in mind:

- `n_bins` trades conditioning resolution against sample density per bin. Check
  `result.n_valid_bins`, which counts the bins that held at least 2 samples per
  parameter. Bins below that are dropped, and `analyze` warns when a parameter
  keeps fewer than half its bins.
- Bins are equal-probability on the marginal's own CDF, so a skewed marginal
  does not by itself starve a tail bin. What empties one is a small `N`, a large
  `n_bins`, or samples that land outside the declared marginal.
- Categorical parameters need no binning. The level code already names the
  conditioning class, so PAWN uses one bin per level, `n_bins` does not apply,
  and relabelling the levels gives the same number.
- The KS distance is a supremum, so it reacts to the single largest gap between
  two CDFs and ignores everything else. It is the sharper instrument when a
  parameter moves one part of the range a lot. Borgonovo delta or the OT index
  sees more of a small shift across the whole distribution.
- First-order only. No total order, no `S2`.

### borgonovo, density-based delta

```python
result = jaxgsa.borgonovo.analyze(problem, X, Y, n_classes=None, grid_size=100,
                                  bandwidth="silverman", n_bootstrap=0,
                                  bias_correct=None, degenerate_tol=...,
                                  degenerate_bandwidth="auto",
                                  slice_chunk_size=None)
result.delta, result.S1
```

Delta is a half-L1 distance between densities, so it lies in [0, 1]. `S1` is the
given-data first-order Sobol index from the same class partition, free with the
same call.

Keep in mind:

- **Continuous outputs only.** `analyze` raises when a column takes at most 20
  distinct values and each value repeats at least 5 times on average, because a
  discrete output has atoms no grid resolves. Use `pawn` or `optimal_transport`
  there. Categorical parameters stay supported; the limit applies to the output.
- If the estimate leaves [0, 1] by more than 0.05 the computation failed and
  `analyze` raises, naming the parameter and the knob that applies to that case.
  The value is never clipped, because a clipped value looks plausible and is
  still wrong. A confidence bound outside the range only warns.
- `n_bootstrap > 0` with `bias_correct` not `False` applies the bias correction,
  including under the `bias_correct=None` default, which warns once per process
  that it did. The corrected estimate can fall marginally below 0 for weak
  parameters. Set `n_bootstrap=0` for the raw plug-in estimate.
- Below about 500 samples, read the ranking and ignore the magnitudes. The
  plug-in estimate is biased upward and the correction can push weak parameters
  below zero.
- `degenerate_tol` says when a conditioning class counts as degenerate, and
  `degenerate_bandwidth` says how wide a kernel it gets. The `"auto"` default
  floors it at `max(0.1 * h_full, grid_step)`, so it never goes below what the
  output grid can integrate. Raising `degenerate_tol` calls more classes
  degenerate and can hand them a narrower kernel than they had, which biases
  delta. `analyze` warns on the run where the floor fires.
- Delta is correlation-inclusive. Use `vkoga` or `kucherenko` to separate direct
  from correlation-borne influence.

### optimal_transport, Wasserstein-based sensitivity

```python
result = jaxgsa.optimal_transport.analyze(
    problem, X, Y,
    mode="univariate",          # or "multivariate" or "trajectory"
    n_partitions=None, standardize_outputs=True,
    epsilon=0.03, max_iter=2000, tol=None,
    dummy=False, n_bootstrap=0, slice_chunk_size=None,
)
result.ot, result.advective, result.diffusive, result.S1
result.ot_dummy, result.above_dummy     # None unless dummy=True
```

`ot` is a normalized expected squared 2-Wasserstein distance in [0, 1]. It
splits exactly into `advective`, the mean-shift part, and `diffusive`, the
spread and shape part. `2 * advective * N/(N-1)` is the given-data first-order
Sobol index, returned as `S1` for convenience. This is the only method here that
says whether a parameter shifts the output or reshapes it.

Modes: `"univariate"` gives per-column indices, `"multivariate"` gives one index
per parameter over the flattened joint output, `"trajectory"` gives one index
per parameter per output over the whole time course.

Keep in mind:

- **In a point-cloud mode, pass `dummy=True`.** `"multivariate"` and
  `"trajectory"` solve entropic transport, and the entropic bias keeps
  irrelevant parameters visibly above zero. Reading those indices without the
  dummy floor will make you believe in parameters that do nothing. Read
  `above_dummy`.
- Cost is `(n_bootstrap + 1) * D * n_partitions` Sinkhorn solves for continuous
  parameters, times `K` in `"trajectory"` mode, plus one single-replicate dummy
  pass. At 100 replicates, 10 parameters, and 25 partitions that is over 25000
  solves. A categorical parameter costs its own level count instead of
  `n_partitions`.
- Valid under correlated inputs, and correlation-inclusive by construction.
- One parameter at a time, so no `S2`, no total order, and no surrogate.

### vkoga, correlated-input variance indices

Given data, and it keeps a surrogate.

```python
import jax
jax.config.update("jax_enable_x64", True)      # before fitting

result = jaxgsa.vkoga.analyze(
    problem, X, Y,
    correlation=None,        # overrides problem.correlation for this call
    gamma=None, ridge=None, max_centers=None, n_folds=10,
    n_outer=512, n_inner=128, n_variance=8192,
    key=jax.random.key(0),   # REQUIRED, the index stage is always Monte Carlo
    batch_size=None,
)
result.S_TC     # total correlated, the input-prioritisation measure
result.S_TU     # total uncorrelated, the input-fixing measure
result.S_U, result.S_C, result.S_IU
result.correlation, result.n_centers, result.gamma, result.ridge, result.rmse
Y_pred = result.predict(X_new, batch_size=2048)
```

Keep in mind:

- **Train on an independent, space-filling design, even when the analysis is
  correlated.** This is the easy way to get wrong answers. A correlated sample
  concentrates on a ridge through the parameter space, but `S_TU` conditions on
  the other parameters and then resamples `X_i` across its whole marginal, which
  is precisely the off-ridge region a correlated training set never visited. The
  surrogate extrapolates exactly where the estimator leans on it hardest. If
  your data is observational, you can still fit the copula from it with
  `fit_correlation`, but read `S_TU`, and so `S_U`, `S_C`, and `S_IU`, as
  carrying that extrapolation error.
- **Use float64.** The coefficient step forms the normal matrix, which squares
  the condition number of the cross kernel, and for small `gamma` that exceeds
  what single precision carries. The surrogate can come out an order of
  magnitude worse. `analyze` warns when x64 is off.
- If you can still run the model, run `kucherenko` instead and get the same two
  quantities with no surrogate error in between. VKOGA is the given-data
  fallback, not the better estimator.
- Cost is dominated by a 10 by 10 grid of k-fold refits. Pass `gamma` and
  `ridge` explicitly once you know good values to skip the search.
  `n_outer`, `n_inner`, and `n_variance` only touch the surrogate, so they are
  cheap to raise.
- `result.shapley()` raises `NotImplementedError`, and there is no `S2`. Use
  `hdmr` when you need to know which interaction carries the variance.

## Export

Every result has `to_dataset(time_coords=None)`, which returns a labeled
`xarray.Dataset` on `param`, `output`, and `time` dimensions. Point estimates
come first, then the `*_lower` and `*_upper` halves of every confidence
interval, so you select a bound by name instead of by integer index.

```python
ds = result.to_dataset(time_coords=np.linspace(0, 10, T))
ds["S1"].sel(param="x1")
```

`output_names` on the `Problem` names the `output` coordinate.

## Benchmarks

`jaxgsa.benchmarks` holds analytical test functions whose Sobol indices are
known in closed form: `ishigami`, `sobol_g`, `linear`, `gaussian_linear`, and
`oakley_ohagan`. Each provides `PROBLEM`, a JAX `evaluate(X)`, the precomputed
`ANALYTICAL_S1`, `ANALYTICAL_ST`, and `ANALYTICAL_S2`, and an
`analytical_indices(...)` for non-default parameters.

```python
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate, ANALYTICAL_S1
```

Use them to check an estimator, or to measure how fast it converges.
