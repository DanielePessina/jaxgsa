# Variance-based methods

Five methods that split the output variance into the share each parameter owns.
All five assume independent inputs and refuse a correlated `Problem`. For
dependent inputs see `dependent-inputs.md`.

`sobol` and `efast` build their own design, so you must be able to run the model
at the points they choose. `pce`, `hdmr` and `shapley` take any `(X, Y)` pairs.

Every example below runs against the Ishigami benchmark, whose indices are known
in closed form:

```python
import jax
import jax.numpy as jnp
import numpy as np
import jaxgsa
from jaxgsa.benchmarks import ishigami

PROBLEM = ishigami.PROBLEM          # x1, x2, x3 all uniform on [-pi, pi]
ishigami.ANALYTICAL_S1              # [0.3139, 0.4424, 0.0   ]
ishigami.ANALYTICAL_ST              # [0.5576, 0.4424, 0.2437]
```

`x3` is the interesting parameter. On its own it does nothing, and it acts only
through its product with `x1`.

## sobol

Exact, model-free variance decomposition on a Saltelli design. This is the
reference method.

```python
samples = jaxgsa.sobol.sample(PROBLEM, n_samples=8192, seed=0)
Y = ishigami.evaluate(samples.samples)
result = jaxgsa.sobol.analyze(samples, Y)

result.S1        # [0.3223, 0.4361, 0.0014]
result.ST        # [0.5560, 0.4417, 0.2413]
result.S2[0, 2]  # 0.2333, the x1-x3 interaction
```

The printed summary is a receipt worth reading on the first run:

```
jaxgsa.sobol.sample: D=3, mode=second-order, base_n=1024, requested_runs>=8192,
  n_runs=8192, n_expanded=8192, duplicates_removed=0 (0.0%), scramble=True
jaxgsa.sobol.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=8192 runs, T=1 x K=1 output slice
    invalid: none found in 1024 Saltelli groups (policy 'raise')
  timing:
    estimators (includes compile on the first call): 0.5518 s
    slice_chunk_size: 1 (resolved from the memory budget)
    estimator: saltelli-jansen
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.556
    2. x2  ST=0.4417
    3. x3  ST=0.2413
```

Signature:

```python
jaxgsa.sobol.sample(problem, n_samples, *, base_n=None, calc_second_order=True,
                    scramble=True, seed=None, verbose=True)
jaxgsa.sobol.analyze(samples, Y, *, estimator="saltelli-jansen",
                     n_bootstrap=0, conf_level=0.95, ci_method="quantile",
                     key=None, slice_chunk_size=None, on_invalid="raise",
                     verbose=True, keep_replicates=False)
```

Estimators: `"saltelli-jansen"` (default), `"jansen"`, `"janon-monod"`,
`"martinez"`, `"mauntz-kucherenko"`, `"azzini-rosati"`.

Result: `S1`, `ST`, `S2`, `estimator`, `problem`, `invalid`, the `*_conf`
intervals, `ci`.

Cost is `N(D + 2)` model runs, or `N(2D + 2)` with `calc_second_order=True`.

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

## efast

The same `S1` and `ST` from Fourier search curves instead of a Saltelli design.

```python
samples = jaxgsa.efast.sample(PROBLEM, n_per_curve=2048, seed=0)
Y = ishigami.evaluate(jnp.asarray(samples.samples))   # 6144 rows, in order
result = jaxgsa.efast.analyze(samples, Y)

result.S1      # [0.3076, 0.4423, 0.0000]
result.ST      # [0.5507, 0.4629, 0.2393]
```

That matched the 8192-run Saltelli estimate above on 25% fewer evaluations, and
its `S1(x3)` came back at `8e-9` against a truth of exactly 0, where the
Saltelli estimator gave 0.0014. The error sits in `ST(x2)`, 0.463 against a
truth of 0.442, which is eFAST's known upward bias on the total order.

Signature:

```python
jaxgsa.efast.sample(problem, n_per_curve, *, M=4, seed=None, verbose=True)
jaxgsa.efast.analyze(samples, Y, *, slice_chunk_size=None,
                     on_invalid="raise", verbose=True)
```

`M`, `n_per_curve` and the problem travel inside `EFASTSamples`, so sampling and
analysis can never be mismatched. Cost is `n_per_curve * D` model runs.

Keep in mind: no `S2`, no bootstrap, and no `on_invalid="drop"`, since removing
a point from an ordered sweep changes what the estimator computes rather than
shrinking the sample. Pick eFAST when the run budget is the binding constraint
and you have measured that `N * D` beats `N'(D+2)` at the accuracy you need.
Otherwise run `sobol`, which gives you more.

## pce

Variance fractions read analytically off a fitted polynomial. Given data, and it
keeps a surrogate.

```python
X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n=4000, seed=0))
Y = ishigami.evaluate(X)

result = jaxgsa.pce.analyze(PROBLEM, X, Y, order=10)

result.explained_variance   # 1.0      <- read this first
result.loo_rmse             # 0.006    <- and this, against Y.std() = 3.692
result.S1                   # [0.3139, 0.4424, 0.0   ]
result.ST                   # [0.5576, 0.4424, 0.2437]

Y_pred = result.predict(X_new, batch_size=2048)
effects = result.shapley()
```

The order sweep is the whole story of this method:

```python
for order in (3, 6, 10):
    r = jaxgsa.pce.analyze(PROBLEM, X, Y, order=order, verbose=False)
    print(order, float(r.explained_variance), float(r.loo_rmse), r.S1)
```

```
3   0.463  2.72   [0.6622 0.0542 0.0006]   <- warns, and is nonsense
6   0.982  0.512  [0.3198 0.4416 0.0001]
10  1.000  0.006  [0.3139 0.4424 0.0   ]   <- matches the analytical values
```

At `order=3` the surrogate captured 46% of the variance and reported
`S1(x1) = 0.662` against a truth of 0.314. The indices are not noisy. They are
the correct indices of a cubic that is not this model.

Signature:

```python
jaxgsa.pce.analyze(problem, X, Y, *, order=3, ridge=1e-8, fit_ratio=0.5,
                   batch_size=None, n_bootstrap=0, conf_level=0.95,
                   ci_method="quantile", key=None, on_invalid="raise",
                   verbose=True, keep_replicates=False)
jaxgsa.pce.effective_order(problem, n_samples, *, order=3, fit_ratio=0.5)
```

Result: `S1`, `S2`, `ST`, `coefficients`, `multi_index`, `order`, `loo_rmse`,
`explained_variance`, `streamed`, `predict`, `shapley`.

Keep in mind:

- **Check the fit before you read the indices.** `analyze` warns when
  `explained_variance` falls below 0.5 on any output slice, and separately when
  `loo_rmse` passes 0.71 times `std(Y)`. A silent run means both passed.
  `indices()` warns about neither, because it traces.
- `explained_variance` is in-sample, so it climbs as you add terms and cannot
  detect an overfit on its own. `loo_rmse` turning back up is the overfit
  signal. Raise `order` until it stops falling.
- A total-degree basis at order p in D parameters has `C(D+p, p)` terms: 286 at
  D=3, p=10, but 3003 at D=10, p=5. `effective_order` returns the order the data
  supports, capped at the `order` you pass, and `analyze` drops to it and
  reports the drop as `result.order`. If it comes back at 1 or 2, PCE is not the
  method for this dataset.
- Polynomials ring around a step, and no `order` fixes it. For a discontinuity,
  threshold or hard saturation use HDMR's B-splines, or a method that fits no
  surrogate at all.
- Correlated inputs are refused, because the orthogonality that makes the
  coefficients readable as variances is orthogonality under the independent
  product measure.

## hdmr

A B-spline decomposition that reports per-term indices, with the ANCOVA split
into a structural share and a coupling share. Given data, and it keeps a
surrogate.

```python
result = jaxgsa.hdmr.analyze(PROBLEM, X, Y, maxorder=2)

result.terms   # ('x1', 'x2', 'x3', 'x1/x2', 'x1/x3', 'x2/x3')
result.Sa      # [0.3004, 0.3638, 0.0006, 0.0013, 0.2313, 0.0020]
result.Sb      # [0.0023, 0.0070, -0.0001, 0.0007, 0.0067, 0.0014]
result.S       # [0.3027, 0.3708, 0.0006, 0.0020, 0.2380, 0.0035]
result.S.sum() # 0.9174   <- read this first
result.ST      # [0.5426, 0.3762, 0.2420]
result.rmse    # 1.166

Y_pred = result.predict(X_new, batch_size=2048)
effects = result.shapley()
```

The per-term view is what HDMR is for. `x1/x3` carries 0.238 of the variance
while `x3` alone carries 0.0006, which names the interaction rather than only
implying it through an `ST - S1` gap.

`result.S1`, `S2` and `S3` re-expose the same terms in dense vector, matrix and
tensor layouts.

Signature:

```python
jaxgsa.hdmr.analyze(problem, X, Y, *, maxorder=2, maxiter=100, m=2,
                    lambdax=0.01, slice_chunk_size=None, batch_size=None,
                    n_bootstrap=0, conf_level=0.95, ci_method="quantile",
                    key=None, on_invalid="raise", verbose=True,
                    keep_replicates=False)
```

Keep in mind:

- **`result.S.sum()` must be near 1 before you rank anything.** The 0.9174 above
  means 8% of the variance is unaccounted for, and `Sa`, `Sb`, `S` and `ST` all
  inherit that shortfall. `analyze` warns below 0.5 and above 1.3, which is the
  Li et al. (2010, Eq. 24) precondition. Raise `maxorder` or `m`, or accept that
  this model does not decompose into low-order terms.
- **Under correlated inputs `ST` is not a total-effect index.** It is the SCSA
  term-membership sum: `Sa + Sb` over every term containing the parameter. It
  can be negative, is not bounded in [0, 1], and does not answer the
  parameter-fixing question. See `dependent-inputs.md`.
- Only the first-order components are backfitted, up to `maxiter` sweeps with an
  early stop once the coefficients settle. Every higher-order component is a
  single ridge solve.
- The surrogate is trained on the outputs you supply, so `predict` and `rmse`
  are on that same scale, with no inverse transform.
- Reach for HDMR over PCE when you want the per-term split, or when the response
  has kinks a polynomial cannot follow. Otherwise PCE fits in one linear solve
  and gives the same numbers with fewer knobs to get wrong.

## shapley

One fair-share number per parameter, summing to 1. Derived from a fitted PCE or
HDMR surrogate; there is no separate Shapley pipeline.

```python
# Canonical form: fit, then allocate.
result = jaxgsa.pce.analyze(PROBLEM, X, Y, order=10, verbose=False).shapley()

result.Sh                  # [0.4357, 0.4424, 0.1219]   sums to 1 exactly
result.Sh.sum()            # 1.0
result.S1                  # [0.3139, 0.4424, 0.0]
result.ST                  # [0.5576, 0.4424, 0.2437]
result.explained_variance  # 1.0    <- read this first

# The HDMR backend, on the same data.
jaxgsa.hdmr.analyze(PROBLEM, X, Y, maxorder=2, verbose=False).shapley()
# Sh [0.4633, 0.4063, 0.1304], explained_variance 0.8994

# One-step convenience wrapper.
jaxgsa.shapley.analyze(PROBLEM, X, Y, backend="pce", order=10)
```

Read the PCE row left to right. `x3` has `S1 ~ 0` and `ST = 0.244`, so
everything it does is an interaction. Shapley splits that interaction with `x1`
and hands `x3` half of it, 0.122. `S1` sums to 0.756, `ST` sums to 1.244, and
`Sh` sums to 1. The analytical Shapley effects for Ishigami are
`[0.4357, 0.4424, 0.1218]`.

Signature:

```python
jaxgsa.shapley.analyze(problem, X, Y, *, backend="pce",
                       include_correlative=False, n_bootstrap=0,
                       conf_level=0.95, ci_method="quantile", key=None,
                       on_invalid="raise", verbose=True,
                       keep_replicates=False, **backend_kwargs)
PCEResult.shapley()
HDMRResult.shapley(*, include_correlative=False)
```

Keep in mind:

- **`explained_variance` decides whether any of this is real.** The effects are
  exact within the surrogate, and a bad surrogate gives exact nonsense: at PCE
  `order=3` the same call returns `Sh = [0.8035, 0.0551, 0.1414]`. The backend's
  own `analyze` warns, so the warning arrives under `jaxgsa.pce` or
  `jaxgsa.hdmr` rather than `jaxgsa.shapley`. For the PCE backend the number is
  a coefficient of determination in [0, 1]; for HDMR it is the decomposed
  fraction and can exceed 1.
- Shapley is not the fixing measure. Splitting an interaction gives a share to
  both participants, so a parameter can be safe to fix and still carry a visible
  `Sh`. Use `ST` for that question.
- Shapley dissolves interactions into the participants by design. For the
  interaction itself read `S2` from `sobol` or `pce`, or HDMR's per-term `S`.
- Backend keyword arguments are forwarded unchecked, so `backend="pce",
  maxorder=3` raises a plain `TypeError` from `pce.analyze`.
- `VKOGAResult.shapley()` raises `NotImplementedError` on purpose, because a
  kernel expansion has no term-wise variance decomposition to allocate from.
- The surrogate-free Song et al. (2016) conditional-variance estimator is not
  implemented.
