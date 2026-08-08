# PCE (Polynomial Chaos Expansion)

This page fits a polynomial surrogate to a set of model runs and reads Sobol
sensitivity indices straight off it. You finish with first-order, total-order
and second-order indices for the Ishigami test function, plus a cheap stand-in
model you can evaluate at new input points.

Polynomial Chaos Expansion fits an orthogonal polynomial surrogate to model
data and extracts Sobol indices analytically from the expansion coefficients
(Sudret, 2008). A surrogate is a cheap function fitted to the model's inputs
and outputs and used in its place. Sobol indices split the output variance
into the share attributable to each input: S1 covers an input acting alone,
ST covers it acting alone or in any interaction, and S2 covers a single pair
acting together. Because the indices fall out directly from the fitted
polynomial, PCE avoids the need for structured Saltelli sampling or variance
decomposition. Saltelli sampling is the specific two-matrix design that the
classical Sobol estimator requires.

When to use PCE:

- You have existing `(X, Y)` data and want Sobol indices without re-sampling.
- You want S1, ST, and S2 from a single surrogate fit.
- You want a cheap emulator for prediction at new input points.
- Your model is smooth enough that a low-order polynomial approximation is
  accurate.

## Import style

The PCE module lives at `jaxgsa.pce`:

```python
from jaxgsa import pce
# pce.analyze(...)
```

## Basic example (Ishigami)

The example below uses the Ishigami function, a three-input benchmark that
ships with jaxgsa. It runs in four steps.

1. Draw 2000 uniform samples inside the problem's bounds. PCE fits a
   least-squares surrogate, so it only needs points that cover the input
   space; any design works.
2. Run the model on those samples. This is the one expensive stage, and it
   happens exactly once because the indices come from the fit rather than
   from extra model calls.
3. Fit the surrogate with `order=4`. The order caps the total polynomial
   degree, so it sets how much curvature and interaction the surrogate can
   represent.
4. Print the indices together with `loo_rmse`. The indices are only as good
   as the fit, so the error estimate belongs next to them.

```python
import jax
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

# Generate samples (any design works -- PCE does not need Saltelli structure)
key = jax.random.PRNGKey(42)
bounds = jnp.array(PROBLEM.bounds)
X = jax.random.uniform(
    key, (2000, PROBLEM.num_vars),
    minval=bounds[:, 0], maxval=bounds[:, 1],
)
Y = evaluate(X)

# Fit PCE and extract Sobol indices
result = jaxgsa.pce.analyze(PROBLEM, X, Y, order=4)

print("S1:", result.S1)          # (D,) = (3,)
print("ST:", result.ST)          # (D,) = (3,)
print("S2:", result.S2)          # (D, D) = (3, 3)
print("order:", result.order)    # effective polynomial degree used
print("LOO RMSE:", result.loo_rmse)
```

D is the number of inputs, so all three printed index arrays are sized by the
problem: one S1 and one ST value per input, and a 3x3 matrix holding one S2
value per input pair. Compare the printed `order` against the 4 that was
requested. A smaller value means the fit was auto-reduced, and the section
below explains when that happens.

## Expansion details

The fitted PCE stores its internal state for inspection:

```python
print("coefficients:", result.coefficients.shape)  # (n_terms,)
print("multi_index:", result.multi_index.shape)     # (n_terms, D)
print("order:", result.order)                       # effective degree
print("loo_rmse:", result.loo_rmse)                 # leave-one-out RMSE
```

- `coefficients` are the fitted weights for each polynomial basis term.
- `multi_index` maps each term to its polynomial degrees per dimension.
- `order` is the effective total degree (may be less than requested if
  auto-reduced to prevent overfitting).
- `loo_rmse` is a leave-one-out cross-validation RMSE computed cheaply from
  the hat matrix, without refitting. Leave-one-out cross-validation scores
  each sample using a fit that excludes it, which measures how the surrogate
  behaves on data it did not see.

The first two shapes share `n_terms`, the number of polynomial basis terms in
the expansion. Every term has one coefficient and one row of per-input
degrees.

## Emulation

`PCEResult.predict()` predicts at new input points using the fitted
polynomial. This is the emulator: it replaces the model with the surrogate,
so new predictions cost a polynomial evaluation instead of a model run.

```python
X_new = jax.random.uniform(
    jax.random.PRNGKey(99), (100, PROBLEM.num_vars),
    minval=bounds[:, 0], maxval=bounds[:, 1],
)
Y_pred = result.predict(X_new)
print("Y_pred shape:", Y_pred.shape)  # (100,)

# Compare against true model
Y_true = evaluate(X_new)
print("Max error:", jnp.max(jnp.abs(Y_pred - Y_true)))
```

The 100 predicted values line up one-to-one with the 100 new input rows. The
printed maximum error is the largest gap between the surrogate and the real
model over those 100 points, so it is a worst case rather than an average.
Read it against the scale of `Y` itself, and against `loo_rmse` from the fit.

## xarray export

`PCEResult.to_dataset()` converts results to a labeled `xarray.Dataset` with
`param` coordinates. Labeling means you select an input by name instead of by
position.

```python
ds = result.to_dataset()
print(ds)
# <xarray.Dataset>
# Dimensions:  (param: 3, param_i: 3, param_j: 3)

print(ds.S1.sel(param="x1"))
print(ds.ST)
print(ds.loo_rmse)  # included when available
```

The dataset carries three dimensions of length 3 because the problem has
three inputs. S1 and ST are indexed by `param`. S2 is a matrix, so it needs
the two separate names `param_i` and `param_j` for its rows and columns.

## Limitations

- Multi-output `(N, K)` and time-series `(N, T, K)` `Y` are supported, but all
  output slices share a single polynomial basis and a single effective `order`
  (they are fitted together in one solve).
- The polynomial order is automatically reduced when the term count would
  exceed `fit_ratio * N` to prevent overfitting.
- Uniform and truncated-Gaussian inputs use Legendre polynomials; untruncated
  Gaussian inputs use Hermite polynomials.

## See also

- [Basic Example](/examples/basic) for the Sobol workflow with structured
  Saltelli sampling.
- [RS-HDMR](/examples/hdmr) for another surrogate-based approach (B-spline
  expansion with ANCOVA decomposition).
- [DGSM](/examples/dgsm) for derivative-based sensitivity bounds.
- [Methods](/guide/methods) for the theory behind PCE and when to choose it
  over other methods.
- [API Reference](/api/#given-data-methods) for full parameter documentation.
