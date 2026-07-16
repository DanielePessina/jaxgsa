# PCE (Polynomial Chaos Expansion)

Polynomial Chaos Expansion fits an orthogonal polynomial surrogate to model
data and extracts Sobol indices analytically from the expansion coefficients
(Sudret, 2008). This avoids the need for structured Saltelli sampling or
variance decomposition -- the indices fall out directly from the fitted
polynomial.

When to use PCE:

- You have existing `(X, Y)` data and want Sobol indices without re-sampling.
- You want S1, ST, and S2 from a single surrogate fit.
- You want a cheap emulator for prediction at new input points.
- Your model is smooth enough that a low-order polynomial approximation is
  accurate.

## Import style

The PCE module lives at `gsax.pce`. You can import it directly or use the
top-level convenience aliases:

```python
# Subpackage import (preferred for PCE-focused scripts)
from gsax import pce
# pce.analyze(...)
# pce.emulate(...)

# Or use the top-level re-exports
import gsax
# gsax.analyze_pce(...)
# gsax.emulate_pce(...)
```

## Basic example (Ishigami)

```python
import jax
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

# Generate samples (any design works -- PCE does not need Saltelli structure)
key = jax.random.PRNGKey(42)
bounds = jnp.array(PROBLEM.bounds)
X = jax.random.uniform(
    key, (2000, PROBLEM.num_vars),
    minval=bounds[:, 0], maxval=bounds[:, 1],
)
Y = evaluate(X)

# Fit PCE and extract Sobol indices
result = gsax.analyze_pce(PROBLEM, X, Y, order=4)

print("S1:", result.S1)          # (D,) = (3,)
print("ST:", result.ST)          # (D,) = (3,)
print("S2:", result.S2)          # (D, D) = (3, 3)
print("order:", result.order)    # effective polynomial degree used
print("LOO RMSE:", result.loo_rmse)
```

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
  the hat matrix, without refitting.

## Emulation

`emulate_pce()` predicts at new input points using the fitted polynomial:

```python
X_new = jax.random.uniform(
    jax.random.PRNGKey(99), (100, PROBLEM.num_vars),
    minval=bounds[:, 0], maxval=bounds[:, 1],
)
Y_pred = gsax.emulate_pce(result, X_new)
print("Y_pred shape:", Y_pred.shape)  # (100,)

# Compare against true model
Y_true = evaluate(X_new)
print("Max error:", jnp.max(jnp.abs(Y_pred - Y_true)))
```

Predictions are batched over rows of `X_new` automatically so very large
prediction sets stay within a fixed transient-memory budget; pass
`batch_size=` to control the batch length explicitly.

## xarray export

`PCEResult.to_dataset()` converts results to a labeled `xarray.Dataset` with
`param` coordinates.

```python
ds = result.to_dataset()
print(ds)
# <xarray.Dataset>
# Dimensions:  (param: 3, param_i: 3, param_j: 3)

print(ds.S1.sel(param="x1"))
print(ds.ST)
print(ds.loo_rmse)  # included when available
```

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
- [API Reference](/api/#pce-workflow) for full parameter documentation.
