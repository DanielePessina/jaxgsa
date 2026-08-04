# PAWN (CDF-based Sensitivity)

PAWN is a distribution-based sensitivity method that measures how much
the entire output CDF shifts when an input is fixed. It uses the
Kolmogorov-Smirnov distance between unconditional and conditional output
distributions — no variance decomposition, no model assumptions.

When to use PAWN:

- You care about distributional changes beyond just variance (tail
  behavior, skewness shifts).
- You want a moment-independent index that is invariant under monotone
  output transformations.
- You have a set of (X, Y) pairs from any sampling strategy — no
  structured design required.

## Import style

```python
# Subpackage import
from jaxgsa import pawn
# pawn.analyze(...)

# Or top-level
import jaxgsa
# jaxgsa.pawn.analyze(...)
```

## Scalar example (Ishigami)

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

# Generate Monte Carlo samples
X = jaxgsa.sampling.monte_carlo(PROBLEM, n=5000, seed=42)
Y = evaluate(jnp.asarray(X))

# Compute PAWN indices (median KS across 10 bins)
result = jaxgsa.pawn.analyze(PROBLEM, jnp.asarray(X), Y)

print("PAWN:", result.pawn)  # (3,)
```

The PAWN index is the median (by default) KS distance across conditioning
bins. Higher values indicate stronger influence on the output distribution.

## Choosing the aggregation statistic

PAWN computes one KS distance per bin, then aggregates across bins. Three
options are available:

```python
# Median (default) — robust to outlier bins
r_med = jaxgsa.pawn.analyze(PROBLEM, X, Y, statistic="median")

# Maximum — captures worst-case shift
r_max = jaxgsa.pawn.analyze(PROBLEM, X, Y, statistic="max")

# Mean — simple average
r_mean = jaxgsa.pawn.analyze(PROBLEM, X, Y, statistic="mean")

print("median:", r_med.pawn)
print("max:   ", r_max.pawn)
print("mean:  ", r_mean.pawn)
```

## Bootstrap confidence intervals

```python
result = jaxgsa.pawn.analyze(
    PROBLEM, X, Y,
    n_bootstrap=100,
    conf_level=0.95,
    seed=0,
)

print("PAWN:", result.pawn)
print("95% CI lower:", result.pawn_conf[0])
print("95% CI upper:", result.pawn_conf[1])
```

## Running on all three benchmark functions

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks import ishigami, sobol_g, linear

# --- Ishigami ---
X_ish = jnp.asarray(jaxgsa.sampling.monte_carlo(ishigami.PROBLEM, n=5000, seed=42))
Y_ish = ishigami.evaluate(X_ish)
r_ish = jaxgsa.pawn.analyze(ishigami.PROBLEM, X_ish, Y_ish)
print("Ishigami PAWN:", r_ish.pawn)

# --- Sobol-G ---
X_sg = jnp.asarray(jaxgsa.sampling.monte_carlo(sobol_g.PROBLEM, n=5000, seed=42))
Y_sg = sobol_g.evaluate(X_sg)
r_sg = jaxgsa.pawn.analyze(sobol_g.PROBLEM, X_sg, Y_sg)
print("Sobol-G PAWN:", r_sg.pawn)

# --- Linear ---
X_lin = jnp.asarray(jaxgsa.sampling.monte_carlo(linear.PROBLEM, n=5000, seed=42))
Y_lin = linear.evaluate(X_lin)
r_lin = jaxgsa.pawn.analyze(linear.PROBLEM, X_lin, Y_lin)
print("Linear PAWN:", r_lin.pawn)
```

## Multi-output example

When Y has shape `(N, K)`, PAWN indices have shape `(K, D)`.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n=3000, seed=42))
Y1 = evaluate(X)
Y2 = jnp.sum(X**2, axis=1)
Y_multi = jnp.column_stack([Y1, Y2])

result = jaxgsa.pawn.analyze(PROBLEM, X, Y_multi)
print("PAWN shape:", result.pawn.shape)  # (2, 3)
```

## xarray export

```python
ds = result.to_dataset()
print(ds)

# With bootstrap CIs
result_ci = jaxgsa.pawn.analyze(PROBLEM, X, Y1, n_bootstrap=50, seed=0)
ds_ci = result_ci.to_dataset()
print(ds_ci)  # includes pawn_lower and pawn_upper
```

## Shape rules

| Y shape | pawn | pawn_conf |
|---|---|---|
| `(N,)` | `(D,)` | `(2, D)` or None |
| `(N, K)` | `(K, D)` | `(2, K, D)` or None |
| `(N, T, K)` | `(T, K, D)` | `(2, T, K, D)` or None |

## Practical caveats

- PAWN needs no structured sampling. Any (X, Y) pairs work, including
  Monte Carlo, Latin Hypercube, or Sobol sequences.
- The number of bins (`n_bins`) trades off conditioning resolution against
  sample density per bin. The default of 10 works well for N >= 1000.
- With very few samples per bin (< 10), the KS statistic becomes noisy.
  Increase N or decrease `n_bins`.
- The KS statistic is bounded in [0, 1] but sensitive to sample size —
  larger N gives sharper discrimination between inputs.
- Categorical inputs work. A categorical parameter needs no binning: its
  level code already names the conditioning class, so PAWN uses one bin
  per level and `n_bins` does not apply to it. Relabeling the levels does
  not change the index. See
  [Categorical Inputs](/examples/categorical-inputs).

## See also

- [Basic Example](/examples/basic) for the Sobol variance-decomposition
  workflow.
- [Methods](/guide/methods) for a comparison of all methods.
- [API Reference](/api/#given-data-methods) for full parameter documentation.
