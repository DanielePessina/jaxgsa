# HSIC (Kernel-Based Sensitivity Analysis)

HSIC (Hilbert-Schmidt Independence Criterion) computes sensitivity indices
using **kernel-based dependence measures** rather than variance decomposition.
It detects any form of statistical dependence — nonlinear, non-monotone,
heteroscedastic — by mapping inputs and outputs into reproducing kernel
Hilbert spaces.

The method produces two index types:

- **R2-HSIC** (first-order): normalized kernel dependence between each input
  and the output, analogous to a kernel correlation coefficient.
- **Total HSIC**: captures dependence through interactions, computed via
  complement product kernels.

When to use HSIC:

- You want a sensitivity measure that captures **any dependence**, not just
  variance-based effects.
- Your inputs may be **correlated** (HSIC works without independence assumptions).
- You have existing (X, Y) sample pairs and want sensitivity indices without
  additional model evaluations — HSIC is a **given-data** method.
- You want a third lens alongside Sobol (variance) and distribution-based
  methods (OT, PAWN).

## Import style

The HSIC module lives at `jaxgsa.hsic`:

```python
from jaxgsa import hsic
# hsic.analyze(...)
```

Note that `monte_carlo` is in `jaxgsa.sampling`, not in `jaxgsa.hsic`. It is
called as `jaxgsa.sampling.monte_carlo()`.

## Key difference from other methods

HSIC is a **given-data** method like RS-HDMR. It takes any (X, Y) sample pairs
— no special sampling design is required. Unlike Sobol indices which decompose
output variance, HSIC measures statistical dependence in a reproducing kernel
Hilbert space. This means:

- HSIC detects nonlinear, non-monotone, and heteroscedastic effects that
  Sobol indices may underweight.
- R2-HSIC indices do **not** sum to 1 (they are individual dependence
  measures, not variance fractions).
- The indices depend on the kernel bandwidth, which is set automatically
  via the median heuristic.

## Scalar example (Ishigami)

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM

# Generate Monte Carlo samples (any sampling works — no special design needed)
X = jaxgsa.sampling.monte_carlo(PROBLEM, n=2048, seed=42)
print("X shape:", X.shape)  # (2048, 3)

# Evaluate the model
Y = jaxgsa.benchmarks.ishigami.evaluate(jnp.asarray(X))
print("Y shape:", Y.shape)  # (2048,)

# Compute HSIC indices
result = jaxgsa.hsic.analyze(PROBLEM, jnp.asarray(X), Y)

print("R2_HSIC:", result.R2_HSIC)   # (D,) = (3,) — normalized first-order
print("T_HSIC:", result.T_HSIC)     # (D,) = (3,) — total-order
print("p_values:", result.p_values) # (D,) = (3,) — permutation p-values
print("hsic_raw:", result.hsic_raw) # (D,) = (3,) — unnormalized HSIC(Xi, Y)
```

## Multi-output example

When your model returns K outputs, the resulting index arrays have shape
`(K, D)`.

```python
import jax.numpy as jnp
import jaxgsa

problem = jaxgsa.Problem.from_dict(
    {
        "x1": (0.0, 1.0),
        "x2": (0.0, 1.0),
        "x3": (0.0, 1.0),
    },
    output_names=("linear", "quadratic"),
)

X = jaxgsa.sampling.monte_carlo(problem, n=2048, seed=42)
Xj = jnp.asarray(X)

# Two outputs: linear combination and sum of squares
Y = jnp.column_stack([
    Xj @ jnp.array([1.0, 2.0, 3.0]),
    jnp.sum(Xj**2, axis=1),
])

result = jaxgsa.hsic.analyze(problem, Xj, Y)

print("R2_HSIC shape:", result.R2_HSIC.shape)  # (K, D) = (2, 3)
print("T_HSIC shape:", result.T_HSIC.shape)    # (K, D) = (2, 3)
```

## xarray export

`HSICResult.to_dataset()` converts results to a labeled `xarray.Dataset`.

```python
ds = result.to_dataset()
print(ds)
# <xarray.Dataset>
# Dimensions:    (output: 2, param: 3)

print(ds.R2_HSIC.sel(param="x1"))
print(ds.p_values.sel(output="linear"))
```

For scalar output, the dataset has dimension `(param,)` only.

## Bandwidth control

By default, HSIC uses the **median heuristic** to set the Gaussian kernel
bandwidth — no tuning required. You can override with a fixed bandwidth for
convergence studies:

```python
# Default: median heuristic (recommended)
result = jaxgsa.hsic.analyze(problem, X, Y)

# Fixed bandwidth for sweep studies
result = jaxgsa.hsic.analyze(problem, X, Y, bandwidth=0.3)
```

## Permutation p-values

Statistical significance is computed via a permutation test with the
Phipson-Smyth correction. The number of permutations controls precision:

```python
# Default: 200 permutations (p-value resolution ≈ 0.005)
result = jaxgsa.hsic.analyze(problem, X, Y, n_perms=200, seed=42)

# Faster with fewer permutations
result = jaxgsa.hsic.analyze(problem, X, Y, n_perms=50, seed=42)
```

A small p-value (< 0.05) indicates that the input has a statistically
significant dependence with the output. The `seed` parameter ensures
reproducibility.

## Shape rules

| `Y` shape | R2_HSIC / T_HSIC / p_values / hsic_raw |
|---|---|
| `(N,)` | `(D,)` |
| `(N, K)` | `(K, D)` |
| `(N, T, K)` | `(T, K, D)` |

D is always the last axis of the index arrays.

## Practical caveats

- HSIC is **O(N²)** in computation and memory (kernel matrices). For N > 8000,
  use `batch_size` to limit peak memory.
- R2-HSIC indices do **not** sum to 1. They are individual dependence measures,
  not variance fractions.
- The total HSIC index uses product kernels across all D inputs. For very
  high D (> 15), the product kernel can underflow in float32.
- For outputs with large magnitude, set `prenormalize=True` to standardize
  Y before kernel computation.
- Inputs are automatically transformed to [0, 1] via their marginal CDF,
  ensuring comparable bandwidths across dimensions.

## See also

- [Basic Example](/examples/basic) for the Sobol workflow with structured
  Saltelli sampling.
- [DGSM](/examples/dgsm) for derivative-based sensitivity measures.
- [eFAST](/examples/efast) for frequency-based variance decomposition.
- [PCE](/examples/pce) for analytical Sobol indices from polynomial expansion
  coefficients.
- [Methods](/guide/methods) for the theory behind HSIC and when to choose it
  over other methods.
- [API Reference](/api/#given-data-methods) for full
  parameter documentation.
