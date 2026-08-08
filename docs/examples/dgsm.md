# DGSM (Derivative-based Global Sensitivity Measures)

By the end of this page you will have, for each input, a range that contains
its total Sobol index, at the cost of one differentiation pass per sample. The
total Sobol index `ST` is the share of output variance an input drives on its
own plus through every interaction it takes part in. Derivative-based global
sensitivity measures (DGSM) do not compute it. They bracket it from above and
below, which is often enough to decide which inputs to drop.

DGSM works from the partial derivatives of the model instead of from a variance
decomposition. For a JAX-differentiable model the derivatives are cheap. One
reverse-mode automatic differentiation pass (`jax.jacrev`) yields all of them
per sample.

The key quantities are the second moment of the partial derivative (importance
measure, nu) and the mean partial derivative (sigma). These yield two-sided
bounds on the total Sobol index ST via the Poincare upper bound and the
Kucherenko-Song lower bound.

When to use DGSM:

- Your model is JAX-differentiable and you want fast sensitivity screening.
- You need upper/lower bounds on total Sobol indices without full variance
  decomposition.
- You want to exploit reverse-mode autodiff to get all D partial derivatives
  in one pass.

## Import style

The DGSM module lives at `jaxgsa.dgsm`:

```python
from jaxgsa import dgsm
# dgsm.analyze(...)
```

`monte_carlo` is in `jaxgsa.sampling`, not in `jaxgsa.dgsm`. Call it as
`jaxgsa.sampling.monte_carlo()`.

## Key difference from other methods

DGSM requires an unbatched function with signature `(D,) -> ()` or
`(D,) -> (K,)`. This is different from the batched `evaluate(X)` functions
used by Sobol, HDMR, and eFAST which accept `(N, D)` input arrays.

The unbatched signature is needed because `jax.jacrev` differentiates a
single-input function. Internally, `jaxgsa.dgsm.analyze` vectorizes the autodiff
over all N samples.

## Scalar example (Ishigami)

There are three steps. First write the model as an unbatched function, because
that is the signature `jax.jacrev` differentiates. Then draw plain Monte Carlo
samples: DGSM averages over the input distribution and needs no structured
design. Then call `analyze`, which differentiates the function at every sample
and reduces the derivatives to the indices.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM

# Define an UNBATCHED function: (D,) -> ()
def ishigami(x):
    return jnp.sin(x[0]) + 7.0 * jnp.sin(x[1])**2 + 0.1 * x[2]**4 * jnp.sin(x[0])

# Generate Monte Carlo samples
X = jaxgsa.sampling.monte_carlo(PROBLEM, n=10000, seed=42)
print("X shape:", X.shape)  # (10000, 3)

# Compute DGSM indices
result = jaxgsa.dgsm.analyze(PROBLEM, ishigami, jnp.asarray(X))

print("nu:", result.nu)                # (D,) = (3,)
print("sigma:", result.sigma)          # (D,) = (3,)
print("upper_bound:", result.upper_bound)  # (D,) = (3,)
print("lower_bound:", result.lower_bound)  # (D,) = (3,)
print("var_y:", result.var_y)          # scalar
```

Read the two bounds as a bracket on `ST`, one entry per input. Where
`upper_bound` for an input is small, that input cannot have a large total
effect, and you may fix it at a nominal value. Where `lower_bound` is well
above zero, the input matters and no further argument is needed. An input whose
bracket is wide is undecided, and Sobol or eFAST will settle it.

`nu` and `sigma` are the raw derivative statistics the bounds are built from,
and `nu` also serves as an importance ranking on its own. `var_y` is the output
variance the bounds are expressed as a fraction of. A `var_y` near zero means
the model barely responds to any input, and the bounds are then ratios of two
small numbers and are not worth reading.

## Multi-output example

When your unbatched function returns K outputs `(D,) -> (K,)`, the resulting
index arrays have shape `(K, D)`.

```python
import jax.numpy as jnp
import jaxgsa

problem = jaxgsa.Problem.from_dict(
    {
        "x1": (-3.14159, 3.14159),
        "x2": (-3.14159, 3.14159),
        "x3": (-3.14159, 3.14159),
    },
    output_names=("output_a", "output_b"),
)


def multi_output_fn(x):
    """Unbatched: (3,) -> (2,)."""
    a = jnp.sin(x[0]) + 7.0 * jnp.sin(x[1])**2
    b = jnp.cos(x[0]) * x[2]
    return jnp.array([a, b])


X = jaxgsa.sampling.monte_carlo(problem, n=10000, seed=42)
result = jaxgsa.dgsm.analyze(problem, multi_output_fn, jnp.asarray(X))

print("nu shape:", result.nu.shape)            # (K, D) = (2, 3)
print("upper_bound shape:", result.upper_bound.shape)  # (K, D) = (2, 3)
```

Row 0 of each array belongs to `output_a` and row 1 to `output_b`, in the order
`output_names` declares. Here `x3` enters only `b`, so its bounds for
`output_a` are zero while its bounds for `output_b` are not. Every output gets
an independent bracket, and an input you may fix for one output may still
matter for another.

## xarray export

`DGSMResult.to_dataset()` converts results to a labeled `xarray.Dataset`.

```python
ds = result.to_dataset()
print(ds)
# <xarray.Dataset>
# Dimensions:      (output: 2, param: 3)

print(ds.upper_bound.sel(param="x1"))
print(ds.nu.sel(output="output_a"))
```

For scalar output, the dataset has dimension `(param,)` only.

## Shape rules

| `fn` signature | nu / sigma / upper / lower | var_y |
|---|---|---|
| `(D,) -> ()` | `(D,)` | `()` |
| `(D,) -> (K,)` | `(K, D)` | `(K,)` |

D is always the last axis of the index arrays.

## Practical caveats

- DGSM requires a JAX-differentiable function. If your model is not
  differentiable in JAX, you can pre-compute the Jacobian externally and pass
  `Y` and `dfdx` arrays directly to `jaxgsa.dgsm.analyze()`.
- The Poincare upper bound can be loose for strongly nonlinear or non-monotone
  responses. The bound becomes tight when the model is nearly monotone in a
  given input.
- For purely additive linear models, the upper and lower bounds collapse to
  the exact total Sobol index.
- The `batch_size` parameter controls batching of the autodiff to limit peak
  memory usage on large sample sets.

## See also

- [Basic Example](/examples/basic) for the Sobol workflow with structured
  Saltelli sampling.
- [eFAST](/examples/efast) for frequency-based variance decomposition.
- [PCE](/examples/pce) for analytical Sobol indices from polynomial expansion
  coefficients.
- [Methods](/guide/methods) for the theory behind DGSM and when to choose it
  over other methods.
- [API Reference](/api/#given-data-methods) for full parameter documentation.
