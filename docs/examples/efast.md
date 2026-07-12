# eFAST (Extended FAST)

eFAST is a frequency-based variance decomposition that computes first-order
(S1) and total-order (ST) Sobol indices from Fourier amplitudes along
sinusoidal search curves. It does **not** produce second-order (S2) interaction
indices.

When to use eFAST instead of Sobol:

- You only need S1 and ST (no S2).
- You want a simpler, non-structured sampling design.
- You are screening a large number of parameters.

## Import style

The eFAST module lives at `gsax.efast`. You can import it directly or use the
top-level convenience aliases:

```python
# Subpackage import (preferred for eFAST-focused scripts)
from gsax import efast
# efast.sample(...)
# efast.analyze(...)

# Or use the top-level re-exports
import gsax
# gsax.sample_efast(...)
# gsax.analyze_efast(...)
```

## Scalar example (Ishigami)

```python
import jax.numpy as jnp
from gsax import efast
from gsax.benchmarks.ishigami import PROBLEM, evaluate

# Generate eFAST samples: N points per search curve, D curves total
X = efast.sample(PROBLEM, N=4096, M=4, seed=42)
print("X shape:", X.shape)  # (4096 * 3, 3) = (12288, 3)

# Evaluate the model
Y = evaluate(jnp.asarray(X))
print("Y shape:", Y.shape)  # (12288,)

# Compute eFAST indices
result = efast.analyze(PROBLEM, Y, M=4)

print("S1:", result.S1)  # (D,) = (3,)
print("ST:", result.ST)  # (D,) = (3,)
print("omega_0:", result.omega_0)
print("M:", result.M)
```

## Multi-output example

When your model returns K outputs per sample, pass Y with shape `(N*D, K)`.
The resulting indices have shape `(K, D)`.

```python
import jax.numpy as jnp
import gsax
from gsax import efast

problem = gsax.Problem.from_dict(
    {
        "amplitude": (0.5, 2.0),
        "frequency": (1.0, 5.0),
        "damping": (0.01, 0.5),
    },
    output_names=("displacement", "velocity"),
)


def multi_output_model(X):
    amp = X[:, 0]
    freq = X[:, 1]
    damping = X[:, 2]
    displacement = amp * jnp.sin(freq) * jnp.exp(-damping)
    velocity = amp * jnp.cos(freq) * jnp.exp(-damping)
    return jnp.stack([displacement, velocity], axis=-1)  # (N*D, K=2)


X = efast.sample(problem, N=4096, seed=42)
Y = multi_output_model(jnp.asarray(X))
print("Y shape:", Y.shape)  # (12288, 2)

result = efast.analyze(problem, Y, M=4)
print("S1 shape:", result.S1.shape)  # (K, D) = (2, 3)
print("ST shape:", result.ST.shape)  # (K, D) = (2, 3)
```

## Time-series example

When your model returns T time steps and K outputs, pass Y with shape
`(N*D, T, K)`. The resulting indices have shape `(T, K, D)`.

```python
import jax.numpy as jnp
import numpy as np
import gsax
from gsax import efast

problem = gsax.Problem.from_dict(
    {
        "amplitude": (0.5, 2.0),
        "frequency": (1.0, 5.0),
        "damping": (0.01, 0.5),
    },
    output_names=("displacement", "velocity"),
)

time_values = np.linspace(0.0, 5.0, 20)


def time_series_model(X):
    amp = X[:, 0, None]
    freq = X[:, 1, None]
    damping = X[:, 2, None]
    tt = jnp.asarray(time_values)[None, :]

    displacement = amp * jnp.sin(2 * jnp.pi * freq * tt) * jnp.exp(-damping * tt)
    velocity = amp * jnp.cos(2 * jnp.pi * freq * tt) * jnp.exp(-damping * tt)
    return jnp.stack([displacement, velocity], axis=-1)  # (N*D, T, K=2)


X = efast.sample(problem, N=4096, seed=42)
Y = time_series_model(jnp.asarray(X))
print("Y shape:", Y.shape)  # (12288, 20, 2)

result = efast.analyze(problem, Y, M=4)
print("S1 shape:", result.S1.shape)  # (T, K, D) = (20, 2, 3)
print("ST shape:", result.ST.shape)  # (T, K, D) = (20, 2, 3)
```

## xarray export

`EFASTResult.to_dataset()` converts results to a labeled `xarray.Dataset`,
just like the Sobol and HDMR result types.

```python
ds = result.to_dataset(time_coords=time_values)
print(ds)
# <xarray.Dataset>
# Dimensions:  (time: 20, output: 2, param: 3)

print(ds.S1.sel(param="amplitude"))
print(ds.ST.sel(output="velocity"))
```

## Shape rules

- `(N*D,)` means scalar output.
- `(N*D, K)` means K output variables with no time dimension.
- `(N*D, T, K)` means T time steps and K outputs.
- Without `problem.output_names`, a 2D array is always treated as `(N*D, K)`.
- With exactly one entry in `problem.output_names`, a 2D array is treated as
  `(N*D, T)` — timepoints of that single output — and flows through as
  `(N*D, T, 1)`. Passing a pre-reshaped `(N*D, T, 1)` array also works.

| Y shape | S1 / ST shape |
|---------|---------------|
| `(N*D,)` | `(D,)` |
| `(N*D, K)` | `(K, D)` |
| `(N*D, T, K)` | `(T, K, D)` |

D is always the last axis.

## Practical caveats

- eFAST does **not** produce S2 (second-order) indices. Use the Sobol workflow
  if you need pairwise interaction estimates.
- The `M` parameter (interference factor) must match between `sample()` and
  `analyze()`. The default is 4 for both.
- `N` must satisfy `N > 4*M^2` (i.e. `N > 64` with the default `M=4`).
- Indices outside `[0, 1]` indicate insufficient samples or near-zero output
  variance.

## See also

- [Basic Example](/examples/basic) for the Sobol workflow with structured
  Saltelli sampling and S2 support.
- [Multi-Output & Time-Series](/examples/multi-output) for the same shape
  conventions on Sobol and HDMR.
- [xarray Labeled Output](/examples/xarray) for named access by parameter,
  output, and time coordinate.
- [Methods](/guide/methods) for the theory behind eFAST and when to choose it
  over other methods.
