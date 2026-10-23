# Differentiable Sobol indices in JAX

jaxgsa Sobol indices can sit inside a JAX computation. You can take the
gradient of an index with respect to the input distribution.

Two functions make this possible:

- `design.transform(theta)` maps the fixed unit-cube design to physical inputs.
  `theta` holds distribution parameters, for example `low` and `high`.
- `jaxgsa.sobol.indices(design, Y)` returns the indices as plain JAX arrays.

`analyze` is not differentiable. It checks values on the host and returns a
labelled result. Use `indices` inside `jax.grad`, `jax.jacrev` or `jax.jit`.

```python
import jax
import jax.numpy as jnp
import numpy as np

import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM

jax.config.update("jax_enable_x64", True)


def ishigami(X):
    x1, x2, x3 = X.T
    return jnp.sin(x1) + 7.0 * jnp.sin(x2) ** 2 + 0.1 * x3**4 * jnp.sin(x1)


design = jaxgsa.sobol.sample(PROBLEM, n_samples=1024, seed=0, verbose=False)
theta = {name: {"low": -np.pi, "high": np.pi} for name in PROBLEM.names}


def first_x1(parameters):
    Y = ishigami(design.transform(parameters))
    return jaxgsa.sobol.indices(design, Y).S1[0]


print("d S1[x1] / d high of x1:", jax.grad(first_x1)(theta)["x1"]["high"])
```

Rules to follow:

- Enable float64 first. Float32 rounding can dominate the derivative.
- The model must be written in JAX and be differentiable.
- Categorical inputs are rejected, because their inverse CDF is a step function.
- Check a gradient against a finite difference before you rely on it.

The [differentiation guide](/guide/differentiation) covers all methods, the
limits of each, and a longer workflow. The
[inverse Sobol example](/examples/inverse-sobol-bounds) fits input ranges to
target indices.
