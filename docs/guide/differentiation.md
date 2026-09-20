# Differentiating sensitivity analyses

jaxgsa uses differentiation in two distinct ways:

1. **DGSM differentiates the model.** It uses model derivatives to measure
   sensitivity across the input distribution.
2. **Traceable estimator cores differentiate the analysis.** They let you ask
   how a reported index changes when the model outputs, input data, or input
   distribution changes.

These answer different questions. Use DGSM when derivatives are the
sensitivity measure you want. Differentiate an estimator core when the
sensitivity index itself is part of a larger JAX computation.

## Differentiating the model with DGSM

Derivative-based global sensitivity measures (DGSM) summarize the model's
local derivatives over a distribution of inputs. Given a one-sample function
`fn`, `dgsm.analyze` can compute its Jacobian with JAX:

```python
import jax.numpy as jnp
import jaxgsa


def model(x):
    return jnp.sin(x[0]) + x[1] ** 2


problem = jaxgsa.Problem.from_dict({"x1": (-1.0, 1.0), "x2": (-1.0, 1.0)})
X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, n=2048, seed=0))
result = jaxgsa.dgsm.analyze(problem, model, X)
```

The function passed to DGSM maps **one input row** to one scalar, a vector, or
an array of outputs. DGSM batches that function over `X` and chooses an
autodiff direction from the shapes: forward mode when the number of output
slices exceeds the number of inputs, and reverse mode otherwise.

If the model is not written in JAX, compute the outputs and Jacobians
elsewhere and pass `Y=` and `dfdx=` instead. This is still DGSM; only the
source of the derivatives changes. See the [DGSM API](/api/dgsm) for shapes,
bounds, and the precomputed-derivative path.

## Differentiating an estimator

For ordinary analysis, use a method's `analyze()` function. It validates the
data, applies the non-finite policy, emits warnings, computes optional
confidence intervals, and returns a labelled result object.

Eleven methods also expose a smaller `indices()` function. It runs the same
estimator on clean inputs but returns bare JAX arrays and performs no
data-dependent host-side diagnostics. This makes it suitable for composition
with transformations such as `jax.jit`, `jax.vmap`, `jax.grad`, `jax.jacfwd`,
and `jax.jacrev`.

| Method | Traceable core | Differentiation note |
| --- | --- | --- |
| Sobol' | `jaxgsa.sobol.indices` | Reverse and forward mode |
| eFAST | `jaxgsa.efast.indices` | Reverse and forward mode |
| Morris | `jaxgsa.morris.indices` | Reverse and forward mode |
| DGSM | `jaxgsa.dgsm.indices` | May differentiate through the model Jacobian |
| HDMR | `jaxgsa.hdmr.indices` | **Forward mode only** |
| PCE | `jaxgsa.pce.indices` | Reverse and forward mode |
| Shapley | `jaxgsa.shapley.indices` | Backend-dependent; see below |
| HSIC | `jaxgsa.hsic.indices` | Reverse and forward mode |
| PAWN | `jaxgsa.pawn.indices` | Traceable, but gradients need care |
| Borgonovo | `jaxgsa.borgonovo.indices` | Reverse and forward mode |
| Optimal transport | `jaxgsa.optimal_transport.indices` | Reverse and forward mode |
| Kucherenko | None | Host NumPy implementation |
| VKOGA | None | Host quasi-Monte Carlo index stage |

“Traceable” is a software property, not a promise that every mathematical
derivative is informative. Discrete choices, ranks, hard bins, and extrema
can make a function piecewise constant or nondifferentiable even when JAX can
trace and compile it.

### Why `indices()` is separate

An analysis wrapper sometimes has to inspect a concrete value: for example,
to decide whether to warn about zero output variance or drop a non-finite
sample. A JAX tracer has no concrete value at tracing time. Keeping those
decisions in `analyze()` and the numerical estimator in `indices()` gives the
two entry points clear contracts:

- use `analyze()` for results you will report;
- use `indices()` inside a transformed computation, after validating inputs.

The core does not replace validation. A NaN that would be caught or described
by `analyze()` may silently propagate through `indices()`. Surrogate cores also
omit fit-quality diagnostics, so check a normal `analyze()` result before
using their derivatives.

## End-to-end Sobol' derivatives

A Sobol' design retains its points in the unit cube. `SobolSamples.transform`
pushes those fixed points through the declared marginal distributions using
JAX. Combining it with a JAX model and `sobol.indices` gives a differentiable
path from distribution parameters to sensitivity indices:

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


samples = jaxgsa.sobol.sample(PROBLEM, n_samples=8192, seed=0)
theta = {
    name: {"low": -np.pi, "high": np.pi}
    for name in PROBLEM.names
}


def first_order(parameters):
    X = samples.transform(parameters)
    Y = ishigami(X)
    S1, _ = jaxgsa.sobol.indices(samples, Y)
    return S1


dS1 = jax.jacrev(first_order)(theta)
print(dS1["x1"]["high"])
```

Each leaf of `dS1` is a vector over all reported first-order indices. Changing
one distribution bound can move every index because they share the same
output variance.

This derivative uses a reparameterisation: the unit-cube design stays fixed
while the inverse CDF changes with `theta`. The model must therefore be
JAX-differentiable. Categorical inputs are rejected because their inverse CDF
is a step function and has no useful pathwise derivative.

Enable JAX float64 before relying on these derivatives. Sobol' indices combine
reductions and ratios whose derivatives can be dominated by float32 rounding.

`transform()` accepts partial overrides, so an optimization can vary one
bound, mean, or variance while leaving every omitted value at the setting in
the `Problem`. Uniform parameters use `low` and `high`; Gaussian parameters
use `mean` and `variance`, with optional truncation bounds.

## Method-specific limits

### HDMR and HDMR-backed Shapley

HDMR's early-stopping backfitting loop blocks reverse-mode differentiation.
Use `jax.jacfwd` with `jaxgsa.hdmr.indices`.

The same rule applies to
`jaxgsa.shapley.indices(..., backend="hdmr")`, because that path allocates the
HDMR decomposition. The default PCE-backed Shapley core supports both forward
and reverse mode.

### DGSM through DGSM

Differentiating `dgsm.indices` can nest an outer derivative around the
Jacobian that DGSM already takes of the model. The model must then be twice
differentiable along that path. If you only want ordinary DGSM values, this
extra requirement does not apply.

### PAWN

`pawn.indices` can be traced and compiled, but PAWN is built from hard
conditioning bins and empirical CDF comparisons. Gradients with respect to
`X` or `Y` are therefore commonly zero between bin or rank changes and
undefined at those changes. Treating such an autodiff result as a smooth
measure of how PAWN changes is usually misleading.

Traceability remains useful for placing PAWN in a compiled or vectorized
pipeline. If derivatives are the goal, prefer an estimator whose arithmetic
is smooth for the variable you intend to perturb, or validate the proposed
derivative against finite perturbations that can cross the relevant bins.

### Kucherenko and VKOGA

Kucherenko is implemented with host NumPy, and VKOGA uses a host-side
quasi-Monte Carlo loop for its index stage. Neither exports `indices()`, so
their analysis cannot be placed inside JAX transformations. Their result
values can still be used as ordinary inputs to later JAX computations; the
gradient simply cannot pass back through the sensitivity analysis itself.

## A practical workflow

1. Run `analyze()` once on representative data and inspect its warnings and
   diagnostics.
2. Reproduce the fields you need with `indices()` on the same clean data.
3. Wrap only the smallest calculation needed by `jit`, `vmap`, or an autodiff
   transform.
4. Use float64 and compare important gradients with finite perturbations.
5. Re-run `analyze()` after optimization or calibration, because the new
   point may have different validity or surrogate-fit diagnostics.

The per-method API pages document each core's signature and returned tuple.
Start with the [API overview](/api/) or the [Sobol' API](/api/sobol) for the
complete reparameterised-design example.
