# Optimal Transport (Wasserstein-Based Sensitivity)

This page runs an optimal-transport sensitivity analysis on the Ishigami test
function, a standard three-input benchmark. You finish with one index per
input, split into the part that comes from shifting the output distribution
and the part that comes from reshaping it.

The optimal-transport index measures how far knowing an input moves the
entire output distribution. It is the class-averaged squared
2-Wasserstein distance between the output distribution conditional on the
input and the unconditional one, normalized by twice the output variance
so it lies in [0, 1] (Borgonovo, Figalli, Plischke & Savaré, 2024). The
2-Wasserstein distance is the smallest mean squared distance needed to move
one distribution onto another. This page calls the whole quantity the OT
index.

A moment-independent index reads the whole output distribution rather than
its variance alone. PAWN and the Borgonovo delta are the other two
moment-independent indices in jaxgsa. Unlike those two, every OT index
splits exactly into two parts:

- **advective** — the location shift of the conditional distribution.
  This equals half the first-order Sobol index. The first-order Sobol index
  (S1) is the share of output variance removed by fixing that one input.
- **diffusive** — everything else: changes in spread, tails, and shape.

An input with a large advective part moves the output; one with a large
diffusive part reshapes it. Ishigami's `x3` below is a concrete example
of the difference.

When to use optimal transport:

- You want a distributional index that stays comparable to variance-based
  results (`2 * advective == S1`).
- You want one number per input for a whole trajectory or multivariate
  output (the `multivariate` and `trajectory` modes) instead of per-time
  indices.
- Your inputs have mixed marginals (uniform + Gaussian) or are
  correlated. Conditioning is rank-based, so marginals never enter, and
  the index is well-defined under dependence (it then measures total,
  correlation-inclusive influence).
- You have any set of (X, Y) pairs. No structured design is required.

## Import style

```python
# Subpackage import
from jaxgsa import optimal_transport
# optimal_transport.analyze(...)

# Or top-level
import jaxgsa
# jaxgsa.optimal_transport.analyze(...)
```

## Scalar example (Ishigami)

The code below does four things.

1. Import the Ishigami benchmark. It ships both the input definition
   (`PROBLEM`) and the model itself (`evaluate`), so no external model is
   needed to follow along.
2. Draw 8192 Monte Carlo samples of the inputs. Optimal transport compares
   sample distributions, so it needs enough points per input class to make
   those distributions meaningful.
3. Run the model on those samples to get the output array `Y`. The estimator
   works from plain (X, Y) pairs, which is why any sampler is allowed here.
4. Call `analyze` and print the total index next to its two parts. Printing
   the parts, not only the total, is what lets you tell a shift apart from a
   reshape.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jaxgsa.sampling.monte_carlo(PROBLEM, n=8192, seed=42)
Y = evaluate(jnp.asarray(X))

result = jaxgsa.optimal_transport.analyze(PROBLEM, jnp.asarray(X), Y)

print("ot:       ", result.ot)         # (3,) total index
print("advective:", result.advective)  # location-shift part
print("diffusive:", result.diffusive)  # spread/shape part
```

On Ishigami, `x3` has a first-order Sobol index of exactly zero: it acts
only through an interaction with `x1`. Its advective component is
correspondingly ~0, but its diffusive component is clearly positive.
Fixing `x3` reshapes the output distribution without moving its mean,
and the decomposition shows that directly.

## How it works

For each input, the sample is split into `n_partitions` equal-frequency
classes by the input's rank, so any monotone transformation of an input
changes nothing. Per class, the squared 2-Wasserstein distance between
the class's conditional output sample and the full sample is computed.
The index is the class-size-weighted average, divided by `2 * Var(Y)`,
the theoretical maximum of that average, which pins the scale to [0, 1].

In the default `mode="univariate"`, each output column uses the exact
closed form of 1-D optimal transport (sorted-quantile coupling), with no
iterative solver at all. The `multivariate` and `trajectory` modes
transport point clouds. They use a pure-JAX log-domain Sinkhorn solver,
an iterative method for transport problems with no closed form.

## Multi-output and time series

All jaxgsa output shapes work, and `mode` chooses the granularity. Here N is
the number of samples, T the number of timepoints, K the number of outputs,
and D the number of inputs.

```python
Y2 = jnp.stack([Y, Y**2], axis=1)        # (N, K=2)
Y3 = jnp.stack([Y2, Y2 + 1.0], axis=1)   # (N, T=2, K=2)

# Per-column indices (default): (K, D) / (T, K, D)
r_uni = jaxgsa.optimal_transport.analyze(PROBLEM, X, Y3)
print(r_uni.ot.shape)  # (2, 2, 3)

# One index per input over the joint output distribution: (D,)
r_mv = jaxgsa.optimal_transport.analyze(PROBLEM, X, Y2, mode="multivariate")
print(r_mv.ot.shape)  # (3,)

# One index per input per output, over each output's whole time course: (K, D)
r_traj = jaxgsa.optimal_transport.analyze(PROBLEM, X, Y3, mode="trajectory")
print(r_traj.ot.shape)  # (2, 3)
```

Read the three printed shapes against the mode that produced them. The
default mode returns `(2, 2, 3)`: one index per timepoint, per output, per
input. `multivariate` collapses the two output columns into a single joint
distribution and returns `(3,)`, one index per input. `trajectory` keeps the
two outputs apart but collapses time, so it returns `(2, 3)`.

`trajectory` fits time-course models (a concentration curve per output,
say): each input gets one index per output summarizing its influence on
the entire curve at once, including temporal correlation, rather than a
per-timepoint average.

In the `multivariate` and `trajectory` modes, outputs are standardized
per column by default (`standardize_outputs=True`) so no output dominates the
transport cost through its units.

## The entropic floor and the dummy baseline

The point-cloud modes use entropic (Sinkhorn) transport. Together with
plain finite-sample noise, this keeps the index of even a totally
irrelevant input strictly above zero. Don't compare these indices
against 0; compare them against an irrelevance baseline:

```python
result = jaxgsa.optimal_transport.analyze(
    PROBLEM, X, Y2, mode="multivariate", dummy=True
)
print("indices:  ", result.ot)
print("baseline: ", result.ot_dummy)  # index of a synthetic, independent input
```

`dummy=True` runs one synthetic input (independent of the output by
construction) through the same estimator. Inputs whose index is not
clearly above `ot_dummy` are indistinguishable from noise. The `epsilon`
parameter trades entropic bias against solver iterations: smaller means
less bias but more iterations.

## Bootstrap confidence intervals

Bootstrapping resamples the (X, Y) pairs with replacement and re-estimates
the index on each resample. The spread of those estimates gives the interval.

```python
result = jaxgsa.optimal_transport.analyze(
    PROBLEM, X, Y, n_bootstrap=200, conf_level=0.95, seed=0
)
print(result.ot_conf)         # (2, 3): [lower, upper]
print(result.advective_conf)  # same for each component
```

The `(2, 3)` shape is a lower and an upper bound for each of the three
inputs, not two outputs. Keep `n_bootstrap` modest in the point-cloud modes:
each replicate re-solves `D * n_partitions` transport problems.

## xarray export

`to_dataset()` returns the indices as a labeled `xarray.Dataset`, with the
input names and output names attached as coordinates.

```python
ds = result.to_dataset()
# variables: ot, advective, diffusive (+ *_lower/*_upper, ot_dummy)
# dims: (param,) / (output, param) / (time, output, param); attrs: mode
```

## References

- Borgonovo, E., Figalli, A., Plischke, E., & Savaré, G. (2024). Global
  sensitivity analysis via optimal transport. *Management Science*.
  doi:10.1287/mnsc.2023.01796
