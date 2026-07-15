# Optimal Transport (Wasserstein-Based Sensitivity)

The optimal-transport index measures how far knowing an input moves the
**entire output distribution**: it is the class-averaged squared
2-Wasserstein distance between the output distribution conditional on the
input and the unconditional one, normalized by twice the output variance
so it lives on a fixed [0, 1] scale (Borgonovo, Figalli, Plischke &
Savaré, 2024).

What sets it apart from the other moment-independent indices (PAWN,
Borgonovo delta) is the built-in **decomposition**: every index splits
exactly into

- an **advective** component — the pure location shift of the conditional
  distribution, which equals *half the first-order Sobol index*, and
- a **diffusive** component — everything else: changes in spread, tails
  and shape.

So the OT analysis tells you not only *how much* an input matters
distributionally, but *how* — by moving the output or by reshaping it.

When to use optimal transport:

- You want a distribution-aware index that still connects rigorously to
  the variance-based world (`2 * advective == S1`).
- You want one physically meaningful number per input for a whole
  **trajectory or multivariate output** (the joint modes), not per-time
  indices.
- Your inputs have **mixed marginals** (uniform + Gaussian) or are
  **correlated** — conditioning is rank-based and distribution-free, and
  the index is well-defined under dependence (it then measures total,
  correlation-inclusive influence).
- You have any set of (X, Y) pairs — no structured design required.

## Import style

```python
# Subpackage import
from gsax import optimal_transport
# optimal_transport.analyze(...)

# Or top-level
import gsax
# gsax.analyze_optimal_transport(...)
```

## Scalar example (Ishigami)

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

X = gsax.sample_mc(PROBLEM, N=8192, seed=42)
Y = evaluate(jnp.asarray(X))

result = gsax.analyze_optimal_transport(PROBLEM, jnp.asarray(X), Y)

print("ot:       ", result.ot)         # (3,) total index
print("advective:", result.advective)  # location-shift part
print("diffusive:", result.diffusive)  # spread/shape part
```

On Ishigami, `x3` has a first-order Sobol index of exactly zero — it acts
only through an interaction with `x1`. Its advective component is
correspondingly ~0, but its diffusive component is clearly positive:
fixing `x3` reshapes the output distribution without moving its mean.
The decomposition makes that mechanism visible directly.

## How it works

For each input, the sample is split into `n_partitions` equal-frequency
classes by the input's **rank** (so any monotone transformation of an
input changes nothing). Per class, the squared 2-Wasserstein distance
between the class's conditional output sample and the full sample is
computed; the index is the class-size-weighted average, divided by
`2 * Var(Y)` — the theoretical maximum of that average, which pins the
scale to [0, 1].

In the default `mode="separate"`, each output column uses the exact
closed form of 1-D optimal transport (sorted-quantile coupling) — no
iterative solver at all. The joint modes transport point clouds with a
pure-JAX log-domain Sinkhorn solver.

## Multi-output and time series

All gsax output shapes work, and `mode` chooses the granularity:

```python
Y2 = jnp.stack([Y, Y**2], axis=1)        # (N, K=2)
Y3 = jnp.stack([Y2, Y2 + 1.0], axis=1)   # (N, T=2, K=2)

# Per-column indices (default): (K, D) / (T, K, D)
r_sep = gsax.analyze_optimal_transport(PROBLEM, X, Y3)
print(r_sep.ot.shape)  # (2, 2, 3)

# One index per input over the *joint* output distribution: (D,)
r_joint = gsax.analyze_optimal_transport(PROBLEM, X, Y2, mode="joint")
print(r_joint.ot.shape)  # (3,)

# One index per input per output, over each output's whole time course: (K, D)
r_traj = gsax.analyze_optimal_transport(PROBLEM, X, Y3, mode="joint-over-time")
print(r_traj.ot.shape)  # (2, 3)
```

`joint-over-time` is the natural mode for time-course models (e.g. a
concentration curve per output): each input gets one index per output
summarizing its influence on the entire trajectory *jointly*, including
temporal correlation — not just a per-timepoint average.

In the joint modes, outputs are standardized per column by default
(`standardize=True`) so no output dominates the transport cost through
its units.

## The entropic floor and the dummy baseline

The joint modes use entropic (Sinkhorn) transport, which — together with
plain finite-sample noise — keeps the index of even a *totally
irrelevant* input strictly above zero. Do not compare joint indices
against 0; compare them against an irrelevance baseline:

```python
result = gsax.analyze_optimal_transport(
    PROBLEM, X, Y2, mode="joint", dummy=True
)
print("indices:  ", result.ot)
print("baseline: ", result.ot_dummy)  # index of a synthetic, independent input
```

`dummy=True` pushes one synthetic input (independent of the output by
construction) through the identical pipeline; inputs whose index is not
clearly above `ot_dummy` are indistinguishable from noise. The `epsilon`
parameter trades entropic bias against solver iterations (smaller = less
bias, slower).

## Bootstrap confidence intervals

```python
result = gsax.analyze_optimal_transport(
    PROBLEM, X, Y, n_bootstrap=200, conf_level=0.95, seed=0
)
print(result.ot_conf)         # (2, 3): [lower, upper]
print(result.advective_conf)  # same for each component
```

Keep `n_bootstrap` modest in the joint modes — each replicate re-solves
`D * n_partitions` transport problems.

## xarray export

```python
ds = result.to_dataset()
# variables: ot, advective, diffusive (+ *_lower/*_upper, ot_dummy)
# dims: (param,) / (output, param) / (time, output, param); attrs: mode
```

## References

- Borgonovo, E., Figalli, A., Plischke, E., & Savaré, G. (2024). Global
  sensitivity analysis via optimal transport. *Management Science*.
  doi:10.1287/mnsc.2023.01796
