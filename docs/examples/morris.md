# Morris (Elementary Effects Screening)

Morris is a global **screening** method: a globalized one-at-a-time design
that measures coarse finite-difference "elementary effects" of each input at
many locations across the whole input domain. It reduces them to three cheap
measures — mu (mean effect), mu_star (mean absolute effect, the headline
importance measure), and sigma (spread, flagging nonlinearity or
interactions) — at a cost of only `r * (D + 1)` model evaluations.

When to use Morris:

- You want a cheap screening pass before committing to a full Sobol run.
- Your model is a black box (if it is JAX-differentiable, consider DGSM,
  which computes the infinitesimal-step analog of mu_star via autodiff).
- You only need a parameter ranking and an interaction flag, not exact
  variance fractions.

A companion marimo notebook lives at
[`examples/morris_gsa.py`](https://github.com/danielepessina/gsax/blob/master/examples/morris_gsa.py).
Run it interactively with `uv run marimo edit examples/morris_gsa.py`.

## Import style

The Morris module lives at `gsax.morris`:

```python
from gsax import morris
# morris.sample(...)
# morris.analyze(...)
```

## Scalar example (Ishigami)

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

# Generate Morris trajectories: r trajectories of D+1 points each.
# Only the unique rows are returned — exact duplicates across trajectories
# are removed, so you evaluate fewer points than r * (D + 1).
sr = gsax.morris.sample(PROBLEM, n_trajectories=50, num_levels=4, seed=42)
print("unique rows:", sr.n_total)          # <= 50 * (3 + 1) = 200
print("expanded rows:", sr.expanded_n_total)  # 200

# Evaluate the model on the unique rows
Y = evaluate(jnp.asarray(sr.samples))

# Compute the screening measures
result = gsax.morris.analyze(sr, Y)

print("mu:", result.mu)            # (D,) mean elementary effect (signed)
print("mu_star:", result.mu_star)  # (D,) mean |elementary effect| — importance
print("sigma:", result.sigma)      # (D,) spread — nonlinearity/interactions
```

Interpreting the measures:

- **mu_star** ranks the parameters. For Ishigami all three inputs come out
  comparable — note that x3 is kept even though its first-order Sobol index
  is near zero, because mu_star is a proxy for the *total-order* index and
  x3 acts through its interaction with x1. Parameters with small mu_star are
  negligible and can be fixed before a more expensive Sobol analysis.
- **sigma** relative to **mu_star** flags *how* a parameter acts. A large
  ratio means the elementary effects vary strongly across the domain — the
  parameter is involved in nonlinearities or interactions (here x3 has the
  largest sigma, consistent with its purely interactive role). The canonical
  diagnostic is the mu_star–sigma scatter plot: negligible parameters sit
  near the origin, additive-linear ones near the mu_star axis, and
  interaction-driven ones high above the diagonal.
- **mu** keeps the sign, so it can cancel to near zero for non-monotonic
  effects — compare mu with mu_star for x2 and x3 in this example. Rank with
  mu_star, not mu.

Bootstrap confidence intervals over trajectories are available via
`num_resamples` (a JAX PRNG key is required):

```python
import jax

result = gsax.morris.analyze(sr, Y, num_resamples=500, key=jax.random.key(0))
print(result.mu_star_conf)  # (2, D) — [lower, upper] bounds
```

## Radial variant

The default `method="trajectory"` walks a `num_levels` grid (Morris 1991).
The alternative `method="radial"` (Campolongo et al. 2011) builds star
designs around scrambled-Sobol' base points, so the steps vary in size and
no grid is involved:

```python
sr_radial = gsax.morris.sample(PROBLEM, n_trajectories=50, method="radial", seed=42)
Y_radial = evaluate(jnp.asarray(sr_radial.samples))
result_radial = gsax.morris.analyze(sr_radial, Y_radial)
print("mu_star (radial):", result_radial.mu_star)
```

`num_levels` is ignored by the radial design. Radial points do not lie on a
coarse grid, so fewer duplicate rows are removed than with the trajectory
design.

## Multi-output example

When your model returns K outputs per sample, pass Y with shape
`(n_total, K)`. The resulting measures have shape `(K, D)`. Time-series
outputs `(n_total, T, K)` produce `(T, K, D)`.

```python
import jax.numpy as jnp
import gsax

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
    return jnp.stack([displacement, velocity], axis=-1)  # (n_total, K=2)


sr = gsax.morris.sample(problem, n_trajectories=50, seed=42)
Y = multi_output_model(jnp.asarray(sr.samples))

result = gsax.morris.analyze(sr, Y)
print("mu_star shape:", result.mu_star.shape)  # (K, D) = (2, 3)
print("sigma shape:", result.sigma.shape)      # (K, D) = (2, 3)
```

## Gaussian inputs

Gaussian marginals are supported through a **truncated-quantile grid**: the
Morris design includes the unit-cube boundaries, which an unbounded inverse
CDF would map to infinity, so each Gaussian coordinate is confined to
`[q, 1 - q]` (`truncation_quantile`, default 0.005 — probing the 0.5%–99.5%
quantile range) before the transform. Uniform marginals are untouched, and
deduplication and prefix-nested downsampling work as usual.

```python
import jax.numpy as jnp
import gsax

problem = gsax.Problem.from_dict(
    {
        "x1": (-1.0, 1.0),
        "x2": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
    }
)

sr = gsax.morris.sample(problem, n_trajectories=50, seed=42)
X = jnp.asarray(sr.samples)
Y = X[:, 0] + X[:, 1] ** 2

result = gsax.morris.analyze(sr, Y)
print("mu_star:", result.mu_star)  # (2,)
```

Elementary effects remain per unit of the original grid coordinate, and
`to_physical_units()` is unavailable for such problems (see below).

## Physical units

Elementary effects are computed in unit-cube coordinates, so mu_star is
directly comparable across parameters regardless of their physical ranges.
`to_physical_units()` returns a rescaled copy — each measure is divided by
the parameter range `high - low`, giving per-physical-unit
(derivative-scale) effects comparable to DGSM's mean derivative. It requires
a uniform-marginal problem (unlike the Gaussian example above):

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

sr = gsax.morris.sample(PROBLEM, n_trajectories=50, seed=42)
result = gsax.morris.analyze(sr, evaluate(jnp.asarray(sr.samples)))

physical = result.to_physical_units()
print(physical.space)    # "physical" (the original result stays "unit")
print(physical.mu_star)  # per-physical-unit effects
```

Calling `to_physical_units()` on a result that is already in physical units
raises `ValueError`. It also raises for problems with non-uniform (Gaussian)
marginals: the inverse-CDF transform is nonlinear, so there is no single
per-parameter range to rescale by, and the measures stay in grid coordinates.

## Downsampling trajectories

Trajectories are generated sequentially, so the first *m* trajectories of an
*r*-trajectory run are identical to drawing *m* trajectories directly with
the same seed. Simulate once at the largest `n_trajectories` and slice down —
no re-simulation needed:

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

sr_full = gsax.morris.sample(PROBLEM, n_trajectories=100, seed=42)
Y_full = evaluate(jnp.asarray(sr_full.samples))

for r in [50, 25, 10]:
    sr_r, Y_r = sr_full.downsample(r, Y_full)
    result = gsax.morris.analyze(sr_r, Y_r)
    print(f"r={r:3d}  mu_star={result.mu_star}")
```

This mirrors `SobolSamples.downsample()` for Sobol designs and is useful
for convergence checks: if the ranking is stable from 25 to 100 trajectories,
25 would have sufficed.

## xarray export

`MorrisResult.to_dataset()` converts results to a labeled `xarray.Dataset`,
just like the Sobol and eFAST result types. The coordinate space is recorded
in the `space` attribute.

```python
ds = result.to_dataset()
print(ds)
# <xarray.Dataset>
# Dimensions:  (output: 2, param: 3)

print(ds.mu_star.sel(param="amplitude"))
print(ds.sigma.sel(output="velocity"))
print(ds.attrs["space"])  # "unit"
```

For time-series results, pass `time_coords` to label the time dimension.
When bootstrap CIs are present, the dataset also contains `mu_lower`,
`mu_upper`, `mu_star_lower`, and so on.

## Shape rules

- `(n_total,)` means scalar output.
- `(n_total, K)` means K output variables with no time dimension.
- `(n_total, T, K)` means T time steps and K outputs.
- Without `problem.output_names`, a 2D array is always treated as
  `(n_total, K)`.
- With exactly one entry in `problem.output_names`, a 2D array is treated as
  `(n_total, T)` — timepoints of that single output — and flows through as
  `(n_total, T, 1)`. Passing a pre-reshaped `(n_total, T, 1)` array also works.

| Y shape | mu / mu_star / sigma shape |
|---------|----------------------------|
| `(n_total,)` | `(D,)` |
| `(n_total, K)` | `(K, D)` |
| `(n_total, T, K)` | `(T, K, D)` |

D is always the last axis.

## Practical caveats

- Gaussian marginals are sampled on a truncated-quantile grid
  (`truncation_quantile`, default 0.005): the design would otherwise hit the
  unit-cube boundaries, which an unbounded inverse CDF maps to infinity.
  `truncation_quantile` must be in `(0, 0.5)` or `sample_morris()` raises
  `ValueError`.
- `to_physical_units()` raises `ValueError` for problems with Gaussian
  marginals — the inverse-CDF transform is nonlinear, so the measures stay
  in grid coordinates.
- Morris does not produce Sobol indices. mu_star is a ranking proxy for the
  total-order index ST, not a variance fraction; sigma flags interactions
  but cannot attribute them to specific pairs.
- Even `num_levels` values (the default is 4) make all grid levels equally
  probable; odd values trigger a warning.
- `Y` must be evaluated on `sr.samples` (the unique rows); `analyze_morris()`
  reconstructs the expanded trajectory layout internally.
- Trajectories containing any non-finite output (NaN/Inf) are dropped as
  whole blocks with a warning. Fewer than 2 remaining trajectories raise an
  error; fewer than 10 trigger a reliability warning.

## See also

- [Basic Example](/examples/basic) for the Sobol workflow with structured
  Saltelli sampling — the natural next step after screening.
- [DGSM](/examples/dgsm) for the autodiff analog of Morris on
  JAX-differentiable models.
- [eFAST](/examples/efast) for frequency-based variance decomposition.
- [xarray Labeled Output](/examples/xarray) for named access by parameter,
  output, and time coordinate.
- [Methods](/guide/methods) for the theory behind Morris and when to choose
  it over other methods.
- [API Reference](/api/#morris-workflow) for full parameter documentation.
