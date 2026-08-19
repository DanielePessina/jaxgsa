# Morris (Elementary Effects Screening)

This page screens a model's inputs with the Morris method. You end with three
arrays, one entry per input, that rank the inputs by importance and flag which
of them behave nonlinearly or interact with each other.

Morris is a global screening method. Screening means it sorts the inputs into
"worth studying" and "safe to fix" instead of quantifying each one exactly. The
design is a globalized one-at-a-time scheme: it changes one input at a time,
but it repeats that from many starting locations across the whole input domain.
Each single change gives an elementary effect, a coarse finite difference of the
output with respect to that input. Morris reduces the collected elementary
effects to three cheap measures:

- mu — the mean effect.
- mu_star — the mean absolute effect, the headline importance measure.
- sigma — the spread, which flags nonlinearity or interactions.

The cost is only `r * (D + 1)` model evaluations, where r is the number of
trajectories and D the number of inputs.

When to use Morris:

- You want a cheap screening pass before committing to a full Sobol run. A
  Sobol run splits the output variance among the inputs and costs far more
  model evaluations.
- Your model is a black box (if it is JAX-differentiable, consider DGSM,
  which computes the infinitesimal-step analog of mu_star via autodiff).
- You only need a parameter ranking and an interaction flag, not exact
  variance fractions.

A companion marimo notebook lives at
[`examples/morris_gsa.py`](https://github.com/danielepessina/jaxgsa/blob/master/examples/morris_gsa.py).
Run it interactively with `uv run marimo edit examples/morris_gsa.py`.

## Import style

The Morris module lives at `jaxgsa.morris`:

```python
from jaxgsa import morris
# morris.sample(...)
# morris.analyze(...)
```

## Scalar example (Ishigami)

Ishigami is a standard three-input test function whose behaviour is known in
advance, so you can check the measures against the right answer. Morris is a
structured method: it builds its own design, so the sampling step and the
analysis step must use the same `MorrisSamples` object.

1. Build the design with `morris.sample`. It lays out r trajectories of D+1
   points, where consecutive points differ in exactly one input.
2. Read `n_runs` and `n_expanded`. `sample` returns only the unique rows, so
   you pay for fewer evaluations than the full layout implies.
3. Run the model on the unique rows. Evaluating the duplicates again would
   cost more and change nothing.
4. Call `morris.analyze` with the design and the outputs. It rebuilds the
   trajectory layout internally and forms the three measures.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

# Generate Morris trajectories: r trajectories of D+1 points each.
# Only the unique rows are returned — exact duplicates across trajectories
# are removed, so you evaluate fewer points than r * (D + 1).
sr = jaxgsa.morris.sample(PROBLEM, n_trajectories=50, num_levels=4, seed=42)
print("unique rows:", sr.n_runs)          # <= 50 * (3 + 1) = 200
print("expanded rows:", sr.n_expanded)  # 200

# Evaluate the model on the unique rows
Y = evaluate(jnp.asarray(sr.samples))

# Compute the screening measures
result = jaxgsa.morris.analyze(sr, Y)

print("mu:", result.mu)            # (D,) mean elementary effect (signed)
print("mu_star:", result.mu_star)  # (D,) mean |elementary effect| — importance
print("sigma:", result.sigma)      # (D,) spread — nonlinearity/interactions
```

`n_expanded` prints 200, which is the full 50 x (3 + 1) layout. `n_runs` is the
number of rows you actually evaluate, and it is at most 200; the gap is the
duplicate points that the grid design produced more than once. The three
measures each have length 3, one entry per input, in the order the parameters
appear in `PROBLEM`.

Interpreting the measures:

- **mu_star** ranks the parameters. For Ishigami all three inputs come out
  comparable — note that x3 is kept even though its first-order Sobol index
  is near zero, because mu_star is a proxy for the total-order index and
  x3 acts through its interaction with x1. Parameters with small mu_star are
  negligible and can be fixed before a more expensive Sobol analysis.
- **sigma** relative to mu_star shows how a parameter acts. A large
  ratio means the elementary effects vary strongly across the domain — the
  parameter is involved in nonlinearities or interactions (here x3 has the
  largest sigma, consistent with its purely interactive role). The canonical
  diagnostic is the mu_star–sigma scatter plot: negligible parameters sit
  near the origin, additive-linear ones near the mu_star axis, and
  interaction-driven ones high above the diagonal.
- **mu** keeps the sign, so it can cancel to near zero for non-monotonic
  effects — compare mu with mu_star for x2 and x3 in this example. Rank with
  mu_star, not mu.

A bootstrap resamples the trajectories many times and recomputes the measures
on each resample. The spread of those values becomes a confidence interval,
which tells you whether two inputs are really ranked apart or only separated by
sampling noise. Ask for it with `num_resamples` (a JAX PRNG key is required):

```python
import jax

result = jaxgsa.morris.analyze(sr, Y, num_resamples=500, key=jax.random.key(0))
print(result.mu_star_conf)  # (2, D) — [lower, upper] bounds
```

`mu_star_conf` has shape `(2, 3)`: row 0 holds the lower bound of each input's
mu_star and row 1 the upper bound. Two inputs whose intervals overlap are not
separated by this run.

## Radial variant

The default `method="trajectory"` walks a `num_levels` grid (Morris 1991). The
alternative `method="radial"` (Campolongo et al. 2011) builds star designs
around scrambled-Sobol' base points. A star design moves out from one base
point along each input in turn, and the base points come from a
low-discrepancy Sobol' sequence with a random scramble applied. The steps
therefore vary in size and no grid is involved:

```python
sr_radial = jaxgsa.morris.sample(PROBLEM, n_trajectories=50, method="radial", seed=42)
Y_radial = evaluate(jnp.asarray(sr_radial.samples))
result_radial = jaxgsa.morris.analyze(sr_radial, Y_radial)
print("mu_star (radial):", result_radial.mu_star)
```

The printed array has the same length and meaning as the trajectory mu_star
above, and you read the ranking the same way. The numbers themselves differ,
because the two designs take different step sizes. `num_levels` is ignored by
the radial design. Radial points do not lie on a coarse grid, so fewer
duplicate rows are removed than with the trajectory design.

## From an existing Sobol design

If you have already run a Sobol analysis, you can get Morris measures out of it
for free. A Saltelli design is the point layout a Sobol analysis uses: two
independent sample matrices `A` and `B`, plus one matrix `AB_j` per input, in
which column j of `A` is replaced by column j of `B`. That layout already
contains the radial structure Morris needs. Within each base point, `A` and
each `AB_j` differ in exactly one parameter. So `SobolSamples.to_morris()`
reinterprets the design without any new model evaluations:

```python
samples = jaxgsa.sobol.sample(PROBLEM, 0, base_n=512, seed=0)
Y = evaluate(jnp.asarray(samples.samples))

sobol_result = jaxgsa.sobol.analyze(samples, Y)
morris_result = jaxgsa.morris.analyze(samples.to_morris(), Y)

print("ST:     ", sobol_result.ST)
print("mu_star:", morris_result.mu_star)
```

The same `Y` feeds both calls. `ST` is the Sobol total-order index, a variance
fraction; `mu_star` is a mean absolute effect on the model's own scale. The two
printed lines are therefore on different scales and should be compared as
rankings, not value by value.

`to_morris()` returns a normal `MorrisSamples` with `method="radial"`, so
everything else works unchanged: bootstrap CIs, multi-output outputs,
`to_dataset()`, `downsample()`, and `save()`. Its `samples` is the same array
you already evaluated, so `n_runs` is unchanged and your existing `Y` stays
valid.

`n_trajectories` is `base_n` for both design variants — one radial block per
base point. A second-order design also holds a block based at `B` (`B` with its
`BA_j` rows), but for additive contributions it is algebraically the same
effect, so harvesting it would inflate the apparent sample size and narrow
bootstrap CIs without improving `mu_star` or `sigma`. It is deliberately unused;
see [Methods](/guide/methods) for the measurement.

See [Methods](/guide/methods) for why this works: Jansen's total-order
estimator and Morris's mu_star are different moments of the same increments
`f(AB_j) - f(A)`.

## Multi-output example

When your model returns K outputs per sample, pass Y with shape
`(n_runs, K)`. The resulting measures have shape `(K, D)`. Time-series
outputs `(n_runs, T, K)` produce `(T, K, D)`, where T is the number of time
steps. The steps match the scalar case, with one addition: the problem carries
`output_names`, which fixes the row order of the measure arrays.

1. Build a `Problem` with three named inputs and two named outputs.
2. Write a model that returns both outputs stacked on the last axis.
3. Sample, evaluate, and analyze exactly as before. One call covers both
   outputs.

```python
import jax.numpy as jnp
import jaxgsa

problem = jaxgsa.Problem.from_dict(
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
    return jnp.stack([displacement, velocity], axis=-1)  # (n_runs, K=2)


sr = jaxgsa.morris.sample(problem, n_trajectories=50, seed=42)
Y = multi_output_model(jnp.asarray(sr.samples))

result = jaxgsa.morris.analyze(sr, Y)
print("mu_star shape:", result.mu_star.shape)  # (K, D) = (2, 3)
print("sigma shape:", result.sigma.shape)      # (K, D) = (2, 3)
```

Both printed shapes are `(2, 3)`: two rows for the two outputs, three columns
for the three inputs. Row 0 belongs to `"displacement"` and row 1 to
`"velocity"`, following `output_names`. So `result.mu_star[1, 0]` is the
importance of `amplitude` for the `"velocity"` output.

## Gaussian inputs

Gaussian marginals are supported through a truncated-quantile grid. A marginal
is the distribution of one input on its own. The Morris design touches the
unit-cube boundaries, and an unbounded inverse CDF maps 0 and 1 to infinity. Each
open side of a Gaussian marginal is therefore pulled in by `q`
(`truncation_quantile`, default 1e-4 — probing the 0.01%–99.99% quantile range)
before the transform. A side the problem already bounds with an explicit `low`
or `high` stays exactly where you put it, so a two-sided truncated Gaussian is
sampled as declared. Uniform marginals are untouched, and deduplication and
prefix-nested downsampling work as usual.

On an unbounded marginal `mu_star` has no `q -> 0` limit. The design always
includes unit levels 0 and 1 exactly, so a smaller `q` always reaches further
into the tail and the effects grow with it. Magnitudes are scale-dependent by
construction there, and only rankings are comparable across truncation
settings. To fix one bounded input model that every method shares, pass
`truncate_gaussians` once:

```python
problem = jaxgsa.Problem.from_dict(params, truncate_gaussians=1e-4)
```

It writes explicit `low`/`high` into every Gaussian that does not already
declare them, at that marginal's own `q` and `1 - q` quantiles.

The next example mixes one uniform input with one Gaussian input and runs the
standard three steps on it.

```python
import jax.numpy as jnp
import jaxgsa

problem = jaxgsa.Problem.from_dict(
    {
        "x1": (-1.0, 1.0),
        "x2": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
    }
)

sr = jaxgsa.morris.sample(problem, n_trajectories=50, seed=42)
X = jnp.asarray(sr.samples)
Y = X[:, 0] + X[:, 1] ** 2

result = jaxgsa.morris.analyze(sr, Y)
print("mu_star:", result.mu_star)  # (2,)
```

`mu_star` has length 2 here, because the problem has two inputs: the uniform
`x1` and the Gaussian `x2`. Elementary effects remain per unit of the original
grid coordinate, and `to_physical_units()` is unavailable for such problems
(see below).

## Physical units

Elementary effects are computed in unit-cube coordinates, so mu_star is
directly comparable across parameters regardless of their physical ranges.
`to_physical_units()` returns a rescaled copy — each measure is divided by
the parameter range `high - low`, giving per-physical-unit
(derivative-scale) effects comparable to DGSM's mean derivative. It requires
a uniform-marginal problem (unlike the Gaussian example above):

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

sr = jaxgsa.morris.sample(PROBLEM, n_trajectories=50, seed=42)
result = jaxgsa.morris.analyze(sr, evaluate(jnp.asarray(sr.samples)))

physical = result.to_physical_units()
print(physical.space)    # "physical" (the original result stays "unit")
print(physical.mu_star)  # per-physical-unit effects
```

`physical.space` prints `"physical"`, which is how you tell the two coordinate
systems apart later. The call returns a copy, so `result` still reports
`"unit"` and still holds the unit-cube measures. `physical.mu_star` holds the
same three inputs, each divided by its own `high - low`, so the ranking can
change when the parameters have very different physical ranges.

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
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

sr_full = jaxgsa.morris.sample(PROBLEM, n_trajectories=100, seed=42)
Y_full = evaluate(jnp.asarray(sr_full.samples))

for r in [50, 25, 10]:
    sr_r, Y_r = sr_full.downsample(r, Y_full)
    result = jaxgsa.morris.analyze(sr_r, Y_r)
    print(f"r={r:3d}  mu_star={result.mu_star}")
```

The loop prints three mu_star rows from one set of model evaluations, at 50,
25, and 10 trajectories. Compare the rows down the column: the values move as r
falls, and what matters is whether the order of the three inputs stays the
same. This mirrors `SobolSamples.downsample()` for Sobol designs and is useful
for convergence checks: if the ranking is stable from 25 to 100 trajectories,
25 would have sufficed.

## xarray export

`MorrisResult.to_dataset()` converts results to a labeled `xarray.Dataset`,
just like the Sobol and eFAST result types. You can then select by parameter
and output name instead of by integer index. The coordinate space is recorded
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

The printed dataset reports `(output: 2, param: 3)`, the same two axes as the
raw `(K, D)` arrays but now named. The first `sel` call returns the mu_star of
`amplitude` for both outputs. The second returns the sigma of all three inputs
for the `"velocity"` output. The `space` attribute prints `"unit"`, so these
measures are in unit-cube coordinates and not per physical unit.

For time-series results, pass `time_coords` to label the time dimension.
When bootstrap CIs are present, the dataset also contains `mu_lower`,
`mu_upper`, `mu_star_lower`, and so on.

## Shape rules

- `(n_runs,)` means scalar output.
- `(n_runs, K)` means K output variables with no time dimension.
- `(n_runs, T, K)` means T time steps and K outputs.
- Without `problem.output_names`, a 2D array is always treated as
  `(n_runs, K)`.
- With exactly one entry in `problem.output_names`, a 2D array is treated as
  `(n_runs, T)` — timepoints of that single output — and flows through as
  `(n_runs, T, 1)`. Passing a pre-reshaped `(n_runs, T, 1)` array also works.

| Y shape | mu / mu_star / sigma shape |
|---------|----------------------------|
| `(n_runs,)` | `(D,)` |
| `(n_runs, K)` | `(K, D)` |
| `(n_runs, T, K)` | `(T, K, D)` |

D is always the last axis.

## Practical caveats

- Gaussian marginals are sampled on a truncated-quantile grid
  (`truncation_quantile`, default 1e-4): the design would otherwise hit the
  unit-cube boundaries, which an unbounded inverse CDF maps to infinity. Only
  open sides are pulled in; an explicit `low` or `high` is kept as written.
  `truncation_quantile` must be in `(0, 0.5)` or `jaxgsa.morris.sample()` raises
  `ValueError`.
- `to_physical_units()` raises `ValueError` for problems with Gaussian
  marginals — the inverse-CDF transform is nonlinear, so the measures stay
  in grid coordinates.
- Morris does not produce Sobol indices. mu_star is a ranking proxy for the
  total-order index ST, not a variance fraction; sigma flags interactions
  but cannot attribute them to specific pairs.
- Even `num_levels` values (the default is 4) make all grid levels equally
  probable; odd values trigger a warning.
- `Y` must be evaluated on `sr.samples` (the unique rows); `jaxgsa.morris.analyze()`
  reconstructs the expanded trajectory layout internally.
- A trajectory containing any non-finite output (NaN/Inf) raises by default.
  Pass `on_invalid="drop"` to remove it instead; a trajectory is dropped as a
  whole block, because an elementary effect is a difference between
  neighbouring rows inside one. Fewer than 2 remaining trajectories raise an
  error; fewer than 10 trigger a reliability warning.
- Measures derived through `SobolSamples.to_morris()` come from the same model
  outputs as that design's Sobol indices, so mu_star and ST agreeing is not an
  independent check of either.
- A derived design is a radial design. It estimates
  `E|f(A with B_j) - f(A)| / |B_j - A_j|`, not the fixed-step-delta grid
  quantity. `jaxgsa.morris.sample()` defaults to `method="trajectory"`, so
  compare a derived result against `morris.sample(..., method="radial")`. On
  Ishigami at r=8192 the derived mu_star is [8.68, 15.01, 6.62] against
  [8.69, 15.02, 6.64] native radial, but [7.59, 7.88, 6.39] native trajectory.
- Derived blocks whose step is unmeasurable are dropped with a warning. At the
  default `scramble=True` this is a non-issue: 0 of 65536 blocks were dropped
  across 8 seeds at D=3. With `scramble=False` the rate falls with `base_n`
  (21.9% at 64, 9.4% at 256, 2.3% at 1024, 1.2% at 4096) and the survivors are
  a biased subsequence — mu_star [8.34, 14.88, 5.55] at base_n=64 against
  [8.68, 15.01, 6.62] scrambled, so x3 reads 16% low. Keep `scramble=True`.
- For unbounded Gaussian marginals a derived mu_star has no fixed scale,
  because how far the design reaches into the tail sets the magnitude and the
  Saltelli design and `morris.sample` reach different distances. Rankings are
  unaffected. Use `Problem.from_dict(..., truncate_gaussians=q)` if magnitudes
  must be comparable across designs.

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
- [API Reference](/api/#structured-methods) for full parameter documentation.
