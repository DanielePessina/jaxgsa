# Borgonovo Delta (Moment-Independent Sensitivity)

By the end of this page you will have one index per input, on a [0, 1] scale,
that says how much fixing that input reshapes the whole distribution of the
output. Variance-based methods only ask how much the output spread shrinks.
This one also catches an input that shifts the tails or splits the output into
two modes without changing the variance much.

Borgonovo delta is called a moment-independent method because it uses no
summary statistic such as mean or variance. It compares densities directly. The
output density with nothing fixed is the unconditional density. The density you
get after fixing one input is the conditional density. Delta is the expected L1
distance between the two, which is the average area between the curves. It
needs no variance decomposition and makes no model assumptions. The same
analysis also returns the given-data first-order Sobol index from the same
conditioning, for comparison at no extra cost. That Sobol index is the share of
output variance the input explains on its own.

When to use Borgonovo delta:

- You care about influence on the whole output distribution (tails,
  skewness, multimodality) beyond just variance.
- You want a moment-independent index on a fixed [0, 1] scale that is
  invariant under monotone output transformations.
- You have a set of (X, Y) pairs from any sampling strategy — no
  structured design required.

::: warning Continuous outputs only
`borgonovo.analyze` supports a continuous output distribution only. The
estimator compares kernel density estimates on a shared output grid, and a
discrete output has atoms that no grid resolves. `analyze` checks the
output first and raises `ValueError` when a column takes at most 20
distinct values and those values are fewer than 1% of the samples. Use
[`jaxgsa.optimal_transport.analyze`](/examples/optimal-transport) for a
discrete output: it compares empirical distributions directly and needs no
density. A continuous output rounded to a few decimals is not refused,
and neither is a constant column, whose exact answer is `delta = S1 = 0`.
Categorical inputs stay supported. The limit applies to the output only.
:::

## Import style

```python
# Subpackage import
from jaxgsa import borgonovo
# borgonovo.analyze(...)

# Or top-level
import jaxgsa
# jaxgsa.borgonovo.analyze(...)
```

## Scalar example (Ishigami)

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

# Generate Monte Carlo samples
X = jaxgsa.sampling.monte_carlo(PROBLEM, n=5000, seed=42)
Y = evaluate(jnp.asarray(X))

# Compute delta and given-data S1 indices
result = jaxgsa.borgonovo.analyze(PROBLEM, jnp.asarray(X), Y)

print("delta:", result.delta)  # (3,)
print("S1:   ", result.S1)     # (3,)
```

The delta index lies in [0, 1], and so do the true index and the raw plug-in
estimate. A value of 0 means fixing the input never changes the output
distribution. Higher values mean stronger influence. The default
bias-corrected estimate in `result.delta` can dip marginally below 0 for weak
or near-noninfluential inputs at small N, and so can the bounds of
`result.delta_conf`. That is expected from the bias correction, not an error.

Compare the two printed arrays entry by entry. On Ishigami, `x3` has a
first-order Sobol index near zero, because it acts only through an interaction
with `x1`. Its delta index is clearly positive. Fixing `x3` reshapes the output
density even though it does not shift the conditional mean. That gap between
`delta` and `S1` is exactly what a moment-independent index adds. Where the two
agree, the input acts on the output mean and variance in the ordinary way, and
delta tells you nothing the Sobol index did not.

## Bias correction and bootstrap confidence intervals

The plug-in delta estimator is biased upward at finite N. A bootstrap fixes
this. It recomputes delta on many resamples of the data, measures how far the
estimate drifts, and subtracts that drift. The defaults are `n_bootstrap=100`
and `bias_correct=True`, so the central estimate is already bias-corrected
(Plischke et al., 2013). Percentile confidence intervals come from the same
replicates.

```python
result = jaxgsa.borgonovo.analyze(
    PROBLEM, X, Y,
    n_bootstrap=100,
    conf_level=0.95,
    seed=0,
)

print("delta:", result.delta)
print("95% CI lower:", result.delta_conf[0])
print("95% CI upper:", result.delta_conf[1])
print("S1 95% CI:", result.S1_conf)
```

Set `n_bootstrap=0` to skip both bias correction and intervals (the raw
plug-in estimate; `delta_conf` and `S1_conf` are `None`), or
`bias_correct=False` to keep the intervals but report the uncorrected
estimate.

## Ground-truth check (Gaussian linear benchmark)

The `gaussian_linear` benchmark has a semi-analytic delta solution
(`ANALYTICAL_DELTA`), so you can validate the estimator against ground
truth rather than another implementation. Run this when you want to know how
much N your own problem needs: shrink `n` until the estimate drifts away from
the analytical values, and you have located the sample size at which the
estimator stops being reliable.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks import gaussian_linear

X = jnp.asarray(jaxgsa.sampling.monte_carlo(gaussian_linear.PROBLEM, n=8000, seed=42))
Y = gaussian_linear.evaluate(X)
result = jaxgsa.borgonovo.analyze(gaussian_linear.PROBLEM, X, Y)

print("estimated: ", result.delta)
print("analytical:", gaussian_linear.ANALYTICAL_DELTA)
```

## Multi-output example

When Y has shape `(N, K)`, delta and S1 indices have shape `(K, D)`.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n=3000, seed=42))
Y1 = evaluate(X)
Y2 = jnp.sum(X**2, axis=1)
Y_multi = jnp.column_stack([Y1, Y2])

result = jaxgsa.borgonovo.analyze(PROBLEM, X, Y_multi)
print("delta shape:", result.delta.shape)  # (2, 3)
```

The shape `(2, 3)` reads as two outputs by three inputs. Row 0 holds the delta
indices for `Y1` and row 1 those for `Y2`. Each output gets its own ranking, and
an input can be the most influential one for `Y1` and the least influential for
`Y2`.

## xarray export

```python
ds = result.to_dataset()
print(ds)  # variables: delta, S1 (+ delta_lower/upper, S1_lower/upper with CIs)

# Without bootstrap: only delta and S1
result_plain = jaxgsa.borgonovo.analyze(PROBLEM, X, Y1, n_bootstrap=0)
ds_plain = result_plain.to_dataset()
print(ds_plain)
```

## Shape rules

| Y shape | delta / S1 | delta_conf / S1_conf |
|---|---|---|
| `(N,)` | `(D,)` | `(2, D)` or None |
| `(N, K)` | `(K, D)` | `(2, K, D)` or None |
| `(N, T, K)` | `(T, K, D)` | `(2, T, K, D)` or None |

## Practical caveats

- Delta needs no structured sampling. Any (X, Y) pairs work, including
  Monte Carlo, Latin Hypercube, or Sobol sequences.
- The number of conditioning classes (`n_classes`) defaults to the
  Plischke sample-size heuristic (roughly `N**(2/7)`, at most 48). More
  classes give finer conditioning but fewer samples per class, so KDE
  estimates get noisier; the default works well for N >= 1000.
- The estimator matches `SALib.analyze.delta` (same partition, bandwidths,
  and grid) but is deterministic given the data — SALib computes its
  central estimate on a random resample — and returns `delta = S1 = 0` for
  a constant output instead of raising.
- Peak memory scales with `slice_chunk_size * D * N * grid_size`; lower
  `slice_chunk_size` for large time-series outputs.
- Delta is a half L1 distance between densities, so it lies in [0, 1]. If
  the returned estimate leaves that range by more than 0.05, the
  computation failed and `analyze` raises `ValueError` naming the
  parameter. The cause is a conditioning class the output grid cannot
  resolve. The message reads what the run actually did and names the knob
  that governs it: `grid_size` and `degenerate_bandwidth` when the class
  was found degenerate and its kernel was floored, or `grid_size` and
  `degenerate_tol` when no class was floored and the floor played no part.
  The value is never clipped, because a clipped value looks plausible and
  is still wrong. A confidence bound outside the range only warns: the
  point estimate is the contract and the interval is a diagnostic.
- `analyze` never refuses `degenerate_tol` or `degenerate_bandwidth` on
  the setting alone. `degenerate_bandwidth` only reaches a class the
  estimator already called degenerate, so on smooth data it cannot change
  the result at any value. Raising `degenerate_tol` above the floor
  fraction does bias the result: a class whose own bandwidth sits between
  the floor and the tolerance is then *narrowed* to the floor, which
  inflates delta for the very classes the higher tolerance said to
  distrust. That is still a valid computation, so it is a bias to know
  about, not an error.

## See also

- [Basic Example](/examples/basic) for the Sobol variance-decomposition
  workflow.
- [PAWN Example](/examples/pawn) for the CDF-based moment-independent
  method.
- [Methods](/guide/methods) for a comparison of all methods.
- [API Reference](/api/#given-data-methods) for full parameter
  documentation.
