# PAWN (CDF-based Sensitivity)

This page computes PAWN sensitivity indices from a set of input-output samples.
You end with one number per input that says how much the shape of the whole
output distribution changes when that input is held fixed.

PAWN is a distribution-based sensitivity method. It compares two cumulative
distribution functions (CDFs) of the output. A CDF gives, for each value, the
probability that the output falls below it. The unconditional CDF uses all the
samples. A conditional CDF uses only the samples in which one input sits inside
a narrow range. The gap between the two is measured with the
Kolmogorov-Smirnov (KS) distance, the largest vertical gap between two CDF
curves. There is no variance decomposition and there are no model assumptions.

When to use PAWN:

- You care about distributional changes beyond just variance (tail
  behavior, skewness shifts).
- You want a moment-independent index that is invariant under monotone
  output transformations. Moment-independent means the index does not read the
  output through its mean or variance. Invariant under monotone
  transformations means that rescaling the output by any order-preserving
  function, such as a log, leaves the index unchanged.
- You have a set of (X, Y) pairs from any sampling strategy — no
  structured design required.

## Import style

```python
# Subpackage import
from jaxgsa import pawn
# pawn.analyze(...)

# Or top-level
import jaxgsa
# jaxgsa.pawn.analyze(...)
```

Both forms reach the same function. Pick one and keep it.

## Scalar example (Ishigami)

Ishigami is a standard three-input test function with a known answer, which
makes it a good place to read the output of a new method. The steps are:

1. Draw samples. PAWN accepts any sampling scheme, so a plain Monte Carlo draw
   is enough.
2. Run the model on those samples. PAWN needs only the paired (X, Y) values.
3. Call `analyze`. It splits each input into bins, measures the KS distance in
   each bin, and reduces the bin distances to one number per input.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

# Generate Monte Carlo samples
X = jaxgsa.sampling.monte_carlo(PROBLEM, n=5000, seed=42)
Y = evaluate(jnp.asarray(X))

# Compute PAWN indices (median KS across 10 bins)
result = jaxgsa.pawn.analyze(PROBLEM, jnp.asarray(X), Y)

print("PAWN:", result.pawn)  # (3,)
```

`result.pawn` has length 3, one entry per input, in the order the parameters
appear in `PROBLEM`. The PAWN index is the median (by default) KS distance
across conditioning bins. A conditioning bin is a slice of one input's range;
inside it, that input is close to fixed while the others still vary. Higher
values indicate stronger influence on the output distribution.

## Choosing the aggregation statistic

PAWN computes one KS distance per bin, then aggregates across bins. The choice
of aggregation decides which bin drives the index, so it changes the ranking
you read. Three options are available:

```python
# Median (default) — robust to outlier bins
r_med = jaxgsa.pawn.analyze(PROBLEM, X, Y, statistic="median")

# Maximum — captures worst-case shift
r_max = jaxgsa.pawn.analyze(PROBLEM, X, Y, statistic="max")

# Mean — simple average
r_mean = jaxgsa.pawn.analyze(PROBLEM, X, Y, statistic="mean")

print("median:", r_med.pawn)
print("max:   ", r_max.pawn)
print("mean:  ", r_mean.pawn)
```

The three printed arrays all have length 3 and come from the same KS distances.
Only the reduction differs. The `max` row is at least as large as the other two
for every input, because it takes the single largest bin distance. An input
whose `max` sits far above its `median` acts strongly in one part of its range
and weakly elsewhere.

## Bootstrap confidence intervals

A bootstrap resamples the data many times and recomputes the index on each
resample. The spread of those values gives an interval, which tells you whether
two inputs are really ranked apart or just separated by sampling noise.

```python
result = jaxgsa.pawn.analyze(
    PROBLEM, X, Y,
    n_bootstrap=100,
    conf_level=0.95,
    seed=0,
)

print("PAWN:", result.pawn)
print("95% CI lower:", result.pawn_conf[0])
print("95% CI upper:", result.pawn_conf[1])
```

`pawn_conf` has shape `(2, D)`: row 0 holds the lower bounds and row 1 the
upper bounds, one per input. If the interval of one input overlaps the interval
of another, the data do not separate those two inputs at the 95% level.

## Running on all three benchmark functions

Running the same call on three benchmark functions shows how the index reads
models of different shape. Each benchmark ships its own `PROBLEM`, so nothing
is shared between the three runs except the method.

1. For each benchmark, draw Monte Carlo samples from that benchmark's own
   `PROBLEM`, which carries its own number of inputs and ranges.
2. Evaluate that benchmark's model on its own samples.
3. Call `analyze` with the matching problem, X, and Y.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks import ishigami, sobol_g, linear

# --- Ishigami ---
X_ish = jnp.asarray(jaxgsa.sampling.monte_carlo(ishigami.PROBLEM, n=5000, seed=42))
Y_ish = ishigami.evaluate(X_ish)
r_ish = jaxgsa.pawn.analyze(ishigami.PROBLEM, X_ish, Y_ish)
print("Ishigami PAWN:", r_ish.pawn)

# --- Sobol-G ---
X_sg = jnp.asarray(jaxgsa.sampling.monte_carlo(sobol_g.PROBLEM, n=5000, seed=42))
Y_sg = sobol_g.evaluate(X_sg)
r_sg = jaxgsa.pawn.analyze(sobol_g.PROBLEM, X_sg, Y_sg)
print("Sobol-G PAWN:", r_sg.pawn)

# --- Linear ---
X_lin = jnp.asarray(jaxgsa.sampling.monte_carlo(linear.PROBLEM, n=5000, seed=42))
Y_lin = linear.evaluate(X_lin)
r_lin = jaxgsa.pawn.analyze(linear.PROBLEM, X_lin, Y_lin)
print("Linear PAWN:", r_lin.pawn)
```

Each printed array has one entry per input of that benchmark, so the three
arrays need not have the same length. The indices are not comparable across
benchmarks, only within one. Read each line as a ranking of that model's own
inputs.

## Multi-output example

When Y has shape `(N, K)`, PAWN indices have shape `(K, D)`. N is the number of
samples, K the number of outputs, and D the number of inputs. Here the two
outputs share one X, so a single `analyze` call covers both.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n=3000, seed=42))
Y1 = evaluate(X)
Y2 = jnp.sum(X**2, axis=1)
Y_multi = jnp.column_stack([Y1, Y2])

result = jaxgsa.pawn.analyze(PROBLEM, X, Y_multi)
print("PAWN shape:", result.pawn.shape)  # (2, 3)
```

The printed shape is `(2, 3)`: two rows for the two stacked outputs, three
columns for the three inputs. Row 0 is the Ishigami output `Y1` and row 1 is
the sum of squares `Y2`, in the column order used by `jnp.column_stack`.

## xarray export

`to_dataset()` converts a result to a labeled `xarray.Dataset`, so you can
select by parameter and output name instead of by integer index.

```python
ds = result.to_dataset()
print(ds)

# With bootstrap CIs
result_ci = jaxgsa.pawn.analyze(PROBLEM, X, Y1, n_bootstrap=50, seed=0)
ds_ci = result_ci.to_dataset()
print(ds_ci)  # includes pawn_lower and pawn_upper
```

The first dataset carries the `pawn` variable alone. The second run asks for
bootstrap intervals, so its dataset carries `pawn_lower` and `pawn_upper` as
well. Confidence bounds reach the dataset only when `n_bootstrap` is set.

## Shape rules

| Y shape | pawn | pawn_conf |
|---|---|---|
| `(N,)` | `(D,)` | `(2, D)` or None |
| `(N, K)` | `(K, D)` | `(2, K, D)` or None |
| `(N, T, K)` | `(T, K, D)` | `(2, T, K, D)` or None |

T is the number of time steps. `pawn_conf` is None when you do not ask for a
bootstrap.

## Practical caveats

- PAWN needs no structured sampling. Any (X, Y) pairs work, including
  Monte Carlo, Latin Hypercube, or Sobol sequences.
- The number of bins (`n_bins`) trades off conditioning resolution against
  sample density per bin. The default of 10 works well for N >= 1000.
- With very few samples per bin (< 10), the KS statistic becomes noisy.
  Increase N or decrease `n_bins`.
- The KS statistic is bounded in [0, 1] but sensitive to sample size —
  larger N gives sharper discrimination between inputs.
- Categorical inputs work. A categorical parameter needs no binning: its
  level code already names the conditioning class, so PAWN uses one bin
  per level and `n_bins` does not apply to it. Relabeling the levels does
  not change the index. See
  [Categorical Inputs](/examples/categorical-inputs).

## See also

- [Basic Example](/examples/basic) for the Sobol variance-decomposition
  workflow.
- [Methods](/guide/methods) for a comparison of all methods.
- [API Reference](/api/#given-data-methods) for full parameter documentation.
