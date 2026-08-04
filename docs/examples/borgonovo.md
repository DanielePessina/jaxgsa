# Borgonovo Delta (Moment-Independent Sensitivity)

Borgonovo delta is a moment-independent sensitivity method that measures
how much the entire output density shifts when an input is fixed. It
is the expected L1 distance between the unconditional output density and
the density conditional on each input — no variance decomposition, no
model assumptions. The same analysis also returns the given-data
first-order Sobol index from the same conditioning, for comparison at no
extra cost.

When to use Borgonovo delta:

- You care about influence on the whole output distribution (tails,
  skewness, multimodality) beyond just variance.
- You want a moment-independent index on a fixed [0, 1] scale that is
  invariant under monotone output transformations.
- You have a set of (X, Y) pairs from any sampling strategy — no
  structured design required.

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

The delta index lies in [0, 1] (as do the true index and the raw plug-in
estimate): 0 means fixing the input never changes the output distribution,
higher values mean stronger influence. The default bias-corrected estimate
returned in `result.delta` — and the bounds of `result.delta_conf` — can dip
marginally below 0 for weak or near-noninfluential inputs at small N; that is
expected from the bias correction, not an error. On Ishigami,
`x3` has a first-order Sobol index near zero (it acts only through an
interaction with `x1`), yet its delta index is clearly positive — fixing
`x3` reshapes the output density even though it does not shift the
conditional mean. That gap between `delta` and `S1` is exactly what a
moment-independent index adds.

## Bias correction and bootstrap confidence intervals

The plug-in delta estimator is biased upward at finite N, so by default
(`n_bootstrap=100`, `bias_correct=True`) the central estimate is
bias-corrected with bootstrap resamples (Plischke et al., 2013) and
percentile confidence intervals come from the same replicates.

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
truth rather than another implementation:

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

## See also

- [Basic Example](/examples/basic) for the Sobol variance-decomposition
  workflow.
- [PAWN Example](/examples/pawn) for the CDF-based moment-independent
  method.
- [Methods](/guide/methods) for a comparison of all methods.
- [API Reference](/api/#given-data-methods) for full parameter
  documentation.
