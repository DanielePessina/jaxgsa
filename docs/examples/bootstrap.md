# Bootstrap Confidence Intervals

A Sobol index computed from a finite sample is an estimate, and a second run
with a different seed gives a slightly different number. By the end of this
page you will have a lower and an upper bound around each index, so you can
tell a real ranking apart from sampling noise.

The bounds come from a bootstrap. The bootstrap draws many resamples of the
same output array with replacement, recomputes the indices on each resample,
and reports the spread of those repeated values as a confidence interval.

## Scalar-output bootstrap

Turn the bootstrap on with two arguments to `analyze`. `num_resamples` sets how
many resamples to draw, and `key` seeds them, because JAX asks for the random
state to be explicit. Every `*_conf` array on the result then holds the bounds
for the matching index array.

```python
import jax
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

sampling_result = jaxgsa.sobol.sample(PROBLEM, n_samples=4096, seed=42)
Y = evaluate(sampling_result.samples)

result = jaxgsa.sobol.analyze(
    sampling_result,
    Y,
    prenormalize=True,
    num_resamples=200,
    conf_level=0.95,
    ci_method="quantile",
    key=jax.random.key(0),
)

print("S1:", result.S1)
print("ST:", result.ST)
print("S1 lower:", result.S1_conf[0])
print("S1 upper:", result.S1_conf[1])
print("ST lower:", result.ST_conf[0])
print("ST upper:", result.ST_conf[1])
print("S2 lower:", result.S2_conf[0])
print("S2 upper:", result.S2_conf[1])
```

Read each index against its own interval, not against the other indices. If the
interval around `x1`'s `S1` overlaps the interval around `x3`'s `S1`, the run
does not support ranking those two inputs, however far apart the point
estimates sit. An interval whose lower bound is above zero is the evidence that
an input matters at all. To narrow an interval, raise `n_samples` in
`sobol.sample` and evaluate the model again. Raising `num_resamples` measures
the same uncertainty more precisely; it does not reduce it.

## Confidence-interval shapes

The bootstrap adds a leading dimension of 2 for `[lower, upper]`:

- scalar output: `S1_conf.shape == (2, D)`
- multi-output: `S1_conf.shape == (2, K, D)`
- time-series multi-output: `S1_conf.shape == (2, T, K, D)`

`S2_conf` follows the same rule with two trailing parameter axes.

`ci_method` only changes how those two endpoints are summarized from the
bootstrap draws:

- `quantile` uses percentile bootstrap endpoints.
- `gaussian` uses symmetric gaussian endpoints from the bootstrap standard
  deviation around the point estimate.

## Practical caveats

- A `jax.random.key(...)` is required when `num_resamples > 0`.
- `prenormalize=True` applies SALib-style output standardization once over the
  sample axis before the bootstrap starts. The resamples reuse that transformed
  output array; they are not re-standardized per resample.
- Set `num_resamples=0` to skip bootstrap entirely when you only need point
  estimates.
- If `calc_second_order=False` during sampling, then `result.S2` and
  `result.S2_conf` are both `None`.
- Bootstrap intervals follow the same output-shape rules as the point estimates,
  so the page on [Multi-Output & Time-Series](/examples/multi-output) is the
  right companion when your model is not scalar.
- Confidence intervals always remain lower/upper endpoint arrays even when
  `prenormalize=True`. `ci_method="gaussian"` is closer to SALib's CI
  construction, but `jaxgsa` still returns endpoints rather than SALib-style
  confidence half-widths.

## See also

- [Save and Reload Samples](/examples/save-load) if you want to bootstrap a
  stored design.
- [Multi-Output & Time-Series](/examples/multi-output) for concrete shape
  examples on `(N, K)` and `(N, T, K)` outputs.
- [xarray Labeled Output](/examples/xarray) for exporting confidence intervals
  as `_lower` and `_upper` dataset variables.
