# Bootstrap Confidence Intervals

Use bootstrap resampling when you need uncertainty bounds for Sobol indices,
not just point estimates.

## Scalar-output bootstrap

```python
import jax
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)
Y = evaluate(sampling_result.samples)

result = gsax.analyze(
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
  construction, but `gsax` still returns endpoints rather than SALib-style
  confidence half-widths.

## See also

- [Save and Reload Samples](/examples/save-load) if you want to bootstrap a
  stored design.
- [Multi-Output & Time-Series](/examples/multi-output) for concrete shape
  examples on `(N, K)` and `(N, T, K)` outputs.
- [xarray Labeled Output](/examples/xarray) for exporting confidence intervals
  as `_lower` and `_upper` dataset variables.
