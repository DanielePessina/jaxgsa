# Basic Example (Ishigami)

Start here if you want the smallest complete Sobol workflow. This page uses the
Ishigami benchmark, then points you to the next examples in the recommended
order.

## Minimal Sobol run

```python
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

# Generate Saltelli samples (unique rows only)
sampling_result = gsax.sample(
    PROBLEM,
    n_samples=4096,
    seed=42,
    calc_second_order=True,
)

# Evaluate your model on the unique rows
Y = evaluate(sampling_result.samples)

# Compute Sobol indices
result = gsax.analyze(sampling_result, Y)

print("S1:", result.S1)
print("ST:", result.ST)
print("S2:", result.S2)
```

Expected output (A=7, B=0.1):

```text
S1: [~0.31, ~0.44, ~0.00]
ST: [~0.56, ~0.44, ~0.24]
```

## What the Ishigami result means

- `x2` has the largest first-order effect, so it explains the biggest share of
  variance by itself.
- `x1` has a moderate main effect but a much larger total effect, which signals
  interactions.
- `x3` has almost no main effect but still matters through interactions with
  `x1`.

## Inspect the unique sample table

`sampling_result.samples_df` is useful when you want to export the sample
matrix, join outputs back onto a run table, or audit which rows were actually
evaluated.

```python
df = sampling_result.samples_df
print(df.head())
#    SampleID        x1        x2        x3
# 0         0 -1.234567  2.345678 -0.123456
# 1         1  0.987654 -1.876543  3.012345
# ...
```

## Practical caveats

- Evaluate `sampling_result.samples`, not an expanded Saltelli matrix. `gsax`
  reconstructs the expanded layout internally.
- `calc_second_order=False` is a good speed and memory tradeoff when you only
  need `S1` and `ST`. In that case `result.S2` is `None`.
- `sample()` may raise the internal base Sobol count until the deduplicated
  sample matrix contains at least `n_samples` unique rows.

## Next examples

Follow these pages in order if you are learning the package:

- [Non-Uniform Inputs](/examples/non-uniform-inputs) for mixed uniform,
  Gaussian, and truncated Gaussian Sobol marginals.
- [Save and Reload Samples](/examples/save-load) for persisting `SamplingResult`
  plus Saltelli reconstruction metadata.
- [Bootstrap Confidence Intervals](/examples/bootstrap) for uncertainty bounds
  and confidence-interval shapes.
- [Multi-Output & Time-Series](/examples/multi-output) for fully runnable
  `(N, T, K)` outputs with named outputs.
- [xarray Labeled Output](/examples/xarray) for turning results into labeled
  datasets and selecting by parameter, output, and time.
- [RS-HDMR Example](/examples/hdmr) for the surrogate-based workflow that works
  with arbitrary `(X, Y)` pairs.
- [Advanced Workflow](/examples/advanced-workflow) for one end-to-end custom
  model that combines Sobol, HDMR, emulator prediction, and `to_dataset()`.

If you want the theory behind the estimators before moving on, read
[Methods](/guide/methods).
