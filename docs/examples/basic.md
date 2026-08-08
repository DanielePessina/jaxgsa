# Basic Example (Ishigami)

By the end of this page you will have run a complete Sobol analysis on a
three-input test function and read three sets of numbers off it: how much of
the output variance each input explains on its own, how much it explains in
total, and how much comes from each pair of inputs. The test function is the
Ishigami function, a standard sensitivity benchmark with known answers.

A Sobol index is a fraction of output variance attributed to an input. The
first-order index `S1` covers the input acting alone. The total index `ST`
covers the input acting alone plus every interaction it takes part in. The
second-order index `S2` covers one pair of inputs acting together.

## Minimal Sobol run

The run has four steps:

1. Build a Saltelli design. The Sobol estimators need a specific pattern of
   sample rows, not free-form random points, so the design comes from
   `jaxgsa.sobol.sample` rather than from your own sampler.
2. Run your model on every row of that design. This is the only expensive
   step, and it is the reason the sampler returns unique rows only.
3. Pass the design object and the outputs to `jaxgsa.sobol.analyze`. The
   design object carries the layout metadata, so the analyzer can tell which
   row belongs where.
4. Read `S1`, `ST` and `S2` off the result.

```python
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

# Generate Saltelli samples (unique rows only)
sampling_result = jaxgsa.sobol.sample(
    PROBLEM,
    n_samples=4096,
    seed=42,
    calc_second_order=True,
)

# Evaluate your model on the unique rows
Y = evaluate(sampling_result.samples)

# Compute Sobol indices
result = jaxgsa.sobol.analyze(sampling_result, Y)

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

Read the two arrays side by side, one input at a time. The three entries are
`x1`, `x2`, `x3` in the order the problem declares them.

- `x2` has `S1` near 0.44 and `ST` near 0.44 as well. The two agree, so `x2`
  acts on its own and takes part in no interaction.
- `x1` has `S1` near 0.31 but `ST` near 0.56. Roughly a quarter of the output
  variance therefore comes from `x1` interacting with something else, not from
  `x1` alone.
- `x3` has `S1` near 0.00 and `ST` near 0.24. Varying `x3` on its own does
  nothing to the output, yet `x3` still drives a quarter of the variance. It
  can only do that through its interaction with `x1`.

The practical reading: you may not freeze `x3` at a nominal value even though
its main effect is zero. A screening method that looks only at main effects
would have thrown it away.

## Free Morris screening from the same run

Morris screening measures an elementary effect: the change in the output when
one input moves by one step and the others stay fixed. It needs a design where
sample rows differ in exactly one input at a time.

A Saltelli design already contains such a design, called a Morris radial (star)
design: within each base point, `A` and each `AB_j` differ in exactly one
parameter. So you can read elementary-effect screening measures straight off a
design you have already evaluated. This needs no extra model runs:

```python
morris_result = jaxgsa.morris.analyze(sampling_result.to_morris(), Y)

print("mu_star:", morris_result.mu_star)  # importance ranking
print("sigma:  ", morris_result.sigma)    # nonlinearity / interaction flag
```

`mu_star` is the mean absolute elementary effect, which ranks inputs by
importance. `sigma` is the spread of the elementary effects for one input.

`sigma` is worth having even when you already have `S2`. A large
`sigma / mu_star` says an input's effect changes across the domain. The two
analyses share the same model outputs, so this is not an independent
confirmation of the Sobol indices. See
[Morris](/examples/morris#from-an-existing-sobol-design) for the derivation and
its caveats.

## Export the unique sample matrix

`sampling_result.samples` stays a plain NumPy array. Write it out with NumPy
when another process needs the design as a table:

```python
import numpy as np

np.savetxt("samples.csv", sampling_result.samples, delimiter=",")
```

## Practical caveats

- Evaluate `sampling_result.samples`, not an expanded Saltelli matrix. `jaxgsa`
  reconstructs the expanded layout internally.
- `calc_second_order=False` is a good speed and memory tradeoff when you only
  need `S1` and `ST`. In that case `result.S2` is `None`.
- `sample()` may raise the internal base Sobol count until the deduplicated
  sample matrix contains at least `n_samples` unique rows.

## Next examples

Follow these pages in order if you are learning the package:

- [Non-Uniform Inputs](/examples/non-uniform-inputs) for mixed uniform,
  Gaussian, and truncated Gaussian Sobol marginals.
- [Save and Reload Samples](/examples/save-load) for persisting `SobolSamples`
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
