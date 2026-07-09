# Getting Started

## Installation

```bash
pip install gsax
# or, with uv:
uv add gsax
```

To install the latest development version from GitHub:

```bash
pip install git+https://github.com/DanielePessina/gsax.git
```

For local development:

```bash
git clone https://github.com/DanielePessina/gsax.git
cd gsax
uv sync --extra dev   # or: pip install -e ".[dev]"
```

## Quick Start

```python
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

# 1. Generate unique Sobol/Saltelli samples
sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)

# 2. Evaluate your model on the unique samples
Y = evaluate(sampling_result.samples)

# 3. Compute Sobol indices
result = gsax.analyze(sampling_result, Y)

print("S1:", result.S1)   # first-order indices
print("ST:", result.ST)   # total-order indices
print("S2:", result.S2)   # second-order interaction matrix
```

Expected output (Ishigami function, A=7, B=0.1):

```
S1: [~0.31, ~0.44, ~0.00]
ST: [~0.56, ~0.44, ~0.24]
```

## Define a Problem

A `Problem` specifies parameter names and bounds:

```python
from gsax import Problem

problem = Problem.from_dict({
    "x1": (-3.14, 3.14),
    "x2": (-3.14, 3.14),
    "x3": (-3.14, 3.14),
})
```

For mixed Sobol inputs, `Problem.from_dict(...)` also accepts tagged
distribution specs. See [Non-Uniform Inputs](/examples/non-uniform-inputs) for
the full `TypedDict` form and the Gaussian truncation rules.

## Save and Reuse Samples

`gsax.sample()` returns a `SamplingResult` that you can persist and reload
later without losing the metadata needed by `gsax.analyze()`:

```python
sampling_result = gsax.sample(problem, n_samples=4096, seed=42)
sampling_result.save("runs/experiment", format="csv")

restored = gsax.load("runs/experiment", format="csv")
Y = my_model(restored.samples)
result = gsax.analyze(restored, Y)
```

This writes a sample file such as `runs/experiment.csv`, a metadata file
`runs/experiment.json`, and an optional `runs/experiment.npz` sidecar when the
expanded Saltelli layout cannot be reconstructed with an identity mapping alone.

## What's Next?

Start with the core workflow, then branch into the example that matches your
next problem:

- [Methods](/guide/methods) -- compare Sobol sampling, RS-HDMR, and PCE before choosing a workflow
- [Basic Example (Ishigami)](/examples/basic) -- run the canonical scalar-output Sobol analysis end to end
- [Non-Uniform Inputs](/examples/non-uniform-inputs) -- mix uniform, Gaussian, and truncated Gaussian Sobol marginals in one `Problem`
- [Save and Reload Samples](/examples/save-load) -- persist a `SamplingResult` and reuse it across runs
- [Bootstrap CIs](/examples/bootstrap) -- quantify uncertainty with confidence intervals around `S1`, `ST`, and `S2`
- [Multi-Output & Time-Series](/examples/multi-output) -- move from scalar outputs to `(N, K)` and `(N, T, K)` analyses
- [xarray Output](/examples/xarray) -- export labeled datasets with named parameters, outputs, and time coordinates
- [RS-HDMR](/examples/hdmr) -- switch to surrogate-based analysis when you already have arbitrary `(X, Y)` pairs
- [Advanced Workflow](/examples/advanced-workflow) -- follow the full custom-model path with named outputs, Sobol, HDMR, emulation, and `to_dataset()`
- [Batch Reactor (notebook)](/examples/batch_reactor) -- a self-contained walkthrough of Sobol GSA on a batch reactor with three uniform inputs $(C_{A,0}, T, \mathrm{pH})$, including bootstrap CIs and time-resolved $S_1$ / $S_T$ / $S_{ij}$
- [API Reference](/api/) -- browse the single-page reference for signatures, shape contracts, and result objects
