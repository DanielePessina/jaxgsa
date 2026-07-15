# Getting Started

gsax answers a practical question: **which of your model's inputs actually
drive its output?** You give it a set of input samples and the outputs your
model produced for them; it returns sensitivity indices that rank the inputs
and expose interactions between them. Everything is computed in JAX, so the
analysis is JIT-compiled and runs on CPU, GPU, or TPU without code changes.

This page walks through one complete analysis with Sobol indices, the most
widely used method. Once it runs, the [Methods guide](/guide/methods) explains
how to choose among the eleven methods gsax provides.

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

## Your First Analysis

The workflow has four steps: define the problem, generate samples, evaluate
your model, analyze. The model here is the Ishigami function, a standard test
function whose sensitivity indices are known exactly — replace it with your
own.

```python
import jax.numpy as jnp
import gsax

# 1. Define the problem: parameter names and their ranges
problem = gsax.Problem.from_dict({
    "x1": (-jnp.pi, jnp.pi),
    "x2": (-jnp.pi, jnp.pi),
    "x3": (-jnp.pi, jnp.pi),
})

# 2. Generate input samples. Sobol analysis needs a specific sample layout
#    (a Saltelli design), so use gsax.sample() rather than random points.
sampling_result = gsax.sample(problem, n_samples=4096, seed=42)

# 3. Evaluate your model at each sampled input
def model(X):  # Ishigami test function — swap in your own model here
    return (
        jnp.sin(X[:, 0])
        + 7.0 * jnp.sin(X[:, 1]) ** 2
        + 0.1 * X[:, 2] ** 4 * jnp.sin(X[:, 0])
    )

Y = model(sampling_result.samples)  # one output value per sample row

# 4. Compute Sobol indices
result = gsax.analyze(sampling_result, Y)

print("S1:", result.S1)   # first-order indices
print("ST:", result.ST)   # total-order indices
```

Expected output:

```
S1: [~0.31, ~0.44, ~0.00]
ST: [~0.56, ~0.44, ~0.24]
```

## Reading the Results

Each index is a fraction of the output variance, one value per parameter:

- **S1 (first-order)** is the share of output variance each input explains
  *on its own*. Here `x2` has the largest direct effect (~0.44).
- **ST (total-order)** adds every interaction the input takes part in. It's
  the right number for asking "can I fix this input to a constant?" — an
  input with ST near zero doesn't matter at all.
- The **gap between ST and S1** is that input's interaction share. `x3` is
  the interesting case: S1 ≈ 0 (no effect alone) but ST ≈ 0.24, so it matters
  only through its interaction with `x1`. `result.S2` holds the pairwise
  interaction matrix if you want to see which pairs are responsible.

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

Plain `(low, high)` tuples mean uniform inputs. For Gaussian or truncated
Gaussian Sobol inputs, `Problem.from_dict(...)` also accepts tagged
distribution specs. See [Non-Uniform Inputs](/examples/non-uniform-inputs) for
the full `TypedDict` form and the Gaussian truncation rules.

## Save and Reuse Samples

Generating samples and evaluating the model are often separate steps — the
model may run on a cluster, or take hours. `gsax.sample()` returns a
`SamplingResult` that you can persist and reload later without losing the
metadata `gsax.analyze()` needs:

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

- [Methods](/guide/methods) -- compare all eleven methods before choosing a workflow
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
