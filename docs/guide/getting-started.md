# Getting Started

jaxgsa answers one practical question: which of your model's inputs drive its
output? You give it a set of input samples and the outputs your model produced
for them. It returns sensitivity indices, that is, numbers that rank the inputs
and show the interactions between them. jaxgsa computes everything in JAX, so
the analysis is JIT-compiled and runs on CPU, GPU, or TPU with no code changes.

This page runs one complete analysis with Sobol indices, the most widely used
method. After it runs, the [Methods guide](/guide/methods) explains how to
choose among the thirteen methods jaxgsa provides.

## Installation

Install the released version from PyPI:

```bash
pip install jaxgsa
# or, with uv:
uv add jaxgsa
```

To install the latest development version from GitHub:

```bash
pip install git+https://github.com/DanielePessina/jaxgsa.git
```

To work on jaxgsa itself, clone the repository and install it in place:

```bash
git clone https://github.com/DanielePessina/jaxgsa.git
cd jaxgsa
uv sync --extra dev   # or: pip install -e ".[dev]"
```

## Your First Analysis

The workflow has four steps:

1. Define the problem: the name and range of each input.
2. Generate the input samples.
3. Evaluate your model at every sample.
4. Compute the Sobol indices.

The model below is the Ishigami function, a standard test function whose
sensitivity indices are known exactly. Replace it with your own model.

```python
import jax.numpy as jnp
import jaxgsa

# 1. Define the problem: parameter names and their ranges
problem = jaxgsa.Problem.from_dict({
    "x1": (-jnp.pi, jnp.pi),
    "x2": (-jnp.pi, jnp.pi),
    "x3": (-jnp.pi, jnp.pi),
})

# 2. Generate input samples. Sobol analysis needs a specific sample layout
#    (a Saltelli design), so use jaxgsa.sobol.sample() rather than random points.
sampling_result = jaxgsa.sobol.sample(problem, n_samples=4096, seed=42)

# 3. Evaluate your model at each sampled input
def model(X):  # Ishigami test function — swap in your own model here
    return (
        jnp.sin(X[:, 0])
        + 7.0 * jnp.sin(X[:, 1]) ** 2
        + 0.1 * X[:, 2] ** 4 * jnp.sin(X[:, 0])
    )

Y = model(sampling_result.samples)  # one output value per sample row

# 4. Compute Sobol indices
result = jaxgsa.sobol.analyze(sampling_result, Y)

print("S1:", result.S1)   # first-order indices
print("ST:", result.ST)   # total-order indices
```

Expected output:

```
S1: [~0.31, ~0.44, ~0.00]
ST: [~0.56, ~0.44, ~0.24]
```

## Reading the Results

Each index is a fraction of the output variance, one value per input:

- **S1 (first-order)** is the share of output variance an input explains on
  its own. Here `x2` has the largest direct effect (~0.44).
- **ST (total-order)** adds every interaction the input takes part in. Use it
  to decide whether you can fix an input to a constant: an input with ST near
  zero has no effect on the output.
- The gap between ST and S1 is that input's interaction share. `x3` shows why
  this matters. Its S1 is about 0, so it has no effect on its own, but its ST
  is about 0.24, so it acts only through its interaction with `x1`. To see
  which pairs cause the interaction, read the pairwise matrix in `result.S2`.

## Define a Problem

A `Problem` gives each input a name and a range. jaxgsa calls these inputs
parameters in code:

```python
from jaxgsa import Problem

problem = Problem.from_dict({
    "x1": (-3.14, 3.14),
    "x2": (-3.14, 3.14),
    "x3": (-3.14, 3.14),
})
```

A plain `(low, high)` tuple means a uniform input. For Gaussian or truncated
Gaussian Sobol inputs, `Problem.from_dict(...)` also accepts tagged
distribution specs. See [Non-Uniform Inputs](/examples/non-uniform-inputs) for
the full `TypedDict` form and the Gaussian truncation rules.

## Save and Reuse Samples

Sampling and model evaluation are often separate steps. The model may run on a
cluster, or it may take hours. `jaxgsa.sobol.sample()` returns a
`SobolSamples` object that you can save and reload later. The reloaded object
keeps the metadata that `jaxgsa.sobol.analyze()` needs:

```python
sampling_result = jaxgsa.sobol.sample(problem, n_samples=4096, seed=42)
sampling_result.save("runs/experiment")

restored = jaxgsa.sobol.SobolSamples.load("runs/experiment")
Y = my_model(restored.samples)
result = jaxgsa.sobol.analyze(restored, Y)
```

This writes `runs/experiment.npz`, containing the sample matrix, problem
definition, and Saltelli reconstruction metadata.

## What's Next?

Start with the core workflow. Then open the page that matches your next
problem:

- [Methods](/guide/methods) -- compare all thirteen methods before choosing a workflow
- [Migrating to 0.4](/guide/migration-0.4) -- update sampling, analysis, prediction, and Shapley calls from 0.3
- [Basic Example (Ishigami)](/examples/basic) -- run the canonical scalar-output Sobol analysis end to end
- [Non-Uniform Inputs](/examples/non-uniform-inputs) -- mix uniform, Gaussian, and truncated Gaussian Sobol marginals in one `Problem`
- [Save and Reload Samples](/examples/save-load) -- persist a `SobolSamples` and reuse it across runs
- [Bootstrap CIs](/examples/bootstrap) -- quantify uncertainty with confidence intervals around `S1`, `ST`, and `S2`
- [Multi-Output & Time-Series](/examples/multi-output) -- move from scalar outputs to `(N, K)` and `(N, T, K)` analyses
- [xarray Output](/examples/xarray) -- export labeled datasets with named parameters, outputs, and time coordinates
- [RS-HDMR](/examples/hdmr) -- switch to surrogate-based analysis when you already have arbitrary `(X, Y)` pairs
- [Advanced Workflow](/examples/advanced-workflow) -- follow the full custom-model path with named outputs, Sobol, HDMR, emulation, and `to_dataset()`
- [Batch Reactor (notebook)](/examples/batch_reactor) -- a self-contained walkthrough of Sobol GSA on a batch reactor with three uniform inputs $(C_{A,0}, T, \mathrm{pH})$, including bootstrap CIs and time-resolved $S_1$ / $S_T$ / $S_{ij}$
- [API Reference](/api/) -- browse the single-page reference for signatures, shape contracts, and result objects
