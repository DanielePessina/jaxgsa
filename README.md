# jaxgsa

**Global sensitivity analysis in JAX**

[![PyPI](https://img.shields.io/pypi/v/jaxgsa)](https://pypi.org/project/jaxgsa/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22099117.svg)](https://doi.org/10.5281/zenodo.22099117)
[![CI](https://github.com/DanielePessina/jaxgsa/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielePessina/jaxgsa/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://danielepessina.github.io/jaxgsa/)
[![jaxgsa studio](https://img.shields.io/badge/jaxgsa-studio-16a6c7)](https://danielepessina.github.io/jaxgsa/studio/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)

jaxgsa helps you find which inputs drive variation in a model's output. You
choose input distributions, evaluate your model at sample points, and use
jaxgsa to estimate each input's effect. It does not run your model for you.

## Why jaxgsa

- **Many outputs in one call.** Time series and multi-output models are
  analysed in one compiled JAX call, not one Python loop per output. On 300
  output slices, Sobol with second-order indices is 35x faster than SALib, and
  HDMR is about 1000x faster
  ([benchmarks](https://danielepessina.github.io/jaxgsa/guide/benchmarks)).
  For one scalar output, SALib is as fast or faster.
- **Gradients come free.** DGSM gets model derivatives from JAX, and the
  estimator cores are differentiable, so an index can sit inside a larger
  JAX computation
  ([differentiation guide](https://danielepessina.github.io/jaxgsa/guide/differentiation)).
- **Fourteen methods behind one interface.** Variance-based, screening,
  moment-independent, kernel and surrogate methods share the same `Problem`,
  sample and result types.
- **Dependent inputs.** Correlated, custom-sampled and categorical inputs are
  supported, with methods whose indices keep their meaning under dependence.
- **CPU, GPU or TPU.** The same code runs on any device JAX supports.

## Install

```bash
uv add jaxgsa
```

Python 3.12 or newer is required. If you do not use uv, install with
`pip install jaxgsa`.

## First analysis

This example uses the Ishigami test model, whose sensitivity indices are
known. Save it as `first_analysis.py` and run `uv run first_analysis.py`:

```python
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

# Draw the structured input values Sobol analysis needs.
design = jaxgsa.sobol.sample(PROBLEM, n_samples=4096, seed=42, verbose=False)

# In your work, replace this call with your model evaluated at design.samples.
Y = evaluate(design.samples)

result = jaxgsa.sobol.analyze(design, Y, verbose=False)
print("S1:", result.S1)
print("ST:", result.ST)
```

`S1` estimates how much output variation an input explains on its own.
`ST` also counts its interactions with other inputs. For this run, `x3`
has `S1` near zero but `ST` around 0.24: it matters through an interaction
with `x1`. Looking at `S1` alone would miss it. These numbers are estimates,
so check their uncertainty before making a close decision.

The [getting-started tutorial](https://danielepessina.github.io/jaxgsa/guide/getting-started)
explains each step and how to interpret the result.

## Capabilities

jaxgsa provides fourteen analysis methods. Which one fits depends on your
model and your question. The table lists what the package supports and
where to find a worked example:

| Capability | Methods | Example |
| --- | --- | --- |
| Generate a sampling design | `sobol`, `efast`, `morris`, `kucherenko` | [Basic example](https://danielepessina.github.io/jaxgsa/examples/basic) |
| Analyse existing `X`, `Y` with no special design | `borgonovo`, `hdmr`, `hsic`, `optimal_transport`, `pawn`, `pce`, `rank`, `shapley`, `vkoga` | [Existing data](https://danielepessina.github.io/jaxgsa/guide/existing-data) |
| Drop failed or NaN model runs (`on_invalid="drop"`) | every method except `efast` | [Failed runs](https://danielepessina.github.io/jaxgsa/guide/failed-runs) |
| Many outputs, or an output over time, in one call | all methods | [Multi-output example](https://danielepessina.github.io/jaxgsa/examples/multi-output) |
| Correlated inputs | `borgonovo`, `hdmr`, `hsic`, `kucherenko`, `optimal_transport`, `pawn`, `rank`, `vkoga`, `shapley` with `backend="hdmr"` | [Dependent inputs](https://danielepessina.github.io/jaxgsa/guide/dependent-inputs) |
| Categorical inputs | `borgonovo`, `optimal_transport`, `pawn`, `sobol` | [Categorical example](https://danielepessina.github.io/jaxgsa/examples/categorical-inputs) |
| Gradients of an index | differentiable estimator cores | [Differentiation guide](https://danielepessina.github.io/jaxgsa/guide/differentiation) |
| Bootstrap confidence intervals (`n_bootstrap`) | every method except `efast` and `hsic` (HSIC reports permutation p-values) | [Bootstrap example](https://danielepessina.github.io/jaxgsa/examples/bootstrap) |

The [methods guide](https://danielepessina.github.io/jaxgsa/guide/methods)
describes what each method measures and its assumptions. The
[API reference](https://danielepessina.github.io/jaxgsa/api/) gives signatures
and result fields.

Scalar, multi-output, and time-series results are supported. Thirteen methods
also accept output channels with separate, irregular time grids. See the
[multi-output](https://danielepessina.github.io/jaxgsa/examples/multi-output)
and [irregular-output](https://danielepessina.github.io/jaxgsa/examples/irregular-outputs)
examples.

## More to read

- [Concepts](https://danielepessina.github.io/jaxgsa/guide/concepts): what
  sensitivity indices mean.
- [Ishigami example](https://danielepessina.github.io/jaxgsa/examples/basic):
  sampling error and convergence.
- [Configuration](https://danielepessina.github.io/jaxgsa/guide/configuration):
  precision, memory budget, and compilation.
- [Benchmarks](https://danielepessina.github.io/jaxgsa/guide/benchmarks):
  measurement methods and performance results.

## jaxgsa studio

The separate [jaxgsa studio](https://danielepessina.github.io/jaxgsa/studio/)
runs ported analyses locally in the browser. Its source and development instructions
are in `web/`. The Python
package and documentation site have separate builds.

## Develop

```bash
git clone https://github.com/DanielePessina/jaxgsa.git
cd jaxgsa
uv sync --extra dev
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md). To run the docs locally:

```bash
npm --prefix docs install
npm --prefix docs run docs:dev
```

The repository also includes a [coding agent skill](skills/jaxgsa/SKILL.md)
covering the API and method-specific caveats.

## Cite jaxgsa

For research, please cite jaxgsa and the primary paper for each sensitivity
method (see the
[methods guide](https://danielepessina.github.io/jaxgsa/guide/methods)).
For the software citation, use the general DOI
[10.5281/zenodo.22099117](https://doi.org/10.5281/zenodo.22099117), which
covers all versions:

In BibTeX:

```bibtex
@software{pessina2026jaxgsa,
  author    = {Pessina, Daniele and Papathanasiou, Maria M.},
  title     = {jaxgsa: Global Sensitivity Analysis in JAX},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22099117},
  url       = {https://doi.org/10.5281/zenodo.22099117}
}
```

Or as a text citation:

> Daniele Pessina and Maria M. Papathanasiou (2026). *jaxgsa: Global
> Sensitivity Analysis in JAX*. Zenodo.
> https://doi.org/10.5281/zenodo.22099117

To cite a specific release, use its version DOI instead. The
[Zenodo record](https://doi.org/10.5281/zenodo.22099117) lists the DOI of
each release. See [CITATION.cff](CITATION.cff) for machine-readable citation
metadata.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
