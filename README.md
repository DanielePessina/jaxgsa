# jaxgsa

**Global sensitivity analysis in JAX**

[![PyPI](https://img.shields.io/pypi/v/jaxgsa)](https://pypi.org/project/jaxgsa/)
[![CI](https://github.com/DanielePessina/jaxgsa/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielePessina/jaxgsa/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://danielepessina.github.io/jaxgsa/)
[![Browser workbench](https://img.shields.io/badge/browser%20workbench-WASM-16a6c7)](https://danielepessina.github.io/jaxgsa/workbench/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)

jaxgsa helps you find which inputs drive variation in a model's output. You
choose input distributions, evaluate your model at sample points, and use
jaxgsa to estimate each input's effect. It does not run your model for you.

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

## Choose a method

jaxgsa provides thirteen analysis methods. A useful starting point depends
on the data you can collect and the question you want answered:

| If you have… | Start with… | Why |
| --- | --- | --- |
| Control over future model runs and independent inputs | [Sobol](https://danielepessina.github.io/jaxgsa/examples/basic) | Quantify individual and total effects, plus pairwise interactions when needed. |
| Existing input and output pairs from a smooth model | [PCE](https://danielepessina.github.io/jaxgsa/examples/pce) | Fit a surrogate and estimate variance-based effects; check the fit error. |
| A small budget and many inputs | [Morris](https://danielepessina.github.io/jaxgsa/examples/morris) | Screen for inputs worth studying in more detail. |
| Correlated inputs | [Methods guide](https://danielepessina.github.io/jaxgsa/guide/methods) | Choose a method whose indices have the meaning you need under dependence. |

The [methods guide](https://danielepessina.github.io/jaxgsa/guide/methods)
covers all thirteen methods and their assumptions. The
[API reference](https://danielepessina.github.io/jaxgsa/api/) gives signatures
and result fields.

Scalar, multi-output, and time-series results are supported. Twelve methods
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

## Browser workbench

The separate [browser workbench](https://danielepessina.github.io/jaxgsa/workbench/)
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

For research, cite the exact jaxgsa version used and the primary paper for
each sensitivity method. Use [CITATION.cff](CITATION.cff) for software citation
metadata and the [methods guide](https://danielepessina.github.io/jaxgsa/guide/methods)
for method references.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
