# jaxgsa

**Global Sensitivity Analysis in JAX**

[![PyPI](https://img.shields.io/pypi/v/jaxgsa)](https://pypi.org/project/jaxgsa/)
[![CI](https://github.com/DanielePessina/jaxgsa/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielePessina/jaxgsa/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://danielepessina.github.io/jaxgsa/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)

jaxgsa tells you which of your model's inputs drive its output. You give it
input samples and the outputs your model produced for them. It returns
sensitivity indices that rank the inputs and show their interactions.

Thirteen methods share one interface and one output contract. Eleven of them
are JIT-compiled and vectorized over the output axes. A model with 50
timesteps and 6 outputs then costs one compiled pass, not 300 Python loop
iterations. Those eleven also export a traceable `indices()`, so you can put
the estimator itself under `jit`, `vmap` and `grad`. The other two,
`kucherenko` and `vkoga`, run on the host in NumPy and SciPy by design. They
have no traceable core.

One method, DGSM, needs a model written in JAX so it can take derivatives
instead of running many samples. It costs about one gradient per sample point
on a scalar output. For a model with several outputs, jaxgsa picks whichever
of forward-mode or reverse-mode automatic differentiation is cheaper: reverse
mode computes one output's full gradient per pass, so it gets expensive as
the output count grows, while forward mode computes one input's effect on
every output per pass, so it gets expensive as the input count grows instead.
Later sections call the output shape `(N, T, K)`, for `N` samples, `T` time
steps, and `K` output channels; jaxgsa compares `T * K` against the number of
inputs to pick the cheaper mode.

jaxgsa does not run your model. You control that. It builds designs and reads
indices off the results.

## Install

```bash
pip install jaxgsa
# or
uv add jaxgsa
```

Python 3.12 or newer. The runtime dependencies are `jax`, `jaxlib`, `numpy`,
`scipy`, and `xarray`. Optional extras: `notebook` (marimo, matplotlib) and
`dev` (pytest, ruff, ty, SALib, POT).

### Coding agent skill

The repository ships an agent skill that teaches a coding agent the jaxgsa API:
the sampling designs, all thirteen methods, and the caveats that decide whether
an index means anything. Install it into a project with

```bash
npx skills add https://github.com/DanielePessina/jaxgsa
```

The skill lives at [`skills/jaxgsa/SKILL.md`](skills/jaxgsa/SKILL.md), with one
reference file per method family under `skills/jaxgsa/reference/`. It needs no
extra setup once installed.

## Quickstart

Sobol indices on the Ishigami function, which has known analytic indices.

```python
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

# 1. Build a Saltelli design. `samples` is the (n_runs, D) array to run your model on.
design = jaxgsa.sobol.sample(PROBLEM, n_samples=16384, seed=42)

# 2. Run the model. Here it is the Ishigami test function.
Y = evaluate(design.samples)  # shape (16384,)

# 3. Read the indices off the design and the outputs.
result = jaxgsa.sobol.analyze(design, Y)

print("S1:", result.S1)
print("S2 x1-x3:", result.S2[0, 2])
```

Both `sample` and `analyze` print a summary by default. This is what the script
writes, verbatim, apart from the timing line, which depends on your machine:

```
jaxgsa.sobol.sample: D=3, mode=second-order, base_n=2048, requested_runs>=16384, n_runs=16384, n_expanded=16384, duplicates_removed=0 (0.0%), scramble=True
jaxgsa.sobol.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=16384 runs, T=1 x K=1 output slice
    invalid: none found in 2048 Saltelli groups (policy 'raise')
  timing:
    estimators (includes compile on the first call): 0.7828 s
    slice_chunk_size: 1 (resolved from the memory budget)
    estimator: saltelli-jansen
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.5559
    2. x2  ST=0.4414
    3. x3  ST=0.2415
S1: [ 0.308098   0.4440502 -0.0113217]
S2 x1-x3: 0.24699646
```

Pass `verbose=False` to any `analyze()` or `sample()` call to silence it. The
summary is worth reading once per new problem, because the `invalid` and
`marginals` lines catch the mistakes that silently ruin an analysis.

Every index is a fraction of the output variance. S1 is an input's direct
effect. ST also counts every interaction the input takes part in. Reading the
numbers above:

- `x2` has the largest direct effect, 0.444 of the variance, and S1 equals ST,
  so it acts alone.
- `x1` has S1 = 0.308 but ST = 0.556. The gap is interaction.
- `x3` has S1 = -0.011, which is zero plus estimator noise. On its own it does
  nothing. Yet ST = 0.242. `result.S2[0, 2] = 0.247` names the partner. The
  whole effect of `x3` runs through `x1`. Fixing `x3` at its nominal value
  would still change the output, so you cannot drop it.

A negative S1 is not a bug. The Saltelli estimator is a difference of
Monte Carlo means, so a true zero comes out slightly either side of zero. Read
it as "no direct effect", and read its size as your noise floor.

The analytic answers are S1 = (0.3139, 0.4424, 0) and
ST = (0.5576, 0.4424, 0.2437). At `n_samples=16384` every index above is within
0.012 of the truth. At `n_samples=4096` the worst error is 0.07, on ST for
`x1`. Monte Carlo error falls with the square root of the sample count, so
budget for it. Quadrupling the runs roughly halves the error.

## When you cannot choose the sample points

Sobol indices normally require the Saltelli design: a specific pattern of
sample points, built and evaluated ahead of time. Nine of the thirteen
methods skip that requirement. They work on whatever (X, Y) pairs you already
have, including runs from an old sweep. Polynomial chaos, one of the nine,
fits a polynomial surrogate to those pairs. The polynomials it uses are
mutually orthogonal, which lets it read exact Sobol indices straight off the
fitted coefficients, with no extra integration step.

```python
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jaxgsa.sampling.monte_carlo(PROBLEM, n=2000, seed=0)  # any (N, D) points
Y = evaluate(X)

result = jaxgsa.pce.analyze(PROBLEM, X, Y, order=6, verbose=False)
print("S1:", result.S1)
print("ST:", result.ST)
print("LOO RMSE:", result.loo_rmse, "  output sd:", Y.std())
```

```
S1: [3.1933489e-01 4.4209677e-01 1.2573625e-04]
ST: [0.5573787  0.44309324 0.23836787]
LOO RMSE: 0.52724934   output sd: 3.5930953
```

2000 unstructured points land within 0.0054 of every analytic index. The
Saltelli run above needed 16384 and did worse. That is the surrogate paying
off, and it is why PCE is the right first try for a smooth model.

The catch is that you are now trusting a fit. `loo_rmse` is the number that
decides whether to trust it. Leave-one-out error of 0.527 against an output
standard deviation of 3.59 means the surrogate reproduces about 98% of the
variance. If `loo_rmse` approaches the output standard deviation, the indices
describe the surrogate and not your model. Raise `order`, add samples, or
switch to `jaxgsa.hdmr`, whose B-spline basis handles kinks that polynomials
cannot.

## Multi-output and time series

Every method takes `Y` as scalar `(N,)`, multi-output `(N, K)`, or time-series
`(N, T, K)`. The output axes are never inferred or transposed, and one call
covers all of them.

```python
import jax.numpy as jnp

import jaxgsa
from jaxgsa import Problem

problem = Problem.from_dict(
    {"amplitude": (0.5, 1.5), "decay": (0.1, 1.0), "freq": (1.0, 3.0)}
)
t = jnp.linspace(0.1, 5.0, 50)


def model(X):  # X is (N, 3)
    a, k, w = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    return (a * jnp.exp(-k * t) * jnp.sin(w * t))[:, :, None]  # (N, T=50, K=1)


design = jaxgsa.sobol.sample(problem, n_samples=8192, seed=0, verbose=False)
result = jaxgsa.sobol.analyze(design, model(jnp.asarray(design.samples)), verbose=False)

print("S1 shape:", result.S1.shape)  # (T, K, D)
print("t=0.6  S1:", result.S1[5, 0])
print("t=4.5  S1:", result.S1[44, 0])
```

```
S1 shape: (50, 1, 3)
t=0.6  S1: [0.6211161  0.18043584 0.16506985]
t=4.5  S1: [-0.00563779 -0.01124666  0.4224577 ]
```

150 index sets, one compiled pass, one set of model runs. The ranking flips
along the trajectory. Early on, amplitude explains 62% of the variance. By
t = 4.5 the signal has decayed and only the phase is left, so frequency
explains 42% and amplitude explains nothing. A single index averaged over time
would have hidden both facts.

Watch for zero-variance slices. If your model returns a constant at t = 0, the
indices there are 0/0 and jaxgsa returns NaN with a `JaxgsaWarning` that names
the slice.

## The thirteen methods

| Method | Own design | Reach for it when |
|---|---|---|
| `sobol` | Saltelli | You can still choose where to run the model and you want the reference variance decomposition, S1, ST, and S2. |
| `pce` | given data | The model is smooth. Fewest samples per unit of accuracy, plus an emulator. |
| `hdmr` | given data | Same job as PCE, with B-splines. Better on kinks and non-polynomial shapes. |
| `shapley` | given data | You want one number per input that sums to exactly 1. Computed from a PCE or HDMR fit, with no permutation Monte Carlo. Also available as `result.shapley()` on those results. |
| `efast` | search curves | You want S1 and ST from a plain N x D design instead of Saltelli matrices. |
| `dgsm` | given data + `fn` | The model is JAX-differentiable. Bounds on ST from gradients, at roughly one gradient per sample. You pass the model itself, or a precomputed Jacobian. |
| `morris` | trajectories | The budget is tight. Ranks D inputs in `r * (D + 1)` runs, so you can drop the dead ones before spending on Sobol. |
| `hsic` | given data | You want a dependence test, including nonlinear and heteroscedastic, with permutation p-values. |
| `pawn` | given data | The output is skewed or heavy-tailed and variance is the wrong summary. CDF-based. |
| `borgonovo` | given data | Same reason as PAWN, measured on the density instead of the CDF. Also returns given-data S1. |
| `optimal_transport` | given data | You want to know how an input matters. Each index splits into a mean shift (= S1/2) and a shape change. |
| `vkoga` | given data | The inputs are correlated and you still want variance fractions, split into correlated and uncorrelated parts. |
| `kucherenko` | conditional copula | The inputs are correlated and you would rather run the real model on a dedicated design than fit a surrogate. |

Every method exposes `analyze()`. The design-based ones also expose `sample()`.
Results carry `to_dataset()` for labeled xarray output. Every method except
eFAST and HSIC reports bootstrap confidence intervals through `n_bootstrap`.

The [methods guide](https://danielepessina.github.io/jaxgsa/guide/methods)
carries the estimators, the references, and a capability table that a test
checks against the code.

## Correlated and categorical inputs

Declare a Gaussian-copula correlation matrix on the `Problem` and
`jaxgsa.sampling.monte_carlo` draws from it. Declare a categorical parameter
with `{"dist": "categorical", "probs": [...], "labels": [...]}` and samples
carry integer level codes.

Neither is universally supported, and jaxgsa refuses rather than approximates.
Sobol, Morris, eFAST, PCE, DGSM, and PCE-backed Shapley raise a `ValueError` on
a correlated problem that names the alternatives. Every method whose indices
would depend on the arbitrary order of category codes raises on a categorical
one. Sobol is fine with categoricals, because the Saltelli column-swap scheme
never looks at the values.

See [correlated inputs](https://danielepessina.github.io/jaxgsa/examples/correlated-inputs)
and [categorical inputs](https://danielepessina.github.io/jaxgsa/examples/categorical-inputs).

## Performance

The gain is vectorization over output slices. SALib analyzes each `(t, k)`
slice in a Python loop. jaxgsa fuses the estimators and maps one compiled
kernel over all `T * K` slices, so its cost is nearly flat in output size while
SALib's is linear.

On the widest gap measured, RS-HDMR on 50 timesteps by 6 outputs, jaxgsa runs
in 27.4 ms against 29.06 s for SALib 1.5.2 on the same Apple M1 Pro. That is
1060x, and the baseline is single-process NumPy on one CPU core, which is what
SALib does by default. Do not read it as a claim against a parallel CPU or a
tuned GPU comparison, where published speedups for Monte Carlo GSA are closer
to 13x. Eight cores would already cut 1060x to roughly 130x.

Most of that ratio is Python loop overhead rather than arithmetic. Give the
same RS-HDMR comparison one output slice instead of 300 and the gap falls to
10.9x. Shrink the work further, to Sobol on a scalar output with no bootstrap,
and SALib wins at 0.2 ms against jaxgsa's 0.9 ms, because JAX dispatch costs more
than the arithmetic does.

So output size is what decides. `T * K = 1` gains little and can lose. Time
series and multi-output work is where jaxgsa pays for itself. Any speedup
quoted without its `T` and `K` is meaningless, including the ones above.

Full tables, methodology, and the script are in the
[benchmarks guide](https://danielepessina.github.io/jaxgsa/guide/benchmarks).

```bash
uv run --extra dev benchmark_salib.py
```

## Documentation

- [Getting started](https://danielepessina.github.io/jaxgsa/guide/getting-started)
- [Methods guide](https://danielepessina.github.io/jaxgsa/guide/methods), including the capability table
- [Configuration](https://danielepessina.github.io/jaxgsa/guide/configuration), including 64-bit floats and the persistent compilation cache
- [API reference](https://danielepessina.github.io/jaxgsa/api/)
- [Examples](https://danielepessina.github.io/jaxgsa/examples/basic), one page per method

One configuration note is worth repeating here. JAX defaults to float32 and
silently downcasts float64 arrays. For precision-sensitive Sobol or HSIC work,
call `jax.config.update("jax_enable_x64", True)` before you create the first
array.

## Development

```bash
git clone https://github.com/DanielePessina/jaxgsa.git
cd jaxgsa
uv sync --extra dev
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

jaxgsa's Sobol sampling and analysis workflow follows
[SALib](https://salib.readthedocs.io/), reimplemented for JAX.

## Citing jaxgsa

If jaxgsa contributed to work you publish, please cite it. GitHub reads
[`CITATION.cff`](CITATION.cff), so the "Cite this repository" button on the
repository page gives you APA or BibTeX directly. The BibTeX is:

```bibtex
@software{pessina_jaxgsa_2026,
  author  = {Pessina, Daniele},
  title   = {{jaxgsa}: Global Sensitivity Analysis in {JAX}},
  version = {1.0.1},
  year    = {2026},
  license = {MIT},
  url     = {https://github.com/DanielePessina/jaxgsa}
}
```

Cite the version you ran, not the latest one. Indices move between releases
when an estimator defect is fixed, and `CHANGELOG.md` records every such
change under "Numbers that moved". `jaxgsa.__version__` tells you what you
have.

The methods themselves have their own primary sources, and a paper reporting
Sobol indices should cite Saltelli or Jansen rather than this package alone.
Every method's page in the [Methods
guide](https://danielepessina.github.io/jaxgsa/guide/methods) ends with its
references.

## License

MIT. See [LICENSE](LICENSE).
