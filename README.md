# jaxgsa

**Global Sensitivity Analysis in JAX**

[![PyPI](https://img.shields.io/pypi/v/jaxgsa)](https://pypi.org/project/jaxgsa/)
[![CI](https://github.com/DanielePessina/jaxgsa/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielePessina/jaxgsa/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://danielepessina.github.io/jaxgsa/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)

`jaxgsa` tells you which of your model's inputs actually drive its output. You give it input samples and the outputs your model produced for them; it returns sensitivity indices that rank the inputs and expose interactions. Everything is computed in JAX, so analyses are JIT-compiled and run on CPU, GPU, or TPU without code changes.

Thirteen complementary methods are included. They fall into four groups.

Variance-based methods split the output variance among the inputs.

| Method | What it does |
|---|---|
| Sobol indices | The standard variance decomposition, via Saltelli sampling. |
| RS-HDMR | Surrogate-based: fits a cheap approximation of your model to any existing input–output pairs and reads the indices off the fit. |
| PCE | Surrogate-based in the same way as RS-HDMR. |
| Shapley effects | A fair, game-theoretic split of the output variance. Computed analytically from a PCE or HDMR surrogate. |
| eFAST | Fourier-based S1 and ST. |
| DGSM | Derivative-based bounds, via JAX autodiff. |
| VKOGA | Variance-based indices for correlated inputs, from a greedy kernel surrogate under a Gaussian copula (Hilhorst et al., 2024). |
| Kucherenko indices | Sobol' indices for dependent inputs. Estimated by evaluating the model on a conditional-copula design (Kucherenko et al., 2012). |

Screening: Morris runs cheap elementary-effects screening, to discard
unimportant inputs early.

Moment-independent methods look at the whole output distribution rather than
just its variance.

| Method | What it does |
|---|---|
| PAWN | CDF-based (Pianosi & Wagener, 2015). |
| Borgonovo delta | Density-based (Borgonovo, 2007). |
| Optimal-transport indices | Wasserstein-based, with an advective/diffusive decomposition (Borgonovo et al., 2024). |

Dependence: HSIC does kernel-based dependence detection.

## Features

- **Sobol indices** via Saltelli sampling with Sobol quasi-random sequences (`scipy.stats.qmc`)
  - First-order (S1: an input's direct share of output variance), total-order (ST: including all its interactions), and second-order (S2: pairwise interactions)
  - Fused JIT kernels and chunked `jit(vmap(...))` execution for bounded memory on large output grids
  - [Up to 668× faster than SALib](#benchmark-results) (HDMR on multi-output workloads)
- **RS-HDMR** (Random Sampling High-Dimensional Model Representation)
  - Works with any set of (X, Y) pairs — no structured sampling required
  - B-spline surrogate with ANCOVA decomposition (Sa, Sb, S, ST)
  - Built-in emulator for prediction at new inputs
  - S1/ST properties for direct comparison with Sobol results. Under
    independent inputs `ST` is the ordinary Sobol' total-order index. Under
    correlated inputs it is the SCSA total of Li et al. (2010, Section 2.2.3):
    it can be negative, it is not bounded in `[0, 1]`, and it must not be used
    to decide whether a parameter can be fixed. Use `jaxgsa.kucherenko` or
    `jaxgsa.vkoga` for a conditional-variance total under dependence.
- **PCE** (Polynomial Chaos Expansion)
  - Analytical Sobol indices from orthogonal polynomial coefficients (Sudret, 2008)
  - Wiener-Askey scheme: Legendre for uniform, Hermite for Gaussian inputs
  - Built-in emulator and leave-one-out cross-validation RMSE
  - Scalar, multi-output, and time-series outputs. All output slices share one basis, fitted in a single multi-right-hand-side solve
- **Shapley effects** (Owen, 2014; Song, Nelson & Staum, 2016)
  - Fair, game-theoretic allocation of output variance. Each interaction's variance is split equally among its participants
  - Computed analytically from a fitted PCE (default) or RS-HDMR surrogate: no permutation Monte Carlo, no extra model runs
  - Works with any set of (X, Y) pairs. Returns Sh alongside S1 and ST from the same surrogate, with S1 <= Sh <= ST
  - Assumes independent inputs (v1). Sh sums to 1. `explained_variance` reports the fraction of Var(Y) the surrogate captured
- **eFAST** (Extended Fourier Amplitude Sensitivity Test)
  - Frequency-based S1 and ST via sinusoidal search curves and Fourier decomposition
  - Supports scalar, multi-output, and time-series outputs
  - Simple N x D sampling design, no cross-matrix structure needed
- **DGSM** (Derivative-based Global Sensitivity Measures)
  - Upper and lower bounds on total Sobol index via JAX reverse-mode autodiff
  - Poincare constants for uniform, Gaussian, and truncated Gaussian inputs
  - Pre-computed Jacobian path for non-JAX models
- **Morris** (elementary-effects screening)
  - Globalized one-at-a-time screening. It gives a mu_star importance ranking and a sigma interaction flag from only `r * (D + 1)` model runs, where r is the trajectory count and D the input count
  - Trajectory (Morris, 1991) and radial (Campolongo et al., 2011) designs with unique-row deduplication
  - Bootstrap confidence intervals over trajectories and prefix-nested trajectory downsampling
- **HSIC** (Hilbert–Schmidt Independence Criterion)
  - Kernel-based dependence: normalized first-order (R2-HSIC) and Total HSIC indices
  - Works with any set of (X, Y) pairs — Gaussian RBF kernels with the median heuristic
  - Detects nonlinear, non-monotone, and heteroscedastic dependence. Reports permutation-test p-values
- **PAWN** — moment-independent, CDF-based sensitivity (Pianosi & Wagener, 2015)
  - Kolmogorov–Smirnov distance between unconditional and conditional output CDFs
  - Tie-aware KS matching `scipy.stats.ks_2samp` for discrete/continuous outputs
  - Median / max / mean aggregation with bootstrap confidence intervals
- **Borgonovo delta** — moment-independent, density-based sensitivity (Borgonovo, 2007)
  - Plischke et al. (2013) given-data estimator: works with any set of (X, Y) pairs
  - Bias-corrected delta plus the given-data first-order Sobol S1 (SALib-compatible)
  - Percentile bootstrap confidence intervals
- **Optimal transport** — Wasserstein-based distributional sensitivity (Borgonovo et al., 2024)
  - Given-data estimator on any (X, Y) pairs; rank-based conditioning handles mixed uniform/Gaussian marginals and correlated inputs
  - Advective (mean-shift, = S1/2) vs diffusive (spread/shape) decomposition of every index
  - Per-column indices via exact 1-D transport (solver-free). Joint point-cloud modes over multivariate/time-series outputs use a pure-JAX log-domain Sinkhorn. A dummy-input irrelevance baseline is included
- **Correlated inputs** — declare a Gaussian-copula correlation matrix on `Problem`
  - Declare it with `correlation=` on the latent or Spearman scale, or with `problem.with_correlation(R)`
  - `jaxgsa.sampling.monte_carlo` draws from it transparently. `correlate()` retrofits it onto existing samples. `fit_correlation()` estimates it from data
  - Correlation-tolerant methods (OT, Borgonovo, HDMR, HSIC, PAWN, VKOGA) analyze it, and the Kucherenko design conditions on it
  - Correlation-naive methods (Sobol, Morris, eFAST, PCE, DGSM, PCE-backed Shapley) refuse it with an actionable error
- **Categorical inputs** — declare unordered discrete marginals with `{"dist": "categorical", "probs": [...], "labels": [...]}`
  - Samples carry integer level codes
  - Optimal transport, Borgonovo, and PAWN condition on one class per level
  - The Saltelli-based Sobol pipeline works unchanged
  - Every code-order-sensitive method (Morris, eFAST, DGSM, PCE, HDMR, HSIC, Shapley) refuses with a clear error
- **VKOGA** — variance-based indices for correlated inputs (Hilhorst et al., 2024; Li et al., 2010)
  - Given-data method: fits a greedy Gaussian-RBF kernel surrogate to whatever (X, Y) pairs you have
  - Gaussian copula for the dependency structure: reads `problem.correlation`, with a per-call matrix override
  - Splits each input's effect into correlated and uncorrelated parts (S_TC, S_TU, S_U, S_C, S_IU). S_TC ranks inputs for measurement, S_TU for fixing
  - Built-in emulator for prediction at new inputs
- **Kucherenko indices** — design-based Sobol' indices for dependent inputs (Kucherenko et al., 2012)
  - `kucherenko.sample` builds a conditional-copula design from `problem.correlation`, and is exempt from the correlated-design error. You evaluate your actual model on it. No surrogate is fitted
  - S1 is correlation-inclusive (VKOGA's S_TC) and ST is correlation-exclusive (VKOGA's S_TU). With independent inputs both reduce to the classic Sobol' indices
  - Cross-validated against VKOGA and the analytic linear-Gaussian closed form in the test suite
- All thirteen methods use one strict output contract: scalar `(N,)`, multi-output `(N, K)`, or time-series `(N, T, K)`
- Bootstrap confidence intervals with JAX-accelerated resampling
- Sobol analysis always standardizes each output slice before the estimators, so a
  large output mean cannot bias S1 or S2
- `standardize_outputs=True` on Morris and DGSM reports their dimensional
  quantities in units of the output standard deviation
- `on_invalid` on every `analyze()`: a non-finite model output raises by default,
  and `"drop"` or `"propagate"` say what to do instead. Every result carries a
  report naming the runs that failed
- **xarray integration** — `to_dataset()` on results for labeled, named dimensions (`param`, `output`, `time`)
- Save and reload sample designs as one NPZ file via `SobolSamples.save()` / `.load()` (and the same on `MorrisSamples`)
- Built-in Ishigami benchmark function with known analytical solutions

## Installation

```bash
pip install jaxgsa
# or, with uv:
uv add jaxgsa
```

To install the latest development version from GitHub:

```bash
pip install git+https://github.com/DanielePessina/jaxgsa.git
```

For local development:

```bash
git clone https://github.com/DanielePessina/jaxgsa.git
cd jaxgsa
uv sync --extra dev   # or: pip install -e ".[dev]"
```

## Configuration

jaxgsa inherits JAX's runtime defaults. Two optional knobs, documented in full in
the [Configuration guide](https://danielepessina.github.io/jaxgsa/guide/configuration):

- **Double precision** — JAX defaults to float32 and silently downcasts `float64`.
  For precision-sensitive Sobol/HSIC work, enable 64-bit floats before the first
  array is created: `jax.config.update("jax_enable_x64", True)`.
- **Persistent compilation cache** — reuse compiled kernels across process
  restarts (sweeps, CI, HPC) by calling
  `jaxgsa.config.enable_compilation_cache("~/.cache/jaxgsa-jax")` once, before your first analysis.

## Quick Start

```python
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

# 1. Generate unique Sobol/Saltelli samples
sampling_result = jaxgsa.sobol.sample(PROBLEM, n_samples=4096, seed=42)
# sampling_result.samples.shape == (n_runs, D)  — D parameters, n_runs unique rows
# sampling_result.n_expanded is the internal Saltelli row count used by analyze()
# by default, sample() also prints a short summary of unique vs expanded rows

# 2. Evaluate your model on the samples
Y = evaluate(sampling_result.samples)  # Y.shape == (n_runs,)

# 3. Compute Sobol indices
result = jaxgsa.sobol.analyze(
    sampling_result,
    Y,
)
# result.S1.shape == (D,)    — first-order indices
# result.ST.shape == (D,)    — total-order indices
# result.S2.shape == (D, D)  — second-order interaction matrix

print("First-order indices (S1):", result.S1)
print("Total-order indices (ST):", result.ST)
print("Second-order indices (S2):")
print(result.S2)
```

Expected output (Ishigami function with A=7, B=0.1):

```
First-order indices (S1): [~0.31, ~0.44, ~0.00]
Total-order indices (ST): [~0.56, ~0.44, ~0.24]
```

Each index is a fraction of the output variance. S1 is an input's direct
effect. ST also counts every interaction it takes part in. Here `x2` has the
largest direct effect. `x3` has no direct effect at all (S1 ≈ 0), but it still
matters through its interaction with `x1` (ST ≈ 0.24). `result.S2` shows which
pairs are responsible.

### RS-HDMR (surrogate-based)

RS-HDMR fits a surrogate to any existing (X, Y) pairs, then computes the
indices from the fit. The surrogate is a cheap spline approximation of your
model. Use it when your model runs already exist and rerunning on a Saltelli
design isn't an option.

```python
import jax
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

# 1. Generate any set of input samples (no structured sampling needed)
key = jax.random.PRNGKey(42)
bounds = jnp.array(PROBLEM.bounds)
X = jax.random.uniform(key, (2000, 3), minval=bounds[:, 0], maxval=bounds[:, 1])

# 2. Evaluate your model
Y = evaluate(X)  # Y.shape == (2000,)

# 3. Compute HDMR sensitivity indices
result = jaxgsa.hdmr.analyze(
    PROBLEM, X, Y,
    maxorder=2,
    slice_chunk_size=64,  # optional: cap the vmap batch (timesteps x outputs) for memory control
)

# Sobol-compatible first-order and total-order indices
print("S1:", result.S1)   # Sa[:D] — structural first-order contribution
print("ST:", result.ST)   # SCSA total per parameter; equals the Sobol' ST
                          # only when the inputs are independent

# HDMR-specific: per-term decomposition
print("Sa:", result.Sa)   # structural (uncorrelated) contribution per term
print("Sb:", result.Sb)   # correlative contribution per term
print("Terms:", result.terms)  # ('x1', 'x2', 'x3', 'x1/x2', 'x1/x3', 'x2/x3')

# 4. Use the fitted surrogate as an emulator
Y_pred = result.predict(X)
# HDMR fits on your output scale, so Y_pred is on that scale too
```

### PCE (analytical Sobol indices from a surrogate)

Polynomial Chaos Expansion fits an orthogonal-polynomial surrogate. It then
reads the Sobol indices straight off the coefficients. No Saltelli design is
needed.

```python
import jax
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

key = jax.random.PRNGKey(42)
bounds = jnp.array(PROBLEM.bounds)
X = jax.random.uniform(key, (2000, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
Y = evaluate(X)

result = jaxgsa.pce.analyze(PROBLEM, X, Y, order=4)
print("S1:", result.S1)              # (D,) first-order
print("ST:", result.ST)              # (D,) total-order
print("LOO RMSE:", result.loo_rmse)  # leave-one-out cross-validation error

Y_pred = result.predict(X)
```

### Shapley effects (fair variance allocation)

Shapley effects split the output variance fairly among the inputs. Each
interaction's variance is shared equally by its participants. jaxgsa computes
them analytically from a fitted PCE (default) or RS-HDMR surrogate, with no
permutation Monte Carlo. Inputs are assumed independent in this version.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jaxgsa.sampling.monte_carlo(PROBLEM, n=2000, seed=42)
Y = evaluate(jnp.asarray(X))

result = jaxgsa.pce.analyze(PROBLEM, jnp.asarray(X), Y).shapley()
print("Sh:", result.Sh)              # (D,) Shapley effects
print("sum:", result.Sh.sum())       # == 1 (Shapley efficiency property)
print("explained:", result.explained_variance)  # fraction of Var(Y) captured
print("S1:", result.S1)              # (D,) first-order, same surrogate
print("ST:", result.ST)              # (D,) total-order — S1 <= Sh <= ST per parameter

# Both backends accept scalar (N,), multi-output (N, K), and
# time-series (N, T, K) Y; backend="hdmr" swaps in the B-spline surrogate
result_hdmr = jaxgsa.hdmr.analyze(PROBLEM, jnp.asarray(X), Y).shapley()
```

### HSIC (kernel-based dependence)

The Hilbert–Schmidt Independence Criterion detects any statistical dependence,
whether nonlinear, non-monotone, or heteroscedastic. It works on any (X, Y)
pairs, including correlated inputs.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jaxgsa.sampling.monte_carlo(PROBLEM, n=2000, seed=42)
Y = evaluate(jnp.asarray(X))

result = jaxgsa.hsic.analyze(PROBLEM, jnp.asarray(X), Y)
print("R2-HSIC:", result.R2_HSIC)    # (D,) normalized first-order dependence
print("Total HSIC:", result.T_HSIC)  # (D,) dependence through interactions
print("p-values:", result.p_values)  # permutation-test significance
```

### PAWN (distribution/CDF-based)

PAWN measures how much the entire output distribution (its CDF) shifts when
an input is fixed. It uses the Kolmogorov–Smirnov distance between the
unconditional and conditional distributions. Because it looks at the whole
distribution rather than just the variance, it is called moment-independent.
It catches effects on tails and extremes that Sobol indices can miss. No
structured sampling is needed.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jaxgsa.sampling.monte_carlo(PROBLEM, n=5000, seed=42)
Y = evaluate(jnp.asarray(X))

result = jaxgsa.pawn.analyze(PROBLEM, jnp.asarray(X), Y, statistic="median")
print("PAWN:", result.pawn)  # (D,) median KS distance across conditioning bins
```

### Morris (elementary-effects screening)

Morris ranks parameters from coarse finite-difference effects sampled across
the whole domain. It is a cheap screening pass before a full Sobol run. jaxgsa
removes exact duplicate design rows, so you evaluate fewer than `r * (D + 1)`
points.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

sr = jaxgsa.morris.sample(PROBLEM, n_trajectories=50, seed=42)
Y = evaluate(jnp.asarray(sr.samples))

result = jaxgsa.morris.analyze(sr, Y)
print("mu_star:", result.mu_star)  # (D,) mean |elementary effect| — importance
print("sigma:", result.sigma)      # (D,) spread — nonlinearity/interactions
```

### Borgonovo delta (density-based, moment-independent)

The Borgonovo delta measures the average shift of the entire output density
when an input is fixed. It is moment-independent like PAWN, but density-based.
The Plischke et al. (2013) "given-data" estimator works on any existing (X, Y)
pairs, with no special sampling design required. It also returns the
first-order Sobol index, estimated from the same data partition.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jaxgsa.sampling.monte_carlo(PROBLEM, n=5000, seed=42)
Y = evaluate(jnp.asarray(X))

result = jaxgsa.borgonovo.analyze(PROBLEM, jnp.asarray(X), Y)
print("delta:", result.delta)  # (D,) bias-corrected delta indices
print("S1:", result.S1)        # (D,) given-data first-order Sobol
```

### Optimal transport (Wasserstein-based, moment-independent)

The OT index measures how far knowing an input moves the entire output
distribution. It is the class-averaged squared 2-Wasserstein distance between
conditional and unconditional outputs, on a [0, 1] scale (Borgonovo et al.,
2024).

Every index splits exactly into an advective part and a diffusive part. The
advective part is the mean shift, equal to half the first-order Sobol index.
The diffusive part covers changes in spread and shape. An input with a large
advective part moves the output. One with a large diffusive part reshapes it.

Conditioning is rank-based, so mixed uniform/Gaussian marginals and correlated
inputs work unchanged.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jaxgsa.sampling.monte_carlo(PROBLEM, n=5000, seed=42)
Y = evaluate(jnp.asarray(X))

result = jaxgsa.optimal_transport.analyze(PROBLEM, jnp.asarray(X), Y)
print("ot:", result.ot)                # (D,) total index
print("advective:", result.advective)  # mean-shift part (= S1 / 2)
print("diffusive:", result.diffusive)  # spread/shape part

# Time-series outputs: one index per input over each output's whole
# trajectory (point-cloud transport via pure-JAX Sinkhorn)
# result = jaxgsa.optimal_transport.analyze(PROBLEM, X, Y_tk, mode="trajectory")
```

## Usage

### Define a problem

A `Problem` specifies the parameter names and their bounds:

```python
from jaxgsa import Problem

# From a dictionary
problem = Problem.from_dict({
    "x1": (-3.14, 3.14),
    "x2": (-3.14, 3.14),
    "x3": (-3.14, 3.14),
})

# Or directly
problem = Problem(
    names=("x1", "x2", "x3"),
    bounds=((-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14)),
)
```

### Generate samples

```python
sampling_result = jaxgsa.sobol.sample(
    problem,
    n_samples=4096,          # minimum desired unique model evaluations
    calc_second_order=True,  # include second-order indices (default)
    scramble=True,           # scramble Sobol sequence (default)
    seed=42,                 # reproducibility
    verbose=True,            # print a short sampling summary (default)
)

# sampling_result.samples is the unique NumPy array you pass to your model
# sampling_result.n_expanded is the internal Saltelli row count
```

### Save and reload samples

If you want to generate samples once and reuse them later, persist the
`SobolSamples` to disk and reconstruct it with its class method:

```python
sampling_result.save("runs/ishigami_samples")

restored = jaxgsa.sobol.SobolSamples.load("runs/ishigami_samples")
Y = my_model(restored.samples)
result = jaxgsa.sobol.analyze(restored, Y)
```

The call writes `runs/ishigami_samples.npz`, containing the unique sample
matrix, problem definition, and Saltelli reconstruction metadata.

### Analyze results

```python
# Y can be:
#   - (n_runs,)       scalar output (single output, no time dimension)
#   - (n_runs, K)     multi-output (K outputs, no time dimension)
#   - (n_runs, T, K)  time-series multi-output (T timesteps, K outputs)
#
# Axes are never inferred or transposed. Use (N, T, 1) for one
# time-varying output and make len(problem.output_names) match K.
Y = my_model(sampling_result.samples)

result = jaxgsa.sobol.analyze(
    sampling_result,
    Y,
    # ci_method="quantile",  # optional bootstrap CI summary method
    slice_chunk_size=64,  # optional: limit vmap batch size for memory control
)

# result.S1, result.ST — sensitivity indices
# result.S2            — second-order interactions (None if not computed)
```

jaxgsa always standardizes the outputs over the sample axis before the Sobol
estimators run. It subtracts the mean of each output slice and divides by its
standard deviation. The S1 and S2 estimators are uncentred products, so a
non-zero output mean biases them. The standardization removes that bias. SALib
does the same thing.

For confidence intervals, set `num_resamples > 0`. Then choose how the
bootstrap distribution is summarized. `ci_method="quantile"` gives percentile
lower/upper endpoints. `ci_method="gaussian"` gives symmetric endpoints from
the bootstrap standard deviation. Either way jaxgsa returns endpoint arrays,
not SALib-style confidence half-widths.

### Multi-output models

For models with multiple outputs, pass a 2D array `(n_runs, K)` evaluated on the unique rows. The returned indices will have shape `(K, D)`:

```python
import jax.numpy as jnp

def multi_output_model(X):
    y1 = jnp.sin(X[:, 0]) + X[:, 1] ** 2
    y2 = X[:, 0] * X[:, 2]
    return jnp.column_stack([y1, y2])

Y = multi_output_model(sampling_result.samples)  # (n_runs, 2)
result = jaxgsa.sobol.analyze(sampling_result, Y)
# result.S1.shape == (2, 3)  — 2 outputs, 3 parameters (K, D)
# result.ST.shape == (2, 3)  — (K, D)
# result.S2.shape == (2, 3, 3)  — (K, D, D)
```

For time-series multi-output models, pass a 3D array `(n_runs, T, K)` evaluated on the unique rows:

```python
def time_series_model(X):
    # Returns shape (n_runs, T, K) — e.g. 50 timesteps, 4 outputs
    ...

Y = time_series_model(sampling_result.samples)  # (n_runs, 50, 4)
result = jaxgsa.sobol.analyze(sampling_result, Y)
# result.S1.shape == (50, 4, D)  — (T, K, D)
# result.ST.shape == (50, 4, D)  — (T, K, D)
# result.S2.shape == (50, 4, D, D)  — (T, K, D, D)
```

### Edge cases: single output or single timestep

How a 2D array is interpreted depends on `problem.output_names`. Without it, a 2D array is `(N, K)`: multiple outputs, no time dimension. With exactly one entry in `output_names`, a 2D array is `(N, T)`, the timepoints of that single output. It then flows through as `(N, T, 1)`:

```python
# Single output, no time dimension — pass a 1D array
Y = my_model(X)          # shape (n_runs,)
result = jaxgsa.sobol.analyze(sampling_result, Y)
# result.S1.shape == (D,)

# Single output WITH time dimension — reshape to (N, T, 1) ...
Y = my_model(X)          # shape (n_runs, T) — e.g. 50 timesteps
Y = Y[:, :, None]        # reshape to (n_runs, 50, 1)
result = jaxgsa.sobol.analyze(sampling_result, Y)
# result.S1.shape == (50, 1, D)  — (T, K=1, D)

# ... or set output_names=["y"] on the problem and pass (N, T) directly:
# with exactly one output name, a 2D array is read as timepoints of that
# output and produces the same (50, 1, D) result.

# Multiple outputs, single timestep — just pass (N, K)
Y = my_model(X)          # shape (n_runs, 4) — 4 outputs
result = jaxgsa.sobol.analyze(sampling_result, Y)
# result.S1.shape == (4, D)  — (K, D)
# No need for a time dimension; (N, 1, 4) also works but is unnecessary.
```

jaxgsa also resolves layouts that are off but unambiguously recoverable. Two cases qualify: a transposed `(K, N)` array, and a 3D `(N, K, T)` array whose middle axis matches `len(output_names)`. jaxgsa fixes them and emits a `JaxgsaWarning` that names the transformation. Ambiguous layouts raise. jaxgsa never guesses.

`JaxgsaWarning` is the category of every warning that jaxgsa raises. It is a subclass of `UserWarning`, so code that already filters on `UserWarning` keeps working. Use the jaxgsa class when you want to select jaxgsa warnings alone:

```python
import warnings
from jaxgsa import JaxgsaWarning

# Silence them.
warnings.filterwarnings("ignore", category=JaxgsaWarning)

# Or turn them into errors. Pick one of the two lines, not both.
warnings.simplefilter("error", JaxgsaWarning)
```

---

## API Reference

The full site reference now lives at
[danielepessina.github.io/jaxgsa/api/](https://danielepessina.github.io/jaxgsa/api/).

Use it for:

- the complete exported surface from `jaxgsa`
- parameter, field, and shape contracts
- validation and error behavior
- `to_dataset()` labeling rules
- Sobol, RS-HDMR, PCE, Shapley, eFAST, DGSM, Morris, HSIC, PAWN, Borgonovo delta, optimal-transport, VKOGA, and Kucherenko workflow examples

Quick map:

- `Problem`, `UniformSpec`, `GaussianSpec`, and `CategoricalSpec`
- `jaxgsa.sobol`: `sample` / `analyze` / `SobolSamples` / `SobolResult`
- `jaxgsa.sampling`: `monte_carlo`
- `jaxgsa.hdmr`: `analyze` / `HDMRResult`
- `jaxgsa.pce`: `analyze` / `PCEResult`
- `jaxgsa.shapley`: `analyze` / `ShapleyResult`
- `jaxgsa.efast`: `sample` / `analyze` / `EFASTResult` / `EFASTSamples`
- `jaxgsa.dgsm`: `analyze` / `DGSMResult` / `poincare_constant` / `axis_constants`
- `jaxgsa.morris`: `sample` / `analyze` / `MorrisResult` / `MorrisSamples`
- `jaxgsa.hsic`: `analyze` / `HSICResult`
- `jaxgsa.pawn`: `analyze` / `PAWNResult`
- `jaxgsa.borgonovo`: `analyze` / `DeltaResult`
- `jaxgsa.optimal_transport`: `analyze` / `OTResult`
- `jaxgsa.vkoga`: `analyze` / `VKOGAResult`
- `jaxgsa.kucherenko`: `sample` / `analyze` / `KucherenkoSamples` / `KucherenkoResult`

Commands are intentionally not duplicated at the package root. Use the method
namespaces shown above. PCE, HDMR, and VKOGA predictions are result methods:
`result.predict(...)`. Shapley effects are result methods on PCE and HDMR:
`result.shapley(...)`. VKOGA offers `predict` but not `shapley`, because a
kernel expansion has no per-parameter-subset variance decomposition. For a
Shapley-style allocation under dependent inputs, use `jaxgsa.hdmr` with
`shapley(include_correlative=True)`. For conditional-variance indices under
dependence, use `jaxgsa.vkoga` or `jaxgsa.kucherenko`.

For runnable walkthroughs, start with the
[Getting Started guide](https://danielepessina.github.io/jaxgsa/guide/getting-started)
and the
[examples section](https://danielepessina.github.io/jaxgsa/examples/basic).

---

## Dependencies

Core runtime dependencies (installed automatically): `jax`, `jaxlib`, `scipy`,
and `xarray`. See [`pyproject.toml`](pyproject.toml) for exact version
bounds.

Optional extras: `notebook` (marimo, matplotlib) and `dev` (pytest, ruff, ty,
SALib, POT).

## License

Released under the MIT License.

See [LICENSE](LICENSE) for details.

## Benchmark Results

jaxgsa vs SALib on a coupled-oscillator model. Methodology:

- Model: D=5 parameters, N=1024 base samples.
- Hardware and versions: Apple M1 Pro CPU, JAX 0.10.2.
- Repeats: every timing is the best of 5 runs, except the slow SALib HDMR path (best of 2).
- JIT: jaxgsa figures are post-JIT steady-state. The one-off XLA compile takes roughly 0.3–1.1 s depending on scenario. It is paid once per process and excluded here. SALib is pure NumPy/SciPy and requires no compilation.
- Scope: the tables below cover the two methods timed against SALib here, Sobol and RS-HDMR.

The other methods are validated for correctness but not timed here: PCE, Shapley, eFAST, DGSM, Morris, HSIC, PAWN, Borgonovo delta, and optimal transport. Borgonovo delta also has a direct SALib counterpart, `SALib.analyze.delta`, and is validated against it in the test suite.

### Sobol — point estimates (no bootstrap)

| Scenario (T×K) | Method | jaxgsa (ms) | SALib (ms) | Speedup |
|---|---|---:|---:|---:|
| 1×1 | analyze (no S2) | 0.7 | 0.2 | 0.3× |
| 1×1 | analyze (S2) | 0.9 | 0.9 | 0.9× |
| 1×6 | analyze (no S2) | 0.9 | 1.4 | 1.5× |
| 1×6 | analyze (S2) | 1.5 | 5.5 | 3.6× |
| 50×1 | analyze (no S2) | 3.0 | 12.4 | 4.1× |
| 50×1 | analyze (S2) | 3.7 | 46.7 | 12.5× |
| 50×6 | analyze (no S2) | 12.1 | 73.4 | 6.1× |
| 50×6 | analyze (S2) | 17.4 | 274.8 | 15.8× |

### Sobol — 300 bootstrap resamples

| Scenario (T×K) | Method | jaxgsa (ms) | SALib (ms) | Speedup |
|---|---|---:|---:|---:|
| 1×1 | analyze (no S2) | 8.2 | 22.2 | 2.7× |
| 1×1 | analyze (S2) | 11.1 | 88.4 | 8.0× |
| 1×6 | analyze (no S2) | 36.0 | 143.5 | 4.0× |
| 1×6 | analyze (S2) | 51.6 | 471.4 | 9.1× |
| 50×1 | analyze (no S2) | 283.4 | 1208.1 | 4.3× |
| 50×1 | analyze (S2) | 414.7 | 3536.2 | 8.5× |
| 50×6 | analyze (no S2) | 1955.7 | 7544.9 | 3.9× |
| 50×6 | analyze (S2) | 2721.1 | 22933.8 | 8.4× |

### HDMR

| Scenario (T×K) | jaxgsa (ms) | SALib (ms) | Speedup |
|---|---:|---:|---:|
| 1×1 | 18.3 | 89.3 | 4.9× |
| 1×6 | 18.8 | 506.1 | 26.9× |
| 50×1 | 20.9 | 4000.7 | 191.6× |
| 50×6 | 39.0 | 26063.1 | 667.7× |

The speedup grows with output dimensionality. SALib loops over each (T, K) slice in Python, while jaxgsa vectorizes with `jax.vmap`. With bootstrap, JIT reuse across resamples adds further gains.

Correctness is validated against analytical Ishigami solutions and SALib on every run. Full benchmark script: [`benchmark_salib.py`](https://github.com/DanielePessina/jaxgsa/blob/master/benchmark_salib.py). See the [docs](https://danielepessina.github.io/jaxgsa/guide/benchmarks) for methodology details.

```bash
uv run --extra dev benchmark_salib.py
```
