# gsax

**Global Sensitivity Analysis in JAX**

[![PyPI](https://img.shields.io/pypi/v/gsax)](https://pypi.org/project/gsax/)
[![CI](https://github.com/DanielePessina/gsax/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielePessina/gsax/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://danielepessina.github.io/gsax/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)

`gsax` tells you which of your model's inputs actually drive its output. You give it input samples and the outputs your model produced for them; it returns sensitivity indices that rank the inputs and expose interactions. Everything is computed in JAX, so analyses are JIT-compiled and run on CPU, GPU, or TPU without code changes.

Eleven complementary methods are included: **Sobol indices** (the standard variance decomposition, via Saltelli sampling), **RS-HDMR** and **PCE** (surrogate-based — they fit a cheap approximation of your model to any existing input–output pairs and read the indices off the fit), **Shapley effects** (a fair, game-theoretic split of the output variance, computed analytically from a PCE or HDMR surrogate), **eFAST** (Fourier-based S1 and ST), **DGSM** (derivative-based bounds via JAX autodiff), **Morris** (cheap elementary-effects screening to discard unimportant inputs early), **HSIC** (kernel-based dependence detection), and three moment-independent measures that look at the whole output distribution rather than just its variance: **PAWN** (CDF-based, Pianosi & Wagener, 2015), the **Borgonovo delta** (density-based, Borgonovo, 2007), and **optimal-transport indices** (Wasserstein-based with an advective/diffusive decomposition, Borgonovo et al., 2024).

## Features

- **Sobol indices** via Saltelli sampling with Sobol quasi-random sequences (`scipy.stats.qmc`)
  - First-order (S1: an input's direct share of output variance), total-order (ST: including all its interactions), and second-order (S2: pairwise interactions)
  - Fused JIT kernels and chunked `jit(vmap(...))` execution for bounded memory on large output grids
  - [**Up to 668× faster than SALib**](#benchmark-results) (HDMR on multi-output workloads)
- **RS-HDMR** (Random Sampling High-Dimensional Model Representation)
  - Works with **any** set of (X, Y) pairs — no structured sampling required
  - B-spline surrogate with ANCOVA decomposition (Sa, Sb, S, ST)
  - Built-in emulator for prediction at new inputs
  - S1/ST properties for direct comparison with Sobol results
- **PCE** (Polynomial Chaos Expansion)
  - Analytical Sobol indices from orthogonal polynomial coefficients (Sudret, 2008)
  - Wiener-Askey scheme: Legendre for uniform, Hermite for Gaussian inputs
  - Built-in emulator and leave-one-out cross-validation RMSE
  - Scalar, multi-output, and time-series outputs — all output slices share one basis, fitted in a single multi-right-hand-side solve
- **Shapley effects** (Owen, 2014; Song, Nelson & Staum, 2016)
  - Fair, game-theoretic allocation of output variance — each interaction's variance split equally among its participants
  - Computed **analytically** from a fitted PCE (default) or RS-HDMR surrogate: no permutation Monte Carlo, no extra model runs
  - Works with **any** set of (X, Y) pairs; returns Sh alongside S1 and ST from the same surrogate (S1 <= Sh <= ST)
  - Assumes independent inputs (v1); Sh sums to 1, and `explained_variance` reports the fraction of Var(Y) the surrogate captured
- **eFAST** (Extended Fourier Amplitude Sensitivity Test)
  - Frequency-based S1 and ST via sinusoidal search curves and Fourier decomposition
  - Supports scalar, multi-output, and time-series outputs
  - Simple N x D sampling design, no cross-matrix structure needed
- **DGSM** (Derivative-based Global Sensitivity Measures)
  - Upper and lower bounds on total Sobol index via JAX reverse-mode autodiff
  - Poincare constants for uniform, Gaussian, and truncated Gaussian inputs
  - Pre-computed Jacobian path for non-JAX models
- **Morris** (elementary-effects screening)
  - Globalized one-at-a-time screening: mu_star importance ranking and sigma interaction flag from only `r * (D + 1)` model runs (r trajectories, D inputs)
  - Trajectory (Morris, 1991) and radial (Campolongo et al., 2011) designs with unique-row deduplication
  - Bootstrap confidence intervals over trajectories and prefix-nested trajectory downsampling
- **HSIC** (Hilbert–Schmidt Independence Criterion)
  - Kernel-based dependence: normalized first-order (R2-HSIC) and Total HSIC indices
  - Works with **any** set of (X, Y) pairs — Gaussian RBF kernels with the median heuristic
  - Detects nonlinear, non-monotone, and heteroscedastic dependence; permutation-test p-values
- **PAWN** — moment-independent, CDF-based sensitivity (Pianosi & Wagener, 2015)
  - Kolmogorov–Smirnov distance between unconditional and conditional output CDFs
  - Tie-aware KS matching `scipy.stats.ks_2samp` for discrete/continuous outputs
  - Median / max / mean aggregation with bootstrap confidence intervals
- **Borgonovo delta** — moment-independent, density-based sensitivity (Borgonovo, 2007)
  - Plischke et al. (2013) given-data estimator: works with **any** set of (X, Y) pairs
  - Bias-corrected delta plus the given-data first-order Sobol S1 (SALib-compatible)
  - Percentile bootstrap confidence intervals
- **Optimal transport** — Wasserstein-based distributional sensitivity (Borgonovo et al., 2024)
  - Given-data estimator on **any** (X, Y) pairs; rank-based conditioning handles mixed uniform/Gaussian marginals and correlated inputs
  - Advective (mean-shift, = S1/2) vs diffusive (spread/shape) decomposition of every index
  - Per-column indices via exact 1-D transport (solver-free) plus joint point-cloud modes over multivariate/time-series outputs (pure-JAX log-domain Sinkhorn); dummy-input irrelevance baseline
- All eleven methods accept scalar `(N,)`, multi-output `(N, K)`, and time-series `(N, T, K)` outputs, with smart layout inference: pass `output_names` on the problem and gsax disambiguates 2-D/3-D layouts, fixing obvious transposes with a warning and raising on ambiguity
- Bootstrap confidence intervals with JAX-accelerated resampling
- Optional `prenormalize=True` mode for SALib-style output standardization before
  Sobol or HDMR analysis
- Automatic data cleaning: non-finite values (NaN/Inf) are detected and dropped by group
- **xarray integration** — `to_dataset()` on results for labeled, named dimensions (`param`, `output`, `time`)
- Save and reload Sobol sample sets with metadata via `SamplingResult.save()` and `gsax.load()`
- Built-in Ishigami benchmark function with known analytical solutions

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

## Configuration

gsax inherits JAX's runtime defaults. Two optional knobs, documented in full in
the [Configuration guide](https://danielepessina.github.io/gsax/guide/configuration):

- **Double precision** — JAX defaults to float32 and silently downcasts `float64`.
  For precision-sensitive Sobol/HSIC work, enable 64-bit floats *before the first
  array is created*: `jax.config.update("jax_enable_x64", True)`.
- **Persistent compilation cache** — reuse compiled kernels across process
  restarts (sweeps, CI, HPC) by calling
  `gsax.enable_compilation_cache("~/.cache/gsax-jax")` once, before your first analysis.

## Quick Start

```python
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

# 1. Generate unique Sobol/Saltelli samples
sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)
# sampling_result.samples.shape == (n_total, D)  — D parameters, n_total unique rows
# sampling_result.expanded_n_total is the internal Saltelli row count used by analyze()
# by default, sample() also prints a short summary of unique vs expanded rows

# 2. Evaluate your model on the samples
Y = evaluate(sampling_result.samples)  # Y.shape == (n_total,)

# 3. Compute Sobol indices
result = gsax.analyze(
    sampling_result,
    Y,
    prenormalize=False,  # default; set True for SALib-style output standardization
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
effect; ST also counts every interaction it takes part in. Here `x2` has the
largest direct effect, while `x3` has no direct effect at all (S1 ≈ 0) but
still matters through its interaction with `x1` (ST ≈ 0.24) — `result.S2`
shows which pairs are responsible.

### RS-HDMR (surrogate-based)

RS-HDMR fits a surrogate — a cheap spline approximation of your model — to any
existing (X, Y) pairs and computes the indices from the fit. Use it when your
model runs already exist and rerunning on a Saltelli design isn't an option.

```python
import jax
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

# 1. Generate any set of input samples (no structured sampling needed)
key = jax.random.PRNGKey(42)
bounds = jnp.array(PROBLEM.bounds)
X = jax.random.uniform(key, (2000, 3), minval=bounds[:, 0], maxval=bounds[:, 1])

# 2. Evaluate your model
Y = evaluate(X)  # Y.shape == (2000,)

# 3. Compute HDMR sensitivity indices
result = gsax.analyze_hdmr(
    PROBLEM, X, Y,
    maxorder=2,
    prenormalize=False,  # default; set True for SALib-style output standardization
    chunk_size=64,  # optional: cap the vmap batch (timesteps x outputs) for memory control
)

# Sobol-compatible first-order and total-order indices
print("S1:", result.S1)   # Sa[:D] — structural first-order contribution
print("ST:", result.ST)   # total-order per parameter

# HDMR-specific: per-term decomposition
print("Sa:", result.Sa)   # structural (uncorrelated) contribution per term
print("Sb:", result.Sb)   # correlative contribution per term
print("Terms:", result.terms)  # ('x1', 'x2', 'x3', 'x1/x2', 'x1/x3', 'x2/x3')

# 4. Use the fitted surrogate as an emulator
Y_pred = gsax.emulate_hdmr(result, X)
# Y_pred stays on the original output scale even when prenormalize=True
```

### PCE (analytical Sobol indices from a surrogate)

Polynomial Chaos Expansion fits an orthogonal-polynomial surrogate and reads
the Sobol indices straight off the coefficients — no Saltelli design needed.

```python
import jax
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

key = jax.random.PRNGKey(42)
bounds = jnp.array(PROBLEM.bounds)
X = jax.random.uniform(key, (2000, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
Y = evaluate(X)

result = gsax.analyze_pce(PROBLEM, X, Y, order=4)
print("S1:", result.S1)              # (D,) first-order
print("ST:", result.ST)              # (D,) total-order
print("LOO RMSE:", result.loo_rmse)  # leave-one-out cross-validation error

Y_pred = gsax.emulate_pce(result, X)  # use the fit as an emulator
```

### Shapley effects (fair variance allocation)

Shapley effects split the output variance fairly among the inputs — each
interaction's variance is shared equally by its participants — computed
analytically from a fitted PCE (default) or RS-HDMR surrogate, with no
permutation Monte Carlo. Inputs are assumed independent in this version.

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

X = gsax.sample_mc(PROBLEM, N=2000, seed=42)
Y = evaluate(jnp.asarray(X))

result = gsax.analyze_shapley(PROBLEM, jnp.asarray(X), Y)  # backend="pce" default
print("Sh:", result.Sh)              # (D,) Shapley effects
print("sum:", result.Sh.sum())       # == 1 (Shapley efficiency property)
print("explained:", result.explained_variance)  # fraction of Var(Y) captured
print("S1:", result.S1)              # (D,) first-order, same surrogate
print("ST:", result.ST)              # (D,) total-order — S1 <= Sh <= ST per parameter

# Both backends accept scalar (N,), multi-output (N, K), and
# time-series (N, T, K) Y; backend="hdmr" swaps in the B-spline surrogate
result_hdmr = gsax.analyze_shapley(PROBLEM, jnp.asarray(X), Y, backend="hdmr")
```

### HSIC (kernel-based dependence)

The Hilbert–Schmidt Independence Criterion detects **any** statistical
dependence — nonlinear, non-monotone, or heteroscedastic — from any (X, Y)
pairs, including correlated inputs.

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

X = gsax.sample_mc(PROBLEM, N=2000, seed=42)
Y = evaluate(jnp.asarray(X))

result = gsax.analyze_hsic(PROBLEM, jnp.asarray(X), Y)
print("R2-HSIC:", result.R2_HSIC)    # (D,) normalized first-order dependence
print("Total HSIC:", result.T_HSIC)  # (D,) dependence through interactions
print("p-values:", result.p_values)  # permutation-test significance
```

### PAWN (distribution/CDF-based)

PAWN measures how much the **entire output distribution** (its CDF) shifts when
an input is fixed, using the Kolmogorov–Smirnov distance between the
unconditional and conditional distributions. Because it looks at the whole
distribution rather than just the variance ("moment-independent"), it catches
effects on tails and extremes that Sobol indices can miss. No structured
sampling is needed.

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

X = gsax.sample_mc(PROBLEM, N=5000, seed=42)
Y = evaluate(jnp.asarray(X))

result = gsax.analyze_pawn(PROBLEM, jnp.asarray(X), Y, statistic="median")
print("PAWN:", result.pawn)  # (D,) median KS distance across conditioning bins
```

### Morris (elementary-effects screening)

Morris ranks parameters from coarse finite-difference effects sampled across
the whole domain — a cheap screening pass before a full Sobol run. Exact
duplicate design rows are removed, so you evaluate fewer than `r * (D + 1)`
points.

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

sr = gsax.sample_morris(PROBLEM, n_trajectories=50, seed=42)
Y = evaluate(jnp.asarray(sr.samples))

result = gsax.analyze_morris(sr, Y)
print("mu_star:", result.mu_star)  # (D,) mean |elementary effect| — importance
print("sigma:", result.sigma)      # (D,) spread — nonlinearity/interactions
```

### Borgonovo delta (density-based, moment-independent)

The Borgonovo delta measures the average shift of the **entire output
density** when an input is fixed — moment-independent like PAWN, but
density-based. The Plischke et al. (2013) "given-data" estimator works on any
existing (X, Y) pairs, no special sampling design required, and also returns
the first-order Sobol index estimated from the same data partition.

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

X = gsax.sample_mc(PROBLEM, N=5000, seed=42)
Y = evaluate(jnp.asarray(X))

result = gsax.analyze_borgonovo(PROBLEM, jnp.asarray(X), Y)
print("delta:", result.delta)  # (D,) bias-corrected delta indices
print("S1:", result.S1)        # (D,) given-data first-order Sobol
```

### Optimal transport (Wasserstein-based, moment-independent)

The OT index measures how far knowing an input moves the entire output
distribution: the class-averaged squared 2-Wasserstein distance between
conditional and unconditional outputs, on a [0, 1] scale (Borgonovo et al.,
2024). Every index splits exactly into an **advective** part (mean shift,
equal to half the first-order Sobol index) and a **diffusive** part (changes
in spread and shape) — an input with a large advective part moves the
output, one with a large diffusive part reshapes it. Conditioning is
rank-based: mixed uniform/Gaussian marginals and correlated inputs work
unchanged.

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

X = gsax.sample_mc(PROBLEM, N=5000, seed=42)
Y = evaluate(jnp.asarray(X))

result = gsax.analyze_optimal_transport(PROBLEM, jnp.asarray(X), Y)
print("ot:", result.ot)                # (D,) total index
print("advective:", result.advective)  # mean-shift part (= S1 / 2)
print("diffusive:", result.diffusive)  # spread/shape part

# Time-series outputs: one index per input over each output's whole
# trajectory (point-cloud transport via pure-JAX Sinkhorn)
# result = gsax.analyze_optimal_transport(PROBLEM, X, Y_tk, mode="trajectory")
```

## Usage

### Define a problem

A `Problem` specifies the parameter names and their bounds:

```python
from gsax import Problem

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
sampling_result = gsax.sample(
    problem,
    n_samples=4096,          # minimum desired unique model evaluations
    calc_second_order=True,  # include second-order indices (default)
    scramble=True,           # scramble Sobol sequence (default)
    seed=42,                 # reproducibility
    verbose=True,            # print a short sampling summary (default)
)

# sampling_result.samples is the unique NumPy array you pass to your model
# sampling_result.samples_df is a pandas DataFrame with SampleID + parameter columns
# sampling_result.expanded_n_total is the internal Saltelli row count
```

### Save and reload samples

If you want to generate samples once and reuse them later, persist the
`SamplingResult` to disk and reconstruct it with `gsax.load()`:

```python
sampling_result.save("runs/ishigami_samples", format="csv")

restored = gsax.load("runs/ishigami_samples", format="csv")
Y = my_model(restored.samples)
result = gsax.analyze(restored, Y)
```

`path` is a file stem, not a full filename. The call above writes:

- `runs/ishigami_samples.csv` with the unique sample matrix
- `runs/ishigami_samples.json` with problem and Saltelli metadata
- `runs/ishigami_samples.npz` only when the internal expanded-to-unique mapping is non-trivial

Supported formats are `csv`, `txt`, `xlsx`, `parquet`, and `pkl`. Use the
same `format` value when calling `gsax.load()`. `xlsx` requires `openpyxl`,
and `parquet` requires `pyarrow`.

### Analyze results

```python
# Y can be:
#   - (n_total,)       scalar output (single output, no time dimension)
#   - (n_total, K)     multi-output (K outputs, no time dimension)
#   - (n_total, T, K)  time-series multi-output (T timesteps, K outputs)
#
# How a 2D array is read depends on problem.output_names:
#   - without output_names, (N, M) is read as (N, K) — M outputs;
#   - with exactly ONE entry in output_names, (N, M) is read as (N, T) —
#     T timepoints of that single output — and flows through as (N, T, 1).
# Obvious layout mistakes (e.g. a transposed (K, N) array) are fixed with
# a UserWarning naming the transformation; ambiguous layouts raise.
# If you have a single output with no time, just pass a 1D array (N,).
Y = my_model(sampling_result.samples)

result = gsax.analyze(
    sampling_result,
    Y,
    prenormalize=False,  # optional SALib-style output standardization
    # ci_method="quantile",  # optional bootstrap CI summary method
    chunk_size=64,  # optional: limit vmap batch size for memory control
)

# result.S1, result.ST — sensitivity indices
# result.S2            — second-order interactions (None if not computed)
```

`prenormalize=True` standardizes the outputs over the sample axis before
analysis, SALib-style; the default `False` analyzes the raw outputs. For
confidence intervals, set `num_resamples > 0` and choose how the bootstrap
distribution is summarized: `ci_method="quantile"` gives percentile
lower/upper endpoints, `ci_method="gaussian"` gives symmetric endpoints from
the bootstrap standard deviation. Either way gsax returns endpoint arrays,
not SALib-style confidence half-widths — even when `prenormalize=True`.

### Multi-output models

For models with multiple outputs, pass a 2D array `(n_total, K)` evaluated on the unique rows. The returned indices will have shape `(K, D)`:

```python
import jax.numpy as jnp

def multi_output_model(X):
    y1 = jnp.sin(X[:, 0]) + X[:, 1] ** 2
    y2 = X[:, 0] * X[:, 2]
    return jnp.column_stack([y1, y2])

Y = multi_output_model(sampling_result.samples)  # (n_total, 2)
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (2, 3)  — 2 outputs, 3 parameters (K, D)
# result.ST.shape == (2, 3)  — (K, D)
# result.S2.shape == (2, 3, 3)  — (K, D, D)
```

For time-series multi-output models, pass a 3D array `(n_total, T, K)` evaluated on the unique rows:

```python
def time_series_model(X):
    # Returns shape (n_total, T, K) — e.g. 50 timesteps, 4 outputs
    ...

Y = time_series_model(sampling_result.samples)  # (n_total, 50, 4)
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (50, 4, D)  — (T, K, D)
# result.ST.shape == (50, 4, D)  — (T, K, D)
# result.S2.shape == (50, 4, D, D)  — (T, K, D, D)
```

### Edge cases: single output or single timestep

How a 2D array is interpreted depends on `problem.output_names`. Without it, a 2D array is `(N, K)` — multiple outputs, no time dimension. With exactly **one** entry in `output_names`, a 2D array is `(N, T)` — timepoints of that single output — and flows through as `(N, T, 1)`:

```python
# Single output, no time dimension — pass a 1D array
Y = my_model(X)          # shape (n_total,)
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (D,)

# Single output WITH time dimension — reshape to (N, T, 1) ...
Y = my_model(X)          # shape (n_total, T) — e.g. 50 timesteps
Y = Y[:, :, None]        # reshape to (n_total, 50, 1)
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (50, 1, D)  — (T, K=1, D)

# ... or set output_names=["y"] on the problem and pass (N, T) directly:
# with exactly one output name, a 2D array is read as timepoints of that
# output and produces the same (50, 1, D) result.

# Multiple outputs, single timestep — just pass (N, K)
Y = my_model(X)          # shape (n_total, 4) — 4 outputs
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (4, D)  — (K, D)
# No need for a time dimension; (N, 1, 4) also works but is unnecessary.
```

gsax also resolves layouts that are off but unambiguously recoverable — a transposed `(K, N)` array, or a 3D `(N, K, T)` array whose middle axis matches `len(output_names)` — fixing them with a `UserWarning` that names the transformation. Ambiguous layouts raise; gsax never guesses.

---

## API Reference

The full site reference now lives at
[danielepessina.github.io/gsax/api/](https://danielepessina.github.io/gsax/api/).

Use it for:

- the complete exported surface from `gsax`
- parameter, field, and shape contracts
- validation and error behavior
- `to_dataset()` labeling rules
- Sobol, RS-HDMR, PCE, Shapley, eFAST, DGSM, Morris, HSIC, PAWN, Borgonovo delta, and optimal-transport workflow examples

Quick map:

- `Problem`
- `sample` / `SamplingResult` / `load`
- `gsax.sobol`: `analyze` / `SAResult`
- `gsax.hdmr`: `analyze` / `emulate` / `HDMRResult` / `HDMREmulator`
- `gsax.pce`: `analyze` / `emulate` / `PCEResult`
- `gsax.shapley`: `analyze` / `ShapleyResult`
- `gsax.efast`: `sample` / `analyze` / `EFASTResult`
- `gsax.dgsm`: `analyze` / `DGSMResult` / `poincare_constant` / `axis_constants`
- `gsax.morris`: `sample` / `analyze` / `MorrisResult` / `MorrisSamplingResult`
- `gsax.hsic`: `analyze` / `HSICResult`
- `gsax.pawn`: `analyze` / `PAWNResult`
- `gsax.borgonovo`: `analyze` / `DeltaResult`
- `gsax.optimal_transport`: `analyze` / `OTResult`

All symbols are also re-exported from the top-level `gsax` namespace
(`gsax.analyze()`, `gsax.analyze_hdmr()`, `gsax.analyze_pce()`,
`gsax.analyze_shapley()`, `gsax.sample_efast()`, `gsax.analyze_efast()`,
`gsax.analyze_dgsm()`, `gsax.sample_morris()`, `gsax.analyze_morris()`,
`gsax.analyze_hsic()`, `gsax.analyze_pawn()`, `gsax.analyze_borgonovo()`,
`gsax.analyze_optimal_transport()`, etc.).

For runnable walkthroughs, start with the
[Getting Started guide](https://danielepessina.github.io/gsax/guide/getting-started)
and the
[examples section](https://danielepessina.github.io/gsax/examples/basic).

---

## Dependencies

Core runtime dependencies (installed automatically): `jax`, `jaxlib`, `pandas`,
`scipy`, and `xarray`. See [`pyproject.toml`](pyproject.toml) for exact version
bounds.

Optional extras: `xlsx` (openpyxl), `parquet` (pyarrow), `notebook` (marimo,
matplotlib), and `dev` (pytest, ruff, ty, SALib). Install with, e.g.,
`pip install "gsax[all]"`.

## License

Released under the MIT License.

See [LICENSE](LICENSE) for details.

## Benchmark Results

gsax vs SALib on a coupled-oscillator model (D=5 parameters, N=1024 base samples), Apple M1 Pro CPU, JAX 0.10.2. Every timing is the best of 5 runs, except the slow SALib HDMR path (best of 2). gsax figures are post-JIT steady-state: the one-off XLA compile — roughly 0.3–1.1 s depending on scenario — is paid once per process and excluded here, whereas SALib (pure NumPy/SciPy) requires no compilation. The timing tables below cover the two methods timed against SALib here (Sobol and RS-HDMR); the other methods (PCE, Shapley, eFAST, DGSM, Morris, HSIC, PAWN, Borgonovo delta, and optimal transport) are validated for correctness but not timed here. Borgonovo delta also has a direct SALib counterpart, `SALib.analyze.delta`, and is validated against it in the test suite.

### Sobol — point estimates (no bootstrap)

| Scenario (T×K) | Method | gsax (ms) | SALib (ms) | Speedup |
|---|---|---:|---:|---:|
| 1×1 | analyze (no S2) | 0.7 | 0.2 | 0.3× |
| 1×1 | analyze (S2) | 0.9 | 0.9 | **0.9×** |
| 1×6 | analyze (no S2) | 0.9 | 1.4 | **1.5×** |
| 1×6 | analyze (S2) | 1.5 | 5.5 | **3.6×** |
| 50×1 | analyze (no S2) | 3.0 | 12.4 | **4.1×** |
| 50×1 | analyze (S2) | 3.7 | 46.7 | **12.5×** |
| 50×6 | analyze (no S2) | 12.1 | 73.4 | **6.1×** |
| 50×6 | analyze (S2) | 17.4 | 274.8 | **15.8×** |

### Sobol — 300 bootstrap resamples

| Scenario (T×K) | Method | gsax (ms) | SALib (ms) | Speedup |
|---|---|---:|---:|---:|
| 1×1 | analyze (no S2) | 8.2 | 22.2 | **2.7×** |
| 1×1 | analyze (S2) | 11.1 | 88.4 | **8.0×** |
| 1×6 | analyze (no S2) | 36.0 | 143.5 | **4.0×** |
| 1×6 | analyze (S2) | 51.6 | 471.4 | **9.1×** |
| 50×1 | analyze (no S2) | 283.4 | 1208.1 | **4.3×** |
| 50×1 | analyze (S2) | 414.7 | 3536.2 | **8.5×** |
| 50×6 | analyze (no S2) | 1955.7 | 7544.9 | **3.9×** |
| 50×6 | analyze (S2) | 2721.1 | 22933.8 | **8.4×** |

### HDMR

| Scenario (T×K) | gsax (ms) | SALib (ms) | Speedup |
|---|---:|---:|---:|
| 1×1 | 18.3 | 89.3 | **4.9×** |
| 1×6 | 18.8 | 506.1 | **26.9×** |
| 50×1 | 20.9 | 4000.7 | **191.6×** |
| 50×6 | 39.0 | 26063.1 | **667.7×** |

The speedup grows with output dimensionality because SALib loops over each (T, K) slice in Python while gsax vectorizes with `jax.vmap`. With bootstrap, JIT reuse across resamples adds further gains.

Correctness is validated against analytical Ishigami solutions and SALib on every run. Full benchmark script: [`benchmark_salib.py`](https://github.com/DanielePessina/gsax/blob/master/benchmark_salib.py). See the [docs](https://danielepessina.github.io/gsax/guide/benchmarks) for methodology details.

```bash
uv run --extra dev benchmark_salib.py
```
