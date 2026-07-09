# gsax

**Global Sensitivity Analysis in JAX**

[![CI](https://github.com/DanielePessina/gsax/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielePessina/gsax/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://danielepessina.github.io/gsax/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)

`gsax` computes global sensitivity indices entirely in JAX, giving you GPU/TPU acceleration and JIT compilation for free. It provides seven complementary methods: **Sobol indices** (via Saltelli sampling), **RS-HDMR** (surrogate-based, works with any input-output pairs), **PCE** (Polynomial Chaos Expansion with analytical Sobol indices), **eFAST** (Extended Fourier Amplitude Sensitivity Test), **DGSM** (Derivative-based Global Sensitivity Measures via JAX autodiff), **HSIC** (the Hilbert–Schmidt Independence Criterion, a kernel-based dependence measure), and **PAWN** (a moment-independent method based on output CDFs, after Pianosi & Wagener, 2015).

## Features

- **Sobol indices** via Saltelli sampling with Sobol quasi-random sequences (`scipy.stats.qmc`)
  - First-order (S1), total-order (ST), and second-order (S2) indices
  - Fused JIT kernels and chunked `jit(vmap(...))` execution for bounded memory on large output grids
  - [**Up to 718× faster than SALib**](#benchmark-results) (HDMR on multi-output workloads)
- **RS-HDMR** (Random Sampling High-Dimensional Model Representation)
  - Works with **any** set of (X, Y) pairs — no structured sampling required
  - B-spline surrogate with ANCOVA decomposition (Sa, Sb, S, ST)
  - Built-in emulator for prediction at new inputs
  - S1/ST properties for direct comparison with Sobol results
- **PCE** (Polynomial Chaos Expansion)
  - Analytical Sobol indices from orthogonal polynomial coefficients (Sudret, 2008)
  - Wiener-Askey scheme: Legendre for uniform, Hermite for Gaussian inputs
  - Built-in emulator and leave-one-out cross-validation RMSE
  - Scalar outputs only
- **eFAST** (Extended Fourier Amplitude Sensitivity Test)
  - Frequency-based S1 and ST via sinusoidal search curves and Fourier decomposition
  - Supports scalar, multi-output, and time-series outputs
  - Simple N x D sampling design, no cross-matrix structure needed
- **DGSM** (Derivative-based Global Sensitivity Measures)
  - Upper and lower bounds on total Sobol index via JAX reverse-mode autodiff
  - Poincare constants for uniform, Gaussian, and truncated Gaussian inputs
  - Pre-computed Jacobian path for non-JAX models
- **HSIC** (Hilbert–Schmidt Independence Criterion)
  - Kernel-based dependence: normalized first-order (R2-HSIC) and Total HSIC indices
  - Works with **any** set of (X, Y) pairs — Gaussian RBF kernels with the median heuristic
  - Detects nonlinear, non-monotone, and heteroscedastic dependence; permutation-test p-values
- **PAWN** — moment-independent, CDF-based sensitivity (Pianosi & Wagener, 2015)
  - Kolmogorov–Smirnov distance between unconditional and conditional output CDFs
  - Tie-aware KS matching `scipy.stats.ks_2samp` for discrete/continuous outputs
  - Median / max / mean aggregation with bootstrap confidence intervals
- Supports scalar, multi-output, and time-series model outputs from the start
- Bootstrap confidence intervals with JAX-accelerated resampling
- Optional `prenormalize=True` mode for SALib-style output standardization before
  Sobol or HDMR analysis
- Automatic data cleaning: non-finite values (NaN/Inf) are detected and dropped by group
- **xarray integration** — `to_dataset()` on results for labeled, named dimensions (`param`, `output`, `time`)
- Save and reload Sobol sample sets with metadata via `SamplingResult.save()` and `gsax.load()`
- Built-in Ishigami benchmark function with known analytical solutions

## Installation

```bash
pip install git+https://github.com/DanielePessina/gsax.git
```

Or for development:

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
# sampling_result.samples.shape == (n_total, D)  where n_total is the unique row count
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

### RS-HDMR (surrogate-based)

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
    chunk_size=64,  # optional: limit T*K vmap batch size for memory control
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

PAWN measures how much the **entire output CDF** shifts when an input is fixed,
using the Kolmogorov–Smirnov distance between the unconditional and conditional
distributions. It is moment-independent and needs no structured sampling.

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

X = gsax.sample_mc(PROBLEM, N=5000, seed=42)
Y = evaluate(jnp.asarray(X))

result = gsax.analyze_pawn(PROBLEM, jnp.asarray(X), Y, statistic="median")
print("PAWN:", result.pawn)  # (D,) median KS distance across conditioning bins
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
# Important: a 2D array is always interpreted as (N, K), never (N, T).
# If you have time-series with a single output, reshape to (N, T, 1).
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

Set `prenormalize=True` when you want global output standardization over the
sample axis before analysis. The default `False` preserves the current gsax
behavior. When bootstrapping with `num_resamples > 0`, use
`ci_method="quantile"` for percentile bootstrap lower/upper endpoints or
`ci_method="gaussian"` for symmetric gaussian lower/upper endpoints computed
from the bootstrap standard deviation. Both options still return endpoint
arrays, not SALib-style confidence half-widths, even when `prenormalize=True`.

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

A 2D array is **always** interpreted as `(N, K)` — multiple outputs, no time dimension. This matters when your model has only one output or only one timestep:

```python
# Single output, no time dimension — pass a 1D array
Y = my_model(X)          # shape (n_total,)
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (D,)

# Single output WITH time dimension — reshape to (N, T, 1)
Y = my_model(X)          # shape (n_total, T) — e.g. 50 timesteps
Y = Y[:, :, None]        # reshape to (n_total, 50, 1)
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (50, 1, D)  — (T, K=1, D)

# Multiple outputs, single timestep — just pass (N, K)
Y = my_model(X)          # shape (n_total, 4) — 4 outputs
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (4, D)  — (K, D)
# No need for a time dimension; (N, 1, 4) also works but is unnecessary.
```

---

## API Reference

The full site reference now lives at
[danielepessina.github.io/gsax/api/](https://danielepessina.github.io/gsax/api/).

Use it for:

- the complete exported surface from `gsax`
- parameter, field, and shape contracts
- validation and error behavior
- `to_dataset()` labeling rules
- Sobol, RS-HDMR, PCE, eFAST, DGSM, HSIC, and PAWN workflow examples

Quick map:

- `Problem`
- `sample` / `SamplingResult` / `load`
- `gsax.sobol`: `analyze` / `SAResult`
- `gsax.hdmr`: `analyze` / `emulate` / `HDMRResult` / `HDMREmulator`
- `gsax.pce`: `analyze` / `emulate` / `PCEResult`
- `gsax.efast`: `sample` / `analyze` / `EFASTResult`
- `gsax.dgsm`: `analyze` / `DGSMResult` / `poincare_constant` / `axis_constants`
- `gsax.hsic`: `analyze` / `HSICResult`
- `gsax.pawn`: `analyze` / `PAWNResult`

All symbols are also re-exported from the top-level `gsax` namespace
(`gsax.analyze()`, `gsax.analyze_hdmr()`, `gsax.analyze_pce()`,
`gsax.sample_efast()`, `gsax.analyze_efast()`, `gsax.analyze_dgsm()`,
`gsax.analyze_hsic()`, `gsax.analyze_pawn()`, etc.).

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

gsax vs SALib on a coupled-oscillator model (D=5 parameters, N=1024 base samples). Post-JIT steady-state timings, best of 5, Apple M3 Pro CPU. Benchmarks cover the two methods with a direct SALib counterpart (Sobol and RS-HDMR); the other methods (PCE, eFAST, DGSM, HSIC, PAWN) are validated for correctness but not timed here.

### Sobol — point estimates (no bootstrap)

| Scenario (T×K) | Method | gsax (ms) | SALib (ms) | Speedup |
|---|---|---:|---:|---:|
| 1×1 | analyze (no S2) | 0.8 | 0.2 | 0.3× |
| 1×1 | analyze (S2) | 0.9 | 0.9 | **1.0×** |
| 1×6 | analyze (no S2) | 1.0 | 1.4 | **1.4×** |
| 1×6 | analyze (S2) | 1.5 | 5.3 | **3.5×** |
| 50×1 | analyze (no S2) | 2.8 | 12.3 | **4.4×** |
| 50×1 | analyze (S2) | 4.9 | 45.8 | **9.4×** |
| 50×6 | analyze (no S2) | 8.8 | 74.3 | **8.5×** |
| 50×6 | analyze (S2) | 14.8 | 276.6 | **18.7×** |

### Sobol — 300 bootstrap resamples

| Scenario (T×K) | Method | gsax (ms) | SALib (ms) | Speedup |
|---|---|---:|---:|---:|
| 1×1 | analyze (no S2) | 5.2 | 28.5 | **5.5×** |
| 1×1 | analyze (S2) | 8.5 | 80.0 | **9.4×** |
| 1×6 | analyze (no S2) | 17.0 | 162.8 | **9.6×** |
| 1×6 | analyze (S2) | 36.4 | 490.5 | **13.5×** |
| 50×1 | analyze (no S2) | 121.7 | 1434.9 | **11.8×** |
| 50×1 | analyze (S2) | 280.6 | 4142.0 | **14.8×** |
| 50×6 | analyze (no S2) | 726.1 | 9384.2 | **12.9×** |
| 50×6 | analyze (S2) | 1666.7 | 26596.8 | **16.0×** |

### HDMR

| Scenario (T×K) | gsax (ms) | SALib (ms) | Speedup |
|---|---:|---:|---:|
| 1×1 | 17.6 | 89.5 | **5.1×** |
| 1×6 | 19.0 | 508.6 | **26.7×** |
| 50×1 | 22.3 | 3990.5 | **178.7×** |
| 50×6 | 37.0 | 28345.8 | **766.3×** |

The speedup grows with output dimensionality because SALib loops over each (T, K) slice in Python while gsax vectorizes with `jax.vmap`. With bootstrap, JIT reuse across resamples adds further gains.

Correctness is validated against analytical Ishigami solutions and SALib on every run. Full benchmark script: [`benchmark_salib.py`](https://github.com/DanielePessina/gsax/blob/master/benchmark_salib.py). See the [docs](https://danielepessina.github.io/gsax/guide/benchmarks) for methodology details.

```bash
uv run python benchmark_salib.py
```
