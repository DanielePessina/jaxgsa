# gsax

**Global Sensitivity Analysis in JAX**

**[Documentation](https://danielepessina.github.io/gsax/)**

`gsax` computes variance-based sensitivity indices entirely in JAX, giving you GPU/TPU acceleration and JIT compilation for free. It provides three complementary methods: **Sobol indices** (via Saltelli sampling), **RS-HDMR** (surrogate-based, works with any input-output pairs), and **PCE** (Polynomial Chaos Expansion with analytical Sobol indices).

## Features

- **Sobol indices** via Saltelli sampling with Sobol quasi-random sequences (`scipy.stats.qmc`)
  - First-order (S1), total-order (ST), and second-order (S2) indices
  - Fused JIT kernels and chunked `jit(vmap(...))` execution for bounded memory on large output grids
  - [**4.7× to 929× faster than SALib**](#benchmark-results) across all output shapes
- **RS-HDMR** (Random Sampling High-Dimensional Model Representation)
  - Works with **any** set of (X, Y) pairs — no structured sampling required
  - B-spline surrogate with ANCOVA decomposition (Sa, Sb, S, ST)
  - Built-in emulator for prediction at new inputs
  - S1/ST properties for direct comparison with Sobol results
- **PCE** (Polynomial Chaos Expansion)
  - Analytical Sobol indices from orthogonal polynomial coefficients (Sudret, 2008)
  - Wiener-Askey scheme: Legendre for uniform, Hermite for Gaussian inputs
  - Built-in emulator and leave-one-out cross-validation RMSE
  - Scalar output only in this version
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
pip install git+https://github.com/danielepessina/gsax.git
```

Or for development:

```bash
git clone https://github.com/danielepessina/gsax.git
cd gsax
pip install -e ".[dev]"
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
- Sobol, RS-HDMR, and PCE workflow examples

Quick map:

- `Problem`
- `sample` / `SamplingResult` / `load`
- `gsax.sobol`: `analyze` / `SAResult`
- `gsax.hdmr`: `analyze` / `emulate` / `HDMRResult` / `HDMREmulator`
- `gsax.pce`: `analyze` / `emulate` / `PCEResult`

All symbols are also re-exported from the top-level `gsax` namespace
(`gsax.analyze()`, `gsax.analyze_hdmr()`, `gsax.analyze_pce()`, etc.).

For runnable walkthroughs, start with the
[Getting Started guide](https://danielepessina.github.io/gsax/guide/getting-started)
and the
[examples section](https://danielepessina.github.io/gsax/examples/basic).

---

## Dependencies

- `jax >= 0.4`
- `jaxlib >= 0.4`
- `scipy >= 1.10`
- `xarray`

## License

See [LICENSE](LICENSE) for details.

## Benchmark Results

gsax vs SALib on a coupled-oscillator model (D=5 parameters, N=1024 base samples). Post-JIT steady-state timings, best of 5, Apple M3 Pro CPU.

| Scenario (T×K) | Method | gsax (ms) | SALib (ms) | Speedup |
|---|---|---:|---:|---:|
| 1×1 | analyze (no S2) | 0.6 | 13.2 | **20.7×** |
| 1×1 | analyze (S2) | 0.9 | 36.7 | **43.0×** |
| 1×1 | analyze_hdmr | 17.8 | 83.1 | **4.7×** |
| 1×6 | analyze (no S2) | 0.9 | 80.9 | **88.2×** |
| 1×6 | analyze (S2) | 1.3 | 280.4 | **216.6×** |
| 1×6 | analyze_hdmr | 19.3 | 501.4 | **26.0×** |
| 50×1 | analyze (no S2) | 2.2 | 661.8 | **300.4×** |
| 50×1 | analyze (S2) | 3.8 | 2092.2 | **554.7×** |
| 50×1 | analyze_hdmr | 23.2 | 4024.6 | **173.5×** |
| 50×6 | analyze (no S2) | 7.6 | 4442.1 | **582.6×** |
| 50×6 | analyze (S2) | 14.3 | 13289.2 | **929.2×** |
| 50×6 | analyze_hdmr | 38.5 | 29115.3 | **757.1×** |

Correctness is validated against analytical Ishigami solutions and SALib on every run. Full benchmark script: [`benchmark_salib.py`](https://github.com/DanielePessina/gsax/blob/dev/benchmark_salib.py). See the [docs](https://danielepessina.github.io/gsax/guide/benchmarks) for methodology details.

```bash
uv run python benchmark_salib.py
```
