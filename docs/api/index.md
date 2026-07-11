# API Reference

This is the canonical reference for the exported `gsax` surface. The package has
eight workflows:

- Sobol: `sample()` -> `analyze()`
- RS-HDMR: `analyze_hdmr()` -> `emulate_hdmr()`
- PCE: `analyze_pce()` -> `emulate_pce()`
- eFAST: `sample_efast()` -> `analyze_efast()`
- DGSM: `sample_mc()` -> `analyze_dgsm()`
- Morris: `sample_morris()` -> `analyze_morris()`
- HSIC: `sample_mc()` -> `analyze_hsic()`
- PAWN: `sample_mc()` -> `analyze_pawn()`

Related docs:

- [Getting Started](/guide/getting-started)
- [Methods](/guide/methods)
- [Examples](/examples/basic)
- [Advanced Workflow](/examples/advanced-workflow)
- [xarray Output](/examples/xarray)

## Package Structure

Since v0.6.0, `gsax` is organized into subpackages:

| Subpackage | Contents |
| --- | --- |
| `gsax.sobol` | `analyze`, `SAResult` |
| `gsax.hdmr` | `analyze`, `emulate`, `HDMRResult`, `HDMREmulator` |
| `gsax.pce` | `analyze`, `emulate`, `PCEResult` |
| `gsax.efast` | `sample`, `analyze`, `EFASTResult` |
| `gsax.dgsm` | `analyze`, `DGSMResult`, `poincare_constant`, `axis_constants` |
| `gsax.morris` | `sample`, `analyze`, `MorrisResult`, `MorrisSamplingResult` |
| `gsax.hsic` | `analyze`, `HSICResult` |
| `gsax.pawn` | `analyze`, `PAWNResult` |

You can import from the subpackages directly:

```python
from gsax.sobol import analyze
from gsax.hdmr import analyze as analyze_hdmr, emulate as emulate_hdmr
from gsax.pce import analyze as analyze_pce, emulate as emulate_pce
from gsax.efast import sample as sample_efast, analyze as analyze_efast
from gsax.dgsm import analyze as analyze_dgsm
from gsax.morris import sample as sample_morris, analyze as analyze_morris
from gsax.hsic import analyze as analyze_hsic
```

All public symbols are also re-exported from the top-level `gsax` namespace for
convenience, so `import gsax; gsax.analyze(...)` still works.

## Exported Surface

Top-level exports from `gsax`:

- [`UniformInputSpec`](#uniforminputspec)
- [`GaussianInputSpec`](#gaussianinputspec)
- [`Problem`](#problem)
- [`sample`](#sample)
- [`SamplingResult`](#samplingresult)
- [`downsample`](#downsample)
- [`verify_prefix`](#verify_prefix)
- [`load`](#load)
- [`enable_compilation_cache`](#enable_compilation_cache)
- [`analyze`](#analyze)
- [`SAResult`](#saresult)
- [`analyze_hdmr`](#analyze_hdmr)
- [`emulate_hdmr`](#emulate_hdmr)
- [`HDMRResult`](#hdmrresult)
- [`HDMREmulator`](#hdmremulator)
- [`analyze_pce`](#analyze_pce)
- [`emulate_pce`](#emulate_pce)
- [`PCEResult`](#pceresult)
- [`sample_efast`](#sample-efast)
- [`analyze_efast`](#analyze-efast)
- [`EFASTResult`](#efastresult)
- [`sample_mc`](#sample-mc)
- [`analyze_dgsm`](#analyze-dgsm)
- [`DGSMResult`](#dgsmresult)
- [`sample_morris`](#sample-morris)
- [`analyze_morris`](#analyze-morris)
- [`MorrisSamplingResult`](#morrissamplingresult)
- [`MorrisResult`](#morrisresult)
- [`analyze_hsic`](#analyze-hsic)
- [`HSICResult`](#hsicresult)
- [`analyze_pawn`](#analyze-pawn)
- [`PAWNResult`](#pawnresult)

## Problem Definition

<a id="problem"></a>
### `Problem`

Immutable dataclass defining parameter names, optional finite bounds, and
optional output names.

```python
@dataclass(frozen=True)
class Problem:
    names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...] | None
    output_names: tuple[str, ...] | None = None
```

| Field / Property | Type | Description |
| --- | --- | --- |
| `names` | `tuple[str, ...]` | Parameter names in model-input order. |
| `bounds` | `tuple[tuple[float, float], ...] \| None` | Finite bounds for uniform-only problems, otherwise `None`. |
| `output_names` | `tuple[str, ...] \| None` | Optional labels for output coordinates in `to_dataset()`. |
| `has_non_uniform_inputs` | `bool` | Whether any parameter uses a non-uniform marginal. |
| `num_vars` | `int` | Property returning `len(names)`. |

Validation and behavior:

- The direct constructor remains the legacy uniform-only path.
- `Problem(names=..., bounds=...)` validates matching lengths and `low < high`.
- `Problem.from_dict(...)` is the canonical constructor for mixed uniform and Gaussian marginals.
- Prefer `output_names` whenever results will be exported with `to_dataset()`.

<a id="uniforminputspec"></a>
#### `UniformInputSpec`

```python
class UniformInputSpec(TypedDict):
    dist: Literal["uniform"]
    low: float
    high: float
```

<a id="gaussianinputspec"></a>
#### `GaussianInputSpec`

```python
class GaussianInputSpec(TypedDict):
    dist: Literal["gaussian"]
    mean: float
    variance: float
    low: NotRequired[float]
    high: NotRequired[float]
```

Gaussian semantics:

- `mean` and `variance` describe the parent Gaussian before truncation.
- `low` and `high` are optional one-sided or two-sided truncation bounds.
- When either bound is present, Sobol sampling uses a true truncated normal transform.

<a id="problem-from-dict"></a>
#### `Problem.from_dict()`

```python
@classmethod
def from_dict(
    cls,
    params: dict[
        str,
        tuple[float, float] | UniformInputSpec | GaussianInputSpec,
    ],
    output_names: tuple[str, ...] | None = None,
) -> Problem
```

`params` keys become `names` in insertion order. Each value may be:

- `(low, high)` as the legacy uniform shorthand
- `UniformInputSpec`
- `GaussianInputSpec`

Minimal example:

```python
import gsax

problem = gsax.Problem.from_dict(
    {
        "amplitude": (0.5, 2.0),
        "frequency": {
            "dist": "gaussian",
            "mean": 3.0,
            "variance": 0.25,
        },
        "damping": {
            "dist": "gaussian",
            "mean": 0.2,
            "variance": 0.01,
            "low": 0.01,
        },
    },
    output_names=("displacement", "velocity"),
)

print(problem.num_vars)  # 3
print(problem.bounds)    # None
```

Related links:

- [Getting Started](/guide/getting-started)
- [Advanced Workflow](/examples/advanced-workflow)
- [Non-Uniform Inputs](/examples/non-uniform-inputs)

## Sobol Workflow

<a id="sample"></a>
### `sample()`

Generate a unique Sobol/Saltelli sample matrix for model evaluation.

```python
def sample(
    problem: Problem,
    n_samples: int,
    *,
    calc_second_order: bool = True,
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
    base_n: int | None = None,
    verbose: bool = True,
) -> SamplingResult
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `problem` | `Problem` | required | Parameter space definition. |
| `n_samples` | `int` | required | Minimum desired number of unique model evaluations. |
| `calc_second_order` | `bool` | `True` | Include BA blocks so `S2` can be computed later. |
| `scramble` | `bool` | `True` | Apply Owen scrambling to the Sobol sequence. |
| `seed` | `int \| np.random.Generator \| None` | `None` | Seed or NumPy generator for reproducibility. |
| `base_n` | `int \| None` | `None` | Explicit Sobol base count (power of 2). When `None`, derived automatically from `n_samples`. |
| `verbose` | `bool` | `True` | Print a compact sampling summary. |

Returns: [`SamplingResult`](#samplingresult)

Shape and behavior:

- `sample()` returns unique rows only, not the expanded Saltelli matrix.
- The returned sample matrix has shape `(n_total, D)`.
- Saltelli construction still happens in the unit cube, then each marginal is
  transformed according to the declared input distribution.
- Uniform inputs use an affine transform from `[0, 1]` into `[low, high]`.
- Gaussian inputs use inverse-CDF transforms, with `truncnorm.ppf` when
  truncation bounds are present.
- `n_samples` is a minimum target, not an exact promise. Internally, `base_n`
  is promoted to the next power of 2 and exact duplicate Saltelli rows are
  removed.
- When `calc_second_order=False`, later Sobol analysis returns `S2=None`.

Minimal example:

```python
import gsax
import jax.numpy as jnp
from gsax.benchmarks.ishigami import PROBLEM, evaluate

sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)
Y = evaluate(jnp.asarray(sampling_result.samples))
result = gsax.analyze(sampling_result, Y)
```

<a id="samplingresult"></a>
### `SamplingResult`

Immutable dataclass returned by `sample()`. It carries the unique rows plus the
metadata needed for `analyze()` to reconstruct the internal Saltelli layout.

```python
@dataclass(frozen=True)
class SamplingResult:
    samples: np.ndarray
    sample_ids: np.ndarray
    expanded_n_total: int
    expanded_to_unique: np.ndarray
    base_n: int
    n_params: int
    calc_second_order: bool
    problem: Problem
```

| Field | Type | Shape / Value | Description |
| --- | --- | --- | --- |
| `samples` | `np.ndarray` | `(n_total, D)` | Unique rows to evaluate with your model. |
| `sample_ids` | `np.ndarray` | `(n_total,)` | Stable integer row IDs aligned with `samples`. |
| `expanded_n_total` | `int` | `N * step` | Expanded Saltelli row count reconstructed internally by `analyze()`. |
| `expanded_to_unique` | `np.ndarray` | `(expanded_n_total,)` | Map from expanded Saltelli rows back to `samples`. |
| `base_n` | `int` | power of 2 | Base Sobol sample count. |
| `n_params` | `int` | `D` | Number of parameters. |
| `calc_second_order` | `bool` | | Whether BA blocks were included. |
| `problem` | `Problem` | | Problem used to generate the samples. |

<a id="samplingresult-n_total"></a>
#### `SamplingResult.n_total`

Property returning `samples.shape[0]`, i.e. the unique-row count.

<a id="samplingresult-samples_df"></a>
#### `SamplingResult.samples_df`

Property returning a pandas `DataFrame` with `SampleID` followed by one column
per parameter. Use it for export, inspection, or joining model outputs back to
inputs.

<a id="samplingresult-save"></a>
#### `SamplingResult.save()`

```python
sampling_result.save("runs/experiment", format="csv")
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | `str \| Path` | required | File stem with no extension. |
| `format` | `str` | `"csv"` | One of `csv`, `txt`, `xlsx`, `parquet`, or `pkl`. |

Behavior and validation:

- Writes `path.<format>` with the unique rows only.
- Writes `path.json` with the `Problem` and Saltelli reconstruction metadata.
- Mixed problems persist their declared input specs in JSON so `load()` can
  rebuild uniform, Gaussian, and truncated Gaussian marginals.
- Writes `path.npz` only when `expanded_to_unique` is not the identity mapping.
- Raises `ValueError` for unsupported formats.
- `xlsx` requires `openpyxl`; `parquet` requires `pyarrow`.

<a id="samplingresult-downsample"></a>
#### `SamplingResult.downsample()`

Return a smaller `SamplingResult` by prefix-slicing to a lower `base_n`.
Optionally pass `Y` (model outputs aligned with `samples`) to get the
corresponding output slice back, similar to how
`sklearn.model_selection.train_test_split` accepts both *X* and *y*.

```python
# Without Y — returns SamplingResult
sr_small = sampling_result.downsample(base_n=8)

# With Y — returns (SamplingResult, Y_small)
sr_small, Y_small = sampling_result.downsample(base_n=8, Y=Y_full)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `base_n` | `int` | required | Target base size (power of 2, `<= self.base_n`). |
| `Y` | `np.ndarray \| None` | `None` | Model outputs with shape `(n_total, ...)`. When provided, the matching prefix is returned alongside the new result. |

Returns: `SamplingResult` when called without `Y`, or `(SamplingResult, np.ndarray)` when `Y` is provided.

Why this works:

- Sobol sequences are **prefix-nested**: at a fixed seed and scramble, the
  first *K* base points of a draw with *N > K* base points are bit-identical
  to drawing *K* base points directly.
- This means you can simulate the model once at the largest `base_n` and
  recover exact results for any smaller power-of-2 `base_n` by slicing — no
  re-simulation needed.
- This property does **not** hold for Latin Hypercube Sampling (LHS), whose
  stratification depends on *N*.

Validation:

- `base_n` must be a power of 2 and `<= self.base_n`.
- When `Y` is provided, it must have at least as many rows as the downsampled
  design requires.
- If `base_n == self.base_n`, the same object is returned (no copy).

Minimal example:

```python
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

# Sample at the largest rung
sr_full = gsax.sample(PROBLEM, n_samples=1, base_n=1024, seed=42)
Y_full = evaluate(sr_full.samples)

# Downsample to smaller rungs — no re-simulation
for base_n in [512, 256, 128, 64]:
    sr_k, Y_k = sr_full.downsample(base_n, Y_full)
    result = gsax.analyze(sr_k, Y_k)
    print(f"base_n={base_n:4d}  S1={result.S1}")
```

<a id="downsample"></a>
### `downsample()`

Module-level convenience wrapper around `SamplingResult.downsample()` that
always takes `Y` and returns both the downsampled result and the output slice.

```python
def downsample(
    sr: SamplingResult,
    Y: np.ndarray,
    base_n: int,
) -> tuple[SamplingResult, np.ndarray]
```

| Parameter | Type | Description |
| --- | --- | --- |
| `sr` | `SamplingResult` | Result from the largest rung. |
| `Y` | `np.ndarray` | Model outputs aligned with `sr.samples`, shape `(sr.n_total, ...)`. |
| `base_n` | `int` | Target base size (power of 2, `<= sr.base_n`). |

Returns: `(sr_small, Y_small)`.

Equivalent to `sr.downsample(base_n, Y)`.

<a id="verify_prefix"></a>
### `verify_prefix()`

Assert that a smaller Sobol design is a bit-exact prefix of a larger one at
the same seed. This validates the mathematical property that makes
`SamplingResult.downsample()` correct.

```python
def verify_prefix(
    problem: Problem,
    base_n_small: int,
    base_n_large: int,
    *,
    calc_second_order: bool = True,
    scramble: bool = True,
    seed: int = 0,
    atol: float = 0.0,
) -> None
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `problem` | `Problem` | required | Problem definition (bounds and distributions). |
| `base_n_small` | `int` | required | Smaller base size (power of 2). |
| `base_n_large` | `int` | required | Larger base size (power of 2, `>= base_n_small`). |
| `calc_second_order` | `bool` | `True` | Saltelli layout order (must match both draws). |
| `scramble` | `bool` | `True` | Whether Owen scrambling is applied (must match both draws). |
| `seed` | `int` | `0` | Integer seed shared by both draws. Must be an `int` so that both `Sobol` engines receive the same Owen scramble. |
| `atol` | `float` | `0.0` | Absolute tolerance; `0.0` demands bit-exact agreement. |

Raises `ValueError` if `base_n_small > base_n_large` or `seed` is not an
integer. Raises `AssertionError` if the prefix property is violated.

Minimal example:

```python
import gsax
from gsax.benchmarks.ishigami import PROBLEM

# Verify that base_n=64 is a prefix of base_n=512 at seed 42
gsax.verify_prefix(PROBLEM, 64, 512, seed=42)
```

<a id="load"></a>
### `load()`

Reconstruct a saved `SamplingResult`.

```python
def load(path: str | Path, *, format: str = "csv") -> SamplingResult
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | `str \| Path` | required | File stem previously passed to `save()`. |
| `format` | `str` | `"csv"` | Must match the format used when saving. |

Validation and behavior:

- Rebuilds `Problem`, `base_n`, `expanded_n_total`, and `expanded_to_unique`.
- Loads both the new `input_specs` metadata and legacy uniform-only metadata
  that stored only `bounds`.
- The sample format is not auto-detected; pass the same `format` explicitly.
- Raises `FileNotFoundError` if the metadata JSON is missing.
- Raises `ValueError` for unsupported formats.

Related links:

- [Save and Reload Samples](/examples/save-load)
- [Methods](/guide/methods)

<a id="analyze"></a>
### `analyze()`

Compute Sobol first-order, total-order, and optional second-order indices from
model outputs evaluated on `SamplingResult.samples`.

```python
def analyze(
    sampling_result: SamplingResult,
    Y: Array,
    *,
    prenormalize: bool = False,
    num_resamples: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    chunk_size: int = 2048,
) -> SAResult
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sampling_result` | `SamplingResult` | required | Result from `sample()`. |
| `Y` | `Array` | required | Model outputs on the unique rows in `sampling_result.samples`. |
| `prenormalize` | `bool` | `False` | Apply SALib-style output standardization over the sample axis before analysis. |
| `num_resamples` | `int` | `0` | Number of bootstrap resamples. |
| `conf_level` | `float` | `0.95` | Confidence level for bootstrap intervals. |
| `ci_method` | `Literal["quantile", "gaussian"]` | `"quantile"` | Bootstrap CI summary method. `quantile` returns percentile endpoints; `gaussian` returns symmetric gaussian endpoints from bootstrap standard deviation. |
| `key` | `Array \| None` | `None` | Required JAX PRNG key when `num_resamples > 0`. |
| `chunk_size` | `int` | `2048` | `(T, K)` output combinations per batch on the no-bootstrap path. |

Accepted output shapes:

- `(n_total,)` for scalar output
- `(n_total, K)` for multi-output
- `(n_total, T, K)` for time-series multi-output

Validation and behavior:

- A 2D array is always interpreted as `(N, K)`, never `(N, T)`.
- For a time-series with one output, reshape to `(N, T, 1)`.
- When `prenormalize=True`, `Y` is centered and scaled once per output slice
  over the sample axis after Saltelli reconstruction and non-finite-group
  cleanup.
- `ci_method` accepts `"quantile"` and `"gaussian"`. The option is ignored
  when `num_resamples == 0` because no CI arrays are produced.
- If `num_resamples > 0`, `key` is required or `ValueError` is raised.
- Sample groups containing any non-finite values are dropped before analysis.
- If every group is invalid, `ValueError("All samples contain non-finite values")`
  is raised.
- Zero-variance slices emit warnings because Sobol indices become undefined.
- Bootstrap intervals always remain lower/upper endpoint arrays, not SALib-style
  half-widths. `ci_method="quantile"` uses percentile endpoints, while
  `ci_method="gaussian"` uses symmetric gaussian endpoints from bootstrap
  standard deviation.

Returns: [`SAResult`](#saresult)

<a id="saresult"></a>
### `SAResult`

Dataclass holding Sobol point estimates, optional bootstrap intervals, and
diagnostic NaN counts.

```python
@dataclass
class SAResult:
    S1: Array
    ST: Array
    S2: Array | None
    problem: Problem
    S1_conf: Array | None = None
    ST_conf: Array | None = None
    S2_conf: Array | None = None
    nan_counts: dict[str, int] | None = None
```

| Field | Shape | Description |
| --- | --- | --- |
| `S1` | `(D,)` / `(K, D)` / `(T, K, D)` | First-order Sobol indices. |
| `ST` | same as `S1` | Total-order Sobol indices. |
| `S2` | `(D, D)` / `(K, D, D)` / `(T, K, D, D)` or `None` | Symmetric second-order matrix with `NaN` diagonal. |
| `S1_conf`, `ST_conf`, `S2_conf` | `(2, ...)` or `None` | Bootstrap lower and upper bounds. |
| `problem` | `Problem` | Problem carried through for labeling and metadata. |
| `nan_counts` | `dict[str, int] \| None` | Diagnostic NaN counts in the result arrays. |

Shape contract:

| `Y` shape passed to `analyze()` | `S1` / `ST` | `S2` |
| --- | --- | --- |
| `(N,)` | `(D,)` | `(D, D)` |
| `(N, K)` | `(K, D)` | `(K, D, D)` |
| `(N, T, K)` | `(T, K, D)` | `(T, K, D, D)` |

`S2` is `None` when `sampling_result.calc_second_order` is `False`. Confidence
interval arrays, when present, prepend a leading dimension of 2 for
`[lower, upper]`.

<a id="saresult-to_dataset"></a>
#### `SAResult.to_dataset()`

```python
ds = result.to_dataset(time_coords=None)
```

Converts Sobol results to a labeled `xarray.Dataset`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `time_coords` | `list \| np.ndarray \| None` | `None` | Coordinate values for the time dimension on 3D results. |

Behavior:

- Uses `problem.names` for parameter coordinates.
- Uses `problem.output_names` when available, otherwise `y0`, `y1`, and so on.
- Splits confidence intervals into `*_lower` and `*_upper` dataset variables.
- Uses `param_i` and `param_j` dimensions for `S2`.

Minimal example:

```python
import jax
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)
Y = evaluate(sampling_result.samples)
result = gsax.analyze(
    sampling_result,
    Y,
    prenormalize=True,
    num_resamples=200,
    key=jax.random.key(0),
)

print(result.S1)
print(result.ST)
print(result.S2 is not None)
print(result.nan_counts)
```

Related links:

- [Bootstrap Confidence Intervals](/examples/bootstrap)
- [xarray Output](/examples/xarray)

## RS-HDMR Workflow

<a id="analyze_hdmr"></a>
### `analyze_hdmr()`

Fit an RS-HDMR surrogate on arbitrary `(X, Y)` pairs and derive ANCOVA-based
sensitivity indices.

```python
def analyze_hdmr(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    prenormalize: bool = False,
    maxorder: int = 2,
    maxiter: int = 100,
    m: int = 2,
    lambdax: float = 0.01,
    chunk_size: int = 2048,
) -> HDMRResult
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `problem` | `Problem` | required | Bounds and names used to normalize `X`. |
| `X` | `Array` | required | Input array with shape `(N, D)`. |
| `Y` | `Array` | required | Output array with shape `(N,)`, `(N, K)`, or `(N, T, K)`. |
| `prenormalize` | `bool` | `False` | Apply SALib-style output standardization over the sample axis before fitting. |
| `maxorder` | `int` | `2` | Maximum HDMR expansion order. |
| `maxiter` | `int` | `100` | Maximum backfitting iterations. |
| `m` | `int` | `2` | Number of B-spline intervals. |
| `lambdax` | `float` | `0.01` | Tikhonov regularization strength. |
| `chunk_size` | `int` | `2048` | Maximum `(T, K)` combinations per batch. |

Validation and behavior:

- `X.shape[1]` must match `problem.num_vars`.
- Non-uniform inputs (Gaussian, truncated Gaussian) are supported via CDF
  mapping to `[0, 1]` before surrogate fitting.
- At least 300 rows are required or `ValueError` is raised.
- `maxorder` must be 1, 2, or 3.
- When `D == 2`, `maxorder` cannot exceed 2.
- `chunk_size` must be at least 1.
- A 2D output array is always treated as `(N, K)`.
- When `prenormalize=True`, `Y` is centered and scaled once per output slice
  over the sample axis before surrogate fitting.

Returns: [`HDMRResult`](#hdmrresult)

<a id="emulate_hdmr"></a>
### `emulate_hdmr()`

Predict at new input points using the surrogate stored in an `HDMRResult`.

```python
def emulate_hdmr(result: HDMRResult, X_new: Array) -> Array
```

| Parameter | Type | Description |
| --- | --- | --- |
| `result` | `HDMRResult` | Must contain `emulator`. |
| `X_new` | `Array` | New input points with shape `(N_new, D)`. |

Validation and behavior:

- Raises `ValueError` when `result.emulator is None`.
- Returns `(N_new,)`, `(N_new, K)`, or `(N_new, T, K)` to match the fitted
  output layout.
- When the result was fit with `prenormalize=True`, predictions are mapped back
  to the original output scale before being returned.
- Not JIT-compatible because `HDMRResult` is not a JAX pytree.

<a id="hdmrresult"></a>
### `HDMRResult`

Dataclass holding ANCOVA-decomposed HDMR sensitivities and optional emulator
artifacts.

```python
@dataclass
class HDMRResult:
    Sa: Array
    Sb: Array
    S: Array
    ST: Array
    problem: Problem
    terms: tuple[str, ...]
    emulator: HDMREmulator | None = None
    select: Array | None = None
    rmse: Array | None = None
```

| Field | Shape | Description |
| --- | --- | --- |
| `Sa` | `(n_terms,)` / `(K, n_terms)` / `(T, K, n_terms)` | Structural contribution per term. |
| `Sb` | same as `Sa` | Correlative contribution per term. |
| `S` | same as `Sa` | Total contribution per term: `Sa + Sb`. |
| `ST` | `(D,)` / `(K, D)` / `(T, K, D)` | Total contribution per parameter. |
| `terms` | `tuple[str, ...]` | Human-readable term labels such as `"x1/x2"`. |
| `emulator` | `HDMREmulator \| None` | Surrogate coefficients and static metadata. |
| `select` | `(n_terms,)` or `None` | F-test selection counts summed across outputs. |
| `rmse` | `()` / `(K,)` / `(T, K)` or `None` | Emulator RMSE without the sample axis. |

<a id="hdmrresult-s1"></a>
#### `HDMRResult.S1`

Property returning the first-order structural contribution extracted from the
first `D` HDMR terms:

```python
hdmr.S1  # shape matches hdmr.ST
```

This is the Sobol-compatible first-order view of an HDMR fit.

<a id="hdmrresult-to_dataset"></a>
#### `HDMRResult.to_dataset()`

```python
ds = hdmr.to_dataset(time_coords=None)
```

Converts HDMR results to a labeled `xarray.Dataset`.

Behavior:

- Uses `term` for `Sa`, `Sb`, `S`, and `select`.
- Uses `param` for `ST`.
- Uses `problem.output_names` when available, otherwise generated labels.
- Uses `time_coords` when passed for 3D results.

<a id="hdmremulator"></a>
### `HDMREmulator`

Typed dictionary stored on `HDMRResult.emulator`.

```python
class HDMREmulator(TypedDict):
    C1: Array
    C2: Array | None
    C3: Array | None
    f0: Array
    prenormalize: bool
    y_mean: Array
    y_std: Array
    m: int
    maxorder: int
    c2: list[tuple[int, int]]
    c3: list[tuple[int, int, int]]
```

| Key | Description |
| --- | --- |
| `C1`, `C2`, `C3` | Fitted B-spline coefficients for first-, second-, and third-order terms. |
| `f0` | Intercept term in the emulator. |
| `prenormalize` | Whether the HDMR fit standardized outputs before fitting. |
| `y_mean`, `y_std` | Per-output-slice statistics used to map prenormalized predictions back to the original scale. |
| `m` | Number of spline intervals used during fitting. |
| `maxorder` | Expansion order used to build the surrogate. |
| `c2`, `c3` | Term-index mappings for pairwise and triple interaction terms. |

Minimal example:

```python
import jax
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

key = jax.random.PRNGKey(42)
bounds = jnp.array(PROBLEM.bounds)
X = jax.random.uniform(key, (2000, PROBLEM.num_vars), minval=bounds[:, 0], maxval=bounds[:, 1])
Y = evaluate(X)

hdmr = gsax.analyze_hdmr(PROBLEM, X, Y, maxorder=2)
Y_pred = gsax.emulate_hdmr(hdmr, X[:5])

print(hdmr.S1)
print(hdmr.ST)
print(Y_pred.shape)
```

Related links:

- [Methods](/guide/methods)
- [RS-HDMR Example](/examples/hdmr)
- [Advanced Workflow](/examples/advanced-workflow)

## PCE Workflow

<a id="analyze_pce"></a>
### `analyze_pce()`

Compute Sobol indices via Polynomial Chaos Expansion. Fits an orthogonal
polynomial surrogate to `(X, Y)` data and extracts first-order, total-order,
and second-order indices analytically from the expansion coefficients
(Sudret, 2008).

```python
def analyze_pce(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    order: int = 3,
    ridge: float = 1e-8,
    fit_ratio: float = 0.5,
) -> PCEResult
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `problem` | `Problem` | required | Parameter names and distributions. |
| `X` | `Array` | required | Input array with shape `(N, D)`. |
| `Y` | `Array` | required | Output array with shape `(N,)` (scalar output only). |
| `order` | `int` | `3` | Maximum total polynomial degree. Automatically reduced if the number of terms would exceed `fit_ratio * N`. |
| `ridge` | `float` | `1e-8` | Tikhonov regularization parameter for least-squares fit. |
| `fit_ratio` | `float` | `0.5` | Maximum ratio of terms to samples before the order is reduced. |

Validation and behavior:

- `Y` must be 1D. Multi-output and time-series outputs are not yet supported.
- `X.shape[1]` must match `problem.num_vars`.
- Uniform and truncated-Gaussian inputs use Legendre polynomials on `[-1, 1]`.
- Untruncated Gaussian inputs use Hermite polynomials standardized to `N(0, 1)`.
- The polynomial order is automatically reduced when the term count would
  exceed `fit_ratio * N` to prevent overfitting.

Returns: [`PCEResult`](#pceresult)

<a id="emulate_pce"></a>
### `emulate_pce()`

Predict at new input points using the fitted PCE.

```python
def emulate_pce(result: PCEResult, X_new: Array) -> Array
```

| Parameter | Type | Description |
| --- | --- | --- |
| `result` | `PCEResult` | Result from `analyze_pce()`. |
| `X_new` | `Array` | New input points with shape `(N_new, D)`. |

Returns `(N_new,)` predicted outputs.

<a id="pceresult"></a>
### `PCEResult`

Dataclass holding PCE-derived Sobol indices and fitted expansion coefficients.

```python
@dataclass
class PCEResult:
    S1: Array
    ST: Array
    S2: Array
    problem: Problem
    coefficients: Array
    multi_index: np.ndarray
    order: int
    loo_rmse: Array | None = None
```

| Field | Shape | Description |
| --- | --- | --- |
| `S1` | `(D,)` | First-order Sobol indices. |
| `ST` | `(D,)` | Total-order Sobol indices. |
| `S2` | `(D, D)` | Second-order interaction matrix with `NaN` diagonal. |
| `coefficients` | `(n_terms,)` | Fitted PCE coefficients. |
| `multi_index` | `(n_terms, D)` | Multi-index array mapping terms to polynomial degrees. |
| `order` | `int` | Effective total polynomial degree used (may be less than requested). |
| `loo_rmse` | `scalar or None` | Leave-one-out cross-validation RMSE. |

<a id="pceresult-to_dataset"></a>
#### `PCEResult.to_dataset()`

```python
ds = result.to_dataset()
```

Converts PCE results to a labeled `xarray.Dataset` with `param` coordinates.
Includes `S1`, `ST`, `S2` (with `param_i` / `param_j` dimensions), and
`loo_rmse` when available.

Minimal example:

```python
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate
from gsax.pce import analyze, emulate

sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)
X = sampling_result.samples
Y = evaluate(X)

result = analyze(PROBLEM, X, Y, order=4)
Y_pred = emulate(result, X[:5])

print(result.S1)
print(result.ST)
print(result.loo_rmse)
```

Related links:

- [Methods](/guide/methods)

## eFAST Workflow

<a id="sample-efast"></a>
### `sample_efast()` {#sample-efast}

Generate eFAST samples along sinusoidal search curves.

```python
def sample_efast(
    problem: Problem,
    N: int,
    *,
    M: int = 4,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `problem` | `Problem` | required | Parameter space definition. |
| `N` | `int` | required | Number of samples per search curve. Must satisfy `N > 4*M^2`. |
| `M` | `int` | `4` | Interference factor (number of harmonics). |
| `seed` | `int \| np.random.Generator \| None` | `None` | Seed or NumPy generator for reproducibility. |

Returns: `np.ndarray` with shape `(N * D, D)`.

Shape and behavior:

- For each of the D parameters, generates N samples along a search curve where
  the focal parameter oscillates at the primary frequency omega_0 and the
  remaining parameters oscillate at lower complementary frequencies.
- The total output is the concatenation of all D search curves.
- Samples are transformed from `[0, 1]` into the problem's physical parameter
  space using CDF-based transforms matching the declared input distributions.
- The primary frequency omega_0 is computed as `(N - 1) // (2 * M)`.

Minimal example:

```python
from gsax import efast
from gsax.benchmarks.ishigami import PROBLEM

X = efast.sample(PROBLEM, N=4096, M=4, seed=42)
print(X.shape)  # (12288, 3)
```

<a id="analyze-efast"></a>
### `analyze_efast()` {#analyze-efast}

Compute eFAST first-order and total-order sensitivity indices from model
outputs evaluated on eFAST samples.

```python
def analyze_efast(
    problem: Problem,
    Y: Array,
    *,
    M: int = 4,
    prenormalize: bool = False,
    chunk_size: int = 2048,
) -> EFASTResult
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `problem` | `Problem` | required | Problem definition with D parameters. |
| `Y` | `Array` | required | Model outputs evaluated at eFAST samples. |
| `M` | `int` | `4` | Interference factor used during sampling. |
| `prenormalize` | `bool` | `False` | Center and scale each output slice to unit variance before computing indices. |
| `chunk_size` | `int` | `2048` | Maximum number of output slices per vmapped batch. |

Accepted output shapes:

- `(N*D,)` for scalar output
- `(N*D, K)` for K output variables
- `(N*D, T, K)` for K outputs over T time steps

Validation and behavior:

- A 2D array is always interpreted as `(N*D, K)`, never `(N*D, T)`.
- For a time-series with one output, reshape to `(N*D, T, 1)`.
- The leading dimension of `Y` must be a multiple of `problem.num_vars`.
- `M` must match the value used during sampling.
- Non-finite values in `Y` will propagate into the computed indices.
- Zero-variance output slices emit warnings.
- Indices outside `[0, 1]` indicate insufficient samples or near-zero output variance.

Returns: [`EFASTResult`](#efastresult)

<a id="efastresult"></a>
### `EFASTResult` {#efastresult}

Dataclass holding eFAST sensitivity indices.

```python
@dataclass
class EFASTResult:
    S1: Array
    ST: Array
    problem: Problem
    omega_0: int = 0
    M: int = 4
```

| Field | Shape | Description |
| --- | --- | --- |
| `S1` | `(D,)` / `(K, D)` / `(T, K, D)` | First-order Sobol indices from Fourier amplitudes at harmonics of omega_0. |
| `ST` | same as `S1` | Total-order Sobol indices from complementary frequencies. |
| `problem` | `Problem` | Problem definition used for the analysis. |
| `omega_0` | `int` | Primary frequency used in the Fourier decomposition. |
| `M` | `int` | Interference factor (number of harmonics summed). |

Shape contract:

| `Y` shape passed to `analyze_efast()` | `S1` / `ST` |
| --- | --- |
| `(N*D,)` | `(D,)` |
| `(N*D, K)` | `(K, D)` |
| `(N*D, T, K)` | `(T, K, D)` |

eFAST does not produce second-order (S2) indices.

<a id="efastresult-to_dataset"></a>
#### `EFASTResult.to_dataset()`

```python
ds = result.to_dataset(time_coords=None)
```

Converts eFAST results to a labeled `xarray.Dataset`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `time_coords` | `list \| np.ndarray \| None` | `None` | Coordinate values for the time dimension on 3D results. |

Behavior:

- Uses `problem.names` for `param` coordinates.
- Uses `problem.output_names` when available, otherwise `y0`, `y1`, and so on.
- For 3D results, defaults to integer time indices when `time_coords` is not provided.
- The dataset contains only `S1` and `ST` variables (no `S2`).

Minimal example:

```python
import jax.numpy as jnp
from gsax import efast
from gsax.benchmarks.ishigami import PROBLEM, evaluate

X = efast.sample(PROBLEM, N=4096, seed=42)
Y = evaluate(jnp.asarray(X))
result = efast.analyze(PROBLEM, Y)

print(result.S1)
print(result.ST)
```

Related links:

- [eFAST Example](/examples/efast)
- [Methods](/guide/methods)
- [xarray Output](/examples/xarray)

## DGSM Workflow

<a id="sample-mc"></a>
### `sample_mc()` {#sample-mc}

Generate plain Monte Carlo samples from the declared input distributions.
Unlike Sobol/Saltelli sampling, these samples have no quasi-random structure.
Suitable for DGSM and other methods that require i.i.d. draws.

```python
def sample_mc(
    problem: Problem,
    N: int,
    *,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `problem` | `Problem` | required | Parameter space definition with distributions. |
| `N` | `int` | required | Number of samples. Must be >= 1. |
| `seed` | `int \| np.random.Generator \| None` | `None` | Seed or NumPy generator for reproducibility. |

Returns: `np.ndarray` with shape `(N, D)`.

Shape and behavior:

- Each row is an i.i.d. draw from the joint input distribution defined by the
  problem's parameter specs.
- Uniform inputs are drawn uniformly on `[low, high]`.
- Gaussian inputs use inverse-CDF transforms, with truncated normal when
  truncation bounds are present.
- The returned array is in the problem's physical units, not the unit cube.

Minimal example:

```python
import gsax
from gsax.benchmarks.ishigami import PROBLEM

X = gsax.sample_mc(PROBLEM, N=10000, seed=42)
print(X.shape)  # (10000, 3)
```

<a id="analyze-dgsm"></a>
### `analyze_dgsm()` {#analyze-dgsm}

Compute DGSM sensitivity measures and Sobol index bounds from a
JAX-differentiable function or pre-computed Jacobians.

Two calling conventions are supported:

**Autodiff path** (primary): pass `fn` and `X`. The function is differentiated
via `jax.jacrev` and evaluated to obtain both the Jacobian and forward outputs.

**Pre-computed path**: pass `Y` and `dfdx`. Useful when the model is not
JAX-differentiable or when the Jacobian has been computed externally.

```python
def analyze_dgsm(
    problem: Problem,
    fn: Callable | None = None,
    X: Array | None = None,
    *,
    Y: Array | None = None,
    dfdx: Array | None = None,
    chunk_size: int | None = None,
) -> DGSMResult
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `problem` | `Problem` | required | Problem definition with D parameters. |
| `fn` | `Callable \| None` | `None` | JAX-differentiable function `(D,) -> ()` or `(D,) -> (T,)`. |
| `X` | `Array \| None` | `None` | Sample matrix `(N, D)` in the problem's physical units. |
| `Y` | `Array \| None` | `None` | Pre-computed forward outputs `(N,)` or `(N, T)`. |
| `dfdx` | `Array \| None` | `None` | Pre-computed Jacobian `(N, D)` or `(N, T, D)`. |
| `chunk_size` | `int \| None` | `None` | Batch size for autodiff to limit peak memory. |

Validation and behavior:

- Provide either `(fn, X)` for the autodiff path or `(Y, dfdx)` for the
  pre-computed path. Providing neither or mixing raises `ValueError`.
- `X.shape[1]` must match `problem.num_vars`.
- `fn` must accept a 1D array of shape `(D,)` and return a scalar `()` or
  a 1D array `(T,)`.
- When `chunk_size` is set, the autodiff path processes samples in batches,
  padding the last chunk to avoid JIT recompilation.
- For the pre-computed path, `dfdx` may be `(N, D)` for scalar output or
  `(N, T, D)` for multi-output, and `Y` row count must match.
- Zero-variance outputs produce `NaN` bounds (division by zero guarded).
- A warning is emitted when upper bounds fall below lower bounds, suggesting
  insufficient samples.

Returns: [`DGSMResult`](#dgsmresult)

Minimal example:

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM

def ishigami(x):
    return jnp.sin(x[0]) + 7.0 * jnp.sin(x[1])**2 + 0.1 * x[2]**4 * jnp.sin(x[0])

X = gsax.sample_mc(PROBLEM, N=10000, seed=42)
result = gsax.analyze_dgsm(PROBLEM, ishigami, jnp.asarray(X))

print(result.nu)           # (3,) — importance measures
print(result.upper_bound)  # (3,) — Poincaré upper bounds on ST
print(result.lower_bound)  # (3,) — Kucherenko-Song lower bounds on ST
```

<a id="dgsmresult"></a>
### `DGSMResult` {#dgsmresult}

Dataclass holding DGSM sensitivity measures and Sobol index bounds.

```python
@dataclass
class DGSMResult:
    nu: Array
    sigma: Array
    upper_bound: Array
    lower_bound: Array
    var_y: Array
    problem: Problem
```

| Field | Shape | Description |
| --- | --- | --- |
| `nu` | `(D,)` / `(K, D)` | $\mathbb{E}[(\partial f / \partial X_i)^2]$, the DGSM importance measure. |
| `sigma` | `(D,)` / `(K, D)` | $\mathbb{E}[\partial f / \partial X_i]$, the mean partial derivative. |
| `upper_bound` | `(D,)` / `(K, D)` | $C_i \cdot \nu_i / \mathrm{Var}(Y)$, Poincaré upper bound on $S_T$. |
| `lower_bound` | `(D,)` / `(K, D)` | $\mathrm{Var}(X_i) \cdot \sigma_i^2 / \mathrm{Var}(Y)$, Kucherenko–Song lower bound on $S_T$. |
| `var_y` | `()` / `(K,)` | Output variance (scalar for single output, per-component for multi-output). |
| `problem` | `Problem` | Problem definition used for the analysis. |

Shape contract: scalar-output models (`fn: (D,) -> ()`) produce `(D,)` index
arrays; multi-output models (`fn: (D,) -> (K,)`) produce `(K, D)`.

<a id="dgsmresult-to_dataset"></a>
#### `DGSMResult.to_dataset()`

```python
ds = result.to_dataset()
```

Converts DGSM results to a labeled `xarray.Dataset`.

Behavior:

- For scalar output (`T = 1`), variables have dimension `(param,)`.
- For multi-output (`T > 1`), variables have dimensions `(output, param)`.
- Uses `problem.names` for `param` coordinates.
- Uses `problem.output_names` when available, otherwise integer indices.
- Dataset contains `nu`, `sigma`, `upper_bound`, and `lower_bound` variables.

Minimal example:

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM

def ishigami(x):
    return jnp.sin(x[0]) + 7.0 * jnp.sin(x[1])**2 + 0.1 * x[2]**4 * jnp.sin(x[0])

X = gsax.sample_mc(PROBLEM, N=10000, seed=42)
result = gsax.analyze_dgsm(PROBLEM, ishigami, jnp.asarray(X))
ds = result.to_dataset()
print(ds)
```

### `poincare_constant()`

Compute the Poincare constant $C(p)$ for a single marginal distribution.

```python
def poincare_constant(
    spec: _NormalizedInputSpec,
    *,
    grid: int = 512,
) -> float
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `spec` | `_NormalizedInputSpec` | required | Normalized input spec tuple `(dist, first, second, low, high)`. |
| `grid` | `int` | `512` | Number of P1 elements for truncated-Normal spectral solve. |

Poincare constants by distribution:

| Distribution | $C_i$ |
| --- | --- |
| Uniform $[a, b]$ | $(b - a)^2 / \pi^2$ |
| Gaussian $\mathcal{N}(\mu, \sigma^2)$ | $\sigma^2$ |
| Truncated Normal | Spectral solve (P1 FEM Neumann eigenproblem) |

### `axis_constants()`

Compute per-axis Poincare constants and marginal variances from a `Problem`.

```python
def axis_constants(problem: Problem) -> tuple[np.ndarray, np.ndarray]
```

| Parameter | Type | Description |
| --- | --- | --- |
| `problem` | `Problem` | Problem definition with D parameters. |

Returns a tuple `(C, Var)` where both arrays have shape `(D,)`:

- `C[i]` is the Poincare constant of the i-th input's marginal.
- `Var[i]` is the marginal variance of the i-th input.

These are used internally by `analyze_dgsm()` to compute the upper and lower
bounds on total Sobol indices.

Related links:

- [DGSM Example](/examples/dgsm)
- [Methods](/guide/methods)

---

## Morris Workflow

<a id="sample-morris"></a>
### `sample_morris()` {#sample-morris}

Generate unique Morris elementary-effects samples for model evaluation.

```python
def sample_morris(
    problem: Problem,
    n_trajectories: int,
    *,
    num_levels: int = 4,
    method: Literal["trajectory", "radial"] = "trajectory",
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
    truncation_quantile: float = 0.005,
    verbose: bool = True,
) -> MorrisSamplingResult
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `problem` | `Problem` | required | Problem definition with uniform and/or Gaussian marginals. |
| `n_trajectories` | `int` | required | Number of trajectories r (>= 2). Each contributes one elementary effect per parameter; typical screening uses 10-50. |
| `num_levels` | `int` | `4` | Grid levels `p` for the trajectory design (step `delta = p / (2 * (p - 1))`). Ignored by the radial design. |
| `method` | `Literal["trajectory", "radial"]` | `"trajectory"` | `"trajectory"` (Morris 1991 grid walks) or `"radial"` (Campolongo 2011 star designs around scrambled-Sobol' base points). |
| `scramble` | `bool` | `True` | Whether to Owen-scramble the Sobol' sequence (radial design only). |
| `seed` | `int \| np.random.Generator \| None` | `None` | Seed or NumPy generator for reproducibility. |
| `truncation_quantile` | `float` | `0.005` | Tail probability `q` excluded on each side of every Gaussian marginal's grid (the default probes the 0.5%-99.5% quantile range). Applied to truncated Gaussians as well for consistency; ignored for uniform marginals. Must be in `(0, 0.5)`. |
| `verbose` | `bool` | `True` | Print a short summary including how many duplicate rows were removed. |

Returns: [`MorrisSamplingResult`](#morrissamplingresult)

Shape and behavior:

- Builds `n_trajectories` one-at-a-time paths of `D + 1` points each, so the
  expanded design always has `n_trajectories * (D + 1)` rows.
- Like Sobol' `sample()`, exact duplicate rows are removed while preserving
  first-occurrence order, and only the unique rows are returned for
  evaluation. Trajectory points live on a coarse `num_levels` grid, so
  duplicates across trajectories are common in low dimensions and
  deduplication saves real model evaluations.
- Gaussian marginals are supported through a truncated-quantile grid: the
  Morris design includes the unit-cube boundaries, which an unbounded inverse
  CDF maps to infinity, so for each Gaussian parameter the unit-cube
  coordinate is confined to `[q, 1 - q]` (`q = truncation_quantile`) before
  the transform. Applied to truncated Gaussians as well for consistency;
  uniform marginals are untouched. Deduplication and prefix-nesting are
  unaffected.
- Elementary effects remain per unit of the original grid coordinate;
  `MorrisResult.to_physical_units()` is unavailable for problems with
  non-uniform marginals because the transform is nonlinear.
- Trajectory design: even `num_levels` values make all grid levels equally
  probable; odd values trigger a warning.
- Radial design: base and auxiliary points come from a scrambled Sobol'
  sequence; a near-zero step raises `ValueError` (use `scramble=True` or a
  different seed).
- `n_trajectories < 2`, `num_levels < 2`, an unknown `method`, or
  `truncation_quantile` outside `(0, 0.5)` raise `ValueError`.

Minimal example:

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

sampling_result = gsax.sample_morris(PROBLEM, n_trajectories=50, seed=42)
Y = evaluate(jnp.asarray(sampling_result.samples))
result = gsax.analyze_morris(sampling_result, Y)
```

<a id="morrissamplingresult"></a>
### `MorrisSamplingResult` {#morrissamplingresult}

Immutable dataclass returned by `sample_morris()`. It carries the unique rows
plus the metadata needed for `analyze_morris()` to reconstruct the expanded
design and locate each elementary effect inside it.

```python
@dataclass(frozen=True)
class MorrisSamplingResult:
    samples: np.ndarray
    expanded_n_total: int
    expanded_to_unique: np.ndarray
    n_trajectories: int
    num_levels: int
    method: Literal["trajectory", "radial"]
    ee_idx_after: np.ndarray
    ee_idx_before: np.ndarray
    ee_delta: np.ndarray
    n_params: int
    problem: Problem
```

| Field | Type | Shape / Value | Description |
| --- | --- | --- | --- |
| `samples` | `np.ndarray` | `(n_unique, D)` | Unique rows to evaluate with your model, in the problem's physical units. |
| `expanded_n_total` | `int` | `r * (D + 1)` | Row count of the full expanded design before deduplication. |
| `expanded_to_unique` | `np.ndarray` | `(expanded_n_total,)` | Map from each expanded row to its row index in `samples`. |
| `n_trajectories` | `int` | `r` | Number of trajectories, the Morris repetition unit. |
| `num_levels` | `int` | `p` | Grid levels used by the trajectory design (unused by the radial design). |
| `method` | `Literal["trajectory", "radial"]` | | Design generator. |
| `ee_idx_after` | `np.ndarray` | `(r, D)` | Expanded-row index of the perturbed point of each elementary effect. |
| `ee_idx_before` | `np.ndarray` | `(r, D)` | Expanded-row index of the reference point of each elementary effect. |
| `ee_delta` | `np.ndarray` | `(r, D)` | Signed unit-cube step of each elementary effect, so that `EE = (Y[after] - Y[before]) / delta`. |
| `n_params` | `int` | `D` | Number of problem dimensions. |
| `problem` | `Problem` | | Problem used to transform the samples. |

<a id="morrissamplingresult-n_total"></a>
#### `MorrisSamplingResult.n_total`

Property returning `samples.shape[0]`, i.e. the unique-row count.

<a id="morrissamplingresult-downsample"></a>
#### `MorrisSamplingResult.downsample()`

Return a smaller `MorrisSamplingResult` by prefix-slicing to fewer
trajectories. Optionally pass `Y` (model outputs aligned with `samples`) to get
the corresponding output slice back, just like `SamplingResult.downsample()`.

```python
# Without Y — returns MorrisSamplingResult
sr_small = sampling_result.downsample(n_trajectories=25)

# With Y — returns (MorrisSamplingResult, Y_small)
sr_small, Y_small = sampling_result.downsample(n_trajectories=25, Y=Y_full)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_trajectories` | `int` | required | Target trajectory count (`2 <= m <= r`). |
| `Y` | `np.ndarray \| None` | `None` | Model outputs with shape `(n_total, ...)`. When provided, the matching prefix is returned alongside the new result. |

Returns: `MorrisSamplingResult` when called without `Y`, or
`(MorrisSamplingResult, np.ndarray)` when `Y` is provided.

Why this works:

- Trajectories are generated sequentially from independent draws (trajectory
  design) or from prefix-nested Sobol' points (radial design), so the first
  *m* trajectories of an *r*-trajectory run are identical to drawing *m*
  trajectories directly with the same seed.
- This means you can simulate the model once at the largest `n_trajectories`
  and recover exact results for any smaller trajectory count by slicing — no
  re-simulation needed.

Validation:

- `n_trajectories` must satisfy `2 <= n_trajectories <= self.n_trajectories`.
- When `Y` is provided, `Y.shape[0]` must match `n_total`.
- If `n_trajectories == self.n_trajectories`, the same object is returned
  (no copy).

Minimal example:

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

# Sample at the largest rung
sr_full = gsax.sample_morris(PROBLEM, n_trajectories=100, seed=42)
Y_full = evaluate(jnp.asarray(sr_full.samples))

# Downsample to smaller rungs — no re-simulation
for r in [50, 25, 10]:
    sr_r, Y_r = sr_full.downsample(r, Y_full)
    result = gsax.analyze_morris(sr_r, Y_r)
    print(f"r={r:3d}  mu_star={result.mu_star}")
```

<a id="analyze-morris"></a>
### `analyze_morris()` {#analyze-morris}

Compute Morris elementary-effects screening measures (mu, mu_star, sigma) from
model outputs evaluated on `MorrisSamplingResult.samples`.

```python
def analyze_morris(
    sampling_result: MorrisSamplingResult,
    Y: Array,
    *,
    prenormalize: bool = False,
    num_resamples: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    chunk_size: int = 2048,
) -> MorrisResult
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sampling_result` | `MorrisSamplingResult` | required | Result from `sample_morris()`. |
| `Y` | `Array` | required | Model outputs on the unique rows in `sampling_result.samples`. |
| `prenormalize` | `bool` | `False` | Standardize each output slice to mean 0 and unit standard deviation over the expanded sample axis before computing elementary effects. |
| `num_resamples` | `int` | `0` | Number of bootstrap resamples (over trajectories, with replacement) for confidence intervals. |
| `conf_level` | `float` | `0.95` | Confidence level for bootstrap intervals. |
| `ci_method` | `Literal["quantile", "gaussian"]` | `"quantile"` | Bootstrap CI endpoint method. `quantile` returns percentile endpoints; `gaussian` returns symmetric endpoints around the estimate. |
| `key` | `Array \| None` | `None` | Required JAX PRNG key when `num_resamples > 0`. |
| `chunk_size` | `int` | `2048` | Bootstrap resamples per vmap batch, bounding peak memory. |

Accepted output shapes:

- `(n_total,)` for scalar output
- `(n_total, K)` for multi-output
- `(n_total, T, K)` for time-series multi-output

Validation and behavior:

- `Y.shape[0]` must match `sampling_result.n_total` (the unique-row count);
  the expanded layout is reconstructed internally.
- A 2D array is always interpreted as `(N, K)`, never `(N, T)`. For a
  time-series with one output, reshape to `(N, T, 1)`.
- Elementary effects are computed in unit-cube coordinates, so `mu_star` is
  directly comparable across parameters regardless of their physical ranges;
  use `MorrisResult.to_physical_units()` for derivative-scale values
  (uniform-marginal problems only).
- Trajectories containing any non-finite value (NaN/Inf) are dropped as whole
  blocks with a warning. Fewer than 2 remaining trajectories raise
  `ValueError`; fewer than 10 trigger a statistical-reliability warning.
- When `prenormalize=True`, `Y` is centered and scaled once per output slice
  over the expanded sample axis after non-finite cleanup.
- If `num_resamples > 0`, `key` is required or `ValueError` is raised.
  Bootstrap resampling is over trajectories, with replacement.
- Zero-variance output slices emit warnings.

Returns: [`MorrisResult`](#morrisresult)

<a id="morrisresult"></a>
### `MorrisResult` {#morrisresult}

Dataclass holding Morris elementary-effects screening measures.

```python
@dataclass
class MorrisResult:
    mu: Array
    mu_star: Array
    sigma: Array
    problem: Problem
    mu_conf: Array | None = None
    mu_star_conf: Array | None = None
    sigma_conf: Array | None = None
    space: Literal["unit", "physical"] = "unit"
```

| Field | Shape | Description |
| --- | --- | --- |
| `mu` | `(D,)` / `(K, D)` / `(T, K, D)` | Mean elementary effect; sign cancellation can mask non-monotonic influence. |
| `mu_star` | same as `mu` | Mean absolute elementary effect (Campolongo et al. 2007), the headline importance measure and a proxy for total-order ranking. |
| `sigma` | same as `mu` | Standard deviation of the elementary effects (ddof=1); large values relative to `mu_star` indicate nonlinearity or interactions. |
| `mu_conf`, `mu_star_conf`, `sigma_conf` | `(2, ...)` or `None` | Bootstrap lower and upper bounds. |
| `problem` | `Problem` | Problem definition used for the analysis. |
| `space` | `Literal["unit", "physical"]` | Coordinate space of the measures, `"unit"` (default) or `"physical"`. |

Shape contract:

| `Y` shape passed to `analyze_morris()` | `mu` / `mu_star` / `sigma` |
| --- | --- |
| `(N,)` | `(D,)` |
| `(N, K)` | `(K, D)` |
| `(N, T, K)` | `(T, K, D)` |

Morris does not produce Sobol indices; `mu_star` ranks parameters as a proxy
for total-order importance. Confidence interval arrays, when present, prepend
a leading dimension of 2 for `[lower, upper]`.

<a id="morrisresult-to_physical_units"></a>
#### `MorrisResult.to_physical_units()`

```python
physical = result.to_physical_units()
```

Returns a copy with all measures rescaled to physical input units
(`space == "physical"`).

Behavior:

- Unit-cube elementary effects divide the output change by a step in `[0, 1]`
  coordinates; dividing each measure by the parameter range `high - low`
  converts it to a per-physical-unit (derivative-scale) effect, comparable to
  DGSM's mean derivative.
- Raises `ValueError` if the result is already in physical units.
- Raises `ValueError` for problems with non-uniform (Gaussian) marginals:
  the inverse-CDF transform is nonlinear, so there is no single per-parameter
  range to rescale by (`problem.bounds` is `None`). Measures for such
  problems stay in grid coordinates.

<a id="morrisresult-to_dataset"></a>
#### `MorrisResult.to_dataset()`

```python
ds = result.to_dataset(time_coords=None)
```

Converts Morris results to a labeled `xarray.Dataset`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `time_coords` | `list \| np.ndarray \| None` | `None` | Coordinate values for the time dimension on 3D results. |

Behavior:

- Uses `problem.names` for `param` coordinates.
- Uses `problem.output_names` when available, otherwise `y0`, `y1`, and so on.
- The dataset contains `mu`, `mu_star`, and `sigma` variables, plus
  `*_lower` / `*_upper` variables when bootstrap CIs are present.
- Records the coordinate space in the `space` dataset attribute
  (`"unit"` or `"physical"`).

Minimal example:

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

sampling_result = gsax.sample_morris(PROBLEM, n_trajectories=50, seed=42)
Y = evaluate(jnp.asarray(sampling_result.samples))
result = gsax.analyze_morris(sampling_result, Y)

print(result.mu_star)  # (3,) — importance ranking
print(result.sigma)    # (3,) — nonlinearity / interactions
print(result.to_dataset())
```

Related links:

- [Morris Example](/examples/morris)
- [Methods](/guide/methods)
- [xarray Output](/examples/xarray)

---

## HSIC (Kernel-Based Sensitivity Analysis)

<a id="analyze-hsic"></a>
### `analyze_hsic()` {#analyze-hsic}

Compute HSIC (Hilbert-Schmidt Independence Criterion) sensitivity indices
from arbitrary (X, Y) sample pairs using Gaussian RBF kernels.

```python
def analyze_hsic(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_perms: int = 200,
    seed: int = 0,
    bandwidth: float | None = None,
    chunk_size: int | None = None,
    prenormalize: bool = False,
) -> HSICResult
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `problem` | `Problem` | required | Problem definition with D parameters. |
| `X` | `Array` | required | Input sample matrix `(N, D)` in physical units. |
| `Y` | `Array` | required | Model output `(N,)`, `(N, K)`, or `(N, T, K)`. |
| `n_perms` | `int` | `200` | Number of permutations for p-value computation. |
| `seed` | `int` | `0` | Random seed for permutation test reproducibility. |
| `bandwidth` | `float \| None` | `None` | Fixed kernel bandwidth. `None` uses the median heuristic. |
| `chunk_size` | `int \| None` | `None` | Block size for N×N kernel matrix computation. |
| `prenormalize` | `bool` | `False` | If `True`, standardize Y before analysis. |

Validation and behavior:

- `X` must be 2-D with `X.shape[1] == problem.num_vars`.
- `X` and `Y` must have the same number of rows.
- `n_perms` must be >= 1.
- Inputs are transformed to [0, 1] via their marginal CDF before kernel
  computation, ensuring comparable bandwidths across dimensions.
- Uses the biased V-statistic HSIC estimator with an efficient trace formula.
- P-values use the Phipson-Smyth correction: `(count + 1) / (n_perms + 1)`.

Returns: [`HSICResult`](#hsicresult)

Minimal example:

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM

X = gsax.sample_mc(PROBLEM, N=2048, seed=42)
Y = gsax.benchmarks.ishigami.evaluate(jnp.asarray(X))
result = gsax.analyze_hsic(PROBLEM, jnp.asarray(X), Y)

print(result.R2_HSIC)   # (3,) — normalized first-order indices
print(result.T_HSIC)     # (3,) — total-order indices
print(result.p_values)   # (3,) — permutation p-values
```

<a id="hsicresult"></a>
### `HSICResult` {#hsicresult}

Dataclass holding HSIC sensitivity analysis results.

```python
@dataclass
class HSICResult:
    R2_HSIC: Array
    T_HSIC: Array
    p_values: Array
    hsic_raw: Array
    problem: Problem
```

| Field | Shape | Description |
| --- | --- | --- |
| `R2_HSIC` | `(D,)` / `(K, D)` / `(T, K, D)` | Normalized first-order HSIC index (CKA normalization), in [0, 1]. |
| `T_HSIC` | `(D,)` / `(K, D)` / `(T, K, D)` | Total-order HSIC index via complement product kernels. |
| `p_values` | `(D,)` / `(K, D)` / `(T, K, D)` | Permutation p-values for R2_HSIC (Phipson-Smyth corrected). |
| `hsic_raw` | `(D,)` / `(K, D)` / `(T, K, D)` | Unnormalized HSIC(X_i, Y) values. |
| `problem` | `Problem` | Problem definition used for the analysis. |

Shape contract follows the same convention as other gsax methods:

| `Y` shape passed to `analyze_hsic()` | Index shapes |
| --- | --- |
| `(N,)` | `(D,)` |
| `(N, K)` | `(K, D)` |
| `(N, T, K)` | `(T, K, D)` |

<a id="hsicresult-to_dataset"></a>
#### `HSICResult.to_dataset()`

```python
ds = result.to_dataset(time_coords=None)
```

Converts HSIC results to a labeled `xarray.Dataset`.

Behavior:

- For scalar output, variables have dimension `(param,)`.
- For multi-output, variables have dimensions `(output, param)`.
- For time-series multi-output, variables have dimensions `(time, output, param)`.
- Uses `problem.names` for `param` coordinates.
- Dataset contains `R2_HSIC`, `T_HSIC`, `p_values`, and `hsic_raw` variables.

---

## PAWN Workflow

<a id="analyze-pawn"></a>
### `analyze_pawn()` {#analyze-pawn}

Compute PAWN sensitivity indices via KS distances between unconditional and
conditional output CDFs (Pianosi & Wagener, 2015).

```python
def analyze_pawn(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_bins: int = 10,
    statistic: Literal["median", "max", "mean"] = "median",
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    seed: int = 0,
    chunk_size: int = 2048,
) -> PAWNResult
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `problem` | `Problem` | required | Problem definition with D parameters. |
| `X` | `Array` | required | Input sample matrix `(N, D)`. |
| `Y` | `Array` | required | Model output `(N,)`, `(N, K)`, or `(N, T, K)`. |
| `n_bins` | `int` | `10` | Number of equal-width conditioning bins per input. |
| `statistic` | `Literal["median", "max", "mean"]` | `"median"` | Aggregation of KS values across bins. |
| `n_bootstrap` | `int` | `0` | Number of bootstrap resamples for confidence intervals. Set to 0 to skip. |
| `conf_level` | `float` | `0.95` | Confidence level for bootstrap intervals. |
| `seed` | `int` | `0` | Random seed for bootstrap resampling. |
| `chunk_size` | `int` | `2048` | Unused, kept for API consistency. |

Validation and behavior:

- `X.shape[1]` must match `problem.num_vars`.
- Inputs are transformed to `[0, 1]` via CDF mapping before binning.
- The unconditional CDF uses all Y values. Conditional CDFs use subsets
  where each input falls in a bin.
- Empty bins are skipped during aggregation.
- `statistic` must be one of `"median"`, `"max"`, or `"mean"`.

Returns: [`PAWNResult`](#pawnresult)

<a id="pawnresult"></a>
### `PAWNResult` {#pawnresult}

Dataclass holding PAWN sensitivity indices and optional bootstrap intervals.

```python
@dataclass
class PAWNResult:
    pawn: Array
    pawn_conf: Array | None
    problem: Problem
```

| Field | Shape | Description |
| --- | --- | --- |
| `pawn` | `(D,)` / `(K, D)` / `(T, K, D)` | PAWN sensitivity index per parameter. |
| `pawn_conf` | `(2, ...)` or `None` | Bootstrap confidence interval `[lower, upper]`, or `None` when `n_bootstrap=0`. |
| `problem` | `Problem` | Problem definition used for the analysis. |

<a id="pawnresult-to_dataset"></a>
#### `PAWNResult.to_dataset()`

```python
ds = result.to_dataset(time_coords=None)
```

Converts PAWN results to a labeled `xarray.Dataset`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `time_coords` | `list \| np.ndarray \| None` | `None` | Coordinate values for the time dimension on 3D results. |

Behavior:

- Uses `problem.names` for `param` coordinates.
- Uses `problem.output_names` when available, otherwise `y0`, `y1`, and so on.
- When `pawn_conf` is present, splits into `pawn_lower` and `pawn_upper` variables.

## Configuration

<a id="enable_compilation_cache"></a>
### `enable_compilation_cache()`

```python
cache_dir = gsax.enable_compilation_cache(
    path,
    *,
    min_compile_time_secs=1.0,
    min_entry_size_bytes=0,
)
```

Opt-in helper that enables JAX's persistent, on-disk compilation cache so compiled
kernels are reused across process restarts (parameter sweeps, CI, HPC batches).
Call it once, before your first `gsax.analyze*` call. Returns the absolute cache
directory path that was configured.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | `str \| Path` | — | On-disk cache directory. `~` is expanded and the result is resolved to an absolute path; created lazily by JAX on the first cache write. |
| `min_compile_time_secs` | `float` | `1.0` | Only cache executables whose compilation took at least this long, so trivial kernels are skipped. |
| `min_entry_size_bytes` | `int` | `0` | Minimum serialized executable size to cache (coerced to `int`). `0` allows a filesystem-specific default. |

**Warning:** the cache directory is effectively executable — anyone who can write
to it can make this process load and run arbitrary compiled code. Never point it
at a world-writable or shared, untrusted location.

See the [Configuration guide](/guide/configuration) for details, including
double-precision (`jax_enable_x64`) guidance.
