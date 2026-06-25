# API Reference

This is the canonical reference for the exported `gsax` surface. The package has
three workflows:

- Sobol: `sample()` -> `analyze()`
- RS-HDMR: `analyze_hdmr()` -> `emulate_hdmr()`
- PCE: `analyze_pce()` -> `emulate_pce()`

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

You can import from the subpackages directly:

```python
from gsax.sobol import analyze
from gsax.hdmr import analyze as analyze_hdmr, emulate as emulate_hdmr
from gsax.pce import analyze as analyze_pce, emulate as emulate_pce
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
- [`load`](#load)
- [`analyze`](#analyze)
- [`SAResult`](#saresult)
- [`analyze_hdmr`](#analyze_hdmr)
- [`emulate_hdmr`](#emulate_hdmr)
- [`HDMRResult`](#hdmrresult)
- [`HDMREmulator`](#hdmremulator)
- [`analyze_pce`](#analyze_pce)
- [`emulate_pce`](#emulate_pce)
- [`PCEResult`](#pceresult)

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
- `problem` must have finite uniform bounds; non-uniform specs are not
  supported by HDMR in this version.
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
