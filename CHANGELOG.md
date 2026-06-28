# Changelog

## 0.8.0

### New features

- Add `gsax.hsic` subpackage: HSIC (Hilbert-Schmidt Independence Criterion)
  kernel-based sensitivity analysis. Detects any form of statistical dependence
  (nonlinear, non-monotone, heteroscedastic) via Gaussian RBF kernels.
  - **R2-HSIC** (first-order): normalized kernel dependence index (Da Veiga 2015).
  - **Total HSIC** (total-effect): augmented product kernels per Larsen &
    Alexanderian (2026), capturing all interaction orders.
  - Permutation p-values with Phipson-Smyth correction.
  - Median heuristic bandwidth (upper-triangle, excluding diagonal zeros).
  - Supports all output shapes: `(N,)`, `(N, K)`, `(N, T, K)`.
  - `HSICResult.to_dataset()` for labeled xarray export.
- Given-data method: reuses any (X, Y) sample pairs, no special sampling
  design required. Works with correlated inputs.
- Optional `chunk_size` for N x N kernel matrix blocking on large problems.
- Input CDF transform ensures comparable bandwidths across dimensions.

## 0.7.0

### New features

- Add `gsax.pawn` subpackage: PAWN distribution-based sensitivity analysis
  (Pianosi & Wagener, 2015) using Kolmogorov-Smirnov distances between
  unconditional and conditional output CDFs. Includes `analyze_pawn()` and
  `PAWNResult` with configurable binning (`n_bins`), three aggregation
  statistics (median/max/mean), and bootstrap confidence intervals.
- PAWN supports all output shapes: scalar `(N,)`, multi-output `(N, K)`,
  and time-series `(N, T, K)`.
- CDF-space binning for proper handling of Gaussian and truncated inputs.
- `PAWNResult.to_dataset()` for labeled xarray export.

## 0.6.0

### Breaking changes

- **Package restructure:** Sobol analysis moved into `gsax.sobol` subpackage.
  HDMR moved from `gsax.expansions.hdmr` to `gsax.hdmr`. PCE moved from
  `gsax.expansions.pce` to `gsax.pce`. Top-level convenience re-exports are
  preserved (`gsax.analyze()`, `gsax.analyze_hdmr()`, `gsax.analyze_pce()`,
  etc.), so `import gsax; gsax.analyze(...)` still works unchanged.
- `Problem._input_specs` is now the public property `Problem.input_specs`.

### New features

- Add `gsax.pce` subpackage: Polynomial Chaos Expansion sensitivity analysis
  with analytical Sobol indices from expansion coefficients (Sudret, 2008).
  Includes `analyze_pce()`, `emulate_pce()`, and `PCEResult` with LOO-CV RMSE.
- PCE supports mixed uniform and Gaussian inputs via the Wiener-Askey scheme
  (Legendre for uniform, Hermite for Gaussian).

### Preferred import style

```python
# Subpackage imports (preferred)
from gsax.sobol import analyze
from gsax.hdmr import analyze, emulate
from gsax.pce import analyze, emulate

# Top-level convenience (still works)
import gsax
gsax.analyze(...)
gsax.analyze_hdmr(...)
gsax.analyze_pce(...)
```

## 0.4.0

- Add `SamplingResult.save()` for serializing samples + metadata to disk
- Add `gsax.load()` for reconstructing `SamplingResult` from saved files
- Supported sample formats: csv, txt, xlsx, parquet, pkl
- Storage-optimized: identity mappings skip the .npz sidecar
