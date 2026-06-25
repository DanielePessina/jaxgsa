# Changelog

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
