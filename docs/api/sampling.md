# Sampling

Use `gsax.sobol.sample(...)` for Saltelli designs and
`gsax.sampling.monte_carlo(...)` for ordinary independent draws.

`gsax.sobol.SobolSamples` provides:

- `samples` and `sample_ids`;
- Saltelli reconstruction metadata;
- `downsample(base_n, Y=None)`;
- `save(path)` and `load(path)` using one NPZ file.

See [Save and Reload Samples](/examples/save-load) and the
[API overview](/api/).
