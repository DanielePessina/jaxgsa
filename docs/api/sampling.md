# Sampling

Use `jaxgsa.sobol.sample(...)` for Saltelli designs and
`jaxgsa.sampling.monte_carlo(...)` for ordinary independent draws.

`jaxgsa.sobol.SobolSamples` provides:

- `samples` and `sample_ids`;
- `n_runs` (unique rows to evaluate, one model run per row) and `n_expanded`
  (pre-deduplication design size);
- Saltelli reconstruction metadata;
- `downsample(base_n, Y=None)`;
- `save(path)` and `load(path)` using one NPZ file.

`jaxgsa.morris.MorrisSamples` shares the same vocabulary (`n_runs`,
`n_expanded`) and the same single-NPZ `save(path)` / `load(path)` format.

See [Save and Reload Samples](/examples/save-load) and the
[API overview](/api/).
