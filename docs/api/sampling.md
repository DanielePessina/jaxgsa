# Sampling

Use `jaxgsa.sobol.sample(...)` for Saltelli designs and
`jaxgsa.sampling.monte_carlo(...)` for ordinary pseudo-random draws.

`jaxgsa.sampling` also covers correlated inputs
(see [Correlated Inputs](/examples/correlated-inputs)):

- `monte_carlo(problem, n, seed=...)` transparently honors
  `problem.correlation` via a Gaussian copula (independent problems keep the
  plain pseudo-random path bit-for-bit);
- `correlate(X, problem, seed=...)` retrofits the declared correlation onto
  an existing sample by rank re-pairing (each column is permuted, never
  altered);
- `fit_correlation(problem, X)` estimates the latent correlation matrix from
  observed data, for `problem.with_correlation(...)`;
- `correlation_from_covariance(cov)` rescales a published covariance matrix
  to the correlation form the API accepts.

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
