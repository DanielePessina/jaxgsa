# Sampling

Use `jaxgsa.sobol.sample(...)` for Saltelli designs and
`jaxgsa.sampling.monte_carlo(...)` for ordinary pseudo-random draws.

## Correlated inputs

`jaxgsa.sampling` also covers correlated inputs. See
[Correlated Inputs](/examples/correlated-inputs).

- `monte_carlo(problem, n, seed=...)` — draw `n` rows, honoring
  `problem.correlation` through a Gaussian copula. An independent problem
  keeps the plain pseudo-random path bit-for-bit.
- `correlate(X, problem, seed=...)` — retrofit the declared correlation onto
  an existing sample by rank re-pairing. Each column is permuted, never
  altered.
- `fit_correlation(problem, X)` — estimate the latent correlation matrix from
  observed data, for use with `problem.with_correlation(...)`.
- `correlation_from_covariance(cov)` — rescale a published covariance matrix
  to the correlation form the API accepts.

One design builder reads `problem.correlation` rather than refusing it.
`jaxgsa.kucherenko.sample(problem, n, seed=...)` places its blocks with the
declared copula's conditionals, so the indices it feeds are valid under
dependence. See the [Kucherenko page](/api/kucherenko). To analyze correlated
data you already have, see [VKOGA](/api/vkoga).

## SobolSamples

`jaxgsa.sobol.SobolSamples` provides:

- `samples` — the input rows to evaluate.
- `sample_ids` — the identifier of each row.
- `n_runs` — the number of unique rows to evaluate, one model run per row.
- `n_expanded` — the design size before deduplication.
- Saltelli reconstruction metadata.
- `downsample(base_n, Y=None)` — a prefix-nested smaller design.
- `save(path)` and `load(path)` — persistence in one NPZ file.

`jaxgsa.morris.MorrisSamples` shares the same vocabulary (`n_runs`,
`n_expanded`) and the same single-NPZ `save(path)` and `load(path)` format.

See [Save and Reload Samples](/examples/save-load) and the
[API overview](/api/).
