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
- `unit` — the same design in the unit cube, before the input distributions
  are applied.
- `transform(theta=None)` — the design in physical units, for the
  distribution parameters you pass.

`jaxgsa.morris.MorrisSamples` shares the same vocabulary (`n_runs`,
`n_expanded`) and the same single-NPZ `save(path)` and `load(path)` format.

### Reusing one design under different input ranges

`unit` holds the quasi-random points before any distribution is applied, so
they do not depend on the input distributions at all. `transform` applies a
set of distribution parameters to them. Pass `None` to get the design the
problem itself describes.

That lets you evaluate the effect of your assumed input ranges without
drawing a new design:

```python
samples = jaxgsa.sobol.sample(problem, n_samples=8192, seed=0)

narrow = samples.transform({"x1": {"low": 0.0, "high": 1.0}, ...})
wide   = samples.transform({"x1": {"low": -1.0, "high": 2.0}, ...})
```

`theta` is a dictionary of dictionaries, keyed by parameter name and then by
the field names of that parameter's distribution: `low` and `high` for a
uniform, `mean` and `variance` for a Gaussian, and all four for a truncated
Gaussian.

`transform` is written in JAX, so it is differentiable with respect to
`theta`. See [Analyze (Sobol)](/api/analyze) for how to get the derivative of
an index.

`transform` raises for a problem with categorical parameters. A categorical
inverse CDF is a step function, so `unit` and `samples` do not have the same
number of rows, and a derivative through it has no meaning.

See [Save and Reload Samples](/examples/save-load) and the
[API overview](/api/).
