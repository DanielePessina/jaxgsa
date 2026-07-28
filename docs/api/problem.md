# Problem

The `Problem` dataclass names your input parameters and gives each a marginal
distribution. Every jaxgsa sampling and analysis function takes one.

Optionally, a Gaussian-copula `correlation` matrix declares dependence
between parameters: pass `correlation=` (with `correlation_kind="latent"`
or `"spearman"`) to the constructor or `from_dict`, or attach one to an
existing (frozen) problem with `problem.with_correlation(R)`. The
`problem.correlation` property returns the validated latent matrix and
`problem.has_correlated_inputs` reports whether a non-identity matrix is
set. See [Correlated Inputs](/examples/correlated-inputs).

The canonical API reference now lives at [API Reference](/api/).

Jump directly to:

- [`jaxgsa.Problem`](/api/#foundational-types)
- [`jaxgsa.Problem.from_dict()`](/api/#foundational-types)

Related docs:

- [Getting Started](/guide/getting-started)
- [Advanced Workflow](/examples/advanced-workflow)
- [Correlated Inputs](/examples/correlated-inputs)
