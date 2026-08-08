# Problem

The `Problem` dataclass names your input parameters and gives each a marginal
distribution. Every jaxgsa sampling and analysis function takes one.

Marginals can be uniform, Gaussian (optionally truncated), or categorical.

## Categorical parameters

Declare a categorical parameter as
`{"dist": "categorical", "probs": [p0, ..., pL-1], "labels": [...]}`. Samples
carry its integer level codes `0 .. L-1` as floats, never physical values. The
optional `labels` are reporting metadata.

- `problem.categorical_labels` — the label tuple for each categorical
  parameter, keyed by parameter name.
- `problem.has_categorical_inputs` — whether the problem declares any
  categorical parameter.

See [Categorical Inputs](/examples/categorical-inputs).

## Correlation

A Gaussian-copula `correlation` matrix declares dependence between parameters.
It is optional. Pass `correlation=` to the constructor or to `from_dict`,
together with `correlation_kind="latent"` or `correlation_kind="spearman"`. To
attach a matrix to an existing (frozen) problem, call
`problem.with_correlation(R)`.

- `problem.correlation` — the validated latent correlation matrix.
- `problem.has_correlated_inputs` — whether a non-identity matrix is set.

A correlation entry that touches a categorical parameter raises. Polychoric
coupling is future work. Drop the entry, or drop the categorical declaration,
to build the problem. See [Correlated Inputs](/examples/correlated-inputs).

## Reference

The canonical API reference lives at [API Reference](/api/).

Jump directly to:

- [`jaxgsa.Problem`](/api/#foundational-types)
- [`jaxgsa.Problem.from_dict()`](/api/#foundational-types)

Related docs:

- [Getting Started](/guide/getting-started)
- [Advanced Workflow](/examples/advanced-workflow)
- [Correlated Inputs](/examples/correlated-inputs)
