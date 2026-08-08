# Problem

The `Problem` dataclass names your input parameters and gives each a marginal
distribution. Every jaxgsa sampling and analysis function takes one.

Marginals can be uniform, Gaussian (optionally truncated), or categorical.
A categorical parameter is declared as
`{"dist": "categorical", "probs": [p0, ..., pL-1], "labels": [...]}`.
Samples carry its integer level codes `0 .. L-1` as floats — codes, never
physical values. The optional `labels` are reporting metadata:
`problem.categorical_labels` maps each categorical parameter name to its
label tuple, and `problem.has_categorical_inputs` reports their presence.
See [Categorical Inputs](/examples/categorical-inputs).

Optionally, a Gaussian-copula `correlation` matrix declares dependence
between parameters: pass `correlation=` (with `correlation_kind="latent"`
or `"spearman"`) to the constructor or `from_dict`, or attach one to an
existing (frozen) problem with `problem.with_correlation(R)`. The
`problem.correlation` property returns the validated latent matrix and
`problem.has_correlated_inputs` reports whether a non-identity matrix is
set. A correlation entry touching a categorical parameter raises
(polychoric coupling is future work). See
[Correlated Inputs](/examples/correlated-inputs).

The canonical API reference now lives at [API Reference](/api/).

Jump directly to:

- [`jaxgsa.Problem`](/api/#foundational-types)
- [`jaxgsa.Problem.from_dict()`](/api/#foundational-types)

Related docs:

- [Getting Started](/guide/getting-started)
- [Advanced Workflow](/examples/advanced-workflow)
- [Correlated Inputs](/examples/correlated-inputs)
