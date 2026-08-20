# Problem

`jaxgsa.Problem` names your input parameters, in the column order your model
expects them, and gives each one a marginal distribution. Every jaxgsa
sampling and analysis function takes one. It is a frozen dataclass, so one
`Problem` is safe to share across analyses and across threads.

## Two constructors

```python
Problem(
    names: tuple[str, ...],
    bounds: tuple[tuple[float, float], ...],
    output_names: tuple[str, ...] | None = None,
    correlation: ArrayLike | None = None,
    correlation_type: Literal["latent", "spearman"] = "latent",
)
```

The direct constructor takes uniform marginals only, as finite `(low, high)`
pairs. It is the short form for the common case.

```python
Problem.from_dict(
    params: dict[str, InputSpecValue],
    output_names: tuple[str, ...] | None = None,
    *,
    truncate_gaussians: float | None = None,
    correlation: ArrayLike | None = None,
    correlation_type: Literal["latent", "spearman"] = "latent",
)
```

`from_dict` takes any mix of uniform, Gaussian, and categorical marginals.
Parameter order follows the dict's insertion order, so the dict order is the
model's column order.

Each value in `params` may be a `(low, high)` tuple, one of the spec
dataclasses (`UniformSpec`, `GaussianSpec`, `CategoricalSpec`), or the
matching plain dict (`UniformInputSpec`, `GaussianInputSpec`,
`CategoricalInputSpec`). The dict form is JSON-expressible, which is what
lets a design save and reload itself.

```python
import jaxgsa

problem = jaxgsa.Problem.from_dict(
    {
        "x1": (0.0, 1.0),
        "x2": jaxgsa.GaussianSpec(mean=0.0, variance=4.0, high=3.0),
        "x3": {"dist": "categorical", "probs": [0.5, 0.5], "labels": ["off", "on"]},
    }
)
problem.names        # ('x1', 'x2', 'x3')
problem.num_vars     # 3
problem.bounds       # None
problem.input_specs
```

```
(UniformSpec(low=0.0, high=1.0),
 GaussianSpec(mean=0.0, variance=4.0, low=None, high=3.0),
 CategoricalSpec(probs=(0.5, 0.5), labels=('off', 'on')))
```

`bounds` came back as `None` here on purpose. It holds `(low, high)` pairs
only while every marginal is uniform. An unbounded Gaussian has no finite
support, and a categorical marginal carries level codes rather than a range,
so as soon as one appears there is no honest pair to report. Read
`input_specs` instead and branch on the spec type with `isinstance`.

`output_names` labels the model's outputs. It is used to name the `output`
coordinate in `to_dataset()` exports, and its length must equal `K`.

### truncate_gaussians

`from_dict(..., truncate_gaussians=q)` gives every Gaussian marginal an
explicit `low` and `high` at its own `q` and `1 - q` quantiles. `q` must be in
`(0, 0.5)`. A side the spec already declares is kept as written, so only open
sides are filled.

This is one switch instead of editing every spec, and the bounds it writes are
real. `jaxgsa.morris.sample` will not squash such a marginal a second time,
and `SobolSamples.to_morris()` stops warning about unbounded tails. Leave it
at `None` and Gaussians stay unbounded.

## Parameter names are validated

Names must be strings and must be unique. Both rules are enforced at
construction, on every surface, because a bad name only fails much later and
much less clearly: `Theta` mappings, xarray `param` coordinates, and saved
design metadata are all keyed by name.

```python
jaxgsa.Problem(names=("a", "a"), bounds=((0.0, 1.0), (0.0, 1.0)))
```

```
ValueError: parameter names must be unique, but ['a'] appear more than once.
Rename the duplicated parameters so every name is distinct.
```

```python
jaxgsa.Problem(names=("a", 2), bounds=((0.0, 1.0), (0.0, 1.0)))
```

```
ValueError: parameter names must be strings, got [2]. Convert each name with
str(...) when declaring the problem.
```

## Categorical parameters

Declare a categorical parameter as
`{"dist": "categorical", "probs": [p0, ..., pL-1], "labels": [...]}`, or with
`CategoricalSpec(probs, labels)`. `probs` needs at least two positive finite
entries and must already sum to 1 within `1e-3`; a spec that does not is
rejected rather than rescaled, because a probability vector that misses by
more than float noise is a mistake, not a scaling convention. `labels` is
optional, must be unique, and is reporting metadata only. Omit it and the
levels are labelled `"0"`, `"1"`, and so on.

Samples carry the integer level codes `0 .. L-1` as floats, never physical
values. That is what makes the codes safe to pass through the same `(N, D)`
float matrix as everything else.

- `problem.categorical_labels` — `{'x3': ('off', 'on')}`, the label tuple per
  categorical parameter, keyed by name.
- `problem.has_categorical_inputs` — `True` if any parameter is categorical.

Four routes accept categorical inputs: `optimal_transport`, `borgonovo`,
`pawn`, and the Saltelli-based `sobol` pipeline. Everything else raises a
`ValueError`, because a derivative, a level spacing, or a Fourier sweep along
an unordered code has no meaning. See
[Categorical Inputs](/examples/categorical-inputs).

## Correlation

A Gaussian-copula `correlation` matrix declares dependence between parameters.
Every marginal is kept exactly as written; only the joint pairing changes.

The same two keywords appear on all three surfaces, spelled the same way:

```python
Problem(names, bounds, correlation=R, correlation_type="latent")
Problem.from_dict(params, correlation=R, correlation_type="latent")
problem.with_correlation(R, correlation_type="latent")   # keyword-only here
```

`correlation_type` says which scale your matrix `R` is expressed on.

- `"latent"` (the default) means `R` is the Pearson correlation of the
  copula's latent normals. It is stored as given.
- `"spearman"` means `R` is a rank correlation. jaxgsa converts it with
  `2 sin(pi rho_s / 6)` before storing.

The distinction is not cosmetic. A published Spearman rank correlation of 0.8
becomes a latent 0.8135, and feeding it in as `"latent"` quietly understates
the dependence:

```python
import numpy as np

R = np.array([[1.0, 0.8], [0.8, 1.0]])
p = jaxgsa.Problem(("a", "b"), ((0.0, 1.0), (0.0, 1.0)))

p.with_correlation(R, correlation_type="spearman").correlation
# array([[1.        , 0.81348899],
#        [0.81348899, 1.        ]])

p.with_correlation(R).correlation          # correlation_type="latent"
# array([[1. , 0.8],
#        [0.8, 1. ]])
```

`problem.correlation` always returns the validated latent matrix, whatever
scale you declared, as a fresh NumPy array. `problem.has_correlated_inputs` is
`True` when a non-identity matrix is set.

`Problem` is frozen, so `with_correlation` returns a copy rather than mutating.
Pass `None` to drop a matrix. The fit-then-attach workflow reads:

```python
R = jaxgsa.sampling.fit_correlation(problem, X_observed)
problem = problem.with_correlation(R)
```

### What validation does to your matrix

The matrix is checked on entry, not at first use. A slightly
non-positive-definite matrix, the usual result of estimating one from data, is
repaired to the nearest positive-definite correlation matrix and the repair is
reported with a `JaxgsaWarning`. A matrix whose repair would have to move any
entry by 0.05 or more is rejected with a `ValueError` instead. The threshold
separates float noise from a matrix that does not describe a joint
distribution.

`jaxgsa.sampling.correlation_from_covariance(cov)` rescales a published
covariance matrix into the correlation form these arguments accept.

### Correlation and categorical parameters do not mix

```python
problem.with_correlation(np.array([[1, 0, 0.5], [0, 1, 0], [0.5, 0, 1.0]]))
```

```
ValueError: problem.correlation couples categorical parameter 'x3', but the
Gaussian copula does not define a coupling for an unordered marginal
(polychoric coupling is future work). Keep the categorical parameter's row and
column at identity, or drop the matrix with problem.with_correlation(None).
```

Keep the categorical row and column at identity and the rest of the matrix is
accepted. The check runs on the matrix as you declared it, before the
positive-definiteness repair, so repair noise is never read as a declared
coupling.

## Properties

| Property | Returns |
| --- | --- |
| `names` | parameter names in model-input order |
| `bounds` | `(low, high)` per parameter, or `None` if any marginal is not uniform |
| `output_names` | output labels, or `None` |
| `num_vars` | `D`, the number of parameters |
| `input_specs` | the canonical spec dataclass per parameter |
| `correlation` | the validated latent matrix, or `None` |
| `has_correlated_inputs` | `True` if a non-identity matrix is set |
| `has_non_uniform_inputs` | `True` if any marginal is not uniform |
| `has_categorical_inputs` | `True` if any marginal is categorical |
| `categorical_labels` | `{name: labels}` for the categorical parameters |

Related docs:

- [API reference](/api/)
- [Getting Started](/guide/getting-started)
- [Screen first, then quantify](/examples/advanced-workflow)
- [Correlated Inputs](/examples/correlated-inputs)
