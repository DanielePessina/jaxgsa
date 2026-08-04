# Non-Uniform Inputs

Use this page when your Sobol analysis needs independent inputs with different
marginals. `jaxgsa.Problem.from_dict(...)` accepts the legacy `(low, high)`
uniform shorthand plus tagged `TypedDict` specs for Gaussian and truncated
Gaussian inputs.

## Define a mixed-input problem

```python
import jax.numpy as jnp
import numpy as np
from scipy.stats import truncnorm

import jaxgsa

problem = jaxgsa.Problem.from_dict(
    {
        "uniform": (0.0, 2.0),
        "gaussian": {
            "dist": "gaussian",
            "mean": 1.0,
            "variance": 2.25,
        },
        "truncated": {
            "dist": "gaussian",
            "mean": 0.5,
            "variance": 1.44,
            "low": -0.5,
            "high": 1.0,
        },
    },
    output_names=("response",),
)
```

Rules for Gaussian specs:

- `mean` and `variance` describe the parent Gaussian before truncation.
- `low` and `high` are optional and may be used independently.
- When either truncation bound is present, `jaxgsa.sobol.sample()` uses a true
  truncated normal transform, not hard clipping.
- An unbounded Gaussian is still bounded in practice, at the library's own
  support clip of +/-7.0345 standard deviations. Inverse-CDF sampling has to
  keep the unit coordinate off 0 and 1, and that clip is the result.
- `Problem.from_dict(params, truncate_gaussians=q)` fills `low` and `high` into
  every Gaussian that does not already declare them, at that marginal's own `q`
  and `1 - q` quantiles. Use it to fix one bounded input model that every
  method shares. Sides you declared yourself are kept as written.

## Run Sobol on a single-timepoint linear model

For a linear model

$$
y = \sum_i a_i x_i
$$

the analytical first-order and total-order Sobol indices are identical:

$$
S_i = \frac{a_i^2 \operatorname{Var}(X_i)}{\sum_j a_j^2 \operatorname{Var}(X_j)}
$$

The snippet below keeps the output layout as `(N, 1, 1)` so you can compare one
timepoint/output slice directly.

```python
coeffs = jnp.array([1.5, -0.75, 2.0])

sampling_result = jaxgsa.sobol.sample(
    problem,
    n_samples=8192,
    calc_second_order=False,
    seed=101,
)

X = jnp.asarray(sampling_result.samples)
Y = (X @ coeffs)[:, None, None]  # (N, 1, 1)

result = jaxgsa.sobol.analyze(sampling_result, Y)

std = np.sqrt(1.44)
a = (-0.5 - 0.5) / std
b = (1.0 - 0.5) / std

variances = np.array(
    [
        (2.0 - 0.0) ** 2 / 12.0,
        2.25,
        truncnorm.var(a, b, loc=0.5, scale=std),
    ]
)
weights = np.square(np.asarray(coeffs)) * variances
analytical = weights / weights.sum()

print("Computed S1:", np.asarray(result.S1[0, 0]))
print("Computed ST:", np.asarray(result.ST[0, 0]))
print("Analytical:", analytical)
```

Expected behavior:

- `result.S1[0, 0, :]` and `result.ST[0, 0, :]` should closely match the
  analytical variance ratios.
- `result.S2` is `None` because `calc_second_order=False`.

## Practical notes

- `problem.bounds` is `None` as soon as any Gaussian spec is present. This is
  expected and signals that the problem is not finite-bounds-only anymore.
- Save/load still works for mixed problems. The JSON metadata records the
  declared input specs so `SobolSamples.load()` reconstructs the same marginals.
- `jaxgsa.hdmr.analyze()` supports non-uniform inputs (Gaussian, truncated Gaussian)
  via CDF mapping to `[0, 1]` before surrogate fitting.
- `jaxgsa.pce.analyze()` picks the polynomial basis from how tight a truncation
  is. A narrow truncation is a different measure, so it goes through the
  truncated CDF and onto Legendre. A wide one — every declared bound at least 5
  standard deviations out — keeps the Hermite basis, because forcing Legendre
  there makes the fit visibly worse (on Oakley-O'Hagan at order 3, the LOO RMSE
  went from 0.93 to 1.70 and the largest S1 error from 0.0023 to 0.0054). Above
  order 7 Legendre is used even for a wide truncation, because the Hermite Gram
  defect against the truncated measure grows with degree.

## See also

- [Basic Example](/examples/basic) for the smallest uniform-only Sobol run.
- [Save and Reload Samples](/examples/save-load) if you want to persist a mixed
  design and analyze it later.
- [API Reference](/api/) for the exact `TypedDict` shapes and `Problem.bounds`
  semantics.
