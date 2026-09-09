# Irregular output grids

When output channels do not share a time grid, pass a ragged `Y`. Each channel
can have its own time values and its own number of samples along time; jaxgsa
automatically buckets the channels, runs the selected analysis on each
channel's regular `(N, T_k, 1)` array, and keeps the channels separate.

This is different from a missing-value mask. Automatic bucketing does not pad
channels onto a common grid and does not invent values at times that were not
measured.

## Pass one `(times, values)` pair per channel

```python
import jax.numpy as jnp
import jaxgsa

problem = jaxgsa.Problem.from_dict(
    {
        "decay": (0.1, 1.0),
        "offset": (-1.0, 1.0),
        "amplitude": (0.5, 2.0),
    },
    output_names=("concentration", "diameter"),
)

t_concentration = jnp.array([0.0, 0.5, 1.5, 3.0, 6.0])
t_diameter = jnp.array([0.0, 1.0, 4.0])

design = jaxgsa.sobol.sample(
    problem,
    n_samples=1024,
    calc_second_order=False,
    seed=0,
    verbose=False,
)
X = jnp.asarray(design.samples)

concentration = jnp.exp(-X[:, 0, None] * t_concentration[None, :]) + X[:, 1, None]
diameter = X[:, 2, None] * jnp.cos(t_diameter[None, :]) + X[:, 1, None]

Y = [
    (t_concentration, concentration),  # values: (N, 5)
    (t_diameter, diameter),             # values: (N, 3)
]

result = jaxgsa.sobol.analyze(design, Y, verbose=False)
```

The ragged form is a list or tuple of `(times, values)` pairs. `times` must be
one-dimensional, finite, and unique. `values` may be `(N, T_k)` or `(N,)` for a
single-time channel. jaxgsa sorts each time grid in ascending order and carries
the corresponding value columns with it.

When `problem.output_names` is declared, a list follows that order. A dict is
also accepted and must contain exactly those keys; its insertion order does not
override the declared output order:

```python
Y_named = {
    "diameter": (t_diameter, diameter),
    "concentration": (t_concentration, concentration),
}
result = jaxgsa.sobol.analyze(design, Y_named, verbose=False)
```

Without `output_names`, channels are named `y0`, `y1`, and so on. Declaring
names is recommended because those names are carried into the result and the
xarray export.

## Read the per-channel results

The returned object has one ordinary method result in `result.channels` for
each output channel. The method's normal shape contract still applies inside a
channel; for example, the Sobol `S1` arrays below retain a singleton output
axis:

```python
result.channels["concentration"].S1.shape  # (5, 1, 3)
result.channels["diameter"].ST.shape       # (3, 1, 3)
result.times["concentration"]              # [0.0, 0.5, 1.5, 3.0, 6.0]
```

`result.to_dataset()` keeps the grids separate rather than padding them:

```python
ds = result.to_dataset()
ds["concentration_S1"].dims  # ("time_concentration", "param")
ds["diameter_ST"].dims       # ("time_diameter", "param")
```

Every channel goes through the selected method's regular validation, estimator,
fit, and optional confidence interval path. If a `key` is supplied for a
bootstrap or permutation analysis, jaxgsa folds it per channel so the channels
do not reuse identical random draws.

## Supported methods and limits

Automatic bucketing is available on these twelve `analyze` entry points:

`borgonovo`, `efast`, `hdmr`, `hsic`, `kucherenko`, `morris`,
`optimal_transport`, `pawn`, `pce`, `shapley`, `sobol`, and `vkoga`.

`dgsm` is the exception. It requires a Jacobian with a fixed output layout, so
the ragged form is rejected; use one of the methods above or analyze each
channel with its own derivative array. There is no mask mode: padding a common
grid would add computation and would require a separate missing-data contract,
neither of which is needed by the per-channel estimators.

If all channels share one grid, the regular `(N, T, K)` array form remains the
shortest path and returns the ordinary method result. Use the ragged form when
the grids genuinely differ or when you want to preserve each channel's own
time coordinate.
