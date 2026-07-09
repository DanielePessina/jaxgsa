# xarray Labeled Output

`to_dataset()` turns `gsax` results into labeled `xarray.Dataset` objects so
you can select by parameter name, output name, time coordinate, and term label
instead of raw integer axes.

## Self-contained setup

```python
import jax
import jax.numpy as jnp
import numpy as np
import gsax

problem = gsax.Problem.from_dict(
    {
        "amplitude": (0.5, 2.0),
        "frequency": (1.0, 5.0),
        "damping": (0.01, 0.5),
    },
    output_names=("displacement", "velocity"),
)

time_values = np.linspace(0.0, 5.0, 30)


def model(X):
    amp = X[:, 0, None]
    freq = X[:, 1, None]
    damping = X[:, 2, None]
    tt = jnp.asarray(time_values)[None, :]

    displacement = amp * jnp.sin(2 * jnp.pi * freq * tt) * jnp.exp(-damping * tt)
    velocity = amp * jnp.cos(2 * jnp.pi * freq * tt) * jnp.exp(-damping * tt)
    return jnp.stack([displacement, velocity], axis=-1)  # (N, T, K)


sampling_result = gsax.sample(problem, n_samples=2048, seed=42)
X_sobol = jnp.asarray(sampling_result.samples)
Y_sobol = model(X_sobol)

sobol = gsax.analyze(
    sampling_result,
    Y_sobol,
    num_resamples=100,
    key=jax.random.key(0),
)

bounds = jnp.array(problem.bounds)
X_hdmr = jax.random.uniform(
    jax.random.key(1),
    (1500, problem.num_vars),
    minval=bounds[:, 0],
    maxval=bounds[:, 1],
)
Y_hdmr = model(X_hdmr)
hdmr = gsax.analyze_hdmr(problem, X_hdmr, Y_hdmr, maxorder=2)

ds_sobol = sobol.to_dataset(time_coords=time_values)
ds_hdmr = hdmr.to_dataset(time_coords=time_values)
```

## Sobol dataset

```python
print(ds_sobol)
# <xarray.Dataset>
# Dimensions:  (time: 30, output: 2, param: 3, param_i: 3, param_j: 3)

print(ds_sobol.S1.sel(param="amplitude"))
print(ds_sobol.ST.sel(output="velocity"))
print(
    ds_sobol.S2.sel(
        time=time_values[10],
        output="displacement",
        param_i="amplitude",
        param_j="frequency",
        method="nearest",
    )
)
```

## Confidence intervals

Bootstrap intervals are split into separate dataset variables:

```python
print(ds_sobol.S1_lower.sel(param="amplitude"))
print(ds_sobol.S1_upper.sel(param="amplitude"))
print(ds_sobol.ST_lower.sel(output="velocity"))
```

## HDMR dataset

`HDMRResult.to_dataset()` uses `term` for `Sa`, `Sb`, `S`, and `select`, while
`ST` stays indexed by `param`.

```python
print(ds_hdmr.ST.sel(param="amplitude"))
print(ds_hdmr.Sa.sel(term="amplitude/frequency"))
print(ds_hdmr.rmse.sel(output="displacement"))
```

## Practical caveats

- If `problem.output_names` is omitted, outputs are labeled `y0`, `y1`, and so
  on.
- Without `time_coords`, `to_dataset()` uses integer time indices.
- Sobol `S2` becomes dataset variables with `param_i` and `param_j`.
- `select` and `rmse` only appear on the HDMR dataset when the result contains
  those fields.

## See also

- [Multi-Output & Time-Series](/examples/multi-output) for the shape rules that
  feed into `to_dataset()`.
- [RS-HDMR Example](/examples/hdmr) for the surrogate workflow before export.
- [Advanced Workflow](/examples/advanced-workflow) for one page that uses both
  Sobol and HDMR datasets on the same custom model.
