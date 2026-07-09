# Advanced Workflow

This page adapts the repo's `development.py` example into a single docs
workflow. It uses a custom `Problem`, a time-series multi-output model, Sobol
analysis, HDMR analysis, HDMR emulation, and labeled `xarray` export.

## Define the problem and model

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
        "phase": (0.0, 2 * np.pi),
        "offset": (-1.0, 1.0),
    },
    output_names=("displacement", "velocity", "envelope"),
)

time_values = np.linspace(0.0, 5.0, 50)


def model(X):
    amp = X[:, 0, None]
    freq = X[:, 1, None]
    damping = X[:, 2, None]
    phase = X[:, 3, None]
    offset = X[:, 4, None]
    tt = jnp.asarray(time_values)[None, :]

    displacement = (
        amp * jnp.sin(2 * jnp.pi * freq * tt + phase) * jnp.exp(-damping * tt)
        + offset
    )
    velocity = amp * jnp.cos(2 * jnp.pi * freq * tt + phase) * jnp.exp(-damping * tt)
    envelope = amp * jnp.exp(-damping * tt)

    return jnp.stack([displacement, velocity, envelope], axis=-1)  # (N, T, K=3)
```

## Run Sobol analysis

```python
sampling_result = gsax.sample(
    problem,
    n_samples=2048,
    seed=42,
    calc_second_order=True,
)

X_sobol = jnp.asarray(sampling_result.samples)
Y_sobol = model(X_sobol)

sobol = gsax.analyze(sampling_result, Y_sobol)

print("Sobol S1 shape:", sobol.S1.shape)  # (T, K, D)
print("Sobol ST shape:", sobol.ST.shape)  # (T, K, D)
print("Sobol S2 shape:", sobol.S2.shape)  # (T, K, D, D)
```

## Run HDMR on arbitrary samples

```python
key = jax.random.PRNGKey(42)
bounds = jnp.array(problem.bounds)
X_hdmr = jax.random.uniform(
    key,
    (2000, problem.num_vars),
    minval=bounds[:, 0],
    maxval=bounds[:, 1],
)
Y_hdmr = model(X_hdmr)

hdmr = gsax.analyze_hdmr(problem, X_hdmr, Y_hdmr, maxorder=2)

print("HDMR S1 shape:", hdmr.S1.shape)  # (T, K, D)
print("HDMR ST shape:", hdmr.ST.shape)  # (T, K, D)
print("HDMR RMSE:", hdmr.rmse)
```

## Predict with the HDMR emulator

```python
Y_pred = gsax.emulate_hdmr(hdmr, X_hdmr[:5])
print("Prediction shape:", Y_pred.shape)  # (5, T, K)
print("Max absolute residual:", jnp.abs(Y_hdmr[:5] - Y_pred).max())
```

## Export labeled datasets

```python
ds_sobol = sobol.to_dataset(time_coords=time_values)
ds_hdmr = hdmr.to_dataset(time_coords=time_values)

print(ds_sobol.S1.sel(param="amplitude", output="displacement"))
print(ds_sobol.S2.sel(param_i="amplitude", param_j="frequency"))
print(ds_hdmr.ST.sel(param="damping", output="velocity"))
print(ds_hdmr.Sa.sel(term="amplitude/frequency"))
```

## Why this example matters

- It shows the recommended `Problem.from_dict(..., output_names=...)` setup.
- The Sobol path uses `sampling_result.samples`, not an expanded Saltelli matrix.
- The HDMR path works from arbitrary random samples and yields an emulator for
  fast prediction.
- Both result types convert cleanly to labeled `xarray.Dataset` objects.

## See also

- [Basic Example](/examples/basic) for the smallest possible Sobol workflow.
- [Multi-Output & Time-Series](/examples/multi-output) for the output-shape
  rules in isolation.
- [xarray Labeled Output](/examples/xarray) for more dataset selection examples.
- [RS-HDMR Example](/examples/hdmr) for a smaller HDMR-only walkthrough.
