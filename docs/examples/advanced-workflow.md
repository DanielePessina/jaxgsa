# Advanced Workflow

By the end of this page you will have run two sensitivity methods on the same
damped-oscillator model, built a fast stand-in for that model, and turned both
sets of results into labeled datasets you can slice by parameter name, output
name, and time. The model has five inputs and returns three outputs at each of
50 time steps.

This page adapts the repo's `development.py` example into a single docs
workflow. It runs in five steps:

1. Declare the inputs and their ranges, name the outputs, and write the model.
2. Run Sobol analysis, which splits output variance among the inputs. It needs
   its own structured design, so it drives its own model runs.
3. Run RS-HDMR analysis on a separate set of random samples. HDMR fits a
   surrogate to whatever `(X, Y)` pairs you already have, so it does not need a
   structured design.
4. Use that surrogate to predict at new inputs, and check the predictions
   against the real model.
5. Export both results to `xarray`, so you can select by name instead of by
   axis position.

## 1. Define the problem and model

`Problem.from_dict` takes the input ranges. Passing `output_names` here means
every later result and dataset carries those names, so you never have to
remember which output index is which.

```python
import jax
import jax.numpy as jnp
import numpy as np
import jaxgsa

problem = jaxgsa.Problem.from_dict(
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

## 2. Run Sobol analysis

A Sobol index is a fraction of output variance attributed to an input. `S1`
covers an input acting alone, `ST` covers it acting alone plus every
interaction it takes part in, and `S2` covers one pair of inputs acting
together. Evaluate the model on `sampling_result.samples`, then hand the
design object and the outputs to `analyze`.

```python
sampling_result = jaxgsa.sobol.sample(
    problem,
    n_samples=2048,
    seed=42,
    calc_second_order=True,
)

X_sobol = jnp.asarray(sampling_result.samples)
Y_sobol = model(X_sobol)

sobol = jaxgsa.sobol.analyze(sampling_result, Y_sobol)

print("Sobol S1 shape:", sobol.S1.shape)  # (T, K, D)
print("Sobol ST shape:", sobol.ST.shape)  # (T, K, D)
print("Sobol S2 shape:", sobol.S2.shape)  # (T, K, D, D)
```

The indices are not one number per input. `S1` and `ST` have shape
`(T, K, D)` = `(50, 3, 5)`: one index for every combination of time step,
output and input. So `sobol.S1[10, 0, 1]` is the first-order index of
`frequency` for `displacement` at the eleventh time step. Sensitivity here is
a time series in its own right, and an input can dominate early and matter
little later. `S2` carries two input axes instead of one, giving
`(50, 3, 5, 5)`, one entry per input pair.

## 3. Run HDMR on arbitrary samples

RS-HDMR fits a surrogate model as a sum of terms: one term per input, plus one
term per input pair when `maxorder=2`. The size of each term gives the
sensitivity index. Because the fit is a regression, any `(X, Y)` pairs will
do. This step uses plain uniform random draws and no structured design.

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

hdmr = jaxgsa.hdmr.analyze(problem, X_hdmr, Y_hdmr, maxorder=2)

print("HDMR S1 shape:", hdmr.S1.shape)  # (T, K, D)
print("HDMR ST shape:", hdmr.ST.shape)  # (T, K, D)
print("HDMR RMSE:", hdmr.rmse)
```

The index shapes match the Sobol ones, `(50, 3, 5)`, so you can compare the two
methods entry by entry. Check `hdmr.rmse` before you trust any of them. It is
the fit error of the surrogate against the training outputs. The indices
describe the surrogate, so a surrogate that fits the model badly gives indices
that describe the wrong function.

## 4. Predict with the HDMR emulator

The fitted surrogate is a stand-in for the model that costs almost nothing to
evaluate. Run it on inputs it was trained on to see how close it stays.

```python
Y_pred = hdmr.predict(X_hdmr[:5])
print("Prediction shape:", Y_pred.shape)  # (5, T, K)
print("Max absolute residual:", jnp.abs(Y_hdmr[:5] - Y_pred).max())
```

`Y_pred` has shape `(5, 50, 3)`: the emulator returns the full time series and
all three outputs, in the same layout as the model itself. The residual line
prints the largest gap over those five rows. Compare it against the spread of
`Y_hdmr`. A residual that is a small part of the output range means the
surrogate captured the response, and with `maxorder=2` a leftover gap means
the model has effects above second order that this expansion cannot represent.

## 5. Export labeled datasets

`to_dataset()` attaches the parameter names, output names, and the time values
to the index arrays. After that you select by name and never index by axis
position.

```python
ds_sobol = sobol.to_dataset(time_coords=time_values)
ds_hdmr = hdmr.to_dataset(time_coords=time_values)

print(ds_sobol.S1.sel(param="amplitude", output="displacement"))
print(ds_sobol.S2.sel(param_i="amplitude", param_j="frequency"))
print(ds_hdmr.ST.sel(param="damping", output="velocity"))
print(ds_hdmr.Sa.sel(term="amplitude/frequency"))
```

Each `.sel(...)` fixes the axes you name and keeps the rest. Naming both a
parameter and an output leaves the time axis alone, so the first and third
lines each print a 50-point time series of one index. The second line names
only the two parameter axes of `S2`, so it keeps time and output. The last line
selects by HDMR term instead of by parameter: `Sa` is indexed by term name, and
`"amplitude/frequency"` is the term for that input pair.

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
