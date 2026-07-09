# Multi-Output & Time-Series

`gsax` accepts scalar, multi-output, and time-series multi-output arrays from
the same API. This page uses one concrete model to show both `(N, K)` and
`(N, T, K)` layouts.

## Fully runnable example

```python
import jax.numpy as jnp
import numpy as np
import gsax

problem = gsax.Problem.from_dict(
    {
        "amplitude": (0.5, 2.0),
        "frequency": (1.0, 5.0),
        "damping": (0.01, 0.5),
        "offset": (-1.0, 1.0),
    },
    output_names=("displacement", "velocity"),
)

time_values = np.linspace(0.0, 5.0, 40)


def oscillator_model(X):
    amp = X[:, 0, None]
    freq = X[:, 1, None]
    damping = X[:, 2, None]
    offset = X[:, 3, None]
    tt = jnp.asarray(time_values)[None, :]

    displacement = (
        amp * jnp.sin(2 * jnp.pi * freq * tt) * jnp.exp(-damping * tt) + offset
    )
    velocity = amp * jnp.cos(2 * jnp.pi * freq * tt) * jnp.exp(-damping * tt)

    return jnp.stack([displacement, velocity], axis=-1)  # (N, T, K=2)


sampling_result = gsax.sample(problem, n_samples=2048, seed=42)
X = jnp.asarray(sampling_result.samples)

Y_time = oscillator_model(X)      # (N, T, K)
Y_snapshot = Y_time[:, -1, :]     # (N, K)

time_result = gsax.analyze(sampling_result, Y_time)
snapshot_result = gsax.analyze(sampling_result, Y_snapshot)

print("Time-series S1 shape:", time_result.S1.shape)      # (T, K, D)
print("Time-series ST shape:", time_result.ST.shape)      # (T, K, D)
print("Snapshot S1 shape:", snapshot_result.S1.shape)     # (K, D)
print("Snapshot ST shape:", snapshot_result.ST.shape)     # (K, D)

print("Displacement sensitivities at the final time step:")
print(time_result.S1[-1, 0, :])

print("Velocity sensitivities for the snapshot:")
print(snapshot_result.S1[1, :])
```

## Shape rules

- `(N,)` means scalar output.
- `(N, K)` means multiple outputs with no time dimension.
- `(N, T, K)` means time-series multi-output.
- A 2D array is always treated as `(N, K)`, never `(N, T)`.
- For a time-series with one output, reshape to `(N, T, 1)`.

## Single-output edge case

```python
# Scalar output
Y_scalar = Y_snapshot[:, 0]      # (N,)
scalar_result = gsax.analyze(sampling_result, Y_scalar)
print(scalar_result.S1.shape)    # (D,)

# Time-series with one output
Y_one_output = Y_time[:, :, :1]  # (N, T, 1)
one_output_result = gsax.analyze(sampling_result, Y_one_output)
print(one_output_result.S1.shape)  # (T, 1, D)
```

## Practical caveats

- Named outputs come from `problem.output_names`, so set them up early if you
  plan to export with `to_dataset()`.
- `calc_second_order=False` removes `S2`, which can be a useful tradeoff for
  large `(T, K)` outputs when you only need `S1` and `ST`.
- The same shape rules apply to `gsax.analyze_hdmr()`.

## See also

- [xarray Labeled Output](/examples/xarray) for named access by parameter,
  output, and time coordinate.
- [RS-HDMR Example](/examples/hdmr) for the same shape rules on the surrogate
  workflow.
- [Advanced Workflow](/examples/advanced-workflow) for a bigger custom model
  that combines Sobol, HDMR, emulator prediction, and dataset export.
