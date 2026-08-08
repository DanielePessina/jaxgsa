# Multi-Output & Time-Series

By the end of this page you will have run one damped-oscillator model twice
through the same `jaxgsa` call, once on a full time history and once on a
single time step, and you will know how to read the index array that comes
back in each case.

A Sobol' index measures how much of the variance in a model output is caused by
one input parameter. `jaxgsa` reports two of them here. `S1` is the first-order
index: the share of output variance explained by that parameter on its own.
`ST` is the total-order index: the share explained by that parameter on its own
plus every interaction it takes part in.

The same `jaxgsa.sobol.analyze()` call accepts scalar, multi-output, and
time-series multi-output arrays. The letters used for the array axes are `N`
for the number of model runs, `T` for timepoints, `K` for outputs, and `D` for
input parameters. This page uses one concrete model to show both the `(N, K)`
and the `(N, T, K)` layout.

## Fully runnable example

The example below runs in five steps.

1. Declare the problem. `Problem.from_dict()` names the four input parameters
   and their ranges. Naming the two outputs here means later index arrays and
   dataset exports can be read by name instead of by position.
2. Write the model. `oscillator_model` returns displacement and velocity at 40
   timepoints, stacked on the last axis. Stacking outputs last is what produces
   the `(N, T, K)` layout that `jaxgsa` expects.
3. Sample. `jaxgsa.sobol.sample()` builds the Saltelli design that the Sobol'
   estimator needs, and returns only the unique rows you have to evaluate.
4. Build two output arrays from one model evaluation. `Y_time` keeps every
   timepoint. `Y_snapshot` keeps only the last one. Reusing the same run means
   the two analyses differ in output layout alone.
5. Analyze both. The same function handles both layouts, so the only difference
   is the shape of the array that comes back.

```python
import jax.numpy as jnp
import numpy as np
import jaxgsa

problem = jaxgsa.Problem.from_dict(
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


sampling_result = jaxgsa.sobol.sample(problem, n_samples=2048, seed=42)
X = jnp.asarray(sampling_result.samples)

Y_time = oscillator_model(X)      # (N, T, K)
Y_snapshot = Y_time[:, -1, :]     # (N, K)

time_result = jaxgsa.sobol.analyze(sampling_result, Y_time)
snapshot_result = jaxgsa.sobol.analyze(sampling_result, Y_snapshot)

print("Time-series S1 shape:", time_result.S1.shape)      # (T, K, D)
print("Time-series ST shape:", time_result.ST.shape)      # (T, K, D)
print("Snapshot S1 shape:", snapshot_result.S1.shape)     # (K, D)
print("Snapshot ST shape:", snapshot_result.ST.shape)     # (K, D)

print("Displacement sensitivities at the final time step:")
print(time_result.S1[-1, 0, :])

print("Velocity sensitivities for the snapshot:")
print(snapshot_result.S1[1, :])
```

## Reading the output

The two analyses return arrays of different rank, and the rank tells you what
the leading axis means.

- `time_result.S1` has shape `(T, K, D)`, so `(40, 2, 4)` here. That is one
  first-order index per timepoint, per output, per input parameter: 320 numbers
  in total. `time_result.S1[-1, 0, :]` selects the last of the 40 timepoints,
  then output 0, which is `displacement` because it is first in
  `output_names`. The four numbers printed are the first-order indices of
  amplitude, frequency, damping, and offset, in the order they were declared
  in `from_dict()`.
- `snapshot_result.S1` has shape `(K, D)`, so `(2, 4)`. The time axis is gone
  because `Y_snapshot` has no time axis. `snapshot_result.S1[1, :]` selects
  output 1, which is `velocity`, and prints its four first-order indices.

The two printed rows are not the same numbers. `time_result.S1[-1, 0, :]` is
displacement at the final time step, and `snapshot_result.S1[1, :]` is velocity
at that same final time step. Displacement carries the `offset` term and
velocity does not, so the two outputs do not have to rank their inputs the same
way.

## Shape rules

`jaxgsa` decides what an array means from its rank, and from
`problem.output_names` when the rank alone is ambiguous.

- `(N,)` means scalar output.
- `(N, K)` means multiple outputs with no time dimension.
- `(N, T, K)` means time-series multi-output.
- Without `problem.output_names`, a 2D array is always treated as `(N, K)`.
- With exactly one entry in `problem.output_names`, a 2D array is treated as
  `(N, T)` — timepoints of that single output — and flows through as
  `(N, T, 1)`. Passing a pre-reshaped `(N, T, 1)` array also works.
- Obvious layout mistakes (e.g. a transposed array) are fixed with a
  `UserWarning`; ambiguous layouts raise.

## Single-output edge case

One output is the case where the rules above are easiest to trip over, so the
snippet below prints the resulting shape in each direction. A truly scalar
output drops both the time and the output axis. A single output measured over
time keeps a length-1 output axis rather than dropping it.

```python
# Scalar output
Y_scalar = Y_snapshot[:, 0]      # (N,)
scalar_result = jaxgsa.sobol.analyze(sampling_result, Y_scalar)
print(scalar_result.S1.shape)    # (D,)

# Time-series with one output
Y_one_output = Y_time[:, :, :1]  # (N, T, 1)
one_output_result = jaxgsa.sobol.analyze(sampling_result, Y_one_output)
print(one_output_result.S1.shape)  # (T, 1, D)
```

The first result is `(D,)`, one index per input parameter and nothing else.
The second is `(T, 1, D)`: the output axis stays, with length 1. Index it as
`one_output_result.S1[:, 0, :]` to get a `(T, D)` array you can plot against
time.

## Practical caveats

- Named outputs come from `problem.output_names`, so set them up early if you
  plan to export with `to_dataset()`.
- `calc_second_order=False` removes `S2`, the second-order indices for
  parameter pairs. Dropping them can be a useful tradeoff for large `(T, K)`
  outputs when you only need `S1` and `ST`.
- The same shape rules apply to `jaxgsa.hdmr.analyze()`.

## See also

- [xarray Labeled Output](/examples/xarray) for named access by parameter,
  output, and time coordinate.
- [RS-HDMR Example](/examples/hdmr) for the same shape rules on the surrogate
  workflow.
- [Advanced Workflow](/examples/advanced-workflow) for a bigger custom model
  that combines Sobol, HDMR, emulator prediction, and dataset export.
