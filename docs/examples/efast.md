# eFAST (Extended FAST)

By the end of this page you will have a first-order and a total-order Sobol
index for every input, from a design that costs one sweep per input. A Sobol
index is a share of output variance attributed to an input. The first-order
index `S1` covers the input acting alone. The total index `ST` covers the input
acting alone plus every interaction it takes part in.

The extended Fourier amplitude sensitivity test (eFAST) gets there by a
different route from the Sobol estimators. It walks the input space along a
sinusoidal search curve, giving each input its own frequency. An input that
drives the output leaves its frequency in the output signal, and the strength
of that frequency in the output's Fourier spectrum is its index. eFAST does not
produce second-order (S2) interaction indices, which report the effect of one
pair of inputs acting together.

When to use eFAST instead of Sobol:

- You only need S1 and ST (no S2).
- You want a simpler, non-structured sampling design.
- You are screening a large number of parameters.

## Import style

The eFAST module lives at `jaxgsa.efast`:

```python
from jaxgsa import efast
# efast.sample(...)
# efast.analyze(...)
```

## Scalar example (Ishigami)

The run has three steps. Build the design with `efast.sample`, which lays out
one search curve per input. Evaluate your model on every row of it. Then call
`efast.analyze` with the design object rather than the raw array, so the
analyzer knows which rows belong to which curve and which frequency to look
for.

```python
import jax.numpy as jnp
from jaxgsa import efast
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

# Generate eFAST samples: n_per_curve points per search curve, D curves total
samples = efast.sample(PROBLEM, n_per_curve=4096, M=4, seed=42)
print("samples.samples shape:", samples.samples.shape)  # (4096 * 3, 3) = (12288, 3)
print("n_runs:", samples.n_runs)  # 12288

# Evaluate the model
Y = evaluate(jnp.asarray(samples.samples))
print("Y shape:", Y.shape)  # (12288,)

# Compute eFAST indices — M and the problem travel inside `samples`
result = efast.analyze(samples, Y)

print("S1:", result.S1)  # (D,) = (3,)
print("ST:", result.ST)  # (D,) = (3,)
print("omega_0:", result.omega_0)
print("M:", result.M)
```

The design cost is visible in the shapes: 4096 points per curve times 3 inputs
is 12288 model runs, and that number grows linearly with the number of inputs.
Compare `S1` and `ST` entry by entry. Where they agree, the input acts alone.
Where `ST` is much larger, the input matters through interactions, though eFAST
will not tell you with which partner. `omega_0` and `M` are the frequency
settings the analysis used, and they are echoed back so a stored result records
how it was produced.

## Multi-output example

When your model returns K outputs per sample, pass Y with shape `(n_runs, K)`.
The resulting indices have shape `(K, D)`.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa import efast

problem = jaxgsa.Problem.from_dict(
    {
        "amplitude": (0.5, 2.0),
        "frequency": (1.0, 5.0),
        "damping": (0.01, 0.5),
    },
    output_names=("displacement", "velocity"),
)


def multi_output_model(X):
    amp = X[:, 0]
    freq = X[:, 1]
    damping = X[:, 2]
    displacement = amp * jnp.sin(freq) * jnp.exp(-damping)
    velocity = amp * jnp.cos(freq) * jnp.exp(-damping)
    return jnp.stack([displacement, velocity], axis=-1)  # (n_runs, K=2)


samples = efast.sample(problem, n_per_curve=4096, seed=42)
Y = multi_output_model(jnp.asarray(samples.samples))
print("Y shape:", Y.shape)  # (12288, 2)

result = efast.analyze(samples, Y)
print("S1 shape:", result.S1.shape)  # (K, D) = (2, 3)
print("ST shape:", result.ST.shape)  # (K, D) = (2, 3)
```

The same design serves both outputs. `result.S1[0]` ranks the three inputs for
`displacement` and `result.S1[1]` ranks them for `velocity`, in the order
`output_names` declares. Extra outputs cost no extra model runs, so return
every quantity you might want to analyze in one pass.

## Time-series example

When your model returns T time steps and K outputs, pass Y with shape
`(n_runs, T, K)`. The resulting indices have shape `(T, K, D)`.

```python
import jax.numpy as jnp
import numpy as np
import jaxgsa
from jaxgsa import efast

problem = jaxgsa.Problem.from_dict(
    {
        "amplitude": (0.5, 2.0),
        "frequency": (1.0, 5.0),
        "damping": (0.01, 0.5),
    },
    output_names=("displacement", "velocity"),
)

time_values = np.linspace(0.0, 5.0, 20)


def time_series_model(X):
    amp = X[:, 0, None]
    freq = X[:, 1, None]
    damping = X[:, 2, None]
    tt = jnp.asarray(time_values)[None, :]

    displacement = amp * jnp.sin(2 * jnp.pi * freq * tt) * jnp.exp(-damping * tt)
    velocity = amp * jnp.cos(2 * jnp.pi * freq * tt) * jnp.exp(-damping * tt)
    return jnp.stack([displacement, velocity], axis=-1)  # (n_runs, T, K=2)


samples = efast.sample(problem, n_per_curve=4096, seed=42)
Y = time_series_model(jnp.asarray(samples.samples))
print("Y shape:", Y.shape)  # (12288, 20, 2)

result = efast.analyze(samples, Y)
print("S1 shape:", result.S1.shape)  # (T, K, D) = (20, 2, 3)
print("ST shape:", result.ST.shape)  # (T, K, D) = (20, 2, 3)
```

Each index is now a curve rather than a number: `result.S1[:, 0, 1]` is the
first-order index of `frequency` for `displacement` across all 20 time steps.
Plot these curves before you rank anything. Damping matters more late in the
trajectory than early, so a single ranking taken at one time step can be
misleading.

## xarray export

`EFASTResult.to_dataset()` converts results to a labeled `xarray.Dataset`,
just like the Sobol and HDMR result types.

```python
ds = result.to_dataset(time_coords=time_values)
print(ds)
# <xarray.Dataset>
# Dimensions:  (time: 20, output: 2, param: 3)

print(ds.S1.sel(param="amplitude"))
print(ds.ST.sel(output="velocity"))
```

## Shape rules

Below, `n_runs = n_per_curve * D` is the total row count of the design
(`samples.n_runs`); `analyze` rejects any `Y` whose leading dimension differs.

- `(n_runs,)` means scalar output.
- `(n_runs, K)` means K output variables with no time dimension.
- `(n_runs, T, K)` means T time steps and K outputs.
- Without `problem.output_names`, a 2D array is always treated as `(n_runs, K)`.
- With exactly one entry in `problem.output_names`, a 2D array is treated as
  `(n_runs, T)` — timepoints of that single output — and flows through as
  `(n_runs, T, 1)`. Passing a pre-reshaped `(n_runs, T, 1)` array also works.

| Y shape | S1 / ST shape |
|---------|---------------|
| `(n_runs,)` | `(D,)` |
| `(n_runs, K)` | `(K, D)` |
| `(n_runs, T, K)` | `(T, K, D)` |

D is always the last axis.

## Practical caveats

- eFAST does not produce S2 (second-order) indices. Use the Sobol workflow
  if you need pairwise interaction estimates.
- The `M` parameter (interference factor) is set once in `sample()` and travels
  inside the returned `EFASTSamples`, so it can never be mismatched at
  `analyze()` time. The default is 4.
- `n_per_curve` must satisfy `n_per_curve >= 4*M^2*(D-1) + 1`, where `D` is the
  number of parameters — the cost of eFAST grows with dimension because every
  parameter needs its own frequency. With the default `M=4` and `D=3` that is
  `n_per_curve >= 129`; at `D=10` it is `n_per_curve >= 577`. Shorter designs
  raise `ValueError` rather than silently reusing a frequency for two
  parameters, which would leave them indistinguishable along the curve.
- Indices outside `[0, 1]` indicate insufficient samples or near-zero output
  variance.

## See also

- [Basic Example](/examples/basic) for the Sobol workflow with structured
  Saltelli sampling and S2 support.
- [Multi-Output & Time-Series](/examples/multi-output) for the same shape
  conventions on Sobol and HDMR.
- [xarray Labeled Output](/examples/xarray) for named access by parameter,
  output, and time coordinate.
- [Methods](/guide/methods) for the theory behind eFAST and when to choose it
  over other methods.
