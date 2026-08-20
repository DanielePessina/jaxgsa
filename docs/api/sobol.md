# Sobol

`jaxgsa.sobol.analyze()` turns model outputs on a Saltelli design into
variance-based Sobol indices: `S1` per parameter, `ST` per parameter, and `S2`
per pair when the design carries second order.

```python
analyze(
    sampling_result: SobolSamples,
    Y: Array,
    *,
    estimator: Estimator = "saltelli-jansen",
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    slice_chunk_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> SobolResult
```

`Y` holds the model outputs at each unique row of `sampling_result.samples`,
in that row order. Accepted shapes are `(n_runs,)`, `(n_runs, K)` and
`(n_runs, T, K)`. Indices are computed independently for every `(t, k)` slice.

```python
import numpy as np, jax.numpy as jnp, jaxgsa

problem = jaxgsa.Problem(("x1", "x2", "x3"), ((-np.pi, np.pi),) * 3)

def ishigami(X):
    return (
        jnp.sin(X[:, 0])
        + 7.0 * jnp.sin(X[:, 1]) ** 2
        + 0.1 * X[:, 2] ** 4 * jnp.sin(X[:, 0])
    )

samples = jaxgsa.sobol.sample(problem, n_samples=8192, seed=0, verbose=False)
result = jaxgsa.sobol.analyze(samples, ishigami(samples.samples), verbose=False)

result.S1        # Array([0.3223, 0.4361, 0.0014], dtype=float32)
result.ST        # Array([0.556 , 0.4417, 0.2413], dtype=float32)
result.S2.shape  # (3, 3)
```

The analytical answer for Ishigami is `S1 = [0.3139, 0.4424, 0.0]` and
`ST = [0.5576, 0.4424, 0.2437]`, so 1024 base samples land within about 0.01.
Drop `verbose=False` to get the printed run summary described in the
[API reference](/api/#printed-run-summaries).

`SobolResult` carries:

| Field | Shape | Notes |
| --- | --- | --- |
| `S1`, `ST` | `(D,)` / `(K, D)` / `(T, K, D)` | mirrors the layout of `Y` |
| `S2` | `(D, D)` / `(K, D, D)` / `(T, K, D, D)` | `None` for a first-order-only design |
| `S1_conf`, `ST_conf`, `S2_conf` | one extra leading axis of length 2 | `None` without a bootstrap |
| `ci` | — | the `CIInfo` record, or `None` |
| `estimator` | — | the estimator name that ran |
| `problem`, `invalid` | — | the problem, and the non-finite report |

`S2` is symmetric with a NaN diagonal. `S2[j, k]` and `S2[k, j]` are two
independent Monte Carlo estimates of the same pairwise index, so jaxgsa
reports their average; the two differ by sampling noise, not
floating-point drift. Averaging usually lowers the error, but not always.
Measured on Ishigami over 200 seeds and all three parameter pairs, the
RMSE falls 25% at `base_n=64` and 13% at `base_n=1024`, and rises 9% at
`base_n=256`, where one pair drew a lower-triangle estimate three times
noisier than its upper one. Averaging is still the better default, because
you cannot tell in advance which triangle is the lucky one. This is a
deliberate departure from SALib, which reports only the upper triangle. A
parameter has no pairwise interaction with itself, and the diagonal is NaN
so that reads louder than a zero would.

Every result supports `to_dataset(time_coords=None)` for a labeled xarray
export.

The bootstrap resamples base points as if they were an i.i.d. sample. A
scrambled Sobol' design converges faster than that, so the interval does
not describe the true spread. It is too narrow at a small design and far
too wide at a large one. Measured coverage of the 95% interval on an
exact-zero index (Ishigami's `S1` for `x3`) is 90.7-91.5% at `base_n=256`
(`"quantile"`) and 92.7-93.0% (`"gaussian"`), so the interval under-covers
there. At `base_n=2048` the bootstrap standard deviation runs 3.2x the
true seed-to-seed spread and coverage reaches 100%. `"azzini-rosati"`
covers 96-97% at `base_n=256`. Read the interval as a rough guide, not as
a calibrated 95% statement. This is the only interval in the package that
was measured for coverage.

## jaxgsa.sobol.indices()

```python
indices(sampling_result, Y, *, estimator="saltelli-jansen", slice_chunk_size=None)
    -> tuple[Array, ...]
```

`indices` returns the same numbers as `analyze`, as plain arrays. It does
nothing else: no check for non-finite outputs, no zero-variance warning, no
`SobolResult`, and no printing.

Use `analyze` for ordinary work. Use `indices` when the calculation has to go
inside a JAX transformation. `analyze` reads values on the host to decide what
its `on_invalid` policy should do, and a policy decision needs a concrete
number, so it cannot run under `jax.jit` or `jax.vmap`. `indices` reads
nothing, so it can.

```python
S1, ST = jaxgsa.sobol.indices(samples, Y)
```

The return is a tuple of two arrays for a first-order design and three
(`S1, ST, S2`) for a second-order one, so unpack it accordingly.

`morris`, `efast` and `dgsm` each expose an `indices()` under the same
contract.

## Choosing an estimator

Both functions take `estimator=` and both default to `"saltelli-jansen"`. The
accepted names are `"saltelli-jansen"`, `"jansen"`, `"janon-monod"`,
`"martinez"`, `"mauntz-kucherenko"` and `"azzini-rosati"`. Anything else raises
a `ValueError` before any array is touched.

```python
result = jaxgsa.sobol.analyze(samples, Y, estimator="azzini-rosati")
```

They all converge to the same indices. They differ in how much sampling noise
they carry at small `N`, and in whether an estimate can fall outside `[0, 1]`.

The default pairs Sobol'-Mauntz first order with Jansen (1999) total order for
two reasons. Jansen's total-order estimator is a mean of squares, so `ST` can
never come back negative. And it is the pairing SALib uses by default, so the
two libraries agree out of the box.

`"azzini-rosati"` reads the `BA` blocks of the design, so it needs
`calc_second_order=True`. It is the only one that does, and the only one that
holds `S1 <= ST` on every sample. Every estimator is plain arithmetic on the
output vectors, so the choice costs `indices` none of its `jit`, `vmap` or
`jacrev` support.

See [Methods](/guide/methods#choosing-a-different-estimator) for the measured
errors behind the default, and for what a negative index estimate means.

## slice_chunk_size

`slice_chunk_size` is the number of `(T, K)` output slices per vmap batch. It
is a memory knob, not an algorithm switch. On the bootstrap path each slice in
a batch carries all `n_bootstrap` of its draws, so one device call covers
`slice_chunk_size * n_bootstrap` estimator evaluations.

`None` (the default) derives the width from the memory budget. A slice costs
about `2 * N * (D + 2)` elements first-order-only, and
`2 * N * (2D + 2) + N * D * D` with second order, because every second-order
estimator forms an `(N, D, D)` outer product. Set it yourself when you hit a
device out-of-memory error, and lower it in factors of two.

It changes no index beyond float noise. The estimator sums over the sample
axis, and XLA schedules that reduction differently for a different batch width,
so two chunk sizes can disagree in the last bits of a float32 result.

## Differentiating an index

Pair `indices` with
[`SobolSamples.transform`](/api/sampling#sobolsamples) to get the derivative
of a Sobol index with respect to the input distribution parameters:

```python
import jax
jax.config.update("jax_enable_x64", True)

theta = {name: {"low": -np.pi, "high": np.pi} for name in problem.names}

def s1(theta):
    Y = ishigami(samples.transform(theta))   # your model must be JAX-differentiable
    return jaxgsa.sobol.indices(samples, Y)[0]

dS1 = jax.jacrev(s1)(theta)
dS1["x1"]["low"]     # Array([ 0.0278, -0.0369,  0.003 ])
dS1["x1"]["high"]    # Array([-0.0174,  0.0378,  0.    ])
```

Each entry is a full `(D,)` vector: widening `x1` downward moves `S1` for
every parameter, not only for `x1`, because the indices share one output
variance.

The gradient reaches `theta` because the design is built as
`X = F inverse (u; theta)` from quasi-random points `u` that do not depend on
`theta`. That is a reparameterisation, so it differentiates.

Two limits are worth stating plainly.

The chain runs through your model, so your model must be differentiable in
JAX. There is no route to this derivative that avoids it.

Enable float64 before you rely on the numbers, as above. In single precision
the derivative is dominated by rounding.

Related docs:

- [Bootstrap Confidence Intervals](/examples/bootstrap)
- [xarray Output](/examples/xarray)
- [Methods](/guide/methods)
