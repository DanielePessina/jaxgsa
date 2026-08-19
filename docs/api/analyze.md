# Analyze (Sobol)

`jaxgsa.sobol.analyze()` computes variance-based Sobol indices (`S1`, `ST`,
and the optional `S2`) from model outputs evaluated on a Saltelli design built
by `jaxgsa.sobol.sample()`.

The canonical API reference lives at [API Reference](/api/).

Jump directly to:

- [`jaxgsa.sobol.analyze()`](/api/#sobol)
- [`jaxgsa.sobol.SobolResult`](/api/#sobol)

## `jaxgsa.sobol.indices()`

`indices(samples, Y)` returns the same numbers as `analyze`, as plain arrays.
It does nothing else: no check for non-finite outputs, no zero-variance
warning, and no `SobolResult`.

Use `analyze` for ordinary work. Use `indices` when you need to put the
calculation inside a JAX transformation. `analyze` reads values on the host to
decide what its `on_invalid` policy should do, and a policy decision needs a
concrete number, so it cannot run under `jax.jit` or `jax.vmap`. `indices`
reads nothing, so it can.

Both call the same estimator, so the numbers are the same.

```python
S1, ST = jaxgsa.sobol.indices(samples, Y)
```

## Choosing an estimator

Both functions take `estimator=`, and both default to `"saltelli-jansen"`,
which is what jaxgsa has always computed. The accepted names are
`"saltelli-jansen"`, `"jansen"`, `"janon-monod"`, `"martinez"`,
`"mauntz-kucherenko"` and `"azzini-rosati"`. Anything else raises a
`ValueError` before any array is touched.

```python
result = jaxgsa.sobol.analyze(samples, Y, estimator="azzini-rosati")
```

`"azzini-rosati"` reads the `BA` blocks of the design, so it needs
`calc_second_order=True`; it is the only one that does, and the only one that
can never report `S1 > ST`. Every estimator is plain arithmetic on the output
vectors, so the choice costs `indices` none of its `jit`, `vmap` or `jacrev`
support.

See [Methods](/guide/methods#choosing-a-different-estimator) for the measured
errors behind the default, and for what a negative index estimate means.

### Differentiating an index

Pair `indices` with
[`SobolSamples.transform`](/api/sampling#sobolsamples) to get the derivative
of a Sobol index with respect to the input distribution parameters:

```python
import jax

def s1(theta):
    Y = model(samples.transform(theta))     # your model must be JAX-differentiable
    return jaxgsa.sobol.indices(samples, Y)[0]

dS1_dtheta = jax.jacrev(s1)(theta)
dS1_dtheta["x1"]["low"]     # d S1 / d (lower bound of x1)
```

The gradient reaches `theta` because the design is built as
`X = F inverse (u; theta)` from quasi-random points `u` that do not depend on
`theta`. That is a reparameterisation, so it differentiates.

Two limits are worth stating plainly.

The chain runs through your model, so your model must be differentiable in
JAX. There is no route to this derivative that avoids it.

Enable float64 before you rely on the numbers:
`jax.config.update("jax_enable_x64", True)`. In single precision the
derivative is dominated by rounding.

Related docs:

- [Bootstrap Confidence Intervals](/examples/bootstrap)
- [xarray Output](/examples/xarray)
- [Methods](/guide/methods)
