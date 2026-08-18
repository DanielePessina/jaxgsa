# DGSM

`jaxgsa.dgsm.analyze()` computes derivative-based global sensitivity measures
with Poincaré bounds on the total Sobol indices. It needs a
JAX-differentiable model or precomputed Jacobians.

## How to call it

There are two ways to call `analyze()`. Use exactly one of them.

- Give it a model and inputs: `analyze(problem, fn, X)`. It differentiates the
  model for you.
- Give it values you already have: `analyze(problem, Y=Y, dfdx=dfdx)`. Use this
  when the model is not JAX-differentiable, or when the Jacobian comes from
  somewhere else.

If you give arguments from both groups, or fill only half of one group, the
call raises a `ValueError` that names the argument. Nothing is dropped in
silence.

## `fn` takes one sample

`fn` is a **one-sample** function. It maps one row of shape `(D,)` to a scalar
`()`, a vector `(K,)`, or an array `(T, K)`. This is different from the other
methods in this package, which call the model on the whole `(N, D)` matrix.
`analyze()` does the batching itself.

Wrap a batch model like this:

```python
result = jaxgsa.dgsm.analyze(problem, lambda x: model(x[None, :])[0], X)
```

A batch model passed straight in raises a `ValueError` that names the expected
signature. The check traces `fn` on one row with `jax.eval_shape`, so it never
runs the model and costs an expensive model nothing.

The check adds the "wrap a batch model" advice only when the failure looks like
a batch callable: the function indexes or reduces over a sample axis the row
does not have, or it returns more than two axes. Any other trace failure is
reported plainly. You get the original error and the expected signature, and no
guess at the cause. A model that already takes one row and fails for its own
reason is therefore not sent to write a wrapper it does not need.

The canonical API reference lives at [API Reference](/api/).

Jump directly to:

- [`jaxgsa.sampling.monte_carlo()`](/api/#given-data-methods)
- [`jaxgsa.dgsm.analyze()`](/api/#given-data-methods)
- [`jaxgsa.dgsm.DGSMResult`](/api/#given-data-methods)
- [`jaxgsa.dgsm.poincare_constant()`](/api/#given-data-methods)
- [`jaxgsa.dgsm.axis_constants()`](/api/#given-data-methods)

Related docs:

- [DGSM Example](/examples/dgsm)
- [Methods](/guide/methods)
