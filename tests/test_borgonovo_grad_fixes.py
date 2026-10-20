"""Gradient regression tests for zero-variance classes and columns."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxgsa import Problem, borgonovo


def _problem_and_x(n: int = 500):
    problem = Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0)})
    x = jnp.asarray(np.random.default_rng(0).random((n, 2)))
    return problem, x


def test_grad_finite_with_constant_classes():
    """A ReLU output has constant classes, and its gradient must be finite."""
    problem, x = _problem_and_x()
    y = jnp.maximum(x[:, 0] - 0.5, 0.0)

    def delta_sum(y_):
        return borgonovo.indices(problem, x, y_).delta.sum()

    def s1_sum(y_):
        return borgonovo.indices(problem, x, y_).S1.sum()

    for fn in (delta_sum, s1_sum):
        g = jax.jit(jax.jacrev(fn))(y)
        assert bool(jnp.all(jnp.isfinite(g)))


def test_constant_output_column_has_zero_gradient():
    """A constant output column gets a gradient of exactly zero."""
    problem, x = _problem_and_x()
    y0 = x[:, 0] + 0.3 * x[:, 1]
    y = jnp.stack([y0, jnp.full_like(y0, 2.0)], axis=1)

    def delta_sum(y_):
        return borgonovo.indices(problem, x, y_).delta.sum()

    g = jax.grad(delta_sum)(y)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert bool(jnp.all(g[:, 1] == 0.0))
