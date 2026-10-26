"""Gradient regression tests for zero-variance classes and columns."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning, Problem, borgonovo


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


def _grad_temp_bytes(n_cols: int, n: int = 1024, d: int = 8) -> int:
    """Compiled XLA temp of ``jit(grad(sum(delta)))`` with respect to Y."""
    problem = Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(d)})
    x = jnp.asarray(np.random.default_rng(0).random((n, d)), jnp.float32)

    def total(y_):
        return borgonovo.indices(problem, x, y_, slice_chunk_size=4).delta.sum()

    y = jax.ShapeDtypeStruct((n, n_cols), jnp.float32)
    stats = jax.jit(jax.grad(total)).lower(y).compile().memory_analysis()
    assert stats is not None
    return stats.temp_size_in_bytes


def test_gradient_memory_follows_the_column_batch():
    """Gradient memory is one column batch, not one working set per column.

    Without the per-column checkpoint, reverse mode through the column walk
    saved every column's class-density residuals, so the temp grew linearly
    with the column count whatever ``slice_chunk_size`` was (4.3 GiB for 64
    columns at N=4096, D=8). With it, eight times the columns must cost
    about the same.
    """
    narrow = _grad_temp_bytes(4)
    wide = _grad_temp_bytes(32)
    assert wide < 1.5 * narrow, (narrow, wide)


# Constants whose float sample variance and centred class means round to a
# tiny nonzero value instead of 0. Before the fix, 0.0539... in float32 at
# N=1000 gave S1 = 0.905 for the constant column.
_ROUNDING_CONSTANTS = (0.053902783001748715, 0.1, 1e6, -3.7)


def _two_column_y(x, c, dtype):
    y0 = jnp.sin(3.0 * x[:, 0]) + x[:, 1] ** 2
    return jnp.stack([y0, jnp.full_like(y0, c)], axis=1).astype(dtype)


@pytest.mark.parametrize("x64", [False, True], ids=["f32", "f64"])
@pytest.mark.parametrize("c", _ROUNDING_CONSTANTS)
def test_constant_output_column_is_exactly_zero(c, x64):
    """A constant output column reports delta = S1 = 0, as the warning promises.

    The kernel uses the warning's own zero-variance test, so analyze (point
    estimate and interval) and indices agree, and the varying column is
    unaffected.
    """
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        problem, x = _problem_and_x(1000)
        x = x.astype(dtype)
        y = _two_column_y(x, c, dtype)
        with pytest.warns(JaxgsaWarning, match="zero variance"):
            res = borgonovo.analyze(problem, x, y, n_bootstrap=10, key=0, verbose=False)
        ix = borgonovo.indices(problem, x, y)
        alone = borgonovo.indices(problem, x, y[:, 0])
        assert isinstance(res, borgonovo.BorgonovoResult)
        assert res.delta_conf is not None and res.S1_conf is not None
        for v in (res.delta[1], res.S1[1], res.delta_conf[:, 1], res.S1_conf[:, 1]):
            np.testing.assert_allclose(np.asarray(v), 0.0, atol=1e-12)
        for v in (ix.delta[1], ix.S1[1]):
            np.testing.assert_allclose(np.asarray(v), 0.0, atol=1e-12)
        rtol = 1e-10 if x64 else 1e-5
        np.testing.assert_allclose(np.asarray(ix.delta[0]), np.asarray(alone.delta), rtol=rtol)
        np.testing.assert_allclose(np.asarray(ix.S1[0]), np.asarray(alone.S1), rtol=rtol)


def test_rounding_constant_column_gradient_is_zero_and_isolated():
    """jit(grad) through a rounding constant column: zero there, unchanged elsewhere."""
    problem, x = _problem_and_x(1000)
    x = x.astype(jnp.float32)
    y = _two_column_y(x, _ROUNDING_CONSTANTS[0], jnp.float32)

    def total(y_):
        out = borgonovo.indices(problem, x, y_)
        return out.delta.sum() + out.S1.sum()

    g = jax.jit(jax.grad(total))(y)
    g_alone = jax.jit(jax.grad(total))(y[:, 0])
    assert bool(jnp.all(jnp.isfinite(g)))
    np.testing.assert_allclose(np.asarray(g[:, 1]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(g[:, 0]), np.asarray(g_alone), rtol=1e-4, atol=1e-6)
