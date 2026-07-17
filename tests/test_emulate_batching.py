"""Batched emulation must match single-shot emulation.

Both ``predict`` and ``predict`` accept a ``batch_size`` kwarg that
splits prediction over row batches to bound transient memory. Each row's
basis contraction is independent, so batching must match the single-shot
path for every output layout up to float32 reassociation (XLA tiles the
term-axis reduction differently for different batch shapes, giving ~1e-6
relative differences).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from gsax import hdmr, pce
from gsax._batching import apply_batched, resolve_batch_size
from gsax.problem import Problem

D = 4
N_FIT = 800
N_NEW = 257  # deliberately not a multiple of any batch size used below


@pytest.fixture(scope="module")
def problem() -> Problem:
    return Problem.from_dict({f"x{i}": (-1.0, 1.0) for i in range(D)})


def _make_xy(shape: str) -> tuple[Array, Array]:
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(N_FIT, D))
    y = np.sum(np.sin(X) + 0.3 * X**2, axis=1)
    if shape == "scalar":
        Y = y
    elif shape == "multi":
        Y = y[:, None] * np.arange(1, 4)[None, :]
    else:
        # time-series (N, T, K): T=5, K=3
        t = np.linspace(0.0, 1.0, 5)[None, :, None]
        k = np.arange(1, 4)[None, None, :]
        Y = y[:, None, None] * (1.0 + t * k)
    return jnp.asarray(X), jnp.asarray(Y)


def _x_new() -> Array:
    rng = np.random.default_rng(1)
    return jnp.asarray(rng.uniform(-1.0, 1.0, size=(N_NEW, D)))


@pytest.mark.parametrize("shape", ["scalar", "multi", "timeseries"])
@pytest.mark.parametrize("batch_size", [1, 100, N_NEW, 10_000])
def test_pce_batched_matches_single_shot(problem, shape, batch_size):
    X, Y = _make_xy(shape)
    result = pce.analyze(problem, X, Y, order=3)
    X_new = _x_new()

    single = result.predict(X_new, batch_size=N_NEW)
    batched = result.predict(X_new, batch_size=batch_size)

    assert batched.shape == single.shape
    np.testing.assert_allclose(np.asarray(batched), np.asarray(single), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", ["scalar", "multi", "timeseries"])
@pytest.mark.parametrize("batch_size", [1, 100, N_NEW, 10_000])
def test_hdmr_batched_matches_single_shot(problem, shape, batch_size):
    X, Y = _make_xy(shape)
    result = hdmr.analyze(problem, X, Y, maxorder=2)
    X_new = _x_new()

    single = result.predict(X_new, batch_size=N_NEW)
    batched = result.predict(X_new, batch_size=batch_size)

    assert batched.shape == single.shape
    np.testing.assert_allclose(np.asarray(batched), np.asarray(single), rtol=1e-5, atol=1e-6)


def test_hdmr_third_order_batched_matches_single_shot(problem):
    X, Y = _make_xy("scalar")
    result = hdmr.analyze(problem, X, Y, maxorder=3, m=2)
    X_new = _x_new()

    single = result.predict(X_new, batch_size=N_NEW)
    batched = result.predict(X_new, batch_size=64)

    np.testing.assert_allclose(np.asarray(batched), np.asarray(single), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", ["scalar", "timeseries"])
def test_auto_batch_default_matches_explicit(problem, shape):
    """batch_size=None (auto) must give the same predictions."""
    X, Y = _make_xy(shape)
    X_new = _x_new()

    r_pce = pce.analyze(problem, X, Y, order=3)
    np.testing.assert_allclose(
        np.asarray(r_pce.predict(X_new)),
        np.asarray(r_pce.predict(X_new, batch_size=N_NEW)),
        rtol=1e-5,
        atol=1e-6,
    )

    r_hdmr = hdmr.analyze(problem, X, Y, maxorder=2)
    np.testing.assert_allclose(
        np.asarray(r_hdmr.predict(X_new)),
        np.asarray(r_hdmr.predict(X_new, batch_size=N_NEW)),
        rtol=1e-5,
        atol=1e-6,
    )


def test_prenormalized_hdmr_batched(problem):
    """Batching must compose with the inverse output standardization."""
    X, Y = _make_xy("scalar")
    result = hdmr.analyze(problem, X, Y, maxorder=2, prenormalize=True)
    X_new = _x_new()

    single = result.predict(X_new, batch_size=N_NEW)
    batched = result.predict(X_new, batch_size=32)

    np.testing.assert_allclose(np.asarray(batched), np.asarray(single), rtol=1e-5, atol=1e-6)


def test_invalid_batch_size_raises(problem):
    X, Y = _make_xy("scalar")
    result = pce.analyze(problem, X, Y, order=3)
    with pytest.raises(ValueError, match="batch_size"):
        result.predict(_x_new(), batch_size=0)


def test_resolve_batch_size_bounds():
    # Explicit values are clamped to the row count.
    assert resolve_batch_size(1000, 50, 200) == 50
    assert resolve_batch_size(1000, 50, 7) == 7
    # Auto: budget-derived, at least 1 even for enormous rows.
    assert resolve_batch_size(10**12, 50, None) == 1
    # Auto: cheap rows fall back to single-shot.
    assert resolve_batch_size(8, 50, None) == 50


def test_apply_batched_host_buffer_matches_single_shot():
    """The host-staged batched path returns a jax Array identical to fn(X)."""
    X = jnp.asarray(np.random.default_rng(3).normal(size=(11, 4)))

    def fn(x: Array) -> Array:
        return jnp.stack([x.sum(axis=1), x.prod(axis=1)], axis=-1)

    single = fn(X)
    # 4 does not divide 11, so the last batch is a short remainder.
    batched = apply_batched(fn, X, 4)
    assert isinstance(batched, jax.Array)
    assert batched.dtype == single.dtype
    np.testing.assert_allclose(np.asarray(batched), np.asarray(single), rtol=1e-6)


def test_pce_batched_jit_compatible(problem):
    """predict stays jit-traceable when batching kicks in."""
    X, Y = _make_xy("scalar")
    result = pce.analyze(problem, X, Y, order=3)
    X_new = _x_new()

    fn = jax.jit(lambda x: result.predict(x, batch_size=100))
    np.testing.assert_allclose(
        np.asarray(fn(X_new)),
        np.asarray(result.predict(X_new, batch_size=N_NEW)),
        rtol=1e-6,
        atol=1e-6,
    )
