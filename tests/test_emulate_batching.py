"""Batched emulation must match single-shot emulation.

Tier T4 (internal consistency) throughout. There is no external oracle for a
batching seam; these tests prove only that the batched path agrees with the
single-shot path, whose values are anchored elsewhere.

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

from jaxgsa import hdmr, pce
from jaxgsa._core.batching import apply_batched, resolve_batch_size
from jaxgsa._core.surrogate import SurrogateResult, _PredictPlan
from jaxgsa.problem import Problem

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
@pytest.mark.parametrize("batch_size", [1, 100])
def test_pce_batched_matches_single_shot(problem, shape, batch_size):
    X, Y = _make_xy(shape)
    result = pce.analyze(problem, X, Y, order=3)
    X_new = _x_new()

    single = result.predict(X_new, batch_size=N_NEW)
    batched = result.predict(X_new, batch_size=batch_size)

    assert batched.shape == single.shape
    np.testing.assert_allclose(np.asarray(batched), np.asarray(single), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", ["scalar", "multi", "timeseries"])
@pytest.mark.parametrize("batch_size", [1, 100])
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


class _FakeSurrogate(SurrogateResult):
    """Minimal SurrogateResult: row sums, no real fit, records batch sizes."""

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.batch_row_counts: list[int] = []

    def _predict_plan(self, X: Array) -> _PredictPlan:
        def kernel(chunk: Array) -> Array:
            self.batch_row_counts.append(int(chunk.shape[0]))
            return chunk.sum(axis=1)

        return _PredictPlan(X=X, bytes_per_row=8, kernel=kernel)

    def shapley(self):
        raise NotImplementedError


def test_pce_hdmr_reject_wrong_shaped_x_identically(problem):
    """Both result types reject malformed X with the same shared error."""
    X, Y = _make_xy("scalar")
    results = [pce.analyze(problem, X, Y, order=2), hdmr.analyze(problem, X, Y, maxorder=1)]

    messages_1d = []
    messages_wide = []
    for result in results:
        with pytest.raises(ValueError, match="2-D") as exc_1d:
            result.predict(jnp.ones(10))
        messages_1d.append(str(exc_1d.value))
        with pytest.raises(ValueError, match="columns") as exc_wide:
            result.predict(jnp.ones((10, D + 2)))
        messages_wide.append(str(exc_wide.value))

    # One shared validator, one wording: pce and hdmr messages are identical.
    assert messages_1d[0] == messages_1d[1]
    assert messages_wide[0] == messages_wide[1]

    # The template validates before any subclass code, so a rejected call
    # never reaches the prediction kernel. Only a surrogate that records its
    # own invocations can observe that, hence the fake.
    fake = _FakeSurrogate(problem)
    with pytest.raises(ValueError, match="2-D"):
        fake.predict(jnp.ones(10))
    with pytest.raises(ValueError, match="columns"):
        fake.predict(jnp.ones((10, D + 2)))
    with pytest.raises(ValueError, match="batch_size"):
        fake.predict(jnp.ones((10, D)), batch_size=0)
    assert fake.batch_row_counts == []  # kernel never ran on invalid input


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
