"""Tests for the PCE streaming fit.

The streamed fit accumulates the normal equations and the exact two-pass LOO
over row batches; it must match the single-pass path to float32 tolerances on
every public result field, engage automatically when the memory budget is
exceeded, and be forceable via ``batch_size``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import pce
from jaxgsa.problem import Problem

# Streamed vs single-pass runs do the same float32 reductions in a different
# order (batched accumulation vs one big matmul, inv vs solve for the hat
# diagonal), so agreement is to float32 noise, not bit-exact.
RTOL = 1e-4
ATOL = 1e-5


@pytest.fixture(autouse=True)
def _restore_memory_budget():
    """Snapshot and restore the global memory budget around every test.

    ``set_memory_budget`` mutates process-global state; leaking a tiny test
    budget would silently switch other tests onto batched/streamed paths.
    """
    from jaxgsa._core import batching

    saved = batching.get_memory_budget()
    yield
    batching._set_memory_budget(saved)


@pytest.fixture(scope="module")
def problem_4d() -> Problem:
    return Problem(names=("x1", "x2", "x3", "x4"), bounds=((-1.0, 1.0),) * 4)


@pytest.fixture(scope="module")
def data_4d(problem_4d):
    """Moderate nonlinear 4-D problem: N=2048, order-3 fit is non-trivial."""
    key = jax.random.PRNGKey(0)
    X = jax.random.uniform(key, shape=(2048, 4), minval=-1.0, maxval=1.0)
    Y = (
        jnp.sin(2.0 * X[:, 0])
        + 0.7 * X[:, 1] ** 3
        + 1.5 * X[:, 2] * X[:, 3]
        + 0.3 * X[:, 0] * X[:, 1]
    )
    return X, Y


@pytest.fixture(scope="module")
def data_4d_ntk(data_4d):
    """(N, T, K) variant: T=2 time steps, K=2 output columns."""
    X, Y = data_4d
    Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=-1)  # (N, K=2)
    Y3 = jnp.stack([Y2, 0.5 * Y2 - 3.0], axis=1)  # (N, T=2, K=2)
    return X, Y3


def _assert_results_match(a, b, *, rtol=RTOL, atol=ATOL):
    """Assert every public PCEResult field matches between two runs."""
    np.testing.assert_allclose(np.asarray(a.coefficients), np.asarray(b.coefficients), rtol, atol)
    np.testing.assert_allclose(np.asarray(a.S1), np.asarray(b.S1), rtol, atol)
    np.testing.assert_allclose(np.asarray(a.ST), np.asarray(b.ST), rtol, atol)
    np.testing.assert_allclose(np.asarray(a.S2), np.asarray(b.S2), rtol, atol, equal_nan=True)
    assert a.explained_variance is not None and b.explained_variance is not None
    np.testing.assert_allclose(
        np.asarray(a.explained_variance), np.asarray(b.explained_variance), rtol, atol
    )
    assert a.loo_rmse is not None and b.loo_rmse is not None
    np.testing.assert_allclose(np.asarray(a.loo_rmse), np.asarray(b.loo_rmse), rtol, atol)
    assert a.order == b.order
    np.testing.assert_array_equal(a.multi_index, b.multi_index)


class TestForcedStreaming:
    """batch_size=int forces the streamed fit; results match single-pass."""

    def test_scalar_output_matches(self, problem_4d, data_4d):
        X, Y = data_4d
        single = pce.analyze(problem_4d, X, Y, order=3)
        streamed = pce.analyze(problem_4d, X, Y, order=3, batch_size=300)
        _assert_results_match(streamed, single)

    def test_ntk_output_matches(self, problem_4d, data_4d_ntk):
        X, Y3 = data_4d_ntk
        single = pce.analyze(problem_4d, X, Y3, order=3)
        streamed = pce.analyze(problem_4d, X, Y3, order=3, batch_size=257)
        assert streamed.S1.shape == (2, 2, 4)
        assert streamed.loo_rmse is not None and streamed.loo_rmse.shape == (2, 2)
        _assert_results_match(streamed, single)

    def test_batch_size_larger_than_n_streams_once(self, problem_4d, data_4d):
        """Tier T4 (internal consistency): batch_size >= N still streams.

        One full batch is still the streamed path, and the reported flag says
        so.
        """
        X, Y = data_4d
        single = pce.analyze(problem_4d, X, Y, order=3)
        streamed = pce.analyze(problem_4d, X, Y, order=3, batch_size=10**6)
        assert streamed.streamed is True
        _assert_results_match(streamed, single)

    def test_invalid_batch_size_raises(self, problem_4d, data_4d):
        X, Y = data_4d
        with pytest.raises(ValueError, match="batch_size"):
            pce.analyze(problem_4d, X, Y, order=3, batch_size=0)


class TestAutoEngage:
    """A small global budget flips the default path to streaming."""

    def test_tiny_budget_streams_and_matches(self, problem_4d, data_4d):
        """Tier T4 (internal consistency): a small budget engages streaming.

        This is the structural guard for the whole file. Every other test here
        compares the two paths, and they agree by design, so a change to the
        memory estimate that stopped the streamed path engaging at all would
        leave those tests green. Asserting the reported path fails instead.
        """
        X, Y = data_4d
        single = pce.analyze(problem_4d, X, Y, order=3)
        assert single.streamed is False

        saved = jaxgsa.config.get_memory_budget()
        try:
            jaxgsa.config.set_memory_budget(64 * 1024, unit="b")  # 64 KiB: forces streaming
            streamed = pce.analyze(problem_4d, X, Y, order=3)
        finally:
            jaxgsa.config.set_memory_budget(saved, unit="b")
        assert jaxgsa.config.get_memory_budget() == saved
        assert streamed.streamed is True
        _assert_results_match(streamed, single)
