"""Tests for the PCE streaming fit and the global memory-budget knob.

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
from jaxgsa.pce import _analyze as pce_analyze_mod
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


class _DesignBuildCounter:
    """Wraps build_design_matrix to count invocations (1 = single-pass)."""

    def __init__(self):
        self.calls = 0
        self._orig = pce_analyze_mod.build_design_matrix

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._orig(*args, **kwargs)


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

    def test_loo_two_pass_exactness(self, problem_4d, data_4d):
        """The streamed two-pass LOO equals the single-pass hat-diagonal LOO."""
        X, Y = data_4d
        single = pce.analyze(problem_4d, X, Y, order=3)
        streamed = pce.analyze(problem_4d, X, Y, order=3, batch_size=100)
        assert single.loo_rmse is not None and streamed.loo_rmse is not None
        np.testing.assert_allclose(
            float(streamed.loo_rmse), float(single.loo_rmse), rtol=RTOL, atol=ATOL
        )

    def test_batch_size_larger_than_n_streams_once(self, problem_4d, data_4d):
        """batch_size >= N still runs the streamed path (one full batch)."""
        X, Y = data_4d
        single = pce.analyze(problem_4d, X, Y, order=3)
        streamed = pce.analyze(problem_4d, X, Y, order=3, batch_size=10**6)
        _assert_results_match(streamed, single)

    def test_invalid_batch_size_raises(self, problem_4d, data_4d):
        X, Y = data_4d
        with pytest.raises(ValueError, match="batch_size"):
            pce.analyze(problem_4d, X, Y, order=3, batch_size=0)

    def test_explicit_batch_size_engages_streaming(self, problem_4d, data_4d, monkeypatch):
        """An explicit batch_size streams even under the default (huge) budget."""
        X, Y = data_4d
        counter = _DesignBuildCounter()
        monkeypatch.setattr(pce_analyze_mod, "build_design_matrix", counter)
        pce.analyze(problem_4d, X, Y, order=3, batch_size=512)
        # Two passes over ceil(2048/512) = 4 batches each.
        assert counter.calls == 8


class TestAutoEngage:
    """A small global budget flips the default path to streaming."""

    def test_tiny_budget_streams_and_matches(self, problem_4d, data_4d, monkeypatch):
        X, Y = data_4d
        single = pce.analyze(problem_4d, X, Y, order=3)

        counter = _DesignBuildCounter()
        monkeypatch.setattr(pce_analyze_mod, "build_design_matrix", counter)
        saved = jaxgsa.config.get_memory_budget()
        try:
            jaxgsa.config.set_memory_budget(64 * 1024)  # 64 KiB: forces streaming
            streamed = pce.analyze(problem_4d, X, Y, order=3)
        finally:
            jaxgsa.config.set_memory_budget(saved)
        assert jaxgsa.config.get_memory_budget() == saved
        # More than one design-block build per pass proves streaming engaged.
        assert counter.calls > 2
        _assert_results_match(streamed, single)

    def test_default_budget_keeps_single_pass(self, problem_4d, data_4d, monkeypatch):
        """Small fits under the default budget never stream (one Phi build)."""
        X, Y = data_4d
        counter = _DesignBuildCounter()
        monkeypatch.setattr(pce_analyze_mod, "build_design_matrix", counter)
        pce.analyze(problem_4d, X, Y, order=3)
        assert counter.calls == 1


class TestMemoryBudgetConfig:
    """set_memory_budget / get_memory_budget contract."""

    def test_get_returns_default(self):
        from jaxgsa._core.batching import DEFAULT_EMULATE_BUDGET_BYTES

        assert jaxgsa.config.get_memory_budget() == DEFAULT_EMULATE_BUDGET_BYTES

    def test_get_returns_what_was_set(self):
        jaxgsa.config.set_memory_budget(123 * 1024**2)
        assert jaxgsa.config.get_memory_budget() == 123 * 1024**2

    @pytest.mark.parametrize("bad", [0, -1, -(512 * 1024**2)])
    def test_rejects_non_positive(self, bad):
        with pytest.raises(ValueError, match="positive"):
            jaxgsa.config.set_memory_budget(bad)

    @pytest.mark.parametrize("bad", [1.5, "512MiB", None, True])
    def test_rejects_non_int(self, bad):
        with pytest.raises(ValueError, match="positive"):
            jaxgsa.config.set_memory_budget(bad)

    def test_budget_applies_to_predict_batching(self):
        """resolve_batch_size reads the global budget for predict batches too."""
        from jaxgsa._core.batching import resolve_batch_size

        jaxgsa.config.set_memory_budget(1000)
        assert resolve_batch_size(100, 50, None) == 10

    def test_per_call_batch_size_overrides_global(self, problem_4d, data_4d):
        """An explicit batch_size wins over the global budget in both directions."""
        from jaxgsa._core.batching import resolve_batch_size

        jaxgsa.config.set_memory_budget(1000)
        # Explicit batch of 40 rows beats the budget-derived 10.
        assert resolve_batch_size(100, 50, 40) == 40
        # And through the public API: a huge budget plus an explicit
        # batch_size still yields correct (streamed) results.
        X, Y = data_4d
        jaxgsa.config.set_memory_budget(2**62)
        single = pce.analyze(problem_4d, X, Y, order=3)
        streamed = pce.analyze(problem_4d, X, Y, order=3, batch_size=333)
        _assert_results_match(streamed, single)
