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

    def test_explicit_batch_size_engages_streaming(self, problem_4d, data_4d):
        """Tier T4 (internal consistency): batch_size forces the streamed path.

        Under the default (huge) memory budget this fit would run in one pass,
        so an explicit ``batch_size`` is the only thing that can make
        ``streamed`` True here. The flag reports the fit path the code took,
        so this checks the code against its own documented dispatch rule and
        nothing outside it.
        """
        X, Y = data_4d
        result = pce.analyze(problem_4d, X, Y, order=3, batch_size=512)
        assert result.streamed is True


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

    def test_default_budget_keeps_single_pass(self, problem_4d, data_4d):
        """Tier T4 (internal consistency): small fits stay on the one-pass path.

        The default budget is far above this fit's estimated footprint, so the
        reported path must be the single-pass one.
        """
        X, Y = data_4d
        assert pce.analyze(problem_4d, X, Y, order=3).streamed is False


class TestSinglePassFitBytes:
    """The PCE auto-engage memory estimate, pinned to hand-computed literals."""

    @pytest.mark.parametrize(
        ("N", "n_terms", "M", "itemsize", "expected"),
        [
            # Scalar fit, N=2048, order-3 in 4-D (n_terms=35), float32:
            #   3 x (N, n_terms) = 3 x 2048 x 35 = 215040 values
            #   2 x (N, M)       = 2 x 2048 x 1  =   4096 values
            #   total 219136 values x 4 bytes    = 876544 bytes
            (2048, 35, 1, 4, 876_544),
            # Multi-output fit with M > n_terms / 2 (16 > 10), N=1000,
            # n_terms=20, float32:
            #   3 x (N, n_terms) = 3 x 1000 x 20 =  60000 values
            #   2 x (N, M)       = 2 x 1000 x 16 =  32000 values
            #   total 92000 values x 4 bytes     = 368000 bytes
            # The rejected two-term variant would give 4 x 1000 x (40 + 32)
            # = 288000 bytes, a 21.7% under-count.
            (1000, 20, 16, 4, 368_000),
            # Time-series fit, N=4000, n_terms=35, M=20 (again M > n_terms/2),
            # float32:
            #   3 x (N, n_terms) = 3 x 4000 x 35 = 420000 values
            #   2 x (N, M)       = 2 x 4000 x 20 = 160000 values
            #   total 580000 values x 4 bytes    = 2320000 bytes
            # The two-term variant would give 4 x 4000 x (70 + 40) = 1760000
            # bytes, a 24.1% under-count: the worst case the docstring names.
            (4000, 35, 20, 4, 2_320_000),
            # Same shape in float64 doubles the estimate, so the itemsize
            # factor is pinned too.
            (4000, 35, 20, 8, 4_640_000),
        ],
    )
    def test_single_pass_fit_bytes_formula(self, N, n_terms, M, itemsize, expected):
        """Tier T4 (internal consistency): the auto-engage memory estimate.

        ``_single_pass_fit_bytes`` charges ``itemsize * N * (3 * n_terms +
        2 * M)``: three ``(N, n_terms)`` arrays resident at the leave-one-out
        peak (the design matrix ``Phi``, the ``gram_inv_PhiT`` product the
        coefficient solve still holds, and the triangular-solve result inside
        ``hat_diagonal``), plus the ``(N, M)`` prediction and residual arrays
        ``loo_error`` materializes. Each case's per-array breakdown is in the
        comment above it.

        Two of the four cases have ``M > n_terms / 2``, which is the
        multi-output and time-series regime. That is where a two-term variant
        of this formula under-counts the peak by up to a quarter, and a
        single-slice case (``M = 1``) cannot separate the two. A measurement
        with ``jax.live_arrays()`` during a real fit found 3.16 times
        ``N * n_terms`` resident, so three is the right count.

        The literals are hand-computed from the array breakdown the function
        documents for itself, not from any paper or outside tool, so this is
        T4 and not T1. What it proves: the code agrees with its own documented
        arithmetic, and a silent change to the formula fails instead of moving
        with it. What it does not prove: that the breakdown names the right
        arrays, or that the estimate matches the memory JAX really allocates.
        Only a measurement can show that.
        """
        from jaxgsa.pce._analyze import _single_pass_fit_bytes

        assert _single_pass_fit_bytes(N=N, n_terms=n_terms, M=M, itemsize=itemsize) == expected


class TestMemoryBudgetConfig:
    """set_memory_budget / get_memory_budget contract."""

    def test_get_returns_default(self):
        from jaxgsa._core.batching import DEFAULT_EMULATE_BUDGET_BYTES

        assert jaxgsa.config.get_memory_budget() == DEFAULT_EMULATE_BUDGET_BYTES

    def test_get_returns_what_was_set(self):
        jaxgsa.config.set_memory_budget(123 * 1024**2, unit="b")
        assert jaxgsa.config.get_memory_budget() == 123 * 1024**2

    @pytest.mark.parametrize("bad", [0, -1, -(512 * 1024**2)])
    def test_rejects_non_positive(self, bad):
        with pytest.raises(ValueError, match="positive"):
            jaxgsa.config.set_memory_budget(bad)

    @pytest.mark.parametrize("bad", ["512MiB", None, True])
    def test_rejects_non_int(self, bad):
        with pytest.raises(ValueError, match="positive"):
            jaxgsa.config.set_memory_budget(bad)

    def test_budget_applies_to_predict_batching(self):
        """resolve_batch_size reads the global budget for predict batches too."""
        from jaxgsa._core.batching import resolve_batch_size

        jaxgsa.config.set_memory_budget(1000, unit="b")
        assert resolve_batch_size(100, 50, None) == 10

    def test_per_call_batch_size_overrides_global(self, problem_4d, data_4d):
        """An explicit batch_size wins over the global budget in both directions."""
        from jaxgsa._core.batching import resolve_batch_size

        jaxgsa.config.set_memory_budget(1000, unit="b")
        # Explicit batch of 40 rows beats the budget-derived 10.
        assert resolve_batch_size(100, 50, 40) == 40
        # And through the public API: a huge budget plus an explicit
        # batch_size still yields correct (streamed) results.
        X, Y = data_4d
        jaxgsa.config.set_memory_budget(2**62, unit="b")
        single = pce.analyze(problem_4d, X, Y, order=3)
        streamed = pce.analyze(problem_4d, X, Y, order=3, batch_size=333)
        _assert_results_match(streamed, single)
