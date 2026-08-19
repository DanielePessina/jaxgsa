"""Tests for HDMR's two batching axes and the one fit path that honours both.

``jaxgsa.hdmr`` cuts the sample axis into row batches (``batch_size``) and the
output-slice axis into chunks (``slice_chunk_size``). There is one fit, so the
unbatched fit is the corner ``batch_size = N``, ``slice_chunk_size = T*K``.
The tests here pin three things: subdividing either axis, or both at once,
must not move a public result field beyond float32 noise; the F-test term set
is discrete and must be picked identically; and both keywords must actually
subdivide the work on every call, which is the defect that made this file
necessary -- ``slice_chunk_size`` used to be read on one fit path only, so
``analyze(batch_size=64, slice_chunk_size=8)`` silently dropped the second.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate
from jaxgsa.hdmr import _fit
from jaxgsa.hdmr import analyze as analyze_hdmr

# Row-batched and single-batch runs do the same float32 reductions in a
# different order, so agreement is to float32 noise, not bit-exact.
# Coefficients and predictions amplify that noise through the (m+3)^order
# basis, hence the looser tolerances there.
RTOL = 1e-4
ATOL = 1e-4
COEF_RTOL = 1e-3
COEF_ATOL = 2e-2
PREDICT_ATOL = 2e-2


@pytest.fixture(autouse=True)
def _restore_memory_budget():
    """Snapshot and restore the global memory budget around every test.

    ``set_memory_budget`` mutates process-global state; leaking a tiny test
    budget would silently switch other tests onto batched paths.
    """
    from jaxgsa._core import batching

    saved = batching.get_memory_budget()
    yield
    batching._set_memory_budget(saved)


@pytest.fixture(scope="module")
def ishigami_data():
    """Random Ishigami (X, Y) data shared by all batching tests."""
    key = jax.random.PRNGKey(7)
    N = 1000
    bounds = jnp.array(PROBLEM.bounds)
    X = jax.random.uniform(key, shape=(N, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
    Y = evaluate(X)
    return X, Y


@pytest.fixture(scope="module")
def ishigami_data_ntk(ishigami_data):
    """(N, T, K) variant: T=2 time steps, K=2 output columns."""
    X, Y = ishigami_data
    Y_alt = jnp.cos(X[:, 1]) + 0.3 * X[:, 0] * X[:, 2]
    Y_tk = jnp.stack(
        [
            jnp.stack([Y, Y_alt], axis=1),
            jnp.stack([0.5 * Y + 3.0, -1.5 * Y_alt], axis=1),
        ],
        axis=1,
    )
    return X, Y_tk


@pytest.fixture(scope="module")
def full_result(ishigami_data):
    """Unbatched maxorder=2 reference fit, computed once."""
    X, Y = ishigami_data
    return analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2)


def _assert_results_match(batched, full):
    """Assert every public HDMRResult field matches between the two runs."""
    np.testing.assert_allclose(np.asarray(batched.Sa), np.asarray(full.Sa), RTOL, ATOL)
    np.testing.assert_allclose(np.asarray(batched.Sb), np.asarray(full.Sb), RTOL, ATOL)
    np.testing.assert_allclose(np.asarray(batched.S), np.asarray(full.S), RTOL, ATOL)
    np.testing.assert_allclose(np.asarray(batched.ST), np.asarray(full.ST), RTOL, ATOL)
    # F-test selection is discrete: a batched fit must pick EXACTLY the same
    # significant-term sets (select counts per term across all slices).
    np.testing.assert_array_equal(np.asarray(batched.select), np.asarray(full.select))
    assert batched.terms == full.terms
    assert batched.rmse is not None and full.rmse is not None
    np.testing.assert_allclose(np.asarray(batched.rmse), np.asarray(full.rmse), RTOL, ATOL)
    # Coefficient-level: the fitted surrogate state must agree too.
    assert batched._fit is not None and full._fit is not None
    np.testing.assert_allclose(
        np.asarray(batched._fit["f0"]), np.asarray(full._fit["f0"]), RTOL, ATOL
    )
    np.testing.assert_allclose(
        np.asarray(batched._fit["C1"]), np.asarray(full._fit["C1"]), COEF_RTOL, COEF_ATOL
    )
    if full._fit["C2"] is not None:
        assert batched._fit["C2"] is not None
        np.testing.assert_allclose(
            np.asarray(batched._fit["C2"]), np.asarray(full._fit["C2"]), COEF_RTOL, COEF_ATOL
        )


class TestRowBatching:
    """batch_size=int splits the sample axis; results match one batch."""

    def test_scalar_output_matches(self, ishigami_data, full_result):
        X, Y = ishigami_data
        batched = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2, batch_size=137)
        _assert_results_match(batched, full_result)

    def test_maxorder1_matches(self, ishigami_data):
        X, Y = ishigami_data
        full = analyze_hdmr(PROBLEM, X, Y, maxorder=1, m=2)
        batched = analyze_hdmr(PROBLEM, X, Y, maxorder=1, m=2, batch_size=250)
        _assert_results_match(batched, full)

    def test_maxorder3_matches(self, ishigami_data):
        X, Y = ishigami_data
        full = analyze_hdmr(PROBLEM, X, Y, maxorder=3, m=2)
        batched = analyze_hdmr(PROBLEM, X, Y, maxorder=3, m=2, batch_size=333)
        _assert_results_match(batched, full)
        # Raw C3 is not comparable element-wise: with a 125-wide tensor-product
        # basis per triple, the per-term Gram is nearly singular, so f32 noise
        # moves coefficients along near-null directions without changing the
        # fitted component functions. Compare the order-3 fit functionally.
        assert full._fit is not None and batched._fit is not None
        assert full._fit["C3"] is not None and batched._fit["C3"] is not None
        np.testing.assert_allclose(
            np.asarray(batched.predict(X[:200])),
            np.asarray(full.predict(X[:200])),
            rtol=1e-3,
            atol=PREDICT_ATOL,
        )

    def test_ntk_output_matches(self, ishigami_data_ntk):
        X, Y_tk = ishigami_data_ntk
        full = analyze_hdmr(PROBLEM, X, Y_tk, maxorder=2, m=2)
        batched = analyze_hdmr(PROBLEM, X, Y_tk, maxorder=2, m=2, batch_size=111)
        n_terms = 3 + 3
        assert batched.Sa.shape == (2, 2, n_terms)
        assert batched.rmse is not None and batched.rmse.shape == (2, 2)
        _assert_results_match(batched, full)

    def test_batch_size_larger_than_n_is_one_batch(self, ishigami_data, full_result):
        """Tier T4 (internal consistency): batch_size >= N is a single batch.

        ``streamed`` reports whether the sample axis was actually split, so a
        batch size the data never reaches leaves it False.
        """
        X, Y = ishigami_data
        single = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2, batch_size=10**6)
        assert single.streamed is False
        _assert_results_match(single, full_result)


class TestAutoEngage:
    """A small global budget splits the sample axis on its own."""

    def test_tiny_budget_batches_and_matches(self, ishigami_data, full_result):
        """Tier T4 (internal consistency): a small budget engages row batching.

        This is the structural guard for the whole file. Every other test here
        compares two batchings, and they agree by design, so a change to the
        memory estimate that stopped row batching engaging at all would leave
        those tests green. Asserting the reported path fails instead.
        """
        X, Y = ishigami_data
        assert full_result.streamed is False
        jaxgsa.config.set_memory_budget(256 * 1024, unit="b")  # 256 KiB
        batched = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2)
        assert batched.streamed is True
        _assert_results_match(batched, full_result)


class TestBothAxesCompose:
    """The defect this file exists for: both keywords work on every call."""

    def _count_blocks(self, monkeypatch) -> list[int]:
        """Count how many (row batch, slice chunk) blocks the fit processes.

        The contract of the two keywords is that they subdivide the work.
        Comparing numbers cannot see a dropped keyword, because a keyword that
        does nothing returns the same numbers -- that is exactly how
        ``slice_chunk_size`` came to be ignored on one of the two old fit
        paths. So this counts the blocks the final statistics pass runs, which
        is ``ceil(N / batch_size) * ceil(T*K / slice_chunk_size)``.
        """
        calls = [0]
        real = _fit._get_stats_step

        def counting(n1: int, n2: int):
            step = real(n1, n2)

            def wrapper(*args, **kwargs):
                calls[0] += 1
                return step(*args, **kwargs)

            return wrapper

        monkeypatch.setattr(_fit, "_get_stats_step", counting)
        return calls

    def test_both_keywords_subdivide_the_work(self, ishigami_data_ntk, monkeypatch):
        """batch_size and slice_chunk_size both take effect on one call."""
        X, Y_tk = ishigami_data_ntk  # N=1000, T*K = 4
        calls = self._count_blocks(monkeypatch)
        analyze_hdmr(PROBLEM, X, Y_tk, maxorder=2, m=2, batch_size=250, slice_chunk_size=1)
        assert calls[0] == 4 * 4  # 4 row batches x 4 slice chunks

    def test_slice_chunk_size_alone_subdivides(self, ishigami_data_ntk, monkeypatch):
        calls = self._count_blocks(monkeypatch)
        X, Y_tk = ishigami_data_ntk
        analyze_hdmr(PROBLEM, X, Y_tk, maxorder=2, m=2, slice_chunk_size=2)
        assert calls[0] == 1 * 2

    def test_ragged_chunks_do_not_change_the_answer(self, ishigami_data_ntk):
        """Widths that do not divide N or T*K are padded, not truncated."""
        X, Y_tk = ishigami_data_ntk  # N=1000, T*K = 4
        full = analyze_hdmr(PROBLEM, X, Y_tk, maxorder=2, m=2)
        ragged = analyze_hdmr(
            PROBLEM, X, Y_tk, maxorder=2, m=2, batch_size=333, slice_chunk_size=3
        )
        _assert_results_match(ragged, full)

    def test_both_axes_under_a_tiny_budget(self, ishigami_data_ntk):
        """An automatic batch_size still leaves slice_chunk_size usable."""
        X, Y_tk = ishigami_data_ntk
        full = analyze_hdmr(PROBLEM, X, Y_tk, maxorder=2, m=2)
        jaxgsa.config.set_memory_budget(256 * 1024, unit="b")
        batched = analyze_hdmr(PROBLEM, X, Y_tk, maxorder=2, m=2, slice_chunk_size=1)
        assert batched.streamed is True
        _assert_results_match(batched, full)
