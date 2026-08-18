"""Tests for the row-streamed HDMR fit and its auto-engage policy.

The streamed fit (``jaxgsa.hdmr._stream``) accumulates the component-regression
Gram blocks, the ANCOVA covariances, and the F-test sums of squares over row
batches; it must match the in-memory kernel to float32 tolerances on every
public result field, pick the SAME F-test term set (selection is discrete),
engage automatically when the memory budget is exceeded, and be forceable via
``batch_size``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate
from jaxgsa.hdmr import analyze as analyze_hdmr

# Streamed vs in-memory runs do the same float32 reductions in a different
# order (row-batched Gram accumulation vs one full-N kernel), so agreement is
# to float32 noise, not bit-exact. Coefficients and predictions amplify that
# noise through the (m+3)^order basis, hence the looser tolerances there.
RTOL = 1e-4
ATOL = 1e-4
COEF_RTOL = 1e-3
COEF_ATOL = 2e-2
PREDICT_ATOL = 2e-2


@pytest.fixture(autouse=True)
def _restore_memory_budget():
    """Snapshot and restore the global memory budget around every test.

    ``set_memory_budget`` mutates process-global state; leaking a tiny test
    budget would silently switch other tests onto streamed paths.
    """
    from jaxgsa._core import batching

    saved = batching.get_memory_budget()
    yield
    batching._set_memory_budget(saved)


@pytest.fixture(scope="module")
def ishigami_data():
    """Random Ishigami (X, Y) data shared by all streaming tests."""
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
    """In-memory maxorder=2 reference fit, computed once."""
    X, Y = ishigami_data
    return analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2)


def _assert_results_match(streamed, full):
    """Assert every public HDMRResult field matches between the two paths."""
    np.testing.assert_allclose(np.asarray(streamed.Sa), np.asarray(full.Sa), RTOL, ATOL)
    np.testing.assert_allclose(np.asarray(streamed.Sb), np.asarray(full.Sb), RTOL, ATOL)
    np.testing.assert_allclose(np.asarray(streamed.S), np.asarray(full.S), RTOL, ATOL)
    np.testing.assert_allclose(np.asarray(streamed.ST), np.asarray(full.ST), RTOL, ATOL)
    # F-test selection is discrete: the streamed fit must pick EXACTLY the
    # same significant-term sets (select counts per term across all slices).
    np.testing.assert_array_equal(np.asarray(streamed.select), np.asarray(full.select))
    assert streamed.terms == full.terms
    assert streamed.rmse is not None and full.rmse is not None
    np.testing.assert_allclose(np.asarray(streamed.rmse), np.asarray(full.rmse), RTOL, ATOL)
    # Coefficient-level: the fitted surrogate state must agree too.
    assert streamed._fit is not None and full._fit is not None
    np.testing.assert_allclose(
        np.asarray(streamed._fit["f0"]), np.asarray(full._fit["f0"]), RTOL, ATOL
    )
    np.testing.assert_allclose(
        np.asarray(streamed._fit["C1"]), np.asarray(full._fit["C1"]), COEF_RTOL, COEF_ATOL
    )
    if full._fit["C2"] is not None:
        assert streamed._fit["C2"] is not None
        np.testing.assert_allclose(
            np.asarray(streamed._fit["C2"]), np.asarray(full._fit["C2"]), COEF_RTOL, COEF_ATOL
        )


class TestForcedStreaming:
    """batch_size=int forces the streamed fit; results match in-memory."""

    def test_scalar_output_matches(self, ishigami_data, full_result):
        X, Y = ishigami_data
        streamed = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2, batch_size=137)
        _assert_results_match(streamed, full_result)

    def test_maxorder1_matches(self, ishigami_data):
        X, Y = ishigami_data
        full = analyze_hdmr(PROBLEM, X, Y, maxorder=1, m=2)
        streamed = analyze_hdmr(PROBLEM, X, Y, maxorder=1, m=2, batch_size=250)
        _assert_results_match(streamed, full)

    def test_maxorder3_matches(self, ishigami_data):
        X, Y = ishigami_data
        full = analyze_hdmr(PROBLEM, X, Y, maxorder=3, m=2)
        streamed = analyze_hdmr(PROBLEM, X, Y, maxorder=3, m=2, batch_size=333)
        _assert_results_match(streamed, full)
        # Raw C3 is not comparable element-wise: with a 125-wide tensor-product
        # basis per triple, the per-term Gram is nearly singular, so f32 noise
        # moves coefficients along near-null directions without changing the
        # fitted component functions. Compare the order-3 fit functionally.
        assert full._fit is not None and streamed._fit is not None
        assert full._fit["C3"] is not None and streamed._fit["C3"] is not None
        np.testing.assert_allclose(
            np.asarray(streamed.predict(X[:200])),
            np.asarray(full.predict(X[:200])),
            rtol=1e-3,
            atol=PREDICT_ATOL,
        )

    def test_ntk_output_matches(self, ishigami_data_ntk):
        X, Y_tk = ishigami_data_ntk
        full = analyze_hdmr(PROBLEM, X, Y_tk, maxorder=2, m=2)
        streamed = analyze_hdmr(PROBLEM, X, Y_tk, maxorder=2, m=2, batch_size=111)
        n_terms = 3 + 3
        assert streamed.Sa.shape == (2, 2, n_terms)
        assert streamed.rmse is not None and streamed.rmse.shape == (2, 2)
        _assert_results_match(streamed, full)

    def test_prenormalize_matches(self, ishigami_data_ntk):
        X, Y_tk = ishigami_data_ntk
        full = analyze_hdmr(PROBLEM, X, Y_tk, maxorder=2, m=2, prenormalize=True)
        streamed = analyze_hdmr(
            PROBLEM, X, Y_tk, maxorder=2, m=2, prenormalize=True, batch_size=200
        )
        _assert_results_match(streamed, full)

    def test_predict_round_trip(self, ishigami_data, full_result):
        """The streamed fit's surrogate predicts like the in-memory one."""
        X, Y = ishigami_data
        streamed = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2, batch_size=137)
        np.testing.assert_allclose(
            np.asarray(streamed.predict(X[:100])),
            np.asarray(full_result.predict(X[:100])),
            rtol=1e-3,
            atol=PREDICT_ATOL,
        )

    def test_batch_size_larger_than_n_streams_once(self, ishigami_data, full_result):
        """Tier T4 (internal consistency): batch_size >= N still streams.

        One full batch is still the streamed path, and the reported flag says
        so.
        """
        X, Y = ishigami_data
        streamed = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2, batch_size=10**6)
        assert streamed.streamed is True
        _assert_results_match(streamed, full_result)

    def test_invalid_batch_size_raises(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="batch_size"):
            analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2, batch_size=0)

    def test_explicit_batch_size_engages_streaming(self, ishigami_data):
        """Tier T4 (internal consistency): batch_size forces the streamed path.

        Under the default (huge) memory budget this fit would run in memory,
        so an explicit ``batch_size`` is the only thing that can make
        ``streamed`` True here. The flag reports the fit path the code took,
        so this checks the code against its own documented dispatch rule and
        nothing outside it.
        """
        X, Y = ishigami_data
        result = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2, batch_size=500)
        assert result.streamed is True


class TestAutoEngage:
    """A small global budget flips the default path to streaming."""

    def test_tiny_budget_streams_and_matches(self, ishigami_data, full_result):
        """Tier T4 (internal consistency): a small budget engages streaming.

        This is the structural guard for the whole file. Every other test here
        compares the two paths, and they agree by design, so a change to the
        memory estimate that stopped the streamed path engaging at all would
        leave those tests green. Asserting the reported path fails instead.
        """
        X, Y = ishigami_data
        assert full_result.streamed is False
        saved = jaxgsa.config.get_memory_budget()
        try:
            jaxgsa.config.set_memory_budget(256 * 1024, unit="b")  # 256 KiB: forces streaming
            streamed = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2)
        finally:
            jaxgsa.config.set_memory_budget(saved, unit="b")
        assert jaxgsa.config.get_memory_budget() == saved
        assert streamed.streamed is True
        _assert_results_match(streamed, full_result)

    def test_default_budget_keeps_in_memory(self, ishigami_data):
        """Tier T4 (internal consistency): small fits stay on the in-memory path.

        The default budget is far above this fit's estimated footprint, so the
        reported path must be the in-memory one.
        """
        X, Y = ishigami_data
        assert analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2).streamed is False

    def test_full_fit_bytes_formula(self):
        """Tier T4 (internal consistency): the auto-engage memory estimate.

        For ``maxorder=2, D=3, m=2`` (so ``m1=5, m2=25, n1=3, n2=3, n=6``) and
        ``N=1000`` float32 values the resident arrays are ``B1`` (1000 x 5 x 3 =
        15000), ``B2`` (1000 x 25 x 3 = 75000), the ``B3`` placeholder (1000)
        and the per-lane kernel transients (1000 x (2 x 3 x 5 + 5 x 6) = 60000).
        That is 151000 values, or 604000 bytes at 4 bytes each.

        The literal is hand-computed from the array breakdown that
        ``_full_fit_bytes`` documents for itself, not from any paper or outside
        tool, so this is T4 and not T1. What it proves: the code agrees with
        its own documented arithmetic, and a silent change to the formula makes
        the test fail instead of moving with it. What it does not prove: that
        the breakdown names the right arrays, or that the estimate matches the
        memory JAX really allocates. Only a measurement can show that.
        """
        from jaxgsa.hdmr._stream import _full_fit_bytes

        got = _full_fit_bytes(N=1000, D=3, m1=5, m2=25, m3=125, n1=3, n2=3, n3=0, n=6, itemsize=4)
        assert got == 604_000


class TestHDMRStaticDataLayout:
    """The twelve-field ``_HDMRStaticData`` NamedTuple's positional layout."""

    def test_field_order_is_unchanged(self):
        """Tier T4 (internal consistency): the positional field order is pinned.

        ``_get_hdmr_static_data`` builds ``_HDMRStaticData`` positionally,
        while every consumer reads it by attribute name. Swapping two names in
        the class body therefore re-maps the values silently, with no error
        anywhere. Same-typed neighbours make that easy to do by accident:
        ``n1, n2, n3, n`` and ``m1, m2, m3`` are all plain ints.

        Append new fields, never insert them.
        """
        from jaxgsa.hdmr._analyze import _HDMRStaticData

        assert _HDMRStaticData._fields == (
            "c1",
            "c2",
            "c3",
            "n1",
            "n2",
            "n3",
            "n",
            "m1",
            "m2",
            "m3",
            "beta2",
            "beta3",
        )

    def test_values_land_on_the_right_names(self):
        """Tier T4 (internal consistency): each name carries its own value.

        The field-order test above catches a rename. This one catches the
        other half: that the positional construction puts each computed value
        under the name its own docstring gives it. For ``D=3, maxorder=2,
        m=2`` the values are fixed by the definitions ``n1 = D``, ``n2 =
        C(D, 2)``, ``n3 = 0``, ``n = n1 + n2 + n3``, ``m1 = m + 3``,
        ``m2 = m1^2``, ``m3 = m1^3``.
        """
        from jaxgsa.hdmr._analyze import _get_hdmr_static_data

        static = _get_hdmr_static_data(3, 2, 2)
        assert static.c1 == (0, 1, 2)
        assert static.c2 == ((0, 1), (0, 2), (1, 2))
        assert static.c3 == ()
        assert (static.n1, static.n2, static.n3, static.n) == (3, 3, 0, 6)
        assert (static.m1, static.m2, static.m3) == (5, 25, 125)
        assert static.beta2.shape == (25, 2)
        assert static.beta3.shape == (0, 3)  # unused below maxorder=3

        # maxorder=3 fills the order-3 fields, so no name is left untested.
        static3 = _get_hdmr_static_data(3, 3, 2)
        assert static3.c3 == ((0, 1, 2),)
        assert (static3.n1, static3.n2, static3.n3, static3.n) == (3, 3, 1, 7)
        assert (static3.m1, static3.m2, static3.m3) == (5, 25, 125)
        assert static3.beta2.shape == (25, 2)
        assert static3.beta3.shape == (125, 3)
