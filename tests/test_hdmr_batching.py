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
    if full._fit["C2"] is None:
        # maxorder=1 has no second-order coefficients on either side.
        assert batched._fit["C2"] is None
    else:
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
    """The defect this file exists for: both keywords work on every call.

    D15 asked for observable behaviour, not a monkeypatch that counts private
    calls: that pattern breaks the moment the loop structure changes, with no
    change in what the fit computes. ``jax.make_jaxpr`` is the observable
    replacement here. Tracing ``_fit_hdmr`` (a public JAX facility on a
    function this file already calls directly, in ``TestPaddingIsInert``)
    inlines one compiled sub-call per row batch times per slice chunk, so the
    traced program's equation count grows with the number of blocks the fit
    actually ran -- exactly the fact that used to go unverified when
    ``slice_chunk_size`` was silently dropped on one of the two old fit
    paths, because a dropped keyword still returns the right numbers.
    """

    def _fit_kwargs(self, N: int, batch_size: int, slice_chunk_size: int) -> dict:
        from jaxgsa.hdmr._analyze import _get_hdmr_static_data
        from jaxgsa.hdmr._engine import _compute_f_crits

        sd = _get_hdmr_static_data(3, 2, 2)
        return dict(
            m=2,
            maxorder=2,
            maxiter=50,
            lambdax=0.01,
            f_crits=_compute_f_crits(0.95, sd.m1, sd.m2, sd.m3, N),
            c2_idx=jnp.asarray(sd.c2, dtype=int),
            c3_idx=jnp.zeros((0, 3), dtype=int),
            beta2=jnp.asarray(sd.beta2, dtype=int),
            beta3=jnp.asarray(sd.beta3, dtype=int),
            n1=sd.n1,
            n2=sd.n2,
            n3=sd.n3,
            n=sd.n,
            m1=sd.m1,
            m2=sd.m2,
            m3=sd.m3,
            batch_size=batch_size,
            slice_chunk_size=slice_chunk_size,
        )

    def _n_traced_equations(self, X_n, Y_nl, *, batch_size: int, slice_chunk_size: int) -> int:
        kw = self._fit_kwargs(X_n.shape[0], batch_size, slice_chunk_size)
        jaxpr = jax.make_jaxpr(lambda x, y: _fit._fit_hdmr(x, y, **kw))(X_n, Y_nl)
        return len(jaxpr.jaxpr.eqns)

    def test_both_keywords_subdivide_the_work(self, ishigami_data_ntk):
        """batch_size and slice_chunk_size must both add structure to the trace.

        N=1000, T*K=4: one row batch and one slice chunk is the baseline.
        Splitting either axis alone must grow the traced program, and
        splitting both together must grow it further than either alone --
        that composition is exactly what a silently-dropped keyword cannot
        produce, because dropping one axis collapses back to the other
        axis's count.
        """
        from jaxgsa._core.transforms import cdf_to_unit_interval

        X, Y_tk = ishigami_data_ntk
        X_n = cdf_to_unit_interval(X, PROBLEM)
        Y_nl = Y_tk.reshape(Y_tk.shape[0], -1)

        baseline = self._n_traced_equations(X_n, Y_nl, batch_size=1000, slice_chunk_size=4)
        rows_only = self._n_traced_equations(X_n, Y_nl, batch_size=250, slice_chunk_size=4)
        slices_only = self._n_traced_equations(X_n, Y_nl, batch_size=1000, slice_chunk_size=1)
        both = self._n_traced_equations(X_n, Y_nl, batch_size=250, slice_chunk_size=1)

        assert rows_only > baseline, "batch_size alone did not grow the trace"
        assert slices_only > baseline, "slice_chunk_size alone did not grow the trace"
        assert both > rows_only, "slice_chunk_size added nothing once batch_size was set"
        assert both > slices_only, "batch_size added nothing once slice_chunk_size was set"

    def test_slice_chunk_size_alone_subdivides(self, ishigami_data_ntk):
        """slice_chunk_size grows the trace even with a single row batch."""
        from jaxgsa._core.transforms import cdf_to_unit_interval

        X, Y_tk = ishigami_data_ntk
        X_n = cdf_to_unit_interval(X, PROBLEM)
        Y_nl = Y_tk.reshape(Y_tk.shape[0], -1)

        baseline = self._n_traced_equations(X_n, Y_nl, batch_size=1000, slice_chunk_size=4)
        chunked = self._n_traced_equations(X_n, Y_nl, batch_size=1000, slice_chunk_size=2)
        assert chunked > baseline

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


class TestPaddingIsInert:
    """Padded rows and padded lanes contribute nothing to a real result.

    Both loops pad their trailing chunk so a jitted step compiles once. That
    is only sound if the padding cannot reach a real number. Comparing a
    padded run against an unpadded one cannot show this, because the two also
    differ in reduction width, and width alone moves float32 results. Each
    test below therefore holds every width fixed and changes only whether a
    slot holds padding or data.

    Tier T4 (internal consistency).
    """

    def _fit_kwargs(self, N: int, slice_chunk_size: int, batch_size: int) -> dict:
        """Static arguments for a direct ``_fit_hdmr`` call on 3 uniform inputs."""
        from jaxgsa.hdmr._analyze import _get_hdmr_static_data
        from jaxgsa.hdmr._engine import _compute_f_crits

        sd = _get_hdmr_static_data(3, 2, 2)
        return dict(
            m=2,
            maxorder=2,
            maxiter=50,
            lambdax=0.01,
            f_crits=_compute_f_crits(0.95, sd.m1, sd.m2, sd.m3, N),
            c2_idx=jnp.asarray(sd.c2, dtype=int),
            c3_idx=jnp.zeros((0, 3), dtype=int),
            beta2=jnp.asarray(sd.beta2, dtype=int),
            beta3=jnp.asarray(sd.beta3, dtype=int),
            n1=sd.n1,
            n2=sd.n2,
            n3=sd.n3,
            n=sd.n,
            m1=sd.m1,
            m2=sd.m2,
            m3=sd.m3,
            batch_size=batch_size,
            slice_chunk_size=slice_chunk_size,
        )

    @staticmethod
    def _assert_lanes_identical(a, b, n_lanes: int) -> None:
        """Assert the first ``n_lanes`` lanes of two fits agree bit for bit."""
        names = ("Sa", "Sb", "S", "select", "rmse", "C1", "C2", "C3", "f0")
        for name, u, v in zip(names, a[:9], b[:9], strict=True):
            if u is None:
                assert v is None, f"{name}: one fit has it, the other does not"
                continue
            np.testing.assert_array_equal(
                np.asarray(u)[:n_lanes], np.asarray(v)[:n_lanes], err_msg=name
            )

    def _three_lanes(self, X, Y):
        return jnp.stack([Y, jnp.cos(X[:, 1]) + 0.3 * X[:, 0] * X[:, 2], 0.5 * Y + 3.0], axis=1)

    def test_a_padded_lane_does_not_touch_the_real_lanes(self, ishigami_data):
        """Three lanes at chunk width 2: slot 3 is padding, or it is data.

        ``slice_chunk_size=2`` on 3 lanes gives chunks [0:2] and [2:4], so the
        last slot of the second chunk is padding. Running 4 real lanes at the
        same chunk size gives the same two widths with that slot filled. Lanes
        0-2 must not notice.
        """
        from jaxgsa._core.transforms import cdf_to_unit_interval
        from jaxgsa.hdmr._fit import _fit_hdmr

        X, Y = ishigami_data
        X_n = cdf_to_unit_interval(X, PROBLEM)
        kw = self._fit_kwargs(X.shape[0], slice_chunk_size=2, batch_size=X.shape[0])
        lanes3 = self._three_lanes(X, Y)
        extra = (jnp.sin(X[:, 0]) * X[:, 2] - 7.0)[:, None]

        padded = _fit_hdmr(X_n, lanes3, **kw)
        filled = _fit_hdmr(X_n, jnp.concatenate([lanes3, extra], axis=1), **kw)
        self._assert_lanes_identical(padded, filled, 3)

    def test_padded_row_content_cannot_reach_an_accumulator(self, ishigami_data, monkeypatch):
        """Fill the row padding with 1e6 instead of 0 and change nothing.

        Every accumulator contracts against a basis the row mask has already
        zeroed, or against a residual the mask zeroes directly. So the value
        sitting in a padded row is unobservable, and the strongest way to say
        that is to put a number in it that would be impossible to miss.
        """
        from jaxgsa._core.transforms import cdf_to_unit_interval
        from jaxgsa.hdmr import _fit as fit_mod

        X, Y = ishigami_data
        X_n = cdf_to_unit_interval(X, PROBLEM)
        # 1000 rows in batches of 300: the trailing batch pads 200 rows.
        kw = self._fit_kwargs(X.shape[0], slice_chunk_size=3, batch_size=300)
        lanes3 = self._three_lanes(X, Y)

        zero_padded = fit_mod._fit_hdmr(X_n, lanes3, **kw)

        def loud_pad(A, n_pad):
            if n_pad == 0:
                return A
            junk = jnp.full((n_pad, *A.shape[1:]), 1.0e6, dtype=A.dtype)
            return jnp.concatenate([A, junk], axis=0)

        monkeypatch.setattr(fit_mod, "_pad_rows", loud_pad)
        loud_padded = fit_mod._fit_hdmr(X_n, lanes3, **kw)
        self._assert_lanes_identical(zero_padded, loud_padded, 3)
