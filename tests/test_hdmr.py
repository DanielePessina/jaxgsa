"""Tests for RS-HDMR sensitivity analysis."""

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks.ishigami import (
    ANALYTICAL_S1,
    ANALYTICAL_ST,
    PROBLEM,
    evaluate,
)
from jaxgsa.hdmr import analyze as analyze_hdmr
from jaxgsa.problem import GaussianInputSpec, Problem
from jaxgsa.sampling import monte_carlo

# One Ishigami data set and one fit per ``order``, shared with
# test_hdmr_batching.py so the whole suite pays for each fit once.


# HDMR builds polynomial surrogates that approximate the true function, so
# tolerances (~30% for S1) are wider than for Sobol sampling methods which
# converge to exact values with sufficient samples.
def _assert_matches_analytical_s1_st(S1: np.ndarray, ST: np.ndarray) -> None:
    """Assert HDMR first- and total-order indices track Ishigami analytics."""
    for i, expected in enumerate(ANALYTICAL_S1):
        if expected == 0.0:
            assert abs(S1[i]) < 0.1, f"S1[{i}] = {S1[i]}, expected ~0"
        else:
            rel_err = abs(S1[i] - expected) / abs(expected)
            assert rel_err < 0.30, (
                f"S1[{i}] = {S1[i]:.4f}, expected {expected}, rel_err={rel_err:.3f}"
            )

    for i, expected in enumerate(ANALYTICAL_ST):
        rel_err = abs(ST[i] - expected) / max(abs(expected), 0.01)
        assert rel_err < 0.25, f"ST[{i}] = {ST[i]:.4f}, expected {expected}, rel_err={rel_err:.3f}"


@pytest.fixture(scope="module")
def hdmr_result(hdmr_by_order):
    """The ``order=2`` fit of the shared Ishigami data."""
    return hdmr_by_order[2]


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------


def test_st_accuracy(hdmr_result):
    """Indices approximate the analytical values, in the documented shapes.

    Y shape (N,) gives term-axis shapes ``(n_terms,)`` and index shapes ``(D,)``.
    """
    ST = np.array(hdmr_result.ST)
    _assert_matches_analytical_s1_st(np.array(hdmr_result.S1), ST)

    D = PROBLEM.n_params
    # First-order terms (D) + second-order interactions C(D,2) = D(D-1)/2.
    n_terms = D + D * (D - 1) // 2
    assert hdmr_result.Sa.shape == (n_terms,)
    assert hdmr_result.Sb.shape == (n_terms,)
    assert hdmr_result.S.shape == (n_terms,)
    assert hdmr_result.ST.shape == (D,)
    assert hdmr_result.rmse is not None
    assert hdmr_result.rmse.shape == ()


def test_hdmr_accepts_gaussian_inputs():
    """HDMR should accept mixed uniform/Gaussian problems via CDF mapping."""
    problem = PROBLEM.from_dict(
        {
            "x1": (0.0, 1.0),
            "x2": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
            "x3": (0.0, 1.0),
        }
    )
    key = jax.random.PRNGKey(99)
    N = 500
    x1 = jax.random.uniform(key, (N,), minval=0.0, maxval=1.0)
    x2 = jax.random.normal(jax.random.PRNGKey(100), (N,))
    x3 = jax.random.uniform(jax.random.PRNGKey(101), (N,), minval=0.0, maxval=1.0)
    X = jnp.column_stack([x1, x2, x3])
    Y = X[:, 0] + X[:, 1] ** 2 + X[:, 2]
    result = analyze_hdmr(problem, X, Y, order=1)
    assert result.S1.shape == (3,)
    assert result.ST.shape == (3,)
    # x2 ~ N(0, 1) enters as x2**2 (variance 2); each uniform adds only 1/12.
    # The spline basis under-resolves the Gaussian tails, so the share reads
    # about 0.7 and not the exact 0.92. It still dominates.
    assert int(np.argmax(np.asarray(result.S1))) == 1
    assert float(result.S1[1]) > 0.5


def test_indices_are_affine_invariant(hdmr_ishigami_xy, hdmr_result):
    """HDMR indices are variance ratios, so ``Y -> a*Y + b`` must not move them.

    Tier T0 (closed form): an ANCOVA index is a ratio of two quantities that
    both scale by ``a**2`` and both ignore ``b``, so invariance is exact up to
    the fit's own conditioning. No keyword buys this and none can switch it
    off.

    The tolerance is 1e-5 rather than float32 epsilon because the fit solves
    normal equations: it accumulates ``B^T B`` and ``B^T Y``, which squares
    the basis' condition number, so the offset ``b`` costs a few low bits.
    Under ``jax_enable_x64`` the same comparison holds to 8e-15, which is what
    says the residue is conditioning and not a term that fails to cancel.
    """
    X, Y = hdmr_ishigami_xy
    Y_affine = 3.0 * Y + 17.0

    base = hdmr_result
    affine = analyze_hdmr(PROBLEM, X, Y_affine, order=2, n_intervals=2)

    np.testing.assert_allclose(np.asarray(base.S1), np.asarray(affine.S1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(base.ST), np.asarray(affine.ST), rtol=1e-5, atol=1e-5)


def _correlated_additive_case():
    """A correlated additive model: N = 1500, rho(x1, x2) = 0.7."""
    rng = np.random.default_rng(0)
    N = 1500
    corr = np.array([[1.0, 0.7, 0.3], [0.7, 1.0, 0.0], [0.3, 0.0, 1.0]])
    problem = Problem(("a", "b", "c"), ((0.0, 1.0),) * 3, correlation=corr)
    Z = rng.multivariate_normal(np.zeros(3), corr, size=N)
    from scipy.stats import norm

    X = jnp.asarray(norm.cdf(Z))
    Y = X[:, 0] + 2.0 * X[:, 1] ** 2 + jnp.sin(3.0 * X[:, 2])
    return problem, X, Y


def test_backfitting_stop_rule_does_not_depend_on_y_scale_or_m():
    """H4 regression: the backfit stop rule is relative, not absolute.

    ``3.0 * Y + 17.0`` above is too large a shift to trigger the old bug: the
    absolute threshold only stopped a correlated fit early when the
    coefficients were small, which happens for a small ``Y`` scale or a
    large basis count ``m`` (the basis is scaled by ``m**3``, so the
    coefficients shrink as ``m**-3``). This reproduces the review's case: a
    correlated additive model, N = 1500, rho(x1, x2) = 0.7. Before the fix,
    ``S1[0]`` read 0.014-0.066 instead of ~0.104 for ``m in (4, 8)`` and for
    ``Y * 1e-3``. This is the cheap case (``m = 4`` at scale 1e-3, one of the
    configurations the old bug broke); the full grid is the slow test below.
    """
    problem, X, Y = _correlated_additive_case()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s1_ref = np.asarray(analyze_hdmr(problem, X, Y, n_intervals=2, max_iter=100).S1)
        small = analyze_hdmr(problem, X, Y * 1e-3, n_intervals=4, max_iter=100)
    np.testing.assert_allclose(np.asarray(small.S1), s1_ref, atol=0.02)


@pytest.mark.slow
def test_backfitting_stop_rule_full_scale_and_m_grid():
    """H4 regression, full grid: ``m`` in (4, 8) times three ``Y`` scales."""
    problem, X, Y = _correlated_additive_case()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s1_ref = np.asarray(analyze_hdmr(problem, X, Y, n_intervals=2, max_iter=100).S1)
        for m in (4, 8):
            for scale in (1e-3, 1.0, 1e3):
                result = analyze_hdmr(problem, X, Y * scale, n_intervals=m, max_iter=100)
                np.testing.assert_allclose(np.asarray(result.S1), s1_ref, atol=0.02)


# ---------------------------------------------------------------------------
# order tests
# ---------------------------------------------------------------------------


def test_order_selects_the_term_count(hdmr_by_order):
    """For D=3 the term count is 3, 6 and 7 at order 1, 2 and 3."""
    assert PROBLEM.n_params == 3
    for order, n_terms in ((1, 3), (2, 6), (3, 7)):
        result = hdmr_by_order[order]
        assert result.Sa.shape == (n_terms,)
        assert len(result.terms) == n_terms


# ---------------------------------------------------------------------------
# Emulator tests
# ---------------------------------------------------------------------------


def test_multi_output_analytical_ishigami_emulator(hdmr_ishigami_xy):
    """Multi-output HDMR should keep per-output coefficients and predictions."""
    X, Y = hdmr_ishigami_xy
    Y_multi = jnp.stack([Y, 2.0 * Y], axis=1)

    result = analyze_hdmr(PROBLEM, X, Y_multi, order=2, n_intervals=2)
    Y_pred = result.predict(X)

    assert result._fit is not None
    assert result._fit["C1"].shape == (2, 5, 3)
    assert result._fit["C2"] is not None
    assert result._fit["C2"].shape == (2, 25, 3)
    assert result._fit["f0"].shape == (2,)
    assert Y_pred.shape == Y_multi.shape

    np.testing.assert_allclose(result.S1[0], result.S1[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.ST[0], result.ST[1], rtol=1e-6, atol=1e-6)
    _assert_matches_analytical_s1_st(np.array(result.S1[0]), np.array(result.ST[0]))
    _assert_matches_analytical_s1_st(np.array(result.S1[1]), np.array(result.ST[1]))

    rmse = np.sqrt(np.mean(np.square(np.array(Y_pred) - np.array(Y_multi)), axis=0))
    assert rmse[0] < 1.5, f"RMSE[0] = {rmse[0]:.3f}, expected < 1.5"
    assert rmse[1] < 3.0, f"RMSE[1] = {rmse[1]:.3f}, expected < 3.0"
    assert rmse[1] > rmse[0]
    assert not np.allclose(np.array(Y_pred[:, 0]), np.array(Y_pred[:, 1]))


def test_time_series_multi_output_emulator_preserves_axes(hdmr_ishigami_xy):
    """Time-series HDMR emulator should preserve both output axes."""
    X, Y = hdmr_ishigami_xy
    Y_tk = jnp.stack(
        [
            jnp.stack([Y, 2.0 * Y], axis=1),
            jnp.stack([0.5 * Y, -1.5 * Y], axis=1),
        ],
        axis=1,
    )

    result = analyze_hdmr(PROBLEM, X, Y_tk, order=2, n_intervals=2)
    Y_pred = result.predict(X)

    assert result._fit is not None
    assert result._fit["C1"].shape == (2, 2, 5, 3)
    assert result._fit["C2"] is not None
    assert result._fit["C2"].shape == (2, 2, 25, 3)
    assert result._fit["f0"].shape == (2, 2)
    assert Y_pred.shape == Y_tk.shape

    rmse = np.sqrt(np.mean(np.square(np.array(Y_pred) - np.array(Y_tk)), axis=0))
    assert np.all(rmse < np.array([[1.5, 3.0], [0.8, 2.3]])), rmse
    assert not np.allclose(np.array(Y_pred[:, 0, 0]), np.array(Y_pred[:, 0, 1]))
    assert not np.allclose(np.array(Y_pred[:, 0, 0]), np.array(Y_pred[:, 1, 0]))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_constant_output_with_float32_rounding_noise_still_reads_as_zero_variance():
    """A constant Y whose naive sample variance is not bit-exact zero still warns.

    Regression for M1 (REVIEW-1.0.md): the old guard tested ``var == 0``,
    which almost never holds for a constant float32 output because the mean
    itself rounds. ``Y = full(N, 0.1)`` has a naive sample variance of about
    2e-16, not zero, so this case would have slipped past the old guard.
    """
    N = 500
    D = PROBLEM.n_params
    key = jax.random.PRNGKey(99)
    bounds = jnp.array(PROBLEM.bounds)
    X = jax.random.uniform(key, shape=(N, D), minval=bounds[:, 0], maxval=bounds[:, 1])
    Y = jnp.full(N, 0.1)
    assert float(jnp.var(Y)) != 0.0, "the naive variance must be nonzero for this to test the fix"
    with pytest.warns(UserWarning, match="zero variance"):
        result = analyze_hdmr(PROBLEM, X, Y, order=2, n_intervals=2)
    assert jnp.all(jnp.isnan(result.Sa))
    assert jnp.all(jnp.isnan(result.ST))


def test_validation_errors():
    """Input validation should raise on bad inputs."""
    # Order 2, n_intervals 2: the largest term has 5**2 = 25 coefficients, so
    # the floor is 26 rows. Below it the fit is not computable.
    X = jax.random.uniform(jax.random.key(0), (25, 3))
    Y = jnp.ones(25)

    with pytest.raises(ValueError, match=r"jaxgsa\.hdmr: N=25 rows is too few"):
        analyze_hdmr(PROBLEM, X, Y)

    X = jnp.ones((500, 3))
    Y = jnp.ones(500)

    with pytest.raises(ValueError, match="order"):
        analyze_hdmr(PROBLEM, X, Y, order=4)

    with pytest.raises(ValueError, match="slice_chunk_size"):
        analyze_hdmr(PROBLEM, X, Y, slice_chunk_size=0)

    with pytest.raises(ValueError, match="batch_size"):
        analyze_hdmr(PROBLEM, X, Y, batch_size=0)

    # 0.4.0 rename: the old kwarg is gone, no shim. (Passed via an Any-typed
    # dict so the static type-checker does not flag the intentionally-invalid
    # name; the TypeError is the runtime contract under test.)
    legacy_kwargs: dict[str, Any] = {"chunk_size": 1}
    with pytest.raises(TypeError):
        analyze_hdmr(PROBLEM, X, Y, **legacy_kwargs)


def test_fit_scalars_rejected_in_analyze_and_indices():
    """T4: n_intervals, max_iter and ridge are validated, in both entry points.

    Each of the three used to reach the fit unchecked and degrade it
    silently. Both ``analyze`` and ``indices`` run the same checks (one
    definition, ``_fit_scalar_checks``), so the two cannot drift.
    """
    from jaxgsa.hdmr import indices as indices_hdmr

    X = jnp.ones((500, 3))
    Y = jnp.ones(500)
    cases = [
        ({"n_intervals": 0}, "n_intervals must be >= 1"),
        ({"n_intervals": -1}, "n_intervals must be >= 1"),
        ({"max_iter": 0}, "max_iter must be >= 1"),
        ({"ridge": -1.0}, "ridge must be >= 0"),
    ]
    for kwargs, match in cases:
        with pytest.raises(ValueError, match=match):
            analyze_hdmr(PROBLEM, X, Y, **kwargs)
        with pytest.raises(ValueError, match=match):
            indices_hdmr(PROBLEM, X, Y, **kwargs)


def test_fit_scalars_valid_edges_pass(hdmr_ishigami_xy):
    """T4: the smallest legal values fit without an error.

    ``n_intervals=1`` is the minimal cubic basis (one knot interval, four functions),
    ``max_iter=1`` is one backfitting sweep, ``ridge=0.0`` is an
    unregularized solve. All three are degraded-but-defined, so they must
    run, not raise.
    """
    X, Y = hdmr_ishigami_xy
    result = analyze_hdmr(PROBLEM, X, Y, order=1, n_intervals=1, max_iter=1, ridge=0.0)
    assert np.all(np.isfinite(np.asarray(result.S1)))


# ---------------------------------------------------------------------------
# S2 / S3 interaction-index properties
# ---------------------------------------------------------------------------


def _term_index(result, label: str) -> int:
    """Return the position of a term label along the term axis."""
    return result.terms.index(label)


def test_s2_property_shape_and_symmetry(hdmr_result):
    """S2 is symmetric, indexed by terms, and has no diagonal."""
    D = PROBLEM.n_params
    S2 = np.array(hdmr_result.S2)
    assert S2.shape == (D, D)
    # Diagonal has no pairwise term, so it stays NaN.
    assert np.all(np.isnan(np.diag(S2)))
    # Off-diagonal pairs are populated (Ishigami, order=2) and symmetric.
    offdiag = S2[~np.eye(D, dtype=bool)]
    assert np.all(np.isfinite(offdiag))
    np.testing.assert_array_equal(S2, S2.T)
    Sa = np.array(hdmr_result.Sa)
    for i in range(D):
        for j in range(i + 1, D):
            k = _term_index(hdmr_result, f"{PROBLEM.names[i]}/{PROBLEM.names[j]}")
            assert S2[i, j] == Sa[k]
            assert S2[j, i] == Sa[k]
    # x1 (index 0) and x3 (index 2) carry Ishigami's only real interaction.
    assert S2[0, 2] > S2[0, 1]
    assert S2[0, 2] > S2[1, 2]


def test_s2_s3_all_nan_when_order_1(hdmr_by_order):
    """With no interaction terms, S2 and S3 are entirely NaN."""
    D = PROBLEM.n_params
    result = hdmr_by_order[1]
    assert result.S2.shape == (D, D)
    assert bool(jnp.all(jnp.isnan(result.S2)))
    assert result.S3.shape == (D, D, D)
    assert bool(jnp.all(jnp.isnan(result.S3)))


def test_s3_property_order_3(hdmr_by_order):
    """S3 is a symmetric (D, D, D) tensor matching the triple's Sa entry."""
    names = PROBLEM.names
    D = PROBLEM.n_params
    result = hdmr_by_order[3]
    S3 = np.array(result.S3)
    Sa = np.array(result.Sa)
    assert S3.shape == (D, D, D)

    # Cells with any repeated axis have no term and stay NaN.
    for i in range(D):
        for j in range(D):
            assert np.isnan(S3[i, i, j])
            assert np.isnan(S3[i, j, i])
            assert np.isnan(S3[j, i, i])

    # The single D=3 triple (x1/x2/x3) is populated symmetrically.
    k = _term_index(result, "/".join(names))
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        assert S3[perm] == Sa[k]


# ---------------------------------------------------------------------------
# Correlated inputs: the SCSA reading of ST
# ---------------------------------------------------------------------------


def _correlated_problem(rho: float = 0.7):
    """Build a 3-parameter uniform problem with x1 and x2 correlated."""
    R = np.eye(3)
    R[0, 1] = R[1, 0] = rho
    return Problem(("x1", "x2", "x3"), ((-np.pi, np.pi),) * 3, correlation=R)


def _linear_interaction(X):
    """Model that uses x1 and x2 (with an interaction) and ignores x3."""
    return X[:, 0] + 2.0 * X[:, 1] + 0.5 * X[:, 0] * X[:, 1]


@pytest.fixture(scope="module")
def correlated_hdmr_fit():
    """Fit HDMR once on a correlated problem: ``(result, recorded warnings)``."""
    problem = _correlated_problem()
    X = jnp.asarray(monte_carlo(problem, 1000, seed=7))
    Y = _linear_interaction(X)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analyze_hdmr(problem, X, Y, order=2, n_intervals=2)
    return result, caught


def test_st_is_scsa_total_under_correlation(correlated_hdmr_fit):
    """ST is sum over terms u containing i of (Sa_u + Sb_u), not a Sobol total.

    Pins the SCSA convention that Li et al. (2010) define in Section 2.2.3,
    which is also the documented HDMR convention. Recomputed here from the per-term
    fields so a change of convention cannot pass silently.
    """
    result, _ = correlated_hdmr_fit
    D = result.problem.n_params

    def _scatter(per_term):
        out = np.zeros(D)
        for term_index, label in enumerate(result.terms):
            for name in label.split("/"):
                out[result.problem.names.index(name)] += per_term[term_index]
        return out

    ST = np.array(result.ST)
    # Exact: ST scatters the per-term total S onto its participating params.
    np.testing.assert_allclose(ST, _scatter(np.array(result.S)), rtol=1e-5, atol=1e-7)
    # M8: S is measured against the fitted expansion (Li et al., 2010), so
    # S = Sa + Sb is an identity, not an approximation. This is float32
    # noise only now, not a few-percent gap from a different output convention
    # under correlation.
    np.testing.assert_allclose(
        np.array(result.S), np.array(result.Sa) + np.array(result.Sb), atol=1e-6
    )
    np.testing.assert_allclose(ST, _scatter(np.array(result.Sa) + np.array(result.Sb)), atol=1e-6)
    # The correlative shares are what make this differ from the structural
    # sum; if they were negligible the test would not be pinning anything.
    assert np.abs(np.array(result.Sb)).max() > 1e-3
    assert not np.allclose(ST, _scatter(np.array(result.Sa)), atol=1e-2)


def test_correlated_problem_warns_once(correlated_hdmr_fit):
    """analyze emits exactly one SCSA warning per call on a correlated problem."""
    _, caught = correlated_hdmr_fit
    scsa = [w for w in caught if "SCSA" in str(w.message)]
    assert len(scsa) == 1
    assert issubclass(scsa[0].category, UserWarning)
    message = str(scsa[0].message)
    assert "kucherenko" in message
    assert "vkoga" in message


# ---------------------------------------------------------------------------
# on_invalid policy
# ---------------------------------------------------------------------------


class TestHDMROnInvalid:
    """T4: ``hdmr.analyze`` applies the shared non-finite policy.

    The check lives in the public wrapper only, next to the zero-variance and
    SCSA warnings and for the same reason: ``_analyze_hdmr_core`` must stay
    free of it so no caller can trigger the policy twice.
    """

    HDMR_KWARGS: dict[str, Any] = dict(order=2, n_intervals=2)

    def test_raise_is_the_default_and_names_the_rows(self, hdmr_ishigami_xy):
        """T4: a non-finite Y refuses the analysis and says which rows carry it."""
        X, Y = hdmr_ishigami_xy
        Y = np.asarray(Y).copy()
        Y[3] = np.nan
        Y[500] = -np.inf
        with pytest.raises(ValueError, match="non-finite") as exc:
            analyze_hdmr(PROBLEM, X, Y, **self.HDMR_KWARGS)
        message = str(exc.value)
        assert "jaxgsa.hdmr.analyze" in message
        assert f"2 of {len(X)} rows" in message
        assert "[3, 500]" in message

    def test_propagate_warns_and_lets_the_nan_reach_the_indices(self, hdmr_ishigami_xy):
        """T4: 'propagate' keeps every row, so the fitted indices go non-finite."""
        X, Y = hdmr_ishigami_xy
        Y = np.asarray(Y).copy()
        Y[3] = np.nan
        with pytest.warns(JaxgsaWarning, match="reaches the indices"):
            result = analyze_hdmr(PROBLEM, X, Y, on_invalid="propagate", **self.HDMR_KWARGS)
        assert not np.all(np.isfinite(np.array(result.ST)))
        assert result.invalid.policy == "propagate"
        assert result.invalid.n_invalid == 1

    def test_drop_removes_the_rows_and_the_fit_is_finite_again(self, hdmr_ishigami_xy):
        """T4: 'drop' fits on the survivors, so every index is finite."""
        X, Y = hdmr_ishigami_xy
        Y = np.asarray(Y).copy()
        Y[3] = np.nan
        with pytest.warns(JaxgsaWarning, match=f"dropped 1 of {len(X)} rows"):
            result = analyze_hdmr(PROBLEM, X, Y, on_invalid="drop", **self.HDMR_KWARGS)
        assert np.all(np.isfinite(np.array(result.ST)))
        assert result.invalid.unit_indices == (3,)
        assert result.invalid.sources == ("Y",)
        _assert_matches_analytical_s1_st(np.array(result.S1), np.array(result.ST))

    def test_a_non_finite_input_is_caught_too(self, hdmr_ishigami_xy):
        """T4: X is checked as well as Y, and ``sources`` says which array.

        A NaN input reaches the CDF transform and then the B-spline basis, so
        it is no less fatal than a NaN output.
        """
        X, Y = hdmr_ishigami_xy
        X = np.asarray(X).copy()
        X[17, 1] = np.inf
        with pytest.raises(ValueError, match=r"in X\.") as exc:
            analyze_hdmr(PROBLEM, X, Y, **self.HDMR_KWARGS)
        assert "[17]" in str(exc.value)
        with pytest.warns(JaxgsaWarning):
            result = analyze_hdmr(PROBLEM, X, Y, on_invalid="drop", **self.HDMR_KWARGS)
        assert result.invalid.sources == ("X",)


class TestBootstrapIntervals:
    """The opt-in confidence intervals.

    Tier T4 throughout. A bootstrap interval has no closed form to check
    against; what these pin is that the interval brackets its estimate, that
    it costs nothing when nobody asks for it, and that the shared interval
    vocabulary is honoured here too.
    """

    @staticmethod
    def _data(n: int = 400, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = jnp.asarray(rng.uniform(-np.pi, np.pi, size=(n, 3)))
        return X, evaluate(X)

    def test_no_bootstrap_by_default(self, hdmr_result):
        """T4: the plainest call reports no interval and pays for none.

        Each replicate refits the whole B-spline expansion, so an
        on-by-default interval would make a routine call an order of
        magnitude slower.
        """
        assert hdmr_result.ci is None
        assert hdmr_result.Sa_conf is None
        assert hdmr_result.Sb_conf is None
        assert hdmr_result.S_conf is None
        assert hdmr_result.ST_conf is None

    def test_the_interval_brackets_the_point_estimate_and_exports(self):
        """T4: lower <= upper, the estimate sits near, and the export splits it.

        A five-draw percentile interval need not contain the estimate, so the
        estimate may sit outside by at most one interval width.
        """
        X, Y = self._data()
        result = analyze_hdmr(
            PROBLEM, X, Y, order=2, max_iter=50, n_bootstrap=5, key=jax.random.key(0)
        )

        for point, conf in (
            (result.Sa, result.Sa_conf),
            (result.Sb, result.Sb_conf),
            (result.S, result.S_conf),
            (result.ST, result.ST_conf),
        ):
            lower, upper = np.asarray(conf[0]), np.asarray(conf[1])
            point = np.asarray(point)
            finite = np.isfinite(point) & np.isfinite(lower) & np.isfinite(upper)
            assert (lower[finite] <= upper[finite]).all()
            width = upper[finite] - lower[finite]
            assert width.max() > 0.0
            assert (point[finite] >= lower[finite] - width - 0.01).all()
            assert (point[finite] <= upper[finite] + width + 0.01).all()

        dataset = result.to_dataset()
        for name in ("Sa", "Sb", "S", "ST"):
            assert f"{name}_lower" in dataset
            assert f"{name}_upper" in dataset

    def test_ci_metadata_and_replicates_are_recorded(self):
        """The interval records its recipe and optionally exposes its draws."""
        X, Y = self._data()
        result = analyze_hdmr(
            PROBLEM,
            X,
            Y,
            order=2,
            max_iter=50,
            n_bootstrap=4,
            conf_level=0.9,
            ci_method="gaussian",
            key=jax.random.key(2),
            keep_replicates=True,
        )

        assert result.ci is not None
        assert (result.ci.level, result.ci.method, result.ci.n_bootstrap) == (0.9, "gaussian", 4)
        assert result.ci.replicates is not None
        assert result.ci.replicates["Sa"].shape == (4, 6)
        assert result.ci.replicates["ST"].shape == (4, 3)

    def test_a_key_is_required(self):
        """T4: no key means no interval, and it is refused rather than seeded."""
        X, Y = self._data()
        with pytest.raises(ValueError, match="key is required"):
            analyze_hdmr(PROBLEM, X, Y, order=2, max_iter=50, n_bootstrap=3)


# ---------------------------------------------------------------------------
# Decomposition completeness
# ---------------------------------------------------------------------------


class TestUnexplainedVarianceWarning:
    """Li et al. (2010, Eq. 24) require the per-term S to sum to about 1.

    Below that, the shortfall is variance the decomposition never captured,
    and Sa, Sb, S and ST all inherit it. Nothing in the index arrays says so,
    which is why analyze reads the sum and speaks.
    """

    @staticmethod
    def _oscillatory():
        """Data a first-order, coarse-basis HDMR cannot follow."""
        problem = Problem(("x1", "x2"), ((0.0, 1.0), (0.0, 1.0)))
        X = jnp.asarray(monte_carlo(problem, 400, seed=0))
        Y = jnp.sin(30.0 * X[:, 0]) + jnp.sin(30.0 * X[:, 1])
        return problem, X, Y

    def test_shortfall_warns_and_names_the_sum(self):
        """A decomposition that captures 7% of the variance says so."""
        problem, X, Y = self._oscillatory()
        with pytest.warns(JaxgsaWarning, match=r"result\.S\.sum\(\) is 0\.\d+"):
            result = analyze_hdmr(problem, X, Y, order=1, n_intervals=2, verbose=False)
        assert float(np.asarray(result.S).sum()) < 0.5

    def test_a_complete_decomposition_is_silent(self, hdmr_result):
        """A fit whose terms account for the variance raises nothing.

        The shared order-2 fit is built with every ``JaxgsaWarning`` promoted
        to an error (see ``_fixtures_dependence.hdmr_by_order``), so reaching this
        test already proves it was silent.
        """
        assert 0.5 <= float(np.asarray(hdmr_result.S).sum()) <= 1.3


def test_below_300_rows_warns_instead_of_raising():
    """N5: between the computable floor and 300 rows the fit runs and warns."""
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.uniform(-np.pi, np.pi, (150, 3)))
    Y = jnp.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    with pytest.warns(JaxgsaWarning, match=r"jaxgsa\.hdmr: N=150 rows is below the 300"):
        result = analyze_hdmr(PROBLEM, X, Y, verbose=False)
    assert np.isfinite(np.asarray(result.S1)).all()


def test_drop_below_floor_refuses_in_the_policy_words():
    """N5: on_invalid="drop" refuses once the survivors fall under the floor."""
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.uniform(-np.pi, np.pi, (60, 3)))
    Y = np.array(jnp.sin(X[:, 0]))
    Y[:40] = np.nan
    with pytest.raises(ValueError, match="every usable row was removed"):
        analyze_hdmr(PROBLEM, X, jnp.asarray(Y), on_invalid="drop", verbose=False)


def test_invalid_order_is_refused_before_the_row_floor_with_drop():
    X = monte_carlo(PROBLEM, 64, seed=0)
    Y = np.asarray(X).sum(axis=1)
    with pytest.raises(ValueError, match="order must be 1, 2, or 3"):
        analyze_hdmr(PROBLEM, X, Y, order=5, on_invalid="drop")


@pytest.mark.parametrize("kind", ["duplicate", "constant", "affine", "sum"])
def test_degenerate_input_raises(kind):
    """A constant, duplicated or linearly dependent input is refused, not split silently."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (500, 3)).astype(np.float32)
    Y = 2 * X[:, 0] + X[:, 1] ** 2
    if kind == "duplicate":
        X[:, 2] = X[:, 0]
    elif kind == "constant":
        X[:, 2] = 0.5
    elif kind == "affine":
        X[:, 2] = 1.0 - 0.5 * X[:, 0]
    else:
        X[:, 2] = (X[:, 0] + X[:, 1]) / 2
    problem = Problem(["a", "b", "c"], [(0, 1)] * 3)
    with pytest.raises(ValueError, match="jaxgsa.hdmr"):
        analyze_hdmr(problem, X, Y, verbose=False)
