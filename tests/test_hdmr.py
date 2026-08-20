"""Tests for RS-HDMR sensitivity analysis."""

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning, kucherenko
from jaxgsa._core.invalid import InvalidReport, InvalidUnit
from jaxgsa.benchmarks.ishigami import (
    ANALYTICAL_S1,
    ANALYTICAL_ST,
    PROBLEM,
    evaluate,
)
from jaxgsa.hdmr import analyze as analyze_hdmr
from jaxgsa.problem import GaussianInputSpec, Problem
from jaxgsa.sampling import monte_carlo


def _clean_report(n_units: int = 0) -> InvalidReport:
    """Build the empty non-finite report a hand-made result needs."""
    return InvalidReport(
        policy="raise",
        unit=InvalidUnit.ROW,
        n_units=n_units,
        n_invalid=0,
        unit_indices=(),
        row_indices=(),
        bad_row_indices=(),
        sources=(),
    )


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
def ishigami_data():
    """Generate random Ishigami (X, Y) data for HDMR tests."""
    key = jax.random.PRNGKey(42)
    N = 2000
    bounds = jnp.array(PROBLEM.bounds)
    X = jax.random.uniform(key, shape=(N, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
    Y = evaluate(X)
    return X, Y


@pytest.fixture(scope="module")
def hdmr_result(ishigami_data):
    """Run HDMR analysis once for all tests."""
    X, Y = ishigami_data
    return analyze_hdmr(
        PROBLEM,
        X,
        Y,
        maxorder=2,
        m=2,
    )


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------


def test_st_accuracy(hdmr_result):
    """Total-order indices should approximate analytical values."""
    ST = np.array(hdmr_result.ST)
    _assert_matches_analytical_s1_st(np.array(hdmr_result.S1), ST)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_shapes_1d(ishigami_data):
    """Y shape (N,) -> index shapes (n_terms,) and (D,)."""
    X, Y = ishigami_data
    result = analyze_hdmr(
        PROBLEM,
        X,
        Y,
        maxorder=2,
        m=2,
    )
    D = PROBLEM.num_vars
    # First-order terms (D) + second-order interactions C(D,2) = D(D-1)/2.
    n_terms = D + D * (D - 1) // 2
    assert result.Sa.shape == (n_terms,)
    assert result.Sb.shape == (n_terms,)
    assert result.S.shape == (n_terms,)
    assert result.ST.shape == (D,)
    assert result.rmse is not None
    assert result.rmse.shape == ()


def test_shapes_2d(ishigami_data):
    """Y shape (N, K) -> index shapes (K, n_terms) and (K, D)."""
    X, Y = ishigami_data
    K = 2
    Y_2d = jnp.stack([Y, Y * 0.5], axis=1)  # (N, 2)
    result = analyze_hdmr(
        PROBLEM,
        X,
        Y_2d,
        maxorder=2,
        m=2,
    )
    D = PROBLEM.num_vars
    n_terms = D + D * (D - 1) // 2
    assert result.Sa.shape == (K, n_terms)
    assert result.ST.shape == (K, D)
    assert result.rmse is not None
    assert result.rmse.shape == (K,)
    assert result._fit is not None
    assert result._fit["C1"].shape == (K, 5, 3)
    assert result._fit["C2"] is not None
    assert result._fit["C2"].shape == (K, 25, 3)
    assert result._fit["C3"] is None
    assert result._fit["f0"].shape == (K,)


def test_shapes_3d(ishigami_data):
    """Y shape (N, T, K) -> index shapes (T, K, n_terms) and (T, K, D)."""
    X, Y = ishigami_data
    T, K = 2, 3
    Y_3d = jnp.broadcast_to(Y[:, None, None], (Y.shape[0], T, K))
    result = analyze_hdmr(
        PROBLEM,
        X,
        Y_3d,
        maxorder=2,
        m=2,
    )
    D = PROBLEM.num_vars
    n_terms = D + D * (D - 1) // 2
    assert result.Sa.shape == (T, K, n_terms)
    assert result.ST.shape == (T, K, D)
    assert result.rmse is not None
    assert result.rmse.shape == (T, K)
    assert result._fit is not None
    assert result._fit["C1"].shape == (T, K, 5, 3)
    assert result._fit["C2"] is not None
    assert result._fit["C2"].shape == (T, K, 25, 3)
    assert result._fit["C3"] is None
    assert result._fit["f0"].shape == (T, K)


def test_slice_chunk_size_regression(ishigami_data):
    """Chunked and unchunked HDMR paths should agree for multi-output Y."""
    X, Y = ishigami_data
    Y_2d = jnp.stack([Y, Y * 0.5], axis=1)

    result_default = analyze_hdmr(PROBLEM, X, Y_2d, maxorder=2, m=2)
    result_chunked = analyze_hdmr(
        PROBLEM,
        X,
        Y_2d,
        maxorder=2,
        m=2,
        slice_chunk_size=1,
    )

    # Chunked and unchunked runs are the same code at two vmap widths, so
    # they differ only by floating-point reduction order. The tolerance
    # absorbs that (and cross-version jax rounding drift) while still
    # catching a genuine chunking bug. Under ``jax_enable_x64`` the two agree
    # to 7e-15.
    np.testing.assert_allclose(
        np.asarray(result_default.Sa), np.asarray(result_chunked.Sa), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(result_default.Sb), np.asarray(result_chunked.Sb), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(result_default.S), np.asarray(result_chunked.S), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(result_default.ST), np.asarray(result_chunked.ST), rtol=1e-5, atol=1e-5
    )
    assert result_default.rmse is not None
    assert result_chunked.rmse is not None
    np.testing.assert_allclose(
        np.asarray(result_default.rmse),
        np.asarray(result_chunked.rmse),
        rtol=1e-5,
        atol=1e-5,
    )
    assert result_default._fit is not None
    assert result_chunked._fit is not None
    np.testing.assert_allclose(
        result_default._fit["C1"], result_chunked._fit["C1"], rtol=1e-4, atol=5e-3
    )
    assert result_default._fit["C2"] is not None
    assert result_chunked._fit["C2"] is not None
    np.testing.assert_allclose(
        np.asarray(result_default._fit["C2"]),
        np.asarray(result_chunked._fit["C2"]),
        rtol=1e-4,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        result_default._fit["f0"], result_chunked._fit["f0"], rtol=1e-4, atol=5e-3
    )
    np.testing.assert_allclose(
        result_default.predict(X),
        result_chunked.predict(X),
        rtol=1e-4,
        atol=3e-3,
    )


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
    result = analyze_hdmr(problem, X, Y, maxorder=1)
    assert result.S1.shape == (3,)
    assert result.ST.shape == (3,)


def test_indices_are_affine_invariant(ishigami_data):
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
    X, Y = ishigami_data
    Y_affine = 3.0 * Y + 17.0

    base = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2)
    affine = analyze_hdmr(PROBLEM, X, Y_affine, maxorder=2, m=2)

    np.testing.assert_allclose(np.asarray(base.S1), np.asarray(affine.S1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(base.ST), np.asarray(affine.ST), rtol=1e-5, atol=1e-5)


def test_backfitting_stop_rule_does_not_depend_on_y_scale_or_m():
    """H4 regression: the backfit stop rule is relative, not absolute.

    ``3.0 * Y + 17.0`` above is too large a shift to trigger the old bug: the
    absolute threshold only stopped a correlated fit early when the
    coefficients were small, which happens for a small ``Y`` scale or a
    large basis count ``m`` (the basis is scaled by ``m**3``, so the
    coefficients shrink as ``m**-3``). This reproduces the review's case: a
    correlated additive model, N = 1500, rho(x1, x2) = 0.7. Before the fix,
    ``S1[0]`` read 0.014-0.066 instead of ~0.104 for ``m in (4, 8)`` and for
    ``Y * 1e-3``.
    """
    rng = np.random.default_rng(0)
    N = 1500
    corr = np.array([[1.0, 0.7, 0.3], [0.7, 1.0, 0.0], [0.3, 0.0, 1.0]])
    problem = Problem(("a", "b", "c"), ((0.0, 1.0),) * 3, correlation=corr)
    Z = rng.multivariate_normal(np.zeros(3), corr, size=N)
    from scipy.stats import norm

    X = jnp.asarray(norm.cdf(Z))
    Y = X[:, 0] + 2.0 * X[:, 1] ** 2 + jnp.sin(3.0 * X[:, 2])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s1_ref = np.asarray(analyze_hdmr(problem, X, Y, m=2, maxiter=100).S1)
        for m in (4, 8):
            for scale in (1e-3, 1.0, 1e3):
                result = analyze_hdmr(problem, X, Y * scale, m=m, maxiter=100)
                np.testing.assert_allclose(np.asarray(result.S1), s1_ref, atol=0.02)


# ---------------------------------------------------------------------------
# maxorder tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("maxorder", "n_terms"), [(1, 3), (2, 6), (3, 7)])
def test_maxorder_selects_the_term_count(maxorder, n_terms, ishigami_data):
    """For D=3 the term count is 3, 6 and 7 at maxorder 1, 2 and 3."""
    X, Y = ishigami_data
    assert PROBLEM.num_vars == 3
    result = analyze_hdmr(PROBLEM, X, Y, maxorder=maxorder, m=2)
    assert result.Sa.shape == (n_terms,)
    assert len(result.terms) == n_terms


# ---------------------------------------------------------------------------
# Emulator tests
# ---------------------------------------------------------------------------


def test_emulator_prediction(hdmr_result, ishigami_data):
    """Emulator predictions on training data should have low RMSE."""
    X, Y = ishigami_data
    Y_pred = hdmr_result.predict(X)
    assert Y_pred.shape == Y.shape
    rmse = float(jnp.sqrt(jnp.mean(jnp.square(Y - Y_pred))))
    # HDMR surrogate should capture most of the variance
    assert rmse < 1.5, f"Emulator RMSE = {rmse:.3f}, expected < 1.5"


def test_predict_rejects_wrong_shaped_x(hdmr_result, ishigami_data):
    """predict must validate X shape instead of silently clamping/truncating."""
    X, _ = ishigami_data
    with pytest.raises(ValueError, match="2-D"):
        hdmr_result.predict(X[:, 0])  # 1-D input
    D = PROBLEM.num_vars
    X_wide = jnp.concatenate([X, X[:, :2]], axis=1)  # (N, D + 2)
    assert X_wide.shape[1] == D + 2
    with pytest.raises(ValueError, match="columns"):
        hdmr_result.predict(X_wide)


def test_emulator_predictions_and_rmse_stay_on_the_caller_scale(ishigami_data):
    """A shifted, scaled Y must come back out of predict() on that same scale."""
    X, Y = ishigami_data
    Y_affine = 3.0 * Y + 17.0

    result = analyze_hdmr(PROBLEM, X, Y_affine, maxorder=2, m=2)
    Y_pred = result.predict(X)
    rmse = float(jnp.sqrt(jnp.mean(jnp.square(Y_affine - Y_pred))))

    assert result._fit is not None
    assert Y_pred.shape == Y_affine.shape
    assert result.rmse is not None
    np.testing.assert_allclose(np.asarray(result.rmse), rmse, rtol=1e-6, atol=1e-6)
    assert abs(float(jnp.mean(Y_pred)) - float(jnp.mean(Y_affine))) < 3.0


def test_multi_output_analytical_ishigami_emulator(ishigami_data):
    """Multi-output HDMR should keep per-output coefficients and predictions."""
    X, Y = ishigami_data
    Y_multi = jnp.stack([Y, 2.0 * Y], axis=1)

    result = analyze_hdmr(PROBLEM, X, Y_multi, maxorder=2, m=2)
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


def test_time_series_multi_output_emulator_preserves_axes(ishigami_data):
    """Time-series HDMR emulator should preserve both output axes."""
    X, Y = ishigami_data
    Y_tk = jnp.stack(
        [
            jnp.stack([Y, 2.0 * Y], axis=1),
            jnp.stack([0.5 * Y, -1.5 * Y], axis=1),
        ],
        axis=1,
    )

    result = analyze_hdmr(PROBLEM, X, Y_tk, maxorder=2, m=2)
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


def test_multi_output_emulator_preserves_non_proportional_outputs(ishigami_data):
    """Distinct outputs should keep distinct sensitivities and predictions."""
    X, Y = ishigami_data
    Y_alt = jnp.sin(X[:, 1]) + 0.1 * X[:, 0] * X[:, 2]
    Y_multi = jnp.stack([Y, Y_alt], axis=1)

    result = analyze_hdmr(PROBLEM, X, Y_multi, maxorder=2, m=2)
    Y_pred = result.predict(X)

    assert result._fit is not None
    assert result._fit["C1"].shape == (2, 5, 3)
    assert result._fit["C2"] is not None
    assert result._fit["C2"].shape == (2, 25, 3)
    assert result._fit["f0"].shape == (2,)
    assert result.rmse is not None
    assert result.rmse.shape == (2,)
    assert Y_pred.shape == Y_multi.shape

    assert not np.allclose(np.array(result.S1[0]), np.array(result.S1[1]))
    assert not np.allclose(np.array(result.ST[0]), np.array(result.ST[1]))
    assert not np.allclose(np.array(Y_pred[:, 0]), np.array(Y_pred[:, 1]))

    rmse = np.sqrt(np.mean(np.square(np.array(Y_pred) - np.array(Y_multi)), axis=0))
    assert np.all(rmse < np.array([1.5, 1.0])), rmse


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_constant_y():
    """Constant Y yields NaN indices and a warning (package-wide convention)."""
    N = 500
    D = PROBLEM.num_vars
    key = jax.random.PRNGKey(99)
    bounds = jnp.array(PROBLEM.bounds)
    X = jax.random.uniform(key, shape=(N, D), minval=bounds[:, 0], maxval=bounds[:, 1])
    Y = jnp.full(N, 5.0)  # constant output (exactly zero variance)
    with pytest.warns(UserWarning, match="zero variance"):
        result = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2)
    # Matches Sobol/PCE: 0/0 indices are NaN, not silent zeros.
    assert jnp.all(jnp.isnan(result.Sa))
    assert jnp.all(jnp.isnan(result.ST))


def test_validation_errors():
    """Input validation should raise on bad inputs."""
    X = jnp.ones((100, 3))
    Y = jnp.ones(100)

    with pytest.raises(ValueError, match="at least 300"):
        analyze_hdmr(PROBLEM, X, Y)

    X = jnp.ones((500, 3))
    Y = jnp.ones(500)

    with pytest.raises(ValueError, match="maxorder"):
        analyze_hdmr(PROBLEM, X, Y, maxorder=4)

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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        # m=0 zeroes the whole basis (val * m**3 in _engine._bspline_basis),
        # so every index came out 0 with no error at all.
        ({"m": 0}, "m must be >= 1"),
        ({"m": -1}, "m must be >= 1"),
        # maxiter=0 silently skipped backfitting and returned unfitted
        # coefficients.
        ({"maxiter": 0}, "maxiter must be >= 1"),
        # lambdax<0 subtracts from the Gram diagonal and yields NaN
        # coefficients.
        ({"lambdax": -1.0}, "lambdax must be >= 0"),
    ],
)
def test_fit_scalars_rejected_in_analyze_and_indices(kwargs, match):
    """T4: m, maxiter and lambdax are validated, in both entry points.

    Each of the three used to reach the fit unchecked and degrade it
    silently. Both ``analyze`` and ``indices`` run the same checks (one
    definition, ``_fit_scalar_checks``), so the two cannot drift.
    """
    from jaxgsa.hdmr import indices as indices_hdmr

    X = jnp.ones((500, 3))
    Y = jnp.ones(500)
    with pytest.raises(ValueError, match=match):
        analyze_hdmr(PROBLEM, X, Y, **kwargs)
    with pytest.raises(ValueError, match=match):
        indices_hdmr(PROBLEM, X, Y, **kwargs)


def test_fit_scalars_valid_edges_pass(ishigami_data):
    """T4: the smallest legal values fit without an error.

    ``m=1`` is the minimal cubic basis (one knot interval, four functions),
    ``maxiter=1`` is one backfitting sweep, ``lambdax=0.0`` is an
    unregularized solve. All three are degraded-but-defined, so they must
    run, not raise.
    """
    X, Y = ishigami_data
    result = analyze_hdmr(PROBLEM, X, Y, maxorder=1, m=1, maxiter=1, lambdax=0.0)
    assert np.all(np.isfinite(np.asarray(result.S1)))


# ---------------------------------------------------------------------------
# S2 / S3 interaction-index properties
# ---------------------------------------------------------------------------


def _term_index(result, label: str) -> int:
    """Return the position of a term label along the term axis."""
    return result.terms.index(label)


def test_s2_property_shape_and_symmetry(hdmr_result):
    """S2 is a symmetric (D, D) matrix with a NaN diagonal."""
    D = PROBLEM.num_vars
    S2 = np.array(hdmr_result.S2)
    assert S2.shape == (D, D)
    # Diagonal has no pairwise term, so it stays NaN.
    assert np.all(np.isnan(np.diag(S2)))
    # Off-diagonal pairs are populated (Ishigami, maxorder=2) and symmetric.
    offdiag = S2[~np.eye(D, dtype=bool)]
    assert np.all(np.isfinite(offdiag))
    np.testing.assert_array_equal(S2, S2.T)


def test_s2_matches_term_slice(hdmr_result):
    """Each S2[i, j] equals the Sa entry of the matching interaction term."""
    names = PROBLEM.names
    Sa = np.array(hdmr_result.Sa)
    S2 = np.array(hdmr_result.S2)
    D = PROBLEM.num_vars
    for i in range(D):
        for j in range(i + 1, D):
            k = _term_index(hdmr_result, f"{names[i]}/{names[j]}")
            assert S2[i, j] == Sa[k]
            assert S2[j, i] == Sa[k]


def test_s2_ishigami_dominant_pair(hdmr_result):
    """Ishigami's x1-x3 interaction should dominate the S2 matrix."""
    S2 = np.array(hdmr_result.S2)
    # x1 (index 0) and x3 (index 2) carry Ishigami's only real interaction.
    assert S2[0, 2] > S2[0, 1]
    assert S2[0, 2] > S2[1, 2]


def test_s2_s3_all_nan_when_maxorder_1(ishigami_data):
    """With no interaction terms, S2 and S3 are entirely NaN."""
    X, Y = ishigami_data
    D = PROBLEM.num_vars
    result = analyze_hdmr(PROBLEM, X, Y, maxorder=1, m=2)
    assert result.S2.shape == (D, D)
    assert bool(jnp.all(jnp.isnan(result.S2)))
    assert result.S3.shape == (D, D, D)
    assert bool(jnp.all(jnp.isnan(result.S3)))


def test_s3_property_maxorder_3(ishigami_data):
    """S3 is a symmetric (D, D, D) tensor matching the triple's Sa entry."""
    X, Y = ishigami_data
    names = PROBLEM.names
    D = PROBLEM.num_vars
    result = analyze_hdmr(PROBLEM, X, Y, maxorder=3, m=2)
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


def test_s2_layout_uses_numeric_interaction_indices():
    """S2 uses stored numeric indices instead of parsing display labels."""
    from jaxgsa.hdmr._result import HDMRResult
    from jaxgsa.problem import Problem

    problem = Problem(names=("x1", "x2", "x3"), bounds=((0.0, 1.0),) * 3)
    result = HDMRResult(
        Sa=jnp.array([1.0, 2.0, 3.0, 9.0]),
        Sb=jnp.zeros(4),
        S=jnp.ones(4),
        ST=jnp.ones(3),
        problem=problem,
        terms=("x1", "x2", "x3", "x1/x2"),
        invalid=_clean_report(),
        _c2=((0, 1),),
    )
    S2 = np.array(result.S2)
    # The (x1, x2) cell is populated from Sa[3], not silently left NaN.
    assert S2[0, 1] == 9.0
    assert S2[1, 0] == 9.0
    assert np.all(np.isnan(np.diag(S2)))
    assert np.isnan(S2[0, 2])  # (x1, x3) has no term -> NaN


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
def correlated_hdmr_result():
    """Fit HDMR once on a correlated problem, warnings suppressed."""
    problem = _correlated_problem()
    X = jnp.asarray(monte_carlo(problem, 1000, seed=7))
    Y = _linear_interaction(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = analyze_hdmr(problem, X, Y, maxorder=2, m=2)
    return result


def test_st_is_scsa_total_under_correlation(correlated_hdmr_result):
    """ST is sum over terms u containing i of (Sa_u + Sb_u), not a Sobol total.

    Pins the SCSA convention that Li et al. (2010) define in Section 2.2.3,
    which is also SALib's HDMR convention. Recomputed here from the per-term
    fields so a change of convention cannot pass silently.
    """
    result = correlated_hdmr_result
    D = result.problem.num_vars

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
    # noise only now, not the few-percent gap SALib's Y-referenced ancova
    # leaves under correlation.
    np.testing.assert_allclose(
        np.array(result.S), np.array(result.Sa) + np.array(result.Sb), atol=1e-6
    )
    np.testing.assert_allclose(ST, _scatter(np.array(result.Sa) + np.array(result.Sb)), atol=1e-6)
    # The correlative shares are what make this differ from the structural
    # sum; if they were negligible the test would not be pinning anything.
    assert np.abs(np.array(result.Sb)).max() > 1e-3
    assert not np.allclose(ST, _scatter(np.array(result.Sa)), atol=1e-2)


def test_st_diverges_from_the_conditional_variance_total():
    """HDMR's SCSA total must not be "fixed" to agree with a real total index.

    ``jaxgsa.kucherenko`` estimates the conditional-variance total
    ``E[Var(Y | X_~i)] / Var(Y)`` from its own design, and that is the
    quantity HDMR's ``ST`` is often mistaken for. The two answer different
    questions under dependence, so they must disagree. This test exists to
    fail loudly if someone later "corrects" either one to match the other.
    """
    problem = _correlated_problem(rho=0.9)

    # HDMR, given data.
    X = jnp.asarray(monte_carlo(problem, 4000, seed=5))
    Y = _linear_interaction(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdmr_ST = np.asarray(analyze_hdmr(problem, X, Y, maxorder=2, m=2).ST)

    # Kucherenko, from its own conditional-copula design.
    samples = kucherenko.sample(problem, 4000, seed=5)
    k_result = kucherenko.analyze(samples, _linear_interaction(jnp.asarray(samples.samples)))
    kucherenko_ST = np.asarray(k_result.ST)

    # Kucherenko's total is a variance fraction: non-negative and bounded.
    assert np.all(kucherenko_ST >= -1e-6)
    assert np.all(kucherenko_ST <= 1.0 + 1e-6)

    # HDMR's SCSA total is neither of those things here, and the correlated
    # pair is where the two readings come apart.
    assert not np.allclose(hdmr_ST[:2], kucherenko_ST[:2], atol=0.05)


def test_correlated_problem_warns_once():
    """analyze emits exactly one SCSA warning per call on a correlated problem."""
    problem = _correlated_problem()
    X = jnp.asarray(monte_carlo(problem, 400, seed=11))
    Y = _linear_interaction(X)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyze_hdmr(problem, X, Y, maxorder=2, m=2)

    scsa = [w for w in caught if "SCSA" in str(w.message)]
    assert len(scsa) == 1
    assert issubclass(scsa[0].category, UserWarning)
    message = str(scsa[0].message)
    assert "kucherenko" in message
    assert "vkoga" in message


def test_independent_problem_does_not_warn(ishigami_data):
    """An independent problem stays silent about the SCSA reading."""
    X, Y = ishigami_data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2)
    assert [w for w in caught if "SCSA" in str(w.message)] == []


# ---------------------------------------------------------------------------
# on_invalid policy
# ---------------------------------------------------------------------------


class TestHDMROnInvalid:
    """T4: ``hdmr.analyze`` applies the shared non-finite policy.

    The check lives in the public wrapper only, next to the zero-variance and
    SCSA warnings and for the same reason: ``_analyze_hdmr_core`` must stay
    free of it so no caller can trigger the policy twice.
    """

    HDMR_KWARGS: dict[str, Any] = dict(maxorder=2, m=2)

    def test_raise_is_the_default_and_names_the_rows(self, ishigami_data):
        """T4: a non-finite Y refuses the analysis and says which rows carry it."""
        X, Y = ishigami_data
        Y = np.asarray(Y).copy()
        Y[3] = np.nan
        Y[500] = -np.inf
        with pytest.raises(ValueError, match="non-finite") as exc:
            analyze_hdmr(PROBLEM, X, Y, **self.HDMR_KWARGS)
        message = str(exc.value)
        assert "jaxgsa.hdmr.analyze" in message
        assert "2 of 2000 rows" in message
        assert "[3, 500]" in message

    def test_propagate_warns_and_lets_the_nan_reach_the_indices(self, ishigami_data):
        """T4: 'propagate' keeps every row, so the fitted indices go non-finite."""
        X, Y = ishigami_data
        Y = np.asarray(Y).copy()
        Y[3] = np.nan
        with pytest.warns(JaxgsaWarning, match="reaches the indices"):
            result = analyze_hdmr(PROBLEM, X, Y, on_invalid="propagate", **self.HDMR_KWARGS)
        assert not np.all(np.isfinite(np.array(result.ST)))
        assert result.invalid.policy == "propagate"
        assert result.invalid.n_invalid == 1

    def test_drop_removes_the_rows_and_the_fit_is_finite_again(self, ishigami_data):
        """T4: 'drop' fits on the survivors, so every index is finite."""
        X, Y = ishigami_data
        Y = np.asarray(Y).copy()
        Y[3] = np.nan
        with pytest.warns(JaxgsaWarning, match="dropped 1 of 2000 rows"):
            result = analyze_hdmr(PROBLEM, X, Y, on_invalid="drop", **self.HDMR_KWARGS)
        assert np.all(np.isfinite(np.array(result.ST)))
        assert result.invalid.unit_indices == (3,)
        assert result.invalid.sources == ("Y",)
        _assert_matches_analytical_s1_st(np.array(result.S1), np.array(result.ST))

    def test_a_non_finite_input_is_caught_too(self, ishigami_data):
        """T4: X is checked as well as Y, and ``sources`` says which array.

        A NaN input reaches the CDF transform and then the B-spline basis, so
        it is no less fatal than a NaN output.
        """
        X, Y = ishigami_data
        X = np.asarray(X).copy()
        X[17, 1] = np.inf
        with pytest.raises(ValueError, match=r"in X\.") as exc:
            analyze_hdmr(PROBLEM, X, Y, **self.HDMR_KWARGS)
        assert "[17]" in str(exc.value)
        with pytest.warns(JaxgsaWarning):
            result = analyze_hdmr(PROBLEM, X, Y, on_invalid="drop", **self.HDMR_KWARGS)
        assert result.invalid.sources == ("X",)

    def test_dropping_below_the_sample_floor_refuses(self, ishigami_data):
        """T4: 'drop' will not fit on fewer rows than HDMR's own 300 minimum.

        Without the floor the surviving sample would fall through to the
        generic "Need at least 300 samples" check, which names neither the
        dropped rows nor the policy that removed them.
        """
        X, Y = ishigami_data
        X, Y = np.asarray(X)[:400], np.asarray(Y)[:400].copy()
        Y[:150] = np.nan
        with pytest.raises(ValueError, match="every usable row was removed") as exc:
            analyze_hdmr(PROBLEM, X, Y, on_invalid="drop", **self.HDMR_KWARGS)
        assert "at least 300" in str(exc.value)

    def test_shapley_carries_the_report_through(self, ishigami_data):
        """T4: the report survives the analytical Shapley step unchanged."""
        X, Y = ishigami_data
        Y = np.asarray(Y).copy()
        Y[3] = np.nan
        with pytest.warns(JaxgsaWarning):
            result = analyze_hdmr(PROBLEM, X, Y, on_invalid="drop", **self.HDMR_KWARGS)
        assert result.shapley().invalid == result.invalid


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

    def test_no_bootstrap_by_default(self):
        """T4: the plainest call reports no interval and pays for none.

        Each replicate refits the whole B-spline expansion, so an
        on-by-default interval would make a routine call an order of
        magnitude slower.
        """
        X, Y = self._data()
        result = analyze_hdmr(PROBLEM, X, Y, maxorder=2, maxiter=50)

        assert result.ci is None
        assert result.Sa_conf is None
        assert result.Sb_conf is None
        assert result.S_conf is None
        assert result.ST_conf is None

    def test_the_interval_brackets_the_point_estimate(self):
        """T4: lower <= upper, and the estimate sits inside, for all four fields."""
        X, Y = self._data()
        result = analyze_hdmr(
            PROBLEM, X, Y, maxorder=2, maxiter=50, n_bootstrap=5, key=jax.random.key(0)
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
            # A five-draw percentile interval need not straddle the estimate
            # exactly, so the comparison carries slack.
            assert (point[finite] >= lower[finite] - 0.25).all()
            assert (point[finite] <= upper[finite] + 0.25).all()

    def test_the_point_estimate_is_the_one_without_a_bootstrap(self):
        """T4: asking for an interval does not move the number it is around."""
        X, Y = self._data()
        plain = analyze_hdmr(PROBLEM, X, Y, maxorder=2, maxiter=50)
        with_ci = analyze_hdmr(
            PROBLEM, X, Y, maxorder=2, maxiter=50, n_bootstrap=3, key=jax.random.key(1)
        )

        np.testing.assert_array_equal(np.asarray(plain.Sa), np.asarray(with_ci.Sa))
        np.testing.assert_array_equal(np.asarray(plain.ST), np.asarray(with_ci.ST))

    def test_ci_records_how_the_interval_was_made(self):
        """T4: the level, the rule and the count travel with the numbers."""
        X, Y = self._data()
        result = analyze_hdmr(
            PROBLEM,
            X,
            Y,
            maxorder=2,
            maxiter=50,
            n_bootstrap=4,
            conf_level=0.9,
            ci_method="gaussian",
            key=jax.random.key(2),
        )

        assert result.ci is not None
        assert (result.ci.level, result.ci.method, result.ci.n_bootstrap) == (0.9, "gaussian", 4)
        assert result.ci.replicates is None

    def test_keep_replicates_retains_every_draw(self):
        """T4: the draws come back with a leading resample axis."""
        X, Y = self._data()
        result = analyze_hdmr(
            PROBLEM,
            X,
            Y,
            maxorder=2,
            maxiter=50,
            n_bootstrap=3,
            key=jax.random.key(3),
            keep_replicates=True,
        )

        assert result.ci is not None
        assert result.ci.replicates is not None
        assert result.ci.replicates["Sa"].shape == (3, 6)
        assert result.ci.replicates["ST"].shape == (3, 3)

    def test_a_key_is_required(self):
        """T4: no key means no interval, and it is refused rather than seeded."""
        X, Y = self._data()
        with pytest.raises(ValueError, match="key is required"):
            analyze_hdmr(PROBLEM, X, Y, maxorder=2, maxiter=50, n_bootstrap=3)

    def test_the_interval_reaches_the_dataset(self):
        """T4: the export splits each interval into named lower/upper halves."""
        X, Y = self._data()
        result = analyze_hdmr(
            PROBLEM, X, Y, maxorder=2, maxiter=50, n_bootstrap=3, key=jax.random.key(5)
        )
        dataset = result.to_dataset()

        for name in ("Sa", "Sb", "S", "ST"):
            assert f"{name}_lower" in dataset
            assert f"{name}_upper" in dataset
