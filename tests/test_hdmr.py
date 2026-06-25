"""Tests for RS-HDMR sensitivity analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gsax.benchmarks.ishigami import (
    ANALYTICAL_S1,
    ANALYTICAL_ST,
    PROBLEM,
    evaluate,
)
from gsax.expansions.hdmr import analyze as analyze_hdmr
from gsax.expansions.hdmr import emulate as emulate_hdmr
from gsax.problem import GaussianInputSpec


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


def test_s1_via_sa(hdmr_result):
    """First-order Sa terms should approximate analytical S1."""
    Sa = np.array(hdmr_result.Sa)
    D = PROBLEM.num_vars
    _assert_matches_analytical_s1_st(Sa[:D], np.array(hdmr_result.ST))


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
    assert result.emulator is not None
    assert set(result.emulator) == {
        "C1",
        "C2",
        "C3",
        "f0",
        "prenormalize",
        "y_mean",
        "y_std",
        "m",
        "maxorder",
        "c2",
        "c3",
    }
    assert result.emulator["C1"].shape == (5, 3)
    assert result.emulator["C2"] is not None
    assert result.emulator["C2"].shape == (25, 3)
    assert result.emulator["C3"] is None
    assert result.emulator["f0"].shape == ()
    assert result.emulator["y_mean"].shape == ()
    assert result.emulator["y_std"].shape == ()
    assert result.emulator["prenormalize"] is False


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
    assert result.emulator is not None
    assert result.emulator["C1"].shape == (K, 5, 3)
    assert result.emulator["C2"] is not None
    assert result.emulator["C2"].shape == (K, 25, 3)
    assert result.emulator["C3"] is None
    assert result.emulator["f0"].shape == (K,)


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
    assert result.emulator is not None
    assert result.emulator["C1"].shape == (T, K, 5, 3)
    assert result.emulator["C2"] is not None
    assert result.emulator["C2"].shape == (T, K, 25, 3)
    assert result.emulator["C3"] is None
    assert result.emulator["f0"].shape == (T, K)


def test_chunk_size_regression(ishigami_data):
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
        chunk_size=1,
    )

    np.testing.assert_allclose(
        np.asarray(result_default.Sa), np.asarray(result_chunked.Sa), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(result_default.Sb), np.asarray(result_chunked.Sb), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(result_default.S), np.asarray(result_chunked.S), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(result_default.ST), np.asarray(result_chunked.ST), rtol=1e-6, atol=1e-6
    )
    assert result_default.rmse is not None
    assert result_chunked.rmse is not None
    np.testing.assert_allclose(
        np.asarray(result_default.rmse),
        np.asarray(result_chunked.rmse),
        rtol=1e-6,
        atol=1e-6,
    )
    assert result_default.emulator is not None
    assert result_chunked.emulator is not None
    np.testing.assert_allclose(
        result_default.emulator["C1"], result_chunked.emulator["C1"], rtol=1e-4, atol=5e-3
    )
    assert result_default.emulator["C2"] is not None
    assert result_chunked.emulator["C2"] is not None
    np.testing.assert_allclose(
        np.asarray(result_default.emulator["C2"]),
        np.asarray(result_chunked.emulator["C2"]),
        rtol=1e-4,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        result_default.emulator["f0"], result_chunked.emulator["f0"], rtol=1e-4, atol=5e-3
    )
    np.testing.assert_allclose(
        emulate_hdmr(result_default, X),
        emulate_hdmr(result_chunked, X),
        rtol=1e-4,
        atol=3e-3,
    )


def test_chunk_size_regression_3d(ishigami_data):
    """Chunked and unchunked HDMR paths should agree for time-series multi-output Y."""
    X, Y = ishigami_data
    Y_alt = jnp.sin(X[:, 1]) + 0.1 * X[:, 0] * X[:, 2]
    Y_tk = jnp.stack(
        [
            jnp.stack([Y, Y_alt], axis=1),
            jnp.stack([0.5 * Y + 0.25 * X[:, 0], -1.5 * Y_alt], axis=1),
        ],
        axis=1,
    )

    result_default = analyze_hdmr(PROBLEM, X, Y_tk, maxorder=2, m=2)
    result_chunked = analyze_hdmr(
        PROBLEM,
        X,
        Y_tk,
        maxorder=2,
        m=2,
        chunk_size=1,
    )

    np.testing.assert_allclose(
        np.asarray(result_default.Sa), np.asarray(result_chunked.Sa), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(result_default.Sb), np.asarray(result_chunked.Sb), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(result_default.S), np.asarray(result_chunked.S), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(result_default.ST), np.asarray(result_chunked.ST), rtol=1e-6, atol=1e-6
    )
    assert result_default.rmse is not None
    assert result_chunked.rmse is not None
    np.testing.assert_allclose(
        np.asarray(result_default.rmse),
        np.asarray(result_chunked.rmse),
        rtol=1e-6,
        atol=1e-6,
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


def test_repeated_calls_identical(ishigami_data):
    """Repeated identical HDMR calls should return identical outputs."""
    X, Y = ishigami_data
    result_first = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2)
    result_second = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2)

    np.testing.assert_allclose(
        np.asarray(result_first.Sa), np.asarray(result_second.Sa), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(result_first.Sb), np.asarray(result_second.Sb), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(result_first.S), np.asarray(result_second.S), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(result_first.ST), np.asarray(result_second.ST), rtol=1e-6, atol=1e-6
    )
    assert result_first.rmse is not None
    assert result_second.rmse is not None
    np.testing.assert_allclose(
        np.asarray(result_first.rmse),
        np.asarray(result_second.rmse),
        rtol=1e-6,
        atol=1e-6,
    )
    assert result_first.emulator is not None
    assert result_second.emulator is not None
    np.testing.assert_allclose(
        np.asarray(result_first.emulator["C1"]),
        np.asarray(result_second.emulator["C1"]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result_first.emulator["C2"]),
        np.asarray(result_second.emulator["C2"]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert result_first.terms == result_second.terms


def test_explicit_prenormalize_false_matches_default(ishigami_data):
    """Explicit prenormalize=False should preserve the default HDMR path."""
    X, Y = ishigami_data
    default = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2)
    explicit = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2, prenormalize=False)

    np.testing.assert_allclose(
        np.asarray(default.Sa), np.asarray(explicit.Sa), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(default.Sb), np.asarray(explicit.Sb), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(np.asarray(default.S), np.asarray(explicit.S), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(default.ST), np.asarray(explicit.ST), rtol=1e-6, atol=1e-6
    )
    assert default.rmse is not None
    assert explicit.rmse is not None
    np.testing.assert_allclose(
        np.asarray(default.rmse),
        np.asarray(explicit.rmse),
        rtol=1e-6,
        atol=1e-6,
    )


def test_prenormalize_is_shift_scale_invariant(ishigami_data):
    """prenormalize=True should make HDMR sensitivities invariant to affine Y changes."""
    X, Y = ishigami_data
    Y_affine = 3.0 * Y + 17.0

    base = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2, prenormalize=True)
    affine = analyze_hdmr(PROBLEM, X, Y_affine, maxorder=2, m=2, prenormalize=True)

    np.testing.assert_allclose(np.asarray(base.S1), np.asarray(affine.S1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(base.ST), np.asarray(affine.ST), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# maxorder tests
# ---------------------------------------------------------------------------


def test_maxorder_1(ishigami_data):
    """maxorder=1 should produce D terms."""
    X, Y = ishigami_data
    result = analyze_hdmr(
        PROBLEM,
        X,
        Y,
        maxorder=1,
        m=2,
    )
    assert result.Sa.shape == (PROBLEM.num_vars,)
    assert len(result.terms) == PROBLEM.num_vars


def test_maxorder_2(ishigami_data):
    """maxorder=2 should produce D + C(D,2) terms."""
    X, Y = ishigami_data
    D = PROBLEM.num_vars
    result = analyze_hdmr(
        PROBLEM,
        X,
        Y,
        maxorder=2,
        m=2,
    )
    expected_n = D + D * (D - 1) // 2
    assert result.Sa.shape == (expected_n,)
    assert len(result.terms) == expected_n


def test_maxorder_3(ishigami_data):
    """maxorder=3 should produce D + C(D,2) + C(D,3) terms."""
    X, Y = ishigami_data
    D = PROBLEM.num_vars
    result = analyze_hdmr(
        PROBLEM,
        X,
        Y,
        maxorder=3,
        m=2,
    )
    expected_n = D + D * (D - 1) // 2 + D * (D - 1) * (D - 2) // 6
    assert result.Sa.shape == (expected_n,)
    assert len(result.terms) == expected_n


# ---------------------------------------------------------------------------
# Emulator tests
# ---------------------------------------------------------------------------


def test_emulator_prediction(hdmr_result, ishigami_data):
    """Emulator predictions on training data should have low RMSE."""
    X, Y = ishigami_data
    Y_pred = emulate_hdmr(hdmr_result, X)
    assert Y_pred.shape == Y.shape
    rmse = float(jnp.sqrt(jnp.mean(jnp.square(Y - Y_pred))))
    # HDMR surrogate should capture most of the variance
    assert rmse < 1.5, f"Emulator RMSE = {rmse:.3f}, expected < 1.5"


def test_emulator_reasonable(hdmr_result, ishigami_data):
    """Emulator output mean should be close to data mean."""
    X, Y = ishigami_data
    Y_pred = emulate_hdmr(hdmr_result, X)
    assert abs(float(jnp.mean(Y_pred)) - float(jnp.mean(Y))) < 1.0


def test_prenormalize_emulator_predictions_and_rmse_stay_on_original_scale(ishigami_data):
    """prenormalized HDMR fits should still predict and report RMSE in original units."""
    X, Y = ishigami_data
    Y_affine = 3.0 * Y + 17.0

    result = analyze_hdmr(PROBLEM, X, Y_affine, maxorder=2, m=2, prenormalize=True)
    Y_pred = emulate_hdmr(result, X)
    rmse = float(jnp.sqrt(jnp.mean(jnp.square(Y_affine - Y_pred))))

    assert result.emulator is not None
    assert result.emulator["prenormalize"] is True
    assert Y_pred.shape == Y_affine.shape
    assert result.rmse is not None
    np.testing.assert_allclose(np.asarray(result.rmse), rmse, rtol=1e-6, atol=1e-6)
    assert abs(float(jnp.mean(Y_pred)) - float(jnp.mean(Y_affine))) < 3.0


def test_multi_output_analytical_ishigami_emulator(ishigami_data):
    """Multi-output HDMR should keep per-output coefficients and predictions."""
    X, Y = ishigami_data
    Y_multi = jnp.stack([Y, 2.0 * Y], axis=1)

    result = analyze_hdmr(PROBLEM, X, Y_multi, maxorder=2, m=2)
    Y_pred = emulate_hdmr(result, X)

    assert result.emulator is not None
    assert result.emulator["C1"].shape == (2, 5, 3)
    assert result.emulator["C2"] is not None
    assert result.emulator["C2"].shape == (2, 25, 3)
    assert result.emulator["f0"].shape == (2,)
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
    Y_pred = emulate_hdmr(result, X)

    assert result.emulator is not None
    assert result.emulator["C1"].shape == (2, 2, 5, 3)
    assert result.emulator["C2"] is not None
    assert result.emulator["C2"].shape == (2, 2, 25, 3)
    assert result.emulator["f0"].shape == (2, 2)
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
    Y_pred = emulate_hdmr(result, X)

    assert result.emulator is not None
    assert result.emulator["C1"].shape == (2, 5, 3)
    assert result.emulator["C2"] is not None
    assert result.emulator["C2"].shape == (2, 25, 3)
    assert result.emulator["f0"].shape == (2,)
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


def test_term_labels(hdmr_result):
    """Term labels should contain parameter names and interaction terms."""
    terms = hdmr_result.terms
    names = PROBLEM.names
    # First-order terms should be parameter names
    for name in names:
        assert name in terms
    # Should have interaction terms
    assert any("/" in t for t in terms)


def test_select_and_rmse(hdmr_result):
    """select and rmse should be present."""
    assert hdmr_result.select is not None
    assert hdmr_result.rmse is not None
    assert hdmr_result.select.shape[0] > 0
    assert isinstance(hdmr_result.emulator["c2"], list)
    assert isinstance(hdmr_result.emulator["c3"], list)


def test_s1_property(hdmr_result):
    """S1 property should extract first D terms of Sa."""
    D = PROBLEM.num_vars
    S1 = hdmr_result.S1
    assert S1.shape == (D,)
    np.testing.assert_array_equal(S1, hdmr_result.Sa[:D])


def test_s1_accuracy(hdmr_result):
    """S1 property should approximate analytical first-order Sobol indices."""
    _assert_matches_analytical_s1_st(np.array(hdmr_result.S1), np.array(hdmr_result.ST))


def test_constant_y():
    """Constant Y should not produce NaN indices."""
    N = 500
    D = PROBLEM.num_vars
    key = jax.random.PRNGKey(99)
    bounds = jnp.array(PROBLEM.bounds)
    X = jax.random.uniform(key, shape=(N, D), minval=bounds[:, 0], maxval=bounds[:, 1])
    Y = jnp.ones(N) * 42.0  # constant output
    result = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2)
    assert not jnp.any(jnp.isnan(result.Sa))
    assert not jnp.any(jnp.isnan(result.ST))


def test_scalar_like_lambdax(ishigami_data):
    """Scalar-like lambdax inputs should still be accepted."""
    X, Y = ishigami_data
    result = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2, lambdax=float(jnp.array(0.01)))
    assert result.Sa.shape[0] == PROBLEM.num_vars + PROBLEM.num_vars * (PROBLEM.num_vars - 1) // 2


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

    with pytest.raises(ValueError, match="chunk_size"):
        analyze_hdmr(PROBLEM, X, Y, chunk_size=0)
