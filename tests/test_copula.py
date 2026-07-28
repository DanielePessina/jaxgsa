"""Tests for the Gaussian-copula backend and the correlated-problem guards."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa._core.copula import (
    _spearman_to_latent,
    canonicalize_correlation,
    correlation_from_covariance,
    fit_gaussian_copula,
    independent_correlation,
    is_independent,
    latent_to_physical,
    physical_to_latent,
    validate_correlation,
)
from jaxgsa.problem import GaussianInputSpec, Problem

# An indefinite candidate: pairwise entries are individually valid but jointly
# inconsistent (eigenvalues of this matrix include a negative one).
_INDEFINITE_R = np.array(
    [
        [1.0, 0.9, 0.9],
        [0.9, 1.0, -0.9],
        [0.9, -0.9, 1.0],
    ]
)


def _uniform_problem(D: int = 2) -> Problem:
    return Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(D)})


# ---------------------------------------------------------------------------
# validate_correlation / repair warning
# ---------------------------------------------------------------------------


def test_validate_correlation_accepts_valid_matrix_unchanged():
    R = np.array([[1.0, 0.5], [0.5, 1.0]])
    np.testing.assert_allclose(validate_correlation(R, 2), R, atol=1e-15)


@pytest.mark.parametrize(
    ("R", "match"),
    [
        (np.eye(3), r"must be \(2, 2\)"),
        (np.array([[1.0, 0.5], [0.2, 1.0]]), "symmetric"),
        (np.array([[2.0, 0.5], [0.5, 1.0]]), "unit diagonal"),
        (np.array([[1.0, 1.5], [1.5, 1.0]]), r"lie in \[-1, 1\]"),
    ],
)
def test_validate_correlation_rejects_structurally_invalid(R, match):
    with pytest.raises(ValueError, match=match):
        validate_correlation(R, 2)


def test_repair_warning_fires_for_indefinite_matrix():
    # UserWarning is not promoted to an error by the pytest config, so the
    # firing case must be asserted explicitly with pytest.warns.
    with pytest.warns(UserWarning, match="not positive definite"):
        repaired = validate_correlation(_INDEFINITE_R, 3, warn_on_repair=True)
    eigenvalues = np.linalg.eigvalsh(repaired)
    assert eigenvalues.min() > 0
    np.testing.assert_allclose(np.diag(repaired), 1.0, atol=1e-12)


def test_repair_warning_reports_min_eigenvalue_and_max_change():
    with pytest.warns(UserWarning) as record:
        validate_correlation(_INDEFINITE_R, 3, warn_on_repair=True)
    message = str(record[0].message)
    min_eig = np.linalg.eigvalsh(_INDEFINITE_R).min()
    assert f"{min_eig:.3e}" in message
    assert "largest entrywise change" in message


def test_repair_stays_silent_for_valid_matrix():
    R = np.array([[1.0, 0.3], [0.3, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_correlation(R, 2, warn_on_repair=True)


def test_repair_stays_silent_without_opt_in():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_correlation(_INDEFINITE_R, 3)  # warn_on_repair defaults to False


# ---------------------------------------------------------------------------
# _spearman_to_latent
# ---------------------------------------------------------------------------


def test_spearman_to_latent_matches_kruskal_formula():
    rho_s = 0.7
    latent = _spearman_to_latent(np.array([[1.0, rho_s], [rho_s, 1.0]]))
    expected = 2.0 * np.sin(np.pi * rho_s / 6.0)
    np.testing.assert_allclose(latent[0, 1], expected, atol=1e-15)
    np.testing.assert_allclose(np.diag(latent), 1.0, atol=0)  # pinned exactly


def test_spearman_to_latent_maps_extremes_to_extremes():
    R = np.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(_spearman_to_latent(R), R, atol=1e-15)
    R = np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(_spearman_to_latent(R), R, atol=1e-15)


# ---------------------------------------------------------------------------
# canonicalize_correlation
# ---------------------------------------------------------------------------


def test_canonicalize_rejects_unknown_kind():
    with pytest.raises(ValueError, match="correlation_kind"):
        canonicalize_correlation(np.eye(2), 2, kind="pearson")


def test_canonicalize_latent_is_identity_on_valid_input():
    R = np.array([[1.0, 0.4], [0.4, 1.0]])
    np.testing.assert_allclose(canonicalize_correlation(R, 2), R, atol=1e-15)


def test_canonicalize_spearman_applies_conversion():
    rho_s = 0.5
    R = canonicalize_correlation(np.array([[1.0, rho_s], [rho_s, 1.0]]), 2, kind="spearman")
    np.testing.assert_allclose(R[0, 1], 2.0 * np.sin(np.pi * rho_s / 6.0), atol=1e-15)


# ---------------------------------------------------------------------------
# correlation_from_covariance
# ---------------------------------------------------------------------------


def test_correlation_from_covariance_round_trips_drd():
    R = np.array(
        [
            [1.0, 0.6, -0.2],
            [0.6, 1.0, 0.1],
            [-0.2, 0.1, 1.0],
        ]
    )
    sigma = np.array([0.5, 2.0, 3.0])
    cov = sigma[:, None] * R * sigma[None, :]
    np.testing.assert_allclose(correlation_from_covariance(cov), R, atol=1e-14)


def test_correlation_from_covariance_pins_unit_diagonal_exactly():
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    R = correlation_from_covariance(cov)
    assert (np.diag(R) == 1.0).all()
    # Exact diagonal means the result survives validate_correlation unchanged.
    np.testing.assert_allclose(validate_correlation(R, 2), R, atol=1e-15)


@pytest.mark.parametrize(
    ("cov", "match"),
    [
        (np.ones((2, 3)), "square"),
        (np.array([[1.0, 0.5], [0.2, 1.0]]), "symmetric"),
        (np.array([[1.0, 0.0], [0.0, -1.0]]), "strictly positive"),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), "strictly positive"),
    ],
)
def test_correlation_from_covariance_rejects_invalid(cov, match):
    with pytest.raises(ValueError, match=match):
        correlation_from_covariance(cov)


# ---------------------------------------------------------------------------
# fit_gaussian_copula / latent transforms
# ---------------------------------------------------------------------------


def test_fit_gaussian_copula_recovers_latent_correlation():
    problem = _uniform_problem(2)
    rho = 0.8
    R = np.array([[1.0, rho], [rho, 1.0]])
    correlated = problem.with_correlation(R)
    X = jaxgsa.sampling.monte_carlo(correlated, 20_000, seed=11)
    fitted = fit_gaussian_copula(problem, X)
    assert abs(fitted[0, 1] - rho) < 0.02
    np.testing.assert_allclose(np.diag(fitted), 1.0, atol=1e-12)


def test_fit_gaussian_copula_rejects_bad_shapes():
    problem = _uniform_problem(2)
    with pytest.raises(ValueError, match=r"must be \(N, 2\)"):
        fit_gaussian_copula(problem, np.zeros((10, 3)))
    with pytest.raises(ValueError, match="at least 3 samples"):
        fit_gaussian_copula(problem, np.zeros((2, 2)))


def test_latent_physical_round_trip():
    problem = Problem.from_dict(
        {
            "u": (-2.0, 3.0),
            "g": GaussianInputSpec(dist="gaussian", mean=1.0, variance=4.0),
        }
    )
    rng = np.random.default_rng(5)
    Z = rng.standard_normal((256, 2))
    X = latent_to_physical(problem, Z)
    Z_back = physical_to_latent(problem, X)
    # float32 JAX CDF on the inverse path limits the achievable precision.
    np.testing.assert_allclose(Z_back, Z, atol=5e-3)


def test_is_independent_and_identity_helper():
    assert is_independent(independent_correlation(4))
    assert not is_independent(np.array([[1.0, 0.2], [0.2, 1.0]]))


# ---------------------------------------------------------------------------
# Hard-error guards: design samplers
# ---------------------------------------------------------------------------


def _correlated_problem(D: int = 2, rho: float = 0.8) -> Problem:
    R = np.full((D, D), rho)
    np.fill_diagonal(R, 1.0)
    return _uniform_problem(D).with_correlation(R)


def test_sobol_sample_rejects_correlated_problem():
    with pytest.raises(ValueError, match=r"jaxgsa\.sobol\.sample.*independent inputs"):
        jaxgsa.sobol.sample(_correlated_problem(), 64, seed=1, verbose=False)


def test_morris_sample_rejects_correlated_problem():
    with pytest.raises(ValueError, match=r"jaxgsa\.morris\.sample.*independent inputs"):
        jaxgsa.morris.sample(_correlated_problem(), n_trajectories=4, seed=1, verbose=False)


def test_efast_sample_rejects_correlated_problem():
    with pytest.raises(ValueError, match=r"jaxgsa\.efast\.sample.*independent inputs"):
        jaxgsa.efast.sample(_correlated_problem(), n_per_curve=128, seed=1)


def test_sampler_guard_message_names_alternatives():
    with pytest.raises(
        ValueError,
        match=r"monte_carlo.*optimal_transport.*borgonovo.*hdmr.*hsic.*pawn",
    ):
        jaxgsa.sobol.sample(_correlated_problem(), 64, seed=1, verbose=False)


def test_identity_correlation_does_not_trip_the_guards():
    problem = _uniform_problem(2).with_correlation(np.eye(2))
    samples = jaxgsa.sobol.sample(problem, 16, seed=1, verbose=False)
    assert samples.n_runs >= 16


# ---------------------------------------------------------------------------
# Hard-error guards: correlation-naive analyzers
# ---------------------------------------------------------------------------


def _correlated_xy(D: int = 2, N: int = 64):
    problem = _correlated_problem(D)
    X = jaxgsa.sampling.monte_carlo(problem, N, seed=7)
    Y = np.asarray(X[:, 0])
    return problem, jnp.asarray(X), jnp.asarray(Y)


def test_pce_analyze_rejects_correlated_problem():
    problem, X, Y = _correlated_xy()
    with pytest.raises(ValueError, match=r"jaxgsa\.pce\.analyze.*independent inputs"):
        jaxgsa.pce.analyze(problem, X, Y)


def test_dgsm_analyze_rejects_correlated_problem_on_both_paths():
    problem, X, Y = _correlated_xy()
    with pytest.raises(ValueError, match=r"jaxgsa\.dgsm\.analyze.*independent inputs"):
        jaxgsa.dgsm.analyze(problem, fn=lambda x: x[0], X=X)
    dfdx = jnp.zeros((X.shape[0], 2))
    with pytest.raises(ValueError, match=r"jaxgsa\.dgsm\.analyze.*independent inputs"):
        jaxgsa.dgsm.analyze(problem, Y=Y, dfdx=dfdx)


def test_shapley_pce_backend_rejects_correlated_problem():
    problem, X, Y = _correlated_xy()
    with pytest.raises(ValueError, match=r"shapley.*backend='pce'.*include_correlative=True"):
        jaxgsa.shapley.analyze(problem, X, Y, backend="pce")


def test_analyzer_guard_message_names_alternatives():
    problem, X, Y = _correlated_xy()
    with pytest.raises(
        ValueError,
        match=r"optimal_transport.*borgonovo.*hdmr.*hsic.*pawn",
    ):
        jaxgsa.pce.analyze(problem, X, Y)


def test_correlation_tolerant_analyzers_accept_correlated_problem():
    problem, X, Y = _correlated_xy(N=512)  # HDMR needs at least 300 samples
    # Each tolerant method must run to completion without the guard tripping.
    jaxgsa.pawn.analyze(problem, X, Y)
    jaxgsa.hsic.analyze(problem, X, Y)
    jaxgsa.borgonovo.analyze(problem, X, Y)
    jaxgsa.optimal_transport.analyze(problem, X, Y)
    jaxgsa.hdmr.analyze(problem, X, Y)
    jaxgsa.shapley.analyze(problem, X, Y, backend="hdmr", include_correlative=True)
