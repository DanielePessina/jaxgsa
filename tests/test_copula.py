"""Tests for the Gaussian-copula backend and the correlated-problem guards."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

import jaxgsa
from jaxgsa._core.copula import (
    _MIN_EIGENVALUE,
    _REPAIR_NOISE,
    _project_to_correlation,
    _safe_cholesky,
    _spearman_to_latent,
    build_conditional_plan,
    canonicalize_correlation,
    correlation_from_covariance,
    fit_gaussian_copula,
    latent_normal_sample,
    latent_to_physical,
)
from jaxgsa._core.validation import _correlation_tolerant_methods
from jaxgsa.problem import GaussianInputSpec, Problem

# An indefinite candidate: pairwise entries are individually valid but jointly
# inconsistent (eigenvalues of this matrix include a negative one). The repair
# moves an entry by 0.4, so a declared matrix like this is rejected outright.
_INDEFINITE_R = np.array(
    [
        [1.0, 0.9, 0.9],
        [0.9, 1.0, -0.9],
        [0.9, -0.9, 1.0],
    ]
)


def _equicorrelated(rho: float, D: int = 3) -> np.ndarray:
    """Return the ``(D, D)`` matrix with ``rho`` off the diagonal.

    Its smallest eigenvalue is ``1 + (D - 1) rho``, so ``rho`` places the
    matrix in any severity band we want to test.
    """
    R = np.full((D, D), float(rho))
    np.fill_diagonal(R, 1.0)
    return R


# Smallest eigenvalue -1e-9, so the repair only lifts it to the 1e-8 floor.
# The largest entrywise change is 9e-9: numerical noise, reported to nobody.
_NOISE_R = _equicorrelated(1.0 - 1e-9)

# Smallest eigenvalue -0.01. The repair moves an entry by 5e-3, which is under
# the 0.05 material threshold, so a declared matrix like this warns.
_MILD_INDEFINITE_R = _equicorrelated(-0.505)


def _uniform_problem(D: int = 2) -> Problem:
    return Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(D)})


# ---------------------------------------------------------------------------
# canonicalize_correlation / repair warning
# ---------------------------------------------------------------------------


def test_canonicalize_correlation_rejects_a_matrix_of_the_wrong_size():
    """The shape check is the one this file owns.

    Asymmetry, a non-unit diagonal and an out-of-range entry are covered on
    both correlation kinds by ``test_canonicalize_rejects_bad_diagonal_on_
    every_kind`` and ``test_canonicalize_rejects_asymmetry_and_out_of_range_
    on_every_kind`` below.
    """
    with pytest.raises(ValueError, match=r"must be \(2, 2\)"):
        canonicalize_correlation(np.eye(3), 2)


def test_repair_warning_fires_for_mildly_indefinite_declared_matrix():
    # UserWarning is not promoted to an error by the pytest config, so the
    # firing case must be asserted explicitly with pytest.warns.
    with pytest.warns(UserWarning, match="not positive definite"):
        repaired = canonicalize_correlation(_MILD_INDEFINITE_R, 3, policy="declared")
    eigenvalues = np.linalg.eigvalsh(repaired)
    assert eigenvalues.min() > 0
    np.testing.assert_allclose(np.diag(repaired), 1.0, atol=1e-12)


def test_repair_of_declared_matrix_is_silent_at_noise_level():
    """A repair that only lifts the eigenvalue floor says nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        repaired = canonicalize_correlation(_NOISE_R, 3, policy="declared")
    assert 0.0 < np.abs(repaired - _NOISE_R).max() < _REPAIR_NOISE


def test_repair_of_declared_matrix_raises_when_material():
    with pytest.raises(ValueError, match="too far to accept"):
        canonicalize_correlation(_INDEFINITE_R, 3, policy="declared")


def test_repair_stays_silent_for_valid_matrix():
    R = np.array([[1.0, 0.3], [0.3, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        canonicalize_correlation(R, 2, policy="declared")


def test_fitted_repair_stays_silent_below_the_material_threshold():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        canonicalize_correlation(_MILD_INDEFINITE_R, 3)  # policy defaults to "fitted"


def test_fitted_repair_warns_but_never_raises_when_material():
    with pytest.warns(UserWarning, match="the fitted correlation matrix"):
        repaired = canonicalize_correlation(_INDEFINITE_R, 3)
    assert np.linalg.eigvalsh(repaired).min() > 0


def test_fit_gaussian_copula_never_raises_on_degenerate_data():
    """A duplicated column makes the rank estimate singular; the fit survives."""
    rng = np.random.default_rng(0)
    column = rng.normal(size=64)
    X = np.column_stack([column, column, rng.normal(size=64)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R = fit_gaussian_copula(_uniform_problem(3), X)
    assert np.linalg.eigvalsh(R).min() > 0


# ---------------------------------------------------------------------------
# severity reported in the units the user declared
# ---------------------------------------------------------------------------


def test_spearman_repair_reports_the_change_on_the_spearman_scale():
    """The user wrote Spearman numbers, so the message must use them.

    This matrix straddles the material threshold: the change is 5.04e-2 on the
    latent scale but 4.99e-2 on the Spearman scale. Reporting the declared
    scale therefore also decides between a warning and an error.
    """
    from jaxgsa._core.copula import _latent_to_spearman, _spearman_to_latent

    declared = _equicorrelated(-0.5325)
    with pytest.warns(UserWarning, match="Spearman scale") as record:
        repaired = canonicalize_correlation(declared, 3, kind="spearman", policy="declared")

    latent_change = np.abs(repaired - _spearman_to_latent(declared)).max()
    spearman_change = np.abs(_latent_to_spearman(repaired) - declared).max()
    assert latent_change >= 0.05 > spearman_change
    assert f"{spearman_change:.3e}" in str(record[0].message)


def test_spearman_round_trip_is_exact():
    """Reporting on the declared scale is only honest if the inverse is exact."""
    from jaxgsa._core.copula import _latent_to_spearman, _spearman_to_latent

    R = np.array([[1.0, 0.7, -0.3], [0.7, 1.0, 0.2], [-0.3, 0.2, 1.0]])
    np.testing.assert_allclose(_latent_to_spearman(_spearman_to_latent(R)), R, atol=1e-15)


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


@pytest.mark.parametrize("bad_diagonal", [0.0, 0.5, 2.0, -1.0])
@pytest.mark.parametrize("kind", ["latent", "spearman"])
def test_canonicalize_rejects_bad_diagonal_on_every_kind(bad_diagonal, kind):
    """Structural checks must run on the declared matrix, before conversion.

    The Spearman conversion pins the diagonal to exactly 1, so a check made
    after it accepted any diagonal at all.
    """
    R = [[bad_diagonal, 0.3], [0.3, bad_diagonal]]
    with pytest.raises(ValueError, match="unit diagonal"):
        canonicalize_correlation(R, 2, kind=kind)


@pytest.mark.parametrize("kind", ["latent", "spearman"])
def test_canonicalize_rejects_asymmetry_and_out_of_range_on_every_kind(kind):
    with pytest.raises(ValueError, match="symmetric"):
        canonicalize_correlation([[1.0, 0.5], [0.2, 1.0]], 2, kind=kind)
    with pytest.raises(ValueError, match=r"lie in \[-1, 1\]"):
        canonicalize_correlation([[1.0, 1.5], [1.5, 1.0]], 2, kind=kind)


# ---------------------------------------------------------------------------
# positive-definiteness repair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "R",
    [
        _INDEFINITE_R,
        np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]),
        np.array([[1.0, 1.0, 0.4], [1.0, 1.0, 0.4], [0.4, 0.4, 1.0]]),  # exactly singular
    ],
)
def test_repair_is_idempotent(R):
    """A repaired matrix must survive a second repair bit for bit.

    One clip-then-renormalise pass leaves the smallest eigenvalue under the
    floor, so a naive repair keeps nudging the matrix on every round trip.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the fitted policy reports large repairs
        once = _project_to_correlation(R)
        twice = _project_to_correlation(once)
    np.testing.assert_array_equal(twice, once)
    assert np.linalg.eigvalsh(once).min() >= _MIN_EIGENVALUE
    np.testing.assert_array_equal(np.diag(once), np.ones(R.shape[0]))


def test_repair_survives_a_problem_metadata_round_trip():
    """Serializing and reloading a repaired correlation must not move it."""
    import json

    from jaxgsa._core.samples import _problem_from_meta, _problem_to_meta

    problem = Problem.from_dict({"x0": (0.0, 1.0), "x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    with pytest.warns(UserWarning, match="not positive definite"):
        problem = problem.with_correlation(_MILD_INDEFINITE_R)
    stored = problem.correlation
    assert stored is not None
    for _ in range(3):
        reloaded = _problem_from_meta(json.loads(json.dumps(_problem_to_meta(problem))))
        assert reloaded.correlation is not None
        np.testing.assert_array_equal(reloaded.correlation, stored)
        problem = reloaded


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
    # Exact diagonal means the result survives canonicalize_correlation unchanged.
    np.testing.assert_allclose(canonicalize_correlation(R, 2), R, atol=1e-15)


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


def test_fit_gaussian_copula_averages_tied_ranks():
    # Regression test for position-dependent tie ranks. Column 0 holds two
    # tied levels in a structured block order. Column 1 increases with the
    # row index. The true Spearman correlation is exactly zero. The old
    # double-argsort fit ranked ties by row position. That gave a spurious
    # rho_s of about -0.305 here (latent rho about -0.318). Average ranks
    # must reproduce scipy.stats.spearmanr exactly.
    problem = _uniform_problem(2)
    x0 = np.repeat([0.0, 1.0, 1.0, 0.0], 50)
    x1 = np.arange(200, dtype=np.float64)
    X = np.column_stack([x0, x1])

    rho_s = scipy.stats.spearmanr(x0, x1).statistic
    expected = 2.0 * np.sin(np.pi * rho_s / 6.0)

    fitted = fit_gaussian_copula(problem, X)
    assert abs(fitted[0, 1] - expected) < 1e-12

    # The public entry point must agree.
    public = jaxgsa.sampling.fit_correlation(problem, X)
    np.testing.assert_allclose(public, fitted, atol=1e-15)


def test_fit_gaussian_copula_rejects_bad_shapes():
    problem = _uniform_problem(2)
    with pytest.raises(ValueError, match=r"must be \(N, 2\)"):
        fit_gaussian_copula(problem, np.zeros((10, 3)))
    with pytest.raises(ValueError, match="at least 3 samples"):
        fit_gaussian_copula(problem, np.zeros((2, 2)))


def test_latent_to_physical_applies_the_marginal_inverse_cdfs():
    problem = Problem.from_dict(
        {
            "u": (-2.0, 3.0),
            "g": GaussianInputSpec(dist="gaussian", mean=1.0, variance=4.0),
        }
    )
    rng = np.random.default_rng(5)
    Z = rng.standard_normal((256, 2))
    X = latent_to_physical(problem, Z)
    # Column-by-column against scipy: X_i = F_i^{-1}(Phi(Z_i)).
    U = scipy.stats.norm.cdf(Z)
    np.testing.assert_allclose(X[:, 0], -2.0 + 5.0 * U[:, 0], atol=1e-12)
    expected_g = scipy.stats.norm.ppf(U[:, 1], loc=1.0, scale=2.0)
    np.testing.assert_allclose(X[:, 1], expected_g, atol=1e-9)


def test_unscrambled_latent_sample_skips_the_origin():
    """The unscrambled sequence starts past the origin, not at a clipped extreme.

    The origin maps through the probit to the clipped ~-7-sigma corner in
    every coordinate — a point no normal sample should contain. The sampler
    fast-forwards past it, so the first returned point is the sequence's
    second point (0.5, ..., 0.5), which maps to the origin of the latent
    space.
    """
    Z = latent_normal_sample(8, 3, seed=0, scramble=False)
    assert np.all(np.isfinite(Z))
    assert np.abs(Z).max() < 6.0
    np.testing.assert_allclose(Z[0], 0.0, atol=1e-12)


def test_scrambled_latent_sample_unchanged_by_origin_skip():
    """The scrambled path is a fixed-seed regression: no fast-forward there."""
    Z = latent_normal_sample(16, 3, seed=42, scramble=True)
    unit = scipy.stats.qmc.Sobol(d=3, scramble=True, seed=42).random(16)
    expected = scipy.stats.norm.ppf(np.clip(unit, 1e-12, 1.0 - 1e-12))
    np.testing.assert_array_equal(Z, expected)


# ---------------------------------------------------------------------------
# Hard-error guards: design samplers
# ---------------------------------------------------------------------------


def _correlated_problem(D: int = 2, rho: float = 0.8) -> Problem:
    R = np.full((D, D), rho)
    np.fill_diagonal(R, 1.0)
    return _uniform_problem(D).with_correlation(R)


def test_sampler_guard_message_names_alternatives():
    """The sampler refusal quotes the generated list, not a copy of it.

    ``tests/test_entry_preamble.py`` makes this claim for the *analysis*
    gate. This is the design gate, whose message is built at a different call
    site. Asserting against ``_correlation_tolerant_methods()`` rather than a
    transcribed ordering means adding a tolerant method cannot make the test
    stale without also making the message wrong.
    """
    with pytest.raises(ValueError) as excinfo:
        jaxgsa.sobol.sample(_correlated_problem(), 64, seed=1, verbose=False)
    message = str(excinfo.value)
    assert _correlation_tolerant_methods() in message
    # The design gate additionally has to say how to draw the samples.
    assert "jaxgsa.sampling.monte_carlo" in message


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


def test_dgsm_analyze_rejects_correlated_problem_on_both_paths():
    problem, X, Y = _correlated_xy()
    with pytest.raises(ValueError, match=r"jaxgsa\.dgsm\.analyze.*independent inputs"):
        jaxgsa.dgsm.analyze(problem, fn=lambda x: x[0], X=X)
    dfdx = jnp.zeros((X.shape[0], 2))
    with pytest.raises(ValueError, match=r"jaxgsa\.dgsm\.analyze.*independent inputs"):
        jaxgsa.dgsm.analyze(problem, Y=Y, dfdx=dfdx)


def test_conditional_plan_carries_the_factor_of_its_own_matrix():
    """T4 internal consistency: ``chol_full`` belongs to the plan's matrix.

    The plan used to be paired with a Cholesky factor computed separately at
    each call site, so a caller could hand one function the conditionals of
    one matrix and the joint factor of another. The factor now travels on the
    plan, and it must reconstruct exactly the matrix the plan was built from.
    """
    R = _equicorrelated(0.6, D=4)
    plan = build_conditional_plan(R)

    assert plan.chol_full.shape == (4, 4)
    # Exact reconstruction: the field is the factor of R, not of anything else.
    np.testing.assert_allclose(plan.chol_full @ plan.chol_full.T, R, atol=1e-12)
    # And it agrees bit-for-bit with the module's own convention.
    np.testing.assert_array_equal(plan.chol_full, _safe_cholesky(R))

    # A different matrix must give a different factor, so the two cannot be
    # silently interchanged.
    other = build_conditional_plan(_equicorrelated(0.2, D=4))
    assert not np.allclose(plan.chol_full, other.chol_full)


def test_safe_cholesky_matches_numpy_when_comfortably_pd():
    """T4 internal consistency: ``_safe_cholesky`` is a no-op on a healthy matrix.

    ``_safe_cholesky`` repairs a matrix that is only just positive definite.
    On a comfortably positive-definite one it must return exactly what
    ``np.linalg.cholesky`` returns, so moving the plan's call sites onto it
    cannot move a number.
    """
    R = _equicorrelated(0.5, D=3)
    np.testing.assert_array_equal(_safe_cholesky(R), np.linalg.cholesky(R))


def test_safe_cholesky_repairs_a_rank_deficient_matrix_silently():
    """T4 internal consistency: the repair branch, and how loud it is.

    This is the direction the no-op test above cannot see. On an exactly
    singular matrix ``np.linalg.cholesky`` raises ``LinAlgError`` while
    ``_safe_cholesky`` lifts the spectrum to ``_MIN_EIGENVALUE`` and returns a
    factor. Moving the plan's call sites onto ``_safe_cholesky`` therefore
    changed the behaviour for a non-positive-definite matrix from "fail" to
    "repair".

    Pinned here: the repair happens, it reconstructs the input to the lift
    tolerance, and it says nothing. Silence is deliberate, not an oversight:
    every matrix that reaches this function through the public API has already
    been through ``canonicalize_correlation``, which reports the repair it
    made on the scale the user declared. A second warning from here would
    report the same defect twice, in units nobody wrote.
    """
    R = np.ones((3, 3))  # rank 1: the extreme non-PD case
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(R)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        L = _safe_cholesky(R)

    assert L.shape == (3, 3)
    assert np.isfinite(L).all()
    np.testing.assert_array_equal(L, np.tril(L))
    # The factor is of the lifted matrix, so it reproduces R only to the size
    # of the lift, and its smallest eigenvalue sits at the floor, not at zero.
    np.testing.assert_allclose(L @ L.T, R, atol=10 * _MIN_EIGENVALUE)
    assert np.linalg.eigvalsh(L @ L.T).min() >= _MIN_EIGENVALUE / 2


def test_safe_cholesky_repair_is_unreachable_through_the_public_api():
    """T4 internal consistency: canonicalization gets there first.

    The behaviour change of the previous test is not reachable from
    ``kucherenko.sample`` or ``vkoga.analyze``. Both take their matrix from
    ``canonicalize_correlation``, directly or through ``problem.correlation``,
    and that already lifts the spectrum off zero. So the matrix handed to
    ``build_conditional_plan`` is positive definite, plain
    ``np.linalg.cholesky`` succeeds on it, and ``_safe_cholesky`` returns the
    same factor: the repair branch never runs.

    The declared matrix here is exactly singular, the worst case a user can
    write. Its repair moves an entry by about 4e-8, far under the noise
    threshold, so ``policy="declared"`` neither raises nor warns and the
    sampler runs on the lifted matrix. That silence is the point of the test:
    the guard against a truly inconsistent matrix is the material-repair band,
    not the Cholesky.
    """
    R_declared = np.ones((2, 2))  # rank 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        problem = _uniform_problem(D=2).with_correlation(R_declared)

    R = np.asarray(problem.correlation)
    assert np.linalg.eigvalsh(R).min() >= _MIN_EIGENVALUE
    # Plain cholesky already works here, so the two functions agree.
    np.testing.assert_array_equal(_safe_cholesky(R), np.linalg.cholesky(R))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        samples = jaxgsa.kucherenko.sample(problem, 64, seed=0).samples
    assert np.isfinite(np.asarray(samples, dtype=float)).all()

    # A genuinely inconsistent declared matrix is rejected before any factor
    # is taken, so no non-PD matrix reaches the sampler at all.
    with pytest.raises(ValueError, match="not positive definite"):
        _uniform_problem(D=3).with_correlation(_INDEFINITE_R)
