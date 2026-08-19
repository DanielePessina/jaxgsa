"""Tests for jaxgsa.vkoga — correlated variance-based GSA via a VKOGA surrogate.

The decisive check is a linear model with Gaussian marginals under a known
copula correlation, where every index of Li et al. (2010) is closed form:
for ``Y = a . X`` with ``X ~ N(0, R)``,

    V(Y)     = a' R a
    S_TC_i   = (R a)_i^2 / V(Y)
    S_TU_i   = a_i^2 (1 - R_i,rest R_rest^-1 R_rest,i) / V(Y)

and the model is additive, so ``S_U = S_TU`` and ``S_IU = 0``. Note that
``(R a)_i = sum_j R_ij a_j`` already contains ``a_i``, because ``R_ii = 1``.

The accuracy tests run under ``jax.enable_x64()``: the kernel solve squares
the condition number of the cross kernel, which float32 cannot carry. The
context manager restores the global flag on exit, so other test files are
unaffected. Shape and error-path tests stay in float32, where ``analyze``
warns about single precision.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

import jaxgsa
from jaxgsa import JaxgsaWarning
from jaxgsa.problem import Problem

# Closed-form linear-Gaussian reference, shared with test_kucherenko.py and
# test_correlated_agreement.py.
from _linear_gaussian import (  # isort: skip
    A_COEF,
    A_COEF_ASYM,
    ASYM_PROBLEM,
    GAUSS_PROBLEM,
    R_ASYM,
    R_GAUSS,
    analytic_indices,
)

# --- cheap uniform model for shape/contract tests ---------------------------

UNIFORM_PROBLEM = Problem(names=("u1", "u2"), bounds=((0.0, 1.0),) * 2)

# Explicit gamma/ridge skip the 10x10 cross-validation grid; small greedy and
# estimator budgets keep each fit well under a second. All sample sizes are
# powers of two so the Sobol' engines stay balanced (and silent).
SMALL_KWARGS = dict(
    gamma=3.0,
    ridge=1e-6,
    max_centers=64,
    n_outer=64,
    n_inner=16,
    n_variance=512,
    seed=0,
)


def _uniform_scalar(X: np.ndarray) -> np.ndarray:
    return X[:, 0] + X[:, 1] ** 2 + 0.5 * X[:, 0] * X[:, 1]


def _uniform_outputs(X: np.ndarray) -> dict[str, np.ndarray]:
    """The same smooth model in every supported output layout."""
    y0 = _uniform_scalar(X)
    Y2 = np.stack([y0, 2.0 * y0, 1.0 - y0], axis=-1)  # (N, K=3)
    Y3 = np.stack([Y2, Y2 + 0.5], axis=1)  # (N, T=2, K=3)
    return {"scalar": y0, "multi": Y2, "time": Y3}


@pytest.fixture(scope="module")
def gauss_result():
    """x64 analysis of the correlated linear-Gaussian model (computed once)."""
    with jax.enable_x64():
        X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 1536, seed=7)
        Y = X @ A_COEF
        return jaxgsa.vkoga.analyze(
            GAUSS_PROBLEM,
            X,
            Y,
            correlation=R_GAUSS,
            gamma=2.0,
            ridge=1e-10,
            max_centers=150,
            n_outer=256,
            n_inner=64,
            n_variance=4096,
            seed=0,
        )


@pytest.fixture(scope="module")
def asym_result():
    """x64 analysis of the D=4 asymmetric-structure case (computed once)."""
    with jax.enable_x64():
        X = jaxgsa.sampling.monte_carlo(ASYM_PROBLEM, 2048, seed=7)
        Y = X @ A_COEF_ASYM
        return jaxgsa.vkoga.analyze(
            ASYM_PROBLEM,
            X,
            Y,
            correlation=R_ASYM,
            gamma=2.0,
            ridge=1e-10,
            max_centers=200,
            n_outer=256,
            n_inner=64,
            n_variance=4096,
            seed=0,
        )


@pytest.fixture(scope="module")
def uniform_fits():
    """Small float32 fits of the uniform model in all three output layouts."""
    X = jaxgsa.sampling.monte_carlo(UNIFORM_PROBLEM, 256, seed=11)
    outputs = _uniform_outputs(X)
    fits = {}
    for label, Y in outputs.items():
        with pytest.warns(UserWarning, match="single precision"):
            fits[label] = jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, **SMALL_KWARGS)
    return X, outputs, fits


# --- closed-form validation --------------------------------------------------


def test_closed_form_total_indices(gauss_result):
    S_TC_true, S_TU_true, var_y = analytic_indices(A_COEF, R_GAUSS)
    # atol tightened from 2e-2 after dropping the iid inner-noise correction
    # (it biased small S_TC low; see _indices._total_correlated).
    np.testing.assert_allclose(np.asarray(gauss_result.S_TC), S_TC_true, atol=1e-2)
    np.testing.assert_allclose(np.asarray(gauss_result.S_TU), S_TU_true, atol=2e-2)
    np.testing.assert_allclose(float(np.asarray(gauss_result.variance)), var_y, rtol=5e-2)
    assert gauss_result.is_correlated
    assert 0 < gauss_result.n_centers <= 150


def test_closed_form_split_identities(gauss_result):
    """Tier T0 (closed form): an additive model has no interaction part.

    The test model is a sum of one-dimensional terms, so every interaction
    index is zero and the uncorrelated index equals its total.
    """
    S_TU = np.asarray(gauss_result.S_TU)
    S_U = np.asarray(gauss_result.S_U)
    # The model is additive: no interactions, and the uncorrelated part is all
    # of the total uncorrelated index.
    np.testing.assert_allclose(np.asarray(gauss_result.S_IU), 0.0, atol=3e-2)
    np.testing.assert_allclose(S_U, S_TU, atol=3e-2)


def test_independent_inputs_recover_sobol_indices():
    """With correlation=None the indices collapse to the classical ones."""
    S1_true = A_COEF**2 / np.sum(A_COEF**2)
    with jax.enable_x64():
        X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 1536, seed=7)
        Y = X @ A_COEF
        result = jaxgsa.vkoga.analyze(
            GAUSS_PROBLEM,
            X,
            Y,
            correlation=None,
            gamma=2.0,
            ridge=1e-10,
            max_centers=150,
            n_outer=256,
            n_inner=64,
            n_variance=4096,
            seed=0,
        )
    assert not result.is_correlated
    np.testing.assert_allclose(result.correlation, np.eye(3), atol=1e-12)
    S_TC = np.asarray(result.S_TC)
    np.testing.assert_allclose(S_TC, S1_true, atol=3e-2)
    np.testing.assert_allclose(np.asarray(result.S_C), 0.0, atol=3e-2)
    # Additive model, no correlation: every index tells the same story.
    np.testing.assert_allclose(np.asarray(result.S_TU), S_TC, atol=3e-2)
    np.testing.assert_allclose(np.asarray(result.S_U), S_TC, atol=3e-2)


def test_asymmetric_correlation_structure(asym_result):
    """A D=4 case with six distinct off-diagonals of mixed sign.

    R_GAUSS has one non-zero off-diagonal, so a transposition of the parameter
    axis could hide inside it. This case cannot be permuted onto itself, so any
    index that landed on the wrong parameter would show.
    """
    S_TC_true, S_TU_true, var_y = analytic_indices(A_COEF_ASYM, R_ASYM)
    S_TC = np.asarray(asym_result.S_TC)
    S_TU = np.asarray(asym_result.S_TU)
    # Measured errors are 0.013 (S_TC) and 0.003 (S_TU); the budgets keep a
    # small margin over that.
    np.testing.assert_allclose(S_TC, S_TC_true, atol=2e-2)
    np.testing.assert_allclose(S_TU, S_TU_true, atol=1e-2)
    # The ordering carries the transposition signal: x4 dominates, x1 is nearly
    # cancelled by its correlations, and no two entries are close enough to
    # swap by chance.
    assert list(np.argsort(S_TC)) == list(np.argsort(S_TC_true))
    assert list(np.argsort(S_TU)) == list(np.argsort(S_TU_true))
    # The surrogate under-resolves the Gaussian tails, so the variance comes
    # out low. Documented in docs/examples/vkoga.md; do not loosen this.
    np.testing.assert_allclose(float(np.asarray(asym_result.variance)), var_y, rtol=5e-2)


def test_parameter_permutation_equivariance():
    """Relabelling the parameters permutes the indices, and nothing else.

    Re-running the same model with its parameters in a different order must
    move every index to the new position of its parameter. An index computed
    against the wrong column would not survive the relabelling.
    """
    perm = np.array([2, 0, 3, 1])
    kwargs = dict(
        gamma=2.0,
        ridge=1e-10,
        max_centers=200,
        n_outer=256,
        n_inner=64,
        n_variance=4096,
        seed=0,
    )
    with jax.enable_x64():
        X = np.asarray(jaxgsa.sampling.monte_carlo(ASYM_PROBLEM, 2048, seed=7))
        Y = X @ A_COEF_ASYM
        base = jaxgsa.vkoga.analyze(ASYM_PROBLEM, X, Y, correlation=R_ASYM, **kwargs)
        # Same model, parameters relabelled: permute the columns, the
        # coefficients, and the correlation matrix together.
        permuted = jaxgsa.vkoga.analyze(
            ASYM_PROBLEM,
            X[:, perm],
            X[:, perm] @ A_COEF_ASYM[perm],
            correlation=R_ASYM[np.ix_(perm, perm)],
            **kwargs,
        )
    for name in ("S_TC", "S_TU", "S_U", "S_C", "S_IU"):
        np.testing.assert_allclose(
            np.asarray(getattr(permuted, name)),
            np.asarray(getattr(base, name))[perm],
            atol=2e-2,
            err_msg=name,
        )


# --- the additive-projection limit --------------------------------------------

# A correlated model with a genuine interaction, where the degree-6 additive
# component functions cannot represent what X_i explains alone. The raw S_U
# then exceeds S_TU and S_IU would go negative; the clip stops that.
R_INTERACTION = np.array(
    [
        [1.0, -0.32, -0.12],
        [-0.32, 1.0, -0.75],
        [-0.12, -0.75, 1.0],
    ]
)


def _interaction_model(X: np.ndarray) -> np.ndarray:
    return X @ np.array([1.0, 0.7, 0.4]) + 0.5 * X[:, 0] * X[:, 1] + 0.3 * np.tanh(X[:, 2])


def test_interaction_model_clips_s_u_and_warns():
    """S_IU never goes negative, and a wide clip is reported.

    On this model the additive projection absorbs interaction variance it
    cannot represent as a function of X_i alone, so the raw S_U overshoots
    S_TU. The estimator clips S_U to S_TU, which keeps S_IU >= 0, and warns
    because the overshoot is far above the noise floor.
    """
    with jax.enable_x64():
        X = np.asarray(jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 1024, seed=5))
        Y = _interaction_model(X)
        with pytest.warns(UserWarning, match="S_U exceeded S_TU"):
            result = jaxgsa.vkoga.analyze(
                GAUSS_PROBLEM,
                X,
                Y,
                correlation=R_INTERACTION,
                gamma=2.0,
                ridge=1e-10,
                max_centers=150,
                n_outer=256,
                n_inner=64,
                n_variance=4096,
                seed=0,
            )
    S_TU = np.asarray(result.S_TU)
    S_U = np.asarray(result.S_U)
    S_IU = np.asarray(result.S_IU)
    # The invariant the clip exists to hold.
    assert np.all(S_U <= S_TU + 1e-12)
    assert np.all(S_IU >= 0.0)
    # x3 is the parameter that overshot; the clip binds there exactly.
    assert S_IU[2] == pytest.approx(0.0, abs=1e-12)
    # S_C is left free: these correlations oppose the direct effects, so a
    # negative correlated part is the honest reading, not an artefact.
    assert np.all(np.asarray(result.S_C) < 0.0)


def test_additive_model_does_not_trigger_the_clip():
    """The clip must stay out of the way on a model the projection can fit.

    An additive model has no interaction variance for f_i to absorb, so S_U
    tracks S_TU on its own and no warning fires.
    """
    with jax.enable_x64():
        X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 1024, seed=5)
        Y = X @ A_COEF
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = jaxgsa.vkoga.analyze(
                GAUSS_PROBLEM,
                X,
                Y,
                correlation=R_INTERACTION,
                gamma=2.0,
                ridge=1e-10,
                max_centers=150,
                n_outer=256,
                n_inner=64,
                n_variance=4096,
                seed=0,
            )
    assert not [w for w in caught if "S_U exceeded" in str(w.message)]
    np.testing.assert_allclose(np.asarray(result.S_IU), 0.0, atol=3e-2)


# --- surrogate-quality diagnostic ---------------------------------------------


def test_oscillatory_model_warns_that_the_surrogate_failed():
    """A 12-cycle sine defeats the kernel, and the user is told.

    The greedy Gaussian kernel cannot resolve this frequency from 512 points.
    Every index is measured against the surrogate, so the reported ranking is
    meaningless — and without the warning it would look ordinary.
    """
    problem = Problem(names=("u1", "u2", "u3"), bounds=((0.0, 1.0),) * 3)
    X = jaxgsa.sampling.monte_carlo(problem, 512, seed=1)
    Y = np.sin(2.0 * np.pi * 12.0 * np.asarray(X)[:, 0]) + 0.5 * np.asarray(X)[:, 1]
    with pytest.warns(UserWarning, match="cross-validated surrogate error"):
        result = jaxgsa.vkoga.analyze(
            problem,
            X,
            Y,
            ridge=1e-6,  # gamma is cross-validated; ridge fixed keeps it to 10 fits
            max_centers=100,
            n_folds=4,
            n_outer=64,
            n_inner=16,
            n_variance=512,
            seed=0,
        )
    # The honest out-of-sample score reaches the result, and it is the thing
    # the warning fired on: the surrogate explains nothing.
    assert result.cv_rmse is not None
    assert result.cv_rmse > 0.5 * float(Y.std())


# --- correlation handling -----------------------------------------------------


def test_problem_correlation_is_read_by_default():
    """analyze() with no correlation argument reads problem.correlation.

    A problem that declares R and a plain problem with R passed per-call must
    produce identical results on the same seed: same matrix, same draws.
    """
    X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 256, seed=2)
    Y = X @ A_COEF
    problem_corr = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    with pytest.warns(UserWarning, match="single precision"):
        from_problem = jaxgsa.vkoga.analyze(problem_corr, X, Y, **SMALL_KWARGS)
    with pytest.warns(UserWarning, match="single precision"):
        from_override = jaxgsa.vkoga.analyze(
            GAUSS_PROBLEM, X, Y, correlation=R_GAUSS, **SMALL_KWARGS
        )
    assert from_problem.is_correlated
    np.testing.assert_array_equal(from_problem.correlation, from_override.correlation)
    for field in ("S_TC", "S_TU", "S_U", "S_C", "S_IU"):
        np.testing.assert_array_equal(
            np.asarray(getattr(from_problem, field)),
            np.asarray(getattr(from_override, field)),
        )


def test_explicit_matrix_overrides_problem_correlation():
    """A per-call matrix wins over the problem's declaration."""
    X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 256, seed=2)
    Y = X @ A_COEF
    problem_corr = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    with pytest.warns(UserWarning, match="single precision"):
        result = jaxgsa.vkoga.analyze(problem_corr, X, Y, correlation=np.eye(3), **SMALL_KWARGS)
    assert not result.is_correlated
    np.testing.assert_allclose(result.correlation, np.eye(3), atol=1e-12)


def test_correlation_string_raises():
    """Every string is rejected; the error names the one supported workflow."""
    X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 32, seed=0)
    Y = X @ A_COEF
    for value in ("empirical", "bogus"):
        with (
            pytest.raises(ValueError, match="fit_correlation"),
            pytest.warns(UserWarning),
        ):
            jaxgsa.vkoga.analyze(GAUSS_PROBLEM, X, Y, correlation=value, **SMALL_KWARGS)


def test_correlation_matrix_validation_errors():
    """A correlation matrix of the wrong size is refused, naming the size."""
    X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 32, seed=0)
    Y = X @ A_COEF
    # The single-precision warning fires before validation raises.
    with pytest.raises(ValueError, match=r"must be \(3, 3\)"), pytest.warns(UserWarning):
        jaxgsa.vkoga.analyze(GAUSS_PROBLEM, X, Y, correlation=np.eye(2), **SMALL_KWARGS)


def test_indefinite_correlation_is_repaired():
    """A mildly indefinite matrix is projected to PD, not rejected."""
    R = np.array([[1.0, 0.52, 0.52], [0.52, 1.0, -0.52], [0.52, -0.52, 1.0]])
    assert np.linalg.eigvalsh(R).min() < 0  # genuinely indefinite as declared
    X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 256, seed=2)
    Y = X @ A_COEF
    with pytest.warns(UserWarning, match="single precision"):
        result = jaxgsa.vkoga.analyze(GAUSS_PROBLEM, X, Y, correlation=R, **SMALL_KWARGS)
    assert np.linalg.eigvalsh(result.correlation).min() > 0
    np.testing.assert_allclose(np.diag(result.correlation), 1.0, atol=1e-12)
    assert result.is_correlated
    assert np.all(np.isfinite(np.asarray(result.S_TC)))


# --- output contract ----------------------------------------------------------


def test_output_shape_contract(uniform_fits):
    _, _, fits = uniform_fits
    D = 2
    expected = {"scalar": (D,), "multi": (3, D), "time": (2, 3, D)}
    for label, result in fits.items():
        index_shape = expected[label]
        for name in ("S_TC", "S_TU", "S_U", "S_C", "S_IU"):
            assert getattr(result, name).shape == index_shape, (label, name)
        # Per-slice diagnostics drop the parameter axis.
        assert result.variance.shape == index_shape[:-1]
        assert result.rmse is not None
        assert result.rmse.shape == index_shape[:-1]


def test_predict_matches_training_layout_and_fit(uniform_fits):
    X, outputs, fits = uniform_fits
    for label, result in fits.items():
        Y = outputs[label]
        pred = np.asarray(result.predict(X))
        assert pred.shape == Y.shape, label
        # Predicting the training rows reproduces the state's own training RMSE.
        rmse = np.sqrt(np.mean((pred - Y) ** 2, axis=0))
        np.testing.assert_allclose(rmse, np.asarray(result.rmse), rtol=1e-2, atol=1e-5)
        assert np.all(rmse < 0.2 * np.std(_uniform_scalar(X)))


def test_predict_new_points_and_batching(uniform_fits):
    _, _, fits = uniform_fits
    result = fits["scalar"]
    X_new = jaxgsa.sampling.monte_carlo(UNIFORM_PROBLEM, 128, seed=5)
    pred = np.asarray(result.predict(X_new))
    assert pred.shape == (128,)
    truth = _uniform_scalar(X_new)
    assert np.sqrt(np.mean((pred - truth) ** 2)) < 0.2 * np.std(truth)
    # Batching only reassociates floating point; same precision, so tight.
    pred_batched = np.asarray(result.predict(X_new, batch_size=37))
    np.testing.assert_allclose(pred_batched, pred, rtol=2e-5, atol=1e-6)


def test_shapley_raises_not_implemented(uniform_fits):
    _, _, fits = uniform_fits
    with pytest.raises(NotImplementedError, match="Shapley"):
        fits["scalar"].shapley()


def test_to_dataset_carries_the_coordinate_and_attribute_values(gauss_result):
    """The dataset's names and dims are snapshot-pinned; the values are not.

    ``test_result_schema.py`` fixes the data-var names, dims and attribute
    keys for every method. What it cannot see is what the coordinates and
    attributes actually hold, so those claims live here.
    """
    ds = gauss_result.to_dataset()
    assert list(ds.coords["param"].values) == ["x1", "x2", "x3"]
    assert list(ds.coords["param_i"].values) == ["x1", "x2", "x3"]
    np.testing.assert_allclose(ds["correlation"].values, R_GAUSS, atol=1e-12)
    assert ds.attrs["is_correlated"] is True
    assert ds.attrs["n_centers"] == gauss_result.n_centers


# --- argument validation ------------------------------------------------------


def test_scalar_argument_validation_raises_early():
    """Bad scalar arguments raise before any expensive work, without warnings."""
    X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 32, seed=0)
    Y = X @ A_COEF
    cases = [
        ({"n_folds": 1}, "n_folds must be >= 2"),
        ({"n_outer": 1}, "n_outer must be >= 2"),
        ({"n_inner": 1}, "n_inner must be >= 2"),
        ({"n_variance": 1}, "n_variance must be >= 2"),
        ({"max_centers": 0}, "max_centers must be >= 1"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"gamma": -1.0}, "gamma must be a finite positive number"),
        ({"gamma": 0.0}, "gamma must be a finite positive number"),
        ({"gamma": float("nan")}, "gamma must be a finite positive number"),
        ({"ridge": -1.0}, "ridge must be a finite positive number"),
        ({"ridge": 0.0}, "ridge must be a finite positive number"),
        ({"ridge": float("inf")}, "ridge must be a finite positive number"),
    ]
    for overrides, message in cases:
        kwargs = dict(SMALL_KWARGS, **overrides)
        # Validation runs first: no fit, no CV, and no precision warning yet.
        with pytest.raises(ValueError, match=message):
            jaxgsa.vkoga.analyze(GAUSS_PROBLEM, X, Y, **kwargs)


def test_odd_sample_sizes_round_up_to_powers_of_two():
    """Non-power-of-two sizes are rounded up; scipy emits no balance warnings."""
    X = jaxgsa.sampling.monte_carlo(UNIFORM_PROBLEM, 128, seed=0)
    Y = _uniform_scalar(X)
    kwargs = dict(SMALL_KWARGS, n_outer=100, n_inner=9, n_variance=1000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, **kwargs)
    assert not [w for w in caught if "balance" in str(w.message)]
    assert np.all(np.isfinite(np.asarray(result.S_TC)))


# --- precision, cross-validation, determinism ---------------------------------


def test_masked_selection_stops_like_a_training_rows_fit():
    """Held-out rows must not leak into the residual stopping rule.

    A greedy sweep with a train_mask must stop at the same centre count as
    the same sweep run on the training rows alone. Before the residual
    re-mask, the held-out rows accumulated minus-interpolant values, the
    stopping norm plateaued, and every cross-validation fit ran to
    max_centers while the final fit stopped early — so the two scored
    hyperparameters under different stopping behaviour.
    """
    import jax.numpy as jnp

    from jaxgsa._core.transforms import cdf_to_unit_interval
    from jaxgsa.vkoga._engine import _select_centers

    with jax.enable_x64():
        X = jaxgsa.sampling.monte_carlo(UNIFORM_PROBLEM, 256, seed=11)
        U = jnp.asarray(cdf_to_unit_interval(jnp.asarray(X), UNIFORM_PROBLEM))
        y = _uniform_scalar(np.asarray(X))
        Y = jnp.asarray((y - y.mean())[:, None])
        mask = np.ones(256, dtype=bool)
        mask[::10] = False  # hold out every tenth row, CV-style

        _, m_masked = _select_centers(
            U,
            Y,
            gamma=3.0,
            max_centers=256,
            tol_residual=0.05,
            train_mask=jnp.asarray(mask),
        )
        _, m_train_only = _select_centers(
            U[np.flatnonzero(mask)],
            Y[np.flatnonzero(mask)],
            gamma=3.0,
            max_centers=256,
            tol_residual=0.05,
        )

    assert int(m_masked) == int(m_train_only)
    # The loose tolerance stops the sweep long before the candidate pool is
    # exhausted; running to the cap is the regression signature.
    assert int(m_masked) < int(mask.sum())


def test_determinism_by_seed():
    X = jaxgsa.sampling.monte_carlo(UNIFORM_PROBLEM, 256, seed=11)
    Y = _uniform_scalar(X)
    R_uni = np.array([[1.0, 0.4], [0.4, 1.0]])

    def run(seed: int):
        kwargs = dict(SMALL_KWARGS, seed=seed)
        with pytest.warns(UserWarning, match="single precision"):
            return jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, correlation=R_uni, **kwargs)

    first, again = run(0), run(0)
    np.testing.assert_array_equal(np.asarray(first.S_TC), np.asarray(again.S_TC))
    np.testing.assert_array_equal(np.asarray(first.S_TU), np.asarray(again.S_TU))

    other = run(1)
    # A different quasi-random stream moves the estimates, but not by much.
    assert not np.allclose(np.asarray(first.S_TC), np.asarray(other.S_TC), atol=1e-6)
    np.testing.assert_allclose(np.asarray(first.S_TC), np.asarray(other.S_TC), atol=0.25)


# --- on_invalid policy --------------------------------------------------------


class TestVKOGAOnInvalid:
    """T4: ``vkoga.analyze`` applies the shared non-finite policy.

    Before the policy, a single non-finite ``Y`` made every cross-validation
    score non-finite and the caller met a ``RuntimeError`` about failed kernel
    solves. The cause it named was wrong, and no message pointed at the bad
    row. The check now runs before any fitting.
    """

    @staticmethod
    def _data(n: int = 256, seed: int = 5):
        """A small smooth sample on the cheap two-parameter uniform problem."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(0.0, 1.0, size=(n, 2))
        return X, _uniform_scalar(X)

    def test_raise_is_the_default_and_names_the_rows(self):
        """T4: a non-finite Y refuses the analysis and says which rows carry it."""
        X, Y = self._data()
        Y = Y.copy()
        Y[4] = np.nan
        Y[9] = np.inf
        with pytest.raises(ValueError, match="non-finite") as exc:
            jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, **SMALL_KWARGS)
        message = str(exc.value)
        assert "jaxgsa.vkoga.analyze" in message
        assert "2 of 256 rows" in message
        assert "[4, 9]" in message

    def test_a_non_finite_y_no_longer_reaches_the_cross_validation(self):
        """T4: the misleading 'kernel solves failed' RuntimeError is unreachable.

        This is the exact regression the policy was added for. With ``gamma``
        and ``ridge`` unset the hyperparameter search runs, and one NaN row
        used to make every score on the grid non-finite. The user then got a
        ``RuntimeError`` blaming conditioning and suggesting float64, with no
        mention of the bad row. It must now be a ``ValueError`` that names the
        row instead.
        """
        X, Y = self._data(n=64)
        Y = Y.copy()
        Y[11] = np.nan
        cv_kwargs = dict(SMALL_KWARGS)
        cv_kwargs.pop("gamma")
        cv_kwargs.pop("ridge")
        cv_kwargs["n_folds"] = 2
        cv_kwargs["max_centers"] = 16
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="non-finite") as exc:
                jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, **cv_kwargs)
        message = str(exc.value)
        assert "cross-validation" not in message
        assert "[11]" in message

    def test_propagate_warns_and_lets_the_nan_reach_the_indices(self):
        """T4: 'propagate' keeps every row, so the indices come out non-finite."""
        X, Y = self._data()
        Y = Y.copy()
        Y[4] = np.nan
        with pytest.warns(JaxgsaWarning, match="reaches the indices"):
            result = jaxgsa.vkoga.analyze(
                UNIFORM_PROBLEM, X, Y, on_invalid="propagate", **SMALL_KWARGS
            )
        assert not np.all(np.isfinite(np.asarray(result.S_TC)))
        assert result.invalid.policy == "propagate"
        assert result.invalid.n_invalid == 1

    def test_drop_removes_the_rows_and_the_fit_is_finite_again(self):
        """T4: 'drop' fits the surrogate on the survivors, so indices are finite."""
        X, Y = self._data()
        Y = Y.copy()
        Y[4] = np.nan
        with pytest.warns(JaxgsaWarning, match="dropped 1 of 256 rows"):
            result = jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, on_invalid="drop", **SMALL_KWARGS)
        assert np.all(np.isfinite(np.asarray(result.S_TC)))
        assert np.all(np.isfinite(np.asarray(result.S_TU)))
        assert result.invalid.unit_indices == (4,)
        assert result.invalid.sources == ("Y",)

    def test_a_non_finite_input_is_caught_too(self):
        """T4: X is checked as well as Y, and ``sources`` says which array.

        A NaN input reaches the marginal CDF map and then the kernel metric,
        so it poisons the surrogate just as thoroughly as a NaN output.
        """
        X, Y = self._data()
        X = X.copy()
        X[6, 0] = np.nan
        with pytest.raises(ValueError, match=r"in X\.") as exc:
            jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, **SMALL_KWARGS)
        assert "[6]" in str(exc.value)
        with pytest.warns(JaxgsaWarning):
            result = jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, on_invalid="drop", **SMALL_KWARGS)
        assert result.invalid.sources == ("X",)

    def test_dropping_below_the_fold_count_refuses(self):
        """T4: 'drop' will not leave fewer rows than there are folds.

        A fold with no rows in it scores nothing, so the floor is ``n_folds``
        and the refusal names the policy rather than failing later inside the
        cross-validation.
        """
        X, Y = self._data(n=16)
        Y = Y.copy()
        Y[8:] = np.nan
        with pytest.raises(ValueError, match="every usable row was removed") as exc:
            jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, on_invalid="drop", **SMALL_KWARGS)
        assert "at least 10" in str(exc.value)
