"""Tests for jaxgsa.vkoga — correlated variance-based GSA via a VKOGA surrogate.

The decisive check is a linear model with Gaussian marginals under a known
copula correlation, where every index of Li et al. (2010) is closed form:
for ``Y = a . X`` with ``X ~ N(0, R)``,

    V(Y)     = a' R a
    S_TC_i   = (a_i + sum_j a_j R_ji)^2 / V(Y)
    S_TU_i   = a_i^2 (1 - R_i,rest R_rest^-1 R_rest,i) / V(Y)

and the model is additive, so ``S_U = S_TU`` and ``S_IU = 0``.

The accuracy tests run under ``jax.enable_x64()``: the kernel solve squares
the condition number of the cross kernel, which float32 cannot carry. The
context manager restores the global flag on exit, so other test files are
unaffected. Shape and error-path tests stay in float32, where ``analyze``
warns about single precision.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jaxgsa
from jaxgsa.problem import Problem

# --- closed-form linear-Gaussian reference ----------------------------------

A_COEF = np.array([2.0, 1.0, 0.5])
RHO = 0.6

GAUSS_PROBLEM = Problem.from_dict(
    {
        "x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x2": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x3": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
    }
)

R_GAUSS = np.eye(3)
R_GAUSS[0, 1] = R_GAUSS[1, 0] = RHO


def _analytic_indices(a: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Closed-form (S_TC, S_TU, V(Y)) for a linear model on N(0, R) inputs."""
    var_y = float(a @ R @ a)
    S_TC = (R @ a) ** 2 / var_y
    D = a.shape[0]
    S_TU = np.empty(D)
    for i in range(D):
        rest = [j for j in range(D) if j != i]
        r = R[rest, i]
        S_TU[i] = a[i] ** 2 * (1.0 - r @ np.linalg.solve(R[np.ix_(rest, rest)], r)) / var_y
    return S_TC, S_TU, var_y


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
    S_TC_true, S_TU_true, var_y = _analytic_indices(A_COEF, R_GAUSS)
    np.testing.assert_allclose(np.asarray(gauss_result.S_TC), S_TC_true, atol=2e-2)
    np.testing.assert_allclose(np.asarray(gauss_result.S_TU), S_TU_true, atol=2e-2)
    np.testing.assert_allclose(float(np.asarray(gauss_result.variance)), var_y, rtol=5e-2)
    assert gauss_result.is_correlated
    assert 0 < gauss_result.n_centers <= 150


def test_closed_form_split_identities(gauss_result):
    S_TC = np.asarray(gauss_result.S_TC)
    S_TU = np.asarray(gauss_result.S_TU)
    S_U = np.asarray(gauss_result.S_U)
    # S_C and S_IU are defined as differences; the identities must be exact.
    np.testing.assert_allclose(np.asarray(gauss_result.S_C), S_TC - S_U, atol=1e-12)
    np.testing.assert_allclose(np.asarray(gauss_result.S_IU), S_TU - S_U, atol=1e-12)
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


# --- correlation handling -----------------------------------------------------


def test_empirical_correlation_is_fitted_from_x():
    L = np.linalg.cholesky(R_GAUSS)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((1024, 3)) @ L.T  # standard-normal marginals, rho12 = 0.6
    Y = X @ A_COEF
    with pytest.warns(UserWarning, match="single precision"):
        result = jaxgsa.vkoga.analyze(GAUSS_PROBLEM, X, Y, correlation="empirical", **SMALL_KWARGS)
    assert result.is_correlated
    np.testing.assert_allclose(result.correlation, result.correlation.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(result.correlation), 1.0, atol=1e-12)
    assert abs(result.correlation[0, 1] - RHO) < 0.1
    # The uncorrelated pairs must not pick up spurious structure.
    assert abs(result.correlation[0, 2]) < 0.1
    assert abs(result.correlation[1, 2]) < 0.1


def test_invalid_correlation_string_raises():
    X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 32, seed=0)
    Y = X @ A_COEF
    with pytest.raises(ValueError, match="empirical"), pytest.warns(UserWarning):
        jaxgsa.vkoga.analyze(GAUSS_PROBLEM, X, Y, correlation="bogus", **SMALL_KWARGS)


def test_correlation_matrix_validation_errors():
    X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 32, seed=0)
    Y = X @ A_COEF

    asymmetric = np.eye(3)
    asymmetric[0, 1] = 0.5
    out_of_range = np.eye(3)
    out_of_range[0, 1] = out_of_range[1, 0] = 1.5
    cases = [
        (np.eye(2), r"must be \(3, 3\)"),
        (asymmetric, "symmetric"),
        (np.eye(3) * 2.0, "unit diagonal"),
        (out_of_range, r"\[-1, 1\]"),
    ]
    for bad_matrix, message in cases:
        # The single-precision warning fires before validation raises.
        with pytest.raises(ValueError, match=message), pytest.warns(UserWarning):
            jaxgsa.vkoga.analyze(GAUSS_PROBLEM, X, Y, correlation=bad_matrix, **SMALL_KWARGS)


def test_indefinite_correlation_is_repaired():
    """A valid-looking but indefinite matrix is projected to PD, not rejected."""
    R = np.array([[1.0, 0.99, 0.99], [0.99, 1.0, -0.99], [0.99, -0.99, 1.0]])
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


def test_to_dataset_schema(gauss_result):
    ds = gauss_result.to_dataset()
    for name in ("S_TC", "S_TU", "S_U", "S_C", "S_IU"):
        assert ds[name].dims == ("param",)
    assert list(ds.coords["param"].values) == ["x1", "x2", "x3"]
    assert ds["variance"].dims == ()
    assert ds["rmse"].dims == ()
    assert ds["correlation"].dims == ("param_i", "param_j")
    assert list(ds.coords["param_i"].values) == ["x1", "x2", "x3"]
    np.testing.assert_allclose(ds["correlation"].values, R_GAUSS, atol=1e-12)
    assert ds.attrs["method"] == "vkoga"
    assert ds.attrs["correlated"] == 1
    assert ds.attrs["n_centers"] == gauss_result.n_centers


def test_to_dataset_time_series_dims(uniform_fits):
    _, _, fits = uniform_fits
    ds = fits["time"].to_dataset(time_coords=[0.5, 1.0])
    assert ds["S_TC"].dims == ("time", "output", "param")
    assert list(ds.coords["time"].values) == [0.5, 1.0]
    assert ds["variance"].dims == ("time", "output")
    assert ds.attrs["correlated"] == 0


# --- precision, cross-validation, determinism ---------------------------------


def test_float32_emits_precision_warning():
    X = jaxgsa.sampling.monte_carlo(UNIFORM_PROBLEM, 128, seed=0)
    Y = _uniform_scalar(X)
    with pytest.warns(UserWarning, match="single precision"):
        jaxgsa.vkoga.analyze(
            UNIFORM_PROBLEM,
            X,
            Y,
            gamma=3.0,
            ridge=1e-6,
            max_centers=32,
            n_outer=64,
            n_inner=16,
            n_variance=512,
        )


def test_cross_validation_path_resolves_gamma():
    """Leaving gamma=None cross-validates it over the built-in grid."""
    X = jaxgsa.sampling.monte_carlo(UNIFORM_PROBLEM, 128, seed=4)
    Y = _uniform_scalar(X)
    with pytest.warns(UserWarning, match="single precision"):
        result = jaxgsa.vkoga.analyze(
            UNIFORM_PROBLEM,
            X,
            Y,
            gamma=None,  # cross-validated; ridge fixed keeps the grid to 10 fits
            ridge=1e-6,
            max_centers=32,
            n_folds=4,
            n_outer=64,
            n_inner=16,
            n_variance=512,
            seed=0,
        )
    assert np.isfinite(result.gamma) and result.gamma > 0
    assert result.ridge == 1e-6
    assert np.all(np.isfinite(np.asarray(result.S_TC)))


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
