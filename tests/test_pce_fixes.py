"""Regression tests for the 1.1.2 PCE fixes."""

from __future__ import annotations

import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import JaxgsaWarning, pce
from jaxgsa._core.transforms import _cdf_by_column, _cdf_by_group
from jaxgsa.pce._analyze import _map_by_column, _map_by_group
from jaxgsa.pce._engine import (
    _design_matrix_dense,
    _design_matrix_sparse,
    build_multi_index,
    sobol_from_coefficients,
)
from jaxgsa.problem import GaussianInputSpec, InputSpecValue


def _mixed_problem(d: int) -> jaxgsa.Problem:
    kinds: list[InputSpecValue] = [
        (0.0, 1.0),
        GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
        GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0, low=-1.0, high=2.0),
        GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0, low=1.0, high=3.0),
        GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0, low=-8.0, high=8.0),
    ]
    return jaxgsa.Problem.from_dict({f"x{i}": kinds[i % len(kinds)] for i in range(d)})


def _inputs(d: int, n: int = 64) -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(0).uniform(0.05, 0.95, (n, d)).astype(np.float32))


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize(
    "types",
    [
        ("uniform",) * 6,
        ("gaussian",) * 6,
        ("uniform", "gaussian", "gaussian", "uniform", "gaussian", "uniform"),
    ],
)
def test_sparse_design_matrix_is_bitwise_equal_to_dense(types, order):
    X = _inputs(len(types)) * 2 - 1
    mi = build_multi_index(len(types), order)
    dense = _design_matrix_dense(X, mi, types, order)
    sparse = _design_matrix_sparse(X, mi, types, order)
    assert np.array_equal(np.asarray(dense), np.asarray(sparse))


@pytest.mark.parametrize("d", [5, 20])
def test_grouped_transforms_are_bitwise_equal_to_column_loops(d):
    problem = _mixed_problem(d)
    X = _inputs(d)
    assert np.array_equal(
        np.asarray(_cdf_by_column(X, problem)), np.asarray(_cdf_by_group(X, problem))
    )
    for order in (3, 8):
        col, col_types = _map_by_column(X, problem, order)
        grp, grp_types = _map_by_group(X, problem, order)
        assert col_types == grp_types
        assert np.array_equal(np.asarray(col), np.asarray(grp))


def test_large_d_fit_matches_small_d_path_values():
    """D above the cut-over runs the grouped/sparse code and stays accurate."""
    d = 20
    problem = jaxgsa.Problem([f"x{i}" for i in range(d)], [(0.0, 1.0)] * d)
    X = np.random.default_rng(1).uniform(0, 1, (1200, d)).astype(np.float32)
    Y = 3 * X[:, 0] + X[:, 1]
    r = pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=2, verbose=False)
    assert isinstance(r, pce.PCEResult)
    assert float(r.S1[0]) == pytest.approx(0.9, abs=1e-3)
    assert float(r.S1[1]) == pytest.approx(0.1, abs=1e-3)


def test_order_one_second_order_indices_are_zero_and_nan_when_constant():
    mi = build_multi_index(4, 1)
    coef = jnp.asarray([[0.5, 1.0, 2.0, 0.0, 0.0]])
    _, _, S2 = sobol_from_coefficients(coef, mi)
    off_diag = np.asarray(S2)[0][~np.eye(4, dtype=bool)]
    assert np.all(off_diag == 0.0)
    _, _, S2c = sobol_from_coefficients(coef, mi, is_constant=jnp.asarray([True]))
    assert np.all(np.isnan(np.asarray(S2c)))


def _three_inputs(n: int = 2000) -> tuple[jaxgsa.Problem, np.ndarray, np.random.Generator]:
    rng = np.random.default_rng(0)
    problem = jaxgsa.Problem(["a", "b", "c"], [(0, 1)] * 3)
    return problem, rng.uniform(0, 1, (n, 3)).astype(np.float32), rng


def _gaussian_fit(order: int, n: int, seed: int, **kwargs):
    gauss = GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0)
    problem = jaxgsa.Problem.from_dict({"a": gauss, "b": gauss})
    X = np.random.default_rng(seed).standard_normal((n, 2))
    if not jax.config.jax_enable_x64:
        X = X.astype(np.float32)
    Y = X[:, 0] + 0.5 * X[:, 1] + X[:, 0] * X[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pce.analyze(
            problem, jnp.asarray(X), jnp.asarray(Y), order=order, verbose=False, **kwargs
        )
    assert isinstance(result, pce.PCEResult)
    return np.asarray(result.S1)


@pytest.mark.parametrize("ridge", [1e-8, 1.0])
def test_duplicated_input_raises_in_both_dtypes(ridge):
    # Read off the inputs, so neither dtype nor ridge can hide it.
    problem, X, _ = _three_inputs()
    X[:, 2] = X[:, 1]
    Y = 2 * X[:, 0] + X[:, 1] ** 2
    with pytest.raises(ValueError, match=r"\[\('b', 'c'\)\] are equal in every row"):
        pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=3, ridge=ridge, verbose=False)


@pytest.mark.parametrize("batch_size", [None, 256])
def test_constant_input_raises_in_both_dtypes(batch_size):
    problem, X, _ = _three_inputs()
    X[:, 2] = 0.5
    Y = 2 * X[:, 0] + X[:, 1] ** 2
    with pytest.raises(ValueError, match=r"\['c'\] have the same value in every row"):
        pce.analyze(
            problem,
            jnp.asarray(X),
            jnp.asarray(Y),
            order=3,
            batch_size=batch_size,
            verbose=False,
        )


def test_integer_inputs_run_through_the_degenerate_input_check():
    # 1.1.1 accepted integer X and returned exact indices. The duplicate
    # check must not reject the dtype.
    rng = np.random.default_rng(1)
    problem = jaxgsa.Problem(["a", "b", "c"], [(0, 9)] * 3)
    dtype = np.int64 if jax.config.jax_enable_x64 else np.int32
    X = rng.integers(0, 10, (1000, 3)).astype(dtype)
    Y = (X[:, 0] + 2 * X[:, 1] + 0.5 * X[:, 2]).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=2, verbose=False)
        shap = jaxgsa.shapley.analyze(problem, jnp.asarray(X), jnp.asarray(Y), backend="pce")
    assert isinstance(result, pce.PCEResult)
    np.testing.assert_allclose(np.asarray(result.S1), [0.1905, 0.7619, 0.0476], atol=1e-3)
    assert isinstance(shap, jaxgsa.shapley.ShapleyResult)
    assert np.all(np.isfinite(np.asarray(shap.Sh)))


def _bool_inputs() -> tuple[jaxgsa.Problem, np.ndarray]:
    rng = np.random.default_rng(1)
    problem = jaxgsa.Problem(["a", "b", "c"], [(0, 1)] * 3)
    return problem, rng.integers(0, 2, (2000, 3)).astype(bool)


def test_boolean_inputs_run_through_the_degenerate_input_check():
    # 1.1.1 accepted boolean X and returned exact indices.
    problem, X = _bool_inputs()
    Y = X[:, 0] + 1.5 * X[:, 1] + 2.0 * X[:, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=1, verbose=False)
        shap = jaxgsa.shapley.analyze(
            problem, jnp.asarray(X), jnp.asarray(Y), backend="pce", order=1
        )
    assert isinstance(result, pce.PCEResult)
    np.testing.assert_allclose(np.asarray(result.S1), [0.1379, 0.3103, 0.5517], atol=1e-3)
    assert isinstance(shap, jaxgsa.shapley.ShapleyResult)
    assert np.all(np.isfinite(np.asarray(shap.Sh)))


@pytest.mark.parametrize(
    ("degenerate", "message"),
    [("constant", "have the same value in every row"), ("duplicate", "are equal in every row")],
)
def test_boolean_degenerate_inputs_raise_value_error(degenerate, message):
    problem, X = _bool_inputs()
    X[:, 2] = True if degenerate == "constant" else X[:, 1]
    Y = X[:, 0] + 1.5 * X[:, 1] + 2.0 * X[:, 2]
    with pytest.raises(ValueError, match=message):
        pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=1, verbose=False)


def test_degenerate_input_message_names_the_shapley_entry_point():
    problem, X, _ = _three_inputs()
    X[:, 2] = X[:, 1]
    Y = 2 * X[:, 0] + X[:, 1] ** 2
    with pytest.raises(ValueError, match=r"^jaxgsa\.shapley\.analyze: "):
        jaxgsa.shapley.analyze(problem, jnp.asarray(X), jnp.asarray(Y), backend="pce")


def test_near_duplicated_input_warns_in_both_dtypes():
    # Copies b up to 1e-3 noise. The share is about 3e-10 in both dtypes, and
    # the QR factor resolves it in float32 too, so both dtypes warn.
    problem, X, rng = _three_inputs()
    X[:, 2] = np.clip(X[:, 1] + 1e-3 * rng.standard_normal(2000), 0, 1).astype(np.float32)
    Y = 2 * X[:, 0] + X[:, 1] ** 2
    with pytest.warns(JaxgsaWarning, match="near singular"):
        pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=3, verbose=False)


def test_three_level_input_is_never_silent():
    # Exactly singular at order 3: the cubic in a three-level input lies in
    # the span of the lower terms. The ridge floors the share near 2e-10, and
    # both dtypes warn.
    problem, X, rng = _three_inputs()
    X[:, 2] = rng.choice(np.array([0.1, 0.5, 0.9], np.float32), 2000)
    Y = 2 * X[:, 0] + X[:, 1] ** 2
    with pytest.warns(JaxgsaWarning, match="near singular"):
        pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=3, verbose=False)


@pytest.mark.parametrize(("order", "n", "seed"), [(20, 1000, 0), (20, 1000, 1), (40, 2000, 1)])
def test_accurate_high_order_hermite_fit_is_not_refused_in_float64(order, n, seed):
    # Pivot shares fall far below any fixed limit at these orders, yet the
    # indices stay close to the truth [0.4444, 0.1111]. They may warn, never
    # raise.
    if not jax.config.jax_enable_x64:
        pytest.skip("float64 only")
    np.testing.assert_allclose(_gaussian_fit(order, n, seed), [0.4444, 0.1111], atol=1e-2)


def test_ridge_regularized_high_order_fit_is_not_refused_in_float32():
    # The share keeps the ridge, so ridge=1 keeps a float32 order-16 fit
    # finite and accurate.
    if jax.config.jax_enable_x64:
        pytest.skip("float32 only")
    np.testing.assert_allclose(_gaussian_fit(16, 1000, 0, ridge=1.0), [0.4444, 0.1111], atol=2e-2)


def test_high_order_hermite_fit_is_not_refused_in_float64():
    # A well-posed order-10 fit has pivot shares near 2e-6. float64 fits it
    # accurately and must not refuse it. float32 cannot fit it at all.
    if not jax.config.jax_enable_x64:
        pytest.skip("float64 only")
    rng = np.random.default_rng(0)
    gauss = GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0)
    problem = jaxgsa.Problem.from_dict({"a": gauss, "b": gauss, "c": gauss})
    X = rng.standard_normal((600, 3))
    Y = X[:, 0] + 0.3 * X[:, 1] + X[:, 2] ** 2
    result = pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=10, verbose=False)
    assert isinstance(result, pce.PCEResult)
    assert np.all(np.isfinite(np.asarray(result.S1)))


def test_well_posed_fit_emits_no_singularity_message():
    rng = np.random.default_rng(0)
    problem = jaxgsa.Problem(["a", "b"], [(0, 1)] * 2)
    X = rng.uniform(0, 1, (500, 2)).astype(np.float32)
    Y = X[:, 0] + X[:, 1] ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=3, verbose=False)


def _wide_truncated_problem() -> jaxgsa.Problem:
    return jaxgsa.Problem.from_dict(
        {
            "a": {"dist": "gaussian", "mean": 0.0, "variance": 1.0, "low": -8.0, "high": 8.0},
            "b": (0.0, 1.0),
        }
    )


def test_high_order_wide_truncation_is_exact_and_silent():
    # 1.1.2 switched this input to a Legendre basis above order 7 and warned.
    # At order 8 it returned S1 = 0.7141 and a cv_rmse of 0.23, where the exact
    # S1 is 0.7273. The basis is now exact for every order and no warning
    # speaks.
    rng = np.random.default_rng(0)
    X = np.stack([rng.standard_normal(3000), rng.uniform(0, 1, 3000)], 1).astype(np.float32)
    Y = X[:, 0] ** 2 + 3 * X[:, 1]
    problem = _wide_truncated_problem()
    for order in (3, 7, 8):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = pce.analyze(
                problem, jnp.asarray(X), jnp.asarray(Y), order=order, verbose=False
            )
        assert isinstance(result, pce.PCEResult)
        assert float(result.S1[0]) == pytest.approx(2 / 2.75, abs=5e-3)
        assert float(result.cv_rmse) < 1e-3


def test_constant_output_and_dropped_rows_emit_no_singularity_message():
    rng = np.random.default_rng(0)
    problem = jaxgsa.Problem(["a", "b", "c"], [(0, 1)] * 3)
    X = rng.uniform(0, 1, (400, 3)).astype(np.float32)
    Y = np.stack([2 * X[:, 0] + X[:, 1] ** 2, np.full(400, 3.3, np.float32)], axis=1)
    X[3, 0] = np.nan
    Y[7, 0] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), on_invalid="drop", verbose=False)
    assert not [w for w in caught if "singular" in str(w.message)]


def test_grouped_transforms_one_sided_tails_keep_column_order():
    specs: list[InputSpecValue] = [
        (0.0, 1.0),
        GaussianInputSpec(dist="gaussian", mean=2.0, variance=0.25, low=2.5),
        GaussianInputSpec(dist="gaussian", mean=0.0, variance=9.0, high=-1.0),
        GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0, low=-2.0),
        GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0, high=8.0),
    ]
    problem = jaxgsa.Problem.from_dict({f"x{i}": specs[(3 * i) % 5] for i in range(20)})
    X = _inputs(20) * 4 - 1
    assert np.array_equal(
        np.asarray(_cdf_by_column(X, problem)), np.asarray(_cdf_by_group(X, problem))
    )
    for order in (3, 8):
        col, col_types = _map_by_column(X, problem, order)
        grp, grp_types = _map_by_group(X, problem, order)
        assert col_types == grp_types
        assert np.array_equal(np.asarray(col), np.asarray(grp), equal_nan=True)


def _warn_sources(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return [(str(w.message), w.filename) for w in caught if issubclass(w.category, JaxgsaWarning)]


def test_pce_warnings_name_the_shapley_entry_point_and_point_at_the_caller():
    # An order reduction, a poor fit and an overfit each warn from inside the
    # PCE. Called through shapley.analyze they must name that function and point
    # at the caller's own line, not at a frame inside the package.
    rng = np.random.default_rng(0)
    problem = jaxgsa.Problem(["a", "b", "c"], [(0, 1)] * 3)
    X = rng.uniform(0, 1, (30, 3)).astype(np.float32)
    Y = np.sin(12 * X[:, 0]) + rng.standard_normal(30).astype(np.float32)
    Xj, Yj = jnp.asarray(X), jnp.asarray(Y)

    def through_shapley():
        jaxgsa.shapley.analyze(problem, Xj, Yj, backend="pce", order=6, verbose=False)

    def direct():
        pce.analyze(problem, Xj, Yj, order=6, verbose=False)

    shap = _warn_sources(through_shapley)
    assert shap, "expected at least the order-reduced warning"
    for message, filename in shap:
        assert message.startswith("jaxgsa.shapley"), message
        assert filename == __file__
    for message, filename in _warn_sources(direct):
        assert message.startswith("jaxgsa.pce"), message
        assert filename == __file__


@pytest.mark.parametrize("ridge", [1e-8, 1.0])
@pytest.mark.parametrize(
    ("make", "parents"),
    [
        (lambda a, b: 1.0 - a, "['a']"),
        (lambda a, b: 2.0 * a, "['a']"),
        (lambda a, b: a + b, "['a', 'b']"),
        (lambda a, b: 0.3 * a + 0.7 * b + 5.0, "['a', 'b']"),
        (lambda a, b: 1000.0 + a, "['a']"),
    ],
)
def test_affine_copy_of_other_inputs_raises_in_both_dtypes(make, parents, ridge):
    # 1.1.2 let these through in float64 with one warning, and the indices of
    # the copy were wrong. The check reads the inputs, so the dtype and the
    # ridge cannot hide it.
    problem, X, _ = _three_inputs()
    X[:, 2] = make(X[:, 0].astype(np.float64), X[:, 1].astype(np.float64)).astype(X.dtype)
    Y = 2 * X[:, 0] + X[:, 1] ** 2
    with pytest.raises(ValueError, match=rf"c = linear combination of {re.escape(parents)}"):
        pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=3, ridge=ridge, verbose=False)


def test_affine_copy_message_names_the_shapley_entry_point():
    problem, X, _ = _three_inputs()
    X[:, 2] = 1.0 - X[:, 0]
    Y = 2 * X[:, 0] + X[:, 1] ** 2
    with pytest.raises(ValueError, match=r"^jaxgsa\.shapley\.analyze: .*linearly dependent"):
        jaxgsa.shapley.analyze(problem, jnp.asarray(X), jnp.asarray(Y), backend="pce")


def test_noisy_copy_above_rounding_error_is_not_called_linearly_dependent():
    # 1e-3 noise is far above the rounding error of either dtype. It is the
    # pivot share's case (test_near_duplicated_input_is_flagged_in_both_dtypes).
    problem, X, rng = _three_inputs()
    X[:, 2] = np.clip(X[:, 0] + 1e-3 * rng.standard_normal(2000), 0, 1).astype(np.float32)
    Y = 2 * X[:, 0] + X[:, 1] ** 2
    try:
        pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=3, verbose=False)
    except ValueError as error:
        assert "linearly dependent" not in str(error)


def test_independent_input_with_large_offset_is_not_called_dependent():
    # 1e5 + U(0, 1) is stored with a spacing of 0.008 in float32, yet it is
    # independent of the others and must fit.
    problem = jaxgsa.Problem(["a", "b", "c"], [(0, 1), (0, 1), (1e5, 1e5 + 1)])
    rng = np.random.default_rng(3)
    X = np.c_[rng.uniform(0, 1, (2000, 2)), 1e5 + rng.uniform(0, 1, 2000)]
    X = X.astype(np.float64 if jax.config.jax_enable_x64 else np.float32)
    Y = X[:, 0] + X[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=1, verbose=False)
    assert isinstance(result, pce.PCEResult)


def _gaussian_three(n: int = 6000):
    gauss = GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0)
    problem = jaxgsa.Problem.from_dict({"a": gauss, "b": gauss, "c": gauss})
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, 3)).astype(np.float32)
    Y = X[:, 0] + 0.5 * X[:, 1] + 0.2 * X[:, 0] * X[:, 2]
    return problem, X, Y


@pytest.mark.parametrize("order", [12, 14])
def test_float32_high_order_hermite_fit_is_exact(order):
    # 1.1.2 solved the normal equations. Float32 lost every digit of the
    # Gram matrix at order 12: the factorisation failed and the fit raised.
    # QR keeps the digits, so the indices are exact to float32 precision.
    problem, X, Y = _gaussian_three()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pce.analyze(
            problem, jnp.asarray(X), jnp.asarray(Y), order=order, fit_ratio=1.0, verbose=False
        )
    assert isinstance(result, pce.PCEResult)
    np.testing.assert_allclose(np.asarray(result.ST), [0.8062, 0.1938, 0.0310], atol=2e-4)


def test_float32_ill_conditioned_fit_warns_about_precision():
    # Order 16 has a share near 1e-8: the indices are only good to about 3e-3
    # in float32, and the fit says so.
    if jax.config.jax_enable_x64:
        pytest.skip("float32 only")
    problem, X, Y = _gaussian_three()
    with pytest.warns(JaxgsaWarning, match="float32 carries about 7 digits"):
        pce.analyze(
            problem, jnp.asarray(X), jnp.asarray(Y), order=16, fit_ratio=1.0, verbose=False
        )


def test_ridge_qr_solves_the_normal_equations():
    from jaxgsa._core.fit_quality import min_pivot_share
    from jaxgsa.pce._engine import hat_diagonal, r_pivot_share, ridge_qr

    rng = np.random.default_rng(0)
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    Phi = jnp.asarray(rng.standard_normal((200, 8)).astype(dtype))
    Y = jnp.asarray(rng.standard_normal((200, 3)).astype(dtype))
    ridge = 0.05
    coefficients, R = ridge_qr(Phi, Y, ridge)
    gram = Phi.T @ Phi + ridge * jnp.eye(8, dtype=Phi.dtype)
    tol = 1e-10 if jax.config.jax_enable_x64 else 5e-4
    np.testing.assert_allclose(
        np.asarray(coefficients), np.asarray(jnp.linalg.solve(gram, Phi.T @ Y)), atol=tol
    )
    np.testing.assert_allclose(np.asarray(R.T @ R), np.asarray(gram), atol=tol * 100)
    np.testing.assert_allclose(
        float(r_pivot_share(R)), float(min_pivot_share(gram)), rtol=1e-3 if tol < 1e-3 else 5e-2
    )
    chol = jnp.linalg.cholesky(gram)
    np.testing.assert_allclose(
        np.asarray(hat_diagonal(Phi, R.T)), np.asarray(hat_diagonal(Phi, chol)), atol=tol
    )


def test_bootstrap_counts_replicates_with_a_near_singular_design():
    # A two-level input makes every fit near singular, so every replicate is
    # counted, and the count is in the warning.
    problem, X, rng = _three_inputs()
    X[:, 2] = rng.choice(np.array([0.2, 0.8], np.float32), 2000)
    Y = 2 * X[:, 0] + X[:, 1] ** 2
    with pytest.warns(JaxgsaWarning, match=r"3 of 3 bootstrap replicates"):
        pce.analyze(
            problem,
            jnp.asarray(X),
            jnp.asarray(Y),
            order=3,
            n_bootstrap=3,
            key=jax.random.key(0),
            verbose=False,
        )


def _truncnorm_response(low, high, order, *, n=4000, power=3):
    """Fit Y = x**power + 3u, x truncated Gaussian, and return S1[0] with its truth."""
    from scipy import stats

    mean, std = 0.0, 1.0
    a = -np.inf if low is None else (low - mean) / std
    b = np.inf if high is None else (high - mean) / std
    dist = stats.truncnorm(a, b, loc=mean, scale=std)
    spec = GaussianInputSpec(dist="gaussian", mean=mean, variance=std**2)
    if low is not None:
        spec["low"] = low
    if high is not None:
        spec["high"] = high
    problem = jaxgsa.Problem.from_dict({"a": spec, "b": (0.0, 1.0)})
    rng = np.random.default_rng(0)
    x = dist.rvs(n, random_state=1)
    u = rng.uniform(0, 1, n)
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    X = np.stack([x, u], 1).astype(dtype)
    Y = (x**power + 3 * u).astype(dtype)
    grid = np.linspace(dist.ppf(1e-13), dist.ppf(1 - 1e-13), 400001)
    pdf = dist.pdf(grid)
    pdf /= np.trapezoid(pdf, grid)
    mean_p = np.trapezoid(grid**power * pdf, grid)
    var_p = np.trapezoid((grid**power - mean_p) ** 2 * pdf, grid)
    truth = var_p / (var_p + 0.75)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=order, verbose=False)
    assert isinstance(result, pce.PCEResult)
    return float(result.S1[0]), truth, float(result.cv_rmse)


@pytest.mark.parametrize(
    ("low", "high"),
    [(-8.0, 8.0), (-3.0, 3.0), (-1.0, 1.0), (-1.0, 2.0), (0.0, None), (3.0, None), (None, -3.0)],
    ids=["+-8", "+-3", "[-1,1]", "[-1,2]", "[0,inf)", "z>=3", "z<=-3"],
)
def test_truncated_gaussian_polynomial_response_is_exact(low, high):
    # A cubic response lies in the span of an order-3 basis that is orthonormal
    # against the truncated measure, so S1 is exact up to float rounding. The
    # old Legendre-through-the-CDF route was off by 4e-3 to 5e-2.
    s1, truth, cv = _truncnorm_response(low, high, order=4)
    tol = 1e-8 if jax.config.jax_enable_x64 else 2e-5
    assert s1 == pytest.approx(truth, abs=tol)
    assert cv < (1e-6 if jax.config.jax_enable_x64 else 1e-2)


def test_very_narrow_truncation_is_finite_and_exact():
    # A truncation 0.1 standard deviations wide, which float32 cannot hold in
    # a raw polynomial basis.
    s1, truth, _ = _truncnorm_response(-0.05, 0.05, order=6)
    assert np.isfinite(s1)
    assert s1 == pytest.approx(truth, abs=1e-6)


@pytest.mark.parametrize(
    ("a", "b"),
    [(-3.0, 3.0), (-1.0, 2.0), (0.0, np.inf), (3.0, np.inf), (-np.inf, -5.0), (-0.05, 0.05)],
)
def test_truncated_family_is_orthonormal_against_its_measure(a, b):
    from scipy import stats

    from jaxgsa.pce._engine import _truncated_1d, truncated_gaussian_family

    order = 8
    mean, std, family = truncated_gaussian_family(a, b, order)
    dist = stats.truncnorm(a, b)
    assert mean == pytest.approx(dist.mean(), abs=1e-9)
    assert std == pytest.approx(dist.std(), abs=1e-9)
    # Degree-16 integrands peak far out in a tail, so the grid reaches 20 sigma.
    grid = np.linspace(max(a, -20.0), min(b, 20.0), 400001)
    pdf = dist.pdf(grid)
    pdf /= np.trapezoid(pdf, grid)
    with jax.enable_x64():
        basis = np.asarray(
            _truncated_1d(
                jnp.asarray((grid - mean) / std, dtype=jnp.float64),
                np.asarray(family.alpha),
                np.asarray(family.root_beta),
                order,
            )
        )
    gram = np.trapezoid(basis[:, :, None] * basis[:, None, :] * pdf[:, None, None], grid, axis=0)
    np.testing.assert_allclose(gram, np.eye(order + 1), atol=2e-6)


def test_wide_truncation_family_reproduces_hermite():
    from jaxgsa.pce._engine import truncated_gaussian_family

    _, std, family = truncated_gaussian_family(-12.0, 12.0, 10)
    assert std == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(family.alpha, 0.0, atol=1e-12)
    np.testing.assert_allclose(family.root_beta[1:], np.sqrt(np.arange(1, 11)), atol=1e-10)


def test_truncated_basis_survives_predict_and_grouped_mapping():
    # D above the grouped-map cut-over, mixing a truncated input with the rest:
    # the grouped map and the column loop must agree, and predict must run.
    d = 20
    problem = _mixed_problem(d)
    X = _inputs(d) * 4 - 1
    for order in (3, 8):
        col, col_types = _map_by_column(X, problem, order)
        grp, grp_types = _map_by_group(X, problem, order)
        assert col_types == grp_types
        np.testing.assert_allclose(np.asarray(col), np.asarray(grp), rtol=1e-6, atol=1e-7)
    problem = jaxgsa.Problem.from_dict(
        {
            "a": GaussianInputSpec(dist="gaussian", mean=1.0, variance=4.0, low=0.0, high=5.0),
            "b": (0.0, 1.0),
        }
    )
    rng = np.random.default_rng(0)
    X = np.stack([rng.uniform(0, 5, 500), rng.uniform(0, 1, 500)], 1).astype(np.float32)
    Y = X[:, 0] ** 2 + X[:, 1]
    result = pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=3, verbose=False)
    assert isinstance(result, pce.PCEResult)
    pred = np.asarray(result.predict(jnp.asarray(X)))
    np.testing.assert_allclose(pred, Y, atol=1e-2, rtol=1e-3)
