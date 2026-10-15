"""Regression tests for the 1.1.2 PCE fixes."""

from __future__ import annotations

import warnings

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


def test_duplicated_input_raises_on_singular_gram():
    rng = np.random.default_rng(0)
    problem = jaxgsa.Problem(["a", "b", "c"], [(0, 1)] * 3)
    X = rng.uniform(0, 1, (2000, 3)).astype(np.float32)
    X[:, 2] = X[:, 1]
    Y = 2 * X[:, 0] + X[:, 1] ** 2
    with pytest.raises(ValueError, match="numerically singular"):
        pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=3, verbose=False)


def test_constant_input_warns_when_loo_fails():
    rng = np.random.default_rng(0)
    problem = jaxgsa.Problem(["a", "b", "c"], [(0, 1)] * 3)
    X = rng.uniform(0, 1, (2000, 3)).astype(np.float32)
    X[:, 2] = 0.5
    Y = 2 * X[:, 0] + X[:, 1] ** 2
    with pytest.warns(JaxgsaWarning, match="numerically singular"):
        pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=3, verbose=False)


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


def test_high_order_wide_truncation_warns_and_names_input():
    rng = np.random.default_rng(0)
    X = np.stack([rng.standard_normal(3000), rng.uniform(0, 1, 3000)], 1).astype(np.float32)
    Y = X[:, 0] ** 2 + 3 * X[:, 1]
    problem = _wide_truncated_problem()
    with pytest.warns(JaxgsaWarning, match=r"truncated Gaussian input\(s\) \['a'\]"):
        pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=8, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=7, verbose=False)


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
