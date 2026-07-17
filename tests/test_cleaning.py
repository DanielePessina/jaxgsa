"""Tests for data cleaning, division guards, and NaN reporting."""

import jax.numpy as jnp
import pytest

import gsax
from gsax.problem import Problem
from gsax.sobol._indices import first_order, second_order, total_order


@pytest.fixture()
def simple_problem():
    return Problem(
        names=("x1", "x2", "x3"),
        bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
    )


# --- Division guard: zero-variance inputs produce NaN, not inf ---


def test_first_order_zero_variance():
    A = jnp.ones(10)
    AB_j = jnp.ones(10)
    B = jnp.ones(10)
    result = first_order(A, AB_j, B)
    assert jnp.isnan(result), f"Expected NaN, got {result}"


def test_total_order_zero_variance():
    A = jnp.ones(10)
    AB_j = jnp.ones(10)
    B = jnp.ones(10)
    result = total_order(A, AB_j, B)
    assert jnp.isnan(result), f"Expected NaN, got {result}"


def test_second_order_zero_variance():
    A = jnp.ones(10)
    AB_j = jnp.ones(10)
    AB_k = jnp.ones(10)
    BA_j = jnp.ones(10)
    B = jnp.ones(10)
    result = second_order(A, AB_j, AB_k, BA_j, B)
    assert jnp.isnan(result), f"Expected NaN, got {result}"


# --- End-to-end: constant Y → NaN indices, nan_counts populated ---


def test_constant_y_produces_nan_counts(simple_problem):
    sr = gsax.sobol.sample(
        simple_problem, n_samples=64, seed=0, calc_second_order=False, verbose=False
    )
    Y = jnp.ones(sr.n_total)
    with pytest.warns(UserWarning, match="zero variance"):
        result = gsax.sobol.analyze(sr, Y)
    assert result.nan_counts is not None
    assert result.nan_counts["S1"] > 0
    assert result.nan_counts["ST"] > 0
    assert jnp.all(jnp.isnan(result.S1))
    assert jnp.all(jnp.isnan(result.ST))


def test_constant_y_second_order_nan(simple_problem):
    sr = gsax.sobol.sample(
        simple_problem, n_samples=64, seed=0, calc_second_order=True, verbose=False
    )
    Y = jnp.ones(sr.n_total)
    with pytest.warns(UserWarning, match="zero variance"):
        result = gsax.sobol.analyze(sr, Y)
    assert result.nan_counts is not None
    assert result.nan_counts["S2"] > 0
    assert jnp.all(jnp.isnan(result.S2))


# --- Input cleaning: non-finite values ---


def test_drop_nonfinite_rows(simple_problem):
    sr = gsax.sobol.sample(
        simple_problem, n_samples=64, seed=0, calc_second_order=False, verbose=False
    )
    Y = jnp.sin(jnp.sum(jnp.asarray(sr.samples), axis=1))
    # Inject NaN into the first group
    Y_bad = Y.at[0].set(jnp.nan)
    with pytest.warns(UserWarning, match="dropped"):
        result = gsax.sobol.analyze(sr, Y_bad)
    # Should still produce finite results from remaining groups
    assert result.S1.shape == (3,)


def test_all_nonfinite_raises(simple_problem):
    sr = gsax.sobol.sample(
        simple_problem, n_samples=64, seed=0, calc_second_order=False, verbose=False
    )
    Y = jnp.full(sr.n_total, jnp.nan)
    with pytest.warns(UserWarning, match="dropped"):
        with pytest.raises(ValueError, match="All samples contain non-finite values"):
            gsax.sobol.analyze(sr, Y)


def test_inf_values_dropped(simple_problem):
    sr = gsax.sobol.sample(
        simple_problem, n_samples=64, seed=0, calc_second_order=False, verbose=False
    )
    Y = jnp.sin(jnp.sum(jnp.asarray(sr.samples), axis=1))
    Y_bad = Y.at[0].set(jnp.inf)
    with pytest.warns(UserWarning, match="dropped"):
        result = gsax.sobol.analyze(sr, Y_bad)
    assert result.S1.shape == (3,)
