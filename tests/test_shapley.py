"""Tests for the Shapley-effect sensitivity analysis module."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import gsax
from gsax.benchmarks import ishigami, linear, sobol_g
from gsax.shapley._engine import build_membership, shapley_from_variances

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ishigami_data():
    """Monte Carlo training data for the Ishigami benchmark."""
    X = jnp.asarray(gsax.sample_mc(ishigami.PROBLEM, 8192, seed=42))
    Y = ishigami.evaluate(X)
    return X, Y


@pytest.fixture(scope="module")
def ishigami_hdmr_shapley(ishigami_data):
    X, Y = ishigami_data
    return gsax.analyze_shapley(ishigami.PROBLEM, X, Y, backend="hdmr")


@pytest.fixture(scope="module")
def ishigami_pce_shapley(ishigami_data):
    X, Y = ishigami_data
    # Ishigami needs a high polynomial degree for the sin^2 term.
    return gsax.analyze_shapley(ishigami.PROBLEM, X, Y, backend="pce", order=9)


@pytest.fixture(scope="module")
def linear_data():
    """Monte Carlo training data for the linear benchmark."""
    X = jnp.asarray(gsax.sample_mc(linear.PROBLEM, 2048, seed=7))
    Y = linear.evaluate(X)
    return X, Y


# ---------------------------------------------------------------------------
# Engine: exact hand-computed values
# ---------------------------------------------------------------------------


def test_engine_exact_hand_computed():
    # Terms x0, x1, {x0,x1}, {x0,x2} with variance fractions 0.3/0.2/0.1/0.4.
    subsets = [(0,), (1,), (0, 1), (0, 2)]
    membership = build_membership(subsets, D=3)
    V = jnp.array([0.3, 0.2, 0.1, 0.4])

    Sh, S1, ST = shapley_from_variances(V, membership)

    np.testing.assert_allclose(np.asarray(Sh), [0.55, 0.25, 0.2], atol=1e-7)
    np.testing.assert_allclose(np.asarray(S1), [0.3, 0.2, 0.0], atol=1e-7)
    np.testing.assert_allclose(np.asarray(ST), [0.8, 0.3, 0.4], atol=1e-7)
    # Efficiency: the Shapley effects redistribute exactly the modelled variance.
    np.testing.assert_allclose(np.asarray(Sh).sum(), np.asarray(V).sum(), atol=1e-7)


def test_engine_batched_leading_dims():
    subsets = [(0,), (1,), (0, 1)]
    membership = build_membership(subsets, D=2)
    V = jnp.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.4]])

    Sh, S1, ST = shapley_from_variances(V, membership)

    assert Sh.shape == S1.shape == ST.shape == (2, 2)
    np.testing.assert_allclose(np.asarray(Sh), [[0.6, 0.4], [0.3, 0.4]], atol=1e-7)


def test_build_membership():
    membership = build_membership([(1,), (0, 2)], D=3)
    expected = np.array([[False, True, False], [True, False, True]])
    np.testing.assert_array_equal(membership, expected)


# ---------------------------------------------------------------------------
# Correctness vs analytical benchmark values
# ---------------------------------------------------------------------------


def test_hdmr_ishigami_vs_analytical(ishigami_hdmr_shapley):
    Sh = np.asarray(ishigami_hdmr_shapley.Sh)
    np.testing.assert_allclose(Sh, ishigami.ANALYTICAL_SHAPLEY, atol=0.12)
    # Normalized by empirical Var(Y): the sum is the explained fraction (<= ~1).
    assert 0.8 < Sh.sum() <= 1.02


def test_pce_ishigami_vs_analytical(ishigami_pce_shapley):
    Sh = np.asarray(ishigami_pce_shapley.Sh)
    np.testing.assert_allclose(Sh, ishigami.ANALYTICAL_SHAPLEY, atol=0.01)
    assert abs(Sh.sum() - 1.0) < 0.02


@pytest.mark.parametrize("backend,kwargs", [("hdmr", {}), ("pce", {"order": 2})])
def test_linear_vs_analytical(linear_data, backend, kwargs):
    X, Y = linear_data
    result = gsax.analyze_shapley(linear.PROBLEM, X, Y, backend=backend, **kwargs)
    Sh = np.asarray(result.Sh)
    np.testing.assert_allclose(Sh, linear.ANALYTICAL_SHAPLEY, atol=0.05)
    # Additive model: no interactions, so Shapley collapses to first-order.
    np.testing.assert_allclose(Sh, np.asarray(result.S1), atol=0.01)


def test_hdmr_sobol_g_vs_analytical():
    X = jnp.asarray(gsax.sample_mc(sobol_g.PROBLEM, 8192, seed=456))
    Y = sobol_g.evaluate(X)
    result = gsax.analyze_shapley(sobol_g.PROBLEM, X, Y)
    np.testing.assert_allclose(np.asarray(result.Sh), sobol_g.ANALYTICAL_SHAPLEY, atol=0.1)


@pytest.mark.parametrize("fixture", ["ishigami_hdmr_shapley", "ishigami_pce_shapley"])
def test_s1_le_sh_le_st(fixture, request):
    result = request.getfixturevalue(fixture)
    S1, Sh, ST = np.asarray(result.S1), np.asarray(result.Sh), np.asarray(result.ST)
    assert np.all(S1 <= Sh + 1e-6)
    assert np.all(Sh <= ST + 1e-6)


# ---------------------------------------------------------------------------
# Kwarg validation
# ---------------------------------------------------------------------------


def test_pce_kwarg_with_hdmr_backend_raises(linear_data):
    X, Y = linear_data
    with pytest.raises(ValueError, match="only apply to backend='pce'"):
        gsax.analyze_shapley(linear.PROBLEM, X, Y, backend="hdmr", order=5)


def test_hdmr_kwarg_with_pce_backend_raises(linear_data):
    X, Y = linear_data
    with pytest.raises(ValueError, match="only apply to backend='hdmr'"):
        gsax.analyze_shapley(linear.PROBLEM, X, Y, backend="pce", maxorder=3)


def test_unknown_backend_raises(linear_data):
    X, Y = linear_data
    with pytest.raises(ValueError, match="backend must be"):
        gsax.analyze_shapley(linear.PROBLEM, X, Y, backend="sobol")  # ty: ignore[invalid-argument-type]


def test_pce_multi_output_raises(linear_data):
    X, Y = linear_data
    Y2 = jnp.stack([Y, 2.0 * Y], axis=1)
    with pytest.raises(ValueError, match="scalar output only"):
        gsax.analyze_shapley(linear.PROBLEM, X, Y2, backend="pce")


# ---------------------------------------------------------------------------
# Output shapes and scale invariance
# ---------------------------------------------------------------------------


def test_multi_output_shapes(linear_data):
    X, Y = linear_data
    # Affine copies: indices are scale/shift invariant, so rows must match.
    Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=1)
    result = gsax.analyze_shapley(linear.PROBLEM, X, Y2)
    assert result.Sh.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(result.Sh[0]), np.asarray(result.Sh[1]), atol=1e-4)


def test_time_series_shapes(linear_data):
    X, Y = linear_data
    Y3 = jnp.stack([jnp.stack([Y, 2.0 * Y], axis=1)] * 2, axis=1)  # (N, T=2, K=2)
    result = gsax.analyze_shapley(linear.PROBLEM, X, Y3)
    assert result.Sh.shape == (2, 2, 3)
    assert result.S1.shape == (2, 2, 3)
    assert result.ST.shape == (2, 2, 3)


# ---------------------------------------------------------------------------
# xarray export
# ---------------------------------------------------------------------------


def test_to_dataset_scalar(ishigami_pce_shapley):
    ds = ishigami_pce_shapley.to_dataset()
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"Sh", "S1", "ST"}
    assert ds["Sh"].dims == ("param",)
    assert list(ds.coords["param"].values) == list(ishigami.PROBLEM.names)


def test_to_dataset_time_series(linear_data):
    X, Y = linear_data
    Y3 = jnp.stack([jnp.stack([Y, 2.0 * Y], axis=1)] * 2, axis=1)
    result = gsax.analyze_shapley(linear.PROBLEM, X, Y3)
    ds = result.to_dataset(time_coords=[0.5, 1.5])
    assert ds["Sh"].dims == ("time", "output", "param")
    assert list(ds.coords["time"].values) == [0.5, 1.5]


def test_repr(ishigami_pce_shapley):
    text = repr(ishigami_pce_shapley)
    assert "ShapleyResult" in text
    assert "pce" in text
