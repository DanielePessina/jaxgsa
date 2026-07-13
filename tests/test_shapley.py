"""Tests for the Shapley-effect sensitivity analysis module."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import gsax
from gsax.benchmarks import ishigami, linear, sobol_g
from gsax.problem import GaussianInputSpec, Problem
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
    # Sh is normalized to sum to 1; the fit quality lives in explained_variance.
    assert abs(Sh.sum() - 1.0) < 1e-5
    assert 0.8 < float(ishigami_hdmr_shapley.explained_variance) <= 1.02


def test_pce_ishigami_vs_analytical(ishigami_pce_shapley):
    Sh = np.asarray(ishigami_pce_shapley.Sh)
    np.testing.assert_allclose(Sh, ishigami.ANALYTICAL_SHAPLEY, atol=0.01)
    assert abs(Sh.sum() - 1.0) < 1e-5


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
    result = gsax.analyze_shapley(sobol_g.PROBLEM, X, Y, backend="hdmr")
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


def test_pce_multi_output_matches_scalar(linear_data):
    """Each column of a multi-output pce run equals the standalone scalar run."""
    X, Y = linear_data
    Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=1)
    multi = gsax.analyze_shapley(linear.PROBLEM, X, Y2, backend="pce", order=2)
    scalar = gsax.analyze_shapley(linear.PROBLEM, X, Y, backend="pce", order=2)
    assert multi.Sh.shape == (2, 3)
    assert multi.explained_variance.shape == (2,)
    for k in range(2):
        np.testing.assert_allclose(multi.Sh[k], scalar.Sh, rtol=1e-5)
        np.testing.assert_allclose(multi.S1[k], scalar.S1, rtol=1e-5)
        np.testing.assert_allclose(multi.ST[k], scalar.ST, rtol=1e-5)
    np.testing.assert_allclose(multi.Sh.sum(axis=-1), 1.0, rtol=1e-6)


def test_default_backend_is_pce(linear_data):
    X, Y = linear_data
    result = gsax.analyze_shapley(linear.PROBLEM, X, Y, order=2)
    assert result.backend == "pce"


# ---------------------------------------------------------------------------
# Output shapes and scale invariance
# ---------------------------------------------------------------------------


def test_multi_output_shapes(linear_data):
    X, Y = linear_data
    # Affine copies: indices are scale/shift invariant, so rows must match.
    Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=1)
    result = gsax.analyze_shapley(linear.PROBLEM, X, Y2, backend="hdmr")
    assert result.Sh.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(result.Sh[0]), np.asarray(result.Sh[1]), atol=1e-4)


def test_time_series_shapes(linear_data):
    X, Y = linear_data
    Y3 = jnp.stack([jnp.stack([Y, 2.0 * Y], axis=1)] * 2, axis=1)  # (N, T=2, K=2)
    result = gsax.analyze_shapley(linear.PROBLEM, X, Y3, backend="hdmr")
    assert result.Sh.shape == (2, 2, 3)
    assert result.S1.shape == (2, 2, 3)
    assert result.ST.shape == (2, 2, 3)


# ---------------------------------------------------------------------------
# xarray export
# ---------------------------------------------------------------------------


def test_to_dataset_scalar(ishigami_pce_shapley):
    ds = ishigami_pce_shapley.to_dataset()
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"Sh", "S1", "ST", "explained_variance"}
    assert ds["Sh"].dims == ("param",)
    assert ds["explained_variance"].dims == ()
    assert list(ds.coords["param"].values) == list(ishigami.PROBLEM.names)


def test_to_dataset_time_series(linear_data):
    X, Y = linear_data
    Y3 = jnp.stack([jnp.stack([Y, 2.0 * Y], axis=1)] * 2, axis=1)
    result = gsax.analyze_shapley(linear.PROBLEM, X, Y3, backend="hdmr")
    ds = result.to_dataset(time_coords=[0.5, 1.5])
    assert ds["Sh"].dims == ("time", "output", "param")
    assert list(ds.coords["time"].values) == [0.5, 1.5]


def test_repr(ishigami_pce_shapley):
    text = repr(ishigami_pce_shapley)
    assert "ShapleyResult" in text
    assert "pce" in text


# ---------------------------------------------------------------------------
# Normalization contract and fit diagnostics (regression tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["ishigami_hdmr_shapley", "ishigami_pce_shapley"])
def test_sh_sums_to_one(fixture, request):
    """Efficiency property: Sh sums to exactly 1 regardless of backend/fit."""
    result = request.getfixturevalue(fixture)
    np.testing.assert_allclose(float(np.asarray(result.Sh).sum()), 1.0, atol=1e-5)


def test_pce_overfit_sums_to_one_and_flags(ishigami_data):
    """An overfit PCE keeps Sh summing to 1; the overshoot shows in ev + warns."""
    X, Y = ishigami_data
    Xs, Ys = X[:64], Y[:64]
    with pytest.warns(UserWarning, match="explained_variance exceeds"):
        result = gsax.analyze_shapley(
            ishigami.PROBLEM, Xs, Ys, backend="pce", order=5, fit_ratio=0.9
        )
    np.testing.assert_allclose(float(np.asarray(result.Sh).sum()), 1.0, atol=1e-4)
    # The over-counted variance surfaces in the diagnostic, not the sum.
    assert float(result.explained_variance) > 1.3


def test_pce_shapley_matches_analyze_pce(ishigami_data):
    """The PCE backend's S1/ST coincide with analyze_pce on the identical fit."""
    X, Y = ishigami_data
    sh = gsax.analyze_shapley(ishigami.PROBLEM, X, Y, backend="pce", order=9)
    pce = gsax.analyze_pce(ishigami.PROBLEM, X, Y, order=9)
    np.testing.assert_allclose(np.asarray(sh.S1), np.asarray(pce.S1), atol=1e-5)
    np.testing.assert_allclose(np.asarray(sh.ST), np.asarray(pce.ST), atol=1e-5)


def test_pce_order_reduction_surfaced():
    """A silently-reduced PCE order is both warned about and exposed on the result."""
    X = jnp.asarray(gsax.sample_mc(ishigami.PROBLEM, 200, seed=11))
    Y = ishigami.evaluate(X)
    # For N=200 (fit_ratio=0.5) the order-7 term count exceeds the budget,
    # so the fit is silently coarsened -- which must now be visible.
    with pytest.warns(UserWarning, match="order reduced"):
        result = gsax.analyze_shapley(ishigami.PROBLEM, X, Y, backend="pce", order=7)
    assert result.order < 7


# ---------------------------------------------------------------------------
# Correlated inputs: optional ANCOVA (Sb) fold-in
# ---------------------------------------------------------------------------


def _correlated_gaussian(rho, coeffs=(1.0, 2.0), sigmas=(1.0, 1.0), *, n=16384, seed=0):
    """Correlated Gaussian design + additive linear output.

    Correlation enters purely through the samples (the empirical-from-X
    contract): X = Z @ chol(Sigma).T with the declared marginals matching the
    per-column variances, and Y = X @ coeffs.
    """
    d = len(coeffs)
    problem = Problem.from_dict(
        {
            f"x{i}": GaussianInputSpec(dist="gaussian", mean=0.0, variance=sigmas[i] ** 2)
            for i in range(d)
        }
    )
    corr = np.eye(d)
    corr[0, 1] = corr[1, 0] = rho
    cov = np.outer(sigmas, sigmas) * corr
    L = np.linalg.cholesky(cov)
    X = np.random.default_rng(seed).standard_normal((n, d)) @ L.T
    Y = X @ np.asarray(coeffs, dtype=float)
    return problem, jnp.asarray(X), jnp.asarray(Y)


def test_correlative_efficiency():
    """Folding Sb in preserves the Shapley efficiency property (Sh sums to 1)."""
    problem, X, Y = _correlated_gaussian(0.6)
    res = gsax.analyze_shapley(problem, X, Y, backend="hdmr", include_correlative=True, m=5)
    np.testing.assert_allclose(float(np.asarray(res.Sh).sum()), 1.0, atol=1e-4)


def test_correlative_matches_ancova_closed_form():
    """Additive linear-Gaussian: folded Sh matches the derived ANCOVA closed form.

    For Y = c1 X1 + c2 X2 with correlation rho, a = c1 sigma1, b = c2 sigma2,
    the Sa+Sb fold-in gives Sh1 = a(a+rho b)/Var(Y), Sh2 = b(b+rho a)/Var(Y)
    (Var(Y) = a^2 + b^2 + 2 rho a b). This differs from the conditional-variance
    Shapley by rho^2 (b^2 - a^2) / (2 Var(Y)) -- so it pins the ANCOVA notion,
    not that estimator.
    """
    rho, (c1, c2), (s1, s2) = 0.5, (1.0, 2.0), (1.0, 1.0)
    problem, X, Y = _correlated_gaussian(rho, (c1, c2), (s1, s2))
    a, b = c1 * s1, c2 * s2
    var_y = a**2 + b**2 + 2 * rho * a * b
    expected = np.array([a * (a + rho * b), b * (b + rho * a)]) / var_y  # [2/7, 5/7]

    res = gsax.analyze_shapley(problem, X, Y, backend="hdmr", include_correlative=True, m=5)
    np.testing.assert_allclose(np.asarray(res.Sh), expected, atol=0.01)


def test_correlative_matches_hdmr_sa_plus_sb():
    """White-box: the fold-in reproduces the Shapley allocation of analyze_hdmr's
    Sa+Sb exactly, independent of what the component functions are."""
    problem, X, Y = _correlated_gaussian(0.5)
    hd = gsax.analyze_hdmr(problem, X, Y, m=5)
    subsets = [(i,) for i in range(problem.num_vars)]
    subsets += [tuple(u) for u in hd.emulator["c2"]]
    subsets += [tuple(u) for u in hd.emulator["c3"]]
    membership = build_membership(subsets, problem.num_vars)
    partial = np.asarray(hd.Sa) + np.asarray(hd.Sb)
    V = partial / partial.sum()
    sh_expected, _, _ = shapley_from_variances(jnp.asarray(V), membership)

    res = gsax.analyze_shapley(problem, X, Y, backend="hdmr", include_correlative=True, m=5)
    np.testing.assert_allclose(np.asarray(res.Sh), np.asarray(sh_expected), atol=1e-5)


def test_correlative_reduces_to_structural_under_independence():
    """With independent X (rho=0) folding Sb in is a no-op (Sb ~ 0), so the
    result matches both the structural run and the analytic Shapley."""
    problem, X, Y = _correlated_gaussian(0.0)
    folded = gsax.analyze_shapley(problem, X, Y, backend="hdmr", include_correlative=True, m=5)
    structural = gsax.analyze_shapley(
        problem, X, Y, backend="hdmr", include_correlative=False, m=5
    )
    np.testing.assert_allclose(np.asarray(folded.Sh), np.asarray(structural.Sh), atol=5e-3)
    # Analytic additive Shapley = S1 = c_j^2 var_j / Var(Y) = [1/5, 4/5].
    np.testing.assert_allclose(np.asarray(folded.Sh), [0.2, 0.8], atol=0.01)


def test_correlative_index_can_be_negative():
    """Strong anti-correlation drives a correlated index negative; Sh still
    sums to 1. Documents the signed-index behaviour."""
    problem, X, Y = _correlated_gaussian(-0.8)
    res = gsax.analyze_shapley(problem, X, Y, backend="hdmr", include_correlative=True, m=5)
    Sh = np.asarray(res.Sh)
    assert np.any(Sh < 0.0)
    np.testing.assert_allclose(float(Sh.sum()), 1.0, atol=1e-4)


def test_correlative_explained_variance_is_r2():
    """Folded explained_variance equals the true emulator R^2 = Var(Y_hat)/Var(Y)
    = sum_j S_j, not the structural sum sum_j Sa_j."""
    problem, X, Y = _correlated_gaussian(0.5)
    res = gsax.analyze_shapley(problem, X, Y, backend="hdmr", include_correlative=True, m=5)
    hd = gsax.analyze_hdmr(problem, X, Y, m=5)
    np.testing.assert_allclose(
        float(res.explained_variance), float(np.asarray(hd.S).sum()), atol=1e-5
    )
    assert 0.0 < float(res.explained_variance) <= 1.05


@pytest.mark.parametrize("backend", ["pce", None])
def test_correlative_requires_hdmr(linear_data, backend):
    """include_correlative is HDMR-only; the PCE backend (explicit or default)
    raises rather than silently ignoring it."""
    X, Y = linear_data
    kwargs = {} if backend is None else {"backend": backend}
    with pytest.raises(ValueError, match="hdmr"):
        gsax.analyze_shapley(linear.PROBLEM, X, Y, include_correlative=True, **kwargs)


def test_correlative_provenance_on_result_and_dataset():
    """The result records the flag and surfaces it as a dataset attr."""
    problem, X, Y = _correlated_gaussian(0.5)
    res = gsax.analyze_shapley(problem, X, Y, backend="hdmr", include_correlative=True, m=5)
    assert res.include_correlative is True
    assert "include_correlative" in repr(res)
    assert res.to_dataset().attrs["include_correlative"] is True


def test_hdmr_order_field(ishigami_hdmr_shapley):
    """The HDMR backend reports its expansion order."""
    assert ishigami_hdmr_shapley.order == 2


@pytest.mark.parametrize("backend", ["hdmr", "pce"])
def test_constant_output_nan_and_warns(ishigami_data, backend):
    """Constant Y yields NaN indices and a zero-variance warning for both backends."""
    X, _ = ishigami_data
    Yc = jnp.full(X.shape[0], 5.0)
    with pytest.warns(UserWarning, match="zero variance"):
        result = gsax.analyze_shapley(ishigami.PROBLEM, X, Yc, backend=backend)
    assert np.all(np.isnan(np.asarray(result.Sh)))
    assert np.all(np.isnan(np.asarray(result.S1)))
    assert np.all(np.isnan(np.asarray(result.ST)))
    assert np.isnan(float(result.explained_variance))


def test_multi_output_explained_variance_per_slice(ishigami_data):
    """explained_variance is reported per output slice and Sh sums to 1 per slice."""
    X, Y = ishigami_data
    Y2 = jnp.stack([Y, 3.0 * Y], axis=1)
    result = gsax.analyze_shapley(ishigami.PROBLEM, X, Y2, backend="hdmr")
    assert result.explained_variance.shape == (2,)
    np.testing.assert_allclose(np.asarray(result.Sh).sum(axis=-1), 1.0, atol=1e-5)


def test_pce_time_series_shapes_and_slices(linear_data):
    """backend='pce' handles (N, T, K); each slice matches the scalar run."""
    X, Y = linear_data
    Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=-1)  # (N, K=2)
    Y3 = jnp.stack([Y2, 0.5 * Y2], axis=1)  # (N, T=2, K=2)
    result = gsax.analyze_shapley(linear.PROBLEM, X, Y3, backend="pce", order=2)
    scalar = gsax.analyze_shapley(linear.PROBLEM, X, Y, backend="pce", order=2)
    assert result.Sh.shape == result.S1.shape == result.ST.shape == (2, 2, 3)
    assert result.explained_variance.shape == (2, 2)
    np.testing.assert_allclose(result.Sh.sum(axis=-1), 1.0, rtol=1e-6)
    for t in range(2):
        for k in range(2):
            np.testing.assert_allclose(result.Sh[t, k], scalar.Sh, rtol=1e-5, atol=1e-6)


def test_pce_time_series_to_dataset(linear_data):
    X, Y = linear_data
    Y3 = jnp.stack([jnp.stack([Y, 2.0 * Y], axis=-1)] * 2, axis=1)
    result = gsax.analyze_shapley(linear.PROBLEM, X, Y3, backend="pce", order=2)
    ds = result.to_dataset(time_coords=[0.1, 0.2])
    assert ds["Sh"].dims == ("time", "output", "param")
    np.testing.assert_allclose(ds.coords["time"].values, [0.1, 0.2])
