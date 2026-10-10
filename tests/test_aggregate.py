"""Tests for ``result.aggregate()``: one index per input across many outputs.

Tier T0 (closed form). For ``Y_t = a_t . X`` with independent ``U(0, 1)``
inputs, ``Var(X_i) = 1/12`` and the model is additive, so

    V_t      = |a_t|^2 / 12
    S1_{i,t} = ST_{i,t} = a_{t,i}^2 / |a_t|^2

The aggregate of Gamboa, Janon, Klein and Lagnoux (2014) is then

    variance weights:  sum_t a_{t,i}^2 / sum_t |a_t|^2
    equal weights:     mean_t S1_{i,t}

Tier T4 (internal consistency) covers the single-output identity, the
replicate-based interval, the refusal on an irregular layout and the optional
``variance`` field of an old Sobol result.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import kucherenko, pce, sobol
from jaxgsa.problem import Problem

# Two outputs of very different variance, so the two weightings disagree.
A = np.array([[1.0, 2.0, 2.0], [8.0, 1.0, 1.0]])
PROBLEM = Problem.from_dict({"x0": (0.0, 1.0), "x1": (0.0, 1.0), "x2": (0.0, 1.0)})

V_TRUE = (A**2).sum(axis=1) / 12.0  # (2,)
S1_TRUE = A**2 / (A**2).sum(axis=1, keepdims=True)  # (2, 3)
AGG_VARIANCE = (A**2).sum(axis=0) / (A**2).sum()  # (3,)
AGG_EQUAL = S1_TRUE.mean(axis=0)  # (3,)


def _model(X):
    return jnp.asarray(X) @ jnp.asarray(A.T, dtype=jnp.asarray(X).dtype)  # (N, 2)


@pytest.fixture(scope="module")
def sobol_res():
    sr = sobol.sample(PROBLEM, n_samples=4096, seed=0, verbose=False)
    return sobol.analyze(sr, _model(sr.samples), verbose=False)


@pytest.fixture(scope="module")
def kuch_res():
    sr = kucherenko.sample(PROBLEM, n_samples=4096, seed=0, verbose=False)
    return kucherenko.analyze(sr, _model(sr.samples), verbose=False)


@pytest.fixture(scope="module")
def pce_res():
    X = jax.random.uniform(jax.random.key(0), (600, 3))
    return pce.analyze(PROBLEM, X, _model(X), order=2, verbose=False)


# --- T0: closed form ---------------------------------------------------------


def test_sobol_variance_field_matches_closed_form(sobol_res):
    np.testing.assert_allclose(np.asarray(sobol_res.variance), V_TRUE, rtol=5e-2)


@pytest.mark.parametrize("fixture", ["sobol_res", "kuch_res"])
def test_aggregate_closed_form_sampling_methods(fixture, request):
    res = request.getfixturevalue(fixture)
    var = res.aggregate()
    eq = res.aggregate(weights="equal")
    assert type(var) is type(res)
    assert var.S1.shape == (1, 3)
    np.testing.assert_allclose(np.asarray(var.S1)[0], AGG_VARIANCE, atol=3e-2)
    np.testing.assert_allclose(np.asarray(var.ST)[0], AGG_VARIANCE, atol=3e-2)
    np.testing.assert_allclose(np.asarray(eq.S1)[0], AGG_EQUAL, atol=3e-2)
    # The two weightings must differ here, or the test says nothing.
    assert np.abs(AGG_VARIANCE - AGG_EQUAL).max() > 0.05


def test_aggregate_closed_form_pce(pce_res):
    var = pce_res.aggregate()
    eq = pce_res.aggregate(weights="equal")
    assert var.S1.shape == (1, 3)
    assert var.S2.shape == (1, 3, 3)
    # A degree-2 expansion holds a linear model exactly.
    np.testing.assert_allclose(np.asarray(var.S1)[0], AGG_VARIANCE, atol=1e-4)
    np.testing.assert_allclose(np.asarray(var.ST)[0], AGG_VARIANCE, atol=1e-4)
    np.testing.assert_allclose(np.asarray(eq.S1)[0], AGG_EQUAL, atol=1e-4)
    # No interaction in an additive model; the diagonal stays NaN.
    s2 = np.asarray(var.S2)[0]
    assert np.isnan(np.diag(s2)).all()
    np.testing.assert_allclose(s2[~np.eye(3, dtype=bool)], 0.0, atol=1e-4)


def test_pce_aggregate_drops_what_does_not_aggregate(pce_res):
    agg = pce_res.aggregate()
    assert agg.loo_rmse is None and agg.explained_variance is None
    assert agg.coefficients.shape == (1, pce_res.coefficients.shape[-1])
    assert np.isnan(np.asarray(agg.coefficients)).all()


# --- T4: algebra on the result's own numbers ---------------------------------


@pytest.mark.parametrize("fixture", ["sobol_res", "kuch_res"])
def test_aggregate_is_the_weighted_sum_of_the_slices(fixture, request):
    res = request.getfixturevalue(fixture)
    v = np.asarray(res.variance)
    s1 = np.asarray(res.S1)
    expected = (v[:, None] * s1).sum(axis=0) / v.sum()
    np.testing.assert_allclose(np.asarray(res.aggregate().S1)[0], expected, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(res.aggregate(weights="equal").S1)[0], s1.mean(0), rtol=1e-5
    )
    np.testing.assert_allclose(float(np.asarray(res.aggregate().variance)[0]), v.sum(), rtol=1e-5)


def test_sobol_second_order_aggregate_is_weighted(sobol_res):
    sr = sobol.sample(PROBLEM, n_samples=512, seed=1, calc_second_order=True, verbose=False)
    res = sobol.analyze(sr, _model(sr.samples), verbose=False)
    v = np.asarray(res.variance)
    expected = (v[:, None, None] * np.asarray(res.S2)).sum(axis=0) / v.sum()
    got = np.asarray(res.aggregate().S2)[0]
    assert got.shape == (3, 3)
    np.testing.assert_allclose(got, expected, rtol=1e-5, equal_nan=True)


def test_sobol_indices_do_not_depend_on_the_variance_field():
    """Adding ``variance`` leaves every index as the estimator made it."""
    sr = sobol.sample(PROBLEM, n_samples=256, seed=2, verbose=False)
    Y = _model(sr.samples)
    res = sobol.analyze(sr, Y, verbose=False)
    plain = dataclasses.replace(res, variance=None)
    np.testing.assert_array_equal(np.asarray(res.S1), np.asarray(plain.S1))
    np.testing.assert_array_equal(np.asarray(res.ST), np.asarray(plain.ST))


def test_sobol_variance_fused_scalar_path_matches_eager():
    """``on_invalid='none'`` defers the row expansion; the variance must agree."""
    sr = sobol.sample(PROBLEM, n_samples=512, seed=3, verbose=False)
    y = _model(sr.samples)[:, 0]
    eager = sobol.analyze(sr, y, verbose=False)
    fused = sobol.analyze(sr, y, on_invalid="none", verbose=False)
    np.testing.assert_allclose(float(fused.variance), float(eager.variance), rtol=1e-5)


def test_aggregate_time_resolved_collapses_both_axes():
    sr = sobol.sample(PROBLEM, n_samples=1024, seed=4, verbose=False)
    Y = jnp.stack([_model(sr.samples), 2.0 * _model(sr.samples)], axis=1)  # (N, T=2, K=2)
    res = sobol.analyze(sr, Y, verbose=False)
    agg = res.aggregate()
    assert agg.S1.shape == (1, 1, 3)
    np.testing.assert_allclose(np.asarray(agg.S1)[0, 0], AGG_VARIANCE, atol=5e-2)
    assert agg.to_dataset().S1.dims == ("time", "output", "param")


# --- single output -----------------------------------------------------------


def test_single_output_is_the_identity():
    sr = sobol.sample(PROBLEM, n_samples=512, seed=5, verbose=False)
    y = _model(sr.samples)[:, 0]
    res = sobol.analyze(sr, y, n_bootstrap=5, key=jax.random.key(0), verbose=False)
    for weights in ("variance", "equal"):
        agg = res.aggregate(weights=weights)
        np.testing.assert_array_equal(np.asarray(agg.S1), np.asarray(res.S1))
        np.testing.assert_array_equal(np.asarray(agg.ST), np.asarray(res.ST))
        np.testing.assert_array_equal(np.asarray(agg.S1_conf), np.asarray(res.S1_conf))
    one_col = sobol.analyze(sr, y[:, None], verbose=False)
    agg = one_col.aggregate()
    assert agg.S1.shape == (1, 3)
    np.testing.assert_array_equal(np.asarray(agg.S1), np.asarray(one_col.S1))


# --- confidence intervals ----------------------------------------------------


def test_replicates_give_an_interval_from_aggregated_draws():
    sr = sobol.sample(PROBLEM, n_samples=512, seed=6, verbose=False)
    Y = _model(sr.samples)
    res = sobol.analyze(
        sr, Y, n_bootstrap=60, key=jax.random.key(1), keep_replicates=True, verbose=False
    )
    agg = res.aggregate()
    assert agg.ci is not None and agg.ci.replicates is not None
    assert agg.ci.n_bootstrap == 60 and agg.ci.level == res.ci.level
    draws = np.asarray(res.ci.replicates["S1"])  # (R, 2, 3)
    v = np.asarray(res.variance)
    manual = (v[None, :, None] * draws).sum(axis=1) / v.sum()  # (R, 3)
    np.testing.assert_allclose(np.asarray(agg.ci.replicates["S1"])[:, 0], manual, rtol=1e-4)
    lo, hi = np.quantile(manual, [0.025, 0.975], axis=0)
    conf = np.asarray(agg.S1_conf)
    assert conf.shape == (2, 1, 3)
    np.testing.assert_allclose(conf[0, 0], lo, atol=5e-3)
    np.testing.assert_allclose(conf[1, 0], hi, atol=5e-3)
    assert (conf[0] <= np.asarray(agg.S1)).all() and (np.asarray(agg.S1) <= conf[1]).all()


def test_without_replicates_the_interval_is_nan():
    sr = sobol.sample(PROBLEM, n_samples=256, seed=7, verbose=False)
    res = sobol.analyze(
        sr, _model(sr.samples), n_bootstrap=5, key=jax.random.key(2), verbose=False
    )
    agg = res.aggregate()
    assert agg.S1_conf.shape == (2, 1, 3)
    assert np.isnan(np.asarray(agg.S1_conf)).all()
    assert np.isnan(np.asarray(agg.ST_conf)).all()


def test_no_bootstrap_stays_without_an_interval(sobol_res):
    agg = sobol_res.aggregate()
    assert agg.S1_conf is None and agg.ci is None


# --- refusals and edge cases -------------------------------------------------


def test_unknown_weights_raise(sobol_res):
    with pytest.raises(ValueError, match="weights"):
        sobol_res.aggregate(weights="median")  # ty: ignore[invalid-argument-type]


def test_irregular_layout_is_refused(sobol_res):
    odd = dataclasses.replace(
        sobol_res, S1=jnp.zeros((2, 2, 2, 3)), ST=jnp.zeros((2, 2, 2, 3)), S2=None
    )
    with pytest.raises(ValueError, match="regular"):
        odd.aggregate()


def test_constant_output_gets_zero_weight():
    sr = sobol.sample(PROBLEM, n_samples=512, seed=8, verbose=False)
    y = _model(sr.samples)
    Y = jnp.stack([y[:, 0], jnp.full_like(y[:, 0], 3.0)], axis=1)
    res = sobol.analyze(sr, Y, verbose=False, on_invalid="none")
    assert np.isnan(np.asarray(res.S1)[1]).all()
    agg = res.aggregate()
    np.testing.assert_allclose(np.asarray(agg.S1)[0], S1_TRUE[0], atol=3e-2)
    assert np.isnan(np.asarray(res.aggregate(weights="equal").S1)).all()


# --- the new Sobol field and old results -------------------------------------


def test_variance_is_exported_to_the_dataset(sobol_res):
    ds = sobol_res.to_dataset()
    assert ds["variance"].dims == ("output",)
    np.testing.assert_allclose(ds["variance"].values, np.asarray(sobol_res.variance))
    assert "variance" in repr(sobol_res)


def test_old_result_without_variance_still_works(sobol_res):
    """A result built before the field existed has ``variance=None``."""
    old = dataclasses.replace(sobol_res, variance=None)
    assert "variance" not in old.to_dataset()
    assert "variance" not in repr(old)
    eq = old.aggregate(weights="equal")
    np.testing.assert_allclose(
        np.asarray(eq.S1)[0], np.asarray(sobol_res.S1).mean(axis=0), rtol=1e-5
    )
    assert eq.variance is None
    with pytest.raises(ValueError, match="variance"):
        old.aggregate()


def test_aggregated_result_round_trips_through_netcdf(sobol_res, tmp_path):
    xr = pytest.importorskip("xarray")
    path = tmp_path / "agg.nc"
    ds = sobol_res.aggregate().to_dataset()
    ds.to_netcdf(path)
    back = xr.open_dataset(path)
    np.testing.assert_allclose(back["S1"].values, ds["S1"].values)
    np.testing.assert_allclose(back["variance"].values, ds["variance"].values)


def test_pce_constant_output_gets_zero_weight():
    X = jax.random.uniform(jax.random.key(1), (600, 3))
    y = _model(X)[:, 0]
    Y = jnp.stack([y, jnp.full_like(y, 3.0)], axis=1)
    res = pce.analyze(PROBLEM, X, Y, order=2, verbose=False)
    assert np.isnan(np.asarray(res.S1)[1]).all()
    np.testing.assert_allclose(np.asarray(res.aggregate().S1)[0], S1_TRUE[0], atol=1e-3)


def test_pce_failed_fit_makes_the_aggregate_nan():
    # A failed fit has non-finite coefficients. Its variance is unknown, not
    # zero, so it must not drop out of the weighted sum.
    X = jax.random.uniform(jax.random.key(1), (600, 3))
    y = _model(X)[:, 0]
    Y = jnp.stack([y, y.at[0].set(jnp.nan)], axis=1)
    res = pce.analyze(PROBLEM, X, Y, order=2, on_invalid="propagate", verbose=False)
    assert not np.isfinite(np.asarray(res.coefficients)[1]).all()
    assert np.isnan(np.asarray(res.aggregate().S1)).all()
