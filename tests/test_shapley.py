"""End-to-end Shapley checks against analytical variance fixtures."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa.benchmarks import ishigami, linear
from jaxgsa.problem import GaussianInputSpec, Problem
from jaxgsa.sampling import monte_carlo


@pytest.fixture(scope="module")
def linear_data():
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n=1024, seed=4))
    return X, linear.evaluate(X)


@pytest.fixture(scope="module")
def ishigami_data():
    X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=2048, seed=5))
    return X, ishigami.evaluate(X)


@pytest.mark.parametrize("backend", ["pce", "hdmr"])
def test_linear_fixture_matches_true_shapley(backend, linear_data):
    X, Y = linear_data
    kwargs = {"backend": backend}
    if backend == "pce":
        kwargs["order"] = 2
    else:
        kwargs.update(maxorder=2, maxiter=20)
    result = jaxgsa.shapley.analyze(linear.PROBLEM, X, Y, **kwargs)
    np.testing.assert_allclose(result.Sh, linear.ANALYTICAL_SHAPLEY, atol=0.05)
    np.testing.assert_allclose(result.Sh.sum(axis=-1), 1.0, atol=1e-5)


def test_ishigami_pce_matches_fixture(ishigami_data):
    X, Y = ishigami_data
    result = jaxgsa.shapley.analyze(ishigami.PROBLEM, X, Y, backend="pce", order=8)
    np.testing.assert_allclose(result.Sh, ishigami.ANALYTICAL_SHAPLEY, atol=0.04)


def test_wrapper_matches_fitted_result_method(linear_data):
    X, Y = linear_data
    wrapper = jaxgsa.shapley.analyze(linear.PROBLEM, X, Y, backend="pce", order=2)
    fitted = jaxgsa.pce.analyze(linear.PROBLEM, X, Y, order=2).shapley()
    np.testing.assert_array_equal(wrapper.Sh, fitted.Sh)
    np.testing.assert_array_equal(wrapper.S1, fitted.S1)
    np.testing.assert_array_equal(wrapper.ST, fitted.ST)


def test_multi_output_and_time_series_shapes(linear_data):
    X, Y = linear_data
    Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=-1)
    Y3 = jnp.stack([Y2, 0.5 * Y2], axis=1)
    result = jaxgsa.shapley.analyze(linear.PROBLEM, X, Y3, backend="pce", order=2)
    assert result.Sh.shape == (2, 2, 3)
    np.testing.assert_allclose(result.Sh.sum(axis=-1), 1.0, atol=1e-5)


def test_bootstrap_returns_bracketed_effects(linear_data):
    X, Y = linear_data
    result = jaxgsa.shapley.analyze(
        linear.PROBLEM, X, Y, backend="pce", order=2, n_bootstrap=4, key=jax.random.key(3)
    )
    assert result.Sh_conf is not None
    assert np.all(np.asarray(result.Sh_conf[0]) <= np.asarray(result.Sh_conf[1]))


def test_correlative_hdmr_matches_closed_form():
    rho = 0.5
    problem = Problem.from_dict(
        {
            "x1": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
            "x2": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
        },
        correlation=np.array([[1.0, rho], [rho, 1.0]]),
    )
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    X = jnp.asarray(np.random.default_rng(7).normal(size=(2048, 2)) @ L.T)
    Y = X[:, 0] + 2.0 * X[:, 1]
    result = jaxgsa.shapley.analyze(
        problem, X, Y, backend="hdmr", include_correlative=True, maxorder=2, maxiter=20
    )
    expected = np.array([2.0 / 7.0, 5.0 / 7.0])
    np.testing.assert_allclose(result.Sh, expected, atol=0.06)


def test_constant_output_is_nan_with_a_warning(ishigami_data):
    X, _ = ishigami_data
    with pytest.warns(UserWarning, match="zero variance"):
        result = jaxgsa.shapley.analyze(ishigami.PROBLEM, X, jnp.ones(X.shape[0]), order=2)
    assert np.all(np.isnan(np.asarray(result.Sh)))


def test_invalid_backend_is_rejected(linear_data):
    X, Y = linear_data
    with pytest.raises(ValueError, match="backend"):
        jaxgsa.shapley.analyze(linear.PROBLEM, X, Y, backend="unknown")


def test_invalid_policy_drop_is_applied_once(linear_data):
    X, Y = linear_data
    dirty = Y.at[5].set(jnp.nan)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = jaxgsa.shapley.analyze(
            linear.PROBLEM, X, dirty, backend="pce", order=2, on_invalid="drop"
        )
    assert result.invalid.n_invalid == 1
    assert sum("non-finite" in str(w.message) for w in caught) == 1
