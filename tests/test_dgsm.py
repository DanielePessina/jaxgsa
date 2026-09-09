"""End-to-end DGSM checks against closed-form derivative fixtures."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks import linear
from jaxgsa.dgsm import analyze
from jaxgsa.dgsm._poincare import marginal_variance, poincare_constant
from jaxgsa.problem import GaussianInputSpec, Problem
from jaxgsa.sampling import monte_carlo

COEFFICIENTS = jnp.asarray([1.0, 2.0, 3.0])


def _linear_fn(x):
    return jnp.dot(COEFFICIENTS, x)


def _quadratic_fn(x):
    return x[0] ** 2 + 0.5 * x[1] ** 3 + 0.25 * x[2]


def _time_series_fn(x):
    base = _linear_fn(x)
    return jnp.stack([jnp.stack([base, 2.0 * base]), jnp.stack([0.5 * base, -base])])


def test_linear_autodiff_matches_exact_derivative_fixture():
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n=2048, seed=4))
    result = analyze(linear.PROBLEM, _linear_fn, X, verbose=False)
    np.testing.assert_allclose(result.nu, COEFFICIENTS**2, atol=1e-5)
    np.testing.assert_allclose(result.sigma, COEFFICIENTS, atol=1e-5)
    np.testing.assert_allclose(result.lower_bound, linear.ANALYTICAL_ST, rtol=0.08)


def test_time_series_autodiff_preserves_output_layout():
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n=512, seed=5))
    result = analyze(linear.PROBLEM, _time_series_fn, X, verbose=False)
    assert result.nu.shape == (2, 2, 3)
    assert result.upper_bound.shape == (2, 2, 3)
    assert result.var_y.shape == (2, 2)


def test_precomputed_jacobian_matches_autodiff():
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n=256, seed=6))
    auto = analyze(linear.PROBLEM, _time_series_fn, X, verbose=False)
    Y = jax.vmap(_time_series_fn)(X)
    dfdx = jax.vmap(jax.jacrev(_time_series_fn))(X)
    precomputed = analyze(linear.PROBLEM, Y=Y, dfdx=dfdx, verbose=False)
    np.testing.assert_allclose(precomputed.nu, auto.nu, rtol=1e-5)
    np.testing.assert_allclose(precomputed.upper_bound, auto.upper_bound, rtol=1e-5)


def test_bootstrap_is_reproducible():
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n=256, seed=7))
    kwargs = dict(n_bootstrap=8, key=jax.random.key(3), verbose=False)
    first = analyze(linear.PROBLEM, _quadratic_fn, X, **kwargs)
    second = analyze(linear.PROBLEM, _quadratic_fn, X, **kwargs)
    assert first.ci is not None
    np.testing.assert_array_equal(first.nu_conf, second.nu_conf)
    assert first.ci.replicates is None


def test_invalid_output_is_named_and_drop_returns_finite_result():
    problem = Problem(("x1", "x2", "x3"), ((0.0, 1.0),) * 3)
    X = jnp.asarray(monte_carlo(problem, n=128, seed=8))
    Y = jax.vmap(_linear_fn)(X).at[7].set(jnp.nan)
    dfdx = jnp.broadcast_to(COEFFICIENTS, (X.shape[0], 3)).at[7, 0].set(jnp.nan)
    with pytest.raises(ValueError, match="non-finite"):
        analyze(problem, Y=Y, dfdx=dfdx, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=JaxgsaWarning)
        result = analyze(problem, Y=Y, dfdx=dfdx, on_invalid="drop", verbose=False)
    assert np.all(np.isfinite(np.asarray(result.nu)))
    assert result.invalid.n_invalid == 1


def test_correlated_inputs_are_rejected():
    problem = jaxgsa.Problem(("a", "b", "c"), ((0.0, 1.0),) * 3).with_correlation(
        np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    X = jnp.asarray(monte_carlo(problem, n=64, seed=9))
    with pytest.raises(ValueError, match="correlat"):
        analyze(problem, _linear_fn, X, verbose=False)


def test_poincare_and_marginal_variance_fixtures():
    uniform = Problem.from_dict({"x": (-jnp.pi, jnp.pi)}).input_specs[0]
    gaussian = Problem.from_dict(
        {"x": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.5)}
    ).input_specs[0]
    assert poincare_constant(uniform) == pytest.approx(4.0)
    assert marginal_variance(uniform) == pytest.approx((2.0 * np.pi) ** 2 / 12.0)
    assert poincare_constant(gaussian) == pytest.approx(1.5)
    assert marginal_variance(gaussian) == pytest.approx(1.5)
