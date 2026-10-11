"""End-to-end DGSM checks against closed-form derivative fixtures."""

from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks import linear
from jaxgsa.dgsm import DGSMIndices, analyze, indices
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
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n_samples=2048, seed=4))
    result = analyze(linear.PROBLEM, X, fn=_linear_fn, verbose=False)
    np.testing.assert_allclose(result.nu, COEFFICIENTS**2, atol=1e-5)
    np.testing.assert_allclose(result.mu, COEFFICIENTS, atol=1e-5)
    np.testing.assert_allclose(result.lower_bound, linear.ANALYTICAL_ST, rtol=0.08)


def test_time_series_autodiff_preserves_output_layout():
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n_samples=512, seed=5))
    result = analyze(linear.PROBLEM, X, fn=_time_series_fn, verbose=False)
    assert result.nu.shape == (2, 2, 3)
    assert result.upper_bound.shape == (2, 2, 3)
    assert result.variance.shape == (2, 2)


def test_precomputed_jacobian_matches_autodiff():
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n_samples=256, seed=6))
    auto = analyze(linear.PROBLEM, X, fn=_time_series_fn, verbose=False)
    Y = jax.vmap(_time_series_fn)(X)
    dfdx = jax.vmap(jax.jacrev(_time_series_fn))(X)
    precomputed = analyze(linear.PROBLEM, X, Y, dfdx=dfdx, verbose=False)
    np.testing.assert_allclose(precomputed.nu, auto.nu, rtol=1e-5)
    np.testing.assert_allclose(precomputed.upper_bound, auto.upper_bound, rtol=1e-5)


def test_bootstrap_is_reproducible():
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n_samples=256, seed=7))
    kwargs = dict(n_bootstrap=8, key=jax.random.key(3), verbose=False)
    first = analyze(linear.PROBLEM, X, fn=_quadratic_fn, **kwargs)
    second = analyze(linear.PROBLEM, X, fn=_quadratic_fn, **kwargs)
    assert first.ci is not None
    np.testing.assert_array_equal(first.nu_conf, second.nu_conf)
    assert first.ci.replicates is None


def test_invalid_output_is_named_and_drop_returns_finite_result():
    problem = Problem(("x1", "x2", "x3"), ((0.0, 1.0),) * 3)
    X = jnp.asarray(monte_carlo(problem, n_samples=128, seed=8))
    Y = jax.vmap(_linear_fn)(X).at[7].set(jnp.nan)
    dfdx = jnp.broadcast_to(COEFFICIENTS, (X.shape[0], 3)).at[7, 0].set(jnp.nan)
    with pytest.raises(ValueError, match="non-finite"):
        analyze(problem, X, Y, dfdx=dfdx, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=JaxgsaWarning)
        result = analyze(problem, X, Y, dfdx=dfdx, on_invalid="drop", verbose=False)
    assert np.all(np.isfinite(np.asarray(result.nu)))
    assert result.invalid.n_invalid == 1


def test_correlated_inputs_are_rejected():
    problem = jaxgsa.Problem(("a", "b", "c"), ((0.0, 1.0),) * 3).with_correlation(
        np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    X = jnp.asarray(monte_carlo(problem, n_samples=64, seed=9))
    with pytest.raises(ValueError, match="correlat"):
        analyze(problem, X, fn=_linear_fn, verbose=False)


def test_poincare_and_marginal_variance_fixtures():
    uniform = Problem.from_dict({"x": (-jnp.pi, jnp.pi)}).input_specs[0]
    gaussian = Problem.from_dict(
        {"x": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.5)}
    ).input_specs[0]
    assert poincare_constant(uniform) == pytest.approx(4.0)
    assert marginal_variance(uniform) == pytest.approx((2.0 * np.pi) ** 2 / 12.0)
    assert poincare_constant(gaussian) == pytest.approx(1.5)
    assert marginal_variance(gaussian) == pytest.approx(1.5)


def test_callable_as_x_raises_type_error():
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n_samples=16, seed=1))
    model: Any = _linear_fn
    with pytest.raises(TypeError, match="pass the model as fn"):
        analyze(linear.PROBLEM, model, fn=None, verbose=False)
    with pytest.raises(TypeError, match="pass the model as fn"):
        analyze(linear.PROBLEM, model, X, verbose=False)


def test_invalid_argument_combinations_raise_value_error():
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n_samples=16, seed=1))
    Y = jax.vmap(_linear_fn)(X)
    dfdx = jnp.broadcast_to(COEFFICIENTS, (X.shape[0], 3))
    with pytest.raises(ValueError, match="not both"):
        analyze(linear.PROBLEM, X, Y, fn=_linear_fn, dfdx=dfdx, verbose=False)
    with pytest.raises(ValueError, match="Provide either"):
        analyze(linear.PROBLEM, X, verbose=False)
    with pytest.raises(ValueError, match="needs both Y and dfdx"):
        analyze(linear.PROBLEM, X, Y, verbose=False)


def test_indices_returns_named_tuple_and_variance_is_exported():
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n_samples=256, seed=2))
    out = indices(linear.PROBLEM, X, fn=_linear_fn)
    assert isinstance(out, DGSMIndices)
    nu, mu, upper, lower, variance = out
    result = analyze(linear.PROBLEM, X, fn=_linear_fn, verbose=False)
    np.testing.assert_allclose(out.mu, result.mu, rtol=1e-5)
    np.testing.assert_allclose(variance, result.variance, rtol=1e-5)
    assert "variance" in result.to_dataset()


def test_indices_validates_x_on_the_precomputed_path_and_still_jits():
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n_samples=64, seed=2))
    Y, dfdx = jax.vmap(jax.value_and_grad(_linear_fn))(X)
    with pytest.raises(ValueError):
        indices(linear.PROBLEM, X[:, :-1], Y, dfdx=dfdx)
    eager = indices(linear.PROBLEM, X, Y, dfdx=dfdx)
    jitted = jax.jit(lambda X, Y, g: indices(linear.PROBLEM, X, Y, dfdx=g))(X, Y, dfdx)
    np.testing.assert_allclose(jitted.mu, eager.mu, rtol=1e-5)
