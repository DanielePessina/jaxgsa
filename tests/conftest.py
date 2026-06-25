"""Shared test fixtures for gsax tests."""

import jax
import jax.numpy as jnp
import pytest

import gsax
from gsax.benchmarks import ishigami, linear, oakley_ohagan, sobol_g


# Sobol estimation error scales as O(1/sqrt(N)).  Higher-dimensional or more
# complex models need proportionally larger sample sizes for convergence.
@pytest.fixture(scope="session")
def ishigami_sobol_result():
    """Ishigami Sobol analysis result (session-scoped, computed once)."""
    sr = gsax.sample(ishigami.PROBLEM, n_samples=2**14 * 8, seed=42, verbose=False)
    Y = ishigami.evaluate(jnp.asarray(sr.samples))
    return gsax.analyze(sr, Y)


@pytest.fixture(scope="session")
def ishigami_hdmr_data():
    """Random (X, Y) pair for Ishigami HDMR tests (session-scoped)."""
    key = jax.random.PRNGKey(42)
    N = 2000
    bounds = jnp.array(ishigami.PROBLEM.bounds)
    X = jax.random.uniform(key, shape=(N, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
    Y = ishigami.evaluate(X)
    return X, Y


@pytest.fixture(scope="session")
def linear_sobol_result():
    """Linear additive model Sobol analysis result (session-scoped)."""
    sr = gsax.sample(linear.PROBLEM, n_samples=2**13 * 5, seed=123, verbose=False)
    Y = linear.evaluate(jnp.asarray(sr.samples))
    return gsax.analyze(sr, Y)


@pytest.fixture(scope="session")
def sobol_g_result():
    """Sobol G-function Sobol analysis result (session-scoped, first-order only)."""
    sr = gsax.sample(
        sobol_g.PROBLEM,
        n_samples=2**14 * 10,
        calc_second_order=False,
        seed=456,
        verbose=False,
    )
    Y = sobol_g.evaluate(jnp.asarray(sr.samples))
    return gsax.analyze(sr, Y)


# Oakley-O'Hagan has 15 Gaussian inputs with correlated interactions,
# requiring ~2^14 * 32 samples for stable convergence of S1 and ST.
@pytest.fixture(scope="session")
def oakley_sobol_result():
    """Oakley & O'Hagan Sobol analysis result (session-scoped, first-order only)."""
    sr = gsax.sample(
        oakley_ohagan.PROBLEM,
        n_samples=2**14 * 32,
        calc_second_order=False,
        seed=789,
        verbose=False,
    )
    Y = oakley_ohagan.evaluate(jnp.asarray(sr.samples))
    return gsax.analyze(sr, Y)
