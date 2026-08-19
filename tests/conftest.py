"""Shared test fixtures and helpers for jaxgsa tests."""

import contextlib
import warnings

import jax
import jax.numpy as jnp
import pytest

import jaxgsa
from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks import ishigami, linear, oakley_ohagan, sobol_g


@contextlib.contextmanager
def single_precision_warning():
    """Assert the float32 warning fires in float32, and is absent under x64.

    ``vkoga.analyze`` and ``hsic.analyze`` warn about single precision only
    when JAX is actually in float32, which is right. Written as a bare
    ``pytest.warns``, such a test fails with ``DID NOT WARN`` the moment the
    suite runs with ``JAX_ENABLE_X64=1`` — the test would be precision-blind,
    not the source. The flag is read here rather than at import so a caller
    that switches precision with a context manager gets the treatment that
    matches the precision actually in force.
    """
    if bool(getattr(jax.config, "jax_enable_x64", False)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
        assert not [w for w in caught if "single precision" in str(w.message)]
    else:
        with pytest.warns(JaxgsaWarning, match="single precision"):
            yield


# Sobol estimation error scales as O(1/sqrt(N)).  Higher-dimensional or more
# complex models need proportionally larger sample sizes for convergence.
@pytest.fixture(scope="session")
def ishigami_sobol_result():
    """Ishigami Sobol analysis result (session-scoped, computed once)."""
    sr = jaxgsa.sobol.sample(ishigami.PROBLEM, n_samples=2**14 * 8, seed=42, verbose=False)
    Y = ishigami.evaluate(jnp.asarray(sr.samples))
    return jaxgsa.sobol.analyze(sr, Y)


@pytest.fixture(scope="session")
def linear_sobol_result():
    """Linear additive model Sobol analysis result (session-scoped)."""
    sr = jaxgsa.sobol.sample(linear.PROBLEM, n_samples=2**13 * 5, seed=123, verbose=False)
    Y = linear.evaluate(jnp.asarray(sr.samples))
    return jaxgsa.sobol.analyze(sr, Y)


@pytest.fixture(scope="session")
def sobol_g_result():
    """Sobol G-function Sobol analysis result (session-scoped, first-order only)."""
    sr = jaxgsa.sobol.sample(
        sobol_g.PROBLEM,
        n_samples=2**14 * 10,
        calc_second_order=False,
        seed=456,
        verbose=False,
    )
    Y = sobol_g.evaluate(jnp.asarray(sr.samples))
    return jaxgsa.sobol.analyze(sr, Y)


# Oakley-O'Hagan has 15 Gaussian inputs with correlated interactions,
# requiring ~2^14 * 32 samples for stable convergence of S1 and ST.
@pytest.fixture(scope="session")
def oakley_sobol_result():
    """Oakley & O'Hagan Sobol analysis result (session-scoped, first-order only)."""
    sr = jaxgsa.sobol.sample(
        oakley_ohagan.PROBLEM,
        n_samples=2**14 * 32,
        calc_second_order=False,
        seed=789,
        verbose=False,
    )
    Y = oakley_ohagan.evaluate(jnp.asarray(sr.samples))
    return jaxgsa.sobol.analyze(sr, Y)
