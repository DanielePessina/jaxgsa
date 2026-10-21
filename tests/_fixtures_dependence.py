"""Shared fits for the HDMR, VKOGA and Kucherenko tests.

``tests/conftest.py`` loads this module as a pytest plugin, so the fixtures
here are available to every test file without an import. They are
session-scoped: one pytest process builds each fit once, and every file that
asks for it shares the fit and the jit compiles behind it.

The values are read-only (JAX arrays, frozen results, read-only numpy). A test
that needs to change one copies it.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks import ishigami
from jaxgsa.hdmr import analyze as analyze_hdmr

from _linear_gaussian import (  # isort: skip
    A_COEF,
    GAUSS_PROBLEM,
    R_GAUSS,
)

# --- HDMR --------------------------------------------------------------------

HDMR_N = 1000


@pytest.fixture(scope="session")
def hdmr_ishigami_xy() -> tuple[jax.Array, jax.Array]:
    """Random Ishigami ``(X, Y)`` data, N = 1000, shared by the HDMR tests."""
    bounds = jnp.array(ishigami.PROBLEM.bounds)
    X = jax.random.uniform(
        jax.random.PRNGKey(7), shape=(HDMR_N, 3), minval=bounds[:, 0], maxval=bounds[:, 1]
    )
    return X, ishigami.evaluate(X)


@pytest.fixture(scope="session")
def hdmr_by_order(hdmr_ishigami_xy):
    """HDMR fits (n_intervals=2) of ``hdmr_ishigami_xy`` at ``order`` 1 and 2, keyed by order.

    The order-2 fit is built with every ``JaxgsaWarning`` promoted to an
    error. That fit is a complete decomposition, so it must stay silent, and
    building it here checks that once instead of in a second fit.
    """
    X, Y = hdmr_ishigami_xy
    fits = {}
    for order in (1, 2):
        with warnings.catch_warnings():
            warnings.simplefilter("error" if order == 2 else "ignore", JaxgsaWarning)
            fits[order] = analyze_hdmr(ishigami.PROBLEM, X, Y, order=order, n_intervals=2)
    return fits


# --- VKOGA and Kucherenko on the correlated linear-Gaussian model ------------

CORRELATED_PROBLEM = GAUSS_PROBLEM.with_correlation(R_GAUSS)


@pytest.fixture(scope="session")
def lg_corr_vkoga():
    """x64 VKOGA analysis of the correlated linear-Gaussian model.

    The surrogate trains on rows drawn from the analysis copula itself. Every
    conditional the estimator draws stays where the joint puts mass, so the
    closed form must come out and the fit must raise no jaxgsa warning. Warnings
    are recorded, not shown, and returned beside the result.
    """
    with jax.enable_x64(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X = jaxgsa.sampling.monte_carlo(CORRELATED_PROBLEM, 1536, seed=7)
        result = jaxgsa.vkoga.analyze(
            CORRELATED_PROBLEM,
            X,
            X @ A_COEF,
            gamma=2.0,
            ridge=1e-10,
            max_centers=150,
            n_outer=256,
            n_inner=64,
            n_variance=4096,
            key=jax.random.key(0),
            verbose=False,
        )
    return result, tuple(caught)


@pytest.fixture(scope="session")
def lg_corr_kucherenko():
    """Kucherenko design (base 4096, seed 1) and analysis of the same model."""
    design = jaxgsa.kucherenko.sample(CORRELATED_PROBLEM, base_n=4096, seed=1, verbose=False)
    return jaxgsa.kucherenko.analyze(design, design.samples @ A_COEF, verbose=False)


# --- Kucherenko on Ishigami --------------------------------------------------


@pytest.fixture(scope="session")
def kuch_ishigami_512():
    """An evaluated Kucherenko design on Ishigami (independent inputs), base 512."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JaxgsaWarning)
        design = jaxgsa.kucherenko.sample(ishigami.PROBLEM, base_n=512, seed=2, verbose=False)
    return design, np.asarray(ishigami.evaluate(design.samples))
