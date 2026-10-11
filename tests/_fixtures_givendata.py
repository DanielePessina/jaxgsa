"""Shared fixtures for the given-data tests (HSIC, OT, Borgonovo, PAWN).

All four methods read one ``(X, Y)`` sample. They share a few session-wide
samples and results here, so one Monte Carlo draw and one jit compile serve
many tests. ``conftest.py`` loads this module as a pytest plugin, so test files
take the fixtures by name with no import. Plain helpers (``BOOT``,
``invalid_sample``) are imported normally.

Keep the shapes and static arguments below fixed. A new ``n_bootstrap`` or
``n_classes`` value costs a new compile, so the result fixtures all use one
bootstrap size, ``BOOT``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import borgonovo, optimal_transport, pawn
from jaxgsa.benchmarks import gaussian_linear, ishigami
from jaxgsa.problem import Problem
from jaxgsa.sampling import monte_carlo

# One bootstrap size for every method, so the compile is shared.
BOOT = 8


@pytest.fixture(scope="session")
def given_ishigami_xy():
    """Ishigami Monte Carlo sample, ``N = 2048``, seed 42."""
    X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=2048, seed=42))
    return X, ishigami.evaluate(X)


@pytest.fixture(scope="session")
def ishigami_xy_4k():
    """Ishigami Monte Carlo sample, ``N = 4096``, seed 42 (OT closed-form checks)."""
    X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=4096, seed=42))
    return X, ishigami.evaluate(X)


@pytest.fixture(scope="session")
def ishigami_xy_series(given_ishigami_xy):
    """``(N, 2)`` and ``(N, 2, 2)`` outputs built from the Ishigami sample."""
    X, Y = given_ishigami_xy
    Y2 = jnp.stack([Y, Y**2], axis=1)
    Y3 = jnp.stack([Y2, Y2 + 1.0], axis=1)
    return X, Y2, Y3


@pytest.fixture(scope="session")
def gaussian_linear_xy():
    """Gaussian linear sample, ``N = 4096``, seed 1 (closed-form smoke tests)."""
    X = jnp.asarray(monte_carlo(gaussian_linear.PROBLEM, n_samples=4096, seed=1))
    return X, gaussian_linear.evaluate(X)


@pytest.fixture(scope="session")
def ot_ishigami(ishigami_xy_4k):
    """OT result on the 4096-row Ishigami sample, no bootstrap."""
    X, Y = ishigami_xy_4k
    return optimal_transport.analyze(ishigami.PROBLEM, X, Y, verbose=False)


@pytest.fixture(scope="session")
def ot_ishigami_boot(ishigami_xy_4k):
    """OT result with ``BOOT`` replicates and the dummy floor (key 0)."""
    X, Y = ishigami_xy_4k
    return optimal_transport.analyze(
        ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, dummy=True, key=jax.random.key(0), verbose=False
    )


@pytest.fixture(scope="session")
def delta_ishigami(given_ishigami_xy):
    """Borgonovo result on the Ishigami sample, no bootstrap."""
    X, Y = given_ishigami_xy
    return borgonovo.analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0, verbose=False)


@pytest.fixture(scope="session")
def delta_ishigami_boot(given_ishigami_xy):
    """Borgonovo result with ``BOOT`` replicates (key 0)."""
    X, Y = given_ishigami_xy
    return borgonovo.analyze(
        ishigami.PROBLEM,
        X,
        Y,
        n_bootstrap=BOOT,
        bias_correct=True,
        key=jax.random.key(0),
        verbose=False,
    )


@pytest.fixture(scope="session")
def pawn_ishigami(given_ishigami_xy):
    """PAWN result on the Ishigami sample, no bootstrap."""
    X, Y = given_ishigami_xy
    return pawn.analyze(ishigami.PROBLEM, X, Y, verbose=False)


@pytest.fixture(scope="session")
def pawn_ishigami_boot(given_ishigami_xy):
    """PAWN result with ``BOOT`` replicates (key 0)."""
    X, Y = given_ishigami_xy
    return pawn.analyze(
        ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, key=jax.random.key(0), verbose=False
    )


def invalid_sample(n: int = 200, seed: int = 0):
    """Build a clean two-parameter sample with a continuous output.

    Used by the shared ``on_invalid`` tests. The output has a little noise, so
    it is continuous for the Borgonovo estimator, and it reads both inputs.

    Args:
        n: Number of rows.
        seed: Seed of the NumPy generator.

    Returns:
        ``(problem, X, Y)`` with ``X`` of shape ``(n, 2)`` and ``Y`` of shape ``(n,)``.
    """
    problem = Problem(names=("a", "b"), bounds=((0.0, 1.0), (0.0, 1.0)))
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 2))
    Y = np.sin(3.0 * X[:, 0]) + 0.3 * X[:, 1] + 0.01 * rng.normal(size=n)
    return problem, jnp.asarray(X), jnp.asarray(Y)
