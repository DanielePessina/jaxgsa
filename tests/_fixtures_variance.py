"""Shared fixtures for the variance-based test files.

``conftest.py`` loads this module as a pytest plugin, so test files take these
fixtures by name with no import. The fixtures are session-scoped and use fixed
shapes, so every test that takes one also shares the jit compiles for that shape.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jaxgsa
from jaxgsa.benchmarks import ishigami


@pytest.fixture(scope="session")
def sobol_ishigami_1k():
    """A second-order Ishigami Sobol design with 2**10 runs, and its outputs.

    Returns:
        A ``(samples, Y)`` pair. ``samples`` is the ``SobolSamples`` design.
        ``Y`` is the ``(n_runs,)`` Ishigami output on its unique rows.
    """
    sr = jaxgsa.sobol.sample(ishigami.PROBLEM, n_samples=2**10, seed=7, verbose=False)
    return sr, ishigami.evaluate(jnp.asarray(sr.samples))


@pytest.fixture(scope="session")
def sobol_ishigami_2k():
    """A second-order Ishigami Sobol design with 2**11 * 8 runs, and its outputs.

    Large enough for bootstrap intervals to bracket their point estimates.

    Returns:
        A ``(samples, Y)`` pair, as for ``sobol_ishigami_1k``.
    """
    sr = jaxgsa.sobol.sample(ishigami.PROBLEM, n_samples=2**11 * 8, seed=42, verbose=False)
    return sr, ishigami.evaluate(jnp.asarray(sr.samples))
