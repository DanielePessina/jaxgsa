"""Shared sample builders for the cross-cutting test files.

These are plain functions, not fixtures, because they take arguments. Import
them by name, as with ``_linear_gaussian``. Each builder is cached, so files
that ask for the same sample share one array and one jit compile (the shapes
match). The arrays are read-only: copy before changing one.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

import jaxgsa
from jaxgsa import Problem

PROBS = [0.5, 0.3, 0.2]
OFFSETS = np.array([0.0, 2.0, -1.0])


def mixed_problem() -> Problem:
    """One uniform and one 3-level categorical parameter."""
    return Problem.from_dict({"x1": (0.0, 1.0), "c": {"dist": "categorical", "probs": PROBS}})


@lru_cache(maxsize=None)
def mixed_data(n: int = 4000, seed: int = 1, noise: float = 0.1):
    """Sample the mixed problem and evaluate the per-level-offset model.

    Args:
        n: Number of rows.
        seed: Seed for the sample and the noise.
        noise: Standard deviation of the additive output noise.

    Returns:
        ``(problem, X, Y)``. ``X`` and ``Y`` are shared and read-only.
    """
    problem = mixed_problem()
    X = np.array(jaxgsa.sampling.monte_carlo(problem, n, seed=seed))
    codes = X[:, 1].astype(int)
    rng = np.random.default_rng(seed)
    Y = X[:, 0] + OFFSETS[codes] + noise * rng.standard_normal(n)
    X.setflags(write=False)
    Y.setflags(write=False)
    return problem, X, Y
