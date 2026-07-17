"""Generic input sampling utilities."""

from __future__ import annotations

import numpy as np

from gsax.problem import Problem
from gsax.sobol._sampling import _transform_samples


def monte_carlo(
    problem: Problem,
    n: int,
    *,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Draw independent Monte Carlo samples from a problem's marginals."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    return _transform_samples(problem, rng.random((n, problem.num_vars)))


__all__ = ["monte_carlo"]
