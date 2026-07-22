"""Generic input sampling utilities."""

from __future__ import annotations

import numpy as np

from jaxgsa._core.sampling import _transform_samples
from jaxgsa.problem import Problem


def monte_carlo(
    problem: Problem,
    n: int,
    *,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Draw independent Monte Carlo samples from a problem's marginals.

    Each column is drawn independently from the corresponding parameter's
    declared input distribution (uniform, Gaussian, or truncated Gaussian)
    via inverse-CDF transformation of pseudo-random uniforms. Unlike
    :func:`jaxgsa.sobol.sample`, this uses plain pseudo-random draws with no
    low-discrepancy structure and no Saltelli layout.

    Args:
        problem: Problem definition with parameter names and distributions;
            its marginals determine how each column is transformed.
        n: Number of samples (rows) to draw. Must be at least 1.
        seed: Seed for NumPy's default RNG, or an existing
            ``np.random.Generator`` to draw from. ``None`` (default) uses
            fresh OS entropy, so repeated calls give different samples;
            pass an int for reproducible draws.

    Returns:
        Array of shape ``(n, D)``, where ``D`` is the number of parameters,
        in physical units (each column transformed through the problem's
        declared marginal distribution).

    Raises:
        ValueError: If ``n`` is less than 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    return _transform_samples(problem, rng.random((n, problem.num_vars)))


__all__ = ["monte_carlo"]
