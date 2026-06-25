"""Shared input transforms for expansion methods (HDMR, PCE)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from gsax.problem import Problem


def cdf_to_unit_interval(X: Array, problem: Problem) -> Array:
    """Map each column of X to [0, 1] via its marginal CDF.

    Uniform inputs use an affine map. Gaussian inputs (truncated or
    not) are mapped through their normal CDF. The result is always in
    [0, 1], suitable for B-spline bases or further affine mapping to
    [-1, 1] for Legendre polynomials.

    Args:
        X: (N, D) input samples in physical units.
        problem: Problem with per-dimension distribution specs.

    Returns:
        (N, D) array with each column in [0, 1].
    """
    from scipy.stats import norm, truncnorm

    D = problem.num_vars
    cols = []

    # CDF-to-unit-interval mapping: apply F(x) per dimension so each column
    # lands in (0, 1).  For Gaussian inputs, standardised truncation bounds
    # a=(lo-mu)/sigma, b=(hi-mu)/sigma follow scipy's truncnorm convention.
    # Clipping output to (1e-12, 1-1e-12) keeps values in the open interval
    # so downstream inverse transforms (ppf) never receive exactly 0 or 1.
    for d in range(D):
        dist, first, second, lo, hi = problem._input_specs[d]
        if dist == "uniform":
            cols.append((X[:, d] - first) / (second - first))
        elif lo is not None or hi is not None:
            mean, variance = first, second
            std = float(jnp.sqrt(variance))
            a = -np.inf if lo is None else (lo - mean) / std
            b = np.inf if hi is None else (hi - mean) / std
            u = jnp.asarray(
                truncnorm.cdf(np.asarray(X[:, d]), a=a, b=b, loc=mean, scale=std)
            )
            cols.append(jnp.clip(u, 1e-12, 1.0 - 1e-12))
        else:
            mean, variance = first, second
            std = float(jnp.sqrt(variance))
            u = jnp.asarray(norm.cdf(np.asarray(X[:, d]), loc=mean, scale=std))
            cols.append(jnp.clip(u, 1e-12, 1.0 - 1e-12))

    return jnp.column_stack(cols)
