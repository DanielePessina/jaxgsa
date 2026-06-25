"""Sobol G-function for sensitivity analysis benchmarking.

A standard D-dimensional multiplicative benchmark with known analytical
Sobol indices for any choice of the importance parameters ``a``.

The G-function is widely used because its analytical variance decomposition
is available in closed form, and the ``a`` vector provides direct control
over each parameter's importance: ``a_j = 0`` makes ``x_j`` highly
influential; large ``a_j`` makes it nearly irrelevant.

References:
    Saltelli, A. and Sobol, I. M. (1995). About the use of rank
    transformation in sensitivity analysis of model output.
    Reliability Engineering & System Safety, 50(3):225-239.
"""

import jax.numpy as jnp
import numpy as np
from jax import Array

from gsax.problem import Problem

DEFAULT_A = (0.0, 1.0, 4.5, 9.0, 99.0, 99.0, 99.0, 99.0)

PROBLEM = Problem.from_dict({f"x{i + 1}": (0.0, 1.0) for i in range(len(DEFAULT_A))})


def evaluate(X: Array, a: tuple[float, ...] = DEFAULT_A) -> Array:
    """Evaluate the Sobol G-function.

    .. math::
        g(\\mathbf{x}) = \\prod_{j=1}^{D}
        \\frac{|4 x_j - 2| + a_j}{1 + a_j}

    Args:
        X: Input array of shape ``(N, D)`` with ``x_j \\in [0, 1]``.
        a: Importance parameters, one per dimension.

    Returns:
        Array of shape ``(N,)`` with function values.
    """
    a_arr = jnp.asarray(a)
    return jnp.prod((jnp.abs(4.0 * X - 2.0) + a_arr) / (1.0 + a_arr), axis=1)


def analytical_indices(
    a: tuple[float, ...] = DEFAULT_A,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytical first-order, total-order, and second-order Sobol indices.

    For the G-function with independent uniform inputs:

    .. math::
        V_j = \\frac{1}{3(1 + a_j)^2}, \\quad
        V(Y) = \\prod_j (1 + V_j) - 1

    Since the G-function is multiplicatively separable, the ANOVA
    decomposition gives closed-form indices at all orders.

    Args:
        a: Importance parameters.

    Returns:
        ``(S1, ST, S2)`` where S1 and ST are ``(D,)`` arrays and S2 is
        a ``(D, D)`` symmetric matrix with NaN on the diagonal.
    """
    a_arr = np.asarray(a, dtype=float)
    D = len(a_arr)

    Vi = 1.0 / (3.0 * (1.0 + a_arr) ** 2)
    VY = np.prod(1.0 + Vi) - 1.0

    S1 = Vi / VY

    ST = np.empty(D)
    for j in range(D):
        others = np.prod(1.0 + np.delete(Vi, j))
        ST[j] = Vi[j] * others / VY

    S2 = np.full((D, D), np.nan)
    for j in range(D):
        for k in range(j + 1, D):
            val = Vi[j] * Vi[k] / VY
            S2[j, k] = val
            S2[k, j] = val

    return S1, ST, S2


ANALYTICAL_S1, ANALYTICAL_ST, ANALYTICAL_S2 = analytical_indices()
