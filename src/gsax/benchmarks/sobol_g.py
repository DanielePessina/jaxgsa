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

# a_j=0 => maximally influential; a_j=99 => nearly inert.
# This mix creates 4 tiers: dominant (x1), moderate (x2), weak (x3-x4), negligible (x5-x8).
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
    # Each factor (|4x_j-2|+a_j)/(1+a_j) is mean-1 and variance 1/(3(1+a_j)^2).
    # The product form means ALL subsets of inputs interact (no purely additive structure).
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

    # Each factor's variance: Vi = Var(g_j) = 1/(3(1+a_j)^2), from integrating
    # Var((|4U-2|+a)/(1+a)) over U~Uniform[0,1].
    Vi = 1.0 / (3.0 * (1.0 + a_arr) ** 2)
    # Total variance via multiplicative ANOVA: V(Y) = prod(1 + V_j) - 1,
    # because the factors are independent and each has mean 1.
    VY = np.prod(1.0 + Vi) - 1.0

    S1 = Vi / VY

    # Total-order index includes all interactions containing x_j.
    # For a product model: ST_j = V_j * prod_{k!=j}(1 + V_k) / V(Y).
    ST = np.empty(D)
    for j in range(D):
        others = np.prod(1.0 + np.delete(Vi, j))
        ST[j] = Vi[j] * others / VY

    # Second-order interaction S2_jk = V_j * V_k / V(Y), a direct consequence
    # of the multiplicative structure (each pair's joint effect factorizes).
    S2 = np.full((D, D), np.nan)
    for j in range(D):
        for k in range(j + 1, D):
            val = Vi[j] * Vi[k] / VY
            S2[j, k] = val
            S2[k, j] = val

    return S1, ST, S2


ANALYTICAL_S1, ANALYTICAL_ST, ANALYTICAL_S2 = analytical_indices()
