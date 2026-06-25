"""Polynomial basis construction and evaluation for PCE.

Implements orthonormal 1-D polynomial bases via three-term recurrence
and multi-dimensional tensor-product basis via multi-index sets.
"""

from __future__ import annotations

from math import comb, factorial

import jax.numpy as jnp
import numpy as np
from jax import Array


def _legendre_1d(x: Array, max_degree: int) -> Array:
    """Evaluate orthonormal Legendre polynomials up to degree max_degree.

    Orthonormal w.r.t. the uniform measure on [-1, 1] (weight = 1/2):
        (1/2) * integral_{-1}^{1} tilde_P_m(x) tilde_P_n(x) dx = delta_{mn}

    Args:
        x: (N,) points in [-1, 1].
        max_degree: maximum polynomial degree.

    Returns:
        (N, max_degree + 1) matrix of basis values.
    """
    N = x.shape[0]
    P = jnp.zeros((N, max_degree + 1))
    P = P.at[:, 0].set(1.0)
    if max_degree >= 1:
        P = P.at[:, 1].set(x)

    # Three-term recurrence: P_{n+1}(x) = ((2n+1)·x·P_n − n·P_{n-1}) / (n+1).
    # Numerically stable vs. direct polynomial evaluation (Bonnet's recursion).
    for n in range(1, max_degree):
        P = P.at[:, n + 1].set(
            ((2 * n + 1) * x * P[:, n] - n * P[:, n - 1]) / (n + 1)
        )

    norms = jnp.sqrt(jnp.array([2.0 * k + 1.0 for k in range(max_degree + 1)]))
    return P * norms[None, :]


def _hermite_1d(x: Array, max_degree: int) -> Array:
    """Evaluate orthonormal probabilist's Hermite polynomials.

    Orthonormal w.r.t. the standard normal measure N(0, 1):
        E[tilde_He_m(X) tilde_He_n(X)] = delta_{mn}

    Args:
        x: (N,) standardized points (zero mean, unit variance).
        max_degree: maximum polynomial degree.

    Returns:
        (N, max_degree + 1) matrix of basis values.
    """
    N = x.shape[0]
    H = jnp.zeros((N, max_degree + 1))
    H = H.at[:, 0].set(1.0)
    if max_degree >= 1:
        H = H.at[:, 1].set(x)

    # Probabilist's Hermite recurrence: He_{n+1}(x) = x·He_n(x) − n·He_{n-1}(x).
    # The 1/sqrt(n!) normalization below makes them orthonormal w.r.t. N(0,1).
    for n in range(1, max_degree):
        H = H.at[:, n + 1].set(x * H[:, n] - n * H[:, n - 1])

    norms = jnp.array([1.0 / jnp.sqrt(float(factorial(k))) for k in range(max_degree + 1)])
    return H * norms[None, :]


def build_multi_index(D: int, p: int) -> np.ndarray:
    """Enumerate the graded total-degree multi-index set.

    Returns all alpha in N^D with |alpha| <= p, ordered by total degree
    then lexicographically. The first row is always the zero index
    (constant term).

    Args:
        D: number of dimensions.
        p: maximum total polynomial degree.

    Returns:
        (n_terms, D) integer array where n_terms = C(D+p, p).
    """
    # Enumerate all D-tuples alpha with |alpha| = sum(alpha_i) <= p (graded
    # total-degree set). Sorted by (total degree, lexicographic) for consistent
    # ordering. Total count = C(D+p, p) by stars-and-bars.
    indices: list[tuple[int, ...]] = []

    def _recurse(depth: int, remaining: int, current: list[int]) -> None:
        if depth == D:
            indices.append(tuple(current))
            return
        for k in range(remaining + 1):
            current.append(k)
            _recurse(depth + 1, remaining - k, current)
            current.pop()

    _recurse(0, p, [])
    result = np.array(sorted(indices, key=lambda a: (sum(a), a)), dtype=np.int32)
    assert result.shape[0] == comb(D + p, p)
    return result


def build_design_matrix(
    X: Array,
    multi_index: np.ndarray,
    input_types: tuple[str, ...],
    max_degree: int,
) -> Array:
    """Build the PCE design matrix Phi.

    Args:
        X: (N, D) input samples, already mapped to the reference domain
            ([-1,1] for uniform, standardized for Gaussian).
        multi_index: (n_terms, D) multi-index array.
        input_types: tuple of "uniform" or "gaussian" per dimension.
        max_degree: maximum 1-D polynomial degree.

    Returns:
        (N, n_terms) design matrix where Phi[n, alpha] = Psi_alpha(X_n).
    """
    N, D = X.shape
    basis_1d: list[Array] = []
    for d in range(D):
        if input_types[d] == "uniform":
            basis_1d.append(_legendre_1d(X[:, d], max_degree))
        else:
            basis_1d.append(_hermite_1d(X[:, d], max_degree))

    # Tensor-product basis: Psi_alpha(x) = prod_{d=1}^{D} phi_{alpha_d}(x_d).
    # The multi-index alpha selects which 1-D polynomial degree per dimension.
    mi = jnp.asarray(multi_index)
    stacked = jnp.stack([basis_1d[d][:, mi[:, d]] for d in range(D)])
    return jnp.prod(stacked, axis=0)


def fit_coefficients(
    Phi: Array,
    Y: Array,
    ridge: float = 0.0,
) -> Array:
    """Fit PCE coefficients by regularized least squares.

    Args:
        Phi: (N, n_terms) design matrix.
        Y: (N,) output values.
        ridge: Tikhonov regularization parameter (0 = unregularized).

    Returns:
        (n_terms,) coefficient vector.
    """
    gram = Phi.T @ Phi
    if ridge > 0:
        gram = gram + ridge * jnp.eye(gram.shape[0])
    return jnp.linalg.solve(gram, Phi.T @ Y)


def sobol_from_coefficients(
    coefficients: Array,
    multi_index: np.ndarray,
) -> tuple[Array, Array, Array]:
    """Compute Sobol indices from PCE coefficients (Sudret 2008).

    Args:
        coefficients: (n_terms,) expansion coefficients.
        multi_index: (n_terms, D) multi-index array.

    Returns:
        (S1, ST, S2) where:
            S1: (D,) first-order indices.
            ST: (D,) total-order indices.
            S2: (D, D) second-order interaction indices (NaN diagonal).
    """
    mi = np.asarray(multi_index)
    D = mi.shape[1]
    c2 = jnp.asarray(coefficients) ** 2

    total_var = jnp.sum(c2[1:])
    inv_var = jnp.where(total_var == 0, jnp.nan, 1.0 / total_var)

    active = mi > 0  # (n_terms, D) bool
    active_count = np.sum(active, axis=1)  # (n_terms,)

    # Sobol indices from PCE coefficients (Sudret, 2008):
    # S1_i  = sum(c_alpha^2 : only x_i active) / Var
    # ST_i  = sum(c_alpha^2 : x_i active, possibly with others) / Var
    # S2_ij = sum(c_alpha^2 : exactly x_i and x_j active) / Var
    only_i_mask = active & (active_count[:, None] == 1)  # (n_terms, D)
    S1 = jnp.asarray(c2 @ only_i_mask) * inv_var  # (D,)

    ST = jnp.asarray(c2 @ active) * inv_var  # (D,)

    S2 = jnp.full((D, D), jnp.nan)
    pair_mask = active_count == 2  # (n_terms,)
    for i in range(D):
        for j in range(i + 1, D):
            mask = active[:, i] & active[:, j] & pair_mask
            val = jnp.sum(c2[mask]) * inv_var
            S2 = S2.at[i, j].set(val)
            S2 = S2.at[j, i].set(val)

    return S1, ST, S2


def loo_error(
    Phi: Array, Y: Array, coefficients: Array, ridge: float = 0.0
) -> Array:
    """Efficient leave-one-out cross-validation error from the hat matrix.

    Uses the identity: ``e_LOO_i = (Y_i - Phi_i @ c) / (1 - H_ii)``
    where ``H = Phi @ (Phi^T Phi + ridge*I)^{-1} @ Phi^T``.

    Args:
        Phi: (N, n_terms) design matrix.
        Y: (N,) outputs.
        coefficients: (n_terms,) fitted coefficients.
        ridge: Tikhonov parameter used during fitting.

    Returns:
        Scalar LOO RMSE.
    """
    # LOO via leverage identity: e_LOO_i = (y_i - y_hat_i) / (1 - H_ii),
    # H = Phi (Phi^T Phi + lambda I)^{-1} Phi^T.  Avoids refitting N times.
    residuals = Y - Phi @ coefficients
    gram = Phi.T @ Phi
    if ridge > 0:
        gram = gram + ridge * jnp.eye(gram.shape[0])
    gram_inv_PhiT = jnp.linalg.solve(gram, Phi.T)  # (P, N)
    leverage = jnp.sum(Phi * gram_inv_PhiT.T, axis=1)  # H_ii
    loo_residuals = residuals / jnp.maximum(1.0 - leverage, 1e-10)
    return jnp.sqrt(jnp.mean(loo_residuals**2))
