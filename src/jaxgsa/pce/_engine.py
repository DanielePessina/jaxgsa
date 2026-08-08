"""Polynomial basis construction and evaluation for PCE.

Implements orthonormal 1-D polynomial bases via three-term recurrence
and multi-dimensional tensor-product basis via multi-index sets.
"""

from __future__ import annotations

from math import comb, factorial

import jax.numpy as jnp
import numpy as np
from jax import Array

# Shared recurrence in _core.legendre; the VKOGA component fit uses it too.
from jaxgsa._core.legendre import legendre_orthonormal as _legendre_1d


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
    # Seed the recurrence: He_0(x) = 1, He_1(x) = x.
    H = H.at[:, 0].set(1.0)
    if max_degree >= 1:
        H = H.at[:, 1].set(x)

    # Probabilist's Hermite recurrence: He_{n+1}(x) = x·He_n(x) − n·He_{n-1}(x).
    # The 1/sqrt(n!) normalization below makes them orthonormal w.r.t. N(0,1).
    for n in range(1, max_degree):
        H = H.at[:, n + 1].set(x * H[:, n] - n * H[:, n - 1])

    # Orthonormalize: E[He_k^2] = k! under N(0,1), so dividing by sqrt(k!)
    # gives unit-norm basis functions needed for variance decomposition.
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
    indices: list[tuple[int, ...]] = []

    # Depth-first enumeration: at each dimension, assign degree 0..remaining,
    # distributing the remaining total degree budget to later dimensions.
    def _recurse(depth: int, remaining: int, current: list[int]) -> None:
        if depth == D:
            indices.append(tuple(current))
            return
        for k in range(remaining + 1):
            current.append(k)
            _recurse(depth + 1, remaining - k, current)
            current.pop()

    _recurse(0, p, [])
    # Sort by (total degree, lex) so the constant term is always index 0.
    result = np.array(sorted(indices, key=lambda a: (sum(a), a)), dtype=np.int32)
    # Sanity check: the graded set has exactly C(D+p, p) members (stars-and-bars).
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
    # Pre-compute all 1-D basis values per dimension (N x max_degree+1 each).
    # Each dimension picks Legendre or Hermite per the Wiener-Askey scheme.
    basis_1d: list[Array] = []
    for d in range(D):
        if input_types[d] == "uniform":
            basis_1d.append(_legendre_1d(X[:, d], max_degree))
        else:
            basis_1d.append(_hermite_1d(X[:, d], max_degree))

    # Tensor-product basis: Psi_alpha(x) = prod_{d=1}^{D} phi_{alpha_d}(x_d).
    # The multi-index alpha selects which 1-D polynomial degree per dimension.
    mi = jnp.asarray(multi_index)
    # Index into the precomputed 1-D tables and multiply across dimensions as
    # a running product: only the accumulator, one gathered (N, n_terms)
    # factor, and the multiply output are live at a time (~3*N*n_terms
    # floats), instead of stacking all D gathered factors at once.
    Phi = basis_1d[0][:, mi[:, 0]]
    for d in range(1, D):
        Phi = Phi * basis_1d[d][:, mi[:, d]]
    return Phi


def sobol_from_coefficients(
    coefficients: Array,
    multi_index: np.ndarray,
) -> tuple[Array, Array, Array]:
    """Compute Sobol indices from PCE coefficients (Sudret 2008).

    Batched over any leading output-slice dims: the term axis is always last,
    and every reduction contracts it in a single vectorized operation (masked
    matmuls and one einsum) — no Python loop over slices or parameter pairs.

    Args:
        coefficients: (..., n_terms) expansion coefficients; leading dims are
            output slices (e.g. (T, K) for time-series outputs).
        multi_index: (n_terms, D) multi-index array, shared by all slices.

    Returns:
        (S1, ST, S2) where:
            S1: (..., D) first-order indices.
            ST: (..., D) total-order indices.
            S2: (..., D, D) second-order interaction indices (NaN diagonal).
    """
    mi = np.asarray(multi_index)
    D = mi.shape[1]
    # Orthonormality ensures each c_alpha^2 equals the partial variance
    # contributed by the basis function Psi_alpha (Parseval's identity).
    c2 = jnp.asarray(coefficients) ** 2

    # Total variance = sum of all c_alpha^2 excluding the constant term
    # (alpha=0), per output slice.
    total_var = jnp.sum(c2[..., 1:], axis=-1)
    # Guard against zero-variance slices (constant output).
    inv_var = jnp.where(total_var == 0, jnp.nan, 1.0 / total_var)

    # "Active" means variable d has nonzero degree in multi-index alpha.
    active = mi > 0  # (n_terms, D) bool
    active_count = np.sum(active, axis=1)  # (n_terms,)

    # Sobol indices from PCE coefficients (Sudret, 2008):
    # S1_i  = sum(c_alpha^2 : only x_i active) / Var
    # ST_i  = sum(c_alpha^2 : x_i active, possibly with others) / Var
    # S2_ij = sum(c_alpha^2 : exactly x_i and x_j active) / Var

    # First-order: terms where exactly one variable is active.
    only_i_mask = active & (active_count[:, None] == 1)  # (n_terms, D)
    S1 = jnp.asarray(c2 @ only_i_mask) * inv_var[..., None]  # (..., D)

    # Total-order: all terms where variable i participates (any interaction order).
    ST = jnp.asarray(c2 @ active) * inv_var[..., None]  # (..., D)

    # Second-order: for each unordered pair (i < j), sum the c_alpha^2 of terms
    # that activate exactly x_i and x_j. Enumerate the D*(D-1)/2 upper-triangle
    # pairs once (numpy, at trace time), so extraction is a single masked matmul
    # over the term axis — the same shape as S1/ST, and half the work of
    # contracting the full symmetric (D, D) tensor.
    iu, ju = np.triu_indices(D, k=1)  # (n_pairs,) each
    pair_mask = active[:, iu] & active[:, ju] & (active_count == 2)[:, None]  # (n_terms, n_pairs)
    s2_upper = jnp.asarray(c2 @ pair_mask) * inv_var[..., None]  # (..., n_pairs)
    # Scatter into the symmetric matrix and mirror; the diagonal is not a pair
    # index, so keep it NaN to match SobolResult's S2.
    S2 = jnp.full((*c2.shape[:-1], D, D), jnp.nan)
    S2 = S2.at[..., iu, ju].set(s2_upper).at[..., ju, iu].set(s2_upper)

    return S1, ST, S2


def loo_error(
    Phi: Array,
    Y: Array,
    coefficients: Array,
    ridge: float = 0.0,
    gram_inv_PhiT: Array | None = None,
) -> Array:
    """Efficient leave-one-out cross-validation error from the hat matrix.

    Uses the identity: ``e_LOO_i = (Y_i - Phi_i @ c) / (1 - H_ii)``
    where ``H = Phi @ (Phi^T Phi + ridge*I)^{-1} @ Phi^T``.

    The leverage ``H_ii`` depends only on ``Phi``, so a single hat-matrix
    diagonal serves every output slice; residuals broadcast over trailing
    slice columns.

    Args:
        Phi: (N, n_terms) design matrix.
        Y: (N,) outputs, or (N, M) with one column per output slice.
        coefficients: (n_terms,) fitted coefficients, or (n_terms, M).
        ridge: Tikhonov parameter used during fitting.
        gram_inv_PhiT: Pre-computed ``(Phi^T Phi + ridge*I)^{-1} Phi^T``.
            If provided, avoids recomputing the Gram factorization.

    Returns:
        Scalar LOO RMSE for 1-D ``Y``, or (M,) per-slice LOO RMSE for 2-D.
    """
    residuals = Y - Phi @ coefficients  # (N,) or (N, M)
    if gram_inv_PhiT is None:
        gram = Phi.T @ Phi
        if ridge > 0:
            gram = gram + ridge * jnp.eye(gram.shape[0])
        gram_inv_PhiT = jnp.linalg.solve(gram, Phi.T)
    # H_ii without forming the full N x N hat matrix: diag(Phi @ gram_inv_PhiT).
    leverage = jnp.sum(Phi * gram_inv_PhiT.T, axis=1)
    # A leverage of exactly 1 (interpolated point) would divide by zero below.
    leverage = jnp.clip(leverage, 0.0, 1.0 - 1e-10)
    denom = 1.0 - leverage
    if residuals.ndim == 2:
        denom = denom[:, None]  # broadcast the shared leverage over slices
    loo_residuals = residuals / denom
    return jnp.sqrt(jnp.mean(loo_residuals**2, axis=0))
