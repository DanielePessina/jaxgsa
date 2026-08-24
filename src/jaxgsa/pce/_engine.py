"""Polynomial basis construction and evaluation for PCE.

Each 1-D orthonormal basis is built from its three-term recurrence: Hermite
for a Gaussian marginal, Legendre for a uniform marginal. The
multi-dimensional basis is the tensor product of those 1-D families, selected
by a multi-index set.
"""

from __future__ import annotations

from math import comb, factorial

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.linalg import solve_triangular

from jaxgsa._core.batching import get_memory_budget

# Shared recurrence in _core.legendre; the VKOGA component fit uses it too.
from jaxgsa._core.legendre import legendre_orthonormal as _legendre_1d
from jaxgsa._core.precision import unit_clip_bounds

# Nominal gap the leverage clip keeps between h_i and 1. The value the clip
# actually uses is dtype-aware — see hat_diagonal.
_LEVERAGE_CLIP = 1e-10


def _hermite_1d(x: Array, max_degree: int) -> Array:
    """Evaluate orthonormal probabilist's Hermite polynomials.

    Orthonormal w.r.t. the standard normal measure N(0, 1):
        E[tilde_He_m(X) tilde_He_n(X)] = delta_{mn}

    Args:
        x: Standardized points (zero mean, unit variance), shape ``(N,)``.
        max_degree: Maximum polynomial degree.

    Returns:
        Matrix of basis values, shape ``(N, max_degree + 1)``.
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
        D: Number of dimensions.
        p: Maximum total polynomial degree.

    Returns:
        Integer array of shape ``(n_terms, D)`` where n_terms = C(D+p, p).
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
        X: Input samples already mapped to the reference domain, shape
            ``(N, D)``: [-1,1] for uniform, standardized for Gaussian.
        multi_index: Multi-index array, shape ``(n_terms, D)``.
        input_types: Tuple of "uniform" or "gaussian" per dimension.
        max_degree: Maximum 1-D polynomial degree.

    Returns:
        Design matrix of shape ``(N, n_terms)`` where
        Phi[n, alpha] = Psi_alpha(X_n).
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
    is_constant: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Compute Sobol indices from PCE coefficients (Sudret 2008).

    Batched over any leading output-slice dims. The term axis is always last,
    and every reduction contracts it in a single vectorized operation (masked
    matmuls and one einsum). No Python loop runs over slices or parameter
    pairs.

    Args:
        coefficients: Expansion coefficients, shape ``(..., n_terms)``.
            Leading dims are output slices, e.g. ``(T, K)`` for time-series
            outputs.
        multi_index: Multi-index array, shape ``(n_terms, D)``, shared by all
            slices.
        is_constant: Boolean array, shape ``(...,)`` matching the leading
            output-slice dims, ``True`` where the caller's ``Y`` slice is
            numerically constant. Pass this whenever ``Y`` is available: the
            fitted non-constant coefficients of a constant slice are rarely
            bit-exact zero in float32, so a fit-only ``total_var == 0`` guard
            (the fallback below) misses most constant slices and returns a
            plausible-looking but meaningless index instead of NaN. When
            ``None``, only the fit-only guard runs.

    Returns:
        Tuple ``(S1, ST, S2)`` where:
            S1: First-order indices, shape ``(..., D)``.
            ST: Total-order indices, shape ``(..., D)``.
            S2: Second-order interaction indices with a NaN diagonal, shape
                ``(..., D, D)``.
    """
    mi = np.asarray(multi_index)
    D = mi.shape[1]
    # Orthonormality ensures each c_alpha^2 equals the partial variance
    # contributed by the basis function Psi_alpha (Parseval's identity).
    c2 = jnp.asarray(coefficients) ** 2

    # Total variance = sum of all c_alpha^2 excluding the constant term
    # (alpha=0), per output slice.
    total_var = jnp.sum(c2[..., 1:], axis=-1)
    zero = total_var == 0
    if is_constant is not None:
        zero = zero | is_constant
    # Guard against zero-variance slices (constant output).
    inv_var = jnp.where(zero, jnp.nan, 1.0 / total_var)

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
    # pairs once (numpy, at trace time), so extraction is masked matmuls over
    # the term axis: the same shape as S1/ST, and half the work of contracting
    # the full symmetric (D, D) tensor.
    #
    # The pair mask is (n_terms, n_pairs), and the matmul promotes it to the
    # coefficient dtype, so as one array it is an unbudgeted quadratic-in-D
    # float transient — ~3.5 GB at D=100, p=3. The pair axis is therefore
    # walked in chunks sized against the memory budget. At small D the budget
    # allows one chunk, and that path runs the exact matmul this function has
    # always run, returned without a concatenate — bit-identical, which the
    # chunking test pins. A split only happens where the transient would not
    # fit; there each chunk computes the same per-column dot products, and
    # only XLA's per-shape reassociation of a reduction can move last bits.
    # The per-pair transient is the promoted mask column plus its boolean
    # original (~2 * n_terms elements) plus one output column per slice.
    iu, ju = np.triu_indices(D, k=1)  # (n_pairs,) each
    n_pairs = iu.shape[0]
    pair2 = (active_count == 2)[:, None]  # (n_terms, 1)
    n_terms = mi.shape[0]
    n_slices = int(np.prod(c2.shape[:-1], dtype=np.int64))
    itemsize = jnp.dtype(c2.dtype).itemsize
    bytes_per_pair = itemsize * (2 * n_terms + n_slices)
    chunk = max(1, min(n_pairs, get_memory_budget() // max(bytes_per_pair, 1)))
    parts = [jnp.zeros((*c2.shape[:-1], 0), dtype=c2.dtype)]  # covers D == 1
    for start in range(0, n_pairs, chunk):
        end = start + chunk
        pair_mask = active[:, iu[start:end]] & active[:, ju[start:end]] & pair2
        parts.append(jnp.asarray(c2 @ pair_mask))  # (..., <=chunk)
    stacked = parts[-1] if len(parts) == 2 else jnp.concatenate(parts, axis=-1)
    s2_upper = stacked * inv_var[..., None]  # (..., n_pairs)
    # Scatter into the symmetric matrix and mirror; the diagonal is not a pair
    # index, so keep it NaN to match SobolResult's S2.
    S2 = jnp.full((*c2.shape[:-1], D, D), jnp.nan)
    S2 = S2.at[..., iu, ju].set(s2_upper).at[..., ju, iu].set(s2_upper)

    return S1, ST, S2


def gram_cholesky(Phi: Array, ridge: float = 0.0) -> Array:
    """Factor the (regularized) Gram matrix of a design block.

    Args:
        Phi: Design matrix, shape ``(N, n_terms)``.
        ridge: Tikhonov parameter added to the diagonal.

    Returns:
        Lower-triangular Cholesky factor ``L`` of ``Phi^T Phi + ridge*I``,
        shape ``(n_terms, n_terms)``.
    """
    gram = Phi.T @ Phi
    if ridge > 0:
        gram = gram + ridge * jnp.eye(gram.shape[0], dtype=gram.dtype)
    return jnp.linalg.cholesky(gram)


def hat_diagonal(Phi: Array, gram_chol: Array) -> Array:
    """Hat-matrix diagonal (leverage) of a design block, from a Cholesky factor.

    This is the single definition of PCE leverage. Both the single-pass fit
    and the row-streamed fit call it, so the two paths cannot drift apart.

    With ``Phi^T Phi + ridge*I = L L^T``, the leverage of row ``i`` is

        ``h_i = phi_i (L L^T)^{-1} phi_i^T = || L^{-1} phi_i ||^2``

    so one triangular solve gives every ``h_i``. The Gram matrix is never
    inverted. That matters here: the default ridge is ``1e-8`` and PCE Gram
    matrices lose conditioning quickly as the polynomial order rises, so an
    explicit inverse would square an already large condition number, and a
    degraded leverage feeds straight back into automatic order selection.
    The triangular solve keeps the error at the level of the factorization
    itself, and ``h_i`` comes out as a sum of squares, hence non-negative by
    construction.

    Args:
        Phi: Design matrix or row block, shape ``(b, n_terms)``.
        gram_chol: Lower-triangular Cholesky factor of the regularized Gram
            matrix, shape ``(n_terms, n_terms)``.

    Returns:
        Leverage per row, shape ``(b,)``, clipped to ``[0, high]`` with
        ``high`` strictly below 1 in the leverage's own dtype. The upper clip
        keeps ``1 - h_i`` away from zero: a row the fit interpolates exactly
        has ``h_i = 1``, which would divide by zero and turn ``loo_rmse``
        into inf/NaN for every output slice. The nominal bound is
        ``1 - 1e-10``, but in float32 that rounds back to exactly 1.0 (the
        spacing below 1.0 is about ``6e-8``), which made the clip a silent
        no-op — the same dtype trap :func:`jaxgsa._core.precision.unit_clip_bounds`
        exists for, so it supplies the bound.
    """
    # (n_terms, b): solve L Z = Phi^T, then h_i = sum_j Z[j, i]^2.
    Z = solve_triangular(gram_chol, Phi.T, lower=True)
    leverage = jnp.sum(Z * Z, axis=0)
    _, high = unit_clip_bounds(_LEVERAGE_CLIP, leverage.dtype)
    return jnp.clip(leverage, 0.0, high)


def loo_error(
    Phi: Array,
    Y: Array,
    coefficients: Array,
    ridge: float = 0.0,
    gram_chol: Array | None = None,
) -> Array:
    """Efficient leave-one-out cross-validation error from the hat matrix.

    Uses the identity: ``e_LOO_i = (Y_i - Phi_i @ c) / (1 - H_ii)``
    where ``H = Phi @ (Phi^T Phi + ridge*I)^{-1} @ Phi^T``.

    The leverage ``H_ii`` depends only on ``Phi``, so a single hat-matrix
    diagonal serves every output slice; residuals broadcast over trailing
    slice columns. :func:`hat_diagonal` computes it.

    Args:
        Phi: Design matrix, shape ``(N, n_terms)``.
        Y: Outputs, shape ``(N,)``, or ``(N, M)`` with one column per output
            slice.
        coefficients: Fitted coefficients, shape ``(n_terms,)`` or
            ``(n_terms, M)``.
        ridge: Tikhonov parameter used during fitting.
        gram_chol: Pre-computed lower-triangular Cholesky factor of
            ``Phi^T Phi + ridge*I``, shape ``(n_terms, n_terms)``. Pass it to
            reuse a factorization the caller already holds. It is the small
            factor, not an ``(n_terms, N)`` product, so reuse costs no memory
            that scales with ``N``.

    Returns:
        Scalar LOO RMSE for 1-D ``Y``, or per-slice LOO RMSE of shape
        ``(M,)`` for 2-D ``Y``.
    """
    residuals = Y - Phi @ coefficients  # (N,) or (N, M)
    if gram_chol is None:
        gram_chol = gram_cholesky(Phi, ridge)
    denom = 1.0 - hat_diagonal(Phi, gram_chol)
    if residuals.ndim == 2:
        denom = denom[:, None]  # broadcast the shared leverage over slices
    loo_residuals = residuals / denom
    return jnp.sqrt(jnp.mean(loo_residuals**2, axis=0))
