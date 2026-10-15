"""Polynomial basis construction and evaluation for PCE.

Each 1-D orthonormal basis is built from its three-term recurrence: Hermite
for a Gaussian marginal, Legendre for a uniform marginal. The
multi-dimensional basis is the tensor product of those 1-D families, selected
by a multi-index set.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
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
        x: Standardized points (zero mean, unit variance), shape ``(N,)`` or
            any ``(...,)``; the recurrence runs elementwise.
        max_degree: Maximum polynomial degree.

    Returns:
        Basis values, shape ``(..., max_degree + 1)``.
    """
    H = jnp.zeros((*x.shape, max_degree + 1))
    # Seed the recurrence: He_0(x) = 1, He_1(x) = x.
    H = H.at[..., 0].set(1.0)
    if max_degree >= 1:
        H = H.at[..., 1].set(x)

    # Probabilist's Hermite recurrence: He_{n+1}(x) = x·He_n(x) − n·He_{n-1}(x).
    # The 1/sqrt(n!) normalization below makes them orthonormal w.r.t. N(0,1).
    for n in range(1, max_degree):
        H = H.at[..., n + 1].set(x * H[..., n] - n * H[..., n - 1])

    # Orthonormalize: E[He_k^2] = k! under N(0,1), so dividing by sqrt(k!)
    # gives unit-norm basis functions needed for variance decomposition.
    norms = jnp.array([1.0 / jnp.sqrt(float(factorial(k))) for k in range(max_degree + 1)])
    return H * norms


def _positive_compositions(total: int, n_parts: int):
    """Yield every tuple of ``n_parts`` positive integers summing to ``total``.

    Choosing the ``n_parts - 1`` internal cut points of a length-``total`` bar
    gives each composition exactly once, so there are ``C(total-1, n_parts-1)``
    of them.

    Args:
        total: The sum each composition must reach. At least 1.
        n_parts: How many positive parts to split it into.

    Yields:
        Tuples of length ``n_parts`` whose entries are >= 1 and sum to
        ``total``.
    """
    for cuts in combinations(range(1, total), n_parts - 1):
        previous = 0
        parts = []
        for cut in cuts:
            parts.append(cut - previous)
            previous = cut
        parts.append(total - previous)
        yield tuple(parts)


@lru_cache(maxsize=32)
def build_multi_index(D: int, p: int) -> np.ndarray:
    """Enumerate the graded total-degree multi-index set.

    Returns all alpha in N^D with |alpha| <= p, ordered by total degree
    then lexicographically. The first row is always the zero index
    (constant term).

    Args:
        D: Number of dimensions.
        p: Maximum total polynomial degree.

    Returns:
        Read-only integer array of shape ``(n_terms, D)`` where
        n_terms = C(D+p, p). The result is cached per ``(D, p)``, so a
        refit or a bootstrap replicate does not enumerate it again.
    """
    # Enumerated by support rather than by dimension. A depth-first walk over
    # the D dimensions recurses once per dimension, so its depth is D and it
    # hits CPython's recursion limit at D ~ 997 for *any* order, including
    # order 1 whose basis is a mere D+1 terms. It also visits every one of the
    # D positions for every term, when a term of total degree <= p has at most
    # p non-zero entries. Both problems go away by choosing which positions
    # are non-zero and then how the degree splits between them.
    n_terms = comb(D + p, p)
    result = np.zeros((n_terms, D), dtype=np.int32)

    # Degree 0 is the constant term, already the all-zero first row.
    row = 1
    degree_bounds = [(0, 1)]
    for degree in range(1, p + 1):
        start = row
        # `n_nonzero` positions each carry at least 1 of the total `degree`.
        for n_nonzero in range(1, min(degree, D) + 1):
            for positions in combinations(range(D), n_nonzero):
                for parts in _positive_compositions(degree, n_nonzero):
                    result[row, positions] = parts
                    row += 1
        degree_bounds.append((start, row))
    assert row == n_terms, f"built {row} terms, expected {n_terms}"

    # Sort by (total degree, lex) so the constant term is always index 0.
    # Degree blocks are already emitted in ascending order, so only the rows
    # within each block need ordering. `np.lexsort` takes its last key as the
    # primary one, hence the reversed column view: column 0 sorts first.
    for start, stop in degree_bounds:
        block = result[start:stop]
        block[:] = block[np.lexsort(block.T[::-1])]
    result.setflags(write=False)
    return result


def _basis_table(
    X: Array, input_types: tuple[str, ...], max_degree: int
) -> tuple[Array, np.ndarray]:
    """Tabulate every 1-D orthonormal basis function at every sample.

    Each family (Legendre for ``"uniform"``, Hermite otherwise) runs once over
    all of its columns, so the traced graph does not grow with the number of
    dimensions. The table lists the uniform columns first, then the Gaussian
    ones, so no permutation back to column order is needed: ``position`` says
    where each dimension went, and callers index through it.

    Args:
        X: Reference-domain inputs, shape ``(N, D)``.
        input_types: Per-dimension ``"uniform"`` or ``"gaussian"`` tags.
        max_degree: Maximum 1-D polynomial degree.

    Returns:
        ``(table, position)``. ``table`` has shape ``(N, D, max_degree + 1)``
        and ``table[n, position[d], k] = phi_k(X[n, d])``. ``position`` is an
        integer array of length ``D``.
    """
    uniform = [d for d, kind in enumerate(input_types) if kind == "uniform"]
    gaussian = [d for d, kind in enumerate(input_types) if kind != "uniform"]
    parts: list[Array] = []
    for family, basis in ((uniform, _legendre_1d), (gaussian, _hermite_1d)):
        if family:
            # Applied to the whole (N, G) block at once: both recurrences are
            # elementwise over leading axes. `vmap` over columns would give
            # the same values eagerly, but XLA fuses it differently under
            # `jit`, and the fused float32 recurrence then differs in the last
            # bit from the per-column program this table replaces.
            parts.append(basis(X[:, jnp.asarray(family)], max_degree))
    position = np.argsort(np.asarray(uniform + gaussian))
    table = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1)
    return table, position


def _design_matrix_dense(
    X: Array,
    multi_index: np.ndarray,
    input_types: tuple[str, ...],
    max_degree: int,
) -> Array:
    """Build the design matrix as a running product over every dimension.

    This is the original construction. Its traced graph grows with ``D``, but
    for small ``D`` it compiles fast, and the float32 values it returns are
    the reference that :func:`build_design_matrix` keeps for those sizes.

    Args:
        X: Reference-domain inputs, shape ``(N, D)``.
        multi_index: Multi-index array, shape ``(n_terms, D)``.
        input_types: Per-dimension ``"uniform"`` or ``"gaussian"`` tags.
        max_degree: Maximum 1-D polynomial degree.

    Returns:
        Design matrix of shape ``(N, n_terms)``.
    """
    D = X.shape[1]
    basis_1d: list[Array] = []
    for d in range(D):
        if input_types[d] == "uniform":
            basis_1d.append(_legendre_1d(X[:, d], max_degree))
        else:
            basis_1d.append(_hermite_1d(X[:, d], max_degree))

    # Index into the precomputed 1-D tables and multiply across dimensions as
    # a running product: only the accumulator, one gathered (N, n_terms)
    # factor, and the multiply output are live at a time (~3*N*n_terms
    # floats), instead of stacking all D gathered factors at once.
    mi = jnp.asarray(multi_index)
    Phi = basis_1d[0][:, mi[:, 0]]
    for d in range(1, D):
        Phi = Phi * basis_1d[d][:, mi[:, d]]
    return Phi


def _design_matrix_sparse(
    X: Array,
    multi_index: np.ndarray,
    input_types: tuple[str, ...],
    max_degree: int,
) -> Array:
    """Build the design matrix from at most ``max_degree`` factors per term.

    A term of total degree <= p has at most p non-zero entries, so each term
    is a product of at most that many table lookups, not of ``D`` of them.
    Padding slots point at degree 0, whose value is exactly 1.0, and
    ``x * 1.0 == x`` in IEEE arithmetic. The non-zero factors are gathered in
    ascending dimension order, the order of the dense product, so in exact
    arithmetic and in float64 the result equals it. The unrolled graph has
    ``width`` multiplies instead of ``D``, so compile time stays flat in ``D``.

    Args:
        X: Reference-domain inputs, shape ``(N, D)``.
        multi_index: Multi-index array, shape ``(n_terms, D)``.
        input_types: Per-dimension ``"uniform"`` or ``"gaussian"`` tags.
        max_degree: Maximum 1-D polynomial degree.

    Returns:
        Design matrix of shape ``(N, n_terms)``.
    """
    mi = np.asarray(multi_index)
    table, position = _basis_table(X, input_types, max_degree)
    n_terms = mi.shape[0]
    width = int(np.max(np.count_nonzero(mi, axis=1)))
    if width == 0:
        return jnp.ones((X.shape[0], n_terms), dtype=table.dtype)
    # A stable argsort on "is zero" lists each row's non-zero dimensions first,
    # in ascending order, then pads with zero-degree slots.
    dims = np.argsort(mi == 0, axis=1, kind="stable")[:, :width]
    degs = np.take_along_axis(mi, dims, axis=1).astype(np.int32)
    dims = position[dims].astype(np.int32)
    Phi = table[:, dims[:, 0], degs[:, 0]]
    for j in range(1, width):
        Phi = Phi * table[:, dims[:, j], degs[:, j]]
    return Phi


# Largest D that keeps the dense running product. Its compile time grows with
# D (24.8 s at D=1000), while the sparse form compiles in well under a second
# at any D. Below the cut the dense form keeps float32 results bit-identical to
# earlier releases: XLA fuses the two forms differently, and its fused
# multiply-add contraction then moves float32 values in the last bit.
_DENSE_MAX_D = 16


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
    if X.shape[1] <= _DENSE_MAX_D:
        return _design_matrix_dense(X, multi_index, input_types, max_degree)
    return _design_matrix_sparse(X, multi_index, input_types, max_degree)


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
    if n_terms == 0 or int(active_count.max()) < 2:
        # An order-1 expansion has no term that activates two variables, so
        # every masked sum below would be an exact zero. Build the zeros
        # directly and skip the O(n_terms * n_pairs) mask work. The product
        # with `inv_var` still runs, so a constant slice still reads NaN.
        stacked = jnp.zeros((*c2.shape[:-1], n_pairs), dtype=c2.dtype)
    else:
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
        has ``h_i = 1``, which would divide by zero and turn ``cv_rmse``
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
