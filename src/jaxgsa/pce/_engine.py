"""Polynomial basis construction and evaluation for PCE.

Each 1-D orthonormal basis is built from its three-term recurrence: Hermite
for a Gaussian marginal, Legendre for a uniform marginal, and a recurrence
computed for the measure itself (:class:`TruncatedFamily`) for a truncated
Gaussian marginal. The multi-dimensional basis is the tensor product of those
1-D families, selected by a multi-index set.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import comb, factorial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.linalg import solve_triangular

from jaxgsa._core.batching import get_memory_budget

# Shared recurrence in _core.legendre; the VKOGA component fit uses it too.
from jaxgsa._core.legendre import legendre_orthonormal as _legendre_1d
from jaxgsa._core.precision import default_float_dtype, unit_clip_bounds

# Nominal gap the leverage clip keeps between h_i and 1. The value the clip
# actually uses is dtype-aware — see hat_diagonal. float32 clips at its
# largest value below 1 whatever this is. In float64 it was 1e-10, which cut
# real leverages of accurate high-order fits (1 - h = 4e-12 at order 20 on
# 3000 Gaussian rows) and made cv_rmse 24 times too small there.
_LEVERAGE_CLIP = 1e-13


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
    # The basis keeps the dtype of ``x``: a float32 input stays float32 under x64.
    if not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(default_float_dtype())
    H = jnp.zeros((*x.shape, max_degree + 1), dtype=x.dtype)
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
    norms = jnp.array(
        [1.0 / np.sqrt(float(factorial(k))) for k in range(max_degree + 1)], dtype=x.dtype
    )
    return H * norms


class TruncatedFamily(NamedTuple):
    """Three-term recurrence of the orthonormal polynomials of a truncated normal.

    The polynomials are orthonormal against the standard normal truncated to
    ``[a, b]`` and then standardised to zero mean and unit variance, so the
    first one is the input itself. With ``p_0 = 1``::

        root_beta[k+1] * p_{k+1}(x) = (x - alpha[k]) * p_k(x)
                                      - root_beta[k] * p_{k-1}(x)

    A tuple of floats, so a family is hashable and can sit in the static
    ``input_types`` of the jitted fit. Nothing about it is stored on a
    result: ``(a, b, order)`` rebuild it exactly.

    Attributes:
        alpha: ``alpha[k]`` for ``k = 0..order-1``.
        root_beta: ``sqrt(beta_k)`` for ``k = 0..order``. The entry at 0 is a
            placeholder and is never read.
    """

    alpha: tuple[float, ...]
    root_beta: tuple[float, ...]


def _quadrature_interval(a: float, b: float) -> tuple[float, float]:
    """Cut a truncation interval to where the normal density has mass.

    Args:
        a: Lower bound in standard units, may be ``-inf``.
        b: Upper bound in standard units, may be ``+inf``.

    Returns:
        ``(lo, hi)`` with ``a <= lo < hi <= b``. An interval that holds 0 is
        cut at 12 standard units. One in a tail is cut where the density has
        fallen by a factor of ``exp(-40)`` from its edge, so a bound far in
        the tail still gets a finite interval.
    """
    lo, hi = a, b
    if a > 0:
        hi = min(b, a + max(12.0, 40.0 / a))
    elif b < 0:
        lo = max(a, b - max(12.0, 40.0 / -b))
    else:
        lo, hi = max(a, -12.0), min(b, 12.0)
    return lo, hi


@lru_cache(maxsize=256)
def truncated_gaussian_family(
    a: float, b: float, order: int
) -> tuple[float, float, TruncatedFamily]:
    """Orthonormal polynomial recurrence for a normal truncated to ``[a, b]``.

    Runs the Stieltjes procedure on the host, in float64, on a Gauss-Legendre
    discretisation of the truncated density, with full re-orthogonalisation.
    The recurrence then runs on device in any dtype with no Cholesky of a
    Gram matrix, so it stays accurate in float32 where a Gram-based
    orthogonalisation would not. Raw Hermite polynomials are not orthogonal
    against a truncated normal, and the defect grows with degree.

    Args:
        a: Lower truncation bound in standard units (``(low - mean) / std``).
        b: Upper truncation bound in standard units. ``-inf`` and ``inf``
            mark an open side.
        order: Highest polynomial degree needed.

    Returns:
        ``(mean, std, family)``: the mean and standard deviation of the
        truncated standard normal, and the recurrence of its standardised
        orthonormal polynomials.
    """
    lo, hi = _quadrature_interval(a, b)
    n_nodes = 400 + 25 * order
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    u = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    nearest = min(max(0.0, lo), hi)
    # Subtract the log density at the nearest point to 0, so a bound far in
    # the tail does not underflow.
    mass = weights * np.exp(-0.5 * (u**2 - nearest**2))
    mass /= mass.sum()
    mean = float(mass @ u)
    std = float(np.sqrt(mass @ (u - mean) ** 2))
    v = (u - mean) / std

    sqrt_mass = np.sqrt(mass)
    basis = np.zeros((order + 1, n_nodes))  # orthonormal vectors sqrt(w) * p_k(v)
    basis[0] = sqrt_mass
    alpha = np.zeros(order)
    root_beta = np.ones(order + 1)
    for k in range(order):
        current = basis[k]
        alpha[k] = float(current @ (v * current))
        nxt = v * current - alpha[k] * current
        if k > 0:
            nxt -= root_beta[k] * basis[k - 1]
        for _ in range(2):  # Gram-Schmidt twice against every earlier vector
            nxt -= basis[: k + 1].T @ (basis[: k + 1] @ nxt)
        root_beta[k + 1] = float(np.linalg.norm(nxt))
        basis[k + 1] = nxt / root_beta[k + 1]
    return mean, std, TruncatedFamily(tuple(alpha.tolist()), tuple(root_beta.tolist()))


def _truncated_1d(x: Array, alpha: np.ndarray, root_beta: np.ndarray, max_degree: int) -> Array:
    """Evaluate the orthonormal polynomials of a truncated normal.

    Args:
        x: Standardised points, shape ``(N,)``, or ``(N, G)`` for ``G``
            columns with one recurrence each.
        alpha: ``TruncatedFamily.alpha``, shape ``(order,)``, or
            ``(G, order)`` for ``G`` columns.
        root_beta: ``TruncatedFamily.root_beta``, shape ``(order + 1,)``, or
            ``(G, order + 1)`` for ``G`` columns.
        max_degree: Maximum polynomial degree, at most the order the
            recurrence was built for.

    Returns:
        Basis values, shape ``(..., max_degree + 1)``.
    """
    x = jnp.asarray(x)
    dtype = jnp.result_type(x.dtype, float)
    x = x.astype(dtype)
    a = jnp.asarray(alpha, dtype=dtype)
    rb = jnp.asarray(root_beta, dtype=dtype)
    terms = [jnp.ones_like(x)]
    if max_degree >= 1:
        terms.append((x - a[..., 0]) / rb[..., 1])
    for k in range(1, max_degree):
        terms.append(((x - a[..., k]) * terms[k] - rb[..., k] * terms[k - 1]) / rb[..., k + 1])
    return jnp.stack(terms, axis=-1)


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


def _family_at(input_types: tuple[str | TruncatedFamily, ...], d: int) -> TruncatedFamily:
    """Return the truncated-normal recurrence of dimension ``d``."""
    family = input_types[d]
    assert isinstance(family, TruncatedFamily)
    return family


def _basis_table(
    X: Array, input_types: tuple[str | TruncatedFamily, ...], max_degree: int
) -> tuple[Array, np.ndarray]:
    """Tabulate every 1-D orthonormal basis function at every sample.

    Each family (Legendre for ``"uniform"``, Hermite for ``"gaussian"``, a
    truncated-normal recurrence for a :class:`TruncatedFamily`) runs once over
    all of its columns, so the traced graph does not grow with the number of
    dimensions. The table lists the uniform columns first, then the Gaussian
    ones, then the truncated ones, so no permutation back to column order is
    needed: ``position`` says where each dimension went, and callers index
    through it.

    Args:
        X: Reference-domain inputs, shape ``(N, D)``.
        input_types: Per-dimension ``"uniform"`` or ``"gaussian"`` tags, or a
            :class:`TruncatedFamily`.
        max_degree: Maximum 1-D polynomial degree.

    Returns:
        ``(table, position)``. ``table`` has shape ``(N, D, max_degree + 1)``
        and ``table[n, position[d], k] = phi_k(X[n, d])``. ``position`` is an
        integer array of length ``D``.
    """
    uniform = [
        d for d, kind in enumerate(input_types) if isinstance(kind, str) and kind == "uniform"
    ]
    gaussian = [
        d for d, kind in enumerate(input_types) if isinstance(kind, str) and kind != "uniform"
    ]
    truncated = [d for d, kind in enumerate(input_types) if isinstance(kind, TruncatedFamily)]
    parts: list[Array] = []
    for family, basis in ((uniform, _legendre_1d), (gaussian, _hermite_1d)):
        if family:
            # Applied to the whole (N, G) block at once: both recurrences are
            # elementwise over leading axes. `vmap` over columns would give
            # the same values eagerly, but XLA fuses it differently under
            # `jit`, and the fused float32 recurrence then differs in the last
            # bit from the per-column program this table replaces.
            parts.append(basis(X[:, jnp.asarray(family)], max_degree))
    if truncated:
        alpha = np.stack([np.asarray(_family_at(input_types, d).alpha) for d in truncated])
        root_beta = np.stack([np.asarray(_family_at(input_types, d).root_beta) for d in truncated])
        parts.append(_truncated_1d(X[:, jnp.asarray(truncated)], alpha, root_beta, max_degree))
    position = np.argsort(np.asarray(uniform + gaussian + truncated))
    table = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1)
    return table, position


def _design_matrix_dense(
    X: Array,
    multi_index: np.ndarray,
    input_types: tuple[str | TruncatedFamily, ...],
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
        kind = input_types[d]
        if isinstance(kind, TruncatedFamily):
            basis_1d.append(
                _truncated_1d(
                    X[:, d], np.asarray(kind.alpha), np.asarray(kind.root_beta), max_degree
                )
            )
        elif kind == "uniform":
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
    input_types: tuple[str | TruncatedFamily, ...],
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
    input_types: tuple[str | TruncatedFamily, ...],
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
    S2 = jnp.full((*c2.shape[:-1], D, D), jnp.nan, dtype=s2_upper.dtype)
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


# Default Tikhonov parameter in float32, absolute. See resolve_ridge.
DEFAULT_RIDGE_F32 = 1e-8

# The float64 default ridge, relative to the mean diagonal of the Gram matrix,
# as a smooth function of the smallest pivot share ``t`` of the design without
# the ridge (see default_ridge_rel):
#
#   t >= RIDGE_BAND_HIGH:  eps. Every real design measured sits here: order-20
#       Hermite 2.4e-15, an input copying another to 1e-3 1.6e-14, inputs on
#       [0, 0.05] at order 4 7e-13, on [0, 0.3] at order 6 9e-8.
#   t <= RIDGE_BAND_LOW:   max(SINGULAR_RIDGE_REL, RIDGE_SHARE_SLOPE * sqrt(t)).
#       A design singular to round-off (two- to five-level inputs: t <= 1e-31)
#       gets 1e-12, the minimum-norm fit. A design a tiny jitter away from
#       singular gets a ridge that grows with the jitter.
#   between:               log-linear in log(t).
#
# Why sqrt(t). Jitter X of a design on 4 or 5 levels by +-j: the singular
# direction gets t ~ (5 j)**2, and the noise along it enters the coefficients
# as sigma * s / lam. The indices stay put (to 5e-3) when the relative ridge is
# at least 100 j, that is about 20 sqrt(t); a fixed ridge lets them jump from
# 0.37 to 0.8 as j passes it. The slope 30 keeps them within 1e-3 up to
# j = 1e-11 (measured, three inputs, order 6, N = 400, noise 0.1).
SINGULAR_RIDGE_REL = 1e-12
RIDGE_SHARE_SLOPE = 30.0
RIDGE_BAND_LOW = 1e-20
RIDGE_BAND_HIGH = 1e-16

# Floor of the float64 ridge the leave-one-out error uses (see loo_fit).
# Measured on two interpolating fits (10 rows, 10 terms): 1e-10 reproduces the
# brute-force leave-one-out error, 1e-11 gives 89% of it and eps gives 1e-4.
LOO_RIDGE_REL = 1e-10


def unridged_share(R: Array) -> Array:
    """Smallest pivot share of a design, read off its QR factor without a ridge.

    As :func:`r_pivot_share`, but a zero column counts as share 0 instead of
    NaN.

    Args:
        R: Triangular factor, ``(n_terms, n_terms)``.

    Returns:
        A scalar in ``[0, 1]``.
    """
    col_sq = jnp.sum(R * R, axis=0)
    ok = col_sq > 0
    return jnp.min(jnp.where(ok, jnp.diagonal(R) ** 2 / jnp.where(ok, col_sq, 1.0), 0.0))


def default_ridge_rel(share: Array) -> Array:
    """The float64 default ridge over the mean Gram diagonal, from the share.

    Continuous in ``share``, so a round-off change of ``X`` cannot switch the
    fit between two ridges. See :data:`SINGULAR_RIDGE_REL` for the three
    pieces and the measurements behind them.

    Args:
        share: Smallest pivot share without the ridge (:func:`unridged_share`).

    Returns:
        The relative ridge, a scalar.
    """
    eps = jnp.finfo(share.dtype).eps
    t = jnp.clip(share, 1e-300, 1.0)
    low = jnp.maximum(SINGULAR_RIDGE_REL, RIDGE_SHARE_SLOPE * jnp.sqrt(t))
    at_low = max(SINGULAR_RIDGE_REL, RIDGE_SHARE_SLOPE * RIDGE_BAND_LOW**0.5)
    frac = (jnp.log(t) - np.log(RIDGE_BAND_LOW)) / np.log(RIDGE_BAND_HIGH / RIDGE_BAND_LOW)
    band = jnp.exp(np.log(at_low) + jnp.clip(frac, 0.0, 1.0) * (jnp.log(eps) - np.log(at_low)))
    return jnp.where(t <= RIDGE_BAND_LOW, low, band)


def resolve_ridge(ridge: float | None, R: Array) -> tuple[Array, Array]:
    """The Tikhonov parameter a fit uses, from the caller's ``ridge``.

    An explicit ``ridge`` is used as given, as an absolute parameter. ``None``
    picks the default for the working dtype:

    - float32: ``1e-8``, absolute. float32 refuses a design whose small
      eigenvalues would meet it (:func:`jaxgsa._core.fit_quality.check_gram_rank`).
    - float64: :func:`default_ridge_rel` of the pivot share times the mean
      diagonal of ``Phi^T Phi``. For every real design that is machine
      epsilon, the round-off of the Gram matrix itself, so the ridge does not
      bias a direction the data resolves. A design that is singular to
      round-off (a two-level input at order 3) gets ``1e-12``: its null
      directions then carry the minimum-norm solution, not noise divided by
      round-off.

    The absolute ``1e-8`` was the float64 default before. When an input does
    not fill its declared bounds, the smallest Gram eigenvalue falls to about
    ``1e-9``, and that ridge shrank real directions. On the declared domain
    the indices moved by up to 0.5 with a perfect ``cv_rmse``. With the
    relative ridge the same fits are exact to 2e-4 (inputs on ``[0, 0.3]``,
    order 6) and 7e-5 (``[0.495, 0.505]``, order 3). On data that fills its
    bounds the indices move by about 2e-13.

    Args:
        ridge: The caller's ridge, or ``None`` for the default.
        R: Triangular QR factor of the design without the ridge rows, shape
            ``(n_terms, n_terms)``. Its column norms are those of ``Phi``.

    Returns:
        ``(lam, share)``: the Tikhonov parameter, a scalar in the dtype of
        ``R``, and the pivot share the rank check should read. For the
        float64 default that is :func:`unridged_share` of ``R``: the default
        ridge can be large next to a near-singular column and would otherwise
        hide it. Otherwise it is NaN, and the caller reads the share off the
        ridge-augmented factor, as before.
    """
    nan = jnp.asarray(jnp.nan, dtype=R.dtype)
    if ridge is not None:
        return jnp.asarray(ridge, dtype=R.dtype), nan
    if R.dtype != jnp.float64:
        return jnp.asarray(DEFAULT_RIDGE_F32, dtype=R.dtype), nan
    mean_diag = jnp.sum(R * R) / R.shape[1]
    share = unridged_share(R)
    return default_ridge_rel(share) * mean_diag, share


def fold_ridge(R: Array, z: Array, lam: Array) -> tuple[Array, Array]:
    """Add the ridge rows ``sqrt(lam) * I`` to a QR factorisation.

    ``R`` and ``z = Q^T Y`` factor a least-squares problem. A QR of the small
    ``(2 * n_terms, n_terms + M)`` stack ``[[R, z], [sqrt(lam) I, 0]]`` gives
    the factor and rotated right-hand side of the same problem with ``lam``
    more ridge, because an orthogonal map of the rows does not change a
    least-squares problem.

    Args:
        R: Upper-triangular factor, ``(n_terms, n_terms)``.
        z: Rotated right-hand side, ``(n_terms, M)``.
        lam: Ridge to add, at least 0.

    Returns:
        ``(R_new, z_new)`` with ``R_new^T R_new = R^T R + lam * I``.
    """
    n_terms = R.shape[1]
    lower = jnp.concatenate(
        [jnp.sqrt(lam).astype(R.dtype) * jnp.eye(n_terms, dtype=R.dtype), jnp.zeros_like(z)],
        axis=1,
    )
    top = jnp.concatenate([R, z], axis=1)
    full = jnp.linalg.qr(jnp.concatenate([top, lower], axis=0), mode="r")
    return full[:n_terms, :n_terms], full[:n_terms, n_terms:]


def qr_with_rhs(A: Array, B: Array) -> tuple[Array, Array]:
    """The triangular factor ``R`` of ``A`` and ``Q^T B``, without forming ``Q``.

    When ``B`` has few columns next to ``A``, one QR of ``[A B]`` in
    ``mode="r"`` gives both: ``R`` is its top-left block and ``Q^T B`` its
    top-right block. Skipping ``Q`` halves the time of the PCE fit (measured
    at N = 65536, 165 terms: 384 ms against 192 ms in float64). For a wide
    ``B`` the QR of ``[A B]`` costs more than forming ``Q``, so ``Q`` is
    formed then: the crossover is at ``M ~ 0.6 * n_terms``.

    Args:
        A: Matrix with at least as many rows as columns, ``(n, p)``.
        B: Right-hand sides, ``(n, M)``, in the dtype of ``A``.

    Returns:
        ``(R, z)`` with ``R`` of shape ``(p, p)`` and ``z = Q^T B`` of shape
        ``(p, M)``.
    """
    p = A.shape[1]
    if 2 * B.shape[1] <= p:
        full = jnp.linalg.qr(jnp.concatenate([A, B], axis=1), mode="r")
        return full[:p, :p], full[:p, p:]
    Q, R = jnp.linalg.qr(A, mode="reduced")
    return R, Q.T @ B


def ridge_qr(Phi: Array, Y: Array, ridge: float | None) -> tuple[Array, Array, Array, Array]:
    """Solve the ridge least-squares problem by QR.

    The coefficients minimise ``|Phi c - Y|^2 + lam * |c|^2``, the same
    Tikhonov problem as the normal equations ``(Phi^T Phi + lam*I) c =
    Phi^T Y``. They come from the QR factorisation of ``[Phi; sqrt(lam)*I]``
    instead (:func:`qr_with_rhs`). The normal equations square the condition
    number of ``Phi``. Float32 loses every digit of an order-12 fit that way,
    and QR keeps them. One factorisation serves every column of ``Y``.

    A fixed ridge (explicit, or the float32 default) goes in as extra rows
    before the QR. The float64 default is read off the factor of ``Phi``
    (:func:`resolve_ridge`), so it is added after, with :func:`fold_ridge`.

    ``R`` satisfies ``R^T R = Phi^T Phi + lam*I``, so ``R.T`` is a valid
    Cholesky factor for :func:`hat_diagonal`, and :func:`r_pivot_share`
    reads the rank diagnostic off it.

    Args:
        Phi: Design matrix, shape ``(N, n_terms)``.
        Y: Outputs, shape ``(N, M)``, one column per output slice, in the
            dtype of ``Phi``.
        ridge: Tikhonov parameter, at least 0, or ``None`` for the default of
            :func:`resolve_ridge`.

    Returns:
        ``(coefficients, R, lam, share)``: coefficients of shape
        ``(n_terms, M)``, the upper-triangular factor ``R`` of shape
        ``(n_terms, n_terms)``, the Tikhonov parameter used and the pivot
        share for the rank check (see :func:`resolve_ridge`).
    """
    n_terms = Phi.shape[1]
    if ridge is None and Phi.dtype == jnp.float64:
        R0, z0 = qr_with_rhs(Phi, Y)
        lam, share = resolve_ridge(None, R0)
        R, z = fold_ridge(R0, z0, lam)
    else:
        lam, _ = resolve_ridge(ridge, jnp.zeros((1, 1), Phi.dtype))
        # The ridge rows go below Phi, with zero right-hand side rows.
        R, z = qr_with_rhs(
            jnp.concatenate([Phi, jnp.sqrt(lam) * jnp.eye(n_terms, dtype=Phi.dtype)], axis=0),
            jnp.concatenate([Y, jnp.zeros((n_terms, Y.shape[1]), dtype=Y.dtype)], axis=0),
        )
        share = r_pivot_share(R)
    coefficients = solve_triangular(R, z, lower=False)
    return coefficients, R, lam, share


def loo_fit(R: Array, coefficients: Array, lam: Array) -> tuple[Array, Array, Array]:
    """The fallback fit for leave-one-out rows the fit interpolates.

    The shortcut ``e_i / (1 - h_i)`` needs both terms well above round-off.
    A row the fit interpolates (as many rows as terms, or a row alone in its
    part of the design) has a residual and ``1 - h_i`` of order ``lam``. With
    the tiny float64 default both are round-off, and the ratio is noise:
    ``cv_rmse`` read 4e-4 where the brute-force leave-one-out error is 5.3.
    :func:`loo_residuals` therefore takes such a row from this fit instead,
    whose ridge is at least :data:`LOO_RIDGE_REL` times the mean Gram
    diagonal (added with :func:`fold_ridge`). With it the shortcut equals the
    brute-force value on those fits.

    Such a row is told apart by its margin ``1 - h_i``: it is made by the
    ridge, so it doubles when the ridge doubles. A real margin, even a tiny
    one (4e-12 on an accurate order-20 fit), does not move. The third
    output, the factor with the ridge doubled, is for that test. Every other
    row keeps the exact shortcut of the fit itself.

    Args:
        R: Factor of the fitted ridge problem, ``(n_terms, n_terms)``.
        coefficients: Fitted coefficients, ``(n_terms, M)``.
        lam: Tikhonov parameter of that fit.

    Returns:
        ``(coefficients, R, R_double)``: the fallback fit, and the factor of
        the fit with ``2 * lam``.
    """
    mean_diag = jnp.sum(R * R) / R.shape[1] - lam
    extra = jnp.maximum(LOO_RIDGE_REL * mean_diag - lam, 0.0)
    R2, z2 = fold_ridge(R, R @ coefficients, extra)
    R_double, _ = fold_ridge(R, jnp.zeros((R.shape[0], 0), R.dtype), lam)
    return solve_triangular(R2, z2, lower=False), R2, R_double


def loo_residuals(
    Phi: Array,
    Y: Array,
    coefficients: Array,
    R: Array,
    fallback: tuple[Array, Array, Array] | None = None,
) -> tuple[Array, Array]:
    """Leave-one-out and in-sample residuals of a fit, row by row.

    Args:
        Phi: Design rows, ``(b, n_terms)``.
        Y: Outputs, ``(b, M)``.
        coefficients: Fitted coefficients, ``(n_terms, M)``.
        R: Factor of the fit, ``(n_terms, n_terms)``.
        fallback: The output of :func:`loo_fit`, or ``None``. When given, a
            row whose margin ``1 - h_i`` is made by the ridge (at least 1.5
            times larger with the ridge doubled, or below ``1e3 * eps``)
            takes its leave-one-out residual from the fallback fit.

    Returns:
        ``(loo, residuals)``, each ``(b, M)``.
    """
    residuals = Y - Phi @ coefficients
    h = hat_diagonal(Phi, R.T)
    loo = residuals / (1.0 - h)[:, None]
    if fallback is not None:
        c2, R2, R_double = fallback
        margin = 1.0 - _raw_leverage(Phi, R.T)
        # A margin at round-off (below 1e3 eps; real ones measured down to
        # 4e-12) counts as made too: there the doubling test is noise, and
        # the streamed factor misread one interpolated row by it.
        tiny = 1e3 * jnp.finfo(margin.dtype).eps
        made = (margin < tiny) | (margin < (1.0 - _raw_leverage(Phi, R_double.T)) / 1.5)
        loo2 = (Y - Phi @ c2) / (1.0 - hat_diagonal(Phi, R2.T))[:, None]
        loo = jnp.where(made[:, None], loo2, loo)
    return loo, residuals


def ridge_sensitivity(R: Array, coefficients: Array, lam: Array, multi_index: np.ndarray) -> Array:
    """How far the total indices move when the ridge is doubled.

    The fit with ``2 * lam`` comes from :func:`fold_ridge` on the small
    factor, with no pass over the rows. Where the ridge does not matter, the
    indices do not move: 1e-7 on an accurate order-20 Hermite fit, where the
    default ridge is 1.4e-3 of the smallest Gram eigenvalue. Where it sets a
    direction of the fit, they move by about as much as the ridge biased
    them: 0.25 with ``ridge=1e-8`` on inputs that fill a third of their
    bounds. The test reads the indices themselves, so a ridge that is large
    next to a direction the indices do not depend on passes.

    Args:
        R: Factor of the ridge-augmented design, ``(n_terms, n_terms)``.
        coefficients: Fitted coefficients, ``(T, K, n_terms)``.
        lam: Tikhonov parameter used for ``R``.
        multi_index: Multi-index of the expansion, ``(n_terms, D)``.

    Returns:
        ``max |ST(2 lam) - ST(lam)|`` over inputs and output slices, a scalar.
        Constant slices are skipped.
    """
    n_terms = coefficients.shape[-1]
    flat = coefficients.reshape(-1, n_terms).T  # (n_terms, T*K)
    R2, z2 = fold_ridge(R, R @ flat, lam)
    doubled = solve_triangular(R2, z2, lower=False).T.reshape(coefficients.shape)
    active = jnp.asarray(np.asarray(multi_index) > 0, dtype=coefficients.dtype)

    def total_index(c: Array) -> Array:
        # As in sobol_from_coefficients, ST only: S2 is not needed here.
        c2 = c**2
        return (c2 @ active) / jnp.sum(c2[..., 1:], axis=-1, keepdims=True)

    diff = jnp.abs(total_index(doubled) - total_index(coefficients))
    return jnp.max(jnp.where(jnp.isfinite(diff), diff, 0.0))


@jax.jit
def r_pivot_share(R: Array) -> Array:
    """The smallest unexplained share of any column, read off a QR factor.

    For column ``j`` of the augmented design ``A``, ``R[j, j]**2`` is the part
    of ``|A_j|**2`` that the columns before it do not explain, and
    ``|A_j|**2`` is the squared norm of column ``j`` of ``R``. The ratio is
    the squared Cholesky pivot of the Gram matrix over its diagonal, but it
    never goes through a Cholesky factorisation, so it stays finite in
    float32.

    Args:
        R: Upper-triangular factor, shape ``(n_terms, n_terms)``.

    Returns:
        ``min_j R[j, j]**2 / sum_i R[i, j]**2``, a scalar in ``[0, 1]``. NaN
        when a column of ``R`` is all zeros.
    """
    return jnp.min(jnp.diagonal(R) ** 2 / jnp.sum(R**2, axis=0))


def hat_diagonal(Phi: Array, gram_chol: Array) -> Array:
    """Hat-matrix diagonal (leverage) of a design block, from a Cholesky factor.

    This is the single definition of PCE leverage. Both the single-pass fit
    and the row-streamed fit call it, so the two paths cannot drift apart.

    With ``Phi^T Phi + ridge*I = L L^T``, the leverage of row ``i`` is

        ``h_i = phi_i (L L^T)^{-1} phi_i^T = || L^{-1} phi_i ||^2``

    so the inverse of the small triangular factor gives every ``h_i``. The
    Gram matrix itself is never inverted. That matters here: the default
    ridge is tiny and PCE Gram matrices lose conditioning quickly as the
    polynomial order rises, so an inverse of the Gram matrix would square an
    already large condition number, and a degraded leverage feeds straight
    back into automatic order selection. The triangular factor keeps the
    error at the level of the factorization itself, and ``h_i`` comes out as
    a sum of squares, hence non-negative by construction.

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
        ``1 - 1e-13``, but in float32 that rounds back to exactly 1.0 (the
        spacing below 1.0 is about ``6e-8``), which made the clip a silent
        no-op — the same dtype trap :func:`jaxgsa._core.precision.unit_clip_bounds`
        exists for, so it supplies the bound.
    """
    # h_i = |L^{-1} phi_i|^2 = |phi_i L^{-T}|^2. Invert the small triangular
    # factor once, then one (b, n_terms) matmul and a last-axis sum. Solving
    # into a (n_terms, b) array and summing over its leading axis was 3.5x
    # slower and held a transposed copy of Phi.
    leverage = _raw_leverage(Phi, gram_chol)
    _, high = unit_clip_bounds(_LEVERAGE_CLIP, leverage.dtype)
    return jnp.clip(leverage, 0.0, high)


def _raw_leverage(Phi: Array, gram_chol: Array) -> Array:
    """The leverage of :func:`hat_diagonal` before the clip."""
    eye = jnp.eye(gram_chol.shape[0], dtype=gram_chol.dtype)
    L_inv = solve_triangular(gram_chol, eye, lower=True)
    Z = Phi @ L_inv.T
    return jnp.sum(Z * Z, axis=1)


def loo_error(
    Phi: Array,
    Y: Array,
    coefficients: Array,
    ridge: float = 0.0,
    gram_chol: Array | None = None,
    n_rows: Array | int | None = None,
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
        n_rows: Number of real rows, when ``Phi`` and ``Y`` carry zero rows
            after them (a padded fit). The sum of squared leave-one-out
            residuals divides by this count. A zero row has leverage 0 and
            residual 0, so it adds nothing to the sum. ``None`` means every
            row is real, and the mean runs over all of them.

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
    if n_rows is None:
        return jnp.sqrt(jnp.mean(loo_residuals**2, axis=0))
    return jnp.sqrt(jnp.sum(loo_residuals**2, axis=0) / n_rows)
