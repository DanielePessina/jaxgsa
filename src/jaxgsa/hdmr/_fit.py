"""The RS-HDMR fit: one path, chunked on rows and on output slices.

There used to be two fits here. An "in-memory" one built the full-``N``
B-spline bases once and ``vmap``ped a per-slice kernel over chunks of output
slices; a "streamed" one accumulated the same normal equations over row
batches and kept every lane in flight. They bounded different axes, so each
honoured one of HDMR's two batching keywords and silently dropped the other.

This module is the single fit, and it chunks both axes. The sample axis is
cut into row batches of ``batch_size`` rows, and the output-slice axis into
chunks of ``slice_chunk_size`` lanes. Row batches are the outer loop, so the
B-spline bases of a batch are built once and reused by every slice chunk. The
old in-memory path is now the degenerate corner ``batch_size = N`` and
``slice_chunk_size = T*K``, which is what the automatic sizes resolve to
whenever the whole fit fits the memory budget.

The fit never needs a full-``N`` basis, because every quantity the estimator
wants is a per-row reduction:

- The first-order backfit only touches the data through the cross-Gram
  ``G1[j, k] = B1_j^T B1_k`` and the moment vector ``b1_j = B1_j^T Y``. Row
  batches accumulate both, and the backfit iteration then runs in coefficient
  space (see :func:`_get_gram_backfit`).
- The higher-order single-shot solves need only the per-term Gram blocks
  ``B_j^T B_j`` and moments ``B_j^T Y_res``, where the sequential residual
  ``Y_res`` is recomputed exactly per batch from the already-solved
  lower-order coefficients.
- The ANCOVA covariances, the F-test sums of squares, and the RMSE are all
  sums over rows of per-row products of the term contributions
  ``f_j = B_j C_j``, accumulated in a final pass.

Both loops are padded: the trailing row batch is padded back to ``batch_size``
with masked-out rows, and the trailing slice chunk to ``slice_chunk_size``
with constant lanes that are sliced off at the end. A jitted step therefore
compiles once per pass rather than twice. Padding is exact in real arithmetic
-- a masked row contributes a hard zero to every accumulator, and a padded
lane is an independent lane -- so it cannot move a number by more than the
reassociation that adding a zero to a float32 reduction can cause.
"""

import functools
import math
from collections.abc import Callable
from functools import lru_cache
from typing import Any, NamedTuple

import jax
import jax.core
import jax.numpy as jnp
from jax import Array

from jaxgsa._core.batching import get_memory_budget, pad_rows, resolve_batch_size
from jaxgsa.hdmr._engine import _build_B1, _build_B2, _build_B3, term_order_map

# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------

# Terms per block in the final statistics pass. Larger blocks mean fewer loop
# trips; smaller blocks mean smaller (row, lane, block) temporaries.
_STATS_TERM_BLOCK = 32


def _basis_bytes_per_row(
    D: int,
    m1: int,
    m2: int,
    m3: int,
    n2: int,
    n3: int,
    itemsize: int,
) -> int:
    """Transient bytes one row costs in the basis build, per row batch.

    This part is shared by every output slice: each batch materializes ``B1``
    (m1 * D per row), ``B2`` with its two gather temporaries (3 * m2 * n2),
    and ``B3`` with its three gather temporaries (4 * m3 * n3), plus the
    projected pair basis and its temporary (2 * m2 * n2).

    Args:
        D: Number of input parameters.
        m1: First-order basis size.
        m2: Second-order basis size.
        m3: Third-order basis size.
        n2: Number of second-order terms.
        n3: Number of third-order terms.
        itemsize: Bytes per element of the working dtype.

    Returns:
        Estimated basis-build bytes per streamed row.
    """
    # The extra 2 * m2 * n2 is the projected pair basis of the stats pass.
    return itemsize * (m1 * D + 5 * m2 * n2 + 4 * m3 * n3)


def _slice_bytes_per_row(n: int, itemsize: int) -> int:
    """Transient bytes one (row, output slice) pair costs in the final pass.

    The statistics pass walks the terms in blocks of at most
    ``_STATS_TERM_BLOCK``. Each block materializes the contributions ``F`` plus
    two elementwise temporaries of the same shape, so 3 * block elements per
    row per lane, plus a few ``(row, lane)`` arrays (the order sums and the
    residuals). The estimate does not grow with the number of terms.

    Args:
        n: Total number of terms.
        itemsize: Bytes per element of the working dtype.

    Returns:
        Estimated bytes per (row, slice) pair.
    """
    return itemsize * (3 * min(n, _STATS_TERM_BLOCK) + 8)


def _fit_bytes_per_row(
    D: int,
    m1: int,
    m2: int,
    m3: int,
    n2: int,
    n3: int,
    n: int,
    L: int,
    itemsize: int,
) -> int:
    """Transient bytes one row costs when every output slice is in flight.

    The sum of the basis build and all ``L`` lanes of the final pass. It is
    what the automatic ``batch_size`` is derived from, so a fit that fits the
    budget resolves to a single batch.

    Args:
        D: Number of input parameters.
        m1: First-order basis size.
        m2: Second-order basis size.
        m3: Third-order basis size.
        n2: Number of second-order terms.
        n3: Number of third-order terms.
        n: Total number of terms.
        L: Number of flattened output slices (T*K).
        itemsize: Bytes per element of the working dtype.

    Returns:
        Estimated transient bytes per row.
    """
    return _basis_bytes_per_row(D, m1, m2, m3, n2, n3, itemsize) + L * _slice_bytes_per_row(
        n, itemsize
    )


def _resolve_slice_chunk_size(
    slice_chunk_size: int | None,
    L: int,
    b: int,
    slice_bytes: int,
    lane_fixed_bytes: int = 0,
) -> int:
    """Resolve how many output slices one vmap handles, against the budget.

    Args:
        slice_chunk_size: The caller's value, or ``None`` to derive one.
        L: Number of flattened output slices.
        b: Resolved rows per batch.
        slice_bytes: Bytes per (row, slice) pair, from
            :func:`_slice_bytes_per_row`.
        lane_fixed_bytes: Bytes one lane holds regardless of the row count,
            such as the pair solver state.

    Returns:
        A chunk width in ``[1, L]``. With an automatic ``b`` this is ``L``,
        because ``b`` was already chosen so that all ``L`` lanes fit; it only
        bites when the caller forces a batch size the budget would not have
        picked.

    Raises:
        ValueError: If an explicit ``slice_chunk_size`` is below 1.
    """
    if slice_chunk_size is not None:
        if slice_chunk_size < 1:
            raise ValueError(f"jaxgsa.hdmr: slice_chunk_size must be >= 1, got {slice_chunk_size}")
        return min(slice_chunk_size, L)
    budget = get_memory_budget()
    auto = budget // max(b * slice_bytes, 1)
    if lane_fixed_bytes > 0:
        auto = min(auto, budget // lane_fixed_bytes)
    return max(1, min(L, auto))


# ---------------------------------------------------------------------------
# Gram-space solvers
# ---------------------------------------------------------------------------


def _ridge_eye(gram: Array, ridge: float, m: int) -> Array:
    """Return the Tikhonov term ``ridge * mean(diag(gram)) * I`` per block.

    The ridge is relative to the mean Gram diagonal of each block. The raw
    Gram scales with ``N`` and with the basis scale ``m**3``, so an absolute
    ``ridge`` would do nothing at large ``N`` and a different amount at each
    ``n_intervals``. Relative to the diagonal, ``ridge=0.01`` always means
    "shrink directions that carry less than 1% of a typical column's energy".

    Args:
        gram: Gram blocks, shape ``(..., m, m)``.
        ridge: Relative ridge strength.
        m: Block size.

    Returns:
        Array of shape ``(..., m, m)``.
    """
    return ridge * _ridge_scale(gram)[..., None, None] * jnp.eye(m, dtype=gram.dtype)


def _ridge_scale(gram: Array) -> Array:
    """Return the mean Gram diagonal per block, the scale of the relative ridge.

    Args:
        gram: Gram blocks, shape ``(..., m, m)``.

    Returns:
        Array of shape ``(...)``.
    """
    return jnp.mean(jnp.diagonal(gram, axis1=-2, axis2=-1), axis=-1)


def _live_pairs(BtB: Array, ridge: float | Array) -> tuple[Array, Array]:
    """Return the per-term ridge and which terms are live.

    A pair whose projected Gram is near zero (an input with almost no spread)
    holds nothing beyond the first order. Its ridge is then tiny and the
    round-off left by the projection makes its block near-singular or
    indefinite, which turns the coefficients of every lane into huge values or
    NaN. Every pair solver drops such a pair alike, with zero coefficients.

    Args:
        BtB: Projected per-term Gram blocks, ``(n2, m2, m2)``.
        ridge: Relative ridge, as in :func:`_ridge_eye`.

    Returns:
        The per-term ridge ``ridge * mean(diag)``, shape ``(n2,)``, and a
        boolean mask of the live terms, shape ``(n2,)``.
    """
    lam = ridge * _ridge_scale(BtB)
    live = lam > jnp.sqrt(jnp.finfo(lam.dtype).eps) * jnp.max(lam)
    return lam, live


def _eigen_cut(G: Array) -> tuple[Array, Array, Array]:
    """Eigen-decompose a symmetric Gram and flag the directions to keep.

    Eigenvalues below ``sqrt(eps)`` of the largest are dropped: the
    first-order bases of different inputs share the constant function, so the
    Gram has a null space.

    Args:
        G: Gram matrix, shape ``(P, P)``.

    Returns:
        The eigenvalues ``w``, the eigenvectors ``V`` and the boolean mask
        ``keep``, shapes ``(P,)``, ``(P, P)`` and ``(P,)``.
    """
    w, V = jnp.linalg.eigh((G + G.T) / 2)
    keep = w > jnp.sqrt(jnp.finfo(G.dtype).eps) * jnp.max(w)
    return w, V, keep


@lru_cache(maxsize=32)
def _get_gram_backfit(m1: int, n1: int, max_iter: int, ridge: float):
    """Build and cache the vmapped Gram-space first-order backfit.

    Coordinate-descent backfitting (Breiman & Friedman, 1985) written in
    coefficient space. With ``M_j = G1[j, j] + ridge * I``, a backfit step
    only needs ``B1_j^T`` of the current partial residual, which the Gram
    blocks already hold:

    ``B1_j^T Y_r = b1res_j - sum_k G1[j, k] C1_k + G1[j, j] C1_j``.

    Tikhonov regularization keeps an ill-conditioned ``B^T B`` from blowing up
    the coefficients when the basis functions are nearly collinear or the
    sample count is low. The ridge is relative to the mean Gram diagonal (see
    :func:`_ridge_eye`). Iteration stops when the largest change in any
    dimension's coefficient energy falls below one part in a thousand of the
    largest energy itself, or after ``max_iter`` sweeps. The test is relative,
    not absolute: the coefficients scale with ``Y`` and with ``m**-3``, so an
    absolute cutoff stops the sweep after one pass whenever ``Y`` is small or
    ``m`` is large (H4).

    Returns:
        A jitted function ``(G1, G1_diag, b1res) -> C1``. ``G1``
        (n1, m1, n1, m1) and ``G1_diag`` (n1, m1, m1) are shared across lanes,
        and ``b1res`` is batched as (L, n1, m1). The result is (L, m1, n1).
    """

    def backfit(G1: Array, G1_diag: Array, b1res: Array) -> Array:
        M = G1_diag + _ridge_eye(G1_diag, ridge, m1)  # (n1, m1, m1)

        # Initial fit: C1_j = M_j^{-1} B1_j^T Y_res, one solve per dimension.
        # Backfitting below corrects for inter-dimension coupling.
        C1 = jax.vmap(jnp.linalg.solve)(M, b1res).T  # (m1, n1)
        var_old = jnp.sum(jnp.square(C1), axis=0)  # (n1,)

        def _update_j(C1: Array, j: int | Array) -> tuple[Array, None]:
            # B1_j^T of the partial residual, assembled from Gram blocks.
            all_contrib = jnp.einsum("mkn,nk->m", G1[j], C1)  # (m1,)
            j_contrib = G1_diag[j] @ C1[:, j]  # (m1,)
            rhs = b1res[j] - all_contrib + j_contrib
            C1 = C1.at[:, j].set(jnp.linalg.solve(M[j], rhs))
            return C1, None

        def _backfit_cond(state: tuple) -> Array:
            _, var, varmax, it = state
            # Relative to the current coefficient energy, not an absolute
            # threshold: the coefficients scale with Y and with m**-3 (the
            # basis is scaled by m**3), so an absolute cutoff stops a
            # correlated fit after one sweep whenever Y is small or m is
            # large, well before the coupling terms have converged.
            return (varmax > 1e-3 * jnp.max(var)) & (it < max_iter)

        def _backfit_body(state: tuple) -> tuple:
            C1, var_old, _, it = state
            C1, _ = jax.lax.scan(_update_j, C1, jnp.arange(n1))
            var_new = jnp.sum(jnp.square(C1), axis=0)
            varmax = jnp.max(jnp.abs(var_new - var_old))
            return C1, var_new, varmax, it + 1

        init_state = (C1, var_old, jnp.array(jnp.inf), jnp.array(0))
        C1, _, _, _ = jax.lax.while_loop(_backfit_cond, _backfit_body, init_state)
        return C1

    return jax.jit(jax.vmap(backfit, in_axes=(None, None, 0)))


@lru_cache(maxsize=32)
def _get_gram_higher_solver(m_basis: int, ridge: float):
    """Build and cache the vmapped Gram-space higher-order solver.

    Each term is solved independently,
    ``C_j = (G_j + ridge * mean(diag G_j) * I)^{-1} m_j``, from its Gram block
    ``G_j`` and moment ``m_j``. The caller passes blocks that are already
    orthogonal to every lower-order term (see :func:`_project_pairs`), so
    same-order terms do not compete for the same leftover.

    Returns:
        A jitted function ``(BtB, BtY) -> C``. ``BtB``
        (n_terms, m_basis, m_basis) is shared across lanes, and ``BtY`` is
        batched as (L, n_terms, m_basis). The result is
        (L, m_basis, n_terms).
    """

    def solve_lane(BtB: Array, BtY: Array) -> Array:
        lam = _ridge_eye(BtB, ridge, m_basis)
        C = jax.vmap(jnp.linalg.solve)(BtB + lam, BtY)
        return C.T  # (m_basis, n_terms)

    return jax.jit(jax.vmap(solve_lane, in_axes=(None, 0)))


@jax.jit
def _first_order_pinv(G1: Array) -> Array:
    """Pseudo-inverse of the flat first-order Gram matrix.

    The first-order bases of different inputs share the constant function (each
    sums to one), so the Gram has a null space. Eigenvalues below
    ``sqrt(eps)`` of the largest are dropped.

    Args:
        G1: Flat Gram, shape ``(P, P)`` with ``P = m1 * n1``.

    Returns:
        The pseudo-inverse, shape ``(P, P)``.
    """
    w, V, keep = _eigen_cut(G1)
    inv = jnp.where(keep, 1.0 / jnp.where(keep, w, 1.0), 0.0)
    return (V * inv[None, :]) @ V.T


def _pair_normal_pass(
    coefs: list[Array],
    A: Array,
    b: int,
    n_row_batches: int,
    X_p: Array,
    w_full: Array,
    build: Callable[..., list[Array]],
    c2_idx: Array,
    c3_idx: Array,
    beta2: Array,
    beta3: Array,
) -> list[Array]:
    """Apply the projected pair normal matrix to every lane chunk.

    Args:
        coefs: Pair coefficients per lane chunk, each ``(cs, n2, m2)``.
        A: Projection of B2 onto the first-order space, ``(P, m2, n2)``.
        b: Rows per batch.
        n_row_batches: Number of row batches.
        X_p: Padded inputs.
        w_full: Row mask.
        build: Basis builder for orders up to 2.
        c2_idx: Pair index table.
        c3_idx: Triple index table.
        beta2: Pair basis index table.
        beta3: Triple basis index table.

    Returns:
        ``Bt^T Bt c`` for each lane chunk, without the ridge term.

    The row batches run in a ``fori_loop``, so only one batch of bases is
    alive at a time. Unrolled in Python inside the CG loop, XLA was free to
    keep every batch's bases alive at once: at D = 64, N = 4096 the float32
    fit went above 3 GiB, and 1.2 GiB with the loop.
    """
    vs = [jnp.einsum("pmj,ljm->lp", A, c) for c in coefs]
    accs = [(jnp.zeros_like(v), jnp.zeros_like(c)) for v, c in zip(vs, coefs, strict=True)]

    def body(i: Array, accs: list[tuple[Array, Array]]) -> list[tuple[Array, Array]]:
        X_b = jax.lax.dynamic_slice_in_dim(X_p, i * b, b)
        w_b = jax.lax.dynamic_slice_in_dim(w_full, i * b, b)
        B1_b, B2_b = build(X_b, w_b, c2_idx, c3_idx, beta2, beta3)[:2]
        return [
            _acc_pair_normal(B1_b, B2_b, c, v, acc)
            for c, v, acc in zip(coefs, vs, accs, strict=True)
        ]

    accs = jax.lax.fori_loop(0, n_row_batches, body, accs)
    return [g2 - jnp.einsum("pmj,lp->ljm", A, g1) for g1, g2 in accs]


def _pair_normal_cached(coefs: list[Array], A: Array, B1: Array, B2: Array) -> list[Array]:
    """Apply the projected pair normal matrix with the bases of all rows held.

    The arithmetic is that of :func:`_pair_normal_pass` with one row batch,
    ``B2 c - B1 (A c)``. Holding the projected basis ``B2 - B1 A`` instead
    was measured to be worse in float32: at D = 8, N = 1024 with one input on
    (0.5, 0.501) the pair CG stalled at relative residual 0.07 and S2 of the
    true pair fell from 0.036 to 0.026.

    Args:
        coefs: Pair coefficients per lane chunk, each ``(cs, n2, m2)``.
        A: Projection of B2 onto the first-order space, ``(P, m2, n2)``.
        B1: First-order basis of all padded rows, ``(N_pad, m1, n1)``.
        B2: Pair basis of all padded rows, ``(N_pad, m2, n2)``.

    Returns:
        ``Bt^T Bt c`` for each lane chunk, without the ridge term.
    """
    out = []
    for c in coefs:
        v = jnp.einsum("pmj,ljm->lp", A, c)
        g1, g2 = _acc_pair_normal(B1, B2, c, v, (jnp.zeros_like(v), jnp.zeros_like(c)))
        out.append(g2 - jnp.einsum("pmj,lp->ljm", A, g1))
    return out


def _pair_bases(
    rows: list[tuple[int, int]],
    X_p: Array,
    w_full: Array,
    build: Callable[..., list[Array]],
    basis_idx: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    """Build the first-order and pair bases of every padded row.

    Args:
        rows: Row batch bounds.
        X_p: Padded inputs.
        w_full: Row mask.
        build: Basis builder for orders up to 2.
        basis_idx: ``(c2_idx, c3_idx, beta2, beta3)``.

    Returns:
        ``B1``, ``(N_pad, m1, n1)``, and ``B2``, ``(N_pad, m2, n2)``, zero on
        padded rows.
    """
    blocks = [build(X_p[lo:hi], w_full[lo:hi], *basis_idx)[:2] for lo, hi in rows]
    return jnp.concatenate([b[0] for b in blocks]), jnp.concatenate([b[1] for b in blocks])


_PAIR_ITER_FACTOR = 10
_PAIR_STALL = 25
_PAIR_WARN_REL = 1e-2


def _pair_tol(eps: float) -> float:
    """Return the relative residual the joint pair CG is held to.

    A looser value (1e-4) was measured to stop early: at D = 15, N = 1024 the
    test NRMSE was 0.14 against 0.05 at 1e-7. In float32 CG cannot reach
    ``10 eps`` (1.2e-6) and runs to the step cap, so the floor there is
    ``100 eps`` (1.2e-5).
    """
    return max(1e-7, 100 * eps) if eps > 1e-10 else 1e-7


def _joint_pair_solve(
    BtB: Array,
    rhs: list[Array],
    *,
    ridge: float,
    max_iter: int,
    sweep: Callable[[list[Array]], list[Array]],
) -> tuple[list[Array], Array, float]:
    """Solve all pair terms jointly by block-Jacobi preconditioned CG.

    Fitting each pair alone against one residual counts the same leftover
    once per pair, so the pairs add up to far more than the residual. The
    joint ridge solution ``(Bt^T Bt + lam) c = Bt^T r`` has no such
    overshoot. It is solved without forming the ``n2 m2`` square matrix: each
    CG step is one pass over the rows (:func:`_pair_normal_pass`) or two
    products with the cached projected basis. The per-term solve is the
    preconditioner and the start.

    Args:
        BtB: Projected per-term Gram blocks, ``(n2, m2, m2)``.
        rhs: ``Bt^T r`` per lane chunk, each ``(cs, n2, m2)``.
        ridge: Relative ridge, as in :func:`_ridge_eye`.
        max_iter: Cap on CG steps.
        sweep: Applies the unregularized normal matrix to a list of chunks.

    Returns:
        The pair coefficients per lane chunk, each ``(cs, m2, n2)``, the
        worst relative residual over all lanes, and the tolerance it was
        held to. Each lane returns its best iterate, by recurrence residual.

    In float32 this solve relies on stopping early. The projected system holds
    round-off in near-null directions (an input that does not fill its
    bounds); a solve that converges further fits them. Measured at D = 8,
    N = 1024 with one input on (0.5, 0.8): spurious S2 up to 0.0067 with a
    factored float32 solve, 0.0019 with an SPD-clipped preconditioner, and
    0.0002 with this CG.
    """
    m_basis = BtB.shape[1]
    lam = _ridge_eye(BtB, ridge, m_basis)  # (n2, m2, m2)
    lam_diag = jnp.diagonal(lam, axis1=-2, axis2=-1)  # (n2, m2)
    # A dead pair is dropped as in the direct solves: its preconditioner block
    # is zero, so it starts at zero and its search direction stays zero, and
    # its residual is masked out.
    _, live = _live_pairs(BtB, ridge)
    eye = jnp.eye(m_basis, dtype=BtB.dtype)
    Minv = jnp.where(
        live[:, None, None], jnp.linalg.inv(jnp.where(live[:, None, None], BtB + lam, eye)), 0.0
    )
    live_l = live[None, :, None]

    def precond(r: Array) -> Array:
        return jnp.einsum("jmn,ljn->ljm", Minv, r)

    def dot(a: list[Array], b: list[Array]) -> list[Array]:
        """Per-lane inner products, one ``(cs,)`` array per lane chunk."""
        return [jnp.sum(x * y, axis=(1, 2)) for x, y in zip(a, b, strict=True)]

    def safe(d: Array) -> Array:
        return jnp.where(d == 0, 1.0, d)

    solver = _get_gram_higher_solver(m_basis, ridge)
    rhs = [jnp.where(live_l, rr, 0.0) for rr in rhs]
    # Start from the independent per-term solution, stored as (cs, n2, m2).
    # A dead block is swapped for the identity, so no near-singular solve runs.
    BtB_live = jnp.where(live[:, None, None], BtB, eye)
    x = [solver(BtB_live, r).transpose(0, 2, 1) for r in rhs]
    Mx = sweep(x)
    r = [
        jnp.where(live_l, b - (m + lam_diag[None] * xx), 0.0)
        for b, m, xx in zip(rhs, Mx, x, strict=True)
    ]
    z = [precond(rr) for rr in r]
    # A zero right-hand side (a constant or padded lane) is already solved.
    bnorm = [safe(jnp.sqrt(d)) for d in dot(rhs, rhs)]
    eps = float(jnp.finfo(bnorm[0].dtype).eps)
    tol = _pair_tol(eps)
    # Only float32 stalls; in float64 a slow stretch is followed by progress.
    stall = _PAIR_STALL if eps > 1e-10 else max_iter + 1

    def rel_res(r: list[Array]) -> Array:
        return jnp.max(
            jnp.stack([jnp.max(jnp.sqrt(d) / b) for d, b in zip(dot(r, r), bnorm, strict=True)])
        )

    def lane_rel(r: list[Array]) -> list[Array]:
        return [jnp.sqrt(d) / b for d, b in zip(dot(r, r), bnorm, strict=True)]

    def cond(state: tuple) -> Array:
        _, r, _, _, k, _, k_best, _, _ = state
        # Stop at the tolerance, at the step cap, or when the residual has not
        # improved for _PAIR_STALL steps (float32 stalls above the tolerance).
        return (rel_res(r) >= tol) & (k < max_iter) & (k - k_best < stall)

    def body(state: tuple) -> tuple:
        x, r, p, rz, k, best, k_best, x_best, lane_best = state
        Mp = sweep(p)
        Ap = [m + lam_diag[None] * pp for m, pp in zip(Mp, p, strict=True)]
        pAp = dot(p, Ap)
        # A lane that has already converged is frozen: its residual is noise,
        # and CG on noise can return a huge step.
        active = [jnp.sqrt(d) / b >= tol for d, b in zip(dot(r, r), bnorm, strict=True)]
        alpha = [jnp.where(on, a / safe(d), 0.0) for on, a, d in zip(active, rz, pAp, strict=True)]
        x = [xx + al[:, None, None] * pp for xx, al, pp in zip(x, alpha, p, strict=True)]
        r = [
            jnp.where(live_l, rr - al[:, None, None] * a, 0.0)
            for rr, al, a in zip(r, alpha, Ap, strict=True)
        ]
        z = [precond(rr) for rr in r]
        rz_new = dot(r, z)
        beta = [
            jnp.where(on, n / safe(o), 0.0) for on, n, o in zip(active, rz_new, rz, strict=True)
        ]
        p = [zz + be[:, None, None] * pp for zz, be, pp in zip(z, beta, p, strict=True)]
        rel_new = rel_res(r)
        improved = rel_new < 0.99 * best
        # Keep each lane's best iterate. Float32 CG often stalls and then
        # drifts, so the last iterate can be worse than an earlier one.
        lr = lane_rel(r)
        better = [n < o for n, o in zip(lr, lane_best, strict=True)]
        x_best = [
            jnp.where(bt[:, None, None], xx, xb)
            for bt, xx, xb in zip(better, x, x_best, strict=True)
        ]
        lane_best = [jnp.minimum(n, o) for n, o in zip(lr, lane_best, strict=True)]
        return (
            x,
            r,
            p,
            rz_new,
            k + 1,
            jnp.where(improved, rel_new, best),
            jnp.where(improved, k + 1, k_best),
            x_best,
            lane_best,
        )

    init = (x, r, z, dot(r, z), jnp.array(0), rel_res(r), jnp.array(0), x, lane_rel(r))
    *_, x_best, lane_best = jax.lax.while_loop(cond, body, init)
    worst = jnp.max(jnp.stack([jnp.max(lb) for lb in lane_best]))
    return [xx.transpose(0, 2, 1) for xx in x_best], worst, tol


def pair_solve_warning(rel: Array | None, max_iter: int) -> str | None:
    """Return the warning text for a joint pair CG that did not converge.

    Args:
        rel: Worst relative residual, or ``None`` when no CG ran. A tracer
            (under jit or vmap) has no value and gives no warning.
        max_iter: The fit's ``max_iter``.

    Returns:
        The message, or ``None`` when there is nothing to warn about.
    """
    if rel is None or isinstance(rel, jax.core.Tracer):
        return None
    tol = _pair_tol(float(jnp.finfo(rel.dtype).eps))
    if float(rel) < max(tol, _PAIR_WARN_REL):
        return None
    return (
        f"jaxgsa.hdmr: the joint second-order solve stopped at relative "
        f"residual {float(rel):.1e}, at the cap of "
        f"{_PAIR_ITER_FACTOR * max_iter} steps (10 x max_iter) or after "
        f"{_PAIR_STALL} steps without progress (float32). Raise ridge, "
        "use float64, or raise max_iter."
    )


@lru_cache(maxsize=32)
def _get_pair_solve(m: int, b: int, n_row_batches: int, max_iter: int, ridge: float):
    """Build and cache the jitted joint pair solve.

    It is cached on its static settings, so a repeated call with the same
    shapes compiles nothing.

    Returns:
        A jitted function ``(X_p, w, c2_idx, c3_idx, beta2, beta3, A, BtB,
        rhs, bases) -> (C2, rel)`` where ``rhs`` and ``C2`` stack the lane
        chunks on the leading axis. ``bases`` is ``(B1, B2)`` of every padded
        row (:func:`_pair_bases`), or ``None`` to rebuild them on every step.
    """
    build = _get_build_bases(m, True, False)

    @jax.jit
    def solve(
        X_p: Array,
        w_full: Array,
        c2_idx: Array,
        c3_idx: Array,
        beta2: Array,
        beta3: Array,
        A: Array,
        BtB: Array,
        rhs: Array,
        bases: tuple[Array, Array] | None,
    ) -> tuple[Array, Array]:
        if bases is None:

            def sweep(cs_: list[Array]) -> list[Array]:
                return _pair_normal_pass(
                    cs_, A, b, n_row_batches, X_p, w_full, build, c2_idx, c3_idx, beta2, beta3
                )

        else:

            def sweep(cs_: list[Array]) -> list[Array]:
                return _pair_normal_cached(cs_, A, *bases)

        C, rel, _ = _joint_pair_solve(BtB, list(rhs), ridge=ridge, max_iter=max_iter, sweep=sweep)
        return jnp.stack(C), rel

    return solve


# ---------------------------------------------------------------------------
# Per-batch accumulation steps
#
# Each is one jitted call, so a row batch costs a handful of dispatches rather
# than a few dozen. At the sizes HDMR runs at, that dispatch overhead is the
# dominant cost of a small fit.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _get_build_bases(m: int, order2: bool, order3: bool):
    """Build and cache the masked per-batch basis builder.

    The row mask is applied to ``B1`` and therefore reaches ``B2`` and ``B3``
    through their tensor products. Every accumulator below contracts against
    one of the three, so a padded row contributes a hard zero everywhere it
    is not masked again explicitly.

    Returns:
        A jitted function ``(X_b, w, c2_idx, c3_idx, beta2, beta3) -> list``
        of the active bases, first order first.
    """

    @jax.jit
    def build(
        X_b: Array,
        w: Array,
        c2_idx: Array,
        c3_idx: Array,
        beta2: Array,
        beta3: Array,
    ) -> list[Array]:
        B1 = _build_B1(X_b, m) * w[:, None, None]  # (b, m1, D)
        out = [B1]
        if order2:
            out.append(_build_B2(B1, c2_idx, beta2))  # (b, m2, n2)
        if order3:
            out.append(_build_B3(B1, c3_idx, beta3))  # (b, m3, n3)
        return out

    return build


def _walk_term_blocks(
    n_t: int,
    tb: int,
    init: Any,
    step: Callable[[Array, Array, Any], Any],
) -> Any:
    """Fold ``step`` over the terms in blocks of ``tb``, counting each term once.

    The last block is clamped to end at ``n_t``, so it shares terms with the
    previous block. ``step`` gets a mask for the terms that are new.

    Args:
        n_t: Number of terms, at least 1.
        tb: Block size, at most ``n_t``.
        init: Initial carry.
        step: ``(start, valid, carry) -> carry``, where ``start`` is the first
            term of the block and ``valid`` is a boolean mask of shape
            ``(tb,)`` for the terms not already seen.

    Returns:
        The final carry.
    """

    def body(i: Array, carry: Any) -> Any:
        start = jnp.minimum(i * tb, n_t - tb)
        return step(start, start + jnp.arange(tb) >= i * tb, carry)

    return jax.lax.fori_loop(0, -(-n_t // tb), body, init)


@jax.jit
def _acc_gram(B_b: Array, BtB: Array, colsum: Array) -> tuple[Array, Array]:
    """Accumulate one basis' per-term Gram block and column sums.

    Both are slice-independent, so this runs once per row batch rather than
    once per slice chunk.

    Args:
        B_b: Basis for this row batch, shape ``(b, m_basis, n_terms)``.
        BtB: Running Gram blocks, shape ``(n_terms, m_basis, m_basis)``.
        colsum: Running column sums, shape ``(m_basis, n_terms)``.

    Returns:
        The updated ``(BtB, colsum)``.
    """
    n_t = B_b.shape[2]
    if n_t == 0:
        return BtB, colsum + jnp.sum(B_b, axis=0)
    tb = min(n_t, _STATS_TERM_BLOCK)

    # A single einsum copies B_b twice to bring the term axis to the front. A
    # loop over term blocks copies one small block at a time instead.
    def step(start: Array, valid: Array, G: Array) -> Array:
        blk = jax.lax.dynamic_slice_in_dim(B_b, start, tb, axis=2)
        cur = jax.lax.dynamic_slice_in_dim(G, start, tb, axis=0)
        upd = jnp.where(
            valid[:, None, None], cur + jnp.einsum("rmj,rnj->jmn", blk, blk).astype(G.dtype), cur
        )
        return jax.lax.dynamic_update_slice_in_dim(G, upd, start, 0)

    return _walk_term_blocks(n_t, tb, BtB, step), colsum + jnp.sum(B_b, axis=0)


@jax.jit
def _acc_cross_gram1(B1_b: Array, G1: Array, colsum1: Array) -> tuple[Array, Array]:
    """Accumulate the first-order cross-Gram and column sums.

    The first order needs the full cross-Gram ``G1[j, m, k, n]``, not just the
    per-term diagonal blocks, because backfitting couples the dimensions.

    Args:
        B1_b: First-order basis for this row batch, shape ``(b, m1, n1)``.
        G1: Running cross-Gram, shape ``(n1, m1, n1, m1)``.
        colsum1: Running column sums, shape ``(m1, n1)``.

    Returns:
        The updated ``(G1, colsum1)``.
    """
    return (
        G1 + jnp.einsum("rmj,rnk->jmkn", B1_b, B1_b),
        colsum1 + jnp.sum(B1_b, axis=0),
    )


@jax.jit
def _acc_cross_gram12(B1_b: Array, B2_b: Array, G12: Array) -> Array:
    """Accumulate the cross-Gram between the flat first-order basis and B2.

    Args:
        B1_b: Masked first-order basis, shape ``(b, m1, n1)``.
        B2_b: Second-order basis, shape ``(b, m2, n2)``.
        G12: Running cross-Gram, shape ``(P, m2, n2)`` with ``P = m1 * n1``.

    Returns:
        The updated ``G12``.
    """
    return G12 + jnp.einsum("rp,rmj->pmj", B1_b.reshape(B1_b.shape[0], -1), B2_b)


@jax.jit
def _acc_gram22(B2_b: Array, G22: Array) -> Array:
    """Accumulate the full pair Gram over all pairs, one row batch.

    Args:
        B2_b: Second-order basis, shape ``(b, m2, n2)``.
        G22: Running Gram, shape ``(K, K)`` with ``K = n2 * m2`` in the flat
            pair layout ``k = j * m2 + m``.

    Returns:
        The updated ``G22``.
    """
    B2f = B2_b.transpose(0, 2, 1).reshape(B2_b.shape[0], -1)
    return G22 + B2f.T @ B2f


def _pair_solver(N: int, L: int, n2: int, m2: int, itemsize: int) -> str:
    """Choose how to solve the projected pair system.

    ``"dense"`` solves the ``K x K`` normal equations, ``K = n2 * m2``.
    ``"dual"`` solves the same ridge problem as an ``N x N`` system in sample
    space, which is smaller when there are more pair coefficients than rows.
    Both give the exact ridge solution, so the smaller one is used when it fits
    the memory budget: about three ``K x K`` arrays for ``"dense"``; two
    ``N x N`` arrays (the projected kernel and its factor) plus the ``N x L``
    residual and weights for ``"dual"``. ``"cg"`` is
    the matrix-free fallback.

    Direct solves run only in float64. In float32 the projected pair system
    carries round-off (in the projection and the right-hand side) in
    near-null directions. An exact solve fits them; CG from the per-term
    start stops before it does. Measured at D = 8, N = 1024 with one input
    on (0.5, 0.8): spurious S2 up to 0.0067 with a float32 factored solve,
    0.0002 with CG. At D = 15, N = 1024, test NRMSE was 1.7 with a float32
    direct solve against 0.17 with CG.

    Args:
        N: Number of rows.
        L: Number of output lanes, padded.
        n2: Number of pair terms.
        m2: Pair basis size.
        itemsize: Bytes per element of the working dtype.

    Returns:
        ``"dense"``, ``"dual"`` or ``"cg"``.
    """
    K = n2 * m2
    budget = get_memory_budget()
    if itemsize < 8:
        return "cg"
    dense_ok = 3 * K * K * itemsize <= budget
    # 10 % margin for the tile work and the tile padding.
    dual_ok = 1.1 * (2 * N * N + 2 * N * L) * itemsize <= budget
    if dual_ok and (N < K or not dense_ok):
        return "dual"
    return "dense" if dense_ok else "cg"


@jax.jit
def _residual1(
    B1_b: Array, C1_c: Array, Y_bc: Array, w: Array, shift_c: Array, mean_z_c: Array
) -> Array:
    """Return the first-order residual ``Y - f0 - B1 C1`` for one block.

    Args:
        B1_b: Masked first-order basis, shape ``(b, m1, n1)``.
        C1_c: First-order coefficients, shape ``(cs, m1, n1)``.
        Y_bc: This block of Y, shape ``(b, cs)``.
        w: Row mask, shape ``(b,)``.
        shift_c: The first row of Y for these lanes, shape ``(cs,)``.
        mean_z_c: Mean of the shifted response per lane, shape ``(cs,)``.

    Returns:
        The residual, shape ``(b, cs)``, zero on padded rows.
    """
    Y_res = (Y_bc - shift_c[None, :]) - mean_z_c[None, :]
    return (Y_res - jnp.einsum("rmj,lmj->rl", B1_b, C1_c)) * w[:, None]


@jax.jit
def _pair_kernel_rows(B1_t: Array, B1: Array, inv_lam: Array) -> Array:
    """Rows of the pair kernel ``sum_p (G_i o G_j) / lam_p``.

    A pair basis is the tensor product of two first-order bases, so its row
    Gram is the elementwise product of theirs: ``B2_p B2_p^T = G_i o G_j``
    with ``G_i = B1_i B1_i^T``. The ``N x K`` pair basis is never built.

    Args:
        B1_t: First-order basis of these rows, shape ``(t, m1, n1)``.
        B1: First-order basis of all rows, shape ``(N, m1, n1)``.
        inv_lam: ``1 / lam_p`` on the upper triangle, zero elsewhere,
            shape ``(n1, n1)``.

    Returns:
        The kernel rows, shape ``(t, N)``.
    """
    G = jnp.einsum("tmi,nmi->itn", B1_t, B1)
    # Two steps, so no (n1, n1, t, N) intermediate is ever formed.
    H = jnp.einsum("ij,jtn->itn", inv_lam, G)
    return jnp.sum(G * H, axis=0)


@functools.partial(jax.jit, static_argnames=("t",))
def _dual_pair_weights(B1: Array, inv_lam: Array, U: Array, Y_res: Array, t: int) -> Array:
    """Return ``P1 (P1 K2 P1 + I)^-1 y`` for every lane.

    ``K2`` is the pair kernel of :func:`_pair_kernel_rows` and
    ``P1 = I - U U^T`` removes the first-order span. The projected kernel is
    written tile by tile into one ``N x N`` buffer, so the unprojected kernel
    is never held: peak memory is that buffer plus its Cholesky factor.

    Args:
        B1: First-order basis of the real rows, shape ``(N, m1, n1)``.
        inv_lam: ``1 / lam_p`` on the upper triangle, shape ``(n1, n1)``.
        U: Orthonormal basis of the first-order span, shape ``(N, r)``.
        Y_res: First-order residuals, shape ``(N, L)``.
        t: Kernel rows per tile.

    Returns:
        The sample-space weights, shape ``(N, L)``.
    """
    N = B1.shape[0]
    n_tiles = -(-N // t)
    Np = n_tiles * t
    # Padded rows have a zero basis, so their kernel rows and columns are zero
    # and the identity decouples them; their residual is zero too.
    B1p, Up, Yp = (pad_rows(A, Np - N) for A in (B1, U, Y_res))

    def rows(i: Array) -> Array:
        return _pair_kernel_rows(jax.lax.dynamic_slice_in_dim(B1p, i * t, t), B1p, inv_lam)

    def ku_tile(i: Array, KU: Array) -> Array:
        return jax.lax.dynamic_update_slice_in_dim(KU, rows(i) @ Up, i * t, axis=0)

    KU = jax.lax.fori_loop(0, n_tiles, ku_tile, jnp.zeros_like(Up))
    # P1 K2 P1 = K2 - Z U^T - U Z^T with Z = K2 U - U (U^T K2 U) / 2.
    Z = KU - 0.5 * Up @ (Up.T @ KU)
    diag = jnp.arange(t)

    def kp_tile(i: Array, Kp: Array) -> Array:
        Zi = jax.lax.dynamic_slice_in_dim(Z, i * t, t)
        Ui = jax.lax.dynamic_slice_in_dim(Up, i * t, t)
        blk = rows(i) - Zi @ Up.T - Ui @ Z.T
        blk = blk.at[diag, i * t + diag].add(1.0)
        return jax.lax.dynamic_update_slice_in_dim(Kp, blk, i * t, axis=0)

    Kp = jax.lax.fori_loop(0, n_tiles, kp_tile, jnp.zeros((Np, Np), dtype=B1.dtype))
    Lc = jnp.linalg.cholesky(Kp, symmetrize_input=False)
    a = jax.scipy.linalg.solve_triangular(Lc, Yp, lower=True)
    a = jax.scipy.linalg.solve_triangular(Lc, a, lower=True, trans=1)
    return (a - Up @ (Up.T @ a))[:N]


@jax.jit
def _acc_pair_dual(B2_b: Array, a_b: Array, acc: Array) -> Array:
    """Accumulate ``B2^T a`` for one row batch.

    Args:
        B2_b: Second-order basis, shape ``(b, m2, n2)``.
        a_b: Sample-space weights of these rows, shape ``(b, L)``.
        acc: Running product, shape ``(L, n2, m2)``.

    Returns:
        The updated product.
    """
    return acc + jnp.einsum("rmj,rl->ljm", B2_b, a_b)


def _dual_pair_solve(
    G1_flat: Array,
    BtB: Array,
    Y_res: Array,
    X_p: Array,
    w_full: Array,
    rows: list[tuple[int, int]],
    basis_args: tuple[int, Array, Array, Array, Array],
    n_chunks: int,
    cs: int,
    ridge: float,
) -> Array:
    """Solve all pair terms jointly in sample space.

    With ``Bt = P1 B2`` the projected pair basis and ``Lam`` the per-term
    ridge, the pair coefficients are ``(Bt^T Bt + Lam)^-1 Bt^T y``. The
    identity ``(B^T B + Lam)^-1 B^T = Lam^-1 B^T (B Lam^-1 B^T + I)^-1``
    turns this into one ``N x N`` system, factored once for every lane. It is
    the exact ridge solution, the same as :func:`_dense_pair_solve`, and is
    smaller whenever there are more pair coefficients than rows.

    Args:
        G1_flat: Flat first-order Gram, shape ``(P, P)``.
        BtB: Projected per-term Gram blocks, ``(n2, m2, m2)``, for the ridge
            scale.
        Y_res: First-order residuals of the real rows, shape ``(N, L)``.
        X_p: Padded inputs, shape ``(N_pad, D)``.
        w_full: Row mask, shape ``(N_pad,)``.
        rows: Row batch bounds.
        basis_args: ``(n_intervals, c2_idx, c3_idx, beta2, beta3)``.
        n_chunks: Number of lane chunks.
        cs: Lanes per chunk.
        ridge: Relative ridge.

    Returns:
        The pair coefficients, ``(n_chunks, cs, m2, n2)``.
    """
    n_intervals, c2_idx, c3_idx, beta2, beta3 = basis_args
    N = Y_res.shape[0]
    n2, m2, _ = BtB.shape
    build1 = _get_build_bases(n_intervals, False, False)
    B1 = jnp.concatenate(
        [build1(X_p[lo:hi], w_full[lo:hi], c2_idx, c3_idx, beta2, beta3)[0] for lo, hi in rows]
    )[:N]
    n1 = B1.shape[2]

    # Orthonormal basis of the first-order span, with the eigenvalue cut of
    # _first_order_pinv; a dropped direction gets a zero column.
    w, V, keep = _eigen_cut(G1_flat)
    U = B1.reshape(N, -1) @ (V * jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, w, 1.0)), 0.0))

    # A dead pair would make 1 / lam huge and the kernel indefinite, so the
    # Cholesky factor of every lane turns NaN. Drop it (see _live_pairs).
    lam, live = _live_pairs(BtB, ridge)  # (n2,)
    inv = jnp.where(live, 1.0 / jnp.where(live, lam, 1.0), 0.0)
    inv_lam = jnp.zeros((n1, n1), dtype=B1.dtype).at[c2_idx[:, 0], c2_idx[:, 1]].set(inv)
    itemsize = jnp.dtype(B1.dtype).itemsize
    # Even tiles, so the padded kernel stays close to N x N.
    n_tiles = -(-N // max(1, min(N, get_memory_budget() // (8 * n1 * N * itemsize))))
    t = -(-N // n_tiles)
    a = pad_rows(_dual_pair_weights(B1, inv_lam, U, Y_res, t), X_p.shape[0] - N)

    build2 = _get_build_bases(n_intervals, True, False)
    acc = jnp.zeros((a.shape[1], n2, m2), dtype=B1.dtype)
    for lo, hi in rows:
        B2_b = build2(X_p[lo:hi], w_full[lo:hi], c2_idx, c3_idx, beta2, beta3)[1]
        acc = _acc_pair_dual(B2_b, a[lo:hi], acc)
    C2 = acc * inv[None, :, None]
    return C2.reshape(n_chunks, cs, n2, m2).transpose(0, 1, 3, 2)


@jax.jit
def _dense_pair_solve(
    G22: Array, G12: Array, A: Array, BtB: Array, rhs: Array, ridge: Array
) -> Array:
    """Solve all pair terms jointly by one direct solve.

    The projected pair Gram is ``G22 - G12^T A`` (the Schur complement of the
    first-order block). It is formed once and solved for every lane, which is
    far cheaper than conjugate gradients whenever ``n2 * m2`` is moderate:
    each CG step is a full pass over the rows, and the projected normal
    matrix is ill-conditioned, so CG needs hundreds of steps and stalls in
    float32.

    Args:
        G22: Full pair Gram, ``(K, K)``, flat layout ``k = j * m2 + m``.
        G12: First-order to pair cross-Gram, ``(P, m2, n2)``.
        A: Projection of the pair basis onto the first-order space,
            ``(P, m2, n2)``.
        BtB: Projected per-term Gram blocks, ``(n2, m2, m2)``, for the ridge
            scale.
        rhs: Right-hand sides, ``(n_chunks, cs, n2, m2)``.
        ridge: Relative ridge, a scalar.

    Returns:
        The pair coefficients, ``(n_chunks, cs, m2, n2)``.
    """
    n_chunks, cs, n2, m2 = rhs.shape
    K = n2 * m2
    A_f = A.transpose(0, 2, 1).reshape(A.shape[0], K)
    G12_f = G12.transpose(0, 2, 1).reshape(G12.shape[0], K)
    Gt = G22 - G12_f.T @ A_f
    Gt = (Gt + Gt.T) / 2
    lam, live = _live_pairs(BtB, ridge)  # (n2,)
    # A dead pair keeps an identity block and a zero right-hand side, so its
    # coefficients are zero and it cannot reach the live pairs.
    live_flat = jnp.repeat(live, m2)
    lam_flat = jnp.repeat(jnp.where(live, lam, 1.0), m2)
    mask = live_flat[:, None] & live_flat[None, :]
    rhs_flat = jnp.where(live_flat[:, None], rhs.reshape(n_chunks * cs, K).T, 0.0)
    sol = jnp.linalg.solve(jnp.where(mask, Gt, 0.0) + jnp.diag(lam_flat), rhs_flat)
    return sol.T.reshape(n_chunks, cs, n2, m2).transpose(0, 1, 3, 2)


@jax.jit
def _acc_pair_normal(
    B1_b: Array,
    B2_b: Array,
    c: Array,
    v: Array,
    acc: tuple[Array, Array],
) -> tuple[Array, Array]:
    """Accumulate ``Bt^T Bt c`` for the projected pair basis, one row batch.

    With ``Bt = B2 - B1 A`` the product is formed without ever building
    ``Bt``: ``t = B2 c - B1 (A c)`` per row, then ``Bt^T t = B2^T t -
    A^T (B1^T t)``. The second part is left to the caller, which holds ``A``.

    Args:
        B1_b: Masked first-order basis, shape ``(b, m1, n1)``.
        B2_b: Second-order basis, shape ``(b, m2, n2)``.
        c: Pair coefficients, shape ``(cs, n2, m2)``.
        v: ``A c``, shape ``(cs, P)``.
        acc: Running ``(B1^T t, B2^T t)``, shapes ``(cs, P)`` and
            ``(cs, n2, m2)``.

    Returns:
        The updated accumulators.
    """
    b, m2, n2 = B2_b.shape
    cs = c.shape[0]
    B1f = B1_b.reshape(b, -1)
    # Flat matrix products: the same products as 3-D einsums were measured
    # 3.6x slower (16 ms against 58 ms at N = 4096, K = 12400).
    B2f = B2_b.reshape(b, m2 * n2)
    t = B2f @ c.transpose(0, 2, 1).reshape(cs, m2 * n2).T - B1f @ v.T  # (b, cs)
    g1, g2 = acc
    g2_new = (t.T @ B2f).reshape(cs, m2, n2).transpose(0, 2, 1)
    return g1 + t.T @ B1f, g2 + g2_new


@jax.jit
def _project_pair_basis(B_all: list[Array], A: Array) -> list[Array]:
    """Replace ``B2`` by ``B2 - B1 A``, which holds no first-order function.

    It is lane-independent, so it is built once per row batch.

    Args:
        B_all: Bases ``[B1, B2, ...]`` for one row batch.
        A: Projection of ``B2`` onto the first-order space, ``(P, m2, n2)``.

    Returns:
        The same list with ``B2`` projected.
    """
    B1f = B_all[0].reshape(B_all[0].shape[0], -1)
    Bt = B_all[1] - jnp.einsum("rp,pmj->rmj", B1f, A)
    return [B_all[0], Bt, *B_all[2:]]


@jax.jit
def _acc_moments1(
    B1_b: Array,
    Y_bc: Array,
    w: Array,
    shift_c: Array,
    carry: tuple[Array, Array, Array],
) -> tuple[Array, Array, Array]:
    """Accumulate the first-order moments and the Y statistics for one chunk.

    Everything here is accumulated on ``z = Y - Y[0]`` rather than on ``Y``.
    Shifting by a constant changes neither the mean's offset nor the variance
    nor ``B1^T (Y - f0)``, and ``Y[0]`` removes the bulk of the mean, so an
    output with a large offset does not lose its low bits to cancellation
    when the accumulated moment is re-centred after the pass. A constant
    slice gives ``z = 0`` exactly, so the ``V_Y == 0`` detection is exact.

    Args:
        B1_b: Masked first-order basis, shape ``(b, m1, n1)``.
        Y_bc: This (row batch, slice chunk) block of Y, shape ``(b, cs)``.
        w: Row mask, shape ``(b,)``.
        shift_c: The first row of Y for these lanes, shape ``(cs,)``.
        carry: Running ``(b1z, sum_z, sum_z2)``.

    Returns:
        The updated carry.
    """
    b1z, sum_z, sum_z2 = carry
    Z = (Y_bc - shift_c[None, :]) * w[:, None]
    return (
        b1z + jnp.einsum("rmj,rl->ljm", B1_b, Z),
        sum_z + jnp.sum(Z, axis=0),
        sum_z2 + jnp.sum(Z * Z, axis=0),
    )


@jax.jit
def _acc_higher_moment(
    B_new: Array,
    B_lower: list[Array],
    C_lower: list[Array],
    Y_bc: Array,
    shift_c: Array,
    mean_z_c: Array,
    BtY: Array,
) -> Array:
    """Accumulate ``B^T Y_res`` for one order, on the exact residual.

    The sequential residual is recomputed from the already-solved lower-order
    coefficients rather than carried, which is what makes the fit exact under
    row batching. ``Y - f0`` is formed as ``(Y - Y[0]) - mean(Y - Y[0])`` for
    the same reason :func:`_acc_moments1` accumulates on the shifted response.

    Args:
        B_new: Basis of the order being solved, shape ``(b, m_basis, n_terms)``.
        B_lower: Bases of the already-solved orders.
        C_lower: Their coefficients, each ``(cs, m_basis, n_terms)``.
        Y_bc: This block of Y, shape ``(b, cs)``.
        shift_c: The first row of Y for these lanes, shape ``(cs,)``.
        mean_z_c: Mean of the shifted response per lane, shape ``(cs,)``.
        BtY: Running moments, shape ``(cs, n_terms, m_basis)``.

    Returns:
        The updated moments.
    """
    Y_res = (Y_bc - shift_c[None, :]) - mean_z_c[None, :]
    for B, C in zip(B_lower, C_lower, strict=True):
        Y_res = Y_res - jnp.einsum("rmj,lmj->rl", B, C)
    return BtY + jnp.einsum("rmj,rl->ljm", B_new, Y_res)


def _stats_step_whole(
    B_list: list[Array],
    C_list: list[Array],
    Y_bc: Array,
    w: Array,
    shift_c: Array,
    mean_z_c: Array,
    mean_f_c: Array,
    mean_y0m_c: Array,
    acc: tuple[Array, Array, Array, Array, Array, Array, Array],
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Fold one block into the statistics with ``F`` held whole.

    Same maths as the blocked step. It forms the ``(b, cs, n)`` contributions
    once, which is fastest but needs about ``3 * b * cs * n`` elements.

    Args:
        B_list: Bases per order, each ``(b, m_k, n_k)``.
        C_list: Coefficients per order, each ``(cs, m_k, n_k)``.
        Y_bc: Outputs, shape ``(b, cs)``.
        w: Row mask, shape ``(b,)``.
        shift_c: Per-lane shift, shape ``(cs,)``.
        mean_z_c: Per-lane mean of the shifted output, shape ``(cs,)``.
        mean_f_c: Term means, shape ``(cs, n)``.
        mean_y0m_c: Means of the leave-one-out sums, shape ``(cs, n)``.
        acc: Running accumulators.

    Returns:
        The updated accumulators.
    """
    acc_var_f, acc_cov_fy, acc_cov_fy0m, acc_ssr, acc_cross, acc_fsq, acc_res2 = acc
    n1 = B_list[0].shape[2]
    n2 = B_list[1].shape[2] if len(B_list) > 1 else 0
    F = jnp.concatenate(
        [jnp.einsum("rmj,lmj->rlj", B, C) for B, C in zip(B_list, C_list, strict=True)],
        axis=2,
    )  # (b, cs, n)
    fc = (F - mean_f_c[None]) * w[:, None, None]
    acc_var_f = acc_var_f + jnp.sum(fc * fc, axis=0)
    y0 = jnp.sum(F, axis=2)
    y0_w = y0 * w[:, None]
    acc_cov_fy = acc_cov_fy + jnp.sum(fc * y0_w[:, :, None], axis=0)
    y0m_c = (y0[:, :, None] - F) - mean_y0m_c[None]  # leave-one-out sums
    acc_cov_fy0m = acc_cov_fy0m + jnp.sum(fc * y0m_c, axis=0)
    r0 = ((Y_bc - shift_c[None, :]) - mean_z_c[None, :]) * w[:, None]
    r1 = r0 - jnp.sum(F[:, :, :n1], axis=2)
    r2 = r1 - jnp.sum(F[:, :, n1 : n1 + n2], axis=2)
    acc_ssr = acc_ssr + jnp.stack(
        [jnp.sum(r0 * r0, axis=0), jnp.sum(r1 * r1, axis=0), jnp.sum(r2 * r2, axis=0)],
        axis=1,
    )
    acc_cross = acc_cross + jnp.concatenate(
        [
            jnp.sum(r0[:, :, None] * F[:, :, :n1], axis=0),
            jnp.sum(r1[:, :, None] * F[:, :, n1 : n1 + n2], axis=0),
            jnp.sum(r2[:, :, None] * F[:, :, n1 + n2 :], axis=0),
        ],
        axis=1,
    )
    acc_fsq = acc_fsq + jnp.sum(F * F, axis=0)
    r_pred = r0 - y0
    acc_res2 = acc_res2 + jnp.sum(r_pred * r_pred, axis=0)
    return acc_var_f, acc_cov_fy, acc_cov_fy0m, acc_ssr, acc_cross, acc_fsq, acc_res2


def _block_sum(B: Array, C: Array, b: int, cs: int, dtype: jnp.dtype) -> Array:
    """Sum the per-term contributions of one order, ``sum_j B_j C_j``.

    One matrix product over all terms would copy ``B`` into a flat layout. A
    loop over term blocks copies only one block at a time.

    Args:
        B: Basis, shape ``(b, m, n_k)``.
        C: Coefficients, shape ``(cs, m, n_k)``.
        b: Rows in the batch.
        cs: Lanes in the chunk.
        dtype: Accumulator dtype.

    Returns:
        The summed contribution, shape ``(b, cs)``.
    """
    n_k = B.shape[2]
    if n_k == 0:
        return jnp.zeros((b, cs), dtype=dtype)
    tb = min(n_k, _STATS_TERM_BLOCK)

    def step(start: Array, valid: Array, total: Array) -> Array:
        B_blk = jax.lax.dynamic_slice_in_dim(B, start, tb, axis=2)
        C_blk = jnp.where(valid, jax.lax.dynamic_slice_in_dim(C, start, tb, axis=2), 0)
        return total + jnp.einsum("rmj,lmj->rl", B_blk, C_blk).astype(dtype)

    return _walk_term_blocks(n_k, tb, jnp.zeros((b, cs), dtype=dtype), step)


@lru_cache(maxsize=2)
def _get_stats_step(blocked: bool):
    """Build and cache the final-pass reduction for one (row batch, chunk).

    The per-term contributions ``F`` have shape ``(b, cs, n)``, which is the
    largest tensor of the fit. XLA would also materialise two more tensors of
    that size (the centred ``F`` and a broadcast row weight), so the step
    never forms ``F`` whole. It first gets the per-order sums with plain
    matrix products, then walks the terms of each order in blocks of at most
    ``_STATS_TERM_BLOCK`` and folds each block into the accumulators. Only one
    block's temporaries are live at a time.

    The blocked path computes each term's contribution twice (once for the
    order sums, once in the block loop), so it is slower. ``blocked=False``
    keeps the single-pass path, which is right whenever the whole
    ``(b, cs, n)`` working set fits the memory budget. The caller picks the
    path; it is part of the cache key.

    Args:
        blocked: Whether to walk the terms in blocks.

    Returns:
        A jitted function that folds one block into the ANCOVA covariances,
        the hierarchical F-test sums of squares and cross-products, and the
        RMSE residual sum.
    """

    @jax.jit
    def step(
        B_list: list[Array],
        C_list: list[Array],
        Y_bc: Array,
        w: Array,
        shift_c: Array,
        mean_z_c: Array,
        mean_f_c: Array,
        mean_y0m_c: Array,
        acc: tuple[Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        if not blocked:
            return _stats_step_whole(
                B_list, C_list, Y_bc, w, shift_c, mean_z_c, mean_f_c, mean_y0m_c, acc
            )
        acc_var_f, acc_cov_fy, acc_cov_fy0m, acc_ssr, acc_cross, acc_fsq, acc_res2 = acc
        b, cs = Y_bc.shape
        # The loops carry fixed dtypes, so promote once up front.
        dtype = jnp.result_type(*B_list, *C_list, Y_bc, w, mean_f_c, mean_y0m_c)

        # Per-order sums of the term contributions, sum_j f_j, with no F.
        zero = jnp.zeros((b, cs), dtype=dtype)
        sums = [_block_sum(B, C, b, cs, dtype) for B, C in zip(B_list, C_list, strict=True)]
        sums = [*sums, zero, zero][:3]
        y0 = sums[0] + sums[1] + sums[2]  # total emulator, sans f0 (= f_hat - f0)

        # ANCOVA (Li et al., 2010). Sa = Var(f_j)/Var(Y) is the structural
        # share, Sb = Cov(f_j, sum_{k != j} f_k)/Var(Y) the correlative
        # remainder, and S = Sa + Sb = Cov(f_j, f_hat)/Var(Y) the total,
        # where f_hat = f0 + sum_k f_k is the fitted expansion. Measuring S
        # against the fitted expansion rather than against Y itself is what
        # makes S = Sa + Sb exact: summing S over every term gives
        # Var(f_hat)/Var(Y), the surrogate's explained-variance fraction,
        # matching the direct sum of Sa + Sb term by term. SALib's `ancova`
        # measures S against Y instead, which leaves a gap of
        # Cov(f_j, Y - f_hat)/Var(Y) that does not cancel under correlated
        # inputs. With independent inputs Sb -> 0 and S -> Sa either way.
        w2 = w * w
        # The block loop works in a (term, lane, row) layout so every sum runs
        # over the minor axis. Summing ``X * y[None]`` over the middle axis of
        # a (term, row, lane) array hits a slow float32 XLA CPU fusion:
        # measured 102 ms in float32 against 7 ms in float64 at
        # (28, 624, 1000), and 2.5 ms over the minor axis.
        w2_ = w2[None, None, :]
        y0_w2_t = (y0 * w2[:, None]).T  # (cs, b)
        y0_t = y0.T

        r0 = ((Y_bc - shift_c[None, :]) - mean_z_c[None, :]) * w[:, None]  # Y - f0
        # F-test for incremental model selection (Li et al., 2002). The null
        # model for a term of order k holds every term of order < k, so the
        # residuals are hierarchical.
        r1 = r0 - sums[0]
        r2 = r1 - sums[1]
        r_null = [r0, r1, r2]

        outs: list[list[Array]] = [[], [], [], [], []]
        off = 0
        for k, (B, C) in enumerate(zip(B_list, C_list, strict=True)):
            n_k = B.shape[2]
            if n_k == 0:
                for lst in outs:
                    lst.append(jnp.zeros((0, cs), dtype=dtype))
                continue
            tb = min(n_k, _STATS_TERM_BLOCK)
            r_k = r_null[k].T[None]  # (1, cs, b)

            def body(
                start: Array,
                valid: Array,
                carry: tuple[Array, ...],
                B: Array = B,
                C: Array = C,
                off: int = off,
                tb: int = tb,
                r_k: Array = r_k,
            ) -> tuple[Array, ...]:
                B_blk = jax.lax.dynamic_slice_in_dim(B, start, tb, axis=2)
                C_blk = jax.lax.dynamic_slice_in_dim(C, start, tb, axis=2)
                mf = jax.lax.dynamic_slice_in_dim(mean_f_c, off + start, tb, axis=1)
                my = jax.lax.dynamic_slice_in_dim(mean_y0m_c, off + start, tb, axis=1)
                Fk = jnp.einsum("rmj,lmj->jlr", B_blk, C_blk)  # (tb, cs, b)
                fcn = Fk - mf.T[:, :, None]
                y0m_c = (y0_t[None] - Fk) - my.T[:, :, None]
                vals = (
                    jnp.sum(fcn * fcn * w2_, axis=2),
                    jnp.sum(fcn * y0_w2_t[None], axis=2),
                    jnp.sum(fcn * (y0m_c * w2_), axis=2),
                    # sum r_null * f_i, each term against its own null residual.
                    jnp.sum(r_k * Fk, axis=2),
                    jnp.sum(Fk * Fk, axis=2),
                )
                new = []
                for acc_k, v in zip(carry, vals, strict=True):
                    cur = jax.lax.dynamic_slice_in_dim(acc_k, start, tb, axis=0)
                    new.append(
                        jax.lax.dynamic_update_slice_in_dim(
                            acc_k,
                            jnp.where(valid[:, None], cur + v.astype(acc_k.dtype), cur),
                            start,
                            0,
                        )
                    )
                return tuple(new)

            init = tuple(jnp.zeros((n_k, cs), dtype=dtype) for _ in range(5))
            res = _walk_term_blocks(n_k, tb, init, body)
            for lst, r in zip(outs, res, strict=True):
                lst.append(r)
            off += n_k

        # Each is (n, cs) after joining the orders; return to (cs, n).
        acc_var_f = acc_var_f + jnp.concatenate(outs[0], axis=0).T
        acc_cov_fy = acc_cov_fy + jnp.concatenate(outs[1], axis=0).T
        acc_cov_fy0m = acc_cov_fy0m + jnp.concatenate(outs[2], axis=0).T
        acc_cross = acc_cross + jnp.concatenate(outs[3], axis=0).T
        acc_fsq = acc_fsq + jnp.concatenate(outs[4], axis=0).T
        acc_ssr = acc_ssr + jnp.stack(
            [jnp.sum(r0 * r0, axis=0), jnp.sum(r1 * r1, axis=0), jnp.sum(r2 * r2, axis=0)],
            axis=1,
        )
        r_pred = r0 - y0  # full-model residual Y - (f0 + sum f_j)
        acc_res2 = acc_res2 + jnp.sum(r_pred * r_pred, axis=0)
        return acc_var_f, acc_cov_fy, acc_cov_fy0m, acc_ssr, acc_cross, acc_fsq, acc_res2

    return step


@lru_cache(maxsize=32)
def _get_finalize(n: int, n1: int, n2: int, m1: int, m2: int, m3: int, N: int):
    """Build and cache the elementwise finish: indices, selection, RMSE.

    Returns:
        A jitted function
        ``(V_Y, Y, w, acc, f_crits) -> (Sa, Sb, S, select, rmse, explained)``
        over every lane at once, where ``w`` is the row mask marking real
        rows. It is elementwise in the lane axis, so it does not need
        chunking.
    """

    @jax.jit
    def finalize(
        V_Y: Array,
        Y: Array,
        w: Array,
        acc: tuple[Array, Array, Array, Array, Array, Array, Array],
        f_crits: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        acc_var_f, acc_cov_fy, acc_cov_fy0m, acc_ssr, acc_cross, acc_fsq, acc_res2 = acc
        # A slice counts as zero-variance when it is numerically constant
        # (max == min, which survives float32 where var == 0 does not) or when
        # its variance is negligible against the machine epsilon of the
        # working dtype relative to the slice's scale (max |Y|); below that
        # the 1e-30 denominator floor turns the indices into finite garbage.
        # Emit NaN (matching Sobol and the zero-variance warning) rather than
        # silent zeros; the safe denominator only avoids the inf.
        # The extremes must come from the real rows only: padded rows carry
        # arbitrary values (zero by default, but a padded slot is required to
        # be unobservable), so the infinities exclude them from both extremes.
        eps = jnp.finfo(V_Y.dtype).eps
        keep = w[:, None] == 1
        const = jnp.max(jnp.where(keep, Y, -jnp.inf), axis=0) == jnp.min(
            jnp.where(keep, Y, jnp.inf), axis=0
        )
        scale = jnp.max(jnp.where(keep, jnp.abs(Y), 0.0), axis=0)
        zero_var = (const | (V_Y < eps * scale**2))[:, None]
        V_Y_safe = jnp.maximum(V_Y, 1e-30)[:, None]
        Sa = jnp.where(zero_var, jnp.nan, (acc_var_f / N) / V_Y_safe)
        S = jnp.where(zero_var, jnp.nan, (acc_cov_fy / N) / V_Y_safe)
        Sb = jnp.where(zero_var, jnp.nan, (acc_cov_fy0m / N) / V_Y_safe)

        # The per-term order map comes from _engine.term_order_map, so the
        # basis counts, critical values and null-model indices have one
        # definition rather than a copy that can drift.
        p1, f_crit_per_term, order_idx = term_order_map(n, n1, n2, m1, m2, m3, f_crits)
        SSR0 = acc_ssr[:, order_idx]  # (L, n) null SSR per term
        # SSR1 = sum (r_null - f_i)^2 = SSR0 - 2 sum r_null f_i + sum f_i^2.
        SSR1 = SSR0 - 2.0 * acc_cross + acc_fsq
        # R <= p makes the test undefined; guard the denominator.
        R_minus_p = jnp.maximum(N - p1, 1.0)
        F_stat = ((SSR0 - SSR1) / p1) / (SSR1 / R_minus_p)
        select = (F_stat > f_crit_per_term[None]).astype(jnp.float32)

        mse = acc_res2 / N
        # 1 - in-sample MSE / Var(Y), NaN on a zero-variance slice like the
        # indices above. It reuses the residual sum the RMSE already reads.
        explained = jnp.where(zero_var[:, 0], jnp.nan, 1.0 - mse / V_Y_safe[:, 0])
        return Sa, Sb, S, select, jnp.sqrt(mse), explained

    return finalize


# ---------------------------------------------------------------------------
# Fit driver
# ---------------------------------------------------------------------------


class _FitHDMR(NamedTuple):
    """The raw fit :func:`_fit_hdmr` returns, one field per name.

    A bare positional tuple let two of its ten fields swap silently at a
    call site with no type error to catch it (the D16 hazard). Every field
    here is read by name instead.

    Attributes:
        Sa: Structural variance fraction per term, shape ``(L, n)``.
        Sb: Correlative variance fraction per term, same shape.
        S: ``Sa + Sb``, same shape.
        select: F-test significance flag per term, same shape.
        rmse: Per-slice fit RMSE, shape ``(L,)``.
        explained_variance: Per-slice ``1 - MSE / Var(Y)``, shape ``(L,)``.
        C1: First-order component coefficients, shape ``(L, m1, n1)``.
        C2: Second-order coefficients, shape ``(L, m2, n2)``, or ``None``
            below ``order=2``.
        C3: Third-order coefficients, shape ``(L, m3, n3)``, or ``None``
            below ``order=3``.
        f0: Grand mean per slice, shape ``(L,)``.
        streamed: Whether the sample axis was actually split into row
            batches.
        pair_rel: Worst relative residual of the joint pair CG, or ``None``
            when a direct solve ran. A tracer under jit or vmap.
    """

    Sa: Array
    Sb: Array
    S: Array
    select: Array
    rmse: Array
    explained_variance: Array
    C1: Array
    C2: Array | None
    C3: Array | None
    f0: Array
    streamed: bool
    pair_rel: Array | None = None


def _fit_hdmr(
    X_n: Array,
    Y_nl: Array,
    *,
    n_intervals: int,
    order: int,
    max_iter: int,
    ridge: float,
    f_crits: Array,
    c2_idx: Array,
    c3_idx: Array,
    beta2: Array,
    beta3: Array,
    n1: int,
    n2: int,
    n3: int,
    n: int,
    m1: int,
    m2: int,
    m3: int,
    batch_size: int | None,
    slice_chunk_size: int | None,
) -> _FitHDMR:
    """Fit RS-HDMR over row batches and output-slice chunks.

    Runs ``1 + (order - 1) + 1`` passes over the row batches. Each pass
    rebuilds the per-batch B-spline bases once and then walks the slice
    chunks, so a full-``N`` basis is never resident and the basis build is
    not repeated per chunk:

    1. Accumulate the first-order cross-Gram ``G1``, the basis column sums,
       the per-lane moments ``B1^T Y`` and the Y statistics (mean, shifted
       variance), then solve the backfit in Gram space -> ``C1``.
    2. (order >= 2) Recompute the exact first-order residual per batch
       from ``C1`` and accumulate the per-term ``B2^T B2`` / ``B2^T Y_res``
       -> ``C2``; likewise a third pass for ``C3`` at order 3.
    3. Final pass: materialize the per-term contributions per batch and
       accumulate the ANCOVA covariances, the F-test sums of squares and
       cross-products, and the RMSE residual sum.

    Args:
        X_n: CDF-normalized inputs in [0, 1], shape ``(N, D)``.
        Y_nl: Response with the output slices flattened to the last axis,
            shape ``(N, L)``. Column ``t*K + k`` is slice ``(t, k)``.
        n_intervals: Number of B-spline intervals.
        order: HDMR expansion order (1, 2, or 3), post-clamping.
        max_iter: Maximum backfitting iterations.
        ridge: Tikhonov regularization strength.
        f_crits: Precomputed critical F values per order, shape ``(3,)``.
        c2_idx: Parameter index pairs, shape ``(n2, 2)``.
        c3_idx: Parameter index triples, shape ``(n3, 3)``.
        beta2: Tensor-product basis index table, shape ``(m2, 2)``.
        beta3: Tensor-product basis index table, shape ``(m3, 3)``.
        n1: Number of first-order terms (= D).
        n2: Number of second-order terms.
        n3: Number of third-order terms.
        n: Total number of terms.
        m1: First-order basis size.
        m2: Second-order basis size.
        m3: Third-order basis size.
        batch_size: Rows per batch, or ``None`` to derive one from the
            active memory budget.
        slice_chunk_size: Output slices per vmap, or ``None`` to derive one
            from the active memory budget.

    Returns:
        A :class:`_FitHDMR` with lane-leading shapes: (L, n) indices and
        selection, (L,) rmse and f0, (L, m1, n1) C1, (L, m2, n2) C2 or None,
        (L, m3, n3) C3 or None. ``streamed`` says whether the sample axis was
        actually split.
    """
    N, D = X_n.shape
    L = Y_nl.shape[1]
    dtype = jnp.result_type(X_n.dtype, Y_nl.dtype)
    itemsize = jnp.dtype(dtype).itemsize

    slice_bytes = _slice_bytes_per_row(n, itemsize)
    b = resolve_batch_size(
        _fit_bytes_per_row(D, m1, m2, m3, n2, n3, n, L, itemsize), N, batch_size
    )
    order2 = order >= 2 and n2 > 0
    order3 = order >= 3 and n3 > 0
    # Pair solver state per lane: rhs, x, best x, r, z, p and the
    # matrix-vector product.
    lane_fixed = 7 * n2 * m2 * itemsize if order2 else 0
    cs = _resolve_slice_chunk_size(slice_chunk_size, L, b, slice_bytes, lane_fixed)

    # Pad both axes so each jitted step compiles once. A padded row is masked
    # to zero in every accumulator; a padded lane is an independent constant
    # slice that is sliced off at the end.
    n_row_batches = math.ceil(N / b)
    n_chunks = math.ceil(L / cs)
    X_p = pad_rows(X_n, n_row_batches * b - N)
    Y_p = pad_rows(Y_nl, n_row_batches * b - N)
    if n_chunks * cs != L:
        Y_p = jnp.concatenate(
            [Y_p, jnp.zeros((Y_p.shape[0], n_chunks * cs - L), dtype=Y_p.dtype)], axis=1
        )
    w_full = jnp.concatenate(
        [jnp.ones((N,), dtype=dtype), jnp.zeros((n_row_batches * b - N,), dtype=dtype)]
    )
    rows = [(i * b, i * b + b) for i in range(n_row_batches)]
    lanes = [(j * cs, j * cs + cs) for j in range(n_chunks)]

    # ---- Pass 1: first-order Gram + moments and Y statistics -------------
    # This pass only needs the first-order basis. Higher-order tensor products
    # are expensive and would otherwise be built and discarded here.
    build_bases1 = _get_build_bases(n_intervals, False, False)
    shift = Y_p[0]  # (L_pad,)
    G1 = jnp.zeros((n1, m1, n1, m1), dtype=dtype)
    colsum1 = jnp.zeros((m1, n1), dtype=dtype)
    zeros_l = jnp.zeros((cs,), dtype=dtype)
    carry1 = [(jnp.zeros((cs, n1, m1), dtype=dtype), zeros_l, zeros_l) for _ in lanes]
    for lo, hi in rows:
        B1_b = build_bases1(X_p[lo:hi], w_full[lo:hi], c2_idx, c3_idx, beta2, beta3)[0]
        G1, colsum1 = _acc_cross_gram1(B1_b, G1, colsum1)
        for j, (s, e) in enumerate(lanes):
            carry1[j] = _acc_moments1(B1_b, Y_p[lo:hi, s:e], w_full[lo:hi], shift[s:e], carry1[j])

    sum_z = jnp.concatenate([c[1] for c in carry1])
    sum_z2 = jnp.concatenate([c[2] for c in carry1])
    mean_z = sum_z / N  # (L_pad,) mean of the shifted response
    f0 = shift + mean_z  # (L_pad,)
    V_Y = jnp.maximum(sum_z2 / N - jnp.square(mean_z), 0.0)  # (L_pad,)

    # b1res_j = B1_j^T (Y - f0) = B1_j^T z - mean(z) * colsum(B1_j), with
    # z = Y - Y[0]. Doing it on z keeps a large output offset out of the
    # subtraction, which a float32 fit of an offset response needs.
    diag = jnp.arange(n1)
    G1_diag = G1[diag, :, diag, :]  # (n1, m1, m1)
    backfit = _get_gram_backfit(m1, n1, max_iter, ridge)
    b1res_chunks = [
        carry1[j][0] - mean_z[s:e, None, None] * colsum1.T[None] for j, (s, e) in enumerate(lanes)
    ]  # each (cs, n1, m1)
    C1_chunks = [backfit(G1, G1_diag, b1res) for b1res in b1res_chunks]  # each (cs, m1, n1)
    P = m1 * n1
    # Flat first-order index p = m * n1 + j, the layout of a (m1, n1) array.
    G1_flat = G1.transpose(1, 0, 3, 2).reshape(P, P)

    # ---- Passes 2/3: higher orders on the exact sequential residual -------
    colsums = [colsum1]
    coef_chunks = [C1_chunks]  # what the residuals and predict() use
    stat_chunks = [C1_chunks]  # what the per-term statistics use
    proj_A: Array | None = None
    pair_rel: Array | None = None
    for term_order, (n_terms, m_basis) in enumerate(((n2, m2), (n3, m3)), start=2):
        if not (order2 if term_order == 2 else order3):
            break
        BtB = jnp.zeros((n_terms, m_basis, m_basis), dtype=dtype)
        colsum = jnp.zeros((m_basis, n_terms), dtype=dtype)
        BtY = [jnp.zeros((cs, n_terms, m_basis), dtype=dtype) for _ in lanes]
        G12 = jnp.zeros((P, m_basis, n_terms), dtype=dtype)
        kind = (
            _pair_solver(N, n_chunks * cs, n_terms, m_basis, itemsize) if term_order == 2 else "cg"
        )
        dense = kind == "dense"
        G22 = jnp.zeros((n_terms * m_basis,) * 2 if dense else (0, 0), dtype=dtype)
        # The dual solve needs the first-order residual in sample space.
        res_rows: list[Array] = []
        # Solving order ``k`` needs the bases up through order ``k`` to
        # reconstruct the exact sequential residual. Do not build a higher
        # order basis that this pass cannot consume.
        build_bases_order = _get_build_bases(n_intervals, term_order >= 2, term_order >= 3)
        for lo, hi in rows:
            B_all = build_bases_order(X_p[lo:hi], w_full[lo:hi], c2_idx, c3_idx, beta2, beta3)
            B_new = B_all[term_order - 1]
            BtB, colsum = _acc_gram(B_new, BtB, colsum)
            if term_order == 2:
                G12 = _acc_cross_gram12(B_all[0], B_new, G12)
                if dense:
                    G22 = _acc_gram22(B_new, G22)
                if kind == "dual":
                    res_rows.append(
                        jnp.concatenate(
                            [
                                _residual1(
                                    B_all[0],
                                    C1_chunks[j],
                                    Y_p[lo:hi, s:e],
                                    w_full[lo:hi],
                                    shift[s:e],
                                    mean_z[s:e],
                                )
                                for j, (s, e) in enumerate(lanes)
                            ],
                            axis=1,
                        )
                    )
            for j, (s, e) in enumerate(lanes):
                BtY[j] = _acc_higher_moment(
                    B_new,
                    B_all[: term_order - 1],
                    [c[j] for c in coef_chunks],
                    Y_p[lo:hi, s:e],
                    shift[s:e],
                    mean_z[s:e],
                    BtY[j],
                )
        # The projected pair blocks are rank-deficient by construction, so they
        # need a ridge floor even when the caller asks for none.
        term_ridge = (
            max(ridge, math.sqrt(float(jnp.finfo(dtype).eps))) if term_order == 2 else ridge
        )
        solver = _get_gram_higher_solver(m_basis, term_ridge)
        if term_order == 2:
            # Project the pair basis off the first-order space, so a pair term
            # holds only what no sum of 1-D functions can explain.
            A = (_first_order_pinv(G1_flat) @ G12.reshape(P, -1)).reshape(G12.shape)
            BtB = BtB - jnp.einsum("pmj,pnj->jmn", A, G12)
            BtB = (BtB + BtB.transpose(0, 2, 1)) / 2
            rhs = []
            for j in range(n_chunks):
                # B1^T of the first-order residual: not exactly zero, because
                # the backfit stops early and the ridge shrinks.
                r1 = b1res_chunks[j].transpose(0, 2, 1).reshape(cs, P) - (
                    C1_chunks[j].reshape(cs, P) @ G1_flat
                )
                rhs.append(BtY[j] - jnp.einsum("pmj,lp->ljm", A, r1))
            C2_stack: Array | None = None
            if dense:
                C2_stack = _dense_pair_solve(
                    G22, G12, A, BtB, jnp.stack(rhs), jnp.asarray(term_ridge, dtype=dtype)
                )
                del G22
            elif kind == "dual":
                C2_stack = _dual_pair_solve(
                    G1_flat,
                    BtB,
                    jnp.concatenate(res_rows)[:N],
                    X_p,
                    w_full,
                    rows,
                    (n_intervals, c2_idx, c3_idx, beta2, beta3),
                    n_chunks,
                    cs,
                    term_ridge,
                )
                del res_rows
            rel = None
            if C2_stack is None:
                # Hold the bases when they fit half the budget: a CG step is
                # then a few matrix products instead of a basis rebuild.
                bases = None
                if 2 * X_p.shape[0] * (n_terms * m_basis + P) * itemsize <= get_memory_budget():
                    bases = _pair_bases(
                        rows,
                        X_p,
                        w_full,
                        _get_build_bases(n_intervals, True, False),
                        (c2_idx, c3_idx, beta2, beta3),
                    )
                C2_stack, rel = _get_pair_solve(
                    n_intervals, b, n_row_batches, _PAIR_ITER_FACTOR * max_iter, term_ridge
                )(X_p, w_full, c2_idx, c3_idx, beta2, beta3, A, BtB, jnp.stack(rhs), bases)
                del bases
            pair_rel = rel
            C2_chunks = list(C2_stack)
            C1_fold = []
            for j in range(n_chunks):
                shift_p = jnp.einsum("pmj,lmj->lp", A, C2_chunks[j])  # (cs, P)
                C1_fold.append((C1_chunks[j].reshape(cs, P) - shift_p).reshape(cs, m1, n1))
            # The pair terms are B2 C2 minus their projection: use the
            # projected basis for the means and the stats pass.
            colsum = colsum - jnp.einsum("p,pmj->mj", colsum1.reshape(P), A)
            proj_A = A
            coef_chunks = [C1_fold, C2_chunks]
            stat_chunks = [C1_chunks, C2_chunks]
            colsums.append(colsum)
        else:
            colsums.append(colsum)
            C3_chunks = [solver(BtB, BtY[j]) for j in range(n_chunks)]
            coef_chunks.append(C3_chunks)
            stat_chunks.append(C3_chunks)

    # ---- Term means from column sums (no extra pass needed) ---------------
    # mean(f_j) = (1^T B_j C_j) / N.
    mean_f = [
        jnp.concatenate(
            [
                jnp.einsum("mj,lmj->lj", cols, C[j]) / N
                for cols, C in zip(colsums, stat_chunks, strict=True)
            ],
            axis=1,
        )
        for j in range(n_chunks)
    ]  # each (cs, n)
    # mean of the leave-one-out sums Y0_minus_j = Y0 - f_j.
    mean_y0m = [jnp.sum(mf, axis=1, keepdims=True) - mf for mf in mean_f]

    # ---- Final pass: ANCOVA / F-test / RMSE reductions ---------------------
    build_bases_final = _get_build_bases(n_intervals, order2, order3)
    # Hold F whole when the (row, lane, term) working set fits half the budget,
    # since that is the faster path. Read the budget here, outside any jit.
    blocked = b * cs * 3 * n * itemsize > get_memory_budget() // 2
    stats_step = _get_stats_step(blocked)
    accs = [
        (
            jnp.zeros((cs, n), dtype=dtype),  # sum (f - mean_f)^2
            jnp.zeros((cs, n), dtype=dtype),  # sum (f - mean_f)(Y - f0)
            jnp.zeros((cs, n), dtype=dtype),  # sum (f - mean_f)(Y0m - mean)
            jnp.zeros((cs, 3), dtype=dtype),  # SSR of the 3 null models
            jnp.zeros((cs, n), dtype=dtype),  # sum r_null * f, per term
            jnp.zeros((cs, n), dtype=dtype),  # sum f^2, per term
            jnp.zeros((cs,), dtype=dtype),  # sum (Y - f0 - Y0)^2
        )
        for _ in lanes
    ]
    for lo, hi in rows:
        B_all = build_bases_final(X_p[lo:hi], w_full[lo:hi], c2_idx, c3_idx, beta2, beta3)
        if proj_A is not None:
            B_all = _project_pair_basis(B_all, proj_A)
        for j, (s, e) in enumerate(lanes):
            accs[j] = stats_step(
                B_all,
                [c[j] for c in stat_chunks],
                Y_p[lo:hi, s:e],
                w_full[lo:hi],
                shift[s:e],
                mean_z[s:e],
                mean_f[j],
                mean_y0m[j],
                accs[j],
            )

    acc = tuple(jnp.concatenate([a[k] for a in accs]) for k in range(7))
    Sa, Sb, S, select, rmse, explained = _get_finalize(n, n1, n2, m1, m2, m3, N)(
        V_Y, Y_p, w_full, acc, f_crits
    )

    C = [jnp.concatenate(chunks)[:L] for chunks in coef_chunks]
    return _FitHDMR(
        Sa=Sa[:L],
        Sb=Sb[:L],
        S=S[:L],
        select=select[:L],
        rmse=rmse[:L],
        explained_variance=explained[:L],
        C1=C[0],
        C2=C[1] if order2 else None,
        C3=C[2] if order3 else None,
        f0=f0[:L],
        streamed=b < N,
        pair_rel=pair_rel,
    )
