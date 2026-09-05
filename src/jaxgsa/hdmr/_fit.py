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

import math
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxgsa._core.batching import get_memory_budget, resolve_batch_size
from jaxgsa.hdmr._engine import _build_B1, _build_B2, _build_B3, term_order_map

# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------


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
    and ``B3`` with its three gather temporaries (4 * m3 * n3).

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
    return itemsize * (m1 * D + 3 * m2 * n2 + 4 * m3 * n3)


def _slice_bytes_per_row(n: int, itemsize: int) -> int:
    """Transient bytes one (row, output slice) pair costs in the final pass.

    The statistics pass materializes the per-term contributions ``F`` plus
    roughly two elementwise temporaries of the same shape, so 3 * n elements
    per row per lane.

    Args:
        n: Total number of terms.
        itemsize: Bytes per element of the working dtype.

    Returns:
        Estimated bytes per (row, slice) pair.
    """
    return itemsize * 3 * n


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
) -> int:
    """Resolve how many output slices one vmap handles, against the budget.

    Args:
        slice_chunk_size: The caller's value, or ``None`` to derive one.
        L: Number of flattened output slices.
        b: Resolved rows per batch.
        slice_bytes: Bytes per (row, slice) pair, from
            :func:`_slice_bytes_per_row`.

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
            raise ValueError(f"slice_chunk_size must be >= 1, got {slice_chunk_size}")
        return min(slice_chunk_size, L)
    auto = get_memory_budget() // max(b * slice_bytes, 1)
    return max(1, min(L, auto))


# ---------------------------------------------------------------------------
# Gram-space solvers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _get_gram_backfit(m1: int, n1: int, maxiter: int, lambdax: float):
    """Build and cache the vmapped Gram-space first-order backfit.

    Coordinate-descent backfitting (Breiman & Friedman, 1985) written in
    coefficient space. With ``M_j = G1[j, j] + lambdax * I``, a backfit step
    only needs ``B1_j^T`` of the current partial residual, which the Gram
    blocks already hold:

    ``B1_j^T Y_r = b1res_j - sum_k G1[j, k] C1_k + G1[j, j] C1_j``.

    Tikhonov regularization keeps an ill-conditioned ``B^T B`` from blowing up
    the coefficients when the basis functions are nearly collinear or the
    sample count is low. Iteration stops when the largest change in any
    dimension's coefficient energy falls below one part in a thousand of the
    largest energy itself, or after ``maxiter`` sweeps. The test is relative,
    not absolute: the coefficients scale with ``Y`` and with ``m**-3``, so an
    absolute cutoff stops the sweep after one pass whenever ``Y`` is small or
    ``m`` is large (H4).

    Returns:
        A jitted function ``(G1, G1_diag, b1res) -> C1``. ``G1``
        (n1, m1, n1, m1) and ``G1_diag`` (n1, m1, m1) are shared across lanes,
        and ``b1res`` is batched as (L, n1, m1). The result is (L, m1, n1).
    """

    def backfit(G1: Array, G1_diag: Array, b1res: Array) -> Array:
        lam_eye = lambdax * jnp.eye(m1)
        M = G1_diag + lam_eye  # (n1, m1, m1)

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
            return (varmax > 1e-3 * jnp.max(var)) & (it < maxiter)

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
def _get_gram_higher_solver(m_basis: int, lambdax: float):
    """Build and cache the vmapped Gram-space higher-order solver.

    Higher-order terms are fit in a single shot, with no backfitting: each
    term is solved independently against the same lower-order residual,
    ``C_j = (B_j^T B_j + lambdax * I)^{-1} B_j^T Y_res``, from the accumulated
    per-term Gram blocks and moments. The lower-order structure is already
    removed, so same-order tensor-product terms overlap little and one
    joint-free pass is accurate in practice.

    Returns:
        A jitted function ``(BtB, BtY) -> C``. ``BtB``
        (n_terms, m_basis, m_basis) is shared across lanes, and ``BtY`` is
        batched as (L, n_terms, m_basis). The result is
        (L, m_basis, n_terms).
    """

    def solve_lane(BtB: Array, BtY: Array) -> Array:
        lam_eye = lambdax * jnp.eye(m_basis)
        C = jax.vmap(lambda btb, bty: jnp.linalg.solve(btb + lam_eye, bty))(BtB, BtY)
        return C.T  # (m_basis, n_terms)

    return jax.jit(jax.vmap(solve_lane, in_axes=(None, 0)))


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
    return BtB + jnp.einsum("rmj,rnj->jmn", B_b, B_b), colsum + jnp.sum(B_b, axis=0)


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


@lru_cache(maxsize=32)
def _get_stats_step(n1: int, n2: int):
    """Build and cache the final-pass reduction for one (row batch, chunk).

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
        acc_var_f, acc_cov_fy, acc_cov_fy0m, acc_ssr, acc_cross, acc_fsq, acc_res2 = acc
        # Per-term contributions f_j = B_j C_j, all orders side by side.
        F = jnp.concatenate(
            [jnp.einsum("rmj,lmj->rlj", B, C) for B, C in zip(B_list, C_list, strict=True)],
            axis=2,
        )  # (b, cs, n)

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
        fc = (F - mean_f_c[None]) * w[:, None, None]
        acc_var_f = acc_var_f + jnp.sum(fc * fc, axis=0)

        y0 = jnp.sum(F, axis=2)  # (b, cs) total emulator, sans f0 (= f_hat - f0)
        y0_w = y0 * w[:, None]
        acc_cov_fy = acc_cov_fy + jnp.sum(fc * y0_w[:, :, None], axis=0)

        y0m_c = (y0[:, :, None] - F) - mean_y0m_c[None]  # leave-one-out sums
        acc_cov_fy0m = acc_cov_fy0m + jnp.sum(fc * y0m_c, axis=0)

        r0 = ((Y_bc - shift_c[None, :]) - mean_z_c[None, :]) * w[:, None]  # Y - f0

        # F-test for incremental model selection (Li et al., 2002). The null
        # model for a term of order k holds every term of order < k, so the
        # residuals are hierarchical.
        r1 = r0 - jnp.sum(F[:, :, :n1], axis=2)
        r2 = r1 - jnp.sum(F[:, :, n1 : n1 + n2], axis=2)
        acc_ssr = acc_ssr + jnp.stack(
            [jnp.sum(r0 * r0, axis=0), jnp.sum(r1 * r1, axis=0), jnp.sum(r2 * r2, axis=0)],
            axis=1,
        )
        # sum r_null * f_i, each term against its own null residual.
        acc_cross = acc_cross + jnp.concatenate(
            [
                jnp.sum(r0[:, :, None] * F[:, :, :n1], axis=0),
                jnp.sum(r1[:, :, None] * F[:, :, n1 : n1 + n2], axis=0),
                jnp.sum(r2[:, :, None] * F[:, :, n1 + n2 :], axis=0),
            ],
            axis=1,
        )
        acc_fsq = acc_fsq + jnp.sum(F * F, axis=0)

        r_pred = r0 - y0  # full-model residual Y - (f0 + sum f_j)
        acc_res2 = acc_res2 + jnp.sum(r_pred * r_pred, axis=0)
        return acc_var_f, acc_cov_fy, acc_cov_fy0m, acc_ssr, acc_cross, acc_fsq, acc_res2

    return step


@lru_cache(maxsize=32)
def _get_finalize(n: int, n1: int, n2: int, m1: int, m2: int, m3: int, N: int):
    """Build and cache the elementwise finish: indices, selection, RMSE.

    Returns:
        A jitted function ``(V_Y, Y, w, acc, f_crits) -> (Sa, Sb, S, select, rmse)``
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
    ) -> tuple[Array, Array, Array, Array, Array]:
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

        return Sa, Sb, S, select, jnp.sqrt(acc_res2 / N)

    return finalize


# ---------------------------------------------------------------------------
# Fit driver
# ---------------------------------------------------------------------------


def _pad_rows(A: Array, n_pad: int) -> Array:
    """Pad ``A`` with ``n_pad`` zero rows on the sample axis."""
    if n_pad == 0:
        return A
    return jnp.concatenate([A, jnp.zeros((n_pad, *A.shape[1:]), dtype=A.dtype)], axis=0)


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
        C1: First-order component coefficients, shape ``(L, m1, n1)``.
        C2: Second-order coefficients, shape ``(L, m2, n2)``, or ``None``
            below ``maxorder=2``.
        C3: Third-order coefficients, shape ``(L, m3, n3)``, or ``None``
            below ``maxorder=3``.
        f0: Grand mean per slice, shape ``(L,)``.
        streamed: Whether the sample axis was actually split into row
            batches.
    """

    Sa: Array
    Sb: Array
    S: Array
    select: Array
    rmse: Array
    C1: Array
    C2: Array | None
    C3: Array | None
    f0: Array
    streamed: bool


def _fit_hdmr(
    X_n: Array,
    Y_nl: Array,
    *,
    m: int,
    maxorder: int,
    maxiter: int,
    lambdax: float,
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

    Runs ``1 + (maxorder - 1) + 1`` passes over the row batches. Each pass
    rebuilds the per-batch B-spline bases once and then walks the slice
    chunks, so a full-``N`` basis is never resident and the basis build is
    not repeated per chunk:

    1. Accumulate the first-order cross-Gram ``G1``, the basis column sums,
       the per-lane moments ``B1^T Y`` and the Y statistics (mean, shifted
       variance), then solve the backfit in Gram space -> ``C1``.
    2. (maxorder >= 2) Recompute the exact first-order residual per batch
       from ``C1`` and accumulate the per-term ``B2^T B2`` / ``B2^T Y_res``
       -> ``C2``; likewise a third pass for ``C3`` at maxorder 3.
    3. Final pass: materialize the per-term contributions per batch and
       accumulate the ANCOVA covariances, the F-test sums of squares and
       cross-products, and the RMSE residual sum.

    Args:
        X_n: CDF-normalized inputs in [0, 1], shape ``(N, D)``.
        Y_nl: Response with the output slices flattened to the last axis,
            shape ``(N, L)``. Column ``t*K + k`` is slice ``(t, k)``.
        m: Number of B-spline intervals.
        maxorder: HDMR expansion order (1, 2, or 3), post-clamping.
        maxiter: Maximum backfitting iterations.
        lambdax: Tikhonov regularization strength.
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
    cs = _resolve_slice_chunk_size(slice_chunk_size, L, b, slice_bytes)

    order2 = maxorder >= 2 and n2 > 0
    order3 = maxorder >= 3 and n3 > 0

    # Pad both axes so each jitted step compiles once. A padded row is masked
    # to zero in every accumulator; a padded lane is an independent constant
    # slice that is sliced off at the end.
    n_row_batches = math.ceil(N / b)
    n_chunks = math.ceil(L / cs)
    X_p = _pad_rows(X_n, n_row_batches * b - N)
    Y_p = _pad_rows(Y_nl, n_row_batches * b - N)
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
    build_bases1 = _get_build_bases(m, False, False)
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
    backfit = _get_gram_backfit(m1, n1, maxiter, lambdax)
    C1_chunks = [
        backfit(G1, G1_diag, carry1[j][0] - mean_z[s:e, None, None] * colsum1.T[None])
        for j, (s, e) in enumerate(lanes)
    ]  # each (cs, m1, n1)

    # ---- Passes 2/3: higher orders on the exact sequential residual -------
    colsums = [colsum1]
    coef_chunks = [C1_chunks]
    for order, (n_terms, m_basis) in enumerate(((n2, m2), (n3, m3)), start=2):
        if not (order2 if order == 2 else order3):
            break
        BtB = jnp.zeros((n_terms, m_basis, m_basis), dtype=dtype)
        colsum = jnp.zeros((m_basis, n_terms), dtype=dtype)
        BtY = [jnp.zeros((cs, n_terms, m_basis), dtype=dtype) for _ in lanes]
        # Solving order ``k`` needs the bases up through order ``k`` to
        # reconstruct the exact sequential residual. Do not build a higher
        # order basis that this pass cannot consume.
        build_bases_order = _get_build_bases(m, order >= 2, order >= 3)
        for lo, hi in rows:
            B_all = build_bases_order(X_p[lo:hi], w_full[lo:hi], c2_idx, c3_idx, beta2, beta3)
            B_new = B_all[order - 1]
            BtB, colsum = _acc_gram(B_new, BtB, colsum)
            for j, (s, e) in enumerate(lanes):
                BtY[j] = _acc_higher_moment(
                    B_new,
                    B_all[: order - 1],
                    [c[j] for c in coef_chunks],
                    Y_p[lo:hi, s:e],
                    shift[s:e],
                    mean_z[s:e],
                    BtY[j],
                )
        solver = _get_gram_higher_solver(m_basis, lambdax)
        colsums.append(colsum)
        coef_chunks.append([solver(BtB, BtY[j]) for j in range(n_chunks)])

    # ---- Term means from column sums (no extra pass needed) ---------------
    # mean(f_j) = (1^T B_j C_j) / N.
    mean_f = [
        jnp.concatenate(
            [
                jnp.einsum("mj,lmj->lj", cols, C[j]) / N
                for cols, C in zip(colsums, coef_chunks, strict=True)
            ],
            axis=1,
        )
        for j in range(n_chunks)
    ]  # each (cs, n)
    # mean of the leave-one-out sums Y0_minus_j = Y0 - f_j.
    mean_y0m = [jnp.sum(mf, axis=1, keepdims=True) - mf for mf in mean_f]

    # ---- Final pass: ANCOVA / F-test / RMSE reductions ---------------------
    build_bases_final = _get_build_bases(m, order2, order3)
    stats_step = _get_stats_step(n1, n2)
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
        for j, (s, e) in enumerate(lanes):
            accs[j] = stats_step(
                B_all,
                [c[j] for c in coef_chunks],
                Y_p[lo:hi, s:e],
                w_full[lo:hi],
                shift[s:e],
                mean_z[s:e],
                mean_f[j],
                mean_y0m[j],
                accs[j],
            )

    acc = tuple(jnp.concatenate([a[k] for a in accs]) for k in range(7))
    Sa, Sb, S, select, rmse = _get_finalize(n, n1, n2, m1, m2, m3, N)(
        V_Y, Y_p, w_full, acc, f_crits
    )

    C = [jnp.concatenate(chunks)[:L] for chunks in coef_chunks]
    return _FitHDMR(
        Sa=Sa[:L],
        Sb=Sb[:L],
        S=S[:L],
        select=select[:L],
        rmse=rmse[:L],
        C1=C[0],
        C2=C[1] if order2 else None,
        C3=C[2] if order3 else None,
        f0=f0[:L],
        streamed=b < N,
    )
