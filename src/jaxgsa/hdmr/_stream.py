"""Row-streamed RS-HDMR fit.

Reproduces the full-N fit in :mod:`jaxgsa.hdmr._engine` without ever holding
the full-N B-spline bases ``B1``/``B2``/``B3`` resident. Every quantity the
per-slice kernel computes is a per-row reduction over the N samples:

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

The streamed result therefore matches the in-memory kernel exactly in real
arithmetic; in float32 the two differ only by summation order.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import Array

from jaxgsa._core.batching import resolve_batch_size
from jaxgsa.hdmr._engine import _build_B1, _build_B2, _build_B3, term_order_map

# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------


def _full_fit_bytes(
    N: int,
    D: int,
    m1: int,
    m2: int,
    m3: int,
    n1: int,
    n2: int,
    n3: int,
    n: int,
    itemsize: int,
) -> int:
    """Estimate the resident bytes of the full-N (non-streamed) HDMR fit.

    Derived from the arrays the in-memory path actually materializes:

    - the shared full-N bases: ``B1`` (N, m1, D), ``B2`` (N, m2, n2) and
      ``B3`` (N, m3, n3) (a (N, 1, 1) placeholder when an order is inactive);
    - per vmap lane, ``_fit_first_order`` holds the transposed basis ``B1_t``
      and the solver matrices ``T1``, each (n1, m1, N) -> ``2 * n1 * m1 * N``;
    - per vmap lane, the ANCOVA/F-test stage keeps ~5 arrays of shape (N, n)
      live at its peak (``Y_em``, its centered copy, the leave-one-out sums
      ``Y0_minus``, their centered copy, and the F-test per-term residuals)
      -> ``5 * N * n``. That is the same constant the automatic
      ``slice_chunk_size`` derivation uses.

    The per-lane terms are counted once: slice chunking can shrink the lane
    count to 1 but can never shrink the shared full-N bases, so this is the
    irreducible footprint of the non-streamed path.

    Args:
        N: Number of sample rows.
        D: Number of input parameters.
        m1: First-order basis size (m + 3).
        m2: Second-order tensor-product basis size (m1**2).
        m3: Third-order tensor-product basis size (m1**3).
        n1: Number of first-order terms (= D).
        n2: Number of second-order terms.
        n3: Number of third-order terms.
        n: Total number of terms (n1 + n2 + n3).
        itemsize: Bytes per element of the working dtype.

    Returns:
        Estimated peak resident bytes of the full-N fit.
    """
    b1_elems = N * m1 * D
    b2_elems = N * m2 * n2 if n2 > 0 else N
    b3_elems = N * m3 * n3 if n3 > 0 else N
    lane_elems = N * (2 * n1 * m1 + 5 * n)
    return itemsize * (b1_elems + b2_elems + b3_elems + lane_elems)


def _streamed_bytes_per_row(
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
    """Per-row transient bytes of one streamed batch.

    Each batch materializes ``B1`` (m1 * D per row), ``B2`` with its two
    gather temporaries (3 * m2 * n2), and ``B3`` with its three gather
    temporaries (4 * m3 * n3). The final statistics pass adds the per-term
    contributions ``F`` plus ~2 elementwise temporaries of the same shape
    (3 * n * L per row).

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
        Estimated transient bytes needed per streamed row.
    """
    elems = m1 * D + 3 * m2 * n2 + 4 * m3 * n3 + 3 * n * L
    return itemsize * elems


# ---------------------------------------------------------------------------
# Gram-space solvers (mirror _engine._fit_first_order / _fit_higher_order)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _get_gram_backfit(m1: int, n1: int, maxiter: int, lambdax: float):
    """Build and cache the vmapped Gram-space first-order backfit.

    Rewrites :func:`jaxgsa.hdmr._engine._fit_first_order` in coefficient space.
    With ``M_j = G1[j, j] + lambdax * I``, the kernel's solver matrices give
    ``T1[j] @ v = M_j^{-1} (B1_j^T v)``, so every backfit step only needs
    ``B1_j^T`` of the current partial residual:

    ``B1_j^T Y_r = b1res_j - sum_k G1[j, k] C1_k + G1[j, j] C1_j``.

    The initial fit, the sweep order, and the convergence test (max change
    in per-dimension coefficient energy vs the same 1e-3 tolerance and
    ``maxiter``) are identical to the in-memory kernel.

    Returns:
        A jitted function ``(G1, G1_diag, b1res) -> C1``. ``G1``
        (n1, m1, n1, m1) and ``G1_diag`` (n1, m1, m1) are shared across lanes,
        and ``b1res`` is batched as (L, n1, m1). The result is (L, m1, n1).
    """

    def backfit(G1: Array, G1_diag: Array, b1res: Array) -> Array:
        lam_eye = lambdax * jnp.eye(m1)
        M = G1_diag + lam_eye  # (n1, m1, m1)

        # Initial fit: C1_j = M_j^{-1} B1_j^T Y_res, one solve per dimension.
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
            _, _, varmax, it = state
            return (varmax > 1e-3) & (it < maxiter)

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

    Mirrors :func:`jaxgsa.hdmr._engine._fit_higher_order`: each term is solved
    independently, ``C_j = (B_j^T B_j + lambdax * I)^{-1} B_j^T Y_res``, from
    the accumulated per-term Gram blocks and moments.

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
# Streamed fit driver
# ---------------------------------------------------------------------------


def _fit_hdmr_streamed(
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
) -> tuple[Array, Array, Array, Array, Array, Array, Array | None, Array | None, Array]:
    """Fit RS-HDMR by streaming row batches; exact up to f32 summation order.

    Runs ``1 + (maxorder - 1) + 1`` passes over row batches of
    ``X_n``/``Y_nl``. Each pass rebuilds the per-batch B-spline bases, so a
    full-N basis is never resident:

    1. Accumulate the first-order cross-Gram ``G1``, basis column sums, the
       per-lane moments ``B1^T Y`` and the Y statistics (mean, shifted
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

    Returns:
        Tuple ``(Sa, Sb, S, select, rmse, C1, C2, C3, f0)`` with lane-leading
        shapes: (L, n) indices and selection, (L,) rmse and f0, (L, m1, n1)
        C1, (L, m2, n2) C2 or None, (L, m3, n3) C3 or None. These match the
        vmapped in-memory kernel's concatenated outputs.
    """
    N, D = X_n.shape
    L = Y_nl.shape[1]
    dtype = jnp.result_type(X_n.dtype, Y_nl.dtype)
    itemsize = jnp.dtype(dtype).itemsize

    bytes_per_row = _streamed_bytes_per_row(D, m1, m2, m3, n2, n3, n, L, itemsize)
    b = resolve_batch_size(bytes_per_row, N, batch_size)

    order2 = maxorder >= 2 and n2 > 0
    order3 = maxorder >= 3 and n3 > 0

    # ---- Pass 1: first-order Gram + moments and Y statistics -------------
    G1 = jnp.zeros((n1, m1, n1, m1), dtype=dtype)
    colsum1 = jnp.zeros((m1, n1), dtype=dtype)
    b1 = jnp.zeros((L, n1, m1), dtype=dtype)
    sum_y = jnp.zeros((L,), dtype=dtype)
    # Shift by the first row before squaring: var(Y) = var(Y - c) for any
    # constant c, and c = Y[0] removes the bulk of the mean so the streamed
    # E[z^2] - E[z]^2 form stays well-conditioned in float32. A constant
    # slice gives z = 0 exactly, so V_Y == 0 detection matches the kernel.
    shift = Y_nl[0]
    sum_z = jnp.zeros((L,), dtype=dtype)
    sum_z2 = jnp.zeros((L,), dtype=dtype)
    for i in range(0, N, b):
        B1_b = _build_B1(X_n[i : i + b], m)  # (b, m1, D)
        Y_b = Y_nl[i : i + b]  # (b, L)
        G1 = G1 + jnp.einsum("rmj,rnk->jmkn", B1_b, B1_b)
        colsum1 = colsum1 + jnp.sum(B1_b, axis=0)
        b1 = b1 + jnp.einsum("rmj,rl->ljm", B1_b, Y_b)
        sum_y = sum_y + jnp.sum(Y_b, axis=0)
        Z = Y_b - shift
        sum_z = sum_z + jnp.sum(Z, axis=0)
        sum_z2 = sum_z2 + jnp.sum(Z * Z, axis=0)

    f0 = sum_y / N  # (L,)
    V_Y = jnp.maximum(sum_z2 / N - jnp.square(sum_z / N), 0.0)  # (L,)

    # b1res_j = B1_j^T (Y - f0) = b1_j - f0 * colsum(B1_j).
    b1res = b1 - f0[:, None, None] * colsum1.T[None]  # (L, n1, m1)
    diag = jnp.arange(n1)
    G1_diag = G1[diag, :, diag, :]  # (n1, m1, m1)
    C1 = _get_gram_backfit(m1, n1, maxiter, lambdax)(G1, G1_diag, b1res)  # (L, m1, n1)

    # ---- Pass 2/3: higher orders on the exact sequential residual ---------
    C2: Array | None = None
    C3: Array | None = None
    colsum2 = jnp.zeros((m2, n2), dtype=dtype) if order2 else None
    colsum3 = jnp.zeros((m3, n3), dtype=dtype) if order3 else None
    if order2:
        BtB2 = jnp.zeros((n2, m2, m2), dtype=dtype)
        BtY2 = jnp.zeros((L, n2, m2), dtype=dtype)
        for i in range(0, N, b):
            B1_b = _build_B1(X_n[i : i + b], m)
            B2_b = _build_B2(B1_b, c2_idx, beta2)  # (b, m2, n2)
            Y_res1_b = (
                Y_nl[i : i + b] - f0[None, :] - jnp.einsum("rmj,lmj->rl", B1_b, C1)
            )  # (b, L)
            BtB2 = BtB2 + jnp.einsum("rmj,rnj->jmn", B2_b, B2_b)
            BtY2 = BtY2 + jnp.einsum("rmj,rl->ljm", B2_b, Y_res1_b)
            assert colsum2 is not None
            colsum2 = colsum2 + jnp.sum(B2_b, axis=0)
        C2 = _get_gram_higher_solver(m2, lambdax)(BtB2, BtY2)  # (L, m2, n2)
    if order3:
        BtB3 = jnp.zeros((n3, m3, m3), dtype=dtype)
        BtY3 = jnp.zeros((L, n3, m3), dtype=dtype)
        for i in range(0, N, b):
            B1_b = _build_B1(X_n[i : i + b], m)
            B2_b = _build_B2(B1_b, c2_idx, beta2)
            B3_b = _build_B3(B1_b, c3_idx, beta3)  # (b, m3, n3)
            assert C2 is not None
            Y_res2_b = (
                Y_nl[i : i + b]
                - f0[None, :]
                - jnp.einsum("rmj,lmj->rl", B1_b, C1)
                - jnp.einsum("rmj,lmj->rl", B2_b, C2)
            )
            BtB3 = BtB3 + jnp.einsum("rmj,rnj->jmn", B3_b, B3_b)
            BtY3 = BtY3 + jnp.einsum("rmj,rl->ljm", B3_b, Y_res2_b)
            assert colsum3 is not None
            colsum3 = colsum3 + jnp.sum(B3_b, axis=0)
        C3 = _get_gram_higher_solver(m3, lambdax)(BtB3, BtY3)  # (L, m3, n3)

    # ---- Term means from column sums (no extra pass needed) ---------------
    # mean(f_j) = (1^T B_j C_j) / N; identical (up to reassociation) to the
    # row means the kernel's _ancova computes from the full-N Y_em.
    mean_parts = [jnp.einsum("mj,lmj->lj", colsum1, C1) / N]
    if order2:
        assert colsum2 is not None and C2 is not None
        mean_parts.append(jnp.einsum("mj,lmj->lj", colsum2, C2) / N)
    if order3:
        assert colsum3 is not None and C3 is not None
        mean_parts.append(jnp.einsum("mj,lmj->lj", colsum3, C3) / N)
    mean_f = jnp.concatenate(mean_parts, axis=1)  # (L, n)
    # mean of the leave-one-out sums Y0_minus_j = Y0 - f_j.
    mean_y0m = jnp.sum(mean_f, axis=1, keepdims=True) - mean_f  # (L, n)

    # ---- Final pass: ANCOVA / F-test / RMSE reductions ---------------------
    acc_var_f = jnp.zeros((L, n), dtype=dtype)  # sum (f - mean_f)^2
    acc_cov_fy = jnp.zeros((L, n), dtype=dtype)  # sum (f - mean_f)(Y - f0)
    acc_cov_fy0m = jnp.zeros((L, n), dtype=dtype)  # sum (f - mean_f)(Y0m - mean)
    acc_ssr = jnp.zeros((L, 3), dtype=dtype)  # SSR of the 3 null models
    acc_cross = jnp.zeros((L, n), dtype=dtype)  # sum r_null * f, per term
    acc_fsq = jnp.zeros((L, n), dtype=dtype)  # sum f^2, per term
    acc_res2 = jnp.zeros((L,), dtype=dtype)  # sum (Y - f0 - Y0)^2
    for i in range(0, N, b):
        B1_b = _build_B1(X_n[i : i + b], m)
        Y_b = Y_nl[i : i + b]  # (b, L)
        f_parts = [jnp.einsum("rmj,lmj->rlj", B1_b, C1)]  # (b, L, n1)
        if order2:
            assert C2 is not None
            B2_b = _build_B2(B1_b, c2_idx, beta2)
            f_parts.append(jnp.einsum("rmj,lmj->rlj", B2_b, C2))
        if order3:
            assert C3 is not None
            B3_b = _build_B3(B1_b, c3_idx, beta3)
            f_parts.append(jnp.einsum("rmj,lmj->rlj", B3_b, C3))
        F = jnp.concatenate(f_parts, axis=2)  # (b, L, n)

        fc = F - mean_f[None]  # centered contributions
        acc_var_f = acc_var_f + jnp.sum(fc * fc, axis=0)

        r0 = Y_b - f0[None, :]  # (b, L): Y centered by its mean
        acc_cov_fy = acc_cov_fy + jnp.sum(fc * r0[:, :, None], axis=0)

        y0 = jnp.sum(F, axis=2)  # (b, L) total emulator (sans f0)
        y0m_c = (y0[:, :, None] - F) - mean_y0m[None]
        acc_cov_fy0m = acc_cov_fy0m + jnp.sum(fc * y0m_c, axis=0)

        # Hierarchical null-model residuals, as in _engine._f_test.
        r1 = r0 - jnp.sum(F[:, :, :n1], axis=2)
        r2 = r1 - jnp.sum(F[:, :, n1 : n1 + n2], axis=2)
        acc_ssr = acc_ssr + jnp.stack(
            [jnp.sum(r0 * r0, axis=0), jnp.sum(r1 * r1, axis=0), jnp.sum(r2 * r2, axis=0)],
            axis=1,
        )
        # sum r_null * f_i with each term's own null residual.
        cross = jnp.concatenate(
            [
                jnp.sum(r0[:, :, None] * F[:, :, :n1], axis=0),
                jnp.sum(r1[:, :, None] * F[:, :, n1 : n1 + n2], axis=0),
                jnp.sum(r2[:, :, None] * F[:, :, n1 + n2 :], axis=0),
            ],
            axis=1,
        )
        acc_cross = acc_cross + cross
        acc_fsq = acc_fsq + jnp.sum(F * F, axis=0)

        r_pred = r0 - y0  # full-model residual Y - (f0 + sum f_j)
        acc_res2 = acc_res2 + jnp.sum(r_pred * r_pred, axis=0)

    # ---- ANCOVA indices (mirrors _engine._ancova) ---------------------------
    zero_var = (V_Y == 0)[:, None]  # (L, 1)
    V_Y_safe = jnp.maximum(V_Y, 1e-30)[:, None]
    Sa = jnp.where(zero_var, jnp.nan, (acc_var_f / N) / V_Y_safe)
    S = jnp.where(zero_var, jnp.nan, (acc_cov_fy / N) / V_Y_safe)
    Sb = jnp.where(zero_var, jnp.nan, (acc_cov_fy0m / N) / V_Y_safe)

    # ---- F-test (mirrors _engine._f_test) ----------------------------------
    # The per-term order map comes from _engine.term_order_map, the same
    # function _f_test calls, so the two paths cannot select different terms
    # because one copy of the expressions drifted.
    p1, f_crit_per_term, order_idx = term_order_map(n, n1, n2, m1, m2, m3, f_crits)
    SSR0 = acc_ssr[:, order_idx]  # (L, n) null SSR per term
    # SSR1 = sum (r_null - f_i)^2 = SSR0 - 2 sum r_null f_i + sum f_i^2.
    SSR1 = SSR0 - 2.0 * acc_cross + acc_fsq
    R_minus_p = jnp.maximum(N - p1, 1.0)
    F_stat = ((SSR0 - SSR1) / p1) / (SSR1 / R_minus_p)
    select = (F_stat > f_crit_per_term[None]).astype(jnp.float32)  # (L, n)

    rmse = jnp.sqrt(acc_res2 / N)  # (L,)

    return Sa, Sb, S, select, rmse, C1, C2, C3, f0
