"""Internal helpers for RS-HDMR sensitivity analysis.

Implements pure-JAX cubic B-spline basis evaluation, tensor product construction,
regularized least squares, backfitting for first-order terms, ANCOVA decomposition,
and F-test model selection.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# B-spline basis
# ---------------------------------------------------------------------------


def _bspline_basis(x: Array, m: int) -> Array:
    """Evaluate m+3 cubic B-spline basis functions at points x in [0, 1].

    Uses the standard piecewise cubic polynomial for uniform B-splines with
    knot spacing 1/m, scaled by m^3 to match SALib convention.

    Args:
        x: (N,) points in [0, 1].
        m: number of B-spline intervals.

    Returns:
        B: (N, m+3) basis matrix.
    """
    n_basis = m + 3
    i = jnp.arange(n_basis, dtype=x.dtype)
    # u = x*m - (k_i - 2) where k_i = i - 1, so u = x*m - i + 3
    u = x[:, None] * m - i[None, :] + 3.0

    # Standard cubic B-spline on support [0, 4]
    p1 = u**3 / 6.0
    p2 = (-3 * u**3 + 12 * u**2 - 12 * u + 4) / 6.0
    p3 = (3 * u**3 - 24 * u**2 + 60 * u - 44) / 6.0
    p4 = (4 - u) ** 3 / 6.0

    val = jnp.where(
        u < 0,
        0.0,
        jnp.where(
            u < 1,
            p1,
            jnp.where(
                u < 2,
                p2,
                jnp.where(
                    u < 3,
                    p3,
                    jnp.where(u <= 4, p4, 0.0),
                ),
            ),
        ),
    )
    return jnp.maximum(val * (m**3), 0.0)


def _build_B1(X_n: Array, m: int) -> Array:
    """Build first-order B-spline basis for all dimensions.

    Args:
        X_n: (N, D) normalized inputs in [0, 1].
        m: number of B-spline intervals.

    Returns:
        B1: (N, m1, D) where m1 = m + 3.
    """
    # vmap over columns of X_n (dimensions)
    B1_T = jax.vmap(partial(_bspline_basis, m=m))(X_n.T)  # (D, N, m1)
    return B1_T.transpose(1, 2, 0)  # (N, m1, D)


def _build_B2(B1: Array, c2: Array, beta: Array) -> Array:
    """Build second-order tensor product basis.

    Args:
        B1: (N, m1, D) first-order basis.
        c2: (n2, 2) int array of parameter index combinations.
        beta: (m1^2, 2) tensor-product basis index table.

    Returns:
        B2: (N, m1^2, n2) second-order basis.
    """
    left = B1[:, :, c2[:, 0]][:, beta[:, 0], :]  # (N, m2, n2)
    right = B1[:, :, c2[:, 1]][:, beta[:, 1], :]  # (N, m2, n2)
    return left * right


def _build_B3(B1: Array, c3: Array, beta: Array) -> Array:
    """Build third-order tensor product basis.

    Args:
        B1: (N, m1, D) first-order basis.
        c3: (n3, 3) int array of parameter index combinations.
        beta: (m1^3, 3) tensor-product basis index table.

    Returns:
        B3: (N, m1^3, n3) third-order basis.
    """
    a = B1[:, :, c3[:, 0]][:, beta[:, 0], :]  # (N, m3, n3)
    b = B1[:, :, c3[:, 1]][:, beta[:, 1], :]
    c = B1[:, :, c3[:, 2]][:, beta[:, 2], :]
    return a * b * c


# ---------------------------------------------------------------------------
# First-order fitting with backfitting
# ---------------------------------------------------------------------------


def _fit_first_order(
    B1: Array,
    Y_res: Array,
    m1: int,
    n1: int,
    maxiter: int,
    lambdax: float,
) -> tuple[Array, Array, Array]:
    """Fit first-order component functions via regularized least squares + backfitting.

    Args:
        B1: (R, m1, n1) first-order basis (subsampled).
        Y_res: (R,) residuals (Y - f0).
        m1: number of basis functions per dimension.
        n1: number of first-order terms (= D).
        maxiter: max backfitting iterations.
        lambdax: Tikhonov regularization parameter.

    Returns:
        Y_em1: (R, n1) first-order emulated contributions.
        Y_res_out: (R,) updated residuals after subtracting first-order terms.
        C1: (m1, n1) fitted coefficients.
    """
    lam_eye = lambdax * jnp.eye(m1)

    # Precompute solver matrices: T1[j] = (B1_j^T B1_j + lam*I)^{-1} B1_j^T
    # BtB: (n1, m1, m1), B1_t: (n1, m1, R)
    B1_t = B1.transpose(2, 1, 0)  # (n1, m1, R)
    BtB = jnp.einsum("jmr,jnr->jmn", B1_t, B1_t)  # (n1, m1, m1)

    T1 = jax.vmap(lambda btb, bt: jnp.linalg.solve(btb + lam_eye, bt))(
        BtB,
        B1_t,
    )  # (n1, m1, R)

    # Initial individual fit: C1[:, j] = T1[j] @ Y_res
    C1 = jnp.einsum("jmr,r->jm", T1, Y_res).T  # (m1, n1)

    # Backfitting via while_loop
    var_old = jnp.sum(jnp.square(C1), axis=0)  # (n1,)

    def _update_j(C1: Array, j: int) -> tuple[Array, None]:
        all_contrib = jnp.einsum("rmj,mj->r", B1, C1)  # (R,)
        j_contrib = B1[:, :, j] @ C1[:, j]  # (R,)
        Y_r = Y_res - all_contrib + j_contrib
        C1 = C1.at[:, j].set(T1[j] @ Y_r)
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

    # Final contributions and residual update
    Y_em1 = jnp.einsum("rmj,mj->rj", B1, C1)  # (R, n1)
    Y_res_out = Y_res - jnp.sum(Y_em1, axis=1)  # (R,)
    return Y_em1, Y_res_out, C1


# ---------------------------------------------------------------------------
# Higher-order fitting (no backfitting)
# ---------------------------------------------------------------------------


def _fit_higher_order(
    B: Array,
    Y_res: Array,
    n_terms: int,
    m_basis: int,
    lambdax: float,
) -> tuple[Array, Array, Array]:
    """Fit second- or third-order terms via regularized least squares.

    Args:
        B: (R, m_basis, n_terms) basis matrix.
        Y_res: (R,) residuals.
        n_terms: number of terms at this order.
        m_basis: number of basis functions per term.
        lambdax: regularization parameter.

    Returns:
        Y_em: (R, n_terms) contributions.
        Y_res_out: (R,) updated residuals.
        C: (m_basis, n_terms) coefficients.
    """
    lam_eye = lambdax * jnp.eye(m_basis)
    BtB = jnp.einsum("rmj,rnj->jmn", B, B)  # (n_terms, m_basis, m_basis)
    BtY = jnp.einsum("rmj,r->jm", B, Y_res)  # (n_terms, m_basis)

    C = jax.vmap(lambda btb, bty: jnp.linalg.solve(btb + lam_eye, bty))(
        BtB,
        BtY,
    )  # (n_terms, m_basis)
    C = C.T  # (m_basis, n_terms)

    Y_em = jnp.einsum("rmj,mj->rj", B, C)  # (R, n_terms)
    Y_res_out = Y_res - jnp.sum(Y_em, axis=1)
    return Y_em, Y_res_out, C


# ---------------------------------------------------------------------------
# ANCOVA decomposition
# ---------------------------------------------------------------------------


def _ancova(Y: Array, Y_em: Array, V_Y: Array) -> tuple[Array, Array, Array]:
    """ANCOVA decomposition of sensitivity indices.

    Args:
        Y: (R,) model output (subsampled).
        Y_em: (R, n) emulated term contributions.
        V_Y: scalar variance of Y.

    Returns:
        S:  (n,) total sensitivity per term.
        Sa: (n,) structural (uncorrelated) contribution.
        Sb: (n,) correlative contribution.
    """
    # Structural: variance of each emulated term / total variance
    V_Y = jnp.maximum(V_Y, 1e-30)
    Sa = jnp.var(Y_em, axis=0) / V_Y  # (n,)

    # Total: covariance of each emulated term with actual Y / total variance
    Y_em_c = Y_em - jnp.mean(Y_em, axis=0, keepdims=True)
    Y_c = Y - jnp.mean(Y)
    S = jnp.mean(Y_em_c * Y_c[:, None], axis=0) / V_Y  # (n,)

    # Correlative: covariance of each term with sum of all other terms / V_Y
    Y0 = jnp.sum(Y_em, axis=1)  # (R,)
    Y0_minus = Y0[:, None] - Y_em  # (R, n)
    Y0m_c = Y0_minus - jnp.mean(Y0_minus, axis=0, keepdims=True)
    Sb = jnp.mean(Y_em_c * Y0m_c, axis=0) / V_Y  # (n,)

    return S, Sa, Sb


# ---------------------------------------------------------------------------
# F-test
# ---------------------------------------------------------------------------


def _f_ppf(q: float, d1: float, d2: float) -> float:
    """Compute F-distribution percent point function via bisection on betainc.

    Pure JAX, no scipy dependency. Only intended for scalar arguments computed
    once outside JIT.
    """
    a, b = d1 / 2.0, d2 / 2.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if float(jax.scipy.special.betainc(a, b, mid)) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    x = (lo + hi) / 2.0
    return float(d2 * x / (d1 * (1.0 - x + 1e-30)))


def _compute_f_crits(alpha: float, m1: int, m2: int, m3: int, R: int) -> Array:
    """Precompute F critical values for each order (outside JIT).

    Returns:
        f_crits: (3,) array [f_crit_order1, f_crit_order2, f_crit_order3].
    """
    crits = []
    for p in (m1, m2, m3):
        if R > p:
            crits.append(_f_ppf(alpha, float(p), float(R - p)))
        else:
            crits.append(float("inf"))
    return jnp.array(crits)


def _f_test(
    Y: Array,
    f0: Array,
    Y_em: Array,
    R: int,
    m1: int,
    m2: int,
    m3: int,
    n1: int,
    n2: int,
    n3: int,
    n: int,
    f_crits: Array,
) -> Array:
    """F-test for model selection (JIT-compatible).

    Args:
        Y: (R,) model output.
        f0: scalar mean.
        Y_em: (R, n) emulated contributions per term.
        R: number of samples.
        m1, m2, m3: basis sizes for orders 1, 2, 3.
        n1, n2, n3, n: term counts.
        f_crits: (3,) precomputed critical F values per order.

    Returns:
        select: (n,) binary array, 1.0 if term is significant.
    """
    # Null residuals per order: each order's null includes all lower-order terms
    Y_res_order0 = Y - f0
    Y_res_order1 = Y_res_order0 - jnp.sum(Y_em[:, :n1], axis=1)
    Y_res_order2 = Y_res_order1 - jnp.sum(Y_em[:, n1 : n1 + n2], axis=1)

    SSR_nulls = jnp.array([
        jnp.sum(jnp.square(Y_res_order0)),
        jnp.sum(jnp.square(Y_res_order1)),
        jnp.sum(jnp.square(Y_res_order2)),
    ])

    term_idx = jnp.arange(n)
    p1 = jnp.where(
        term_idx < n1,
        m1,
        jnp.where(term_idx < n1 + n2, m2, m3),
    ).astype(jnp.float32)
    f_crit_per_term = jnp.where(
        term_idx < n1,
        f_crits[0],
        jnp.where(term_idx < n1 + n2, f_crits[1], f_crits[2]),
    )
    order_idx = jnp.where(term_idx < n1, 0, jnp.where(term_idx < n1 + n2, 1, 2))

    def _test_term(i: int) -> Array:
        SSR0 = SSR_nulls[order_idx[i]]
        Y_res_null = jnp.where(
            order_idx[i] == 0,
            Y_res_order0,
            jnp.where(order_idx[i] == 1, Y_res_order1, Y_res_order2),
        )
        Y_res1 = Y_res_null - Y_em[:, i]
        SSR1 = jnp.sum(jnp.square(Y_res1))
        p = p1[i]
        R_minus_p = jnp.maximum(R - p, 1.0)
        F_stat = ((SSR0 - SSR1) / p) / (SSR1 / R_minus_p)
        return (F_stat > f_crit_per_term[i]).astype(jnp.float32)

    select = jax.vmap(_test_term)(jnp.arange(n))  # (n,)
    return select


# ---------------------------------------------------------------------------
# Single bootstrap iteration kernel
# ---------------------------------------------------------------------------


def _make_hdmr_kernel(
    maxorder: int,
    m1: int,
    n1: int,
    maxiter: int,
    m2: int,
    m3: int,
    n2: int,
    n3: int,
    n: int,
    lambdax: float,
    R: int,
):
    """Create an HDMR fitting kernel for a given maxorder.

    All integer/float parameters are captured in the closure so they are
    concrete when this kernel is wrapped by a cached outer ``jit(vmap(...))``.

    Returns a function: kernel(B1_sub, B2_sub, B3_sub, Y_sub, f_crits)
    """

    if maxorder == 1:

        def kernel(B1_sub, _B2_sub, _B3_sub, Y_sub, f_crits):
            f0 = jnp.mean(Y_sub)
            V_Y = jnp.var(Y_sub)
            Y_res = Y_sub - f0

            Y_em1, Y_res, C1 = _fit_first_order(
                B1_sub,
                Y_res,
                m1,
                n1,
                maxiter,
                lambdax,
            )
            Y_em = Y_em1

            S, Sa, Sb = _ancova(Y_sub, Y_em, V_Y)
            select = _f_test(
                Y_sub,
                f0,
                Y_em,
                R,
                m1,
                m2,
                m3,
                n1,
                n2,
                n3,
                n,
                f_crits,
            )
            Y_pred = f0 + jnp.sum(Y_em, axis=1)
            rmse = jnp.sqrt(jnp.mean(jnp.square(Y_sub - Y_pred)))
            dummy = jnp.zeros((1, 1))
            return Sa, Sb, S, select, rmse, C1, dummy, dummy, f0

        return kernel

    elif maxorder == 2:

        def kernel(B1_sub, B2_sub, _B3_sub, Y_sub, f_crits):
            f0 = jnp.mean(Y_sub)
            V_Y = jnp.var(Y_sub)
            Y_res = Y_sub - f0

            Y_em1, Y_res, C1 = _fit_first_order(
                B1_sub,
                Y_res,
                m1,
                n1,
                maxiter,
                lambdax,
            )
            Y_em2, Y_res, C2 = _fit_higher_order(
                B2_sub,
                Y_res,
                n2,
                m2,
                lambdax,
            )
            Y_em = jnp.concatenate([Y_em1, Y_em2], axis=1)

            S, Sa, Sb = _ancova(Y_sub, Y_em, V_Y)
            select = _f_test(
                Y_sub,
                f0,
                Y_em,
                R,
                m1,
                m2,
                m3,
                n1,
                n2,
                n3,
                n,
                f_crits,
            )
            Y_pred = f0 + jnp.sum(Y_em, axis=1)
            rmse = jnp.sqrt(jnp.mean(jnp.square(Y_sub - Y_pred)))
            dummy = jnp.zeros((1, 1))
            return Sa, Sb, S, select, rmse, C1, C2, dummy, f0

        return kernel

    else:  # maxorder == 3

        def kernel(B1_sub, B2_sub, B3_sub, Y_sub, f_crits):
            f0 = jnp.mean(Y_sub)
            V_Y = jnp.var(Y_sub)
            Y_res = Y_sub - f0

            Y_em1, Y_res, C1 = _fit_first_order(
                B1_sub,
                Y_res,
                m1,
                n1,
                maxiter,
                lambdax,
            )
            Y_em2, Y_res, C2 = _fit_higher_order(
                B2_sub,
                Y_res,
                n2,
                m2,
                lambdax,
            )
            Y_em3, _, C3 = _fit_higher_order(
                B3_sub,
                Y_res,
                n3,
                m3,
                lambdax,
            )
            Y_em = jnp.concatenate([Y_em1, Y_em2, Y_em3], axis=1)

            S, Sa, Sb = _ancova(Y_sub, Y_em, V_Y)
            select = _f_test(
                Y_sub,
                f0,
                Y_em,
                R,
                m1,
                m2,
                m3,
                n1,
                n2,
                n3,
                n,
                f_crits,
            )
            Y_pred = f0 + jnp.sum(Y_em, axis=1)
            rmse = jnp.sqrt(jnp.mean(jnp.square(Y_sub - Y_pred)))
            return Sa, Sb, S, select, rmse, C1, C2, C3, f0

        return kernel
