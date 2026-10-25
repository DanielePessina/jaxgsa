"""Pure-JAX VKOGA: greedy kernel interpolation with a Newton basis.

Implements the surrogate-based sensitivity analysis of Hilhorst et al.
(2024), who name their kernel expansion VKOGA (Vectorial Kernel Orthogonal
Greedy Algorithm). The surrogate is a Gaussian RBF expansion. Its centres are
chosen by the P-greedy rule, which maximises the power function and does not
read ``Y``; this is the target-independent selection rule of De Marchi,
Schaback & Wendland (2005), not the target-dependent f-greedy rule of Wirtz
& Haasdonk (2013), whose own "VKOGA" name this module borrows for
consistency with Hilhorst et al. Its coefficients come from an
RKHS-regularised least-squares solve over the selected centres, following
Santin & Haasdonk's presentation of the Newton basis.

Two properties make this cheap enough to drive a Monte-Carlo sensitivity loop.
First, the P-greedy selection does not depend on ``Y``, so one greedy sweep
serves every output slice. That is the "vectorial" part: one set of centres,
one Newton basis, and one coefficient matrix with a column per slice
(a separable matrix-valued kernel). Second, prediction is a single kernel GEMM
against a handful of centres.

Everything here is a pure function on arrays. There are no result classes and
no ``Problem``. Outputs arrive as ``(n, S)`` with ``S = T*K`` output slices
already flattened by the caller.

The coefficient step solves the regularised least-squares problem in the
Newton basis the greedy builds, not through the kernel-translate normal matrix
``A_nm^T A_nm``. That matrix squares the condition number of the cross-kernel,
which for a Gaussian kernel exceeds what float32 can carry at ordinary
settings; the Newton basis is the stable pivoted-Cholesky factor of the
kernel, so the solve keeps float32 results close to float64 ones. The
remaining float32 limit is the greedy itself: its power function
``1 - sum(V**2)`` reaches float32 rounding for small ``gamma`` or many
centres, and the greedy then stops with fewer centres than float64 selects.
The code carries ``X.dtype`` throughout, so float64 inputs under
``jax.config.update("jax_enable_x64", True)`` need no other change.

References:
    De Marchi, Schaback & Wendland (2005). Adv. Comput. Math. 23:317-330.
        The P-greedy centre-selection rule.
    Santin & Haasdonk (2021). "Kernel Methods for Surrogate Modeling", in
        Model Order Reduction, Volume 1: System- and Data-Driven Methods and
        Algorithms, De Gruyter, pp. 311-354. The Newton basis and the
        RKHS-regularised coefficient solve.
    Wirtz & Haasdonk (2013). Dolomites Res. Notes Approx. 6:83-100. Source of
        the "VKOGA" name, for a target-dependent f-greedy variant this
        module does not implement.
    Hilhorst, Quicken, van de Vosse & Huberts (2024). Int. J. Numer. Meth.
        Biomed. Engng. 40(2):e3797. Surrogate-based sensitivity analysis
        built on this expansion.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, NamedTuple

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

from jaxgsa._core.batching import get_memory_budget

# Paper defaults (Hilhorst et al. 2024): tau_P = 5e-8 on the power function,
# tau_R = 1e-4 on the interpolation residual, lambda = 1e-10 on the RKHS term.
# tau_P is absolute: the Gaussian kernel has K(x, x) = 1, so the power
# function is already scale-free. tau_R is applied relative to each output
# slice's norm, because an absolute residual rule would depend on the scale
# of Y.
_DEFAULT_RIDGE = 1e-10
_DEFAULT_TOL_POWER = 5e-8
_DEFAULT_TOL_RESIDUAL = 1e-4

# Relative jitter on the active diagonal of the Newton-basis normal matrix
# V^T V. With the default lambda = 1e-10 the ridge term does almost nothing,
# and late Newton columns carry little power, so V^T V can still be close to
# singular once many centres are selected.
_JITTER = 1e-8

# How the greedy updates the squared power function after each centre: rebuilt
# from the whole Newton basis ("exact") or by subtracting the new column
# ("incremental"). See jaxgsa.vkoga.analyze.
GreedyUpdate = Literal["exact", "incremental"]

# Narrowest basis width of the greedy loop's first phase (see _phase_widths).
_FIRST_PHASE_WIDTH = 64

# Relative jitter, in machine epsilons of the working dtype, for the retry
# solve after a non-finite first solve.
_JITTER_EPS = 10.0


class _VKOGAState(NamedTuple):
    """A fitted VKOGA surrogate.

    The arrays are allocated at the static ``max_centers`` size, because the
    greedy loop runs under ``lax.while_loop`` and JAX needs static shapes.
    Slots beyond ``n_centers`` carry a duplicated centre with an exactly zero
    coefficient row. :func:`_predict_vkoga` is therefore correct whether or not
    the caller first slices the state down to ``n_centers`` rows.

    Attributes:
        centers: Selected centres, shape ``(m, D)`` with ``m = max_centers``
            as returned.
        coefficients: Expansion coefficients, shape ``(m, S)``, one column per
            output slice. Rows ``>= n_centers`` are zero.
        gamma: Scalar Gaussian shape parameter the surrogate was fitted with.
        ridge: Scalar RKHS regularisation parameter (the paper's lambda).
        n_centers: Number of centres the greedy loop actually selected.
        rmse: Root-mean-square error of the fit on its own training rows, shape
            ``(S,)``. It is a diagnostic, not a generalisation estimate.
    """

    centers: Array
    coefficients: Array
    gamma: Array
    ridge: Array
    n_centers: Array
    rmse: Array


def _gaussian_kernel(X1: Array, X2: Array, gamma: Array | float) -> Array:
    """Evaluate the Gaussian RBF kernel between two point sets.

    ``K(x1, x2) = exp(-gamma * ||x1 - x2||^2)``.

    Args:
        X1: First point set, shape ``(n1, D)``.
        X2: Second point set, shape ``(n2, D)``.
        gamma: Positive shape parameter. Larger means narrower kernels.

    Returns:
        Kernel matrix, shape ``(n1, n2)``.
    """
    # Squared distances via the ||a||^2 + ||b||^2 - 2a.b expansion: one GEMM
    # instead of an (n1, n2, D) difference tensor, which is what keeps
    # prediction affordable when n1 runs into the millions. The price is
    # cancellation for near-coincident points. In float32 the expansion can go
    # slightly negative, which would make the kernel exceed 1 and corrupt the
    # power function p2 = 1 - ... below, so clamp at zero.
    sq1 = jnp.sum(X1 * X1, axis=1)
    sq2 = jnp.sum(X2 * X2, axis=1)
    d2 = sq1[:, None] + sq2[None, :] - 2.0 * (X1 @ X2.T)
    return jnp.exp(-gamma * jnp.maximum(d2, 0.0))


@functools.partial(jax.jit, static_argnames=("max_centers", "greedy_update"))
def _select_centers(
    X: Array,
    Y: Array,
    *,
    gamma: Array | float,
    max_centers: int,
    greedy_update: GreedyUpdate = "exact",
    tol_power: Array | float = _DEFAULT_TOL_POWER,
    tol_residual: Array | float = _DEFAULT_TOL_RESIDUAL,
    train_mask: Array | None = None,
) -> tuple[Array, Array]:
    """Select kernel centres by the P-greedy rule.

    The selection half of :func:`_newton_greedy`, for callers that do not
    need the Newton basis. See there for the rule and the argument contract.

    Returns:
        ``(idx, m)``: the int32 buffer of selected row indices, shape
        ``(max_centers,)``, and the scalar count actually selected.
    """
    idx, m, _ = _newton_greedy(
        X,
        Y,
        gamma=gamma,
        max_centers=max_centers,
        greedy_update=greedy_update,
        tol_power=tol_power,
        tol_residual=tol_residual,
        train_mask=train_mask,
    )
    return idx, m


@functools.partial(jax.jit, static_argnames=("max_centers", "greedy_update"))
def _newton_greedy(
    X: Array,
    Y: Array,
    *,
    gamma: Array | float,
    max_centers: int,
    greedy_update: GreedyUpdate = "exact",
    tol_power: Array | float = _DEFAULT_TOL_POWER,
    tol_residual: Array | float = _DEFAULT_TOL_RESIDUAL,
    train_mask: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Select kernel centres by the P-greedy rule and return the Newton basis.

    Centres are picked one at a time at the current maximiser of the power
    function (Hilhorst et al. 2024, eq. 8). The rule is expressed in the Newton
    basis of Pazouki & Schaback, which keeps the basis nested: adding a centre
    appends one column and never touches the earlier ones. For the Gaussian
    kernel the diagonal is 1, so the squared power function at every training
    point is ``p2 = 1 - rowsum(V**2)`` with ``V`` the Newton basis evaluated at
    those points.

    Selection depends on ``X``, ``gamma``, and the stopping rules. It does not
    depend on ``ridge``. ``Y`` enters only through the residual stopping rule,
    so all output slices share one set of centres. :func:`_cross_validate` uses
    that ridge independence: one sweep per ``(fold, gamma)`` serves the whole
    ridge grid.

    Args:
        X: Training inputs, shape ``(n, D)``.
        Y: Training outputs, shape ``(n, S)``, output slices already flattened.
        gamma: Positive Gaussian shape parameter.
        greedy_update: ``"exact"`` or ``"incremental"``; see
            :func:`jaxgsa.vkoga.analyze`.
        max_centers: Hard cap on the number of centres, and the static size of
            the returned index buffer. Silently capped at ``n``.
        tol_power: Power-function stopping rule (tau_P). Stop once
            ``sqrt(max(p2)) <= tol_power``. It is absolute, because
            ``K(x, x) = 1`` makes the power function scale-free.
        tol_residual: Relative residual stopping rule (tau_R). Stop once every
            slice's residual L2 norm falls to ``tol_residual`` times that
            slice's ``||Y||`` over the training rows. A zero-norm slice counts
            as converged from the start.
        train_mask: Boolean mask, shape ``(n,)``, or ``None``. ``False`` rows
            are excluded as candidate centres and from the residual rule.
            :func:`_cross_validate` uses it to hold folds out without changing
            shapes.

    Returns:
        ``(idx, m, V)``. ``idx`` is the int32 buffer of selected row indices,
        shape ``(max_centers,)``. ``m`` is the scalar count of centres actually
        selected. Slots ``>= m`` are unused. ``V`` is the Newton basis at every
        row (held-out rows included), shape ``(n, max_centers)``; its columns
        ``>= m`` are zero. :func:`_solve_ridge` solves for the coefficients in
        this basis.
    """
    n, _ = X.shape
    # Never ask for more centres than there are points to pick from: the
    # greedy loop would run out of candidates and stall on masked -inf.
    max_centers = min(max_centers, n)

    mask = jnp.ones((n,), dtype=bool) if train_mask is None else train_mask.astype(bool)

    dtype = X.dtype
    # Rows that must never be chosen as centres: held-out rows start blocked,
    # and each selected point is blocked afterwards so it cannot repeat.
    blocked0 = ~mask
    # Held-out rows contribute nothing to the residual stopping rule either.
    residual0 = jnp.where(mask[:, None], Y, 0.0)

    # Per-slice norms of the training outputs, for the relative residual rule.
    # A zero-norm slice would divide 0/0, so give it scale 1: its ratio is then
    # 0 and it counts as converged.
    y_norm = jnp.sqrt(jnp.sum(residual0**2, axis=0))
    res_scale = jnp.where(y_norm > 0.0, y_norm, 1.0)

    eps = jnp.finfo(dtype).eps
    n_slices = Y.shape[1]
    # The residual rule needs only "is any slice still above tau_R". One
    # slice that is above it answers that, so only that "witness" slice's
    # residual is carried over all n rows. The other slices are carried as
    # their Newton coefficients C, shape (max_centers, S): the residual at
    # the next centre is then one (max_centers,) x (max_centers, S) product,
    # not an (n, S) update per step. The full (n, S) residual is rebuilt,
    # as one GEMM, only when the witness drops to tau_R. Then the rule is
    # decided on every slice and the worst slice becomes the next witness.
    # Y[i] - V[i] @ C rounds differently from a running (n, S) update, so the
    # stopping step can differ by one centre when tau_R lands within rounding
    # of a residual norm. The selection order itself does not read Y.
    rel0 = y_norm / res_scale
    witness0 = jnp.argmax(rel0).astype(jnp.int32)
    # The basis V only has m non-zero columns after m steps, but every step
    # reads all of its columns. So V (and C) start narrow and widen in a
    # few static phases, each its own while loop. The extra columns of a
    # wider phase are exact zeros, which add nothing to the products and
    # sums below. On the checked cases (f32 and f64) the selected centres
    # matched a single full-width loop exactly.
    widths = _phase_widths(max_centers)
    state = (
        jnp.zeros((n, widths[0]), dtype=dtype),  # Newton basis at all points
        jnp.zeros((widths[0], n_slices), dtype=dtype),  # Newton coefficients
        residual0[:, witness0],  # residual of the witness slice
        witness0,
        blocked0,
        jnp.zeros((max_centers,), dtype=jnp.int32),  # selected row indices
        jnp.zeros((), dtype=jnp.int32),  # centres selected so far
        # K(x, x) = 1 for the Gaussian kernel, so the power function starts at
        # 1 everywhere; the empty basis subtracts nothing.
        jnp.ones((), dtype=dtype),
        jnp.max(rel0),
        # Squared power function at every point, carried so the loop body reads
        # it instead of rebuilding it. The empty basis subtracts nothing, so an
        # open point starts at 1 and a blocked point is never a candidate.
        jnp.where(blocked0, -jnp.inf, jnp.ones((n,), dtype=dtype)),
    )

    def _cond(state: tuple) -> Array:
        _, _, _, _, _, _, m, power, res, _ = state
        # All three stopping rules, first to trigger wins.
        return (m < max_centers) & (power > tol_power) & (res > tol_residual)

    def _all_slices(V: Array, C: Array) -> tuple[Array, Array, Array]:
        """Rebuild every slice's residual and pick the worst as the witness."""
        residual = jnp.where(mask[:, None], residual0 - V @ C, 0.0)
        rel = jnp.sqrt(jnp.sum(residual**2, axis=0)) / res_scale
        worst = jnp.argmax(rel).astype(jnp.int32)
        return rel[worst], residual[:, worst], worst

    def _body(state: tuple) -> tuple:
        V, C, r_w, w, blocked, idx, m, _, _, p2 = state

        i = jnp.argmax(p2)
        # p2 = 1 - sum(V^2) is a cancellation: once the basis nearly spans the
        # space the true value is ~0 and float32 rounding pushes it negative.
        # The loop condition guarantees p2[i] > tol_power^2 on entry, so this
        # floor only ever fires as a guard against a NaN/inf division.
        sqrt_p = jnp.sqrt(jnp.maximum(p2[i], eps))

        # Newton basis update (Pazouki & Schaback): orthogonalise the new
        # kernel translate against the existing basis, then normalise by the
        # power function at the new centre.
        k_col = _gaussian_kernel(X, X[i][None, :], gamma)[:, 0]
        V_i = jax.lax.optimization_barrier(V[i])
        v = (k_col - V @ V_i) / sqrt_p
        # At its own centre the new basis function equals the power there,
        # so set that entry exactly. Recomputed as (1 - V_i @ V_i) / sqrt_p it
        # can round to zero in float32 once the power nears rounding level,
        # and a zero there makes the triangular factor in _solve_ridge
        # singular.
        v = v.at[i].set(sqrt_p)

        # Interpolation residual in the same basis: one coefficient row per
        # output slice, since all slices share this basis. The residual at
        # the new centre is Y[i] minus the interpolant so far, V[i] @ C. The
        # witness residual is re-masked after its update: v is dense, so the
        # update writes minus-interpolant values into held-out rows. Without
        # the mask those rows accumulate and the stopping norm below never
        # converges inside cross-validation.
        c = (residual0[i] - V_i @ C) / sqrt_p  # (S,)
        C = C.at[m].set(c)
        r_w = jnp.where(mask, r_w - v * c[w], 0.0)

        V = V.at[:, m].set(v)
        # Also block every exact copy of the new centre. At a copy the true
        # squared power is 0, but the computed 1 - sum(V^2) is rounding noise.
        # In float32 that noise (~1e-7) is above tol_power**2, so the greedy
        # would pick the copy too and make the Gram singular. The test compares
        # coordinates, because the kernel value of a copy is only 1 up to
        # rounding. On data without duplicate rows it blocks only row i.
        blocked = blocked | jnp.all(X == X[i][None, :], axis=1)
        blocked = blocked.at[i].set(True)
        # argmax returns int64 when x64 is enabled; keep the index buffer int32
        # so the scatter never hits a narrowing-cast promotion error.
        idx = idx.at[m].set(i.astype(idx.dtype))
        m = m + 1

        # Stopping quantities for the next iteration. Clamp the power at zero
        # so exhausted candidates (all -inf) terminate instead of producing a
        # NaN under the square root.
        # "exact" rebuilds p2 from the whole basis each step. "incremental"
        # subtracts the new column's square only: one fewer read of V per
        # step, but different rounding, and the greedy is sensitive enough to
        # rounding that it then often picks different centres.
        if greedy_update == "incremental":
            p2_next = jnp.where(blocked, -jnp.inf, p2 - v * v)
        else:
            p2_next = jnp.where(blocked, -jnp.inf, 1.0 - jnp.sum(V * V, axis=1))
        power = jnp.sqrt(jnp.maximum(jnp.max(p2_next), 0.0))
        # Held-out rows are re-masked to exactly zero above, and selected
        # points are interpolated exactly, so summing over all rows equals
        # summing over the active training rows. Relative to each slice's own
        # norm, so the rule is scale-free. While the witness is above tau_R
        # the maximum over the slices is too, so the loop goes on either way.
        res = jnp.sqrt(jnp.sum(r_w**2)) / res_scale[w]
        if n_slices > 1:
            res, r_w, w = jax.lax.cond(
                res > tol_residual,
                lambda: (res, r_w, w),
                lambda: _all_slices(V, C),
            )
        return (V, C, r_w, w, blocked, idx, m, power, res, p2_next)

    for width in widths:
        V, C, *rest = state
        pad = width - V.shape[1]
        state = (
            jnp.pad(V, ((0, 0), (0, pad))),
            jnp.pad(C, ((0, pad), (0, 0))),
            *rest,
        )
        state = jax.lax.while_loop(lambda s, width=width: _cond(s) & (s[6] < width), _body, state)
    V, *_, idx, m, _, _, _ = state
    return idx, m, V


def _phase_widths(max_centers: int) -> tuple[int, ...]:
    """Return the static basis widths the greedy loop grows through.

    Widths double from ``_FIRST_PHASE_WIDTH`` and end at ``max_centers``, so
    a step never reads more than about twice the columns it needs.

    Args:
        max_centers: Final number of basis columns.

    Returns:
        Increasing widths, the last equal to ``max_centers``.
    """
    widths = []
    width = _FIRST_PHASE_WIDTH
    while width < max_centers:
        widths.append(width)
        width *= 2
    widths.append(max_centers)
    return tuple(widths)


class _NormalEquations(NamedTuple):
    """Ridge-independent pieces of the regularised coefficient solve.

    The system is posed in the Newton basis ``V`` the greedy built, not in the
    kernel translates. With ``A_nm = V L^T`` and ``A_mm = L L^T``, where ``L``
    is ``V`` at the centres (lower triangular), the RKHS-regularised problem
    ``min ||Y - A_nm alpha||^2 + ridge * alpha^T A_mm alpha`` becomes
    ``min ||Y - V beta||^2 + ridge * ||beta||^2`` with ``beta = L^T alpha``.
    The two are equal in exact arithmetic. In floating point they are not:
    ``A_nm^T A_nm`` squares the condition number of the cross-kernel, which
    for a Gaussian kernel exceeds ``1 / eps`` of float32 at ordinary settings
    (measured: ``6e8`` on Ishigami, ``N=600``, 60 centres). The Newton basis
    is the stable pivoted-Cholesky factor of the kernel, so ``V^T V`` stays
    well conditioned (``cond(V) = 145`` on the same case).

    Everything here depends on ``X``, ``Y``, the selected centres, and
    ``gamma``, but not on ``ridge``. :func:`_cross_validate` builds one per
    ``(fold, gamma)`` and reuses it for every ridge in the grid. Only the cheap
    assembly-and-solve in :func:`_solve_ridge` repeats.

    Attributes:
        active: Boolean mask of the slots the greedy filled, shape
            ``(max_centers,)``.
        centers: Selected centres, shape ``(max_centers, D)``, padded with
            duplicates.
        L: Newton basis at the centres, lower triangular, shape
            ``(max_centers, max_centers)``. Inactive slots hold the identity.
        VtV: ``V^T V`` over the training rows, shape
            ``(max_centers, max_centers)``.
        VtY: ``V^T Y`` over the training rows, shape ``(max_centers, S)``.
        m: Scalar count of centres actually selected.
    """

    active: Array
    centers: Array
    L: Array
    VtV: Array
    VtY: Array
    m: Array


def _normal_equations(
    X: Array,
    Y: Array,
    idx: Array,
    m: Array,
    V: Array,
    *,
    train_mask: Array | None = None,
) -> _NormalEquations:
    """Build the ridge-independent normal-equation matrices in the Newton basis.

    Args:
        X: Training inputs, shape ``(n, D)``.
        Y: Training outputs, shape ``(n, S)``, output slices already flattened.
        idx: Selected row indices from :func:`_newton_greedy`, shape
            ``(max_centers,)``.
        m: Scalar count of centres actually selected.
        V: Newton basis from :func:`_newton_greedy`, shape
            ``(n, max_centers)``.
        train_mask: Boolean mask, shape ``(n,)``, or ``None``. ``False`` rows
            are excluded from the solve.

    Returns:
        A :class:`_NormalEquations` bundle.
    """
    n, _ = X.shape
    max_centers = idx.shape[0]
    mask = jnp.ones((n,), dtype=bool) if train_mask is None else train_mask.astype(bool)

    # Unused slots point at row 0. Their V columns are exact zeros, and an
    # identity block in L (plus a 1 on their diagonal in _solve_ridge) forces
    # their coefficients to be exactly zero, so the duplicate centre is inert.
    active = jnp.arange(max_centers) < m
    both = active[:, None] & active[None, :]
    # A Newton basis function vanishes at every earlier centre, so V[idx] is
    # lower triangular up to rounding; tril makes that exact.
    L = jnp.where(both, jnp.tril(V[idx]), 0.0) + jnp.diag(jnp.where(active, 0.0, 1.0))
    V_train = jnp.where(mask[:, None], V, 0.0)
    Y_train = jnp.where(mask[:, None], Y, 0.0)
    return _NormalEquations(
        active=active,
        centers=X[idx],
        L=L.astype(V.dtype),
        VtV=V_train.T @ V_train,
        VtY=V_train.T @ Y_train,
        m=m,
    )


def _solve_ridge(eq: _NormalEquations, ridge: Array | float) -> Array:
    """Assemble the regularised system for one ridge value and solve it.

    Solves ``(V^T V + ridge * I + JITTER * diag_mean * I) beta = V^T Y`` and
    maps back to kernel-translate coefficients by ``L^T alpha = beta``.

    Args:
        eq: Ridge-independent matrices from :func:`_normal_equations`.
        ridge: RKHS regularisation parameter (the paper's lambda).

    Returns:
        Expansion coefficients, shape ``(max_centers, S)``. Inactive rows are
        zero.
    """
    G = eq.VtV + jnp.diag(jnp.where(eq.active, ridge, 0.0)).astype(eq.VtV.dtype)
    # Relative jitter: scale-free against the number of rows, which moves the
    # magnitude of V^T V by orders of magnitude.
    diag_mean = jnp.sum(jnp.where(eq.active, jnp.diagonal(G), 0.0)) / jnp.maximum(eq.m, 1)
    beta = jnp.linalg.solve(G + jnp.diag(jnp.where(eq.active, _JITTER * diag_mean, 1.0)), eq.VtY)
    # The default jitter is below float32 eps, so a singular system can still
    # give NaN there. Only then, re-solve with a larger jitter. A finite first
    # solve is returned unchanged, and the retry costs nothing unless it runs.
    finite = jnp.all(jnp.isfinite(beta))
    if isinstance(finite, jax.core.Tracer):
        # Traced (the cross-validation scorer): branch on the device.
        beta = jax.lax.cond(
            finite, lambda: beta, lambda: _retry_solve(G, eq.VtY, eq.active, diag_mean)
        )
    elif not bool(finite):
        # Eager (the final fit): branch on the host. An eager lax.cond would
        # compile again on every call, because its branch closures are new.
        beta = _retry_solve(G, eq.VtY, eq.active, diag_mean)
    return jax.scipy.linalg.solve_triangular(eq.L.T, beta, lower=False)


def _retry_solve(G: Array, rhs: Array, active: Array, diag_mean: Array) -> Array:
    """Solve the ridge system again with a jitter of ``_JITTER_EPS`` epsilons.

    Args:
        G: Regularised normal matrix without jitter, ``(max_centers, max_centers)``.
        rhs: Right-hand side, ``(max_centers, S)``.
        active: Boolean mask of the filled centre slots, ``(max_centers,)``.
        diag_mean: Mean of the active diagonal of ``G``.

    Returns:
        Newton-basis coefficients ``beta``, shape ``(max_centers, S)``.
    """
    eps_jitter = jnp.maximum(_JITTER, _JITTER_EPS * jnp.finfo(G.dtype).eps)
    return jnp.linalg.solve(G + jnp.diag(jnp.where(active, eps_jitter * diag_mean, 1.0)), rhs)


def _solve_coefficients(
    X: Array,
    Y: Array,
    idx: Array,
    m: Array,
    V: Array,
    *,
    gamma: Array | float,
    ridge: Array | float,
    train_mask: Array | None = None,
) -> _VKOGAState:
    """Solve for the expansion coefficients over already-selected centres.

    Coefficients solve the RKHS-regularised least-squares problem over the
    selected centres::

        min_alpha ||Y - A_nm alpha||^2 + ridge * alpha^T A_mm alpha

    ``A_nm`` is the ``(n, m)`` cross-kernel between training points and
    centres, and ``A_mm`` is the ``(m, m)`` centre Gram. When ``m = n`` this
    is the paper's eq. (6), ``(A + ridge*I) alpha = Y``. The solve itself runs
    in the Newton basis ``V`` (see :class:`_NormalEquations`), because the
    kernel-translate normal matrix ``A_nm^T A_nm`` is too ill-conditioned for
    float32.

    A ``JITTER * diag_mean * I`` term (see :func:`_solve_ridge`) is added to
    the Newton-basis system unconditionally. ``diag_mean`` is the mean active
    diagonal of ``V^T V + ridge * I``, so the term acts as an extra ridge of
    ``JITTER * diag_mean``: about 3e-8 to 2e-7 at 1024 training rows on
    uniform inputs. A caller-supplied ``ridge`` below that scale is therefore
    not the regularisation in force, and the reported ``VKOGAResult.ridge``
    undercounts it.

    Args:
        X: Training inputs, shape ``(n, D)``.
        Y: Training outputs, shape ``(n, S)``, output slices already flattened.
        idx: Selected row indices from :func:`_newton_greedy`, shape
            ``(max_centers,)``.
        m: Scalar count of centres actually selected.
        V: Newton basis from :func:`_newton_greedy`, shape
            ``(n, max_centers)``.
        gamma: Positive Gaussian shape parameter the centres were selected
            with.
        ridge: RKHS regularisation parameter (the paper's lambda).
        train_mask: Boolean mask, shape ``(n,)``, or ``None``. ``False`` rows
            are excluded from the solve and the RMSE diagnostic.

    Returns:
        The fitted :class:`_VKOGAState`.
    """
    n, _ = X.shape
    n_slices = Y.shape[1]
    dtype = X.dtype

    mask = jnp.ones((n,), dtype=bool) if train_mask is None else train_mask.astype(bool)
    n_train = jnp.sum(mask)

    eq = _normal_equations(X, Y, idx, m, V, train_mask=mask)
    centers = eq.centers
    alpha = _solve_ridge(eq, ridge)  # (max_centers, S)

    # Training-fit diagnostic, per slice, over the rows actually fitted. It
    # evaluates the kernel translates, as predict does, so it scores the
    # surrogate the caller gets.
    A_nm = _gaussian_kernel(X, centers, gamma) * eq.active[None, :]
    sq_err = jnp.where(mask[:, None], (Y - A_nm @ alpha) ** 2, 0.0)
    rmse = jnp.sqrt(jnp.sum(sq_err, axis=0) / jnp.maximum(n_train, 1))
    rmse = jnp.where(n_train > 0, rmse, jnp.full((n_slices,), jnp.nan))

    return _VKOGAState(
        centers=centers,
        coefficients=alpha,
        gamma=jnp.asarray(gamma, dtype=dtype),
        ridge=jnp.asarray(ridge, dtype=dtype),
        n_centers=m,
        rmse=rmse,
    )


def _fit_vkoga(
    X: Array,
    Y: Array,
    *,
    gamma: Array | float,
    max_centers: int,
    greedy_update: GreedyUpdate = "exact",
    ridge: Array | float = _DEFAULT_RIDGE,
    tol_power: Array | float = _DEFAULT_TOL_POWER,
    tol_residual: Array | float = _DEFAULT_TOL_RESIDUAL,
    train_mask: Array | None = None,
) -> _VKOGAState:
    """Fit a vectorial Gaussian-RBF surrogate by P-greedy centre selection.

    The fit runs in two stages. :func:`_select_centers` picks the centres, then
    :func:`_solve_coefficients` solves the regularised system over them. See
    those functions for the algorithmic details and the argument contracts.

    Args:
        X: Training inputs, shape ``(n, D)``.
        Y: Training outputs, shape ``(n, S)``, output slices already flattened.
        gamma: Positive Gaussian shape parameter.
        greedy_update: ``"exact"`` or ``"incremental"``; see
            :func:`jaxgsa.vkoga.analyze`.
        max_centers: Hard cap on the number of centres, and the static array
            size of the returned state. Silently capped at ``n``.
        ridge: RKHS regularisation parameter (the paper's lambda).
        tol_power: Absolute power-function stopping tolerance (tau_P).
        tol_residual: Relative residual stopping tolerance (tau_R), taken
            against each output slice's own norm.
        train_mask: Boolean mask, shape ``(n,)``, or ``None``. ``False`` rows
            are held out.

    Returns:
        The fitted :class:`_VKOGAState`.
    """
    idx, m, V = _newton_greedy(
        X,
        Y,
        gamma=gamma,
        max_centers=max_centers,
        greedy_update=greedy_update,
        tol_power=tol_power,
        tol_residual=tol_residual,
        train_mask=train_mask,
    )
    return _solve_coefficients(X, Y, idx, m, V, gamma=gamma, ridge=ridge, train_mask=train_mask)


def _predict_vkoga(state: _VKOGAState, X_new: Array) -> Array:
    """Evaluate a fitted VKOGA surrogate at new points.

    The function has no host-side control flow and no dependence on
    ``n_centers``, so it is safe to ``jit`` and ``vmap`` and to call inside a
    Monte-Carlo loop over millions of rows. Unused centre slots carry zero
    coefficients, so they drop out of the matrix product.

    Args:
        state: Fitted surrogate.
        X_new: Evaluation points, shape ``(n_new, D)``.

    Returns:
        Predictions, shape ``(n_new, S)``, one column per output slice.
    """
    return _gaussian_kernel(X_new, state.centers, state.gamma) @ state.coefficients


def _content_digest(*arrays: ArrayLike) -> bytes:
    """Return a digest of the dtype, shape and bytes of ``arrays``.

    A cache keyed on this digest holds 16 bytes per entry, not a host copy of
    the data.

    Args:
        *arrays: Arrays to hash, in order.

    Returns:
        A 16-byte BLAKE2b digest.
    """
    h = hashlib.blake2b(digest_size=16)
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.data)
    return h.digest()


@functools.partial(
    jax.jit,
    static_argnames=("max_centers", "n_folds", "tol_power", "tol_residual", "greedy_update"),
)
def _cv_fold_sse(
    X: Array,
    Y: Array,
    fold_of: Array,
    fold: Array,
    ridge_vec: Array,
    gamma: Array,
    *,
    max_centers: int,
    greedy_update: GreedyUpdate = "exact",
    n_folds: int,
    tol_power: float,
    tol_residual: float,
) -> tuple[Array, Array]:
    """Held-out squared errors of one ``(gamma, fold)`` pair, for every ridge.

    The data are arguments, not frozen constants, so the program compiles once
    per shape, dtype and static setting and is reused by every new dataset
    and every grid point.

    Args:
        X: Training inputs, shape ``(n, D)``.
        Y: Training outputs, shape ``(n, S)``.
        fold_of: Fold index of each row, shape ``(n,)``.
        fold: Scalar index of the held-out fold.
        ridge_vec: Ridge grid, shape ``(R,)``.
        gamma: Scalar Gaussian shape parameter.
        greedy_update: ``"exact"`` or ``"incremental"``; see
            :func:`jaxgsa.vkoga.analyze`.
        max_centers: Centre cap passed through to each fit.
        n_folds: Number of folds.
        tol_power: Absolute power-function stopping tolerance.
        tol_residual: Relative residual stopping tolerance.

    Returns:
        ``(sse_slice, sse_total)`` of shapes ``(R, S)`` and ``(R,)``: the
        held-out sum of squared errors per slice, and over all slices.
    """
    n = X.shape[0]
    train = fold_of != fold
    # One greedy sweep per (fold, gamma): selection is ridge-free.
    idx, m, V = _newton_greedy(
        X,
        Y,
        gamma=gamma,
        max_centers=max_centers,
        greedy_update=greedy_update,
        tol_power=tol_power,
        tol_residual=tol_residual,
        train_mask=train,
    )
    # The normal-equation matrices and the prediction kernel are also
    # ridge-free: build them once per (fold, gamma) and reuse them for
    # the whole ridge grid. Only the assembly-and-solve repeats.
    eq = _normal_equations(X, Y, idx, m, V, train_mask=train)
    block = -(-n // n_folds)
    (held_idx,) = jnp.nonzero(~train, size=block, fill_value=0)
    valid = jnp.arange(block) < jnp.sum(~train)
    K_pred = _gaussian_kernel(X[held_idx], eq.centers, gamma)
    Y_held = Y[held_idx]

    def _ridge_sse(ridge: Array) -> tuple[Array, Array]:
        err = Y_held - K_pred @ _solve_ridge(eq, ridge)
        held = jnp.where(valid[:, None], err**2, 0.0)
        # Per-slice sums for cv_rmse; the pooled sum is reduced
        # straight from the full array so the grid choice keeps
        # its original summation order.
        return jnp.sum(held, axis=0), jnp.sum(held)

    return jax.lax.map(_ridge_sse, ridge_vec)  # ((n_ridges, S), (n_ridges,))


@functools.partial(jax.jit, static_argnames=("n",))
def _cv_reduce(sse_slice: Array, sse_total: Array, *, n: int) -> tuple[Array, Array]:
    """Pool the per-fold squared errors into RMSE scores.

    Args:
        sse_slice: Per-slice sums, shape ``(G, F, R, S)``.
        sse_total: Pooled sums, shape ``(G, F, R)``.
        n: Training rows. The folds partition them, so every row is held out
            exactly once.

    Returns:
        ``(per_slice, pooled)`` of shapes ``(G, R, S)`` and ``(G, R)``.
    """
    n_slices = sse_slice.shape[-1]
    per_slice = jnp.sqrt(jnp.sum(sse_slice, axis=1) / n)
    pooled = jnp.sqrt(jnp.sum(sse_total, axis=1) / (n * n_slices))
    return per_slice, pooled


def _cv_workers(
    n: int, n_slices: int, max_centers: int, n_ridges: int, itemsize: int, n_tasks: int
) -> int:
    """Return how many ``(gamma, fold)`` programs may run at the same time.

    Each program is one long greedy loop of small kernels, so one program
    keeps about one core busy. Running several from a thread pool uses the
    other cores. Each running program holds its own buffers, so the count
    is also capped by the memory budget.

    Args:
        n: Training rows.
        n_slices: Output slices ``S``.
        max_centers: Centre cap.
        n_ridges: Ridge grid size.
        itemsize: Bytes per element of the working dtype.
        n_tasks: Number of ``(gamma, fold)`` programs to run.

    Returns:
        Worker count in ``[1, n_tasks]``.
    """
    # The Newton basis, the cross-kernel and the masked copies of it (about
    # 4 n x m), the outputs and their masked copy (2 n x S), the centre
    # matrices (3 m x m), and the right-hand sides and solutions of the
    # ridge grid ((R + 4) m x S).
    per_task = itemsize * (
        4 * n * max_centers
        + 2 * n * n_slices
        + 3 * max_centers * max_centers
        + (n_ridges + 4) * max_centers * n_slices
    )
    by_memory = get_memory_budget() // max(per_task, 1)
    return int(max(1, min(n_tasks, _usable_cpus(), by_memory)))


def _usable_cpus() -> int:
    """Return the number of CPUs this process may run on.

    Respects affinity and cgroup limits where the platform reports them,
    and falls back to the machine's CPU count.

    Returns:
        A CPU count of at least 1.
    """
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        return process_cpu_count() or 1
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        return len(sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def _thread_local_config() -> list[tuple[Any, Any]] | None:
    """Capture the JAX config values set on the calling thread only.

    Context managers such as ``jax.enable_x64()`` or ``jax.default_device()``
    set a thread-local value, which a pool thread does not see. Each pair
    here is re-entered in the worker by :func:`_in_config`.

    Returns:
        ``(state, value)`` pairs for every config option with a thread-local
        value, or ``None`` if this JAX version does not expose them. The
        caller then runs everything on its own thread.
    """
    try:
        from jax._src import config as jax_config  # noqa: PLC0415

        unset = jax_config.config_ext.unset
        states = list(jax_config.config_states.values())
        return [(state, value) for state in states if (value := state.get_local()) is not unset]
    except AttributeError:
        return None


def _in_config(
    local: list[tuple[Any, Any]], fn: Callable[[], tuple[Array, Array]]
) -> tuple[Array, Array]:
    """Run ``fn`` with the captured thread-local config values entered.

    Args:
        local: Pairs from :func:`_thread_local_config`.
        fn: Zero-argument function to run.

    Returns:
        ``fn()``.
    """
    from jax._src import config as jax_config  # noqa: PLC0415

    with contextlib.ExitStack() as stack:
        for state, value in local:
            stack.enter_context(jax_config.StateContextManager(state, value))
        return fn()


def _cv_scores(
    X: Array,
    Y: Array,
    fold_of: Array,
    ridge_vec: Array,
    gammas: np.ndarray,
    *,
    max_centers: int,
    greedy_update: GreedyUpdate = "exact",
    n_folds: int,
    tol_power: float,
    tol_residual: float,
) -> tuple[Array, Array]:
    """Score every ``(gamma, ridge)`` pair by k-fold held-out RMSE.

    Each ``(gamma, fold)`` pair is one call of :func:`_cv_fold_sse`. The
    pairs are independent, so after the first call (which compiles the
    program) they run from a thread pool, several at a time, each under the
    caller's thread-local JAX config. Each pair runs the same program
    whichever thread runs it, so the scores do not depend on the number of
    workers.

    Args:
        X: Training inputs, shape ``(n, D)``.
        Y: Training outputs, shape ``(n, S)``.
        fold_of: Fold index of each row, shape ``(n,)``.
        ridge_vec: Ridge grid, shape ``(R,)``.
        gammas: Gamma grid, host array of shape ``(G,)`` in ``X.dtype``.
        greedy_update: ``"exact"`` or ``"incremental"``; see
            :func:`jaxgsa.vkoga.analyze`.
        max_centers: Centre cap passed through to each fit.
        n_folds: Number of folds.
        tol_power: Absolute power-function stopping tolerance.
        tol_residual: Relative residual stopping tolerance.

    Returns:
        ``(per_slice, pooled)`` of shapes ``(G, R, S)`` and ``(G, R)``.
    """
    n, n_slices = Y.shape
    tasks = [(g, f) for g in range(len(gammas)) for f in range(n_folds)]
    fold_dtype = fold_of.dtype

    def _run(task: tuple[int, int]) -> tuple[Array, Array]:
        g, f = task
        out = _cv_fold_sse(
            X,
            Y,
            fold_of,
            np.asarray(f, dtype=fold_dtype),
            ridge_vec,
            gammas[g],
            max_centers=max_centers,
            greedy_update=greedy_update,
            n_folds=n_folds,
            tol_power=tol_power,
            tol_residual=tol_residual,
        )
        return jax.block_until_ready(out)

    # The first call compiles; the pool then reuses that one program.
    results = [_run(tasks[0])]
    workers = _cv_workers(
        n, n_slices, max_centers, int(ridge_vec.shape[0]), X.dtype.itemsize, len(tasks) - 1
    )
    # Pool threads do not inherit thread-local JAX config (for example
    # ``with jax.enable_x64():``), so each task re-enters the caller's values.
    local = _thread_local_config()
    if workers > 1 and local is not None:
        pool = ThreadPoolExecutor(max_workers=workers)
        futures = [pool.submit(_in_config, local, functools.partial(_run, t)) for t in tasks[1:]]
        try:
            results.extend(future.result() for future in futures)
        finally:
            # On an error, drop the queued tasks so it surfaces promptly.
            pool.shutdown(wait=True, cancel_futures=True)
    else:
        results.extend(_run(task) for task in tasks[1:])
    sse_slice = jnp.stack([r[0] for r in results]).reshape(
        len(gammas), n_folds, *results[0][0].shape
    )
    sse_total = jnp.stack([r[1] for r in results]).reshape(len(gammas), n_folds, -1)
    return _cv_reduce(sse_slice, sse_total, n=n)


def _cross_validate(
    X: Array,
    Y: Array,
    *,
    gammas: ArrayLike,
    ridges: ArrayLike,
    max_centers: int,
    greedy_update: GreedyUpdate = "exact",
    n_folds: int = 10,
    tol_power: float = _DEFAULT_TOL_POWER,
    tol_residual: float = _DEFAULT_TOL_RESIDUAL,
    seed: int | None = None,
) -> tuple[Array, Array]:
    """Score a ``(gamma, ridge)`` grid by k-fold held-out RMSE.

    Each fold refits from scratch on the other ``k-1`` folds. The greedy centre
    selection is part of what is being validated, so it must not see the
    held-out rows. The folds partition the rows, so every row is predicted
    exactly once out-of-sample. The returned score is the RMSE over that full
    set of out-of-sample predictions. The score is kept per output slice, and
    the pooled score is returned beside it, reduced on device as
    ``sqrt(sum(sse) / (n * S))``.

    Centre selection, the normal-equation matrices, and the prediction
    kernel do not depend on ``ridge``, so each ``(fold, gamma)`` pair builds
    them once and reuses them for every ridge in the grid. Only the cheap
    assembly-and-solve repeats per ridge.

    The caller supplies the grids. No range is baked in here.

    Args:
        X: Training inputs, shape ``(n, D)``.
        Y: Training outputs, shape ``(n, S)``, output slices already flattened.
        gammas: Candidate Gaussian shape parameters, any 1-D array-like.
        ridges: Candidate RKHS regularisation parameters, any 1-D array-like.
        greedy_update: ``"exact"`` or ``"incremental"``; see
            :func:`jaxgsa.vkoga.analyze`.
        max_centers: Centre cap passed through to each fit.
        n_folds: Number of folds, at least 2. Capped at ``n``.
        tol_power: Absolute power-function stopping tolerance (tau_P).
        tol_residual: Relative residual stopping tolerance (tau_R).
        seed: Seed for the row permutation used to build the folds. ``None``
            keeps the sample order, which is the right choice for a
            low-discrepancy design. Pass an int if the rows are ordered by
            anything that correlates with the output.

    Returns:
        A pair ``(per_slice, pooled)``. ``per_slice`` is the out-of-sample
        RMSE per output slice, shape ``(len(gammas), len(ridges), S)``.
        ``pooled`` is the RMSE over all slices, shape
        ``(len(gammas), len(ridges))``. Take ``argmin`` of ``pooled`` to pick
        the hyperparameters.

    Raises:
        ValueError: If ``n_folds < 2``. A single fold holds out every row, so
            all grid scores tie and the argmin is meaningless.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 for cross-validation, got {n_folds}")
    gamma_grid = np.atleast_1d(np.asarray(gammas, dtype=np.float64)).ravel()
    ridge_grid = np.atleast_1d(np.asarray(ridges, dtype=np.float64)).ravel()

    n = X.shape[0]
    n_folds = min(n_folds, n)

    # Interleaved fold assignment (row i -> fold i % k) after an optional
    # permutation: contiguous blocks would correlate with sample order for a
    # sequential design, whereas interleaving spreads each fold evenly.
    order = np.arange(n) if seed is None else np.random.default_rng(seed).permutation(n)
    fold_of = jnp.asarray(np.argsort(order) % n_folds)
    ridge_vec = jnp.asarray(ridge_grid, dtype=X.dtype)

    return _cv_scores(
        X,
        Y,
        fold_of,
        ridge_vec,
        gamma_grid.astype(X.dtype),
        max_centers=max_centers,
        greedy_update=greedy_update,
        n_folds=n_folds,
        tol_power=tol_power,
        tol_residual=tol_residual,
    )
