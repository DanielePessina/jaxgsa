"""Pure-JAX VKOGA: greedy kernel interpolation with a Newton basis.

Implements the Vectorial Kernel Orthogonal Greedy Algorithm (Wirtz &
Haasdonk, 2013; Santin & Haasdonk) as used for surrogate-based sensitivity
analysis by Hilhorst et al. (2024): a Gaussian RBF surrogate whose centres
are chosen by the P-greedy rule (maximise the power function) and whose
coefficients come from an RKHS-regularised least-squares solve over the
selected centres.

Two properties make this cheap enough to drive a Monte-Carlo sensitivity
loop. First, the P-greedy selection is *y-independent*, so a single greedy
sweep serves every output slice -- that is the "vectorial" part: one set of
centres, one Newton basis, one coefficient matrix with a column per slice
(separable matrix-valued kernel). Second, prediction is a single kernel
GEMM against a handful of centres.

Everything here is a pure function on arrays: no result classes, no
``Problem``. Outputs arrive as ``(n, S)`` with ``S = T*K`` output slices
already flattened by the caller.

Precision: the coefficient step forms the normal matrix ``A_nm^T A_nm``,
which squares the condition number of the cross-kernel. Under JAX's default
float32 that is the accuracy ceiling of this module -- it bites whenever the
kernel Gram is ill-conditioned, i.e. small ``gamma`` or a centre count close
to ``n``, where the fitted surrogate can be an order of magnitude worse than
the same equations solved in double. Enabling
``jax.config.update("jax_enable_x64", True)`` removes the ceiling; the code
carries ``X.dtype`` throughout and needs no other change. Grid-searching the
hyperparameters with :func:`_cross_validate` also mitigates it, since the
scores are computed with the same arithmetic and so penalise the
ill-conditioned corner of the grid.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

# Paper defaults (Hilhorst et al. 2024): tau_P = 5e-8 on the power function,
# tau_R = 1e-4 on the interpolation residual, lambda = 1e-10 on the RKHS term.
_DEFAULT_RIDGE = 1e-10
_DEFAULT_TOL_POWER = 5e-8
_DEFAULT_TOL_RESIDUAL = 1e-4

# Relative jitter on the active diagonal of the normal matrix. The greedy
# centres are chosen to be maximally distinct, but with the default
# lambda = 1e-10 the RKHS term does almost nothing, so the kernel Gram can
# still be numerically singular in float32 once many centres are selected.
_JITTER = 1e-8


class _VKOGAState(NamedTuple):
    """A fitted VKOGA surrogate.

    The arrays are allocated at the *static* ``max_centers`` size because the
    greedy loop runs under ``lax.while_loop`` and JAX needs static shapes.
    Slots beyond ``n_centers`` carry a duplicated centre with an exactly zero
    coefficient row, so :func:`_predict_vkoga` is correct whether or not the
    caller first slices the state down to ``n_centers`` rows.

    Attributes:
        centers: (m, D) selected centres, ``m = max_centers`` as returned.
        coefficients: (m, S) expansion coefficients, one column per output
            slice; rows ``>= n_centers`` are zero.
        gamma: scalar Gaussian shape parameter the surrogate was fitted with.
        ridge: scalar RKHS regularisation parameter (the paper's lambda).
        n_centers: number of centres the greedy loop actually selected.
        rmse: (S,) per-slice root-mean-square error of the fit on its own
            training rows -- a diagnostic, not a generalisation estimate.
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
        X1: (n1, D) first point set.
        X2: (n2, D) second point set.
        gamma: positive shape parameter; larger means narrower kernels.

    Returns:
        (n1, n2) kernel matrix.
    """
    # Squared distances via the ||a||^2 + ||b||^2 - 2a.b expansion: one GEMM
    # instead of an (n1, n2, D) difference tensor, which is what keeps
    # prediction affordable when n1 runs into the millions. The price is
    # cancellation for near-coincident points -- in float32 the expansion can
    # go slightly negative, which would make the kernel exceed 1 and corrupt
    # the power function p2 = 1 - ... below, so clamp at zero.
    sq1 = jnp.sum(X1 * X1, axis=1)
    sq2 = jnp.sum(X2 * X2, axis=1)
    d2 = sq1[:, None] + sq2[None, :] - 2.0 * (X1 @ X2.T)
    return jnp.exp(-gamma * jnp.maximum(d2, 0.0))


def _fit_vkoga(
    X: Array,
    Y: Array,
    *,
    gamma: Array | float,
    max_centers: int,
    ridge: Array | float = _DEFAULT_RIDGE,
    tol_power: Array | float = _DEFAULT_TOL_POWER,
    tol_residual: Array | float = _DEFAULT_TOL_RESIDUAL,
    train_mask: Array | None = None,
) -> _VKOGAState:
    """Fit a vectorial Gaussian-RBF surrogate by P-greedy centre selection.

    Centres are picked one at a time at the current maximiser of the power
    function (Hilhorst et al. 2024, eq. 8), expressed in the Newton basis of
    Pazouki & Schaback so that the basis is *nested*: adding a centre appends
    one column and never touches the earlier ones. For the Gaussian kernel
    the diagonal is 1, so the squared power function at every training point
    is simply ``p2 = 1 - rowsum(V**2)`` with ``V`` the Newton basis evaluated
    at those points.

    Selection is driven only by ``X``; ``Y`` enters through the residual
    stopping rule and the final coefficient solve. All output slices
    therefore share one basis and one set of centres.

    Coefficients solve the RKHS-regularised normal equations over the
    selected centres::

        (A_nm^T A_nm + ridge * A_mm) alpha = A_nm^T Y

    where ``A_nm`` is the (n, m) cross-kernel between training points and
    centres and ``A_mm`` the (m, m) centre Gram. When ``m = n`` this collapses
    to the paper's eq. (6): ``A_nm = A_mm = A`` is symmetric, so the system is
    ``A (A + ridge*I) alpha = A Y``, i.e. ``(A + ridge*I) alpha = Y``.

    Args:
        X: (n, D) training inputs.
        Y: (n, S) training outputs, output slices already flattened.
        gamma: positive Gaussian shape parameter.
        max_centers: hard cap on the number of centres; also the static array
            size of the returned state. Silently capped at the number of
            usable training rows.
        ridge: RKHS regularisation parameter (the paper's lambda).
        tol_power: stop once ``sqrt(max(p2)) <= tol_power`` (tau_P).
        tol_residual: stop once the largest per-slice residual norm falls to
            ``tol_residual`` (tau_R).
        train_mask: optional (n,) boolean mask; ``False`` rows are excluded
            both as candidate centres and from the coefficient solve. Used by
            :func:`_cross_validate` to hold folds out without changing shapes.

    Returns:
        The fitted :class:`_VKOGAState`.
    """
    n, _ = X.shape
    n_slices = Y.shape[1]
    # Never ask for more centres than there are points to pick from: the
    # greedy loop would run out of candidates and stall on masked -inf.
    max_centers = min(max_centers, n)

    mask = jnp.ones((n,), dtype=bool) if train_mask is None else train_mask.astype(bool)
    n_train = jnp.sum(mask)

    dtype = X.dtype
    # Rows that must never be chosen as centres: held-out rows start blocked,
    # and each selected point is blocked afterwards so it cannot repeat.
    blocked0 = ~mask
    # Held-out rows contribute nothing to the residual stopping rule either.
    residual0 = jnp.where(mask[:, None], Y, 0.0)

    eps = jnp.finfo(dtype).eps
    init = (
        jnp.zeros((n, max_centers), dtype=dtype),  # Newton basis at all points
        residual0,
        blocked0,
        jnp.zeros((max_centers,), dtype=jnp.int32),  # selected row indices
        jnp.zeros((), dtype=jnp.int32),  # centres selected so far
        # K(x, x) = 1 for the Gaussian kernel, so the power function starts at
        # 1 everywhere; the empty basis subtracts nothing.
        jnp.ones((), dtype=dtype),
        jnp.max(jnp.sqrt(jnp.sum(residual0**2, axis=0))),
    )

    def _cond(state: tuple) -> Array:
        _, _, _, _, m, power, res = state
        # All three stopping rules, first to trigger wins.
        return (m < max_centers) & (power > tol_power) & (res > tol_residual)

    def _body(state: tuple) -> tuple:
        V, residual, blocked, idx, m, _, _ = state

        p2 = jnp.where(blocked, -jnp.inf, 1.0 - jnp.sum(V * V, axis=1))
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
        v = (k_col - V @ V[i]) / sqrt_p

        # Interpolation residual in the same basis: one coefficient row per
        # output slice, since all slices share this basis.
        c = residual[i] / sqrt_p  # (S,)
        residual = residual - jnp.outer(v, c)

        V = V.at[:, m].set(v)
        blocked = blocked.at[i].set(True)
        # argmax returns int64 when x64 is enabled; keep the index buffer int32
        # so the scatter never hits a narrowing-cast promotion error.
        idx = idx.at[m].set(i.astype(idx.dtype))
        m = m + 1

        # Stopping quantities for the next iteration. Clamp the power at zero
        # so exhausted candidates (all -inf) terminate instead of producing a
        # NaN under the square root.
        p2_next = jnp.where(blocked, -jnp.inf, 1.0 - jnp.sum(V * V, axis=1))
        power = jnp.sqrt(jnp.maximum(jnp.max(p2_next), 0.0))
        # Selected points are interpolated exactly, so their rows are already
        # ~0 and summing over all rows equals summing over the remaining ones.
        res = jnp.max(jnp.sqrt(jnp.sum(residual**2, axis=0)))
        return (V, residual, blocked, idx, m, power, res)

    _, _, _, idx, m, _, _ = jax.lax.while_loop(_cond, _body, init)

    # --- coefficients over the selected centres -----------------------------
    # Unused slots point at row 0; masking their kernel columns to zero (and
    # putting a 1 on their diagonal below) forces their coefficients to be
    # exactly zero, so the duplicate centre is inert.
    active = jnp.arange(max_centers) < m
    centers = X[idx]
    A_nm = _gaussian_kernel(X, centers, gamma) * active[None, :] * mask[:, None]
    A_mm = _gaussian_kernel(centers, centers, gamma) * active[None, :] * active[:, None]

    G = A_nm.T @ A_nm + ridge * A_mm
    # Relative jitter: scale-free against gamma and the number of rows, which
    # both move the magnitude of A_nm^T A_nm by orders of magnitude.
    diag_mean = jnp.sum(jnp.where(active, jnp.diagonal(G), 0.0)) / jnp.maximum(m, 1)
    G = G + jnp.diag(jnp.where(active, _JITTER * diag_mean, 1.0))

    Y_train = jnp.where(mask[:, None], Y, 0.0)
    alpha = jnp.linalg.solve(G, A_nm.T @ Y_train)  # (max_centers, S)

    # Training-fit diagnostic, per slice, over the rows actually fitted.
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


def _predict_vkoga(state: _VKOGAState, X_new: Array) -> Array:
    """Evaluate a fitted VKOGA surrogate at new points.

    Free of host-side control flow and of any dependence on ``n_centers``, so
    it is safe to ``jit``/``vmap`` and to call inside a Monte-Carlo loop over
    millions of rows. Unused centre slots carry zero coefficients, so they
    drop out of the matrix product.

    Args:
        state: fitted surrogate.
        X_new: (n_new, D) evaluation points.

    Returns:
        (n_new, S) predictions, one column per output slice.
    """
    return _gaussian_kernel(X_new, state.centers, state.gamma) @ state.coefficients


def _cross_validate(
    X: Array,
    Y: Array,
    *,
    gammas: ArrayLike,
    ridges: ArrayLike,
    max_centers: int,
    n_folds: int = 10,
    tol_power: float = _DEFAULT_TOL_POWER,
    tol_residual: float = _DEFAULT_TOL_RESIDUAL,
    seed: int | None = None,
) -> Array:
    """Score a ``(gamma, ridge)`` grid by k-fold held-out RMSE.

    Each fold refits from scratch on the other ``k-1`` folds -- the greedy
    centre selection is part of what is being validated, so it must not see
    the held-out rows. Because the folds partition the rows, every row is
    predicted exactly once out-of-sample and the returned score is the RMSE
    over that full set of out-of-sample predictions, pooled across output
    slices.

    The grids are caller-supplied: no range is baked in here.

    Args:
        X: (n, D) inputs.
        Y: (n, S) outputs, output slices already flattened.
        gammas: candidate Gaussian shape parameters, any 1-D array-like.
        ridges: candidate RKHS regularisation parameters, any 1-D array-like.
        max_centers: centre cap passed through to each fit.
        n_folds: number of folds; capped at ``n``.
        tol_power: power-function stopping tolerance (tau_P).
        tol_residual: residual stopping tolerance (tau_R).
        seed: seed for the row permutation used to build the folds. ``None``
            keeps the sample order, which is what you want for a
            low-discrepancy design; pass an int if the rows are ordered by
            anything that correlates with the output.

    Returns:
        (len(gammas), len(ridges)) array of pooled out-of-sample RMSE. Take
        ``argmin`` over the flattened grid to pick the hyperparameters.
    """
    gamma_grid = np.atleast_1d(np.asarray(gammas, dtype=np.float64)).ravel()
    ridge_grid = np.atleast_1d(np.asarray(ridges, dtype=np.float64)).ravel()

    n = X.shape[0]
    n_slices = Y.shape[1]
    n_folds = min(n_folds, n)

    # Interleaved fold assignment (row i -> fold i % k) after an optional
    # permutation: contiguous blocks would correlate with sample order for a
    # sequential design, whereas interleaving spreads each fold evenly.
    order = np.arange(n) if seed is None else np.random.default_rng(seed).permutation(n)
    fold_of = jnp.asarray(np.argsort(order) % n_folds)
    fold_ids = jnp.arange(n_folds)

    @jax.jit
    def _score(gamma: Array, ridge: Array) -> Array:
        def _fold_sse(fold: Array) -> Array:
            train = fold_of != fold
            state = _fit_vkoga(
                X,
                Y,
                gamma=gamma,
                max_centers=max_centers,
                ridge=ridge,
                tol_power=tol_power,
                tol_residual=tol_residual,
                train_mask=train,
            )
            err = Y - _predict_vkoga(state, X)
            return jnp.sum(jnp.where(train[:, None], 0.0, err**2))

        # lax.map runs the folds sequentially: vmap would hold n_folds copies
        # of the (n, n) kernel work live at once, which is the memory wall
        # here long before it is a speed win.
        sse = jnp.sum(jax.lax.map(_fold_sse, fold_ids))
        return jnp.sqrt(sse / (n * n_slices))

    # The grids are host-side and static, so loop over them in Python; the
    # jitted body is traced once and reused for every pair.
    scores = [
        [_score(jnp.asarray(g, dtype=X.dtype), jnp.asarray(r, dtype=X.dtype)) for r in ridge_grid]
        for g in gamma_grid
    ]
    return jnp.asarray(scores)
