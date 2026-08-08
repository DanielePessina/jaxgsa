"""Pure-JAX VKOGA: greedy kernel interpolation with a Newton basis.

Implements the Vectorial Kernel Orthogonal Greedy Algorithm (Wirtz &
Haasdonk, 2013; Santin & Haasdonk) as used for surrogate-based sensitivity
analysis by Hilhorst et al. (2024). The surrogate is a Gaussian RBF expansion.
Its centres are chosen by the P-greedy rule, which maximises the power
function. Its coefficients come from an RKHS-regularised least-squares solve
over the selected centres.

Two properties make this cheap enough to drive a Monte-Carlo sensitivity loop.
First, the P-greedy selection does not depend on ``Y``, so one greedy sweep
serves every output slice. That is the "vectorial" part: one set of centres,
one Newton basis, and one coefficient matrix with a column per slice
(a separable matrix-valued kernel). Second, prediction is a single kernel GEMM
against a handful of centres.

Everything here is a pure function on arrays. There are no result classes and
no ``Problem``. Outputs arrive as ``(n, S)`` with ``S = T*K`` output slices
already flattened by the caller.

The coefficient step forms the normal matrix ``A_nm^T A_nm``, which squares the
condition number of the cross-kernel. Under JAX's default float32 that is the
accuracy ceiling of this module. It bites whenever the kernel Gram is
ill-conditioned, which means a small ``gamma`` or a centre count close to
``n``. There the fitted surrogate can be an order of magnitude worse than the
same equations solved in double precision. Enabling
``jax.config.update("jax_enable_x64", True)`` removes the ceiling, and the code
carries ``X.dtype`` throughout so it needs no other change. Grid-searching the
hyperparameters with :func:`_cross_validate` also helps, because the scores are
computed in the same arithmetic and so penalise the ill-conditioned corner of
the grid.
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
# tau_P is absolute: the Gaussian kernel has K(x, x) = 1, so the power
# function is already scale-free. tau_R is applied relative to each output
# slice's norm, because an absolute residual rule would depend on the scale
# of Y.
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


def _select_centers(
    X: Array,
    Y: Array,
    *,
    gamma: Array | float,
    max_centers: int,
    tol_power: Array | float = _DEFAULT_TOL_POWER,
    tol_residual: Array | float = _DEFAULT_TOL_RESIDUAL,
    train_mask: Array | None = None,
) -> tuple[Array, Array]:
    """Select kernel centres by the P-greedy rule.

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
        ``(idx, m)``. ``idx`` is the int32 buffer of selected row indices,
        shape ``(max_centers,)``. ``m`` is the scalar count of centres actually
        selected. Slots ``>= m`` are unused.
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
    init = (
        jnp.zeros((n, max_centers), dtype=dtype),  # Newton basis at all points
        residual0,
        blocked0,
        jnp.zeros((max_centers,), dtype=jnp.int32),  # selected row indices
        jnp.zeros((), dtype=jnp.int32),  # centres selected so far
        # K(x, x) = 1 for the Gaussian kernel, so the power function starts at
        # 1 everywhere; the empty basis subtracts nothing.
        jnp.ones((), dtype=dtype),
        jnp.max(y_norm / res_scale),
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
        # output slice, since all slices share this basis. Re-mask after the
        # update: v is dense, so the outer product writes minus-interpolant
        # values into held-out rows. Without the mask those rows accumulate
        # and the stopping norm below never converges inside cross-validation.
        c = residual[i] / sqrt_p  # (S,)
        residual = jnp.where(mask[:, None], residual - jnp.outer(v, c), 0.0)

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
        # Held-out rows are re-masked to exactly zero above, and selected
        # points are interpolated exactly, so summing over all rows equals
        # summing over the active training rows. Relative to each slice's own
        # norm, so the rule is scale-free.
        res = jnp.max(jnp.sqrt(jnp.sum(residual**2, axis=0)) / res_scale)
        return (V, residual, blocked, idx, m, power, res)

    _, _, _, idx, m, _, _ = jax.lax.while_loop(_cond, _body, init)
    return idx, m


class _NormalEquations(NamedTuple):
    """Ridge-independent pieces of the regularised coefficient solve.

    Everything here depends on ``X``, ``Y``, the selected centres, and
    ``gamma``, but not on ``ridge``. :func:`_cross_validate` builds one per
    ``(fold, gamma)`` and reuses it for every ridge in the grid. Only the cheap
    assembly-and-solve in :func:`_solve_ridge` repeats.

    Attributes:
        active: Boolean mask of the slots the greedy filled, shape
            ``(max_centers,)``.
        centers: Selected centres, shape ``(max_centers, D)``, padded with
            duplicates.
        A_nm: Masked cross-kernel between training points and centres, shape
            ``(n, max_centers)``.
        A_mm: Masked centre Gram, shape ``(max_centers, max_centers)``.
        AtA: ``A_nm^T A_nm``, shape ``(max_centers, max_centers)``.
        AtY: ``A_nm^T Y`` over the training rows, shape ``(max_centers, S)``.
        m: Scalar count of centres actually selected.
    """

    active: Array
    centers: Array
    A_nm: Array
    A_mm: Array
    AtA: Array
    AtY: Array
    m: Array


def _normal_equations(
    X: Array,
    Y: Array,
    idx: Array,
    m: Array,
    *,
    gamma: Array | float,
    train_mask: Array | None = None,
) -> _NormalEquations:
    """Build the ridge-independent normal-equation matrices.

    Args:
        X: Training inputs, shape ``(n, D)``.
        Y: Training outputs, shape ``(n, S)``, output slices already flattened.
        idx: Selected row indices from :func:`_select_centers`, shape
            ``(max_centers,)``.
        m: Scalar count of centres actually selected.
        gamma: Positive Gaussian shape parameter the centres were selected
            with.
        train_mask: Boolean mask, shape ``(n,)``, or ``None``. ``False`` rows
            are excluded from the solve.

    Returns:
        A :class:`_NormalEquations` bundle.
    """
    n, _ = X.shape
    max_centers = idx.shape[0]
    mask = jnp.ones((n,), dtype=bool) if train_mask is None else train_mask.astype(bool)

    # Unused slots point at row 0; masking their kernel columns to zero (and
    # putting a 1 on their diagonal in _solve_ridge) forces their coefficients
    # to be exactly zero, so the duplicate centre is inert.
    active = jnp.arange(max_centers) < m
    centers = X[idx]
    A_nm = _gaussian_kernel(X, centers, gamma) * active[None, :] * mask[:, None]
    A_mm = _gaussian_kernel(centers, centers, gamma) * active[None, :] * active[:, None]
    Y_train = jnp.where(mask[:, None], Y, 0.0)
    return _NormalEquations(
        active=active,
        centers=centers,
        A_nm=A_nm,
        A_mm=A_mm,
        AtA=A_nm.T @ A_nm,
        AtY=A_nm.T @ Y_train,
        m=m,
    )


def _solve_ridge(eq: _NormalEquations, ridge: Array | float) -> Array:
    """Assemble the regularised system for one ridge value and solve it.

    Args:
        eq: Ridge-independent matrices from :func:`_normal_equations`.
        ridge: RKHS regularisation parameter (the paper's lambda).

    Returns:
        Expansion coefficients, shape ``(max_centers, S)``. Inactive rows are
        zero.
    """
    G = eq.AtA + ridge * eq.A_mm
    # Relative jitter: scale-free against gamma and the number of rows, which
    # both move the magnitude of A_nm^T A_nm by orders of magnitude.
    diag_mean = jnp.sum(jnp.where(eq.active, jnp.diagonal(G), 0.0)) / jnp.maximum(eq.m, 1)
    G = G + jnp.diag(jnp.where(eq.active, _JITTER * diag_mean, 1.0))
    return jnp.linalg.solve(G, eq.AtY)


def _solve_coefficients(
    X: Array,
    Y: Array,
    idx: Array,
    m: Array,
    *,
    gamma: Array | float,
    ridge: Array | float,
    train_mask: Array | None = None,
) -> _VKOGAState:
    """Solve for the expansion coefficients over already-selected centres.

    Coefficients solve the RKHS-regularised normal equations over the
    selected centres::

        (A_nm^T A_nm + ridge * A_mm) alpha = A_nm^T Y

    ``A_nm`` is the ``(n, m)`` cross-kernel between training points and
    centres, and ``A_mm`` is the ``(m, m)`` centre Gram. When ``m = n`` this
    collapses to the paper's eq. (6). ``A_nm = A_mm = A`` is then symmetric, so
    the system is ``A (A + ridge*I) alpha = A Y``, that is
    ``(A + ridge*I) alpha = Y``.

    Args:
        X: Training inputs, shape ``(n, D)``.
        Y: Training outputs, shape ``(n, S)``, output slices already flattened.
        idx: Selected row indices from :func:`_select_centers`, shape
            ``(max_centers,)``.
        m: Scalar count of centres actually selected.
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

    eq = _normal_equations(X, Y, idx, m, gamma=gamma, train_mask=mask)
    centers = eq.centers
    alpha = _solve_ridge(eq, ridge)  # (max_centers, S)

    # Training-fit diagnostic, per slice, over the rows actually fitted.
    sq_err = jnp.where(mask[:, None], (Y - eq.A_nm @ alpha) ** 2, 0.0)
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
    idx, m = _select_centers(
        X,
        Y,
        gamma=gamma,
        max_centers=max_centers,
        tol_power=tol_power,
        tol_residual=tol_residual,
        train_mask=train_mask,
    )
    return _solve_coefficients(X, Y, idx, m, gamma=gamma, ridge=ridge, train_mask=train_mask)


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

    Each fold refits from scratch on the other ``k-1`` folds. The greedy centre
    selection is part of what is being validated, so it must not see the
    held-out rows. The folds partition the rows, so every row is predicted
    exactly once out-of-sample. The returned score is the RMSE over that full
    set of out-of-sample predictions, pooled across output slices.

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
        max_centers: Centre cap passed through to each fit.
        n_folds: Number of folds, at least 2. Capped at ``n``.
        tol_power: Absolute power-function stopping tolerance (tau_P).
        tol_residual: Relative residual stopping tolerance (tau_R).
        seed: Seed for the row permutation used to build the folds. ``None``
            keeps the sample order, which is the right choice for a
            low-discrepancy design. Pass an int if the rows are ordered by
            anything that correlates with the output.

    Returns:
        Pooled out-of-sample RMSE, shape ``(len(gammas), len(ridges))``. Take
        ``argmin`` over the flattened grid to pick the hyperparameters.

    Raises:
        ValueError: If ``n_folds < 2``. A single fold holds out every row, so
            all grid scores tie and the argmin is meaningless.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 for cross-validation, got {n_folds}")
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
    ridge_vec = jnp.asarray(ridge_grid, dtype=X.dtype)

    @jax.jit
    def _score_gamma(gamma: Array) -> Array:
        def _fold_sse(fold: Array) -> Array:
            train = fold_of != fold
            # One greedy sweep per (fold, gamma): selection is ridge-free.
            idx, m = _select_centers(
                X,
                Y,
                gamma=gamma,
                max_centers=max_centers,
                tol_power=tol_power,
                tol_residual=tol_residual,
                train_mask=train,
            )
            # The normal-equation matrices and the prediction kernel are also
            # ridge-free: build them once per (fold, gamma) and reuse them for
            # the whole ridge grid. Only the assembly-and-solve repeats.
            eq = _normal_equations(X, Y, idx, m, gamma=gamma, train_mask=train)
            K_pred = _gaussian_kernel(X, eq.centers, gamma)

            def _ridge_sse(ridge: Array) -> Array:
                err = Y - K_pred @ _solve_ridge(eq, ridge)
                return jnp.sum(jnp.where(train[:, None], 0.0, err**2))

            return jax.lax.map(_ridge_sse, ridge_vec)  # (n_ridges,)

        # lax.map runs the folds sequentially: vmap would hold n_folds copies
        # of the (n, n) kernel work live at once, which is the memory wall
        # here long before it is a speed win.
        sse = jnp.sum(jax.lax.map(_fold_sse, fold_ids), axis=0)
        return jnp.sqrt(sse / (n * n_slices))

    # The gamma grid is host-side and static, so loop over it in Python; the
    # jitted body is traced once and reused for every value.
    scores = [_score_gamma(jnp.asarray(g, dtype=X.dtype)) for g in gamma_grid]
    return jnp.asarray(scores)
