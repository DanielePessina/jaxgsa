"""Log-domain Sinkhorn solver for entropic optimal transport.

This module implements the standard entropic-regularization scheme for
discrete optimal transport (Cuturi 2013; Peyre & Cuturi 2019,
"Computational Optimal Transport"). The scaling updates run in the log
domain for numerical stability.

The ``"multivariate"`` and ``"trajectory"`` modes of
:func:`jaxgsa.optimal_transport.analyze` use this solver to transport the
unconditional output point cloud onto each conditional class. The
``"univariate"`` mode never needs a solver, because 1-D optimal transport
has a closed form through sorted quantiles.

The solver is a pure jit/vmap-compatible function. It never raises on
non-convergence, because exceptions cannot cross a ``lax.while_loop``.
Instead it returns the final marginal residual. The caller can then warn
once per analysis after a single host sync.

Derivatives do not go through the loop. ``jax.lax.custom_root`` keeps the
loop's values and differentiates the fixed point of the scaling update by the
implicit function theorem. Reverse and forward mode both work, and both give
the derivative of the converged solution, not of the iterate where the loop
stopped (Luise et al. 2018; Eisenberger et al. 2022). GMRES solves the
linear system of that derivative. Second derivatives run, but no test
covers them.

References:
    Cuturi (2013). Sinkhorn distances: lightspeed computation of optimal
    transport. NeurIPS 26.

    Peyre & Cuturi (2019). Computational Optimal Transport. Foundations
    and Trends in Machine Learning 11(5-6):355-607.

    Luise, Rudi, Pontil & Ciliberto (2018). Differential properties of
    Sinkhorn approximation for learning with Wasserstein distance. NeurIPS 31.
    arXiv:1805.11897.

    Eisenberger, Toker, Leal-Taixe, Bernard & Cremers (2022). A unified
    framework for implicit Sinkhorn differentiation. CVPR.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.sparse.linalg import gmres
from jax.scipy.special import logsumexp

# Iterations between convergence checks: measuring the marginal residual
# costs as much as one scaling update, so checking every iteration would
# tax the whole loop by ~50%.
_CHECK_EVERY = 10

# Krylov vectors GMRES keeps before it restarts, in the implicit-derivative
# solve.
_GMRES_RESTART = 30


def _sinkhorn_w2(
    C: Array,
    log_b: Array,
    epsilon: Array,
    max_iter: Array,
    tol: Array,
    scale: Array,
) -> tuple[Array, Array]:
    """Entropic OT cost from a uniform source onto one masked target class.

    Solves the entropic optimal-transport problem between the uniform
    distribution on the ``N`` source points and the target histogram
    ``exp(log_b)`` over ``P`` (padded) target points. It then reports the
    unregularized transport cost ``<P, C>`` of the entropic plan, not the
    epsilon-regularized objective. The value therefore approaches the
    exact squared Wasserstein distance as ``epsilon -> 0``.

    The caller passes ``scale``, the same normalizer that turns the index
    itself into a [0, 1] quantity (``V = 2 * Var`` or ``2 * tr(Cov)``), so
    ``epsilon`` is a regularization strength relative to that one fixed
    scale. Scaling by the cost matrix's own per-class maximum instead (an
    earlier version of this function did that) makes the effective
    regularization outlier-driven and different for every class, which
    left a class-dependent entropic bias that did not cancel between
    parameters. Updates run in the log domain. ``log_b = -inf`` handles
    padded target columns exactly: they carry zero mass and stay inert in
    every ``logsumexp``.

    Args:
        C: Cost matrix of squared Euclidean distances, shape ``(N, P)``.
        log_b: Log target weights, shape ``(P,)``. The value is
            ``-log(n_m)`` for the class's ``n_m`` valid entries and
            ``-inf`` at padded entries.
        epsilon: Entropic regularization strength relative to ``scale``
            (scalar).
        max_iter: Iteration cap (scalar).
        tol: Stopping tolerance on the L1 violation of the target
            marginal (scalar).
        scale: The cost normalizer (scalar), the same one the caller
            divides the index by. Must be positive; the caller guards the
            degenerate (zero-variance) case before calling.

    Returns:
        A tuple ``(cost, err)``. ``cost`` is the transport cost ``<P, C>``
        on the original cost scale. ``err`` is the L1 target-marginal
        violation at exit, refreshed on the final iteration, and
        ``err <= tol`` means the solve converged.
    """
    dtype = C.dtype
    N = C.shape[0]
    b = jnp.exp(log_b)
    # Regularization relative to the index's own normalizer, not the cost
    # matrix's per-class maximum: see the docstring above.
    scale = jnp.maximum(scale, jnp.asarray(jnp.finfo(dtype).tiny, dtype))
    log_K = -(C / scale) / epsilon  # (N, P) log-kernel
    log_a = jnp.full((N,), -jnp.log(N), dtype=dtype)

    def _err(log_f: Array, log_g: Array) -> Array:
        """L1 violation of the target marginal of the implied plan."""
        log_plan = log_f[:, None] + log_K + log_g[None, :]
        return jnp.abs(jnp.exp(logsumexp(log_plan, axis=0)) - b).sum()

    def _body(state: tuple[Array, Array, Array, Array]):
        log_f, log_g, it, err = state
        # Alternate scaling: after the f-update the source marginal is
        # exact, so convergence is measured on the target side.
        log_g = log_b - logsumexp(log_K + log_f[:, None], axis=0)
        log_f = log_a - logsumexp(log_K + log_g[None, :], axis=1)
        it = it + 1
        # Measuring the residual costs a third full (N, P) reduction per
        # iteration, so amortize it: refresh every _CHECK_EVERY iterations
        # and on the final one; the carried value persists in between (it
        # is > tol by construction, or the loop would have exited).
        err = jax.lax.cond(
            (it % _CHECK_EVERY == 0) | (it >= max_iter),
            lambda: _err(log_f, log_g),
            lambda: err,
        )
        return (log_f, log_g, it, err)

    def _cond(state: tuple[Array, Array, Array, Array]) -> Array:
        _, _, it, err = state
        return (err > tol) & (it < max_iter)

    # Padded target columns hold log_g = -inf. The implicit derivative below
    # needs a finite root variable, so it works on log_g with the pads set
    # to 0 and restores -inf before every use.
    valid = jnp.isfinite(log_b)
    neg_inf = jnp.asarray(-jnp.inf, dtype=dtype)

    def _f_from_g(g_safe: Array) -> Array:
        """Source half-update: the exact source marginal for a given log_g."""
        log_g = jnp.where(valid, g_safe, neg_inf)
        return log_a - logsumexp(log_K + log_g[None, :], axis=1)

    def _residual(g_safe: Array) -> Array:
        """One full scaling update of log_g minus log_g. Zero at the fixed point."""
        g_next = log_b - logsumexp(log_K + _f_from_g(g_safe)[:, None], axis=0)
        return jnp.where(valid, g_next, 0.0) - g_safe

    def _solve(_residual_fn, _guess: Array) -> tuple[Array, Array]:
        """Run the stopping-rule loop. Its iterates are the reported values."""
        init = (
            jnp.zeros(N, dtype=dtype),
            jnp.zeros(log_b.shape, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=dtype),
        )
        _, log_g, _, err = jax.lax.while_loop(_cond, _body, _body(init))
        return jnp.where(valid, log_g, 0.0), err

    n_valid = jnp.maximum(valid.sum(), 1)

    def _project(x: Array) -> Array:
        """Remove the constant shift of log_g on the valid columns."""
        return jnp.where(valid, x - jnp.where(valid, x, 0.0).sum() / n_valid, 0.0)

    def _krylov(linear: Callable[[Array], Array], rhs: Array) -> Array:
        """Solve ``linear(x) = rhs`` by GMRES on the mean-zero subspace.

        ``linear`` is ``J - I`` or its transpose ``J^T - I``, with ``J`` the
        Jacobian of one scaling update. Both are singular. ``J`` keeps the
        constant shift of log_g, because the potentials are only defined up
        to that shift. ``J^T`` keeps a different direction: the target
        weights ``b``, not the constant vector.

        The solve works on the mean-zero vectors for both. Let ``Pi``
        remove the mean on the valid columns. The right-hand side has no
        component along the constant vector, because the transport cost
        does not change under the shift, so ``Pi`` does not change it.
        ``Pi J`` and ``Pi J^T`` have the same spectrum: ``0`` on the
        constant vector, then ``lambda_2 < 1`` and smaller. So the projected
        system is nonsingular for ``J`` and for ``J^T``.

        A plain fixed-point iteration converges at the rate ``lambda_2``.
        That rate is close to 1 for small epsilon, so the iteration stops
        far from the solution. GMRES does not depend on that rate. Its
        tolerance ``tol`` is relative to the norm of the right-hand side.
        """
        x = gmres(
            lambda v: _project(linear(_project(v))),
            _project(rhs),
            tol=tol,
            restart=_GMRES_RESTART,
            maxiter=max_iter,
            solve_method="incremental",
        )[0]
        return _project(x)

    def _tangent_solve(linear: Callable[[Array], Array], rhs: Array) -> Array:
        """Solve the linear system of the implicit derivative.

        ``custom_linear_solve`` gives JAX an explicit transpose solve, so
        reverse mode never has to transpose the ``while_loop`` inside
        :func:`_krylov`.
        """
        return jax.lax.custom_linear_solve(linear, rhs, solve=_krylov, transpose_solve=_krylov)

    # custom_root keeps the loop's values and differentiates the fixed point
    # by the implicit function theorem, in forward and reverse mode. Only the
    # residual's inputs (C, scale, epsilon, log_b) carry derivatives; a
    # derivative in log_b is meaningful only for changes that keep its total
    # mass. The loop itself is never differentiated.
    g_safe, err = jax.lax.custom_root(
        _residual, jnp.zeros(log_b.shape, dtype=dtype), _solve, _tangent_solve, has_aux=True
    )
    log_f = _f_from_g(g_safe)
    log_g = jnp.where(valid, g_safe, neg_inf)

    log_plan = log_f[:, None] + log_K + log_g[None, :]
    # <P, C> on the scaled cost, mapped back to the original scale. Since
    # log_K = -(C / scale) / epsilon, C / scale = -epsilon * log_K, so the
    # cost reads off log_K and C need not stay live. exp of -inf (padded
    # columns) is exactly 0, and log_K is finite there, so pads add nothing.
    cost = -(jnp.exp(log_plan) * log_K).sum() * (scale * epsilon)
    return cost, err
