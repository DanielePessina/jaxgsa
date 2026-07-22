"""Log-domain Sinkhorn solver for entropic optimal transport.

Implements the standard entropic-regularization scheme for discrete
optimal transport (Cuturi 2013; Peyre & Cuturi 2019, "Computational
Optimal Transport") with log-domain scaling updates for numerical
stability. Used by the multivariate and trajectory modes of
:func:`jaxgsa.optimal_transport.analyze`
to transport the unconditional output point cloud onto each conditional
class; the ``"univariate"`` mode never needs a solver (1-D optimal transport
has a closed form via sorted quantiles).

The solver is a pure jit/vmap-compatible function: it never raises on
non-convergence (exceptions cannot cross a ``lax.while_loop``); instead it
returns the final marginal residual so the caller can warn once per
analysis after a single host sync.

References:
    Cuturi (2013). Sinkhorn distances: lightspeed computation of optimal
    transport. NeurIPS 26.

    Peyre & Cuturi (2019). Computational Optimal Transport. Foundations
    and Trends in Machine Learning 11(5-6):355-607.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

# Iterations between convergence checks: measuring the marginal residual
# costs as much as one scaling update, so checking every iteration would
# tax the whole loop by ~50%.
_CHECK_EVERY = 10


def _sinkhorn_w2(
    C: Array,
    log_b: Array,
    epsilon: Array,
    max_iter: Array,
    tol: Array,
) -> tuple[Array, Array]:
    """Entropic OT cost from a uniform source onto one masked target class.

    Solves the entropic optimal-transport problem between the uniform
    distribution on the ``N`` source points and the target histogram
    ``exp(log_b)`` over ``P`` (padded) target points, then reports the
    *unregularized* transport cost ``<P, C>`` of the entropic plan --
    not the epsilon-regularized objective -- so the value approaches the
    exact squared Wasserstein distance as ``epsilon -> 0``.

    The cost matrix is scaled by its maximum before solving (so
    ``epsilon`` is a scale-free regularization strength) and the returned
    cost is rescaled back. Updates run in the log domain; padded target
    columns are handled exactly by ``log_b = -inf`` (zero mass, inert in
    every ``logsumexp``).

    Args:
        C: Cost matrix ``(N, P)`` (squared Euclidean distances).
        log_b: Log target weights ``(P,)``: ``-log(n_m)`` for the class's
            ``n_m`` valid entries and ``-inf`` at padded entries.
        epsilon: Entropic regularization strength relative to the
            max-scaled cost (scalar).
        max_iter: Iteration cap (scalar).
        tol: Stopping tolerance on the L1 violation of the target
            marginal (scalar).

    Returns:
        A tuple ``(cost, err)`` where ``cost`` is the transport cost
        ``<P, C>`` on the original cost scale and ``err`` is the L1
        target-marginal violation at exit (``err <= tol`` means
        converged; it is refreshed on the final iteration).
    """
    dtype = C.dtype
    N = C.shape[0]
    b = jnp.exp(log_b)
    # Scale-free regularization: epsilon acts on a cost normalized to [0, 1].
    c_max = jnp.maximum(C.max(), jnp.asarray(jnp.finfo(dtype).tiny, dtype))
    log_K = -(C / c_max) / epsilon  # (N, P) log-kernel
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

    init = (
        jnp.zeros(N, dtype=dtype),
        jnp.zeros(log_b.shape, dtype=dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=dtype),
    )
    log_f, log_g, _, err = jax.lax.while_loop(_cond, _body, _body(init))

    log_plan = log_f[:, None] + log_K + log_g[None, :]
    # <P, C> on the scaled cost, mapped back to the original scale. exp of
    # -inf (padded columns) is exactly 0, so pads contribute nothing.
    cost = (jnp.exp(log_plan) * (C / c_max)).sum() * c_max
    return cost, err
