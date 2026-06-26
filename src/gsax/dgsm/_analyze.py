"""DGSM analysis: compute derivative-based sensitivity measures and bounds.

Computes the DGSM moments (nu, sigma) from a JAX-differentiable function
via reverse-mode autodiff, then derives Poincare upper bounds and
Kucherenko-Song lower bounds on the total Sobol index ST.

References:
    Sobol' & Kucherenko (2009). Math. Comp. Sim. 79:3009-3017.
    Kucherenko & Song (2016). Rel. Eng. Sys. Safety 148:81-95.
    Lamboni et al. (2013). Math. Comp. Sim. 93:53-61.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from gsax.dgsm._poincare import axis_constants
from gsax.dgsm._result import DGSMResult
from gsax.problem import Problem


def _promote_jac(jac: Array) -> Array:
    """Promote scalar-output Jacobian (N, D) to (N, 1, D) for uniform handling."""
    if jac.ndim == 2:
        return jac[:, None, :]
    return jac


def _compute_moments(
    fn: Callable,
    X: Array,
    chunk_size: int | None = None,
) -> tuple[Array, Array, Array]:
    """Compute DGSM moments and forward outputs via reverse-mode autodiff.

    Args:
        fn: JAX-differentiable function (D,) -> () or (D,) -> (T,).
        X: Sample matrix (N, D).
        chunk_size: If given, process in batches to limit memory.

    Returns:
        (Y, sigma, nu) where Y is (N,) or (N, T), sigma and nu are (T, D).
    """

    # Reverse-mode Jacobian is efficient when T (outputs) < D (inputs).
    # has_aux=True returns both Jacobian and forward output from a single
    # forward+backward pass, avoiding a redundant model evaluation.
    def _fn_aux(x: Array) -> tuple[Array, Array]:
        y = fn(x)
        return y, y

    combined = jax.jit(jax.vmap(jax.jacrev(_fn_aux, has_aux=True)))

    N = X.shape[0]

    if not chunk_size or chunk_size <= 0 or N <= chunk_size:
        jac_raw, Y = combined(X)
        jac = _promote_jac(jac_raw)
        return Y, jnp.mean(jac, axis=0), jnp.mean(jac**2, axis=0)

    # Chunked path: accumulate running sums to bound peak memory.
    # The first full chunk triggers XLA compilation; subsequent chunks reuse it.
    sum_jac: Array | None = None
    sum_jac2: Array | None = None
    Y_parts: list[Array] = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        actual_len = end - start

        # Pad ragged last chunk to avoid JIT recompilation for a new shape
        X_chunk = X[start:end]
        if actual_len < chunk_size:
            pad_size = chunk_size - actual_len
            X_chunk = jnp.concatenate([X_chunk, jnp.zeros((pad_size, X.shape[1]))], axis=0)

        jac_raw, Y_chunk_full = combined(X_chunk)
        Y_parts.append(Y_chunk_full[:actual_len])

        jac = _promote_jac(jac_raw)[:actual_len]
        sj = jnp.sum(jac, axis=0)
        sj2 = jnp.sum(jac**2, axis=0)
        sum_jac = sj if sum_jac is None else sum_jac + sj
        sum_jac2 = sj2 if sum_jac2 is None else sum_jac2 + sj2

    Y = jnp.concatenate(Y_parts, axis=0)
    assert sum_jac is not None and sum_jac2 is not None
    return Y, sum_jac / N, sum_jac2 / N


def analyze(
    problem: Problem,
    fn: Callable | None = None,
    X: Array | None = None,
    *,
    Y: Array | None = None,
    dfdx: Array | None = None,
    chunk_size: int | None = None,
) -> DGSMResult:
    """Compute DGSM sensitivity indices and Sobol index bounds.

    Two calling conventions are supported:

    **Autodiff path** (primary): pass ``fn`` and ``X``. The function is
    differentiated via ``jax.jacrev`` and evaluated to obtain both
    the Jacobian and forward outputs.

    **Pre-computed path**: pass ``Y`` and ``dfdx``. Useful when the model
    is not JAX-differentiable or when the Jacobian has been computed
    externally.

    Args:
        problem: Problem definition with D parameters.
        fn: JAX-differentiable function ``(D,) -> ()`` or ``(D,) -> (T,)``.
        X: Sample matrix ``(N, D)`` in the problem's physical units.
        Y: Forward model outputs ``(N,)`` or ``(N, T)``.
        dfdx: Pre-computed Jacobian ``(N, D)`` or ``(N, T, D)``.
        chunk_size: Batch size for autodiff (limits memory).

    Returns:
        DGSMResult with nu, sigma, upper_bound, lower_bound, and var_y.
    """
    D = problem.num_vars

    if fn is not None and X is not None:
        X = jnp.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, D), got ndim={X.ndim}")
        if X.shape[1] != D:
            raise ValueError(f"X has {X.shape[1]} columns but problem has {D} parameters")
        Y_out, sigma, nu = _compute_moments(fn, X, chunk_size=chunk_size)
    elif Y is not None and dfdx is not None:
        # Pre-computed path: user supplies Jacobian and forward outputs directly
        Y_out = jnp.asarray(Y)
        dfdx_arr = jnp.asarray(dfdx)
        if dfdx_arr.ndim == 2:
            dfdx_arr = dfdx_arr[:, None, :]  # scalar -> (N, 1, D)
        if dfdx_arr.ndim != 3:
            raise ValueError(f"dfdx must be 2-D (N, D) or 3-D (N, T, D), got ndim={dfdx_arr.ndim}")
        if dfdx_arr.shape[-1] != D:
            raise ValueError(
                f"dfdx last dimension ({dfdx_arr.shape[-1]}) must match problem.num_vars ({D})"
            )
        if dfdx_arr.shape[0] != Y_out.shape[0]:
            raise ValueError(
                f"dfdx rows ({dfdx_arr.shape[0]}) must match Y rows ({Y_out.shape[0]})"
            )
        sigma = jnp.mean(dfdx_arr, axis=0)  # E[df/dx_i]
        nu = jnp.mean(dfdx_arr**2, axis=0)  # E[(df/dx_i)^2]
    else:
        raise ValueError("Provide either (fn, X) or (Y, dfdx)")

    # Var(Y) per output component, needed as denominator for both bounds
    var_y = jnp.atleast_1d(jnp.var(Y_out, axis=0))

    # Per-axis constants: C for Poincare upper bound, Var for lower bound
    C, Var = axis_constants(problem)
    C_jnp = jnp.asarray(C)
    Var_jnp = jnp.asarray(Var)

    # Guard against zero variance (constant output) -> NaN bounds
    denom = jnp.where(var_y == 0, jnp.nan, var_y)
    denom = denom[:, None]  # (T, 1) for broadcasting with (T, D)

    # Upper: ST_i <= C_i * nu_i / Var(Y)  (Sobol-Kucherenko inequality)
    upper = C_jnp[None, :] * nu / denom
    # Lower: ST_i >= Var_i * sigma_i^2 / Var(Y)  (Kucherenko-Song)
    lower = Var_jnp[None, :] * sigma**2 / denom

    # Sanity check: upper should be >= lower within numerical tolerance
    if jnp.any(jnp.isfinite(upper) & jnp.isfinite(lower) & (upper < lower * 0.9)):
        warnings.warn(
            "DGSM: some upper bounds are below lower bounds, suggesting "
            "insufficient samples or numerical issues",
            stacklevel=2,
        )

    return DGSMResult(
        nu=nu,
        sigma=sigma,
        upper_bound=upper,
        lower_bound=lower,
        var_y=var_y,
        problem=problem,
    )
