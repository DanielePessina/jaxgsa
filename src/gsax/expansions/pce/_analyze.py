"""PCE analysis and emulation entry points."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from gsax._transforms import cdf_to_unit_interval
from gsax.expansions.pce._engine import (
    build_design_matrix,
    build_multi_index,
    fit_coefficients,
    loo_error,
    sobol_from_coefficients,
)
from gsax.expansions.pce._result import PCEResult
from gsax.problem import Problem


def _map_to_reference(X: Array, problem: Problem) -> tuple[Array, tuple[str, ...]]:
    """Map physical inputs to the orthogonal polynomial reference domain.

    Uniform and truncated-Gaussian inputs use Legendre (mapped to [-1,1]).
    Untruncated Gaussian inputs use Hermite (standardized to N(0,1)).
    """
    D = problem.num_vars
    U = cdf_to_unit_interval(X, problem)

    cols = []
    input_types: list[str] = []
    for d in range(D):
        dist, first, second, lo, hi = problem._input_specs[d]
        if dist == "uniform" or lo is not None or hi is not None:
            cols.append(2.0 * U[:, d] - 1.0)
            input_types.append("uniform")
        else:
            mean, variance = first, second
            std = jnp.sqrt(variance)
            cols.append((X[:, d] - mean) / std)
            input_types.append("gaussian")

    return jnp.column_stack(cols), tuple(input_types)


def _auto_order(D: int, N: int, max_order: int, fit_ratio: float) -> int:
    """Reduce polynomial order so the term count fits within the sample budget."""
    from math import comb

    cap = max(1, int(fit_ratio * N))
    order = max_order
    while order >= 1 and comb(D + order, order) > cap:
        order -= 1
    return max(order, 1)


def analyze_pce(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    order: int = 3,
    ridge: float = 1e-8,
    fit_ratio: float = 0.5,
) -> PCEResult:
    """Compute Sobol indices via polynomial chaos expansion.

    Fits an orthogonal polynomial surrogate to (X, Y) data and extracts
    first-order, total-order, and second-order Sobol indices directly
    from the expansion coefficients (Sudret, 2008).

    Args:
        problem: Parameter names and distributions.
        X: (N, D) input samples.
        Y: (N,) model outputs (scalar output only for now).
        order: Maximum total polynomial degree. Automatically reduced
            if the number of terms would exceed ``fit_ratio * N``.
        ridge: Tikhonov regularization parameter for least-squares fit.
        fit_ratio: Maximum ratio of terms to samples before the order
            is reduced.

    Returns:
        PCEResult with S1, ST, S2, fitted coefficients, and LOO RMSE.
    """
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)

    if Y.ndim != 1:
        raise ValueError(
            f"PCE currently supports scalar output only (Y.ndim must be 1), got {Y.ndim}"
        )

    N, D = X.shape
    if D != problem.num_vars:
        raise ValueError(
            f"X has {D} columns but problem defines {problem.num_vars} parameters"
        )

    effective_order = _auto_order(D, N, order, fit_ratio)
    mi = build_multi_index(D, effective_order)

    X_ref, input_types = _map_to_reference(X, problem)
    Phi = build_design_matrix(X_ref, mi, input_types, effective_order)
    coeffs = fit_coefficients(Phi, Y, ridge=ridge)

    S1, ST, S2 = sobol_from_coefficients(coeffs, mi)

    loo = loo_error(Phi, Y, coeffs, ridge=ridge)

    return PCEResult(
        S1=S1,
        ST=ST,
        S2=S2,
        problem=problem,
        coefficients=coeffs,
        multi_index=mi,
        order=effective_order,
        loo_rmse=loo,
    )


def emulate_pce(result: PCEResult, X_new: Array) -> Array:
    """Predict at new input points using the fitted PCE.

    Args:
        result: PCEResult from ``analyze_pce``.
        X_new: (N_new, D) new input points.

    Returns:
        (N_new,) predicted outputs.
    """
    X_new = jnp.asarray(X_new)

    X_ref, input_types = _map_to_reference(X_new, result.problem)
    Phi = build_design_matrix(X_ref, result.multi_index, input_types, result.order)
    return Phi @ result.coefficients
