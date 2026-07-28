"""Poincare constants and marginal variances for DGSM bounds.

The Poincare constant C(p) of a distribution p is the sharpest factor
for which the Poincare inequality ``Var(g(X)) <= C(p) * E[g'(X)^2]``
holds for every smooth function g. The Sobol-Kucherenko inequality
applies it per input to bound the total Sobol index as

    ST_i <= C(p_i) * nu_i / Var(Y)

where C(p_i) is the Poincare constant of the i-th marginal and
``nu_i = E[(df/dx_i)^2]``.

Poincare constants by marginal type:
    Uniform [a, b]:      C = (b - a)^2 / pi^2
    Normal  N(mu, s^2):  C = s^2
    Truncated Normal:     spectral solve (P1 FEM Neumann eigenproblem)

References:
    Sobol' & Kucherenko (2009). Math. Comp. Sim. 79:3009-3017.
    Lamboni et al. (2013). Math. Comp. Sim. 87:44-54.
    Roustant et al. (2017). Stat. Comp. 27:879-894.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import truncnorm

from jaxgsa.problem import Problem, _NormalizedInputSpec


def poincare_constant(spec: _NormalizedInputSpec, *, grid: int = 512) -> float:
    """Poincare constant C(p) for a single marginal.

    Args:
        spec: Normalized input spec tuple (dist, first, second, low, high,
            categorical).
        grid: Number of P1 elements for truncated-Normal spectral solve.

    Returns:
        The (optimal) Poincare constant.
    """
    dist, first, second, low, high, _ = spec
    if dist == "uniform":
        return (second - first) ** 2 / math.pi**2
    if dist == "gaussian":
        sigma2 = second
        if low is None and high is None:
            return sigma2
        std = math.sqrt(sigma2)
        fallback_lo = first - 8 * std if low is None else low
        fallback_hi = first + 8 * std if high is None else high
        return _truncnorm_poincare(first, std, fallback_lo, fallback_hi, grid)
    if dist == "categorical":
        raise ValueError(
            "A categorical marginal has no Poincare constant (the inequality "
            "needs a continuous density); jaxgsa.dgsm does not support "
            "categorical parameters"
        )
    raise ValueError(f"Unknown distribution type {dist!r}")


def _truncnorm_poincare(mu: float, sigma: float, a: float, b: float, grid: int) -> float:
    """Optimal Poincare constant of N(mu, sigma^2) truncated to [a, b].

    Solves the weighted Neumann eigenproblem ``int rho g' h' = lam int rho g h``
    on [a, b] with a P1 finite-element basis. The constant is 1/lambda_1
    where lambda_1 is the spectral gap (smallest positive eigenvalue).

    Args:
        mu: Mean of the underlying Gaussian.
        sigma: Standard deviation of the underlying Gaussian.
        a: Lower truncation bound.
        b: Upper truncation bound.
        grid: Number of finite elements.

    Returns:
        The optimal Poincare constant.
    """
    from scipy.linalg import eigh

    x = np.linspace(a, b, grid + 1)
    h = np.diff(x)
    xm = 0.5 * (x[:-1] + x[1:])
    w = np.exp(-0.5 * ((xm - mu) / sigma) ** 2)
    n = grid + 1
    stiff = np.zeros((n, n))
    mass = np.zeros((n, n))
    for e in range(grid):
        we, he = w[e], h[e]
        stiff[e : e + 2, e : e + 2] += (we / he) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        mass[e : e + 2, e : e + 2] += (we * he / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    vals = np.sort(eigh(stiff, mass, eigvals_only=True))
    return float(1.0 / vals[1])


def marginal_variance(spec: _NormalizedInputSpec) -> float:
    """Marginal variance of a single input distribution.

    Used for the Kucherenko-Song lower bound on ST:
        ST_i >= Var_i * w_i^2 / Var(Y)

    Args:
        spec: Normalized input spec tuple.

    Returns:
        The variance of the marginal distribution.
    """
    dist, first, second, low, high, _ = spec
    if dist == "uniform":
        return (second - first) ** 2 / 12.0
    if dist == "gaussian":
        sigma2 = second
        if low is None and high is None:
            return sigma2
        mu = first
        sd = math.sqrt(sigma2)
        a_std = -np.inf if low is None else (low - mu) / sd
        b_std = np.inf if high is None else (high - mu) / sd
        return float(truncnorm.var(a_std, b_std, loc=mu, scale=sd))
    if dist == "categorical":
        raise ValueError(
            "A categorical marginal's level codes have no variance meaningful "
            "to the DGSM bounds; jaxgsa.dgsm does not support categorical "
            "parameters"
        )
    raise ValueError(f"Unknown distribution type {dist!r}")


def axis_constants(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis (Poincare constant, marginal variance) from a Problem.

    Args:
        problem: Problem definition with D parameters.

    Returns:
        (C, Var) each of shape (D,).
    """
    C = np.array(
        [poincare_constant(spec) for spec in problem.input_specs],
        dtype=np.float64,
    )
    Var = np.array(
        [marginal_variance(spec) for spec in problem.input_specs],
        dtype=np.float64,
    )
    return C, Var
