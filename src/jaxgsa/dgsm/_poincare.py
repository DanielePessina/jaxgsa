"""Poincare constants and marginal variances for DGSM bounds.

The Poincare constant C(p) of a distribution p is the sharpest factor for
which the Poincare inequality ``Var(g(X)) <= C(p) * E[g'(X)^2]`` holds for
every smooth function g. The Sobol-Kucherenko inequality applies it per input
to bound the total Sobol index as

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
from functools import lru_cache

import numpy as np
from scipy.stats import truncnorm

from jaxgsa.problem import CategoricalSpec, GaussianSpec, InputSpec, Problem, UniformSpec


def poincare_constant(spec: InputSpec) -> float:
    """Return the Poincare constant C(p) of a single marginal.

    Args:
        spec: Marginal spec of one parameter, as carried by
            :attr:`jaxgsa.Problem.input_specs`: a :class:`jaxgsa.UniformSpec`
            or :class:`jaxgsa.GaussianSpec`.

    Returns:
        The optimal Poincare constant.

    Raises:
        ValueError: If the marginal is categorical, which has no Poincare
            constant, or if the distribution type is unknown.
    """
    if isinstance(spec, UniformSpec):
        return (spec.high - spec.low) ** 2 / math.pi**2
    if isinstance(spec, GaussianSpec):
        if spec.low is None and spec.high is None:
            return spec.variance
        std = math.sqrt(spec.variance)
        # An open side gets a far-tail stand-in bound; the spectral solve needs
        # a finite interval, and 8 sigma carries the whole mass to float precision.
        fallback_lo = spec.mean - 8 * std if spec.low is None else spec.low
        fallback_hi = spec.mean + 8 * std if spec.high is None else spec.high
        return _truncnorm_poincare(spec.mean, std, fallback_lo, fallback_hi, 512)
    if isinstance(spec, CategoricalSpec):
        raise ValueError(
            "A categorical marginal has no Poincare constant (the inequality "
            "needs a continuous density); jaxgsa.dgsm does not support "
            "categorical parameters"
        )
    raise ValueError(f"Unknown input spec {spec!r}")


@lru_cache(maxsize=None)
def _truncnorm_poincare(mu: float, sigma: float, a: float, b: float, grid: int) -> float:
    """Return the optimal Poincare constant of N(mu, sigma^2) on [a, b].

    Solves the weighted Neumann eigenproblem ``int rho g' h' = lam int rho g h``
    on [a, b] with a P1 finite-element basis. The constant is 1/lambda_1, where
    lambda_1 is the spectral gap: the smallest positive eigenvalue.

    The solve is a pure function of five hashable scalars and it costs a dense
    generalized eigensolve, so the result is memoised with
    ``functools.lru_cache``: repeated ``analyze`` calls on the same truncated
    Gaussian marginal pay for the FEM once per process. Zero numeric change —
    the cached value is the value the solve returned.

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


def marginal_variance(spec: InputSpec) -> float:
    """Return the marginal variance of a single input distribution.

    The Kucherenko-Song lower bound on ST uses it:
        ST_i >= Var_i * w_i^2 / Var(Y)

    Args:
        spec: Marginal spec of one parameter, as carried by
            :attr:`jaxgsa.Problem.input_specs`: a :class:`jaxgsa.UniformSpec`
            or :class:`jaxgsa.GaussianSpec`.

    Returns:
        The variance of the marginal distribution.

    Raises:
        ValueError: If the marginal is categorical, whose level codes have no
            variance meaningful to the bounds, or if the distribution type is
            unknown.
    """
    if isinstance(spec, UniformSpec):
        return (spec.high - spec.low) ** 2 / 12.0
    if isinstance(spec, GaussianSpec):
        if spec.low is None and spec.high is None:
            return spec.variance
        sd = math.sqrt(spec.variance)
        a_std = -np.inf if spec.low is None else (spec.low - spec.mean) / sd
        b_std = np.inf if spec.high is None else (spec.high - spec.mean) / sd
        return float(truncnorm.var(a_std, b_std, loc=spec.mean, scale=sd))
    if isinstance(spec, CategoricalSpec):
        raise ValueError(
            "A categorical marginal's level codes have no variance meaningful "
            "to the DGSM bounds; jaxgsa.dgsm does not support categorical "
            "parameters"
        )
    raise ValueError(f"Unknown input spec {spec!r}")


def axis_constants(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-axis Poincare constant and marginal variance.

    Args:
        problem: Problem definition with D parameters.

    Returns:
        A tuple ``(C, Var)``. ``C`` holds the Poincare constants and ``Var`` the
        marginal variances, each of shape ``(D,)``.
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
