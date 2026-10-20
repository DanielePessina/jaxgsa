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
    Lamboni et al. (2013). Math. Comp. Sim. 87:45-54.
    Roustant et al. (2017). Electron. J. Statist. 11(2):3081-3119.
        doi: 10.1214/17-EJS1310.
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
        lo, hi = _effective_support(spec.mean, std, spec.low, spec.high)
        return _truncnorm_poincare(spec.mean, std, lo, hi, 512)
    if isinstance(spec, CategoricalSpec):
        raise ValueError(
            "A categorical marginal has no Poincare constant (the inequality "
            "needs a continuous density); jaxgsa.dgsm does not support "
            "categorical parameters"
        )
    raise ValueError(f"Unknown input spec {spec!r}")


# Tail mass cut off each end of the effective support. The weight there is far
# below float64 resolution of the spectral solve, so the constant is unchanged.
_TAIL_MASS = 1e-15

# Number of standard deviations around the mean that holds the whole mass of an
# open or ordinary truncated side to float precision.
_OPEN_SIDE_SD = 8.0

# A declared bound this many standard deviations from the mean, on the side
# that excludes the mean, leaves the mass piled against the bound. A fixed
# stand-in interval no longer holds it, so the quantile cut takes over. It also
# takes over for a bound past ``_OPEN_SIDE_SD`` on the mean's own side, where
# the weight underflows.
_FAR_BOUND_SD = 5.0


# Interval width, in standard deviations, below which the closed-form
# truncated variance cancels badly and the quadrature is used instead.
_NARROW_SD = 0.1


def _is_far(mean: float, std: float, low: float | None, high: float | None) -> bool:
    """Return whether a declared bound sits far from the mass.

    The Poincare constant and the marginal variance both switch to the
    quantile-cut path on this test, so they cannot drift apart.

    Args:
        mean: Mean of the underlying Gaussian.
        std: Standard deviation of the underlying Gaussian.
        low: Declared lower bound, or ``None`` if open.
        high: Declared upper bound, or ``None`` if open.

    Returns:
        ``True`` if a bound excludes the mean by more than 5 std, or lies more
        than 8 std from the mean.
    """
    z_low = None if low is None else (low - mean) / std
    z_high = None if high is None else (high - mean) / std
    far_above = z_low is not None and z_low > _FAR_BOUND_SD
    far_below = z_high is not None and -z_high > _FAR_BOUND_SD
    beyond_stand_in = (z_low is not None and -z_low > _OPEN_SIDE_SD) or (
        z_high is not None and z_high > _OPEN_SIDE_SD
    )
    return far_above or far_below or beyond_stand_in


def _effective_support(
    mean: float, std: float, low: float | None, high: float | None
) -> tuple[float, float]:
    """Return the interval the spectral solve runs on.

    Ordinarily each side is its declared bound, or ``mean +/- 8 std`` if the
    side is open. That is exact for any bound near the mean, and it keeps a
    truncation and its mirror image equal.

    The declared bounds can instead sit far from the mass, where the Gaussian
    weight underflows (a bound past 8 std from the mean). Or one bound can
    exclude the mean by more than 5 std. Then the mass piles against the bound nearest the mean,
    and a fixed stand-in such as ``mean +/- 8 std`` can lie on the wrong side
    of the declared bound on the other side. Only in that case are the bounds
    cut to the quantiles of the truncated distribution itself that leave
    ``1e-15`` of the mass in each tail. Those come from
    :class:`scipy.stats.truncnorm`, which works in log space and stays
    accurate in the far tail.

    Args:
        mean: Mean of the underlying Gaussian.
        std: Standard deviation of the underlying Gaussian.
        low: Declared lower bound, or ``None`` if open.
        high: Declared upper bound, or ``None`` if open.

    Returns:
        ``(lo, hi)``, inside the declared bounds, with ``lo < hi``.
    """
    lo = mean - _OPEN_SIDE_SD * std if low is None else low
    hi = mean + _OPEN_SIDE_SD * std if high is None else high
    far = _is_far(mean, std, low, high)
    if not far and (low is None) == (high is None):
        return lo, hi
    a_std = -np.inf if low is None else (low - mean) / std
    b_std = np.inf if high is None else (high - mean) / std
    # Mirror image, so the lower cut is as accurate as the upper one.
    lo_std = -float(truncnorm.isf(_TAIL_MASS, -b_std, -a_std))
    hi_std = float(truncnorm.isf(_TAIL_MASS, a_std, b_std))
    q_lo = mean + std * lo_std
    q_hi = mean + std * hi_std
    if far:
        lo, hi = q_lo, q_hi
        if low is not None:
            lo = max(lo, low)
        if high is not None:
            hi = min(hi, high)
        return lo, hi
    # Ordinary one-sided truncation: the open side reaches the wider of
    # mean +/- 8 std and the tail quantile. This matches the far path at the
    # switch, so the constant is continuous in the bound.
    if high is None:
        hi = max(hi, q_hi)
    if low is None:
        lo = min(lo, q_lo)
    return lo, hi


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
    z2 = ((xm - mu) / sigma) ** 2
    # exp(-0.5 * z**2) underflows to 0 for z beyond about 37 (float64), which
    # would make the mass matrix singular and eigh raise LinAlgError. The
    # eigenvalue solved for is a ratio of the same weight applied to both
    # matrices, so subtracting a constant off z**2 before exponentiating
    # rescales stiff and mass by the same factor and leaves it unchanged.
    w = np.exp(-0.5 * (z2 - z2.min()))
    n = grid + 1
    stiff = np.zeros((n, n))
    mass = np.zeros((n, n))
    for e in range(grid):
        we, he = w[e], h[e]
        stiff[e : e + 2, e : e + 2] += (we / he) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        mass[e : e + 2, e : e + 2] += (we * he / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    vals = np.sort(eigh(stiff, mass, eigvals_only=True))
    return float(1.0 / vals[1])


# Gauss-Legendre nodes for the truncated-Gaussian variance. The weight is a
# smooth Gaussian on an interval at most about 16 standard deviations wide,
# so this many nodes reach float64 resolution.
_GL_NODES = 256


def _std_truncnorm_variance(a: float, b: float) -> float:
    """Return the variance of a standard normal truncated to ``[a, b]``.

    The closed form ``1 + (a phi(a) - b phi(b)) / Z - (...)^2`` subtracts
    nearly equal numbers on a narrow interval or far in a tail, and it can
    return a negative variance. This quadrature never subtracts like that. It is
    used only when a bound is far from the mass (see ``_is_far``), or when
    the interval is narrower than ``_NARROW_SD`` standard deviations. It
    cuts the interval to its effective support, weights Gauss-Legendre nodes
    by the density (shifted so the heaviest weight is 1), and takes the
    second moment about the weighted mean.

    Args:
        a: Lower bound in standard units, possibly ``-inf``.
        b: Upper bound in standard units, possibly ``inf``.

    Returns:
        The variance, which is positive.
    """
    lo, hi = _effective_support(0.0, 1.0, None if np.isinf(a) else a, None if np.isinf(b) else b)
    nodes, weights = np.polynomial.legendre.leggauss(_GL_NODES)
    x = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    z2 = x**2
    w = weights * np.exp(-0.5 * (z2 - z2.min()))
    w /= w.sum()
    mean = float(np.sum(w * x))
    return float(np.sum(w * (x - mean) ** 2))


def marginal_variance(spec: InputSpec) -> float:
    """Return the marginal variance of a single input distribution.

    The reported ``lower_bound`` uses it:
        lower_bound_i = Var_i * w_i^2 / Var(Y)

    That expression is a proven lower bound on ST_i for a Gaussian marginal
    (Kucherenko & Song 2016, Theorem 6, Section 4.1, eq. 31). For a uniform
    or truncated-Gaussian marginal it is an estimate, not a bound: see
    :class:`jaxgsa.dgsm.DGSMResult`.

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
        # The closed form also loses all digits on a very narrow interval, so
        # that case takes the quadrature too.
        narrow = b_std - a_std < _NARROW_SD
        if narrow or _is_far(spec.mean, sd, spec.low, spec.high):
            return sd**2 * _std_truncnorm_variance(a_std, b_std)
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
