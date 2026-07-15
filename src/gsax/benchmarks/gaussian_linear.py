"""Gaussian linear additive model for sensitivity analysis benchmarking.

A weighted sum of independent Gaussian inputs. Like :mod:`gsax.benchmarks.linear`
it is purely additive (S1 == ST, zero interactions), but the Gaussian marginals
make the output and every conditional output Gaussian too, so the Borgonovo
delta index has a semi-analytic solution: the L1 distance between two Gaussian
densities is closed-form, and the outer expectation over each input reduces to
a 1-D Gauss-Hermite quadrature.

Useful for:
    - Ground-truth validation of moment-independent (delta) estimators,
      independent of any reference implementation.
    - The same S1 == ST / zero-interaction checks as the uniform linear model.
"""

import math

import numpy as np
from jax import Array
from scipy.stats import norm

from gsax.benchmarks.linear import _additive_sobol_indices
from gsax.benchmarks.linear import evaluate as _linear_evaluate
from gsax.problem import GaussianInputSpec, Problem

# Increasing weights (1, 2, 3) test whether the method correctly ranks
# input importance: x3 should have the largest index, x1 the smallest.
DEFAULT_COEFFS = (1.0, 2.0, 3.0)
DEFAULT_VARIANCES = (1.0, 1.0, 1.0)

PROBLEM = Problem.from_dict(
    {
        f"x{i + 1}": GaussianInputSpec(dist="gaussian", mean=0.0, variance=v)
        for i, v in enumerate(DEFAULT_VARIANCES)
    }
)


def evaluate(
    X: Array,
    coeffs: tuple[float, ...] = DEFAULT_COEFFS,
) -> Array:
    """Evaluate the Gaussian linear additive model.

    A weighted sum of the inputs, identical to
    :func:`gsax.benchmarks.linear.evaluate`; kept here as a thin wrapper so the
    Gaussian benchmark exposes ``evaluate`` under its own module namespace.

    .. math::
        f(\\mathbf{x}) = \\sum_{j=1}^{D} c_j \\, x_j

    Args:
        X: Input array of shape ``(N, D)``.
        coeffs: Coefficient per dimension.

    Returns:
        Array of shape ``(N,)`` with function values.
    """
    return _linear_evaluate(X, coeffs)


def analytical_indices(
    coeffs: tuple[float, ...] = DEFAULT_COEFFS,
    variances: tuple[float, ...] = DEFAULT_VARIANCES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytical Sobol indices for the Gaussian linear model.

    For ``Y = sum c_j x_j`` with independent ``x_j ~ N(mu_j, sigma_j^2)``:

    .. math::
        V_j = c_j^2 \\sigma_j^2, \\quad
        S_{1,j} = S_{T,j} = V_j / V(Y), \\quad
        S_{2,jk} = 0

    The means do not enter any index (all indices are translation invariant).

    Args:
        coeffs: Coefficient per dimension.
        variances: Variance of each Gaussian input.

    Returns:
        ``(S1, ST, S2)`` where S1 == ST (no interactions) and S2 is
        all-zero off-diagonal with NaN on the diagonal.
    """
    # The Gaussian marginals feed their variances straight into the shared
    # additive-model formula (the uniform benchmark derives var_x from bounds).
    return _additive_sobol_indices(
        np.asarray(coeffs, dtype=float),
        np.asarray(variances, dtype=float),
    )


def _gaussian_l1(
    mu1: float,
    v1: float,
    mu2: np.ndarray,
    v2: float,
) -> np.ndarray:
    """L1 distance between ``N(mu1, v1)`` and each ``N(mu2[k], v2)``.

    Two Gaussian densities with distinct variances cross at exactly two
    points (the roots of the quadratic obtained from equating log-densities);
    between them one density dominates, outside the other does. Because both
    densities integrate to one, the L1 distance is twice the absolute
    difference of the CDF increments over the crossing interval.

    When the variances coincide (within a small relative tolerance) that
    quadratic degenerates because ``a = 1/v1 - 1/v2 -> 0`` (its roots become
    ``0/0 -> NaN``). Equal-variance Gaussians differ only in mean and cross
    once, at the midpoint, so the L1 distance has the closed form
    ``2 * (2 * Phi(|mu2 - mu1| / (2 * sqrt(v))) - 1)`` -- exactly 0 when the
    means also coincide.

    Args:
        mu1: Mean of the first Gaussian.
        v1: Variance of the first Gaussian.
        mu2: Means of the second Gaussian (vectorized).
        v2: Variance of the second Gaussian.

    Returns:
        Array of L1 distances, same shape as ``mu2``, each in ``[0, 2]``.
    """
    mu2 = np.asarray(mu2, dtype=float)

    # Equal (or numerically indistinguishable) variances: the two-root formula
    # below divides by a -> 0, so use the single-crossing closed form instead.
    if abs(v1 - v2) <= 1e-9 * max(v1, v2):
        s = math.sqrt(0.5 * (v1 + v2))
        return 2.0 * (2.0 * norm.cdf(np.abs(mu2 - mu1) / (2.0 * s)) - 1.0)

    # Equate log-densities: (y-mu1)^2/v1 + ln v1 = (y-mu2)^2/v2 + ln v2
    a = 1.0 / v1 - 1.0 / v2
    b = -2.0 * (mu1 / v1 - mu2 / v2)
    c = mu1**2 / v1 - mu2**2 / v2 + math.log(v1 / v2)
    disc = b**2 - 4.0 * a * c  # > 0 whenever v1 != v2
    sqrt_disc = np.sqrt(disc)
    r1 = (-b - sqrt_disc) / (2.0 * a)
    r2 = (-b + sqrt_disc) / (2.0 * a)
    lo = np.minimum(r1, r2)
    hi = np.maximum(r1, r2)

    s1 = math.sqrt(v1)
    s2 = math.sqrt(v2)
    dP = norm.cdf((hi - mu1) / s1) - norm.cdf((lo - mu1) / s1)
    dQ = norm.cdf((hi - mu2) / s2) - norm.cdf((lo - mu2) / s2)
    return 2.0 * np.abs(dP - dQ)


def analytical_delta(
    coeffs: tuple[float, ...] = DEFAULT_COEFFS,
    variances: tuple[float, ...] = DEFAULT_VARIANCES,
    *,
    quad_order: int = 61,
) -> np.ndarray:
    """Compute Borgonovo delta indices for the Gaussian linear model.

    With ``Y ~ N(mu_Y, v_Y)`` and ``Y | X_i = x ~ N(mu + c_i x, v_Y - c_i^2 s_i^2)``,
    the inner L1 distance between the unconditional and conditional densities
    is closed-form (:func:`_gaussian_l1`); the outer expectation

    .. math::
        \\delta_i = \\tfrac{1}{2}\\, \\mathbb{E}_{X_i}\\!\\left[
            \\lVert f_Y - f_{Y|X_i} \\rVert_1 \\right]

    is evaluated with Gauss-Hermite quadrature over ``X_i``. Means do not
    affect the result (translation invariance), so only coefficients and
    variances are parameters.

    Args:
        coeffs: Coefficient per dimension.
        variances: Variance of each Gaussian input.
        quad_order: Number of Gauss-Hermite nodes for the outer expectation.

    Returns:
        Array of shape ``(D,)`` with delta indices in ``[0, 1]``.
    """
    c = np.asarray(coeffs, dtype=float)
    var_x = np.asarray(variances, dtype=float)
    var_y = float((c**2 * var_x).sum())

    # Physicists' Gauss-Hermite: E[g(Z)] = pi^{-1/2} sum_k w_k g(sqrt(2) t_k)
    nodes, weights = np.polynomial.hermite.hermgauss(quad_order)

    delta = np.zeros(len(c))
    for i, (ci, vi) in enumerate(zip(c, var_x)):
        if ci == 0.0 or vi == 0.0:
            # A zero coefficient or a constant (zero-variance) input leaves the
            # conditional density unchanged -> delta is exactly 0.
            continue
        v_cond = var_y - ci**2 * vi
        if v_cond <= 0.0:
            # Y is a deterministic function of X_i alone: the conditional
            # density is a Dirac mass, the L1 distance is 2, delta is 1.
            delta[i] = 1.0
            continue
        x = math.sqrt(2.0 * vi) * nodes
        l1 = _gaussian_l1(0.0, var_y, ci * x, v_cond)
        delta[i] = 0.5 * float((weights * l1).sum()) / math.sqrt(math.pi)

    return delta


def analytical_ot(
    coeffs: tuple[float, ...] = DEFAULT_COEFFS,
    variances: tuple[float, ...] = DEFAULT_VARIANCES,
) -> np.ndarray:
    """Compute optimal-transport sensitivity indices for the Gaussian linear model.

    With ``Y ~ N(0, v)`` and ``Y | X_i = x ~ N(c_i x, v - c_i^2 s_i^2)``
    (all Gaussian), the squared 2-Wasserstein distance between two
    Gaussians is closed-form, ``(mu1 - mu2)^2 + (sqrt(v1) - sqrt(v2))^2``,
    and the outer expectation over ``X_i`` is elementary
    (``E[c_i^2 X_i^2] = c_i^2 s_i^2``):

    .. math::
        \\iota_i = \\frac{c_i^2 \\sigma_i^2
            + (\\sqrt{v} - \\sqrt{v - c_i^2 \\sigma_i^2})^2}{2 v}

    The first numerator term is the advective (mean-shift) part -- exactly
    ``S1_i / 2`` after normalization -- and the second is the diffusive
    (spread) part. This is the point-conditioning (``M -> infinity``)
    limit of the partition estimator.

    Args:
        coeffs: Coefficient per dimension.
        variances: Variance of each Gaussian input.

    Returns:
        Array of shape ``(D,)`` with OT indices in ``[0, 1]``.
    """
    c = np.asarray(coeffs, dtype=float)
    var_x = np.asarray(variances, dtype=float)
    var_y = float((c**2 * var_x).sum())

    ot = np.zeros(len(c))
    for i, (ci, vi) in enumerate(zip(c, var_x)):
        if ci == 0.0 or vi == 0.0:
            # A zero coefficient or a constant input leaves the conditional
            # distribution unchanged -> the OT index is exactly 0.
            continue
        v_cond = var_y - ci**2 * vi
        if v_cond <= 0.0:
            # Y is a deterministic function of X_i alone: advective part
            # c_i^2 s_i^2 = v and diffusive part (sqrt(v) - 0)^2 = v -> 1.
            ot[i] = 1.0
            continue
        advective = ci**2 * vi
        diffusive = (math.sqrt(var_y) - math.sqrt(v_cond)) ** 2
        ot[i] = (advective + diffusive) / (2.0 * var_y)

    return ot


ANALYTICAL_S1, ANALYTICAL_ST, ANALYTICAL_S2 = analytical_indices()
ANALYTICAL_DELTA = analytical_delta()
ANALYTICAL_OT = analytical_ot()
