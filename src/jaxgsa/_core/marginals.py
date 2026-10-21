"""Inverse CDFs and CDFs of the lognormal, Weibull, triangular and beta marginals.

Each family has three implementations here:

* a host inverse CDF and CDF in float64 NumPy/SciPy, which build every stored
  design and invert it (:func:`host_ppf`, :func:`host_cdf`);
* a JAX CDF with broadcastable per-column parameters, which the given-data
  methods and PCE use to map inputs onto ``[0, 1]`` (:func:`jax_cdf`);
* a JAX inverse CDF that is differentiable in the distribution parameters,
  used by :meth:`jaxgsa.sobol.SobolSamples.transform` (:func:`jax_ppf`).

Callers clip the unit coordinate before they call an inverse CDF, as they do
for the Gaussian family. The SciPy parameter mapping is:

========== ===================================================================
lognormal  ``lognorm(s=sigma, scale=exp(mu))``
weibull    ``weibull_min(c=shape, scale=scale)``
triangular ``triang(c=(mode - low) / (high - low), loc=low, scale=high - low)``
beta       ``beta(a, b, loc=low, scale=high - low)``
========== ===================================================================
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import betainc, ndtr, ndtri
from scipy import special

from jaxgsa.problem import (
    BetaSpec,
    ContinuousFamilySpec,
    LognormalSpec,
    TriangularSpec,
    WeibullSpec,
)

# Distribution fields of each family, in the order :func:`jax_cdf` takes them.
FAMILY_FIELDS: dict[str, tuple[str, ...]] = {
    "lognormal": ("mu", "sigma"),
    "weibull": ("shape", "scale"),
    "triangular": ("low", "mode", "high"),
    "beta": ("a", "b", "low", "high"),
}


def family_params(spec: ContinuousFamilySpec) -> tuple[float, ...]:
    """Return the spec's parameters in :data:`FAMILY_FIELDS` order.

    Args:
        spec: A lognormal, Weibull, triangular or beta spec.

    Returns:
        The parameters as Python floats.
    """
    return tuple(float(getattr(spec, name)) for name in FAMILY_FIELDS[spec.dist])


def _triangular_c(low: Any, mode: Any, high: Any) -> Any:
    """Return the relative mode position ``(mode - low) / (high - low)``."""
    return (mode - low) / (high - low)


# ---------------------------------------------------------------------------
# Host (float64) inverse CDF and CDF
# ---------------------------------------------------------------------------


def host_ppf(spec: ContinuousFamilySpec, u: np.ndarray) -> np.ndarray:
    """Map clipped unit-interval values through the inverse CDF, in float64.

    Args:
        spec: A lognormal, Weibull, triangular or beta spec.
        u: Unit-interval values, already clipped off 0 and 1.

    Returns:
        Physical values, same shape as ``u``, dtype float64.

    Raises:
        TypeError: If ``spec`` is not one of the four families.
    """
    u = np.asarray(u, dtype=np.float64)
    if isinstance(spec, LognormalSpec):
        return np.exp(spec.mu + spec.sigma * special.ndtri(u))
    if isinstance(spec, WeibullSpec):
        # log1p, not log(1 - u): near u = 0 the plain form loses every digit.
        return spec.scale * (-np.log1p(-u)) ** (1.0 / spec.shape)
    if isinstance(spec, TriangularSpec):
        width = spec.high - spec.low
        c = _triangular_c(spec.low, spec.mode, spec.high)
        left = spec.low + np.sqrt(u * width * (spec.mode - spec.low))
        right = spec.high - np.sqrt((1.0 - u) * width * (spec.high - spec.mode))
        return np.where(u < c, left, right)
    if isinstance(spec, BetaSpec):
        return spec.low + (spec.high - spec.low) * special.betaincinv(spec.a, spec.b, u)
    raise TypeError(f"unsupported marginal {type(spec).__name__}")


def host_cdf(spec: ContinuousFamilySpec, x: np.ndarray) -> np.ndarray:
    """Map physical values through the CDF, in float64.

    Values outside the support map to 0 or 1.

    Args:
        spec: A lognormal, Weibull, triangular or beta spec.
        x: Physical values.

    Returns:
        CDF values in ``[0, 1]``, same shape as ``x``, dtype float64.

    Raises:
        TypeError: If ``spec`` is not one of the four families.
    """
    x = np.asarray(x, dtype=np.float64)
    if isinstance(spec, LognormalSpec):
        positive = x > 0
        safe = np.where(positive, x, 1.0)
        return np.where(positive, special.ndtr((np.log(safe) - spec.mu) / spec.sigma), 0.0)
    if isinstance(spec, WeibullSpec):
        t = np.maximum(x, 0.0) / spec.scale
        return -np.expm1(-(t**spec.shape))
    if isinstance(spec, TriangularSpec):
        z = np.clip((x - spec.low) / (spec.high - spec.low), 0.0, 1.0)
        c = _triangular_c(spec.low, spec.mode, spec.high)
        return np.asarray(_triangular_cdf_unit(np, z, c))
    if isinstance(spec, BetaSpec):
        z = np.clip((x - spec.low) / (spec.high - spec.low), 0.0, 1.0)
        return special.betainc(spec.a, spec.b, z)
    raise TypeError(f"unsupported marginal {type(spec).__name__}")


def _triangular_cdf_unit(xp: Any, z: Any, c: Any) -> Any:
    """Triangular CDF on the unit support, for ``z`` already in ``[0, 1]``.

    The degenerate modes ``c = 0`` and ``c = 1`` are allowed, so neither
    branch divides by its own width unguarded; the untaken branch reads a
    width of 1 (double-where), which also keeps its gradient finite.

    Args:
        xp: Array namespace, ``numpy`` or ``jax.numpy``.
        z: Values on the unit support.
        c: Relative mode position in ``[0, 1]``.

    Returns:
        CDF values in ``[0, 1]``.
    """
    left_ok = c > 0
    right_ok = c < 1
    c_left = xp.where(left_ok, c, 1.0)
    c_right = xp.where(right_ok, 1.0 - c, 1.0)
    left = z * z / c_left
    right = 1.0 - (1.0 - z) * (1.0 - z) / c_right
    return xp.where(z < c, left, right)


# ---------------------------------------------------------------------------
# JAX CDF (for the given-data methods and PCE)
# ---------------------------------------------------------------------------


def jax_cdf(dist: str, x: Array, params: tuple[Any, ...]) -> Array:
    """Map physical values through the CDF on device, keeping an out-of-support signal.

    A value inside the support maps into ``[0, 1]``. A value outside it maps
    outside ``[0, 1]``, as a uniform column does, so callers such as PAWN can
    drop it instead of folding it into an edge bin. The caller clips the
    in-range values off 0 and 1.

    Args:
        dist: Family tag: ``"lognormal"``, ``"weibull"``, ``"triangular"`` or
            ``"beta"``.
        x: Physical values, any shape.
        params: Parameters in :data:`FAMILY_FIELDS` order, each a scalar or an
            array that broadcasts against ``x`` (one value per column).

    Returns:
        Array of the same shape as ``x``.

    Raises:
        TypeError: If ``dist`` is not one of the four families.
    """
    if dist == "lognormal":
        mu, sigma = params
        positive = x > 0
        safe = jnp.where(positive, x, 1.0)
        u = ndtr((jnp.log(safe) - mu) / sigma)
        # x == 0 is the closed end of the support (CDF 0); x < 0 is outside.
        return jnp.where(positive, u, jnp.where(x == 0, 0.0, x))
    if dist == "weibull":
        shape, scale = params
        inside = x >= 0
        t = jnp.where(inside, x, 0.0) / scale
        u = -jnp.expm1(-(t**shape))
        return jnp.where(inside, u, x)
    if dist in ("triangular", "beta"):
        low, high = (params[0], params[2]) if dist == "triangular" else (params[2], params[3])
        z = (x - low) / (high - low)
        inside = (z >= 0) & (z <= 1)
        zc = jnp.clip(z, 0.0, 1.0)
        if dist == "triangular":
            u = _triangular_cdf_unit(jnp, zc, _triangular_c(low, params[1], high))
        else:
            a, b = params[0], params[1]
            u = betainc(jnp.asarray(a, zc.dtype), jnp.asarray(b, zc.dtype), zc)
        return jnp.where(inside, u, z)
    raise TypeError(f"unsupported marginal {dist}")


def stacked_params(specs: list[ContinuousFamilySpec], dtype: Any) -> tuple[Array, ...]:
    """Stack the parameters of same-family specs into per-column arrays.

    The values are gathered in float64 on the host and cast to ``dtype``,
    which is what a Python-float operand becomes inside a single-column op.

    Args:
        specs: Specs of one family, one per column.
        dtype: Target dtype of the columns.

    Returns:
        One array of shape ``(len(specs),)`` per family parameter.
    """
    values = np.asarray([family_params(spec) for spec in specs], dtype=np.float64)
    return tuple(jnp.asarray(values[:, j], dtype=dtype) for j in range(values.shape[1]))


# ---------------------------------------------------------------------------
# JAX inverse CDF (for SobolSamples.transform)
# ---------------------------------------------------------------------------


def jax_ppf(
    spec: ContinuousFamilySpec,
    u: Array,
    fields: dict[str, Any],
    *,
    beta_z: np.ndarray | None = None,
) -> Array:
    """Map clipped unit-interval values through the inverse CDF, differentiably.

    Lognormal, Weibull and triangular are closed forms in pure ``jnp``, so the
    result is differentiable in every parameter. The beta inverse CDF has no
    JAX implementation, so the caller computes ``z = betaincinv(a, b, u)`` on
    the host and passes it as ``beta_z``. The result ``low + (high - low) * z``
    is then differentiable in ``low`` and ``high`` only.

    Args:
        spec: A lognormal, Weibull, triangular or beta spec, supplying the
            default parameter values.
        u: Unit-interval values, already clipped off 0 and 1.
        fields: Parameter overrides by field name. Values may be tracers.
        beta_z: Host-computed standard-beta quantiles of ``u``. Required for
            a beta spec, ignored otherwise.

    Returns:
        Physical values, same shape as ``u``.

    Raises:
        TypeError: If ``spec`` is not one of the four families.
        ValueError: If ``spec`` is a beta spec and ``beta_z`` is missing.
    """
    if isinstance(spec, LognormalSpec):
        mu = fields.get("mu", spec.mu)
        sigma = fields.get("sigma", spec.sigma)
        return jnp.exp(mu + sigma * ndtri(u))
    if isinstance(spec, WeibullSpec):
        shape = fields.get("shape", spec.shape)
        scale = fields.get("scale", spec.scale)
        return scale * (-jnp.log1p(-u)) ** (1.0 / shape)
    if isinstance(spec, TriangularSpec):
        low = fields.get("low", spec.low)
        mode = fields.get("mode", spec.mode)
        high = fields.get("high", spec.high)
        width = high - low
        c = _triangular_c(low, mode, high)
        # Double-where: both branches run everywhere, and the sqrt of a
        # non-positive argument in the untaken branch would poison the
        # gradient with 0 * inf. Clamp each argument before the sqrt.
        tiny = jnp.finfo(u.dtype).tiny
        left_arg = jnp.maximum(u * width * (mode - low), tiny)
        right_arg = jnp.maximum((1.0 - u) * width * (high - mode), tiny)
        left = low + jnp.sqrt(left_arg)
        right = high - jnp.sqrt(right_arg)
        return jnp.where(u < c, left, right)
    if isinstance(spec, BetaSpec):
        if beta_z is None:
            raise ValueError("a beta inverse CDF needs host-computed quantiles (beta_z)")
        low = fields.get("low", spec.low)
        high = fields.get("high", spec.high)
        return low + (high - low) * jnp.asarray(beta_z, dtype=u.dtype)
    raise TypeError(f"unsupported marginal {type(spec).__name__}")


def beta_unit_quantiles(spec: BetaSpec, u: np.ndarray) -> np.ndarray:
    """Return the standard-beta quantiles ``betaincinv(a, b, u)`` in float64.

    Args:
        spec: The beta spec supplying ``a`` and ``b``.
        u: Unit-interval values, already clipped off 0 and 1.

    Returns:
        Quantiles on ``[0, 1]``, same shape as ``u``.
    """
    return special.betaincinv(spec.a, spec.b, np.asarray(u, dtype=np.float64))
