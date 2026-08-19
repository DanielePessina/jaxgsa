"""Shared low-level sampling helpers.

This module holds four groups of helper. Marginal transforms map the unit
cube to the problem's declared distributions and back. A second, JAX-native
copy of the forward transform does the same job differentiably, so a design
can be re-expressed as a function of its distribution parameters. Power-of-2
utilities size a Sobol' sequence. One deduplication routine drops repeated
design rows while keeping their first-occurrence order. The Sobol', Morris,
and eFAST samplers use them, as does :func:`jaxgsa.sampling.monte_carlo`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Mapping, TypeAlias

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import ndtr, ndtri
from scipy.stats import norm, truncnorm

from jaxgsa._core.precision import unit_clip_bounds
from jaxgsa.problem import CategoricalSpec, UniformSpec

if TYPE_CHECKING:
    from jaxgsa.problem import Problem


# Distribution parameters keyed by parameter name, then by field name. The
# field names are exactly those of the public input specs
# (:class:`jaxgsa.problem.UniformSpec` and
# :class:`jaxgsa.problem.GaussianSpec`), so a user writes the same words
# here as when the problem was declared. The values may be Python floats or
# JAX scalars, which is what makes the mapping a pytree that ``jax.grad`` and
# ``jax.jacrev`` can differentiate with respect to.
Theta: TypeAlias = Mapping[str, Mapping[str, Any]]


# Absolute support bound the library puts on every unbounded marginal.
#
# Unit-cube coordinates are clipped into ``[UNIT_CLIP, 1 - UNIT_CLIP]`` before
# an inverse CDF runs, because ``ppf(0)`` and ``ppf(1)`` return -inf and +inf.
# That clip is what bounds the support: no Gaussian draw can ever leave the
# range +/-7.0345 standard deviations from its mean.
#
# The value stays at 1e-12 on purpose. It is Hermite-safe to about degree 8,
# and the moment error it introduces is about 1e-10, far below any sampling
# error a user will see. A larger clip would visibly truncate the marginal. A
# smaller one runs into the tail resolution of the scipy quantile functions.
#
# The value is a float64 one. A device path that clips in float32 cannot use
# ``1 - UNIT_CLIP`` as written, because that expression rounds back to exactly
# 1.0 there; it goes through
# :func:`jaxgsa._core.precision.unit_clip_bounds` instead.
UNIT_CLIP = 1e-12


def _is_power_of_2(n: int) -> bool:
    """Check whether ``n`` is a positive power of 2."""
    return n >= 1 and (n & (n - 1)) == 0


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is >= ``n``."""
    if n <= 0:
        return 1
    # Bit-length trick: (n-1).bit_length() gives the position of the highest
    # set bit, so 1 << that yields the smallest power of 2 >= n.
    return 1 << (n - 1).bit_length()


def _power_of_2_error(name: str, value: int, *, reason: str) -> str:
    """Build a ValueError message for a value that must be a power of 2.

    Names the two nearest valid powers of 2 so the user can correct the
    value without guessing.

    Args:
        name: Parameter name to report (e.g. ``"base_n"``).
        value: The offending value.
        reason: Short justification shown in parentheses (e.g.
            ``"Sobol' sequence balance"``).

    Returns:
        The formatted error message.
    """
    if value < 2:
        lower, upper = 1, 2
    else:
        lower = 1 << (value.bit_length() - 1)
        upper = lower * 2
    return (
        f"{name} must be a power of 2 ({reason}); got {value} — nearest valid: {lower} or {upper}"
    )


def _transform_uniform(unit_values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Affine-map unit-interval samples into a finite uniform range."""
    return unit_values * (high - low) + low


def _transform_gaussian(
    unit_values: np.ndarray,
    mean: float,
    variance: float,
    *,
    low: float | None,
    high: float | None,
) -> np.ndarray:
    """Transform unit-interval samples into Gaussian or truncated Gaussian values."""
    # Inverse-CDF sampling (probability integral transform): if U ~ Uniform(0,1)
    # then F^{-1}(U) ~ F.  Clipping to (UNIT_CLIP, 1-UNIT_CLIP) prevents ppf
    # (percent-point function = quantile = inverse CDF) from returning +/-inf
    # at the boundaries.
    clipped = np.clip(unit_values, UNIT_CLIP, 1.0 - UNIT_CLIP)
    std = math.sqrt(variance)
    if low is None and high is None:
        return mean + std * norm.ppf(clipped)

    # Standardised truncation bounds a=(lo-mu)/sigma, b=(hi-mu)/sigma follow
    # scipy's truncnorm convention (standard-normal scale).
    a = -np.inf if low is None else (low - mean) / std
    b = np.inf if high is None else (high - mean) / std
    return truncnorm.ppf(clipped, a=a, b=b, loc=mean, scale=std)


def _transform_categorical(unit_values: np.ndarray, probs: tuple[float, ...]) -> np.ndarray:
    """Transform unit-interval samples into integer level codes (as floats).

    Step-function inverse CDF: level ``l`` is selected when the unit value
    falls into the ``l``-th cumulative-probability bin, so level frequencies
    follow ``probs`` exactly in distribution. Codes ``0 .. L-1`` are the
    sample values; labels never enter the sample matrix.
    """
    cum = np.cumsum(np.asarray(probs, dtype=np.float64))
    codes = np.searchsorted(cum, unit_values, side="right")
    # Float cumsum can end slightly below 1.0; clamp the (measure-zero)
    # overflow onto the last level.
    return np.minimum(codes, len(cum) - 1).astype(np.float64)


def _transform_samples(problem: Problem, samples_unit: np.ndarray) -> np.ndarray:
    """Transform unit-cube samples into the problem's declared marginals."""
    # Pre-allocate output; each column is filled independently by its marginal's inverse CDF
    transformed = np.empty_like(samples_unit, dtype=np.float64)

    for idx, spec in enumerate(problem.input_specs):
        if isinstance(spec, UniformSpec):
            transformed[:, idx] = _transform_uniform(samples_unit[:, idx], spec.low, spec.high)
        elif isinstance(spec, CategoricalSpec):
            transformed[:, idx] = _transform_categorical(samples_unit[:, idx], spec.probs)
        else:
            transformed[:, idx] = _transform_gaussian(
                samples_unit[:, idx],
                spec.mean,
                spec.variance,
                low=spec.low,
                high=spec.high,
            )

    return transformed


# ---------------------------------------------------------------------------
# Differentiable (JAX) copy of the forward transform
#
# The NumPy/SciPy transforms above stay the reference implementation: they run
# in float64 regardless of the JAX x64 flag, and every stored design is built
# with them, so their exact output is part of the library's baseline. The
# functions below reproduce the same mathematics in pure ``jnp`` so a design
# can be re-expressed as a differentiable function of its distribution
# parameters. With ``jax_enable_x64`` on they agree with the SciPy path to
# floating-point precision (measured max absolute difference 9e-16 on a
# uniform / Gaussian / truncated-Gaussian problem), not bitwise: ``ndtri`` and
# ``scipy.stats.norm.ppf`` are different rational approximations of the same
# function and differ in the last unit in the last place.
#
# There is deliberately no JAX copy of the categorical transform. Its inverse
# CDF is a step function, so its derivative is zero almost everywhere and
# undefined on the jumps. Callers reject categorical parameters instead.
# ---------------------------------------------------------------------------

# Which fields of a theta entry each marginal family accepts. A field outside
# this set is a typo or a distribution mix-up, and silently ignoring it would
# return a gradient of zero for a parameter the user believes they varied.
_THETA_FIELDS: dict[str, frozenset[str]] = {
    "uniform": frozenset({"low", "high"}),
    "gaussian": frozenset({"mean", "variance", "low", "high"}),
}


def _jax_transform_uniform(unit_values: Array, low: Any, high: Any) -> Array:
    """Affine-map unit-interval samples into a finite uniform range."""
    # Factor order matches _transform_uniform so the two agree bitwise in x64.
    return unit_values * (high - low) + low


def _jax_transform_gaussian(
    unit_values: Array,
    mean: Any,
    variance: Any,
    *,
    low: Any | None,
    high: Any | None,
) -> Array:
    """Transform unit-interval samples into Gaussian or truncated Gaussian values.

    The truncated inverse CDF is written out rather than taken from SciPy:
    rescale the unit value into the probability window the truncation leaves,
    then invert the standard normal. ``ndtr``/``ndtri`` are JAX primitives with
    derivative rules, so the whole chain differentiates. Whether a bound exists
    is a static Python fact, so that branch never blocks tracing; only the
    bound's *value* flows through the graph.

    Written directly, ``ndtri(Phi(a) + u (Phi(b) - Phi(a)))`` fails when the
    truncation window sits in the upper tail: ``Phi`` of both bounds rounds to
    exactly 1 (from about 8.2 standard deviations in float64), the window width
    becomes 0, and the whole column silently turns into ``ndtri(1) = inf``. The
    remedy is the same one :func:`jaxgsa._core.transforms._truncnorm_cdf` uses
    in the forward direction — reflect to the survival-function side, where the
    same probabilities are small and well separated. By the symmetry of the
    normal, ``truncnorm_ppf(u; a, b) == -truncnorm_ppf(1 - u; -b, -a)`` in
    standard units, so the upper-tail window is computed as a lower-tail one on
    the negated axis. The side is chosen branchlessly with ``jnp.where`` on the
    sign of the window midpoint, so the function stays ``jit``-, ``vmap``- and
    ``grad``-safe even when the bounds are traced. Each side's inputs are
    sanitised before ``ndtri`` (the double-``where`` idiom), because an ``inf``
    in the *untaken* branch would still poison reverse-mode gradients.

    Residual limit: when even the well-conditioned side underflows — both
    ``ndtr`` endpoints identically 0, which needs the whole window beyond about
    ``38.5`` standard deviations from the mean in float64 (about ``14`` in
    float32) — the truncated distribution is genuinely unrepresentable in the
    dtype and the result is infinite. The float64 host twin
    (:func:`_transform_gaussian` via ``scipy.stats.truncnorm.ppf``) hits the
    same wall at the same point.

    Args:
        unit_values: Unit-interval samples, any shape.
        mean: Mean of the (untruncated) Gaussian. May be a tracer.
        variance: Variance of the (untruncated) Gaussian. May be a tracer.
        low: Lower truncation bound, or ``None`` for unbounded below.
        high: Upper truncation bound, or ``None`` for unbounded above.

    Returns:
        Samples in physical units, same shape as ``unit_values``.
    """
    # Same clip as _transform_gaussian: ndtri(0) and ndtri(1) are -inf/+inf,
    # and an infinite sample would poison every downstream index. The host
    # twin runs in float64, where ``1 - UNIT_CLIP`` is a number below 1; this
    # copy runs in the caller's dtype, where in float32 it is not, so the
    # upper bound is brought inside the dtype. The lower bound is left alone:
    # float32 represents 1e-12 exactly, and widening it for symmetry would
    # discard information and move numbers for nothing. See unit_clip_bounds.
    unit_values = jnp.asarray(unit_values)
    clip_lo, clip_hi = unit_clip_bounds(UNIT_CLIP, unit_values.dtype)
    clipped = jnp.clip(unit_values, clip_lo, clip_hi)
    std = jnp.sqrt(variance)
    if low is None and high is None:
        return mean + std * ndtri(clipped)

    # The reflected side works with the complement of the unit coordinate. It
    # is re-clipped because the complement of the asymmetric clip is not the
    # clip: in float32, ``1 - UNIT_CLIP`` rounds back to exactly 1, and
    # ``ndtri(1)`` is the very infinity the clip exists to prevent.
    complement = jnp.clip(1.0 - clipped, clip_lo, clip_hi)

    if low is None:
        # (-inf, high]: the direct form is well conditioned everywhere. Phi of
        # the bound only loses precision near 1, which here means the window
        # covers essentially the whole line — the untruncated regime, where a
        # width of exactly 1 is harmless.
        return mean + std * ndtri(clipped * ndtr((high - mean) / std))
    if high is None:
        # [low, inf): the mirror image, always through the survival function.
        return mean + std * -ndtri(complement * ndtr((mean - low) / std))

    alpha = (low - mean) / std
    beta = (high - mean) / std
    # Reflect when the window midpoint sits above the mean. Near the mean both
    # forms are exact, so the exact crossover point is immaterial.
    upper = alpha + beta > 0.0
    # Double-where: each side's ndtr/ndtri chain also runs where the *other*
    # side is selected, and on a far-tail window the wrong side computes
    # ndtri(1) = inf. jnp.where discards the value, but reverse-mode AD would
    # still propagate a 0 * inf = nan cotangent through it, so the untaken
    # side's inputs are replaced with a benign window first.
    alpha_direct = jnp.where(upper, -1.0, alpha)
    beta_direct = jnp.where(upper, 1.0, beta)
    alpha_reflect = jnp.where(upper, alpha, -1.0)
    beta_reflect = jnp.where(upper, beta, 1.0)

    a = ndtr(alpha_direct)
    b = ndtr(beta_direct)
    direct = ndtri(a + clipped * (b - a))

    sf_high = ndtr(-beta_reflect)
    sf_low = ndtr(-alpha_reflect)
    reflected = -ndtri(sf_high + complement * (sf_low - sf_high))

    return mean + std * jnp.where(upper, reflected, direct)


def _validate_theta(problem: Problem, theta: Theta) -> None:
    """Check that a theta override names real parameters and real fields.

    Args:
        problem: Problem whose declared marginals theta overrides.
        theta: Distribution parameters keyed by parameter name, then field.

    Raises:
        ValueError: If theta names a parameter the problem does not declare,
            or a field the parameter's marginal family does not use.
    """
    for name, fields in theta.items():
        if name not in problem.names:
            raise ValueError(
                f"theta names parameter {name!r}, which the problem does not declare. "
                f"Known parameters: {list(problem.names)}"
            )
        dist = problem.input_specs[problem.names.index(name)].dist
        allowed = _THETA_FIELDS.get(dist, frozenset())
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise ValueError(
                f"theta[{name!r}] sets {unknown}, which a {dist!r} marginal does not use. "
                f"Allowed fields: {sorted(allowed)}"
            )


def _jax_transform_samples(
    problem: Problem, samples_unit: Array | np.ndarray, theta: Theta | None = None
) -> Array:
    """Transform unit-cube samples into declared marginals, differentiably.

    This is the JAX counterpart of :func:`_transform_samples`. It reads the
    distribution parameters from ``theta`` where the caller supplies them and
    from ``problem`` everywhere else, so a caller can differentiate the design
    with respect to a few fields without restating the rest.

    Args:
        problem: Problem definition supplying the default parameter values.
        samples_unit: Unit-cube samples, shape ``(N, D)``.
        theta: Optional overrides keyed by parameter name, then field name.
            ``None`` means "use the problem's own values throughout".

    Returns:
        Samples in physical units, shape ``(N, D)``.

    Raises:
        ValueError: If any parameter is categorical, or if ``theta`` names an
            unknown parameter or field.
    """
    if theta is None:
        theta = {}
    else:
        _validate_theta(problem, theta)

    samples_unit = jnp.asarray(samples_unit)
    columns = []
    for idx, (name, spec) in enumerate(zip(problem.names, problem.input_specs)):
        if isinstance(spec, CategoricalSpec):
            raise ValueError(
                f"Parameter {name!r} is categorical; its inverse CDF is a step "
                "function, which has no useful derivative"
            )
        fields = theta.get(name, {})
        unit_column = samples_unit[:, idx]
        if isinstance(spec, UniformSpec):
            columns.append(
                _jax_transform_uniform(
                    unit_column, fields.get("low", spec.low), fields.get("high", spec.high)
                )
            )
        else:
            columns.append(
                _jax_transform_gaussian(
                    unit_column,
                    fields.get("mean", spec.mean),
                    fields.get("variance", spec.variance),
                    low=fields.get("low", spec.low),
                    high=fields.get("high", spec.high),
                )
            )

    return jnp.stack(columns, axis=1)


def _inverse_transform_uniform(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Affine-map a finite uniform range back onto the unit interval."""
    return (values - low) / (high - low)


def _inverse_transform_gaussian(
    values: np.ndarray,
    mean: float,
    variance: float,
    *,
    low: float | None,
    high: float | None,
) -> np.ndarray:
    """Map Gaussian or truncated-Gaussian values back onto the unit interval."""
    # Forward CDF, the inverse of the ppf used by _transform_gaussian. Values
    # produced by that function were clipped to (UNIT_CLIP, 1-UNIT_CLIP)
    # beforehand, so the round trip cannot land exactly on 0 or 1 here.
    std = math.sqrt(variance)
    if low is None and high is None:
        return np.asarray(norm.cdf((values - mean) / std), dtype=np.float64)

    a = -np.inf if low is None else (low - mean) / std
    b = np.inf if high is None else (high - mean) / std
    return np.asarray(truncnorm.cdf(values, a=a, b=b, loc=mean, scale=std), dtype=np.float64)


def _inverse_transform_samples(problem: Problem, samples_physical: np.ndarray) -> np.ndarray:
    """Recover unit-cube coordinates from samples in the problem's marginals.

    Inverse of :func:`_transform_samples`, computed in float64 throughout.
    Precision matters because the recovered coordinates are differenced to
    form the denominator of an elementary effect. The float32 JAX equivalent
    (:func:`jaxgsa._core.transforms.cdf_to_unit_interval`) loses too many
    digits to divide by safely.

    Args:
        problem: Problem definition whose marginals produced the samples.
        samples_physical: Samples in physical units, shape ``(N, D)``.

    Returns:
        Unit-cube coordinates of shape ``(N, D)``, dtype float64.

    Raises:
        ValueError: If any parameter is categorical. Its step CDF maps a whole
            probability bin onto one code, so no unit-cube coordinate can be
            recovered.
    """
    unit = np.empty_like(samples_physical, dtype=np.float64)

    for idx, spec in enumerate(problem.input_specs):
        if isinstance(spec, CategoricalSpec):
            raise ValueError(
                f"Parameter {problem.names[idx]!r} is categorical; its step CDF is "
                "many-to-one, so physical samples cannot be mapped back onto the "
                "unit cube"
            )
        if isinstance(spec, UniformSpec):
            unit[:, idx] = _inverse_transform_uniform(
                samples_physical[:, idx], spec.low, spec.high
            )
        else:
            unit[:, idx] = _inverse_transform_gaussian(
                samples_physical[:, idx],
                spec.mean,
                spec.variance,
                low=spec.low,
                high=spec.high,
            )

    return unit


def _stable_unique_rows(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate rows while preserving first-occurrence order.

    Args:
        samples: ``(N, D)`` design matrix in physical units.

    Returns:
        ``(unique_samples, expanded_to_unique)`` where ``expanded_to_unique``
        maps each original row position in ``samples`` back to the retained
        unique row index.
    """
    # Ensure C-contiguous layout so the byte view below is well defined.
    samples = np.ascontiguousarray(samples)
    n_rows, n_cols = samples.shape

    if n_rows == 0:
        return (
            np.empty((0, n_cols), dtype=samples.dtype),
            np.empty(0, dtype=np.int64),
        )
    if n_cols == 0:
        # Every row holds the same (empty) value, so one row survives.
        return samples[:1].copy(), np.zeros(n_rows, dtype=np.int64)

    # Compare whole rows as raw bytes. This keeps the exact-match semantics of
    # the old ``row.tobytes()`` key: two rows merge only when they are bitwise
    # equal, so 0.0 and -0.0 stay distinct and equal NaN patterns still merge.
    # Exact deduplication is what we want here: if two rows are bitwise equal,
    # evaluating the model twice is wasteful.
    #
    # The void view turns each row into one Python ``bytes`` object, and
    # ``tolist()`` builds all N of them in one C loop. That is the expensive
    # part of the old per-row Python loop, which paid for a fresh array view
    # plus a ``tobytes()`` call on every row. Hashing the keys in a dict then
    # costs far less than sorting them.
    #
    # Measured on an M1 Pro at the hot case (N=2**20, D=20, float64): the old
    # per-row loop took 1099 ms, ``np.unique`` over the same void view took
    # 1128 ms (it falls back to a generic comparison sort with memcmp
    # callbacks, so it loses to hashing once rows are wide), ``np.lexsort``
    # over the columns took 5431 ms, and this form took 325 ms. Do not replace
    # this with a sort-based form without re-measuring.
    keys = samples.view(np.dtype((np.void, samples.dtype.itemsize * n_cols))).ravel().tolist()

    # ``setdefault`` numbers the rows in first-appearance order: the label of a
    # new row is the number of distinct rows seen before it.
    seen: dict[bytes, int] = {}
    setdefault = seen.setdefault
    expanded_to_unique = np.fromiter(
        (setdefault(key, len(seen)) for key in keys),
        dtype=np.int64,
        count=n_rows,
    )

    # A row is a first occurrence exactly when its label is higher than every
    # label before it, because the labels rise by one on each new row.
    is_first = np.empty(n_rows, dtype=bool)
    is_first[0] = True
    np.greater(
        expanded_to_unique[1:],
        np.maximum.accumulate(expanded_to_unique)[:-1],
        out=is_first[1:],
    )
    unique_samples = samples[np.flatnonzero(is_first)]
    return unique_samples, expanded_to_unique
