"""Shared low-level sampling helpers.

This module holds five groups of helper. Argument guards check what a
sampler or a design-based ``analyze()`` was given. Marginal transforms map the unit
cube to the problem's declared distributions and back. A second, JAX-native
copy of the forward transform does the same job differentiably, so a design
can be re-expressed as a function of its distribution parameters. Power-of-2
utilities size a Sobol' sequence. One deduplication routine drops repeated
design rows while keeping their first-occurrence order. The Sobol', Morris,
and eFAST samplers use them, as does :func:`jaxgsa.sampling.monte_carlo`.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, TypeAlias

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import ndtr, ndtri
from scipy.stats import norm, truncnorm

from jaxgsa._core.precision import unit_clip_bounds
from jaxgsa.problem import (
    _CONTINUOUS_FAMILY_SPECS,
    BetaSpec,
    CategoricalSpec,
    GaussianSpec,
    Problem,
    UniformSpec,
)


def check_seed(seed: object, who: str) -> None:
    """Refuse a JAX key passed as a sampler ``seed``.

    A sampler draws on the host with NumPy and SciPy, so it takes an ``int``
    or a ``numpy.random.Generator``. A JAX key would otherwise fail deep in
    NumPy with an error that does not name the mistake.

    Args:
        seed: The value the user passed as ``seed=``.
        who: Qualified function name for the message, such as
            ``"jaxgsa.sobol.sample"``.

    Raises:
        TypeError: If ``seed`` is a JAX array, which includes typed keys.
    """
    if isinstance(seed, jax.Array):
        raise TypeError(
            f"{who}: seed takes an int or numpy Generator; jax keys go to analyze(key=...)"
        )


def check_budget_choice(n_samples: int | None, base_n: int | None, where: str) -> None:
    """Require exactly one of ``n_samples`` and ``base_n``.

    Args:
        n_samples: Minimum run budget, or ``None``.
        base_n: Exact base-point count, or ``None``.
        where: Fully qualified name of the calling sampler, for the message.

    Raises:
        TypeError: If a given value is not an int (``bool`` included).
        ValueError: If both or neither are given, or a value is below 1.
    """
    if n_samples is None and base_n is None:
        raise ValueError(
            f"{where}: give exactly one of n_samples (minimum number of model "
            "runs) or base_n (exact number of base points). Neither was given."
        )
    if n_samples is not None and base_n is not None:
        raise ValueError(
            f"{where}: give exactly one of n_samples (minimum number of model "
            "runs) or base_n (exact number of base points), not both."
        )
    for name, value in (("n_samples", n_samples), ("base_n", base_n)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{where}: {name} must be an int, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"{where}: {name} must be >= 1, got {value}")


def require_design(obj: object, cls: type, function: str, sampler: str) -> None:
    """Require the design class that ``function`` analyzes.

    Args:
        obj: The first argument the caller passed.
        cls: The design class the function accepts.
        function: Fully qualified name of the analyzing function.
        sampler: Fully qualified name of the sampler that builds ``cls``.

    Raises:
        TypeError: If ``obj`` is not an instance of ``cls``. The message names
            the type received and points given-data users to the right methods.
    """
    if isinstance(obj, cls):
        return
    received = "a Problem" if isinstance(obj, Problem) else f"a {type(obj).__name__}"
    raise TypeError(
        f"{function} takes the {cls.__name__} from {sampler} as its first "
        f"argument, not {received}. For given-data X and Y, use a given-data "
        "method such as jaxgsa.rank or jaxgsa.pce."
    )


def check_seed_scramble(scramble: bool, seed: Any, where: str) -> None:
    """Reject a seed when the Sobol' sequence is not scrambled.

    Args:
        scramble: Whether the sequence is scrambled.
        seed: The seed the caller passed.
        where: Fully qualified name of the calling sampler, for the message.

    Raises:
        ValueError: If ``scramble`` is false and a seed is given.
    """
    if not scramble and seed is not None:
        raise ValueError(
            f"{where}: seed has no effect with scramble=False. "
            "The unscrambled Sobol' sequence is deterministic, so the seed "
            "would do nothing. Use scramble=True, or drop the seed."
        )


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


def _transform_samples(
    problem: Problem, samples_unit: np.ndarray, *, out: np.ndarray | None = None
) -> np.ndarray:
    """Transform unit-cube samples into the problem's declared marginals.

    Args:
        problem: Problem supplying one marginal per column.
        samples_unit: ``(N, D)`` points on the unit cube.
        out: Optional ``(N, D)`` float64 array to write the result into. It
            may be ``samples_unit`` itself: every column is computed from its
            own unit column in full before it is written back, so the
            in-place transform gives the same values and saves a full copy of
            the design. ``None`` allocates a new array.

    Returns:
        ``(N, D)`` float64 samples in physical units (``out`` when given).
    """
    # Each column is filled independently by its marginal's inverse CDF.
    transformed = np.empty_like(samples_unit, dtype=np.float64) if out is None else out

    for idx, spec in enumerate(problem.input_specs):
        if isinstance(spec, UniformSpec):
            transformed[:, idx] = _transform_uniform(samples_unit[:, idx], spec.low, spec.high)
        elif isinstance(spec, CategoricalSpec):
            transformed[:, idx] = _transform_categorical(samples_unit[:, idx], spec.probs)
        elif isinstance(spec, GaussianSpec):
            transformed[:, idx] = _transform_gaussian(
                samples_unit[:, idx],
                spec.mean,
                spec.variance,
                low=spec.low,
                high=spec.high,
            )
        elif isinstance(spec, _CONTINUOUS_FAMILY_SPECS):
            # Same clip as the Gaussian: the lognormal and Weibull inverse
            # CDFs are infinite at u = 1, and it is harmless for the bounded
            # triangular and beta families.
            clipped = np.clip(samples_unit[:, idx], UNIT_CLIP, 1.0 - UNIT_CLIP)
            transformed[:, idx] = spec.ppf(clipped)

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
    # Each continuous family declares its own. The beta shape parameters a
    # and b are not among them; see _validate_theta for the refusal.
    **{cls.dist: cls.theta_fields for cls in _CONTINUOUS_FAMILY_SPECS},
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
    ``38`` standard deviations from the mean in float64 (about ``14`` in
    float32) — the result is infinite. The float64 host twin
    (:func:`_transform_gaussian` via ``scipy.stats.truncnorm.ppf``) does not
    hit this wall: scipy's algorithm stays finite past 1000 standard
    deviations (measured). For a declared truncation window that far out,
    ``SobolSamples.samples`` (built on the scipy host path) and
    ``.transform()`` (built on this function) therefore disagree for the
    affected rows — finite against ``inf``.

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
            a field the parameter's marginal family does not use, or a field
            set to ``None``.
    """
    for name, fields in theta.items():
        if name not in problem.names:
            raise ValueError(
                f"theta names parameter {name!r}, which the problem does not declare. "
                f"Known parameters: {list(problem.names)}"
            )
        dist = problem.input_specs[problem.names.index(name)].dist
        if dist == "beta":
            shape_fields = sorted(set(fields) & {"a", "b"})
            if shape_fields:
                raise ValueError(
                    f"theta[{name!r}] sets the beta shape parameter(s) {shape_fields}, "
                    "which transform() cannot differentiate: JAX has no inverse of the "
                    "regularized incomplete beta function, and its betainc has no "
                    "gradient in a or b. Only 'low' and 'high' of a beta input can "
                    "be varied."
                )
        allowed = _THETA_FIELDS.get(dist, frozenset())
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise ValueError(
                f"theta[{name!r}] sets {unknown}, which a {dist!r} marginal does not use. "
                f"Allowed fields: {sorted(allowed)}"
            )
        none_fields = [field for field, value in fields.items() if value is None]
        if none_fields:
            raise ValueError(
                f"theta[{name!r}] sets {none_fields} to None, which a {dist!r} "
                "marginal does not accept. Omit the field to keep the "
                "problem's declared value."
            )


def beta_unit_quantiles(
    problem: Problem, samples_unit: Array | np.ndarray
) -> dict[int, np.ndarray]:
    """Compute the standard-beta quantiles of every beta column, once, on the host.

    The beta inverse CDF has no JAX implementation. Its standard quantiles
    depend only on the unit design and the fixed shape parameters, so a
    design computes them once and passes them to every
    :func:`_jax_transform_samples` call.

    Args:
        problem: Problem whose beta parameters are read.
        samples_unit: Unit-cube samples, shape ``(N, D)``. Must be concrete
            when the problem has a beta parameter.

    Returns:
        ``betaincinv(a, b, u)`` in float64, keyed by column index, for each
        beta column. Empty when the problem has no beta parameter.

    Raises:
        ValueError: If the problem has a beta parameter and ``samples_unit``
            is traced.
    """
    beta = [
        name
        for name, spec in zip(problem.names, problem.input_specs)
        if isinstance(spec, BetaSpec)
    ]
    if beta and isinstance(samples_unit, jax.core.Tracer):
        raise ValueError(
            f"Parameter {beta[0]!r} is beta-distributed; its inverse CDF is computed "
            "on the host, so the unit design must be a concrete array, not a "
            "traced one"
        )
    quantiles: dict[int, np.ndarray] = {}
    for idx, spec in enumerate(problem.input_specs):
        if isinstance(spec, BetaSpec):
            unit_column = np.clip(
                np.asarray(samples_unit[:, idx], dtype=np.float64), UNIT_CLIP, 1.0 - UNIT_CLIP
            )
            quantiles[idx] = spec.unit_quantiles(unit_column)
    return quantiles


def _jax_transform_samples(
    problem: Problem,
    samples_unit: Array | np.ndarray,
    theta: Theta | None = None,
    unit_quantiles: Mapping[int, np.ndarray] | None = None,
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
        unit_quantiles: The beta columns' standard quantiles, from
            :func:`beta_unit_quantiles` on the same ``samples_unit``. A design
            computes them once and passes them on every call. ``None``
            computes them here, which needs a concrete ``samples_unit``.

    Returns:
        Samples in physical units, shape ``(N, D)``.

    Raises:
        ValueError: If any parameter is categorical, if ``theta`` names an
            unknown parameter or field, or if the problem has a beta
            parameter, ``unit_quantiles`` is ``None`` and ``samples_unit`` is
            traced.
    """
    if theta is None:
        theta = {}
    else:
        _validate_theta(problem, theta)

    if unit_quantiles is None:
        unit_quantiles = beta_unit_quantiles(problem, samples_unit)

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
        elif isinstance(spec, GaussianSpec):
            columns.append(
                _jax_transform_gaussian(
                    unit_column,
                    fields.get("mean", spec.mean),
                    fields.get("variance", spec.variance),
                    low=fields.get("low", spec.low),
                    high=fields.get("high", spec.high),
                )
            )
        elif isinstance(spec, _CONTINUOUS_FAMILY_SPECS):
            # Same clip as the Gaussian, brought inside the column dtype. The
            # Weibull shape gradient is NaN at u = 0, and the lognormal and
            # Weibull inverse CDFs are infinite at u = 1.
            clip_lo, clip_hi = unit_clip_bounds(UNIT_CLIP, unit_column.dtype)
            columns.append(
                spec.jax_ppf(
                    jnp.clip(unit_column, clip_lo, clip_hi),
                    fields,
                    unit_quantiles=unit_quantiles.get(idx),
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
        elif isinstance(spec, GaussianSpec):
            unit[:, idx] = _inverse_transform_gaussian(
                samples_physical[:, idx],
                spec.mean,
                spec.variance,
                low=spec.low,
                high=spec.high,
            )
        elif isinstance(spec, _CONTINUOUS_FAMILY_SPECS):
            unit[:, idx] = spec.cdf(samples_physical[:, idx])

    return unit


# Multipliers of the row fingerprint in _row_fingerprints. Any odd 64-bit
# constants work: the fingerprint only has to spread distinct rows apart, and
# every group it forms is verified bit for bit before it is trusted.
_FP_MUL = np.uint64(0x9E3779B97F4A7C15)
_FP_MIX = np.uint64(0xBF58476D1CE4E5B9)
# Rows fingerprinted per block: a block of row words stays in cache across
# the column loop, and the per-block temporaries stay a few MB.
_FP_BLOCK_ROWS = 16384


def _row_fingerprints(words: np.ndarray) -> np.ndarray:
    """Hash each row of a ``(N, W)`` uint64 array to one uint64.

    Equal rows always get equal fingerprints; distinct rows almost always get
    distinct ones. The caller verifies every collision exactly, so the hash
    quality affects only speed, never the result.

    Args:
        words: ``(N, W)`` C-contiguous uint64 view of the rows.

    Returns:
        ``(N,)`` uint64 fingerprints.
    """
    n_rows, n_words = words.shape
    out = np.empty(n_rows, dtype=np.uint64)
    shift = np.uint64(31)
    with np.errstate(over="ignore"):
        for start in range(0, n_rows, _FP_BLOCK_ROWS):
            block = words[start : start + _FP_BLOCK_ROWS]
            h = np.full(block.shape[0], n_words, dtype=np.uint64)
            for j in range(n_words):
                h ^= block[:, j]
                h *= _FP_MUL
                h ^= h >> shift
            h *= _FP_MIX
            h ^= h >> shift
            out[start : start + block.shape[0]] = h
    return out


def _unique_rows_by_fingerprint(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Label rows by first occurrence, grouping them by a verified fingerprint.

    Rows are grouped by :func:`_row_fingerprints`, and every row is then
    compared word for word with the first row of its group. Equal rows always
    share a fingerprint, so when every comparison holds the grouping is
    exactly the bitwise grouping of :func:`_stable_unique_rows`. The extra
    memory is a few ``(N,)`` integer arrays, where hashing the rows as Python
    ``bytes`` keys holds a second full copy of the design.

    Args:
        samples: ``(N, D)`` C-contiguous array with ``N >= 1``, ``D >= 1``
            and a row size that is a multiple of 8 bytes.

    Returns:
        ``(is_first, expanded_to_unique)`` as :func:`_stable_unique_rows`
        defines them, or ``None`` when two distinct rows share a fingerprint
        (the caller then falls back to exact keys).
    """
    n_rows = samples.shape[0]
    words = samples.view(np.uint64).reshape(n_rows, -1)
    fp = _row_fingerprints(words)
    # A stable sort keeps each fingerprint group in row order, so the first
    # entry of a group is its first occurrence.
    order = np.argsort(fp, kind="stable")
    sorted_fp = fp[order]
    del fp
    group_start = np.empty(n_rows, dtype=bool)
    group_start[0] = True
    np.not_equal(sorted_fp[1:], sorted_fp[:-1], out=group_start[1:])
    del sorted_fp
    # first_of[r]: the first row of r's fingerprint group.
    group_id = np.cumsum(group_start) - 1
    first_of = np.empty(n_rows, dtype=np.int64)
    first_of[order] = order[np.flatnonzero(group_start)][group_id]
    del order, group_start, group_id
    # Verify, a block of rows at a time so the comparison never holds a
    # full-size boolean array.
    for start in range(0, n_rows, _FP_BLOCK_ROWS):
        stop = min(start + _FP_BLOCK_ROWS, n_rows)
        reps = first_of[start:stop]
        if not np.array_equal(words[start:stop], words[reps]):
            return None
    is_first = first_of == np.arange(n_rows)
    # The label of a first occurrence is the number of first occurrences
    # before it; every other row takes the label of its group's first row.
    label_of_first = np.cumsum(is_first) - 1
    return is_first, label_of_first[first_of]


def _stable_unique_rows(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate rows while preserving first-occurrence order.

    Two rows merge only when they are bitwise equal, so 0.0 and -0.0 stay
    distinct and equal NaN patterns merge.

    When no row repeats, the unique rows are the input rows, and the input
    array is returned as the unique array rather than copied (after
    ``np.ascontiguousarray``). The callers build the input for this call and
    never write to it afterwards, so sharing it is safe and saves a full copy
    of the design.

    Args:
        samples: ``(N, D)`` design matrix in physical units.

    Returns:
        ``(unique_samples, expanded_to_unique)`` where ``expanded_to_unique``
        maps each original row position in ``samples`` back to the retained
        unique row index.
    """
    # Ensure C-contiguous layout so the byte views below are well defined.
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

    found = None
    if (samples.dtype.itemsize * n_cols) % 8 == 0:
        found = _unique_rows_by_fingerprint(samples)
    if found is None:
        found = _unique_rows_by_bytes(samples)
    is_first, expanded_to_unique = found
    if is_first.all():
        return samples, expanded_to_unique
    return samples[np.flatnonzero(is_first)], expanded_to_unique


def _unique_rows_by_bytes(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label rows by first occurrence, keyed on their raw bytes.

    The exact fallback of :func:`_stable_unique_rows`, for a row size that is
    not a multiple of 8 bytes or a fingerprint collision. It holds every row
    as a Python ``bytes`` key, a second full copy of the design.

    Args:
        samples: ``(N, D)`` C-contiguous array with ``N >= 1`` and ``D >= 1``.

    Returns:
        ``(is_first, expanded_to_unique)``: which rows are first occurrences,
        and each row's first-occurrence label.
    """
    n_rows, n_cols = samples.shape
    # Compare whole rows as raw bytes. This keeps the exact-match semantics of
    # the old ``row.tobytes()`` key: two rows merge only when they are bitwise
    # equal. The void view turns each row into one Python ``bytes`` object,
    # and ``tolist()`` builds all N of them in one C loop.
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
    seen: dict[object, int] = {}
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
    return is_first, expanded_to_unique
