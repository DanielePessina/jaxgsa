"""Defines the ``Problem`` dataclass and accepted input specifications."""

from __future__ import annotations

import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NotRequired,
    Self,
    TypeAlias,
    TypedDict,
    cast,
)

import jax.core
import jax.numpy as jnp
import jax.scipy.special as jax_special
import numpy as np
from jax import Array
from scipy import special

if TYPE_CHECKING:
    import numpy.typing as npt

    from jaxgsa._core.copula import RepairPolicy
    from jaxgsa._core.dependence import InputSampler


class UniformInputSpec(TypedDict):
    """Uniform marginal distribution between ``low`` and ``high``.

    Pass to :meth:`Problem.from_dict` as
    ``{"dist": "uniform", "low": 0.0, "high": 1.0}``; a bare ``(low, high)``
    tuple is accepted as shorthand for the same thing.
    """

    dist: Literal["uniform"]
    low: float
    high: float


class GaussianInputSpec(TypedDict):
    """Gaussian (normal) marginal distribution, optionally truncated.

    ``variance`` is the variance, not the standard deviation. Provide ``low``
    and/or ``high`` to truncate the support. Omit both for an unbounded
    Gaussian.
    """

    dist: Literal["gaussian"]
    mean: float
    variance: float
    low: NotRequired[float]
    high: NotRequired[float]


class CategoricalInputSpec(TypedDict):
    """Categorical (unordered discrete) marginal distribution.

    Pass to :meth:`Problem.from_dict` as
    ``{"dist": "categorical", "probs": [0.5, 0.3, 0.2]}``. The parameter
    takes one of ``L = len(probs)`` levels. Samples carry the integer level
    codes ``0 .. L-1`` (as floats), never physical values. ``probs`` must be
    positive and sum to 1 (small rounding error is renormalized). Optional
    ``labels`` (strings or numbers, one per level) are stored on the
    ``Problem`` for reporting only; read them back from
    :attr:`Problem.categorical_labels`.
    """

    dist: Literal["categorical"]
    probs: list[float]
    labels: NotRequired[list[str | int | float]]


class LognormalInputSpec(TypedDict):
    """Lognormal marginal distribution: ``log(X)`` is normal.

    Pass to :meth:`Problem.from_dict` as
    ``{"dist": "lognormal", "mu": 0.0, "sigma": 0.5}``. ``mu`` and ``sigma``
    are the mean and standard deviation of the underlying normal, not of
    ``X``.
    """

    dist: Literal["lognormal"]
    mu: float
    sigma: float


class WeibullInputSpec(TypedDict):
    """Weibull marginal distribution on ``[0, inf)``.

    Pass to :meth:`Problem.from_dict` as
    ``{"dist": "weibull", "shape": 2.0, "scale": 1.0}``.
    """

    dist: Literal["weibull"]
    shape: float
    scale: float


class TriangularInputSpec(TypedDict):
    """Triangular marginal distribution on ``[low, high]`` with peak at ``mode``.

    Pass to :meth:`Problem.from_dict` as
    ``{"dist": "triangular", "low": 0.0, "mode": 0.3, "high": 1.0}``.
    """

    dist: Literal["triangular"]
    low: float
    mode: float
    high: float


class BetaInputSpec(TypedDict):
    """Beta marginal distribution, rescaled to ``[low, high]``.

    Pass to :meth:`Problem.from_dict` as
    ``{"dist": "beta", "a": 2.0, "b": 5.0, "low": 0.0, "high": 1.0}``.
    ``low`` and ``high`` default to 0 and 1, as in :class:`BetaSpec`.
    """

    dist: Literal["beta"]
    a: float
    b: float
    low: NotRequired[float]
    high: NotRequired[float]


# Hashable canonical storage for a validated latent correlation matrix.
_CorrelationTuple: TypeAlias = tuple[tuple[float, ...], ...]

# Renormalize probs whose sum is off by at most this much; raise beyond it.
_PROB_SUM_TOL = 1e-3

# Accepted keys of a dict input spec, per declared ``"dist"``. Anything else
# in the dict is a typo or a distribution mix-up, and silently ignoring it
# would drop a parameter the user believes they declared. The continuous
# families add their own keys, from their ``fields``, below their classes.
_INPUT_SPEC_KEYS: dict[str, frozenset[str]] = {
    "uniform": frozenset({"dist", "low", "high"}),
    "gaussian": frozenset({"dist", "mean", "variance", "low", "high"}),
    "categorical": frozenset({"dist", "probs", "labels"}),
}

# A correlation entry that touches a categorical parameter counts as a real
# coupling only above this magnitude. The tolerance absorbs float noise (a
# declared coupling below 1e-8 is physically meaningless), while the
# positive-definiteness repair itself is checked pre-repair and its output
# re-zeroed, so repair noise never reaches this comparison.
_CATEGORICAL_COUPLING_TOL = 1e-8


@dataclass(frozen=True)
class UniformSpec:
    """Uniform marginal distribution on ``[low, high]``.

    This is the marginal every direct ``Problem(names, bounds)`` parameter
    gets. Both bounds are finite and ``low`` is strictly below ``high``.

    Attributes:
        low: Lower bound of the support.
        high: Upper bound of the support, strictly above ``low``.
    """

    low: float
    high: float

    #: Distribution tag, matching the ``"dist"`` key of the dict input form.
    dist: ClassVar[Literal["uniform"]] = "uniform"
    #: Family name as error messages print it.
    display_name: ClassVar[str] = "uniform"

    def __post_init__(self) -> None:
        """Coerce the bounds to Python floats and check their order.

        Raises:
            ValueError: If either bound is not finite, or ``low`` is not
                strictly below ``high``.
        """
        # Coerce to Python float to keep JAX tracers and numpy scalars out of
        # this hashable jit-cache metadata.
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "high", float(self.high))
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError(
                f"Uniform input requires finite bounds, got {(self.low, self.high)!r}"
            )
        # `not <` instead of `>=` to also catch NaN (NaN comparisons are always False).
        if not self.low < self.high:
            raise ValueError(f"Uniform input requires low < high, got {(self.low, self.high)!r}")

    def support(self) -> tuple[float, float]:
        """Return the support ``(low, high)``."""
        return (self.low, self.high)


@dataclass(frozen=True)
class GaussianSpec:
    """Gaussian (normal) marginal distribution, optionally truncated.

    ``low`` and ``high`` are each optional, so the support has four states:

    * both ``None``: the marginal is the unbounded Gaussian.
    * ``low`` set only: the marginal is truncated on the left, and the right
      tail runs to ``+inf``.
    * ``high`` set only: the marginal is truncated on the right, and the left
      tail runs to ``-inf``.
    * both set: the marginal is truncated on both sides, and its support is
      the finite interval ``[low, high]``.

    All four states are supported. ``Problem.from_dict(...,
    truncate_gaussians=q)`` turns the first three into the fourth by filling
    each open side with that marginal's own ``q`` quantile.

    Attributes:
        mean: Mean of the underlying (untruncated) Gaussian. Must be finite.
        variance: Variance of the underlying Gaussian, strictly positive and
            finite. This is the variance, not the standard deviation.
        low: Lower truncation bound, or ``None`` for an open left tail. Must
            be finite when given.
        high: Upper truncation bound, or ``None`` for an open right tail.
            Must be finite when given.
    """

    mean: float
    variance: float
    low: float | None = None
    high: float | None = None

    #: Distribution tag, matching the ``"dist"`` key of the dict input form.
    dist: ClassVar[Literal["gaussian"]] = "gaussian"
    #: Family name as error messages print it.
    display_name: ClassVar[str] = "Gaussian"

    def __post_init__(self) -> None:
        """Coerce the parameters to Python floats and validate them.

        Raises:
            ValueError: If ``mean`` is not finite, if ``variance`` is not
                finite and positive, if a given bound is not finite, or if
                both bounds are given and ``low`` is not strictly below
                ``high``.
        """
        # Coerce to Python float to prevent JAX tracers or numpy scalars from
        # leaking into metadata.
        object.__setattr__(self, "mean", float(self.mean))
        object.__setattr__(self, "variance", float(self.variance))
        object.__setattr__(self, "low", None if self.low is None else float(self.low))
        object.__setattr__(self, "high", None if self.high is None else float(self.high))

        if not math.isfinite(self.mean):
            raise ValueError(f"Gaussian input requires a finite mean, got {self.mean!r}")
        if not math.isfinite(self.variance) or self.variance <= 0:
            raise ValueError(
                f"Gaussian input requires variance > 0 and finite, got {self.variance!r}"
            )
        if self.low is not None and not math.isfinite(self.low):
            raise ValueError(f"Gaussian input low bound must be finite, got {self.low!r}")
        if self.high is not None and not math.isfinite(self.high):
            raise ValueError(f"Gaussian input high bound must be finite, got {self.high!r}")
        if self.low is not None and self.high is not None and not self.low < self.high:
            raise ValueError(
                f"Truncated Gaussian input requires low < high, got {(self.low, self.high)!r}"
            )

    def support(self) -> tuple[float, float]:
        """Return the support, with ``-inf`` or ``inf`` on an open side."""
        return (
            -math.inf if self.low is None else self.low,
            math.inf if self.high is None else self.high,
        )


@dataclass(frozen=True)
class CategoricalSpec:
    """Categorical (unordered discrete) marginal distribution.

    The parameter takes one of ``L = len(probs)`` levels. Samples carry the
    integer level codes ``0 .. L-1`` (as floats), never physical values. Code
    ``i`` maps to ``labels[i]``.

    ``probs`` and ``labels`` are stored as tuples, so the spec stays immutable
    and hashable and a ``Problem`` remains usable as jit-cache metadata. The
    constructor accepts any sequence and converts it.

    Attributes:
        probs: Level probabilities, one per level, at least two. Each must be
            positive and finite, and they must sum to 1. A sum within ``1e-3``
            of 1 is renormalized; anything further off raises.
        labels: Level labels, one per level, in level-code order. Defaults to
            ``("0", ..., "L-1")``, which is what the empty default requests.
            Labels are reporting metadata only: jaxgsa never relabels sample
            arrays with them.
    """

    probs: tuple[float, ...]
    labels: tuple[str, ...] = ()

    #: Distribution tag, matching the ``"dist"`` key of the dict input form.
    dist: ClassVar[Literal["categorical"]] = "categorical"
    #: Family name as error messages print it.
    display_name: ClassVar[str] = "categorical"

    def __post_init__(self) -> None:
        """Normalize the probabilities and labels into validated tuples.

        Raises:
            ValueError: If fewer than two levels are given, any probability is
                not positive and finite, the probabilities do not sum to 1
                within tolerance, or ``labels`` has the wrong length or
                duplicate entries.
            TypeError: If ``labels`` is a bare string.
        """
        # Coerce to Python floats to prevent JAX tracers or numpy scalars from
        # leaking into hashable metadata.
        prob_values = tuple(float(p) for p in np.asarray(self.probs, dtype=np.float64).ravel())
        n_levels = len(prob_values)
        if n_levels < 2:
            raise ValueError(f"Categorical input requires at least 2 levels, got {n_levels}")
        if not all(math.isfinite(p) and p > 0 for p in prob_values):
            raise ValueError(f"Categorical probs must be positive and finite, got {prob_values!r}")
        total = sum(prob_values)
        if abs(total - 1.0) > _PROB_SUM_TOL:
            raise ValueError(
                f"Categorical probs must sum to 1 (within {_PROB_SUM_TOL}), got sum={total!r}"
            )
        object.__setattr__(self, "probs", tuple(p / total for p in prob_values))

        if not self.labels:
            object.__setattr__(self, "labels", tuple(str(level) for level in range(n_levels)))
            return
        _reject_bare_str(self.labels, "Categorical labels")
        label_values = tuple(str(label) for label in self.labels)
        if len(label_values) != n_levels:
            raise ValueError(
                f"Categorical labels length {len(label_values)} does not match "
                f"the {n_levels} levels declared by probs"
            )
        if len(set(label_values)) != n_levels:
            raise ValueError(f"Categorical labels must be unique, got {label_values!r}")
        object.__setattr__(self, "labels", label_values)

    def support(self) -> tuple[float, float]:
        """Return the range of the level codes, ``(0, L - 1)``."""
        return (0.0, float(len(self.probs) - 1))


def _finite_float(value: float, family: str, field_name: str) -> float:
    """Coerce a spec parameter to a finite Python float.

    Args:
        value: The parameter as given.
        family: Family name for the message, such as ``"Lognormal"``.
        field_name: Parameter name for the message.

    Returns:
        ``value`` as a Python float.

    Raises:
        ValueError: If the value is not finite.
    """
    # Coerce to Python float to keep JAX tracers and numpy scalars out of
    # hashable jit-cache metadata.
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{family} input requires a finite {field_name}, got {result!r}")
    return result


def _positive_float(value: float, family: str, field_name: str) -> float:
    """Coerce a spec parameter to a finite, strictly positive Python float.

    Args:
        value: The parameter as given.
        family: Family name for the message, such as ``"Weibull"``.
        field_name: Parameter name for the message.

    Returns:
        ``value`` as a Python float.

    Raises:
        ValueError: If the value is not finite or not above 0.
    """
    result = _finite_float(value, family, field_name)
    if not result > 0:
        raise ValueError(f"{family} input requires {field_name} > 0, got {result!r}")
    return result


def _triangular_c(low: Any, mode: Any, high: Any) -> Any:
    """Return the relative mode position ``(mode - low) / (high - low)``."""
    return (mode - low) / (high - low)


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


def _bounded_jax_cdf(z: Array, unit_cdf: Array) -> Array:
    """Keep an out-of-support signal for a family on ``[low, high]``.

    Args:
        z: Values rescaled onto the unit support, ``(x - low) / (high - low)``.
        unit_cdf: The CDF evaluated at ``z`` clipped into ``[0, 1]``.

    Returns:
        ``unit_cdf`` where ``z`` is inside ``[0, 1]``, and ``z`` itself (which
        lies outside ``[0, 1]``) everywhere else.
    """
    inside = (z >= 0) & (z <= 1)
    return jnp.where(inside, unit_cdf, z)


class _ContinuousFamily(ABC):
    """Behaviour the lognormal, Weibull, triangular and beta specs share.

    Each family owns its own maths, so callers dispatch on the spec and never
    on its type:

    * :meth:`ppf` and :meth:`cdf`: the host inverse CDF and CDF in float64
      NumPy/SciPy. They build every stored design and invert it.
    * :meth:`jax_cdf`: a JAX CDF with broadcastable per-column parameters. The
      given-data methods and PCE use it to map inputs onto ``[0, 1]``.
    * :meth:`jax_ppf`: a JAX inverse CDF, differentiable in the distribution
      parameters. :meth:`jaxgsa.sobol.SobolSamples.transform` uses it.

    Callers clip the unit coordinate before an inverse CDF, as they do for the
    Gaussian family. The SciPy parameter mapping is in each class docstring.
    """

    #: Distribution tag, matching the ``"dist"`` key of the dict input form.
    dist: ClassVar[str]
    #: Family name as error messages print it, such as ``"Weibull"``.
    display_name: ClassVar[str]
    #: Parameter names, in dict-form order and the order :meth:`jax_cdf` takes.
    fields: ClassVar[tuple[str, ...]]
    #: The fields :meth:`jax_ppf` can differentiate, and so ``theta`` may set.
    theta_fields: ClassVar[frozenset[str]]

    def params(self) -> tuple[float, ...]:
        """Return the parameters in :attr:`fields` order, as Python floats."""
        return tuple(float(getattr(self, name)) for name in self.fields)

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> Self:
        """Build the spec from its dict form.

        Args:
            spec: The dict form, with a ``"dist"`` key and the family's
                fields. A field with a default may be left out.

        Returns:
            The validated spec.

        Raises:
            KeyError: Naming the first required field that ``spec`` lacks.
            ValueError: If a parameter fails the family's own validation.
        """
        required = [
            f.name
            for f in dataclasses.fields(cast(Any, cls))
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        ]
        for name in required:
            if name not in spec:
                raise KeyError(name)
        return cls(**{name: spec[name] for name in cls.fields if name in spec})

    @abstractmethod
    def support(self) -> tuple[float, float]:
        """Return the support ``(low, high)``, with ``math.inf`` for an open side."""

    @abstractmethod
    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Map clipped unit-interval values through the inverse CDF, in float64.

        Args:
            u: Unit-interval values, already clipped off 0 and 1.

        Returns:
            Physical values, same shape as ``u``, dtype float64.
        """

    @abstractmethod
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Map physical values through the CDF, in float64.

        Values outside the support map to 0 or 1.

        Args:
            x: Physical values.

        Returns:
            CDF values in ``[0, 1]``, same shape as ``x``, dtype float64.
        """

    @staticmethod
    @abstractmethod
    def jax_cdf(x: Array, params: tuple[Any, ...]) -> Array:
        """Map physical values through the CDF on device, keeping an out-of-support signal.

        A value inside the support maps into ``[0, 1]``. A value outside it
        maps outside ``[0, 1]``, as a uniform column does, so callers such as
        PAWN can drop it instead of folding it into an edge bin. The caller
        clips the in-range values off 0 and 1.

        Args:
            x: Physical values, any shape.
            params: Parameters in :attr:`fields` order, each a scalar or an
                array that broadcasts against ``x`` (one value per column).

        Returns:
            Array of the same shape as ``x``.
        """

    @abstractmethod
    def jax_ppf(
        self, u: Array, fields: Mapping[str, Any], unit_quantiles: np.ndarray | None = None
    ) -> Array:
        """Map clipped unit-interval values through the inverse CDF, differentiably.

        Args:
            u: Unit-interval values, already clipped off 0 and 1.
            fields: Parameter overrides by field name. Values may be tracers.
                A field it does not name keeps the spec's own value.
            unit_quantiles: Host-computed standard quantiles of ``u``. Only
                :class:`BetaSpec` reads them; see :meth:`BetaSpec.jax_ppf`.

        Returns:
            Physical values, same shape as ``u``.
        """


@dataclass(frozen=True)
class LognormalSpec(_ContinuousFamily):
    """Lognormal marginal distribution: ``log(X) ~ N(mu, sigma**2)``.

    This is SciPy's ``lognorm(s=sigma, scale=exp(mu))``. The support is
    ``(0, inf)``. Sampling clips the unit coordinate to
    ``[UNIT_CLIP, 1 - UNIT_CLIP]``, so draws stay finite. In float32 the
    largest draw overflows when ``mu + 5.3 * sigma`` exceeds about 88.

    Attributes:
        mu: Mean of the underlying normal (of ``log X``). Must be finite.
        sigma: Standard deviation of the underlying normal, strictly positive
            and finite.
    """

    mu: float
    sigma: float

    #: Distribution tag, matching the ``"dist"`` key of the dict input form.
    dist: ClassVar[Literal["lognormal"]] = "lognormal"
    #: Family name as error messages print it.
    display_name: ClassVar[str] = "lognormal"
    fields: ClassVar[tuple[str, ...]] = ("mu", "sigma")
    theta_fields: ClassVar[frozenset[str]] = frozenset(fields)

    def __post_init__(self) -> None:
        """Coerce the parameters to Python floats and validate them.

        Raises:
            ValueError: If ``mu`` is not finite or ``sigma`` is not finite
                and positive.
        """
        object.__setattr__(self, "mu", _finite_float(self.mu, "Lognormal", "mu"))
        object.__setattr__(self, "sigma", _positive_float(self.sigma, "Lognormal", "sigma"))

    def support(self) -> tuple[float, float]:
        """Return the support ``[0, inf)``."""
        return (0.0, math.inf)

    def to_dict(self) -> LognormalInputSpec:
        """Return the JSON-friendly dict form."""
        return LognormalInputSpec(dist="lognormal", mu=self.mu, sigma=self.sigma)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF in float64. See :meth:`_ContinuousFamily.ppf`."""
        u = np.asarray(u, dtype=np.float64)
        return np.exp(self.mu + self.sigma * special.ndtri(u))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """CDF in float64. See :meth:`_ContinuousFamily.cdf`."""
        x = np.asarray(x, dtype=np.float64)
        positive = x > 0
        safe = np.where(positive, x, 1.0)
        return np.where(positive, special.ndtr((np.log(safe) - self.mu) / self.sigma), 0.0)

    @staticmethod
    def jax_cdf(x: Array, params: tuple[Any, ...]) -> Array:
        """JAX CDF. See :meth:`_ContinuousFamily.jax_cdf`."""
        mu, sigma = params
        positive = x > 0
        safe = jnp.where(positive, x, 1.0)
        u = jax_special.ndtr((jnp.log(safe) - mu) / sigma)
        # x == 0 is the closed end of the support (CDF 0); x < 0 is outside.
        return jnp.where(positive, u, jnp.where(x == 0, 0.0, x))

    def jax_ppf(
        self, u: Array, fields: Mapping[str, Any], unit_quantiles: np.ndarray | None = None
    ) -> Array:
        """Closed-form inverse CDF, differentiable in ``mu`` and ``sigma``."""
        mu = fields.get("mu", self.mu)
        sigma = fields.get("sigma", self.sigma)
        return jnp.exp(mu + sigma * jax_special.ndtri(u))


@dataclass(frozen=True)
class WeibullSpec(_ContinuousFamily):
    """Weibull marginal distribution with CDF ``1 - exp(-(x / scale)**shape)``.

    This is SciPy's ``weibull_min(c=shape, scale=scale)``. The support is
    ``[0, inf)``.

    Attributes:
        shape: Shape parameter ``k``, strictly positive and finite.
        scale: Scale parameter ``lambda``, strictly positive and finite.
    """

    shape: float
    scale: float

    #: Distribution tag, matching the ``"dist"`` key of the dict input form.
    dist: ClassVar[Literal["weibull"]] = "weibull"
    #: Family name as error messages print it.
    display_name: ClassVar[str] = "Weibull"
    fields: ClassVar[tuple[str, ...]] = ("shape", "scale")
    theta_fields: ClassVar[frozenset[str]] = frozenset(fields)

    def __post_init__(self) -> None:
        """Coerce the parameters to Python floats and validate them.

        Raises:
            ValueError: If ``shape`` or ``scale`` is not finite and positive.
        """
        object.__setattr__(self, "shape", _positive_float(self.shape, "Weibull", "shape"))
        object.__setattr__(self, "scale", _positive_float(self.scale, "Weibull", "scale"))

    def support(self) -> tuple[float, float]:
        """Return the support ``[0, inf)``."""
        return (0.0, math.inf)

    def to_dict(self) -> WeibullInputSpec:
        """Return the JSON-friendly dict form."""
        return WeibullInputSpec(dist="weibull", shape=self.shape, scale=self.scale)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF in float64. See :meth:`_ContinuousFamily.ppf`."""
        u = np.asarray(u, dtype=np.float64)
        # log1p, not log(1 - u): near u = 0 the plain form loses every digit.
        return self.scale * (-np.log1p(-u)) ** (1.0 / self.shape)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """CDF in float64. See :meth:`_ContinuousFamily.cdf`."""
        x = np.asarray(x, dtype=np.float64)
        t = np.maximum(x, 0.0) / self.scale
        return -np.expm1(-(t**self.shape))

    @staticmethod
    def jax_cdf(x: Array, params: tuple[Any, ...]) -> Array:
        """JAX CDF. See :meth:`_ContinuousFamily.jax_cdf`."""
        shape, scale = params
        inside = x >= 0
        t = jnp.where(inside, x, 0.0) / scale
        u = -jnp.expm1(-(t**shape))
        return jnp.where(inside, u, x)

    def jax_ppf(
        self, u: Array, fields: Mapping[str, Any], unit_quantiles: np.ndarray | None = None
    ) -> Array:
        """Closed-form inverse CDF, differentiable in ``shape`` and ``scale``."""
        shape = fields.get("shape", self.shape)
        scale = fields.get("scale", self.scale)
        return scale * (-jnp.log1p(-u)) ** (1.0 / shape)


@dataclass(frozen=True)
class TriangularSpec(_ContinuousFamily):
    """Triangular marginal distribution on ``[low, high]`` with peak at ``mode``.

    This is SciPy's ``triang(c=(mode - low) / (high - low), loc=low,
    scale=high - low)``. ``mode`` may equal ``low`` or ``high``, as SciPy
    allows; the density is then a right-angled triangle.

    Attributes:
        low: Lower end of the support. Must be finite.
        mode: Peak of the density, with ``low <= mode <= high``.
        high: Upper end of the support, strictly above ``low``.
    """

    low: float
    mode: float
    high: float

    #: Distribution tag, matching the ``"dist"`` key of the dict input form.
    dist: ClassVar[Literal["triangular"]] = "triangular"
    #: Family name as error messages print it.
    display_name: ClassVar[str] = "triangular"
    fields: ClassVar[tuple[str, ...]] = ("low", "mode", "high")
    theta_fields: ClassVar[frozenset[str]] = frozenset(fields)

    def __post_init__(self) -> None:
        """Coerce the parameters to Python floats and validate them.

        Raises:
            ValueError: If any parameter is not finite, ``low`` is not
                strictly below ``high``, or ``mode`` is outside
                ``[low, high]``.
        """
        object.__setattr__(self, "low", _finite_float(self.low, "Triangular", "low"))
        object.__setattr__(self, "mode", _finite_float(self.mode, "Triangular", "mode"))
        object.__setattr__(self, "high", _finite_float(self.high, "Triangular", "high"))
        if not self.low < self.high:
            raise ValueError(
                f"Triangular input requires low < high, got {(self.low, self.high)!r}"
            )
        if not self.low <= self.mode <= self.high:
            raise ValueError(
                "Triangular input requires low <= mode <= high, got "
                f"{(self.low, self.mode, self.high)!r}"
            )

    def support(self) -> tuple[float, float]:
        """Return the support ``(low, high)``."""
        return (self.low, self.high)

    def to_dict(self) -> TriangularInputSpec:
        """Return the JSON-friendly dict form."""
        return TriangularInputSpec(dist="triangular", low=self.low, mode=self.mode, high=self.high)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF in float64. See :meth:`_ContinuousFamily.ppf`."""
        u = np.asarray(u, dtype=np.float64)
        width = self.high - self.low
        c = _triangular_c(self.low, self.mode, self.high)
        left = self.low + np.sqrt(u * width * (self.mode - self.low))
        right = self.high - np.sqrt((1.0 - u) * width * (self.high - self.mode))
        return np.where(u < c, left, right)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """CDF in float64. See :meth:`_ContinuousFamily.cdf`."""
        x = np.asarray(x, dtype=np.float64)
        z = np.clip((x - self.low) / (self.high - self.low), 0.0, 1.0)
        c = _triangular_c(self.low, self.mode, self.high)
        return np.asarray(_triangular_cdf_unit(np, z, c))

    @staticmethod
    def jax_cdf(x: Array, params: tuple[Any, ...]) -> Array:
        """JAX CDF. See :meth:`_ContinuousFamily.jax_cdf`."""
        low, mode, high = params
        z = (x - low) / (high - low)
        zc = jnp.clip(z, 0.0, 1.0)
        return _bounded_jax_cdf(z, _triangular_cdf_unit(jnp, zc, _triangular_c(low, mode, high)))

    def jax_ppf(
        self, u: Array, fields: Mapping[str, Any], unit_quantiles: np.ndarray | None = None
    ) -> Array:
        """Closed-form inverse CDF, differentiable in ``low``, ``mode`` and ``high``."""
        low = fields.get("low", self.low)
        mode = fields.get("mode", self.mode)
        high = fields.get("high", self.high)
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


@dataclass(frozen=True)
class BetaSpec(_ContinuousFamily):
    """Beta marginal distribution, rescaled to ``[low, high]``.

    This is SciPy's ``beta(a, b, loc=low, scale=high - low)``. The defaults
    give the standard beta on ``[0, 1]``.

    Attributes:
        a: First shape parameter, strictly positive and finite.
        b: Second shape parameter, strictly positive and finite.
        low: Lower end of the support. Must be finite.
        high: Upper end of the support, strictly above ``low``.
    """

    a: float
    b: float
    low: float = 0.0
    high: float = 1.0

    #: Distribution tag, matching the ``"dist"`` key of the dict input form.
    dist: ClassVar[Literal["beta"]] = "beta"
    #: Family name as error messages print it.
    display_name: ClassVar[str] = "beta"
    fields: ClassVar[tuple[str, ...]] = ("a", "b", "low", "high")
    # The shape parameters a and b are fixed: JAX has no betaincinv, and
    # jax.scipy.special.betainc has no gradient in a or b. The inverse CDF is
    # therefore computed on the host and only the affine map onto
    # [low, high] is traced.
    theta_fields: ClassVar[frozenset[str]] = frozenset({"low", "high"})

    def __post_init__(self) -> None:
        """Coerce the parameters to Python floats and validate them.

        Raises:
            ValueError: If ``a`` or ``b`` is not finite and positive, a bound
                is not finite, or ``low`` is not strictly below ``high``.
        """
        object.__setattr__(self, "a", _positive_float(self.a, "Beta", "a"))
        object.__setattr__(self, "b", _positive_float(self.b, "Beta", "b"))
        object.__setattr__(self, "low", _finite_float(self.low, "Beta", "low"))
        object.__setattr__(self, "high", _finite_float(self.high, "Beta", "high"))
        if not self.low < self.high:
            raise ValueError(f"Beta input requires low < high, got {(self.low, self.high)!r}")

    def support(self) -> tuple[float, float]:
        """Return the support ``(low, high)``."""
        return (self.low, self.high)

    def to_dict(self) -> BetaInputSpec:
        """Return the JSON-friendly dict form."""
        return BetaInputSpec(dist="beta", a=self.a, b=self.b, low=self.low, high=self.high)

    def unit_quantiles(self, u: np.ndarray) -> np.ndarray:
        """Return the standard-beta quantiles ``betaincinv(a, b, u)`` in float64.

        Args:
            u: Unit-interval values, already clipped off 0 and 1.

        Returns:
            Quantiles on ``[0, 1]``, same shape as ``u``.
        """
        return special.betaincinv(self.a, self.b, np.asarray(u, dtype=np.float64))

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF in float64. See :meth:`_ContinuousFamily.ppf`."""
        return self.low + (self.high - self.low) * self.unit_quantiles(u)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """CDF in float64. See :meth:`_ContinuousFamily.cdf`."""
        x = np.asarray(x, dtype=np.float64)
        z = np.clip((x - self.low) / (self.high - self.low), 0.0, 1.0)
        return special.betainc(self.a, self.b, z)

    @staticmethod
    def jax_cdf(x: Array, params: tuple[Any, ...]) -> Array:
        """JAX CDF. See :meth:`_ContinuousFamily.jax_cdf`."""
        a, b, low, high = params
        z = (x - low) / (high - low)
        zc = jnp.clip(z, 0.0, 1.0)
        u = jax_special.betainc(jnp.asarray(a, zc.dtype), jnp.asarray(b, zc.dtype), zc)
        return _bounded_jax_cdf(z, u)

    def jax_ppf(
        self, u: Array, fields: Mapping[str, Any], unit_quantiles: np.ndarray | None = None
    ) -> Array:
        """Inverse CDF, differentiable in ``low`` and ``high`` only.

        The beta inverse CDF has no JAX implementation, so the standard
        quantiles ``z = betaincinv(a, b, u)`` come from the host. The result
        ``low + (high - low) * z`` is then differentiable in ``low`` and
        ``high``. Pass ``unit_quantiles`` (from :meth:`unit_quantiles`) to
        reuse quantiles computed once per design. Without it they are computed
        here, which needs a concrete ``u``.

        Raises:
            ValueError: If ``unit_quantiles`` is missing and ``u`` is traced.
        """
        if unit_quantiles is None:
            if isinstance(u, jax.core.Tracer):
                raise ValueError(
                    "a beta inverse CDF needs its standard quantiles computed on the "
                    "host, so the unit values must be a concrete array, not a traced one"
                )
            unit_quantiles = self.unit_quantiles(np.asarray(u))
        low = fields.get("low", self.low)
        high = fields.get("high", self.high)
        return low + (high - low) * jnp.asarray(unit_quantiles, dtype=u.dtype)


# Canonical stored form of one parameter's marginal. This is what
# `Problem.input_specs` returns, and what every internal consumer reads.
InputSpec: TypeAlias = (
    UniformSpec
    | GaussianSpec
    | CategoricalSpec
    | LognormalSpec
    | WeibullSpec
    | TriangularSpec
    | BetaSpec
)

# The four continuous families added in 1.3. Each owns its maths (see
# ``_ContinuousFamily``), so the samplers and transforms share one branch for
# all four.
ContinuousFamilySpec: TypeAlias = LognormalSpec | WeibullSpec | TriangularSpec | BetaSpec
_CONTINUOUS_FAMILY_SPECS = (LognormalSpec, WeibullSpec, TriangularSpec, BetaSpec)
_INPUT_SPEC_KEYS.update(
    {cls.dist: frozenset({"dist", *cls.fields}) for cls in _CONTINUOUS_FAMILY_SPECS}
)

# Spec class of each continuous family, by its ``dist`` tag. The one source of
# the family's display name and class name in error messages.
_FAMILY_BY_DIST: dict[str, type[ContinuousFamilySpec]] = {
    cls.dist: cls for cls in _CONTINUOUS_FAMILY_SPECS
}

# Public-facing input union -- users pass one of these per parameter.
InputSpecValue: TypeAlias = (
    tuple[float, float]
    | list[float]
    | UniformInputSpec
    | GaussianInputSpec
    | CategoricalInputSpec
    | LognormalInputSpec
    | WeibullInputSpec
    | TriangularInputSpec
    | BetaInputSpec
    | UniformSpec
    | GaussianSpec
    | CategoricalSpec
    | LognormalSpec
    | WeibullSpec
    | TriangularSpec
    | BetaSpec
)


def _normalize_input_spec(spec: InputSpecValue) -> InputSpec:
    """Normalize any accepted user input form into a canonical spec dataclass.

    Raises:
        ValueError: If ``spec`` is not a tuple, a dict with a ``"dist"`` key,
            or one of the spec dataclasses, if a dict with a known
            ``"dist"`` is missing one of that distribution's required keys,
            or if it carries keys outside that distribution's declared set.
    """
    if isinstance(spec, UniformSpec | GaussianSpec | CategoricalSpec | ContinuousFamilySpec):
        return spec  # already canonical and validated by its own __post_init__
    if isinstance(spec, tuple | list):  # bare (low, high) pair is shorthand for uniform
        if len(spec) != 2:
            raise ValueError("Bounds must have exactly two values: (low, high)")
        return UniformSpec(spec[0], spec[1])
    if not isinstance(spec, dict):
        raise ValueError(
            f"Unsupported input spec {spec!r}; pass a (low, high) tuple, a "
            "UniformSpec, GaussianSpec, CategoricalSpec, LognormalSpec, "
            "WeibullSpec, TriangularSpec, or BetaSpec instance, or a dict with "
            f"a 'dist' key (one of {sorted(_INPUT_SPEC_KEYS)})"
        )
    if "dist" not in spec:
        raise ValueError(
            f"Input spec dict {spec!r} has no 'dist' key; give one of "
            f"{sorted(_INPUT_SPEC_KEYS)}, or pass a bare "
            "(low, high) tuple for a uniform marginal"
        )

    allowed = _INPUT_SPEC_KEYS.get(spec["dist"])
    if allowed is not None:
        unknown = sorted(set(spec) - allowed)
        if unknown:
            raise ValueError(
                f"{spec['dist']!r} input spec dict has unknown key(s) {unknown}; "
                f"allowed keys: {sorted(allowed)}"
            )

    try:
        if spec["dist"] == "uniform":
            return UniformSpec(spec["low"], spec["high"])
        if spec["dist"] == "gaussian":
            return GaussianSpec(
                spec["mean"],
                spec["variance"],
                low=spec.get("low"),
                high=spec.get("high"),
            )
        if spec["dist"] == "categorical":
            labels = spec.get("labels")
            _reject_bare_str(labels, "Categorical labels")
            return CategoricalSpec(
                tuple(spec["probs"]),
                labels=() if labels is None else tuple(str(label) for label in labels),
            )
        family = _FAMILY_BY_DIST.get(spec["dist"])
        if family is not None:
            return family.from_dict(spec)
    except KeyError as exc:
        raise ValueError(
            f"{spec['dist']!r} input spec dict {spec!r} is missing {exc.args[0]!r}"
        ) from exc

    raise ValueError(f"Unsupported input distribution {spec['dist']!r}")


def _truncate_gaussian_spec(spec: InputSpec, q: float) -> InputSpec:
    """Fill the open sides of a Gaussian spec with its own ``q`` quantiles.

    Only Gaussian marginals are touched. Uniform and categorical marginals
    are already bounded, and a side the user declared always wins over the
    automatic bound.

    Args:
        spec: Normalized input spec.
        q: Tail probability to cut from each open side.

    Returns:
        The spec with every open Gaussian side bounded at that marginal's own
        ``q`` (low) and ``1 - q`` (high) quantile.
    """
    from scipy.stats import norm

    if not isinstance(spec, GaussianSpec):
        return spec
    if spec.low is not None and spec.high is not None:
        return spec

    std = math.sqrt(spec.variance)
    low = spec.low
    high = spec.high
    if low is None:
        low = float(norm.ppf(q, loc=spec.mean, scale=std))
    if high is None:
        high = float(norm.ppf(1.0 - q, loc=spec.mean, scale=std))
    return GaussianSpec(spec.mean, spec.variance, low=low, high=high)


def _derive_bounds(
    input_specs: tuple[InputSpec, ...],
) -> tuple[tuple[float, float], ...] | None:
    """Return finite bounds for uniform-only problems, otherwise ``None``."""
    # Bounds only meaningful for uniform-only problems; Gaussian inputs have
    # infinite (or truncated) support and categorical inputs carry codes, so
    # neither maps to simple lo/hi bounds.
    bounds: list[tuple[float, float]] = []
    for spec in input_specs:
        if not isinstance(spec, UniformSpec):
            return None
        bounds.append((spec.low, spec.high))
    return tuple(bounds)


def _canonical_correlation(
    correlation: "npt.ArrayLike | None",
    names: tuple[str, ...],
    input_specs: tuple[InputSpec, ...],
    correlation_type: Literal["latent", "spearman"] = "latent",
    *,
    policy: "RepairPolicy",
) -> _CorrelationTuple | None:
    """Validate a correlation matrix and freeze it into hashable form.

    ``None`` passes through and means independent inputs. Any other matrix is
    converted from ``correlation_type`` to the latent scale and validated, which includes
    the positive-definiteness repair. The result is stored as a nested tuple
    of floats, so ``Problem`` stays frozen and hashable. ``policy`` grades how
    loudly the repair reports itself. Every ``Problem`` surface passes
    ``"declared"``.

    Categorical parameters need care around the repair. The
    categorical-coupling check runs on the matrix as the caller declared it,
    before the repair. The repair's eigendecomposition fills decoupled
    categorical rows with float-level noise, and the check must not read that
    noise as a declared coupling. After the repair, the categorical rows and
    columns are reset to exact identity, so the stored matrix carries exact
    zeros. The reset keeps the matrix positive definite: the result is
    block-diagonal, built from an identity block and a principal submatrix of
    the repaired matrix.

    Args:
        correlation: ``(D, D)`` candidate correlation matrix, or ``None`` for
            independent inputs.
        names: Parameter names in model-input order.
        input_specs: Normalized input spec per parameter, in the same order
            as ``names``.
        correlation_type: Scale ``correlation`` is expressed on,
            ``"latent"`` or ``"spearman"``.
        policy: Repair policy passed through to the copula validator.

    Returns:
        The validated latent matrix as a nested tuple of floats, or ``None``
        when ``correlation`` is ``None``.

    Raises:
        ValueError: If ``correlation`` couples a categorical parameter, or if
            it fails the copula validation.
    """
    if correlation_type not in ("latent", "spearman"):
        raise ValueError(
            f"correlation_type must be 'latent' or 'spearman', got {correlation_type!r}"
        )
    if correlation is None:
        return None
    # Imported lazily: jaxgsa._core.copula imports this module at load time.
    from jaxgsa._core.copula import _force_categorical_identity, canonicalize_correlation

    n_params = len(names)
    R_raw = np.asarray(correlation)
    if np.iscomplexobj(R_raw):
        raise TypeError(
            "correlation must be real, got a complex matrix. Remove the imaginary "
            "part before declaring it."
        )
    R_declared = np.asarray(R_raw, dtype=np.float64)
    if R_declared.shape == (n_params, n_params):
        # Check the declared matrix; a wrong shape falls through to the
        # shape error inside canonicalize_correlation. The sign-preserving
        # Spearman conversion cannot turn a zero coupling into a non-zero
        # one, so checking before the conversion is equivalent.
        _check_correlation_touches_categorical(names, input_specs, R_declared)
    R = canonicalize_correlation(
        R_declared,
        n_params,
        correlation_type=correlation_type,
        policy=policy,
        method="jaxgsa.Problem",
    )
    cat_dims = [d for d, _ in _categorical_dims_from_specs(input_specs)]
    R = _force_categorical_identity(R, cat_dims)
    return tuple(tuple(float(value) for value in row) for row in R)


def _normalized_input_to_dict(
    spec: InputSpec,
) -> (
    UniformInputSpec
    | GaussianInputSpec
    | CategoricalInputSpec
    | LognormalInputSpec
    | WeibullInputSpec
    | TriangularInputSpec
    | BetaInputSpec
):
    """Convert a canonical input spec into a JSON-friendly mapping."""
    if isinstance(spec, _CONTINUOUS_FAMILY_SPECS):
        return spec.to_dict()
    if isinstance(spec, UniformSpec):
        return UniformInputSpec(dist="uniform", low=spec.low, high=spec.high)
    if isinstance(spec, CategoricalSpec):
        label_list: list[str | int | float] = list(spec.labels)
        return CategoricalInputSpec(dist="categorical", probs=list(spec.probs), labels=label_list)

    # Only GaussianSpec is left. Reconstruct the TypedDict for JSON
    # serialization (used by SobolSamples.save).
    payload: GaussianInputSpec = GaussianInputSpec(
        dist="gaussian",
        mean=spec.mean,
        variance=spec.variance,
    )
    if spec.low is not None:
        payload["low"] = spec.low
    if spec.high is not None:
        payload["high"] = spec.high
    return payload


def _check_correlation_touches_categorical(
    names: tuple[str, ...],
    input_specs: tuple[InputSpec, ...],
    correlation: "npt.ArrayLike | None",
) -> None:
    """Reject a correlation matrix that couples a categorical parameter.

    The Gaussian copula has no defined coupling for an unordered marginal
    (that needs a polychoric model, which is future work). Identity rows and
    columns are fine: they declare the categorical parameter independent.
    Entries within ``_CATEGORICAL_COUPLING_TOL`` of zero count as zero.

    Only the rows are checked. A stored matrix is symmetric, so its rows
    already cover the columns. A declared matrix whose coupling sits in the
    column only is asymmetric, and it fails the symmetry validation instead.

    Args:
        names: Parameter names in model-input order.
        input_specs: Normalized input spec per parameter, in the same order
            as ``names``.
        correlation: ``(D, D)`` correlation matrix, or ``None`` to skip the
            check.

    Raises:
        ValueError: If any off-diagonal row entry of ``correlation`` above
            the tolerance touches a categorical parameter.
    """
    if correlation is None:
        return
    R = np.asarray(correlation, dtype=np.float64)
    for d, spec in enumerate(input_specs):
        if not isinstance(spec, CategoricalSpec):
            continue
        off_diag = np.delete(R[d], d)
        if np.any(np.abs(off_diag) > _CATEGORICAL_COUPLING_TOL):
            raise ValueError(
                f"problem.correlation couples categorical parameter {names[d]!r}, "
                "but the Gaussian copula does not define a coupling for an "
                "unordered marginal (polychoric coupling is future work). Keep "
                "the categorical parameter's row and column at identity, or "
                "drop the matrix with problem.with_correlation(None)."
            )


def _reject_bare_str(value: object, label: str) -> None:
    """Refuse a bare ``str`` where a sequence of strings is expected.

    ``tuple("temp")`` is ``('t', 'e', 'm', 'p')``, so a single name passed
    without a tuple around it would silently become four names.

    Args:
        value: The user-supplied sequence.
        label: Argument name for the message.

    Raises:
        TypeError: If ``value`` is a ``str`` or ``bytes``.
    """
    if isinstance(value, str | bytes):
        raise TypeError(
            f"{label} must be a sequence of strings, not a single string {value!r}. "
            f"Write ({value!r},) for one name."
        )


def _output_names_tuple(output_names: Sequence[str] | None) -> tuple[str, ...] | None:
    """Return ``output_names`` as a tuple, refusing a bare string.

    Args:
        output_names: User-supplied output labels, or ``None``.

    Returns:
        The labels as a tuple, or ``None``.

    Raises:
        TypeError: If ``output_names`` is a ``str``.
        ValueError: If any label is not a string, or two labels are equal.
    """
    if output_names is None:
        return None
    _reject_bare_str(output_names, "output_names")
    labels = tuple(output_names)
    _validate_names(labels, label="output names")
    return labels


def _validate_names(names: tuple[str, ...], label: str = "parameter names") -> None:
    """Reject parameter names that are not strings or not unique.

    A non-string name breaks every keyed surface downstream — ``Theta``
    mappings, dataset ``param`` coordinates, saved-design metadata — and a
    duplicate makes those keys ambiguous, so both are refused at
    construction, where the fix is one line away.

    Args:
        names: Parameter names in model-input order, or output labels.
        label: What the names are, for the error messages.

    Raises:
        ValueError: If any name is not a ``str`` (convert it with ``str(...)``
            when declaring the problem), or if two entries share a name
            (rename one of them).
    """
    non_str = [name for name in names if not isinstance(name, str)]
    if non_str:
        raise ValueError(
            f"{label} must be strings, got {non_str!r}. Convert each "
            "entry with str(...) when declaring the problem."
        )
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(
            f"{label} must be unique, but {duplicates!r} appear more "
            "than once. Rename the duplicated entries so every name is "
            "distinct."
        )


def _categorical_dims_from_specs(
    input_specs: tuple[InputSpec, ...],
) -> tuple[tuple[int, int], ...]:
    """Return ``(dimension index, level count)`` per categorical spec."""
    return tuple(
        (d, len(spec.probs))
        for d, spec in enumerate(input_specs)
        if isinstance(spec, CategoricalSpec)
    )


Dependence: TypeAlias = Literal["independent", "gaussian", "custom"]
"""How a problem's inputs depend on each other (see :attr:`Problem.dependence`)."""


def _validated_sampler(
    sampler: InputSampler | None,
    dimension: int | None,
    n_params: int,
    has_correlation: bool,
) -> tuple[InputSampler | None, int | None]:
    """Check a custom sampler declaration and resolve its uniform dimension.

    Args:
        sampler: User transformation from uniform points to physical rows.
        dimension: Uniform coordinates per row, or ``None`` for ``n_params``.
        n_params: Number of problem parameters.
        has_correlation: Whether a Gaussian-copula matrix is also declared.

    Returns:
        ``(sampler, dimension)``, both ``None`` when no sampler is declared.

    Raises:
        TypeError: If ``sampler`` is not callable.
        ValueError: If ``dimension`` is given without a sampler or is not a
            positive integer, or if a correlation is also declared.
    """
    if sampler is None:
        if dimension is not None:
            raise ValueError("a sampler dimension requires a sampler")
        return None, None
    if not callable(sampler):
        raise TypeError("sampler must be callable")
    if has_correlation:
        raise ValueError(
            "A problem takes either a Gaussian-copula correlation or a custom "
            "sampler, not both. Drop one with problem.with_correlation(None) or "
            "problem.with_sampler(None)."
        )
    resolved = n_params if dimension is None else dimension
    if isinstance(resolved, bool) or not isinstance(resolved, (int, np.integer)) or resolved < 1:
        raise ValueError(f"sampler_dimension must be a positive integer, got {dimension!r}")
    return sampler, int(resolved)


def _categorical_dims(problem: "Problem") -> tuple[tuple[int, int], ...]:
    """Return ``(dimension index, level count)`` per categorical parameter."""
    return _categorical_dims_from_specs(problem.input_specs)


# frozen for hashability and safety; init=False because we define a custom __init__ for validation.
@dataclass(frozen=True, init=False)
class Problem:
    """Immutable description of a model's uncertain input parameters.

    Every jaxgsa sampling and analysis function takes a ``Problem`` that names
    the input parameters (in the order the model expects them) and gives each
    one a marginal distribution. Instances are frozen, so one ``Problem`` can
    safely be shared across analyses.

    The direct constructor accepts uniform marginals only, given as finite
    ``(low, high)`` bounds. Use :meth:`from_dict` when you need mixed
    uniform, Gaussian, and categorical marginals.

    A Gaussian-copula ``correlation`` matrix optionally couples the
    marginals. Samples then follow the declared dependence structure, and
    each parameter still keeps its marginal exactly as written.
    ``jaxgsa.sampling.monte_carlo`` honors the matrix transparently.

    A custom ``sampler`` replaces the Gaussian copula with any other joint
    distribution (see :class:`jaxgsa.sampling.InputSampler`). A problem
    declares one or the other, never both. Methods whose indices assume
    independent inputs refuse a dependent problem with a ``ValueError``.

    Attributes:
        names: Parameter names in model-input order.
        bounds: Per-parameter ``(low, high)`` tuples when every marginal is
            uniform, or ``None`` as soon as one is not. A Gaussian support has
            no meaningful finite bounds, and a categorical marginal carries
            level codes rather than a range.
        output_names: Optional labels for the model's outputs, used to name
            the ``output`` coordinate in ``to_dataset()`` exports.
    """

    names: tuple[str, ...]
    _input_specs: tuple[InputSpec, ...] = field(repr=False)
    output_names: tuple[str, ...] | None = None
    _correlation: _CorrelationTuple | None = field(repr=False, default=None)
    _sampler: InputSampler | None = field(repr=False, default=None)
    _sampler_dimension: int | None = field(repr=False, default=None)

    def __init__(
        self,
        names: Sequence[str],
        bounds: Sequence[Sequence[float]],
        *,
        output_names: Sequence[str] | None = None,
        correlation: "npt.ArrayLike | None" = None,
        correlation_type: Literal["latent", "spearman"] = "latent",
        sampler: InputSampler | None = None,
        sampler_dimension: int | None = None,
    ) -> None:
        """Create a uniform-only problem from finite bounds.

        Args:
            names: Parameter names in model-input order.
            bounds: One ``(low, high)`` pair per parameter, in the same order
                as ``names``. Each parameter is sampled uniformly between its
                bounds.
            output_names: Optional output labels used by ``to_dataset()``.
            correlation: Optional ``(D, D)`` Gaussian-copula correlation
                matrix declaring the dependence between parameters. ``None``
                (default) means independent inputs. Validated on entry. A
                slightly non-positive-definite matrix is repaired and reported
                with a ``JaxgsaWarning``. A matrix whose repair would have to
                move an entry by 0.05 or more is rejected with a
                ``ValueError``.
            correlation_type: Scale ``correlation`` is expressed on:
                ``"latent"`` (default) for the Pearson correlation of the
                copula's latent normals, ``"spearman"`` for a rank
                correlation (converted via ``2 sin(pi rho_s / 6)``).
            sampler: Optional custom joint and conditional transformation
                from uniform points to physical rows (see
                :class:`jaxgsa.sampling.InputSampler`). Cannot be combined
                with ``correlation``.
            sampler_dimension: Uniform coordinates the sampler consumes per
                row. ``None`` (default) means one per parameter.
        """
        _reject_bare_str(names, "names")
        normalized_names = tuple(names)
        normalized_bounds = tuple((float(low), float(high)) for low, high in bounds)
        if len(normalized_names) != len(normalized_bounds):
            raise ValueError(
                "names and bounds must have the same length, got "
                f"{len(normalized_names)} and {len(normalized_bounds)}"
            )

        input_specs = tuple(UniformSpec(low, high) for low, high in normalized_bounds)
        self._set_fields(
            names=normalized_names,
            input_specs=input_specs,
            output_names=_output_names_tuple(output_names),
            # User-declared matrices deserve the repair warning.
            correlation=_canonical_correlation(
                correlation,
                normalized_names,
                input_specs,
                correlation_type,
                policy="declared",
            ),
            sampler=sampler,
            sampler_dimension=sampler_dimension,
        )

    @classmethod
    def from_dict(
        cls,
        params: dict[str, InputSpecValue],
        *,
        output_names: Sequence[str] | None = None,
        truncate_gaussians: float | None = None,
        correlation: "npt.ArrayLike | None" = None,
        correlation_type: Literal["latent", "spearman"] = "latent",
        sampler: InputSampler | None = None,
        sampler_dimension: int | None = None,
    ) -> "Problem":
        """Create a ``Problem`` from per-parameter distribution specs.

        This is the most general constructor: it accepts any mix of uniform,
        Gaussian, categorical, lognormal, Weibull, triangular, and beta
        marginals. Parameter order follows the
        dict's insertion order, which must match the column order of the
        model's input matrix.

        Args:
            params: Mapping from parameter name to one of:
                a ``(low, high)`` tuple (shorthand for uniform),
                a :class:`UniformInputSpec`, a :class:`GaussianInputSpec`,
                a :class:`CategoricalInputSpec`, a :class:`LognormalInputSpec`,
                a :class:`WeibullInputSpec`, a :class:`TriangularInputSpec`,
                a :class:`BetaInputSpec`, or the matching spec dataclass.
            output_names: Optional output labels used by ``to_dataset()``.
            truncate_gaussians: Optional tail probability ``q`` in
                ``(0, 0.5)``. ``None`` (the default) leaves every Gaussian
                unbounded, which is the historical behaviour. Give a float and
                each Gaussian marginal gets an explicit ``low`` and ``high``
                at its own ``q`` and ``1 - q`` quantiles. This is the single
                place to opt into one bounded input model that every method
                then shares. A side the spec already declares is kept as
                written, so only open sides are filled.

                A marginal bounded this way carries real bounds, not a
                display convention. :func:`jaxgsa.morris.sample` therefore
                does not squash it a second time, and
                :meth:`jaxgsa.sobol.SobolSamples.to_morris` stops warning
                about unbounded tails.
            correlation: Optional ``(D, D)`` Gaussian-copula correlation
                matrix declaring the dependence between parameters; rows and
                columns follow the dict's insertion order. ``None`` (default)
                means independent inputs.
            correlation_type: Scale ``correlation`` is expressed on:
                ``"latent"`` (default) or ``"spearman"``.
            sampler: Optional custom joint and conditional transformation
                from uniform points to physical rows (see
                :class:`jaxgsa.sampling.InputSampler`). Cannot be combined
                with ``correlation``.
            sampler_dimension: Uniform coordinates the sampler consumes per
                row. ``None`` (default) means one per parameter.

        Returns:
            A normalized ``Problem`` instance.

        Raises:
            ValueError: If ``truncate_gaussians`` is not in ``(0, 0.5)``.
        """
        names = tuple(params.keys())
        input_specs = tuple(_normalize_input_spec(spec) for spec in params.values())
        if truncate_gaussians is not None:
            q = float(truncate_gaussians)
            # `not <` also rejects NaN, whose comparisons are always False.
            if not 0.0 < q < 0.5:
                raise ValueError(
                    f"truncate_gaussians must be in (0, 0.5), got {truncate_gaussians!r}"
                )
            input_specs = tuple(_truncate_gaussian_spec(spec, q) for spec in input_specs)
        return cls._from_normalized_inputs(
            names=names,
            input_specs=input_specs,
            output_names=_output_names_tuple(output_names),
            correlation=_canonical_correlation(
                correlation,
                names,
                input_specs,
                correlation_type,
                policy="declared",
            ),
            sampler=sampler,
            sampler_dimension=sampler_dimension,
        )

    def with_correlation(
        self,
        correlation: "npt.ArrayLike | None",
        *,
        correlation_type: Literal["latent", "spearman"] = "latent",
    ) -> "Problem":
        """Return a copy of this problem with the given correlation matrix.

        ``Problem`` is frozen, so a correlation attached after construction
        goes through this copy constructor. That is the fit-then-attach
        workflow with ``jaxgsa.sampling.fit_correlation``:

        .. code-block:: python

            R = jaxgsa.sampling.fit_correlation(problem, X_observed)
            problem = problem.with_correlation(R)

        Args:
            correlation: ``(D, D)`` Gaussian-copula correlation matrix, or
                ``None`` to drop a previously declared correlation.
            correlation_type: Scale ``correlation`` is expressed on:
                ``"latent"`` (default) or ``"spearman"``.

        Returns:
            A new ``Problem`` with the same marginals, names, and output
            names, and the validated correlation attached.
        """
        return Problem._from_normalized_inputs(
            names=self.names,
            input_specs=self._input_specs,
            output_names=self.output_names,
            correlation=_canonical_correlation(
                correlation,
                self.names,
                self._input_specs,
                correlation_type,
                policy="declared",
            ),
            sampler=self._sampler,
            sampler_dimension=self._sampler_dimension,
        )

    def with_sampler(
        self,
        sampler: InputSampler | None,
        *,
        sampler_dimension: int | None = None,
    ) -> "Problem":
        """Return a copy of this problem with the given custom sampler.

        ``Problem`` is frozen, so a sampler attached after construction goes
        through this copy constructor:

        .. code-block:: python

            problem = problem.with_sampler(sample_modular, sampler_dimension=2)

        Args:
            sampler: Joint and conditional transformation from uniform points
                to physical rows (see :class:`jaxgsa.sampling.InputSampler`),
                or ``None`` to drop a previously declared sampler.
            sampler_dimension: Uniform coordinates the sampler consumes per
                row. ``None`` (default) means one per parameter.

        Returns:
            A new ``Problem`` with the same marginals, names, and output
            names, and the sampler attached.

        Raises:
            ValueError: If the problem declares a correlation.
        """
        return Problem._from_normalized_inputs(
            names=self.names,
            input_specs=self._input_specs,
            output_names=self.output_names,
            correlation=self._correlation,
            sampler=sampler,
            sampler_dimension=sampler_dimension,
        )

    def with_output_names(self, output_names: tuple[str, ...] | None) -> "Problem":
        """Return a copy of this problem with the given output labels.

        ``Problem`` is frozen, so changing the output labels goes through
        this copy constructor. The irregular-output path uses it to analyze
        one output channel at a time under a single-name label.

        Args:
            output_names: New labels for the model's outputs, or ``None`` to
                drop previously declared labels.

        Returns:
            A new ``Problem`` with the same marginals, names, and
            correlation, and the given output labels.
        """
        return Problem._from_normalized_inputs(
            names=self.names,
            input_specs=self._input_specs,
            output_names=_output_names_tuple(output_names),
            correlation=self._correlation,
            sampler=self._sampler,
            sampler_dimension=self._sampler_dimension,
        )

    @classmethod
    def _from_normalized_inputs(
        cls,
        *,
        names: tuple[str, ...],
        input_specs: tuple[InputSpec, ...],
        output_names: tuple[str, ...] | None = None,
        correlation: _CorrelationTuple | None = None,
        sampler: InputSampler | None = None,
        sampler_dimension: int | None = None,
    ) -> "Problem":
        """Create a problem from internal normalized input specs.

        ``correlation`` must already be in canonical validated form (the
        nested-tuple output of :func:`_canonical_correlation`); this internal
        path performs no correlation validation of its own.
        """
        if len(names) != len(input_specs):
            raise ValueError(
                "names and input specs must have the same length, got "
                f"{len(names)} and {len(input_specs)}"
            )

        obj = object.__new__(cls)  # bypass __init__ to construct from already-validated specs
        obj._set_fields(
            names=tuple(names),
            input_specs=tuple(input_specs),
            output_names=output_names,
            correlation=correlation,
            sampler=sampler,
            sampler_dimension=sampler_dimension,
        )
        return obj

    def _set_fields(
        self,
        *,
        names: tuple[str, ...],
        input_specs: tuple[InputSpec, ...],
        output_names: tuple[str, ...] | None,
        correlation: _CorrelationTuple | None,
        sampler: InputSampler | None = None,
        sampler_dimension: int | None = None,
    ) -> None:
        """Assign validated frozen dataclass fields in one place."""
        if not names:
            raise ValueError("a Problem needs at least one parameter, got none")
        _validate_names(names)
        _check_correlation_touches_categorical(names, input_specs, correlation)
        sampler, sampler_dimension = _validated_sampler(
            sampler, sampler_dimension, len(names), correlation is not None
        )
        # Bypass frozen dataclass protection -- only called during construction.
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "output_names", output_names)
        object.__setattr__(self, "_input_specs", input_specs)
        object.__setattr__(self, "_correlation", correlation)
        object.__setattr__(self, "_sampler", sampler)
        object.__setattr__(self, "_sampler_dimension", sampler_dimension)

    def __repr__(self) -> str:
        """Show every parameter's marginal, not just uniform bounds.

        The generated dataclass ``repr`` would show only ``names`` and
        ``output_names``: :attr:`bounds` is a property, and ``_input_specs``
        and ``_correlation`` are marked ``repr=False`` because their tuple
        form is not meant for a human to read directly. A uniform-only
        problem is still shown by its bounds, the familiar and compact form;
        any other problem is shown by its full :attr:`input_specs`, so a
        Gaussian or categorical marginal is never silently missing.
        """
        parts = [f"names={self.names!r}"]
        if self.bounds is not None:
            parts.append(f"bounds={self.bounds!r}")
        else:
            parts.append(f"input_specs={self._input_specs!r}")
        if self.output_names is not None:
            parts.append(f"output_names={self.output_names!r}")
        if self._correlation is not None:
            parts.append("correlation=<declared>")
        if self._sampler is not None:
            parts.append(f"sampler=<custom, dimension={self._sampler_dimension}>")
        return f"Problem({', '.join(parts)})"

    @property
    def input_specs(self) -> tuple[InputSpec, ...]:
        """Canonical marginal spec of each parameter, in model-input order.

        Each entry is a :class:`UniformSpec`, :class:`GaussianSpec`,
        :class:`CategoricalSpec`, :class:`LognormalSpec`, :class:`WeibullSpec`,
        :class:`TriangularSpec`, or :class:`BetaSpec`, whatever input form declared it. Read the
        fields by name, and tell the families apart with ``isinstance``.
        """
        return self._input_specs

    @property
    def bounds(self) -> tuple[tuple[float, float], ...] | None:
        """Per-parameter ``(low, high)`` tuples, or ``None``.

        Derived from :attr:`input_specs` on every read: ``None`` as soon as
        one marginal is not uniform, since a Gaussian support has no
        meaningful finite bounds and a categorical marginal carries level
        codes rather than a range.
        """
        return _derive_bounds(self._input_specs)

    @property
    def correlation(self) -> "np.ndarray | None":
        """Latent Gaussian-copula correlation matrix, or ``None`` if independent.

        Always expressed on the latent scale regardless of the
        ``correlation_type`` it was declared with. Returns a fresh
        ``(D, D)`` float64 array; mutating it does not affect the problem.
        """
        if self._correlation is None:
            return None
        return np.asarray(self._correlation, dtype=np.float64)

    @property
    def sampler(self) -> InputSampler | None:
        """Custom joint and conditional sampler, or ``None``."""
        return self._sampler

    @property
    def sampler_dimension(self) -> int | None:
        """Uniform coordinates the custom sampler consumes per row, or ``None``."""
        return self._sampler_dimension

    @property
    def dependence(self) -> Dependence:
        """How the inputs depend on each other.

        ``"custom"`` when a :attr:`sampler` is declared, ``"gaussian"`` when a
        non-identity :attr:`correlation` is declared, and ``"independent"``
        otherwise. An identity matrix counts as independent.
        """
        if self._sampler is not None:
            return "custom"
        if self._correlation is None:
            return "independent"
        from jaxgsa._core.copula import is_independent

        if is_independent(np.asarray(self._correlation, dtype=np.float64)):
            return "independent"
        return "gaussian"

    @property
    def has_non_uniform_inputs(self) -> bool:
        """Return ``True`` when any parameter uses a non-uniform marginal."""
        return any(not isinstance(spec, UniformSpec) for spec in self._input_specs)

    @property
    def has_categorical_inputs(self) -> bool:
        """Return ``True`` when any parameter uses a categorical marginal."""
        return any(isinstance(spec, CategoricalSpec) for spec in self._input_specs)

    @property
    def categorical_labels(self) -> dict[str, tuple[str, ...]]:
        """Level labels of every categorical parameter, keyed by name.

        Samples always carry the integer level codes ``0 .. L-1`` (as
        floats). Code ``i`` of a parameter maps to ``labels[i]``. The labels
        are reporting metadata only, and jaxgsa never relabels arrays. The
        dict is empty when the problem has no categorical parameters.
        """
        return {
            self.names[d]: spec.labels
            for d, spec in enumerate(self._input_specs)
            if isinstance(spec, CategoricalSpec)
        }

    @property
    def n_params(self) -> int:
        """Return the number of parameters."""
        return len(self.names)
