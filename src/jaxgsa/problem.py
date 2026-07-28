"""Defines the ``Problem`` dataclass and accepted input specifications."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NotRequired, TypeAlias, TypedDict

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


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

    Note that ``variance`` is the variance, not the standard deviation.
    Provide ``low`` and/or ``high`` to truncate the support; omit both for an
    unbounded Gaussian.
    """

    dist: Literal["gaussian"]
    mean: float
    variance: float
    low: NotRequired[float]
    high: NotRequired[float]


# Public-facing union type -- users pass one of these per parameter.
InputSpecValue: TypeAlias = tuple[float, float] | UniformInputSpec | GaussianInputSpec
# Hashable canonical storage for a validated latent correlation matrix.
_CorrelationTuple: TypeAlias = tuple[tuple[float, ...], ...]
# Internal canonical form: (dist, param1, param2, lo_bound, hi_bound).
# For uniform: param1=low, param2=high. For gaussian: param1=mean, param2=variance.
_NormalizedInputSpec: TypeAlias = tuple[
    Literal["uniform", "gaussian"],
    float,
    float,
    float | None,
    float | None,
]


def _make_uniform_spec(low: float, high: float) -> _NormalizedInputSpec:
    """Validate and normalize a uniform input specification."""
    low = float(low)
    high = float(high)
    # `not <` instead of `>=` to also catch NaN (NaN comparisons are always False).
    if not low < high:
        raise ValueError(f"Uniform input requires low < high, got {(low, high)!r}")
    return ("uniform", low, high, None, None)


def _make_gaussian_spec(
    mean: float,
    variance: float,
    *,
    low: float | None = None,
    high: float | None = None,
) -> _NormalizedInputSpec:
    """Validate and normalize a Gaussian input specification."""
    # Coerce to Python float to prevent JAX tracers or numpy scalars from leaking into metadata.
    mean = float(mean)
    variance = float(variance)
    low = None if low is None else float(low)
    high = None if high is None else float(high)

    if variance <= 0:
        raise ValueError(f"Gaussian input requires variance > 0, got {variance!r}")
    if low is not None and high is not None and not low < high:
        raise ValueError(f"Truncated Gaussian input requires low < high, got {(low, high)!r}")

    return ("gaussian", mean, variance, low, high)


def _normalize_input_spec(spec: InputSpecValue) -> _NormalizedInputSpec:
    """Normalize tuple or TypedDict user input into a private immutable spec."""
    if isinstance(spec, tuple):  # bare (low, high) tuple is shorthand for uniform
        if len(spec) != 2:
            raise ValueError("Tuple input specs must have exactly two values: (low, high)")
        return _make_uniform_spec(spec[0], spec[1])

    if spec["dist"] == "uniform":
        return _make_uniform_spec(spec["low"], spec["high"])
    if spec["dist"] == "gaussian":
        return _make_gaussian_spec(
            spec["mean"],
            spec["variance"],
            low=spec.get("low"),
            high=spec.get("high"),
        )

    raise ValueError(f"Unsupported input distribution {spec['dist']!r}")


def _truncate_gaussian_spec(spec: _NormalizedInputSpec, q: float) -> _NormalizedInputSpec:
    """Fill the open sides of a Gaussian spec with its own ``q`` quantiles.

    Uniform specs and sides the user already declared are returned unchanged,
    so an explicit ``low`` or ``high`` always wins over the automatic bound.

    Args:
        spec: Normalized input spec.
        q: Tail probability to cut from each open side.

    Returns:
        The spec with every open Gaussian side bounded at that marginal's own
        ``q`` (low) and ``1 - q`` (high) quantile.
    """
    from scipy.stats import norm

    dist, mean, variance, low, high = spec
    if dist != "gaussian" or (low is not None and high is not None):
        return spec

    std = math.sqrt(variance)
    if low is None:
        low = float(norm.ppf(q, loc=mean, scale=std))
    if high is None:
        high = float(norm.ppf(1.0 - q, loc=mean, scale=std))
    return ("gaussian", mean, variance, low, high)


def _derive_bounds(
    input_specs: tuple[_NormalizedInputSpec, ...],
) -> tuple[tuple[float, float], ...] | None:
    """Return finite bounds for uniform-only problems, otherwise ``None``."""
    # Bounds only meaningful for uniform-only problems; Gaussian inputs have
    # infinite (or truncated) support that doesn't map to simple lo/hi bounds.
    bounds: list[tuple[float, float]] = []
    for dist, first, second, _, _ in input_specs:
        if dist != "uniform":
            return None
        bounds.append((first, second))
    return tuple(bounds)


def _canonical_correlation(
    correlation: "npt.ArrayLike | None",
    n_params: int,
    kind: Literal["latent", "spearman"] = "latent",
    *,
    warn_on_repair: bool,
) -> _CorrelationTuple | None:
    """Validate a correlation matrix and freeze it into hashable form.

    ``None`` passes through (independent inputs). Anything else is converted
    from ``kind`` to the latent scale, validated (with positive-definiteness
    repair), and stored as a nested tuple of floats so ``Problem`` stays
    frozen and hashable.
    """
    if correlation is None:
        return None
    # Imported lazily: jaxgsa._core.copula imports this module at load time.
    from jaxgsa._core.copula import canonicalize_correlation

    R = canonicalize_correlation(correlation, n_params, kind=kind, warn_on_repair=warn_on_repair)
    return tuple(tuple(float(value) for value in row) for row in R)


def _normalized_input_to_dict(spec: _NormalizedInputSpec) -> UniformInputSpec | GaussianInputSpec:
    """Convert a normalized immutable input spec into a JSON-friendly mapping."""
    dist, first, second, low, high = spec
    if dist == "uniform":
        return UniformInputSpec(dist="uniform", low=first, high=second)

    # Reconstruct the TypedDict for JSON serialization (used by SobolSamples.save).
    payload: GaussianInputSpec = GaussianInputSpec(
        dist="gaussian",
        mean=first,
        variance=second,
    )
    if low is not None:
        payload["low"] = low
    if high is not None:
        payload["high"] = high
    return payload


# frozen for hashability and safety; init=False because we define a custom __init__ for validation.
@dataclass(frozen=True, init=False)
class Problem:
    """Immutable description of a model's uncertain input parameters.

    Every jaxgsa sampling and analysis function takes a ``Problem`` that names
    the input parameters (in the order the model expects them) and gives each
    one a marginal distribution. Instances are frozen, so one ``Problem`` can
    safely be shared across analyses.

    The direct constructor accepts uniform marginals only, given as finite
    ``(low, high)`` bounds. Use :meth:`from_dict` when you need mixed uniform
    and Gaussian marginals.

    Optionally, a Gaussian-copula ``correlation`` matrix couples the
    marginals: samples then follow the declared dependence structure while
    each parameter keeps its marginal exactly as written.
    ``jaxgsa.sampling.monte_carlo`` honors it transparently; methods whose
    indices assume independent inputs refuse a correlated problem with a
    ``ValueError``.

    Attributes:
        names: Parameter names in model-input order.
        bounds: Per-parameter ``(low, high)`` tuples when every marginal is
            uniform, or ``None`` as soon as any marginal is Gaussian (whose
            support has no meaningful finite bounds).
        output_names: Optional labels for the model's outputs, used to name
            the ``output`` coordinate in ``to_dataset()`` exports.
    """

    names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...] | None
    _input_specs: tuple[_NormalizedInputSpec, ...] = field(repr=False)
    output_names: tuple[str, ...] | None = None
    _correlation: _CorrelationTuple | None = field(repr=False, default=None)

    def __init__(
        self,
        names: tuple[str, ...],
        bounds: tuple[tuple[float, float], ...],
        output_names: tuple[str, ...] | None = None,
        correlation: "npt.ArrayLike | None" = None,
        correlation_kind: Literal["latent", "spearman"] = "latent",
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
                (default) means independent inputs. Validated on entry; a
                non-positive-definite matrix is repaired with a
                ``UserWarning``.
            correlation_kind: Scale ``correlation`` is expressed on:
                ``"latent"`` (default) for the Pearson correlation of the
                copula's latent normals, ``"spearman"`` for a rank
                correlation (converted via ``2 sin(pi rho_s / 6)``).
        """
        normalized_names = tuple(names)
        normalized_bounds = tuple((float(low), float(high)) for low, high in bounds)
        if len(normalized_names) != len(normalized_bounds):
            raise ValueError(
                "names and bounds must have the same length, got "
                f"{len(normalized_names)} and {len(normalized_bounds)}"
            )

        input_specs = tuple(_make_uniform_spec(low, high) for low, high in normalized_bounds)
        self._set_fields(
            names=normalized_names,
            input_specs=input_specs,
            output_names=output_names,
            # User-declared matrices deserve the repair warning.
            correlation=_canonical_correlation(
                correlation, len(normalized_names), correlation_kind, warn_on_repair=True
            ),
        )

    @classmethod
    def from_dict(
        cls,
        params: dict[str, InputSpecValue],
        output_names: tuple[str, ...] | None = None,
        *,
        truncate_gaussians: float | None = None,
        correlation: "npt.ArrayLike | None" = None,
        correlation_kind: Literal["latent", "spearman"] = "latent",
    ) -> "Problem":
        """Create a ``Problem`` from per-parameter distribution specs.

        This is the most general constructor: it accepts any mix of uniform
        and Gaussian marginals. Parameter order follows the dict's insertion
        order, which must match the column order of the model's input matrix.

        Args:
            params: Mapping from parameter name to one of:
                a ``(low, high)`` tuple (shorthand for uniform),
                a :class:`UniformInputSpec`, or a :class:`GaussianInputSpec`.
            output_names: Optional output labels used by ``to_dataset()``.
            truncate_gaussians: Optional tail probability ``q`` in
                ``(0, 0.5)``. ``None`` (the default) leaves every Gaussian
                unbounded, which is the historical behaviour. Give a float and
                each Gaussian marginal gets an explicit ``low`` and ``high``
                at its own ``q`` and ``1 - q`` quantiles. This is the single
                place to opt into one bounded input model that every method
                then shares. A side the spec already declares is kept as
                written, so only open sides are filled.

                A marginal bounded this way is *genuinely* bounded, so
                :func:`jaxgsa.morris.sample` does not squash it a second time
                and :meth:`jaxgsa.sobol.SobolSamples.to_morris` stops warning
                about unbounded tails.
            correlation: Optional ``(D, D)`` Gaussian-copula correlation
                matrix declaring the dependence between parameters; rows and
                columns follow the dict's insertion order. ``None`` (default)
                means independent inputs.
            correlation_kind: Scale ``correlation`` is expressed on:
                ``"latent"`` (default) or ``"spearman"``.

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
            output_names=output_names,
            correlation=_canonical_correlation(
                correlation, len(names), correlation_kind, warn_on_repair=True
            ),
        )

    def with_correlation(
        self,
        correlation: "npt.ArrayLike | None",
        *,
        kind: Literal["latent", "spearman"] = "latent",
    ) -> "Problem":
        """Return a copy of this problem with the given correlation matrix.

        ``Problem`` is frozen, so attaching a correlation after construction
        — the fit-then-attach workflow with
        ``jaxgsa.sampling.fit_correlation`` — goes through this copy
        constructor:

        .. code-block:: python

            R = jaxgsa.sampling.fit_correlation(problem, X_observed)
            problem = problem.with_correlation(R)

        Args:
            correlation: ``(D, D)`` Gaussian-copula correlation matrix, or
                ``None`` to drop a previously declared correlation.
            kind: Scale ``correlation`` is expressed on: ``"latent"``
                (default) or ``"spearman"``.

        Returns:
            A new ``Problem`` with the same marginals, names, and output
            names, and the validated correlation attached.
        """
        return Problem._from_normalized_inputs(
            names=self.names,
            input_specs=self._input_specs,
            output_names=self.output_names,
            correlation=_canonical_correlation(
                correlation, self.num_vars, kind, warn_on_repair=True
            ),
        )

    @classmethod
    def _from_normalized_inputs(
        cls,
        *,
        names: tuple[str, ...],
        input_specs: tuple[_NormalizedInputSpec, ...],
        output_names: tuple[str, ...] | None = None,
        correlation: _CorrelationTuple | None = None,
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
        )
        return obj

    def _set_fields(
        self,
        *,
        names: tuple[str, ...],
        input_specs: tuple[_NormalizedInputSpec, ...],
        output_names: tuple[str, ...] | None,
        correlation: _CorrelationTuple | None,
    ) -> None:
        """Assign validated frozen dataclass fields in one place."""
        # Bypass frozen dataclass protection -- only called during construction.
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "bounds", _derive_bounds(input_specs))
        normalized = tuple(output_names) if output_names is not None else None
        object.__setattr__(self, "output_names", normalized)
        object.__setattr__(self, "_input_specs", input_specs)
        object.__setattr__(self, "_correlation", correlation)

    @property
    def input_specs(self) -> tuple[_NormalizedInputSpec, ...]:
        """Normalized input distribution specs for each parameter."""
        return self._input_specs

    @property
    def correlation(self) -> "np.ndarray | None":
        """Latent Gaussian-copula correlation matrix, or ``None`` if independent.

        Always expressed on the latent scale regardless of the
        ``correlation_kind`` it was declared with. Returns a fresh
        ``(D, D)`` float64 array; mutating it does not affect the problem.
        """
        if self._correlation is None:
            return None
        return np.asarray(self._correlation, dtype=np.float64)

    @property
    def has_correlated_inputs(self) -> bool:
        """Return ``True`` when a non-identity correlation matrix is declared."""
        if self._correlation is None:
            return False
        from jaxgsa._core.copula import is_independent

        return not is_independent(np.asarray(self._correlation, dtype=np.float64))

    @property
    def has_non_uniform_inputs(self) -> bool:
        """Return ``True`` when any parameter uses a non-uniform marginal."""
        return any(spec[0] != "uniform" for spec in self._input_specs)

    @property
    def num_vars(self) -> int:
        """Return the number of parameters."""
        return len(self.names)
