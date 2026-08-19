"""Defines the ``Problem`` dataclass and accepted input specifications."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, NotRequired, TypeAlias, TypedDict

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from jaxgsa._core.copula import RepairPolicy


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


# Hashable canonical storage for a validated latent correlation matrix.
_CorrelationTuple: TypeAlias = tuple[tuple[float, ...], ...]

# Renormalize probs whose sum is off by at most this much; raise beyond it.
_PROB_SUM_TOL = 1e-3

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

    def __post_init__(self) -> None:
        """Coerce the bounds to Python floats and check their order.

        Raises:
            ValueError: If ``low`` is not strictly below ``high``.
        """
        # Coerce to Python float to keep JAX tracers and numpy scalars out of
        # this hashable jit-cache metadata.
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "high", float(self.high))
        # `not <` instead of `>=` to also catch NaN (NaN comparisons are always False).
        if not self.low < self.high:
            raise ValueError(f"Uniform input requires low < high, got {(self.low, self.high)!r}")


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
        mean: Mean of the underlying (untruncated) Gaussian.
        variance: Variance of the underlying Gaussian, strictly positive.
            This is the variance, not the standard deviation.
        low: Lower truncation bound, or ``None`` for an open left tail.
        high: Upper truncation bound, or ``None`` for an open right tail.
    """

    mean: float
    variance: float
    low: float | None = None
    high: float | None = None

    #: Distribution tag, matching the ``"dist"`` key of the dict input form.
    dist: ClassVar[Literal["gaussian"]] = "gaussian"

    def __post_init__(self) -> None:
        """Coerce the parameters to Python floats and validate them.

        Raises:
            ValueError: If ``variance`` is not positive, or if both bounds are
                given and ``low`` is not strictly below ``high``.
        """
        # Coerce to Python float to prevent JAX tracers or numpy scalars from
        # leaking into metadata.
        object.__setattr__(self, "mean", float(self.mean))
        object.__setattr__(self, "variance", float(self.variance))
        object.__setattr__(self, "low", None if self.low is None else float(self.low))
        object.__setattr__(self, "high", None if self.high is None else float(self.high))

        if self.variance <= 0:
            raise ValueError(f"Gaussian input requires variance > 0, got {self.variance!r}")
        if self.low is not None and self.high is not None and not self.low < self.high:
            raise ValueError(
                f"Truncated Gaussian input requires low < high, got {(self.low, self.high)!r}"
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

    def __post_init__(self) -> None:
        """Normalize the probabilities and labels into validated tuples.

        Raises:
            ValueError: If fewer than two levels are given, any probability is
                not positive and finite, the probabilities do not sum to 1
                within tolerance, or ``labels`` has the wrong length or
                duplicate entries.
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
        label_values = tuple(str(label) for label in self.labels)
        if len(label_values) != n_levels:
            raise ValueError(
                f"Categorical labels length {len(label_values)} does not match "
                f"the {n_levels} levels declared by probs"
            )
        if len(set(label_values)) != n_levels:
            raise ValueError(f"Categorical labels must be unique, got {label_values!r}")
        object.__setattr__(self, "labels", label_values)


# Canonical stored form of one parameter's marginal. This is what
# `Problem.input_specs` returns, and what every internal consumer reads.
InputSpec: TypeAlias = UniformSpec | GaussianSpec | CategoricalSpec

# Public-facing input union -- users pass one of these per parameter.
InputSpecValue: TypeAlias = (
    tuple[float, float]
    | UniformInputSpec
    | GaussianInputSpec
    | CategoricalInputSpec
    | UniformSpec
    | GaussianSpec
    | CategoricalSpec
)


def _normalize_input_spec(spec: InputSpecValue) -> InputSpec:
    """Normalize any accepted user input form into a canonical spec dataclass."""
    if isinstance(spec, UniformSpec | GaussianSpec | CategoricalSpec):
        return spec  # already canonical and validated by its own __post_init__
    if isinstance(spec, tuple):  # bare (low, high) tuple is shorthand for uniform
        if len(spec) != 2:
            raise ValueError("Tuple input specs must have exactly two values: (low, high)")
        return UniformSpec(spec[0], spec[1])

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
        return CategoricalSpec(
            tuple(spec["probs"]),
            labels=() if labels is None else tuple(str(label) for label in labels),
        )

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
    if correlation is None:
        return None
    # Imported lazily: jaxgsa._core.copula imports this module at load time.
    from jaxgsa._core.copula import _force_categorical_identity, canonicalize_correlation

    n_params = len(names)
    R_declared = np.asarray(correlation, dtype=np.float64)
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
) -> UniformInputSpec | GaussianInputSpec | CategoricalInputSpec:
    """Convert a canonical input spec into a JSON-friendly mapping."""
    if isinstance(spec, UniformSpec):
        return UniformInputSpec(dist="uniform", low=spec.low, high=spec.high)
    if isinstance(spec, CategoricalSpec):
        label_list: list[str | int | float] = list(spec.labels)
        return CategoricalInputSpec(dist="categorical", probs=list(spec.probs), labels=label_list)

    # Reconstruct the TypedDict for JSON serialization (used by SobolSamples.save).
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


def _validate_names(names: tuple[str, ...]) -> None:
    """Reject parameter names that are not strings or not unique.

    A non-string name breaks every keyed surface downstream — ``Theta``
    mappings, dataset ``param`` coordinates, saved-design metadata — and a
    duplicate makes those keys ambiguous, so both are refused at
    construction, where the fix is one line away.

    Args:
        names: Parameter names in model-input order.

    Raises:
        ValueError: If any name is not a ``str`` (convert it with ``str(...)``
            when declaring the problem), or if two parameters share a name
            (rename one of them).
    """
    non_str = [name for name in names if not isinstance(name, str)]
    if non_str:
        raise ValueError(
            f"parameter names must be strings, got {non_str!r}. Convert each "
            "name with str(...) when declaring the problem."
        )
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(
            f"parameter names must be unique, but {duplicates!r} appear more "
            "than once. Rename the duplicated parameters so every name is "
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
    ``jaxgsa.sampling.monte_carlo`` honors the matrix transparently. Methods
    whose indices assume independent inputs refuse a correlated problem with
    a ``ValueError``.

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
    bounds: tuple[tuple[float, float], ...] | None
    _input_specs: tuple[InputSpec, ...] = field(repr=False)
    output_names: tuple[str, ...] | None = None
    _correlation: _CorrelationTuple | None = field(repr=False, default=None)

    def __init__(
        self,
        names: tuple[str, ...],
        bounds: tuple[tuple[float, float], ...],
        output_names: tuple[str, ...] | None = None,
        correlation: "npt.ArrayLike | None" = None,
        correlation_type: Literal["latent", "spearman"] = "latent",
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
        """
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
            output_names=output_names,
            # User-declared matrices deserve the repair warning.
            correlation=_canonical_correlation(
                correlation,
                normalized_names,
                input_specs,
                correlation_type,
                policy="declared",
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
        correlation_type: Literal["latent", "spearman"] = "latent",
    ) -> "Problem":
        """Create a ``Problem`` from per-parameter distribution specs.

        This is the most general constructor: it accepts any mix of uniform,
        Gaussian, and categorical marginals. Parameter order follows the
        dict's insertion order, which must match the column order of the
        model's input matrix.

        Args:
            params: Mapping from parameter name to one of:
                a ``(low, high)`` tuple (shorthand for uniform),
                a :class:`UniformInputSpec`, a :class:`GaussianInputSpec`,
                or a :class:`CategoricalInputSpec`.
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
                correlation,
                names,
                input_specs,
                correlation_type,
                policy="declared",
            ),
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
        )

    @classmethod
    def _from_normalized_inputs(
        cls,
        *,
        names: tuple[str, ...],
        input_specs: tuple[InputSpec, ...],
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
        input_specs: tuple[InputSpec, ...],
        output_names: tuple[str, ...] | None,
        correlation: _CorrelationTuple | None,
    ) -> None:
        """Assign validated frozen dataclass fields in one place."""
        _validate_names(names)
        _check_correlation_touches_categorical(names, input_specs, correlation)
        # Bypass frozen dataclass protection -- only called during construction.
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "bounds", _derive_bounds(input_specs))
        normalized = tuple(output_names) if output_names is not None else None
        object.__setattr__(self, "output_names", normalized)
        object.__setattr__(self, "_input_specs", input_specs)
        object.__setattr__(self, "_correlation", correlation)

    @property
    def input_specs(self) -> tuple[InputSpec, ...]:
        """Canonical marginal spec of each parameter, in model-input order.

        Each entry is a :class:`UniformSpec`, :class:`GaussianSpec`, or
        :class:`CategoricalSpec`, whatever input form declared it. Read the
        fields by name, and tell the families apart with ``isinstance``.
        """
        return self._input_specs

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
    def has_correlated_inputs(self) -> bool:
        """Return ``True`` when a non-identity correlation matrix is declared."""
        if self._correlation is None:
            return False
        from jaxgsa._core.copula import is_independent

        return not is_independent(np.asarray(self._correlation, dtype=np.float64))

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
    def num_vars(self) -> int:
        """Return the number of parameters."""
        return len(self.names)
