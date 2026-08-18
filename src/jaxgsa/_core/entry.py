"""One preamble for every ``analyze()``.

Nothing in this module is public API. See :mod:`jaxgsa._core`.

Before this module, every entry point hand-wrote the same ten steps: validate
the scalar arguments, refuse a problem the method cannot handle, resolve the
non-finite policy, check the sample, drop what the policy says to drop,
promote ``Y`` to ``(N, T, K)``, and warn about a constant output slice. Forty
to sixty of the eighty to a hundred and sixty lines in a typical
``_analyze.py`` were that skeleton, written thirteen times.

Thirteen copies drifted, and the drift is where the bugs were. Two methods
never validated ``conf_level``, so a level of ``-0.2`` came back as an
inverted interval. One validated ``slice_chunk_size`` on one of its three
paths. Three accepted a negative resample count and silently returned no
interval at all. Six never warned about a constant output slice, and returned
NaN without a word. Three validated their scalar arguments *after* the
host-side data check, so a typo cost a full pass over the sample before it was
reported.

:func:`prepare` is the one copy. It runs the steps in the order that makes
those bugs impossible: every scalar argument is settled before any array is
touched, and the capability gates come from the method's own
:class:`~jaxgsa._core.registry.MethodSpec` rather than from a flag at the call
site. What each ``analyze()`` keeps is its estimator.

Four methods do not fit the common shape, and the module accommodates them
rather than bending them:

* The four design-based methods (sobol, morris, efast, kucherenko) have no
  ``X``, and their capability gates fire in ``sample()``. There is no design
  to analyze that the sampler did not already agree to build, so
  :func:`prepare` does not gate them a second time.
* ``dgsm`` takes two different argument groups, and the array in its ``Y``
  slot is not always the model output. It uses :func:`check_scalars` and
  :func:`gates` directly and keeps its own data check.
* ``shapley`` forwards ``on_invalid`` to a backend instead of applying it. It
  has no ``invalid_unit``, so it uses :func:`gates` only.
* ``efast`` cannot drop anything. That is read off its unit — a search curve
  is an ordered sweep read by a discrete Fourier transform — and not passed in
  as a flag.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from jaxgsa._core.invalid import (
    InvalidReport,
    InvalidUnit,
    OnInvalid,
    check_invalid,
    resolve_policy,
)
from jaxgsa._core.validation import (
    _prepare_Y,
    _raise_categorical_analysis,
    _raise_correlated_analysis,
    _validate_output,
    _validate_x,
    _warn_zero_variance_slices,
)

if TYPE_CHECKING:
    from jaxgsa._core.registry import MethodSpec
    from jaxgsa.problem import Problem

__all__ = [
    "Context",
    "ScalarCheck",
    "at_least",
    "check_scalars",
    "gates",
    "in_open_interval",
    "one_of",
    "prepare",
    "require",
]


@dataclass(frozen=True)
class ScalarCheck:
    """One already-evaluated verdict on one scalar argument.

    The condition and its message are both computed at the call site, which
    keeps the check readable next to the argument it is about. Building the
    message eagerly costs one f-string per argument and buys the guarantee
    this module exists for: the checks are a *value* the entry point hands to
    :func:`prepare`, so they cannot end up after the data check by accident.

    Attributes:
        ok: Whether the argument is acceptable.
        message: What to say when it is not.
    """

    ok: bool
    message: str


def require(ok: bool, message: str) -> ScalarCheck:
    """Build a check with a message the standard helpers cannot phrase."""
    return ScalarCheck(bool(ok), message)


def at_least(name: str, value: float | None, minimum: float) -> ScalarCheck:
    """Check ``value >= minimum``, passing ``None`` through as "not set"."""
    return ScalarCheck(
        value is None or value >= minimum,
        f"{name} must be >= {minimum}, got {value}",
    )


def in_open_interval(name: str, value: float | None, low: float, high: float) -> ScalarCheck:
    """Check ``low < value < high``, passing ``None`` through as "not set"."""
    return ScalarCheck(
        value is None or low < value < high,
        f"{name} must be in ({low}, {high}), got {value}",
    )


def one_of(name: str, value: object, options: Sequence[object]) -> ScalarCheck:
    """Check that ``value`` is one of ``options``."""
    listed = ", ".join(repr(o) for o in options)
    return ScalarCheck(
        value in options,
        f"{name} must be one of {{{listed}}}, got {value!r}",
    )


def check_scalars(checks: Sequence[ScalarCheck]) -> None:
    """Raise on the first failing scalar argument.

    Args:
        checks: The verdicts, in the order the caller wants them reported.

    Raises:
        ValueError: On the first check that did not pass.
    """
    for check in checks:
        if not check.ok:
            raise ValueError(check.message)


def gates(
    spec: MethodSpec,
    problem: Problem,
    *,
    method: str,
    check: tuple[str, ...] = ("correlation", "categorical"),
) -> None:
    """Refuse a problem the method has declared it cannot handle.

    The two capability questions are answered by the method's registry
    record, not by an argument at the call site. A method therefore cannot
    claim one thing in the registry and do another here.

    Args:
        spec: The calling method's registry record.
        problem: The problem the caller passed.
        method: Fully qualified analyzer name, for the message. Passed
            separately because ``shapley`` gates on behalf of a backend and
            wants its own name in the error.
        check: Which capabilities to gate on. ``shapley`` narrows this to the
            categorical half: whether it tolerates a correlation depends on
            the backend the caller picked, which is not a property of the
            method and so is not in its record.

    Raises:
        ValueError: If the problem trips a capability the spec refuses. A
            problem that trips both gets the combined message.
    """
    if "correlation" in check and spec.correlation == "refuses":
        _raise_correlated_analysis(problem, method)
    if "categorical" in check and spec.categorical == "refuses":
        _raise_categorical_analysis(problem, method)


@dataclass(frozen=True)
class Context:
    """What the preamble settled, for the estimator to work from.

    Both ranks of ``Y`` are carried on purpose. ``Y3`` is what a kernel that
    loops over ``(t, k)`` slices wants; ``Y`` is the caller's own rank, which
    the scalar fast paths still branch on and which gives the zero-variance
    warning its natural labels.

    Attributes:
        method: Fully qualified analyzer name, for any further messages.
        problem: The problem, unchanged.
        X: The input matrix with the dropped rows removed, or ``None`` for a
            design-based method.
        Y: The validated output in the caller's own rank, with the dropped
            rows removed.
        Y3: The same data promoted to ``(N, T, K)``.
        squeeze_time: Whether ``Y3``'s T axis was inserted by the promotion.
        squeeze_output: Whether ``Y3``'s K axis was inserted.
        keep: Per-unit mask of what survived. Already applied to ``X`` and
            ``Y`` when the unit is one row; left to the caller otherwise,
            because only the method knows how its groups are laid out.
        invalid: What the non-finite check found, and what it did.
        policy: The validated ``on_invalid`` policy.
    """

    method: str
    problem: Problem
    X: Array | None
    Y: Array
    Y3: Array
    squeeze_time: bool
    squeeze_output: bool
    keep: npt.NDArray[np.bool_]
    invalid: InvalidReport
    policy: OnInvalid

    @property
    def inputs(self) -> Array:
        """The input matrix, for a method that was given one.

        ``X`` is optional because a design-based method has none, so every
        given-data caller would otherwise narrow the type itself. Reading it
        through here states the expectation once instead of at each call
        site.

        Returns:
            The input matrix with the dropped rows already removed.

        Raises:
            TypeError: If this context came from a method with no ``X``. That
                is a mistake in the analyzer, not in the caller's data, so it
                is not a ``ValueError``.
        """
        if self.X is None:
            raise TypeError(
                f"{self.method}: no input matrix was passed to prepare(), so "
                "Context.inputs is unavailable. A design-based method should "
                "read its design instead."
            )
        return self.X


def prepare(
    spec: MethodSpec,
    problem: Problem,
    Y: Any,
    *,
    X: Any = None,
    on_invalid: object = "raise",
    checks: Sequence[ScalarCheck] = (),
    method: str | None = None,
    n_expected: int | None = None,
    expand: Callable[[Array], Array] | None = None,
    n_units: int | None = None,
    unit_of_row: npt.NDArray[np.intp] | None = None,
    min_kept: int = 1,
    source_names: tuple[str, str] = ("X", "Y"),
    warn_zero_variance: bool = True,
    zero_variance_outcome: str = "nan",
) -> Context:
    """Run the shared analysis preamble and return what it settled.

    The order is the point of the function:

    1. ``on_invalid`` and every other scalar argument, before any array work.
    2. The capability gates, from ``spec``.
    3. The shape contracts on ``X`` and ``Y``, then any design expansion.
    4. The non-finite check, and the removal it calls for.
    5. The promotion of ``Y`` to ``(N, T, K)``.
    6. The zero-variance warning.

    Steps 1 and 2 touch no array, so a mistyped argument costs nothing. Step 4
    reads values on the host and is the first expensive thing that happens.

    Args:
        spec: The calling method's registry record. Supplies the capability
            gates, the unit the non-finite check works in, and the default
            method name.
        problem: The problem definition.
        Y: The caller's model output.
        X: The input matrix, or ``None`` for a design-based method.
        on_invalid: The caller's non-finite policy, unvalidated.
        checks: Verdicts on the method's own scalar arguments, reported in
            order before anything else runs.
        method: Fully qualified analyzer name. Defaults to
            ``f"jaxgsa.{spec.name}.analyze"``.
        n_expected: Rows ``Y`` must have. Defaults to ``X``'s row count; pass
            it explicitly for a design-based method, whose row count comes
            from the design.
        expand: How to rebuild the full design layout from one output per
            unique design row, normally ``SamplingResult.expand_outputs``. It
            runs after the shape contract and before the non-finite check, so
            the check sees whole units. ``None`` where the caller's rows are
            already the design's rows.
        n_units: Units the non-finite check reports on. Defaults to ``Y``'s
            row count, which is right whenever one row is one unit.
        unit_of_row: For each row, the unit it belongs to. ``None`` when one
            row is one unit.
        min_kept: Fewest surviving units the estimator can still work with.
        source_names: What to call the two arrays in messages.
        warn_zero_variance: Whether to check for a constant output slice. Off
            only where the caller has already computed the variance and warns
            with it itself.
        zero_variance_outcome: Which consequence the warning reports; see
            :data:`jaxgsa._core.validation._ZERO_VARIANCE_OUTCOMES`.

    Returns:
        The :class:`Context` for the estimator to work from.

    Raises:
        ValueError: From any of the six steps. The first one to fail wins.
    """
    method = method or f"jaxgsa.{spec.name}.analyze"
    unit = spec.invalid_unit
    # Defensive: no registered method reaches this. Every spec that calls
    # prepare declares an invalid_unit, and shapley, the one spec that does
    # not, uses gates() instead. Nothing tests this branch; it is here so a
    # new method that forgets the field fails with a sentence rather than an
    # AttributeError deeper in the check.
    if unit is None:  # pragma: no cover - defensive
        raise TypeError(
            f"{method}: {spec.name} declares no invalid_unit, so it cannot use prepare"
        )

    # A search curve is an ordered sweep read by a discrete Fourier transform.
    # Removing one, or a point inside one, changes what the estimator computes
    # rather than shrinking the sample. That is a property of the unit, so it
    # is read off the unit instead of being passed in.
    policy = resolve_policy(
        on_invalid,
        method=method,
        unit=unit,
        allow_drop=unit is not InvalidUnit.CURVE,
    )
    check_scalars(checks)

    # The design-based methods gated in sample(). There is no design to
    # analyze that the sampler did not already agree to build, so re-gating
    # here would only add a second message for the same refusal.
    if not spec.is_design_based:
        gates(spec, problem, method=method)

    if X is not None:
        X = _validate_x_array(problem, X)
        if n_expected is None:
            n_expected = int(X.shape[0])
    Y = _validate_output(Y, n_expected, problem)
    if expand is not None:
        # The user evaluated the model once per unique row. The estimator, and
        # the unit the check works in, both live in the expanded layout.
        Y = expand(Y)

    keep, invalid = check_invalid(
        policy=policy,
        method=method,
        unit=unit,
        n_units=int(Y.shape[0]) if n_units is None else n_units,
        Y=Y,
        X=X,
        unit_of_row=unit_of_row,
        min_kept=min_kept,
        source_names=source_names,
    )
    # One row is one unit only for InvalidUnit.ROW. For a grouped design the
    # compaction has to respect the group layout, and only the method knows
    # it, so the mask is handed back unapplied.
    if unit is InvalidUnit.ROW and not keep.all():
        mask = np.asarray(keep)
        Y = Y[mask]
        if X is not None:
            X = X[mask]

    Y3, squeeze_time, squeeze_output = _prepare_Y(Y)

    if warn_zero_variance:
        # The warning has to see what the estimator will see. For a grouped
        # unit the caller has not compacted yet, so the surviving rows are
        # selected here; leaving the dropped rows in would hide a constant
        # slice behind the non-finite values that were removed.
        Y_kept = Y
        if unit_of_row is not None and not keep.all():
            Y_kept = Y[np.asarray(keep)[unit_of_row]]
        # Warned on the caller's own rank, not on Y3: a (N, K) output then
        # gets "k=2" rather than the fabricated "(t=0, k=2)" that the
        # inserted singleton time axis would produce.
        _warn_zero_variance_slices(
            Y_kept,
            output_names=problem.output_names,
            outcome=zero_variance_outcome,
            stacklevel=3,
        )

    return Context(
        method=method,
        problem=problem,
        X=X,
        Y=Y,
        Y3=Y3,
        squeeze_time=squeeze_time,
        squeeze_output=squeeze_output,
        keep=keep,
        invalid=invalid,
        policy=policy,
    )


def _validate_x_array(problem: Problem, X: Any) -> Array:
    """Bring ``X`` onto the device and check the ``(N, D)`` contract."""
    X = jnp.asarray(X)
    _validate_x(problem, X)
    return X
