"""One preamble for every ``analyze()``.

Nothing in this module is public API. See :mod:`jaxgsa._core`.

Every ``analyze()`` shares the same six preamble steps: validate the scalar
arguments, refuse a problem the method cannot handle, check the shape
contracts on ``X`` and ``Y``, check the sample for non-finite values and drop
what the policy says to drop, promote ``Y`` to ``(N, T, K)``, and warn about a
constant output slice.

:func:`prepare` is the one copy of that skeleton, so the steps cannot drift
between methods. It runs them in the order that keeps mistakes cheap: every
scalar argument is settled before any array is touched, and the capability
gates come from the method's own :class:`~jaxgsa._core.registry.MethodSpec`
rather than from a flag at the call site. What each ``analyze()`` keeps is its
estimator.

Reading ``sobol/_analyze.py`` (or any other method) alongside its call to
:func:`prepare` needs two things this module states once rather than at every
call site. First, ``ctx.Y`` is the *expanded* layout — one row per design row,
not one row per unique run — whenever the caller passed ``expand``; a method
built on a compacted, one-row-per-unique-run design has to compact it back
itself, the way ``sobol`` and ``morris`` do after reading ``ctx.keep``.
Second, ``ctx.keep`` is applied to ``X`` and ``Y`` already only when the
non-finite unit is a single row (``InvalidUnit.ROW``); for a grouped unit
(a Saltelli group, a Morris trajectory, an eFAST curve) the compaction has to
respect the group layout, which only the calling method knows, so ``keep`` is
returned unapplied and the caller drops the group itself.

Twelve methods know their model output by the time they call, and use
:func:`prepare` in one call. ``dgsm`` does not: on its autodiff path the model
output is what the preamble would validate, and it does not exist until the
model has been differentiated, so a misspelled ``on_invalid`` must not cost
that sweep. It settles its own scalar arguments up front, with the same
:func:`resolve_policy`, :func:`check_scalars` and :func:`gates` this module
runs internally, computes its output, and then calls :func:`prepare` once,
in full, exactly as every other method does. The second, cheap pass over the
scalar arguments inside that call costs nothing next to the differentiation
that came before it.

Three methods still do not fit the common shape, and the module accommodates
them rather than bending them:

* The four design-based methods (sobol, morris, efast, kucherenko) have no
  ``X``, and their capability gates fire in ``sample()``. There is no design
  to analyze that the sampler did not already agree to build, so
  :func:`prepare` does not gate them a second time.
* ``shapley`` forwards ``on_invalid`` to a backend instead of applying it. It
  has no ``invalid_unit``, so it uses :func:`gates` only.
* ``efast`` cannot drop anything. That is read off its unit — a search curve
  is an ordered sweep read by a discrete Fourier transform — and not passed in
  as a flag.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
from jaxgsa._core.precision import warn_on_float64_downcast
from jaxgsa._core.validation import (
    YLayout,
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
    "validate_inputs",
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
        layout: Which rank the caller passed, so :meth:`squeeze` can undo the
            promotion on each result field.
        keep: Per-unit mask of what survived. Already applied to ``X`` and
            ``Y`` when the unit is one row; left to the caller otherwise,
            because only the method knows how its groups are laid out.
        invalid: What the non-finite check found, and what it did.
        policy: The validated ``on_invalid`` policy.
        extra: The further sample-axis arrays the caller handed to
            :func:`prepare`, under the same names, checked with ``Y`` and
            compacted with it.
        deferred_expand: The design's output re-expansion, held back from
            the eager preamble. Only set under ``on_invalid="none"`` (the
            non-finite check never reads the expanded rows then), so the
            expansion can ride along with the estimator instead of
            materialising the expanded array first. ``None`` when the
            preamble ran it, or when no expansion is needed.
    """

    method: str
    problem: Problem
    X: Array | None
    Y: Array
    Y3: Array
    layout: YLayout
    keep: npt.NDArray[np.bool_]
    invalid: InvalidReport
    policy: OnInvalid
    extra: Mapping[str, Array] = field(default_factory=dict)
    deferred_expand: Callable[[Array], Array] | None = None

    def squeeze(self, arr: Array, *, n_trailing: int = 1) -> Array:
        """Undo the promotion :func:`prepare` applied, for one result field.

        Every estimator computes on ``Y3`` and so produces ``(..., T, K, ...)``
        fields, which have to come back out at the rank the caller passed.
        Reading the layout off the context means a call site names only the
        thing it cannot know, ``n_trailing``.

        Args:
            arr: One result field, with its ``(T, K)`` slice axes immediately
                before ``n_trailing`` trailing axes.
            n_trailing: Number of axes after K. ``1`` for a ``(..., T, K, D)``
                per-parameter field, ``2`` for a ``(..., T, K, D, D)`` pair
                matrix, ``0`` for a per-slice ``(..., T, K)`` scalar.

        Returns:
            The field at the caller's own rank.
        """
        return self.layout.squeeze(arr, n_trailing=n_trailing)


def prepare(
    spec: MethodSpec,
    problem: Problem,
    Y: Any,
    *,
    X: Any = None,
    on_invalid: object = "raise",
    checks: Sequence[ScalarCheck] = (),
    method: str | None = None,
    extra: Mapping[str, Any] | None = None,
    n_expected: int | None = None,
    expand: Callable[[Array], Array] | None = None,
    n_units: int | None = None,
    unit_of_row: npt.NDArray[np.intp] | None = None,
    unit_stride: int | None = None,
    row_labels: npt.NDArray[np.intp] | None = None,
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

    Twelve methods call this once, with their model output already in hand.
    ``dgsm`` cannot: on its autodiff path the output *is* the thing step 4
    would validate, and it does not exist until the model has been
    differentiated, which a misspelled ``on_invalid`` must not have to pay
    for. It runs :func:`resolve_policy`, :func:`check_scalars` and
    :func:`gates` itself first — the same three calls steps 1 and 2 make here
    — then differentiates the model, then calls this function once, in full,
    with the result. The second pass over the scalar arguments inside that
    call is cheap next to the differentiation that came before it, and it
    keeps one copy of the six steps and one order for them, rather than a
    second copy for the two-phase case.

    Args:
        spec: The calling method's registry record. Supplies the capability
            gates, the unit the non-finite check works in, and the default
            method name.
        problem: The problem definition.
        Y: The caller's model output.
        X: The input matrix, or ``None`` for a design-based method.
        on_invalid: The caller's non-finite policy, unvalidated. Under
            ``"none"`` (see :mod:`jaxgsa._core.invalid`) the non-finite scan
            does not run, no row is dropped, and the constant-slice warning is
            skipped with it: the analysis runs on the data exactly as given.
            The shape contracts still run, because they cost no scan.
        checks: Verdicts on the method's own scalar arguments, reported in
            order before anything else runs.
        method: Fully qualified analyzer name. Defaults to
            ``f"jaxgsa.{spec.name}.analyze"``.
        extra: Further arrays whose leading axis is the same sample axis as
            ``Y``'s, by name. They are read as model-side evidence: each is
            checked for non-finite values in the ``Y`` slot, and each is
            compacted with the same mask, so a method with a third array does
            not maintain its own copy of the drop. ``dgsm`` passes its
            Jacobian here.
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
        unit_stride: Rows per unit, when ``unit_of_row`` is the
            equal-contiguous-blocks pattern
            ``np.repeat(np.arange(n_units), unit_stride)``. Forwarded to the
            non-finite check, which collapses the per-unit verdict on the
            device instead of counting bad rows on the host; verified
            exactly, so a wrong stride falls back to the generic path.
            ``None`` (the default) always takes the generic path.
        row_labels: For each row checked here, the row the caller holds.
            A design-based method checks the expanded layout, but the caller
            evaluated one output per unique run, so the report has to name the
            rows they can act on. ``None`` when the two numberings agree.
        min_kept: Fewest surviving units the estimator can still work with.
        source_names: What to call ``X`` and ``Y`` in the non-finite report.
        warn_zero_variance: Whether to check for a constant output slice. Off
            only where the caller has already computed the variance and warns
            with it itself.
        zero_variance_outcome: Which consequence the warning reports; see
            :data:`jaxgsa._core.validation._ZERO_VARIANCE_OUTCOMES`.

    Returns:
        The :class:`Context` for the estimator to work from.

    Raises:
        ValueError: From any of the six steps. The first one to fail wins.
        TypeError: If the spec declares no ``invalid_unit``.
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

    # Before any jnp.asarray: that call is where a float64 array loses its
    # extra digits with x64 off, and the dtype it had is unreadable
    # afterwards. X is not a candidate; see warn_on_float64_downcast.
    warn_on_float64_downcast(method, {"Y": Y})
    extra_arrays = {name: jnp.asarray(a) for name, a in (extra or {}).items()}

    if X is not None:
        X = validate_inputs(problem, X)
        if n_expected is None:
            n_expected = int(X.shape[0])
    Y = _validate_output(Y, n_expected, problem)
    deferred_expand: Callable[[Array], Array] | None = None
    if expand is not None:
        # The user evaluated the model once per unique row. The estimator,
        # and the unit the check works in, both live in the expanded layout.
        # Under on_invalid='none' the check never reads the expanded rows,
        # so the expansion is deferred to the estimator (which may fuse it
        # with the standardization; see Context.deferred_expand); under
        # every other policy it must happen here, before the check sees the
        # data, and the context carries the already-expanded layout.
        if policy == "none":
            deferred_expand = expand
        else:
            Y = expand(Y)

    keep, invalid = check_invalid(
        policy=policy,
        method=method,
        unit=unit,
        n_units=int(Y.shape[0]) if n_units is None else n_units,
        Y=Y,
        X=X,
        extras=extra_arrays.values(),
        unit_of_row=unit_of_row,
        unit_stride=unit_stride,
        row_labels=row_labels,
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
        extra_arrays = {name: a[mask] for name, a in extra_arrays.items()}

    Y3, layout = _prepare_Y(Y)

    if warn_zero_variance and policy != "none":
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
        # Four frames reach the user: the helper, this function, the
        # ``analyze()`` that called it, and the user's own call. Every
        # ``prepare`` call site sits directly in a public ``analyze()``, so
        # the count is the same for all of them. The non-finite warnings next
        # door skip by file prefix instead, which needs no count at all;
        # _warn_zero_variance_slices takes a stacklevel and lives in
        # validation.py, so it cannot do the same until that helper changes.
        _warn_zero_variance_slices(
            Y_kept,
            output_names=problem.output_names,
            outcome=zero_variance_outcome,
            stacklevel=4,
            method=method,
        )

    return Context(
        method=method,
        problem=problem,
        X=X,
        Y=Y,
        Y3=Y3,
        layout=layout,
        keep=keep,
        invalid=invalid,
        policy=policy,
        extra=extra_arrays,
        deferred_expand=deferred_expand,
    )


def validate_inputs(problem: Problem, X: Any) -> Array:
    """Bring ``X`` onto the device and check the ``(N, D)`` contract.

    :func:`prepare` calls this on the ``X`` it is given. It is named rather
    than private because a method that *feeds* ``X`` to the model has to
    check it before differentiating: ``dgsm`` differentiates the model on
    ``X``, and a mis-shaped sample should be reported before the autodiff
    sweep rather than after it. Checking again inside ``prepare`` costs one
    shape comparison.

    Args:
        problem: The problem definition.
        X: The caller's input matrix.

    Returns:
        The validated matrix, on the device.

    Raises:
        ValueError: If ``X`` is not ``(N, D)`` for the problem's ``D``.
    """
    X = jnp.asarray(X)
    _validate_x(problem, X)
    return X
