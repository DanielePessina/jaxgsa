"""One policy for non-finite model outputs and inputs.

A model that fails on some of its runs returns ``NaN`` or ``inf``. Every
``analyze()`` entry point answers that the same way, through one keyword,
``on_invalid``, with four values:

``"raise"``
    Refuse the analysis. This is the default. An index computed from a partial
    sample is a different quantity from the one the user asked for, and
    ``analyze()`` is cheap to run again once the failed runs are understood.

``"propagate"``
    Compute anyway and let the non-finite value flow into the indices. The
    result is loudly wrong rather than quietly wrong, which is sometimes what
    you want while debugging a model.

``"drop"``
    Remove the affected data and analyze what is left. What "affected data"
    means depends on the design; see :class:`InvalidUnit`.

``"none"``
    Skip the non-finite check entirely. The analysis runs on the data exactly
    as given, and the constant-slice warning is skipped with it. On a design
    with millions of expanded rows the two scans are most of the fixed cost
    of an analysis, so this is the fast path for a user who has already
    sanitized the output. A ``NaN`` that reaches the estimator flows into the
    indices, and a constant slice turns into ``NaN`` indices without a word.
    The shape contracts still run: a misaligned output would silently corrupt
    a block split, and checking the shape costs no data scan.

Whatever the policy, the count and the position of the affected data are
recorded in an :class:`InvalidReport` — under ``"none"`` the check never
ran, so the report always says clean. The report is the part that matters in
practice: it names the runs to investigate.

Note:
    ``"drop"`` is not available everywhere, and the restrictions are not
    stylistic. See :func:`resolve_policy`.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, TypeAlias, cast

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from jaxgsa._core.warning_types import JaxgsaWarning

# Every jaxgsa source file starts with this prefix. A preamble warning raised
# from deep inside the package (this module, called through prepare()) skips
# every frame under it, so Python attributes the warning to the caller's own
# line instead of to a helper several calls up. A hand-counted stacklevel
# breaks the moment another internal call is added between the warning and
# the caller; this does not, because it is a property of the file, not of the
# call depth.
_PACKAGE_DIR = str(Path(__file__).resolve().parent.parent) + os.sep

__all__ = [
    "InvalidReport",
    "InvalidUnit",
    "OnInvalid",
    "check_invalid",
    "resolve_policy",
]

OnInvalid: TypeAlias = Literal["raise", "propagate", "drop", "none"]

_POLICIES: tuple[str, ...] = ("raise", "propagate", "drop", "none")

# Below this many surviving units the estimate is not worth reporting without
# saying so. One floor, applied everywhere a "drop" policy can leave a sample
# this small.
_LOW_SURVIVOR_WARN = 10

# How many individual positions an error message or repr lists before it
# summarizes. Long enough to paste into a debugger, short enough to read.
_MAX_LISTED = 10


class InvalidUnit(Enum):
    """The block of data that one non-finite value invalidates.

    A sensitivity design is not always a list of independent rows. Where it is
    not, removing a single row leaves the remaining rows misaligned, and the
    estimator then reads the wrong values without any error. The unit records
    what has to be removed together.

    Attributes:
        ROW: One sample. Given-data methods use this: any ``(X, Y)`` pair
            stands alone, so one row can leave without disturbing the others.
        SALTELLI_GROUP: One base point and all of its cross-matrix
            evaluations, ``D + 2`` or ``2D + 2`` contiguous rows. A partial
            group would corrupt the A/B/AB/BA split.
        TRAJECTORY: One Morris trajectory, ``D + 1`` contiguous rows. The
            elementary effects are differences between neighbours inside the
            trajectory, so a gap in the middle invents an effect that was
            never measured.
        BASE_POINT: One Kucherenko base point, which appears once in each of
            the ``2D + 1`` conditional blocks rather than contiguously.
        CURVE: One eFAST search curve. Listed for completeness only: a curve
            is an ordered sweep read by a discrete Fourier transform, so it
            cannot be dropped at all, and neither can a point inside it.
    """

    ROW = "row"
    SALTELLI_GROUP = "Saltelli group"
    TRAJECTORY = "trajectory"
    BASE_POINT = "base point"
    CURVE = "search curve"

    @property
    def plural(self) -> str:
        """Return the plural noun, for messages that count units.

        Spelled out rather than derived, because "trajectory" does not take a
        bare "s" and a message that says "trajectorys" reads as a bug in the
        library.
        """
        return _PLURALS[self]


_PLURALS: dict[InvalidUnit, str] = {
    InvalidUnit.ROW: "rows",
    InvalidUnit.SALTELLI_GROUP: "Saltelli groups",
    InvalidUnit.TRAJECTORY: "trajectories",
    InvalidUnit.BASE_POINT: "base points",
    InvalidUnit.CURVE: "search curves",
}


@dataclass(frozen=True)
class InvalidReport:
    """What the non-finite check found, and what was done about it.

    Every analysis result carries one of these, whichever policy ran. A report
    with ``n_invalid == 0`` means the check ran and found nothing, which is not
    the same as the check never running.

    The positions are the point of the report. A count tells a user that
    something failed; the indices tell them which model evaluation to look at.
    Indices always refer to the array **as it was passed in**, before anything
    was removed, because that is the numbering the caller can act on.

    Attributes:
        policy: The policy that ran: ``"raise"``, ``"propagate"``,
            ``"drop"`` or ``"none"``. A report with ``policy="raise"`` only
            exists when nothing was found, since otherwise the call raised. A
            report with ``policy="none"`` means the check never ran: it is
            always clean, because nothing was looked at.
        unit: The block of data one bad value invalidates. See
            :class:`InvalidUnit`.
        n_units: Total number of units in the sample before any removal.
        n_invalid: Number of units that held at least one non-finite value.
        unit_indices: Positions of those units, in the original numbering.
        row_indices: Every row the affected units occupy, in the caller's
            numbering. For ``InvalidUnit.ROW`` this equals ``unit_indices``.
            For a grouped design it is larger, because one bad value condemns
            its whole group. These are the rows that ``"drop"`` removes.
        bad_row_indices: The rows that actually held a non-finite value, in
            the caller's numbering. Always a subset of ``row_indices``, and
            usually a much smaller one: an eFAST search curve is one unit of
            256 rows, and this says which of them failed. These are the model
            runs to investigate.
        sources: Which arrays held non-finite values. Empty when nothing was
            found. The names are usually ``"X"`` and ``"Y"``, but a caller
            that checks something else can rename them; see the
            ``source_names`` argument of :func:`check_invalid`.
    """

    policy: OnInvalid
    unit: InvalidUnit
    n_units: int
    n_invalid: int
    unit_indices: tuple[int, ...]
    row_indices: tuple[int, ...]
    bad_row_indices: tuple[int, ...]
    sources: tuple[str, ...]

    @property
    def n_kept(self) -> int:
        """Return the number of units that survived the check."""
        return self.n_units - self.n_invalid

    @property
    def any_invalid(self) -> bool:
        """Return whether the check found anything."""
        return self.n_invalid > 0

    def __repr__(self) -> str:
        """Summarize the finding, listing positions only when there are few."""
        if not self.any_invalid:
            return f"InvalidReport(clean, n_units={self.n_units}, policy={self.policy!r})"
        return (
            f"InvalidReport(policy={self.policy!r}, unit={self.unit.value!r}, "
            f"n_invalid={self.n_invalid}/{self.n_units}, "
            f"sources={self.sources}, "
            f"unit_indices={_abbreviate(self.unit_indices)})"
        )


def _abbreviate(values: tuple[int, ...]) -> str:
    """Render index positions, shortening a long list to its first few."""
    if len(values) <= _MAX_LISTED:
        return str(list(values))
    head = ", ".join(str(v) for v in values[:_MAX_LISTED])
    return f"[{head}, ... {len(values) - _MAX_LISTED} more]"


def _clean_report(unit: InvalidUnit, n_units: int, policy: OnInvalid) -> InvalidReport:
    """Build the report for a sample with nothing wrong in it."""
    return InvalidReport(
        policy=policy,
        unit=unit,
        n_units=n_units,
        n_invalid=0,
        unit_indices=(),
        row_indices=(),
        bad_row_indices=(),
        sources=(),
    )


def resolve_policy(
    on_invalid: object,
    *,
    method: str,
    unit: InvalidUnit,
    allow_drop: bool = True,
) -> OnInvalid:
    """Validate an ``on_invalid`` argument for one method.

    Args:
        on_invalid: The value the caller passed.
        method: Fully qualified name of the calling function, used in
            messages, for example ``"jaxgsa.efast.analyze"``.
        unit: The block of data one bad value invalidates. Named in the
            message when ``"drop"`` is refused, so the refusal says what would
            have had to be removed.
        allow_drop: Whether ``"drop"`` is meaningful for this method. Pass
            ``False`` where removing data would silently change what the
            estimator computes rather than merely shrinking the sample.

    Returns:
        The validated policy.

    Raises:
        ValueError: If the value is not one of the four policies, or if it is
            ``"drop"`` where dropping is not defined.
    """
    if not isinstance(on_invalid, str) or on_invalid not in _POLICIES:
        listed = ", ".join(repr(p) for p in _POLICIES)
        raise ValueError(f"{method}: on_invalid must be one of {listed}, got {on_invalid!r}")
    # Re-state the value as a literal. The membership test above already
    # guarantees it, but a type checker cannot narrow a `str` through `in`
    # against a tuple, and one cast reads better than three return branches.
    policy = cast(OnInvalid, on_invalid)
    if policy == "drop" and not allow_drop:
        raise ValueError(
            f"{method}: on_invalid='drop' is not available for this method, because "
            f"its design cannot lose a {unit.value} and stay valid. Removing one "
            "would change what the estimator computes, without any error. Use "
            "on_invalid='raise' to refuse the sample, or on_invalid='propagate' to "
            "let the non-finite value reach the indices. To drop failed runs, "
            "re-run the model on a repaired design instead."
        )
    return policy


def _finite_by_unit(
    array: Array | None,
    *,
    n_units: int,
    unit_of_row: npt.NDArray[np.intp] | None,
    unit_stride: int | None = None,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_] | None]:
    """Return per-unit and per-row masks, False where a bad value sits.

    Args:
        array: Array whose leading axis is the sample axis, or ``None`` to
            report everything finite.
        n_units: Number of units to report on.
        unit_of_row: For each row of ``array``, the unit it belongs to. Pass
            ``None`` when rows and units are the same thing.
        unit_stride: Rows per unit, when the caller knows the map is the
            equal-contiguous-blocks pattern ``repeat(arange(n_units),
            unit_stride)``. Enables a device-side collapse that skips the
            per-row host verdicts; see the fast path below. ``None`` (the
            default) always takes the generic weighted-``bincount`` path.

    Returns:
        A tuple ``(unit_ok, row_ok)``. ``unit_ok`` has shape ``(n_units,)`` and
        is True where the unit is clean. ``row_ok`` has one entry per row of
        ``array`` and is True where that row is clean, or is ``None`` when
        there was no array to check **or when the contiguous fast path ran**
        (the caller needs row verdicts only for the invalid report, which is
        the rare path, and re-derives them there). The row verdict is kept
        because a unit can span hundreds of rows and only one of them failed;
        the report names both.
    """
    if array is None:
        return np.ones(n_units, dtype=bool), None

    # A row is bad if any element in it is non-finite. `isfinite` rejects both
    # NaN and the infinities, which is what we want: an infinite output breaks
    # a variance just as thoroughly as a NaN does.
    #
    # Reduce on whichever device the array already sits on, and bring back only
    # the one bit per row. Pulling the whole array to the host would move
    # `n_rows * T * K` floats to decide `n_rows` booleans, and for a Saltelli
    # design that array is the expanded one, not the caller's.
    #
    # The row scan is computed exactly once, inside whichever branch uses it:
    # both scans feed the same verdict, and eager JAX executes every op it
    # traces, so computing the row mask before the branch would run the
    # million-row scan twice on designs that take the fast path below.
    if unit_of_row is None:
        row_ok = np.asarray(jnp.isfinite(array.reshape(array.shape[0], -1)).all(axis=1))
        return row_ok, row_ok

    if unit_stride is not None and _is_contiguous_blocks(unit_of_row, n_units, unit_stride):
        # The units are equal contiguous blocks of rows, so the verdict can
        # collapse straight to one boolean per unit on the device. Only
        # ``n_units`` booleans cross to the host, instead of one per row:
        # the alternative, ``bincount``-ing a float weight per row, is the
        # single most expensive host op in the entry path of a large
        # design (measured ~5 ms over a million rows, vs under a
        # millisecond for this path at scalar width). The collapse is two
        # reductions: row verdicts over the trailing (contiguous) axis,
        # then the block verdicts over the short unit axis. Collapsing
        # straight from the raw array -- ``(n_units, unit_stride,
        # trailing).all(axis=(1, 2))`` -- reduces a strided middle axis
        # instead, which XLA vectorises poorly and measured 8-10x slower
        # on wide outputs. The per-row verdict is returned as ``None`` on
        # purpose: the caller needs it only for the invalid report, which
        # is the rare path, and re-derives it there.
        row_ok_dev = jnp.all(jnp.isfinite(array.reshape(array.shape[0], -1)), axis=1)
        unit_ok = np.asarray(jnp.all(row_ok_dev.reshape(n_units, unit_stride), axis=1))
        return unit_ok, None

    row_ok = np.asarray(jnp.isfinite(array.reshape(array.shape[0], -1)).all(axis=1))
    # Scatter the row verdicts onto their units: a unit is clean only when
    # every one of its rows is. `bincount` counts the bad rows per unit in one
    # pass, which beats `np.logical_and.at` by more than a factor of ten, and
    # works for any row map -- contiguous blocks and strided ones alike.
    n_bad_per_unit = np.bincount(unit_of_row, weights=~row_ok, minlength=n_units)
    return n_bad_per_unit == 0, row_ok


def _is_contiguous_blocks(
    unit_of_row: npt.NDArray[np.intp], n_units: int, unit_stride: int
) -> bool:
    """Report whether ``unit_of_row`` groups units into equal contiguous blocks.

    The fast path of :func:`_finite_by_unit` relies on the row map being
    ``np.repeat(np.arange(n_units), unit_stride)``: that is what lets the
    per-unit verdict be computed with a device reshape-reduce instead of a
    host weighted ``bincount`` over every row. The check is exact (no
    tolerance), so a caller that passes a wrong ``unit_stride`` gets the
    correct slow path, never a wrong mask.

    Args:
        unit_of_row: Row-to-unit map, one entry per row.
        n_units: Number of units.
        unit_stride: Rows per unit the caller claims.

    Returns:
        True only when the map is exactly the equal-contiguous-blocks pattern.
    """
    if unit_of_row.shape[0] != n_units * unit_stride:
        return False
    expected = np.repeat(np.arange(n_units), unit_stride)
    return bool(np.array_equal(unit_of_row, expected))


def _unit_of_row_for_policy(
    on_invalid: object, build: Callable[[], npt.NDArray[np.intp]]
) -> npt.NDArray[np.intp] | None:
    """Build the row map only where the non-finite check will read it.

    Under ``on_invalid='none'`` :func:`check_invalid` returns before it ever
    touches the map (a per-row map over the expanded design is exactly the
    scan the policy skips, and building it on the host costs ~0.3-0.6 ms at
    million-row scales), so the call sites pass ``None`` there instead of
    paying for a dead array. The three design-based methods each build a
    different map (contiguous blocks for sobol and morris, interleaved
    base points for kucherenko), so the map itself stays at the call site;
    only the skip decision is shared.

    Args:
        on_invalid: The caller's unvalidated policy value.
        build: How to build the map, called only when a check will read it.

    Returns:
        The built map, or ``None`` under ``on_invalid='none'``.
    """
    if on_invalid == "none":
        return None
    return build()


def check_invalid(
    *,
    policy: OnInvalid,
    method: str,
    unit: InvalidUnit,
    n_units: int,
    Y: npt.ArrayLike | None = None,
    X: npt.ArrayLike | None = None,
    extras: Iterable[npt.ArrayLike] = (),
    unit_of_row: npt.NDArray[np.intp] | None = None,
    unit_stride: int | None = None,
    row_labels: npt.NDArray[np.intp] | None = None,
    min_kept: int = 1,
    source_names: tuple[str, str] = ("X", "Y"),
) -> tuple[npt.NDArray[np.bool_], InvalidReport]:
    """Apply the non-finite policy and describe what was found.

    This function decides and reports. It does not remove anything: it returns
    the mask of units to keep, and the caller applies it to whatever arrays and
    bookkeeping it owns. Morris, for one, has to re-index its elementary-effect
    tables alongside the outputs, and only Morris knows that.

    Args:
        policy: The validated policy, from :func:`resolve_policy`. Under
            ``"none"`` the scan does not run at all: the verdict is that every
            unit is clean, the report is the clean report, and the caller
            applies no mask. The shape consistency check on the companion
            arrays still runs, because a misnumbered array would silently
            corrupt an estimator even when nothing is scanned.
        method: Fully qualified name of the calling function, for messages.
        unit: The block of data one bad value invalidates.
        n_units: Number of units in the sample.
        Y: Model output, or ``None`` to skip it. Its leading axis is the
            sample axis.
        X: Input matrix, or ``None`` to skip it. Checked jointly with ``Y``,
            so a bad input takes its own output with it.
        extras: Further model-side arrays to check alongside ``Y``. Their
            leading axis must match ``Y``; only per-row finite flags are
            combined, so large companion arrays do not get concatenated.
        unit_of_row: For each row, the unit it belongs to. Pass ``None`` when
            one row is one unit.
        unit_stride: Rows per unit, when the caller knows ``unit_of_row`` is
            ``np.repeat(np.arange(n_units), unit_stride)``, i.e. units are
            equal contiguous blocks of rows. Lets the per-unit verdict be
            computed on the device, skipping a host ``bincount`` over every
            row -- the dominant fixed cost of the entry path on designs with
            millions of expanded rows. The pattern is verified exactly, so a
            wrong stride falls back to the generic path, never a wrong mask.
            ``None`` (the default) always takes the generic path.
        row_labels: For each row of the arrays given here, the row of the
            **caller's** array it came from. Pass it whenever the two differ,
            so the report points at rows the caller can actually find. Sobol
            and Morris are the cases: both analyze an expanded design, but the
            user evaluated one output per unique run, and "row 18 of 24" means
            nothing to someone holding 16 rows.
        min_kept: Fewest surviving units the caller can still work with.
            Enforced under ``"drop"`` only, because an estimate from too little
            data is not an estimate. ``"raise"`` has already raised by then,
            and ``"propagate"`` removes nothing, so what it computes from is
            the whole sample.
        source_names: What to call the two arrays in messages and in
            :attr:`InvalidReport.sources`, as ``(name_of_X, name_of_Y)``.
            Override it where the defaults would mislead. DGSM is the case
            that needed it: on its autodiff path the array in the ``Y`` slot
            carries the model output **and** its derivative, so a report
            saying ``"Y"`` would send a user to look at an output array that
            is entirely finite.

    Returns:
        A tuple ``(keep, report)``. ``keep`` is a boolean array of shape
        ``(n_units,)``; under ``"propagate"`` it is all True, so a caller can
        apply it unconditionally.

    Raises:
        ValueError: Under ``"raise"`` when anything was found, and under
            ``"drop"`` when fewer than ``min_kept`` units would remain.
    """
    y_array = _as_array(Y)
    x_array = _as_array(X)
    extra_arrays = tuple(_as_array(array) for array in extras)
    _check_extra_row_counts(y_array, x_array, extra_arrays, unit_of_row=unit_of_row)

    if policy == "none":
        # No scan: the caller has asserted the data is clean. The keep mask is
        # all-True and the report is the clean one, so the estimator's book-
        # keeping sees exactly what it sees under "propagate" on a clean
        # sample, without a single pass over the expanded rows. The contract
        # the other policies hold -- keep + report decide the rest -- is
        # intact, so the callers below need no "none" branches of their own.
        return np.ones(n_units, dtype=bool), _clean_report(unit, n_units, policy)

    y_ok, y_rows = _finite_by_unit(
        y_array, n_units=n_units, unit_of_row=unit_of_row, unit_stride=unit_stride
    )
    for extra_array in extra_arrays:
        extra_ok, extra_rows = _finite_by_unit(
            extra_array,
            n_units=n_units,
            unit_of_row=unit_of_row,
            unit_stride=unit_stride,
        )
        y_ok = y_ok & extra_ok
        if y_rows is None:
            y_rows = extra_rows
        elif extra_rows is not None:
            y_rows = y_rows & extra_rows
    x_ok, x_rows = _finite_by_unit(
        x_array, n_units=n_units, unit_of_row=unit_of_row, unit_stride=unit_stride
    )
    clean = y_ok & x_ok

    n_invalid = int((~clean).sum())
    if n_invalid == 0:
        return np.ones(n_units, dtype=bool), _clean_report(unit, n_units, policy)

    # The report names the bad rows, which the fast path deliberately left on
    # the device. Re-derive the per-row verdicts only here, on the rare path
    # that actually uses them; the fast path is then no slower than the
    # generic one when something is wrong.
    if y_rows is None:
        _, y_rows = _finite_by_unit(y_array, n_units=n_units, unit_of_row=unit_of_row)
    if x_rows is None:
        _, x_rows = _finite_by_unit(x_array, n_units=n_units, unit_of_row=unit_of_row)

    x_name, y_name = source_names
    sources = tuple(name for name, ok in ((x_name, x_ok), (y_name, y_ok)) if bool((~ok).any()))
    unit_indices = tuple(int(i) for i in np.flatnonzero(~clean))
    report = InvalidReport(
        policy=policy,
        unit=unit,
        n_units=n_units,
        n_invalid=n_invalid,
        unit_indices=unit_indices,
        row_indices=_rows_of_units(unit_indices, unit_of_row, n_units, row_labels),
        bad_row_indices=_bad_rows(y_rows, x_rows, row_labels),
        sources=sources,
    )

    if policy == "raise":
        raise ValueError(_raise_message(method, report))

    n_kept = n_units - n_invalid
    if policy == "drop" and n_kept < min_kept:
        raise ValueError(
            f"{method}: every usable {unit.value} was removed. "
            f"{n_invalid} of {n_units} {unit.plural} hold a non-finite value in "
            f"{_and_list(sources)}, leaving {n_kept}, and this method needs at "
            f"least {min_kept}. {_positions_line(report)}"
        )

    if policy == "propagate":
        warnings.warn(
            f"{method}: {n_invalid} of {n_units} {unit.plural} hold a non-finite "
            f"value in {_and_list(sources)}. on_invalid='propagate', so the value "
            "reaches the indices and they will be non-finite too. "
            f"{_positions_line(report)}",
            category=JaxgsaWarning,
            skip_file_prefixes=(_PACKAGE_DIR,),
        )
        return np.ones(n_units, dtype=bool), report

    warnings.warn(
        f"{method}: dropped {n_invalid} of {n_units} {unit.plural} that hold a "
        f"non-finite value in {_and_list(sources)}. The indices are computed from "
        f"the remaining {n_kept}. {_positions_line(report)}",
        category=JaxgsaWarning,
        skip_file_prefixes=(_PACKAGE_DIR,),
    )
    if n_kept < _LOW_SURVIVOR_WARN:
        warnings.warn(
            f"{method}: only {n_kept} {unit.plural} remain after dropping. "
            "Sensitivity indices from a sample this small are not reliable.",
            category=JaxgsaWarning,
            skip_file_prefixes=(_PACKAGE_DIR,),
        )
    return clean, report


def _as_array(array: npt.ArrayLike | None) -> Array | None:
    """Present an input as a JAX array, or pass ``None`` through.

    The decision this feeds is host-side by necessity: which units survive
    decides an array shape, and a traced shape cannot depend on a value. Only
    the per-row verdict crosses back to the host, though, never the data. See
    :func:`_finite_by_unit`.
    """
    if array is None:
        return None
    return jnp.asarray(array)


def _check_extra_row_counts(
    Y: Array | None,
    X: Array | None,
    extras: tuple[Array | None, ...],
    *,
    unit_of_row: npt.NDArray[np.intp] | None,
) -> None:
    """Check that companion arrays have the same sample axis as the data."""
    extra_arrays = tuple(array for array in extras if array is not None)
    if not extra_arrays:
        return

    for array in extra_arrays:
        if array.ndim == 0:
            raise ValueError("arrays checked for invalid values must have a sample axis")

    reference = Y if Y is not None else X
    expected = (
        len(unit_of_row)
        if unit_of_row is not None
        else int(reference.shape[0])
        if reference is not None
        else None
    )
    if expected is None:
        return
    for array in extra_arrays:
        if int(array.shape[0]) != expected:
            raise ValueError(
                "arrays checked for invalid values must have the same number of sample rows"
            )


def _rows_of_units(
    unit_indices: tuple[int, ...],
    unit_of_row: npt.NDArray[np.intp] | None,
    n_units: int,
    row_labels: npt.NDArray[np.intp] | None = None,
) -> tuple[int, ...]:
    """Return the rows the given units occupy, in the caller's numbering."""
    if unit_of_row is None:
        rows = np.asarray(unit_indices, dtype=np.intp)
    else:
        wanted = np.zeros(n_units, dtype=bool)
        wanted[list(unit_indices)] = True
        rows = np.flatnonzero(wanted[unit_of_row])
    if row_labels is not None:
        # A design that repeats a run gives several internal rows the same
        # caller row, so translating can produce duplicates. Report each
        # caller row once, in order.
        rows = np.unique(np.asarray(row_labels)[rows])
    return tuple(int(i) for i in rows)


def _bad_rows(
    y_rows: npt.NDArray[np.bool_] | None,
    x_rows: npt.NDArray[np.bool_] | None,
    row_labels: npt.NDArray[np.intp] | None,
) -> tuple[int, ...]:
    """Return the rows that actually held a bad value, in caller numbering.

    Distinct from the rows of the condemned units. A Saltelli group is 22 rows
    and an eFAST curve is hundreds; naming all of them tells a user nothing
    about which model run to look at.
    """
    verdicts = [v for v in (y_rows, x_rows) if v is not None]
    if not verdicts:
        return ()
    bad = np.zeros(verdicts[0].shape, dtype=bool)
    for verdict in verdicts:
        bad |= ~verdict
    rows = np.flatnonzero(bad)
    if row_labels is not None:
        rows = np.unique(np.asarray(row_labels)[rows])
    return tuple(int(i) for i in rows)


def _and_list(names: tuple[str, ...]) -> str:
    """Join array names for a message: ``X``, ``Y``, or ``X and Y``."""
    return " and ".join(names) if names else "the sample"


def _positions_line(report: InvalidReport) -> str:
    """Describe where the bad values sit.

    For a grouped design the failing rows and the condemned rows are different
    facts, and the failing ones are what a user acts on, so they come first.
    """
    if report.unit is InvalidUnit.ROW:
        return f"Affected rows: {_abbreviate(report.row_indices)}."
    return (
        f"Non-finite rows: {_abbreviate(report.bad_row_indices)}. "
        f"They condemn {report.unit.plural} {_abbreviate(report.unit_indices)}, "
        f"which covers {len(report.row_indices)} rows."
    )


def _raise_message(method: str, report: InvalidReport) -> str:
    """Build the message for the default policy."""
    return (
        f"{method}: {report.n_invalid} of {report.n_units} {report.unit.plural} "
        f"hold a non-finite value (NaN or inf) in {_and_list(report.sources)}. "
        f"{_positions_line(report)} "
        "An index computed from the rest is a different quantity from the one "
        "you asked for, so this raises by default. Investigate those runs, or "
        "pass on_invalid='drop' to analyze the remainder, or "
        "on_invalid='propagate' to let the value reach the indices."
    )
