"""Irregularly-sampled output support: coercion, per-channel analysis, result assembly.

Nothing in this module is public API. See :mod:`jaxgsa._core`.

The irregular input form lets each output channel live on its own (possibly
non-uniform) time grid. :func:`analyze_irregular` routes one analysis call to
the calling method's regular ``analyze`` once per channel, each on a narrowed
sub-problem with ``Y = (N, T_k, 1)``, and assembles the per-channel results
into an :class:`IrregularResult`.

A per-channel analysis is a fully regular analysis: the same kernels, the
same validation, the same bootstrap. The irregular path therefore costs
exactly the sum of its per-channel regular analyses plus a host-side
coercion pass, which is the performance guarantee the shared-grid protocol
relies on.

Ragged ``Y`` is always handled by bucketing: each channel runs on its own grid
through the existing kernels. There is no padding, no fabricated values, and
no public mode switch. A masked/padded path would be a different estimator,
not an optimization of this one.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core import verbose as _verbose

if TYPE_CHECKING:
    from jaxgsa.problem import Problem


IrregularChannel = tuple[Any, Any] | list[Any]
"""One ``(times, values)`` pair in a ragged output container."""

IrregularY = Sequence[IrregularChannel] | Mapping[str, IrregularChannel]
"""The list, tuple, or name-keyed mapping accepted as irregular ``Y``."""


def _is_channel_pair(item: Any) -> bool:
    """Report whether one element looks like a ``(times, values)`` pair.

    A pair is two sequences of rank at least one: ``times`` is 1-D and
    ``values`` is 1-D or 2-D. A two-element tuple of scalars (which would
    otherwise read as a 2-D array of outputs) is deliberately not a pair, so
    the existing ``(N, K)`` array path stays unambiguous.

    Args:
        item: One element of a ragged ``Y`` container.

    Returns:
        ``True`` when ``item`` is a 2-sequence whose two entries are both
        rank-one-or-more array-like values.
    """
    if not isinstance(item, (tuple, list)) or len(item) != 2:
        return False
    return np.ndim(item[0]) >= 1 and np.ndim(item[1]) >= 1


def _is_irregular_y(Y: Any) -> bool:
    """Report whether ``Y`` is the ragged per-channel input form.

    The ragged form is a ``list``/``tuple`` of ``(times, values)`` pairs, or
    a ``dict`` mapping output name to such a pair. Anything else — a plain
    array, a list of scalars, a list of arrays — is the regular input and
    reads ``False`` here.

    Args:
        Y: The caller's output argument, before any ``jnp.asarray``.

    Returns:
        ``True`` when ``Y`` is a non-empty ragged container of channel pairs.
    """
    if isinstance(Y, dict):
        return bool(Y) and all(_is_channel_pair(value) for value in Y.values())
    if isinstance(Y, (list, tuple)):
        return bool(Y) and all(_is_channel_pair(item) for item in Y)
    return False


def _fwd_kwargs(local: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """Copy the named keyword arguments out of an ``analyze`` frame.

    Called at the top of a public ``analyze``, where ``local`` is
    ``locals()`` and therefore holds only the function's parameters. ``names``
    lists the method's own keyword-only parameters (never ``verbose``, which
    the engine controls). The copy is what lets the engine
    call the method's regular ``analyze`` per channel with every caller
    setting intact.

    Args:
        local: A mapping of the calling frame's locals, normally ``locals()``.
        names: Parameter names to copy.

    Returns:
        ``{name: value}`` for every name present in ``local``.
    """
    return {name: local[name] for name in names if name in local}


def _coerce_irregular(
    Y: Any,
    problem: "Problem",
    n_expected: int,
) -> list[tuple[str, Array, Array]]:
    """Validate the ragged input and normalize it to per-channel tensors.

    Accepts a ``dict`` of ``name -> (times, values)`` or a ``list``/``tuple``
    of ``(times, values)`` pairs. Each channel's ``times`` is sorted
    ascending (a stable sort that carries ``values`` along), ``values`` is
    kept sample-first ``(N, T_k)``, and the channel count and the sample
    count are checked against the problem and the caller's ``n_expected``.

    Args:
        Y: The ragged output argument, already recognized by
            :func:`_is_irregular_y`.
        problem: Problem definition, whose ``output_names`` (when declared)
            fix the channel labels and must match the channels.
        n_expected: Sample count every channel must have, normally
            ``X.shape[0]`` or a design's ``n_runs``.

    Returns:
        A list of ``(name, times, values)`` tuples in output order, with
        ``times`` of shape ``(T_k,)`` ascending and ``values`` of shape
        ``(N, T_k)``.

    Raises:
        ValueError: If the channel count does not match ``output_names``,
            a dict's keys disagree with them, a pair is malformed, times
            are not finite or not unique, or a channel's sample count
            differs from ``n_expected``.
    """
    if isinstance(Y, dict):
        declared = problem.output_names
        if declared is not None:
            keys = list(Y)
            if set(keys) != set(declared):
                raise ValueError(
                    "irregular Y dict keys must match problem.output_names, got "
                    f"{keys} vs {list(declared)}"
                )
            # Output order follows the declared names, not dict insertion.
            channels = [(name, Y[name]) for name in declared]
        else:
            channels = [(str(name), pair) for name, pair in Y.items()]
    else:
        pairs = list(Y)
        if problem.output_names is not None:
            if len(pairs) != len(problem.output_names):
                raise ValueError(
                    f"irregular Y has {len(pairs)} channel(s) but problem.output_names "
                    f"declares {len(problem.output_names)}"
                )
            channels = list(zip(problem.output_names, pairs, strict=True))
        else:
            channels = [(f"y{i}", pair) for i, pair in enumerate(pairs)]

    names = [name for name, _ in channels]
    if len(set(names)) != len(names):
        raise ValueError(f"irregular Y output names must be unique, got {names}")

    normalized: list[tuple[str, Array, Array]] = []
    for name, pair in channels:
        if not _is_channel_pair(pair):
            raise ValueError(
                f"output channel {name!r}: expected a (times, values) pair, "
                f"got {type(pair).__name__}"
            )
        times_raw, values_raw = pair[0], pair[1]
        times = np.asarray(times_raw)
        values = np.asarray(values_raw)
        if times.ndim != 1:
            raise ValueError(f"output channel {name!r}: times must be 1-D, got ndim={times.ndim}")
        if times.shape[0] == 0:
            raise ValueError(f"output channel {name!r}: times must have at least one entry")
        if not np.all(np.isfinite(times)):
            raise ValueError(f"output channel {name!r}: times must be finite")
        if values.ndim not in (1, 2):
            raise ValueError(
                f"output channel {name!r}: values must be (N,) or (N, T_k), got ndim={values.ndim}"
            )
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[0] != n_expected:
            raise ValueError(
                f"output channel {name!r}: values has {values.shape[0]} sample rows but "
                f"{n_expected} were expected"
            )
        if times.shape[0] != values.shape[1]:
            raise ValueError(
                f"output channel {name!r}: times has {times.shape[0]} points but values "
                f"has {values.shape[1]} columns"
            )
        # Sort by time, carrying values along, so the analysis and the result
        # coordinates agree with the caller's intended time order. The sort
        # and the duplicate check run on the dtype jnp will actually store,
        # so two float64 times that would collide after a float32 downcast
        # are caught instead of silently merging into one coordinate.
        order = np.argsort(times, kind="stable")
        times = jnp.asarray(times[order])
        values = jnp.asarray(values[:, order])
        if np.any(np.diff(np.asarray(times)) == 0):
            raise ValueError(f"output channel {name!r}: times must not repeat")
        normalized.append((name, times, values))
    return normalized


@dataclass(frozen=True, repr=False)
class IrregularResult:
    """Per-channel sensitivity results for irregularly sampled outputs.

    Returned by every ``analyze`` call whose ``Y`` is ragged. One regular
    result per output channel, each computed on its own grid through the
    method's ordinary kernels. :meth:`to_dataset` merges the channels into a
    single :class:`xarray.Dataset` with one variable per channel, each on its
    own ``time`` coordinate.

    Attributes:
        channels: Mapping of output name to that channel's regular result.
        times: Mapping of output name to that channel's ``(T_k,)`` grid.
        problem: The problem the analysis ran on, with the full
            ``output_names``.
    """

    channels: Mapping[str, Any]
    times: Mapping[str, Array]
    problem: "Problem"

    def to_dataset(self) -> xr.Dataset:
        """Merge the per-channel results into one labeled Dataset.

        Each channel's result is exported with its own ``time`` coordinate,
        then the singleton ``output`` axis is dropped and every variable and
        the ``time`` dimension are renamed under the channel name. xarray
        supports per-variable dimensions, so the merged dataset keeps each
        channel's grid exactly — no padding, no fabricated time points.

        Returns:
            An :class:`xarray.Dataset` with one ``<channel>_<field>`` variable
            per field per channel, on ``time_<channel>`` and ``param`` axes.
        """
        datasets: list[xr.Dataset] = []
        first_attrs: dict[str, Any] | None = None
        for name, res in self.channels.items():
            ds = res.to_dataset(time_coords=np.asarray(self.times[name]))
            if "output" in ds.dims:
                ds = ds.squeeze("output", drop=True)
            renames = {var: f"{name}_{var}" for var in ds.data_vars}
            if "time" in ds.dims:
                renames["time"] = f"time_{name}"
            if "term" in ds.dims:
                renames["term"] = f"term_{name}"
            ds = ds.rename(renames)
            if first_attrs is None:
                first_attrs = dict(ds.attrs)
            ds.attrs = {}
            datasets.append(ds)
        merged = xr.merge(datasets)
        if first_attrs is not None:
            merged.attrs = first_attrs
        return merged

    def __repr__(self) -> str:
        """Summarize the channels, one regular result per line."""
        lines = [f"IrregularResult({len(self.channels)} output channel(s))"]
        for name, res in self.channels.items():
            lines.append(f"  {name}: {res!r}")
        return "\n".join(lines)


def analyze_irregular(
    analyze_fn: Callable,
    *,
    channel_data: Any,
    problem: "Problem",
    Y: Any,
    n_expected: int,
    design_based: bool,
    verbose: bool,
    kwargs: Mapping[str, Any],
) -> IrregularResult:
    """Run one method's analysis per output channel and assemble the results.

    The engine behind ragged ``Y`` on every public ``analyze``.
    It validates the ragged input, then calls ``analyze_fn`` once per channel
    on a narrowed sub-problem with ``Y = (N, T_k, 1)`` — a fully regular
    analysis, so every channel's result carries its own bootstrap, CI and
    method-specific behavior for free.

    Args:
        analyze_fn: The calling method's public ``analyze``. Called
            recursively per channel with a regular 3-D ``Y``, so it never
            re-enters this engine.
        channel_data: The method's first positional argument — ``X`` for a
            given-data method, the sampling result for a design-based one.
        problem: The problem the caller passed, with the full
            ``output_names``.
        Y: The ragged output argument.
        n_expected: Sample count every channel must have.
        design_based: Whether ``channel_data`` is a sampling result (whose
            ``problem`` is narrowed via :func:`dataclasses.replace`) rather
            than the input matrix.
        verbose: Whether to print a per-channel summary after the work.
        kwargs: The method's own keyword arguments, forwarded unchanged to
            each per-channel call (``verbose`` is forced ``False`` there).

    Returns:
        The assembled :class:`IrregularResult`.

    Raises:
        ValueError: If the ragged input fails :func:`_coerce_irregular`.
    """
    if not _is_irregular_y(Y):
        raise ValueError(
            "ragged Y must be a list of (times, values) pairs or a dict keyed "
            "by output name, not a "
            f"{type(Y).__name__}. Pass a regular rectangular array instead."
        )

    channels = _coerce_irregular(Y, problem, n_expected)
    results: dict[str, Any] = {}
    times: dict[str, Array] = {}
    for i, (name, t_k, values) in enumerate(channels):
        sub_problem = problem.with_output_names((name,))
        y3 = values[:, :, None]
        # Fold the caller's key per channel so the bootstrap resamples (and
        # the hsic permutations, vkoga latent draws) are independent across
        # channels. Forwarding the same key verbatim would give every channel
        # bit-identical resamples, correlating their confidence intervals.
        channel_kwargs = dict(kwargs)
        key = kwargs.get("key")
        if key is not None:
            channel_kwargs["key"] = jax.random.fold_in(key, i)
        if design_based:
            channel = dataclasses.replace(channel_data, problem=sub_problem)
            results[name] = analyze_fn(channel, y3, verbose=False, **channel_kwargs)
        else:
            results[name] = analyze_fn(
                sub_problem, channel_data, y3, verbose=False, **channel_kwargs
            )
        times[name] = t_k

    result = IrregularResult(channels=results, times=times, problem=problem)
    if verbose:
        _verbose.emit(f"jaxgsa irregular analysis: {len(results)} output channel(s)")
        for name, res in results.items():
            _verbose.emit(f"  {name}: {res!r}")
    return result
