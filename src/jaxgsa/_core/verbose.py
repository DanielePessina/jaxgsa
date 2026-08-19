"""One voice for every ``verbose=True`` entry point.

Nothing in this module is public API. See :mod:`jaxgsa._core`.

Every ``analyze()`` and every ``sample()`` takes ``verbose: bool = True``.
With it on, ``analyze()`` prints three sections — the problem and the data,
the timings, and a top-k results summary — and ``sample()`` prints one line
that narrates the design. This module is the one place that formats those
sections, so seventeen entry points cannot drift into seventeen styles.

The rules the formatting follows:

* Plain :func:`print` to stdout, through the single seam :func:`emit`. No
  logging module, no color. The test suite silences itself by replacing
  ``emit`` with a no-op.
* Numbers get four significant digits.
* The summary never computes anything the analysis did not already compute.
  Everything printed is read off values the caller is already holding.
* Nothing here runs under ``jax.jit``. The entry points print on the host,
  after the device work, and the pure ``indices()`` cores stay silent.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from jaxgsa._core.invalid import InvalidReport
    from jaxgsa.problem import Problem

__all__ = [
    "analysis_summary",
    "emit",
    "format_value",
    "sample_summary",
    "stop",
    "tic",
]

# How many parameters the results section ranks. min(D, _TOP_K) are shown.
_TOP_K = 5

# How many parameter names the problem line lists before it abbreviates.
_MAX_NAMES = 8

_INDENT = "  "


def emit(text: str) -> None:
    """Print one line of verbose output.

    The single seam every verbose line goes through. The test suite replaces
    this function with a no-op so a thousand analyses do not drown the test
    log, and ``tests/test_verbose.py`` restores it to assert on the output.

    Args:
        text: The line, without a trailing newline.
    """
    print(text)


def tic() -> float:
    """Start a wall-clock measurement.

    Returns:
        The current :func:`time.perf_counter` reading.
    """
    return time.perf_counter()


def stop(t0: float, *arrays: Any) -> float:
    """Finish a wall-clock measurement, after the device work is really done.

    JAX dispatches asynchronously, so a clock stopped without synchronizing
    would time the dispatch, not the computation. Every argument that knows
    ``block_until_ready`` is blocked on before the clock is read; host-side
    values (NumPy arrays, ``None``) pass through untouched.

    Args:
        t0: The reading :func:`tic` returned.
        *arrays: The results of the timed computation.

    Returns:
        Elapsed wall-clock seconds.
    """
    for a in arrays:
        block = getattr(a, "block_until_ready", None)
        if block is not None:
            block()
    return time.perf_counter() - t0


def format_value(x: float) -> str:
    """Render one number at four significant digits."""
    x = float(x)
    if not np.isfinite(x):
        return str(x)
    return f"{x:.4g}"


def _preview_names(names: Sequence[str]) -> str:
    """List parameter names, abbreviating a long list to its first few."""
    if len(names) <= _MAX_NAMES:
        return ", ".join(names)
    head = ", ".join(names[:_MAX_NAMES])
    return f"{head}, ... {len(names) - _MAX_NAMES} more"


def _marginal_counts(problem: Problem) -> str:
    """Count the marginal kinds, e.g. ``"uniform=2, gaussian=1"``.

    Insertion order follows first appearance in the problem, so the line is
    stable for a given problem. A categorical count appearing here is also
    the "categorical presence" signal the problem section promises.
    """
    counts: dict[str, int] = {}
    for spec in problem.input_specs:
        counts[spec.dist] = counts.get(spec.dist, 0) + 1
    return ", ".join(f"{kind}={n}" for kind, n in counts.items())


def _invalid_line(invalid: InvalidReport | None) -> str:
    """Say what the non-finite check found and what was done about it."""
    if invalid is None:
        return "invalid: check delegated to the backend"
    if not invalid.any_invalid:
        return (
            f"invalid: none found in {invalid.n_units} "
            f"{invalid.unit.plural} (policy '{invalid.policy}')"
        )
    if invalid.policy == "drop":
        return (
            f"invalid: {invalid.n_invalid} of {invalid.n_units} "
            f"{invalid.unit.plural} dropped, {invalid.n_kept} kept"
        )
    return (
        f"invalid: {invalid.n_invalid} of {invalid.n_units} "
        f"{invalid.unit.plural} hold non-finite values (policy '{invalid.policy}')"
    )


def analysis_summary(
    *,
    method: str,
    problem: Problem,
    n_runs: int,
    T: int,
    K: int,
    invalid: InvalidReport | None,
    timings: Sequence[tuple[str, float]],
    index_name: str,
    values: Any,
    conf: Any | None = None,
    notes: Sequence[str] = (),
) -> None:
    """Print the three verbose sections for one finished analysis.

    Section one is the problem and the data: dimensionality, parameter
    names, marginal kinds, correlation status, output layout, and what the
    non-finite check found. Section two is the timings the caller measured,
    plus any notes (resolved batch or chunk widths, backend names). Section
    three ranks the top ``min(D, 5)`` parameters by the method's headline
    index, with intervals when the caller bootstrapped.

    Args:
        method: Fully qualified entry-point name, e.g.
            ``"jaxgsa.sobol.analyze"``.
        problem: The problem the analysis ran on.
        n_runs: Rows of model output the estimator worked from, after any
            drop.
        T: Time steps in the promoted ``(N, T, K)`` layout.
        K: Output columns in the promoted layout.
        invalid: What the non-finite check found, or ``None`` for a method
            that delegates the check to a backend.
        timings: ``(label, seconds)`` pairs, in print order. The caller
            measures them with :func:`tic` and :func:`stop`.
        notes: Extra lines for the timing section, already formatted, e.g.
            ``"slice_chunk_size: 8 (resolved from the memory budget)"``.
        index_name: The headline index the results section ranks by.
        values: The headline index array, shaped ``(..., D)`` in the
            caller's own rank.
        conf: The matching ``(2, ..., D)`` [lower, upper] interval array, or
            ``None`` when the caller did not bootstrap.
    """
    emit(method)
    emit(f"{_INDENT}problem: D={problem.num_vars} ({_preview_names(problem.names)})")
    emit(f"{_INDENT}{_INDENT}marginals: {_marginal_counts(problem)}")
    correlation = (
        "correlated (Gaussian copula)" if problem.has_correlated_inputs else "independent"
    )
    emit(f"{_INDENT}{_INDENT}correlation: {correlation}")
    slices = "slice" if T * K == 1 else "slices"
    emit(f"{_INDENT}{_INDENT}output: N={n_runs} runs, T={T} x K={K} output {slices}")
    emit(f"{_INDENT}{_INDENT}{_invalid_line(invalid)}")
    emit(f"{_INDENT}timing:")
    for label, seconds in timings:
        emit(f"{_INDENT}{_INDENT}{label}: {format_value(seconds)} s")
    for note in notes:
        emit(f"{_INDENT}{_INDENT}{note}")
    _results_section(problem=problem, index_name=index_name, values=values, conf=conf)


def _results_section(
    *,
    problem: Problem,
    index_name: str,
    values: Any,
    conf: Any | None,
) -> None:
    """Rank the top ``min(D, 5)`` parameters by the headline index."""
    D = problem.num_vars
    flat = np.asarray(values, dtype=np.float64).reshape(-1, D)
    n_slices = flat.shape[0]
    point = flat.mean(axis=0)

    lower = upper = None
    if conf is not None:
        endpoints = np.asarray(conf, dtype=np.float64).reshape(2, -1, D).mean(axis=1)
        lower, upper = endpoints[0], endpoints[1]

    k = min(D, _TOP_K)
    # NaN sorts to the bottom, not the top: a NaN index (zero-variance
    # slice, say) must not outrank every finite one.
    order = np.argsort(np.nan_to_num(point, nan=-np.inf))[::-1][:k]
    slice_note = f", mean over {n_slices} output slices" if n_slices > 1 else ""
    emit(f"{_INDENT}results: top {k} of {D} parameters by {index_name}{slice_note}")
    width = max(len(problem.names[i]) for i in order)
    for rank, i in enumerate(order, start=1):
        line = (
            f"{_INDENT}{_INDENT}{rank}. {problem.names[i]:<{width}}"
            f"  {index_name}={format_value(point[i])}"
        )
        if lower is not None and upper is not None:
            line += f"  [{format_value(lower[i])}, {format_value(upper[i])}]"
        emit(line)


def sample_summary(
    *,
    method: str,
    problem: Problem,
    entries: Sequence[tuple[str, object]],
) -> None:
    """Print one line that narrates a generated design.

    Samplers keep the one-line style the sobol and morris samplers already
    had: the entry-point name, then ``key=value`` pairs describing the
    design — runs generated, and deduplication savings where the design
    deduplicates.

    Args:
        method: Fully qualified entry-point name, e.g.
            ``"jaxgsa.sobol.sample"``.
        problem: The problem the design was drawn for.
        entries: ``(key, value)`` pairs, in print order. Values are rendered
            with ``str``; format fractions at the call site.
    """
    pairs = ", ".join(f"{key}={value}" for key, value in entries)
    emit(f"{method}: D={problem.num_vars}, {pairs}")
