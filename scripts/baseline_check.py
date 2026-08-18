"""Compare the current jaxgsa against the stored numerical baseline.

Re-runs :mod:`baseline_dump` and diffs the result against
``scripts/baseline/baseline-0.8.0.json``. Every changed field is reported
with the old value, the new value, and the absolute difference. The script
exits non-zero on any difference.

A changed number after a "plumbing only" refactor is a wiring error, not a
tolerance problem. There is no tolerance here on purpose.

Run it with::

    uv run scripts/baseline_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from baseline_dump import DEFAULT_OUT, build_dump

MAX_REPORTED_ELEMENTS = 10
"""How many differing elements of one array to print before summarizing."""


def _fmt(value: Any) -> str:
    """Format a scalar for the report."""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _compare_values(path: str, old: Any, new: Any, out: list[str]) -> None:
    """Append a line per difference between two encoded values.

    Args:
        path: Dotted location of the value, used in the report.
        old: Value from the stored baseline.
        new: Value from the fresh run.
        out: List that difference lines are appended to.
    """
    if isinstance(old, dict) and isinstance(new, dict):
        if old.get("__array__") and new.get("__array__"):
            _compare_arrays(path, old, new, out)
            return
        for key in sorted(set(old) | set(new)):
            if key not in old:
                out.append(f"{path}.{key}: ADDED (not in baseline)")
            elif key not in new:
                out.append(f"{path}.{key}: REMOVED (present in baseline)")
            else:
                _compare_values(f"{path}.{key}", old[key], new[key], out)
        return

    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            out.append(f"{path}: length {len(old)} -> {len(new)}")
            return
        for i, (o, n) in enumerate(zip(old, new, strict=True)):
            _compare_values(f"{path}[{i}]", o, n, out)
        return

    if type(old) is not type(new):
        out.append(f"{path}: {_fmt(old)} -> {_fmt(new)} (type changed)")
        return

    if _scalar_differs(old, new):
        line = f"{path}: {_fmt(old)} -> {_fmt(new)}"
        if isinstance(old, float) and isinstance(new, float):
            line += f"  (delta {new - old!r})"
        out.append(line)


def _scalar_differs(old: Any, new: Any) -> bool:
    """Return True when two scalars differ, treating NaN as equal to NaN."""
    if isinstance(old, float) and isinstance(new, float):
        if old != old and new != new:  # both NaN
            return False
    return old != new


def _flatten(values: Any) -> list[Any]:
    """Flatten a nested list of scalars."""
    if not isinstance(values, list):
        return [values]
    flat: list[Any] = []
    for v in values:
        flat.extend(_flatten(v))
    return flat


def _compare_arrays(path: str, old: dict[str, Any], new: dict[str, Any], out: list[str]) -> None:
    """Compare two encoded arrays element by element or by digest."""
    if old["shape"] != new["shape"]:
        out.append(f"{path}: shape {old['shape']} -> {new['shape']}")
        return
    if old["dtype"] != new["dtype"]:
        out.append(f"{path}: dtype {old['dtype']} -> {new['dtype']}")
        return

    if "sha256" in old or "sha256" in new:
        if old.get("sha256") != new.get("sha256"):
            out.append(
                f"{path}: bytes differ (sha256 {old.get('sha256', '?')[:12]}"
                f" -> {new.get('sha256', '?')[:12]}), shape {old['shape']}"
            )
        return

    o_flat = _flatten(old["values"])
    n_flat = _flatten(new["values"])
    diffs = [
        (i, o, n)
        for i, (o, n) in enumerate(zip(o_flat, n_flat, strict=True))
        if _scalar_differs(o, n)
    ]
    for i, o, n in diffs[:MAX_REPORTED_ELEMENTS]:
        delta = ""
        if isinstance(o, float) and isinstance(n, float):
            delta = f"  (delta {n - o!r})"
        out.append(f"{path}[flat {i}]: {_fmt(o)} -> {_fmt(n)}{delta}")
    if len(diffs) > MAX_REPORTED_ELEMENTS:
        out.append(
            f"{path}: {len(diffs)} of {len(o_flat)} elements differ"
            f" ({len(diffs) - MAX_REPORTED_ELEMENTS} more not listed)"
        )


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Diff a stored baseline body against a fresh one.

    Args:
        baseline: The ``results`` mapping loaded from the JSON file.
        current: The mapping returned by ``build_dump``.

    Returns:
        One line per difference. Empty means the two agree exactly.
    """
    out: list[str] = []
    _compare_values("results", baseline, current, out)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the check.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv``.

    Returns:
        0 when the current run matches the baseline exactly, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_OUT, help="baseline JSON path")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        print(f"baseline not found: {args.baseline}", file=sys.stderr)
        print("run: uv run scripts/baseline_dump.py", file=sys.stderr)
        return 1

    stored = json.loads(args.baseline.read_text())
    header = stored.get("header", {})
    current = build_dump(verbose=not args.quiet)

    diffs = compare(stored["results"], current)
    if not diffs:
        print(
            f"baseline check PASSED against {args.baseline.name} ({header.get('git_commit', '?')})"
        )
        return 0

    print(f"baseline check FAILED: {len(diffs)} difference(s)")
    print(f"baseline was recorded at commit {header.get('git_commit', '?')}")
    for line in diffs:
        print(f"  {line}")
    print("\nA changed number is a wiring error, not a tolerance issue.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
