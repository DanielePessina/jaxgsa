"""Compare the current jaxgsa against the stored numerical baseline.

Re-runs :mod:`baseline_dump` and diffs the result against the stored JSON.
Every changed field is reported with the old value, the new value, and the
absolute difference. The script exits non-zero on any difference.

Differences come in two kinds and they mean opposite things, so they are
reported separately:

**Changed values.** A number, shape, dtype or length that moved. After a
change declared "plumbing only" this is a wiring error, not a tolerance
problem. There is no tolerance here on purpose.

**Schema changes.** A field that a result gained or lost. This is not a
wiring error: it is a deliberate change to the result surface, and the fix
is to regenerate the baseline once the new field is intended. Reporting it
in the same list as a moved number buries the one that matters.

Run it with::

    uv run scripts/baseline_check.py

**The comparison is only valid within one machine.** The stored file records
float32 results to the last bit, and those bits depend on the CPU. Comparing a
dump made on Apple silicon against one made on x86-64 reports about a thousand
moved values with deltas around 1e-5 to 1e-7, none of which is a code change:
different XLA kernels reassociate the same arithmetic differently. So the
stored baseline is a *local* instrument, checked by the developer who owns the
machine that produced it, in the same spirit as the project's rule that
oracles run locally and never in CI.

CI therefore does not use the stored file. It dumps the base commit and the
head commit on one runner and diffs those two, with ``--current``. Same
machine on both sides, no tolerance, and it answers the question that matters
on a pull request: did *this change* move a number.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from baseline_dump import DEFAULT_OUT, build_dump

MAX_REPORTED_ELEMENTS = 10
"""How many differing elements of one array to print before summarizing."""


@dataclass
class Diffs:
    """Differences found, split by what they mean.

    Attributes:
        values: Numbers, shapes, dtypes or lengths that moved. Each one is a
            wiring error until proved otherwise.
        schema: Fields the current run gained or lost relative to the stored
            baseline. These call for regenerating the baseline, not for a fix
            to the library.
    """

    values: list[str] = field(default_factory=list)
    schema: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Return True when anything at all differs."""
        return bool(self.values or self.schema)


def _fmt(value: Any) -> str:
    """Format a scalar for the report."""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _compare_values(path: str, old: Any, new: Any, diffs: Diffs) -> None:
    """Record every difference between two encoded values.

    Args:
        path: Dotted location of the value, used in the report.
        old: Value from the stored baseline.
        new: Value from the fresh run.
        diffs: Collector the difference lines are appended to. A gained or
            lost key goes to ``schema``; everything else goes to ``values``.
    """
    if isinstance(old, dict) and isinstance(new, dict):
        if old.get("__array__") and new.get("__array__"):
            _compare_arrays(path, old, new, diffs)
            return
        for key in sorted(set(old) | set(new)):
            if key not in old:
                diffs.schema.append(f"{path}.{key}: ADDED (not in baseline)")
            elif key not in new:
                diffs.schema.append(f"{path}.{key}: REMOVED (present in baseline)")
            else:
                _compare_values(f"{path}.{key}", old[key], new[key], diffs)
        return

    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            diffs.values.append(f"{path}: length {len(old)} -> {len(new)}")
            return
        for i, (o, n) in enumerate(zip(old, new, strict=True)):
            _compare_values(f"{path}[{i}]", o, n, diffs)
        return

    if type(old) is not type(new):
        diffs.values.append(f"{path}: {_fmt(old)} -> {_fmt(new)} (type changed)")
        return

    if _scalar_differs(old, new):
        line = f"{path}: {_fmt(old)} -> {_fmt(new)}"
        if isinstance(old, float) and isinstance(new, float):
            line += f"  (delta {new - old!r})"
        diffs.values.append(line)


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


def _compare_arrays(path: str, old: dict[str, Any], new: dict[str, Any], diffs: Diffs) -> None:
    """Compare two encoded arrays element by element or by digest."""
    if old["shape"] != new["shape"]:
        diffs.values.append(f"{path}: shape {old['shape']} -> {new['shape']}")
        return
    if old["dtype"] != new["dtype"]:
        diffs.values.append(f"{path}: dtype {old['dtype']} -> {new['dtype']}")
        return

    if "sha256" in old or "sha256" in new:
        if old.get("sha256") != new.get("sha256"):
            diffs.values.append(
                f"{path}: bytes differ (sha256 {old.get('sha256', '?')[:12]}"
                f" -> {new.get('sha256', '?')[:12]}), shape {old['shape']}"
            )
        return

    o_flat = _flatten(old["values"])
    n_flat = _flatten(new["values"])
    changed = [
        (i, o, n)
        for i, (o, n) in enumerate(zip(o_flat, n_flat, strict=True))
        if _scalar_differs(o, n)
    ]
    for i, o, n in changed[:MAX_REPORTED_ELEMENTS]:
        delta = ""
        if isinstance(o, float) and isinstance(n, float):
            delta = f"  (delta {n - o!r})"
        diffs.values.append(f"{path}[flat {i}]: {_fmt(o)} -> {_fmt(n)}{delta}")
    if len(changed) > MAX_REPORTED_ELEMENTS:
        diffs.values.append(
            f"{path}: {len(changed)} of {len(o_flat)} elements differ"
            f" ({len(changed) - MAX_REPORTED_ELEMENTS} more not listed)"
        )


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> Diffs:
    """Diff a stored baseline body against a fresh one.

    Args:
        baseline: The ``results`` mapping loaded from the JSON file.
        current: The mapping returned by ``build_dump``.

    Returns:
        The differences, split into moved values and schema changes. Falsy
        when the two agree exactly.
    """
    diffs = Diffs()
    _compare_values("results", baseline, current, diffs)
    return diffs


def main(argv: list[str] | None = None) -> int:
    """Run the check.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv``.

    Returns:
        ``0`` when the current run matches the baseline exactly, ``1`` when a
        value moved, and ``2`` when only the schema changed. Both failures are
        non-zero, so either one stops CI, but they call for different fixes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_OUT, help="baseline JSON path")
    parser.add_argument(
        "--current",
        type=Path,
        default=None,
        help=(
            "Compare a dump file that already exists instead of building one. "
            "Used by CI to diff two commits dumped on the same runner."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        print(f"baseline not found: {args.baseline}", file=sys.stderr)
        print("run: uv run scripts/baseline_dump.py", file=sys.stderr)
        return 1

    stored = json.loads(args.baseline.read_text())
    header = stored.get("header", {})
    if args.current is not None:
        if not args.current.exists():
            print(f"current dump not found: {args.current}", file=sys.stderr)
            return 1
        current = json.loads(args.current.read_text())["results"]
    else:
        current = build_dump(verbose=not args.quiet)

    diffs = compare(stored["results"], current)
    recorded_at = header.get("git_commit", "?")
    if not diffs:
        print(f"baseline check PASSED against {args.baseline.name} ({recorded_at})")
        return 0

    print(f"baseline check FAILED against {args.baseline.name}, recorded at {recorded_at}")

    if diffs.values:
        print(f"\n{len(diffs.values)} value(s) moved. Each one is a wiring error:")
        for line in diffs.values:
            print(f"  {line}")
        print("\nA changed number is a wiring error, not a tolerance issue.")

    if diffs.schema:
        print(f"\n{len(diffs.schema)} schema change(s). No number moved here:")
        for line in diffs.schema:
            print(f"  {line}")
        print(
            "\nA gained or lost field is a deliberate change to the result "
            "surface. When it is intended, regenerate the baseline:\n"
            "  uv run scripts/baseline_dump.py --out scripts/baseline/baseline-<version>.json"
        )

    return 1 if diffs.values else 2


if __name__ == "__main__":
    raise SystemExit(main())
