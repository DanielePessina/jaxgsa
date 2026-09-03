"""Export-schema snapshot for every jaxgsa result class.

``tests/test_result_schema.py`` compares the ``to_dataset()`` schema of every
result against ``tests/data/result_dataset_schema.json``. This script writes
that file. It builds the results with the same fixtures the test uses, so the
snapshot and the test can never drift apart.

Nothing here is a number: the snapshot records dimension, coordinate and
variable names only. ``scripts/baseline_dump.py`` guards the numbers.

Run it with::

    uv run scripts/dump_result_schema.py

Add ``--out PATH`` to write somewhere else, and ``--check`` to compare the
freshly built snapshot against the stored one without writing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_TESTS = Path(__file__).resolve().parent.parent / "tests"

DEFAULT_OUT = _TESTS / "data" / "result_dataset_schema.json"


def _fixtures() -> ModuleType:
    """Import the shared fixture module ``tests/_result_fixtures.py``.

    ``tests`` is not a package and is not installed. pytest puts the directory
    on ``sys.path`` for the test that reads the snapshot, and this script does
    the same, so both consumers build their results from one module.

    Returns:
        The imported fixture module.
    """
    if str(_TESTS) not in sys.path:
        sys.path.insert(0, str(_TESTS))
    import _result_fixtures

    return _result_fixtures


def build_snapshot(verbose: bool = True) -> dict[str, dict[str, Any]]:
    """Build one schema entry per fixture case.

    Args:
        verbose: Print progress to stderr.

    Returns:
        Mapping from ``"<builder>@<shape>"`` to the recorded schema, plus the
        name of the result class that produced it.
    """
    fixtures = _fixtures()
    snapshot: dict[str, dict[str, Any]] = {}
    for name, shape in fixtures.CASES:
        key = f"{name}@{shape}"
        if verbose:
            print(f"  {key}", file=sys.stderr, flush=True)
        result = fixtures.build(name, shape)
        entry: dict[str, Any] = {"type": type(result).__name__}
        entry.update(fixtures.dataset_schema(result))
        snapshot[key] = entry
    return snapshot


def to_json(snapshot: dict[str, dict[str, Any]]) -> str:
    """Serialize a snapshot.

    Args:
        snapshot: The mapping from :func:`build_snapshot`.

    Returns:
        The JSON text, with sorted keys so the bytes are reproducible.
    """
    return json.dumps(snapshot, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the schema snapshot.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv``.

    Returns:
        Process exit code. ``1`` if ``--check`` found a difference.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSON path")
    parser.add_argument(
        "--check", action="store_true", help="compare against the stored file, write nothing"
    )
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args(argv)
    verbose = not args.quiet

    text = to_json(build_snapshot(verbose=verbose))

    if args.check:
        if not args.out.exists():
            print(f"{args.out} does not exist", file=sys.stderr)
            return 1
        if args.out.read_text() != text:
            print(f"{args.out} is out of date; run this script without --check", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date", file=sys.stderr)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
