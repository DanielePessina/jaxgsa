"""Tests for the numerical-baseline comparison in ``scripts/baseline_check.py``.

The baseline check is the gate that decides whether a change declared
"plumbing only" moved a number. It reports two kinds of difference that mean
opposite things, and it must not confuse them:

- a **value** that moved is a wiring error in the library
- a **schema** change is a field the result surface gained or lost, and calls
  for regenerating the baseline instead

Tier T4 (behavioural contract). There is no external oracle here: these tests
pin the reporting split and the exit codes, not any numerical result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# ``scripts`` is not a package and is not installed; the scripts import each
# other by plain module name, the same way they do when run directly.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import baseline_check  # noqa: E402  # isort: skip
from baseline_check import Diffs, compare  # noqa: E402  # isort: skip


def _array(values: list[float], shape: list[int] | None = None) -> dict[str, Any]:
    """Build an encoded array in the form ``baseline_dump.encode`` produces."""
    return {
        "__array__": True,
        "shape": shape if shape is not None else [len(values)],
        "dtype": "float64",
        "values": values,
    }


@pytest.fixture
def baseline() -> dict[str, Any]:
    """A minimal stored baseline with one array field and one scalar field."""
    return {"method": {"S1": _array([1.0, 2.0]), "streamed": False}}


class TestNoDifference:
    def test_identical_input_compares_clean(self, baseline):
        diffs = compare(baseline, baseline)
        assert diffs.values == []
        assert diffs.schema == []
        assert not diffs

    def test_nan_equals_nan(self):
        """A NaN index is a legitimate result and must not read as a change."""
        nan = float("nan")
        one = {"m": {"S1": _array([nan, 1.0])}}
        assert not compare(one, one)


class TestValueChanges:
    """Everything here is a wiring error, and must land in ``values``."""

    def test_a_moved_number_is_a_value_change(self, baseline):
        current = {"method": {"S1": _array([1.0, 2.5]), "streamed": False}}
        diffs = compare(baseline, current)
        assert diffs.schema == []
        assert len(diffs.values) == 1
        assert "2.0 -> 2.5" in diffs.values[0]
        assert "delta" in diffs.values[0]

    def test_a_changed_shape_is_a_value_change(self, baseline):
        current = {"method": {"S1": _array([1.0, 2.0, 3.0]), "streamed": False}}
        diffs = compare(baseline, current)
        assert diffs.schema == []
        assert len(diffs.values) == 1
        assert "shape" in diffs.values[0]

    def test_a_changed_dtype_is_a_value_change(self, baseline):
        current = {"method": {"S1": {**_array([1.0, 2.0]), "dtype": "float32"}, "streamed": False}}
        diffs = compare(baseline, current)
        assert diffs.schema == []
        assert "dtype" in diffs.values[0]

    def test_a_changed_scalar_is_a_value_change(self, baseline):
        current = {"method": {"S1": _array([1.0, 2.0]), "streamed": True}}
        diffs = compare(baseline, current)
        assert diffs.schema == []
        assert len(diffs.values) == 1
        assert "streamed" in diffs.values[0]


class TestSchemaChanges:
    """A gained or lost field is not a wiring error, and must not read as one."""

    def test_a_gained_field_is_a_schema_change(self, baseline):
        current = {"method": {"S1": _array([1.0, 2.0]), "streamed": False, "invalid": {"n": 0}}}
        diffs = compare(baseline, current)
        assert diffs.values == []
        assert len(diffs.schema) == 1
        assert "ADDED" in diffs.schema[0]

    def test_a_lost_field_is_a_schema_change(self, baseline):
        current = {"method": {"S1": _array([1.0, 2.0])}}
        diffs = compare(baseline, current)
        assert diffs.values == []
        assert len(diffs.schema) == 1
        assert "REMOVED" in diffs.schema[0]

    def test_a_gained_method_is_a_schema_change(self, baseline):
        """Adding a fourteenth method must not read as a wiring error."""
        current = {**baseline, "new_method": {"S1": _array([0.5])}}
        diffs = compare(baseline, current)
        assert diffs.values == []
        assert len(diffs.schema) == 1


class TestComparingTwoStoredDumps:
    """`--current` is how CI diffs two commits dumped on one runner.

    The stored baseline cannot be used in CI: it records float32 bits, and
    those depend on the CPU, so an Apple-silicon file compared against an
    x86-64 run reports about a thousand moved values with no code change.
    """

    def _write(self, path, results):
        path.write_text(json.dumps({"header": {"git_commit": "test"}, "results": results}))

    def test_two_identical_dumps_pass(self, tmp_path, baseline, capsys):
        base, head = tmp_path / "base.json", tmp_path / "head.json"
        self._write(base, baseline)
        self._write(head, baseline)
        code = baseline_check.main(["--baseline", str(base), "--current", str(head)])
        assert code == 0
        assert "PASSED" in capsys.readouterr().out

    def test_a_moved_value_exits_one(self, tmp_path, baseline, capsys):
        base, head = tmp_path / "base.json", tmp_path / "head.json"
        self._write(base, baseline)
        self._write(head, {"method": {"S1": _array([1.0, 2.5]), "streamed": False}})
        assert baseline_check.main(["--baseline", str(base), "--current", str(head)]) == 1
        assert "value(s) moved" in capsys.readouterr().out

    def test_a_schema_change_alone_exits_two(self, tmp_path, baseline, capsys):
        base, head = tmp_path / "base.json", tmp_path / "head.json"
        self._write(base, baseline)
        self._write(head, {"method": {**baseline["method"], "invalid": {"n": 0}}})
        assert baseline_check.main(["--baseline", str(base), "--current", str(head)]) == 2
        out = capsys.readouterr().out
        assert "schema change(s)" in out
        assert "regenerate" in out

    def test_a_missing_current_file_is_refused(self, tmp_path, baseline):
        base = tmp_path / "base.json"
        self._write(base, baseline)
        assert (
            baseline_check.main(["--baseline", str(base), "--current", str(tmp_path / "no")]) == 1
        )


class TestTheTwoKindsStaySeparate:
    def test_a_gained_field_does_not_hide_a_moved_number(self, baseline):
        """The regression this split exists to prevent.

        Fifty added fields alongside one moved number is exactly the state
        this repository was in while the on_invalid report was landing. In one
        undifferentiated list the moved number is the fifty-first line.
        """
        current = {"method": {"S1": _array([1.0, 999.0]), "streamed": False, "invalid": {"n": 0}}}
        diffs = compare(baseline, current)
        assert len(diffs.values) == 1, "the moved number must be reported on its own"
        assert len(diffs.schema) == 1
        assert "999.0" in diffs.values[0]

    def test_diffs_is_falsy_only_when_both_are_empty(self):
        assert not Diffs()
        assert Diffs(values=["x"])
        assert Diffs(schema=["x"])
