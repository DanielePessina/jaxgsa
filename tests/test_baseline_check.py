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
import baseline_dump  # noqa: E402  # isort: skip
from baseline_check import compare  # noqa: E402  # isort: skip

from jaxgsa._core.registry import methods  # noqa: E402  # isort: skip


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


class TestAllowSchemaChange:
    """CI asks one question: did this change move a number?

    A result that gains a field is a deliberate edit to the public surface,
    reviewed in the diff like any other. Blocking it on the numerical guard
    would mean every PR that touches a result class goes red for the wrong
    reason. Moved values must still fail.
    """

    def _write(self, path, results):
        path.write_text(json.dumps({"header": {"git_commit": "test"}, "results": results}))

    def test_a_schema_change_passes_when_allowed(self, tmp_path, baseline, capsys):
        base, head = tmp_path / "base.json", tmp_path / "head.json"
        self._write(base, baseline)
        self._write(head, {"method": {**baseline["method"], "ci": {"level": 0.95}}})
        code = baseline_check.main(
            ["--baseline", str(base), "--current", str(head), "--allow-schema-change"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "schema change(s)" in out, "the change must still be reported, just not fatal"

    def test_a_moved_value_still_fails_when_schema_changes_are_allowed(
        self, tmp_path, baseline, capsys
    ):
        """The flag must not become a way to wave a wiring error through."""
        base, head = tmp_path / "base.json", tmp_path / "head.json"
        self._write(base, baseline)
        self._write(
            head,
            {"method": {"S1": _array([1.0, 999.0]), "streamed": False, "ci": {"level": 0.95}}},
        )
        code = baseline_check.main(
            ["--baseline", str(base), "--current", str(head), "--allow-schema-change"]
        )
        assert code == 1
        assert "999.0" in capsys.readouterr().out


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


class TestEveryMethodIsDumped:
    """The one registration surface that had no guard.

    ``scripts/baseline_dump.py`` lists its runners by hand. A method left out
    of both tables produces no entry in the base dump and none in the head
    dump, so the CI diff is clean and the method ships with no numerical
    regression guard at all. Every other surface — the docs matrix, the
    result-schema snapshot, the result fixtures — is checked against the
    registry. This makes that one answerable too.

    The runner bodies cannot be derived: they carry per-method keywords such
    as ``n_variance=512``. The coverage can.
    """

    def test_the_runner_tables_cover_the_registry(self):
        covered = set(baseline_dump.DESIGN_METHODS) | set(baseline_dump.GIVEN_DATA_METHODS)
        registered = set(methods())
        variants = set(baseline_dump.MODE_VARIANTS)
        assert registered <= covered, (
            f"no baseline runner: {sorted(registered - covered)} — add one to "
            "scripts/baseline_dump.py, in DESIGN_METHODS for a method with its own "
            "sampler and in GIVEN_DATA_METHODS otherwise, then regenerate the stored "
            "baseline. A method in neither table is dumped by neither side of the CI "
            "diff, so it ships with no numerical guard."
        )
        assert covered - registered == variants, (
            "runner with no registered method and no MODE_VARIANTS entry: "
            f"{sorted(covered - registered - variants)}. A runner that pins an extra "
            "mode of an existing method belongs in MODE_VARIANTS, which says which "
            "method it varies. Any other extra key is a typo."
        )

    def test_every_mode_variant_names_a_registered_method(self):
        """A variant pins another mode of a method, so that method must exist."""
        registered = set(methods())
        for variant, base in baseline_dump.MODE_VARIANTS.items():
            assert base in registered, f"{variant} varies {base!r}, which is not registered"
            assert variant not in registered, (
                f"{variant} is a registered method, so it is not a variant of {base!r}"
            )

    def test_each_runner_sits_in_the_table_that_matches_its_calling_convention(self):
        """The two tables call their runners with different arguments.

        A design-based method in ``GIVEN_DATA_METHODS`` would be handed an
        ``X`` and a ``Y`` it never asked for.
        """
        for name, spec in methods().items():
            table = "DESIGN_METHODS" if spec.is_design_based else "GIVEN_DATA_METHODS"
            other = "GIVEN_DATA_METHODS" if spec.is_design_based else "DESIGN_METHODS"
            assert name in getattr(baseline_dump, table), (
                f"{name} has sample={spec.sample!r} so its runner belongs in {table}"
            )
            for variant, base in baseline_dump.MODE_VARIANTS.items():
                if base == name:
                    assert variant in getattr(baseline_dump, table), (
                        f"{variant} varies {name}, so its runner belongs in {table} too"
                    )
            assert name not in getattr(baseline_dump, other), (
                f"{name} is listed in {other}, which calls it the wrong way"
            )
