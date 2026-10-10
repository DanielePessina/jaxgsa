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


def _method(s1=(1.0, 2.0), **fields: Any) -> dict[str, Any]:
    """One method entry shaped like ``baseline``, with fields replaced or added."""
    return {"method": {"S1": _array(list(s1)), "streamed": False, **fields}}


def test_identical_input_compares_clean_and_nan_equals_nan(baseline):
    """No difference reads as none. A NaN index is a result, not a change."""
    diffs = compare(baseline, baseline)
    assert diffs.values == []
    assert diffs.schema == []
    assert not diffs
    one = {"m": {"S1": _array([float("nan"), 1.0])}}
    assert not compare(one, one)


def test_each_kind_of_value_change_lands_in_values(baseline):
    """A moved number, shape, dtype or scalar is a wiring error, so ``values``."""
    cases = {
        "2.0 -> 2.5": _method(s1=(1.0, 2.5)),
        "shape": _method(s1=(1.0, 2.0, 3.0)),
        "dtype": {"method": {"S1": {**_array([1.0, 2.0]), "dtype": "float32"}, "streamed": False}},
        "streamed": _method(streamed=True),
    }
    for expected, current in cases.items():
        diffs = compare(baseline, current)
        assert diffs.schema == [], expected
        assert len(diffs.values) == 1, expected
        assert expected in diffs.values[0], expected
    moved = compare(baseline, cases["2.0 -> 2.5"])
    assert "delta" in moved.values[0]


def test_a_gained_or_lost_field_lands_in_schema(baseline):
    """A gained or lost field is not a wiring error, and must not read as one."""
    gained = compare(baseline, _method(invalid={"n": 0}))
    assert gained.values == []
    assert len(gained.schema) == 1
    assert "ADDED" in gained.schema[0]

    lost = compare(baseline, {"method": {"S1": _array([1.0, 2.0])}})
    assert lost.values == []
    assert len(lost.schema) == 1
    assert "REMOVED" in lost.schema[0]


def test_a_gained_field_does_not_hide_a_moved_number(baseline):
    """The regression the values/schema split exists to prevent.

    Fifty added fields alongside one moved number is exactly the state this
    repository was in while the on_invalid report was landing. In one
    undifferentiated list the moved number is the fifty-first line.
    """
    diffs = compare(baseline, _method(s1=(1.0, 999.0), invalid={"n": 0}))
    assert len(diffs.values) == 1, "the moved number must be reported on its own"
    assert len(diffs.schema) == 1
    assert "999.0" in diffs.values[0]


def test_exit_codes_and_reports_for_two_stored_dumps(tmp_path, baseline, capsys):
    """`--current` is how CI diffs two commits dumped on one runner.

    The stored baseline cannot be used in CI: it records float32 bits, and
    those depend on the CPU, so an Apple-silicon file compared against an
    x86-64 run reports about a thousand moved values with no code change.
    CI asks one question, did this change move a number, so
    ``--allow-schema-change`` waves a gained field through but never a moved
    value.
    """
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"

    def run(current, *flags):
        for path, results in ((base, baseline), (head, current)):
            path.write_text(json.dumps({"header": {"git_commit": "test"}, "results": results}))
        code = baseline_check.main(["--baseline", str(base), "--current", str(head), *flags])
        return code, capsys.readouterr().out

    code, out = run(baseline)
    assert code == 0
    assert "PASSED" in out

    code, out = run(_method(s1=(1.0, 2.5)))
    assert code == 1
    assert "value(s) moved" in out

    code, out = run(_method(invalid={"n": 0}))
    assert code == 2
    assert "schema change(s)" in out
    assert "regenerate" in out

    code, out = run(_method(ci={"level": 0.95}), "--allow-schema-change")
    assert code == 0
    assert "schema change(s)" in out, "the change must still be reported, just not fatal"

    code, out = run(_method(s1=(1.0, 999.0), ci={"level": 0.95}), "--allow-schema-change")
    assert code == 1, "the flag must not become a way to wave a wiring error through"
    assert "999.0" in out

    missing = baseline_check.main(["--baseline", str(base), "--current", str(tmp_path / "no")])
    assert missing == 1


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
