"""Tier T4 (behavioural contract): the documented matrix matches the registry.

jaxgsa used to state its compatibility matrix in three places. The three
disagreed. ``docs/guide/methods.md`` had a categorical row with thirteen cells
where every other row had fourteen, so one column silently rendered blank.
``docs/index.md`` credited DGSM with a sampler it does not have.
``docs/api/index.md`` listed eight of the nine given-data methods. Nothing
caught any of it, because a table in Markdown is answerable to nobody.

There is now one table, under ``### Method capabilities`` in
``docs/guide/methods.md``. These tests parse it and check every cell against
:func:`jaxgsa._core.registry.methods`, which ``tests/test_registry.py`` in
turn checks against what the methods actually do. A fourteenth method, or a
changed capability, fails here rather than leaving the documentation stale.

The pattern comes from ``tests/test_warning_category.py``, the one
cross-cutting test in this suite that could not drift.

No external oracle: these pin documentation against the library's own
declarations, not against any numerical result.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from jaxgsa._core.registry import methods

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
METHODS_PAGE = DOCS / "guide" / "methods.md"
API_PAGE = DOCS / "api" / "index.md"

CAPABILITY_HEADING = "### Method capabilities"
GIVEN_DATA_HEADING = "## Given-Data Methods"

EXPECTED_COLUMNS = [
    "Method",
    "Reports",
    "Own design",
    "Correlated",
    "Categorical",
    "Bootstrap CI",
]

YES = "✓"
NO = "✗"
DASH = "—"
FOOTNOTE_MARKERS = "†‡§"

NAME_IN_CELL = re.compile(r"\[`([a-z_]+)`\]")


def _table_after(page: pathlib.Path, heading: str) -> list[list[str]]:
    """Read the first Markdown table that follows a heading.

    Args:
        page: The Markdown file to read.
        heading: The exact heading line the table sits under.

    Returns:
        One list of stripped cells per table line, header row first and the
        ``|---|`` separator dropped.

    Raises:
        AssertionError: If the heading is absent, or no table follows it.
    """
    lines = page.read_text().splitlines()
    assert heading in lines, f"{page.name} no longer has the heading {heading!r}"
    start = lines.index(heading)

    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)

    assert rows, f"{page.name} has no table under {heading!r}"
    return rows


def _capability_rows() -> dict[str, list[str]]:
    """Parse the capability table into one cell list per method name.

    Returns:
        The data rows, keyed by the method name read out of the first cell.

    Raises:
        AssertionError: If the columns changed, or a row names no method.
    """
    header, *body = _table_after(METHODS_PAGE, CAPABILITY_HEADING)
    assert header == EXPECTED_COLUMNS, (
        f"the capability table's columns changed to {header}; this test reads "
        f"them by position, so update EXPECTED_COLUMNS with the new meaning"
    )

    rows: dict[str, list[str]] = {}
    for cells in body:
        assert len(cells) == len(EXPECTED_COLUMNS), (
            f"row {cells[0]!r} has {len(cells)} cells, not "
            f"{len(EXPECTED_COLUMNS)}; a short row renders a blank column"
        )
        match = NAME_IN_CELL.match(cells[0])
        assert match is not None, (
            f"the first cell of a capability row must be a linked method name "
            f"like [`sobol`](#...), not {cells[0]!r}"
        )
        rows[match.group(1)] = cells
    return rows


def _mark(cell: str, column: str, name: str) -> bool:
    """Read a tick-or-cross cell, ignoring any footnote marker.

    Args:
        cell: The raw cell text.
        column: The column name, for the failure message.
        name: The method name, for the failure message.

    Returns:
        True for a tick, False for a cross.

    Raises:
        AssertionError: If the cell holds neither.
    """
    bare = cell.strip(FOOTNOTE_MARKERS + " ")
    assert bare in (YES, NO), (
        f"the {column!r} cell for {name} is {cell!r}; it must be {YES} or "
        f"{NO}, optionally followed by a footnote marker"
    )
    return bare == YES


ROWS = _capability_rows()
SPECS = sorted(methods().values(), key=lambda s: s.name)
IDS = [s.name for s in SPECS]


class TestTheTableCoversEveryMethod:
    """A fourteenth method cannot be left out of the documentation."""

    def test_the_rows_are_exactly_the_registered_methods(self):
        documented = set(ROWS)
        registered = set(methods())
        assert documented == registered, (
            f"undocumented: {sorted(registered - documented)}; "
            f"documented but not registered: {sorted(documented - registered)}"
        )


class TestEveryCellIsTrue:
    """Each documented cell is answerable to the registry."""

    @pytest.mark.parametrize("spec", SPECS, ids=IDS)
    def test_the_reports_cell_says_something(self, spec):
        """The registry declares no list of what a method reports.

        The cell is prose — "bounds on $S_T$", "dependence measure",
        "$W_2^2$ index" — and none of those is answerable to a field on the
        result class, so the wording cannot be checked. What can be checked is
        that the cell holds prose at all: a blank cell, or a capability mark
        that slipped one column to the left, is the defect that put this test
        module here in the first place.
        """
        cell = ROWS[spec.name][EXPECTED_COLUMNS.index("Reports")]
        assert cell, f"the Reports cell for {spec.name} is empty"
        assert cell.strip(FOOTNOTE_MARKERS + " ") not in (YES, NO, DASH), (
            f"the Reports cell for {spec.name} is {cell!r}; a tick, a cross or "
            f"a dash there means a capability mark landed in the wrong column"
        )

    @pytest.mark.parametrize("spec", SPECS, ids=IDS)
    def test_the_own_design_cell(self, spec):
        cell = ROWS[spec.name][EXPECTED_COLUMNS.index("Own design")]
        documented = _mark(cell, "Own design", spec.name)
        assert documented == spec.is_design_based, (
            f"the docs say {spec.name} "
            f"{'has' if documented else 'has no'} sampler of its own, but the "
            f"registry says sample={spec.sample!r}"
        )

    @pytest.mark.parametrize("spec", SPECS, ids=IDS)
    def test_the_correlated_cell(self, spec):
        cell = ROWS[spec.name][EXPECTED_COLUMNS.index("Correlated")]
        documented = _mark(cell, "Correlated", spec.name)
        assert documented == (spec.correlation == "accepts"), (
            f"the docs say {spec.name} "
            f"{'accepts' if documented else 'refuses'} a correlated problem, "
            f"but the registry declares correlation={spec.correlation!r}"
        )

    @pytest.mark.parametrize("spec", SPECS, ids=IDS)
    def test_the_categorical_cell(self, spec):
        cell = ROWS[spec.name][EXPECTED_COLUMNS.index("Categorical")]
        documented = _mark(cell, "Categorical", spec.name)
        assert documented == (spec.categorical == "accepts"), (
            f"the docs say {spec.name} "
            f"{'accepts' if documented else 'refuses'} a categorical problem, "
            f"but the registry declares categorical={spec.categorical!r}"
        )

    @pytest.mark.parametrize("spec", SPECS, ids=IDS)
    def test_the_bootstrap_cell_names_the_real_keyword(self, spec):
        """Two spellings are in use, so the table must give the right one."""
        cell = ROWS[spec.name][EXPECTED_COLUMNS.index("Bootstrap CI")]
        documented = None if cell == DASH else cell.strip("`")
        assert documented == spec.bootstrap, (
            f"the docs give {spec.name} the bootstrap keyword "
            f"{documented!r}, but the registry declares {spec.bootstrap!r}"
        )


class TestTheProseAgreesWithTheTable:
    """The counts written out in words drift as easily as the cells."""

    def test_the_given_data_table_lists_every_given_data_method(self):
        """``docs/api/index.md`` once listed eight of the nine."""
        rows = _table_after(API_PAGE, GIVEN_DATA_HEADING)[1:]
        listed = {cells[0].strip("`").removeprefix("jaxgsa.") for cells in rows}
        expected = {s.name for s in SPECS if not s.is_design_based}
        assert listed == expected, (
            f"missing from the given-data table: {sorted(expected - listed)}; "
            f"listed but design-based or unknown: {sorted(listed - expected)}"
        )

    def test_the_given_data_table_names_the_right_result_class(self):
        rows = _table_after(API_PAGE, GIVEN_DATA_HEADING)[1:]
        documented = {
            cells[0].strip("`").removeprefix("jaxgsa."): cells[2].strip("`") for cells in rows
        }
        for spec in SPECS:
            if spec.is_design_based:
                continue
            assert documented[spec.name] == spec.result.__name__
