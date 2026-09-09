"""Tests for the shared non-finite policy in ``jaxgsa._core.invalid``.

These test the policy machinery itself. Each analysis module has its own tests
for the fact that it applies the policy at all.
"""

import numpy as np
import pytest

from jaxgsa import JaxgsaWarning
from jaxgsa._core.invalid import (
    InvalidUnit,
    check_invalid,
    resolve_policy,
)

METHOD = "jaxgsa.example.analyze"


def _rows(*bad_rows: int, n: int = 6, width: int = 2) -> np.ndarray:
    """Build an (n, width) array that is finite except at the named rows."""
    array = np.arange(n * width, dtype=float).reshape(n, width)
    for row in bad_rows:
        array[row, 0] = np.nan
    return array


class TestResolvePolicy:
    """T4: validation of the on_invalid argument."""

    def test_rejects_anything_else(self):
        """T4: an unknown value names the input and lists what is accepted.

        ``True`` is included because ``bool`` is an ``int`` subclass and a bare
        truthy value must not be read as a policy.
        """
        for bad in ("Raise", "skip", "", None, 0, True, ["drop"]):
            with pytest.raises(ValueError, match="on_invalid must be one of") as exc:
                resolve_policy(bad, method=METHOD, unit=InvalidUnit.ROW)
            message = str(exc.value)
            assert repr(bad) in message
            assert METHOD in message
            for policy in ("raise", "propagate", "drop", "none"):
                assert repr(policy) in message

    def test_drop_refused_where_it_is_undefined(self):
        """T4: refusing 'drop' says which unit could not be removed, and why.

        eFAST is the real case: a search curve is an ordered sweep read by a
        Fourier transform, so removing a point changes what the estimator
        computes rather than shrinking the sample.
        """
        with pytest.raises(ValueError, match="not available for this method") as exc:
            resolve_policy("drop", method=METHOD, unit=InvalidUnit.CURVE, allow_drop=False)
        message = str(exc.value)
        assert "search curve" in message
        assert "'raise'" in message and "'propagate'" in message

    def test_other_policies_survive_allow_drop_false(self):
        """T4: forbidding 'drop' does not forbid the other two."""
        for policy in ("raise", "propagate"):
            assert (
                resolve_policy(policy, method=METHOD, unit=InvalidUnit.CURVE, allow_drop=False)
                == policy
            )


class TestCleanSample:
    """T4: a sample with nothing wrong in it."""

    def test_clean_sample_keeps_everything_under_every_policy(self, recwarn):
        """T4: no finding means no removal, no warning, and an empty report."""
        for policy in ("raise", "propagate", "drop"):
            keep, report = check_invalid(
                policy=policy,
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                Y=_rows(),
                X=_rows(),
            )
            assert keep.all()
            assert report.n_invalid == 0
            assert not report.any_invalid
            assert report.n_kept == 6
            assert report.sources == ()
        assert len(recwarn) == 0


class TestNonePolicy:
    """T4: on_invalid='none' skips the scan entirely."""

    def test_resolve_accepts_none_for_every_unit(self):
        """T4: 'none' survives allow_drop=False, where 'drop' does not."""
        for unit in InvalidUnit:
            assert resolve_policy("none", method=METHOD, unit=unit, allow_drop=False) == "none"

    def test_nan_data_is_not_looked_at(self):
        """T4: bad data under 'none' keeps everything and reports clean.

        The check never runs, so the verdict is the clean one even though two
        rows hold NaN. The keep mask stays all-True, so the caller applies no
        compaction, and the report cannot name rows that were never scanned.
        """
        keep, report = check_invalid(
            policy="none",
            method=METHOD,
            unit=InvalidUnit.ROW,
            n_units=6,
            Y=_rows(2, 5),
            X=_rows(3),
        )
        assert keep.all()
        assert report.policy == "none"
        assert report.n_invalid == 0
        assert not report.any_invalid
        assert report.n_kept == 6
        assert report.sources == ()

    def test_nan_extra_arrays_are_not_looked_at(self):
        """T4: companion arrays are skipped with Y -- no raise, no drop.

        The shape consistency check still runs: an extra array with a
        different row count is a design error, not a data-quality question.
        """
        keep, report = check_invalid(
            policy="none",
            method=METHOD,
            unit=InvalidUnit.SALTELLI_GROUP,
            n_units=2,
            Y=np.arange(12.0).reshape(12, 1),
            extras=[np.full((12, 3), np.nan)],
            unit_of_row=np.repeat(np.arange(2), 6),
        )
        assert keep.all()
        assert report.n_invalid == 0
        with pytest.raises(ValueError, match="same number of sample rows"):
            check_invalid(
                policy="none",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                Y=_rows(),
                extras=[np.zeros((5, 1))],
            )


class TestRaisePolicy:
    """T4: the default policy."""

    def test_raise_names_bad_sources_counts_positions_and_the_way_out(self):
        """T4: the message carries what a user needs to act, not just a count."""
        with pytest.raises(ValueError) as exc:
            check_invalid(
                policy="raise",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                Y=_rows(1, 4),
            )
        message = str(exc.value)
        assert METHOD in message
        assert "2 of 6 rows" in message
        assert "[1, 4]" in message
        assert "on_invalid='drop'" in message
        assert "on_invalid='propagate'" in message
        with pytest.raises(ValueError, match=r"\bin X\b") as x_exc:
            check_invalid(
                policy="raise",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                Y=_rows(),
                X=_rows(2),
            )
        assert "Y" not in str(x_exc.value).split("Non-finite rows")[0].replace("NaN", "")
        with pytest.raises(ValueError, match="X and Y"):
            check_invalid(
                policy="raise",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                Y=_rows(0),
                X=_rows(3),
            )


class TestPropagatePolicy:
    """T4: computing anyway."""

    def test_propagate_keeps_everything_and_warns_even_below_the_floor(self):
        """T4: nothing is removed, and the warning says the indices will be bad."""
        with pytest.warns(JaxgsaWarning, match="reaches the indices"):
            keep, report = check_invalid(
                policy="propagate",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                Y=_rows(2, 5),
            )
        assert keep.all()
        assert report.n_invalid == 2
        assert report.unit_indices == (2, 5)
        # min_kept governs dropping only; propagate removes nothing.
        with pytest.warns(JaxgsaWarning):
            keep, _ = check_invalid(
                policy="propagate",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                Y=_rows(0, 1, 2, 3, 4),
                min_kept=4,
            )
        assert keep.all()


class TestDropPolicy:
    """T4: removing the affected data."""

    def test_drop_masks_exactly_the_bad_rows(self):
        """T4: the returned mask is False at the affected units and True elsewhere.

        The sample is small, so the low-survivor warning fires too. Both are
        expected, and asserting only the first would let a stray third warning
        through unnoticed.
        """
        with pytest.warns(JaxgsaWarning) as record:
            keep, report = check_invalid(
                policy="drop",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                Y=_rows(1, 3),
            )
        messages = [str(w.message) for w in record]
        assert any("dropped 2 of 6 rows" in m for m in messages)
        assert len(messages) == 2
        assert list(keep) == [True, False, True, False, True, True]
        assert report.n_kept == 4

    def test_drop_raises_when_too_little_remains(self):
        """T4: below min_kept the call refuses instead of returning a stub."""
        with pytest.raises(ValueError, match="every usable row was removed") as exc:
            check_invalid(
                policy="drop",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                Y=_rows(0, 1, 2, 3, 4),
                min_kept=4,
            )
        assert "leaving 1" in str(exc.value)
        assert "at least 4" in str(exc.value)


class TestGroupedDesigns:
    """T4: designs where one bad value invalidates a block of rows."""

    def test_one_bad_row_invalidates_its_whole_group(self):
        """T4: a contiguous grouped design removes the group, not the row.

        Four groups of three rows. Row 4 sits in group 1, so group 1 goes and
        rows 3, 4 and 5 go with it.
        """
        unit_of_row = np.repeat(np.arange(4), 3)
        with pytest.warns(JaxgsaWarning):
            keep, report = check_invalid(
                policy="drop",
                method=METHOD,
                unit=InvalidUnit.SALTELLI_GROUP,
                n_units=4,
                Y=_rows(4, n=12),
                unit_of_row=unit_of_row,
            )
        assert list(keep) == [True, False, True, True]
        assert report.unit_indices == (1,)
        assert report.row_indices == (3, 4, 5)

    def test_non_contiguous_units_report_the_right_rows(self):
        """T4: a unit whose rows are strided is still reported correctly.

        Kucherenko's base points appear once per conditional block rather than
        in a contiguous run, so the row report must follow the actual layout
        and not assume a block.
        """
        # 3 blocks of 4 base points: row r belongs to base point r % 4.
        unit_of_row = np.tile(np.arange(4), 3)
        with pytest.warns(JaxgsaWarning):
            _, report = check_invalid(
                policy="drop",
                method=METHOD,
                unit=InvalidUnit.BASE_POINT,
                n_units=4,
                Y=_rows(6, n=12),
                unit_of_row=unit_of_row,
            )
        # Row 6 is base point 2, which also occupies rows 2 and 10.
        assert report.unit_indices == (2,)
        assert report.row_indices == (2, 6, 10)


class TestWhatCountsAsInvalid:
    """T4: which values the check rejects."""

    def test_nan_and_both_infinities_are_all_invalid(self):
        """T4: an infinite output breaks a variance as thoroughly as a NaN."""
        for bad in (np.nan, np.inf, -np.inf):
            Y = _rows()
            Y[2, 1] = bad
            with pytest.raises(ValueError, match="non-finite"):
                check_invalid(policy="raise", method=METHOD, unit=InvalidUnit.ROW, n_units=6, Y=Y)

    def test_a_bad_value_anywhere_in_a_row_condemns_the_row(self):
        """T4: the check flattens trailing axes, so (N, T, K) outputs work."""
        Y = np.ones((5, 3, 2))
        Y[3, 2, 1] = np.nan
        with pytest.raises(ValueError, match="1 of 5 rows"):
            check_invalid(policy="raise", method=METHOD, unit=InvalidUnit.ROW, n_units=5, Y=Y)


class TestBadRowIndices:
    """T4: the rows that actually failed, as opposed to the rows condemned.

    ``row_indices`` answers "what does 'drop' remove?". ``bad_row_indices``
    answers "which model run do I go and look at?". For a grouped design those
    are very different lists, and conflating them was the defect this field
    was added to fix: an eFAST curve is 257 rows and naming all of them tells
    a user nothing.
    """

    def test_for_a_row_design_the_two_lists_agree(self):
        """T4: when one row is one unit there is nothing to distinguish."""
        with pytest.warns(JaxgsaWarning):
            _, report = check_invalid(
                policy="propagate",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                Y=_rows(1, 4),
            )
        assert report.bad_row_indices == (1, 4)
        assert report.bad_row_indices == report.row_indices
        assert report.bad_row_indices == report.unit_indices

    def test_for_a_grouped_design_it_names_only_the_failing_row(self):
        """T4: one bad row in a group of three is reported as one row.

        The group still loses all three rows, and ``row_indices`` still says
        so. Only ``bad_row_indices`` is allowed to shrink.
        """
        unit_of_row = np.repeat(np.arange(4), 3)
        with pytest.warns(JaxgsaWarning):
            _, report = check_invalid(
                policy="propagate",
                method=METHOD,
                unit=InvalidUnit.SALTELLI_GROUP,
                n_units=4,
                Y=_rows(4, n=12),
                unit_of_row=unit_of_row,
            )
        assert report.bad_row_indices == (4,)
        assert report.row_indices == (3, 4, 5)
        assert len(report.bad_row_indices) < len(report.row_indices)


class TestCallerRowNumbering:
    """T4: reported rows are the caller's rows, not the expanded design's.

    Sobol and Morris analyze an expanded design built by indexing the user's
    outputs with ``expanded_to_unique``. Reporting an expanded position points
    at a row the user's array does not have.
    """

    # Twelve expanded rows over four units of three, but only nine distinct
    # runs: the last unit repeats the runs of the first.
    ROW_LABELS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2])
    UNIT_OF_ROW = np.repeat(np.arange(4), 3)

    def test_the_failing_row_is_translated_back(self):
        """T4: expanded row 10 is reported as the caller row it came from."""
        with pytest.warns(JaxgsaWarning):
            _, report = check_invalid(
                policy="propagate",
                method=METHOD,
                unit=InvalidUnit.SALTELLI_GROUP,
                n_units=4,
                Y=_rows(10, n=12),
                unit_of_row=self.UNIT_OF_ROW,
                row_labels=self.ROW_LABELS,
            )
        assert report.unit_indices == (3,)
        assert report.bad_row_indices == (1,)

    def test_the_condemned_rows_are_translated_and_deduplicated(self):
        """T4: a repeated run is named once, in the caller's numbering.

        Unit 3 occupies expanded rows 9, 10 and 11, which are the same three
        model runs as unit 0. The caller has nine rows, so a report naming
        rows 9 to 11 would be unusable.
        """
        with pytest.warns(JaxgsaWarning):
            _, report = check_invalid(
                policy="propagate",
                method=METHOD,
                unit=InvalidUnit.SALTELLI_GROUP,
                n_units=4,
                Y=_rows(10, n=12),
                unit_of_row=self.UNIT_OF_ROW,
                row_labels=self.ROW_LABELS,
            )
        assert report.row_indices == (0, 1, 2)
        assert max(report.row_indices) < len(np.unique(self.ROW_LABELS))


class TestSourceNames:
    """T4: renaming the arrays the report talks about."""

    def test_source_names_replace_the_default_labels(self):
        """T4: a caller checking something other than X and Y can say so.

        DGSM is the real case. On its autodiff path it puts the model output
        and its derivative into one slot, so a report saying "Y" would send a
        reader to an output array that is finite everywhere.
        """
        with pytest.raises(ValueError, match="Y or its derivative") as exc:
            check_invalid(
                policy="raise",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                Y=_rows(2),
                source_names=("X", "Y or its derivative"),
            )
        assert "in Y or its derivative" in str(exc.value)

    def test_renamed_sources_reach_the_report(self):
        """T4: the report carries the caller's names, not the defaults."""
        with pytest.warns(JaxgsaWarning):
            _, report = check_invalid(
                policy="propagate",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                X=_rows(1),
                Y=_rows(3),
                source_names=("inputs", "outputs"),
            )
        assert report.sources == ("inputs", "outputs")
