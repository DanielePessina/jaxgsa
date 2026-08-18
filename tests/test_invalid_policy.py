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

    @pytest.mark.parametrize("bad", ["Raise", "skip", "", None, 0, True, ["drop"]])
    def test_rejects_anything_else(self, bad):
        """T4: an unknown value names the input and lists what is accepted.

        ``True`` is included because ``bool`` is an ``int`` subclass and a bare
        truthy value must not be read as a policy.
        """
        with pytest.raises(ValueError, match="on_invalid must be one of") as exc:
            resolve_policy(bad, method=METHOD, unit=InvalidUnit.ROW)
        message = str(exc.value)
        assert repr(bad) in message
        assert METHOD in message
        for policy in ("raise", "propagate", "drop"):
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

    @pytest.mark.parametrize("policy", ["raise", "propagate"])
    def test_other_policies_survive_allow_drop_false(self, policy):
        """T4: forbidding 'drop' does not forbid the other two."""
        assert (
            resolve_policy(policy, method=METHOD, unit=InvalidUnit.CURVE, allow_drop=False)
            == policy
        )


class TestCleanSample:
    """T4: a sample with nothing wrong in it."""

    @pytest.mark.parametrize("policy", ["raise", "propagate", "drop"])
    def test_clean_sample_keeps_everything_under_every_policy(self, policy, recwarn):
        """T4: no finding means no removal, no warning, and an empty report."""
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


class TestRaisePolicy:
    """T4: the default policy."""

    def test_raise_names_counts_positions_and_the_way_out(self):
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

    def test_raise_distinguishes_x_from_y(self):
        """T4: the message says which array held the bad value."""
        with pytest.raises(ValueError, match=r"\bin X\b") as exc:
            check_invalid(
                policy="raise",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                Y=_rows(),
                X=_rows(2),
            )
        assert "Y" not in str(exc.value).split("Affected")[0].replace("NaN", "")

    def test_raise_reports_both_arrays_when_both_are_bad(self):
        """T4: bad values in X and in Y are both named."""
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

    def test_propagate_keeps_everything_and_warns(self):
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

    def test_propagate_never_raises_on_a_small_survivor_count(self):
        """T4: min_kept governs dropping only; propagate removes nothing.

        Under propagate the sample is untouched, so 'too little data remains'
        cannot apply. Enforcing the floor here would refuse a call that was
        explicitly asked to compute anyway.
        """
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

    def test_drop_warns_twice_when_few_survive(self):
        """T4: a second warning fires when the surviving sample is tiny.

        The first warning says what was removed. The second says the result is
        not worth trusting, which is a different fact and easy to miss if the
        two are merged.
        """
        Y = _rows(*range(9), n=12)
        with pytest.warns(JaxgsaWarning) as record:
            check_invalid(
                policy="drop",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=12,
                Y=Y,
            )
        messages = [str(w.message) for w in record]
        assert any("dropped 9 of 12" in m for m in messages)
        assert any("not reliable" in m for m in messages)

    def test_a_bad_input_takes_its_output_with_it(self):
        """T4: X and Y are checked jointly, so the pair stays aligned.

        Dropping a row of X without its matching row of Y would misalign every
        later row, which the estimator cannot detect.
        """
        with pytest.warns(JaxgsaWarning):
            keep, report = check_invalid(
                policy="drop",
                method=METHOD,
                unit=InvalidUnit.ROW,
                n_units=6,
                Y=_rows(1),
                X=_rows(4),
            )
        assert list(keep) == [True, False, True, True, False, True]
        assert report.sources == ("X", "Y")


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

    def test_grouped_message_lists_both_units_and_rows(self):
        """T4: for a grouped design the message gives group and row positions.

        A user reads group indices to understand the design and row indices to
        find the model evaluation, so a grouped failure needs both.
        """
        unit_of_row = np.repeat(np.arange(4), 3)
        with pytest.raises(ValueError) as exc:
            check_invalid(
                policy="raise",
                method=METHOD,
                unit=InvalidUnit.TRAJECTORY,
                n_units=4,
                Y=_rows(7, n=12),
                unit_of_row=unit_of_row,
            )
        message = str(exc.value)
        assert "Affected trajectories: [2]" in message
        assert "Rows: [6, 7, 8]" in message


class TestWhatCountsAsInvalid:
    """T4: which values the check rejects."""

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_nan_and_both_infinities_are_all_invalid(self, bad):
        """T4: an infinite output breaks a variance as thoroughly as a NaN."""
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

    def test_skipped_arrays_do_not_make_a_sample_dirty(self):
        """T4: passing None for X checks Y alone."""
        keep, report = check_invalid(
            policy="raise",
            method=METHOD,
            unit=InvalidUnit.ROW,
            n_units=6,
            Y=_rows(),
            X=None,
        )
        assert keep.all()
        assert report.sources == ()


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
