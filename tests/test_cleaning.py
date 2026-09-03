"""Tests for data cleaning, division guards, and NaN reporting."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import JaxgsaWarning
from jaxgsa._core.invalid import InvalidUnit
from jaxgsa.problem import Problem


@pytest.fixture()
def simple_problem():
    return Problem(
        names=("x1", "x2", "x3"),
        bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
    )


# --- End-to-end: constant Y → NaN indices ---


def test_constant_y_produces_nan_indices(simple_problem):
    """A constant output has zero variance, so every index divides by zero."""
    sr = jaxgsa.sobol.sample(
        simple_problem, n_samples=64, seed=0, calc_second_order=False, verbose=False
    )
    Y = jnp.ones(sr.n_runs)
    with pytest.warns(UserWarning, match="zero variance"):
        result = jaxgsa.sobol.analyze(sr, Y)
    assert jnp.all(jnp.isnan(result.S1))
    assert jnp.all(jnp.isnan(result.ST))
    # A constant Y is finite, so the non-finite check has nothing to report.
    assert result.invalid.n_invalid == 0


def test_constant_y_second_order_nan(simple_problem):
    """Zero variance carries through to the second-order indices too."""
    with warnings.catch_warnings():
        # n_samples=64 at D=3 resolves to base_n=8; the degenerate-design
        # warning is not this test's subject.
        warnings.simplefilter("ignore", JaxgsaWarning)
        sr = jaxgsa.sobol.sample(
            simple_problem, n_samples=64, seed=0, calc_second_order=True, verbose=False
        )
    Y = jnp.ones(sr.n_runs)
    with pytest.warns(UserWarning, match="zero variance"):
        result = jaxgsa.sobol.analyze(sr, Y)
    assert jnp.all(jnp.isnan(result.S2))


def test_empty_y_raises_a_clear_value_error(simple_problem):
    """An empty output must be refused up front, not crash in the estimators."""
    sr = jaxgsa.sobol.sample(
        simple_problem, n_samples=64, seed=0, calc_second_order=False, verbose=False
    )
    with pytest.raises(ValueError, match="Y must have at least one row"):
        jaxgsa.sobol.analyze(sr, jnp.zeros((0,)))


# --- Input cleaning: non-finite values ---


def test_all_nonfinite_raises(simple_problem):
    sr = jaxgsa.sobol.sample(
        simple_problem, n_samples=64, seed=0, calc_second_order=False, verbose=False
    )
    Y = jnp.full(sr.n_runs, jnp.nan)
    with pytest.raises(ValueError, match="every usable Saltelli group was removed"):
        jaxgsa.sobol.analyze(sr, Y, on_invalid="drop")


# --- The shared on_invalid policy, at Saltelli-group granularity ---


def _finite_Y(sr) -> np.ndarray:
    """Evaluate a smooth model on the unique design rows."""
    return np.sin(np.sum(np.asarray(sr.samples), axis=1))


def _poison_one_group(sr, Y: np.ndarray, group: int) -> tuple[np.ndarray, list[int]]:
    """Set one NaN that condemns exactly the given Saltelli group.

    The design is deduplicated, so one unique row can feed several expanded
    rows. Picking a unique row that feeds only one expanded row keeps the
    damage inside a single group, which is what the granularity test needs.

    Returns:
        The poisoned copy of ``Y``, and the expanded rows of that group.
    """
    e2u = np.asarray(sr.expanded_to_unique)
    counts = np.bincount(e2u, minlength=Y.shape[0])
    step = sr.n_expanded // sr.base_n
    rows = list(range(group * step, (group + 1) * step))
    for row in rows:
        if counts[e2u[row]] == 1:
            poisoned = Y.copy()
            poisoned[e2u[row]] = np.nan
            return poisoned, rows
    raise AssertionError(f"no uniquely-mapped row in Saltelli group {group}")


class TestOnInvalidPolicy:
    """T4 (behavioural): sobol.analyze applies the shared non-finite policy."""

    def _sample(self, problem):
        return jaxgsa.sobol.sample(
            problem, n_samples=64, seed=0, calc_second_order=False, verbose=False
        )

    def test_raise_is_the_default_and_names_units_and_rows(self, simple_problem):
        """T4: the default refuses, and says which group and which rows."""
        sr = self._sample(simple_problem)
        Y, rows = _poison_one_group(sr, _finite_Y(sr), group=2)
        with pytest.raises(ValueError, match=f"1 of {sr.base_n} Saltelli groups") as exc:
            jaxgsa.sobol.analyze(sr, jnp.asarray(Y))
        message = str(exc.value)
        assert "jaxgsa.sobol.analyze" in message
        assert "They condemn Saltelli groups [2]" in message
        assert f"which covers {len(rows)} rows" in message

    def test_propagate_warns_and_lets_the_value_reach_the_indices(self, simple_problem):
        """T4: nothing is removed, and the indices come back non-finite."""
        sr = self._sample(simple_problem)
        Y, _ = _poison_one_group(sr, _finite_Y(sr), group=2)
        with pytest.warns(jaxgsa.JaxgsaWarning, match="reaches the indices"):
            result = jaxgsa.sobol.analyze(sr, jnp.asarray(Y), on_invalid="propagate")
        assert not np.all(np.isfinite(np.asarray(result.S1)))
        assert result.invalid.policy == "propagate"
        assert result.invalid.n_invalid == 1
        assert result.invalid.unit_indices == (2,)

    def test_drop_removes_the_group_and_leaves_finite_indices(self, simple_problem):
        """T4: the surviving groups give a finite estimate."""
        sr = self._sample(simple_problem)
        Y, _ = _poison_one_group(sr, _finite_Y(sr), group=2)
        with pytest.warns(jaxgsa.JaxgsaWarning, match="dropped 1 of 16 Saltelli groups"):
            result = jaxgsa.sobol.analyze(sr, jnp.asarray(Y), on_invalid="drop")
        assert np.all(np.isfinite(np.asarray(result.S1)))
        assert result.invalid.n_kept == sr.base_n - 1

    def test_one_bad_value_condemns_its_whole_group(self, simple_problem):
        """T4: the unit is the group, not the row.

        One NaN sits in one expanded row. The report must name every row of
        that group, because a row-level drop would shift the A/B/AB split of
        every later group and corrupt the estimate with no error.
        """
        sr = self._sample(simple_problem)
        Y, rows = _poison_one_group(sr, _finite_Y(sr), group=5)
        with pytest.warns(jaxgsa.JaxgsaWarning):
            result = jaxgsa.sobol.analyze(sr, jnp.asarray(Y), on_invalid="drop")
        assert result.invalid.unit_indices == (5,)
        assert result.invalid.row_indices == tuple(rows)
        assert len(rows) == sr.n_expanded // sr.base_n

    def test_drop_matches_a_design_that_never_held_the_group(self, simple_problem):
        """T4: dropping group 5 equals analyzing the survivors by hand.

        The reference rebuilds the expanded outputs, deletes the group, and
        runs the Jansen total-order formula over the remainder, so it checks
        the compaction rather than restating it.
        """
        sr = self._sample(simple_problem)
        Y = _finite_Y(sr)
        poisoned, rows = _poison_one_group(sr, Y, group=5)
        with pytest.warns(jaxgsa.JaxgsaWarning):
            dropped = jaxgsa.sobol.analyze(sr, jnp.asarray(poisoned), on_invalid="drop")

        expanded = Y[np.asarray(sr.expanded_to_unique)]
        kept = np.delete(expanded, rows, axis=0)
        A = kept[0::5]
        B = kept[4::5]
        AB = np.stack([kept[j::5] for j in (1, 2, 3)], axis=1)
        # The estimators pool A and B for the variance; see
        # sobol._estimators._pooled_inv_var.
        var = np.var(np.concatenate([A, B]))
        expected_st = 0.5 * np.mean((A[:, None] - AB) ** 2, axis=0) / var
        np.testing.assert_allclose(np.asarray(dropped.ST), expected_st, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("policy", ["raise", "propagate", "drop"])
    def test_a_clean_sample_reports_nothing_and_stays_silent(
        self, simple_problem, policy, recwarn
    ):
        """T4: a clean run gives an empty report under every policy."""
        sr = self._sample(simple_problem)
        result = jaxgsa.sobol.analyze(sr, jnp.asarray(_finite_Y(sr)), on_invalid=policy)
        assert result.invalid.n_invalid == 0
        assert not result.invalid.any_invalid
        assert result.invalid.unit is InvalidUnit.SALTELLI_GROUP
        assert [w for w in recwarn if issubclass(w.category, jaxgsa.JaxgsaWarning)] == []
