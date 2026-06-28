"""Tests for PAWN sensitivity analysis."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from gsax.benchmarks import ishigami
from gsax.pawn import PAWNResult, analyze
from gsax.problem import Problem
from gsax.sampling import sample_mc


@pytest.fixture(scope="module")
def ishigami_data():
    """Generate Ishigami test data."""
    N = 5000
    X = jnp.asarray(sample_mc(ishigami.PROBLEM, N=N, seed=42))
    Y = ishigami.evaluate(X)
    return X, Y


class TestPAWNBasic:
    def test_returns_pawn_result(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, seed=0)
        assert isinstance(result, PAWNResult)

    def test_shape_scalar_output(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, seed=0)
        assert result.pawn.shape == (3,)
        assert result.pawn_conf is None

    def test_values_in_unit_interval(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, seed=0)
        pawn = np.asarray(result.pawn)
        assert np.all(pawn >= 0.0)
        assert np.all(pawn <= 1.0)

    def test_x1_x2_more_important_than_x3(self, ishigami_data):
        """x1 and x2 should have higher PAWN than x3 (weak first-order)."""
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, seed=0)
        pawn = np.asarray(result.pawn)
        assert pawn[0] > pawn[2], "x1 should be more important than x3"
        assert pawn[1] > pawn[2], "x2 should be more important than x3"


class TestPAWNSALibComparison:
    def test_matches_salib_median(self, ishigami_data):
        """Compare against SALib's PAWN implementation."""
        from SALib.analyze import pawn as salib_pawn

        X_np = np.asarray(ishigami_data[0])
        Y_np = np.asarray(ishigami_data[1])

        salib_problem = {
            "num_vars": 3,
            "names": ["x1", "x2", "x3"],
            "bounds": [[-np.pi, np.pi]] * 3,
        }
        salib_result = salib_pawn.analyze(
            salib_problem, X_np, Y_np, S=10, seed=0
        )
        salib_median = salib_result["median"]

        gsax_result = analyze(
            ishigami.PROBLEM,
            ishigami_data[0],
            ishigami_data[1],
            n_bins=10,
            statistic="median",
            seed=0,
        )
        gsax_pawn = np.asarray(gsax_result.pawn)

        np.testing.assert_allclose(gsax_pawn, salib_median, atol=0.01)


class TestPAWNStatistics:
    def test_max_statistic(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, statistic="max", seed=0)
        pawn = np.asarray(result.pawn)
        assert np.all(pawn >= 0.0)
        assert np.all(pawn <= 1.0)

    def test_mean_statistic(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, statistic="mean", seed=0)
        pawn = np.asarray(result.pawn)
        assert np.all(pawn >= 0.0)
        assert np.all(pawn <= 1.0)

    def test_max_geq_median(self, ishigami_data):
        X, Y = ishigami_data
        r_median = analyze(ishigami.PROBLEM, X, Y, statistic="median", seed=0)
        r_max = analyze(ishigami.PROBLEM, X, Y, statistic="max", seed=0)
        assert np.all(np.asarray(r_max.pawn) >= np.asarray(r_median.pawn) - 1e-6)


class TestPAWNBootstrap:
    def test_bootstrap_produces_conf(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(
            ishigami.PROBLEM, X, Y, n_bootstrap=20, conf_level=0.95, seed=0
        )
        assert result.pawn_conf is not None
        assert result.pawn_conf.shape == (2, 3)

    def test_bootstrap_lower_leq_upper(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(
            ishigami.PROBLEM, X, Y, n_bootstrap=20, conf_level=0.95, seed=0
        )
        assert result.pawn_conf is not None
        lower = np.asarray(result.pawn_conf[0])
        upper = np.asarray(result.pawn_conf[1])
        assert np.all(lower <= upper + 1e-6)


class TestPAWNMultiOutput:
    def test_multi_output_shape(self):
        problem = Problem(
            names=("x1", "x2", "x3"),
            bounds=((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
        )
        N = 2000
        X = jnp.asarray(sample_mc(problem, N=N, seed=42))
        Y1 = ishigami.evaluate(X)
        Y2 = jnp.sum(X**2, axis=1)
        Y_multi = jnp.column_stack([Y1, Y2])

        result = analyze(problem, X, Y_multi, seed=0)
        assert result.pawn.shape == (2, 3)

    def test_time_series_shape(self):
        problem = Problem(
            names=("x1", "x2", "x3"),
            bounds=((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
        )
        N = 2000
        X = jnp.asarray(sample_mc(problem, N=N, seed=42))
        Y1 = ishigami.evaluate(X)
        Y2 = jnp.sum(X**2, axis=1)
        Y_3d = jnp.stack(
            [jnp.column_stack([Y1, Y2]), jnp.column_stack([Y2, Y1])], axis=1
        )
        assert Y_3d.shape == (N, 2, 2)

        result = analyze(problem, X, Y_3d, seed=0)
        assert result.pawn.shape == (2, 2, 3)


class TestPAWNToDataset:
    def test_scalar_dataset(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, seed=0)
        ds = result.to_dataset()
        assert "pawn" in ds.data_vars
        assert list(ds.coords["param"].values) == list(ishigami.PROBLEM.names)
        assert "output" not in ds.dims

    def test_multi_output_dataset(self):
        problem = Problem(
            names=("x1", "x2", "x3"),
            bounds=((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
        )
        N = 1000
        X = jnp.asarray(sample_mc(problem, N=N, seed=42))
        Y1 = ishigami.evaluate(X)
        Y2 = jnp.sum(X**2, axis=1)
        Y_multi = jnp.column_stack([Y1, Y2])

        result = analyze(problem, X, Y_multi, seed=0)
        ds = result.to_dataset()
        assert "output" in ds.dims
        assert ds["pawn"].shape == (2, 3)

    def test_bootstrap_dataset(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(
            ishigami.PROBLEM, X, Y, n_bootstrap=10, seed=0
        )
        ds = result.to_dataset()
        assert "pawn_lower" in ds.data_vars
        assert "pawn_upper" in ds.data_vars


class TestPAWNValidation:
    def test_x_wrong_ndim(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="2-D"):
            analyze(problem, jnp.ones(10), jnp.ones(10))

    def test_x_wrong_columns(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="columns"):
            analyze(problem, jnp.ones((10, 3)), jnp.ones(10))

    def test_row_mismatch(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="rows"):
            analyze(problem, jnp.ones((10, 1)), jnp.ones(5))
