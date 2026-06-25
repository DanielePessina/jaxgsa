"""Tests for xarray Dataset conversion of SAResult and HDMRResult."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gsax
from gsax.hdmr import HDMRResult
from gsax.problem import Problem
from gsax.sobol._result import SAResult


@pytest.fixture
def problem():
    return Problem.from_dict({"x1": (0, 1), "x2": (0, 1), "x3": (0, 1)})


# ── SAResult tests ──────────────────────────────────────────────────────────


class TestSAResultToDataset:
    def test_1d(self, problem):
        """Scalar output → (param,) dims only."""
        r = SAResult(
            S1=jnp.array([0.1, 0.2, 0.3]),
            ST=jnp.array([0.4, 0.5, 0.6]),
            S2=None,
            problem=problem,
        )
        ds = r.to_dataset()
        assert list(ds.S1.dims) == ["param"]
        assert list(ds.coords["param"].values) == ["x1", "x2", "x3"]
        np.testing.assert_allclose(ds.S1.values, [0.1, 0.2, 0.3])

    def test_2d_default_output_names(self, problem):
        """Multi-output → (output, param) with auto-generated y0, y1."""
        r = SAResult(
            S1=jnp.ones((2, 3)),
            ST=jnp.ones((2, 3)),
            S2=None,
            problem=problem,
        )
        ds = r.to_dataset()
        assert list(ds.S1.dims) == ["output", "param"]
        assert list(ds.coords["output"].values) == ["y0", "y1"]

    def test_3d(self, problem):
        """Time-resolved → (time, output, param) with integer time coords."""
        r = SAResult(
            S1=jnp.ones((4, 2, 3)),
            ST=jnp.ones((4, 2, 3)),
            S2=None,
            problem=problem,
        )
        ds = r.to_dataset()
        assert list(ds.S1.dims) == ["time", "output", "param"]
        assert list(ds.coords["time"].values) == [0, 1, 2, 3]

    def test_custom_output_names(self):
        """Problem with explicit output_names passes through."""
        p = Problem(
            names=("x1", "x2"),
            bounds=((0, 1), (0, 1)),
            output_names=("temp", "pressure"),
        )
        r = SAResult(
            S1=jnp.ones((2, 2)),
            ST=jnp.ones((2, 2)),
            S2=None,
            problem=p,
        )
        ds = r.to_dataset()
        assert list(ds.coords["output"].values) == ["temp", "pressure"]

    def test_custom_time_coords(self, problem):
        """Float time coords passed to to_dataset()."""
        r = SAResult(
            S1=jnp.ones((3, 2, 3)),
            ST=jnp.ones((3, 2, 3)),
            S2=None,
            problem=problem,
        )
        ds = r.to_dataset(time_coords=[0.0, 0.5, 1.0])
        np.testing.assert_allclose(ds.coords["time"].values, [0.0, 0.5, 1.0])

    def test_s2_dims(self, problem):
        """S2 uses (param_i, param_j) for last two dims."""
        D = 3
        r = SAResult(
            S1=jnp.ones(D),
            ST=jnp.ones(D),
            S2=jnp.ones((D, D)),
            problem=problem,
        )
        ds = r.to_dataset()
        assert list(ds.S2.dims) == ["param_i", "param_j"]
        assert list(ds.coords["param_i"].values) == ["x1", "x2", "x3"]

    def test_confidence_intervals(self, problem):
        """CIs split into _lower/_upper variables."""
        D = 3
        r = SAResult(
            S1=jnp.ones(D),
            ST=jnp.ones(D),
            S2=None,
            problem=problem,
            S1_conf=jnp.stack([jnp.zeros(D), jnp.ones(D)]),
            ST_conf=jnp.stack([jnp.zeros(D), 2 * jnp.ones(D)]),
        )
        ds = r.to_dataset()
        assert "S1_lower" in ds
        assert "S1_upper" in ds
        np.testing.assert_allclose(ds.S1_lower.values, 0.0)
        np.testing.assert_allclose(ds.S1_upper.values, 1.0)
        np.testing.assert_allclose(ds.ST_upper.values, 2.0)

    def test_s2_confidence(self, problem):
        """S2 CIs also split correctly."""
        D = 3
        r = SAResult(
            S1=jnp.ones(D),
            ST=jnp.ones(D),
            S2=jnp.ones((D, D)),
            problem=problem,
            S1_conf=jnp.stack([jnp.zeros(D), jnp.ones(D)]),
            ST_conf=jnp.stack([jnp.zeros(D), jnp.ones(D)]),
            S2_conf=jnp.stack([jnp.zeros((D, D)), jnp.ones((D, D))]),
        )
        ds = r.to_dataset()
        assert "S2_lower" in ds
        assert "S2_upper" in ds
        assert list(ds.S2_lower.dims) == ["param_i", "param_j"]

    def test_output_names_length_mismatch(self):
        """Mismatched output_names raises ValueError."""
        p = Problem(
            names=("x1", "x2"),
            bounds=((0, 1), (0, 1)),
            output_names=("temp",),  # only 1, but K=2
        )
        r = SAResult(
            S1=jnp.ones((2, 2)),
            ST=jnp.ones((2, 2)),
            S2=None,
            problem=p,
        )
        with pytest.raises(ValueError, match="output_names length"):
            r.to_dataset()

    @pytest.mark.parametrize("ci_method", ["quantile", "gaussian"])
    def test_analyze_bootstrap_export_preserves_lower_upper_variables(self, ci_method):
        """Real bootstrap analyze() output exports lower/upper CI variables."""
        problem = Problem.from_dict(
            {"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)},
            output_names=("response",),
        )
        sampling_result = gsax.sample(
            problem,
            n_samples=256,
            calc_second_order=True,
            seed=7,
            verbose=False,
        )
        X = jnp.asarray(sampling_result.samples)
        Y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2 + X[:, 0] * X[:, 2]

        result = gsax.analyze(
            sampling_result,
            Y,
            num_resamples=20,
            conf_level=0.9,
            ci_method=ci_method,
            key=jax.random.key(123),
        )
        assert result.S1_conf is not None
        assert result.ST_conf is not None
        assert result.S2_conf is not None
        ds = result.to_dataset()

        assert "S1_lower" in ds
        assert "S1_upper" in ds
        assert "ST_lower" in ds
        assert "ST_upper" in ds
        assert "S2_lower" in ds
        assert "S2_upper" in ds

        assert list(ds.S1_lower.dims) == ["param"]
        assert list(ds.ST_upper.dims) == ["param"]
        assert list(ds.S2_lower.dims) == ["param_i", "param_j"]

        np.testing.assert_allclose(ds.S1.values, np.asarray(result.S1))
        np.testing.assert_allclose(ds.S1_lower.values, np.asarray(result.S1_conf[0]))
        np.testing.assert_allclose(ds.S1_upper.values, np.asarray(result.S1_conf[1]))
        np.testing.assert_allclose(ds.ST_lower.values, np.asarray(result.ST_conf[0]))
        np.testing.assert_allclose(ds.ST_upper.values, np.asarray(result.ST_conf[1]))
        np.testing.assert_allclose(
            ds.S2_lower.values,
            np.asarray(result.S2_conf[0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            ds.S2_upper.values,
            np.asarray(result.S2_conf[1]),
            equal_nan=True,
        )
        assert list(ds.coords["param"].values) == ["x1", "x2", "x3"]
        assert list(ds.coords["param_i"].values) == ["x1", "x2", "x3"]
        assert list(ds.coords["param_j"].values) == ["x1", "x2", "x3"]


# ── HDMRResult tests ────────────────────────────────────────────────────────


class TestHDMRResultToDataset:
    def test_basic(self, problem):
        """Term-indexed Sa/Sb/S + param-indexed ST."""
        terms = ("x1", "x2", "x3", "x1/x2")
        n_terms = len(terms)
        D = 3
        r = HDMRResult(
            Sa=jnp.ones(n_terms),
            Sb=jnp.zeros(n_terms),
            S=jnp.ones(n_terms),
            ST=jnp.ones(D),
            problem=problem,
            terms=terms,
        )
        ds = r.to_dataset()
        assert list(ds.Sa.dims) == ["term"]
        assert list(ds.ST.dims) == ["param"]
        assert list(ds.coords["term"].values) == list(terms)
        assert list(ds.coords["param"].values) == ["x1", "x2", "x3"]

    def test_with_select_and_rmse(self, problem):
        """select and rmse are included when present."""
        terms = ("x1", "x2", "x3")
        r = HDMRResult(
            Sa=jnp.ones(3),
            Sb=jnp.zeros(3),
            S=jnp.ones(3),
            ST=jnp.ones(3),
            problem=problem,
            terms=terms,
            select=jnp.array([1, 0, 1]),
            rmse=jnp.array(0.05),
        )
        ds = r.to_dataset()
        assert "select" in ds
        assert "rmse" in ds
        assert ds.select.dims == ("term",)
        assert ds.rmse.dims == ()

    def test_2d(self, problem):
        """Multi-output HDMR → (output, term) / (output, param)."""
        terms = ("x1", "x2", "x3")
        r = HDMRResult(
            Sa=jnp.ones((2, 3)),
            Sb=jnp.zeros((2, 3)),
            S=jnp.ones((2, 3)),
            ST=jnp.ones((2, 3)),
            problem=problem,
            terms=terms,
        )
        ds = r.to_dataset()
        assert list(ds.Sa.dims) == ["output", "term"]
        assert list(ds.ST.dims) == ["output", "param"]


# ── Problem tests ───────────────────────────────────────────────────────────


class TestProblemOutputNames:
    def test_field_stored(self):
        p = Problem(
            names=("x1",),
            bounds=((0, 1),),
            output_names=("temp",),
        )
        assert p.output_names == ("temp",)

    def test_from_dict_output_names(self):
        p = Problem.from_dict(
            {"x1": (0, 1)},
            output_names=("temp", "pressure"),
        )
        assert p.output_names == ("temp", "pressure")

    def test_default_none(self):
        p = Problem.from_dict({"x1": (0, 1)})
        assert p.output_names is None
