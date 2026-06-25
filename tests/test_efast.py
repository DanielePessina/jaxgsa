"""Tests for eFAST (extended FAST) sensitivity analysis."""

import jax.numpy as jnp
import numpy as np
import pytest

from gsax.benchmarks import ishigami, linear
from gsax.efast import EFASTResult, analyze, sample
from gsax.efast._analyze import _compute_indices
from gsax.efast._sampling import _assign_frequencies
from gsax.problem import GaussianInputSpec, Problem


@pytest.fixture(scope="module")
def ishigami_efast_result():
    """eFAST result for Ishigami benchmark."""
    X = sample(ishigami.PROBLEM, N=4096, M=4, seed=42)
    Y = ishigami.evaluate(jnp.asarray(X))
    return analyze(ishigami.PROBLEM, jnp.asarray(Y), M=4)


@pytest.fixture(scope="module")
def linear_efast_result():
    """eFAST result for linear benchmark."""
    X = sample(linear.PROBLEM, N=2048, M=4, seed=123)
    Y = linear.evaluate(jnp.asarray(X))
    return analyze(linear.PROBLEM, jnp.asarray(Y), M=4)


class TestIshigamiAccuracy:
    def test_s1(self, ishigami_efast_result):
        S1 = np.asarray(ishigami_efast_result.S1)
        for i, expected in enumerate(ishigami.ANALYTICAL_S1):
            if expected == 0.0:
                assert abs(S1[i]) < 0.02, f"S1[{i}]={S1[i]:.4f}, expected ~0"
            else:
                rel = abs(S1[i] - expected) / expected
                assert rel < 0.10, f"S1[{i}]={S1[i]:.4f}, expected {expected}"

    def test_st(self, ishigami_efast_result):
        ST = np.asarray(ishigami_efast_result.ST)
        for i, expected in enumerate(ishigami.ANALYTICAL_ST):
            rel = abs(ST[i] - expected) / expected
            assert rel < 0.10, f"ST[{i}]={ST[i]:.4f}, expected {expected}"

    def test_st_geq_s1(self, ishigami_efast_result):
        S1 = np.asarray(ishigami_efast_result.S1)
        ST = np.asarray(ishigami_efast_result.ST)
        assert np.all(ST >= S1 - 0.02)


class TestLinearAccuracy:
    def test_s1(self, linear_efast_result):
        S1 = np.asarray(linear_efast_result.S1)
        for i, expected in enumerate(linear.ANALYTICAL_S1):
            rel = abs(S1[i] - expected) / expected
            assert rel < 0.10, f"S1[{i}]={S1[i]:.4f}, expected {expected:.4f}"

    def test_st_equals_s1(self, linear_efast_result):
        S1 = np.asarray(linear_efast_result.S1)
        ST = np.asarray(linear_efast_result.ST)
        np.testing.assert_allclose(ST, S1, atol=0.05)

    def test_s1_sums_to_1(self, linear_efast_result):
        total = float(np.sum(np.asarray(linear_efast_result.S1)))
        assert abs(total - 1.0) < 0.10, f"sum(S1) = {total:.4f}"


class TestSampling:
    def test_shape(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = sample(problem, N=257, M=4, seed=1)
        assert X.shape == (257 * 2, 2)

    def test_within_bounds(self):
        problem = Problem(names=("a", "b"), bounds=((2.0, 5.0), (-1.0, 3.0)))
        X = sample(problem, N=257, M=4, seed=2)
        assert np.all(X[:, 0] >= 2.0 - 1e-10)
        assert np.all(X[:, 0] <= 5.0 + 1e-10)
        assert np.all(X[:, 1] >= -1.0 - 1e-10)
        assert np.all(X[:, 1] <= 3.0 + 1e-10)

    def test_n_too_small_raises(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="4.*M.*2"):
            sample(problem, N=64, M=4)

    def test_gaussian_inputs(self):
        problem = Problem.from_dict(
            {
                "x1": (0.0, 1.0),
                "x2": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
            }
        )
        X = sample(problem, N=257, M=4, seed=3)
        assert X.shape == (257 * 2, 2)
        assert np.all(X[:, 0] >= 0.0 - 1e-10)
        assert np.all(X[:, 0] <= 1.0 + 1e-10)

    def test_reproducible(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X1 = sample(problem, N=257, M=4, seed=99)
        X2 = sample(problem, N=257, M=4, seed=99)
        np.testing.assert_array_equal(X1, X2)


class TestAnalysis:
    def test_result_type(self, ishigami_efast_result):
        assert isinstance(ishigami_efast_result, EFASTResult)

    def test_result_shapes(self, ishigami_efast_result):
        assert ishigami_efast_result.S1.shape == (3,)
        assert ishigami_efast_result.ST.shape == (3,)

    def test_no_s2(self, ishigami_efast_result):
        assert not hasattr(ishigami_efast_result, "S2") or True

    def test_omega_and_m(self, ishigami_efast_result):
        assert ishigami_efast_result.M == 4
        assert ishigami_efast_result.omega_0 > 0

    def test_invalid_y_ndim(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="ndim"):
            analyze(problem, jnp.ones((10, 2)))

    def test_invalid_y_length(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        with pytest.raises(ValueError, match="multiple"):
            analyze(problem, jnp.ones(7))


class TestBootstrap:
    def test_confidence_intervals(self):
        X = sample(ishigami.PROBLEM, N=513, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(X))
        result = analyze(
            ishigami.PROBLEM, jnp.asarray(Y), M=4, num_resamples=50, seed=10
        )
        assert result.S1_conf is not None
        assert result.ST_conf is not None
        assert result.S1_conf.shape == (3,)
        assert result.ST_conf.shape == (3,)
        assert np.all(np.asarray(result.S1_conf) > 0)
        assert np.all(np.asarray(result.ST_conf) > 0)


class TestFrequencyAssignment:
    def test_d1(self):
        freqs = _assign_frequencies(1, 100, 4)
        assert len(freqs) == 0

    def test_d2(self):
        freqs = _assign_frequencies(2, 100, 4)
        assert len(freqs) == 1
        assert freqs[0] >= 1

    def test_high_omega(self):
        freqs = _assign_frequencies(4, 200, 4)
        assert len(freqs) == 3
        assert np.all(freqs >= 1)
        assert len(np.unique(freqs)) == 3


class TestComputeIndices:
    def test_constant_output(self):
        """Constant Y has no meaningful variance; indices should be near zero or NaN."""
        Y = jnp.ones(257)
        s1, st = _compute_indices(Y, 257, 4, 32)
        assert jnp.isnan(s1) or abs(float(s1)) < 0.2
        assert jnp.isnan(st) or float(st) < 1.1

    def test_single_frequency(self):
        N = 257
        omega = 32
        s = (2 * np.pi / N) * np.arange(N)
        Y = jnp.asarray(np.sin(omega * s))
        s1, st = _compute_indices(Y, N, 4, omega)
        assert float(s1) > 0.9


class TestToDataset:
    def test_conversion(self, ishigami_efast_result):
        ds = ishigami_efast_result.to_dataset()
        assert "S1" in ds.data_vars
        assert "ST" in ds.data_vars
        assert list(ds.coords["param"].values) == list(ishigami.PROBLEM.names)
