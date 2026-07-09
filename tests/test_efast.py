"""Tests for eFAST (extended FAST) sensitivity analysis."""

import jax.numpy as jnp
import numpy as np
import pytest

from gsax.benchmarks import ishigami, linear, sobol_g
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
        assert set(vars(ishigami_efast_result).keys()) == {"S1", "ST", "problem", "omega_0", "M"}

    def test_omega_and_m(self, ishigami_efast_result):
        assert ishigami_efast_result.M == 4
        assert ishigami_efast_result.omega_0 > 0

    def test_invalid_y_length(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        with pytest.raises(ValueError, match="multiple"):
            analyze(problem, jnp.ones(7))

    def test_invalid_m_raises(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = sample(problem, N=257, M=4, seed=1)
        Y = jnp.ones(X.shape[0])
        with pytest.raises(ValueError, match="M must be >= 1"):
            analyze(problem, Y, M=0)


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
        assert jnp.isnan(s1) or abs(float(s1)) < 0.1
        assert jnp.isnan(st) or float(st) < 1.0

    def test_single_frequency(self):
        N = 257
        omega = 32
        s = (2 * np.pi / N) * np.arange(N)
        Y = jnp.asarray(np.sin(omega * s))
        s1, st = _compute_indices(Y, N, 4, omega)
        assert float(s1) > 0.9


class TestMultiOutputShapes:
    """Tests for multi-output and time-series Y shapes."""

    @pytest.fixture(scope="class")
    def ishigami_multi(self):
        """Ishigami scalar result and multi-output Y (K=3)."""
        X = sample(ishigami.PROBLEM, N=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(X))
        scalar_result = analyze(ishigami.PROBLEM, jnp.asarray(Y), M=4)
        Y_multi = jnp.stack([Y, 2 * Y, 0.5 * Y], axis=-1)
        return Y, Y_multi, scalar_result

    def test_2d_multi_output(self, ishigami_multi):
        """Y shape (N*D, K) produces S1/ST shape (K, D)."""
        _, Y_multi, _ = ishigami_multi
        result = analyze(ishigami.PROBLEM, Y_multi, M=4)
        D = ishigami.PROBLEM.num_vars
        K = 3
        assert result.S1.shape == (K, D)
        assert result.ST.shape == (K, D)

    def test_3d_time_series(self, ishigami_multi):
        """Y shape (N*D, T, K) produces S1/ST shape (T, K, D)."""
        Y_scalar, _, _ = ishigami_multi
        T, K = 5, 2
        slices = []
        for t in range(T):
            scale_0 = float(t + 1)
            scale_1 = float(t + 1) * 0.5
            slices.append(jnp.stack([scale_0 * Y_scalar, scale_1 * Y_scalar], axis=-1))
        Y_3d = jnp.stack(slices, axis=1)  # (N*D, T, K)
        result = analyze(ishigami.PROBLEM, Y_3d, M=4)
        D = ishigami.PROBLEM.num_vars
        assert result.S1.shape == (T, K, D)
        assert result.ST.shape == (T, K, D)

    def test_scalar_unchanged(self, ishigami_multi):
        """1-D Y still produces (D,) indices."""
        Y_scalar, _, scalar_result = ishigami_multi
        D = ishigami.PROBLEM.num_vars
        assert scalar_result.S1.shape == (D,)
        assert scalar_result.ST.shape == (D,)


class TestMultiOutputAccuracy:
    """Scaled copies of Ishigami must produce the same indices as scalar."""

    def test_multi_output_accuracy(self):
        X = sample(ishigami.PROBLEM, N=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(X))
        scalar_result = analyze(ishigami.PROBLEM, jnp.asarray(Y), M=4)
        Y_multi = jnp.stack([Y, 2 * Y, 0.5 * Y], axis=-1)
        multi_result = analyze(ishigami.PROBLEM, Y_multi, M=4)
        np.testing.assert_allclose(multi_result.S1[0], scalar_result.S1, atol=1e-5)
        np.testing.assert_allclose(multi_result.ST[0], scalar_result.ST, atol=1e-5)


class TestPrenormalize:
    """Prenormalize should handle large constant offsets."""

    def test_prenormalize(self):
        X = sample(ishigami.PROBLEM, N=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(X))
        baseline = analyze(ishigami.PROBLEM, jnp.asarray(Y), M=4)
        Y_shifted = Y + 1e6
        result = analyze(ishigami.PROBLEM, jnp.asarray(Y_shifted), M=4, prenormalize=True)
        np.testing.assert_allclose(np.asarray(result.S1), np.asarray(baseline.S1), atol=0.02)
        np.testing.assert_allclose(np.asarray(result.ST), np.asarray(baseline.ST), atol=0.02)


class TestChunkSize:
    """Chunk size should not affect results."""

    def test_chunk_size(self):
        X = sample(ishigami.PROBLEM, N=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(X))
        Y_multi = jnp.stack([Y, 2 * Y, 0.5 * Y], axis=-1)
        default_result = analyze(ishigami.PROBLEM, Y_multi, M=4)
        chunked_result = analyze(ishigami.PROBLEM, Y_multi, M=4, chunk_size=1)
        np.testing.assert_allclose(
            np.asarray(chunked_result.S1),
            np.asarray(default_result.S1),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(chunked_result.ST),
            np.asarray(default_result.ST),
            atol=1e-6,
        )


class TestToDatasetMultiOutput:
    """Tests for to_dataset with multi-output and time-series results."""

    def test_2d_dataset(self):
        """Multi-output result exports with ('output', 'param') dims."""
        X = sample(ishigami.PROBLEM, N=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(X))
        Y_multi = jnp.stack([Y, 2 * Y, 0.5 * Y], axis=-1)
        result = analyze(ishigami.PROBLEM, Y_multi, M=4)
        ds = result.to_dataset()
        assert set(ds["S1"].dims) == {"output", "param"}
        assert list(ds.coords["param"].values) == list(ishigami.PROBLEM.names)
        assert len(ds.coords["output"]) == 3

    def test_3d_dataset_with_time_coords(self):
        """Time-series result exports with ('time', 'output', 'param') dims."""
        X = sample(ishigami.PROBLEM, N=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(X))
        T = 5
        slices = []
        for t in range(T):
            slices.append(jnp.stack([float(t + 1) * Y, float(t + 1) * 0.5 * Y], axis=-1))
        Y_3d = jnp.stack(slices, axis=1)
        result = analyze(ishigami.PROBLEM, Y_3d, M=4)
        time_coords = [0.0, 0.1, 0.2, 0.3, 0.4]
        ds = result.to_dataset(time_coords=time_coords)
        assert set(ds["S1"].dims) == {"time", "output", "param"}
        assert list(ds.coords["time"].values) == time_coords
        assert list(ds.coords["param"].values) == list(ishigami.PROBLEM.names)

    def test_repr(self):
        """repr() includes shape info."""
        X = sample(ishigami.PROBLEM, N=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(X))
        Y_multi = jnp.stack([Y, 2 * Y], axis=-1)
        result = analyze(ishigami.PROBLEM, Y_multi, M=4)
        r = repr(result)
        assert "EFASTResult" in r
        assert "S1" in r


class TestToDataset:
    def test_conversion(self, ishigami_efast_result):
        ds = ishigami_efast_result.to_dataset()
        assert "S1" in ds.data_vars
        assert "ST" in ds.data_vars
        assert list(ds.coords["param"].values) == list(ishigami.PROBLEM.names)


class TestSobolGAccuracy:
    """Validate eFAST against the Sobol G-function benchmark (8-D)."""

    @pytest.fixture(scope="module")
    def sobol_g_result(self):
        """eFAST result for Sobol G-function benchmark."""
        X = sample(sobol_g.PROBLEM, N=4096, M=4, seed=42)
        Y = sobol_g.evaluate(jnp.asarray(X))
        return analyze(sobol_g.PROBLEM, jnp.asarray(Y), M=4)

    def test_s1(self, sobol_g_result):
        S1 = np.asarray(sobol_g_result.S1)
        analytical = np.asarray(sobol_g.ANALYTICAL_S1)
        for i in range(len(analytical)):
            if analytical[i] > 0.01:
                rel = abs(S1[i] - analytical[i]) / analytical[i]
                assert rel < 0.20, (
                    f"S1[{i}]={S1[i]:.4f}, expected {analytical[i]:.4f}, rel error {rel:.2%}"
                )
            else:
                assert abs(S1[i]) < 0.02, f"S1[{i}]={S1[i]:.4f}, expected ~0"

    def test_st(self, sobol_g_result):
        ST = np.asarray(sobol_g_result.ST)
        analytical = np.asarray(sobol_g.ANALYTICAL_ST)
        for i in range(len(analytical)):
            if analytical[i] > 0.01:
                rel = abs(ST[i] - analytical[i]) / analytical[i]
                assert rel < 0.20, (
                    f"ST[{i}]={ST[i]:.4f}, expected {analytical[i]:.4f}, rel error {rel:.2%}"
                )

    def test_ranking(self, sobol_g_result):
        """First three params should be ordered by importance (a_j = 0, 1, 4.5)."""
        S1 = np.asarray(sobol_g_result.S1)
        assert S1[0] > S1[1], f"S1[0]={S1[0]:.4f} should be > S1[1]={S1[1]:.4f}"
        assert S1[1] > S1[2], f"S1[1]={S1[1]:.4f} should be > S1[2]={S1[2]:.4f}"


def test_single_param():
    """D=1 edge case: single parameter problem."""
    problem = Problem(names=("x",), bounds=((0.0, 1.0),))
    X = sample(problem, N=257, M=4, seed=1)
    Y = jnp.sin(jnp.asarray(X[:, 0]))
    result = analyze(problem, Y, M=4)
    assert result.S1.shape == (1,)
    assert result.ST.shape == (1,)
    assert float(result.S1[0]) > 0.5
    assert float(result.ST[0]) > 0.5
