"""Tests for HSIC (kernel-based) sensitivity analysis."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from gsax.benchmarks import ishigami, linear, sobol_g
from gsax.hsic import HSICResult, analyze
from gsax.problem import GaussianInputSpec, Problem
from gsax.sampling import monte_carlo


@pytest.fixture(scope="module")
def linear_hsic_result():
    """HSIC result for linear benchmark."""
    X = monte_carlo(linear.PROBLEM, n=1024, seed=42)
    Y = linear.evaluate(jnp.asarray(X))
    return analyze(linear.PROBLEM, jnp.asarray(X), Y, seed=42, n_perms=100)


@pytest.fixture(scope="module")
def ishigami_hsic_result():
    """HSIC result for Ishigami benchmark."""
    X = monte_carlo(ishigami.PROBLEM, n=1024, seed=42)
    Y = ishigami.evaluate(jnp.asarray(X))
    return analyze(ishigami.PROBLEM, jnp.asarray(X), Y, seed=42, n_perms=100)


@pytest.fixture(scope="module")
def sobol_g_hsic_result():
    """HSIC result for Sobol G-function benchmark."""
    X = monte_carlo(sobol_g.PROBLEM, n=1024, seed=42)
    Y = sobol_g.evaluate(jnp.asarray(X))
    return analyze(sobol_g.PROBLEM, jnp.asarray(X), Y, seed=42, n_perms=100)


class TestLinearHSIC:
    def test_result_type(self, linear_hsic_result):
        assert isinstance(linear_hsic_result, HSICResult)

    def test_shapes(self, linear_hsic_result):
        r = linear_hsic_result
        assert r.R2_HSIC.shape == (3,)
        assert r.T_HSIC.shape == (3,)
        assert r.p_values.shape == (3,)
        assert r.hsic_raw.shape == (3,)

    def test_ranking(self, linear_hsic_result):
        """Linear model: x3 most important, x1 least."""
        r2 = np.asarray(linear_hsic_result.R2_HSIC)
        assert r2[2] > r2[1] > r2[0]

    def test_r2_bounded(self, linear_hsic_result):
        r2 = np.asarray(linear_hsic_result.R2_HSIC)
        assert np.all(r2 >= 0.0)
        assert np.all(r2 <= 1.0 + 1e-6)

    def test_total_bounded(self, linear_hsic_result):
        """With augmented kernels, total indices should be in [0, 1] for additive models."""
        t = np.asarray(linear_hsic_result.T_HSIC)
        assert np.all(np.isfinite(t))
        assert np.all(t >= -0.1)
        assert np.all(t <= 1.0 + 0.1)

    def test_all_significant(self, linear_hsic_result):
        """All inputs are influential in the linear model."""
        p = np.asarray(linear_hsic_result.p_values)
        assert np.all(p < 0.05)

    def test_total_ranking(self, linear_hsic_result):
        """For additive model, total order should also rank x3 > x2 > x1."""
        t = np.asarray(linear_hsic_result.T_HSIC)
        assert t[2] > t[1] > t[0]


class TestIshigamiHSIC:
    def test_x1_strongest_hsic(self, ishigami_hsic_result):
        """x1 appears in both sin(x1) and B*x3^4*sin(x1), giving highest HSIC."""
        r2 = np.asarray(ishigami_hsic_result.R2_HSIC)
        assert r2[0] > r2[2]

    def test_all_positive(self, ishigami_hsic_result):
        r2 = np.asarray(ishigami_hsic_result.R2_HSIC)
        assert np.all(r2 > 0.0)

    def test_r2_bounded(self, ishigami_hsic_result):
        r2 = np.asarray(ishigami_hsic_result.R2_HSIC)
        assert np.all(r2 >= 0.0)
        assert np.all(r2 <= 1.0 + 1e-6)

    def test_total_finite(self, ishigami_hsic_result):
        t = np.asarray(ishigami_hsic_result.T_HSIC)
        assert np.all(np.isfinite(t))


class TestSobolGHSIC:
    def test_first_four_ranking(self, sobol_g_hsic_result):
        """First 4 inputs should be ranked x1 > x2 > x3 > x4."""
        r2 = np.asarray(sobol_g_hsic_result.R2_HSIC)
        assert r2[0] > r2[1] > r2[2] > r2[3]

    def test_negligible_inputs(self, sobol_g_hsic_result):
        """Inputs 5-8 (a=99) should have very small indices."""
        r2 = np.asarray(sobol_g_hsic_result.R2_HSIC)
        for i in range(4, 8):
            assert r2[i] < r2[0] * 0.1

    def test_total_ranking(self, sobol_g_hsic_result):
        """Total order should also rank x1 > x2 > x3 > x4."""
        t = np.asarray(sobol_g_hsic_result.T_HSIC)
        assert t[0] > t[1] > t[2] > t[3]


class TestMultiOutput:
    def test_shapes(self):
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n=512, seed=7)
        Xj = jnp.asarray(X)
        Y = jnp.column_stack([Xj @ jnp.array([1.0, 2.0, 3.0]), jnp.sum(Xj**2, axis=1)])
        result = analyze(problem, Xj, Y, n_perms=50, seed=7)
        assert result.R2_HSIC.shape == (2, 3)
        assert result.T_HSIC.shape == (2, 3)
        assert result.p_values.shape == (2, 3)
        assert result.hsic_raw.shape == (2, 3)

    def test_linear_output_matches_scalar(self):
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n=1024, seed=8)
        Xj = jnp.asarray(X)
        Y_scalar = Xj @ jnp.array([1.0, 2.0, 3.0])
        Y_multi = jnp.column_stack([Y_scalar, jnp.sum(Xj**2, axis=1)])
        r_scalar = analyze(problem, Xj, Y_scalar, n_perms=50, seed=8)
        r_multi = analyze(problem, Xj, Y_multi, n_perms=50, seed=8)
        np.testing.assert_allclose(
            np.asarray(r_scalar.R2_HSIC),
            np.asarray(r_multi.R2_HSIC[0]),
            atol=1e-4,
        )


class TestTimeSeries:
    def test_shapes_3d(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=256, seed=9)
        Xj = jnp.asarray(X)
        T, K = 3, 2
        Y = jnp.zeros((256, T, K))
        for t in range(T):
            for k in range(K):
                Y = Y.at[:, t, k].set(Xj[:, 0] * (t + 1) + Xj[:, 1] * (k + 1))
        result = analyze(problem, Xj, Y, n_perms=20, seed=9)
        assert result.R2_HSIC.shape == (T, K, 2)
        assert result.T_HSIC.shape == (T, K, 2)


class TestBandwidthOverride:
    def test_fixed_bandwidth(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=256, seed=10)
        Xj = jnp.asarray(X)
        Y = Xj[:, 0] * 2.0 + Xj[:, 1]
        r1 = analyze(problem, Xj, Y, bandwidth=0.3, n_perms=20, seed=10)
        r2 = analyze(problem, Xj, Y, bandwidth=0.3, n_perms=20, seed=10)
        np.testing.assert_array_equal(np.asarray(r1.R2_HSIC), np.asarray(r2.R2_HSIC))

    def test_different_bandwidth_different_result(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=256, seed=11)
        Xj = jnp.asarray(X)
        Y = Xj[:, 0] * 2.0 + Xj[:, 1]
        r1 = analyze(problem, Xj, Y, bandwidth=0.1, n_perms=20, seed=11)
        r2 = analyze(problem, Xj, Y, bandwidth=1.0, n_perms=20, seed=11)
        assert not np.allclose(np.asarray(r1.R2_HSIC), np.asarray(r2.R2_HSIC))


class TestPrenormalize:
    def test_prenormalize_runs(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=256, seed=12)
        Xj = jnp.asarray(X)
        Y = Xj[:, 0] * 100.0 + Xj[:, 1]
        result = analyze(problem, Xj, Y, prenormalize=True, n_perms=20, seed=12)
        assert result.R2_HSIC.shape == (2,)


class TestChunked:
    def test_chunked_matches_unchunked(self):
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n=512, seed=13)
        Xj = jnp.asarray(X)
        Y = linear.evaluate(Xj)
        r_full = analyze(problem, Xj, Y, n_perms=50, seed=13)
        r_chunked = analyze(problem, Xj, Y, n_perms=50, seed=13, chunk_size=128)
        np.testing.assert_allclose(
            np.asarray(r_full.R2_HSIC),
            np.asarray(r_chunked.R2_HSIC),
            atol=1e-4,
        )
        np.testing.assert_allclose(
            np.asarray(r_full.T_HSIC),
            np.asarray(r_chunked.T_HSIC),
            atol=1e-4,
        )


class TestReproducibility:
    def test_same_seed_same_result(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=256, seed=14)
        Xj = jnp.asarray(X)
        Y = Xj[:, 0] + Xj[:, 1] ** 2
        r1 = analyze(problem, Xj, Y, seed=42, n_perms=50)
        r2 = analyze(problem, Xj, Y, seed=42, n_perms=50)
        np.testing.assert_array_equal(np.asarray(r1.p_values), np.asarray(r2.p_values))

    def test_different_seed_may_differ(self):
        """Different seeds should produce different permutation sequences."""
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n=256, seed=15)
        Xj = jnp.asarray(X)
        # x3 is independent of Y, so its p-value is not pinned at 0
        Y = Xj[:, 0] * 0.5
        r1 = analyze(problem, Xj, Y, seed=1, n_perms=50)
        r2 = analyze(problem, Xj, Y, seed=2, n_perms=50)
        # At least one p-value should differ between seeds
        assert not np.array_equal(np.asarray(r1.p_values), np.asarray(r2.p_values))


class TestValidation:
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

    def test_n_perms_zero_raises(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="n_perms"):
            analyze(problem, jnp.ones((10, 1)), jnp.ones(10), n_perms=0)

    def test_bandwidth_zero_raises(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="bandwidth"):
            analyze(problem, jnp.ones((10, 1)), jnp.ones(10), bandwidth=0.0)

    def test_bandwidth_negative_raises(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="bandwidth"):
            analyze(problem, jnp.ones((10, 1)), jnp.ones(10), bandwidth=-1.0)

    def test_bandwidth_nan_raises(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="bandwidth"):
            analyze(problem, jnp.ones((10, 1)), jnp.ones(10), bandwidth=float("nan"))

    def test_bandwidth_inf_raises(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="bandwidth"):
            analyze(problem, jnp.ones((10, 1)), jnp.ones(10), bandwidth=float("inf"))

    def test_too_few_samples_raises(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="N must be"):
            analyze(problem, jnp.ones((2, 1)), jnp.ones(2))


class TestToDataset:
    def test_scalar_output(self, linear_hsic_result):
        ds = linear_hsic_result.to_dataset()
        assert "R2_HSIC" in ds.data_vars
        assert "T_HSIC" in ds.data_vars
        assert "p_values" in ds.data_vars
        assert "hsic_raw" in ds.data_vars
        assert list(ds.coords["param"].values) == list(linear.PROBLEM.names)
        assert "output" not in ds.dims

    def test_multi_output(self):
        problem = Problem(
            names=("x1", "x2"),
            bounds=((0, 1), (0, 1)),
            output_names=("temp", "pressure"),
        )
        X = monte_carlo(problem, n=256, seed=16)
        Xj = jnp.asarray(X)
        Y = jnp.column_stack([Xj[:, 0], Xj[:, 1]])
        result = analyze(problem, Xj, Y, n_perms=20, seed=16)
        ds = result.to_dataset()
        assert "output" in ds.dims
        assert list(ds.coords["output"].values) == ["temp", "pressure"]
        assert ds["R2_HSIC"].shape == (2, 2)

    def test_timeseries_output(self):
        problem = Problem(names=("x1",), bounds=((0, 1),))
        X = monte_carlo(problem, n=128, seed=17)
        Xj = jnp.asarray(X)
        Y = jnp.ones((128, 2, 1)) * Xj[:, 0:1, None]
        result = analyze(problem, Xj, Y, n_perms=10, seed=17)
        ds = result.to_dataset(time_coords=[0.0, 0.5])
        assert "time" in ds.dims
        assert list(ds.coords["time"].values) == [0.0, 0.5]


class TestGaussianInputs:
    def test_gaussian_problem(self):
        problem = Problem.from_dict(
            {
                "x1": (0.0, 1.0),
                "x2": GaussianInputSpec(
                    dist="gaussian", mean=0.0, variance=1.0, low=-3.0, high=3.0
                ),
            }
        )
        X = monte_carlo(problem, n=512, seed=18)
        Xj = jnp.asarray(X)
        Y = Xj[:, 0] * 2.0 + Xj[:, 1]
        result = analyze(problem, Xj, Y, n_perms=20, seed=18)
        assert result.R2_HSIC.shape == (2,)
        r2 = np.asarray(result.R2_HSIC)
        assert np.all(r2 > 0.0)
        assert np.all(np.isfinite(r2))


class TestSingleParam:
    def test_single_input(self):
        problem = Problem(names=("x",), bounds=((0.0, 1.0),))
        X = monte_carlo(problem, n=512, seed=19)
        Xj = jnp.asarray(X)
        Y = jnp.sin(Xj[:, 0])
        result = analyze(problem, Xj, Y, n_perms=50, seed=19)
        assert result.R2_HSIC.shape == (1,)
        r2 = np.asarray(result.R2_HSIC)
        assert r2[0] > 0.5
        p = np.asarray(result.p_values)
        assert p[0] < 0.05


class TestIndependentInput:
    def test_independent_input_low_hsic(self):
        """An input independent of the output should have low R2-HSIC."""
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=1024, seed=20)
        Xj = jnp.asarray(X)
        Y = jnp.sin(Xj[:, 0] * 6.0)
        result = analyze(problem, Xj, Y, n_perms=100, seed=20)
        r2 = np.asarray(result.R2_HSIC)
        assert r2[0] > r2[1] * 3.0
        p = np.asarray(result.p_values)
        assert p[0] < 0.05
        assert p[1] > 0.05

    def test_irrelevant_input_total_near_zero(self):
        """Total HSIC for an irrelevant input should be near zero."""
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=1024, seed=21)
        Xj = jnp.asarray(X)
        Y = Xj[:, 0]
        result = analyze(problem, Xj, Y, n_perms=20, seed=21)
        t = np.asarray(result.T_HSIC)
        assert t[0] > 0.5
        assert abs(t[1]) < 0.3


class TestSingleParamTotal:
    def test_d1_total_equals_one(self):
        """With D=1, the complement is ones → T_HSIC should be ~1.0."""
        problem = Problem(names=("x",), bounds=((0.0, 1.0),))
        X = monte_carlo(problem, n=512, seed=22)
        Xj = jnp.asarray(X)
        Y = jnp.sin(Xj[:, 0])
        result = analyze(problem, Xj, Y, n_perms=20, seed=22)
        t = np.asarray(result.T_HSIC)
        assert t[0] > 0.8


class TestZeroVarianceOutput:
    def test_constant_output_produces_nan(self):
        """Constant Y should produce NaN indices, consistent with the warning."""
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=64, seed=23)
        Xj = jnp.asarray(X)
        Y = jnp.ones(64)
        result = analyze(problem, Xj, Y, n_perms=5, seed=23)
        r2 = np.asarray(result.R2_HSIC)
        t = np.asarray(result.T_HSIC)
        assert np.all(np.isnan(r2))
        assert np.all(np.isnan(t))
