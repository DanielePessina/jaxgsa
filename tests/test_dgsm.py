"""Tests for DGSM (derivative-based global sensitivity measures)."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from gsax.benchmarks import ishigami, linear, sobol_g
from gsax.dgsm import analyze
from gsax.dgsm._poincare import axis_constants, marginal_variance, poincare_constant
from gsax.problem import GaussianInputSpec, Problem
from gsax.sampling import monte_carlo

A, B = 7.0, 0.1


def _ishigami_single(x):
    """Unbatched Ishigami: (3,) -> (1,)."""
    return jnp.array([jnp.sin(x[0]) + A * jnp.sin(x[1]) ** 2 + B * x[2] ** 4 * jnp.sin(x[0])])


def _linear_single(x):
    """Unbatched linear: (3,) -> (1,)."""
    c = jnp.array([1.0, 2.0, 3.0])
    return jnp.array([jnp.dot(c, x)])


def _multi_output(x):
    """Unbatched multi-output: (3,) -> (2,)."""
    c = jnp.array([1.0, 2.0, 3.0])
    return jnp.array([jnp.dot(c, x), jnp.sum(x**2)])


def _sobol_g_single(x):
    """Unbatched Sobol G-function: (8,) -> ()."""
    a = jnp.array([0.0, 1.0, 4.5, 9.0, 99.0, 99.0, 99.0, 99.0])
    terms = (jnp.abs(4.0 * x - 2.0) + a) / (1.0 + a)
    return jnp.prod(terms)


@pytest.fixture(scope="module")
def ishigami_dgsm_result():
    """DGSM result for Ishigami benchmark."""
    X = monte_carlo(ishigami.PROBLEM, n=50_000, seed=42)
    return analyze(ishigami.PROBLEM, _ishigami_single, jnp.asarray(X))


@pytest.fixture(scope="module")
def linear_dgsm_result():
    """DGSM result for linear benchmark."""
    X = monte_carlo(linear.PROBLEM, n=10_000, seed=123)
    return analyze(linear.PROBLEM, _linear_single, jnp.asarray(X))


class TestPoincare:
    def test_uniform_constant(self):
        spec = ("uniform", -math.pi, math.pi, None, None)
        C = poincare_constant(spec)
        assert C == pytest.approx((2 * math.pi) ** 2 / math.pi**2)
        assert C == pytest.approx(4.0)

    def test_gaussian_constant(self):
        spec = ("gaussian", 0.0, 1.5, None, None)
        assert poincare_constant(spec) == pytest.approx(1.5)

    def test_truncated_gaussian_spectral(self):
        spec = ("gaussian", 0.0, 1.0, -3.0, 3.0)
        C = poincare_constant(spec)
        assert C < 1.0
        assert C > 0.0

    def test_spectral_recovers_uniform(self):
        from gsax.dgsm._poincare import _truncnorm_poincare

        got = _truncnorm_poincare(0.0, 1e6, -math.pi, math.pi, 512)
        assert got == pytest.approx(4.0, rel=2e-3)

    def test_spectral_wide_gaussian_collapses_to_sigma2(self):
        from gsax.dgsm._poincare import _truncnorm_poincare

        sigma = 1.3
        got = _truncnorm_poincare(0.0, sigma, -6 * sigma, 6 * sigma, 700)
        assert got == pytest.approx(sigma**2, rel=3e-2)


class TestMarginalVariance:
    def test_uniform(self):
        spec = ("uniform", 0.0, 1.0, None, None)
        assert marginal_variance(spec) == pytest.approx(1.0 / 12.0)

    def test_uniform_wide(self):
        spec = ("uniform", -math.pi, math.pi, None, None)
        assert marginal_variance(spec) == pytest.approx((2 * math.pi) ** 2 / 12.0)

    def test_gaussian(self):
        spec = ("gaussian", 0.0, 2.5, None, None)
        assert marginal_variance(spec) == pytest.approx(2.5)

    def test_truncated_gaussian(self):
        spec = ("gaussian", 0.0, 1.0, -1.0, 1.0)
        v = marginal_variance(spec)
        assert 0 < v < 1.0


class TestAxisConstants:
    def test_shapes(self):
        problem = Problem(names=("a", "b", "c"), bounds=((0, 1), (0, 1), (0, 1)))
        C, Var = axis_constants(problem)
        assert C.shape == (3,)
        assert Var.shape == (3,)

    def test_uniform_values(self):
        problem = Problem(names=("x",), bounds=((0.0, 1.0),))
        C, Var = axis_constants(problem)
        assert C[0] == pytest.approx(1.0 / math.pi**2)
        assert Var[0] == pytest.approx(1.0 / 12.0)


class TestSampleMC:
    def test_shape(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=100, seed=1)
        assert X.shape == (100, 2)

    def test_within_bounds(self):
        problem = Problem(names=("a", "b"), bounds=((2.0, 5.0), (-1.0, 3.0)))
        X = monte_carlo(problem, n=1000, seed=2)
        assert np.all(X[:, 0] >= 2.0 - 1e-10)
        assert np.all(X[:, 0] <= 5.0 + 1e-10)
        assert np.all(X[:, 1] >= -1.0 - 1e-10)
        assert np.all(X[:, 1] <= 3.0 + 1e-10)

    def test_reproducible(self):
        problem = Problem(names=("x1",), bounds=((0, 1),))
        X1 = monte_carlo(problem, n=50, seed=99)
        X2 = monte_carlo(problem, n=50, seed=99)
        np.testing.assert_array_equal(X1, X2)

    def test_gaussian_inputs(self):
        problem = Problem.from_dict(
            {
                "x1": (0.0, 1.0),
                "x2": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
            }
        )
        X = monte_carlo(problem, n=500, seed=3)
        assert X.shape == (500, 2)
        assert np.all(X[:, 0] >= 0.0 - 1e-10)
        assert np.all(X[:, 0] <= 1.0 + 1e-10)


class TestLinearDGSM:
    def test_nu_exact(self, linear_dgsm_result):
        """Linear f(x) = c . x => df/dx_i = c_i => nu_i = c_i^2 exactly."""
        nu = np.asarray(linear_dgsm_result.nu)
        assert nu.shape == (1, 3)
        c = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(nu[0], c**2, atol=1e-5)

    def test_sigma_exact(self, linear_dgsm_result):
        """Linear f(x) = c . x => E[df/dx_i] = c_i exactly."""
        sigma = np.asarray(linear_dgsm_result.sigma)
        c = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(sigma[0], c, atol=1e-5)

    def test_upper_bound_geq_st(self, linear_dgsm_result):
        upper = np.asarray(linear_dgsm_result.upper_bound)
        ST = np.array(linear.ANALYTICAL_ST)
        assert np.all(upper[0] >= ST - 1e-3)

    def test_lower_bound_close_to_st(self, linear_dgsm_result):
        """For a linear model, the lower bound should approximate ST closely."""
        lower = np.asarray(linear_dgsm_result.lower_bound)
        ST = np.array(linear.ANALYTICAL_ST)
        # MC noise in Var(Y) means the bound can slightly exceed analytical ST
        np.testing.assert_allclose(lower[0], ST, rtol=0.05)

    def test_bracket_contains_st(self, linear_dgsm_result):
        """The DGSM bracket [lower, upper] should approximately contain ST."""
        lower = np.asarray(linear_dgsm_result.lower_bound)[0]
        upper = np.asarray(linear_dgsm_result.upper_bound)[0]
        ST = np.array(linear.ANALYTICAL_ST)
        # MC noise tolerance: bounds are exact in expectation, noisy in practice
        for i in range(3):
            assert lower[i] <= ST[i] + 0.02, f"lower[{i}]={lower[i]:.4f} > ST={ST[i]:.4f}"
            assert upper[i] >= ST[i] - 0.02, f"upper[{i}]={upper[i]:.4f} < ST={ST[i]:.4f}"


class TestIshigamiDGSM:
    def test_nu_analytic(self, ishigami_dgsm_result):
        """Compare nu against analytically derived Ishigami DGSM."""
        pi = math.pi
        nu_analytic = np.array(
            [
                0.5 * (1 + 2 * B * pi**4 / 5 + B**2 * pi**8 / 9),
                A**2 / 2,
                8 * B**2 * pi**6 / 7,
            ]
        )
        nu = np.asarray(ishigami_dgsm_result.nu)[0]
        np.testing.assert_allclose(nu, nu_analytic, rtol=3e-2)

    def test_upper_bound_holds(self, ishigami_dgsm_result):
        upper = np.asarray(ishigami_dgsm_result.upper_bound)[0]
        ST = np.array(ishigami.ANALYTICAL_ST)
        assert np.all(upper >= ST - 1e-2)

    def test_lower_bound_holds(self, ishigami_dgsm_result):
        lower = np.asarray(ishigami_dgsm_result.lower_bound)[0]
        ST = np.array(ishigami.ANALYTICAL_ST)
        assert np.all(lower <= ST + 1e-2)


class TestMultiOutput:
    def test_shapes(self):
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n=500, seed=7)
        result = analyze(problem, _multi_output, jnp.asarray(X))
        assert result.nu.shape == (2, 3)
        assert result.sigma.shape == (2, 3)
        assert result.upper_bound.shape == (2, 3)
        assert result.lower_bound.shape == (2, 3)
        assert result.var_y.shape == (2,)

    def test_linear_output_matches_scalar(self):
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n=2000, seed=8)
        Xj = jnp.asarray(X)
        result_multi = analyze(problem, _multi_output, Xj)
        result_linear = analyze(problem, _linear_single, Xj)
        np.testing.assert_allclose(
            np.asarray(result_multi.nu)[0],
            np.asarray(result_linear.nu)[0],
            atol=1e-5,
        )


class TestScalarOutput:
    def test_scalar_fn(self):
        """fn returning a scalar () instead of (1,) should squeeze to (D,)."""

        def f_scalar(x):
            return jnp.dot(jnp.array([1.0, 2.0, 3.0]), x)

        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n=500, seed=9)
        result = analyze(problem, f_scalar, jnp.asarray(X))
        assert result.nu.shape == (3,)
        assert result.sigma.shape == (3,)
        assert result.upper_bound.shape == (3,)
        assert result.lower_bound.shape == (3,)
        assert result.var_y.shape == ()
        c = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(np.asarray(result.nu), c**2, atol=1e-4)


class TestPrecomputed:
    def test_precomputed_matches_autodiff(self):
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n=1000, seed=10)
        Xj = jnp.asarray(X)
        result_auto = analyze(problem, _linear_single, Xj)

        import jax

        jacfn = jax.vmap(jax.jacrev(_linear_single))
        dfdx = jacfn(Xj)
        Y = jax.vmap(_linear_single)(Xj)

        result_pre = analyze(problem, Y=Y, dfdx=dfdx)
        np.testing.assert_allclose(
            np.asarray(result_auto.nu), np.asarray(result_pre.nu), atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(result_auto.sigma), np.asarray(result_pre.sigma), atol=1e-5
        )

    def test_missing_args_raises(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="Provide either"):
            analyze(problem)

    def test_singleton_pair_scalar(self):
        """Y=(N,) requires an exact (N, D) Jacobian."""
        import jax

        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1),) * 3)
        X = jnp.asarray(monte_carlo(problem, n=400, seed=7))

        def fn(x):  # (3,) -> ()
            return jnp.dot(jnp.array([1.0, 2.0, 3.0]), x)

        Y = jax.vmap(fn)(X)  # (N,)
        dfdx = jax.vmap(jax.jacrev(fn))(X)  # (N, D)
        assert analyze(problem, Y=Y, dfdx=dfdx).nu.shape == (3,)
        with pytest.raises(ValueError, match="ndim"):
            analyze(problem, Y=Y, dfdx=dfdx[:, None, :])

    def test_singleton_pair_2d(self):
        """Y=(N, 1) requires an exact (N, 1, D) Jacobian."""
        import jax

        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1),) * 3)
        X = jnp.asarray(monte_carlo(problem, n=400, seed=8))

        def fn(x):
            return jnp.dot(jnp.array([1.0, 2.0, 3.0]), x)

        Y = jax.vmap(fn)(X)[:, None]  # (N, 1)
        dfdx = jax.vmap(jax.jacrev(fn))(X)[:, None, :]
        result = analyze(problem, Y=Y, dfdx=dfdx)
        assert result.nu.shape == (1, 3)


class TestChunked:
    def test_chunked_matches_unchunked(self):
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n=500, seed=11)
        Xj = jnp.asarray(X)
        result_full = analyze(problem, _linear_single, Xj)
        result_chunked = analyze(problem, _linear_single, Xj, batch_size=100)
        np.testing.assert_allclose(
            np.asarray(result_full.nu),
            np.asarray(result_chunked.nu),
            atol=1e-5,
        )

    def test_ragged_chunk_matches(self):
        """N not divisible by batch_size should still produce correct results."""
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n=503, seed=14)
        Xj = jnp.asarray(X)
        result_full = analyze(problem, _linear_single, Xj)
        result_ragged = analyze(problem, _linear_single, Xj, batch_size=100)
        np.testing.assert_allclose(
            np.asarray(result_full.nu),
            np.asarray(result_ragged.nu),
            atol=1e-5,
        )

    def test_batch_size_kwarg_accepted(self):
        """The 0.4 name `batch_size` is accepted explicitly."""
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1),) * 3)
        Xj = jnp.asarray(monte_carlo(problem, n=50, seed=3))
        result = analyze(problem, _linear_single, Xj, batch_size=16)
        assert result.nu.shape == (1, 3)

    def test_old_chunk_size_kwarg_raises(self):
        """The pre-0.4 `chunk_size` name is gone — no shim."""
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1),) * 3)
        Xj = jnp.asarray(monte_carlo(problem, n=50, seed=3))
        old_kwargs: dict[str, Any] = {"chunk_size": 16}
        with pytest.raises(TypeError):
            analyze(problem, _linear_single, Xj, **old_kwargs)


class TestValidation:
    def test_x_wrong_ndim(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="2-D"):
            analyze(problem, _linear_single, jnp.ones(10))

    def test_x_wrong_columns(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="columns"):
            analyze(problem, _linear_single, jnp.ones((10, 3)))

    def test_dfdx_wrong_ndim(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="ndim"):
            analyze(problem, Y=jnp.ones(10), dfdx=jnp.ones(10))

    def test_dfdx_wrong_columns(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="last dimension"):
            analyze(problem, Y=jnp.ones(10), dfdx=jnp.ones((10, 3)))

    def test_dfdx_row_mismatch(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="rows"):
            analyze(problem, Y=jnp.ones(10), dfdx=jnp.ones((5, 1)))

    def test_sample_mc_n_zero_raises(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="n must be >= 1"):
            monte_carlo(problem, n=0)


class TestToDataset:
    def test_scalar_output(self):
        """Truly scalar fn (returning ()) produces a dataset with no output dim."""

        def f_scalar(x):
            return jnp.dot(jnp.array([1.0, 2.0, 3.0]), x)

        problem = linear.PROBLEM
        X = monte_carlo(problem, n=500, seed=42)
        result = analyze(problem, f_scalar, jnp.asarray(X))
        ds = result.to_dataset()
        assert "nu" in ds.data_vars
        assert "sigma" in ds.data_vars
        assert "upper_bound" in ds.data_vars
        assert "lower_bound" in ds.data_vars
        assert list(ds.coords["param"].values) == list(linear.PROBLEM.names)
        assert "output" not in ds.dims

    def test_multi_output(self):
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n=200, seed=12)
        result = analyze(problem, _multi_output, jnp.asarray(X))
        ds = result.to_dataset()
        assert "output" in ds.dims
        assert ds["nu"].shape == (2, 3)

    def test_output_names_used(self):
        problem = Problem(
            names=("x1", "x2", "x3"),
            bounds=((0, 1), (0, 1), (0, 1)),
            output_names=("temp", "pressure"),
        )
        X = monte_carlo(problem, n=200, seed=13)
        result = analyze(problem, _multi_output, jnp.asarray(X))
        ds = result.to_dataset()
        assert list(ds.coords["output"].values) == ["temp", "pressure"]


class TestSobolGDGSM:
    @pytest.fixture(scope="module")
    def sobol_g_dgsm_result(self):
        """DGSM result for Sobol G-function benchmark."""
        X = monte_carlo(sobol_g.PROBLEM, n=50_000, seed=42)
        return analyze(sobol_g.PROBLEM, _sobol_g_single, jnp.asarray(X))

    def test_upper_bound_holds(self, sobol_g_dgsm_result):
        upper = np.asarray(sobol_g_dgsm_result.upper_bound)
        ST = np.array(sobol_g.ANALYTICAL_ST)
        for i in range(8):
            assert upper[i] >= ST[i] - 0.02, f"upper[{i}]={upper[i]:.4f} < ST={ST[i]:.4f}"

    def test_lower_bound_holds(self, sobol_g_dgsm_result):
        lower = np.asarray(sobol_g_dgsm_result.lower_bound)
        ST = np.array(sobol_g.ANALYTICAL_ST)
        for i in range(4):
            assert lower[i] <= ST[i] + 0.02, f"lower[{i}]={lower[i]:.4f} > ST={ST[i]:.4f}"

    def test_ranking(self, sobol_g_dgsm_result):
        nu = np.asarray(sobol_g_dgsm_result.nu)
        assert nu[0] > nu[1] > nu[2] > nu[3]


def test_single_param():
    problem = Problem(names=("x",), bounds=((0.0, 1.0),))
    X = monte_carlo(problem, n=5000, seed=1)

    def fn(x):
        return jnp.sin(x[0])

    result = analyze(problem, fn, jnp.asarray(X))
    assert result.nu.shape == (1,)
    assert result.sigma.shape == (1,)
    assert result.upper_bound.shape == (1,)
    assert result.lower_bound.shape == (1,)


# ---------------------------------------------------------------------------
# Time-series outputs
# ---------------------------------------------------------------------------


class TestTimeSeries:
    """DGSM supports (D,) -> (T, K) functions and 4-D precomputed Jacobians."""

    @staticmethod
    def _fn_scalar(x):
        return jnp.sum(x**2) + 2.0 * x[0]

    # linear.PROBLEM has D=3 inputs; draw plain MC samples the same way the
    # other DGSM tests do.

    @classmethod
    def _fn_ts(cls, x):
        base = cls._fn_scalar(x)
        # (T=3, K=2): affine images of one scalar model so every slice has
        # identical (normalized) sensitivity structure.
        scale = jnp.array([[1.0, 2.0], [0.5, 1.0], [3.0, 0.1]])
        return scale * base

    def _X(self, n=512):
        return jnp.asarray(monte_carlo(linear.PROBLEM, n=n, seed=3))

    def test_time_series_fn_shapes(self):
        X = self._X()
        result = analyze(linear.PROBLEM, self._fn_ts, X)
        assert result.nu.shape == (3, 2, 3)
        assert result.sigma.shape == (3, 2, 3)
        assert result.upper_bound.shape == (3, 2, 3)
        assert result.lower_bound.shape == (3, 2, 3)
        assert result.var_y.shape == (3, 2)

    def test_time_series_slices_match_scalar(self):
        X = self._X()
        ts = analyze(linear.PROBLEM, self._fn_ts, X)
        scalar = analyze(linear.PROBLEM, self._fn_scalar, X)
        # Bounds are scale-invariant (nu and Var(Y) scale together).
        for t in range(3):
            for k in range(2):
                np.testing.assert_allclose(
                    np.asarray(ts.upper_bound[t, k]),
                    np.asarray(scalar.upper_bound),
                    rtol=1e-4,
                )
                np.testing.assert_allclose(
                    np.asarray(ts.lower_bound[t, k]),
                    np.asarray(scalar.lower_bound),
                    rtol=1e-4,
                )

    def test_precomputed_4d_matches_autodiff(self):
        import jax

        X = self._X()
        auto = analyze(linear.PROBLEM, self._fn_ts, X)
        Y = jax.vmap(self._fn_ts)(X)
        dfdx = jax.vmap(jax.jacrev(self._fn_ts))(X)  # (N, T, K, D)
        pre = analyze(linear.PROBLEM, Y=Y, dfdx=dfdx)
        np.testing.assert_allclose(np.asarray(pre.nu), np.asarray(auto.nu), rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(pre.upper_bound), np.asarray(auto.upper_bound), rtol=1e-5
        )

    def test_mismatched_y_dfdx_ndim_raises(self):
        import jax

        X = self._X(64)
        Y = jax.vmap(self._fn_ts)(X)  # (N, T, K)
        dfdx_3d = jax.vmap(jax.jacrev(self._fn_scalar))(X)[:, None, :]  # (N, 1, D)
        with pytest.raises(ValueError, match="ndim"):
            analyze(linear.PROBLEM, Y=Y, dfdx=dfdx_3d)

    def test_precomputed_swapped_layout_is_rejected(self):
        """The output axis must remain last for both Y and dfdx."""
        import jax

        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1),) * 3, output_names=("a", "b"))
        X = jnp.asarray(monte_carlo(problem, n=400, seed=9))
        Y = jax.vmap(self._fn_ts)(X)  # (N, T=3, K=2)
        dfdx = jax.vmap(jax.jacrev(self._fn_ts))(X)  # (N, T, K, D)
        analyze(problem, Y=Y, dfdx=dfdx)
        with pytest.raises(ValueError, match="output_names"):
            analyze(
                problem,
                Y=jnp.swapaxes(Y, 1, 2),  # (N, K, T)
                dfdx=jnp.swapaxes(dfdx, 1, 2),  # (N, K, T, D)
            )

    def test_precomputed_inconsistent_dfdx_raises(self):
        """Canonical Y with a transposed dfdx (T != K) is rejected, not swapped."""
        import jax

        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1),) * 3, output_names=("a", "b"))
        X = jnp.asarray(monte_carlo(problem, n=64, seed=1))
        Y = jax.vmap(self._fn_ts)(X)  # (N, 3, 2) canonical
        dfdx = jax.vmap(jax.jacrev(self._fn_ts))(X)  # (N, 3, 2, D)
        with pytest.raises(ValueError, match="does not match"):
            analyze(problem, Y=Y, dfdx=jnp.swapaxes(dfdx, 1, 2))  # (N, 2, 3, D)

    def test_autodiff_explicit_single_output_timeseries(self):
        """A time-varying single output is returned explicitly as (T, 1)."""
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1),) * 3, output_names=("p",))
        X = jnp.asarray(monte_carlo(problem, n=400, seed=2))

        def fn(x):  # (3,) -> (T=4, K=1)
            base = jnp.dot(jnp.array([1.0, 2.0, 3.0]), x)
            return (jnp.array([1.0, 2.0, 0.5, 3.0]) * base)[:, None]

        result = analyze(problem, fn, X)
        assert result.nu.shape == (4, 1, 3)
        # Affine images of one scalar model share the same normalized bound.
        ub = np.asarray(result.upper_bound)
        for t in range(1, 4):
            np.testing.assert_allclose(ub[t, 0], ub[0, 0], rtol=1e-4)

    def test_chunked_time_series_matches_unchunked(self):
        X = self._X(200)
        full = analyze(linear.PROBLEM, self._fn_ts, X)
        chunked = analyze(linear.PROBLEM, self._fn_ts, X, batch_size=64)
        np.testing.assert_allclose(np.asarray(chunked.nu), np.asarray(full.nu), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(chunked.sigma), np.asarray(full.sigma), rtol=1e-4)

    def test_to_dataset_time_series(self):
        X = self._X()
        ds = analyze(linear.PROBLEM, self._fn_ts, X).to_dataset(time_coords=[0.0, 0.5, 1.0])
        assert ds["nu"].dims == ("time", "output", "param")
        np.testing.assert_allclose(ds.coords["time"].values, [0.0, 0.5, 1.0])
