"""Tests for DGSM (derivative-based global sensitivity measures)."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning
from jaxgsa._core.invalid import InvalidUnit
from jaxgsa.benchmarks import ishigami, linear, sobol_g
from jaxgsa.dgsm import analyze
from jaxgsa.dgsm._poincare import axis_constants, marginal_variance, poincare_constant
from jaxgsa.problem import GaussianInputSpec, Problem
from jaxgsa.sampling import monte_carlo

A, B = 7.0, 0.1


def _spec(input_spec):
    """Return the normalized spec of one marginal, built the public way.

    The test never spells the private spec layout itself. It declares the
    marginal with the ordinary public input spec, builds a ``Problem``, and
    reads the normalized spec back off it.

    Args:
        input_spec: Any value ``Problem.from_dict`` accepts for one
            parameter, such as a ``(low, high)`` tuple or a
            :class:`GaussianInputSpec`.

    Returns:
        The normalized spec of that single marginal.
    """
    return Problem.from_dict({"x": input_spec}).input_specs[0]


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
        """Tier T0 (closed form): the Poincare constant of ``U(-pi, pi)``.

        For a uniform density on an interval of width ``L`` the constant is
        ``(L / pi) ** 2``. Here ``L = 2 * pi``, so the value is exactly 4.
        """
        spec = _spec((-math.pi, math.pi))
        C = poincare_constant(spec)
        assert C == pytest.approx(4.0)

    def test_gaussian_constant(self):
        spec = _spec(GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.5))
        assert poincare_constant(spec) == pytest.approx(1.5)

    def test_truncated_gaussian_spectral(self):
        spec = _spec(
            GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0, low=-3.0, high=3.0)
        )
        C = poincare_constant(spec)
        assert C < 1.0
        assert C > 0.0

    def test_spectral_recovers_uniform(self):
        from jaxgsa.dgsm._poincare import _truncnorm_poincare

        got = _truncnorm_poincare(0.0, 1e6, -math.pi, math.pi, 512)
        assert got == pytest.approx(4.0, rel=2e-3)

    def test_spectral_wide_gaussian_collapses_to_sigma2(self):
        from jaxgsa.dgsm._poincare import _truncnorm_poincare

        sigma = 1.3
        got = _truncnorm_poincare(0.0, sigma, -6 * sigma, 6 * sigma, 700)
        assert got == pytest.approx(sigma**2, rel=3e-2)


class TestMarginalVariance:
    def test_uniform(self):
        spec = _spec((0.0, 1.0))
        assert marginal_variance(spec) == pytest.approx(1.0 / 12.0)

    def test_uniform_wide(self):
        spec = _spec((-math.pi, math.pi))
        assert marginal_variance(spec) == pytest.approx((2 * math.pi) ** 2 / 12.0)

    def test_gaussian(self):
        spec = _spec(GaussianInputSpec(dist="gaussian", mean=0.0, variance=2.5))
        assert marginal_variance(spec) == pytest.approx(2.5)

    def test_truncated_gaussian(self):
        spec = _spec(
            GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0, low=-1.0, high=1.0)
        )
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
        """Tier T4 (behavioural contract): no argument group at all is rejected."""
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


def _three_uniform_problem():
    """Return a three-parameter unit-cube problem, built the public way."""
    return Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)})


class TestCallStyleResolution:
    """The two argument groups are exclusive, and each must be complete."""

    def test_over_specified_call_raises_and_names_arguments(self):
        """Tier T4 (behavioural contract): X plus (Y, dfdx) is rejected.

        This used to take the pre-computed branch and drop ``X`` in silence,
        which also dropped the bounds check that runs only on ``X``.
        """
        import jax

        problem = _three_uniform_problem()
        X = jnp.asarray(monte_carlo(problem, n=64, seed=3))
        Y = jax.vmap(_linear_single)(X)
        dfdx = jax.vmap(jax.jacrev(_linear_single))(X)

        with pytest.raises(ValueError, match="not both") as excinfo:
            analyze(problem, X=X, Y=Y, dfdx=dfdx)
        message = str(excinfo.value)
        assert "X" in message
        assert "Y" in message and "dfdx" in message

    def test_fn_plus_precomputed_raises(self):
        """Tier T4 (behavioural contract): fn alongside (Y, dfdx) is rejected."""
        problem = _three_uniform_problem()
        with pytest.raises(ValueError, match="not both") as excinfo:
            analyze(problem, _linear_single, Y=jnp.ones((5, 1)), dfdx=jnp.ones((5, 1, 3)))
        assert "fn" in str(excinfo.value)

    def test_fn_without_x_raises(self):
        """Tier T4 (behavioural contract): a half-filled autodiff group is rejected."""
        problem = _three_uniform_problem()
        with pytest.raises(ValueError, match="autodiff path needs both") as excinfo:
            analyze(problem, _linear_single)
        assert "X was not" in str(excinfo.value)

    def test_x_without_fn_raises(self):
        """Tier T4 (behavioural contract): X on its own is rejected."""
        problem = _three_uniform_problem()
        X = jnp.asarray(monte_carlo(problem, n=16, seed=4))
        with pytest.raises(ValueError, match="autodiff path needs both") as excinfo:
            analyze(problem, X=X)
        assert "fn was not" in str(excinfo.value)

    def test_y_without_dfdx_raises(self):
        """Tier T4 (behavioural contract): a half-filled pre-computed group is rejected."""
        problem = _three_uniform_problem()
        with pytest.raises(ValueError, match="pre-computed path needs both") as excinfo:
            analyze(problem, Y=jnp.ones((5, 1)))
        assert "dfdx was not" in str(excinfo.value)

    def test_dfdx_without_y_raises(self):
        """Tier T4 (behavioural contract): dfdx on its own is rejected."""
        problem = _three_uniform_problem()
        with pytest.raises(ValueError, match="pre-computed path needs both") as excinfo:
            analyze(problem, dfdx=jnp.ones((5, 1, 3)))
        assert "Y was not" in str(excinfo.value)

    def test_both_valid_styles_still_run(self):
        """Tier T4 (behavioural contract): the guard changes no valid call.

        Both complete groups must still be accepted, and must still agree on
        the moments. This is the regression guard for the resolution step: a
        guard that rejected a legal call would fail here.
        """
        import jax

        problem = _three_uniform_problem()
        X = jnp.asarray(monte_carlo(problem, n=256, seed=5))
        Y = jax.vmap(_linear_single)(X)
        dfdx = jax.vmap(jax.jacrev(_linear_single))(X)

        result_auto = analyze(problem, _linear_single, X)
        result_pre = analyze(problem, Y=Y, dfdx=dfdx)
        np.testing.assert_allclose(
            np.asarray(result_auto.nu), np.asarray(result_pre.nu), atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(result_auto.upper_bound), np.asarray(result_pre.upper_bound), atol=1e-5
        )


class TestBatchCallableRejected:
    """``fn`` takes one row, not the whole sample matrix."""

    def test_batch_callable_raises_clear_error(self):
        """Tier T4 (behavioural contract): a batch model gives a named error.

        A callable that indexes a sample axis used to die with an
        ``IndexError`` raised inside ``jax.jacrev``, which named nothing the
        caller could act on.
        """
        problem = _three_uniform_problem()
        X = jnp.asarray(monte_carlo(problem, n=32, seed=6))

        def batch_model(x):
            """Batch convention: (N, D) -> (N,)."""
            return x[:, 0] + 2.0 * x[:, 1] + 3.0 * x[:, 2]

        with pytest.raises(ValueError, match="one-sample function") as excinfo:
            analyze(problem, batch_model, X)
        message = str(excinfo.value)
        assert "(3,)" in message
        assert "model(x[None, :])[0]" in message

    def test_batch_callable_with_extra_axis_raises(self):
        """Tier T4 (behavioural contract): a traceable batch model is caught too.

        A batch callable that broadcasts instead of indexing traces cleanly on
        one row, but returns one axis too many. The output-rank check catches
        that case, which the trace alone cannot.
        """
        problem = _three_uniform_problem()
        X = jnp.asarray(monte_carlo(problem, n=32, seed=7))

        def batch_model(x):
            """Batch convention: (N, D) -> (N, T, K)."""
            return x[..., None, None] * jnp.ones((2, 2))

        with pytest.raises(ValueError, match="more than two axes"):
            analyze(problem, batch_model, X)

    def test_generic_trace_failure_omits_wrapper_advice(self):
        """Tier T4 (behavioural contract): an unrelated bug is not blamed on batching.

        This model already follows the one-sample convention. It fails to
        trace for its own reason: it multiplies a ``(3,)`` row by a ``(4,)``
        array. The guard must report that error and the expected signature,
        and must not tell the caller to wrap a batch model. That advice would
        send a caller with an ordinary model bug the wrong way.
        """
        problem = _three_uniform_problem()
        X = jnp.asarray(monte_carlo(problem, n=16, seed=9))

        def buggy_point_model(x):
            """One-sample convention, with an unrelated internal shape bug."""
            return jnp.sum(x * jnp.ones(4))

        with pytest.raises(ValueError) as excinfo:
            analyze(problem, buggy_point_model, X)
        message = str(excinfo.value)
        # The real cause is reported, by type and by text.
        assert "TypeError" in message
        assert "broadcasting" in message
        # The expected signature is still stated.
        assert "one-sample function" in message
        # The batch-wrapper advice is not, because nothing points to batching.
        assert "model(x[None, :])[0]" not in message
        assert "Wrap a batch model" not in message

    def test_valid_time_series_callable_accepted(self):
        """Tier T4 (behavioural contract): the guard passes a legal (T, K) model.

        Two output axes are the largest a one-sample function may return, so
        this is the boundary the rank check must not reject.
        """
        problem = _three_uniform_problem()
        X = jnp.asarray(monte_carlo(problem, n=64, seed=8))

        def point_model(x):
            """One-sample convention: (D,) -> (T, K)."""
            return jnp.outer(jnp.array([1.0, 2.0]), x[:2])

        result = analyze(problem, point_model, X)
        assert result.nu.shape == (2, 2, 3)


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


# --- The shared on_invalid policy -------------------------------------------


def _two_uniform_problem():
    """Return a two-parameter unit-cube problem, built the public way."""
    return Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})


def _grid_X(n=24):
    """Return a deterministic (n, 2) sample whose first column spans (0, 1).

    A fixed grid, not a random draw, so a test can name the rows it makes bad
    and read the same rows back out of the report.
    """
    first = np.linspace(0.02, 0.98, n)
    second = np.linspace(1.0, 2.0, n)
    return jnp.asarray(np.stack([first, second], axis=1))


def _plain(x):
    """Well-behaved model: (2,) -> (), finite output and finite derivative."""
    return 3.0 * x[0] + 0.5 * x[1]


def _nan_output_at(bad_rows, X):
    """Return a model that gives a NaN **output** on the named rows of X.

    The derivative stays finite on those rows, so this isolates the output
    half of the check. The model recognises its row by the first column of
    ``X``, which ``_grid_X`` makes unique.
    """
    keys = jnp.asarray(np.asarray(X)[list(bad_rows), 0])

    def fn(x):
        is_bad = jnp.any(keys == x[0])
        return jnp.where(is_bad, jnp.nan, 3.0 * x[0] + 0.5 * x[1])

    return fn


def _nan_jacobian(x):
    """Finite output, NaN derivative where ``x[0] < 0.5``: (2,) -> ().

    This is the textbook ``jnp.where`` gradient trap. The forward pass selects
    the safe branch, so the output is finite everywhere. The backward pass
    still evaluates ``d/dx sqrt(negative)``, which is NaN, and ``0 * NaN`` is
    NaN. A check that only looked at Y would let it through and return a
    ``nu`` that is silently NaN.
    """
    safe = x[0] >= 0.5
    return jnp.where(safe, jnp.sqrt(x[0] - 0.5), 0.0) + 0.5 * x[1]


def _nan_jacobian_at(bad_rows, X):
    """Return a model with a NaN **derivative** on the named rows of X.

    Same gradient trap as :func:`_nan_jacobian`, but aimed at chosen rows
    rather than at a half of the input range. The output stays finite
    everywhere, so only the derivative check can see these rows.
    """
    keys = jnp.asarray(np.asarray(X)[list(bad_rows), 0])

    def fn(x):
        is_bad = jnp.any(keys == x[0])
        branch = jnp.sqrt(x[0] - jnp.where(is_bad, 10.0, 0.0))
        return jnp.where(is_bad, 0.5 * x[1], branch + 0.5 * x[1])

    return fn


def _precomputed(fn, X):
    """Return ``(Y, dfdx)`` for the pre-computed calling convention."""
    import jax

    return jax.vmap(fn)(X), jax.vmap(jax.jacrev(fn))(X)


class TestInvalidPolicyPrecomputed:
    """T4: the on_invalid policy on the (Y, dfdx) calling convention."""

    def test_raise_is_the_default(self):
        """T4: a non-finite output refuses the analysis without being asked."""
        X = _grid_X()
        Y, dfdx = _precomputed(_nan_output_at([3], X), X)
        with pytest.raises(ValueError, match="non-finite"):
            analyze(_two_uniform_problem(), Y=Y, dfdx=dfdx)

    def test_raise_names_the_rows(self):
        """T4: the refusal lists the positions the caller has to investigate."""
        X = _grid_X()
        Y, dfdx = _precomputed(_nan_output_at([3, 11], X), X)
        with pytest.raises(ValueError) as exc:
            analyze(_two_uniform_problem(), Y=Y, dfdx=dfdx, on_invalid="raise")
        assert "[3, 11]" in str(exc.value)
        assert "jaxgsa.dgsm.analyze" in str(exc.value)

    def test_propagate_lets_the_value_reach_the_indices(self):
        """T4: 'propagate' warns, keeps every row, and returns NaN indices."""
        X = _grid_X()
        Y, dfdx = _precomputed(_nan_output_at([3], X), X)
        with pytest.warns(JaxgsaWarning, match="propagate"):
            result = analyze(_two_uniform_problem(), Y=Y, dfdx=dfdx, on_invalid="propagate")
        assert result.invalid.n_invalid == 1
        assert result.invalid.n_kept == X.shape[0] - 1
        assert not np.isfinite(np.asarray(result.var_y))

    def test_drop_matches_the_sample_with_those_rows_removed(self):
        """T4: 'drop' gives what analyzing the surviving rows alone gives."""
        X = _grid_X()
        bad = [3, 11]
        Y, dfdx = _precomputed(_nan_output_at(bad, X), X)
        keep = np.setdiff1d(np.arange(X.shape[0]), bad)

        with pytest.warns(JaxgsaWarning, match="dropped"):
            dropped = analyze(_two_uniform_problem(), Y=Y, dfdx=dfdx, on_invalid="drop")
        reference = analyze(_two_uniform_problem(), Y=Y[keep], dfdx=dfdx[keep])

        np.testing.assert_allclose(np.asarray(dropped.nu), np.asarray(reference.nu), rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(dropped.sigma), np.asarray(reference.sigma), rtol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(dropped.var_y), np.asarray(reference.var_y), rtol=1e-6
        )
        assert dropped.invalid.unit_indices == tuple(bad)

    def test_a_non_finite_jacobian_is_caught_with_a_finite_output(self):
        """T4: dfdx is checked too, not only Y.

        ``nu`` is built from the derivative, so a NaN there poisons the index
        whatever the output does.
        """
        X = _grid_X()
        Y, dfdx = _precomputed(_plain, X)
        dfdx = dfdx.at[5, 0].set(jnp.nan)
        assert bool(jnp.all(jnp.isfinite(Y)))

        with pytest.raises(ValueError, match="non-finite") as exc:
            analyze(_two_uniform_problem(), Y=Y, dfdx=dfdx)
        assert "[5]" in str(exc.value)

        with pytest.warns(JaxgsaWarning, match="dropped"):
            result = analyze(_two_uniform_problem(), Y=Y, dfdx=dfdx, on_invalid="drop")
        assert result.invalid.unit_indices == (5,)
        # Not plain "Y": the output array is finite everywhere here, and only
        # the derivative is not. A report saying "Y" would send a reader to
        # look at the wrong array.
        assert result.invalid.sources == ("Y or its derivative",)
        assert bool(np.all(np.isfinite(np.asarray(result.nu))))

    def test_drop_refuses_when_nothing_usable_is_left(self):
        """T4: a variance needs two rows, so dropping down to one still raises."""
        X = _grid_X(n=4)
        Y, dfdx = _precomputed(_nan_output_at([0, 1, 2], X), X)
        with pytest.raises(ValueError, match="every usable row was removed"):
            analyze(_two_uniform_problem(), Y=Y, dfdx=dfdx, on_invalid="drop")


class TestInvalidPolicyAutodiff:
    """T4: the on_invalid policy on the (fn, X) calling convention.

    This path builds Y itself, inside a jitted vmap, and reduces the Jacobian
    to running totals as it goes. The policy therefore has to act inside that
    loop, not after it.
    """

    def test_raise_is_the_default(self):
        """T4: a NaN the model itself produced refuses the analysis."""
        X = _grid_X()
        with pytest.raises(ValueError, match="non-finite"):
            analyze(_two_uniform_problem(), _nan_output_at([7], X), X)

    def test_propagate_lets_the_value_reach_the_indices(self):
        """T4: 'propagate' warns and returns the non-finite index."""
        X = _grid_X()
        with pytest.warns(JaxgsaWarning, match="propagate"):
            result = analyze(
                _two_uniform_problem(), _nan_output_at([7], X), X, on_invalid="propagate"
            )
        assert result.invalid.n_invalid == 1
        assert not np.isfinite(np.asarray(result.var_y))

    def test_drop_matches_the_sample_with_those_rows_removed(self):
        """T4: the drop happens before the moments accumulate.

        This is the load-bearing claim of the autodiff path. ``sum_jac`` and
        ``sum_jac2`` are running totals over batches, so a bad row folded into
        them cannot be taken out again: masking after the reduction would
        divide the wrong total by the wrong count. The assertion is equality
        with the same model run on the clean rows only.
        """
        X = _grid_X()
        bad = [2, 9]
        fn = _nan_output_at(bad, X)
        keep = np.setdiff1d(np.arange(X.shape[0]), bad)

        with pytest.warns(JaxgsaWarning, match="dropped"):
            dropped = analyze(_two_uniform_problem(), fn, X, on_invalid="drop")
        reference = analyze(_two_uniform_problem(), fn, X[keep])

        np.testing.assert_allclose(np.asarray(dropped.nu), np.asarray(reference.nu), rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(dropped.sigma), np.asarray(reference.sigma), rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(dropped.var_y), np.asarray(reference.var_y), rtol=1e-5
        )
        assert dropped.invalid.unit_indices == tuple(bad)

    def test_a_non_finite_jacobian_is_caught_with_a_finite_output(self):
        """T4: the Jacobian the model produced is checked, not only its output.

        ``_nan_jacobian`` is finite everywhere in the forward pass, so nothing
        in Y says anything is wrong. Only the derivative is NaN.
        """
        import jax

        X = _grid_X()
        bad = np.flatnonzero(np.asarray(X)[:, 0] < 0.5)
        assert bad.size > 2
        assert bool(jnp.all(jnp.isfinite(jax.vmap(_nan_jacobian)(X))))

        with pytest.raises(ValueError, match="non-finite") as exc:
            analyze(_two_uniform_problem(), _nan_jacobian, X)
        assert "jaxgsa.dgsm.analyze" in str(exc.value)

        with pytest.warns(JaxgsaWarning, match="dropped"):
            result = analyze(_two_uniform_problem(), _nan_jacobian, X, on_invalid="drop")
        assert result.invalid.unit_indices == tuple(int(i) for i in bad)
        # Not plain "Y": the output array is finite everywhere here, and only
        # the derivative is not. A report saying "Y" would send a reader to
        # look at the wrong array.
        assert result.invalid.sources == ("Y or its derivative",)
        assert bool(np.all(np.isfinite(np.asarray(result.nu))))

    def test_a_dropped_nan_jacobian_matches_the_clean_subsample(self):
        """T4: dropping a NaN **derivative** also happens before the totals.

        A NaN Jacobian is the strict case. Its row contributes NaN to
        ``sum_jac`` the instant it is summed, so if the mask were applied
        after the reduction ``nu`` would be NaN and this equality could not
        hold at all.
        """
        X = _grid_X()
        keep = np.flatnonzero(np.asarray(X)[:, 0] >= 0.5)

        with pytest.warns(JaxgsaWarning, match="dropped"):
            dropped = analyze(_two_uniform_problem(), _nan_jacobian, X, on_invalid="drop")
        reference = analyze(_two_uniform_problem(), _nan_jacobian, X[keep])

        np.testing.assert_allclose(np.asarray(dropped.nu), np.asarray(reference.nu), rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(dropped.sigma), np.asarray(reference.sigma), rtol=1e-5
        )

    def test_a_non_finite_input_is_reported_as_an_input(self):
        """T4: X is checked as well, and the report separates it from Y."""
        X = np.asarray(_grid_X()).copy()
        X[4, 1] = np.nan
        with pytest.raises(ValueError, match="non-finite") as exc:
            analyze(_two_uniform_problem(), _plain, jnp.asarray(X))
        assert "X" in str(exc.value)

        with pytest.warns(JaxgsaWarning, match="dropped"):
            result = analyze(_two_uniform_problem(), _plain, jnp.asarray(X), on_invalid="drop")
        assert result.invalid.unit_indices == (4,)
        assert "X" in result.invalid.sources


class TestInvalidPolicyShared:
    """T4: behaviour that must agree across both calling conventions."""

    @pytest.mark.parametrize("policy", ["raise", "propagate", "drop"])
    def test_clean_autodiff_sample_reports_nothing(self, policy, recwarn):
        """T4: a clean sample gives an empty report and no warning."""
        result = analyze(_two_uniform_problem(), _plain, _grid_X(), on_invalid=policy)
        assert result.invalid.n_invalid == 0
        assert result.invalid.policy == policy
        assert result.invalid.unit is InvalidUnit.ROW
        assert result.invalid.n_units == 24
        assert not any("non-finite" in str(w.message) for w in recwarn)

    @pytest.mark.parametrize("policy", ["raise", "propagate", "drop"])
    def test_clean_precomputed_sample_reports_nothing(self, policy, recwarn):
        """T4: the same, on the pre-computed convention."""
        X = _grid_X()
        Y, dfdx = _precomputed(_plain, X)
        result = analyze(_two_uniform_problem(), Y=Y, dfdx=dfdx, on_invalid=policy)
        assert result.invalid.n_invalid == 0
        assert result.invalid.policy == policy
        assert not any("non-finite" in str(w.message) for w in recwarn)

    @pytest.mark.parametrize("bad", ["Raise", "skip", "", None, 0, True])
    def test_a_bad_policy_value_is_rejected(self, bad):
        """T4: an unknown on_invalid is refused, on either convention."""
        X = _grid_X()
        with pytest.raises(ValueError, match="on_invalid must be one of"):
            analyze(_two_uniform_problem(), _plain, X, on_invalid=bad)
        Y, dfdx = _precomputed(_plain, X)
        with pytest.raises(ValueError, match="on_invalid must be one of"):
            analyze(_two_uniform_problem(), Y=Y, dfdx=dfdx, on_invalid=bad)

    def test_the_policy_is_checked_before_the_model_runs(self):
        """T4: a misspelled policy costs no model evaluation.

        The autodiff path is the expensive one. Validating on_invalid first
        means a typo is caught without differentiating anything.
        """
        calls = []

        def counted(x):
            calls.append(1)
            return _plain(x)

        with pytest.raises(ValueError, match="on_invalid must be one of"):
            analyze(_two_uniform_problem(), counted, _grid_X(), on_invalid="nope")
        assert calls == []


class TestInvalidPolicyBatched:
    """T4: batching does not change what the policy does."""

    @pytest.mark.parametrize("bad_row", [1, 15, 29])
    def test_a_bad_row_is_dropped_wherever_its_batch_falls(self, bad_row):
        """T4: batch 1, batch 2 and the ragged last batch behave alike.

        The running totals are per batch, so a row in the middle batch is the
        case that a naive "mask at the end" implementation gets wrong. The
        model poisons the derivative rather than the output, which is the case
        that cannot survive being summed.
        """
        X = _grid_X(n=30)
        fn = _nan_jacobian_at([bad_row], X)
        keep = np.setdiff1d(np.arange(30), [bad_row])

        with pytest.warns(JaxgsaWarning, match="dropped"):
            dropped = analyze(_two_uniform_problem(), fn, X, batch_size=10, on_invalid="drop")
        reference = analyze(_two_uniform_problem(), fn, X[keep])

        assert dropped.invalid.unit_indices == (bad_row,)
        np.testing.assert_allclose(np.asarray(dropped.nu), np.asarray(reference.nu), rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(dropped.sigma), np.asarray(reference.sigma), rtol=1e-5
        )

    def test_batched_and_unbatched_drops_agree(self):
        """T4: the same sample gives the same indices with and without batching."""
        X = _grid_X(n=30)
        fn = _nan_jacobian_at([4, 17], X)
        with pytest.warns(JaxgsaWarning, match="dropped"):
            batched = analyze(_two_uniform_problem(), fn, X, batch_size=7, on_invalid="drop")
        with pytest.warns(JaxgsaWarning, match="dropped"):
            whole = analyze(_two_uniform_problem(), fn, X, on_invalid="drop")
        assert batched.invalid.unit_indices == whole.invalid.unit_indices == (4, 17)
        np.testing.assert_allclose(np.asarray(batched.nu), np.asarray(whole.nu), rtol=1e-5)

    def test_a_ragged_last_batch_does_not_invent_a_bad_row(self):
        """T4: the zero padding of the last batch is never checked.

        The pad rows would look non-finite to a model like ``_nan_jacobian``,
        which fails at ``x[0] < 0.5``. They are cut off before the mask is
        built, so a clean sample stays clean.
        """
        X = jnp.asarray(np.stack([np.linspace(0.6, 0.99, 17), np.linspace(1.0, 2.0, 17)], axis=1))
        result = analyze(_two_uniform_problem(), _nan_jacobian, X, batch_size=5)
        assert result.invalid.n_invalid == 0
