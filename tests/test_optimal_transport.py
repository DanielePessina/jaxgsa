"""Tests for optimal-transport sensitivity analysis."""

from __future__ import annotations

from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gsax
from gsax.benchmarks import gaussian_linear, ishigami
from gsax.optimal_transport import OTResult, analyze
from gsax.sampling import sample_mc


@pytest.fixture(scope="module")
def ishigami_data():
    """Generate Ishigami test data (N large enough for the published anchor)."""
    N = 8192
    X = jnp.asarray(sample_mc(ishigami.PROBLEM, N=N, seed=42))
    Y = ishigami.evaluate(X)
    return X, Y


@pytest.fixture(scope="module")
def gaussian_linear_data():
    """Generate Gaussian linear test data (large N for ground-truth checks)."""
    N = 32000
    X = jnp.asarray(sample_mc(gaussian_linear.PROBLEM, N=N, seed=1))
    Y = gaussian_linear.evaluate(X)
    return X, Y


@pytest.fixture(scope="module")
def multi_output_data(ishigami_data):
    """Ishigami-based multi-output (N, 2) and time-series (N, 2, 2) outputs."""
    X, Y = ishigami_data
    Y2 = jnp.stack([Y, Y**2], axis=1)  # (N, 2)
    Y3 = jnp.stack([Y2, Y2 + 1.0], axis=1)  # (N, 2, 2)
    return X, Y2, Y3


class TestOTBasic:
    def test_returns_ot_result(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y)
        assert isinstance(result, OTResult)
        assert result.mode == "univariate"

    def test_shape_scalar_output(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y)
        assert result.ot.shape == (3,)
        assert result.advective.shape == (3,)
        assert result.diffusive.shape == (3,)
        assert result.ot_conf is None
        assert result.advective_conf is None
        assert result.diffusive_conf is None
        assert result.ot_dummy is None

    def test_values_in_unit_interval(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y)
        for arr in (result.ot, result.advective, result.diffusive):
            a = np.asarray(arr)
            assert np.all(a >= 0.0) and np.all(a <= 1.0)

    def test_decomposition_sums_to_total(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y)
        np.testing.assert_allclose(
            np.asarray(result.ot),
            np.asarray(result.advective + result.diffusive),
            atol=1e-6,
        )

    def test_x1_x2_more_important_than_x3(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y)
        ot = np.asarray(result.ot)
        assert ot[0] > ot[2]
        assert ot[1] > ot[2]

    def test_deterministic(self, ishigami_data):
        X, Y = ishigami_data
        r1 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=10, seed=7)
        r2 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=10, seed=7)
        np.testing.assert_array_equal(np.asarray(r1.ot), np.asarray(r2.ot))
        assert r1.ot_conf is not None and r2.ot_conf is not None
        np.testing.assert_array_equal(np.asarray(r1.ot_conf), np.asarray(r2.ot_conf))

    def test_chunk_size_invariance(self, multi_output_data):
        X, _, Y3 = multi_output_data
        r_default = analyze(ishigami.PROBLEM, X, Y3)
        r_chunked = analyze(ishigami.PROBLEM, X, Y3, chunk_size=1)
        np.testing.assert_allclose(np.asarray(r_default.ot), np.asarray(r_chunked.ot), atol=1e-7)

    def test_n_partitions_changes_result(self, ishigami_data):
        X, Y = ishigami_data
        r25 = analyze(ishigami.PROBLEM, X, Y, n_partitions=25)
        r10 = analyze(ishigami.PROBLEM, X, Y, n_partitions=10)
        assert not np.allclose(np.asarray(r25.ot), np.asarray(r10.ot))

    def test_monotone_transform_invariance(self, ishigami_data):
        """Rank-based conditioning: monotone input transforms change nothing."""
        X, Y = ishigami_data
        X_warp = X.at[:, 0].set(jnp.exp(X[:, 0]))
        r_raw = analyze(ishigami.PROBLEM, X, Y)
        r_warp = analyze(ishigami.PROBLEM, X_warp, Y)
        np.testing.assert_array_equal(np.asarray(r_raw.ot), np.asarray(r_warp.ot))


class TestOTPOTComparison:
    """Numerical parity against POT (MIT-licensed reference)."""

    def test_pot_wasserstein_1d_semantics_pin(self):
        """Pin that ot.wasserstein_1d(p=2) returns W_2^2, not W_2.

        Two point masses at 0 vs two at 1: W_2^2 = 1.0 (and W_2 = 1.0 too),
        so add an asymmetric case where they differ: {0} vs {2} gives
        W_2^2 = 4, W_2 = 2.
        """
        ot = pytest.importorskip("ot")
        val = ot.wasserstein_1d(np.array([0.0]), np.array([2.0]), p=2)
        assert val == pytest.approx(4.0), "POT changed wasserstein_1d semantics"

    def test_per_bin_w2_matches_pot(self):
        """Per-bin 1-D W2^2 vs POT in the exact regime (n_m divides N)."""
        ot = pytest.importorskip("ot")
        from gsax._partition import _build_class_indices, _class_layout
        from gsax.optimal_transport._analyze import _ot_1d_kernel, _quantile_ranks

        N, M = 1000, 25  # n_m = 40 divides N -> exact quantile coupling
        key = jax.random.PRNGKey(1)
        X = jax.random.uniform(key, (N, 2))
        Y = jnp.sin(3 * X[:, 0]) + 0.3 * jax.random.normal(jax.random.PRNGKey(2), (N,))

        take, mask, sizes = _class_layout(N, M)
        all_idx = jnp.arange(N, dtype=jnp.int32)[None, :]
        cls_idx = _build_class_indices(X, all_idx, jnp.asarray(take))
        ot_idx, _, _, _ = _ot_1d_kernel(
            Y[:, None],
            all_idx,
            cls_idx,
            jnp.asarray(mask),
            jnp.asarray(sizes),
            jnp.asarray(_quantile_ranks(sizes, N)),
        )

        y_np = np.asarray(Y, dtype=np.float64)
        V = 2 * np.var(y_np, ddof=1)
        cls = np.asarray(cls_idx[0])
        ref = np.zeros(2)
        for d in range(2):
            acc = 0.0
            for m in range(M):
                cond = y_np[cls[d, m][mask[m]]]
                acc += sizes[m] / N * ot.wasserstein_1d(y_np, cond, p=2)
            ref[d] = acc / V
        np.testing.assert_allclose(np.asarray(ot_idx[0, 0]), ref, atol=1e-4)

    def test_sinkhorn_matches_pot_and_approaches_emd(self):
        """Solver-level parity vs ot.sinkhorn2 and monotone approach to ot.emd2."""
        ot = pytest.importorskip("ot")
        from gsax.optimal_transport._solver import _sinkhorn_w2

        rng = np.random.default_rng(0)
        N, P, E = 200, 40, 2
        Zu = rng.normal(size=(N, E))
        Zc = rng.normal(loc=0.5, size=(P, E))
        C = ((Zu[:, None, :] - Zc[None, :, :]) ** 2).sum(-1)
        a = np.full(N, 1.0 / N)
        b = np.full(P, 1.0 / P)
        log_b = jnp.log(jnp.full(P, 1.0 / P))

        emd = ot.emd2(a, b, C)
        prev = np.inf
        for eps in (0.05, 0.02, 0.008):
            # POT solves on the same max-scaled cost our solver uses
            # internally, so the eps values mean the same thing.
            ref = ot.sinkhorn2(a, b, C / C.max(), reg=eps, numItermax=20000, stopThr=1e-12)
            ref *= C.max()
            ours, err = _sinkhorn_w2(
                jnp.asarray(C), log_b, jnp.asarray(eps), jnp.asarray(20000), jnp.asarray(1e-6)
            )
            assert float(err) <= 1e-6
            assert float(ours) == pytest.approx(ref, rel=1e-4)
            assert float(ours) < prev  # entropic cost decreases toward exact
            prev = float(ours)
        assert prev > emd  # entropic plan cost upper-bounds the exact cost
        assert prev == pytest.approx(emd, rel=0.25)  # loose: eps=0.008 is still biased

    def test_sinkhorn_pad_invariance(self):
        """Padded target columns (zero mass, zeroed cost) change nothing.

        Pads are clamped duplicates of a real sample, so the kernel zeroes
        their cost columns before solving -- otherwise an outlier pad
        would set the max-cost scale and change the effective epsilon.
        This mimics that contract with deliberately extreme pad points.
        """
        from gsax.optimal_transport._solver import _sinkhorn_w2

        rng = np.random.default_rng(3)
        N, P, P_pad, E = 100, 20, 27, 3
        Zu = rng.normal(size=(N, E))
        Zc = rng.normal(size=(P, E))
        Zc_pad = np.vstack([Zc, np.full((P_pad - P, E), 50.0)])  # outlier pads
        C = ((Zu[:, None, :] - Zc[None, :, :]) ** 2).sum(-1)
        C_pad = ((Zu[:, None, :] - Zc_pad[None, :, :]) ** 2).sum(-1)
        C_pad[:, P:] = 0.0  # the kernel's mask-zeroing of pad columns
        log_b = np.full(P_pad, -np.inf)
        log_b[:P] = -np.log(P)

        args = (jnp.asarray(0.02), jnp.asarray(5000), jnp.asarray(1e-6))
        ref, _ = _sinkhorn_w2(jnp.asarray(C), jnp.log(jnp.full(P, 1.0 / P)), *args)
        padded, _ = _sinkhorn_w2(jnp.asarray(C_pad), jnp.asarray(log_b), *args)
        assert float(padded) == pytest.approx(float(ref), abs=1e-6)


class TestOTAnalytic:
    def test_gaussian_additive_closed_form(self):
        """Y = X1 + X2 with N(0, s_i^2): exact continuum index per input."""
        s1, s2 = 1.0, 2.0
        v = s1**2 + s2**2
        key1, key2 = jax.random.split(jax.random.PRNGKey(11))
        N = 32000
        X = jnp.stack(
            [
                s1 * jax.random.normal(key1, (N,)),
                s2 * jax.random.normal(key2, (N,)),
            ],
            axis=1,
        )
        Y = X[:, 0] + X[:, 1]
        problem = gsax.Problem(("x1", "x2"), ((-6.0, 6.0), (-12.0, 12.0)))
        result = analyze(problem, X, Y, n_partitions=50)

        expected = np.array(
            [
                (s1**2 + (np.sqrt(v) - s2) ** 2) / (2 * v),
                (s2**2 + (np.sqrt(v) - s1) ** 2) / (2 * v),
            ]
        )
        np.testing.assert_allclose(np.asarray(result.ot), expected, atol=0.015)
        # advective = c_i^2 s_i^2 / (2v), diffusive = (sqrt(v) - s_other)^2 / (2v)
        exp_adv = np.array([s1**2, s2**2]) / (2 * v)
        exp_diff = np.array([(np.sqrt(v) - s2) ** 2, (np.sqrt(v) - s1) ** 2]) / (2 * v)
        np.testing.assert_allclose(np.asarray(result.advective), exp_adv, atol=0.01)
        np.testing.assert_allclose(np.asarray(result.diffusive), exp_diff, atol=0.01)

    def test_gaussian_linear_analytical_ot(self, gaussian_linear_data):
        """Convergence to the closed-form ANALYTICAL_OT constant."""
        X, Y = gaussian_linear_data
        result = analyze(gaussian_linear.PROBLEM, X, Y, n_partitions=50)
        np.testing.assert_allclose(np.asarray(result.ot), gaussian_linear.ANALYTICAL_OT, atol=0.02)

    def test_advective_is_half_s1(self, gaussian_linear_data):
        """2 * advective == first-order Sobol index (exact tie to Sobol)."""
        X, Y = gaussian_linear_data
        result = analyze(gaussian_linear.PROBLEM, X, Y, n_partitions=50)
        np.testing.assert_allclose(
            2.0 * np.asarray(result.advective), gaussian_linear.ANALYTICAL_S1, atol=0.02
        )

    def test_advective_is_half_s1_ishigami(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y)
        np.testing.assert_allclose(
            2.0 * np.asarray(result.advective), ishigami.ANALYTICAL_S1, atol=0.05
        )

    def test_ishigami_published_anchor(self, ishigami_data):
        """Published Var(Y)-normalized W2^2 for Ishigami X1 is ~0.423.

        gsax normalizes by 2*Var(Y), so the anchor is compared at 2*ot.
        """
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y)
        assert 2.0 * float(result.ot[0]) == pytest.approx(0.423, abs=0.03)

    def test_mixed_uniform_gaussian_marginals(self):
        """Rank-based conditioning handles mixed input marginals unchanged."""
        problem = gsax.Problem.from_dict(
            {
                "xu": (-1.0, 1.0),
                "xg": gsax.GaussianInputSpec(dist="gaussian", mean=0.0, variance=4.0),
            }
        )
        X = jnp.asarray(sample_mc(problem, N=16000, seed=5))
        Y = X[:, 0] + X[:, 1]
        result = analyze(problem, X, Y, n_partitions=40)
        ot = np.asarray(result.ot)
        assert np.all(ot > 0.0) and np.all(ot <= 1.0)
        # xg carries variance 4 vs the uniform's 1/3 -> must dominate.
        assert ot[1] > ot[0]
        # Advective tie to the analytic S1 holds for any marginals.
        var_u, var_g = 1.0 / 3.0, 4.0
        s1 = np.array([var_u, var_g]) / (var_u + var_g)
        np.testing.assert_allclose(2.0 * np.asarray(result.advective), s1, atol=0.03)

    def test_correlated_inputs_closed_form(self):
        """Correlated Gaussian inputs: index matches the conditional-normal formula.

        For jointly Gaussian (X, Y), Y | X_i = x is Gaussian with regression
        mean and residual variance, so the continuum OT index is
        ``(cov_i^2/s_i^2 + (sqrt(v_y) - sqrt(v_y - cov_i^2/s_i^2))^2) / (2 v_y)``
        with ``cov_i = Cov(Y, X_i)``. The index then measures total
        (correlation-inclusive) influence.
        """
        rho = 0.5
        coeffs = np.array([4.0, -2.0, 1.0])
        Sigma = np.full((3, 3), rho) + (1 - rho) * np.eye(3)
        L = np.linalg.cholesky(Sigma)
        rng = np.random.default_rng(17)
        X = rng.normal(size=(40000, 3)) @ L.T
        Y = X @ coeffs

        v_y = float(coeffs @ Sigma @ coeffs)
        cov = Sigma @ coeffs  # Cov(Y, X_i); inputs have unit variance
        adv = cov**2  # cov_i^2 / s_i^2 with s_i^2 = 1
        v_cond = v_y - adv
        expected = (adv + (np.sqrt(v_y) - np.sqrt(v_cond)) ** 2) / (2 * v_y)

        problem = gsax.Problem(("x1", "x2", "x3"), ((-6.0, 6.0),) * 3)
        result = analyze(problem, jnp.asarray(X), jnp.asarray(Y), n_partitions=50)
        np.testing.assert_allclose(np.asarray(result.ot), expected, atol=0.02)


class TestOTPointCloud:
    def test_multivariate_shape(self, multi_output_data):
        X, Y2, _ = multi_output_data
        result = analyze(ishigami.PROBLEM, X, Y2, mode="multivariate", n_partitions=10)
        assert result.ot.shape == (3,)
        assert result.mode == "multivariate"

    def test_trajectory_shape(self, multi_output_data):
        X, _, Y3 = multi_output_data
        result = analyze(ishigami.PROBLEM, X, Y3, mode="trajectory", n_partitions=10)
        assert result.ot.shape == (2, 3)  # (K, D)
        assert result.mode == "trajectory"

    def test_multivariate_decomposition_sums_to_total(self, multi_output_data):
        X, Y2, _ = multi_output_data
        result = analyze(ishigami.PROBLEM, X, Y2, mode="multivariate", n_partitions=10)
        np.testing.assert_allclose(
            np.asarray(result.ot),
            np.asarray(result.advective + result.diffusive),
            atol=1e-5,
        )

    def test_multivariate_scalar_close_to_univariate(self, ishigami_data):
        """On scalar Y small-eps small-eps point-cloud transport approaches the exact 1-D index."""
        X, Y = ishigami_data
        # Subsample: the multivariate mode solves N x (N/M) transport problems.
        Xs, Ys = X[:2000], Y[:2000]
        sep = analyze(ishigami.PROBLEM, Xs, Ys, n_partitions=10)
        joint = analyze(
            ishigami.PROBLEM, Xs, Ys, mode="multivariate", n_partitions=10, epsilon=0.005
        )
        sep_ot = np.asarray(sep.ot)
        joint_ot = np.asarray(joint.ot)
        # Entropic + finite-sample bias is strictly upward.
        assert np.all(joint_ot > sep_ot - 1e-4)
        np.testing.assert_allclose(joint_ot, sep_ot, atol=0.12)

    def test_standardize_matters_for_mismatched_scales(self, multi_output_data):
        X, Y2, _ = multi_output_data
        Y_scaled = Y2.at[:, 1].multiply(1e4)
        r_std = analyze(ishigami.PROBLEM, X, Y_scaled, mode="multivariate", n_partitions=10)
        r_raw = analyze(
            ishigami.PROBLEM, X, Y_scaled, mode="multivariate", n_partitions=10, standardize=False
        )
        assert not np.allclose(np.asarray(r_std.ot), np.asarray(r_raw.ot), atol=1e-3)

    def test_unconverged_sinkhorn_warns(self, multi_output_data):
        X, Y2, _ = multi_output_data
        with pytest.warns(UserWarning, match="Sinkhorn"):
            analyze(
                ishigami.PROBLEM,
                X[:500],
                Y2[:500],
                mode="multivariate",
                n_partitions=5,
                max_iter=1,
            )


class TestOTBootstrap:
    def test_conf_shapes_scalar(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=20)
        for conf in (result.ot_conf, result.advective_conf, result.diffusive_conf):
            assert conf is not None
            assert conf.shape == (2, 3)
            assert np.all(np.asarray(conf[0]) <= np.asarray(conf[1]))

    def test_conf_shapes_multi_output(self, multi_output_data):
        X, Y2, _ = multi_output_data
        result = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=20)
        assert result.ot_conf is not None
        assert result.ot_conf.shape == (2, 2, 3)

    def test_point_estimate_independent_of_bootstrap(self, ishigami_data):
        X, Y = ishigami_data
        r0 = analyze(ishigami.PROBLEM, X, Y)
        rb = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=15)
        np.testing.assert_array_equal(np.asarray(r0.ot), np.asarray(rb.ot))

    def test_different_seeds_differ(self, ishigami_data):
        X, Y = ishigami_data
        r1 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=15, seed=1)
        r2 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=15, seed=2)
        assert r1.ot_conf is not None and r2.ot_conf is not None
        assert not np.array_equal(np.asarray(r1.ot_conf), np.asarray(r2.ot_conf))

    def test_multivariate_bootstrap(self, multi_output_data):
        X, Y2, _ = multi_output_data
        result = analyze(
            ishigami.PROBLEM,
            X[:1500],
            Y2[:1500],
            mode="multivariate",
            n_partitions=5,
            n_bootstrap=5,
        )
        assert result.ot_conf is not None
        assert result.ot_conf.shape == (2, 3)


class TestOTMultiOutput:
    def test_multi_output_shape(self, multi_output_data):
        X, Y2, _ = multi_output_data
        result = analyze(ishigami.PROBLEM, X, Y2)
        assert result.ot.shape == (2, 3)

    def test_time_series_shape(self, multi_output_data):
        X, _, Y3 = multi_output_data
        result = analyze(ishigami.PROBLEM, X, Y3)
        assert result.ot.shape == (2, 2, 3)

    def test_first_column_matches_scalar_run(self, multi_output_data, ishigami_data):
        X, Y2, _ = multi_output_data
        _, Y = ishigami_data
        r_multi = analyze(ishigami.PROBLEM, X, Y2)
        r_scalar = analyze(ishigami.PROBLEM, X, Y)
        np.testing.assert_allclose(np.asarray(r_multi.ot[0]), np.asarray(r_scalar.ot), atol=1e-7)


class TestOTEdgeCases:
    def test_constant_output_gives_zero(self, ishigami_data):
        X, _ = ishigami_data
        Y_const = jnp.ones(X.shape[0])
        result = analyze(ishigami.PROBLEM, X, Y_const)
        np.testing.assert_array_equal(np.asarray(result.ot), np.zeros(3))
        assert not np.any(np.isnan(np.asarray(result.ot)))

    def test_constant_column_among_varying(self, ishigami_data):
        X, Y = ishigami_data
        Y2 = jnp.stack([Y, jnp.ones_like(Y)], axis=1)
        result = analyze(ishigami.PROBLEM, X, Y2)
        assert not np.any(np.isnan(np.asarray(result.ot)))
        np.testing.assert_array_equal(np.asarray(result.ot[1]), np.zeros(3))
        assert np.all(np.asarray(result.ot[0]) > 0)

    def test_ragged_bins(self, ishigami_data):
        """N not divisible by n_partitions still runs and stays in [0, 1]."""
        X, Y = ishigami_data
        Xr, Yr = X[:1013], Y[:1013]  # prime N
        result = analyze(ishigami.PROBLEM, Xr, Yr, n_partitions=25)
        ot = np.asarray(result.ot)
        assert np.all(ot >= 0.0) and np.all(ot <= 1.0)

    def test_ragged_close_to_divisible(self, ishigami_data):
        X, Y = ishigami_data
        r_div = analyze(ishigami.PROBLEM, X[:1000], Y[:1000], n_partitions=25)
        r_rag = analyze(ishigami.PROBLEM, X[:1013], Y[:1013], n_partitions=25)
        np.testing.assert_allclose(np.asarray(r_div.ot), np.asarray(r_rag.ot), atol=0.05)

    def test_dummy_baseline(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, dummy=True)
        assert result.ot_dummy is not None
        assert result.ot_dummy.shape == ()
        # A synthetic independent input must sit far below real inputs.
        assert float(result.ot_dummy) < float(np.asarray(result.ot).min())
        assert float(result.ot_dummy) < 0.05

    def test_dummy_multivariate_mode(self, multi_output_data):
        X, Y2, _ = multi_output_data
        result = analyze(
            ishigami.PROBLEM, X[:1500], Y2[:1500], mode="multivariate", n_partitions=5, dummy=True
        )
        assert result.ot_dummy is not None
        assert result.ot_dummy.shape == ()


class TestOTValidation:
    def test_x_not_2d(self, ishigami_data):
        _, Y = ishigami_data
        with pytest.raises(ValueError, match="2-D"):
            analyze(ishigami.PROBLEM, jnp.ones(10), Y[:10])

    def test_wrong_param_count(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="parameters"):
            analyze(ishigami.PROBLEM, X[:, :2], Y)

    def test_row_mismatch(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="sample rows"):
            analyze(ishigami.PROBLEM, X, Y[:100])

    def test_bad_mode(self, ishigami_data):
        X, Y = ishigami_data
        # cast so the deliberately invalid literal exercises the runtime check
        with pytest.raises(ValueError, match="mode"):
            analyze(ishigami.PROBLEM, X, Y, mode=cast(Literal["univariate"], "sliced"))

    def test_trajectory_needs_3d(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="trajectory"):
            analyze(ishigami.PROBLEM, X, Y, mode="trajectory")

    def test_bad_n_partitions(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="n_partitions"):
            analyze(ishigami.PROBLEM, X, Y, n_partitions=1)
        with pytest.raises(ValueError, match="n_partitions"):
            analyze(ishigami.PROBLEM, X, Y, n_partitions=X.shape[0])

    def test_bad_epsilon(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="epsilon"):
            analyze(ishigami.PROBLEM, X, Y, epsilon=0.0)

    def test_bad_max_iter(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="max_iter"):
            analyze(ishigami.PROBLEM, X, Y, max_iter=0)

    def test_bad_tol(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="tol"):
            analyze(ishigami.PROBLEM, X, Y, tol=0.0)

    def test_bad_n_bootstrap(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="n_bootstrap"):
            analyze(ishigami.PROBLEM, X, Y, n_bootstrap=-1)

    def test_bad_conf_level(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="conf_level"):
            analyze(ishigami.PROBLEM, X, Y, n_bootstrap=10, conf_level=1.5)

    def test_bad_chunk_size(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="chunk_size"):
            analyze(ishigami.PROBLEM, X, Y, chunk_size=0)


class TestOTXarray:
    def test_scalar_dims(self, ishigami_data):
        X, Y = ishigami_data
        ds = analyze(ishigami.PROBLEM, X, Y).to_dataset()
        assert set(ds.data_vars) == {"ot", "advective", "diffusive"}
        assert ds["ot"].dims == ("param",)
        assert ds.attrs["mode"] == "univariate"

    def test_time_series_dims(self, multi_output_data):
        X, _, Y3 = multi_output_data
        ds = analyze(ishigami.PROBLEM, X, Y3).to_dataset(time_coords=[0.5, 1.0])
        assert ds["ot"].dims == ("time", "output", "param")
        assert list(ds.coords["time"].values) == [0.5, 1.0]

    def test_trajectory_dims(self, multi_output_data):
        X, _, Y3 = multi_output_data
        ds = analyze(ishigami.PROBLEM, X, Y3, mode="trajectory", n_partitions=10).to_dataset()
        assert ds["ot"].dims == ("output", "param")
        assert ds.attrs["mode"] == "trajectory"

    def test_conf_and_dummy_vars(self, ishigami_data):
        X, Y = ishigami_data
        ds = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=10, dummy=True).to_dataset()
        assert {"ot_lower", "ot_upper", "advective_lower", "diffusive_upper"} <= set(ds.data_vars)
        assert "ot_dummy" in ds.data_vars
        assert ds["ot_dummy"].dims == ()
