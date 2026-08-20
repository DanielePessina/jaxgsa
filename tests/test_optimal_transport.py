"""Tests for optimal-transport sensitivity analysis."""

from __future__ import annotations

import warnings
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa._core.partition import _build_class_indices, _class_layout
from jaxgsa.benchmarks import gaussian_linear, ishigami
from jaxgsa.optimal_transport import analyze
from jaxgsa.sampling import monte_carlo


@pytest.fixture(scope="module")
def ishigami_data():
    """Generate Ishigami test data (N large enough for the published anchor)."""
    N = 8192
    X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=N, seed=42))
    Y = ishigami.evaluate(X)
    return X, Y


@pytest.fixture(scope="module")
def gaussian_linear_data():
    """Generate Gaussian linear test data (large N for ground-truth checks)."""
    N = 32000
    X = jnp.asarray(monte_carlo(gaussian_linear.PROBLEM, n=N, seed=1))
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
    def test_scalar_output_selects_univariate_mode(self, ishigami_data):
        """Tier T4 (internal consistency): a scalar output picks the univariate mode."""
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y)
        assert result.mode == "univariate"

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
        r1 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=10, key=jax.random.key(7))
        r2 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=10, key=jax.random.key(7))
        np.testing.assert_array_equal(np.asarray(r1.ot), np.asarray(r2.ot))
        assert r1.ot_conf is not None and r2.ot_conf is not None
        np.testing.assert_array_equal(np.asarray(r1.ot_conf), np.asarray(r2.ot_conf))

    def test_slice_chunk_size_invariance(self, multi_output_data):
        X, _, Y3 = multi_output_data
        r_default = analyze(ishigami.PROBLEM, X, Y3)
        r_chunked = analyze(ishigami.PROBLEM, X, Y3, slice_chunk_size=1)
        np.testing.assert_allclose(np.asarray(r_default.ot), np.asarray(r_chunked.ot), atol=1e-7)

    def test_memory_budget_narrows_the_working_set(self, multi_output_data):
        """A small budget still returns the same answer.

        The default chunk width comes from the active memory budget and the
        working set in bytes, so a budget of 1 MiB has to cut the output
        into narrower chunks than the default 512 MiB does. The budget
        changes how the work is cut up, never what it computes. The
        tolerance is there because a narrower vmap lets XLA reassociate the
        float32 reductions, the same reason the chunk-size test beside this
        one carries one.
        """
        X, _, Y3 = multi_output_data
        r_default = analyze(ishigami.PROBLEM, X, Y3)
        jaxgsa.config.set_memory_budget(1, unit="mib")
        try:
            r_small = analyze(ishigami.PROBLEM, X, Y3)
        finally:
            jaxgsa.config.set_memory_budget(512, unit="mib")
        np.testing.assert_allclose(np.asarray(r_default.ot), np.asarray(r_small.ot), atol=1e-7)

    def test_unstable_class_sort_matches_a_stable_one(self, ishigami_data):
        """The kernel's unstable sort must return the stable sort's array.

        Ordering the class members is the most expensive step in this
        method, and the kernel asks an unstable sort for it because it
        wants the order statistics and never which tied element came
        first. Two float32 values that compare equal are the same bits,
        so the two sorts have to agree element for element. This pins
        that at the shapes and the data the kernel actually sees, NaNs
        included, because an unstable sort that moved a NaN or a signed
        zero somewhere else would be a silent change of answer.
        """
        X, Y = ishigami_data
        n = X.shape[0]
        take, _ = _class_layout(n, 8)
        idx = jnp.arange(n, dtype=jnp.int32)[None, :]
        cls_idx = _build_class_indices(X, idx, jnp.asarray(take))[0]  # (D, M, P)
        for values in (Y, Y.at[:5].set(jnp.nan), Y.at[:5].set(-0.0).at[5:10].set(0.0)):
            y_cls = values[cls_idx]
            stable = jnp.sort(y_cls, axis=-1)
            unstable = jax.lax.sort(y_cls, dimension=-1, is_stable=False)
            np.testing.assert_array_equal(np.asarray(stable), np.asarray(unstable), strict=True)

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

    @pytest.mark.parametrize(
        ("N", "M", "label"),
        [
            (1000, 25, "divisible: n_m = 40 divides N"),
            (1013, 25, "not divisible: n_m does not divide N (Section 5)"),
        ],
    )
    def test_per_bin_w2_matches_pot(self, N, M, label):
        """Per-bin 1-D W2^2 vs POT, divisible and not.

        The exact quantile coupling (:func:`_quantile_rank_split`) couples
        every mass point to at most two class members with a split weight,
        so it is exact for any class size, not only one that divides N. The
        prior version of this test covered the divisible case only.
        """
        ot = pytest.importorskip("ot")
        from jaxgsa._core.partition import _build_class_indices, _class_layout
        from jaxgsa.optimal_transport._analyze import _ot_1d_kernel

        key = jax.random.PRNGKey(1)
        X = jax.random.uniform(key, (N, 2))
        Y = jnp.sin(3 * X[:, 0]) + 0.3 * jax.random.normal(jax.random.PRNGKey(2), (N,))

        take, sizes = _class_layout(N, M)
        all_idx = jnp.arange(N, dtype=jnp.int32)[None, :]
        cls_idx = _build_class_indices(X, all_idx, jnp.asarray(take))
        # Canonical partition-group layout: (cls_idx, counts) with
        # length-1 leading axes on the shared counts.
        group = (cls_idx, jnp.asarray(sizes)[None, None, :])
        ot_idx, _, _, _ = _ot_1d_kernel(Y[:, None], all_idx, (group,), (None,))

        y_np = np.asarray(Y, dtype=np.float64)
        V = 2 * np.var(y_np, ddof=1)
        cls = np.asarray(cls_idx[0])
        ref = np.zeros(2)
        for d in range(2):
            acc = 0.0
            for m in range(M):
                cond = y_np[cls[d, m][: sizes[m]]]
                acc += sizes[m] / N * ot.wasserstein_1d(y_np, cond, p=2)
            ref[d] = acc / V
        np.testing.assert_allclose(np.asarray(ot_idx[0, 0]), ref, atol=1e-4)

    def test_sinkhorn_matches_pot_and_approaches_emd(self):
        """Solver-level parity vs ot.sinkhorn2 and monotone approach to ot.emd2."""
        ot = pytest.importorskip("ot")
        from jaxgsa.optimal_transport._solver import _sinkhorn_w2

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
            # Pass the cost matrix's own max as the scale, so POT's
            # convention (dividing by C.max()) and ours mean the same eps.
            ref = ot.sinkhorn2(a, b, C / C.max(), reg=eps, numItermax=20000, stopThr=1e-12)
            ref *= C.max()
            ours, err = _sinkhorn_w2(
                jnp.asarray(C),
                log_b,
                jnp.asarray(eps),
                jnp.asarray(20000),
                jnp.asarray(1e-6),
                jnp.asarray(C.max()),
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
        their cost columns before solving. This mimics that contract with
        deliberately extreme pad points, holding the scale fixed across
        both calls -- the caller's own normalizer (M3), not each cost
        matrix's own max, so an outlier pad cannot even reach the scale.
        """
        from jaxgsa.optimal_transport._solver import _sinkhorn_w2

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

        args = (jnp.asarray(0.02), jnp.asarray(5000), jnp.asarray(1e-6), jnp.asarray(C.max()))
        ref, _ = _sinkhorn_w2(jnp.asarray(C), jnp.log(jnp.full(P, 1.0 / P)), *args)
        padded, _ = _sinkhorn_w2(jnp.asarray(C_pad), jnp.asarray(log_b), *args)
        assert float(padded) == pytest.approx(float(ref), abs=1e-6)

    def test_multivariate_mode_matches_pot_exact_emd(self):
        """T2: the whole 'multivariate' pipeline against POT's exact emd2.

        Nothing in the suite checked the point-cloud modes against an
        external oracle end to end (only the solver alone, above), which is
        exactly the gap M3 slipped through: the Sinkhorn regularization was
        scaled by each class's own max cost instead of the index's own
        normalizer, biasing 'ot' by tens of percent, and no test caught it.
        This drives epsilon small enough that the entropic bias is under
        the tolerance, and cross-checks jaxgsa's whole per-class weighting
        and normalization against POT's exact transport, not just the
        solver kernel.
        """
        ot = pytest.importorskip("ot")

        N, M = 300, 5
        rng = np.random.default_rng(7)
        X = rng.uniform(size=(N, 2))
        Y = np.stack(
            [
                np.sin(3 * X[:, 0]) + 0.2 * rng.normal(size=N),
                0.5 * X[:, 0] + X[:, 1] ** 2 + 0.2 * rng.normal(size=N),
            ],
            axis=1,
        )
        problem = jaxgsa.Problem(("x1", "x2"), ((0.0, 1.0), (0.0, 1.0)))
        # epsilon is deliberately small (tight entropic bias) so 20000
        # iterations do not always reach tol; that non-convergence warning
        # is expected here and not what this test checks.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=jaxgsa.JaxgsaWarning)
            result = analyze(
                problem,
                jnp.asarray(X),
                jnp.asarray(Y),
                mode="multivariate",
                n_partitions=M,
                epsilon=0.002,
                max_iter=20000,
                tol=1e-6,
                verbose=False,
            )

        V = 2.0 * np.var(Y, axis=0, ddof=1).sum()
        edges = np.floor(np.linspace(0.0, N, M + 1)).astype(np.int64)
        ref = np.zeros(2)
        for d in range(2):
            order = np.argsort(X[:, d])
            for m in range(M):
                cls = order[edges[m] : edges[m + 1]]
                w = cls.shape[0] / N
                a = np.full(N, 1.0 / N)
                b = np.full(cls.shape[0], 1.0 / cls.shape[0])
                C = ((Y[:, None, :] - Y[cls][None, :, :]) ** 2).sum(-1)
                ref[d] += w * ot.emd2(a, b, C)
            ref[d] /= V

        np.testing.assert_allclose(np.asarray(result.ot), ref, atol=0.01)


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
        problem = jaxgsa.Problem(("x1", "x2"), ((-6.0, 6.0), (-12.0, 12.0)))
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

    def test_ishigami_x1_matches_pot_exact_emd(self, ishigami_data):
        """T2: jaxgsa's ot(x1) on the Ishigami benchmark against POT's exact emd2.

        A previous version of this test asserted a literal ``0.423`` labeled
        "Published", with no citation anywhere in the repository or the
        commit that added it, and no reviewer could find it in the OT
        paper's own Ishigami table. An unsourced number is not an oracle,
        so this replaces it with a live comparison against POT on the same
        benchmark data, at the default ``n_partitions``.
        """
        ot = pytest.importorskip("ot")
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y)

        N = X.shape[0]
        M = min(25, N // 2)
        y_np = np.asarray(Y, dtype=np.float64)
        V = 2.0 * np.var(y_np, ddof=1)
        order = np.argsort(np.asarray(X[:, 0]))
        edges = np.floor(np.linspace(0.0, N, M + 1)).astype(np.int64)
        ref = 0.0
        for m in range(M):
            cls = order[edges[m] : edges[m + 1]]
            ref += cls.shape[0] / N * ot.wasserstein_1d(y_np, y_np[cls], p=2)
        ref /= V

        assert float(result.ot[0]) == pytest.approx(ref, rel=1e-3)

    def test_mixed_uniform_gaussian_marginals(self):
        """Rank-based conditioning handles mixed input marginals unchanged."""
        problem = jaxgsa.Problem.from_dict(
            {
                "xu": (-1.0, 1.0),
                "xg": jaxgsa.GaussianInputSpec(dist="gaussian", mean=0.0, variance=4.0),
            }
        )
        X = jnp.asarray(monte_carlo(problem, n=16000, seed=5))
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

        problem = jaxgsa.Problem(("x1", "x2", "x3"), ((-6.0, 6.0),) * 3)
        result = analyze(problem, jnp.asarray(X), jnp.asarray(Y), n_partitions=50)
        np.testing.assert_allclose(np.asarray(result.ot), expected, atol=0.02)


class TestOTPointCloud:
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

    def test_standardize_outputs_matters_for_mismatched_scales(self, multi_output_data):
        X, Y2, _ = multi_output_data
        Y_scaled = Y2.at[:, 1].multiply(1e4)
        r_std = analyze(ishigami.PROBLEM, X, Y_scaled, mode="multivariate", n_partitions=10)
        r_raw = analyze(
            ishigami.PROBLEM,
            X,
            Y_scaled,
            mode="multivariate",
            n_partitions=10,
            standardize_outputs=False,
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
        result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=20, key=jax.random.key(0))
        for conf in (result.ot_conf, result.advective_conf, result.diffusive_conf):
            assert conf is not None
            assert conf.shape == (2, 3)
            assert np.all(np.asarray(conf[0]) <= np.asarray(conf[1]))

    def test_conf_shapes_multi_output(self, multi_output_data):
        X, Y2, _ = multi_output_data
        result = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=20, key=jax.random.key(0))
        assert result.ot_conf is not None
        assert result.ot_conf.shape == (2, 2, 3)

    def test_point_estimate_independent_of_bootstrap(self, ishigami_data):
        """The two calls compile the scanned kernel for a different replicate
        count (1 vs. 16), and XLA is free to fuse a reduction differently per
        compiled shape even though replicate 0 is the identical computation
        both times, so this compares at a tight float32 tolerance rather than
        bit-for-bit -- the same one-ULP hazard as ``indices`` vs. ``analyze``
        below, not a dependence on the bootstrap draws themselves.
        """
        X, Y = ishigami_data
        r0 = analyze(ishigami.PROBLEM, X, Y)
        rb = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=15, key=jax.random.key(0))
        np.testing.assert_allclose(np.asarray(r0.ot), np.asarray(rb.ot), rtol=1e-6, atol=1e-7)

    def test_bootstrap_without_a_key_raises(self, ishigami_data):
        """Tier T4. A bootstrap needs a key, and an int seed is not one."""
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="key is required"):
            analyze(ishigami.PROBLEM, X, Y, n_bootstrap=4)

    def test_dummy_without_a_key_raises(self, ishigami_data):
        """Tier T4. The dummy column is the method's other random draw.

        It is independent of the bootstrap, so it needs a key of its own
        even when no interval was asked for.
        """
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="key is required"):
            analyze(ishigami.PROBLEM, X, Y, dummy=True)

    def test_gaussian_ci_is_centred_on_the_estimate(self, ishigami_data):
        """Tier T0 (closed form): the Gaussian interval is symmetric.

        ``estimate +/- z * sd`` puts the point estimate exactly at the
        midpoint of the two endpoints, which the percentile interval does
        not do. That is what tells the two branches apart.
        """
        X, Y = ishigami_data
        result = analyze(
            ishigami.PROBLEM, X, Y, n_bootstrap=10, ci_method="gaussian", key=jax.random.key(0)
        )
        assert result.ci is not None
        assert result.ci.method == "gaussian"
        assert result.ot_conf is not None
        midpoint = 0.5 * (np.asarray(result.ot_conf[0]) + np.asarray(result.ot_conf[1]))
        np.testing.assert_allclose(midpoint, np.asarray(result.ot), atol=1e-6)

    def test_unknown_ci_method_rejected(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="ci_method"):
            analyze(
                ishigami.PROBLEM,
                X,
                Y,
                n_bootstrap=4,
                ci_method=cast(Any, "bca"),
                key=jax.random.key(0),
            )

    def test_different_keys_differ(self, ishigami_data):
        X, Y = ishigami_data
        r1 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=15, key=jax.random.key(1))
        r2 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=15, key=jax.random.key(2))
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
            key=jax.random.key(0),
        )
        assert result.ot_conf is not None
        assert result.ot_conf.shape == (2, 3)


class TestOTMultiOutput:
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
        result = analyze(ishigami.PROBLEM, X, Y, dummy=True, key=jax.random.key(0))
        assert result.ot_dummy is not None
        # All three ishigami parameters are continuous, so they share one
        # floor, broadcast to (D,) -- same shape as ot (M4).
        assert result.ot_dummy.shape == (3,)
        dummy = np.asarray(result.ot_dummy)
        np.testing.assert_allclose(dummy, dummy[0])
        # A synthetic independent input must sit far below real inputs.
        assert float(dummy[0]) < float(np.asarray(result.ot).min())
        assert float(dummy[0]) < 0.05

    def test_dummy_multivariate_mode(self, multi_output_data):
        X, Y2, _ = multi_output_data
        result = analyze(
            ishigami.PROBLEM,
            X[:1500],
            Y2[:1500],
            mode="multivariate",
            n_partitions=5,
            dummy=True,
            key=jax.random.key(0),
        )
        assert result.ot_dummy is not None
        assert result.ot_dummy.shape == (3,)


class TestOTValidation:
    def test_x_not_2d(self, ishigami_data):
        _, Y = ishigami_data
        with pytest.raises(ValueError, match="2-D"):
            analyze(ishigami.PROBLEM, jnp.ones(10), Y[:10])

    def test_wrong_param_count(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="parameters"):
            analyze(ishigami.PROBLEM, X[:, :2], Y)

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

    def test_negative_epsilon(self, ishigami_data):
        """Tier T4. A negative epsilon is still refused up front."""
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="epsilon"):
            analyze(ishigami.PROBLEM, X, Y, epsilon=-0.01)

    def test_nan_epsilon(self, ishigami_data):
        """Tier T4. epsilon=NaN is refused at both entry points.

        The check used to be ``not epsilon <= 0``, which NaN passes
        (``NaN <= 0`` is False). NaN then poisoned ``log_K``, the Sinkhorn
        loop exited at once (``NaN > tol`` is False), the convergence
        warning counted zero failures, and the user got NaN indices with
        zero diagnostics. The positive check ``epsilon > 0`` is also False
        for NaN, so it raises.
        """
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="epsilon"):
            analyze(ishigami.PROBLEM, X, Y, epsilon=float("nan"))
        with pytest.raises(ValueError, match="epsilon"):
            jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, Y, epsilon=float("nan"))

    def test_nan_tol(self, ishigami_data):
        """Tier T4. tol=NaN is refused at both entry points."""
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="tol"):
            analyze(ishigami.PROBLEM, X, Y, tol=float("nan"))
        with pytest.raises(ValueError, match="tol"):
            jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, Y, tol=float("nan"))

    def test_valid_epsilon_and_tol_pass(self, ishigami_data):
        """Tier T4. Valid epsilon and tol still pass validation."""
        X, Y = ishigami_data
        Y2 = jnp.stack([Y[:512], Y[:512] ** 2], axis=1)
        result = analyze(
            ishigami.PROBLEM, X[:512], Y2, mode="multivariate", epsilon=0.05, tol=1e-4
        )
        assert np.all(np.isfinite(np.asarray(result.ot)))

    def test_bad_tol(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="tol"):
            analyze(ishigami.PROBLEM, X, Y, tol=0.0)

    def test_bad_conf_level(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="conf_level"):
            analyze(ishigami.PROBLEM, X, Y, n_bootstrap=10, conf_level=1.5)

    def test_bad_slice_chunk_size(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="slice_chunk_size"):
            analyze(ishigami.PROBLEM, X, Y, slice_chunk_size=0)


class TestOTXarray:
    def test_time_coords_kwarg_sets_the_time_coordinate(self, multi_output_data):
        """The dims are snapshot-covered; the kwarg's coordinate values are not."""
        X, _, Y3 = multi_output_data
        ds = analyze(ishigami.PROBLEM, X, Y3).to_dataset(time_coords=[0.5, 1.0])
        assert list(ds.coords["time"].values) == [0.5, 1.0]


class TestQuantileRankSplit:
    def test_matches_float64_host_reference_exactly(self):
        """The in-kernel int32 long division must match float64 exactly.

        Includes boundary-heavy cases like (N=1000, c=800), where the true
        value i * c / N hits exact integers and a naive float32 product can
        floor to the wrong index.
        """
        from jaxgsa.optimal_transport._analyze import _quantile_rank_split

        rng = np.random.default_rng(0)
        for N in (7, 100, 1000, 1024, 8000, 100_000):
            sizes = sorted(
                {0, 1, 2, 800, N // 3, N // 2, N - 1, N}
                | {int(s) for s in rng.integers(0, N + 1, size=8)}
            )
            counts = np.array([s for s in sizes if s <= N], dtype=np.int64)
            i_grid = np.arange(N, dtype=np.float64)
            c = counts[:, None].astype(np.float64)
            q = np.floor(i_grid * c / N).astype(np.int64)
            r = (i_grid * c - q * N).astype(np.int64)
            j_left_ref = np.minimum(q, np.maximum(counts[:, None] - 1, 0))
            j_right_ref = np.minimum(q + 1, np.maximum(counts[:, None] - 1, 0))
            width_left = np.minimum(N - r, counts[:, None])
            frac_left_ref = width_left / np.maximum(counts[:, None], 1)

            j_left, j_right, frac_left = _quantile_rank_split(jnp.asarray(counts), N)
            np.testing.assert_array_equal(np.asarray(j_left), j_left_ref)
            np.testing.assert_array_equal(np.asarray(j_right), j_right_ref)
            np.testing.assert_allclose(np.asarray(frac_left), frac_left_ref, atol=1e-6)

    def test_recovers_the_exact_1d_wasserstein_when_not_divisible(self):
        """The split weights reproduce POT's exact emd2, class size not dividing N."""
        ot = pytest.importorskip("ot")
        from jaxgsa.optimal_transport._analyze import _quantile_rank_split

        N = 37
        c = 6  # 37 / 6 does not divide evenly
        rng = np.random.default_rng(4)
        y = np.sort(rng.normal(size=N))
        cond = np.sort(rng.normal(loc=1.0, size=c))

        j_left, j_right, frac_left = _quantile_rank_split(jnp.asarray([c]), N)
        j_left, j_right, frac_left = (np.asarray(a)[0] for a in (j_left, j_right, frac_left))
        q_left, q_right = cond[j_left], cond[j_right]
        got = (frac_left * (y - q_left) ** 2 + (1.0 - frac_left) * (y - q_right) ** 2).mean()

        ref = ot.wasserstein_1d(y, cond, p=2)
        assert got == pytest.approx(ref, rel=1e-6)


class TestOTCorrelatedInputs:
    """Certify the documented correlation-inclusive reading of the OT index.

    The estimator partitions on the rank classes of each column of X and
    compares conditional output distributions. It never reads
    ``problem.correlation``, so its value is a property of the (X, Y) sample
    alone — which is exactly why it stays valid when the inputs are
    dependent: the index then measures total (direct plus correlation-borne)
    influence.
    """

    RHO = 0.8

    @pytest.fixture(scope="class")
    def correlated_data(self):
        """Y = X1 only, with corr(X1, X2) = 0.8 declared and honored."""
        R = np.array([[1.0, self.RHO], [self.RHO, 1.0]])
        problem = jaxgsa.Problem.from_dict(
            {
                "x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
                "x2": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
            }
        ).with_correlation(R)
        X = monte_carlo(problem, 4096, seed=0)
        Y = jnp.asarray(X)[:, 0]  # x2 never enters the model
        return problem, X, Y

    def test_unused_but_correlated_input_gets_nonzero_index(self, correlated_data):
        """The correlation-inclusive reading: x2 matters through x1."""
        problem, X, Y = correlated_data
        result = analyze(problem, X, Y)
        ot = np.asarray(result.ot)
        # x1 determines Y outright; x2 sees it only through the coupling.
        assert ot[0] > 0.8
        assert ot[1] > 0.15  # clearly non-zero, far above estimator noise
        assert ot[1] < ot[0]

    def test_indices_ignore_the_declared_matrix_bitwise(self, correlated_data):
        """Stripping the correlation changes nothing: distribution-freeness in X.

        The estimator conditions on rank classes of the observed sample, so
        the declared matrix must not enter the computation at all. Bit
        equality proves it never does.
        """
        problem, X, Y = correlated_data
        with_matrix = analyze(problem, X, Y)
        without_matrix = analyze(problem.with_correlation(None), X, Y)
        np.testing.assert_array_equal(np.asarray(with_matrix.ot), np.asarray(without_matrix.ot))
        np.testing.assert_array_equal(
            np.asarray(with_matrix.advective), np.asarray(without_matrix.advective)
        )
        np.testing.assert_array_equal(
            np.asarray(with_matrix.diffusive), np.asarray(without_matrix.diffusive)
        )


def _invalid_sample(n: int = 200, seed: int = 0):
    """Build a clean two-parameter OT sample for the on_invalid tests."""
    problem = jaxgsa.Problem(("a", "b"), ((0.0, 1.0), (0.0, 1.0)))
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 2))
    Y = np.sin(3.0 * X[:, 0]) + 0.3 * X[:, 1]
    return problem, jnp.asarray(X), jnp.asarray(Y)


class TestOTInvalidPolicy:
    """T4 (behaviour): jaxgsa.optimal_transport.analyze honours on_invalid."""

    def test_raise_is_the_default_and_names_the_rows(self):
        """T4: a non-finite Y row refuses the analysis and says which row it is."""
        problem, X, Y = _invalid_sample()
        Y = Y.at[9].set(jnp.nan)
        with pytest.raises(ValueError) as exc:
            analyze(problem, X, Y)
        message = str(exc.value)
        assert "jaxgsa.optimal_transport.analyze" in message
        assert "1 of 200 rows" in message
        assert "[9]" in message

    def test_propagate_warns_and_the_indices_go_non_finite(self):
        """T4: 'propagate' keeps the row and the NaN reaches the indices.

        The index is a class-averaged squared Wasserstein distance normalized
        by the output variance, and both halves average over the whole
        sample, so one NaN output carries through. It must arrive as NaN and
        not as 0: the normalizer's ``V > 0`` guard turns a non-positive
        variance into an exact 0 for a constant output, and a NaN variance
        fails that same test. Reporting 0 there would read as "no influence"
        instead of "this did not compute".
        """
        problem, X, Y = _invalid_sample()
        Y = Y.at[9].set(jnp.nan)
        with pytest.warns(jaxgsa.JaxgsaWarning, match="reaches the indices"):
            result = analyze(problem, X, Y, on_invalid="propagate")
        assert result.invalid.policy == "propagate"
        assert result.invalid.unit_indices == (9,)
        assert not np.all(np.isfinite(np.asarray(result.ot)))

    def test_drop_removes_the_row_and_returns_finite_indices(self):
        """T4: 'drop' analyzes the remainder and the indices come back usable."""
        problem, X, Y = _invalid_sample()
        Y = Y.at[9].set(jnp.nan)
        with pytest.warns(jaxgsa.JaxgsaWarning, match="dropped 1 of 200 rows"):
            result = analyze(problem, X, Y, on_invalid="drop")
        assert result.invalid.n_kept == 199
        assert np.all(np.isfinite(np.asarray(result.ot)))

    def test_a_bad_x_is_caught_and_named(self):
        """T4: a non-finite input is caught too, and the report says it was X."""
        problem, X, Y = _invalid_sample()
        X = X.at[2, 1].set(jnp.inf)
        with pytest.raises(ValueError, match=r"in X\b") as exc:
            analyze(problem, X, Y)
        assert "[2]" in str(exc.value)
        with pytest.warns(jaxgsa.JaxgsaWarning):
            result = analyze(problem, X, Y, on_invalid="drop")
        assert result.invalid.sources == ("X",)

    def test_the_check_reads_the_real_sample_not_the_dummy_column(self):
        """T4: ``dummy=True`` adds a synthetic column, and it is not what is checked.

        The dummy is drawn inside ``analyze`` and is finite by construction.
        The report must still count only the user's own parameters, so its
        row positions stay the positions of the model runs.
        """
        problem, X, Y = _invalid_sample()
        X_bad = X.at[2, 1].set(jnp.nan)
        with pytest.warns(jaxgsa.JaxgsaWarning):
            result = analyze(
                problem, X_bad, Y, dummy=True, key=jax.random.key(0), on_invalid="drop"
            )
        assert result.invalid.sources == ("X",)
        assert result.invalid.unit_indices == (2,)
        assert result.ot_dummy is not None
        assert np.all(np.isfinite(np.asarray(result.ot_dummy)))

        # A clean sample with a dummy column reports nothing at all.
        clean = analyze(problem, X, Y, dummy=True, key=jax.random.key(0))
        assert clean.invalid.n_invalid == 0


class TestIndicesPureCore:
    """The transformable core ``optimal_transport.indices``.

    Two claims. The core returns exactly what ``analyze`` reports on clean
    outputs, in every mode, and it survives ``jit``, ``vmap`` and
    ``jit(jacrev(...))``, which ``analyze`` cannot.
    """

    def test_matches_analyze_univariate(self, ishigami_data):
        """Tier T4: ``indices`` returns the three index fields of ``analyze``.

        ``analyze`` delegates to the same kernel runner, so this guards the
        delegation wiring rather than the estimator maths.
        """
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y)
        ot, adv, diff = jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, Y)

        np.testing.assert_array_equal(np.asarray(ot), np.asarray(result.ot))
        np.testing.assert_array_equal(np.asarray(adv), np.asarray(result.advective))
        np.testing.assert_array_equal(np.asarray(diff), np.asarray(result.diffusive))

    def test_matches_analyze_in_every_mode(self, multi_output_data):
        """Tier T4: the ``mode`` branch is static, and each branch agrees.

        ``mode`` is a Python string, so the branch stays inside the core:
        one mode is traced per concrete value, exactly as ``analyze``
        compiles one.
        """
        X, Y2, Y3 = multi_output_data
        # The joint modes solve one Sinkhorn problem per class, so they run
        # on a slice of the sample rather than all of it.
        X_small, Y3_small = X[:1024], Y3[:1024]
        cases: list[tuple[Any, Any, Literal["univariate", "multivariate", "trajectory"]]] = [
            (X, Y2, "univariate"),
            (X, Y3, "univariate"),
            (X_small, Y3_small, "multivariate"),
            (X_small, Y3_small, "trajectory"),
        ]
        for inputs, outputs, mode in cases:
            result = analyze(ishigami.PROBLEM, inputs, outputs, mode=mode, n_partitions=8)
            ot, adv, diff = jaxgsa.optimal_transport.indices(
                ishigami.PROBLEM, inputs, outputs, mode=mode, n_partitions=8
            )
            assert ot.shape == result.ot.shape, mode
            np.testing.assert_array_equal(np.asarray(ot), np.asarray(result.ot))
            np.testing.assert_array_equal(np.asarray(adv), np.asarray(result.advective))
            np.testing.assert_array_equal(np.asarray(diff), np.asarray(result.diffusive))

    def test_standardize_outputs_reaches_the_joint_core(self, multi_output_data):
        """Tier T4: ``standardize_outputs`` is honoured identically by core and ``analyze``.

        The keyword is arithmetic over the sample axis, not policy, so the
        core carries it. Turning it off must change the joint indices and
        must change them the same way in both entry points.
        """
        X, _, Y3 = multi_output_data
        X, Y3 = X[:1024], Y3[:1024]
        kwargs: dict[str, Any] = {"mode": "multivariate", "n_partitions": 8}
        on = jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, Y3, **kwargs)[0]
        off = jaxgsa.optimal_transport.indices(
            ishigami.PROBLEM, X, Y3, standardize_outputs=False, **kwargs
        )[0]
        reference = analyze(ishigami.PROBLEM, X, Y3, standardize_outputs=False, **kwargs)

        assert not np.allclose(np.asarray(on), np.asarray(off))
        np.testing.assert_array_equal(np.asarray(off), np.asarray(reference.ot))

    def test_no_bootstrap_and_no_dummy_in_the_core(self, ishigami_data):
        """Tier T4: the core is the point estimate, and takes no key.

        The bootstrap interval and the ``dummy`` baseline both draw
        randomness and both describe the point estimate rather than being
        it, so both stay in ``analyze``.
        """
        X, Y = ishigami_data
        ot, _, _ = jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, Y)
        with_ci = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=4, key=jax.random.key(0))

        # The interval is centred on exactly this estimate. Compared at a
        # tight float32 tolerance, not bit-for-bit: the two calls compile
        # the scanned kernel for a different replicate count (1 vs. 5), and
        # XLA is free to fuse a reduction differently per compiled shape
        # even though replicate 0 is the identical computation both times.
        np.testing.assert_allclose(np.asarray(ot), np.asarray(with_ci.ot), rtol=1e-6, atol=1e-7)
        with pytest.raises(TypeError):
            cast(Any, jaxgsa.optimal_transport.indices)(
                ishigami.PROBLEM, X, Y, n_bootstrap=4, key=jax.random.key(0)
            )
        with pytest.raises(TypeError):
            cast(Any, jaxgsa.optimal_transport.indices)(ishigami.PROBLEM, X, Y, dummy=True)

    def test_is_jittable(self, ishigami_data):
        """Tier T4: ``jit`` traces ``indices`` over both X and Y."""
        X, Y = ishigami_data
        jitted = jax.jit(lambda x, y: jaxgsa.optimal_transport.indices(ishigami.PROBLEM, x, y))
        ot_jit = jitted(X, Y)[0]
        ot = jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, Y)[0]

        np.testing.assert_allclose(np.asarray(ot_jit), np.asarray(ot), rtol=1e-5)

    def test_is_vmappable(self):
        """Tier T4: ``vmap`` maps ``indices`` over a batch of output vectors."""
        N = 1024
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=N, seed=7))
        batch = jnp.stack([ishigami.evaluate(X), X[:, 0], X[:, 1] ** 2])

        mapped = jax.vmap(lambda y: jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, y))(
            batch
        )

        assert mapped[0].shape == (3, ishigami.PROBLEM.num_vars)
        for row, outputs in enumerate(batch):
            ot = jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, outputs)[0]
            np.testing.assert_allclose(np.asarray(mapped[0][row]), np.asarray(ot), rtol=1e-4)

    def test_jit_of_jacrev(self):
        """Tier T4: ``jit(jacrev(...))`` compiles a gradient through the core.

        The 1-D coupling is a sort and a gather, so the index is piecewise
        smooth in Y and its derivative is defined away from ties. The rank
        partition is not differentiable in X, which is why the derivative is
        taken in Y.
        """
        with jax.enable_x64():
            N = 512
            X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=N, seed=3))
            Y = ishigami.evaluate(X)

            def total_ot(outputs):
                return jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, outputs)[0].sum()

            jac = jax.jit(jax.jacrev(total_ot))(Y)
            eager = jax.jacrev(total_ot)(Y)

        assert jac.shape == (N,)
        assert np.all(np.isfinite(np.asarray(jac)))
        np.testing.assert_allclose(np.asarray(jac), np.asarray(eager), rtol=1e-8)

    def test_categorical_is_refused_with_a_pointer_to_analyze(self):
        """Tier T4: a categorical parameter cannot be traced, so the core says so."""
        problem = jaxgsa.Problem.from_dict(
            {"x": (0.0, 1.0), "c": {"dist": "categorical", "probs": [0.5, 0.5]}}
        )
        rng = np.random.default_rng(0)
        n = 400
        codes = rng.integers(0, 2, size=n).astype(np.float64)
        x = rng.uniform(0.0, 1.0, size=n)
        X = jnp.asarray(np.stack([x, codes], axis=1))
        Y = jnp.asarray(codes + 0.3 * x)

        with pytest.raises(ValueError, match="continuous parameters only"):
            jaxgsa.optimal_transport.indices(problem, X, Y)

    def test_no_policy_fires_in_the_core(self):
        """Tier T4: a constant output warns in ``analyze`` and is silent here."""
        problem = jaxgsa.Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0)})
        X = jnp.asarray(monte_carlo(problem, n=400, seed=5))
        constant = jnp.full((400,), 2.5)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ot, adv, diff = jaxgsa.optimal_transport.indices(problem, X, constant)

        for arr in (ot, adv, diff):
            np.testing.assert_array_equal(np.asarray(arr), np.zeros(2))
        with pytest.warns(jaxgsa.JaxgsaWarning):
            analyze(problem, X, constant)


class TestS1AndAboveDummy:
    """The derived convenience fields: ``S1`` and ``above_dummy``."""

    def test_s1_matches_the_analytic_first_order_index(self, gaussian_linear_data):
        """Tier T0 (closed form): S1 estimates the true Sobol first order."""
        X, Y = gaussian_linear_data
        result = analyze(gaussian_linear.PROBLEM, X, Y)
        np.testing.assert_allclose(np.asarray(result.S1), gaussian_linear.ANALYTICAL_S1, atol=0.02)

    def test_s1_is_advective_on_the_ddof0_convention(self, ishigami_data):
        """Tier T4: S1 == 2 * advective * N / (N - 1), exactly.

        The advective numerator is the population (ddof=0) between-class
        variance while the OT normalizer uses ddof=1, so the rescale is the
        one constant that puts both on borgonovo's ddof=0 S1 convention.
        """
        X, Y = ishigami_data
        N = X.shape[0]
        result = analyze(ishigami.PROBLEM, X, Y)
        np.testing.assert_allclose(
            np.asarray(result.S1),
            2.0 * (N / (N - 1)) * np.asarray(result.advective),
            rtol=1e-6,
        )

    def test_s1_agrees_with_borgonovo_s1(self, ishigami_data):
        """Tier T2 (cross-estimator): both given-data S1 estimates coincide."""
        X, Y = ishigami_data
        ot_result = analyze(ishigami.PROBLEM, X, Y)
        delta_result = jaxgsa.borgonovo.analyze(ishigami.PROBLEM, X, Y)
        np.testing.assert_allclose(
            np.asarray(ot_result.S1), np.asarray(delta_result.S1), atol=0.02
        )

    def test_s1_conf_is_the_exactly_rescaled_advective_conf(self, ishigami_data):
        X, Y = ishigami_data
        N = X.shape[0]
        result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=8, key=jax.random.key(0))
        np.testing.assert_allclose(
            np.asarray(result.S1_conf),
            2.0 * (N / (N - 1)) * np.asarray(result.advective_conf),
            rtol=1e-6,
        )

    def test_s1_conf_is_none_without_a_bootstrap(self, ishigami_data):
        X, Y = ishigami_data
        assert analyze(ishigami.PROBLEM, X, Y).S1_conf is None

    def test_above_dummy_requires_the_dummy_baseline(self, ishigami_data):
        X, Y = ishigami_data
        assert analyze(ishigami.PROBLEM, X, Y).above_dummy is None

    def test_above_dummy_is_the_clipped_excess_over_the_floor(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, dummy=True, key=jax.random.key(1))
        above = np.asarray(result.above_dummy)
        assert above.shape == np.asarray(result.ot).shape
        assert (above >= 0.0).all()
        np.testing.assert_allclose(
            above,
            np.maximum(np.asarray(result.ot) - np.asarray(result.ot_dummy), 0.0),
        )
        # The influential Ishigami parameters clear the floor.
        assert above[0] > 0.1
        assert above[1] > 0.1

    def test_above_dummy_matches_ots_shape_in_trajectory_mode(self, multi_output_data):
        X, Y2, Y3 = multi_output_data
        result = analyze(
            ishigami.PROBLEM,
            X,
            Y3,
            mode="trajectory",
            dummy=True,
            key=jax.random.key(2),
            n_partitions=10,
        )
        above = np.asarray(result.above_dummy)
        assert above.shape == np.asarray(result.ot).shape  # (K, D)
        assert np.asarray(result.ot_dummy).shape == np.asarray(result.ot).shape  # (K, D)
        np.testing.assert_allclose(
            above,
            np.maximum(np.asarray(result.ot) - np.asarray(result.ot_dummy), 0.0),
        )

    def test_fields_reach_the_dataset(self, ishigami_data):
        X, Y = ishigami_data
        ds = analyze(ishigami.PROBLEM, X, Y, dummy=True, key=jax.random.key(3)).to_dataset()
        assert "S1" in ds
        assert "above_dummy" in ds
        assert ds["S1"].dims == ("param",)
        assert ds["above_dummy"].dims == ("param",)
