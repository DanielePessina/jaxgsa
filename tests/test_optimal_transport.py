"""Tests for optimal-transport sensitivity analysis."""

from __future__ import annotations

import warnings
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _fixtures_givendata import BOOT, invalid_sample

import jaxgsa
from jaxgsa._core.partition import _build_class_indices, _class_layout
from jaxgsa.benchmarks import gaussian_linear, ishigami
from jaxgsa.optimal_transport import analyze
from jaxgsa.sampling import monte_carlo


@pytest.fixture(scope="module")
def gaussian_linear_32k():
    """Gaussian linear sample at ``N = 32000``, for the slow tight closed-form check."""
    X = jnp.asarray(monte_carlo(gaussian_linear.PROBLEM, n_samples=32000, seed=1))
    return X, gaussian_linear.evaluate(X)


class TestOTBasic:
    def test_plain_result_properties(self, ot_ishigami):
        """A scalar output picks the univariate mode, and the result is well formed.

        The index and its two parts lie in [0, 1], ``ot = advective + diffusive``,
        and x1 and x2 outrank the weak x3.
        """
        result = ot_ishigami
        assert result.mode == "univariate"
        for arr in (result.ot, result.advective, result.diffusive):
            a = np.asarray(arr)
            assert np.all(a >= 0.0) and np.all(a <= 1.0)
        np.testing.assert_allclose(
            np.asarray(result.ot),
            np.asarray(result.advective + result.diffusive),
            atol=1e-6,
        )
        ot = np.asarray(result.ot)
        assert ot[0] > ot[2]
        assert ot[1] > ot[2]

    def test_the_key_decides_the_bootstrap(self, ishigami_xy_4k, ot_ishigami_boot):
        """The same key gives the same result, a different key a different interval."""
        X, Y = ishigami_xy_4k
        same = analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_bootstrap=BOOT,
            dummy=True,
            key=jax.random.key(0),
            verbose=False,
        )
        other = analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_bootstrap=BOOT,
            dummy=True,
            key=jax.random.key(1),
            verbose=False,
        )
        assert ot_ishigami_boot.ot_conf is not None
        assert same.ot_conf is not None and other.ot_conf is not None
        np.testing.assert_array_equal(np.asarray(same.ot), np.asarray(ot_ishigami_boot.ot))
        np.testing.assert_array_equal(
            np.asarray(same.ot_conf), np.asarray(ot_ishigami_boot.ot_conf)
        )
        assert not np.array_equal(np.asarray(other.ot_conf), np.asarray(same.ot_conf))

    def test_unstable_class_sort_matches_a_stable_one(self, given_ishigami_xy):
        """The kernel's unstable sort must return the stable sort's array.

        Ordering the class members is the most expensive step in this
        method, and the kernel asks an unstable sort for it because it
        wants the order statistics and never which tied element came
        first. Two float32 values that compare equal are the same bits,
        so the two sorts have to agree element for element. This pins
        that at the kernel's layout, NaNs and signed zeros included,
        because an unstable sort that moved a NaN or a signed zero
        somewhere else would be a silent change of answer. A small
        sample is enough: the sort does not depend on N.
        """
        X, Y = given_ishigami_xy
        X, Y = X[:256], Y[:256]
        n = X.shape[0]
        take, _ = _class_layout(n, 8)
        idx = jnp.arange(n, dtype=jnp.int32)[None, :]
        cls_idx = _build_class_indices(X, idx, jnp.asarray(take))[0]  # (D, M, P)
        for values in (Y, Y.at[:5].set(jnp.nan), Y.at[:5].set(-0.0).at[5:10].set(0.0)):
            y_cls = values[cls_idx]
            stable = jnp.sort(y_cls, axis=-1)
            unstable = jax.lax.sort(y_cls, dimension=-1, is_stable=False)
            np.testing.assert_array_equal(np.asarray(stable), np.asarray(unstable), strict=True)

    def test_slice_chunk_size_invariance(self, ishigami_xy_series):
        X, _, Y3 = ishigami_xy_series
        r_default = analyze(ishigami.PROBLEM, X, Y3)
        r_chunked = analyze(ishigami.PROBLEM, X, Y3, slice_chunk_size=1)
        np.testing.assert_allclose(np.asarray(r_default.ot), np.asarray(r_chunked.ot), atol=1e-7)

    def test_memory_budget_narrows_the_working_set(self, ishigami_xy_series):
        """A small budget still returns the same answer.

        The default chunk width comes from the active memory budget and the
        working set in bytes, so a budget of 1 MiB has to cut the output
        into narrower chunks than the default 512 MiB does. The budget
        changes how the work is cut up, never what it computes. The
        tolerance is there because a narrower vmap lets XLA reassociate the
        float32 reductions, the same reason the chunk-size test beside this
        one carries one.
        """
        X, _, Y3 = ishigami_xy_series
        r_default = analyze(ishigami.PROBLEM, X, Y3)
        jaxgsa.config.set_memory_budget(1, unit="mib")
        try:
            r_small = analyze(ishigami.PROBLEM, X, Y3)
        finally:
            jaxgsa.config.set_memory_budget(512, unit="mib")
        np.testing.assert_allclose(np.asarray(r_default.ot), np.asarray(r_small.ot), atol=1e-7)

    def test_n_classes_changes_result(self, ishigami_xy_4k):
        X, Y = ishigami_xy_4k
        r25 = analyze(ishigami.PROBLEM, X, Y, n_classes=25)
        r10 = analyze(ishigami.PROBLEM, X, Y, n_classes=10)
        assert not np.allclose(np.asarray(r25.ot), np.asarray(r10.ot))

    def test_monotone_transform_invariance(self, ishigami_xy_4k, ot_ishigami):
        """Rank-based conditioning: monotone input transforms change nothing."""
        X, Y = ishigami_xy_4k
        X_warp = X.at[:, 0].set(jnp.exp(X[:, 0]))
        r_warp = analyze(ishigami.PROBLEM, X_warp, Y)
        np.testing.assert_array_equal(np.asarray(ot_ishigami.ot), np.asarray(r_warp.ot))


class TestOTSolverProperties:
    """Internal properties of the transport solver and partition plumbing."""

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

    def test_sinkhorn_reverse_gradient_matches_central_difference(self):
        """Tier T4: the implicit reverse gradient matches a central difference.

        The target has unequal weights and padded columns, so the transpose
        solve meets its real null direction ``b``, not the constant vector.
        The gradients in ``C``, ``scale`` and ``epsilon`` must match a
        central difference, and a padded column must get an exact zero.
        """
        from jaxgsa.optimal_transport._solver import _sinkhorn_w2

        with jax.enable_x64():
            rng = np.random.default_rng(0)
            N, P = 30, 12
            source = rng.normal(size=(N, 2))
            target = rng.normal(size=(P, 2))
            valid = np.zeros(P, dtype=bool)
            valid[[0, 2, 3, 5, 6, 8, 9, 11]] = True
            weights = rng.uniform(0.2, 2.0, valid.sum())
            log_b = np.full(P, -np.inf)
            log_b[valid] = np.log(weights / weights.sum())
            cost = ((source[:, None] - target[None]) ** 2).sum(-1) * valid[None, :]
            C, log_b = jnp.asarray(cost), jnp.asarray(log_b)

            def f(C, scale, epsilon):
                return _sinkhorn_w2(
                    C, log_b, epsilon, jnp.asarray(200_000), jnp.asarray(1e-13), scale
                )[0]

            scale, epsilon = jnp.asarray(2.0), jnp.asarray(0.03)
            grad_C, grad_scale, grad_eps = jax.grad(f, argnums=(0, 1, 2))(C, scale, epsilon)
            direction = jnp.asarray(rng.normal(size=C.shape) * valid[None, :])
            h = 1e-6
            fd_C = (
                f(C + h * direction, scale, epsilon) - f(C - h * direction, scale, epsilon)
            ) / (2 * h)
            fd_scale = (f(C, scale + h, epsilon) - f(C, scale - h, epsilon)) / (2 * h)
            h_eps = h * 0.03
            fd_eps = (f(C, scale, epsilon + h_eps) - f(C, scale, epsilon - h_eps)) / (2 * h_eps)
            reverse_C = jnp.vdot(grad_C, direction)

        np.testing.assert_allclose(float(reverse_C), float(fd_C), rtol=1e-6)
        np.testing.assert_allclose(float(grad_scale), float(fd_scale), rtol=1e-6)
        np.testing.assert_allclose(float(grad_eps), float(fd_eps), rtol=1e-6)
        assert np.all(np.asarray(grad_C)[:, ~valid] == 0.0)

    def test_empty_class_gives_a_finite_gradient(self):
        """An empty class must not leak NaN into a reverse-mode gradient.

        An empty class has no valid target column. The kernel discards its
        solve with ``jnp.where``, but reverse mode still differentiates the
        discarded branch. So the kernel gives that class a dummy column, and
        the gradient stays finite.
        """
        from jaxgsa.optimal_transport._analyze import _joint_kernel

        with jax.enable_x64():
            N = 40
            Z = jnp.asarray(np.random.default_rng(1).normal(size=(N, 2)))
            # One parameter, two classes: all N rows, then an empty class.
            cls_idx = jnp.stack([jnp.arange(N), jnp.zeros(N, dtype=int)])
            cls_idx = cls_idx[None, None].astype(jnp.int32)  # (R, D, M, P)
            counts = jnp.asarray([[[float(N), 0.0]]])  # (R, D, M)

            def total(Z):
                return _joint_kernel(
                    Z,
                    jnp.arange(N)[None],
                    ((cls_idx, counts),),
                    (jnp.arange(2, dtype=jnp.int32),),
                    jnp.asarray(0.1),
                    jnp.asarray(2000),
                    jnp.asarray(1e-9),
                )[0].sum()

            value, grad = jax.value_and_grad(total)(Z)

        assert np.isfinite(float(value))
        assert np.all(np.isfinite(np.asarray(grad)))


def _gaussian_additive(n: int):
    """``Y = X1 + X2`` with ``N(0, 1)`` and ``N(0, 4)`` inputs, and its closed form."""
    s1, s2 = 1.0, 2.0
    v = s1**2 + s2**2
    key1, key2 = jax.random.split(jax.random.PRNGKey(11))
    X = jnp.stack(
        [s1 * jax.random.normal(key1, (n,)), s2 * jax.random.normal(key2, (n,))],
        axis=1,
    )
    problem = jaxgsa.Problem(("x1", "x2"), ((-6.0, 6.0), (-12.0, 12.0)))
    expected = np.array(
        [
            (s1**2 + (np.sqrt(v) - s2) ** 2) / (2 * v),
            (s2**2 + (np.sqrt(v) - s1) ** 2) / (2 * v),
        ]
    )
    # advective = c_i^2 s_i^2 / (2v), diffusive = (sqrt(v) - s_other)^2 / (2v)
    exp_adv = np.array([s1**2, s2**2]) / (2 * v)
    exp_diff = np.array([(np.sqrt(v) - s2) ** 2, (np.sqrt(v) - s1) ** 2]) / (2 * v)
    return problem, X, X[:, 0] + X[:, 1], expected, exp_adv, exp_diff


class TestOTAnalytic:
    @pytest.mark.parametrize(
        ("n", "tol"),
        [
            pytest.param(4096, 0.05, id="cheap"),
            pytest.param(32000, 0.015, marks=pytest.mark.slow, id="tight"),
        ],
    )
    def test_gaussian_additive_closed_form(self, n, tol):
        """Y = X1 + X2 with N(0, s_i^2): exact continuum index per input.

        At ``N = 4096`` the Monte Carlo sd is about 0.005 and the partition
        bias 0.01 to 0.02, so 0.05 leaves margin. The tight case uses the
        original 32000 rows.
        """
        problem, X, Y, expected, exp_adv, exp_diff = _gaussian_additive(n)
        result = analyze(problem, X, Y, n_classes=50)
        np.testing.assert_allclose(np.asarray(result.ot), expected, atol=tol)
        np.testing.assert_allclose(np.asarray(result.advective), exp_adv, atol=tol)
        np.testing.assert_allclose(np.asarray(result.diffusive), exp_diff, atol=tol)

    def test_gaussian_linear_closed_form(self, gaussian_linear_xy):
        """Convergence to the closed-form constants, and the tie to Sobol S1.

        ``ot`` meets ``ANALYTICAL_OT``. ``2 * advective`` and ``S1`` both meet
        the analytic first-order Sobol index.
        """
        X, Y = gaussian_linear_xy
        result = analyze(gaussian_linear.PROBLEM, X, Y, n_classes=50)
        np.testing.assert_allclose(np.asarray(result.ot), gaussian_linear.ANALYTICAL_OT, atol=0.05)
        np.testing.assert_allclose(
            2.0 * np.asarray(result.advective), gaussian_linear.ANALYTICAL_S1, atol=0.05
        )
        np.testing.assert_allclose(np.asarray(result.S1), gaussian_linear.ANALYTICAL_S1, atol=0.05)

    @pytest.mark.slow
    def test_gaussian_linear_closed_form_tight(self, gaussian_linear_32k):
        """The same closed-form checks at ``N = 32000`` and a tight tolerance."""
        X, Y = gaussian_linear_32k
        result = analyze(gaussian_linear.PROBLEM, X, Y, n_classes=50)
        np.testing.assert_allclose(np.asarray(result.ot), gaussian_linear.ANALYTICAL_OT, atol=0.02)
        np.testing.assert_allclose(
            2.0 * np.asarray(result.advective), gaussian_linear.ANALYTICAL_S1, atol=0.02
        )
        np.testing.assert_allclose(np.asarray(result.S1), gaussian_linear.ANALYTICAL_S1, atol=0.02)

    def test_advective_is_half_s1_ishigami(self, ot_ishigami):
        np.testing.assert_allclose(
            2.0 * np.asarray(ot_ishigami.advective), ishigami.ANALYTICAL_S1, atol=0.05
        )

    @pytest.mark.slow
    def test_mixed_uniform_gaussian_marginals(self):
        """Rank-based conditioning handles mixed input marginals unchanged."""
        problem = jaxgsa.Problem.from_dict(
            {
                "xu": (-1.0, 1.0),
                "xg": jaxgsa.GaussianInputSpec(dist="gaussian", mean=0.0, variance=4.0),
            }
        )
        X = jnp.asarray(monte_carlo(problem, n_samples=16000, seed=5))
        Y = X[:, 0] + X[:, 1]
        result = analyze(problem, X, Y, n_classes=40)
        ot = np.asarray(result.ot)
        assert np.all(ot > 0.0) and np.all(ot <= 1.0)
        # xg carries variance 4 vs the uniform's 1/3 -> must dominate.
        assert ot[1] > ot[0]
        # Advective tie to the analytic S1 holds for any marginals.
        var_u, var_g = 1.0 / 3.0, 4.0
        s1 = np.array([var_u, var_g]) / (var_u + var_g)
        np.testing.assert_allclose(2.0 * np.asarray(result.advective), s1, atol=0.03)

    @pytest.mark.slow
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
        result = analyze(problem, jnp.asarray(X), jnp.asarray(Y), n_classes=50)
        np.testing.assert_allclose(np.asarray(result.ot), expected, atol=0.02)


class TestOTPointCloud:
    @pytest.mark.slow
    def test_multivariate_decomposition_sums_to_total(self, ishigami_xy_series):
        X, Y2, _ = ishigami_xy_series
        result = analyze(ishigami.PROBLEM, X, Y2, mode="multivariate", n_classes=10)
        np.testing.assert_allclose(
            np.asarray(result.ot),
            np.asarray(result.advective + result.diffusive),
            atol=1e-5,
        )

    @pytest.mark.parametrize(
        ("n", "n_classes", "epsilon", "atol"),
        [
            pytest.param(300, 4, 0.05, 0.05, id="cheap"),
            pytest.param(2000, 10, 0.005, 0.12, marks=pytest.mark.slow, id="tight"),
        ],
    )
    def test_multivariate_scalar_close_to_univariate(
        self, ishigami_xy_4k, n, n_classes, epsilon, atol
    ):
        """On scalar Y, point-cloud transport approaches the exact 1-D index.

        The multivariate mode solves ``N x (N / M)`` transport problems, so
        the test subsamples. The entropic and finite-sample bias is strictly
        upward.
        """
        X, Y = ishigami_xy_4k
        Xs, Ys = X[:n], Y[:n]
        sep = analyze(ishigami.PROBLEM, Xs, Ys, n_classes=n_classes)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*entropic transport", category=jaxgsa.JaxgsaWarning
            )
            joint = analyze(
                ishigami.PROBLEM,
                Xs,
                Ys,
                mode="multivariate",
                n_classes=n_classes,
                epsilon=epsilon,
            )
        sep_ot = np.asarray(sep.ot)
        joint_ot = np.asarray(joint.ot)
        assert np.all(joint_ot > sep_ot - 1e-4)
        np.testing.assert_allclose(joint_ot, sep_ot, atol=atol)

    @pytest.mark.slow
    def test_multivariate_matches_pot_sinkhorn(self):
        """Tier T1 (reference implementation): the point-cloud index against POT.

        The oracle rebuilds the multivariate index from POT's log-domain
        Sinkhorn solver on a 64-point, two-output cloud. Class ``m`` of a
        parameter holds the rows with ordinal rank in ``[16 m, 16 m + 16)``.
        The cost is the squared Euclidean distance. The entropic plan comes
        from ``ot.sinkhorn`` with the regularization ``epsilon`` applied to
        the cost divided by ``V = 2 * tr(Cov)``, the normalizer the package
        uses. The reported cost is ``<P, C>`` of that plan, and the index is
        the class-weighted mean of the costs over ``V``. The advective part
        is the class-weighted squared shift of the class means over ``V``.
        """
        ot_lib = pytest.importorskip("ot")
        epsilon = 0.1
        n, n_classes, per_class = 64, 4, 16
        with jax.enable_x64():
            X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=n, seed=3), dtype=jnp.float64)
            base = ishigami.evaluate(X)
            Y = jnp.stack([base, X[:, 0] * X[:, 1]], axis=1)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*entropic transport", category=jaxgsa.JaxgsaWarning
                )
                result = analyze(
                    ishigami.PROBLEM,
                    X,
                    Y,
                    mode="multivariate",
                    n_classes=n_classes,
                    epsilon=epsilon,
                    tol=1e-10,
                    max_iter=20000,
                    standardize_outputs=False,
                    verbose=False,
                )

        X_np = np.asarray(X)
        Z = np.asarray(Y, dtype=np.float64)
        Z = Z - Z.mean(axis=0)
        V = 2.0 * Z.var(axis=0, ddof=1).sum()
        a = np.full(n, 1.0 / n)
        b = np.full(per_class, 1.0 / per_class)
        expected_ot = np.zeros(X_np.shape[1])
        expected_adv = np.zeros(X_np.shape[1])
        for d in range(X_np.shape[1]):
            order = np.argsort(X_np[:, d], kind="stable")
            for m in range(n_classes):
                members = Z[order[m * per_class : (m + 1) * per_class]]
                C = ((Z[:, None, :] - members[None, :, :]) ** 2).sum(-1)
                plan = ot_lib.sinkhorn(
                    a,
                    b,
                    C / V,
                    reg=epsilon,
                    method="sinkhorn_log",
                    numItermax=20000,
                    stopThr=1e-12,
                )
                expected_ot[d] += (plan * C).sum() / n_classes / V
                shift = members.mean(axis=0) - Z.mean(axis=0)
                expected_adv[d] += (shift**2).sum() / n_classes / V
        np.testing.assert_allclose(np.asarray(result.ot), expected_ot, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(result.advective), expected_adv, rtol=1e-6)

    @pytest.mark.slow
    def test_standardize_outputs_matters_for_mismatched_scales(self, ishigami_xy_series):
        X, Y2, _ = ishigami_xy_series
        Y_scaled = Y2.at[:, 1].multiply(1e4)
        r_std = analyze(ishigami.PROBLEM, X, Y_scaled, mode="multivariate", n_classes=10)
        r_raw = analyze(
            ishigami.PROBLEM,
            X,
            Y_scaled,
            mode="multivariate",
            n_classes=10,
            standardize_outputs=False,
        )
        assert not np.allclose(np.asarray(r_std.ot), np.asarray(r_raw.ot), atol=1e-3)

    def test_unconverged_sinkhorn_warns(self, ishigami_xy_series):
        X, Y2, _ = ishigami_xy_series
        with pytest.warns(UserWarning, match="Sinkhorn"):
            analyze(
                ishigami.PROBLEM,
                X[:256],
                Y2[:256],
                mode="multivariate",
                n_classes=4,
                max_iter=1,
            )


class TestOTBootstrap:
    def test_conf_shapes_scalar(self, ot_ishigami_boot):
        result = ot_ishigami_boot
        for conf in (result.ot_conf, result.advective_conf, result.diffusive_conf):
            assert conf is not None
            assert conf.shape == (2, 3)
            assert np.all(np.asarray(conf[0]) <= np.asarray(conf[1]))

    def test_conf_shapes_multi_output(self, ishigami_xy_series):
        X, Y2, _ = ishigami_xy_series
        result = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=BOOT, key=jax.random.key(0))
        assert result.ot_conf is not None
        assert result.ot_conf.shape == (2, 2, 3)

    def test_point_estimate_independent_of_bootstrap(self, ot_ishigami, ot_ishigami_boot):
        """The point estimate does not move when a bootstrap is added.

        The two calls compile the scanned kernel for a different replicate
        count (1 vs. 9), and XLA is free to fuse a reduction differently per
        compiled shape even though replicate 0 is the identical computation
        both times, so this compares at a tight float32 tolerance rather than
        bit-for-bit -- the same one-ULP hazard as ``indices`` vs. ``analyze``
        below, not a dependence on the bootstrap draws themselves.
        """
        np.testing.assert_allclose(
            np.asarray(ot_ishigami.ot), np.asarray(ot_ishigami_boot.ot), rtol=1e-6, atol=1e-7
        )

    def test_the_two_random_draws_each_need_a_key(self, ishigami_xy_4k):
        """Tier T4. A bootstrap needs a key, and so does the dummy column.

        An int seed is not a key. The dummy column is the method's other
        random draw. It is independent of the bootstrap, so it needs a key of
        its own even when no interval was asked for.
        """
        X, Y = ishigami_xy_4k
        with pytest.raises(ValueError, match="key is required"):
            analyze(ishigami.PROBLEM, X, Y, n_bootstrap=BOOT)
        with pytest.raises(ValueError, match="key is required"):
            analyze(ishigami.PROBLEM, X, Y, dummy=True)

    def test_gaussian_ci_is_centred_on_the_estimate(self, ishigami_xy_4k):
        """Tier T0 (closed form): the Gaussian interval is symmetric.

        ``estimate +/- z * sd`` puts the point estimate exactly at the
        midpoint of the two endpoints, which the percentile interval does
        not do. That is what tells the two branches apart.
        """
        X, Y = ishigami_xy_4k
        result = analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_bootstrap=BOOT,
            ci_method="gaussian",
            key=jax.random.key(0),
        )
        assert result.ci is not None
        assert result.ci.method == "gaussian"
        assert result.ot_conf is not None
        midpoint = 0.5 * (np.asarray(result.ot_conf[0]) + np.asarray(result.ot_conf[1]))
        np.testing.assert_allclose(midpoint, np.asarray(result.ot), atol=1e-6)

    def test_unknown_ci_method_rejected(self, ishigami_xy_4k):
        X, Y = ishigami_xy_4k
        with pytest.raises(ValueError, match="ci_method"):
            analyze(
                ishigami.PROBLEM,
                X,
                Y,
                n_bootstrap=BOOT,
                ci_method=cast(Any, "bca"),
                key=jax.random.key(0),
            )

    @pytest.mark.slow
    def test_multivariate_bootstrap(self, ishigami_xy_series):
        X, Y2, _ = ishigami_xy_series
        result = analyze(
            ishigami.PROBLEM,
            X[:1500],
            Y2[:1500],
            mode="multivariate",
            n_classes=5,
            n_bootstrap=5,
            key=jax.random.key(0),
        )
        assert result.ot_conf is not None
        assert result.ot_conf.shape == (2, 3)


class TestOTMultiOutput:
    def test_first_column_matches_scalar_run(self, ishigami_xy_series, given_ishigami_xy):
        X, Y2, _ = ishigami_xy_series
        _, Y = given_ishigami_xy
        r_multi = analyze(ishigami.PROBLEM, X, Y2)
        r_scalar = analyze(ishigami.PROBLEM, X, Y)
        np.testing.assert_allclose(np.asarray(r_multi.ot[0]), np.asarray(r_scalar.ot), atol=1e-7)


class TestOTEdgeCases:
    def test_constant_output_gives_zero(self, ishigami_xy_4k):
        X, _ = ishigami_xy_4k
        Y_const = jnp.ones(X.shape[0])
        result = analyze(ishigami.PROBLEM, X, Y_const)
        np.testing.assert_array_equal(np.asarray(result.ot), np.zeros(3))
        assert not np.any(np.isnan(np.asarray(result.ot)))

    def test_constant_column_among_varying(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        Y2 = jnp.stack([Y, jnp.ones_like(Y)], axis=1)
        result = analyze(ishigami.PROBLEM, X, Y2)
        assert not np.any(np.isnan(np.asarray(result.ot)))
        np.testing.assert_array_equal(np.asarray(result.ot[1]), np.zeros(3))
        assert np.all(np.asarray(result.ot[0]) > 0)

    def test_ragged_close_to_divisible(self, ishigami_xy_4k):
        """A ragged N (prime) stays in [0, 1] and tracks the divisible N."""
        X, Y = ishigami_xy_4k
        r_div = analyze(ishigami.PROBLEM, X[:1000], Y[:1000], n_classes=25)
        r_rag = analyze(ishigami.PROBLEM, X[:1013], Y[:1013], n_classes=25)  # prime N
        ot = np.asarray(r_rag.ot)
        assert np.all(ot >= 0.0) and np.all(ot <= 1.0)
        np.testing.assert_allclose(np.asarray(r_div.ot), ot, atol=0.05)

    def test_dummy_baseline(self, ot_ishigami_boot):
        result = ot_ishigami_boot
        assert result.ot_dummy is not None
        # All three ishigami parameters are continuous, so they share one
        # floor, broadcast to (D,) -- same shape as ot (M4).
        assert result.ot_dummy.shape == (3,)
        dummy = np.asarray(result.ot_dummy)
        np.testing.assert_allclose(dummy, dummy[0])
        # A synthetic independent input must sit far below real inputs.
        assert float(dummy[0]) < float(np.asarray(result.ot).min())
        assert float(dummy[0]) < 0.05

    def test_categorical_dummy_uses_the_observed_level_sizes(self):
        """A categorical floor is matched to that column's own level counts.

        The levels here have unequal sizes (about 600, 300 and 100 rows). A
        floor built with equal-frequency class sizes instead compares the
        output with a wrongly weighted mixture and gives a value near 0.5,
        far above the real index of a weak input.
        """
        problem = jaxgsa.Problem.from_dict(
            {
                "a": (-np.pi, np.pi),
                "b": (-np.pi, np.pi),
                "c": {"dist": "categorical", "probs": [0.6, 0.3, 0.1]},
            }
        )
        rng = np.random.default_rng(0)
        X = rng.uniform(-np.pi, np.pi, (1000, 3))
        X[:, 2] = rng.choice(3, 1000, p=[0.6, 0.3, 0.1])
        Y = np.sin(X[:, 0]) + 7 * np.sin(X[:, 1]) ** 2 + 2 * X[:, 2]
        result = analyze(problem, jnp.asarray(X), jnp.asarray(Y), n_classes=7, dummy=True, key=3)
        assert result.ot_dummy is not None
        assert float(np.asarray(result.ot_dummy)[2]) < 0.05

    def test_dummy_multivariate_mode(self, ishigami_xy_series):
        X, Y2, _ = ishigami_xy_series
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*entropic transport", category=jaxgsa.JaxgsaWarning
            )
            result = analyze(
                ishigami.PROBLEM,
                X[:256],
                Y2[:256],
                mode="multivariate",
                n_classes=4,
                dummy=True,
                key=jax.random.key(0),
            )
        assert result.ot_dummy is not None
        assert result.ot_dummy.shape == (3,)


class TestOTValidation:
    @pytest.mark.parametrize(
        ("case", "kwargs", "match"),
        [
            ("x_1d", {}, "2-D"),
            ("two_columns", {}, "parameters"),
            ("full", {"mode": cast(Literal["univariate"], "sliced")}, "mode"),
            ("full", {"mode": "trajectory"}, "trajectory"),
            ("full", {"n_classes": 1}, "n_classes"),
            ("full", {"n_classes": "N"}, "n_classes"),
            ("full", {"epsilon": 0.0}, "epsilon"),
            ("full", {"epsilon": -0.01}, "epsilon"),
            ("full", {"epsilon": float("nan")}, "epsilon"),
            ("full", {"tol": 0.0}, "tol"),
            ("full", {"tol": float("nan")}, "tol"),
            ("full", {"n_bootstrap": 10, "conf_level": 1.5}, "conf_level"),
            ("full", {"slice_chunk_size": 0}, "slice_chunk_size"),
        ],
        ids=[
            "x_not_2d",
            "wrong_param_count",
            "bad_mode",
            "trajectory_needs_3d",
            "n_classes_one",
            "n_classes_n",
            "epsilon_zero",
            "epsilon_negative",
            "epsilon_nan",
            "tol_zero",
            "tol_nan",
            "bad_conf_level",
            "bad_slice_chunk_size",
        ],
    )
    def test_bad_arguments_are_refused(self, given_ishigami_xy, case, kwargs, match):
        """Each malformed argument raises early and names what is wrong.

        ``epsilon=NaN`` and ``tol=NaN`` used to slip through a ``not x <= 0``
        check, because ``NaN <= 0`` is False. NaN then poisoned the log-kernel,
        the Sinkhorn loop exited at once, and the user got NaN indices with no
        diagnostics. The mode literal is cast so the deliberately invalid
        value reaches the runtime check without a static type error.
        """
        X, Y = given_ishigami_xy
        if case == "x_1d":
            X, Y = jnp.ones(10), Y[:10]
        elif case == "two_columns":
            X = X[:, :2]
        kwargs = {k: (X.shape[0] if v == "N" else v) for k, v in kwargs.items()}
        with pytest.raises(ValueError, match=match):
            analyze(ishigami.PROBLEM, X, Y, **kwargs)

    @pytest.mark.parametrize("bad", [{"epsilon": float("nan")}, {"tol": float("nan")}])
    def test_nan_settings_are_refused_by_the_core_too(self, given_ishigami_xy, bad):
        X, Y = given_ishigami_xy
        with pytest.raises(ValueError, match=next(iter(bad))):
            jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, Y, **bad)


class TestOTXarray:
    def test_time_coords_kwarg_sets_the_time_coordinate(self, ishigami_xy_series):
        """The dims are snapshot-covered; the kwarg's coordinate values are not."""
        X, _, Y3 = ishigami_xy_series
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
        for N in (7, 1000, 100_000):
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

    def test_frac_left_float64_under_x64(self):
        """With x64 on and a float64 dtype, the overlap fractions stay float64."""
        from jaxgsa.optimal_transport._analyze import _quantile_rank_split

        N = 23
        sizes = np.array([7, 13, 3])
        with jax.enable_x64(True):
            _, _, frac = _quantile_rank_split(jnp.asarray(sizes), N, jnp.float64)
            assert frac.dtype == jnp.float64
            r = (np.arange(N)[None] * sizes[:, None]) % N
            exact = np.minimum(N - r, sizes[:, None]) / sizes[:, None]
            np.testing.assert_allclose(np.asarray(frac), exact, rtol=0, atol=1e-15)


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


class TestOTDummyColumnInvalidPolicy:
    """The OT-specific part of ``on_invalid``.

    The raise, propagate and drop cases shared by all given-data methods live
    in ``test_given_data_on_invalid.py``.
    """

    def test_the_check_reads_the_real_sample_not_the_dummy_column(self):
        """T4: ``dummy=True`` adds a synthetic column, and it is not what is checked.

        The dummy is drawn inside ``analyze`` and is finite by construction.
        The report must still count only the user's own parameters, so its
        row positions stay the positions of the model runs.
        """
        problem, X, Y = invalid_sample()
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

    @pytest.mark.parametrize(
        ("kind", "mode"),
        [
            ("scalar", "univariate"),
            ("matrix", "univariate"),
            ("series", "univariate"),
            ("series", "multivariate"),
            ("series", "trajectory"),
        ],
    )
    def test_matches_analyze(self, given_ishigami_xy, ishigami_xy_series, kind, mode):
        """Tier T4: ``indices`` returns the three index fields of ``analyze``.

        ``analyze`` delegates to the same kernel runner, so this guards the
        delegation wiring rather than the estimator maths. ``mode`` is a
        Python string, so the branch stays inside the core: one mode is
        traced per concrete value, exactly as ``analyze`` compiles one. The
        joint modes solve one Sinkhorn problem per class, so they run on a
        slice of the sample with fewer classes. The result keeps the rank of
        the output.
        """
        X, Y2, Y3 = ishigami_xy_series
        outputs = {"scalar": given_ishigami_xy[1], "matrix": Y2, "series": Y3}[kind]
        n_classes = 8
        if mode != "univariate":
            X, outputs, n_classes = X[:256], outputs[:256], 4
        typed_mode = cast(Literal["univariate", "multivariate", "trajectory"], mode)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*entropic transport", category=jaxgsa.JaxgsaWarning
            )
            result = analyze(ishigami.PROBLEM, X, outputs, mode=typed_mode, n_classes=n_classes)
            out = jaxgsa.optimal_transport.indices(
                ishigami.PROBLEM, X, outputs, mode=typed_mode, n_classes=n_classes
            )
        ot, adv, diff, s1 = out
        assert ot.shape == result.ot.shape
        np.testing.assert_array_equal(np.asarray(ot), np.asarray(result.ot))
        np.testing.assert_array_equal(np.asarray(adv), np.asarray(result.advective))
        np.testing.assert_array_equal(np.asarray(diff), np.asarray(result.diffusive))
        np.testing.assert_allclose(np.asarray(s1), np.asarray(result.S1), rtol=1e-6)
        assert isinstance(out, jaxgsa.optimal_transport.OTIndices)
        assert out.S1 is s1

    def test_standardize_outputs_reaches_the_joint_core(self, ishigami_xy_series):
        """Tier T4: ``standardize_outputs`` is honoured identically by core and ``analyze``.

        The keyword is arithmetic over the sample axis, not policy, so the
        core carries it. Turning it off must change the joint indices and
        must change them the same way in both entry points.
        """
        X, _, Y3 = ishigami_xy_series
        X, Y3 = X[:256], Y3[:256]
        kwargs: dict[str, Any] = {"mode": "multivariate", "n_classes": 4}
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*entropic transport", category=jaxgsa.JaxgsaWarning
            )
            on = jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, Y3, **kwargs)[0]
            off = jaxgsa.optimal_transport.indices(
                ishigami.PROBLEM, X, Y3, standardize_outputs=False, **kwargs
            )[0]
            reference = analyze(ishigami.PROBLEM, X, Y3, standardize_outputs=False, **kwargs)

        assert not np.allclose(np.asarray(on), np.asarray(off))
        np.testing.assert_array_equal(np.asarray(off), np.asarray(reference.ot))

    def test_no_bootstrap_and_no_dummy_in_the_core(self, ishigami_xy_4k, ot_ishigami_boot):
        """Tier T4: the core is the point estimate, and takes no key.

        The bootstrap interval and the ``dummy`` baseline both draw
        randomness and both describe the point estimate rather than being
        it, so both stay in ``analyze``.
        """
        X, Y = ishigami_xy_4k
        ot, *_ = jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, Y)

        # The interval is centred on exactly this estimate. Compared at a
        # tight float32 tolerance, not bit-for-bit: the two calls compile
        # the scanned kernel for a different replicate count, and XLA is
        # free to fuse a reduction differently per compiled shape even
        # though replicate 0 is the identical computation both times.
        np.testing.assert_allclose(
            np.asarray(ot), np.asarray(ot_ishigami_boot.ot), rtol=1e-6, atol=1e-7
        )
        with pytest.raises(TypeError):
            cast(Any, jaxgsa.optimal_transport.indices)(
                ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, key=jax.random.key(0)
            )
        with pytest.raises(TypeError):
            cast(Any, jaxgsa.optimal_transport.indices)(ishigami.PROBLEM, X, Y, dummy=True)

    def test_is_jittable(self, ishigami_xy_4k):
        """Tier T4: ``jit`` traces ``indices`` over both X and Y."""
        X, Y = ishigami_xy_4k
        jitted = jax.jit(lambda x, y: jaxgsa.optimal_transport.indices(ishigami.PROBLEM, x, y))
        ot_jit = jitted(X, Y)[0]
        ot = jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, Y)[0]

        np.testing.assert_allclose(np.asarray(ot_jit), np.asarray(ot), rtol=1e-5)

    def test_is_vmappable(self):
        """Tier T4: ``vmap`` maps ``indices`` over a batch of output vectors."""
        N = 1024
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=N, seed=7))
        batch = jnp.stack([ishigami.evaluate(X), X[:, 0], X[:, 1] ** 2])

        mapped = jax.vmap(lambda y: jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, y))(
            batch
        )

        assert mapped[0].shape == (3, ishigami.PROBLEM.n_params)
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
            X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=N, seed=3))
            Y = ishigami.evaluate(X)

            def total_ot(outputs):
                return jaxgsa.optimal_transport.indices(ishigami.PROBLEM, X, outputs)[0].sum()

            jac = jax.jit(jax.jacrev(total_ot))(Y)
            eager = jax.jacrev(total_ot)(Y)

        assert jac.shape == (N,)
        assert np.all(np.isfinite(np.asarray(jac)))
        np.testing.assert_allclose(np.asarray(jac), np.asarray(eager), rtol=1e-8)

    @pytest.mark.parametrize("mode", ["multivariate", "trajectory"])
    def test_point_cloud_modes_differentiate_in_both_modes(self, mode):
        """Tier T4: the Sinkhorn modes give the same derivative in reverse and forward mode.

        The solver stops on a ``while_loop``, which reverse mode cannot pass
        through. The derivative comes from the implicit function theorem at
        the fixed point instead, so ``jax.grad``, ``jax.jvp`` and a central
        finite difference must agree. Before 1.1.2, ``jax.grad`` raised here.
        """
        with jax.enable_x64():
            X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=120, seed=11))
            if mode == "multivariate":
                Y = jnp.stack([ishigami.evaluate(X), X[:, 0] * X[:, 1]], axis=1)
            else:
                Y = jnp.stack([X[:, 0] * t + X[:, 2] * (1 - t) for t in (0.0, 0.5, 1.0)], axis=1)
                Y = Y[..., None]

            def loss(outputs):
                ot = jaxgsa.optimal_transport.indices(
                    ishigami.PROBLEM,
                    X,
                    outputs,
                    mode=mode,
                    epsilon=0.1,
                    tol=1e-13,
                    max_iter=50_000,
                )[0]
                return (ot * jnp.array([1.0, -0.7, 0.4])).sum()

            direction = jax.random.normal(jax.random.key(5), Y.shape)
            reverse = jnp.vdot(jax.jit(jax.grad(loss))(Y), direction)
            _, forward = jax.jvp(loss, (Y,), (direction,))
            h = 1e-6
            central = (loss(Y + h * direction) - loss(Y - h * direction)) / (2 * h)

        np.testing.assert_allclose(float(reverse), float(forward), rtol=1e-8)
        np.testing.assert_allclose(float(reverse), float(central), rtol=1e-5)

    def test_default_epsilon_gradient_matches_a_tight_reference(self):
        """Tier T4: at the default epsilon, the default gradient is accurate.

        At ``epsilon = 0.03`` the scaling update contracts slowly, so a
        loose linear solve gave a gradient off by about 1e-3. The default
        settings must give the same gradient in reverse and forward mode,
        and match a gradient with a much tighter ``tol``.
        """
        with jax.enable_x64():
            X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=120, seed=11))
            Y = jnp.stack([ishigami.evaluate(X), X[:, 0] * X[:, 1]], axis=1)

            def make_loss(**solver):
                def loss(outputs):
                    ot = jaxgsa.optimal_transport.indices(
                        ishigami.PROBLEM, X, outputs, mode="multivariate", **solver
                    )[0]
                    return (ot * jnp.array([1.0, -0.7, 0.4])).sum()

                return loss

            loss = make_loss()
            direction = jax.random.normal(jax.random.key(5), Y.shape)
            reverse = jnp.vdot(jax.jit(jax.grad(loss))(Y), direction)
            _, forward = jax.jvp(loss, (Y,), (direction,))
            tight = make_loss(tol=1e-13, max_iter=100_000)
            reference = jnp.vdot(jax.jit(jax.grad(tight))(Y), direction)

        np.testing.assert_allclose(float(reverse), float(forward), rtol=1e-10)
        np.testing.assert_allclose(float(reverse), float(reference), rtol=1e-6)

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
        X = jnp.asarray(monte_carlo(problem, n_samples=400, seed=5))
        constant = jnp.full((400,), 2.5)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ot, adv, diff, s1 = jaxgsa.optimal_transport.indices(problem, X, constant)

        for arr in (ot, adv, diff, s1):
            np.testing.assert_array_equal(np.asarray(arr), np.zeros(2))
        with pytest.warns(jaxgsa.JaxgsaWarning):
            analyze(problem, X, constant)


class TestS1AndAboveDummy:
    """The derived convenience fields: ``S1`` and ``above_dummy``."""

    def test_s1_is_advective_on_the_ddof0_convention(self, ishigami_xy_4k, ot_ishigami):
        """Tier T4: S1 == 2 * advective * N / (N - 1), exactly.

        The advective numerator is the population (ddof=0) between-class
        variance while the OT normalizer uses ddof=1, so the rescale is the
        one constant that puts both on borgonovo's ddof=0 S1 convention.
        """
        N = ishigami_xy_4k[0].shape[0]
        np.testing.assert_allclose(
            np.asarray(ot_ishigami.S1),
            2.0 * (N / (N - 1)) * np.asarray(ot_ishigami.advective),
            rtol=1e-6,
        )

    def test_s1_agrees_with_borgonovo_s1(self, ishigami_xy_4k, ot_ishigami):
        """Tier T2 (cross-estimator): both given-data S1 estimates coincide."""
        X, Y = ishigami_xy_4k
        delta_result = jaxgsa.borgonovo.analyze(ishigami.PROBLEM, X, Y)
        np.testing.assert_allclose(
            np.asarray(ot_ishigami.S1), np.asarray(delta_result.S1), atol=0.03
        )

    def test_s1_conf_is_the_exactly_rescaled_advective_conf(
        self, ishigami_xy_4k, ot_ishigami_boot
    ):
        N = ishigami_xy_4k[0].shape[0]
        np.testing.assert_allclose(
            np.asarray(ot_ishigami_boot.S1_conf),
            2.0 * (N / (N - 1)) * np.asarray(ot_ishigami_boot.advective_conf),
            rtol=1e-6,
        )

    def test_the_optional_fields_need_their_inputs(self, ot_ishigami):
        """Without a bootstrap there is no ``S1_conf``, and without a dummy no ``above_dummy``."""
        assert ot_ishigami.S1_conf is None
        assert ot_ishigami.above_dummy is None

    def test_above_dummy_is_the_clipped_excess_over_the_floor(self, ot_ishigami_boot):
        result = ot_ishigami_boot
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

    @pytest.mark.slow
    def test_above_dummy_matches_ots_shape_in_trajectory_mode(self, ishigami_xy_series):
        X, Y2, Y3 = ishigami_xy_series
        result = analyze(
            ishigami.PROBLEM,
            X,
            Y3,
            mode="trajectory",
            dummy=True,
            key=jax.random.key(2),
            n_classes=10,
        )
        above = np.asarray(result.above_dummy)
        assert above.shape == np.asarray(result.ot).shape  # (K, D)
        assert np.asarray(result.ot_dummy).shape == np.asarray(result.ot).shape  # (K, D)
        np.testing.assert_allclose(
            above,
            np.maximum(np.asarray(result.ot) - np.asarray(result.ot_dummy), 0.0),
        )

    def test_fields_reach_the_dataset(self, ot_ishigami_boot):
        ds = ot_ishigami_boot.to_dataset()
        assert "S1" in ds
        assert "above_dummy" in ds
        assert ds["S1"].dims == ("param",)
        assert ds["above_dummy"].dims == ("param",)


# ---------------------------------------------------------------------------
# Entropic bias floor
# ---------------------------------------------------------------------------


class TestDummyFloorWarning:
    """The point-cloud modes need a dummy floor to be readable.

    Entropic transport biases every index upward by an amount that does not
    vanish with N, so a parameter the model never reads still scores visibly
    above zero. Nothing in the index array separates the bias from the
    signal, and ``dummy=False`` is the default.
    """

    @staticmethod
    def _data():
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=256, seed=0))
        Y = ishigami.evaluate(X)
        return X, jnp.stack([Y, 2.0 * Y], axis=-1)

    @pytest.mark.parametrize("mode", ["multivariate", "trajectory"])
    def test_point_cloud_without_a_dummy_warns(self, mode):
        """Both entropic modes say the floor is unmeasured."""
        X, Y2 = self._data()
        Y = Y2[:, None, :] if mode == "trajectory" else Y2
        with pytest.warns(jaxgsa.JaxgsaWarning, match="entropic transport"):
            analyze(
                ishigami.PROBLEM,
                X,
                Y,
                mode=cast(Any, mode),
                n_classes=4,
                verbose=False,
            )

    def test_a_dummy_floor_silences_it(self):
        """Asking for the floor is the fix, so it stops warning."""
        X, Y2 = self._data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyze(
                ishigami.PROBLEM,
                X,
                Y2,
                mode="multivariate",
                n_classes=4,
                dummy=True,
                key=jax.random.key(0),
                verbose=False,
            )
        assert not any("entropic transport" in str(w.message) for w in caught)
        assert result.ot_dummy is not None
        assert result.above_dummy is not None

    def test_univariate_is_not_affected(self):
        """The 1-D mode reads a closed-form cost, so it has no floor to warn about."""
        X, Y2 = self._data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analyze(ishigami.PROBLEM, X, Y2, verbose=False)
        assert not any("entropic transport" in str(w.message) for w in caught)


class TestJointMemoryWarning:
    """The joint Sinkhorn path warns when one solve exceeds the memory budget."""

    @staticmethod
    def _run(n=200):
        X = jnp.asarray(np.random.default_rng(0).uniform(-np.pi, np.pi, (n, 3)))
        Y = jnp.stack([X[:, 0], X[:, 1] ** 2], axis=1)
        return analyze(
            ishigami.PROBLEM, X, Y, mode="multivariate", n_classes=4, max_iter=5, dummy=False
        )

    def test_estimate_scales_as_blocks(self):
        from jaxgsa.optimal_transport._analyze import _joint_peak_bytes

        assert _joint_peak_bytes(16384, 656, jnp.float32) == 4 * 16384 * 656 * 4
        assert _joint_peak_bytes(100, 10, jnp.float64) == 4 * 100 * 10 * 8

    def test_warns_when_budget_is_tiny(self):
        from jaxgsa._core import batching

        old = batching.get_memory_budget()
        batching._set_memory_budget(1000)
        try:
            with pytest.warns(jaxgsa.JaxgsaWarning, match="Sinkhorn solve is expected to need"):
                self._run()
        finally:
            batching._set_memory_budget(old)

    def test_silent_under_default_budget(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._run()
        assert not [w for w in caught if "expected to need" in str(w.message)]
