"""Tests for rank-based first-order Sobol' indices (Gamboa et al., 2022)."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _linear_gaussian import (
    A_COEF,
    GAUSS_PROBLEM,
    R_GAUSS,
    analytic_indices,
    gaussian_sampler,
)
from _r_oracle import run_r

import jaxgsa
from jaxgsa import JaxgsaWarning, rank
from jaxgsa.benchmarks import ishigami
from jaxgsa.problem import Problem
from jaxgsa.sampling import monte_carlo

N_BIG = 2**15
BOOT = 8


@pytest.fixture(scope="module")
def ishigami_big():
    """Ishigami Monte Carlo sample large enough for a 0.03 absolute check."""
    X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=N_BIG, seed=7))
    return X, ishigami.evaluate(X)


@pytest.fixture(scope="module")
def ishigami_small():
    X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=1024, seed=3))
    return X, ishigami.evaluate(X)


def _reference_s1(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Eq. (20) of Gamboa et al., written as plain loops over a stable sort."""
    n = len(y)
    out = []
    for d in range(X.shape[1]):
        order = np.argsort(X[:, d], kind="stable")
        pairs = [(order[r], order[(r + 1) % n]) for r in range(n)]
        cross = sum(y[a] * y[b] for a, b in pairs) / n
        out.append((cross - y.mean() ** 2) / (np.mean(y**2) - y.mean() ** 2))
    return np.array(out)


class TestAnalytic:
    """T0: closed forms."""

    def test_ishigami_first_order(self, ishigami_big):
        X, Y = ishigami_big
        result = rank.analyze(ishigami.PROBLEM, X, Y, verbose=False)
        s1_true, _, _ = ishigami.analytical_indices()
        np.testing.assert_allclose(np.asarray(result.S1), s1_true, atol=0.03)

    def test_linear_gaussian_with_correlation(self):
        """``S1_i = (R a)_i^2 / (a' R a)`` holds under a Gaussian copula."""
        problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
        X = jnp.asarray(monte_carlo(problem, n=N_BIG, seed=11))
        Y = X @ jnp.asarray(A_COEF, dtype=X.dtype)
        result = rank.analyze(problem, X, Y, verbose=False)
        s_tc, _, _ = analytic_indices(A_COEF, R_GAUSS)
        np.testing.assert_allclose(np.asarray(result.S1), s_tc, atol=0.03)

    def test_linear_gaussian_with_custom_sampler(self):
        problem = GAUSS_PROBLEM.with_sampler(gaussian_sampler(R_GAUSS))
        X = jnp.asarray(monte_carlo(problem, n=N_BIG, seed=12))
        Y = X @ jnp.asarray(A_COEF, dtype=X.dtype)
        result = rank.analyze(problem, X, Y, verbose=False)
        s_tc, _, _ = analytic_indices(A_COEF, R_GAUSS)
        np.testing.assert_allclose(np.asarray(result.S1), s_tc, atol=0.03)

    def test_a_noiseless_monotone_output_gives_one(self):
        """Sorted by X_0, Y = X_0 is a line, so neighbours nearly share a value."""
        problem = Problem(names=("a", "b"), bounds=((0.0, 1.0), (0.0, 1.0)))
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(4000, 2))
        result = rank.analyze(problem, jnp.asarray(X), jnp.asarray(X[:, 0]), verbose=False)
        s1 = np.asarray(result.S1)
        assert s1[0] > 0.99
        assert abs(s1[1]) < 0.05


class TestEstimator:
    def test_matches_a_loop_reference_of_eq_20(self, ishigami_small):
        """T0: the vectorised kernel equals the written-out formula."""
        X, Y = ishigami_small
        result = rank.analyze(ishigami.PROBLEM, X, Y, verbose=False)
        ref = _reference_s1(np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64))
        np.testing.assert_allclose(np.asarray(result.S1), ref, atol=2e-5)

    def test_the_last_rank_pairs_with_the_first(self):
        """A three-row case worked by hand, including the wrap-around pair."""
        problem = Problem(names=("a",), bounds=((0.0, 1.0),))
        X = jnp.asarray([[0.3], [0.1], [0.2]])
        Y = jnp.asarray([3.0, 1.0, 2.0])
        # Sorted by X: Y = (1, 2, 3). Pairs: (1,2), (2,3), (3,1). Mean Y = 2.
        # Cross mean = (2 + 6 + 3) / 3 = 11/3. Numerator = 11/3 - 4 = -1/3.
        # Variance = (1 + 4 + 9) / 3 - 4 = 2/3. S1 = -1/2.
        result = rank.analyze(problem, X, Y, verbose=False)
        np.testing.assert_allclose(np.asarray(result.S1), [-0.5], atol=1e-6)

    def test_row_order_does_not_matter_without_ties(self):
        X = jnp.asarray(np.random.default_rng(0).uniform(-np.pi, np.pi, size=(1024, 3)))
        assert all(len(np.unique(np.asarray(X[:, d]))) == X.shape[0] for d in range(3))
        Y = ishigami.evaluate(X)
        perm = np.random.default_rng(0).permutation(X.shape[0])
        a = rank.analyze(ishigami.PROBLEM, X, Y, verbose=False)
        b = rank.analyze(ishigami.PROBLEM, X[perm], Y[perm], verbose=False)
        np.testing.assert_allclose(np.asarray(a.S1), np.asarray(b.S1), atol=1e-6)

    def test_ties_are_broken_by_row_order(self):
        """A stable sort makes tied inputs deterministic."""
        problem = Problem(names=("a",), bounds=((0.0, 1.0),))
        X = jnp.asarray([[0.5]] * 6 + [[0.1], [0.9]])
        Y = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 7.0])
        first = rank.analyze(problem, X, Y, verbose=False)
        again = rank.analyze(problem, X, Y, verbose=False)
        ref = _reference_s1(np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64))
        np.testing.assert_array_equal(np.asarray(first.S1), np.asarray(again.S1))
        np.testing.assert_allclose(np.asarray(first.S1), ref, atol=1e-5)

    def test_constant_output_warns_and_gives_nan(self):
        problem = Problem(names=("a",), bounds=((0.0, 1.0),))
        X = jnp.asarray(np.linspace(0.0, 1.0, 50).reshape(-1, 1))
        with pytest.warns(JaxgsaWarning):
            result = rank.analyze(problem, X, jnp.full((50,), 3.0), verbose=False)
        assert np.all(np.isnan(np.asarray(result.S1)))


class TestShapes:
    def test_output_layouts_match_per_slice_scalar_runs(self, ishigami_small):
        X, Y = ishigami_small
        Y2 = jnp.stack([Y, Y**2], axis=1)
        Y3 = jnp.stack([Y2, Y2 + 1.0, 2.0 * Y2], axis=1)  # (N, 3, 2)
        scalar = np.asarray(rank.analyze(ishigami.PROBLEM, X, Y, verbose=False).S1)
        assert scalar.shape == (3,)
        r2 = np.asarray(rank.analyze(ishigami.PROBLEM, X, Y2, verbose=False).S1)
        r3 = np.asarray(rank.analyze(ishigami.PROBLEM, X, Y3, verbose=False).S1)
        assert r2.shape == (2, 3)
        assert r3.shape == (3, 2, 3)
        np.testing.assert_allclose(r2[0], scalar, atol=1e-6)
        sq = np.asarray(rank.analyze(ishigami.PROBLEM, X, Y**2, verbose=False).S1)
        np.testing.assert_allclose(r2[1], sq, atol=1e-6)
        # Adding a constant or scaling changes no index.
        np.testing.assert_allclose(r3[0], r2, atol=1e-5)
        np.testing.assert_allclose(r3[2], r2, atol=1e-5)

    def test_the_result_exports_a_dataset(self, ishigami_small):
        X, Y = ishigami_small
        result = rank.analyze(ishigami.PROBLEM, X, Y, verbose=False)
        assert "S1" in result.to_dataset()


class TestRefusals:
    CATEGORICAL = Problem.from_dict(
        {
            "a": jaxgsa.CategoricalInputSpec(dist="categorical", probs=[0.5, 0.5]),
            "b": jaxgsa.UniformInputSpec(dist="uniform", low=0.0, high=1.0),
        }
    )

    def test_categorical_inputs_are_refused_by_analyze(self):
        X = jnp.asarray(monte_carlo(self.CATEGORICAL, n=64, seed=0))
        with pytest.raises(ValueError, match="categorical"):
            rank.analyze(self.CATEGORICAL, X, X[:, 1], verbose=False)

    def test_categorical_inputs_are_refused_by_indices(self):
        X = jnp.asarray(monte_carlo(self.CATEGORICAL, n=64, seed=0))
        with pytest.raises(ValueError, match="categorical"):
            rank.indices(self.CATEGORICAL, X, X[:, 1])

    def test_too_few_rows_are_refused(self):
        problem = Problem(names=("a",), bounds=((0.0, 1.0),))
        with pytest.raises(ValueError):
            rank.analyze(problem, jnp.asarray([[0.1], [0.2]]), jnp.asarray([1.0, 2.0]))

    def test_mismatched_rows_are_refused(self, ishigami_small):
        X, Y = ishigami_small
        with pytest.raises(ValueError):
            rank.analyze(ishigami.PROBLEM, X, Y[:-1], verbose=False)


class TestBootstrap:
    @pytest.fixture(scope="class")
    def boot(self, ishigami_small):
        X, Y = ishigami_small
        return rank.analyze(
            ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, key=jax.random.key(0), verbose=False
        )

    def test_no_bootstrap_means_no_interval(self, ishigami_small):
        X, Y = ishigami_small
        result = rank.analyze(ishigami.PROBLEM, X, Y, verbose=False)
        assert result.S1_conf is None and result.ci is None

    def test_interval_shape_and_order(self, boot):
        assert boot.S1_conf is not None
        assert boot.S1_conf.shape == (2, 3)
        assert np.all(np.asarray(boot.S1_conf[0]) <= np.asarray(boot.S1_conf[1]) + 1e-6)

    def test_bootstrap_without_a_key_raises(self, ishigami_small):
        X, Y = ishigami_small
        with pytest.raises(ValueError, match="key is required"):
            rank.analyze(ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, verbose=False)

    @pytest.mark.parametrize("level", [0.0, 1.0, 1.5, -0.1])
    def test_conf_level_must_be_inside_zero_one(self, ishigami_small, level):
        X, Y = ishigami_small
        with pytest.raises(ValueError, match="conf_level"):
            rank.analyze(
                ishigami.PROBLEM,
                X,
                Y,
                n_bootstrap=BOOT,
                conf_level=level,
                key=jax.random.key(0),
                verbose=False,
            )

    def test_negative_n_bootstrap_raises(self, ishigami_small):
        X, Y = ishigami_small
        with pytest.raises(ValueError, match="n_bootstrap"):
            rank.analyze(ishigami.PROBLEM, X, Y, n_bootstrap=-1, verbose=False)

    def test_unknown_ci_method_raises(self, ishigami_small):
        X, Y = ishigami_small
        with pytest.raises(ValueError, match="ci_method"):
            rank.analyze(
                ishigami.PROBLEM,
                X,
                Y,
                n_bootstrap=BOOT,
                ci_method=cast(Any, "bca"),
                key=jax.random.key(0),
                verbose=False,
            )

    def test_the_key_decides_the_interval(self, ishigami_small, boot):
        X, Y = ishigami_small
        same = rank.analyze(
            ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, key=jax.random.key(0), verbose=False
        )
        other = rank.analyze(
            ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, key=jax.random.key(5), verbose=False
        )
        assert boot.S1_conf is not None and same.S1_conf is not None and other.S1_conf is not None
        np.testing.assert_array_equal(np.asarray(same.S1_conf), np.asarray(boot.S1_conf))
        assert not np.array_equal(np.asarray(other.S1_conf), np.asarray(same.S1_conf))
        np.testing.assert_array_equal(np.asarray(other.S1), np.asarray(same.S1))

    def test_gaussian_interval_holds_the_symmetric_interval(self, ishigami_small):
        # The reported interval is the hull of a symmetric interval around the
        # estimate and one around the bias-corrected estimate, so it holds
        # the symmetric one and is no longer centred on the estimate.
        X, Y = ishigami_small
        result = rank.analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_bootstrap=BOOT,
            ci_method="gaussian",
            key=jax.random.key(0),
            verbose=False,
            keep_replicates=True,
        )
        assert result.S1_conf is not None and result.ci is not None
        lo, hi = np.asarray(result.S1_conf)
        half = 1.959964 * np.std(np.asarray(result.ci.replicates["S1"]), axis=0, ddof=1)
        s1 = np.asarray(result.S1)
        assert np.all(lo <= s1 - half + 1e-6)
        assert np.all(hi >= s1 + half - 1e-6)

    def test_replicates_are_kept_on_request(self, ishigami_small):
        X, Y = ishigami_small
        result = rank.analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_bootstrap=BOOT,
            key=jax.random.key(0),
            keep_replicates=True,
            verbose=False,
        )
        assert result.ci is not None and result.ci.replicates is not None
        assert result.ci.replicates["S1"].shape == (BOOT, 3)

    def test_the_interval_is_not_biased_upward_by_repeated_rows(self, ishigami_big):
        """A resample with replacement would put the null parameter near 0.37.

        Ishigami's ``x3`` has a true first-order index of 0. The interval of
        a bias-free resample must cover 0 and must stay well below 0.3.
        """
        X, Y = ishigami_big
        X, Y = X[:4000], Y[:4000]
        result = rank.analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_bootstrap=40,
            ci_method="gaussian",
            key=jax.random.key(1),
            keep_replicates=True,
            verbose=False,
        )
        assert result.S1_conf is not None and result.ci is not None
        assert result.ci.replicates is not None
        lower, upper = (float(result.S1_conf[0, 2]), float(result.S1_conf[1, 2]))
        assert lower < 0.0 < upper or abs(lower) < 0.05
        assert upper < 0.15
        assert float(jnp.mean(result.ci.replicates["S1"][:, 2])) < 0.1

    def test_the_interval_width_matches_the_spread_of_the_estimator(self):
        """T0 (simulation): the rescaled spread is not too narrow.

        The standard deviation of the estimate over independent samples is the
        truth. The mean half-width implied by the resamples must reach at
        least 80% of it, and must not exceed twice of it.
        """
        problem = ishigami.PROBLEM
        n, n_rep = 1500, 24
        estimates, spreads = [], []
        for seed in range(n_rep):
            X = jnp.asarray(monte_carlo(problem, n=n, seed=100 + seed))
            Y = ishigami.evaluate(X)
            r = rank.analyze(
                problem,
                X,
                Y,
                n_bootstrap=BOOT,
                key=jax.random.key(seed),
                keep_replicates=True,
                verbose=False,
            )
            assert r.ci is not None and r.ci.replicates is not None
            estimates.append(np.asarray(r.S1))
            spreads.append(np.asarray(r.ci.replicates["S1"]).std(axis=0, ddof=1))
        true_sd = np.std(np.stack(estimates), axis=0, ddof=1)
        implied = np.mean(np.stack(spreads), axis=0)
        ratio = implied / true_sd
        assert np.all(ratio > 0.8), ratio
        assert np.all(ratio < 2.0), ratio


class TestIrregular:
    def test_each_channel_matches_a_direct_run(self):
        problem = Problem.from_dict({"x0": (0.0, 1.0), "x1": (0.0, 1.0)}, output_names=("conc",))
        rng = np.random.default_rng(0)
        X = jnp.asarray(rng.uniform(size=(200, 2)))
        y = jnp.asarray(rng.normal(size=(200, 4)) + np.asarray(X[:, :1]))
        grid = jnp.array([0.0, 1.0, 2.0, 3.0])
        irregular = rank.analyze(problem, X, [(grid, y)], verbose=False)
        direct = rank.analyze(problem, X, y[:, :, None], verbose=False)
        np.testing.assert_allclose(
            np.asarray(irregular.channels["conc"].S1), np.asarray(direct.S1), atol=1e-6
        )


class TestPureCore:
    def test_indices_equals_analyze(self, ishigami_small):
        X, Y = ishigami_small
        (s1,) = rank.indices(ishigami.PROBLEM, X, Y)
        result = rank.analyze(ishigami.PROBLEM, X, Y, verbose=False)
        np.testing.assert_array_equal(np.asarray(s1), np.asarray(result.S1))

    def test_jit_and_vmap(self, ishigami_small):
        X, Y = ishigami_small
        (expected,) = rank.indices(ishigami.PROBLEM, X, Y)
        (jitted,) = jax.jit(lambda x, y: rank.indices(ishigami.PROBLEM, x, y))(X, Y)
        np.testing.assert_allclose(np.asarray(jitted), np.asarray(expected), atol=1e-6)
        Ys = jnp.stack([Y, Y**2])
        (batched,) = jax.vmap(lambda y: rank.indices(ishigami.PROBLEM, X, y))(Ys)
        assert batched.shape == (2, 3)
        np.testing.assert_allclose(np.asarray(batched[0]), np.asarray(expected), atol=1e-6)

    def test_the_gradient_with_respect_to_y_is_finite(self, ishigami_small):
        X, Y = ishigami_small
        grad = jax.grad(lambda y: rank.indices(ishigami.PROBLEM, X, y)[0].sum())(Y)
        assert grad.shape == Y.shape
        assert bool(jnp.all(jnp.isfinite(grad)))


@pytest.mark.slow
class TestROracle:
    """T3: compare with ``sensitivity::sobolrank`` run as a black box."""

    SCRIPT = """
    x <- sobolrank(model = NULL, X = as.data.frame(X))
    tell(x, as.numeric(Y))
    S <- as.numeric(unlist(x$S))
    """

    @pytest.mark.parametrize("n", [500, 3000])
    def test_matches_r_sobolrank(self, n):
        """R divides by the sample variance (n - 1). jaxgsa follows the paper (n).

        The two differ by the exact factor ``n / (n - 1)``, so the comparison
        rescales once and then asserts a tight match.
        """
        rng = np.random.default_rng(n)
        X = rng.uniform(-np.pi, np.pi, size=(n, 3))
        Y = np.asarray(ishigami.evaluate(jnp.asarray(X)), dtype=np.float64)
        # Feed both sides the same float32 values that JAX holds.
        X32 = np.asarray(jnp.asarray(X), dtype=np.float64)
        Y32 = np.asarray(jnp.asarray(Y), dtype=np.float64)
        out = run_r(
            self.SCRIPT, arrays={"X": X32, "Y": Y32}, outputs=("S",), packages=("sensitivity",)
        )
        ours = np.asarray(
            rank.analyze(ishigami.PROBLEM, jnp.asarray(X), jnp.asarray(Y), verbose=False).S1
        )
        np.testing.assert_allclose(ours * (n - 1) / n, out["S"], atol=2e-4)


def test_interval_covers_a_deterministic_relation():
    # For Y = x1 the estimate is about 1 - 6/N, a bias far larger than the
    # spread of the estimator. The bias-corrected interval must still cover
    # the true value 1.
    problem = Problem.from_dict(
        {f"x{i}": {"dist": "uniform", "low": 0.0, "high": 1.0} for i in (1, 2)}
    )
    X = monte_carlo(problem, 1000, seed=0)
    Y = jnp.asarray(X)[:, 0]
    res = rank.analyze(problem, X, Y, n_bootstrap=100, key=jax.random.key(0), verbose=False)
    lo, hi = np.asarray(res.S1_conf)[:, 0]
    assert float(res.S1[0]) < 1.0 - 3.0 / 1000
    assert lo <= 1.0 <= hi
