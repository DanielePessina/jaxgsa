"""Tests for the samplers: design layout, marginals, correlation, downsampling.

Tier T4 (internal consistency) except where noted: most tests pin design
invariants — uniqueness, nesting, determinism, bounds. The truncated-Gaussian
moment checks compare live against ``scipy.stats.truncnorm`` (Tier T2), and
the copula tests check recovered rank correlations against the declared
targets (T4: the target is our own input).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import truncnorm

import jaxgsa
from jaxgsa._core.sampling import _next_power_of_2
from jaxgsa.problem import GaussianInputSpec, InputSpecValue, Problem, UniformInputSpec
from jaxgsa.sampling import correlate, fit_correlation, monte_carlo
from jaxgsa.sobol import sample
from jaxgsa.sobol._sampling import _saltelli_step


def test_next_power_of_2():
    assert _next_power_of_2(1) == 1
    assert _next_power_of_2(2) == 2
    assert _next_power_of_2(3) == 4
    assert _next_power_of_2(5) == 8
    assert _next_power_of_2(1024) == 1024
    assert _next_power_of_2(1025) == 2048


def test_sample_returns_unique_rows():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)})
    result = sample(p, n_samples=100, seed=42, verbose=False)
    assert result.n_runs >= 100
    assert result.samples.shape == (result.n_runs, p.num_vars)
    assert np.unique(result.samples, axis=0).shape[0] == result.n_runs
    assert result.sample_ids.tolist() == list(range(result.n_runs))
    assert result.expanded_to_unique.shape == (result.n_expanded,)
    assert result.expanded_to_unique.max() < result.n_runs
    assert result.n_params == p.num_vars
    assert result.calc_second_order is True


def test_sample_within_bounds():
    p = Problem.from_dict({"x1": (-5.0, 5.0), "x2": (0.0, 10.0)})
    result = sample(p, n_samples=200, seed=42, verbose=False)
    assert np.all(result.samples[:, 0] >= -5.0)
    assert np.all(result.samples[:, 0] <= 5.0)
    assert np.all(result.samples[:, 1] >= 0.0)
    assert np.all(result.samples[:, 1] <= 10.0)


def test_power_of_2_enforcement():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=100, seed=42, verbose=False)
    assert result.base_n & (result.base_n - 1) == 0
    with pytest.raises(ValueError, match=r"power of 2 .*got 1000.*nearest valid: 512 or 1024"):
        sample(p, n_samples=100, base_n=1000, seed=42, verbose=False)


def test_single_parameter_mapping_collapses_duplicates():
    p = Problem.from_dict({"x1": (0.0, 1.0)})

    first_only = sample(p, n_samples=16, calc_second_order=False, seed=42, verbose=False)
    step = _saltelli_step(p.num_vars, False)
    for i in range(first_only.base_n):
        group = first_only.expanded_to_unique[i * step : (i + 1) * step]
        assert group[1] == group[2]
        assert group[0] != group[1]

    second_order = sample(p, n_samples=16, calc_second_order=True, seed=42, verbose=False)
    step = _saltelli_step(p.num_vars, True)
    for i in range(second_order.base_n):
        group = second_order.expanded_to_unique[i * step : (i + 1) * step]
        assert group[0] == group[2]
        assert group[1] == group[3]
        assert group[0] != group[1]


def test_two_parameter_second_order_mapping_collapses_cross_duplicates():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=32, calc_second_order=True, seed=42, verbose=False)
    step = _saltelli_step(p.num_vars, True)

    for i in range(result.base_n):
        group = result.expanded_to_unique[i * step : (i + 1) * step]
        assert group[1] == group[4]
        assert group[2] == group[3]
        assert len(set(group.tolist())) == 4


def test_reconstructing_expanded_samples_matches_mapping():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=64, seed=42, verbose=False)
    reconstructed = result.samples[result.expanded_to_unique]
    assert reconstructed.shape == (result.n_expanded, p.num_vars)
    assert np.unique(reconstructed, axis=0).shape[0] == result.n_runs


@pytest.mark.verbose_output
def test_sample_verbose_false_is_silent(capsys):
    """The marker keeps the emit seam live; otherwise this asserts nothing."""
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sample(p, n_samples=32, seed=42, verbose=False)
    out = capsys.readouterr().out
    assert out == ""


def test_uniform_columns_stay_within_bounds_for_mixed_problem():
    p = Problem.from_dict(
        {
            "uniform": UniformInputSpec(dist="uniform", low=-3.0, high=2.0),
            "gaussian": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
        }
    )

    result = sample(p, n_samples=256, seed=1, verbose=False)
    assert np.all(result.samples[:, 0] >= -3.0)
    assert np.all(result.samples[:, 0] <= 2.0)


def test_gaussian_column_matches_target_mean_and_variance():
    p = Problem.from_dict(
        {
            "x1": GaussianInputSpec(dist="gaussian", mean=1.5, variance=2.25),
            "x2": (0.0, 1.0),
        }
    )

    result = sample(p, n_samples=4096, calc_second_order=False, seed=123, verbose=False)
    gaussian = result.samples[:, 0]

    assert abs(np.mean(gaussian) - 1.5) < 0.05
    assert abs(np.var(gaussian) - 2.25) < 0.08


def test_truncated_gaussian_columns_respect_one_sided_and_two_sided_bounds():
    p = Problem.from_dict(
        {
            "lower_only": GaussianInputSpec(
                dist="gaussian",
                mean=0.0,
                variance=1.0,
                low=-0.25,
            ),
            "upper_only": GaussianInputSpec(
                dist="gaussian",
                mean=0.0,
                variance=1.0,
                high=0.5,
            ),
            "two_sided": GaussianInputSpec(
                dist="gaussian",
                mean=0.0,
                variance=1.0,
                low=-1.0,
                high=1.0,
            ),
        }
    )

    result = sample(p, n_samples=512, calc_second_order=False, seed=7, verbose=False)
    assert np.all(result.samples[:, 0] >= -0.25)
    assert np.all(result.samples[:, 1] <= 0.5)
    assert np.all(result.samples[:, 2] >= -1.0)
    assert np.all(result.samples[:, 2] <= 1.0)


def test_two_sided_truncated_gaussian_matches_target_variance_formula():
    p = Problem.from_dict(
        {
            "x": GaussianInputSpec(
                dist="gaussian",
                mean=0.5,
                variance=1.44,
                low=-0.5,
                high=1.5,
            )
        }
    )

    result = sample(p, n_samples=4096, calc_second_order=False, seed=99, verbose=False)
    observed = np.var(result.samples[:, 0])
    std = np.sqrt(1.44)
    a = (-0.5 - 0.5) / std
    b = (1.5 - 0.5) / std
    expected = truncnorm.var(a, b, loc=0.5, scale=std)
    assert abs(observed - expected) < 0.03


# ---------------------------------------------------------------------------
# Prefix downsampling tests
# ---------------------------------------------------------------------------


class TestSamplingResultDownsample:
    """Tests for SobolSamples.downsample()."""

    def _make_sr(self, D: int = 3, base_n: int = 32, second_order: bool = True, seed: int = 42):
        names: dict[str, InputSpecValue] = {f"x{i}": (0.0, 1.0) for i in range(D)}
        p = Problem.from_dict(names)
        return sample(
            p, n_samples=1, base_n=base_n, calc_second_order=second_order, seed=seed, verbose=False
        )

    def test_identity_when_same_base_n(self):
        sr = self._make_sr(base_n=16)
        assert sr.downsample(16) is sr

    def test_samples_are_prefix(self):
        sr_full = self._make_sr(base_n=64)
        sr_small = sr_full.downsample(16)
        assert np.array_equal(sr_small.samples, sr_full.samples[: sr_small.n_runs])

    def test_multiple_rungs_are_nested(self):
        sr_full = self._make_sr(base_n=64)
        sr_32 = sr_full.downsample(32)
        sr_16 = sr_full.downsample(16)
        sr_8 = sr_full.downsample(8)
        assert sr_8.n_runs <= sr_16.n_runs <= sr_32.n_runs <= sr_full.n_runs
        assert np.array_equal(sr_8.samples, sr_16.samples[: sr_8.n_runs])
        assert np.array_equal(sr_16.samples, sr_32.samples[: sr_16.n_runs])

    def test_upsample_raises(self):
        sr = self._make_sr(base_n=16)
        with pytest.raises(ValueError, match="Cannot upsample"):
            sr.downsample(32)

    def test_non_power_of_two_raises(self):
        sr = self._make_sr(base_n=16)
        with pytest.raises(ValueError, match=r"power of 2 .*nearest valid: 8 or 16"):
            sr.downsample(12)

    def test_with_Y_returns_tuple(self):
        sr_full = self._make_sr(base_n=32)
        Y = np.arange(sr_full.n_runs * 4, dtype=np.float64).reshape(sr_full.n_runs, 4)
        sr_small, Y_small = sr_full.downsample(8, Y)
        assert Y_small.shape == (sr_small.n_runs, 4)
        assert np.array_equal(Y_small, Y[: sr_small.n_runs])

    def test_with_Y_identity_returns_same_Y(self):
        sr_full = self._make_sr(base_n=16)
        Y = np.ones((sr_full.n_runs, 3))
        sr_same, Y_same = sr_full.downsample(16, Y)
        assert sr_same is sr_full
        assert Y_same is Y

    def test_with_Y_misaligned_raises(self):
        sr_full = self._make_sr(base_n=32)
        Y_wrong = np.zeros((sr_full.n_runs + 5, 3))
        with pytest.raises(ValueError, match="does not match n_runs"):
            sr_full.downsample(8, Y_wrong)

    def test_samples_do_not_share_memory(self):
        sr_full = self._make_sr(base_n=32)
        sr_small = sr_full.downsample(8)
        assert not np.shares_memory(sr_full.samples, sr_small.samples)

    def test_Y_does_not_share_memory(self):
        sr_full = self._make_sr(base_n=32)
        Y = np.ones((sr_full.n_runs, 3))
        _, Y_small = sr_full.downsample(8, Y)
        assert not np.shares_memory(Y, Y_small)

    def test_downsample_is_bit_identical_to_direct_draw(self):
        """Prefix property: downsampling to K equals drawing K base points directly.

        This backs the ``downsample`` docstring claim that the first K base
        points of a draw with N > K base points are bit-identical to drawing
        K base points with the same seed and scramble.
        """
        p = Problem.from_dict(
            {
                "uniform": UniformInputSpec(dist="uniform", low=-2.0, high=3.0),
                "gaussian": GaussianInputSpec(
                    dist="gaussian", mean=1.0, variance=4.0, low=-1.0, high=4.0
                ),
            }
        )
        N, K, seed = 64, 16, 1234

        sr_small = sample(p, n_samples=1, base_n=N, seed=seed, verbose=False).downsample(K)
        sr_direct = sample(p, n_samples=1, base_n=K, seed=seed, verbose=False)

        np.testing.assert_array_equal(sr_small.samples, sr_direct.samples)
        np.testing.assert_array_equal(sr_small.expanded_to_unique, sr_direct.expanded_to_unique)
        np.testing.assert_array_equal(sr_small.sample_ids, sr_direct.sample_ids)
        assert sr_small.n_expanded == sr_direct.n_expanded
        assert sr_small.base_n == sr_direct.base_n == K


# ---------------------------------------------------------------------------
# Correlated sampling (Gaussian copula on Problem.correlation)
# ---------------------------------------------------------------------------


def _spearman_of(X: np.ndarray) -> np.ndarray:
    """Sample Spearman rank-correlation matrix of the columns of X."""
    ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float64)
    return np.corrcoef(ranks, rowvar=False)


def test_monte_carlo_gaussian_marginals_reproduce_mvn_covariance():
    """All-Gaussian marginals + latent R: the copula *is* the MVN N(mu, DRD)."""
    rho = 0.6
    sigma = np.array([1.5, 2.0])
    problem = Problem.from_dict(
        {
            "a": GaussianInputSpec(dist="gaussian", mean=1.0, variance=float(sigma[0] ** 2)),
            "b": GaussianInputSpec(dist="gaussian", mean=-2.0, variance=float(sigma[1] ** 2)),
        },
        correlation=[[1.0, rho], [rho, 1.0]],
    )
    X = monte_carlo(problem, 100_000, seed=42)
    expected = sigma[:, None] * np.array([[1.0, rho], [rho, 1.0]]) * sigma[None, :]
    np.testing.assert_allclose(np.cov(X, rowvar=False), expected, atol=0.08)
    np.testing.assert_allclose(np.mean(X, axis=0), [1.0, -2.0], atol=0.03)


def test_spearman_kind_recovers_rank_correlation_with_uniform_marginals():
    rho_s = 0.7
    R = [[1.0, rho_s], [rho_s, 1.0]]
    spearman_problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=R, correlation_type="spearman"
    )
    X = monte_carlo(spearman_problem, 100_000, seed=3)
    assert abs(_spearman_of(X)[0, 1] - rho_s) < 0.01

    # The latent kind would *not* hit the rank target: a latent 0.7 yields
    # a Spearman correlation of (6/pi) asin(0.7/2) ~ 0.683, which is what
    # justifies exposing the kind switch at all.
    latent_problem = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=R)
    X_latent = monte_carlo(latent_problem, 100_000, seed=3)
    expected_spearman = (6.0 / np.pi) * np.arcsin(rho_s / 2.0)
    assert abs(_spearman_of(X_latent)[0, 1] - expected_spearman) < 0.01
    assert abs(_spearman_of(X_latent)[0, 1] - rho_s) > 0.01


def test_correlated_sampling_preserves_marginals():
    """Coupling must not disturb the declared marginals (moments + bounds)."""
    R = np.array(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ]
    )
    problem = Problem.from_dict(
        {
            "uniform": UniformInputSpec(dist="uniform", low=-3.0, high=2.0),
            "gaussian": GaussianInputSpec(dist="gaussian", mean=1.5, variance=2.25),
            "two_sided": GaussianInputSpec(
                dist="gaussian", mean=0.5, variance=1.44, low=-0.5, high=1.5
            ),
        },
        correlation=R,
    )
    X = monte_carlo(problem, 16_384, seed=123)

    assert np.all(X[:, 0] >= -3.0)
    assert np.all(X[:, 0] <= 2.0)
    assert abs(np.mean(X[:, 1]) - 1.5) < 0.05
    assert abs(np.var(X[:, 1]) - 2.25) < 0.08
    assert np.all(X[:, 2] >= -0.5)
    assert np.all(X[:, 2] <= 1.5)
    std = np.sqrt(1.44)
    a = (-0.5 - 0.5) / std
    b = (1.5 - 0.5) / std
    assert abs(np.var(X[:, 2]) - truncnorm.var(a, b, loc=0.5, scale=std)) < 0.03


def test_identity_correlation_is_bitwise_identical_to_independent_path():
    """An identity R short-circuits to the plain pseudo-random path."""
    independent = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    identity = independent.with_correlation(np.eye(2))
    np.testing.assert_array_equal(
        monte_carlo(independent, 512, seed=9), monte_carlo(identity, 512, seed=9)
    )


def test_correlated_monte_carlo_determinism_and_generator_seed():
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, 0.8], [0.8, 1.0]]
    )
    np.testing.assert_array_equal(
        monte_carlo(problem, 128, seed=5), monte_carlo(problem, 128, seed=5)
    )
    assert not np.array_equal(monte_carlo(problem, 128, seed=5), monte_carlo(problem, 128, seed=6))
    from_generator = monte_carlo(problem, 128, seed=np.random.default_rng(5))
    np.testing.assert_array_equal(from_generator, monte_carlo(problem, 128, seed=5))


def test_correlated_monte_carlo_rejects_nonpositive_n():
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, 0.8], [0.8, 1.0]]
    )
    with pytest.raises(ValueError, match="n must be >= 1"):
        monte_carlo(problem, 0)


def test_correlate_is_a_per_column_permutation_hitting_the_target():
    rho = 0.8
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, rho], [rho, 1.0]]
    )
    X = monte_carlo(problem.with_correlation(None), 8192, seed=21)
    X_corr = correlate(X, problem, seed=22)

    # Exact permutation of each column: sample values are fully preserved.
    np.testing.assert_array_equal(np.sort(X_corr, axis=0), np.sort(X, axis=0))
    # And the re-pairing achieves the declared latent correlation.
    assert abs(fit_correlation(problem, X_corr)[0, 1] - rho) < 0.05


def test_correlate_determinism_and_validation():
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, 0.5], [0.5, 1.0]]
    )
    X = monte_carlo(problem.with_correlation(None), 256, seed=1)
    np.testing.assert_array_equal(correlate(X, problem, seed=2), correlate(X, problem, seed=2))

    with pytest.raises(ValueError, match="requires problem.correlation"):
        correlate(X, problem.with_correlation(None))
    with pytest.raises(ValueError, match=r"X must be \(N, 2\)"):
        correlate(X[:, :1], problem)


def test_correlate_sampling_error_matches_iman_conover():
    """Pin the variance reduction the de-correlation step buys.

    A plain correlated normal score matrix carries its own sampling noise into
    the re-pairing. At N = 50 that gave a standard deviation of about 0.065 in
    the achieved rank correlation, plus a bias of about -0.006. Iman-Conover
    removes both: about 0.024 and no bias. The thresholds below sit between
    the two, so the old implementation fails this test.
    """
    from scipy.stats import spearmanr

    rho = 0.8
    n, replicates = 50, 150
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, rho], [rho, 1.0]]
    )
    # Spearman rank correlation implied by a latent Pearson rho under the copula.
    target = 6.0 / np.pi * np.arcsin(rho / 2.0)

    achieved = np.array(
        [
            spearmanr(
                correlate(monte_carlo(problem.with_correlation(None), n, seed=r), problem, seed=r)
            ).statistic
            for r in range(replicates)
        ]
    )
    assert achieved.std() < 0.040  # old implementation: ~0.065
    assert abs(achieved.mean() - target) < 0.004  # old implementation: ~-0.006


def test_correlate_rejects_non_finite_X():
    """Tier T4 (input-contract): NaN rows must be rejected, not silently paired.

    np.sort puts NaN last, so every NaN row would deterministically take the
    highest van der Waerden scores and bias the correlation of the finite
    rows even if the NaNs were dropped afterwards.
    """
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, 0.5], [0.5, 1.0]]
    )
    X = monte_carlo(problem.with_correlation(None), 64, seed=5)
    X[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        correlate(X, problem, seed=5)
    X[3, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        correlate(X, problem, seed=5)


def test_correlate_handles_degenerate_row_counts():
    """Too few rows to estimate corr(M): fall back, do not divide by zero."""
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, 0.5], [0.5, 1.0]]
    )
    for n in (1, 2, 3):
        X = monte_carlo(problem.with_correlation(None), n, seed=n)
        out = correlate(X, problem, seed=n)
        assert np.isfinite(out).all()
        np.testing.assert_array_equal(np.sort(out, axis=0), np.sort(X, axis=0))


def test_fit_correlation_public_wrapper_recovers_declared_matrix():
    rho = 0.7
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, rho], [rho, 1.0]]
    )
    X = monte_carlo(problem, 20_000, seed=17)
    fitted = fit_correlation(problem, X)
    assert fitted.shape == (2, 2)
    assert abs(fitted[0, 1] - rho) < 0.02


def test_correlated_end_to_end_ot_borgonovo_hdmr():
    """Y = X1 with corr(X1, X2) = 0.8: the unused X2 must earn a clearly
    non-zero index under the correlation-inclusive given-data methods, and
    HDMR's ANCOVA Sb term must flag the correlation-induced variance."""
    rho = 0.8
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, rho], [rho, 1.0]]
    )
    X = monte_carlo(problem, 2048, seed=99)
    Y = jnp.asarray(X[:, 0])
    Xj = jnp.asarray(X)

    ot = jaxgsa.optimal_transport.analyze(problem, Xj, Y)
    assert float(ot.ot[0]) > 0.5  # Y is fully determined by X1
    assert float(ot.ot[1]) > 0.1  # X2 unused, but correlated with X1

    delta = jaxgsa.borgonovo.analyze(problem, Xj, Y, key=jax.random.key(0))
    assert float(delta.delta[0]) > 0.5
    assert float(delta.delta[1]) > 0.1

    # For Y = X1 backfitting attributes everything to the x1 component, so
    # the correlative share is clearest on an additive model of both inputs:
    # each first-order term covaries with the other through corr(X1, X2).
    Y_sum = jnp.asarray(X[:, 0] + X[:, 1])
    hdmr = jaxgsa.hdmr.analyze(problem, Xj, Y_sum)
    assert float(jnp.max(jnp.abs(hdmr.Sb))) > 0.1

    # Same model on an independent draw: the correlative share collapses.
    independent = problem.with_correlation(None)
    X0 = monte_carlo(independent, 2048, seed=98)
    hdmr_indep = jaxgsa.hdmr.analyze(
        independent, jnp.asarray(X0), jnp.asarray(X0[:, 0] + X0[:, 1])
    )
    assert float(jnp.max(jnp.abs(hdmr_indep.Sb))) < 0.05
