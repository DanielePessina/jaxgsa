"""Tests for the Morris elementary-effects screening method."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning
from jaxgsa._core.invalid import InvalidUnit
from jaxgsa.benchmarks import ishigami, linear, sobol_g
from jaxgsa.morris import analyze, sample
from jaxgsa.problem import GaussianInputSpec, Problem

UNIT_PROBLEM = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))

COEFFS = np.array([1.0, 2.0, 3.0])


def _multi_output(X: np.ndarray) -> jax.Array:
    """(N, 3) -> (N, 2): linear combination and sum of squares."""
    return jnp.stack([jnp.asarray(X @ COEFFS), jnp.asarray(np.sum(X**2, axis=1))], axis=-1)


def _numpy_reference(sr, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent NumPy implementation of the Morris measures."""
    Y_exp = np.asarray(Y, dtype=np.float64)[sr.expanded_to_unique]
    ee = (Y_exp[sr.ee_idx_after] - Y_exp[sr.ee_idx_before]) / sr.ee_delta
    return ee.mean(axis=0), np.abs(ee).mean(axis=0), ee.std(axis=0, ddof=1)


@pytest.fixture(scope="module")
def ishigami_morris_result():
    """Trajectory-design Morris result for the Ishigami benchmark."""
    sr = sample(ishigami.PROBLEM, n_trajectories=100, seed=42, verbose=False)
    Y = ishigami.evaluate(jnp.asarray(sr.samples))
    return sr, Y, analyze(sr, Y)


@pytest.fixture(scope="module")
def linear_morris_result():
    """Trajectory-design Morris result for the linear benchmark."""
    sr = sample(linear.PROBLEM, n_trajectories=20, seed=0, verbose=False)
    Y = linear.evaluate(jnp.asarray(sr.samples))
    return analyze(sr, Y)


class TestSampling:
    def test_shapes_and_bookkeeping(self):
        sr = sample(UNIT_PROBLEM, n_trajectories=10, seed=1, verbose=False)
        D = 3
        assert sr.n_expanded == 10 * (D + 1)
        assert sr.samples.shape[1] == D
        assert sr.samples.shape[0] == sr.n_runs <= sr.n_expanded
        assert sr.expanded_to_unique.shape == (sr.n_expanded,)
        for arr in (sr.ee_idx_after, sr.ee_idx_before, sr.ee_delta):
            assert arr.shape == (10, D)

    def test_dedup_roundtrip(self):
        """Unique rows indexed by the map must rebuild the expanded design."""
        sr = sample(UNIT_PROBLEM, n_trajectories=15, seed=3, verbose=False)
        expanded = sr.samples[sr.expanded_to_unique]
        # Every consecutive pair within a trajectory differs in one coordinate
        grouped = expanded.reshape(15, 4, 3)
        diffs = grouped[:, 1:, :] - grouped[:, :-1, :]
        changed = np.abs(diffs) > 1e-12
        np.testing.assert_array_equal(changed.sum(axis=2), np.ones((15, 3)))

    def test_trajectory_rows_on_grid(self):
        """All trajectory points must sit on the p-level grid."""
        p = 4
        sr = sample(UNIT_PROBLEM, n_trajectories=25, num_levels=p, seed=5, verbose=False)
        levels = sr.samples * (p - 1)
        np.testing.assert_allclose(levels, np.round(levels), atol=1e-12)

    def test_step_size_is_delta(self):
        p = 4
        delta = p / (2 * (p - 1))
        sr = sample(UNIT_PROBLEM, n_trajectories=25, num_levels=p, seed=6, verbose=False)
        expanded = sr.samples[sr.expanded_to_unique].reshape(25, 4, 3)
        steps = expanded[:, 1:, :] - expanded[:, :-1, :]
        nonzero = steps[np.abs(steps) > 1e-12]
        np.testing.assert_allclose(np.abs(nonzero), delta, atol=1e-12)

    def test_ee_bookkeeping_matches_design(self):
        """ee_delta must equal the actual coordinate change between rows."""
        for method in ("trajectory", "radial"):
            sr = sample(UNIT_PROBLEM, n_trajectories=8, method=method, seed=9, verbose=False)
            expanded = sr.samples[sr.expanded_to_unique]
            for j in range(8):
                for i in range(3):
                    actual = (
                        expanded[sr.ee_idx_after[j, i], i] - expanded[sr.ee_idx_before[j, i], i]
                    )
                    assert actual == pytest.approx(sr.ee_delta[j, i], abs=1e-12)

    def test_prefix_property_trajectory(self):
        sr_large = sample(UNIT_PROBLEM, n_trajectories=30, seed=42, verbose=False)
        sr_small = sample(UNIT_PROBLEM, n_trajectories=10, seed=42, verbose=False)
        n_rows = sr_small.n_expanded
        np.testing.assert_array_equal(
            sr_large.samples[sr_large.expanded_to_unique][:n_rows],
            sr_small.samples[sr_small.expanded_to_unique],
        )

    def test_prefix_property_radial(self):
        sr_large = sample(UNIT_PROBLEM, n_trajectories=30, method="radial", seed=42, verbose=False)
        sr_small = sample(UNIT_PROBLEM, n_trajectories=10, method="radial", seed=42, verbose=False)
        n_rows = sr_small.n_expanded
        np.testing.assert_array_equal(
            sr_large.samples[sr_large.expanded_to_unique][:n_rows],
            sr_small.samples[sr_small.expanded_to_unique],
        )

    def test_reproducible(self):
        sr1 = sample(UNIT_PROBLEM, n_trajectories=12, seed=7, verbose=False)
        sr2 = sample(UNIT_PROBLEM, n_trajectories=12, seed=7, verbose=False)
        np.testing.assert_array_equal(sr1.samples, sr2.samples)
        np.testing.assert_array_equal(sr1.ee_delta, sr2.ee_delta)

    def test_physical_units(self):
        problem = Problem(names=("a", "b"), bounds=((2.0, 5.0), (-1.0, 3.0)))
        sr = sample(problem, n_trajectories=20, seed=2, verbose=False)
        assert np.all(sr.samples[:, 0] >= 2.0 - 1e-10)
        assert np.all(sr.samples[:, 0] <= 5.0 + 1e-10)
        assert np.all(sr.samples[:, 1] >= -1.0 - 1e-10)
        assert np.all(sr.samples[:, 1] <= 3.0 + 1e-10)

    def test_verbose_reports_duplicates(self, capsys):
        sample(UNIT_PROBLEM, n_trajectories=10, seed=1)
        out = capsys.readouterr().out
        assert "duplicates_removed" in out
        assert "method=trajectory" in out

    def test_odd_num_levels_warns(self):
        with pytest.warns(UserWarning, match="odd"):
            sample(UNIT_PROBLEM, n_trajectories=5, num_levels=5, seed=1, verbose=False)

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError, match="n_trajectories"):
            sample(UNIT_PROBLEM, n_trajectories=1, verbose=False)
        with pytest.raises(ValueError, match="num_levels"):
            sample(UNIT_PROBLEM, n_trajectories=5, num_levels=1, verbose=False)
        with pytest.raises(ValueError, match="method"):
            sample(UNIT_PROBLEM, n_trajectories=5, method=cast(Any, "star"), verbose=False)

    def test_radial_tiny_delta_raises(self, monkeypatch):
        import jaxgsa.morris._sampling as morris_sampling

        class DegenerateSobol:
            def __init__(self, d, scramble, seed):
                self.d = d

            def random(self, n):
                return np.full((n, self.d), 0.5)

        monkeypatch.setattr(morris_sampling, "Sobol", DegenerateSobol)
        with pytest.raises(ValueError, match="near-zero step"):
            sample(UNIT_PROBLEM, n_trajectories=4, method="radial", verbose=False)


class TestGaussianInputs:
    PROBLEM = Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "x2": GaussianInputSpec(dist="gaussian", mean=1.0, variance=4.0),
        }
    )

    def test_samples_finite_and_truncated(self):
        from scipy.stats import norm

        sr = sample(self.PROBLEM, n_trajectories=25, seed=1, verbose=False)
        assert np.all(np.isfinite(sr.samples))
        lo, hi = norm.ppf([1e-4, 1 - 1e-4], loc=1.0, scale=2.0)
        assert np.all(sr.samples[:, 1] >= lo - 1e-9)
        assert np.all(sr.samples[:, 1] <= hi + 1e-9)
        # The uniform dimension is untouched: still on the exact p=4 grid
        levels = sr.samples[:, 0] * 3
        np.testing.assert_allclose(levels, np.round(levels), atol=1e-12)

    def test_truncation_quantile_respected(self):
        from scipy.stats import norm

        q = 0.05
        sr = sample(self.PROBLEM, n_trajectories=25, seed=1, truncation_quantile=q, verbose=False)
        lo, hi = norm.ppf([q, 1 - q], loc=1.0, scale=2.0)
        # Grid levels 0 and 1 map exactly onto the truncation quantiles
        assert sr.samples[:, 1].min() == pytest.approx(lo)
        assert sr.samples[:, 1].max() == pytest.approx(hi)

    def test_two_sided_truncated_gaussian_is_not_squashed_again(self):
        """A Gaussian with explicit low/high is already bounded — leave it alone."""
        from scipy.stats import norm

        lo, hi = norm.ppf([0.01, 0.99], loc=1.0, scale=2.0)
        problem = Problem.from_dict(
            {
                "x1": (0.0, 1.0),
                "x2": GaussianInputSpec(
                    dist="gaussian", mean=1.0, variance=4.0, low=float(lo), high=float(hi)
                ),
            }
        )
        sr = sample(problem, n_trajectories=25, seed=1, verbose=False)
        # Grid levels 0 and 1 must reach the declared bounds exactly, not a
        # range narrowed by a second truncation.
        assert sr.samples[:, 1].min() == pytest.approx(lo)
        assert sr.samples[:, 1].max() == pytest.approx(hi)

    def test_one_sided_truncation_squashes_only_the_open_side(self):
        """A declared ``low`` is honoured exactly; the open high side is pulled in."""
        from scipy.stats import norm

        lo = float(norm.ppf(0.01, loc=1.0, scale=2.0))
        q = 0.05
        problem = Problem.from_dict(
            {
                "x1": (0.0, 1.0),
                "x2": GaussianInputSpec(dist="gaussian", mean=1.0, variance=4.0, low=lo),
            }
        )
        sr = sample(problem, n_trajectories=25, seed=1, truncation_quantile=q, verbose=False)
        # scipy's truncnorm renormalises within [lo, inf), so the high grid
        # level lands at the 1-q quantile of the *truncated* marginal.
        from scipy.stats import truncnorm

        a = (lo - 1.0) / 2.0
        hi = float(truncnorm.ppf(1.0 - q, a=a, b=np.inf, loc=1.0, scale=2.0))
        assert sr.samples[:, 1].min() == pytest.approx(lo)
        assert sr.samples[:, 1].max() == pytest.approx(hi)

    def test_trajectory_delta_matches_the_squashed_step(self):
        """A model linear in the unit coordinate recovers its exact coefficients.

        The squash rescales every step on an open side, so ``ee_delta`` must be
        rescaled with it. If it is not, every Gaussian effect is off by the
        squash factor.
        """
        from scipy.stats import norm

        problem = Problem.from_dict(
            {
                "x1": (0.0, 1.0),
                "x2": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
            }
        )
        # A large q makes the bug, if reintroduced, impossible to miss.
        sr = sample(problem, n_trajectories=40, seed=1, truncation_quantile=0.1, verbose=False)
        X = np.asarray(sr.samples)
        Y = jnp.asarray(3.0 * X[:, 0] + 5.0 * norm.cdf(X[:, 1]))
        res = analyze(sr, Y)
        np.testing.assert_allclose(np.asarray(res.mu), [3.0, 5.0], rtol=1e-5)

    def test_analysis_detects_gaussian_input(self):
        sr = sample(self.PROBLEM, n_trajectories=30, seed=2, verbose=False)
        Y = jnp.asarray(sr.samples[:, 0] + 0.5 * sr.samples[:, 1])
        res = analyze(sr, Y)
        assert np.all(np.isfinite(np.asarray(res.mu_star)))
        assert np.all(np.asarray(res.mu_star) > 0)

    def test_radial_gaussian_finite(self):
        sr = sample(self.PROBLEM, n_trajectories=10, method="radial", seed=4, verbose=False)
        assert np.all(np.isfinite(sr.samples))
        Y = jnp.asarray(np.sum(sr.samples, axis=1))
        res = analyze(sr, Y)
        assert np.all(np.isfinite(np.asarray(res.mu_star)))

    def test_to_physical_units_raises(self):
        sr = sample(self.PROBLEM, n_trajectories=10, seed=3, verbose=False)
        res = analyze(sr, jnp.asarray(np.sum(sr.samples, axis=1)))
        with pytest.raises(ValueError, match="finite uniform bounds"):
            res.to_physical_units()

    def test_invalid_truncation_quantile_raises(self):
        for bad_q in (0.0, 0.5, -0.1):
            with pytest.raises(ValueError, match="truncation_quantile"):
                sample(self.PROBLEM, n_trajectories=5, truncation_quantile=bad_q, verbose=False)


class TestDownsample:
    def test_matches_direct_draw(self):
        for method in ("trajectory", "radial"):
            sr_large = sample(
                UNIT_PROBLEM, n_trajectories=30, method=method, seed=11, verbose=False
            )
            sr_direct = sample(
                UNIT_PROBLEM, n_trajectories=12, method=method, seed=11, verbose=False
            )
            sr_small = sr_large.downsample(12)
            np.testing.assert_array_equal(sr_small.samples, sr_direct.samples)
            np.testing.assert_array_equal(
                sr_small.expanded_to_unique, sr_direct.expanded_to_unique
            )
            np.testing.assert_array_equal(sr_small.ee_idx_after, sr_direct.ee_idx_after)
            np.testing.assert_array_equal(sr_small.ee_delta, sr_direct.ee_delta)

    def test_with_outputs(self):
        sr = sample(linear.PROBLEM, n_trajectories=20, seed=1, verbose=False)
        Y = np.asarray(linear.evaluate(jnp.asarray(sr.samples)))
        sr_small, Y_small = sr.downsample(8, Y)
        assert Y_small.shape[0] == sr_small.n_runs
        res_small = analyze(sr_small, jnp.asarray(Y_small))
        sr_direct = sample(linear.PROBLEM, n_trajectories=8, seed=1, verbose=False)
        res_direct = analyze(sr_direct, linear.evaluate(jnp.asarray(sr_direct.samples)))
        np.testing.assert_allclose(
            np.asarray(res_small.mu_star), np.asarray(res_direct.mu_star), atol=1e-6
        )

    def test_same_size_returns_self(self):
        sr = sample(UNIT_PROBLEM, n_trajectories=5, seed=1, verbose=False)
        assert sr.downsample(5) is sr

    def test_downsample_clears_the_earlier_block_loss(self):
        """Tier T4 (internal consistency): the caller asked for this size.

        ``n_blocks_dropped`` counts trajectories the user asked for and did
        not get. ``downsample`` gives the caller exactly the count they name,
        so an earlier loss no longer applies and must not reach ``analyze``.
        """
        sr = sample(UNIT_PROBLEM, n_trajectories=20, seed=1, verbose=False)
        sr = replace(sr, n_blocks_dropped=6)
        sr_small = sr.downsample(8)
        assert sr_small.n_trajectories == 8
        assert sr_small.n_blocks_dropped == 0

    def test_same_size_downsample_also_clears_the_block_loss(self):
        """Tier T4 (internal consistency): the answer cannot hinge on the size.

        Asking for the size you already have is still asking for a size and
        receiving it, so it clears the count for the same reason a smaller
        request does. Were it otherwise, ``downsample(r)`` and
        ``downsample(r - 1)`` would give different answers about the same
        design, and one trajectory in the argument would decide whether
        ``analyze`` warns.
        """
        sr = sample(UNIT_PROBLEM, n_trajectories=5, seed=1, verbose=False)
        sr = replace(sr, n_blocks_dropped=3)
        assert sr.downsample(5).n_blocks_dropped == 0
        assert sr.downsample(4).n_blocks_dropped == 0

    def test_downsampled_design_is_silent_in_analyze(self):
        """Tier T4 (internal consistency): no "requested" count is invented.

        With the loss carried forward, ``analyze`` reported more requested
        trajectories than the caller ever asked for.
        """
        sr = sample(linear.PROBLEM, n_trajectories=20, seed=1, verbose=False)
        sr = replace(sr, n_blocks_dropped=6)
        Y = np.asarray(linear.evaluate(jnp.asarray(sr.samples)))
        sr_small, Y_small = sr.downsample(8, Y)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            analyze(sr_small, jnp.asarray(Y_small))

    def test_invalid_raises(self):
        sr = sample(UNIT_PROBLEM, n_trajectories=5, seed=1, verbose=False)
        with pytest.raises(ValueError, match="upsample"):
            sr.downsample(6)
        with pytest.raises(ValueError, match=">= 2"):
            sr.downsample(1)
        with pytest.raises(ValueError, match="n_runs"):
            sr.downsample(3, np.ones(sr.n_runs + 1))


class TestLinearAccuracy:
    def test_mu_star_exact(self, linear_morris_result):
        """Linear f => unit-cube EE_i = c_i * (hi - lo) = c_i exactly."""
        np.testing.assert_allclose(np.asarray(linear_morris_result.mu_star), COEFFS, atol=1e-5)

    def test_mu_equals_mu_star(self, linear_morris_result):
        """Monotone increasing linear model: no sign cancellation."""
        np.testing.assert_allclose(
            np.asarray(linear_morris_result.mu),
            np.asarray(linear_morris_result.mu_star),
            atol=1e-5,
        )

    def test_sigma_zero(self, linear_morris_result):
        """Constant elementary effects: sigma vanishes."""
        np.testing.assert_allclose(np.asarray(linear_morris_result.sigma), 0.0, atol=1e-5)

    def test_to_physical_units_recovers_coefficients(self, linear_morris_result):
        res_phys = linear_morris_result.to_physical_units()
        np.testing.assert_allclose(np.asarray(res_phys.mu_star), COEFFS, atol=1e-5)
        assert res_phys.space == "physical"


class TestIshigamiAccuracy:
    def test_all_parameters_detected(self, ishigami_morris_result):
        """No Ishigami input is negligible; screening must not discard any."""
        _, _, res = ishigami_morris_result
        assert np.all(np.asarray(res.mu_star) > 1.0)

    def test_ranking_consistent_with_st(self, ishigami_morris_result):
        """mu_star ranking matches the analytical total-order ranking."""
        _, _, res = ishigami_morris_result
        mu_star = np.asarray(res.mu_star)
        st_order = np.argsort(np.asarray(ishigami.ANALYTICAL_ST))
        np.testing.assert_array_equal(np.argsort(mu_star), st_order)

    def test_x3_flagged_nonlinear(self, ishigami_morris_result):
        """x3 acts only through interaction: sigma must dominate mu_star."""
        _, _, res = ishigami_morris_result
        mu_star = np.asarray(res.mu_star)
        sigma = np.asarray(res.sigma)
        assert sigma[2] / mu_star[2] > 1.0

    def test_matches_numpy_reference(self, ishigami_morris_result):
        sr, Y, res = ishigami_morris_result
        mu_ref, mu_star_ref, sigma_ref = _numpy_reference(sr, np.asarray(Y))
        np.testing.assert_allclose(np.asarray(res.mu), mu_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(res.mu_star), mu_star_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(res.sigma), sigma_ref, rtol=1e-3, atol=1e-4)

    def test_radial_matches_numpy_reference(self):
        sr = sample(ishigami.PROBLEM, n_trajectories=50, method="radial", seed=3, verbose=False)
        Y = ishigami.evaluate(jnp.asarray(sr.samples))
        res = analyze(sr, Y)
        mu_ref, mu_star_ref, sigma_ref = _numpy_reference(sr, np.asarray(Y))
        np.testing.assert_allclose(np.asarray(res.mu), mu_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(res.mu_star), mu_star_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(res.sigma), sigma_ref, rtol=1e-3, atol=1e-4)
        assert np.all(np.asarray(res.mu_star) > 1.0)


class TestSobolGAccuracy:
    def test_ranking(self):
        """mu_star must reproduce the G-function importance ordering."""
        sr = sample(sobol_g.PROBLEM, n_trajectories=100, seed=42, verbose=False)
        Y = sobol_g.evaluate(jnp.asarray(sr.samples))
        mu_star = np.asarray(analyze(sr, Y).mu_star)
        # a = (0, 1, 4.5, 9, 99, ...): importance strictly decreasing in a
        assert mu_star[0] > mu_star[1] > mu_star[2] > mu_star[3]
        assert np.all(mu_star[3] > mu_star[4:])


class TestSALibParity:
    def test_trajectory_matches_salib(self):
        """SALib analyzing the identical expanded design must agree exactly."""
        from SALib.analyze import morris as salib_morris

        num_levels = 4
        sr = sample(
            sobol_g.PROBLEM, n_trajectories=25, num_levels=num_levels, seed=42, verbose=False
        )
        Y_unique = sobol_g.evaluate(jnp.asarray(sr.samples))
        res = analyze(sr, Y_unique)

        # Re-expand to the full design; unit-cube bounds make jaxgsa's samples
        # identical to unit-cube coordinates, sidestepping SALib's rescaling.
        X_expanded = sr.samples[sr.expanded_to_unique]
        Y_expanded = np.asarray(Y_unique)[sr.expanded_to_unique]
        problem_dict = {
            "num_vars": sr.n_params,
            "names": list(sobol_g.PROBLEM.names),
            "bounds": [[0.0, 1.0]] * sr.n_params,
        }
        Si = salib_morris.analyze(
            problem_dict,
            X_expanded,
            Y_expanded.astype(np.float64),
            num_levels=num_levels,
            num_resamples=2,
            seed=0,
        )

        np.testing.assert_allclose(np.asarray(res.mu), Si["mu"], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(res.mu_star), Si["mu_star"], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(res.sigma), Si["sigma"], rtol=1e-4, atol=1e-4)


class TestMultiOutput:
    def test_shapes(self):
        sr = sample(UNIT_PROBLEM, n_trajectories=15, seed=4, verbose=False)
        Y = _multi_output(sr.samples)
        res = analyze(sr, Y)
        assert res.mu.shape == (2, 3)
        assert res.mu_star.shape == (2, 3)
        assert res.sigma.shape == (2, 3)

    def test_slices_match_scalar_runs(self):
        sr = sample(UNIT_PROBLEM, n_trajectories=15, seed=4, verbose=False)
        Y = _multi_output(sr.samples)
        res_multi = analyze(sr, Y)
        for k in range(2):
            res_k = analyze(sr, Y[:, k])
            np.testing.assert_allclose(
                np.asarray(res_multi.mu_star)[k], np.asarray(res_k.mu_star), atol=1e-6
            )
            np.testing.assert_allclose(
                np.asarray(res_multi.sigma)[k], np.asarray(res_k.sigma), atol=1e-6
            )

    def test_time_series_shapes_and_slices(self):
        sr = sample(UNIT_PROBLEM, n_trajectories=15, seed=4, verbose=False)
        Y2 = _multi_output(sr.samples)  # (N, 2)
        Y3 = jnp.stack([Y2, 2.0 * Y2], axis=1)  # (N, T=2, K=2)
        res = analyze(sr, Y3)
        assert res.mu_star.shape == (2, 2, 3)
        res_flat = analyze(sr, Y2)
        np.testing.assert_allclose(
            np.asarray(res.mu_star)[0], np.asarray(res_flat.mu_star), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(res.mu_star)[1], 2.0 * np.asarray(res_flat.mu_star), atol=1e-5
        )


class TestBootstrap:
    def test_conf_shapes_and_bracket(self, ishigami_morris_result):
        sr, Y, _ = ishigami_morris_result
        res = analyze(sr, Y, num_resamples=200, key=jax.random.PRNGKey(0))
        for est, conf in [
            (res.mu, res.mu_conf),
            (res.mu_star, res.mu_star_conf),
            (res.sigma, res.sigma_conf),
        ]:
            assert conf is not None
            assert conf.shape == (2, 3)
            assert np.all(np.asarray(conf[0]) <= np.asarray(est) + 1e-6)
            assert np.all(np.asarray(conf[1]) >= np.asarray(est) - 1e-6)

    def test_deterministic_under_key(self, ishigami_morris_result):
        sr, Y, _ = ishigami_morris_result
        r1 = analyze(sr, Y, num_resamples=50, key=jax.random.PRNGKey(3))
        r2 = analyze(sr, Y, num_resamples=50, key=jax.random.PRNGKey(3))
        np.testing.assert_array_equal(np.asarray(r1.mu_star_conf), np.asarray(r2.mu_star_conf))

    def test_chunked_matches_unchunked(self, ishigami_morris_result):
        sr, Y, _ = ishigami_morris_result
        r_full = analyze(sr, Y, num_resamples=64, key=jax.random.PRNGKey(5))
        r_chunk = analyze(sr, Y, num_resamples=64, key=jax.random.PRNGKey(5), chunk_size=10)
        # float32 reductions vary slightly with the vmap batch size
        np.testing.assert_allclose(
            np.asarray(r_full.mu_star_conf), np.asarray(r_chunk.mu_star_conf), atol=1e-5
        )

    def test_gaussian_ci_method(self, ishigami_morris_result):
        sr, Y, _ = ishigami_morris_result
        res = analyze(sr, Y, num_resamples=100, key=jax.random.PRNGKey(1), ci_method="gaussian")
        assert res.mu_star_conf is not None
        assert np.all(np.asarray(res.mu_star_conf[1]) >= np.asarray(res.mu_star_conf[0]))

    def test_missing_key_raises(self, ishigami_morris_result):
        sr, Y, _ = ishigami_morris_result
        with pytest.raises(ValueError, match="key is required"):
            analyze(sr, Y, num_resamples=10)

    def test_invalid_ci_method_raises(self, ishigami_morris_result):
        sr, Y, _ = ishigami_morris_result
        with pytest.raises(ValueError, match="ci_method"):
            analyze(sr, Y, num_resamples=10, key=jax.random.PRNGKey(0), ci_method=cast(Any, "bad"))


class TestPhysicalUnits:
    def test_scaling_uses_ranges(self, ishigami_morris_result):
        """Ishigami ranges are 2*pi: physical measures shrink accordingly."""
        _, _, res = ishigami_morris_result
        res_phys = res.to_physical_units()
        np.testing.assert_allclose(
            np.asarray(res_phys.mu_star),
            np.asarray(res.mu_star) / (2 * np.pi),
            rtol=1e-6,
        )

    def test_conf_scaled_too(self, ishigami_morris_result):
        sr, Y, _ = ishigami_morris_result
        res = analyze(sr, Y, num_resamples=50, key=jax.random.PRNGKey(2))
        res_phys = res.to_physical_units()
        assert res_phys.mu_star_conf is not None
        np.testing.assert_allclose(
            np.asarray(res_phys.mu_star_conf),
            np.asarray(res.mu_star_conf) / (2 * np.pi),
            rtol=1e-6,
        )

    def test_double_call_raises(self, ishigami_morris_result):
        _, _, res = ishigami_morris_result
        with pytest.raises(ValueError, match="already in physical units"):
            res.to_physical_units().to_physical_units()


class TestToDataset:
    def test_scalar_output(self, ishigami_morris_result):
        _, _, res = ishigami_morris_result
        ds = res.to_dataset()
        for var in ("mu", "mu_star", "sigma"):
            assert var in ds.data_vars
        assert list(ds.coords["param"].values) == list(ishigami.PROBLEM.names)
        assert "output" not in ds.dims
        assert ds.attrs["space"] == "unit"

    def test_conf_variables(self, ishigami_morris_result):
        sr, Y, _ = ishigami_morris_result
        res = analyze(sr, Y, num_resamples=50, key=jax.random.PRNGKey(0))
        ds = res.to_dataset()
        for var in ("mu_lower", "mu_upper", "mu_star_lower", "mu_star_upper"):
            assert var in ds.data_vars

    def test_multi_output_names(self):
        problem = Problem(
            names=("x1", "x2", "x3"),
            bounds=((0, 1), (0, 1), (0, 1)),
            output_names=("temp", "pressure"),
        )
        sr = sample(problem, n_trajectories=10, seed=4, verbose=False)
        res = analyze(sr, _multi_output(sr.samples))
        ds = res.to_dataset()
        assert list(ds.coords["output"].values) == ["temp", "pressure"]
        assert ds["mu_star"].shape == (2, 3)

    def test_time_series_coords(self):
        sr = sample(UNIT_PROBLEM, n_trajectories=10, seed=4, verbose=False)
        Y2 = _multi_output(sr.samples)
        Y3 = jnp.stack([Y2, 2.0 * Y2], axis=1)
        ds = analyze(sr, Y3).to_dataset(time_coords=[0.5, 1.0])
        assert list(ds.coords["time"].values) == [0.5, 1.0]
        assert ds["mu_star"].dims == ("time", "output", "param")


class TestThinningWarning:
    """The warning must follow the cause of thinning, not the surviving count."""

    def _linear_Y(self, sr) -> jnp.ndarray:
        return jnp.asarray(linear.evaluate(jnp.asarray(sr.samples)))

    def test_small_deliberate_design_is_silent(self):
        """r = 8 is below the floor, but the user asked for it. Say nothing."""
        sr = sample(linear.PROBLEM, n_trajectories=8, seed=1, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            analyze(sr, self._linear_Y(sr))

    def test_downsample_to_a_small_design_is_silent(self):
        """Downsampling is deliberate too, so it must not trip the warning."""
        sr = sample(linear.PROBLEM, n_trajectories=20, seed=1, verbose=False)
        Y = np.asarray(linear.evaluate(jnp.asarray(sr.samples)))
        sr_small, Y_small = sr.downsample(8, Y)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            analyze(sr_small, jnp.asarray(Y_small))

    def test_lost_blocks_warn_above_the_floor(self):
        """40 of 44 remain: well above the floor, but 4 blocks were lost."""
        sr = sample(linear.PROBLEM, n_trajectories=40, seed=1, verbose=False)
        sr = replace(sr, n_blocks_dropped=4)
        with pytest.warns(UserWarning) as record:
            analyze(sr, self._linear_Y(sr))
        message = str(record[0].message)
        assert "40 of the 44 requested trajectories remain" in message
        assert "no measurable step" in message
        assert "statistically unreliable" not in message

    def test_lost_blocks_and_small_design_warn_with_the_floor(self):
        """Asked for 8, got 4: name the loss and add the reliability note."""
        sr = sample(linear.PROBLEM, n_trajectories=4, seed=1, verbose=False)
        sr = replace(sr, n_blocks_dropped=4)
        with pytest.warns(UserWarning) as record:
            analyze(sr, self._linear_Y(sr))
        message = str(record[0].message)
        assert "4 of the 8 requested trajectories remain" in message
        assert "no measurable step" in message
        assert "statistically unreliable" in message

    def test_nonfinite_loss_and_block_loss_are_reported_separately(self):
        """Each cause gets its own warning: the design's, and the policy's.

        The upstream loss is a property of the design, so ``analyze`` reports
        it. The non-finite loss belongs to the shared ``on_invalid`` policy,
        which reports it in its own message with the affected positions.
        """
        sr = sample(ishigami.PROBLEM, n_trajectories=12, method="radial", seed=8, verbose=False)
        sr = replace(sr, n_blocks_dropped=2)
        Y = np.asarray(ishigami.evaluate(jnp.asarray(sr.samples)), dtype=np.float64)
        Y[int(sr.expanded_to_unique[3 * (sr.n_params + 1)])] = np.nan
        with pytest.warns(UserWarning) as record:
            analyze(sr, jnp.asarray(Y), on_invalid="drop")
        messages = [str(w.message) for w in record]
        assert any("12 of the 14 requested trajectories remain" in m for m in messages)
        assert any("no measurable step" in m for m in messages)
        assert any("dropped 1 of 12 trajectories" in m for m in messages)


class TestValidation:
    def test_y_length_mismatch_raises(self):
        sr = sample(UNIT_PROBLEM, n_trajectories=5, seed=1, verbose=False)
        with pytest.raises(ValueError, match="sample rows"):
            analyze(sr, jnp.ones(sr.n_runs + 3))

    def test_y_wrong_ndim_raises(self):
        """4-D Y is outside the (n,), (n,K), (n,T,K) contract and must error."""
        for D in (1, 3):
            problem = Problem(names=tuple(f"x{i}" for i in range(D)), bounds=((0.0, 1.0),) * D)
            sr = sample(problem, n_trajectories=5, seed=1, verbose=False)
            with pytest.raises(ValueError, match="must be 1-D"):
                analyze(sr, jnp.ones((sr.n_runs, 2, 3, 5)))

    def test_zero_variance_warns_measures_are_zero(self):
        """Constant output warns about 0 measures (not NaN) and returns zeros."""
        sr = sample(UNIT_PROBLEM, n_trajectories=10, seed=1, verbose=False)
        with pytest.warns(UserWarning, match="zero variance.*will be 0"):
            res = analyze(sr, jnp.full(sr.n_runs, 7.0))
        np.testing.assert_array_equal(np.asarray(res.mu_star), np.zeros(3))
        assert not np.any(np.isnan(np.asarray(res.mu_star)))

    def test_nonfinite_trajectory_dropped(self):
        """Poisoning one radial trajectory must drop exactly that trajectory."""
        sr = sample(ishigami.PROBLEM, n_trajectories=12, method="radial", seed=8, verbose=False)
        Y = np.asarray(ishigami.evaluate(jnp.asarray(sr.samples)), dtype=np.float64)
        poisoned_row = int(sr.expanded_to_unique[3 * (sr.n_params + 1)])
        Y_bad = Y.copy()
        Y_bad[poisoned_row] = np.nan

        with pytest.warns(UserWarning, match="dropped 1 of 12"):
            res = analyze(sr, jnp.asarray(Y_bad), on_invalid="drop")

        # Manual reference: recompute measures from the 11 finite trajectories
        Y_exp = Y[sr.expanded_to_unique]
        ee = (Y_exp[sr.ee_idx_after] - Y_exp[sr.ee_idx_before]) / sr.ee_delta
        keep = np.arange(12) != 3
        np.testing.assert_allclose(
            np.asarray(res.mu_star), np.abs(ee[keep]).mean(axis=0), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            np.asarray(res.sigma), ee[keep].std(axis=0, ddof=1), rtol=1e-3, atol=1e-4
        )

    def test_all_nonfinite_raises(self):
        sr = sample(UNIT_PROBLEM, n_trajectories=5, seed=1, verbose=False)
        Y = jnp.full(sr.n_runs, jnp.nan)
        with pytest.raises(ValueError, match="every usable trajectory was removed"):
            analyze(sr, Y, on_invalid="drop")

    def test_prenormalize_scales_measures(self):
        sr = sample(linear.PROBLEM, n_trajectories=20, seed=0, verbose=False)
        Y = np.asarray(linear.evaluate(jnp.asarray(sr.samples)))
        Yj = jnp.asarray(Y)
        res_raw = analyze(sr, Yj)
        res_norm = analyze(sr, Yj, prenormalize=True)
        Y_exp = Y[sr.expanded_to_unique]
        scale = Y_exp.std()
        np.testing.assert_allclose(
            np.asarray(res_norm.mu_star),
            np.asarray(res_raw.mu_star) / scale,
            rtol=1e-4,
        )


class TestOnInvalidPolicy:
    """T4 (behavioural): morris.analyze applies the shared non-finite policy."""

    R = 12

    def _design_and_Y(self):
        """A radial Ishigami design plus its finite outputs on the unique rows."""
        sr = sample(
            ishigami.PROBLEM, n_trajectories=self.R, method="radial", seed=8, verbose=False
        )
        Y = np.asarray(ishigami.evaluate(jnp.asarray(sr.samples)), dtype=np.float64)
        return sr, Y

    def _poison(self, sr, Y: np.ndarray, trajectory: int) -> tuple[np.ndarray, list[int]]:
        """Set one NaN inside one trajectory, and return its expanded rows."""
        rows_per_traj = sr.n_params + 1
        rows = list(range(trajectory * rows_per_traj, (trajectory + 1) * rows_per_traj))
        poisoned = Y.copy()
        poisoned[int(sr.expanded_to_unique[rows[0]])] = np.nan
        return poisoned, rows

    def test_raise_is_the_default_and_names_the_trajectory_and_its_rows(self):
        """T4: the default refuses, and locates the failure in the design."""
        sr, Y = self._design_and_Y()
        bad, rows = self._poison(sr, Y, trajectory=3)
        with pytest.raises(ValueError, match=f"1 of {self.R} trajectories") as exc:
            analyze(sr, jnp.asarray(bad))
        message = str(exc.value)
        assert "jaxgsa.morris.analyze" in message
        assert "Affected trajectories: [3]" in message
        assert f"Rows: {rows}" in message

    def test_propagate_warns_and_lets_the_value_reach_the_measures(self):
        """T4: nothing is removed, so the screening measures come back NaN."""
        sr, Y = self._design_and_Y()
        bad, _ = self._poison(sr, Y, trajectory=3)
        with pytest.warns(JaxgsaWarning, match="reaches the indices"):
            res = analyze(sr, jnp.asarray(bad), on_invalid="propagate")
        assert not np.all(np.isfinite(np.asarray(res.mu_star)))
        assert res.invalid.n_invalid == 1

    def test_one_bad_value_condemns_its_whole_trajectory(self):
        """T4: the unit is the trajectory, not the row.

        A trajectory is a chain of D+1 points, and each elementary effect is a
        difference between two neighbours in that chain. Removing one row
        would leave the neighbours adjacent and invent an effect that spans a
        step nobody took, so the whole block of D+1 rows has to go.
        """
        sr, Y = self._design_and_Y()
        bad, rows = self._poison(sr, Y, trajectory=3)
        with pytest.warns(JaxgsaWarning):
            res = analyze(sr, jnp.asarray(bad), on_invalid="drop")
        assert res.invalid.unit_indices == (3,)
        assert res.invalid.row_indices == tuple(rows)
        assert len(rows) == sr.n_params + 1

    @pytest.mark.parametrize("trajectory", [0, 5, 11])
    def test_drop_reindexes_the_elementary_effects(self, trajectory):
        """T4: dropping trajectory t equals never having generated it.

        This is the part the shared helper cannot do. ``ee_idx_after`` and
        ``ee_idx_before`` hold *global* expanded-row indices, so compacting
        the output array shifts every later block and leaves those indices
        pointing at the wrong rows. The reference gathers the effects with the
        original, uncompacted indices and then removes the trajectory, so a
        re-indexing mistake shows up as a wrong measure rather than as an
        error. Trajectory 0 is included because its block offset is zero,
        which is the one case a missing shift would still get right.
        """
        sr, Y = self._design_and_Y()
        bad, _ = self._poison(sr, Y, trajectory=trajectory)
        with pytest.warns(JaxgsaWarning):
            res = analyze(sr, jnp.asarray(bad), on_invalid="drop")

        Y_exp = Y[sr.expanded_to_unique]
        ee = (Y_exp[sr.ee_idx_after] - Y_exp[sr.ee_idx_before]) / sr.ee_delta
        keep = np.arange(self.R) != trajectory
        np.testing.assert_allclose(np.asarray(res.mu), ee[keep].mean(axis=0), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            np.asarray(res.mu_star), np.abs(ee[keep]).mean(axis=0), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            np.asarray(res.sigma), ee[keep].std(axis=0, ddof=1), rtol=1e-3, atol=1e-4
        )

    def test_dropping_several_trajectories_reindexes_all_survivors(self):
        """T4: the re-indexing holds when the survivors are not contiguous."""
        sr, Y = self._design_and_Y()
        bad = Y.copy()
        for trajectory in (1, 4, 9):
            bad, _ = self._poison(sr, bad, trajectory=trajectory)
        with pytest.warns(JaxgsaWarning) as record:
            res = analyze(sr, jnp.asarray(bad), on_invalid="drop")
        assert any("dropped 3 of 12 trajectories" in str(w.message) for w in record)

        Y_exp = Y[sr.expanded_to_unique]
        ee = (Y_exp[sr.ee_idx_after] - Y_exp[sr.ee_idx_before]) / sr.ee_delta
        keep = np.isin(np.arange(self.R), (1, 4, 9), invert=True)
        np.testing.assert_allclose(
            np.asarray(res.mu_star), np.abs(ee[keep]).mean(axis=0), rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("policy", ["raise", "propagate", "drop"])
    def test_a_clean_sample_reports_nothing_and_stays_silent(self, policy, recwarn):
        """T4: a clean run gives an empty report under every policy."""
        sr, Y = self._design_and_Y()
        res = analyze(sr, jnp.asarray(Y), on_invalid=policy)
        assert res.invalid.n_invalid == 0
        assert res.invalid.unit is InvalidUnit.TRAJECTORY
        assert res.invalid.n_units == self.R
        assert [w for w in recwarn if issubclass(w.category, JaxgsaWarning)] == []

    def test_a_bad_on_invalid_value_is_rejected(self):
        """T4: an unknown policy name is refused before anything is computed."""
        sr, Y = self._design_and_Y()
        with pytest.raises(ValueError, match="on_invalid must be one of"):
            analyze(sr, jnp.asarray(Y), on_invalid="skip")


def test_single_param():
    problem = Problem(names=("x",), bounds=((0.0, 2.0),))
    sr = sample(problem, n_trajectories=10, seed=1, verbose=False)
    Y = jnp.sin(jnp.asarray(sr.samples[:, 0]))
    res = analyze(sr, Y)
    assert res.mu.shape == (1,)
    assert res.mu_star.shape == (1,)
    assert res.sigma.shape == (1,)
    assert res.mu_star[0] > 0
