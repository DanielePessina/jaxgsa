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
from jaxgsa.morris import MorrisIndices, analyze, indices, sample
from jaxgsa.problem import GaussianInputSpec, Problem

UNIT_PROBLEM = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))

COEFFS = np.array([1.0, 2.0, 3.0])


def _multi_output(X: np.ndarray) -> jax.Array:
    """(N, 3) -> (N, 2): linear combination and sum of squares."""
    return jnp.stack([jnp.asarray(X @ COEFFS), jnp.asarray(np.sum(X**2, axis=1))], axis=-1)


@pytest.fixture(scope="module")
def ishigami_morris_result():
    """Trajectory-design Morris result for the Ishigami benchmark."""
    sr = sample(ishigami.PROBLEM, n_trajectories=100, seed=42, verbose=False)
    Y = ishigami.evaluate(jnp.asarray(sr.samples))
    return sr, Y, analyze(sr, Y)


@pytest.fixture(scope="module")
def linear_morris_result_full():
    """Trajectory-design Morris design, outputs and result for the linear benchmark."""
    sr = sample(linear.PROBLEM, n_trajectories=20, seed=0, verbose=False)
    Y = linear.evaluate(jnp.asarray(sr.samples))
    return sr, Y, analyze(sr, Y)


@pytest.fixture(scope="module")
def linear_morris_result(linear_morris_result_full):
    """Trajectory-design Morris result for the linear benchmark."""
    return linear_morris_result_full[2]


@pytest.fixture(scope="module")
def radial_ishigami_r12():
    """A radial Ishigami design with 12 trajectories, and its finite outputs."""
    sr = sample(ishigami.PROBLEM, n_trajectories=12, method="radial", seed=8, verbose=False)
    Y = np.asarray(ishigami.evaluate(jnp.asarray(sr.samples)), dtype=np.float64)
    return sr, Y


class TestSampling:
    @pytest.mark.parametrize("method", ["trajectory", "radial"])
    def test_design_structure(self, method):
        """Shapes, grid, step size and the ee bookkeeping agree with the design."""
        p, n, D = 4, 25, 3
        sr = sample(
            UNIT_PROBLEM, n_trajectories=n, n_levels=p, method=method, seed=5, verbose=False
        )
        assert sr.n_expanded == n * (D + 1)
        assert sr.samples.shape[1] == D
        assert sr.samples.shape[0] == sr.n_runs <= sr.n_expanded
        assert sr.expanded_to_unique.shape == (sr.n_expanded,)
        for arr in (sr.ee_idx_after, sr.ee_idx_before, sr.ee_delta):
            assert arr.shape == (n, D)

        expanded = sr.samples[sr.expanded_to_unique]
        if method == "trajectory":
            # All points sit on the p-level grid.
            levels = sr.samples * (p - 1)
            np.testing.assert_allclose(levels, np.round(levels), atol=1e-12)
            # Every consecutive pair within a trajectory differs in one
            # coordinate, and the step is delta.
            grouped = expanded.reshape(n, D + 1, D)
            steps = grouped[:, 1:, :] - grouped[:, :-1, :]
            changed = np.abs(steps) > 1e-12
            np.testing.assert_array_equal(changed.sum(axis=2), np.ones((n, D)))
            delta = p / (2 * (p - 1))
            np.testing.assert_allclose(np.abs(steps[changed]), delta, atol=1e-12)

        # ee_delta equals the actual coordinate change between the paired rows.
        for j in range(n):
            for i in range(D):
                actual = expanded[sr.ee_idx_after[j, i], i] - expanded[sr.ee_idx_before[j, i], i]
                assert actual == pytest.approx(sr.ee_delta[j, i], abs=1e-12)

    @pytest.mark.parametrize("method", ["trajectory", "radial"])
    def test_prefix_property(self, method):
        """A smaller design is the prefix of a larger one with the same seed."""
        sr_large = sample(UNIT_PROBLEM, n_trajectories=30, method=method, seed=42, verbose=False)
        sr_small = sample(UNIT_PROBLEM, n_trajectories=10, method=method, seed=42, verbose=False)
        n_rows = sr_small.n_expanded
        np.testing.assert_array_equal(
            sr_large.samples[sr_large.expanded_to_unique][:n_rows],
            sr_small.samples[sr_small.expanded_to_unique],
        )

    def test_physical_units(self):
        problem = Problem(names=("a", "b"), bounds=((2.0, 5.0), (-1.0, 3.0)))
        sr = sample(problem, n_trajectories=20, seed=2, verbose=False)
        assert np.all(sr.samples[:, 0] >= 2.0 - 1e-10)
        assert np.all(sr.samples[:, 0] <= 5.0 + 1e-10)
        assert np.all(sr.samples[:, 1] >= -1.0 - 1e-10)
        assert np.all(sr.samples[:, 1] <= 3.0 + 1e-10)

    def test_odd_n_levels_warns(self):
        with pytest.warns(UserWarning, match="odd"):
            sample(UNIT_PROBLEM, n_trajectories=5, n_levels=5, seed=1, verbose=False)

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError, match="n_trajectories"):
            sample(UNIT_PROBLEM, n_trajectories=1, verbose=False)
        with pytest.raises(ValueError, match="n_levels"):
            sample(UNIT_PROBLEM, n_trajectories=5, n_levels=1, verbose=False)
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
        sr = sample(self.PROBLEM, n_trajectories=25, seed=1, verbose=False)
        gaussian = sr.samples[:, 1]
        assert np.all(np.isfinite(sr.samples))
        assert gaussian.min() < 1.0 < gaussian.max()
        assert gaussian.min() + gaussian.max() == pytest.approx(2.0)
        # The uniform dimension is untouched: still on the exact p=4 grid
        levels = sr.samples[:, 0] * 3
        np.testing.assert_allclose(levels, np.round(levels), atol=1e-12)

    def test_truncation_quantile_respected(self):
        q = 0.05
        sr = sample(self.PROBLEM, n_trajectories=25, seed=1, truncation_quantile=q, verbose=False)
        default = sample(self.PROBLEM, n_trajectories=25, seed=1, verbose=False)
        values = sr.samples[:, 1]
        default_values = default.samples[:, 1]
        # A larger removed tail pulls both open sides inward.
        assert values.min() > default_values.min()
        assert values.max() < default_values.max()
        assert values.min() + values.max() == pytest.approx(2.0)

    def test_two_sided_truncated_gaussian_is_not_squashed_again(self):
        """A Gaussian with explicit low/high is already bounded — leave it alone."""
        lo, hi = -3.0, 5.0
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
        lo = -1.0
        q = 0.05
        problem = Problem.from_dict(
            {
                "x1": (0.0, 1.0),
                "x2": GaussianInputSpec(dist="gaussian", mean=1.0, variance=4.0, low=lo),
            }
        )
        sr = sample(problem, n_trajectories=25, seed=1, truncation_quantile=q, verbose=False)
        looser = sample(
            problem, n_trajectories=25, seed=1, truncation_quantile=0.01, verbose=False
        )
        values = sr.samples[:, 1]
        looser_values = looser.samples[:, 1]
        assert values.min() == pytest.approx(lo)
        assert values.max() > 1.0
        assert values.max() < looser_values.max()

    def test_trajectory_delta_matches_the_squashed_step(self):
        """A model linear in the unit coordinate recovers its exact coefficients.

        The squash rescales every step on an open side, so ``ee_delta`` must be
        rescaled with it. If it is not, every Gaussian effect is off by the
        squash factor.
        """
        problem = Problem.from_dict(
            {
                "x1": (0.0, 1.0),
                "x2": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
            }
        )
        # A large q makes the bug, if reintroduced, impossible to miss.
        sr = sample(problem, n_trajectories=40, seed=1, truncation_quantile=0.1, verbose=False)
        X = np.asarray(sr.samples)
        Xj = jnp.asarray(X)
        Y = 3.0 * Xj[:, 0] + 5.0 * jax.scipy.special.ndtr(Xj[:, 1])
        res = analyze(sr, Y)
        np.testing.assert_allclose(np.asarray(res.mu), [3.0, 5.0], rtol=1e-5)

    def test_to_physical_units_needs_finite_bounds(self):
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
    def test_linear_model_is_recovered_exactly(self, linear_morris_result):
        """Linear f => unit-cube EE_i = c_i * (hi - lo) = c_i exactly.

        Monotone increasing, so ``mu`` equals ``mu_star`` (no sign
        cancellation). The effects are constant, so ``sigma`` vanishes. In
        physical units the coefficients come back.
        """
        res = linear_morris_result
        np.testing.assert_allclose(np.asarray(res.mu_star), COEFFS, atol=1e-5)
        np.testing.assert_allclose(np.asarray(res.mu), np.asarray(res.mu_star), atol=1e-5)
        np.testing.assert_allclose(np.asarray(res.sigma), 0.0, atol=1e-5)
        res_phys = res.to_physical_units()
        np.testing.assert_allclose(np.asarray(res_phys.mu_star), COEFFS, atol=1e-5)
        assert res_phys.space == "physical"


class TestIshigamiAccuracy:
    def test_screening_matches_the_known_structure(self, ishigami_morris_result):
        """No input is negligible, the ranking follows ST, and x3 reads as nonlinear.

        x3 acts only through interaction, so sigma must dominate mu_star.
        """
        _, _, res = ishigami_morris_result
        mu_star = np.asarray(res.mu_star)
        sigma = np.asarray(res.sigma)
        assert np.all(mu_star > 1.0)
        np.testing.assert_array_equal(
            np.argsort(mu_star), np.argsort(np.asarray(ishigami.ANALYTICAL_ST))
        )
        assert sigma[2] / mu_star[2] > 1.0


class TestSobolGAccuracy:
    def test_ranking(self):
        """mu_star must reproduce the G-function importance ordering."""
        sr = sample(sobol_g.PROBLEM, n_trajectories=100, seed=42, verbose=False)
        Y = sobol_g.evaluate(jnp.asarray(sr.samples))
        mu_star = np.asarray(analyze(sr, Y).mu_star)
        # a = (0, 1, 4.5, 9, 99, ...): importance strictly decreasing in a
        assert mu_star[0] > mu_star[1] > mu_star[2] > mu_star[3]
        assert np.all(mu_star[3] > mu_star[4:])


class TestMultiOutput:
    def test_slices_dataset_and_core_agree_on_multi_output(self):
        """Slices match scalar runs, the dataset is labelled, and ``indices`` agrees."""
        sr = sample(UNIT_PROBLEM, n_trajectories=15, seed=4, verbose=False)
        Y = _multi_output(sr.samples)  # (N, 2)
        res_multi = analyze(sr, Y)
        for k in range(2):
            res_k = analyze(sr, Y[:, k])
            np.testing.assert_allclose(
                np.asarray(res_multi.mu_star)[k], np.asarray(res_k.mu_star), atol=1e-6
            )
            np.testing.assert_allclose(
                np.asarray(res_multi.sigma)[k], np.asarray(res_k.sigma), atol=1e-6
            )

        # A time axis: (N, T=2, K=2), the second step twice the first.
        Y3 = jnp.stack([Y, 2.0 * Y], axis=1)
        res = analyze(sr, Y3)
        assert res.mu_star.shape == (2, 2, 3)
        np.testing.assert_allclose(
            np.asarray(res.mu_star)[0], np.asarray(res_multi.mu_star), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(res.mu_star)[1], 2.0 * np.asarray(res_multi.mu_star), atol=1e-5
        )

        # The pure core returns the same numbers at the caller's rank.
        mu, mu_star, sigma = indices(sr, Y3)
        assert mu.shape == (2, 2, sr.n_params)
        np.testing.assert_array_equal(np.asarray(mu), np.asarray(res.mu))
        np.testing.assert_array_equal(np.asarray(mu_star), np.asarray(res.mu_star))
        np.testing.assert_array_equal(np.asarray(sigma), np.asarray(res.sigma))

        # Names label the dataset axes.
        problem = Problem(
            names=("x1", "x2", "x3"),
            bounds=((0, 1), (0, 1), (0, 1)),
            output_names=("temp", "pressure"),
        )
        sr_named = sample(problem, n_trajectories=15, seed=4, verbose=False)
        ds = analyze(sr_named, Y).to_dataset()
        assert list(ds.coords["param"].values) == list(problem.names)
        assert list(ds.coords["output"].values) == ["temp", "pressure"]
        assert ds["mu_star"].shape == (2, 3)
        ds3 = analyze(sr_named, Y3).to_dataset(time_coords=[0.5, 1.0])
        assert list(ds3.coords["time"].values) == [0.5, 1.0]
        assert ds3["mu_star"].dims == ("time", "output", "param")


class TestBootstrap:
    def test_intervals_bracket_repeat_chunk_and_scale(self, ishigami_morris_result):
        """One R=32 run: bracket, key-determinism, chunking, gaussian rule, physical units."""
        sr, Y, _ = ishigami_morris_result
        key = jax.random.PRNGKey(0)
        res = analyze(sr, Y, n_bootstrap=32, key=key)
        for est, conf in [
            (res.mu, res.mu_conf),
            (res.mu_star, res.mu_star_conf),
            (res.sigma, res.sigma_conf),
        ]:
            assert conf is not None
            assert conf.shape == (2, 3)
            assert np.all(np.asarray(conf[0]) <= np.asarray(est) + 1e-6)
            assert np.all(np.asarray(conf[1]) >= np.asarray(est) - 1e-6)

        again = analyze(sr, Y, n_bootstrap=32, key=key)
        np.testing.assert_array_equal(np.asarray(res.mu_star_conf), np.asarray(again.mu_star_conf))

        chunked = analyze(sr, Y, n_bootstrap=32, key=key, resample_chunk_size=10)
        # float32 reductions vary slightly with the vmap batch size
        np.testing.assert_allclose(
            np.asarray(res.mu_star_conf), np.asarray(chunked.mu_star_conf), atol=1e-5
        )

        gaussian = analyze(sr, Y, n_bootstrap=32, key=key, ci_method="gaussian")
        assert gaussian.mu_star_conf is not None
        assert np.all(np.asarray(gaussian.mu_star_conf[1]) >= np.asarray(gaussian.mu_star_conf[0]))

        # The interval scales with the point estimate: Ishigami ranges are 2*pi.
        res_phys = res.to_physical_units()
        assert res_phys.mu_star_conf is not None
        np.testing.assert_allclose(
            np.asarray(res_phys.mu_star_conf),
            np.asarray(res.mu_star_conf) / (2 * np.pi),
            rtol=1e-6,
        )

    def test_bad_arguments_raise(self, ishigami_morris_result):
        sr, Y, _ = ishigami_morris_result
        with pytest.raises(ValueError, match="key is required"):
            analyze(sr, Y, n_bootstrap=10)
        with pytest.raises(ValueError, match="ci_method"):
            analyze(sr, Y, n_bootstrap=10, key=jax.random.PRNGKey(0), ci_method=cast(Any, "bad"))


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

    def test_double_call_returns_self(self, ishigami_morris_result):
        _, _, res = ishigami_morris_result
        phys = res.to_physical_units()
        assert phys.to_physical_units() is phys


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
        # ``np.array`` rather than ``np.asarray``: a JAX array converts to a
        # read-only NumPy view, and only a dtype change forces the copy that
        # makes it writable. Asking for float64 was that copy under the
        # float32 default and a no-op under x64, so the write failed there.
        # Keep the model's own dtype and copy on purpose instead.
        Y = np.array(ishigami.evaluate(jnp.asarray(sr.samples)))
        Y[int(sr.expanded_to_unique[3 * (sr.n_params + 1)])] = np.nan
        with pytest.warns(UserWarning) as record:
            analyze(sr, jnp.asarray(Y), on_invalid="drop")
        messages = [str(w.message) for w in record]
        assert any("12 of the 14 requested trajectories remain" in m for m in messages)
        assert any("no measurable step" in m for m in messages)
        assert any("dropped 1 of 12 trajectories" in m for m in messages)


class TestValidation:
    def test_zero_variance_warns_measures_are_zero(self):
        """Constant output warns about 0 measures (not NaN) and returns zeros."""
        sr = sample(UNIT_PROBLEM, n_trajectories=10, seed=1, verbose=False)
        with pytest.warns(UserWarning, match="zero variance.*will be 0"):
            res = analyze(sr, jnp.full(sr.n_runs, 7.0))
        np.testing.assert_array_equal(np.asarray(res.mu_star), np.zeros(3))
        assert not np.any(np.isnan(np.asarray(res.mu_star)))

    def test_all_nonfinite_raises(self):
        sr = sample(UNIT_PROBLEM, n_trajectories=5, seed=1, verbose=False)
        Y = jnp.full(sr.n_runs, jnp.nan)
        with pytest.raises(ValueError, match="every usable trajectory was removed"):
            analyze(sr, Y, on_invalid="drop")

    def test_standardize_outputs_and_affine_changes(self, linear_morris_result_full):
        """Output scaling: raw measures scale with ``a``, standardized ones do not move.

        Tier T0 (closed form): an elementary effect is a difference quotient,
        so under ``Y -> a*Y + b`` the shift cancels and the scale passes
        straight through. Morris reports dimensional quantities and does not
        pretend otherwise. With ``standardize_outputs=True`` the measures
        are divided by the output standard deviation, which scales by ``a``
        too, so the same affine change moves nothing.
        """
        sr, Y, base = linear_morris_result_full
        a, b = 3.0, 1e3

        res_norm = analyze(sr, Y, standardize_outputs=True)
        scale = np.asarray(Y)[sr.expanded_to_unique].std()
        np.testing.assert_allclose(
            np.asarray(res_norm.mu_star), np.asarray(base.mu_star) / scale, rtol=1e-4
        )

        affine = analyze(sr, a * Y + b)
        for name in ("mu", "mu_star", "sigma"):
            np.testing.assert_allclose(
                np.asarray(getattr(affine, name)),
                a * np.asarray(getattr(base, name)),
                rtol=1e-4,
                atol=1e-4,
            )

        affine_norm = analyze(sr, a * Y + b, standardize_outputs=True)
        for name in ("mu", "mu_star", "sigma"):
            np.testing.assert_allclose(
                np.asarray(getattr(affine_norm, name)),
                np.asarray(getattr(res_norm, name)),
                rtol=1e-3,
                atol=1e-4,
            )


class TestOnInvalidPolicy:
    """T4 (behavioural): morris.analyze applies the shared non-finite policy."""

    R = 12

    @pytest.fixture(autouse=True)
    def _design(self, radial_ishigami_r12):
        self.sr, self.Y = radial_ishigami_r12

    def _design_and_Y(self):
        """A radial Ishigami design plus its finite outputs on the unique rows."""
        return self.sr, self.Y

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
        assert "They condemn trajectories [3]" in message
        assert f"which covers {len(rows)} rows" in message

    def test_propagate_warns_and_lets_the_value_reach_the_measures(self):
        """T4: nothing is removed, so the screening measures come back NaN."""
        sr, Y = self._design_and_Y()
        bad, _ = self._poison(sr, Y, trajectory=3)
        with pytest.warns(JaxgsaWarning, match="reaches the indices"):
            res = analyze(sr, jnp.asarray(bad), on_invalid="propagate")
        assert not np.all(np.isfinite(np.asarray(res.mu_star)))
        assert res.invalid.n_invalid == 1

    def test_drop_condemns_whole_trajectories_and_reindexes_the_effects(self):
        """T4: dropping trajectories equals never having generated them.

        The unit is the trajectory, not the row. A trajectory is a chain of
        D+1 points, and each elementary effect is a difference between two
        neighbours in that chain. Removing one row would leave the neighbours
        adjacent and invent an effect that spans a step nobody took, so the
        whole block of D+1 rows has to go.

        The re-indexing is the part the shared helper cannot do.
        ``ee_idx_after`` and ``ee_idx_before`` hold *global* expanded-row
        indices, so compacting the output array shifts every later block and
        leaves those indices pointing at the wrong rows. The reference
        gathers the effects with the original, uncompacted indices and then
        removes the trajectories, so a re-indexing mistake shows up as a
        wrong measure rather than as an error. Trajectory 0 is included
        because its block offset is zero, which is the one case a missing
        shift would still get right. The survivors are not contiguous.
        """
        sr, Y = self._design_and_Y()
        dropped = (0, 5, 11)
        bad = Y.copy()
        rows: list[int] = []
        for trajectory in dropped:
            bad, traj_rows = self._poison(sr, bad, trajectory=trajectory)
            rows.extend(traj_rows)
        with pytest.warns(JaxgsaWarning) as record:
            res = analyze(sr, jnp.asarray(bad), on_invalid="drop")
        assert any("dropped 3 of 12 trajectories" in str(w.message) for w in record)
        assert res.invalid.unit_indices == dropped
        assert res.invalid.row_indices == tuple(rows)
        assert len(rows) == len(dropped) * (sr.n_params + 1)

        Y_exp = Y[sr.expanded_to_unique]
        ee = (Y_exp[sr.ee_idx_after] - Y_exp[sr.ee_idx_before]) / sr.ee_delta
        keep = np.isin(np.arange(self.R), dropped, invert=True)
        np.testing.assert_allclose(np.asarray(res.mu), ee[keep].mean(axis=0), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            np.asarray(res.mu_star), np.abs(ee[keep]).mean(axis=0), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            np.asarray(res.sigma), ee[keep].std(axis=0, ddof=1), rtol=1e-3, atol=1e-4
        )

    def test_a_clean_sample_reports_nothing_and_stays_silent(self, recwarn):
        """T4: a clean run gives an empty report under every policy."""
        sr, Y = self._design_and_Y()
        for policy in ("raise", "propagate", "drop"):
            res = analyze(sr, jnp.asarray(Y), on_invalid=policy)
            assert res.invalid.n_invalid == 0
            assert res.invalid.unit is InvalidUnit.TRAJECTORY
            assert res.invalid.n_units == self.R
        assert [w for w in recwarn if issubclass(w.category, JaxgsaWarning)] == []


def test_single_param():
    problem = Problem(names=("x",), bounds=((0.0, 2.0),))
    sr = sample(problem, n_trajectories=10, seed=1, verbose=False)
    Y = jnp.sin(jnp.asarray(sr.samples[:, 0]))
    res = analyze(sr, Y)
    assert res.mu.shape == (1,)
    assert res.mu_star.shape == (1,)
    assert res.sigma.shape == (1,)
    assert res.mu_star[0] > 0


# ---------------------------------------------------------------------------
# indices(): the pure, transformable core
# ---------------------------------------------------------------------------


class TestIndicesCore:
    """The traceable core against the policy-carrying ``analyze``.

    Tier T4 throughout (behavioural contract). ``analyze`` is not an external
    oracle for the elementary-effect arithmetic -- the accuracy of that is
    checked against closed forms elsewhere in this file. What is checked here
    is the split itself: the core must return exactly the numbers the full
    entry point reports, and it must survive the three transformations
    ``analyze`` cannot.
    """

    @staticmethod
    def _design_and_Y(n_trajectories: int = 20, seed: int = 7):
        """A trajectory design on the Ishigami problem, plus its outputs."""
        sr = sample(ishigami.PROBLEM, n_trajectories=n_trajectories, seed=seed, verbose=False)
        return sr, ishigami.evaluate(jnp.asarray(sr.samples))

    def test_matches_analyze_and_returns_point_estimates_only(self):
        """T4: the three measures equal ``analyze``'s fields exactly, with no interval slot.

        The reporting-units flag agrees with ``analyze`` as well. It is kept
        on the core because it reduces over the *expanded* sample axis, which
        the caller never holds, so there is no equivalent pre-pass on the
        unique rows.

        The bootstrap is policy: which interval to report, at what level, by
        which endpoint rule. It stays on ``analyze``, which keeps
        ``n_bootstrap`` and reports intervals as before.
        """
        sr, Y = self._design_and_Y()
        out = indices(sr, Y)
        result = analyze(sr, Y)

        assert len(out) == 3
        assert all(arr.shape == (sr.n_params,) for arr in out)
        for got, field in zip(out, ("mu", "mu_star", "sigma"), strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(getattr(result, field)))

        std_out = indices(sr, Y, standardize_outputs=True)
        std_result = analyze(sr, Y, standardize_outputs=True)
        for got, field in zip(std_out, ("mu", "mu_star", "sigma"), strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(getattr(std_result, field)))

        # analyze still offers the interval the core does not.
        boot = analyze(sr, Y, n_bootstrap=32, key=jax.random.key(0))
        assert boot.mu_conf is not None
        assert boot.mu_conf.shape == (2, sr.n_params)

    def test_zero_dimensional_Y_is_refused(self):
        """T4: the rank contract of ``analyze`` applies to ``indices`` too.

        A 0-D ``Y`` used to die with an IndexError inside the expansion;
        it now raises the same ValueError ``analyze`` raises.
        """
        sr, _ = self._design_and_Y()
        with pytest.raises(ValueError, match=r"Y must be 1-D \(N,\), 2-D \(N, K\), or 3-D"):
            indices(sr, jnp.asarray(1.0))

    def test_is_traceable_under_jit_vmap_and_eval_shape(self):
        """T4: ``jit``, ``vmap`` and an abstract trace all work on the core.

        The design is a closure constant: it holds NumPy index maps, and
        ``indices`` never reads a value out of ``Y`` on the host. A host read
        would raise ``TracerArrayConversionError`` or
        ``TracerBoolConversionError`` under an abstract trace, so tracing with
        ``eval_shape`` is the check that every array operation is traceable.
        """
        sr, Y = self._design_and_Y()
        jitted = jax.jit(lambda outputs: indices(sr, outputs))
        for traced, eager in zip(jitted(Y), indices(sr, Y), strict=True):
            np.testing.assert_allclose(np.asarray(traced), np.asarray(eager), rtol=1e-6)

        X = jnp.asarray(sr.samples)
        batch = jnp.stack([ishigami.evaluate(X), X[:, 0], X[:, 1] ** 2])
        mu_batch, mu_star_batch, sigma_batch = jax.vmap(lambda out: indices(sr, out))(batch)
        assert mu_batch.shape == (3, sr.n_params)
        for row, outputs in enumerate(batch):
            mu, mu_star, sigma = indices(sr, outputs)
            np.testing.assert_allclose(np.asarray(mu_batch[row]), np.asarray(mu), rtol=1e-5)
            np.testing.assert_allclose(
                np.asarray(mu_star_batch[row]), np.asarray(mu_star), rtol=1e-5
            )
            np.testing.assert_allclose(np.asarray(sigma_batch[row]), np.asarray(sigma), rtol=1e-5)

        shapes = jax.eval_shape(lambda outputs: indices(sr, outputs), Y)
        assert [s.shape for s in shapes] == [(sr.n_params,)] * 3

    def test_jit_of_jacrev_of_the_full_chain(self):
        """T4: ``jit(jacrev(...))`` compiles a model-parameter-to-measure chain.

        ``mu_star`` is differentiable in the model parameter even though it
        holds an absolute value, because the effects are away from zero here.
        Finite differences are the reference, so this runs in float64.
        """
        with jax.enable_x64():
            sr = sample(UNIT_PROBLEM, n_trajectories=8, seed=2, verbose=False)
            X = jnp.asarray(sr.samples, dtype=jnp.float64)

            def total_mu_star(a):
                """Sum of mu_star for the model ``y = a * x1 + x2^2 + x3``."""
                Y = a * X[:, 0] + X[:, 1] ** 2 + X[:, 2]
                return indices(sr, Y)[1].sum()

            grad = jax.jit(jax.jacrev(total_mu_star))(1.5)
            eager = jax.jacrev(total_mu_star)(1.5)

            step = 1e-6
            fd = (total_mu_star(1.5 + step) - total_mu_star(1.5 - step)) / (2 * step)

        assert np.isfinite(float(grad))
        np.testing.assert_allclose(float(grad), float(eager), rtol=1e-10)
        np.testing.assert_allclose(float(grad), float(fd), rtol=1e-6)


class TestInterfaceCleanup:
    def test_first_argument_must_be_morris_samples(self):
        for fn in (analyze, indices):
            with pytest.raises(TypeError, match="takes the MorrisSamples.*not a Problem"):
                fn(cast(Any, UNIT_PROBLEM), jnp.zeros(8))

    def test_indices_is_a_named_tuple(self):
        sr = sample(UNIT_PROBLEM, n_trajectories=8, seed=1, verbose=False)
        Y = jnp.asarray(np.sum(sr.samples, axis=1))
        out = indices(sr, Y)
        assert isinstance(out, MorrisIndices)
        mu, mu_star, sigma = out
        np.testing.assert_allclose(out.mu_star, mu_star)

    def test_to_physical_units_is_idempotent(self):
        problem = Problem(names=("a", "b"), bounds=((0, 2), (0, 4)))
        sr = sample(problem, n_trajectories=8, seed=1, verbose=False)
        res = analyze(sr, jnp.asarray(np.sum(sr.samples, axis=1)), verbose=False)
        phys = res.to_physical_units()
        assert phys.to_physical_units() is phys

    def test_samples_repr_is_compact(self):
        sr = sample(UNIT_PROBLEM, n_trajectories=8, seed=1, verbose=False)
        text = repr(sr)
        assert "n_runs=" in text and "n_trajectories=8" in text
        assert "array" not in text and len(text) < 200


def test_morris_load_rejects_other_samples_file(tmp_path):
    from jaxgsa import sobol

    path = tmp_path / "sobol.npz"
    sobol.sample(ishigami.PROBLEM, base_n=8, seed=0).save(path)
    with pytest.raises(ValueError, match="holds a SobolSamples; load it with jaxgsa.sobol"):
        type(sample(ishigami.PROBLEM, 4, seed=0)).load(path)
