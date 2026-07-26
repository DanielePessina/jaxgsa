"""Tests for deriving Morris elementary effects from a Saltelli design.

``SobolSamples.to_morris()`` reinterprets an existing Sobol/Saltelli design as
a radial Morris design. The decisive check is the Jansen identity: the
elementary-effect increments are literally the increments Jansen's total-order
estimator squares, so recomputing ``ST`` from the derived effects must
reproduce ``sobol.analyze``'s own ``ST``.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats.qmc import Sobol

from jaxgsa import morris, sobol
from jaxgsa.benchmarks import ishigami
from jaxgsa.problem import GaussianInputSpec, Problem

BASE_N = 512
D = 3


@pytest.fixture(scope="module")
def derived():
    """Second-order Saltelli design on Ishigami, plus its derived Morris design."""
    s = sobol.sample(ishigami.PROBLEM, 0, base_n=BASE_N, seed=0, verbose=False)
    Y = ishigami.evaluate(jnp.asarray(s.samples))
    return s, Y, s.to_morris(verbose=False)


def _numpy_reference(sr, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent NumPy implementation of the Morris measures."""
    Y_exp = np.asarray(Y, dtype=np.float64)[sr.expanded_to_unique]
    ee = (Y_exp[sr.ee_idx_after] - Y_exp[sr.ee_idx_before]) / sr.ee_delta
    return ee.mean(axis=0), np.abs(ee).mean(axis=0), ee.std(axis=0, ddof=1)


class TestDesignStructure:
    def test_one_block_per_base_point(self, derived):
        s, _, m = derived
        assert m.n_trajectories == BASE_N
        assert m.method == "radial"
        assert m.n_expanded == BASE_N * (D + 1)
        assert m.ee_delta.shape == (BASE_N, D)
        # The evaluated rows are reused as-is, so existing Y stays valid.
        assert m.samples is s.samples
        assert m.n_runs == s.n_runs

    def test_block_count_independent_of_second_order(self):
        """The B-based block is a near-duplicate, so it is deliberately unused."""
        s = sobol.sample(
            ishigami.PROBLEM, 0, base_n=BASE_N, calc_second_order=False, seed=0, verbose=False
        )
        m = s.to_morris(verbose=False)
        assert m.n_trajectories == BASE_N
        assert m.n_expanded == BASE_N * (D + 1)

    def test_ba_rows_would_duplicate_additive_effects(self):
        """Pin the reason B-blocks are skipped: for additive terms they are identical.

        ``(f(BA_j) - f(B)) / (A_j - B_j)`` reduces to the same difference
        quotient as ``(f(AB_j) - f(A)) / (B_j - A_j)`` whenever parameter j's
        contribution is additive, so harvesting them would inflate the apparent
        sample size without adding information.
        """
        problem = Problem(names=("x1", "x2"), bounds=((0.0, 1.0),) * 2)
        s = sobol.sample(problem, 0, base_n=128, seed=3, verbose=False)
        Y = np.asarray(s.samples[:, 0] + s.samples[:, 1] ** 2, dtype=np.float64)

        step = 2 * 2 + 2
        starts = np.arange(128) * step
        e2u = s.expanded_to_unique
        unit = np.asarray(s.samples, dtype=np.float64)  # unit cube already
        for j in range(2):
            a_row, ab_row = e2u[starts], e2u[starts + 1 + j]
            b_row, ba_row = e2u[starts + step - 1], e2u[starts + 2 + 1 + j]
            ee_a = (Y[ab_row] - Y[a_row]) / (unit[ab_row, j] - unit[a_row, j])
            ee_b = (Y[ba_row] - Y[b_row]) / (unit[ba_row, j] - unit[b_row, j])
            np.testing.assert_allclose(ee_a, ee_b, atol=1e-9)

    def test_blocks_perturb_exactly_one_parameter(self, derived):
        _, _, m = derived
        expanded = m.samples[m.expanded_to_unique].reshape(m.n_trajectories, D + 1, D)
        base = expanded[:, 0, :]
        for j in range(D):
            differs = expanded[:, 1 + j, :] != base
            # Exactly parameter j changes between the base point and row 1+j.
            assert np.all(differs[:, j])
            assert not np.any(np.delete(differs, j, axis=1))

    def test_delta_recovers_the_sobol_draw(self, derived):
        _, _, m = derived
        # Regenerate the underlying draw: A is the first D dims, B the last D.
        base = Sobol(d=2 * D, scramble=True, seed=0).random(BASE_N)
        delta_true = base[:, D:] - base[:, :D]
        np.testing.assert_allclose(m.ee_delta, delta_true, atol=1e-12)


class TestJansenIdentity:
    def test_matches_sobol_ST(self, derived):
        s, Y, m = derived
        sr = sobol.analyze(s, Y)

        # sobol.analyze estimates ST from the A-based increments, which is
        # exactly what the derived blocks hold, normalized by the pooled
        # Var(concat(A, B)) with ddof=0.
        Y_exp = np.asarray(s.expand_outputs(Y), dtype=np.float64)
        step = 2 * D + 2
        offsets = np.arange(BASE_N) * step
        pooled = np.concatenate([Y_exp[offsets], Y_exp[offsets + step - 1]])

        Y_m = np.asarray(m.expand_outputs(Y), dtype=np.float64)
        increments = Y_m[m.ee_idx_after] - Y_m[m.ee_idx_before]
        ST_from_ee = 0.5 * (increments**2).mean(axis=0) / pooled.var(ddof=0)

        np.testing.assert_allclose(ST_from_ee, np.asarray(sr.ST), rtol=1e-5, atol=1e-6)

    def test_increments_equal_delta_times_ee(self, derived):
        _, Y, m = derived
        Y_m = np.asarray(m.expand_outputs(Y), dtype=np.float64)
        ee = (Y_m[m.ee_idx_after] - Y_m[m.ee_idx_before]) / m.ee_delta
        np.testing.assert_allclose(
            ee * m.ee_delta, Y_m[m.ee_idx_after] - Y_m[m.ee_idx_before], rtol=1e-12
        )


class TestAccuracy:
    def test_matches_numpy_reference(self, derived):
        _, Y, m = derived
        mu, mu_star, sigma = _numpy_reference(m, np.asarray(Y))
        result = morris.analyze(m, Y)
        np.testing.assert_allclose(result.mu, mu, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(result.mu_star, mu_star, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(result.sigma, sigma, rtol=1e-4, atol=1e-4)

    def test_all_parameters_detected(self, derived):
        _, Y, m = derived
        result = morris.analyze(m, Y)
        assert np.all(np.asarray(result.mu_star) > 0.5)
        assert result.space == "unit"

    def test_linear_model_recovers_coefficients(self):
        """On a unit-cube linear model every EE equals the coefficient exactly."""
        coeffs = np.array([1.0, 2.0, 3.0])
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0.0, 1.0),) * 3)
        s = sobol.sample(problem, 0, base_n=64, seed=1, verbose=False)
        Y = jnp.asarray(np.asarray(s.samples) @ coeffs)
        result = morris.analyze(s.to_morris(verbose=False), Y)
        np.testing.assert_allclose(np.asarray(result.mu_star), coeffs, atol=1e-4)
        np.testing.assert_allclose(np.asarray(result.sigma), 0.0, atol=1e-4)


class TestMultiOutput:
    def test_shapes_and_bootstrap(self, derived):
        _, Y, m = derived
        Y3 = jnp.stack([Y, 2.0 * Y], axis=-1)[:, None, :]  # (N, T=1, K=2)
        result = morris.analyze(m, Y3, num_resamples=20, key=jax.random.key(0))
        assert result.mu_star.shape == (1, 2, D)
        assert result.mu_star_conf is not None
        assert result.mu_star_conf.shape == (2, 1, 2, D)
        # Second output is 2x the first, so its measures scale by 2.
        np.testing.assert_allclose(
            np.asarray(result.mu_star[0, 1]), 2.0 * np.asarray(result.mu_star[0, 0]), rtol=1e-4
        )

    def test_downsample_then_analyze(self, derived):
        _, Y, m = derived
        m_small, Y_small = m.downsample(64, np.asarray(Y))
        assert m_small.n_trajectories == 64
        result = morris.analyze(m_small, jnp.asarray(Y_small))
        assert result.mu_star.shape == (D,)


class TestGaussianInputs:
    PROBLEM = Problem.from_dict(
        {
            "uni": (0.0, 1.0),
            "unbounded": GaussianInputSpec(dist="gaussian", mean=2.0, variance=4.0),
            "truncated": GaussianInputSpec(
                dist="gaussian", mean=0.0, variance=1.0, low=-2.0, high=2.0
            ),
        }
    )

    def test_delta_roundtrip_is_exact(self):
        s = sobol.sample(self.PROBLEM, 0, base_n=256, seed=1, verbose=False)
        with pytest.warns(UserWarning, match="unbounded gaussian"):
            m = s.to_morris(verbose=False)
        base = Sobol(d=6, scramble=True, seed=1).random(256)
        # Inverting the marginal transform in float64 must be exact enough to
        # divide by: uniform is affine, gaussian round-trips through the CDF.
        np.testing.assert_allclose(m.ee_delta, base[:, 3:] - base[:, :3], atol=1e-12)

    def test_warning_names_only_unbounded_params(self):
        s = sobol.sample(self.PROBLEM, 0, base_n=64, seed=1, verbose=False)
        with pytest.warns(UserWarning, match=r"\['unbounded'\]"):
            s.to_morris(verbose=False)

    def test_one_sided_truncation_still_warns(self):
        """GaussianInputSpec takes low and/or high; one bound leaves a tail open."""
        for spec in (
            GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0, low=-2.0),
            GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0, high=2.0),
        ):
            problem = Problem.from_dict({"half": spec, "uni": (0.0, 1.0)})
            s = sobol.sample(problem, 0, base_n=64, seed=1, verbose=False)
            with pytest.warns(UserWarning, match=r"\['half'\]"):
                s.to_morris(verbose=False)

    def test_bounded_problem_is_silent(self):
        s = sobol.sample(ishigami.PROBLEM, 0, base_n=64, seed=1, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s.to_morris(verbose=False)


class TestDegenerateBlocks:
    def test_unscrambled_blocks_are_dropped(self):
        """An unscrambled Sobol' sequence repeats values across coordinate pairs.

        A and B come from the same row of the same sequence, so ``A_j == B_j``
        happens often (always for the all-zeros first row), giving a zero step.
        """
        s = sobol.sample(ishigami.PROBLEM, 0, base_n=64, scramble=False, seed=0, verbose=False)
        with pytest.warns(UserWarning, match="radial blocks whose step is below"):
            m = s.to_morris(verbose=False)
        assert m.n_trajectories < 64
        assert m.n_trajectories == m.ee_delta.shape[0]
        assert m.n_expanded == m.n_trajectories * (D + 1)
        assert np.all(np.abs(m.ee_delta) > 0)
        # Bookkeeping must stay self-consistent after the drop.
        Y = ishigami.evaluate(jnp.asarray(s.samples))
        mu, mu_star, sigma = _numpy_reference(m, np.asarray(Y))
        result = morris.analyze(m, Y)
        np.testing.assert_allclose(result.mu_star, mu_star, rtol=1e-4, atol=1e-4)

    def test_raises_when_too_few_blocks_survive(self):
        problem = Problem(names=("x",), bounds=((0.0, 1.0),))
        s = sobol.sample(problem, 0, base_n=2, scramble=False, seed=0, verbose=False)
        with pytest.warns(UserWarning), pytest.raises(ValueError, match="at least 2"):
            s.to_morris(verbose=False)


def test_verbose_summary(capsys):
    s = sobol.sample(ishigami.PROBLEM, 0, base_n=64, seed=0, verbose=False)
    s.to_morris()
    out = capsys.readouterr().out
    assert "to_morris" in out
    assert "blocks=64" in out
    assert "0 new model runs" in out
