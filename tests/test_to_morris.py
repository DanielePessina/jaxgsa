"""Tests for deriving Morris elementary effects from a Saltelli design.

Tier T4 (internal consistency) throughout. The central check is an identity
between two of our own estimators, not an external oracle: no library
publishes Morris effects derived from a Saltelli design.

``SobolSamples.to_morris()`` reinterprets an existing Sobol/Saltelli design as
a radial Morris design. The central check is the Jansen identity: the
elementary-effect increments are literally the increments Jansen's total-order
estimator squares, so recomputing ``ST`` from the derived effects must
reproduce ``sobol.analyze``'s own ``ST``.

That identity does *not* validate every bookkeeping decision. It squares the
increments and never touches ``ee_delta``, so swapping ``ee_idx_before`` with
``ee_idx_after`` leaves it bit-identical — as it does ``mu_star`` and
``sigma``. ``TestOrientation`` covers that gap through the sign of ``mu``.
"""

from __future__ import annotations

import dataclasses
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
        """The B-based block is deliberately unused, whatever the design variant."""
        s = sobol.sample(
            ishigami.PROBLEM, 0, base_n=BASE_N, calc_second_order=False, seed=0, verbose=False
        )
        m = s.to_morris(verbose=False)
        assert m.n_trajectories == BASE_N
        assert m.n_expanded == BASE_N * (D + 1)

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


class TestOrientation:
    """Before/after bookkeeping, which the Jansen identity cannot see.

    Every other check in this file squares the increment or takes its absolute
    value, so swapping ``ee_idx_before`` with ``ee_idx_after`` leaves the
    output bit-identical. Only the *sign* of the mean elementary effect
    distinguishes the two, and only on a model whose response is monotone.
    """

    @staticmethod
    def _signed_mu(problem, model, *, swap: bool) -> np.ndarray:
        s = sobol.sample(problem, 0, base_n=BASE_N, seed=0, verbose=False)
        Y = model(jnp.asarray(s.samples))
        m = s.to_morris(verbose=False)
        if swap:
            m = dataclasses.replace(m, ee_idx_before=m.ee_idx_after, ee_idx_after=m.ee_idx_before)
        return np.asarray(morris.analyze(m, Y).mu)

    def test_monotone_model_gives_positive_mu(self):
        problem = Problem(names=("x1", "x2"), bounds=((0.0, 1.0), (0.0, 1.0)))

        def model(X):
            return 4.0 * X[:, 0] + 2.0 * X[:, 1]

        mu = self._signed_mu(problem, model, swap=False)
        # The derived design must report the true slopes, sign included.
        np.testing.assert_allclose(mu, [4.0, 2.0], rtol=1e-4)

    def test_swapping_before_and_after_flips_the_sign(self):
        problem = Problem(names=("x1", "x2"), bounds=((0.0, 1.0), (0.0, 1.0)))

        def model(X):
            return 4.0 * X[:, 0] + 2.0 * X[:, 1]

        mu = self._signed_mu(problem, model, swap=True)
        np.testing.assert_allclose(mu, [-4.0, -2.0], rtol=1e-4)


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


class TestAccuracy:
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
        result = morris.analyze(m, Y3, n_bootstrap=20, key=jax.random.key(0))
        assert result.mu_star.shape == (1, 2, D)
        assert result.mu_star_conf is not None
        assert result.mu_star_conf.shape == (2, 1, 2, D)
        # Second output is 2x the first, so its measures scale by 2.
        np.testing.assert_allclose(
            np.asarray(result.mu_star[0, 1]), 2.0 * np.asarray(result.mu_star[0, 0]), rtol=1e-4
        )


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
        # The loss is recorded, so analyze() can report it as a loss and not
        # as a design the user asked to be small.
        assert m.n_blocks_dropped == 64 - m.n_trajectories
        assert m.n_trajectories == m.ee_delta.shape[0]
        assert m.n_expanded == m.n_trajectories * (D + 1)
        assert np.all(np.abs(m.ee_delta) > 0)
        # Bookkeeping must stay self-consistent after the drop.
        Y = ishigami.evaluate(jnp.asarray(s.samples))
        Y_exp = np.asarray(Y, dtype=np.float64)[m.expanded_to_unique]
        ee = (Y_exp[m.ee_idx_after] - Y_exp[m.ee_idx_before]) / m.ee_delta
        result = morris.analyze(m, Y)
        np.testing.assert_allclose(result.mu_star, np.abs(ee).mean(axis=0), rtol=1e-4, atol=1e-4)

    def test_reliability_warning_fires_on_the_drop_path(self):
        """Dropping blocks for a zero step must trip the same floor as NaN cleaning."""
        problem = Problem(names=("x1", "x2"), bounds=((0.0, 1.0), (0.0, 1.0)))
        s = sobol.sample(problem, 0, base_n=8, scramble=False, seed=0, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = s.to_morris(verbose=False)
        assert 2 <= m.n_trajectories < 10
        Y = jnp.asarray(np.sum(np.asarray(s.samples), axis=1))
        with pytest.warns(UserWarning, match="statistically unreliable"):
            morris.analyze(m, Y)

    def test_raises_when_too_few_blocks_survive(self):
        problem = Problem(names=("x",), bounds=((0.0, 1.0),))
        s = sobol.sample(problem, 0, base_n=2, scramble=False, seed=0, verbose=False)
        with pytest.warns(UserWarning), pytest.raises(ValueError, match="at least 2"):
            s.to_morris(verbose=False)
