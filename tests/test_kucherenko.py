"""Tests for jaxgsa.kucherenko — design-based Sobol' indices for dependent inputs.

The decisive check is the linear-Gaussian closed form: for ``Y = a . X`` with
``X ~ N(0, R)``,

    V(Y)   = a' R a
    S1_i   = (R a)_i^2 / V(Y)
    ST_i   = a_i^2 (1 - R_i,rest R_rest^-1 R_rest,i) / V(Y)

Tolerances are set from the measured estimator error at the test sample sizes
(max ~1.5e-3 over eight seeds at base_n = 4096) with more than 3x headroom.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jaxgsa
from jaxgsa import JaxgsaWarning
from jaxgsa._core.invalid import InvalidUnit
from jaxgsa.benchmarks import ishigami
from jaxgsa.problem import Problem

# Closed-form linear-Gaussian reference, shared with test_vkoga.py and
# test_correlated_agreement.py.
from _linear_gaussian import (  # isort: skip
    A_COEF,
    A_COEF_ASYM,
    ASYM_PROBLEM,
    GAUSS_PROBLEM,
    R_ASYM,
    R_GAUSS,
    RHO,
    analytic_indices,
)


@pytest.fixture(scope="module")
def correlated_result():
    """Kucherenko analysis of the correlated linear-Gaussian model."""
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, 4096, seed=1)
    Y = ks.samples @ A_COEF
    return jaxgsa.kucherenko.analyze(ks, Y)


# --- closed-form validation --------------------------------------------------


def test_closed_form_linear_gaussian(correlated_result):
    S1_true, ST_true, var_y = analytic_indices(A_COEF, R_GAUSS)
    np.testing.assert_allclose(np.asarray(correlated_result.S1), S1_true, atol=5e-3)
    np.testing.assert_allclose(np.asarray(correlated_result.ST), ST_true, atol=5e-3)
    np.testing.assert_allclose(float(np.asarray(correlated_result.variance)), var_y, rtol=2e-2)
    assert correlated_result.is_correlated
    # Under correlation ST < S1 is the expected picture for coupled inputs.
    assert float(correlated_result.ST[0]) < float(correlated_result.S1[0])


def test_asymmetric_correlation_structure():
    """A D=4 case with six distinct off-diagonals of mixed sign.

    R_GAUSS has one non-zero off-diagonal, so the parameter axis could be
    transposed inside it without changing the result. This structure cannot be
    permuted onto itself, so any index landing on the wrong parameter shows.
    """
    S1_true, ST_true, var_y = analytic_indices(A_COEF_ASYM, R_ASYM)
    ks = jaxgsa.kucherenko.sample(ASYM_PROBLEM.with_correlation(R_ASYM), 4096, seed=1)
    result = jaxgsa.kucherenko.analyze(ks, ks.samples @ A_COEF_ASYM)
    # Measured errors are 7e-4 (S1) and 2e-5 (ST); the budgets keep headroom.
    np.testing.assert_allclose(np.asarray(result.S1), S1_true, atol=5e-3)
    np.testing.assert_allclose(np.asarray(result.ST), ST_true, atol=5e-3)
    np.testing.assert_allclose(float(np.asarray(result.variance)), var_y, rtol=2e-2)


def test_parameter_permutation_equivariance():
    """Relabelling the parameters permutes the indices, and nothing else."""
    perm = np.array([2, 0, 3, 1])
    ks = jaxgsa.kucherenko.sample(ASYM_PROBLEM.with_correlation(R_ASYM), 4096, seed=1)
    base = jaxgsa.kucherenko.analyze(ks, ks.samples @ A_COEF_ASYM)

    # The same model with its parameters in a different order: permute the
    # correlation matrix and the coefficients together.
    ks_perm = jaxgsa.kucherenko.sample(
        ASYM_PROBLEM.with_correlation(R_ASYM[np.ix_(perm, perm)]), 4096, seed=1
    )
    permuted = jaxgsa.kucherenko.analyze(ks_perm, ks_perm.samples @ A_COEF_ASYM[perm])

    np.testing.assert_allclose(np.asarray(permuted.S1), np.asarray(base.S1)[perm], atol=5e-3)
    np.testing.assert_allclose(np.asarray(permuted.ST), np.asarray(base.ST)[perm], atol=5e-3)


def test_independent_ishigami_matches_sobol():
    """Identity correlation reproduces the classic Saltelli estimates."""
    ks = jaxgsa.kucherenko.sample(ishigami.PROBLEM, 4096, seed=3)
    Y = ishigami.evaluate(np.asarray(ks.samples))
    result = jaxgsa.kucherenko.analyze(ks, Y)
    assert not result.is_correlated

    sr = jaxgsa.sobol.sample(ishigami.PROBLEM, n_samples=2**15, seed=42, verbose=False)
    sob = jaxgsa.sobol.analyze(sr, ishigami.evaluate(np.asarray(sr.samples)))

    # Both estimators against the analytic values, and against each other,
    # within Monte-Carlo error (measured max ~5e-3 at this size).
    np.testing.assert_allclose(np.asarray(result.S1), ishigami.ANALYTICAL_S1, atol=2e-2)
    np.testing.assert_allclose(np.asarray(result.ST), ishigami.ANALYTICAL_ST, atol=2e-2)
    np.testing.assert_allclose(np.asarray(result.S1), np.asarray(sob.S1), atol=2e-2)
    np.testing.assert_allclose(np.asarray(result.ST), np.asarray(sob.ST), atol=2e-2)


def test_large_output_mean_does_not_corrupt_the_indices():
    """A +1e8 output offset must leave every index unchanged.

    The S1 estimator subtracts one shared shift from both product factors,
    so the covariance never competes with O(mean^2) terms in rounding. The
    uncentered product form lost it: at this configuration the offset moved
    S1 by up to 0.11 (and zeroed small indices). Measured drift after the
    fix is < 3e-11; 1e-8 leaves ~300x headroom. Runs under x64 so the f32
    representation of Y + 1e8 does not mask the estimator behaviour.
    """
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, 2048, seed=1)
    Y = np.asarray(ks.samples @ A_COEF, dtype=np.float64)
    with jax.enable_x64():
        base = jaxgsa.kucherenko.analyze(ks, Y)
        shifted = jaxgsa.kucherenko.analyze(ks, Y + 1e8)
    np.testing.assert_allclose(np.asarray(shifted.S1), np.asarray(base.S1), atol=1e-8)
    np.testing.assert_allclose(np.asarray(shifted.ST), np.asarray(base.ST), atol=1e-8)
    np.testing.assert_allclose(float(shifted.variance), float(base.variance), rtol=1e-9)


# --- design contracts ---------------------------------------------------------


def test_design_block_structure():
    """The joint block is shared; each conditional block keeps its fixed part."""
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, 256, seed=0)
    n, D = ks.base_n, ks.n_params
    assert ks.n_runs == n * (2 * D + 1)
    assert ks.n_expanded == ks.n_runs
    np.testing.assert_array_equal(ks.expanded_to_unique, np.arange(ks.n_runs))

    X = ks.samples
    A = X[:n]
    for i in range(D):
        first = X[n * (1 + i) : n * (2 + i)]
        total = X[n * (1 + D + i) : n * (2 + D + i)]
        others = [j for j in range(D) if j != i]
        # First-order block: column i is copied from the joint block.
        np.testing.assert_allclose(first[:, i], A[:, i], atol=1e-12)
        assert not np.allclose(first[:, others], A[:, others])
        # Total block: the other columns are copied from the joint block.
        np.testing.assert_allclose(total[:, others], A[:, others], atol=1e-12)
        assert not np.allclose(total[:, i], A[:, i])


def test_conditional_blocks_follow_the_copula():
    """Redrawn columns must reproduce the declared conditional correlation."""
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, 8192, seed=5)
    n = ks.base_n
    X = ks.samples
    # In the first-order block for x1, x2 is redrawn given x1: the pair
    # correlation must match rho, and x3 must stay uncorrelated.
    first_x1 = X[n : 2 * n]
    corr = np.corrcoef(first_x1, rowvar=False)
    assert abs(corr[0, 1] - RHO) < 0.05
    assert abs(corr[0, 2]) < 0.05
    # Marginals survive the conditional construction.
    np.testing.assert_allclose(first_x1.mean(axis=0), 0.0, atol=0.05)
    np.testing.assert_allclose(first_x1.std(axis=0), 1.0, atol=0.05)


def test_sample_size_rounds_up_to_power_of_two():
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 1000, seed=0)
    assert ks.base_n == 1024


def test_correlated_problem_is_accepted():
    """kucherenko.sample is exempt from the correlated-design guard."""
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, 64, seed=0)
    assert ks.problem.has_correlated_inputs
    # The guarded samplers refuse the same problem.
    with pytest.raises(ValueError, match="independent"):
        jaxgsa.sobol.sample(problem, 64, verbose=False)


def test_sampler_validation_errors():
    with pytest.raises(ValueError, match="at least 2 parameters"):
        jaxgsa.kucherenko.sample(Problem(names=("x",), bounds=((0.0, 1.0),)), 64)
    with pytest.raises(ValueError, match="n_samples"):
        jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 1)


def test_seed_with_scramble_false_raises():
    """Tier T4: the inert seed/scramble combination is refused.

    The unscrambled Sobol' sequence is deterministic, so a seed passed with
    ``scramble=False`` would do nothing. The policy is to raise on a setting
    that cannot do what it says, never to ignore it silently.
    """
    with pytest.raises(ValueError, match="scramble=False"):
        jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 64, scramble=False, seed=0)


def test_seed_interface_matches_the_other_samplers():
    """Tier T4: ``seed`` takes an int or a Generator, and ``None`` is the default.

    An explicit int seed reproduces the design; a ``np.random.Generator`` is
    accepted the same way ``sobol.sample`` and ``morris.sample`` accept one.
    ``scramble=False`` without a seed stays valid and deterministic.
    """
    a = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 64, seed=7)
    b = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 64, seed=7)
    np.testing.assert_array_equal(a.samples, b.samples)

    from_rng = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 64, seed=np.random.default_rng(7))
    assert from_rng.samples.shape == a.samples.shape

    plain = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 64, scramble=False)
    plain_again = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 64, scramble=False)
    np.testing.assert_array_equal(plain.samples, plain_again.samples)


# --- analyze contracts --------------------------------------------------------


def test_output_shape_contract():
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 128, seed=0)
    y0 = ks.samples @ A_COEF
    Y2 = np.stack([y0, 2.0 * y0, 1.0 - y0], axis=-1)  # (N, K=3)
    Y3 = np.stack([Y2, Y2 + 0.5], axis=1)  # (N, T=2, K=3)
    D = 3
    for Y, expected in ((y0, (D,)), (Y2, (3, D)), (Y3, (2, 3, D))):
        result = jaxgsa.kucherenko.analyze(ks, Y)
        assert result.S1.shape == expected
        assert result.ST.shape == expected
        assert result.variance.shape == expected[:-1]


def test_row_count_mismatch_raises():
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 128, seed=0)
    with pytest.raises(ValueError):
        jaxgsa.kucherenko.analyze(ks, np.zeros(ks.n_runs - 1))


def test_zero_variance_slice_warns_and_yields_nan():
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 128, seed=0)
    Y = np.stack([ks.samples @ A_COEF, np.ones(ks.n_runs)], axis=-1)
    with pytest.warns(UserWarning, match="zero variance"):
        result = jaxgsa.kucherenko.analyze(ks, Y)
    assert np.all(np.isfinite(np.asarray(result.S1)[0]))
    assert np.all(np.isnan(np.asarray(result.S1)[1]))
    assert np.all(np.isnan(np.asarray(result.ST)[1]))


class TestOnInvalidPolicy:
    """T4 (behavioural): kucherenko.analyze applies the shared non-finite policy.

    The design is block-major: the joint block first, then the ``2D``
    conditional blocks, each of ``N`` rows in the same base-point order. Base
    point ``k`` therefore occupies rows ``k``, ``N + k``, ``2N + k`` and so on,
    which is the case where a contiguous-block assumption would report the
    wrong rows.
    """

    N = 128

    def _design_and_Y(self):
        """A linear-Gaussian design plus its finite outputs, one row per run."""
        ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, self.N, seed=0)
        return ks, np.asarray(ks.samples @ A_COEF, dtype=np.float64)

    def _rows_of(self, ks, base_point: int) -> list[int]:
        """Return every row that the given base point occupies."""
        n_blocks = 2 * ks.n_params + 1
        return [block * self.N + base_point for block in range(n_blocks)]

    def test_raise_is_the_default_and_names_the_base_point_and_its_rows(self):
        """T4: the default refuses, and lists the strided rows of the unit."""
        ks, Y = self._design_and_Y()
        Y[9] = np.nan
        with pytest.raises(ValueError, match=f"1 of {self.N} base points") as exc:
            jaxgsa.kucherenko.analyze(ks, Y)
        message = str(exc.value)
        assert "jaxgsa.kucherenko.analyze" in message
        assert "They condemn base points [9]" in message
        assert f"which covers {len(self._rows_of(ks, 9))} rows" in message

    def test_propagate_warns_and_lets_the_value_reach_the_indices(self):
        """T4: nothing is removed, so the indices come back non-finite."""
        ks, Y = self._design_and_Y()
        Y[9] = np.nan
        with pytest.warns(JaxgsaWarning, match="reaches the indices"):
            result = jaxgsa.kucherenko.analyze(ks, Y, on_invalid="propagate")
        assert not np.all(np.isfinite(np.asarray(result.S1)))
        assert result.invalid.policy == "propagate"

    def test_one_bad_value_condemns_every_row_of_its_base_point(self):
        """T4: the unit is the base point, and its rows are not contiguous.

        The bad value sits in the joint block. Every conditional estimator
        pairs that joint row with the conditional rows of the same base point,
        so all ``2D + 1`` of them have to go — and a contiguous-block reading
        of the layout would name ``2D`` rows that belong to other base points.
        """
        ks, Y = self._design_and_Y()
        Y[9] = np.nan
        with pytest.warns(JaxgsaWarning):
            result = jaxgsa.kucherenko.analyze(ks, Y, on_invalid="drop")
        assert result.invalid.unit_indices == (9,)
        assert result.invalid.row_indices == tuple(self._rows_of(ks, 9))
        assert len(result.invalid.row_indices) == 2 * ks.n_params + 1

    def test_a_bad_value_in_a_conditional_block_names_the_same_base_point(self):
        """T4: the unit is found from the row layout, not from the block index.

        Row ``2N + 9`` sits in the second conditional block. It belongs to
        base point 9, exactly like row 9 does.
        """
        ks, Y = self._design_and_Y()
        Y[2 * self.N + 9] = np.nan
        with pytest.warns(JaxgsaWarning):
            result = jaxgsa.kucherenko.analyze(ks, Y, on_invalid="drop")
        assert result.invalid.unit_indices == (9,)
        assert result.invalid.row_indices == tuple(self._rows_of(ks, 9))

    def test_drop_removes_the_base_point_and_leaves_finite_indices(self):
        """T4: the estimate is computed from the surviving base points.

        The reference re-runs the analysis on a design whose base point 9 was
        never poisoned but whose outputs are identical elsewhere, so the two
        agree to within the loss of one base point out of 128.
        """
        ks, Y = self._design_and_Y()
        poisoned = Y.copy()
        poisoned[9] = np.nan
        with pytest.warns(JaxgsaWarning, match=f"dropped 1 of {self.N} base points"):
            dropped = jaxgsa.kucherenko.analyze(ks, poisoned, on_invalid="drop")
        assert np.all(np.isfinite(np.asarray(dropped.S1)))
        assert np.all(np.isfinite(np.asarray(dropped.ST)))
        assert dropped.invalid.n_kept == self.N - 1
        clean = jaxgsa.kucherenko.analyze(ks, Y)
        np.testing.assert_allclose(np.asarray(dropped.S1), np.asarray(clean.S1), atol=5e-2)

    @pytest.mark.parametrize("policy", ["raise", "propagate", "drop"])
    def test_a_clean_sample_reports_nothing_and_stays_silent(self, policy, recwarn):
        """T4: a clean run gives an empty report under every policy."""
        ks, Y = self._design_and_Y()
        result = jaxgsa.kucherenko.analyze(ks, Y, on_invalid=policy)
        assert result.invalid.n_invalid == 0
        assert result.invalid.n_units == self.N
        assert result.invalid.unit is InvalidUnit.BASE_POINT
        # The precision warning is about the arithmetic, not the data: this
        # Y is float64 on purpose, so with x64 off the preamble says it is
        # about to be truncated. A clean sample must still report nothing.
        left = [
            w
            for w in recwarn
            if issubclass(w.category, JaxgsaWarning) and "truncated" not in str(w.message)
        ]
        assert left == []

    def test_a_bad_on_invalid_value_is_rejected(self):
        """T4: an unknown policy name is refused before anything is computed."""
        ks, Y = self._design_and_Y()
        with pytest.raises(ValueError, match="on_invalid must be one of"):
            jaxgsa.kucherenko.analyze(ks, Y, on_invalid="skip")


# --- persistence and reporting ------------------------------------------------


def test_save_load_round_trip(tmp_path):
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, 128, seed=0)
    path = tmp_path / "design.npz"
    ks.save(path)
    loaded = jaxgsa.kucherenko.KucherenkoSamples.load(path)
    np.testing.assert_array_equal(loaded.samples, ks.samples)
    assert loaded.base_n == ks.base_n
    assert loaded.n_params == ks.n_params
    np.testing.assert_allclose(loaded.problem.correlation, R_GAUSS, atol=1e-12)
    # The loaded design analyzes identically.
    Y = ks.samples @ A_COEF
    a = jaxgsa.kucherenko.analyze(ks, Y)
    b = jaxgsa.kucherenko.analyze(loaded, Y)
    np.testing.assert_array_equal(np.asarray(a.S1), np.asarray(b.S1))


# --- bootstrap confidence intervals ------------------------------------------


def _small_design(n=512, seed=2):
    """Return an evaluated Ishigami design, small enough to resample quickly."""
    problem = ishigami.PROBLEM
    ks = jaxgsa.kucherenko.sample(problem, n, seed=seed)
    return ks, ishigami.evaluate(ks.samples)


class TestBootstrap:
    """Base-point resampling, the one unit this design can drop safely."""

    def test_no_bootstrap_leaves_every_interval_unset(self):
        """The plainest call reports no interval and no CI record."""
        ks, Y = _small_design()
        result = jaxgsa.kucherenko.analyze(ks, Y)
        assert result.ci is None
        assert result.S1_conf is None
        assert result.ST_conf is None

    def test_a_bootstrap_needs_a_key(self):
        """No key is silently invented, per the vocabulary."""
        ks, Y = _small_design()
        with pytest.raises(ValueError, match="key is required"):
            jaxgsa.kucherenko.analyze(ks, Y, n_bootstrap=8)

    def test_the_point_estimate_does_not_move(self):
        """T4: asking for an interval must not change the number it brackets."""
        ks, Y = _small_design()
        plain = jaxgsa.kucherenko.analyze(ks, Y)
        with_ci = jaxgsa.kucherenko.analyze(ks, Y, n_bootstrap=16, key=jax.random.key(0))
        np.testing.assert_array_equal(np.asarray(with_ci.S1), np.asarray(plain.S1))
        np.testing.assert_array_equal(np.asarray(with_ci.ST), np.asarray(plain.ST))
        np.testing.assert_array_equal(np.asarray(with_ci.variance), np.asarray(plain.variance))

    def test_intervals_bracket_the_estimate_and_variance_has_none(self):
        """Both indices get an interval; the denominator deliberately does not."""
        ks, Y = _small_design()
        result = jaxgsa.kucherenko.analyze(ks, Y, n_bootstrap=64, key=jax.random.key(1))
        assert result.ci.n_bootstrap == 64
        assert result.ci.level == 0.95
        assert result.ci.method == "quantile"
        assert not hasattr(result, "variance_conf")
        for name in ("S1", "ST"):
            conf = np.asarray(getattr(result, f"{name}_conf"))
            point = np.asarray(getattr(result, name))
            assert conf.shape == (2, *point.shape)
            assert (conf[0] <= point + 1e-5).all()
            assert (point <= conf[1] + 1e-5).all()

    def test_a_replicate_keeps_each_conditional_block_beside_its_base_point(self):
        """The unit is the base point, and this is what says so numerically.

        A replicate that resampled rows independently in each block would
        pair ``f(y_k, z_k)`` with a conditional row drawn around a different
        base point. ``ST`` is the Jansen squared difference of exactly that
        pair, so a misaligned replicate inflates it towards
        ``2 * V(Y) / V(Y) = 2``. Every replicate here stays in range.
        """
        ks, Y = _small_design()
        result = jaxgsa.kucherenko.analyze(
            ks, Y, n_bootstrap=32, key=jax.random.key(2), keep_replicates=True
        )
        ST_draws = np.asarray(result.ci.replicates["ST"])
        assert ST_draws.shape == (32, ks.n_params)
        assert (ST_draws >= -0.05).all()
        assert (ST_draws <= 1.2).all()

    def test_the_interval_narrows_as_the_design_grows(self):
        """T4: a bootstrap interval must respond to the sample size."""
        small = jaxgsa.kucherenko.analyze(
            *_small_design(n=256, seed=3), n_bootstrap=64, key=jax.random.key(3)
        )
        large = jaxgsa.kucherenko.analyze(
            *_small_design(n=2048, seed=3), n_bootstrap=64, key=jax.random.key(3)
        )
        small_width = np.asarray(small.S1_conf[1] - small.S1_conf[0])
        large_width = np.asarray(large.S1_conf[1] - large.S1_conf[0])
        assert large_width.mean() < small_width.mean()

    def test_gaussian_endpoints_are_symmetric_about_the_estimate(self):
        """T0: the normal-approximation interval is ``estimate +/- z*sd``."""
        ks, Y = _small_design()
        result = jaxgsa.kucherenko.analyze(
            ks, Y, n_bootstrap=64, ci_method="gaussian", key=jax.random.key(4)
        )
        conf = np.asarray(result.S1_conf)
        point = np.asarray(result.S1)
        np.testing.assert_allclose(conf[1] - point, point - conf[0], rtol=1e-5)

    def test_keep_replicates_off_by_default(self):
        """The draws are large, so they are kept only when asked for."""
        ks, Y = _small_design()
        result = jaxgsa.kucherenko.analyze(ks, Y, n_bootstrap=8, key=jax.random.key(5))
        assert result.ci.replicates is None

    def test_a_time_series_keeps_its_layout(self):
        """T4: an interval mirrors the output rank the caller passed."""
        ks, Y = _small_design(n=256)
        Y_ts = np.stack([Y, 2.0 * Y], axis=-1)[:, None, :]  # (n_runs, 1, 2)
        result = jaxgsa.kucherenko.analyze(ks, Y_ts, n_bootstrap=16, key=jax.random.key(6))
        assert np.asarray(result.S1).shape == (1, 2, ks.n_params)
        assert np.asarray(result.S1_conf).shape == (2, 1, 2, ks.n_params)

    def test_the_dataset_carries_the_endpoints(self):
        """An interval is only reported if it also exports."""
        ks, Y = _small_design(n=256)
        result = jaxgsa.kucherenko.analyze(ks, Y, n_bootstrap=8, key=jax.random.key(7))
        ds = result.to_dataset()
        for name in ("S1", "ST"):
            assert f"{name}_lower" in ds
            assert f"{name}_upper" in ds
