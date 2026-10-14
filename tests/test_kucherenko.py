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

import functools
import warnings

import jax
import numpy as np
import pytest

import jaxgsa
from jaxgsa import JaxgsaWarning
from jaxgsa._core.invalid import InvalidUnit
from jaxgsa.problem import Problem

# Closed-form linear-Gaussian reference, shared with test_vkoga.py. The
# correlated and Ishigami designs are session fixtures (``_fixtures_dependence``).
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


# --- closed-form validation --------------------------------------------------


def test_closed_form_linear_gaussian(lg_corr_kucherenko):
    S1_true, ST_true, var_y = analytic_indices(A_COEF, R_GAUSS)
    np.testing.assert_allclose(np.asarray(lg_corr_kucherenko.S1), S1_true, atol=5e-3)
    np.testing.assert_allclose(np.asarray(lg_corr_kucherenko.ST), ST_true, atol=5e-3)
    np.testing.assert_allclose(float(np.asarray(lg_corr_kucherenko.variance)), var_y, rtol=2e-2)
    assert lg_corr_kucherenko.dependence == "gaussian"
    # Under correlation ST < S1 is the expected picture for coupled inputs.
    assert float(lg_corr_kucherenko.ST[0]) < float(lg_corr_kucherenko.S1[0])


def test_asymmetric_correlation_structure():
    """A D=4 case with six distinct off-diagonals of mixed sign.

    R_GAUSS has one non-zero off-diagonal, so the parameter axis could be
    transposed inside it without changing the result. This structure cannot be
    permuted onto itself, so any index landing on the wrong parameter shows.
    """
    S1_true, ST_true, var_y = analytic_indices(A_COEF_ASYM, R_ASYM)
    ks = jaxgsa.kucherenko.sample(ASYM_PROBLEM.with_correlation(R_ASYM), base_n=4096, seed=1)
    result = jaxgsa.kucherenko.analyze(ks, ks.samples @ A_COEF_ASYM)
    # Measured errors are 7e-4 (S1) and 2e-5 (ST); the budgets keep headroom.
    np.testing.assert_allclose(np.asarray(result.S1), S1_true, atol=5e-3)
    np.testing.assert_allclose(np.asarray(result.ST), ST_true, atol=5e-3)
    np.testing.assert_allclose(float(np.asarray(result.variance)), var_y, rtol=2e-2)


def test_analytic_indices_accepts_a_covariance_matrix():
    """The closed form must not depend on the input scale.

    ``Y = a . X`` with ``X ~ N(0, D R D)`` is the same model as
    ``Y = (D a) . Z`` with ``Z ~ N(0, R)``. The indices must match. The
    old unit-diagonal formula gave ``S_TC = [3.654, 0.654]`` for the 2-D
    case below.
    """
    scales = np.array([2.0, 0.5, 3.0, 1.5])
    sigma = np.diag(scales) @ R_ASYM @ np.diag(scales)
    S_TC, S_TU, var_y = analytic_indices(A_COEF_ASYM, sigma)
    S_TC_ref, S_TU_ref, var_y_ref = analytic_indices(scales * A_COEF_ASYM, R_ASYM)
    np.testing.assert_allclose(S_TC, S_TC_ref, rtol=1e-12)
    np.testing.assert_allclose(S_TU, S_TU_ref, rtol=1e-12)
    np.testing.assert_allclose(var_y, var_y_ref, rtol=1e-12)

    # Hand values: Sigma a = [5.2, 2.2], V = 7.4, Var(X_i | X_~i) = Sigma_ii - 1.44 / Sigma_~i~i.
    S_TC2, S_TU2, _ = analytic_indices(np.ones(2), np.array([[4.0, 1.2], [1.2, 1.0]]))
    np.testing.assert_allclose(S_TC2, [27.04 / 29.6, 4.84 / 7.4], rtol=1e-12)
    np.testing.assert_allclose(S_TU2, [(4.0 - 1.44) / 7.4, (1.0 - 0.36) / 7.4], rtol=1e-12)


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
    ks = jaxgsa.kucherenko.sample(problem, base_n=2048, seed=1)
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
    ks = jaxgsa.kucherenko.sample(problem, base_n=256, seed=0)
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
    ks = jaxgsa.kucherenko.sample(problem, base_n=8192, seed=5)
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


def test_n_samples_is_a_run_budget_rounded_up_to_a_power_of_two_base():
    """n_samples is the minimum number of runs, as in sobol.sample."""
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 1000, seed=0, verbose=False)
    # D=3 gives 7 blocks: 143 base points needed, rounded up to 256.
    assert ks.base_n == 256
    assert ks.n_runs == 256 * 7 >= 1000
    exact = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 256 * 7, seed=0, verbose=False)
    assert exact.base_n == 256
    one_more = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 256 * 7 + 1, seed=0, verbose=False)
    assert one_more.base_n == 512


def test_base_n_sets_the_base_points_exactly_and_gives_the_same_design():
    by_budget = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 1000, seed=3, verbose=False)
    by_base = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, base_n=256, seed=3, verbose=False)
    np.testing.assert_array_equal(by_budget.samples, by_base.samples)


@pytest.mark.parametrize("sampler", ["kucherenko", "sobol"])
def test_exactly_one_of_n_samples_and_base_n(sampler):
    sample = getattr(jaxgsa, sampler).sample
    with pytest.raises(ValueError, match="n_samples.*base_n"):
        sample(GAUSS_PROBLEM, verbose=False)
    with pytest.raises(ValueError, match="n_samples.*base_n"):
        sample(GAUSS_PROBLEM, 1024, base_n=128, verbose=False)


def test_base_n_must_be_a_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        jaxgsa.kucherenko.sample(GAUSS_PROBLEM, base_n=100, verbose=False)


def test_tiny_base_n_warns_about_the_floor():
    with pytest.warns(JaxgsaWarning, match="floor of 16"):
        jaxgsa.kucherenko.sample(GAUSS_PROBLEM, base_n=8, seed=0, verbose=False)


def test_verbose_line_names_the_request_only_for_a_budget(monkeypatch):
    from jaxgsa._core import verbose as _verbose

    lines: list[str] = []
    monkeypatch.setattr(_verbose, "emit", lines.append)
    correlated = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    jaxgsa.kucherenko.sample(correlated, 1000, seed=0)
    assert "requested_runs>=1000" in lines[-1]
    jaxgsa.kucherenko.sample(correlated, base_n=256, seed=0)
    assert "requested_runs" not in lines[-1]


def test_analyze_refuses_a_problem_with_a_pointer_to_given_data_methods():
    with pytest.raises(TypeError, match="KucherenkoSamples.*not a Problem.*jaxgsa.rank"):
        jaxgsa.kucherenko.analyze(GAUSS_PROBLEM, np.zeros(8))  # type: ignore[arg-type]


def test_correlated_problem_is_accepted():
    """kucherenko.sample is exempt from the correlated-design guard."""
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, base_n=64, seed=0)
    assert ks.problem.dependence == "gaussian"
    # The guarded samplers refuse the same problem.
    with pytest.raises(ValueError, match="independent"):
        jaxgsa.sobol.sample(problem, 64, verbose=False)


def test_an_independent_problem_warns_and_quotes_the_cheaper_design():
    """T4: sample() says an uncorrelated problem is overpaying, with numbers.

    The design degenerates to a Saltelli column-swap scheme under no
    correlation, so the user pays base_n(2D+1) for indices sobol reaches from
    base_n(D+2). The warning must carry all three run counts, because the
    point of it is the budget comparison rather than the fact itself.
    """
    D = GAUSS_PROBLEM.n_params
    base_n = 64
    with pytest.warns(JaxgsaWarning, match="the problem declares no dependence") as caught:
        jaxgsa.kucherenko.sample(GAUSS_PROBLEM, base_n=base_n, seed=0)
    message = str(caught[0].message)
    assert f"{base_n * (2 * D + 1)} runs" in message
    assert f"{base_n * (D + 2)} runs" in message
    assert f"{base_n * (2 * D + 2)} runs" in message
    assert "jaxgsa.sobol.sample" in message
    # A declared correlation is the intended use and must stay silent.
    correlated = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        jaxgsa.kucherenko.sample(correlated, base_n=base_n, seed=0)
    assert [w for w in quiet if "the problem declares no dependence" in str(w.message)] == []


@pytest.mark.parametrize(
    "problem, kwargs, message",
    [
        (Problem(names=("x",), bounds=((0.0, 1.0),)), {"n_samples": 64}, "at least 2 parameters"),
        (GAUSS_PROBLEM, {"n_samples": 1}, "n_samples"),
        # Tier T4: the inert seed/scramble combination is refused. The
        # unscrambled Sobol' sequence is deterministic, so a seed passed with
        # ``scramble=False`` would do nothing. The policy is to raise on a
        # setting that cannot do what it says, never to ignore it silently.
        (GAUSS_PROBLEM, {"n_samples": 64, "scramble": False, "seed": 0}, "scramble=False"),
    ],
    ids=["one-parameter", "too-few-samples", "seed-with-scramble-false"],
)
def test_sampler_guards(problem, kwargs, message):
    with pytest.raises(ValueError, match=message):
        jaxgsa.kucherenko.sample(problem, **kwargs, verbose=False)


# --- analyze contracts --------------------------------------------------------


def test_output_shape_contract():
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, base_n=128, seed=0)
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
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, base_n=128, seed=0)
    with pytest.raises(ValueError):
        jaxgsa.kucherenko.analyze(ks, np.zeros(ks.n_runs - 1))


def test_zero_variance_slice_warns_and_yields_nan():
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, base_n=128, seed=0)
    Y = np.stack([ks.samples @ A_COEF, np.ones(ks.n_runs)], axis=-1)
    with pytest.warns(UserWarning, match="zero variance"):
        result = jaxgsa.kucherenko.analyze(ks, Y)
    assert np.all(np.isfinite(np.asarray(result.S1)[0]))
    assert np.all(np.isnan(np.asarray(result.S1)[1]))
    assert np.all(np.isnan(np.asarray(result.ST)[1]))


@functools.cache
def _linear_gaussian_design(n: int):
    """A linear-Gaussian design and its finite outputs, one row per run (read-only)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JaxgsaWarning)
        ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, base_n=n, seed=0, verbose=False)
    return ks, np.asarray(ks.samples @ A_COEF, dtype=np.float64)


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
        """A linear-Gaussian design plus a fresh copy of its finite outputs."""
        ks, Y = _linear_gaussian_design(self.N)
        return ks, Y.copy()

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

    def test_a_clean_sample_reports_nothing_and_stays_silent(self, recwarn):
        """T4: a clean run gives an empty report under every policy."""
        ks, Y = self._design_and_Y()
        for policy in ("raise", "propagate", "drop"):
            result = jaxgsa.kucherenko.analyze(ks, Y, on_invalid=policy)
            assert result.invalid.n_invalid == 0
            assert result.invalid.n_units == self.N
            assert result.invalid.unit is InvalidUnit.BASE_POINT
        # Two warnings are about something other than the data and are
        # expected here. The precision one is about the arithmetic: this Y is
        # float64 on purpose, so with x64 off the preamble says it is about to
        # be truncated. The sample-time one is about the design budget:
        # GAUSS_PROBLEM carries no correlation, so sample() points at
        # jaxgsa.sobol. Neither is an invalid-data report, and a clean sample
        # must still report nothing.
        left = [
            w
            for w in recwarn
            if issubclass(w.category, JaxgsaWarning)
            and "truncated" not in str(w.message)
            and "jaxgsa.kucherenko.sample" not in str(w.message)
        ]
        assert left == []


# --- persistence and reporting ------------------------------------------------


def test_save_load_round_trip(tmp_path):
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, base_n=128, seed=0)
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


class TestBootstrap:
    """Base-point resampling, the one unit this design can drop safely."""

    def test_no_bootstrap_leaves_every_interval_unset_and_key_is_required(self, kuch_ishigami_512):
        """The plainest call reports no interval and no CI record."""
        ks, Y = kuch_ishigami_512
        result = jaxgsa.kucherenko.analyze(ks, Y)
        assert result.ci is None
        assert result.S1_conf is None
        assert result.ST_conf is None
        with pytest.raises(ValueError, match="key is required"):
            jaxgsa.kucherenko.analyze(ks, Y, n_bootstrap=8)

    def test_intervals_bracket_the_estimate_and_variance_has_none(self, kuch_ishigami_512):
        """Both indices get an interval; the denominator deliberately does not."""
        ks, Y = kuch_ishigami_512
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

    def test_a_replicate_keeps_each_conditional_block_beside_its_base_point(
        self, kuch_ishigami_512
    ):
        """The unit is the base point, and this is what says so numerically.

        A replicate that resampled rows independently in each block would
        pair ``f(y_k, z_k)`` with a conditional row drawn around a different
        base point. ``ST`` is the Jansen squared difference of exactly that
        pair, so a misaligned replicate inflates it towards
        ``2 * V(Y) / V(Y) = 2``. Every replicate here stays in range.
        """
        ks, Y = kuch_ishigami_512
        result = jaxgsa.kucherenko.analyze(
            ks, Y, n_bootstrap=32, key=jax.random.key(2), keep_replicates=True
        )
        ST_draws = np.asarray(result.ci.replicates["ST"])
        assert ST_draws.shape == (32, ks.n_params)
        assert (ST_draws >= -0.05).all()
        assert (ST_draws <= 1.2).all()

    def test_gaussian_endpoints_are_symmetric_about_the_estimate(self, kuch_ishigami_512):
        """T0: the normal-approximation interval is ``estimate +/- z*sd``."""
        ks, Y = kuch_ishigami_512
        result = jaxgsa.kucherenko.analyze(
            ks, Y, n_bootstrap=64, ci_method="gaussian", key=jax.random.key(4)
        )
        conf = np.asarray(result.S1_conf)
        point = np.asarray(result.S1)
        np.testing.assert_allclose(conf[1] - point, point - conf[0], rtol=1e-5)

    def test_a_time_series_keeps_its_layout(self, kuch_ishigami_512):
        """T4: an interval mirrors the output rank the caller passed."""
        ks, Y = kuch_ishigami_512
        Y_ts = np.stack([Y, 2.0 * Y], axis=-1)[:, None, :]  # (n_runs, 1, 2)
        result = jaxgsa.kucherenko.analyze(ks, Y_ts, n_bootstrap=16, key=jax.random.key(6))
        assert np.asarray(result.S1).shape == (1, 2, ks.n_params)
        assert np.asarray(result.S1_conf).shape == (2, 1, 2, ks.n_params)

    def test_the_dataset_carries_the_endpoints(self, kuch_ishigami_512):
        """An interval is only reported if it also exports."""
        ks, Y = kuch_ishigami_512
        result = jaxgsa.kucherenko.analyze(ks, Y, n_bootstrap=8, key=jax.random.key(7))
        ds = result.to_dataset()
        for name in ("S1", "ST"):
            assert f"{name}_lower" in ds
            assert f"{name}_upper" in ds
