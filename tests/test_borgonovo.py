"""Tests for Borgonovo delta sensitivity analysis."""

from __future__ import annotations

import warnings
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa.benchmarks import gaussian_linear, ishigami
from jaxgsa.borgonovo import analyze
from jaxgsa.borgonovo._analyze import _DEGENERATE_BW_FRACTION, _plischke_n_classes
from jaxgsa.problem import Problem
from jaxgsa.sampling import monte_carlo


@pytest.fixture(scope="module")
def degenerate_class_data():
    """A problem with one genuinely degenerate conditioning class.

    The categorical parameter ``c`` has three levels. Level 0 maps every
    sample to the single output value 5.0, so that conditioning class has
    exactly zero variance and zero bandwidth. It is the only kind of class
    the floor predicate fires on at the default ``degenerate_tol``. The
    other two levels keep a genuine spread from the continuous parameter,
    so the column has 1369 distinct values and is not refused as a discrete
    output.

    Level 0's output value is also the column maximum, so it sits exactly on
    the last grid point. That is the worst case for a floored kernel: the
    trapezoid rule lands on the spike rather than missing it.
    """
    problem = Problem.from_dict(
        {"x": (0.0, 1.0), "c": {"dist": "categorical", "probs": [1 / 3, 1 / 3, 1 / 3]}}
    )
    rng = np.random.default_rng(0)
    n = 2000
    codes = rng.integers(0, 3, size=n).astype(np.float64)
    x = rng.uniform(0.0, 1.0, size=n)
    X = jnp.asarray(np.stack([x, codes], axis=1))
    Y = jnp.asarray(np.where(codes == 0, 5.0, codes + 0.3 * x))
    return problem, X, Y


@pytest.fixture(scope="module")
def ishigami_data():
    """Generate Ishigami test data."""
    N = 5000
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


class TestDeltaBasic:
    def test_shape_scalar_output(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0)
        assert result.delta.shape == (3,)
        assert result.S1.shape == (3,)
        assert result.delta_conf is None
        assert result.S1_conf is None

    def test_plugin_values_in_unit_interval(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0)
        delta = np.asarray(result.delta)
        s1 = np.asarray(result.S1)
        assert np.all(delta >= 0.0) and np.all(delta <= 1.0)
        assert np.all(s1 >= 0.0) and np.all(s1 <= 1.0)

    def test_x1_x2_more_important_than_x3(self, ishigami_data):
        """x1 and x2 should have higher delta than x3 (weak first-order)."""
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0)
        delta = np.asarray(result.delta)
        assert delta[0] > delta[2], "x1 should be more important than x3"
        assert delta[1] > delta[2], "x2 should be more important than x3"

    def test_deterministic_given_seed(self, ishigami_data):
        X, Y = ishigami_data
        r1 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=20, seed=3)
        r2 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=20, seed=3)
        np.testing.assert_array_equal(np.asarray(r1.delta), np.asarray(r2.delta))
        assert r1.delta_conf is not None and r2.delta_conf is not None
        np.testing.assert_array_equal(np.asarray(r1.delta_conf), np.asarray(r2.delta_conf))

    def test_n_classes_override_changes_result(self, ishigami_data):
        X, Y = ishigami_data
        r_default = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0)
        r_coarse = analyze(ishigami.PROBLEM, X, Y, n_classes=4, n_bootstrap=0)
        assert not np.allclose(np.asarray(r_default.delta), np.asarray(r_coarse.delta))

    def test_slice_chunk_size_invariance(self, ishigami_data):
        """Chunked column processing must not change the result."""
        X, Y = ishigami_data
        Y2 = jnp.stack([Y, Y**2, jnp.sin(Y)], axis=1)
        r_full = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=10, seed=0)
        r_chunked = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=10, seed=0, slice_chunk_size=1)
        np.testing.assert_allclose(
            np.asarray(r_full.delta), np.asarray(r_chunked.delta), rtol=1e-6
        )


class TestDeltaSALibComparison:
    def test_plugin_matches_salib_calc_delta(self, ishigami_data):
        """Unit-level parity: plug-in estimator vs SALib's calc_delta."""
        from SALib.analyze import delta as salib_delta

        X_np = np.asarray(ishigami_data[0])
        Y_np = np.asarray(ishigami_data[1])
        N = len(Y_np)

        M = _plischke_n_classes(N)
        m = np.linspace(0, N, M + 1)
        Ygrid = np.linspace(Y_np.min(), Y_np.max(), 100)
        salib_plugin = np.array(
            [salib_delta.calc_delta(Y_np, Ygrid, X_np[:, i], m) for i in range(3)]
        )

        result = analyze(ishigami.PROBLEM, ishigami_data[0], ishigami_data[1], n_bootstrap=0)
        np.testing.assert_allclose(np.asarray(result.delta), salib_plugin, atol=1e-5)

    def test_matches_salib_end_to_end(self, ishigami_data):
        """Bias-corrected delta and S1 vs SALib.analyze.delta.

        SALib's central estimates are computed on a random resample of the
        data (jaxgsa deliberately uses the original sample), so the tolerance
        covers that resampling noise on top of bootstrap RNG differences.
        """
        from SALib.analyze import delta as salib_delta

        X_np = np.asarray(ishigami_data[0])
        Y_np = np.asarray(ishigami_data[1])

        salib_problem = {
            "num_vars": 3,
            "names": ["x1", "x2", "x3"],
            "bounds": [[-np.pi, np.pi]] * 3,
        }
        salib_result = salib_delta.analyze(salib_problem, X_np, Y_np, num_resamples=100, seed=7)

        result = analyze(ishigami.PROBLEM, ishigami_data[0], ishigami_data[1], seed=0)
        np.testing.assert_allclose(np.asarray(result.delta), salib_result["delta"], atol=0.03)
        np.testing.assert_allclose(np.asarray(result.S1), salib_result["S1"], atol=0.02)


class TestGaussianLinearBenchmark:
    def test_analytical_delta_matches_brute_force(self):
        """Closed-form Gaussian L1 + quadrature vs dense numeric integration."""
        # scipy.integrate.trapezoid works on numpy 1.x and 2.x (np.trapezoid
        # is 2.0-only), so this test does not force a numpy>=2 floor.
        from scipy.integrate import trapezoid
        from scipy.stats import norm

        c = np.asarray(gaussian_linear.DEFAULT_COEFFS)
        v = np.asarray(gaussian_linear.DEFAULT_VARIANCES)
        var_y = float((c**2 * v).sum())

        brute = np.zeros(len(c))
        for i in range(len(c)):
            xs = np.linspace(-8.0, 8.0, 1601) * np.sqrt(v[i])
            ys = np.linspace(-10.0, 10.0, 3201) * np.sqrt(var_y)
            v_cond = var_y - c[i] ** 2 * v[i]
            f_y = norm.pdf(ys, scale=np.sqrt(var_y))
            l1 = np.array(
                [
                    trapezoid(np.abs(f_y - norm.pdf(ys, loc=c[i] * x, scale=np.sqrt(v_cond))), ys)
                    for x in xs
                ]
            )
            w = norm.pdf(xs, scale=np.sqrt(v[i]))
            brute[i] = 0.5 * trapezoid(l1 * w, xs)

        np.testing.assert_allclose(gaussian_linear.ANALYTICAL_DELTA, brute, atol=1e-4)

    def test_analytical_delta_zero_coefficient(self):
        """A zero coefficient means no influence: delta must be exactly 0."""
        delta = gaussian_linear.analytical_delta(coeffs=(0.0, 1.0, 2.0))
        assert delta[0] == 0.0
        assert np.all(delta[1:] > 0.0)

    def test_analytical_delta_zero_variance(self):
        """A constant (zero-variance) input yields delta 0, not NaN."""
        delta = gaussian_linear.analytical_delta(coeffs=(1.0, 2.0, 3.0), variances=(0.0, 1.0, 1.0))
        assert np.all(np.isfinite(delta))
        assert delta[0] == 0.0
        assert np.all(delta[1:] > 0.0)

    def test_analytical_delta_tiny_coefficient_finite(self):
        """A near-zero coefficient stays finite (no 1/v1-1/v2 cancellation)."""
        delta = gaussian_linear.analytical_delta(coeffs=(1e-9, 1.0, 2.0))
        assert np.all(np.isfinite(delta))
        assert delta[0] >= 0.0

    def test_estimator_converges_to_analytical_delta(self, gaussian_linear_data):
        X, Y = gaussian_linear_data
        result = analyze(gaussian_linear.PROBLEM, X, Y, seed=0)
        np.testing.assert_allclose(
            np.asarray(result.delta), gaussian_linear.ANALYTICAL_DELTA, atol=0.02
        )

    def test_estimator_matches_analytical_s1(self, gaussian_linear_data):
        X, Y = gaussian_linear_data
        result = analyze(gaussian_linear.PROBLEM, X, Y, n_bootstrap=0)
        np.testing.assert_allclose(np.asarray(result.S1), gaussian_linear.ANALYTICAL_S1, atol=0.02)

    def test_analytical_sobol_structure(self):
        S1, ST, S2 = gaussian_linear.analytical_indices()
        np.testing.assert_allclose(S1, ST)
        np.testing.assert_allclose(S1.sum(), 1.0)
        assert np.all(np.isnan(np.diag(S2)))
        off_diag = S2[~np.eye(len(S1), dtype=bool)]
        np.testing.assert_allclose(off_diag, 0.0)


class TestDeltaBootstrap:
    def test_bootstrap_produces_conf(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=20, conf_level=0.95, seed=0)
        assert result.delta_conf is not None
        assert result.S1_conf is not None
        assert result.delta_conf.shape == (2, 3)
        assert result.S1_conf.shape == (2, 3)

    def test_bootstrap_lower_leq_upper(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=20, conf_level=0.95, seed=0)
        assert result.delta_conf is not None
        assert np.all(np.asarray(result.delta_conf[0]) <= np.asarray(result.delta_conf[1]) + 1e-6)
        assert result.S1_conf is not None
        assert np.all(np.asarray(result.S1_conf[0]) <= np.asarray(result.S1_conf[1]) + 1e-6)

    def test_corrected_delta_within_conf(self, ishigami_data):
        """The bias-corrected estimate is the mean of the CI replicates."""
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=50, conf_level=0.99, seed=0)
        delta = np.asarray(result.delta)
        assert result.delta_conf is not None
        assert np.all(delta >= np.asarray(result.delta_conf[0]) - 1e-6)
        assert np.all(delta <= np.asarray(result.delta_conf[1]) + 1e-6)

    def test_bias_correct_false_returns_plugin(self, ishigami_data):
        X, Y = ishigami_data
        r_plugin = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0)
        r_uncorrected = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=20, bias_correct=False)
        np.testing.assert_allclose(
            np.asarray(r_uncorrected.delta), np.asarray(r_plugin.delta), rtol=1e-6
        )
        assert r_uncorrected.delta_conf is not None

    def test_bias_correction_reduces_delta(self, ishigami_data):
        """The plug-in estimator is biased upward; correction should lower it."""
        X, Y = ishigami_data
        r_plugin = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0)
        r_corrected = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=100, seed=0)
        assert np.all(np.asarray(r_corrected.delta) <= np.asarray(r_plugin.delta) + 1e-6)


class TestDeltaMultiOutput:
    def test_multi_output_shape(self, ishigami_data):
        X, Y = ishigami_data
        Y2 = jnp.stack([Y, Y**2], axis=1)
        result = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=10, seed=0)
        assert result.delta.shape == (2, 3)
        assert result.S1.shape == (2, 3)
        assert result.delta_conf is not None
        assert result.delta_conf.shape == (2, 2, 3)

    def test_time_series_shape(self, ishigami_data):
        X, Y = ishigami_data
        Y3 = jnp.stack(
            [jnp.stack([Y, Y**2], axis=1), jnp.stack([Y + 1.0, jnp.abs(Y)], axis=1)],
            axis=1,
        )
        assert Y3.shape == (Y.shape[0], 2, 2)
        result = analyze(ishigami.PROBLEM, X, Y3, n_bootstrap=10, seed=0)
        assert result.delta.shape == (2, 2, 3)
        assert result.delta_conf is not None
        assert result.delta_conf.shape == (2, 2, 2, 3)

    def test_multi_output_consistent_with_scalar(self, ishigami_data):
        """Column 0 of a multi-output run must equal the scalar-output run."""
        X, Y = ishigami_data
        Y2 = jnp.stack([Y, Y**2], axis=1)
        r_scalar = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=10, seed=0)
        r_multi = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=10, seed=0)
        np.testing.assert_allclose(
            np.asarray(r_multi.delta[0]), np.asarray(r_scalar.delta), rtol=1e-6
        )
        np.testing.assert_allclose(np.asarray(r_multi.S1[0]), np.asarray(r_scalar.S1), rtol=1e-6)


class TestDeltaEdgeCases:
    def test_constant_output_yields_zero(self, ishigami_data):
        X, _ = ishigami_data
        Y_const = jnp.ones(X.shape[0])
        result = analyze(ishigami.PROBLEM, X, Y_const, n_bootstrap=5, seed=0)
        np.testing.assert_allclose(np.asarray(result.delta), 0.0)
        np.testing.assert_allclose(np.asarray(result.S1), 0.0)
        assert np.all(np.isfinite(np.asarray(result.delta)))

    def test_constant_column_among_varying(self, ishigami_data):
        X, Y = ishigami_data
        Y2 = jnp.stack([Y, jnp.ones_like(Y)], axis=1)
        result = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=0)
        delta = np.asarray(result.delta)
        np.testing.assert_allclose(delta[1], 0.0)
        assert np.all(delta[0] > 0.0)
        assert np.all(np.isfinite(delta))


class TestDeltaValidation:
    def test_rejects_1d_x(self, ishigami_data):
        _, Y = ishigami_data
        with pytest.raises(ValueError, match="2-D"):
            analyze(ishigami.PROBLEM, jnp.ones(len(Y)), Y)

    def test_rejects_wrong_param_count(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="parameters"):
            analyze(ishigami.PROBLEM, X[:, :2], Y)

    def test_rejects_mismatched_rows(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="rows"):
            analyze(ishigami.PROBLEM, X, Y[:-1])

    def test_rejects_bad_n_classes(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="n_classes"):
            analyze(ishigami.PROBLEM, X, Y, n_classes=1)

    def test_rejects_bad_conf_level(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="conf_level"):
            analyze(ishigami.PROBLEM, X, Y, conf_level=1.5)

    def test_rejects_bad_bandwidth(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="bandwidth"):
            analyze(ishigami.PROBLEM, X, Y, bandwidth=-0.5)

    def test_rejects_bad_grid_size(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="grid_size"):
            analyze(ishigami.PROBLEM, X, Y, grid_size=1)


class TestDeltaXarray:
    def test_scalar_dataset(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=10, seed=0)
        ds = result.to_dataset()
        assert set(ds.data_vars) == {
            "delta",
            "S1",
            "delta_lower",
            "delta_upper",
            "S1_lower",
            "S1_upper",
        }
        assert ds["delta"].dims == ("param",)
        assert list(ds.coords["param"].values) == ["x1", "x2", "x3"]

    def test_no_bootstrap_dataset(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0)
        ds = result.to_dataset()
        assert set(ds.data_vars) == {"delta", "S1"}

    def test_multi_output_dataset(self, ishigami_data):
        X, Y = ishigami_data
        Y2 = jnp.stack([Y, Y**2], axis=1)
        result = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=0)
        ds = result.to_dataset()
        assert ds["delta"].dims == ("output", "param")
        assert ds["delta"].shape == (2, 3)


class TestPlischkeHeuristic:
    def test_matches_reference_class_counts(self):
        """Tier T2 (external library): class counts of the Plischke heuristic.

        Provenance:

        * Tier: T2.
        * Oracle: SALib, ``SALib.analyze.delta.analyze``, which applies the rule
          ``min(ceil(N ** (2 / (7 + tanh((1500 - N) / 500)))), 48)``.
        * Version: SALib 1.5.2.
        * Run on: 2026-08-18.
        * Script: ``scripts/oracles/salib_delta_class_counts.py``.

        SALib is a development extra, so the oracle runs locally and the values
        are typed in here as literals. The test then compares against fixed
        numbers and never against our own code.
        """
        expected = {
            100: 4,
            500: 5,
            1000: 6,
            1500: 9,
            5000: 18,
            10000: 22,
            100000: 47,
        }
        for n_samples, n_classes in expected.items():
            assert _plischke_n_classes(n_samples) == n_classes, n_samples

    def test_saturates_at_48(self):
        assert _plischke_n_classes(10**6) == 48


class TestDeltaRegression:
    """Regression coverage for confirmed code-review findings."""

    def test_rare_event_indicator_is_a_discrete_output(self):
        """A 0/1 indicator is discrete, so the estimator refuses it.

        ``analyze`` supports a continuous output only. A rare-event
        indicator has two atoms, which no output grid resolves.
        """
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=1000, seed=1))
        Y_np = np.zeros(1000)
        Y_np[np.random.default_rng(0).integers(1000)] = 1.0
        with pytest.raises(ValueError, match="continuous output distribution only"):
            analyze(ishigami.PROBLEM, X, jnp.asarray(Y_np), n_bootstrap=0)

    def test_rare_event_bias_correction_not_inflated(self):
        """A rare-event output must not make the bias correction inflate delta.

        Near-constant bootstrap resamples (frequent when Y is a rare-event
        magnitude) previously contributed spurious zero replicates, dragging
        ``mean(d_boot)`` below ``d_hat`` and pushing the corrected estimate
        *above* the plug-in value for an input whose true delta is ~0. The
        output carries a small continuous background so it stays a
        continuous distribution; the spike still dominates it by two orders
        of magnitude.
        """
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=1000, seed=1))
        rng = np.random.default_rng(0)
        Y_np = 0.01 * rng.standard_normal(1000)
        Y_np[rng.integers(1000)] = 1.0
        Y = jnp.asarray(Y_np)
        plug = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0)
        corrected = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=200, seed=0)
        c = np.asarray(corrected.delta)
        assert np.all(np.isfinite(c))
        assert np.all(c <= np.asarray(plug.delta) + 0.02)

    def test_constant_column_degenerate_replicates_stay_zero(self):
        """A constant column keeps delta = 0 through every bootstrap replicate.

        Every resample of a constant column is itself constant, so this
        exercises the degenerate-replicate branch of the bias correction.
        """
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=800, seed=2))
        Y = jnp.stack([ishigami.evaluate(X), jnp.ones(800)], axis=1)
        result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=32, seed=0)
        np.testing.assert_allclose(np.asarray(result.delta)[1], 0.0)
        assert result.delta_conf is not None
        np.testing.assert_allclose(np.asarray(result.delta_conf)[:, 1], 0.0)

    def test_ranking_uses_original_x_under_x64(self):
        """Ranks must be computed on the original X, not a Y-dtype downcast.

        Runs in a subprocess so enabling x64 cannot leak into the rest of
        the suite. With float64 X carrying structure below the float32 ulp
        and a float32 Y, the dominant input's delta must stay large.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import jax
            jax.config.update("jax_enable_x64", True)
            import numpy as np
            import jax.numpy as jnp
            from jaxgsa.benchmarks import ishigami
            from jaxgsa.borgonovo import analyze

            rng = np.random.default_rng(0)
            N = 1500
            x0 = 1e9 + rng.random(N)  # ulp ~64 at 1e9 >> unit spacing
            X = np.stack([x0, rng.random(N), rng.random(N)], axis=1)
            Y = np.sin(2 * np.pi * (x0 - 1e9))
            d32 = np.asarray(
                analyze(ishigami.PROBLEM, jnp.asarray(X),
                        jnp.asarray(Y, dtype=jnp.float32), n_bootstrap=0).delta
            )
            d64 = np.asarray(
                analyze(ishigami.PROBLEM, jnp.asarray(X),
                        jnp.asarray(Y, dtype=jnp.float64), n_bootstrap=0).delta
            )
            assert d32[0] > 0.3, d32
            assert np.allclose(d32, d64, atol=1e-4), (d32, d64)
            """
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


class TestDeltaBandwidthValidation:
    def test_unknown_string_bandwidth_message(self, ishigami_data):
        X, Y = ishigami_data
        # cast so the deliberately invalid value (a string other than
        # "silverman") exercises runtime validation without a static type error.
        with pytest.raises(ValueError, match="'silverman' or a positive float"):
            analyze(ishigami.PROBLEM, X, Y, bandwidth=cast(float, "scott"))

    def test_bool_bandwidth_rejected(self, ishigami_data):
        X, Y = ishigami_data
        # bool is an int subclass; cast so the runtime rejection is exercised
        # without a static type error.
        with pytest.raises(ValueError, match="'silverman' or a positive float"):
            analyze(ishigami.PROBLEM, X, Y, bandwidth=cast(float, True))

    def test_float_bandwidth_accepted(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, bandwidth=0.5, n_bootstrap=0)
        assert np.all(np.isfinite(np.asarray(result.delta)))


class TestDeltaYNdimValidation:
    def test_rejects_4d_y(self, ishigami_data):
        X, _ = ishigami_data
        with pytest.raises(ValueError, match="Y must be 1-D"):
            analyze(ishigami.PROBLEM, X, jnp.ones((X.shape[0], 2, 3, 4)))

    def test_rejects_0d_y(self, ishigami_data):
        X, _ = ishigami_data
        with pytest.raises(ValueError, match="Y must be 1-D"):
            analyze(ishigami.PROBLEM, X, jnp.asarray(3.0))


class TestDegenerateClassBandwidthFloor:
    def test_continuous_fixture_never_engages_floor(self, ishigami_data):
        """Smooth continuous data must not trip the degenerate-class floor.

        The floor predicate only fires for numerically zero class variance,
        so continuous results stay bit-identical to the un-floored kernel
        (verified against the 0.6.0 outputs during development).
        """
        X, Y = ishigami_data
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analyze(ishigami.PROBLEM, X, Y, n_bootstrap=8, seed=0)
        assert not any("bandwidth" in str(w.message) for w in caught)


class TestDegenerateBandwidthIsNotAPrecondition:
    """``degenerate_bandwidth`` is checked on the result, not on the setting.

    Tier T4 (internal consistency) for every test here, and that is the
    right tier: these are behavioural contracts about which configurations
    the estimator accepts, not numbers an external oracle could confirm.
    The delta values these tests compare are checked against SALib
    elsewhere in this file.

    A kernel narrower than one output-grid step does not by itself break
    the run. Two independent conditions have to hold first. The floor only
    reaches a class the estimator already found degenerate, and on smooth
    data no class is. Even on a degenerate class, aliasing needs a grid
    point to land on the spike. ``analyze`` therefore refuses the returned
    delta, never the setting.
    """

    @pytest.mark.parametrize(
        ("grid_size", "bandwidth"),
        [(100, 0.01), (100, 1e-8), (100, 0.1), (50, 0.3)],
    )
    def test_explicit_bandwidth_that_cannot_apply_is_accepted(
        self, ishigami_data, grid_size, bandwidth
    ):
        """Tier T4. A setting that cannot change the result must not raise.

        On smooth continuous data no conditioning class is degenerate, so
        the floor is never applied and ``degenerate_bandwidth`` is dead
        code for this run. Every value must therefore be accepted, and
        every result must be bit-identical to ``"auto"``. All four cases
        here were refused by the 0.8.0 up-front guard, including 0.1,
        which is the fraction ``"auto"`` itself uses.
        """
        X, Y = ishigami_data
        explicit = analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_bootstrap=0,
            grid_size=grid_size,
            degenerate_bandwidth=bandwidth,
        )
        auto = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0, grid_size=grid_size)
        np.testing.assert_array_equal(np.asarray(explicit.delta), np.asarray(auto.delta))
        np.testing.assert_array_equal(np.asarray(explicit.S1), np.asarray(auto.S1))

    def test_sub_grid_step_bandwidth_on_a_degenerate_class_can_still_work(
        self, degenerate_class_data
    ):
        """Tier T4. Even an applied floor below one grid step can be fine.

        This fixture does have a degenerate class, so the floor really is
        applied. One grid step is about 0.108 of the full-sample
        bandwidth. A floor of 0.05, less than half of that, still returns
        a delta within 0.01 of the ``"auto"`` answer. The width alone
        therefore does not decide whether the run fails, which is why
        ``analyze`` cannot refuse it up front.
        """
        problem, X, Y = degenerate_class_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto = analyze(problem, X, Y, n_bootstrap=0)
            narrow = analyze(problem, X, Y, n_bootstrap=0, degenerate_bandwidth=0.05)
        assert np.all(np.asarray(narrow.delta) >= 0.0)
        assert np.all(np.asarray(narrow.delta) <= 1.0)
        np.testing.assert_allclose(np.asarray(narrow.delta), np.asarray(auto.delta), atol=0.01)

    def test_aliased_floor_raises_and_names_the_knobs(self, degenerate_class_data):
        """Tier T4. A floor that does alias is caught on the returned delta.

        A floor of 1e-3 of the full-sample bandwidth is two orders of
        magnitude below one grid step, and this fixture's point mass sits
        exactly on a grid point, so the trapezoid rule integrates the
        spike and delta leaves [0, 1]. The message must name the setting
        that caused it, the grid step it is measured against, and the
        fraction that would fix it.
        """
        problem, X, Y = degenerate_class_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=r"delta is a half L1 distance") as excinfo:
                analyze(problem, X, Y, n_bootstrap=0, degenerate_bandwidth=1e-3)
        message = str(excinfo.value)
        assert "degenerate_bandwidth=0.001" in message
        assert "output-grid step" in message
        # The fix is the fraction of h_full that equals one grid step.
        assert "at least 0.108" in message
        assert "grid_size (currently 100)" in message
        assert "not clipped" in message

    def test_message_names_the_floored_column(self, degenerate_class_data):
        """Tier T4. With many output columns, the message says which one.

        Column 0 carries the point mass; column 1 is a smooth transform of
        the continuous parameter and has no degenerate class. Only column
        0 can be the one reported.
        """
        problem, X, Y = degenerate_class_data
        smooth = jnp.asarray(np.asarray(X)[:, 0] * 2.0 + 1.0)
        Y2 = jnp.stack([Y, smooth], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=r"delta is a half L1 distance") as excinfo:
                analyze(problem, X, Y2, n_bootstrap=0, degenerate_bandwidth=1e-3)
        assert "k=0" in str(excinfo.value)

    def test_advice_switches_when_no_class_was_floored(self, degenerate_class_data):
        """Tier T4. Without a floored class the floor is the wrong advice.

        ``degenerate_tol=1e-12`` stops the predicate firing, so the narrow
        class keeps its own bandwidth and the floor is never applied.
        Pointing the caller at ``degenerate_bandwidth`` would then be
        wrong, because that setting had no effect on this run. The message
        must name ``degenerate_tol`` instead.

        An exact point mass gets bandwidth 0, whose density is dropped;
        that under-counts delta rather than inflating it. Drive the
        failure from the other side instead, with a class narrow enough to
        alias but not exactly zero.
        """
        problem, X, Y = degenerate_class_data
        rng = np.random.default_rng(1)
        jitter = np.where(np.asarray(X)[:, 1] == 0, rng.normal(0.0, 1e-7, Y.shape[0]), 0.0)
        Y_jitter = jnp.asarray(np.asarray(Y) + jitter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=r"delta is a half L1 distance") as excinfo:
                analyze(problem, X, Y_jitter, n_bootstrap=0, degenerate_tol=1e-12)
        message = str(excinfo.value)
        assert "degenerate_tol=1e-12" in message
        assert "never applied" in message
        assert "grid_size (currently 100)" in message

    def test_auto_resolves_to_one_grid_step_when_the_grid_binds(self, degenerate_class_data):
        """Tier T4. The real property behind the old vacuous ``"auto"`` test.

        The old test called ``analyze`` with ``"auto"`` and asserted the
        result was finite. That could never fail: the guard returned early
        for ``"auto"`` before it compared anything.

        The property that matters is that ``"auto"`` resolves to
        ``max(0.1 * h_full, grid_step)``, so it is never below one grid
        step. Test it through the answer, not by restating the formula.
        On this fixture one grid step is about 0.108 of the full-sample
        bandwidth, so the grid-step term is the larger one and it is the
        term ``"auto"`` must pick. Passing that exact fraction explicitly
        must therefore reproduce the ``"auto"`` delta, while passing half
        of it must not: a floor that ignored the grid step would make the
        two agree.
        """
        problem, X, Y = degenerate_class_data
        y = np.asarray(Y, dtype=np.float32)
        n = y.shape[0]
        h_full = (0.75 * n) ** -0.2 * y.std(ddof=1)
        one_step = float((y.max() - y.min()) / 99 / h_full)
        assert one_step > _DEGENERATE_BW_FRACTION, "fixture must let the grid step bind"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto = analyze(problem, X, Y, n_bootstrap=0)
            at_step = analyze(problem, X, Y, n_bootstrap=0, degenerate_bandwidth=one_step)
            half_step = analyze(problem, X, Y, n_bootstrap=0, degenerate_bandwidth=one_step / 2)
        np.testing.assert_allclose(np.asarray(at_step.delta), np.asarray(auto.delta), rtol=1e-4)
        assert not np.allclose(np.asarray(half_step.delta), np.asarray(auto.delta), rtol=1e-4)

    def test_auto_floor_still_aliases_under_a_tiny_bandwidth_factor(self, degenerate_class_data):
        """Tier T4. The ``"auto"`` floor can alias when something else does.

        The ``"auto"`` floor is ``max(0.1 * h_full, grid_step)``, so it is
        never narrower than one grid step and cannot alias on its own.
        A second cause can still push delta out of ``[0, 1]``. A tiny
        ``bandwidth`` factor shrinks the full-sample bandwidth far below
        the grid step, so the *unconditional* density is a spike the grid
        cannot integrate, while the point-mass class is floored as usual.

        The message must then quote the floor it really applied, which is
        one grid step here, and name ``grid_size`` alone. It must not
        point at ``degenerate_bandwidth``, which is not the knob in this
        branch. Raising ``grid_size`` must also actually fix the run,
        which the last block checks.
        """
        problem, X, Y = degenerate_class_data
        y = np.asarray(Y, dtype=np.float32)
        one_step = float((y.max() - y.min()) / 99)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=r"delta is a half L1 distance") as excinfo:
                analyze(problem, X, Y, n_bootstrap=0, bandwidth=1e-3)
        message = str(excinfo.value)
        assert "max(0.1 * h_full, grid_step)" in message
        assert f"grid_step) = {one_step:.3g}" in message
        assert "at least one grid step wide" in message
        assert "grid_size (currently 100)" in message
        assert "Raise degenerate_bandwidth" not in message
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fixed = analyze(problem, X, Y, n_bootstrap=0, bandwidth=1e-3, grid_size=400)
        assert np.all(np.asarray(fixed.delta) >= 0.0)
        assert np.all(np.asarray(fixed.delta) <= 1.0)

    def test_constant_column_is_exempt(self, ishigami_data):
        """Tier T4. A constant column has no density to integrate.

        Its grid step and its bandwidth are both zero, so no class can
        fall below the tolerance and no floor is ever applied. Its exact
        answer is delta = S1 = 0, and that contract must survive any
        explicit bandwidth.
        """
        X, Y = ishigami_data
        Y2 = jnp.stack([jnp.ones(Y.shape[0]), jnp.ones(Y.shape[0])], axis=1)
        result = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=0, degenerate_bandwidth=1e-8)
        np.testing.assert_allclose(np.asarray(result.delta), 0.0)
