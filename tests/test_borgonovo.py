"""Tests for Borgonovo delta sensitivity analysis."""

from __future__ import annotations

import inspect
import warnings
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _fixtures_givendata import BOOT

import jaxgsa
from jaxgsa import JaxgsaWarning
from jaxgsa._core.partition import build_partition_groups
from jaxgsa.benchmarks import gaussian_linear, ishigami
from jaxgsa.borgonovo import analyze
from jaxgsa.borgonovo._analyze import (
    _DEGENERATE_BW_FRACTION,
    _DEGENERATE_BW_TOL,
    _estimate_transient_bytes,
    _get_delta_kernel,
    _plischke_n_classes,
    _resolve_slice_chunks,
)
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


class TestDeltaBasic:
    def test_plain_result_properties(self, delta_ishigami):
        """Plug-in delta and S1 lie in [0, 1], and x1 and x2 outrank the weak x3."""
        delta = np.asarray(delta_ishigami.delta)
        s1 = np.asarray(delta_ishigami.S1)
        assert np.all(delta >= 0.0) and np.all(delta <= 1.0)
        assert np.all(s1 >= 0.0) and np.all(s1 <= 1.0)
        assert delta[0] > delta[2], "x1 should be more important than x3"
        assert delta[1] > delta[2], "x2 should be more important than x3"

    def test_the_key_decides_the_bootstrap(self, given_ishigami_xy, delta_ishigami_boot):
        """Tier T4. The same key gives the same result, a different key a different one."""
        X, Y = given_ishigami_xy
        same = analyze(
            ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, bias_correct=True, key=jax.random.key(0)
        )
        other = analyze(
            ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, bias_correct=True, key=jax.random.key(2)
        )
        assert delta_ishigami_boot.delta_conf is not None
        assert same.delta_conf is not None and other.delta_conf is not None
        np.testing.assert_array_equal(
            np.asarray(same.delta), np.asarray(delta_ishigami_boot.delta)
        )
        np.testing.assert_array_equal(
            np.asarray(same.delta_conf), np.asarray(delta_ishigami_boot.delta_conf)
        )
        assert not np.array_equal(np.asarray(other.delta_conf), np.asarray(same.delta_conf))

    def test_n_classes_override_changes_result(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        r_default = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0)
        r_coarse = analyze(ishigami.PROBLEM, X, Y, n_classes=4, n_bootstrap=0)
        assert not np.allclose(np.asarray(r_default.delta), np.asarray(r_coarse.delta))

    def test_slice_chunk_size_invariance(self, given_ishigami_xy):
        """Chunked column processing must not change the result."""
        X, Y = given_ishigami_xy
        Y2 = jnp.stack([Y, Y**2, jnp.sin(Y)], axis=1)
        r_full = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=BOOT, key=jax.random.key(0))
        r_chunked = analyze(
            ishigami.PROBLEM, X, Y2, n_bootstrap=BOOT, key=jax.random.key(0), slice_chunk_size=1
        )
        np.testing.assert_allclose(
            np.asarray(r_full.delta), np.asarray(r_chunked.delta), rtol=1e-6
        )


class TestGridTiling:
    """The output grid is evaluated in tiles to bound peak memory.

    Tiling is a memory decision only. Every grid point sums over its own
    class members and over nothing else, so the tile width changes no
    term of any sum and no sum reads a term that belongs to another one.
    That is the guarantee, and it is a guarantee about the algorithm.

    It is not a guarantee about the compiled code. The reduction runs
    over an array of shape ``(tile_width, n_samples)``, and XLA picks how
    to vectorize it from that whole shape, not from the reduced axis
    alone. A two-row tile and a hundred-row tile get different emitted
    loops, so the additions along the sample axis land in a different
    order and float32 addition is not associative. The tile width is the
    trip count, so the tile width reaches the last bit of the answer.

    So these tests compare with a tolerance, and the tolerance is
    ``rtol=1e-6``. That is roughly ten times the largest gap measured
    across the tile widths below, which sits at one to two float32 ULP of
    the values involved. It is also three or more orders of magnitude
    below what a real tiling bug costs: dropping a tile, mis-joining the
    tiles, or letting the padding survive changes a density by the share
    of the grid it touches, which is percent-level, not ULP-level. The
    tolerance passes reordered addition and fails everything else.
    """

    @staticmethod
    def _kernel_inputs(X, Y_cols, n_bootstrap=4, seed=0):
        """Build the arguments the jitted delta kernel takes."""
        n = X.shape[0]
        identity = jnp.arange(n, dtype=jnp.int32)[None, :]
        boot = jax.random.randint(
            jax.random.PRNGKey(seed), (n_bootstrap, n), 0, n, dtype=jnp.int32
        )
        all_idx = jnp.concatenate([identity, boot], axis=0)
        groups, _, _ = build_partition_groups(
            ishigami.PROBLEM, X, all_idx, _plischke_n_classes(n), ()
        )
        return Y_cols, all_idx, tuple(groups)

    @pytest.mark.parametrize("grid_chunk", [3, 7, 50])
    def test_tiled_grid_matches_the_untiled_result(self, given_ishigami_xy, grid_chunk):
        """Any tile width returns the untiled result to float32 rounding.

        The widths span the awkward cases on purpose: 3 is far narrower than
        any vectorization XLA would choose, 7 does not divide the 100-point
        grid and so exercises the padding, and 50 divides it exactly. See the
        class docstring for why this is a tolerance and not an equality. The
        rounding claim does not depend on N, so the test uses 1024 rows.
        """
        X, Y = given_ishigami_xy
        X, Y = X[:1024], Y[:1024]
        Y_cols = jnp.stack([Y, Y**2, jnp.sin(Y)], axis=1)
        args = self._kernel_inputs(X, Y_cols)
        untiled = _get_delta_kernel(100, None, 1e-2, None, 100)(*args)
        tiled = _get_delta_kernel(100, None, 1e-2, None, grid_chunk)(*args)
        for a, b in zip(untiled, tiled):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)

    @pytest.mark.parametrize("col_batch", [1, 2, 3])
    def test_column_batching_matches_the_vmapped_result(self, given_ishigami_xy, col_batch):
        """Walking the columns in lax.map batches returns the vmapped result.

        Three batch widths on five columns cover a batch of one, an even
        split with a remainder, and a remainder batch.
        """
        X, Y = given_ishigami_xy
        X, Y = X[:1024], Y[:1024]
        Y_cols = jnp.stack([Y, Y**2, jnp.sin(Y), jnp.cos(Y), Y**3], axis=1)
        args = self._kernel_inputs(X, Y_cols)
        full = _get_delta_kernel(100, None, 1e-2, None, 100)(*args)
        batched = _get_delta_kernel(100, None, 1e-2, None, 100, col_batch)(*args)
        for a, b in zip(full, batched):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize(
        ("n", "total", "grid", "budget_mib"),
        [
            (4096, 32, 100, 512),  # the batch divides the column count
            (4096, 37, 100, 512),  # a short last batch
            (20000, 40, 20, 8),  # a tight budget with a short last batch
            (20000, 16, 400, 512),  # a large grid
        ],
    )
    def test_compiled_temp_memory_matches_the_estimate(self, dtype, n, total, grid, budget_mib):
        """The estimate tracks the compiled temp memory from above and below.

        Compile only: nothing runs, so this costs no sample-sized memory.
        The estimate is what the batch width is chosen from. Temp memory
        above 1.1 times it means a budget can be overrun. Temp memory far
        below it means the estimate wastes budget, so a floor is checked too.
        """
        D, n_rep = 10, 2
        n_classes = _plischke_n_classes(n)
        pts = -(-n // n_classes)
        S = jax.ShapeDtypeStruct
        groups = (
            (S((n_rep, D, n_classes, pts + 1), jnp.int32), S((n_rep, D, n_classes), jnp.int32)),
        )
        Y = S((n, total), jax.dtypes.canonicalize_dtype(dtype))
        jaxgsa.config.set_memory_budget(budget_mib, unit="mib")
        try:
            cs, grid_chunk = _resolve_slice_chunks(groups, Y, grid, None)
        finally:
            jaxgsa.config.set_memory_budget(512, unit="mib")
        kernel = _get_delta_kernel(grid, None, _DEGENERATE_BW_TOL, None, grid_chunk, cs)
        compiled = kernel.lower(Y, S((n_rep, n), jnp.int32), groups).compile()
        temp = compiled.memory_analysis().temp_size_in_bytes
        # With x64 off a float64 struct still compiles as float32.
        itemsize = jnp.dtype(jax.dtypes.canonicalize_dtype(dtype)).itemsize
        estimate = _estimate_transient_bytes(groups, n, total, itemsize, cs)
        assert temp <= 1.1 * estimate
        assert temp >= 0.5 * estimate

    def test_memory_budget_narrows_the_working_set(self, given_ishigami_xy):
        """A small budget still returns the same answer.

        The budget changes how the work is cut up, never what it
        computes. Cutting it up reassociates the float32 reductions XLA
        emits, so this carries a tolerance for the same reason the tile
        width above does. The tolerance is looser here because a small
        budget moves the slice chunk as well as the tile width, and the
        slice chunk crosses the bootstrap replicates.
        """
        X, Y = given_ishigami_xy
        Y2 = jnp.stack([Y, Y**2, jnp.sin(Y)], axis=1)
        r_default = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=BOOT, key=jax.random.key(0))
        jaxgsa.config.set_memory_budget(1, unit="mib")
        try:
            r_small = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=BOOT, key=jax.random.key(0))
        finally:
            jaxgsa.config.set_memory_budget(512, unit="mib")
        np.testing.assert_allclose(
            np.asarray(r_default.delta), np.asarray(r_small.delta), rtol=1e-5, atol=1e-7
        )


class TestGaussianLinearBenchmark:
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

    def test_estimator_converges_to_analytical_delta_and_s1(self, gaussian_linear_xy):
        """Tier T0 (closed form): delta and S1 land near the analytical values.

        The KDE plug-in delta is biased upward, by a measured 0.011 to 0.019 at
        ``N = 4096``, so the delta tolerance is wider than the one a
        32000-row sample needs. The S1 estimate has no KDE bias.
        """
        X, Y = gaussian_linear_xy
        result = analyze(gaussian_linear.PROBLEM, X, Y, n_bootstrap=0)
        np.testing.assert_allclose(
            np.asarray(result.delta), gaussian_linear.ANALYTICAL_DELTA, atol=0.04
        )
        np.testing.assert_allclose(np.asarray(result.S1), gaussian_linear.ANALYTICAL_S1, atol=0.05)

    def test_analytical_sobol_structure(self):
        S1, ST, S2 = gaussian_linear.analytical_indices()
        np.testing.assert_allclose(S1, ST)
        np.testing.assert_allclose(S1.sum(), 1.0)
        assert np.all(np.isnan(np.diag(S2)))
        off_diag = S2[~np.eye(len(S1), dtype=bool)]
        np.testing.assert_allclose(off_diag, 0.0)


class TestDeltaBootstrap:
    def test_bootstrap_lower_leq_upper(self, delta_ishigami_boot):
        result = delta_ishigami_boot
        assert result.delta_conf is not None
        assert np.all(np.asarray(result.delta_conf[0]) <= np.asarray(result.delta_conf[1]) + 1e-6)
        assert result.S1_conf is not None
        assert np.all(np.asarray(result.S1_conf[0]) <= np.asarray(result.S1_conf[1]) + 1e-6)

    def test_bootstrap_without_a_key_raises(self, given_ishigami_xy):
        """Tier T4. Asking for a bootstrap without a key is refused.

        The refusal is tied to asking, not to calling: ``n_bootstrap``
        defaults to 0, so a plain call draws no randomness and needs no key.
        Both halves are asserted, because a version that demanded a key
        unconditionally would pass the first check and still be wrong.
        """
        X, Y = given_ishigami_xy
        with pytest.raises(ValueError, match="key is required"):
            analyze(ishigami.PROBLEM, X, Y, n_bootstrap=BOOT)
        analyze(ishigami.PROBLEM, X, Y)

    def test_gaussian_ci_is_centred_on_the_estimate(self, given_ishigami_xy):
        """Tier T0 (closed form): the Gaussian interval is symmetric.

        ``estimate +/- z * sd`` puts the point estimate exactly at the
        midpoint of the two endpoints, which the percentile interval does
        not do. Delta is bias-corrected, so the estimate the interval is
        centred on is the corrected one the result reports.
        """
        X, Y = given_ishigami_xy
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
        assert result.delta_conf is not None and result.S1_conf is not None
        delta_mid = 0.5 * (np.asarray(result.delta_conf[0]) + np.asarray(result.delta_conf[1]))
        np.testing.assert_allclose(delta_mid, np.asarray(result.delta), atol=1e-6)
        s1_mid = 0.5 * (np.asarray(result.S1_conf[0]) + np.asarray(result.S1_conf[1]))
        np.testing.assert_allclose(s1_mid, np.asarray(result.S1), atol=1e-6)

    def test_unknown_ci_method_rejected(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        with pytest.raises(ValueError, match="ci_method"):
            analyze(
                ishigami.PROBLEM,
                X,
                Y,
                n_bootstrap=BOOT,
                ci_method=cast(Any, "bca"),
                key=jax.random.key(0),
            )

    def test_corrected_delta_within_conf(self, delta_ishigami_boot):
        """The bias-corrected estimate sits inside its own interval."""
        result = delta_ishigami_boot
        delta = np.asarray(result.delta)
        assert result.delta_conf is not None
        assert np.all(delta >= np.asarray(result.delta_conf[0]) - 1e-6)
        assert np.all(delta <= np.asarray(result.delta_conf[1]) + 1e-6)

    def test_bias_correct_false_returns_plugin(self, given_ishigami_xy, delta_ishigami):
        X, Y = given_ishigami_xy
        r_uncorrected = analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_bootstrap=BOOT,
            bias_correct=False,
            key=jax.random.key(0),
        )
        np.testing.assert_allclose(
            np.asarray(r_uncorrected.delta), np.asarray(delta_ishigami.delta), rtol=1e-6
        )
        assert r_uncorrected.delta_conf is not None

    def test_bias_correction_is_two_plugin_minus_replicate_mean(
        self, given_ishigami_xy, delta_ishigami, delta_ishigami_boot
    ):
        """The corrected delta is ``2 * plugin - mean(raw replicates)`` (Plischke eq. 30).

        This is an exact identity, so it holds for any key. A check that the
        correction lowers delta would not: the measured bias is only 0.001 to
        0.01, and x2's bias is about 0, so with ``BOOT`` replicates the sign
        changes with the key.
        """
        X, Y = given_ishigami_xy
        raw = analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_bootstrap=BOOT,
            bias_correct=False,
            keep_replicates=True,
            key=jax.random.key(0),
            verbose=False,
        )
        assert raw.ci is not None and raw.ci.replicates is not None
        replicate_mean = np.mean(np.asarray(raw.ci.replicates["delta"]), axis=0)
        np.testing.assert_allclose(
            np.asarray(delta_ishigami_boot.delta),
            2.0 * np.asarray(delta_ishigami.delta) - replicate_mean,
            rtol=1e-5,
            atol=1e-6,
        )


class TestDeltaMultiOutput:
    def test_time_series_shape(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        Y3 = jnp.stack(
            [jnp.stack([Y, Y**2], axis=1), jnp.stack([Y + 1.0, jnp.abs(Y)], axis=1)],
            axis=1,
        )
        assert Y3.shape == (Y.shape[0], 2, 2)
        result = analyze(ishigami.PROBLEM, X, Y3, n_bootstrap=BOOT, key=jax.random.key(0))
        assert result.delta.shape == (2, 2, 3)
        assert result.delta_conf is not None
        assert result.delta_conf.shape == (2, 2, 2, 3)

    def test_multi_output_consistent_with_scalar(self, given_ishigami_xy, delta_ishigami_boot):
        """Column 0 of a multi-output run must equal the scalar-output run."""
        X, Y = given_ishigami_xy
        Y2 = jnp.stack([Y, Y**2], axis=1)
        r_multi = analyze(
            ishigami.PROBLEM, X, Y2, n_bootstrap=BOOT, bias_correct=True, key=jax.random.key(0)
        )
        np.testing.assert_allclose(
            np.asarray(r_multi.delta[0]), np.asarray(delta_ishigami_boot.delta), rtol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(r_multi.S1[0]), np.asarray(delta_ishigami_boot.S1), rtol=1e-6, atol=1e-7
        )


class TestDeltaEdgeCases:
    def test_constant_output_yields_zero(self, given_ishigami_xy):
        X, _ = given_ishigami_xy
        Y_const = jnp.ones(X.shape[0])
        result = analyze(ishigami.PROBLEM, X, Y_const, n_bootstrap=BOOT, key=jax.random.key(0))
        np.testing.assert_allclose(np.asarray(result.delta), 0.0)
        np.testing.assert_allclose(np.asarray(result.S1), 0.0)
        assert np.all(np.isfinite(np.asarray(result.delta)))

    def test_constant_input_column_yields_exact_zero(self, given_ishigami_xy, delta_ishigami):
        """A constant input cannot shift the output distribution: delta = S1 = 0.

        The plug-in estimate approaches the true 0 only by sampling noise
        (measured delta[0] ~ 0.043 on Ishigami), so the analysis masks the
        parameter to exactly 0 and warns. The other parameters must come back
        bit-identical, because their conditioning classes are unchanged.
        """
        X, Y = given_ishigami_xy
        ref = delta_ishigami
        X_const = X.at[:, 0].set(0.5)
        with pytest.warns(JaxgsaWarning, match="constant"):
            result = analyze(ishigami.PROBLEM, X_const, Y, n_bootstrap=0)
        delta = np.asarray(result.delta)
        s1 = np.asarray(result.S1)
        assert delta[0] == 0.0
        assert s1[0] == 0.0
        np.testing.assert_array_equal(delta[1:], np.asarray(ref.delta)[1:])
        np.testing.assert_array_equal(s1[1:], np.asarray(ref.S1)[1:])


class TestDeltaValidation:
    @pytest.mark.parametrize(
        ("case", "kwargs", "match"),
        [
            ("x_1d", {}, "2-D"),
            ("two_columns", {}, "parameters"),
            ("y_0d", {}, "Y must be 1-D"),
            ("full", {"n_classes": 1}, "n_classes"),
            ("full", {"bandwidth": -0.5}, "bandwidth"),
            ("full", {"bandwidth": cast(float, "scott")}, "'silverman' or a positive float"),
            ("full", {"bandwidth": cast(float, True)}, "'silverman' or a positive float"),
        ],
        ids=[
            "x_not_2d",
            "wrong_param_count",
            "y_0d",
            "n_classes",
            "negative_bandwidth",
            "unknown_string_bandwidth",
            "bool_bandwidth",
        ],
    )
    def test_bad_arguments_are_refused(self, given_ishigami_xy, case, kwargs, match):
        """Each malformed argument raises early and names what is wrong.

        The string and bool bandwidths are cast so the deliberately invalid
        values reach the runtime check without a static type error.
        """
        X, Y = given_ishigami_xy
        if case == "x_1d":
            X = jnp.ones(len(Y))
        elif case == "two_columns":
            X = X[:, :2]
        elif case == "y_0d":
            Y = jnp.asarray(3.0)
        with pytest.raises(ValueError, match=match):
            analyze(ishigami.PROBLEM, X, Y, **kwargs)

    def test_float_bandwidth_accepted(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        result = analyze(ishigami.PROBLEM, X, Y, bandwidth=0.5, n_bootstrap=0)
        assert np.all(np.isfinite(np.asarray(result.delta)))


class TestDeltaXarray:
    def test_no_bootstrap_dataset(self, delta_ishigami):
        ds = delta_ishigami.to_dataset()
        assert set(ds.data_vars) == {"delta", "S1"}
        assert list(ds.coords["param"].values) == ["x1", "x2", "x3"]


class TestPlischkeHeuristic:
    def test_matches_reference_class_counts(self):
        """Tier T2 (permissive-licence library, recorded values): Plischke class counts.

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


class TestDeltaRegression:
    """Regression coverage for confirmed code-review findings."""

    def test_rare_event_indicator_is_a_discrete_output(self):
        """A 0/1 indicator is discrete, so the estimator refuses it.

        ``analyze`` supports a continuous output only. A rare-event
        indicator has two atoms, which no output grid resolves.
        """
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=1000, seed=1))
        Y_np = np.zeros(1000)
        Y_np[np.random.default_rng(0).integers(1000)] = 1.0
        with pytest.raises(ValueError, match="continuous output distribution only"):
            analyze(ishigami.PROBLEM, X, jnp.asarray(Y_np), n_bootstrap=0)

    def test_discrete_guard_refuses_binary_output_at_small_n(self):
        """Tier T4. A binary output is refused at N=150, not just at large N.

        The old rule also required ``n_distinct < 0.01 * N``, which no
        small sample can satisfy: a binary output passed at N <= 200, and
        analyze returned a delta that describes ``grid_size`` rather than
        the model. The rule is now an average-multiplicity test.
        """
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=150, seed=2))
        binary = jnp.asarray(np.random.default_rng(1).integers(0, 2, size=150).astype(np.float64))
        with pytest.raises(ValueError, match="continuous output distribution only"):
            analyze(ishigami.PROBLEM, X, binary)

    def test_discrete_guard_refuses_ten_level_output_at_n_500(self):
        """Tier T4. A 10-level output is refused at N=500.

        Under the old fraction rule this needed N > 1000. At N=500 each
        level repeats 50 times on average, which continuity cannot do.
        """
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=500, seed=3))
        ten = jnp.asarray(np.random.default_rng(2).integers(0, 10, size=500).astype(np.float64))
        with pytest.raises(ValueError, match="continuous output distribution only"):
            analyze(ishigami.PROBLEM, X, ten)

    def test_discrete_guard_accepts_continuous_output_at_n_100(self):
        """Tier T4. A continuous output at N=100 is accepted.

        It has ~100 distinct values, above the 20-value cap, so it is
        structurally safe whatever the multiplicity term says.
        """
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=100, seed=4))
        result = analyze(ishigami.PROBLEM, X, ishigami.evaluate(X))
        assert np.all(np.isfinite(np.asarray(result.delta)))

    def test_discrete_guard_large_n_baseline_unchanged(self, given_ishigami_xy):
        """Tier T4. The large-N behavior of the guard does not move.

        The multiplicity rule is a strict superset of the old fraction
        rule (``d < 0.01*N`` implies ``5*d <= N``), so a rounded
        continuous output with many distinct values stays accepted, and
        the N=1000 rare-event refusal above stays a refusal.
        """
        X, Y = given_ishigami_xy
        rounded = jnp.round(Y, 1)
        n_distinct = np.unique(np.asarray(rounded)).size
        assert n_distinct > 20  # rounding to one decimal keeps many values
        result = analyze(ishigami.PROBLEM, X, rounded)
        assert np.all(np.isfinite(np.asarray(result.delta)))

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
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=1000, seed=1))
        rng = np.random.default_rng(0)
        Y_np = 0.01 * rng.standard_normal(1000)
        Y_np[rng.integers(1000)] = 1.0
        Y = jnp.asarray(Y_np)
        plug = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0)
        corrected = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=50, key=jax.random.key(0))
        c = np.asarray(corrected.delta)
        assert np.all(np.isfinite(c))
        assert np.all(c <= np.asarray(plug.delta) + 0.02)

    def test_constant_column_degenerate_replicates_stay_zero(self):
        """A constant column keeps delta = 0 through every bootstrap replicate.

        Every resample of a constant column is itself constant, so this
        exercises the degenerate-replicate branch of the bias correction.
        """
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=800, seed=2))
        Y = jnp.stack([ishigami.evaluate(X), jnp.ones(800)], axis=1)
        result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, key=jax.random.key(0))
        np.testing.assert_allclose(np.asarray(result.delta)[1], 0.0)
        assert result.delta_conf is not None
        np.testing.assert_allclose(np.asarray(result.delta_conf)[:, 1], 0.0)

    def test_ranking_uses_original_x_under_x64(self):
        """Ranks must be computed on the original X, not a Y-dtype downcast.

        With float64 X carrying structure below the float32 ulp and a
        float32 Y, the dominant input's delta must stay large.
        """
        with jax.enable_x64():
            rng = np.random.default_rng(0)
            N = 1500
            x0 = 1e9 + rng.random(N)  # ulp ~64 at 1e9 >> unit spacing
            X = jnp.asarray(
                np.stack([x0, rng.random(N), rng.random(N)], axis=1), dtype=jnp.float64
            )
            Y = np.sin(2 * np.pi * (x0 - 1e9))
            d32 = np.asarray(
                analyze(
                    ishigami.PROBLEM, X, jnp.asarray(Y, dtype=jnp.float32), n_bootstrap=0
                ).delta
            )
            d64 = np.asarray(
                analyze(
                    ishigami.PROBLEM, X, jnp.asarray(Y, dtype=jnp.float64), n_bootstrap=0
                ).delta
            )
        assert d32[0] > 0.3, d32
        np.testing.assert_allclose(d32, d64, atol=1e-4)


class TestDegenerateBandwidthIsNotAPrecondition:
    """``degenerate_bandwidth`` is checked on the result, not on the setting.

    Tier T4 (internal consistency) for every test here, and that is the
    right tier: these are behavioural contracts about which configurations
    the estimator accepts, not numbers an external oracle could confirm.
    The delta values are checked only through jaxgsa's own invariants.

    A kernel narrower than one output-grid step does not by itself break
    the run. Two independent conditions have to hold first. The floor only
    reaches a class the estimator already found degenerate, and on smooth
    data no class is. Even on a degenerate class, aliasing needs a grid
    point to land on the spike. ``analyze`` therefore refuses the returned
    delta, never the setting.
    """

    @pytest.mark.parametrize(
        ("grid_size", "bandwidth"),
        [(100, 0.1), (50, 0.3)],
    )
    def test_explicit_bandwidth_that_cannot_apply_is_accepted(
        self, given_ishigami_xy, grid_size, bandwidth
    ):
        """Tier T4. A setting that cannot change the result must not raise.

        On smooth continuous data no conditioning class is degenerate, so
        the floor is never applied and ``degenerate_bandwidth`` is dead
        code for this run. Every value must therefore be accepted, and
        every result must match ``"auto"`` to rounding. Both cases here
        were refused by the 0.8.0 up-front guard, including 0.1, which is
        the fraction ``"auto"`` itself uses.

        Not bit-identity: passing the bandwidth explicitly skips the
        arithmetic that derives it, so the two paths can land one unit in
        the last place apart. The tolerance has to be read in the dtype the
        test actually runs in, which is float32 unless something else
        enabled x64 first. One float32 ulp at a delta of 0.21 is 1.5e-8, so
        the earlier ``rtol=1e-12`` was a float64 figure (2.8e-17 measured)
        applied to a float32 result. It held on arm64, where the two paths
        agree exactly, and failed on x86-64, where they differ by that one
        ulp. ``rtol=1e-6`` leaves room for a few ulp and still refuses any
        difference that would change a reported digit.
        """
        X, Y = given_ishigami_xy
        explicit = analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_bootstrap=0,
            grid_size=grid_size,
            degenerate_bandwidth=bandwidth,
        )
        auto = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0, grid_size=grid_size)
        np.testing.assert_allclose(
            np.asarray(explicit.delta), np.asarray(auto.delta), rtol=1e-6, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(explicit.S1), np.asarray(auto.S1), rtol=1e-6, atol=0.0
        )

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
        assert "not clipped" in message

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

        The message must not point at ``degenerate_bandwidth``, which is
        not the knob in this branch. Raising ``grid_size`` must also
        actually fix the run, which the last block checks.
        """
        problem, X, Y = degenerate_class_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=r"delta is a half L1 distance") as excinfo:
                analyze(problem, X, Y, n_bootstrap=0, bandwidth=1e-3)
        message = str(excinfo.value)
        assert "at least one grid step wide" in message
        assert "Raise degenerate_bandwidth" not in message
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fixed = analyze(problem, X, Y, n_bootstrap=0, bandwidth=1e-3, grid_size=400)
        assert np.all(np.asarray(fixed.delta) >= 0.0)
        assert np.all(np.asarray(fixed.delta) <= 1.0)

    def test_constant_column_is_exempt(self, given_ishigami_xy):
        """Tier T4. A constant column has no density to integrate.

        Its grid step and its bandwidth are both zero, so no class can
        fall below the tolerance and no floor is ever applied. Its exact
        answer is delta = S1 = 0, and that contract must survive any
        explicit bandwidth.
        """
        X, Y = given_ishigami_xy
        Y2 = jnp.stack([jnp.ones(Y.shape[0]), jnp.ones(Y.shape[0])], axis=1)
        result = analyze(ishigami.PROBLEM, X, Y2, n_bootstrap=0, degenerate_bandwidth=1e-8)
        np.testing.assert_allclose(np.asarray(result.delta), 0.0)


class TestIndicesPureCore:
    def test_indices_is_a_named_tuple(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        out = jaxgsa.borgonovo.indices(ishigami.PROBLEM, X, Y)
        assert isinstance(out, jaxgsa.borgonovo.BorgonovoIndices)
        assert out._fields == ("delta", "S1")

    def test_scalar_error_names_the_method(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        with pytest.raises(ValueError, match=r"^jaxgsa\.borgonovo: grid_size"):
            jaxgsa.borgonovo.indices(ishigami.PROBLEM, X, Y, grid_size=1)

    """The transformable core ``borgonovo.indices``.

    Two claims. The core returns exactly what ``analyze`` reports on clean
    outputs, and it survives ``jit``, ``vmap`` and ``jit(jacrev(...))``,
    which ``analyze`` cannot.
    """

    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_matches_analyze(self, given_ishigami_xy, rank):
        """Tier T4: ``indices`` returns the ``delta`` and ``S1`` of ``analyze``.

        ``analyze`` delegates to the same kernel runner, so this guards the
        delegation wiring rather than the estimator maths. The output comes
        back at the caller's rank: ``(D,)``, ``(K, D)`` or ``(T, K, D)``.
        """
        X, Y = given_ishigami_xy
        Y2 = jnp.stack([Y, Y**2], axis=1)
        outputs = {1: Y, 2: Y2, 3: jnp.stack([Y2, Y2 + 1.0], axis=1)}[rank]
        result = analyze(ishigami.PROBLEM, X, outputs, grid_size=40)
        delta, S1 = jaxgsa.borgonovo.indices(ishigami.PROBLEM, X, outputs, grid_size=40)
        assert delta.shape == result.delta.shape
        np.testing.assert_array_equal(np.asarray(delta), np.asarray(result.delta))
        np.testing.assert_array_equal(np.asarray(S1), np.asarray(result.S1))

    def test_bias_correction_is_not_in_the_core(
        self, given_ishigami_xy, delta_ishigami, delta_ishigami_boot
    ):
        """Tier T4: the core is the plug-in estimate, never the corrected one.

        The Plischke correction needs bootstrap replicates, so it is policy
        and stays in ``analyze``. The core must therefore equal
        ``analyze(..., n_bootstrap=0)`` and differ from the corrected value.
        """
        X, Y = given_ishigami_xy
        delta, _ = jaxgsa.borgonovo.indices(ishigami.PROBLEM, X, Y)

        np.testing.assert_array_equal(np.asarray(delta), np.asarray(delta_ishigami.delta))
        assert not np.allclose(np.asarray(delta), np.asarray(delta_ishigami_boot.delta))
        assert "n_bootstrap" not in inspect.signature(jaxgsa.borgonovo.indices).parameters

    def test_is_jittable(self, given_ishigami_xy):
        """Tier T4: ``jit`` traces ``indices`` over both X and Y."""
        X, Y = given_ishigami_xy
        jitted = jax.jit(lambda x, y: jaxgsa.borgonovo.indices(ishigami.PROBLEM, x, y))
        delta_jit, S1_jit = jitted(X, Y)
        delta, S1 = jaxgsa.borgonovo.indices(ishigami.PROBLEM, X, Y)

        np.testing.assert_allclose(np.asarray(delta_jit), np.asarray(delta), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(S1_jit), np.asarray(S1), rtol=1e-5)

    def test_is_vmappable(self):
        """Tier T4: ``vmap`` maps ``indices`` over a batch of output vectors."""
        N = 512
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=N, seed=7))
        batch = jnp.stack([ishigami.evaluate(X), X[:, 0], X[:, 1] ** 2])

        mapped = jax.vmap(
            lambda y: jaxgsa.borgonovo.indices(ishigami.PROBLEM, X, y, grid_size=30)
        )(batch)

        assert mapped[0].shape == (3, ishigami.PROBLEM.n_params)
        for row, outputs in enumerate(batch):
            delta, S1 = jaxgsa.borgonovo.indices(ishigami.PROBLEM, X, outputs, grid_size=30)
            np.testing.assert_allclose(np.asarray(mapped[0][row]), np.asarray(delta), rtol=1e-4)
            np.testing.assert_allclose(np.asarray(mapped[1][row]), np.asarray(S1), rtol=1e-4)

    def test_jit_of_jacrev(self):
        """Tier T4: ``jit(jacrev(...))`` compiles a gradient through the core.

        The KDE, the trapezoid rule and the class weights are all smooth in
        Y, so a derivative of ``sum(delta)`` with respect to the outputs is
        defined. The rank partition is not differentiable in X, which is why
        the derivative is taken in Y.
        """
        with jax.enable_x64():
            N = 256
            X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=N, seed=3))
            Y = ishigami.evaluate(X)

            def total_delta(outputs):
                return jaxgsa.borgonovo.indices(ishigami.PROBLEM, X, outputs, grid_size=20)[
                    0
                ].sum()

            jac = jax.jit(jax.jacrev(total_delta))(Y)
            eager = jax.jacrev(total_delta)(Y)

        assert jac.shape == (N,)
        assert np.all(np.isfinite(np.asarray(jac)))
        np.testing.assert_allclose(np.asarray(jac), np.asarray(eager), rtol=1e-8)

    def test_categorical_is_refused_with_a_pointer_to_analyze(self, degenerate_class_data):
        """Tier T4: a categorical parameter cannot be traced, so the core says so.

        The categorical class layout is built on the host from the observed
        level counts, so neither the class count nor the padded width is
        known from a tracer. ``analyze`` is the supported route.
        """
        problem, X, Y = degenerate_class_data
        with pytest.raises(ValueError, match="continuous parameters only"):
            jaxgsa.borgonovo.indices(problem, X, Y)

    def test_no_policy_fires_in_the_core(self):
        """Tier T4: the core neither warns nor raises where ``analyze`` does.

        A constant output column trips the zero-variance warning in
        ``analyze``, and a two-valued one is refused outright as a discrete
        output. The core reports the same numbers for the constant case and
        computes the discrete one without complaint.
        """
        problem = Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0)})
        X = jnp.asarray(monte_carlo(problem, n_samples=400, seed=5))
        constant = jnp.full((400,), 2.5)
        discrete = jnp.where(X[:, 0] > 0.5, 1.0, 0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            delta, S1 = jaxgsa.borgonovo.indices(problem, X, constant)
            jaxgsa.borgonovo.indices(problem, X, discrete)

        np.testing.assert_array_equal(np.asarray(delta), np.zeros(2))
        np.testing.assert_array_equal(np.asarray(S1), np.zeros(2))
        with pytest.raises(ValueError, match="continuous output distribution"):
            analyze(problem, X, discrete)


class TestBiasCorrectFlag:
    """``bias_correct`` is a plain bool: False is the plain estimate always."""

    def test_default_is_false_and_bootstrap_only_adds_intervals(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        plain = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", JaxgsaWarning)
            boot = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, key=jax.random.key(0))
        np.testing.assert_allclose(np.asarray(boot.delta), np.asarray(plain.delta), rtol=1e-6)

    def test_true_changes_delta(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        key = jax.random.key(0)
        off = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, key=key)
        on = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, bias_correct=True, key=key)
        assert not np.allclose(np.asarray(on.delta), np.asarray(off.delta), rtol=1e-6)

    def test_true_without_bootstrap_raises(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        with pytest.raises(ValueError, match="jaxgsa.borgonovo: bias_correct=True"):
            analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0, bias_correct=True)

    def test_non_bool_raises_type_error(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        with pytest.raises(TypeError, match="bias_correct must be True or False"):
            analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0, bias_correct=None)  # type: ignore[arg-type]
