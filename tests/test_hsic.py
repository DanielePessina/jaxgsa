"""Tests for HSIC (kernel-based) sensitivity analysis."""

from __future__ import annotations

import inspect
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import single_precision_warning

from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks import ishigami, linear, sobol_g
from jaxgsa.hsic import analyze, indices
from jaxgsa.hsic._analyze import (
    _build_one_kernel,
    _linear_quantile_by_selection,
    _median_bandwidth_sq,
    _resolve_bandwidth_sq,
)
from jaxgsa.problem import Problem
from jaxgsa.sampling import monte_carlo


@pytest.fixture(scope="module")
def linear_hsic_result():
    """HSIC result for linear benchmark."""
    X = monte_carlo(linear.PROBLEM, n=1024, seed=42)
    Y = linear.evaluate(jnp.asarray(X))
    return analyze(linear.PROBLEM, jnp.asarray(X), Y, key=jax.random.key(42), n_perms=100)


@pytest.fixture(scope="module")
def ishigami_hsic_result():
    """HSIC result for Ishigami benchmark."""
    X = monte_carlo(ishigami.PROBLEM, n=1024, seed=42)
    Y = ishigami.evaluate(jnp.asarray(X))
    return analyze(ishigami.PROBLEM, jnp.asarray(X), Y, key=jax.random.key(42), n_perms=100)


@pytest.fixture(scope="module")
def sobol_g_hsic_result():
    """HSIC result for Sobol G-function benchmark."""
    X = monte_carlo(sobol_g.PROBLEM, n=1024, seed=42)
    Y = sobol_g.evaluate(jnp.asarray(X))
    return analyze(sobol_g.PROBLEM, jnp.asarray(X), Y, key=jax.random.key(42), n_perms=100)


class TestLinearHSIC:
    def test_ranking(self, linear_hsic_result):
        """Linear model: x3 most important, x1 least."""
        r2 = np.asarray(linear_hsic_result.R2_HSIC)
        assert r2[2] > r2[1] > r2[0]

    def test_r2_bounded(self, linear_hsic_result):
        r2 = np.asarray(linear_hsic_result.R2_HSIC)
        assert np.all(r2 >= 0.0)
        assert np.all(r2 <= 1.0 + 1e-6)

    def test_total_bounded(self, linear_hsic_result):
        """With augmented kernels, total indices should be in [0, 1] for additive models."""
        t = np.asarray(linear_hsic_result.T_HSIC)
        assert np.all(np.isfinite(t))
        assert np.all(t >= -0.1)
        assert np.all(t <= 1.0 + 0.1)

    def test_all_significant(self, linear_hsic_result):
        """All inputs are influential in the linear model."""
        p = np.asarray(linear_hsic_result.p_values)
        assert np.all(p < 0.05)

    def test_total_ranking(self, linear_hsic_result):
        """For additive model, total order should also rank x3 > x2 > x1."""
        t = np.asarray(linear_hsic_result.T_HSIC)
        assert t[2] > t[1] > t[0]


class TestIshigamiHSIC:
    def test_x1_strongest_hsic(self, ishigami_hsic_result):
        """x1 appears in both sin(x1) and B*x3^4*sin(x1), giving highest HSIC."""
        r2 = np.asarray(ishigami_hsic_result.R2_HSIC)
        assert r2[0] > r2[2]

    def test_total_finite(self, ishigami_hsic_result):
        t = np.asarray(ishigami_hsic_result.T_HSIC)
        assert np.all(np.isfinite(t))


class TestSobolGHSIC:
    def test_first_four_ranking(self, sobol_g_hsic_result):
        """First 4 inputs should be ranked x1 > x2 > x3 > x4."""
        r2 = np.asarray(sobol_g_hsic_result.R2_HSIC)
        assert r2[0] > r2[1] > r2[2] > r2[3]

    def test_negligible_inputs(self, sobol_g_hsic_result):
        """Inputs 5-8 (a=99) should have very small indices."""
        r2 = np.asarray(sobol_g_hsic_result.R2_HSIC)
        for i in range(4, 8):
            assert r2[i] < r2[0] * 0.1

    def test_total_ranking(self, sobol_g_hsic_result):
        """Total order should also rank x1 > x2 > x3 > x4."""
        t = np.asarray(sobol_g_hsic_result.T_HSIC)
        assert t[0] > t[1] > t[2] > t[3]


class TestMultiOutput:
    def test_linear_output_matches_scalar(self):
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n=1024, seed=8)
        Xj = jnp.asarray(X)
        Y_scalar = Xj @ jnp.array([1.0, 2.0, 3.0])
        Y_multi = jnp.column_stack([Y_scalar, jnp.sum(Xj**2, axis=1)])
        r_scalar = analyze(problem, Xj, Y_scalar, n_perms=50, key=jax.random.key(8))
        r_multi = analyze(problem, Xj, Y_multi, n_perms=50, key=jax.random.key(8))
        np.testing.assert_allclose(
            np.asarray(r_scalar.R2_HSIC),
            np.asarray(r_multi.R2_HSIC[0]),
            atol=1e-4,
        )


class TestBandwidthOverride:
    def test_different_bandwidth_different_result(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=256, seed=11)
        Xj = jnp.asarray(X)
        Y = Xj[:, 0] * 2.0 + Xj[:, 1]
        r1 = analyze(problem, Xj, Y, bandwidth=0.1, n_perms=20, key=jax.random.key(11))
        r2 = analyze(problem, Xj, Y, bandwidth=1.0, n_perms=20, key=jax.random.key(11))
        assert not np.allclose(np.asarray(r1.R2_HSIC), np.asarray(r2.R2_HSIC))

    def test_unit_multiplier_is_the_bare_median_heuristic(self):
        """Tier T0 (closed form): ``bandwidth=1.0`` scales nothing.

        ``1.0`` is the default because it is the identity of the convention:
        the resolved squared bandwidth is ``m**2 * median_sq``, and squaring
        and multiplying by one are both exact in IEEE arithmetic. So the
        default path is the median heuristic itself, bit for bit, and no
        recorded number moves because the keyword changed meaning.
        """
        x = jnp.asarray(np.random.default_rng(2).normal(size=97))
        np.testing.assert_array_equal(
            np.asarray(_resolve_bandwidth_sq(x, 1.0)),
            np.asarray(_median_bandwidth_sq(x)),
        )

    @pytest.mark.parametrize("multiplier", [0.25, 1.0, 4.0])
    def test_multiplier_scales_the_median_width(self, multiplier):
        """Tier T0 (closed form): the float multiplies, it does not replace.

        The property that makes HSIC scale-free is that every width carries
        the data's own scale. That holds only if the caller's number is a
        factor on the heuristic rather than a width in the data's units, so
        this is the algebraic statement of the convention.
        """
        x = jnp.asarray(np.random.default_rng(3).normal(size=64) * 17.0)
        np.testing.assert_allclose(
            float(_resolve_bandwidth_sq(x, multiplier)),
            multiplier**2 * float(_median_bandwidth_sq(x)),
            rtol=1e-12,
        )


class TestAffineInvariance:
    """``Y -> a*Y + b`` must not move an index, at any multiplier.

    The bandwidth is a multiplier on the median heuristic, and the heuristic
    tracks the spread of the data, so an affine change of units cancels out
    of the Gaussian kernel exactly. That is the whole reason the keyword is
    relative: under the older convention, where a float was an absolute
    width, the same rescaling moved ``R2_HSIC`` by 0.56 relative because the
    fixed width did not follow ``a``.
    """

    @staticmethod
    def _case():
        """Ishigami inputs and outputs, in float64 so the check has digits."""
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=256, seed=3), dtype=jnp.float64)
        return ishigami.PROBLEM, X, ishigami.evaluate(X)

    @pytest.mark.parametrize("multiplier", [1.0, 0.35, 3.0])
    def test_rescaling_the_output_changes_nothing(self, multiplier):
        """Tier T0 (closed form): the Gaussian kernel is unchanged by units.

        Both the default and an explicit multiplier are checked. The explicit
        one is the case that used to fail, so it is the point of the test: a
        caller who overrides the bandwidth must not thereby lose the
        invariance the default has.
        """
        with jax.enable_x64():
            problem, X, Y = self._case()
            key = jax.random.key(0)
            base = indices(problem, X, Y, n_perms=5, key=key, bandwidth=multiplier)
            moved = indices(problem, X, 3.7 * Y - 12.0, n_perms=5, key=key, bandwidth=multiplier)
        for a, b in zip(base, moved, strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12)

    def test_analyze_agrees_with_the_core_on_the_invariance(self):
        """Tier T4 (internal consistency): the checked path is invariant too.

        ``analyze`` adds a preamble but no arithmetic, so the invariance must
        survive it. This is the form a caller actually sees.
        """
        with jax.enable_x64():
            problem, X, Y = self._case()
            base = analyze(problem, X, Y, n_perms=5, key=jax.random.key(0), bandwidth=0.6)
            moved = analyze(
                problem, X, -2.5 * Y + 4.0, n_perms=5, key=jax.random.key(0), bandwidth=0.6
            )
        np.testing.assert_allclose(
            np.asarray(base.R2_HSIC), np.asarray(moved.R2_HSIC), rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(base.T_HSIC), np.asarray(moved.T_HSIC), rtol=1e-10, atol=1e-12
        )


class TestMedianBandwidth:
    """The median heuristic reads the median of the off-diagonal distances."""

    @pytest.mark.parametrize("n", [4, 5, 6, 257])
    def test_quantile_equals_upper_triangle_median(self, n):
        """Tier T1 (reference implementation): the index-adjusted quantile.

        The oracle is ``jnp.median(dists_sq[jnp.triu_indices(n, k=1)])``: the
        previous implementation of the median heuristic, re-expressed here as
        an explicit strict-upper-triangle median. That is a reference
        implementation, not a closed form, so this is T1 and not T0. It is
        still the strongest check in this file: it computes the same quantity
        a completely different way, over four sample sizes.

        The bandwidth is defined as the median of the ``(N^2 - N) / 2`` strict
        upper-triangle squared distances. ``_median_bandwidth_sq`` reaches it
        with a quantile of the full matrix instead, so the two must agree.
        ``jnp.quantile`` and ``jnp.median`` can round differently at an even
        element count, hence a relative tolerance and not exact equality. The
        parametrisation covers both parities of ``(N^2 - N) / 2`` and the
        smallest N the analyzer accepts.
        """
        x = jnp.asarray(np.random.default_rng(n).normal(size=n))
        dists_sq = (x[:, None] - x[None, :]) ** 2
        reference = jnp.median(dists_sq[jnp.triu_indices(n, k=1)])
        np.testing.assert_allclose(float(_median_bandwidth_sq(x)), float(reference), rtol=1e-6)

    @pytest.mark.parametrize("n", [4, 5, 33, 128])
    @pytest.mark.parametrize("scale", [1.0, 1e8])
    def test_selection_matches_the_sorting_quantile_bit_for_bit(self, n, scale):
        """Tier T1 (reference implementation): selection reproduces the sort.

        The bandwidth reads two order statistics of the ``(N, N)`` distance
        matrix. ``jnp.quantile`` gets them by sorting; the estimator gets them
        by bisecting on the float bit pattern, which is what makes HSIC fast.
        The oracle is therefore ``jnp.quantile`` itself, and the tolerance is
        zero: a selection that returned a neighbouring element, or blended it
        differently, would be a different bandwidth and a different index.
        The large scale is here because the blend can land on an exact tie
        between two float32 values, which is where a hand-written weighted
        sum stops agreeing with the library one.
        """
        x = jnp.asarray(np.random.default_rng(n).normal(size=n) * scale, dtype=jnp.float32)
        dists_sq = (x[:, None] - x[None, :]) ** 2
        q = (n**2 + n - 1) / (2 * (n**2 - 1))
        np.testing.assert_array_equal(
            np.asarray(_linear_quantile_by_selection(dists_sq, q)),
            np.asarray(jnp.quantile(dists_sq, q)),
        )

    @pytest.mark.parametrize(
        "poison", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
    )
    def test_non_finite_values_match_the_sorting_quantile(self, poison):
        """Tier T1 (reference implementation): non-finite parity with the sort.

        The selection bisects on the float bit pattern, which orders finite
        non-negative floats correctly but says nothing useful about NaN. The
        estimator therefore carries an explicit NaN branch, and infinities
        ride on the bit ordering alone. Both behaviours only matter because
        they must agree with ``jnp.quantile``, so ``jnp.quantile`` is the
        oracle and the tolerance is zero.

        This reaches the analyzer through invalid data, which is the only way
        a non-finite distance appears in practice.
        """
        values = jnp.asarray(np.abs(np.random.default_rng(0).normal(size=64)), dtype=jnp.float32)
        values = values.at[7].set(poison)
        q = 0.5
        expected = np.asarray(jnp.quantile(values, q))
        actual = np.asarray(_linear_quantile_by_selection(values, q))
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        if not np.isnan(expected):
            np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("n", [64, 257])
    def test_the_width_is_the_median_distance_not_the_rms_distance(self, n):
        """Tier T0 (closed form): sqrt of the median square is the median.

        The heuristic returns the median of the *squared* off-diagonal
        distances and the kernel uses it directly as ``sigma**2``, so the
        standard deviation in force is ``sqrt(median(d**2))``. That reads like
        a root-mean-square, and it is not one: squaring is monotone on
        distances, so it is the median distance itself. The docstring says
        "median pairwise distance", and this is what makes that true.

        The RMS is the foil. It is 50% larger here, so a test that only
        checked the median would pass against either quantity; asserting the
        gap is what gives this teeth.
        """
        x = np.random.default_rng(n).normal(size=n)
        upper = np.abs(x[:, None] - x[None, :])[np.triu_indices(n, 1)]
        sigma = float(np.sqrt(float(_median_bandwidth_sq(jnp.asarray(x)))))
        np.testing.assert_allclose(sigma, float(np.median(upper)), rtol=1e-7)
        assert float(np.sqrt(np.mean(upper**2))) > 1.3 * sigma

    def test_diagonal_zeros_would_bias_a_plain_median(self):
        """Tier T4 (internal consistency): the index adjustment does work.

        This test bites: a plain median of the full matrix includes the N
        diagonal zeros and returns a smaller bandwidth. If the quantile
        position were dropped to 0.5, the test above would fail.
        """
        x = jnp.asarray(np.random.default_rng(0).normal(size=65))
        dists_sq = (x[:, None] - x[None, :]) ** 2
        assert float(jnp.median(dists_sq)) < 0.99 * float(_median_bandwidth_sq(x))

    def test_the_resolved_bandwidth_is_squared_once(self):
        """Tier T0 (closed form): the kernel is exp(-d^2 / 2 sigma^2).

        ``sigma`` is the multiplier times the median-heuristic width, so the
        closed form is written against the resolved width rather than against
        the caller's number. A build that squared the multiplier twice, or
        forgot the heuristic, would miss this.
        """
        x = jnp.asarray([0.0, 1.0, 2.0])
        multiplier = 0.5
        sigma_sq = multiplier**2 * float(_median_bandwidth_sq(x))
        K = _build_one_kernel(x, multiplier)
        expected = np.exp(-np.array([0.0, 1.0, 4.0]) / (2.0 * sigma_sq))
        np.testing.assert_allclose(np.asarray(K[0]), expected, rtol=1e-6)


class TestSliceIndependence:
    """Analysing many slices together must not disturb any one of them."""

    @staticmethod
    def _four_slice_case():
        """Build a four-output problem with slices of unlike shape."""
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        Xj = jnp.asarray(monte_carlo(problem, n=256, seed=5))
        Y = jnp.column_stack(
            [
                linear.evaluate(Xj),
                Xj[:, 0] ** 2,
                jnp.sin(3.0 * Xj[:, 1]),
                jnp.sum(Xj, axis=1),
            ]
        )
        return problem, Xj, Y

    @pytest.mark.parametrize("column", [0, 1, 2, 3])
    def test_each_slice_equals_its_own_scalar_analysis(self, column):
        """Tier T4 (internal consistency): the slice map shares nothing.

        Every output slice is computed by the same atomic kernel, mapped one
        slice at a time, so a slice must return exactly what it returns when
        it is the only output there is. Exactly, not nearly: the map runs the
        same kernel on the same bits, and each slice draws its permutations
        from its own key, so any difference here would mean state leaking
        between slices.
        """
        problem, Xj, Y = self._four_slice_case()
        together = analyze(problem, Xj, Y, n_perms=20, key=jax.random.key(5))
        alone = analyze(problem, Xj, Y[:, column], n_perms=20, key=jax.random.key(5))
        for field in ("R2_HSIC", "T_HSIC", "p_values", "hsic_raw"):
            np.testing.assert_array_equal(
                np.asarray(getattr(together, field))[column],
                np.asarray(getattr(alone, field)),
            )

    def test_slice_count_does_not_change_a_shared_slice(self):
        """Tier T4 (internal consistency): adding outputs changes no index.

        The same first column analysed beside one other column and beside
        three must give the same answer. This is the regression guard for a
        future chunked map: a chunk width that reached the answer would show
        up here as soon as the two runs fell into different chunks.
        """
        problem, Xj, Y = self._four_slice_case()
        two = analyze(problem, Xj, Y[:, :2], n_perms=20, key=jax.random.key(5))
        four = analyze(problem, Xj, Y, n_perms=20, key=jax.random.key(5))
        np.testing.assert_array_equal(np.asarray(two.R2_HSIC)[0], np.asarray(four.R2_HSIC)[0])
        np.testing.assert_array_equal(np.asarray(two.T_HSIC)[0], np.asarray(four.T_HSIC)[0])


class TestNoBatchingKeyword:
    def test_batch_size_is_gone_from_both_entry_points(self):
        """Tier T4 (internal consistency): HSIC takes no batching keyword.

        The old ``batch_size`` row-blocked one kernel *build* while the
        resident stacks — ~(2D+1)*N^2 floats — stayed whole, so it never
        bounded peak memory and chunked-vmap was measured useless. A keyword
        that cannot do what its name promises is removed rather than kept;
        the memory story lives in the docstrings instead.
        """
        assert "batch_size" not in inspect.signature(analyze).parameters
        assert "batch_size" not in inspect.signature(indices).parameters
        for text in (analyze.__doc__, indices.__doc__):
            assert text is not None
            assert "N^2" in text  # the docstring carries the memory story


class TestReproducibility:
    def test_same_key_same_result(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=256, seed=14)
        Xj = jnp.asarray(X)
        Y = Xj[:, 0] + Xj[:, 1] ** 2
        r1 = analyze(problem, Xj, Y, key=jax.random.key(42), n_perms=50)
        r2 = analyze(problem, Xj, Y, key=jax.random.key(42), n_perms=50)
        np.testing.assert_array_equal(np.asarray(r1.p_values), np.asarray(r2.p_values))

    def test_different_key_may_differ(self):
        """Different keys should produce different permutation sequences."""
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n=256, seed=15)
        Xj = jnp.asarray(X)
        # x3 is independent of Y, so its p-value is not pinned at 0
        Y = Xj[:, 0] * 0.5
        r1 = analyze(problem, Xj, Y, key=jax.random.key(1), n_perms=50)
        r2 = analyze(problem, Xj, Y, key=jax.random.key(2), n_perms=50)
        # At least one p-value should differ between keys
        assert not np.array_equal(np.asarray(r1.p_values), np.asarray(r2.p_values))


class TestValidation:
    def test_n_perms_zero_raises(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="n_perms"):
            analyze(problem, jnp.ones((10, 1)), jnp.ones(10), n_perms=0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_bandwidth_must_be_finite_and_positive(self, bad):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="bandwidth"):
            analyze(problem, jnp.ones((10, 1)), jnp.ones(10), bandwidth=bad)

    def test_key_is_required(self):
        """The permutation p-values are random, so there is no default key."""
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        Xj = jnp.asarray(monte_carlo(problem, n=64, seed=1))
        with pytest.raises(ValueError, match="key is required"):
            analyze(problem, Xj, Xj[:, 0] * 2.0, n_perms=5)

    def test_too_few_samples_raises(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="N must be"):
            analyze(problem, jnp.ones((2, 1)), jnp.ones(2))


class TestSinglePrecisionWarning:
    """T4: float32 is called out, because the V-statistic cannot carry it.

    HSIC subtracts three sums of the same magnitude over the whole (N, N)
    kernel matrix. What is left is small, so cancellation decides most of the
    digits and the summation order decides the rest: reordering the sample
    rows moves an index by about 2e-4 relative at N = 2048 in float32, and by
    about 2e-13 in float64.
    """

    def _sample(self, n: int = 256):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        Xj = jnp.asarray(monte_carlo(problem, n=n, seed=31))
        return problem, Xj, jnp.sin(3.0 * Xj[:, 0]) + Xj[:, 1]

    def test_float32_warns(self):
        """The warning fires in float32, whatever the suite's ambient flag.

        ``jax.enable_x64(False)`` rather than the ambient default, because the
        x64 CI job runs this same file with ``JAX_ENABLE_X64=1``. Left
        implicit, this test would assert the *absence* of the warning there
        and stop checking the thing it is named for. Forcing the precision
        keeps one contract per test and checks both of them in both jobs.
        """
        with jax.enable_x64(False):
            problem, Xj, Y = self._sample()
            with single_precision_warning():
                analyze(problem, Xj, Y, n_perms=5, key=jax.random.key(0))

    def test_x64_is_silent(self):
        """And it is suppressed under x64 — the other half of the contract.

        Asserted, not skipped: suppressing the warning correctly is as much
        the behaviour as raising it.
        """
        with jax.enable_x64():
            problem, Xj, Y = self._sample()
            with single_precision_warning():
                analyze(problem, Xj, Y, n_perms=5, key=jax.random.key(0))

    def test_reordering_the_rows_is_the_thing_it_warns_about(self):
        """The claim in the warning, measured: float32 is order-dependent.

        A permutation of the sample cannot change HSIC, so any difference is
        the arithmetic alone. float64 keeps it at the noise floor; float32
        does not.

        Both precisions are named explicitly. The float32 half used to ride on
        the ambient flag, which made it pass for the wrong reason under the
        x64 CI job: both measurements ran in float64, came back at 1.3e-14,
        and the comparison of one against a hundred times itself failed. The
        two arms of a precision comparison have to set their own precision.
        """
        problem, Xj, Y = self._sample(n=512)
        perm = np.random.default_rng(0).permutation(512)

        def shift() -> float:
            X_np, Y_np = np.asarray(Xj, np.float64), np.asarray(Y, np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", JaxgsaWarning)
                straight = analyze(
                    problem,
                    jnp.asarray(X_np),
                    jnp.asarray(Y_np),
                    n_perms=5,
                    key=jax.random.key(0),
                )
                shuffled = analyze(
                    problem,
                    jnp.asarray(X_np[perm]),
                    jnp.asarray(Y_np[perm]),
                    n_perms=5,
                    key=jax.random.key(0),
                )
            a = np.asarray(straight.R2_HSIC, np.float64)
            b = np.asarray(shuffled.R2_HSIC, np.float64)
            return float(np.max(np.abs(a - b) / np.abs(a)))

        with jax.enable_x64(False):
            shift_32 = shift()
        with jax.enable_x64():
            shift_64 = shift()
        assert shift_64 < 1e-10
        assert shift_32 > 100.0 * shift_64


class TestToDataset:
    def test_timeseries_output(self):
        problem = Problem(names=("x1",), bounds=((0, 1),))
        X = monte_carlo(problem, n=128, seed=17)
        Xj = jnp.asarray(X)
        Y = jnp.ones((128, 2, 1)) * Xj[:, 0:1, None]
        result = analyze(problem, Xj, Y, n_perms=10, key=jax.random.key(17))
        ds = result.to_dataset(time_coords=[0.0, 0.5])
        assert "time" in ds.dims
        assert list(ds.coords["time"].values) == [0.0, 0.5]


class TestIndependentInput:
    def test_independent_input_low_hsic(self):
        """An input independent of the output should have low R2-HSIC."""
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=1024, seed=20)
        Xj = jnp.asarray(X)
        Y = jnp.sin(Xj[:, 0] * 6.0)
        result = analyze(problem, Xj, Y, n_perms=100, key=jax.random.key(20))
        r2 = np.asarray(result.R2_HSIC)
        assert r2[0] > r2[1] * 3.0
        p = np.asarray(result.p_values)
        assert p[0] < 0.05
        assert p[1] > 0.05

    def test_irrelevant_input_total_near_zero(self):
        """Total HSIC for an irrelevant input should be near zero."""
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=1024, seed=21)
        Xj = jnp.asarray(X)
        Y = Xj[:, 0]
        result = analyze(problem, Xj, Y, n_perms=20, key=jax.random.key(21))
        t = np.asarray(result.T_HSIC)
        assert t[0] > 0.5
        assert abs(t[1]) < 0.3


class TestSingleParamTotal:
    def test_d1_total_equals_one(self):
        """With D=1, the complement is ones → T_HSIC should be ~1.0."""
        problem = Problem(names=("x",), bounds=((0.0, 1.0),))
        X = monte_carlo(problem, n=512, seed=22)
        Xj = jnp.asarray(X)
        Y = jnp.sin(Xj[:, 0])
        result = analyze(problem, Xj, Y, n_perms=20, key=jax.random.key(22))
        t = np.asarray(result.T_HSIC)
        assert t[0] > 0.8


class TestZeroVarianceOutput:
    def test_constant_output_produces_nan(self):
        """Constant Y should produce NaN indices, consistent with the warning."""
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n=64, seed=23)
        Xj = jnp.asarray(X)
        Y = jnp.ones(64)
        result = analyze(problem, Xj, Y, n_perms=5, key=jax.random.key(23))
        r2 = np.asarray(result.R2_HSIC)
        t = np.asarray(result.T_HSIC)
        assert np.all(np.isnan(r2))
        assert np.all(np.isnan(t))


def _invalid_sample(n: int = 64, seed: int = 0):
    """Build a clean two-parameter HSIC sample for the on_invalid tests."""
    problem = Problem(names=("a", "b"), bounds=((0.0, 1.0), (0.0, 1.0)))
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 2))
    Y = np.sin(3.0 * X[:, 0]) + 0.5 * X[:, 1]
    return problem, jnp.asarray(X), jnp.asarray(Y)


class TestHSICInvalidPolicy:
    """T4 (behaviour): jaxgsa.hsic.analyze honours the shared on_invalid policy."""

    def test_raise_is_the_default_and_names_the_rows(self):
        """T4: a non-finite Y row refuses the analysis and says which row it is.

        Before 0.10 HSIC checked neither X nor Y for finiteness, so a failed
        model run reached the kernel matrices and came back as a NaN index
        with nothing said about where it came from.
        """
        problem, X, Y = _invalid_sample()
        Y = Y.at[11].set(jnp.nan)
        with pytest.raises(ValueError) as exc:
            analyze(problem, X, Y, n_perms=5)
        message = str(exc.value)
        assert "jaxgsa.hsic.analyze" in message
        assert "1 of 64 rows" in message
        assert "[11]" in message

    def test_propagate_warns_and_the_indices_go_non_finite(self):
        """T4: 'propagate' keeps the row, and the NaN reaches the indices.

        The Gaussian kernel and the centering both average over the sample,
        so one NaN output contaminates every entry. That is the intended
        outcome of the policy: loudly wrong beats quietly wrong.
        """
        problem, X, Y = _invalid_sample()
        Y = Y.at[11].set(jnp.nan)
        with pytest.warns(JaxgsaWarning, match="reaches the indices"):
            result = analyze(
                problem, X, Y, n_perms=5, key=jax.random.key(0), on_invalid="propagate"
            )
        assert result.invalid.policy == "propagate"
        assert result.invalid.unit_indices == (11,)
        assert not np.all(np.isfinite(np.asarray(result.R2_HSIC)))

    def test_drop_removes_the_row_and_returns_finite_indices(self):
        """T4: 'drop' analyzes the remainder and the indices come back usable."""
        problem, X, Y = _invalid_sample()
        Y = Y.at[11].set(jnp.nan)
        with pytest.warns(JaxgsaWarning, match="dropped 1 of 64 rows"):
            result = analyze(problem, X, Y, n_perms=5, key=jax.random.key(0), on_invalid="drop")
        assert result.invalid.n_kept == 63
        assert np.all(np.isfinite(np.asarray(result.R2_HSIC)))

    def test_a_bad_x_is_caught_and_named(self):
        """T4: a non-finite input is caught too, and the report says it was X."""
        problem, X, Y = _invalid_sample()
        X = X.at[4, 1].set(-jnp.inf)
        with pytest.raises(ValueError, match=r"in X\b") as exc:
            analyze(problem, X, Y, n_perms=5)
        assert "[4]" in str(exc.value)
        with pytest.warns(JaxgsaWarning):
            result = analyze(problem, X, Y, n_perms=5, key=jax.random.key(0), on_invalid="drop")
        assert result.invalid.sources == ("X",)


# ---------------------------------------------------------------------------
# indices(): the pure, transformable core
# ---------------------------------------------------------------------------


class TestIndicesCore:
    """The traceable core against the policy-carrying ``analyze``.

    Tier T4 throughout (behavioural contract). ``analyze`` is not an external
    oracle for the HSIC arithmetic -- that is checked against closed forms and
    independence cases elsewhere in this file. What is checked here is the
    split: the core must return exactly the numbers the full entry point
    reports, and it must survive the three transformations ``analyze`` cannot.
    """

    N = 128
    PERMS = 8

    @staticmethod
    def _sample(problem, n, seed):
        """Monte-Carlo inputs for ``problem`` as a device array."""
        return jnp.asarray(monte_carlo(problem, n=n, seed=seed))

    def _ishigami(self):
        """Inputs and scalar outputs for the Ishigami benchmark."""
        X = self._sample(ishigami.PROBLEM, self.N, 11)
        return ishigami.PROBLEM, X, ishigami.evaluate(X)

    def test_matches_analyze(self):
        """T4: all four arrays equal ``analyze``'s fields exactly."""
        problem, X, Y = self._ishigami()
        key = jax.random.key(3)

        out = indices(problem, X, Y, n_perms=self.PERMS, key=key)
        result = analyze(problem, X, Y, n_perms=self.PERMS, key=key)

        assert len(out) == 4
        fields = (result.R2_HSIC, result.T_HSIC, result.p_values, result.hsic_raw)
        for produced, expected in zip(out, fields, strict=True):
            np.testing.assert_array_equal(np.asarray(produced), np.asarray(expected))

    def test_matches_analyze_multi_output(self):
        """T4: an ``(N, T, K)`` output comes back at the caller's rank."""
        problem, X, base = self._ishigami()
        Y = jnp.stack(
            [
                jnp.stack([base, 2.0 * base + X[:, 0]], axis=-1),
                jnp.stack([base**2, X[:, 1]], axis=-1),
            ],
            axis=1,
        )
        assert Y.shape == (self.N, 2, 2)
        key = jax.random.key(5)

        out = indices(problem, X, Y, n_perms=self.PERMS, key=key)
        result = analyze(problem, X, Y, n_perms=self.PERMS, key=key)

        assert out[0].shape == (2, 2, problem.num_vars)
        fields = (result.R2_HSIC, result.T_HSIC, result.p_values, result.hsic_raw)
        for produced, expected in zip(out, fields, strict=True):
            np.testing.assert_array_equal(np.asarray(produced), np.asarray(expected))

    def test_matches_analyze_with_a_bandwidth_multiplier(self):
        """T4: the static ``bandwidth`` keyword changes nothing about the split.

        ``bandwidth`` is a Python scalar read while the graph is traced, so
        it is baked into the compiled kernel. The core must still agree with
        ``analyze`` under a non-default setting.
        """
        problem, X, Y = self._ishigami()
        key = jax.random.key(7)

        out = indices(problem, X, Y, n_perms=self.PERMS, key=key, bandwidth=0.3)
        result = analyze(problem, X, Y, n_perms=self.PERMS, key=key, bandwidth=0.3)

        fields = (result.R2_HSIC, result.T_HSIC, result.p_values, result.hsic_raw)
        for produced, expected in zip(out, fields, strict=True):
            np.testing.assert_array_equal(np.asarray(produced), np.asarray(expected))

    def test_offers_no_bootstrap_and_says_why(self):
        """T4: HSIC declares no bootstrap, and the reason is written down.

        The permutation p-values are the uncertainty statement. A row
        bootstrap of a V-statistic would repeat rows onto the kernel diagonal
        and bias the resampled index upward, so the absence is a decision and
        has to read as one.
        """
        from jaxgsa import hsic
        from jaxgsa.hsic import _analyze as hsic_module

        assert hsic.SPEC.bootstrap is None
        assert "n_bootstrap" not in inspect.signature(analyze).parameters
        assert "n_bootstrap" not in inspect.signature(indices).parameters
        # ... but a key is still required, because the permutation test draws.
        assert inspect.signature(analyze).parameters["key"] is not None

        for text in (hsic_module.__doc__, analyze.__doc__):
            assert text is not None
            assert "bootstrap" in text
            assert "V-statistic" in text

    def test_is_jittable(self):
        """T4: ``jit`` traces the core and returns the eager values.

        The tolerance is loose on purpose. An outer ``jit`` lets XLA fuse
        across the whole core instead of across each eagerly dispatched op,
        which reassociates the three large sums of the V-statistic. In
        float32 that cancellation is what the module docstring warns about,
        and it moves the last few digits. The claim under test is that the
        traced graph computes the same quantity, not that float32 addition is
        associative.
        """
        problem, X, Y = self._ishigami()
        key = jax.random.key(3)

        jitted = jax.jit(
            lambda inputs, outputs, k: indices(problem, inputs, outputs, n_perms=self.PERMS, key=k)
        )
        for traced, eager in zip(
            jitted(X, Y, key),
            indices(problem, X, Y, n_perms=self.PERMS, key=key),
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(traced), np.asarray(eager), rtol=1e-4, atol=1e-6)

    def test_is_vmappable(self):
        """T4: ``vmap`` maps the core over a batch of output vectors."""
        problem, X, base = self._ishigami()
        batch = jnp.stack([base, X[:, 0], X[:, 1] ** 2])
        key = jax.random.key(9)

        stacked = jax.vmap(
            lambda outputs: indices(problem, X, outputs, n_perms=self.PERMS, key=key)
        )(batch)

        assert stacked[0].shape == (3, problem.num_vars)
        for row, outputs in enumerate(batch):
            eager = indices(problem, X, outputs, n_perms=self.PERMS, key=key)
            for produced, expected in zip(stacked, eager, strict=True):
                np.testing.assert_allclose(
                    np.asarray(produced[row]), np.asarray(expected), rtol=1e-4, atol=1e-6
                )

    def test_jit_of_jacrev_of_the_full_chain(self):
        """T4: ``jit(jacrev(...))`` compiles a model-parameter-to-index chain.

        Every gradient here is taken at a frozen bandwidth, and there is no
        way to ask for anything else: the median heuristic selects an order
        statistic by bisecting bit patterns, which carries no derivative, and
        the caller's ``bandwidth`` is a multiplier on that same selection.
        ``test_the_median_heuristic_carries_no_derivative`` pins the zero this
        rests on.

        So a finite difference of this call is a *different* quantity: it
        moves the median along with the data, and the returned gradient does
        not. It is still worth comparing against, because the two agree to the
        size of the term AD omits — measured at 8% here — which is loose
        enough to admit the missing term and tight enough to catch a wrong
        sign or a mis-wired chain. The exact oracle is ``jacrev`` run eagerly,
        which the ``jit`` must reproduce to the last bit. Runs in float64
        because the finite difference needs the digits.
        """
        with jax.enable_x64():
            X = self._sample(linear.PROBLEM, 64, 4).astype(jnp.float64)

            def total_r2(a):
                """Sum of R2-HSIC for the model ``y = a * x1 + x2``."""
                Y = a * X[:, 0] + X[:, 1]
                return indices(
                    linear.PROBLEM,
                    X,
                    Y,
                    n_perms=4,
                    key=jax.random.key(0),
                    bandwidth=0.4,
                )[0].sum()

            grad = jax.jit(jax.jacrev(total_r2))(2.0)
            eager = jax.jacrev(total_r2)(2.0)

            step = 1e-5
            fd = (total_r2(2.0 + step) - total_r2(2.0 - step)) / (2 * step)

        assert np.isfinite(float(grad))
        np.testing.assert_allclose(float(grad), float(eager), rtol=1e-10)
        np.testing.assert_allclose(float(grad), float(fd), rtol=0.15)

    def test_the_median_heuristic_carries_no_derivative(self):
        """T4: the bandwidth is frozen in every gradient, and cannot be freed.

        This is the fact the ``bandwidth`` docstring promises, and it is worth
        a test of its own because the promise inverted when the keyword became
        a multiplier. Under the old absolute convention a caller could escape
        the median by naming a width; under this one every width is a multiple
        of the same selected order statistic, so the escape hatch is gone and
        the docstring has to say so.

        The zero is exact rather than approximate: the selection reads the
        value back through ``bitcast_convert_type``, which has no tangent at
        all. If anyone ever makes the selection differentiable, this fails,
        and the docstring must be rewritten with it.
        """
        with jax.enable_x64():
            X = self._sample(linear.PROBLEM, 64, 4).astype(jnp.float64)

            def median_sq(a):
                """Median-heuristic squared width of ``y = a * x1 + x2``."""
                return _median_bandwidth_sq(a * X[:, 0] + X[:, 1])

            assert float(jax.grad(median_sq)(2.0)) == 0.0

            step = 1e-5
            fd = (median_sq(2.0 + step) - median_sq(2.0 - step)) / (2 * step)

        # The zero is a choice, not a coincidence: the width really does move.
        assert abs(float(fd)) > 1e-3

    def test_reads_no_array_value_on_the_host(self):
        """T4: every array operation inside the core is traceable.

        A host read of ``X`` or ``Y`` would raise a tracer conversion error
        under an abstract trace, so tracing with ``eval_shape`` alone is the
        check: nothing concrete is available to read.
        """
        problem, X, Y = self._ishigami()

        shapes = jax.eval_shape(
            lambda inputs, outputs: indices(
                problem, inputs, outputs, n_perms=self.PERMS, key=jax.random.key(1)
            ),
            X,
            Y,
        )

        assert [s.shape for s in shapes] == [(problem.num_vars,)] * 4

    @pytest.mark.parametrize(
        ("label", "second_spec"),
        [
            ("uniform", (0.0, 1.0)),
            ("gaussian", {"dist": "gaussian", "mean": 0.3, "variance": 1.2}),
            (
                "truncated_gaussian",
                {"dist": "gaussian", "mean": 0.3, "variance": 1.2, "low": -1.5, "high": 2.0},
            ),
        ],
    )
    def test_x_is_traceable_for_every_marginal_family(self, label, second_spec):
        """T4: ``X`` traces whatever the marginals are, and jit changes nothing.

        ``X`` is a jit *argument* here, not a closure constant, so every
        column reaches ``cdf_to_unit_interval`` as a tracer. That is the only
        arrangement that exercises the transform under trace, and the two
        defects it used to hide were different. A truncated Gaussian went
        through ``scipy.stats.truncnorm.cdf``, which converts a tracer to a
        host array. Separately, every Gaussian -- truncated or not -- took
        ``float(jnp.sqrt(spec.variance))``, which returns a tracer inside a
        trace and raises on the ``float()``. So the untruncated case is not
        redundant with the truncated one: it is the only one that would have
        caught the second defect, and both passed eagerly.

        The uniform row is the control. It was always traceable, so it fails
        only if the fix broke the affine path.
        """
        problem = Problem.from_dict({"x1": (0.0, 1.0), "x2": second_spec})
        X = self._sample(problem, self.N, 21)
        Y = jnp.asarray(X[:, 0] + X[:, 1] ** 2)
        key = jax.random.key(13)

        def core(inputs, outputs):
            """The core with X and Y both live arguments."""
            return indices(problem, inputs, outputs, n_perms=self.PERMS, key=key)

        # Tracing alone proves no host read: a conversion would raise here.
        shapes = jax.eval_shape(core, X, Y)
        assert [s.shape for s in shapes] == [(2,)] * 4, label

        for traced, eager in zip(jax.jit(core)(X, Y), core(X, Y), strict=True):
            np.testing.assert_allclose(np.asarray(traced), np.asarray(eager), rtol=1e-4, atol=1e-6)

    def test_emits_no_warning_where_analyze_does(self):
        """T4: the core carries none of the preamble's diagnostics.

        ``analyze`` warns about single precision on every float32 run. The
        core says nothing: a warning is a policy decision, and it is the
        preamble's to make.
        """
        problem, X, Y = self._ishigami()

        with warnings.catch_warnings():
            warnings.simplefilter("error", JaxgsaWarning)
            indices(problem, X, Y, n_perms=self.PERMS, key=jax.random.key(1))
