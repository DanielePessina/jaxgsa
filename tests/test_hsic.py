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
    _hsic_v,
    _median_bandwidth_sq,
    _pairwise_quantile_by_selection,
)
from jaxgsa.problem import Problem
from jaxgsa.sampling import monte_carlo


@pytest.fixture(scope="module")
def linear_hsic_result():
    """HSIC result for linear benchmark (``N = 512``, 30 permutations).

    The smallest possible p-value is ``1 / (n_perms + 1)``, so the
    ``p < 0.05`` check needs at least 20 permutations. 30 leaves margin.
    """
    X = monte_carlo(linear.PROBLEM, n_samples=512, seed=42)
    Y = linear.evaluate(jnp.asarray(X))
    return analyze(linear.PROBLEM, jnp.asarray(X), Y, key=jax.random.key(42), n_perms=30)


@pytest.fixture(scope="module")
def ishigami_hsic_result(given_ishigami_xy):
    """HSIC result for Ishigami benchmark (first 512 rows of the shared sample)."""
    X, Y = given_ishigami_xy
    return analyze(ishigami.PROBLEM, X[:512], Y[:512], key=jax.random.key(42), n_perms=10)


@pytest.fixture(scope="module")
def sobol_g_hsic_result():
    """HSIC result for Sobol G-function benchmark.

    ``N`` stays at 1024: the ``r2[2] > r2[3]`` check compares two small true
    indices (0.011 and 0.003) and sits close to the bias of a null input at
    smaller ``N``. No p-value is asserted, so 10 permutations are enough.
    """
    X = monte_carlo(sobol_g.PROBLEM, n_samples=1024, seed=42)
    Y = sobol_g.evaluate(jnp.asarray(X))
    return analyze(sobol_g.PROBLEM, jnp.asarray(X), Y, key=jax.random.key(42), n_perms=10)


class TestLinearHSIC:
    def test_ranking_bounds_and_significance(self, linear_hsic_result):
        """Linear model: x3 most important, x1 least, all significant, all bounded.

        With augmented kernels the total indices of an additive model lie in
        [0, 1] up to estimator noise, and they rank the inputs the same way.
        """
        r2 = np.asarray(linear_hsic_result.R2)
        assert r2[2] > r2[1] > r2[0]
        assert np.all(r2 >= 0.0) and np.all(r2 <= 1.0 + 1e-6)
        t = np.asarray(linear_hsic_result.T)
        assert np.all(np.isfinite(t))
        assert np.all(t >= -0.1) and np.all(t <= 1.0 + 0.1)
        assert t[2] > t[1] > t[0]
        assert np.all(np.asarray(linear_hsic_result.p_values) < 0.05)


class TestIshigamiHSIC:
    def test_x1_strongest_hsic(self, ishigami_hsic_result):
        """x1 appears in both sin(x1) and B*x3^4*sin(x1), giving highest HSIC."""
        r2 = np.asarray(ishigami_hsic_result.R2)
        assert r2[0] > r2[2]
        assert np.all(np.isfinite(np.asarray(ishigami_hsic_result.T)))


class TestSobolGHSIC:
    def test_ranking_and_negligible_inputs(self, sobol_g_hsic_result):
        """First 4 inputs rank x1 > x2 > x3 > x4, and inputs 5-8 (a=99) are negligible."""
        r2 = np.asarray(sobol_g_hsic_result.R2)
        assert r2[0] > r2[1] > r2[2] > r2[3]
        for i in range(4, 8):
            assert r2[i] < r2[0] * 0.1
        t = np.asarray(sobol_g_hsic_result.T)
        assert t[0] > t[1] > t[2] > t[3]


class TestAgainstNumpyVStatistic:
    """Tier T1 (reference implementation): ``analyze`` against plain NumPy.

    The oracle below builds every kernel from scratch in float64 NumPy. It
    uses the median heuristic with the strict upper-triangle median, the
    centred-kernel form of the V-statistic ``tr(K H L H) / n^2``, and the
    augmented kernels ``1 + center(K)`` for the total index. The package
    reaches the same numbers through a quantile of the full matrix and the
    expanded ``U/n^2 - 2V/n^3 + W/n^4`` form, so agreement checks the
    normalisation and ``T`` through the public API.
    """

    @staticmethod
    def _numpy_indices(X, y):
        n = len(y)

        def rbf(x):
            d2 = (x[:, None] - x[None, :]) ** 2
            sigma_sq = np.median(d2[np.triu_indices(n, 1)])
            return np.exp(-d2 / (2.0 * sigma_sq))

        def center(K):
            return K - K.mean(axis=1, keepdims=True) - K.mean(axis=0, keepdims=True) + K.mean()

        def hsic(K, L):
            return float(np.sum(center(K) * L)) / n**2

        L = rbf(y)
        Ks = [rbf(X[:, d]) for d in range(X.shape[1])]
        augmented = [1.0 + center(K) for K in Ks]
        full = np.prod(augmented, axis=0)
        raw = np.array([hsic(K, L) for K in Ks])
        r2 = raw / np.sqrt(np.array([hsic(K, K) for K in Ks]) * hsic(L, L))
        total = np.array(
            [
                1.0
                - hsic(np.prod([A for j, A in enumerate(augmented) if j != d], axis=0), L)
                / hsic(full, L)
                for d in range(len(Ks))
            ]
        )
        return r2, total, raw

    def test_r2_t_and_raw_hsic_match(self):
        with jax.enable_x64():
            X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=64, seed=1), dtype=jnp.float64)
            Y = ishigami.evaluate(X)
            result = analyze(ishigami.PROBLEM, X, Y, n_perms=5, key=jax.random.key(0))
        r2, total, raw = self._numpy_indices(np.asarray(X), np.asarray(Y))
        np.testing.assert_allclose(np.asarray(result.R2), r2, rtol=1e-8)
        np.testing.assert_allclose(np.asarray(result.T), total, rtol=1e-8)
        np.testing.assert_allclose(np.asarray(result.hsic_raw), raw, rtol=1e-8)


class TestMultiOutput:
    def test_linear_output_matches_scalar(self):
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        X = monte_carlo(problem, n_samples=256, seed=8)
        Xj = jnp.asarray(X)
        Y_scalar = Xj @ jnp.array([1.0, 2.0, 3.0])
        Y_multi = jnp.column_stack([Y_scalar, jnp.sum(Xj**2, axis=1)])
        r_scalar = analyze(problem, Xj, Y_scalar, n_perms=10, key=jax.random.key(8))
        r_multi = analyze(problem, Xj, Y_multi, n_perms=10, key=jax.random.key(8))
        np.testing.assert_allclose(
            np.asarray(r_scalar.R2),
            np.asarray(r_multi.R2[0]),
            atol=1e-4,
        )


class TestBandwidthOverride:
    def test_different_bandwidth_different_result(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n_samples=256, seed=11)
        Xj = jnp.asarray(X)
        Y = Xj[:, 0] * 2.0 + Xj[:, 1]
        r1 = analyze(problem, Xj, Y, bandwidth=0.1, n_perms=20, key=jax.random.key(11))
        r2 = analyze(problem, Xj, Y, bandwidth=1.0, n_perms=20, key=jax.random.key(11))
        assert not np.allclose(np.asarray(r1.R2), np.asarray(r2.R2))

    @pytest.mark.parametrize("multiplier", [0.25, 1.0, 4.0])
    def test_multiplier_scales_the_median_width(self, multiplier):
        """Tier T0 (closed form): the float multiplies, it does not replace.

        The property that makes HSIC scale-free is that every width carries
        the data's own scale. That holds only if the caller's number is a
        factor on the heuristic rather than a width in the data's units, so
        this is the algebraic statement of the convention.
        """
        x = jnp.asarray(np.random.default_rng(3).normal(size=64) * 17.0)
        # Square the multiplier in the data's own dtype, as the build does.
        # Rounding it through a Python float first would move the last bit of
        # sigma_sq and force a loose tolerance on an exact statement.
        sigma_sq = jnp.asarray(multiplier, dtype=x.dtype) ** 2 * _median_bandwidth_sq(x)
        direct = jnp.exp(-((x[:, None] - x[None, :]) ** 2) / (2.0 * sigma_sq))
        np.testing.assert_array_equal(
            np.asarray(_build_one_kernel(x, multiplier)),
            np.asarray(direct),
        )


class TestAffineInvariance:
    """``Y -> a*Y + b`` must not move an index, at any multiplier.

    The bandwidth is a multiplier on the median heuristic, and the heuristic
    tracks the spread of the data, so an affine change of units cancels out
    of the Gaussian kernel exactly. That is the whole reason the keyword is
    relative: under the older convention, where a float was an absolute
    width, the same rescaling moved ``R2`` by 0.56 relative because the
    fixed width did not follow ``a``.
    """

    @staticmethod
    def _case():
        """Ishigami inputs and outputs, in float64 so the check has digits."""
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=256, seed=3), dtype=jnp.float64)
        return ishigami.PROBLEM, X, ishigami.evaluate(X)

    @pytest.mark.parametrize("multiplier", [1.0, 0.35])
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
            np.asarray(base.R2), np.asarray(moved.R2), rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(np.asarray(base.T), np.asarray(moved.T), rtol=1e-10, atol=1e-12)


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

    @pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 9, 31, 50, 127, 128, 333])
    @pytest.mark.parametrize(
        "kind", ["normal", "ties", "constant", "dupes", "huge", "tiny", "negative"]
    )
    @pytest.mark.parametrize("x64", [False, True], ids=["f32", "f64"])
    def test_pairwise_selection_matches_the_sorting_quantile_bit_for_bit(self, n, kind, x64):
        """Tier T1 (reference implementation): no matrix, same order statistic.

        ``_pairwise_quantile_by_selection`` finds the median-heuristic order
        statistic from the sorted sample alone. The oracle is
        ``jnp.quantile`` over the explicit ``(N, N)`` matrix, which sorts it. The tolerance is
        zero, over odd and even sizes, ties, a constant column, heavy
        duplication, overflow to infinity, subnormal squares and negatives.
        """
        rng = np.random.default_rng(n)
        raw = {
            "normal": rng.normal(size=n),
            "ties": rng.integers(0, 3, size=n).astype(float),
            "constant": np.full(n, 0.7),
            "dupes": np.repeat(rng.normal(size=max(n // 4, 1)), 4)[:n],
            "huge": rng.normal(size=n) * 1e30,
            "tiny": rng.normal(size=n) * 1e-21,
            "negative": -np.abs(rng.normal(size=n)) - 5.0,
        }[kind]
        raw = np.resize(raw, n)
        q = (n**2 + n - 1) / (2 * (n**2 - 1))
        with jax.enable_x64(x64):
            x = jnp.asarray(raw, dtype=jnp.float64 if x64 else jnp.float32)
            dists_sq = (x[:, None] - x[None, :]) ** 2
            expected = np.asarray(jnp.quantile(dists_sq, q))
            actual = np.asarray(_pairwise_quantile_by_selection(x, q))
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(
        "poison", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
    )
    def test_pairwise_selection_is_nan_for_non_finite_input(self, poison):
        """Tier T1 (reference implementation): non-finite input gives NaN, as the matrix does.

        ``inf - inf`` on the diagonal puts a NaN in the distance matrix, so
        the reference returns NaN for any infinity as well as for NaN.
        """
        x = jnp.asarray(np.random.default_rng(0).normal(size=40), dtype=jnp.float32)
        x = x.at[7].set(poison)
        dists_sq = (x[:, None] - x[None, :]) ** 2
        q = (40**2 + 40 - 1) / (2 * (40**2 - 1))
        assert np.isnan(np.asarray(jnp.quantile(dists_sq, q)))
        assert np.isnan(np.asarray(_pairwise_quantile_by_selection(x, q)))

    @pytest.mark.slow
    def test_half_integer_position_matches_the_sorting_quantile(self):
        """Tier T1 (reference implementation): the plan dtype is the default float.

        ``jnp.quantile`` computes the target position ``q * (n - 1)`` in the
        dtype ``q`` promotes to -- the default float dtype -- and casts only
        the finished value back to the array's dtype. The selection has to
        copy that split.

        This is the case that tells the two apart. With x64 on, a
        ``float32`` matrix of ``4096**2`` entries puts the median position
        at ``8390655.5``, which ``float32`` cannot hold. Reading the plan in
        the *array's* dtype collapses it to the single rank ``8390655``,
        while ``jnp.quantile`` blends in ``float64``. A ``float32`` ramp
        keeps every squared distance exact (``4095**2 < 2**24``).
        """
        n = 4096
        with jax.enable_x64():
            x = jnp.arange(n, dtype=jnp.float32)
            dists_sq = (x[:, None] - x[None, :]) ** 2
            q = (n**2 + n - 1) / (2 * (n**2 - 1))
            expected = np.asarray(jnp.quantile(dists_sq, q))
            actual = np.asarray(_pairwise_quantile_by_selection(x, q))
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


class TestSliceIndependence:
    """Analysing many slices together must not disturb any one of them."""

    @staticmethod
    def _four_slice_case():
        """Build a four-output problem with slices of unlike shape."""
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        Xj = jnp.asarray(monte_carlo(problem, n_samples=256, seed=5))
        Y = jnp.column_stack(
            [
                linear.evaluate(Xj),
                Xj[:, 0] ** 2,
                jnp.sin(3.0 * Xj[:, 1]),
                jnp.sum(Xj, axis=1),
            ]
        )
        return problem, Xj, Y

    def test_slices_share_nothing(self):
        """Tier T4 (internal consistency): the slice map shares nothing.

        Every output slice is computed by the same atomic kernel, mapped one
        slice at a time, so a slice must return exactly what it returns when
        it is the only output there is. Exactly, not nearly: the map runs the
        same kernel on the same bits, and each slice draws its permutations
        from its own key, so any difference here would mean state leaking
        between slices. The same holds when the first column sits beside one
        other column instead of three: a chunk width that reached the answer
        would show up as soon as the two runs fell into different chunks.
        """
        problem, Xj, Y = self._four_slice_case()
        together = analyze(problem, Xj, Y, n_perms=20, key=jax.random.key(5))
        two = analyze(problem, Xj, Y[:, :2], n_perms=20, key=jax.random.key(5))
        for column in range(4):
            alone = analyze(problem, Xj, Y[:, column], n_perms=20, key=jax.random.key(5))
            for field in ("R2", "T", "p_values", "hsic_raw"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(together, field))[column],
                    np.asarray(getattr(alone, field)),
                )
        np.testing.assert_array_equal(np.asarray(two.R2)[0], np.asarray(together.R2)[0])
        np.testing.assert_array_equal(np.asarray(two.T)[0], np.asarray(together.T)[0])


class TestReproducibility:
    def test_the_key_decides_the_permutations(self):
        """The same key gives the same p-values, and a different key a different draw."""
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        Xj = jnp.asarray(monte_carlo(problem, n_samples=256, seed=15))
        # x2 and x3 are independent of Y, so their p-values are not pinned at the minimum.
        Y = Xj[:, 0] * 0.5
        r1 = analyze(problem, Xj, Y, key=jax.random.key(1), n_perms=20)
        r1_again = analyze(problem, Xj, Y, key=jax.random.key(1), n_perms=20)
        r2 = analyze(problem, Xj, Y, key=jax.random.key(2), n_perms=20)
        np.testing.assert_array_equal(np.asarray(r1.p_values), np.asarray(r1_again.p_values))
        assert not np.array_equal(np.asarray(r1.p_values), np.asarray(r2.p_values))


class TestValidation:
    @pytest.mark.parametrize(
        ("n_rows", "kwargs", "match"),
        [
            (10, {"n_perms": 0, "key": jax.random.key(0)}, "n_perms"),
            (10, {"bandwidth": 0.0, "key": jax.random.key(0)}, "bandwidth"),
            (10, {"bandwidth": -1.0, "key": jax.random.key(0)}, "bandwidth"),
            (10, {"bandwidth": float("nan"), "key": jax.random.key(0)}, "bandwidth"),
            (10, {"bandwidth": float("inf"), "key": jax.random.key(0)}, "bandwidth"),
            (2, {"key": jax.random.key(0)}, "N must be"),
        ],
        ids=[
            "n_perms_zero",
            "bandwidth_zero",
            "bandwidth_negative",
            "bandwidth_nan",
            "bandwidth_inf",
            "too_few_samples",
        ],
    )
    def test_bad_arguments_are_refused(self, n_rows, kwargs, match):
        """Each malformed argument raises early and names what is wrong.

        The permutation p-values are random, so there is no default key.
        """
        problem = Problem(names=("x",), bounds=((0, 1),))
        X = jnp.asarray(np.random.default_rng(0).uniform(size=(n_rows, 1)))
        with pytest.raises(ValueError, match=match):
            analyze(problem, X, X[:, 0] * 2.0, **kwargs)


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
        Xj = jnp.asarray(monte_carlo(problem, n_samples=n, seed=31))
        return problem, Xj, jnp.sin(3.0 * Xj[:, 0]) + Xj[:, 1]

    @pytest.mark.parametrize("x64", [False, True], ids=["float32_warns", "x64_is_silent"])
    def test_the_warning_follows_the_precision(self, x64):
        """The warning fires in float32 and is absent under x64.

        The precision is forced with ``jax.enable_x64`` rather than read from
        the ambient default, because the x64 CI job runs this same file with
        ``JAX_ENABLE_X64=1``. Left implicit, the float32 case would assert
        the *absence* of the warning there and stop checking what it is named
        for. ``single_precision_warning`` picks the matching contract.
        """
        with jax.enable_x64(x64):
            problem, Xj, Y = self._sample()
            with single_precision_warning():
                analyze(problem, Xj, Y, n_perms=5, key=jax.random.key(0))

    @pytest.mark.slow
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
            a = np.asarray(straight.R2, np.float64)
            b = np.asarray(shuffled.R2, np.float64)
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
        X = monte_carlo(problem, n_samples=128, seed=17)
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
        X = monte_carlo(problem, n_samples=256, seed=20)
        Xj = jnp.asarray(X)
        Y = jnp.sin(Xj[:, 0] * 6.0)
        result = analyze(problem, Xj, Y, n_perms=50, key=jax.random.key(20))
        r2 = np.asarray(result.R2)
        assert r2[0] > r2[1] * 3.0
        p = np.asarray(result.p_values)
        assert p[0] < 0.05
        assert p[1] > 0.05

    def test_irrelevant_input_total_near_zero(self):
        """Total HSIC for an irrelevant input should be near zero."""
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n_samples=256, seed=21)
        Xj = jnp.asarray(X)
        Y = Xj[:, 0]
        result = analyze(problem, Xj, Y, n_perms=10, key=jax.random.key(21))
        t = np.asarray(result.T)
        assert t[0] > 0.5
        assert abs(t[1]) < 0.3


class TestSingleParamTotal:
    def test_d1_total_equals_one(self):
        """With D=1, the complement is ones → T should be ~1.0."""
        problem = Problem(names=("x",), bounds=((0.0, 1.0),))
        X = monte_carlo(problem, n_samples=512, seed=22)
        Xj = jnp.asarray(X)
        Y = jnp.sin(Xj[:, 0])
        result = analyze(problem, Xj, Y, n_perms=20, key=jax.random.key(22))
        t = np.asarray(result.T)
        assert t[0] > 0.8


class TestZeroVarianceOutput:
    def test_constant_output_produces_nan(self):
        """Constant Y should produce NaN indices, consistent with the warning."""
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X = monte_carlo(problem, n_samples=64, seed=23)
        Xj = jnp.asarray(X)
        Y = jnp.ones(64)
        result = analyze(problem, Xj, Y, n_perms=5, key=jax.random.key(23))
        r2 = np.asarray(result.R2)
        t = np.asarray(result.T)
        assert np.all(np.isnan(r2))
        assert np.all(np.isnan(t))


class TestConstantInput:
    def test_constant_input_column_yields_exact_zero(self):
        """A constant input has provably zero dependence: R2 is exactly 0.

        A constant column's Gaussian kernel is identically 1, so the
        self-HSIC vanishes and the normalized index is the ratio of two
        round-off residues — before the fix it came out as spurious values
        above 1 (R2 = 1.31 at N = 100). The kernel now masks those
        parameters to exactly 0, matching PAWN, and the other parameters are
        unaffected.
        """
        problem = Problem(names=("x1", "x2", "x3"), bounds=((0, 1), (0, 1), (0, 1)))
        rng = np.random.default_rng(0)
        n = 128
        X = rng.uniform(size=(n, 3))
        X[:, 0] = 0.5
        Y = rng.normal(size=n)
        with pytest.warns(JaxgsaWarning, match="constant"):
            result = analyze(
                problem, jnp.asarray(X), jnp.asarray(Y), n_perms=20, key=jax.random.key(0)
            )
        r2 = np.asarray(result.R2)
        assert r2[0] == 0.0
        assert np.all(r2[1:] >= -1e-3)
        assert np.all(r2[1:] <= 1.0)


def _population_hsic_gaussian(sx2, sy2, rho, gx2, gy2):
    """Closed-form population HSIC of a jointly Gaussian pair, Gaussian kernels.

    ``(X, Y)`` are jointly Gaussian, mean 0, ``Var(X) = sx2``, ``Var(Y) =
    sy2``, ``Corr(X, Y) = rho``. The kernels are Gaussian RBF with *fixed*
    (not median-heuristic) squared bandwidths ``gx2``, ``gy2``.

    The population HSIC is
    ``E[k(X,X')l(Y,Y')] - 2*E[f(X)g(Y)] + E[k(X,X')]*E[l(Y,Y')]``,
    where ``(X,Y)`` and ``(X',Y')`` are iid copies of the joint, ``f(x) =
    E_x'[k(x,x')]``, ``g(y) = E_y'[l(y,y')]``. Every expectation here is over
    jointly Gaussian variables against a Gaussian kernel, so each reduces to
    the moment-generating function of a Gaussian quadratic form,
    ``E[exp(-Z^T A Z)] = det(I + 2*A*Sigma)^(-1/2)`` for ``Z ~ N(0, Sigma)``:

    - The first term integrates out ``U = X - X'``, ``V = Y - Y'``, which are
      jointly Gaussian with covariance ``2*Sigma``.
    - ``f`` and ``g`` are themselves Gaussian functions of ``x`` and ``y``
      (a Gaussian density convolved with a Gaussian kernel is Gaussian), so
      the middle term integrates ``f(X)g(Y)`` against the original ``Sigma``.
    - The last term is the ``D = 1`` case of the same formula, applied to
      ``X - X'`` and ``Y - Y'`` separately.

    Verified against a Monte Carlo V-statistic built from scratch with fixed
    bandwidths (not the median heuristic). One draw is a poor check on its
    own: at ``sx2, sy2, rho = 1, 2, 0.6`` and ``n = 4000`` the V-statistic's
    relative spread over 40 seeds is 5%, so a single draw agreeing to 1% is
    luck. Averaging 12 independent draws at ``n = 1500`` on the config the
    test below uses cuts that spread to 1.0% relative, with the largest
    deviation over 40 meta-seeds at 2.6%. The averaged estimate then centres
    on the closed form to +0.15%, which is the check that the formula is
    right.

    Args:
        sx2: Population variance of X.
        sy2: Population variance of Y.
        rho: Correlation of X and Y, in (-1, 1).
        gx2: Fixed squared bandwidth of the X kernel.
        gy2: Fixed squared bandwidth of the Y kernel.

    Returns:
        The population HSIC, a Python float.
    """

    def _term(a_diag, sigma):
        a = np.diag(a_diag)
        m = np.eye(2) + 2 * a @ sigma
        return 1.0 / np.sqrt(np.linalg.det(m))

    sxy = rho * np.sqrt(sx2 * sy2)
    sigma = np.array([[sx2, sxy], [sxy, sy2]])
    term1 = _term([1 / (2 * gx2), 1 / (2 * gy2)], 2 * sigma)

    ek = np.sqrt(gx2) / np.sqrt(gx2 + 2 * sx2)
    el = np.sqrt(gy2) / np.sqrt(gy2 + 2 * sy2)
    term2 = ek * el

    ax, ay = sx2 + gx2, sy2 + gy2
    cross_scale = (np.sqrt(gx2) / np.sqrt(ax)) * (np.sqrt(gy2) / np.sqrt(ay))
    cross = cross_scale * _term([1 / (2 * ax), 1 / (2 * ay)], sigma)

    return float(term1 - 2 * cross + term2)


class TestClosedFormGaussianPair:
    """Tier T0: the empirical V-statistic against a genuine population closed form.

    A jointly Gaussian pair under Gaussian kernels is one of the few cases
    where the *population* HSIC has a closed form (every expectation is a
    Gaussian quadratic-form moment-generating function; see
    :func:`_population_hsic_gaussian`). The bandwidth has to be fixed rather
    than the median heuristic jaxgsa always uses, so this calls the internal
    kernel-build and V-statistic pieces directly rather than going through
    ``analyze`` or ``indices``. That is what makes this T0: :class:`TestUnbiasedV`
    and the others near it check the estimator's own consistency (T4), and
    the earlier ``bandwidth``/kernel tests are closed-form on the kernel
    build alone, not on the resulting HSIC value.
    """

    @pytest.mark.parametrize(
        ("n", "n_draws", "tol"),
        [
            pytest.param(800, 3, 0.15, id="cheap"),
            pytest.param(1500, 12, 0.05, marks=pytest.mark.slow, id="tight"),
        ],
    )
    def test_v_statistic_matches_the_population_closed_form(self, n, n_draws, tol):
        """The averaged V-statistic lands on the population target.

        A single V-statistic draw is far too noisy to pin the closed form
        down: at ``n = 4000`` its relative spread is 5%, so a 2% tolerance on
        one draw passes or fails on the seed. The tight case averages 12
        independent draws at ``n = 1500``, which measures at 1.0% relative
        spread (2.6% worst case over 40 meta-seeds), and asserts 5%. The
        cheap case averages 3 draws at ``n = 800``. One draw has about 11%
        relative spread there, so three have about 6%. Four groups of three
        seeds measured at most 3.6% off, and the bound is 15%.
        """
        sx2 = sy2 = 1.0
        rho = 0.99
        gx2 = gy2 = 1.0
        target = _population_hsic_gaussian(sx2, sy2, rho, gx2, gy2)

        with jax.enable_x64():
            chol = np.linalg.cholesky(
                np.array([[sx2, rho * np.sqrt(sx2 * sy2)], [rho * np.sqrt(sx2 * sy2), sy2]])
            )
            draws = []
            for seed in range(n_draws):
                z = np.random.default_rng(seed).standard_normal((n, 2)) @ chol.T
                x = jnp.asarray(z[:, 0])
                y = jnp.asarray(z[:, 1])
                k = jnp.exp(-((x[:, None] - x[None, :]) ** 2) / (2.0 * gx2))
                ly = jnp.exp(-((y[:, None] - y[None, :]) ** 2) / (2.0 * gy2))
                draws.append(float(_hsic_v(k, ly)))
        got = float(np.mean(draws))

        assert abs(got - target) / target < tol


# ---------------------------------------------------------------------------
# indices(): the pure, transformable core
# ---------------------------------------------------------------------------


class TestIndicesCore:
    """The traceable core against the policy-carrying ``analyze``.

    Tier T4 throughout (behavioural contract). ``analyze`` is not an external
    oracle for the HSIC arithmetic -- that is checked against a genuine
    population closed form in :class:`TestClosedFormGaussianPair` and
    against independence cases elsewhere in this file. What is checked here
    is the split: the core must return exactly the numbers the full entry
    point reports, and it must survive the three transformations ``analyze``
    cannot.
    """

    N = 128
    PERMS = 8

    @staticmethod
    def _sample(problem, n, seed):
        """Monte-Carlo inputs for ``problem`` as a device array."""
        return jnp.asarray(monte_carlo(problem, n_samples=n, seed=seed))

    def _ishigami(self):
        """Inputs and scalar outputs for the Ishigami benchmark."""
        X = self._sample(ishigami.PROBLEM, self.N, 11)
        return ishigami.PROBLEM, X, ishigami.evaluate(X)

    @pytest.mark.parametrize("case", ["scalar", "multi_output", "bandwidth"])
    def test_matches_analyze(self, case):
        """T4: all four arrays equal ``analyze``'s fields exactly.

        Three variants share one test. ``multi_output`` has an ``(N, T, K)``
        output, which comes back at the caller's rank. ``bandwidth`` uses a
        non-default multiplier: it is a Python scalar read while the graph is
        traced and baked into the compiled kernel, so the core must still
        agree with ``analyze`` there.

        The core also carries none of the preamble's diagnostics.
        ``analyze`` warns about single precision on every float32 run, and
        the core says nothing, because a warning is the preamble's decision.
        """
        problem, X, base = self._ishigami()
        kwargs = {"bandwidth": 0.3} if case == "bandwidth" else {}
        Y = base
        if case == "multi_output":
            Y = jnp.stack(
                [
                    jnp.stack([base, 2.0 * base + X[:, 0]], axis=-1),
                    jnp.stack([base**2, X[:, 1]], axis=-1),
                ],
                axis=1,
            )
            assert Y.shape == (self.N, 2, 2)
        key = jax.random.key(3)

        with warnings.catch_warnings():
            warnings.simplefilter("error", JaxgsaWarning)
            out = indices(problem, X, Y, n_perms=self.PERMS, key=key, **kwargs)
        result = analyze(problem, X, Y, n_perms=self.PERMS, key=key, **kwargs)

        assert len(out) == 4
        assert out[0].shape == result.R2.shape
        fields = (result.R2, result.T, result.p_values, result.hsic_raw)
        for produced, expected in zip(out, fields, strict=True):
            np.testing.assert_array_equal(np.asarray(produced), np.asarray(expected))

    def test_offers_no_bootstrap_and_no_batching(self):
        """T4: HSIC declares no bootstrap and no batching keyword.

        The permutation p-values are the uncertainty statement. A row
        bootstrap of a V-statistic would repeat rows onto the kernel diagonal
        and bias the resampled index upward, so the absence is a decision.
        The old ``batch_size`` row-blocked one kernel *build* while the
        resident stacks stayed whole, so it never bounded peak memory and it
        was removed. A key is still required, because the permutation test
        draws.
        """
        from jaxgsa import hsic

        assert hsic._SPEC.bootstrap is None
        for fn in (analyze, indices):
            params = inspect.signature(fn).parameters
            assert "n_bootstrap" not in params
            assert "batch_size" not in params
            assert "key" in params

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
        """T4: ``vmap`` maps the core over a batch of output vectors.

        The tolerance here is ``rtol=1e-3, atol=1e-5``, and it is looser
        than the ``jit`` test above by design rather than by accident.

        ``vmap`` gives every ``(N, N)`` reduction in the core a batch
        axis, and XLA is then free to reassociate those reductions across
        the batch. :func:`jaxgsa.hsic._analyze._get_hsic_kernel` records
        that measurement: a batch axis over the output slices moved
        indices by up to 1e-3 relative, which is why the shipped map is a
        sequential :func:`jax.lax.map` and takes no chunk width.

        What makes HSIC that sensitive is the estimator, not the
        transform. The V-statistic is ``U/n^2 - 2V/n^3 + W/n^4``, and on
        this sample the three terms are all near 0.3 to 0.65 while the
        index they produce runs from 3e-4 to 1e-1. So a weak parameter
        cancels away about 2000 parts in 2001 of the sums that make it,
        and a change of summation order at the float32 epsilon of those
        sums lands on the index magnified by that same 2000. Measured
        here: the worst relative gap between the two paths is 2.6e-4 in
        float32 and 4.0e-13 in float64, and both are the machine epsilon
        of their dtype times the cancellation ratio. That is float32
        conditioning, not a difference between the two graphs.

        A user reads the same fact from ``analyze``: it warns in single
        precision that only three or four digits survive and that small
        indices are not reliable. A transform test cannot be tighter than
        the number the library itself refuses to promise, so 1e-3 is that
        documented figure and the atol covers the near-zero indices,
        where no relative tolerance can mean anything in float32.
        """
        problem, X, base = self._ishigami()
        batch = jnp.stack([base, X[:, 0], X[:, 1] ** 2])
        key = jax.random.key(9)

        stacked = jax.vmap(
            lambda outputs: indices(problem, X, outputs, n_perms=self.PERMS, key=key)
        )(batch)

        assert stacked[0].shape == (3, problem.n_params)
        for row, outputs in enumerate(batch):
            eager = indices(problem, X, outputs, n_perms=self.PERMS, key=key)
            for produced, expected in zip(stacked, eager, strict=True):
                np.testing.assert_allclose(
                    np.asarray(produced[row]), np.asarray(expected), rtol=1e-3, atol=1e-5
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
        through a host-only CDF implementation, which converted a tracer to a
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


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestBandwidthProvenance:
    """The index moves with the bandwidth, so the result must carry it.

    A stored ``R2`` without its bandwidth does not say what it measured:
    the same data at 0.25 and at the default 1.0 can rank the parameters
    differently. This is the same reason ``SobolResult`` carries
    ``estimator``.
    """

    @staticmethod
    def _data():
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=200, seed=0))
        return X, ishigami.evaluate(X)

    def test_result_records_the_run_settings(self):
        """Settings come back as passed, defaults are recorded too, and the export carries them."""
        X, Y = self._data()
        result = analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_perms=20,
            bandwidth=0.25,
            key=jax.random.key(0),
            verbose=False,
        )
        assert result.bandwidth == 0.25
        assert result.n_perms == 20
        attrs = result.to_dataset().attrs
        assert attrs["bandwidth"] == 0.25
        assert attrs["n_perms"] == 20

        default = analyze(ishigami.PROBLEM, X, Y, n_perms=20, key=jax.random.key(0), verbose=False)
        assert default.bandwidth == 1.0


# ---------------------------------------------------------------------------
# Interface cleanup: named tuple, ci, key signature, no dtype warning
# ---------------------------------------------------------------------------


def _small_problem():
    problem = Problem(names=("a", "b"), bounds=[(0.0, 1.0)] * 2)
    X = monte_carlo(problem, 64, seed=0)
    return problem, X, X[:, 0] ** 2 + 0.1 * X[:, 1]


def test_indices_returns_a_named_tuple_that_unpacks():
    from jaxgsa.hsic import HSICIndices

    problem, X, Y = _small_problem()
    out = indices(problem, X, Y, n_perms=3, key=0)
    assert isinstance(out, HSICIndices)
    assert out._fields == ("R2", "T", "p_values", "hsic_raw")
    R2, T, p_values, hsic_raw = out
    assert R2 is out.R2 and T is out.T and p_values is out.p_values and hsic_raw is out.hsic_raw


def test_result_has_ci_none_and_renamed_fields():
    problem, X, Y = _small_problem()
    with single_precision_warning():
        result = analyze(problem, X, Y, n_perms=3, key=0, verbose=False)
    assert result.ci is None
    assert result.R2.shape == (2,) and result.T.shape == (2,)
    assert not hasattr(result, "R2_HSIC")
    assert {"R2", "T"} <= set(result.to_dataset().data_vars)


def test_key_is_keyword_only_without_default():
    for fn in (analyze, indices):
        param = inspect.signature(fn).parameters["key"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty


def test_no_dtype_truncation_warning_in_single_precision():
    """The radix selection must not request int64 while x64 is off."""
    problem, X, Y = _small_problem()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyze(problem, X, Y, n_perms=3, key=0, verbose=False)
    user = [w for w in caught if issubclass(w.category, UserWarning) and "dtype" in str(w.message)]
    assert user == []
