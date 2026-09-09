"""Tests for the polynomial chaos expansion (PCE) sensitivity analysis module."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning, pce
from jaxgsa.benchmarks import ishigami, linear
from jaxgsa.pce._analyze import _auto_order
from jaxgsa.pce._engine import (
    _hermite_1d,
    _legendre_1d,
    build_design_matrix,
    build_multi_index,
    gram_cholesky,
    hat_diagonal,
)
from jaxgsa.problem import GaussianInputSpec, InputSpecValue, Problem

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def linear_pce_data():
    """Generate training data for the linear benchmark model."""
    key = jax.random.PRNGKey(0)
    N = 500
    bounds = jnp.array(linear.PROBLEM.bounds)
    X = jax.random.uniform(key, shape=(N, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
    Y = linear.evaluate(X)
    return X, Y


@pytest.fixture(scope="module")
def linear_pce_result(linear_pce_data):
    """PCE analysis result for the linear benchmark model."""
    X, Y = linear_pce_data
    return pce.analyze(linear.PROBLEM, X, Y, order=2)


@pytest.fixture(scope="module")
def ishigami_pce_data():
    """Generate training data for the Ishigami benchmark model."""
    key = jax.random.PRNGKey(42)
    N = 2000
    bounds = jnp.array(ishigami.PROBLEM.bounds)
    X = jax.random.uniform(key, shape=(N, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
    Y = ishigami.evaluate(X)
    return X, Y


@pytest.fixture(scope="module")
def ishigami_pce_result(ishigami_pce_data):
    """PCE analysis result for the Ishigami benchmark model."""
    X, Y = ishigami_pce_data
    return pce.analyze(ishigami.PROBLEM, X, Y, order=6)


@pytest.fixture(scope="module")
def gaussian_pce_data():
    """Generate training data for a linear model with Gaussian inputs."""
    problem = Problem.from_dict(
        {
            "g1": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
            "g2": GaussianInputSpec(dist="gaussian", mean=2.0, variance=4.0),
        }
    )
    key = jax.random.PRNGKey(99)
    N = 500
    # Sample from standard normal and scale
    Z = jax.random.normal(key, shape=(N, 2))
    X = jnp.column_stack([Z[:, 0], 2.0 * Z[:, 1] + 2.0])
    coeffs = jnp.array([1.0, 2.0])
    Y = X @ coeffs
    return problem, X, Y


@pytest.fixture(scope="module")
def gaussian_pce_result(gaussian_pce_data):
    """PCE analysis result for the Gaussian-input linear model."""
    problem, X, Y = gaussian_pce_data
    return pce.analyze(problem, X, Y, order=2)


# ---------------------------------------------------------------------------
# 1. Linear model analytical correctness
# ---------------------------------------------------------------------------


class TestLinearModel:
    """PCE on a linear additive model should recover exact Sobol indices."""

    def test_s1_matches_analytical(self, linear_pce_result):
        """S1 should match the analytical first-order indices for a linear model."""
        S1_analytical, _, _ = linear.analytical_indices()
        np.testing.assert_allclose(
            np.asarray(linear_pce_result.S1),
            S1_analytical,
            atol=0.02,
            rtol=0.05,
        )

    def test_st_equals_s1(self, linear_pce_result):
        """ST should equal S1 for a purely additive model (no interactions)."""
        np.testing.assert_allclose(
            np.asarray(linear_pce_result.ST),
            np.asarray(linear_pce_result.S1),
            atol=0.02,
            rtol=0.05,
        )

    def test_s2_near_zero(self, linear_pce_result):
        """Off-diagonal S2 should be approximately zero for a linear model."""
        S2 = np.asarray(linear_pce_result.S2)
        D = S2.shape[0]
        for i in range(D):
            for j in range(i + 1, D):
                assert abs(S2[i, j]) < 0.02, f"S2[{i},{j}] = {S2[i, j]:.4f}, expected ~0"


# ---------------------------------------------------------------------------
# 2. Ishigami S1/ST within tolerance
# ---------------------------------------------------------------------------


class TestIshigami:
    """PCE on the Ishigami function should approximate known indices."""

    def test_s1_within_tolerance(self, ishigami_pce_result):
        """Tier T0 (closed form): every analytical S1 entry of the Ishigami function.

        Non-zero entries hold to 30% relative error. The analytical ``S1[x3]``
        is exactly zero, so it takes an absolute bound; the estimate is about
        3e-5, well inside it.
        """
        S1 = np.asarray(ishigami_pce_result.S1)
        analytical = np.array(ishigami.ANALYTICAL_S1)
        for i in range(len(analytical)):
            if analytical[i] > 0.01:
                rel_err = abs(S1[i] - analytical[i]) / analytical[i]
                assert rel_err < 0.30, (
                    f"S1[{i}]: PCE={S1[i]:.4f}, analytical={analytical[i]:.4f}, "
                    f"rel_err={rel_err:.2%}"
                )
            else:
                assert abs(S1[i]) < 0.05, f"S1[{i}]={S1[i]:.6f}, expected ~0"

    def test_st_within_tolerance(self, ishigami_pce_result):
        """Tier T0 (closed form): every analytical ST entry of the Ishigami function.

        No analytical ST entry of the Ishigami function is near zero: the
        smallest is ``ST[x3] = 0.2437``, which comes entirely from the x1-x3
        interaction. Every entry therefore takes the same relative bound and
        none is skipped.
        """
        ST = np.asarray(ishigami_pce_result.ST)
        analytical = np.array(ishigami.ANALYTICAL_ST)
        assert np.all(analytical > 0.01)
        for i in range(len(analytical)):
            rel_err = abs(ST[i] - analytical[i]) / analytical[i]
            assert rel_err < 0.30, (
                f"ST[{i}]: PCE={ST[i]:.4f}, analytical={analytical[i]:.4f}, rel_err={rel_err:.2%}"
            )


# ---------------------------------------------------------------------------
# 3. Emulator round-trip
# ---------------------------------------------------------------------------


class TestEmulatorRoundTrip:
    """emulate() should predict training outputs accurately for a polynomial model."""

    def test_linear_emulator_reproduces_training_data(self, linear_pce_result, linear_pce_data):
        """Emulator predictions on training data should be close to actual outputs."""
        X, Y = linear_pce_data
        Y_pred = linear_pce_result.predict(X)
        np.testing.assert_allclose(
            np.asarray(Y_pred),
            np.asarray(Y),
            atol=0.05,
            rtol=0.05,
        )

    def test_ishigami_emulator_reasonable(self, ishigami_pce_result, ishigami_pce_data):
        """Emulator R-squared on Ishigami training data should be high."""
        X, Y = ishigami_pce_data
        Y_pred = ishigami_pce_result.predict(X)
        ss_res = float(jnp.sum((Y - Y_pred) ** 2))
        ss_tot = float(jnp.sum((Y - jnp.mean(Y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot
        assert r_squared > 0.90, f"R^2 = {r_squared:.4f}, expected > 0.90"


# ---------------------------------------------------------------------------
# 4. auto_order reduction
# ---------------------------------------------------------------------------


class TestAutoOrder:
    """Verify that _auto_order reduces the polynomial order for high D / low N."""

    def test_order_is_bounded_by_the_available_sample_budget(self):
        """Large expansions are reduced but never below the linear basis."""
        # D=20, N=100, order=5 => C(25, 5) = 53130 >> 50 = 0.5*100
        reduced = _auto_order(D=20, N=100, max_order=5, fit_ratio=0.5)
        assert reduced < 5, f"Expected order < 5, got {reduced}"
        # D=3, N=500, order=3 => C(6, 3) = 20 << 250 = 0.5*500
        preserved = _auto_order(D=3, N=500, max_order=3, fit_ratio=0.5)
        assert preserved == 3, f"Expected order=3, got {preserved}"
        result = _auto_order(D=100, N=10, max_order=10, fit_ratio=0.5)
        assert result >= 1, f"Expected order >= 1, got {result}"


# ---------------------------------------------------------------------------
# 5. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """PCE should raise ValueError for invalid inputs."""

    def test_x_column_mismatch_raises(self):
        """X with wrong number of columns should raise ValueError."""
        problem = linear.PROBLEM  # 3 params
        X = jnp.ones((10, 5))  # 5 columns != 3 params
        Y = jnp.ones(10)
        with pytest.raises(ValueError, match="columns"):
            pce.analyze(problem, X, Y)

    def test_underdetermined_fit_raises(self):
        """M2: fewer rows than the order-1 expansion has terms must raise.

        N = 3, D = 3: before the fix this fitted an underdetermined system
        silently and returned a plausible-looking but meaningless S1.
        """
        problem = linear.PROBLEM  # 3 params
        rng = np.random.default_rng(0)
        X = rng.uniform(0.0, 1.0, size=(3, 3))
        Y = X.sum(axis=1)
        with pytest.raises(ValueError, match="cannot fit"):
            pce.analyze(problem, X, Y, verbose=False)
        with pytest.raises(ValueError, match="cannot fit"):
            pce.indices(problem, X, Y)

    def test_fit_ratio_above_one_raises(self):
        """M2: fit_ratio > 1 is meaningless (more terms than rows per term)."""
        problem = linear.PROBLEM
        rng = np.random.default_rng(0)
        X = rng.uniform(0.0, 1.0, size=(12, 3))
        Y = X.sum(axis=1)
        with pytest.raises(ValueError, match="fit_ratio"):
            pce.analyze(problem, X, Y, fit_ratio=2.0, verbose=False)
        with pytest.raises(ValueError, match="fit_ratio"):
            pce.indices(problem, X, Y, fit_ratio=2.0)

    def test_constant_output_with_float32_rounding_noise_still_reads_as_zero_variance(self):
        """M1: a constant Y whose naive sample variance is not bit-exact zero still warns.

        The old guard tested ``var == 0``, which almost never holds for a
        constant float32 output because the mean itself rounds.
        ``Y = full(N, 0.1)`` has a naive sample variance of about 2e-16, not
        zero, and used to return a plausible-looking finite S1 with no
        warning instead of NaN.
        """
        problem = linear.PROBLEM
        rng = np.random.default_rng(0)
        X = jnp.asarray(rng.uniform(0.0, 1.0, size=(500, 3)))
        Y = jnp.full(500, 0.1)
        assert float(jnp.var(Y)) != 0.0, (
            "the naive variance must be nonzero for this to test the fix"
        )
        with pytest.warns(UserWarning, match="zero variance"):
            result = pce.analyze(problem, X, Y, verbose=False)
        assert jnp.all(jnp.isnan(result.S1))
        assert jnp.all(jnp.isnan(result.ST))


# ---------------------------------------------------------------------------
# 6. S2 matrix properties
# ---------------------------------------------------------------------------


class TestS2Properties:
    """S2 interaction matrix should have correct structural properties."""

    def test_diagonal_is_nan_and_matrix_is_symmetric(self, linear_pce_result):
        """The structural S2 invariants hold together."""
        S2 = np.asarray(linear_pce_result.S2)
        assert np.all(np.isnan(np.diag(S2))), f"S2 diagonal should be NaN, got {np.diag(S2)}"
        D = S2.shape[0]
        upper = np.triu_indices(D, k=1)
        lower = (upper[1], upper[0])
        np.testing.assert_allclose(S2[upper], S2[lower], atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# 7. LOO RMSE
# ---------------------------------------------------------------------------


class TestLooRmse:
    """Leave-one-out RMSE should be finite and small for well-fitted models."""

    def test_loo_small_for_linear(self, linear_pce_result, linear_pce_data):
        """LOO RMSE should be small relative to the output range for a linear model."""
        _, Y = linear_pce_data
        y_range = float(jnp.max(Y) - jnp.min(Y))
        loo = float(linear_pce_result.loo_rmse)
        assert loo / y_range < 0.05, f"LOO RMSE / range = {loo / y_range:.4f}, expected < 0.05"

    def test_leverage_clip_holds_in_both_dtypes(self):
        """T4: an interpolated row's leverage stays strictly below 1.

        A square invertible ``Phi`` has ``H = I``, so every leverage is
        exactly 1 before the clip. The old bound ``1 - 1e-10`` rounds back
        to exactly 1.0 in float32 (the spacing below 1.0 is ~6e-8), which
        made the clip a no-op and let ``1 - h_i`` reach zero. The bound is
        now taken inside the leverage's own dtype, so the invariant must
        hold in float32 and float64 alike.
        """
        eye32 = jnp.eye(6, dtype=jnp.float32)
        lev32 = hat_diagonal(eye32, gram_cholesky(eye32, 0.0))
        assert lev32.dtype == jnp.float32
        assert np.all(np.asarray(lev32) < 1.0)

        with jax.enable_x64():
            eye64 = jnp.eye(6, dtype=jnp.float64)
            lev64 = hat_diagonal(eye64, gram_cholesky(eye64, 0.0))
            assert lev64.dtype == jnp.float64
            assert np.all(np.asarray(lev64) < 1.0)

    def test_interpolating_fit_keeps_loo_finite(self):
        """T4: a fit with as many terms as rows reports a finite LOO RMSE.

        With ``N == C(D + order, order)`` and ``fit_ratio=1.0`` the least
        squares fit interpolates, every leverage is (up to the tiny ridge)
        1, and the LOO shortcut divides by ``1 - h_i``. In float32 the old
        no-op clip let that become residual/0 and an inf/NaN ``loo_rmse``
        propagated to every slice. The number is legitimately huge — an
        interpolating fit has no honest LOO — but it must be finite, in
        float32 and under x64 alike.
        """

        def _fit():
            key = jax.random.PRNGKey(3)
            D, order = 3, 2
            n_terms = 10  # C(D + order, order) = C(5, 2)
            bounds = jnp.array(ishigami.PROBLEM.bounds)
            X = jax.random.uniform(
                key, shape=(n_terms, D), minval=bounds[:, 0], maxval=bounds[:, 1]
            )
            return pce.analyze(
                ishigami.PROBLEM, X, ishigami.evaluate(X), order=order, fit_ratio=1.0
            )

        result32 = _fit()
        assert result32.loo_rmse is not None
        assert np.all(np.isfinite(np.asarray(result32.loo_rmse)))

        with jax.enable_x64():
            result64 = _fit()
            assert result64.loo_rmse is not None
            assert np.all(np.isfinite(np.asarray(result64.loo_rmse)))


# ---------------------------------------------------------------------------
# 8. Gaussian inputs (Hermite basis)
# ---------------------------------------------------------------------------


class TestGaussianInputs:
    """PCE with Gaussian inputs should use the Hermite basis correctly."""

    def test_gaussian_s1_reasonable(self, gaussian_pce_result):
        """S1 for the Gaussian linear model should match analytical values.

        For f = c1*x1 + c2*x2 with independent inputs:
        S1_j = c_j^2 * Var(x_j) / sum(c_k^2 * Var(x_k)).
        """
        # c = [1, 2], Var = [1, 4] => contributions = [1, 16], S1 = [1/17, 16/17]
        expected = np.array([1.0 / 17.0, 16.0 / 17.0])
        S1 = np.asarray(gaussian_pce_result.S1)
        np.testing.assert_allclose(S1, expected, atol=0.05, rtol=0.1)

    def test_gaussian_emulator(self, gaussian_pce_result, gaussian_pce_data):
        """Emulator should predict well on Gaussian training data."""
        problem, X, Y = gaussian_pce_data
        Y_pred = gaussian_pce_result.predict(X)
        np.testing.assert_allclose(np.asarray(Y_pred), np.asarray(Y), atol=0.5, rtol=0.1)

    def test_hermite_and_legendre_bases_are_orthonormal(self):
        """The two one-dimensional bases have their declared inner products."""
        key = jax.random.PRNGKey(123)
        N = 200000
        x = jax.random.normal(key, shape=(N,))
        H = _hermite_1d(x, 3)
        np.testing.assert_allclose(np.asarray((H.T @ H) / N), np.eye(4), atol=0.05)
        x = jax.random.uniform(jax.random.PRNGKey(456), shape=(50000,), minval=-1.0, maxval=1.0)
        P = _legendre_1d(x, 4)
        np.testing.assert_allclose(np.asarray((P.T @ P) / 50000), np.eye(5), atol=0.05)

    def test_basis_dtype_promotes_float32_inputs_under_x64(self):
        """A float32 x must not downgrade the PCE basis to float32 under x64.

        Both recurrences run in the default float dtype, exactly like the
        Hermite seed matrix. A float32 sample matrix then still yields a
        float64 design matrix, and mixed uniform/Gaussian problems get one
        consistent basis dtype.
        """
        with jax.enable_x64():
            x = jnp.linspace(-0.9, 0.9, 8, dtype=jnp.float32)
            assert _hermite_1d(x, 3).dtype == jnp.float64
            assert _legendre_1d(x, 3).dtype == jnp.float64
            X = jnp.stack([x, x], axis=1)
            Phi = build_design_matrix(X, build_multi_index(2, 2), ("uniform", "gaussian"), 2)
            assert Phi.dtype == jnp.float64
        # The NumPy path promotes the same way (float64 is NumPy's default).
        P = _legendre_1d(np.linspace(-0.9, 0.9, 8, dtype=np.float32), 3)
        assert P.dtype == np.float64


class TestTruncatedGaussianBasis:
    """A wide truncation must keep the Hermite basis; a narrow one must not."""

    @staticmethod
    def _problem(D, *, q=None):
        params: dict[str, InputSpecValue] = {
            f"x{i + 1}": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0)
            for i in range(D)
        }
        return Problem.from_dict(params, truncate_gaussians=q)

    def test_truncation_selects_the_stable_basis_by_width_and_order(self):
        from jaxgsa.pce._analyze import _map_to_reference

        problem = self._problem(2, q=1e-12)  # |z| = 7.03
        X = jax.random.normal(jax.random.PRNGKey(0), (32, 2))
        _, input_types = _map_to_reference(X, problem, 3)
        assert input_types == ("gaussian", "gaussian")
        problem = self._problem(2, q=1e-3)  # |z| = 3.09
        X = jax.random.normal(jax.random.PRNGKey(0), (32, 2)) * 0.5
        _, input_types = _map_to_reference(X, problem, 3)
        assert input_types == ("uniform", "uniform")
        problem = self._problem(2, q=1e-12)
        X = jax.random.normal(jax.random.PRNGKey(0), (32, 2))
        _, input_types = _map_to_reference(X, problem, 10)
        assert input_types == ("uniform", "uniform")

    def test_a_saturated_cdf_stays_off_the_upper_edge_in_float32(self):
        """T4: the unit clip has to bite in the dtype the map runs in.

        The Legendre route clips its CDF values into
        ``[UNIT_CLIP, 1 - UNIT_CLIP]`` so a sample at or past a truncation
        bound stays distinguishable from the bound itself. At
        ``UNIT_CLIP = 1e-12`` the upper half of that clip did nothing in
        float32, because ``1 - 1e-12`` rounds back to exactly 1.0 there,
        while the same expression in float64 clipped properly.

        Only the upper edge is asserted. The lower one is exactly ``-1.0`` in
        float32 whatever the clip does, because the affine ``2u - 1`` that
        follows it cannot represent ``2e-12 - 1`` as anything else, and that
        is a property of the map rather than of the clip.
        """
        from jaxgsa.pce._analyze import _map_to_reference

        problem = self._problem(1, q=1e-3)  # bounds at |z| = 3.09
        X = jnp.asarray([[0.0], [3.09], [40.0]], dtype=jnp.float32)
        xi, input_types = _map_to_reference(X, problem, 3)
        assert input_types == ("uniform",)
        assert np.all(np.asarray(xi) < 1.0)

    def test_wide_truncation_matches_unbounded_accuracy(self):
        """Oakley-O'Hagan: a wide truncation must not degrade the fit.

        The old rule routed every truncated Gaussian to Legendre, which has to
        approximate the steep inverse normal CDF and lost roughly a factor 2 in
        LOO RMSE and 7x in S1 error.
        """
        from jaxgsa.benchmarks import oakley_ohagan

        X = jax.random.normal(jax.random.PRNGKey(0), (4000, 15))
        Y = oakley_ohagan.evaluate(X)
        analytic = oakley_ohagan.analytical_indices()[0]

        unbounded = pce.analyze(self._problem(15), X, Y, order=3)
        wide = pce.analyze(self._problem(15, q=1e-12), X, Y, order=3)

        # Same basis, so the two fits agree to numerical noise.
        np.testing.assert_allclose(np.asarray(wide.S1), np.asarray(unbounded.S1), atol=1e-6)
        assert float(wide.loo_rmse) == pytest.approx(float(unbounded.loo_rmse), rel=1e-4)
        assert np.max(np.abs(np.asarray(wide.S1) - analytic)) < 0.003


# ---------------------------------------------------------------------------
# 9. to_dataset()
# ---------------------------------------------------------------------------


class TestToDataset:
    """PCEResult.to_dataset() should produce a well-structured xarray Dataset."""

    def test_dataset_values_match_result(self, linear_pce_result):
        """Dataset values should match the PCEResult attributes."""
        ds = linear_pce_result.to_dataset()
        np.testing.assert_allclose(ds["S1"].values, np.asarray(linear_pce_result.S1), rtol=1e-10)
        np.testing.assert_allclose(ds["ST"].values, np.asarray(linear_pce_result.ST), rtol=1e-10)


# ---------------------------------------------------------------------------
# Engine unit tests
# ---------------------------------------------------------------------------


class TestEngine:
    """Unit tests for low-level PCE engine functions."""

    def test_multi_index_is_unique_ordered_and_scales_to_large_dimension(self):
        """D above CPython's recursion limit must build, not raise.

        The enumerator used to recurse once per dimension, so it died with
        RecursionError at D ~ 997 for any order, including order 1 whose
        basis is only D+1 terms. 1000 is CPython's default limit.
        """
        from math import comb

        D = 1000
        mi = build_multi_index(D, 1)
        assert mi.shape == (comb(D + 1, 1), D)
        np.testing.assert_array_equal(mi[0], np.zeros(D, dtype=mi.dtype))
        np.testing.assert_array_equal(mi[1:], np.eye(D, dtype=mi.dtype)[::-1])

        mi = build_multi_index(4, 3)
        degrees = mi.sum(axis=1)
        assert np.all(np.diff(degrees) >= 0), "total degree must not decrease"
        rows = [tuple(int(v) for v in row) for row in mi]
        assert rows == sorted(rows, key=lambda a: (sum(a), a))
        assert len(set(rows)) == len(rows), "multi-indices must be distinct"


# ---------------------------------------------------------------------------
# 8. Multi-output and time-series support
# ---------------------------------------------------------------------------


class TestMultiOutput:
    """PCE fits every (t, k) output slice against one shared basis."""

    def test_multi_output_and_time_series_shapes(self, linear_pce_data):
        X, Y = linear_pce_data
        Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=-1)
        result = pce.analyze(linear.PROBLEM, X, Y2, order=2)
        n_terms = result.multi_index.shape[0]
        assert result.S1.shape == (2, 3)
        assert result.ST.shape == (2, 3)
        assert result.S2.shape == (2, 3, 3)
        assert result.coefficients.shape == (2, n_terms)
        assert result.loo_rmse is not None and result.loo_rmse.shape == (2,)
        Y3 = jnp.stack([Y2, 0.5 * Y2], axis=1)  # (N, T=2, K=2)
        series = pce.analyze(linear.PROBLEM, X, Y3, order=2)
        assert series.S1.shape == (2, 2, 3)
        assert series.S2.shape == (2, 2, 3, 3)
        assert series.coefficients.shape == (2, 2, n_terms)
        assert series.loo_rmse is not None and series.loo_rmse.shape == (2, 2)

    def test_emulate_multi_output_round_trip(self, linear_pce_data):
        """predict mirrors the training layout and reproduces linear Y."""
        X, Y = linear_pce_data
        Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=-1)
        Y3 = jnp.stack([Y2, 0.5 * Y2], axis=1)
        res2 = pce.analyze(linear.PROBLEM, X, Y2, order=2)
        res3 = pce.analyze(linear.PROBLEM, X, Y3, order=2)
        pred2 = res2.predict(X[:50])
        pred3 = res3.predict(X[:50])
        assert pred2.shape == (50, 2)
        assert pred3.shape == (50, 2, 2)
        # The linear benchmark is exactly representable at order 2.
        np.testing.assert_allclose(np.asarray(pred2), np.asarray(Y2[:50]), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(pred3), np.asarray(Y3[:50]), rtol=1e-5)

    def test_to_dataset_time_series_with_coords(self, linear_pce_data):
        X, Y = linear_pce_data
        Y3 = jnp.stack([jnp.stack([Y, 2.0 * Y], axis=-1)] * 2, axis=1)
        ds = pce.analyze(linear.PROBLEM, X, Y3, order=2).to_dataset(time_coords=[0.0, 0.5])
        assert ds["S1"].dims == ("time", "output", "param")
        assert ds["S2"].dims == ("time", "output", "param_i", "param_j")
        assert ds["loo_rmse"].dims == ("time", "output")
        np.testing.assert_allclose(ds.coords["time"].values, [0.0, 0.5])


# ---------------------------------------------------------------------------
# 10. on_invalid policy
# ---------------------------------------------------------------------------


def _invalid_data(n: int = 400, seed: int = 3):
    """Build a smooth 3-parameter sample the default PCE order fits well."""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(linear.PROBLEM.bounds)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, 3))
    Y = np.asarray(linear.evaluate(jnp.asarray(X)))
    return X, Y


class TestPCEOnInvalid:
    """T4: ``pce.analyze`` applies the shared non-finite policy.

    A PCE fit has no defence of its own against a NaN. It reaches the normal
    equations, and one bad row is enough to make every coefficient, every
    leave-one-out RMSE and every index non-finite.
    """

    def test_raise_is_the_default_and_names_the_rows(self):
        """T4: a non-finite Y refuses the analysis and says which rows carry it."""
        X, Y = _invalid_data()
        Y = Y.copy()
        Y[7] = np.nan
        Y[19] = np.inf
        with pytest.raises(ValueError, match="non-finite") as exc:
            pce.analyze(linear.PROBLEM, X, Y)
        message = str(exc.value)
        assert "jaxgsa.pce.analyze" in message
        assert "2 of 400 rows" in message
        assert "[7, 19]" in message

    def test_propagate_warns_and_lets_the_nan_reach_the_indices(self):
        """T4: 'propagate' keeps every row, so the indices come out non-finite.

        This is the debugging policy: the answer must be loudly wrong rather
        than quietly wrong, so both the warning and the NaN indices matter.
        """
        X, Y = _invalid_data()
        Y = Y.copy()
        Y[7] = np.nan
        with pytest.warns(JaxgsaWarning, match="reaches the indices"):
            result = pce.analyze(linear.PROBLEM, X, Y, on_invalid="propagate")
        assert not np.all(np.isfinite(np.asarray(result.S1)))
        assert result.invalid.policy == "propagate"
        assert result.invalid.n_invalid == 1
        assert result.invalid.n_kept == 399

    def test_drop_removes_the_rows_and_the_fit_is_finite_again(self):
        """T4: 'drop' fits on the survivors, so every index is finite."""
        X, Y = _invalid_data()
        Y = Y.copy()
        Y[7] = np.nan
        with pytest.warns(JaxgsaWarning, match="dropped 1 of 400 rows"):
            result = pce.analyze(linear.PROBLEM, X, Y, on_invalid="drop", order=2)
        assert np.all(np.isfinite(np.asarray(result.S1)))
        assert result.loo_rmse is not None
        assert np.all(np.isfinite(np.asarray(result.loo_rmse)))
        assert result.invalid.unit_indices == (7,)
        assert result.invalid.sources == ("Y",)

    def test_a_non_finite_input_is_caught_too(self):
        """T4: X is checked as well as Y, and ``sources`` says which array.

        A NaN input is just as fatal as a NaN output, because it enters the
        design matrix. A user who only sanitized Y needs to be told where to
        look.
        """
        X, Y = _invalid_data()
        X = X.copy()
        X[11, 2] = np.nan
        with pytest.raises(ValueError, match=r"in X\.") as exc:
            pce.analyze(linear.PROBLEM, X, Y)
        assert "[11]" in str(exc.value)
        with pytest.warns(JaxgsaWarning):
            result = pce.analyze(linear.PROBLEM, X, Y, on_invalid="drop", order=2)
        assert result.invalid.sources == ("X",)


class TestBootstrapIntervals:
    """The opt-in confidence intervals.

    Tier T4 throughout. A bootstrap interval has no closed form to check
    against; what these pin is that the interval brackets the estimate it is
    an interval for, that it costs nothing when nobody asks for it, and that
    the vocabulary the rest of the package uses is honoured here too.
    """

    @staticmethod
    def _data(n=256, seed=0):
        rng = np.random.default_rng(seed)
        X = jnp.asarray(rng.uniform(-np.pi, np.pi, size=(n, 3)))
        return X, ishigami.evaluate(X)

    def test_no_bootstrap_by_default(self):
        """T4: the plainest call reports no interval and pays for none.

        Each replicate refits the expansion, so an on-by-default interval
        would make a routine call an order of magnitude slower.
        """
        X, Y = self._data()
        result = pce.analyze(ishigami.PROBLEM, X, Y, order=3)

        assert result.ci is None
        assert result.S1_conf is None
        assert result.ST_conf is None
        assert result.S2_conf is None

    def test_the_interval_brackets_the_point_estimate(self):
        """T4: lower <= estimate <= upper, for all three index arrays."""
        X, Y = self._data()
        result = pce.analyze(ishigami.PROBLEM, X, Y, order=3, n_bootstrap=8, key=jax.random.key(0))

        for point, conf in (
            (result.S1, result.S1_conf),
            (result.ST, result.ST_conf),
            (result.S2, result.S2_conf),
        ):
            lower, upper = np.asarray(conf[0]), np.asarray(conf[1])
            point = np.asarray(point)
            finite = np.isfinite(point) & np.isfinite(lower) & np.isfinite(upper)
            assert (lower[finite] <= upper[finite]).all()
            # The percentile interval is read off the replicate distribution,
            # which is centred near but not exactly on the point estimate, so
            # a small slack is expected on a run of eight.
            assert (point[finite] >= lower[finite] - 0.2).all()
            assert (point[finite] <= upper[finite] + 0.2).all()

    def test_ci_metadata_and_replicates_are_recorded(self):
        """The interval records its recipe and optionally exposes its draws."""
        X, Y = self._data()
        result = pce.analyze(
            ishigami.PROBLEM,
            X,
            Y,
            order=2,
            n_bootstrap=6,
            conf_level=0.8,
            ci_method="gaussian",
            key=jax.random.key(2),
            keep_replicates=True,
        )

        assert result.ci is not None
        assert result.ci.level == 0.8
        assert result.ci.method == "gaussian"
        assert result.ci.n_bootstrap == 6
        assert result.ci.replicates is not None
        assert result.ci.replicates["S1"].shape == (6, 3)
        assert result.ci.replicates["S2"].shape == (6, 3, 3)

    def test_a_key_is_required(self):
        """T4: no key means no interval, and it is refused rather than seeded.

        Falling back to a constant key would silently correlate every nested
        or repeated bootstrap.
        """
        X, Y = self._data()
        with pytest.raises(ValueError, match="key is required"):
            pce.analyze(ishigami.PROBLEM, X, Y, n_bootstrap=4)

    def test_the_same_key_gives_the_same_interval(self):
        """T4: the draw is a function of the key, not of the call."""
        X, Y = self._data()
        kwargs = dict(order=2, n_bootstrap=4, key=jax.random.key(7))
        first = pce.analyze(ishigami.PROBLEM, X, Y, **kwargs)
        second = pce.analyze(ishigami.PROBLEM, X, Y, **kwargs)

        np.testing.assert_array_equal(np.asarray(first.S1_conf), np.asarray(second.S1_conf))

    def test_a_time_series_gets_intervals_at_the_same_rank(self):
        """T4: the interval keeps the result's own layout, plus the [lo, hi] axis."""
        X, Y = self._data()
        Y3 = jnp.stack([jnp.stack([Y, 2.0 * Y], axis=-1)], axis=1)
        result = pce.analyze(
            ishigami.PROBLEM, X, Y3, order=2, n_bootstrap=3, key=jax.random.key(4)
        )

        assert result.S1.shape == (1, 2, 3)
        assert result.S1_conf.shape == (2, 1, 2, 3)
        assert result.S2_conf.shape == (2, 1, 2, 3, 3)


class TestExplainedVariance:
    """``explained_variance`` is the in-sample R^2, so it stays within [0, 1].

    Regression cover for a fault where the numerator was the sum of squared
    non-constant coefficients, which is the surrogate's variance under the
    *input measure*, while the denominator was the *sample* variance of ``Y``.
    At finite ``N`` the empirical Gram is not the identity, so the two
    disagree, and an order-8 Ishigami fit on ``N=2000`` reported 1.0709 for a
    fit whose true R^2 was 0.9997.
    """

    def test_high_order_ishigami_does_not_exceed_one(self, ishigami_pce_data):
        """The exact case that used to report 1.0709."""
        X, Y = ishigami_pce_data
        result = pce.analyze(ishigami.PROBLEM, X, Y, order=8, verbose=False)
        ev = float(np.asarray(result.explained_variance))
        assert 0.0 <= ev <= 1.0, f"explained_variance out of range: {ev}"
        assert ev > 0.99, f"an order-8 Ishigami fit should be near-perfect, got {ev}"

    def test_matches_r_squared(self, ishigami_pce_data):
        """It equals ``1 - SS_res / SS_tot`` of the fitted values."""
        X, Y = ishigami_pce_data
        result = pce.analyze(ishigami.PROBLEM, X, Y, order=3, verbose=False)
        residual = Y - result.predict(X)
        r2 = 1.0 - float(jnp.sum(residual**2)) / float(jnp.sum((Y - jnp.mean(Y)) ** 2))
        np.testing.assert_allclose(
            float(np.asarray(result.explained_variance)), r2, rtol=1e-3, atol=1e-4
        )

    def test_constant_output_explained_variance_and_shapley_are_nan(self):
        """A constant slice reports NaN everywhere, Shapley included.

        Regression for the constant-output Shapley bug: the old guard tested
        ``total_var == 0``, which misses a constant float32 slice whose mean
        rounds (``Y = full(N, 0.1)`` has variance ~2e-16), so ``shapley()``
        normalized by that rounding noise and returned plausible-looking
        finite Sh instead of NaN.
        """
        problem = linear.PROBLEM
        rng = np.random.default_rng(0)
        X = jnp.asarray(rng.uniform(0.0, 1.0, size=(500, 3)))
        Y = jnp.full(500, 0.1)
        assert float(jnp.var(Y)) != 0.0, (
            "the naive variance must be nonzero for this to test the fix"
        )
        with pytest.warns(UserWarning, match="zero variance"):
            result = pce.analyze(problem, X, Y, verbose=False)
        assert result.explained_variance is not None
        assert jnp.isnan(result.explained_variance)
        assert jnp.all(jnp.isnan(result.shapley().Sh))

    def test_varying_output_explained_variance_and_shapley_are_finite(self):
        """A varying slice must not be masked: R^2 stays ~1 and Sh stays finite.

        Guards against over-masking, the mirror image of the constant-output
        bug.
        """
        problem = linear.PROBLEM
        bounds = jnp.array(problem.bounds)
        X = jax.random.uniform(
            jax.random.PRNGKey(3),
            shape=(500, problem.num_vars),
            minval=bounds[:, 0],
            maxval=bounds[:, 1],
        )
        Y = 2.0 + 3.0 * X[:, 0] - 1.5 * X[:, 1]
        result = pce.analyze(problem, X, Y, order=1, verbose=False)
        np.testing.assert_allclose(float(np.asarray(result.explained_variance)), 1.0, atol=1e-4)
        Sh = result.shapley().Sh
        assert jnp.all(jnp.isfinite(Sh))
        np.testing.assert_allclose(float(Sh.sum()), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Fit-quality warnings
# ---------------------------------------------------------------------------


class TestFitQualityWarnings:
    """analyze must say when the fit is too poor for its own indices.

    Every PCE index is exact within the fitted polynomial, so a polynomial
    that is not this model gives indices that are exactly wrong and look
    exactly right. The default ``order=3`` does not resolve a strong
    nonlinearity, which makes the silent version of this the most likely way
    to read a wrong number off the library.
    """

    def test_underfit_warns_and_names_explained_variance(self, ishigami_pce_data):
        """order=3 on Ishigami explains under half the variance, and says so."""
        X, Y = ishigami_pce_data
        with pytest.warns(JaxgsaWarning, match=r"explained_variance is 0\.\d+"):
            result = pce.analyze(ishigami.PROBLEM, X, Y, order=3, verbose=False)
        assert float(np.asarray(result.explained_variance)) < 0.5

    def test_overfit_warns_on_the_loo_ratio(self, ishigami_pce_data):
        """A fit with too few rows per term is caught out of sample only."""
        X, Y = ishigami_pce_data
        Xs, Ys = X[:64], Y[:64]
        with pytest.warns(JaxgsaWarning, match=r"loo_rmse is \d+\.\d+ times std\(Y\)"):
            result = pce.analyze(ishigami.PROBLEM, Xs, Ys, order=5, fit_ratio=0.9, verbose=False)
        # The in-sample number reports nothing wrong at all, which is the
        # reason the out-of-sample check exists.
        assert float(np.asarray(result.explained_variance)) > 0.9

    def test_healthy_fit_is_silent(self, ishigami_pce_data):
        """A well-resolved fit raises neither warning."""
        X, Y = ishigami_pce_data
        with warnings.catch_warnings():
            warnings.simplefilter("error", JaxgsaWarning)
            pce.analyze(ishigami.PROBLEM, X, Y, order=10, verbose=False)
