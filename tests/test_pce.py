"""Tests for the polynomial chaos expansion (PCE) sensitivity analysis module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from gsax import pce
from gsax.benchmarks import ishigami, linear
from gsax.pce._analyze import _auto_order
from gsax.pce._engine import (
    _hermite_1d,
    _legendre_1d,
    build_design_matrix,
    build_multi_index,
    loo_error,
    sobol_from_coefficients,
)
from gsax.problem import GaussianInputSpec, Problem

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

    def test_s1_sums_to_one(self, linear_pce_result):
        """S1 should sum to approximately 1 for a purely additive model."""
        total = float(jnp.sum(linear_pce_result.S1))
        assert abs(total - 1.0) < 0.05, f"sum(S1) = {total}, expected ~1.0"


# ---------------------------------------------------------------------------
# 2. Ishigami S1/ST within tolerance
# ---------------------------------------------------------------------------


class TestIshigami:
    """PCE on the Ishigami function should approximate known indices."""

    def test_s1_within_tolerance(self, ishigami_pce_result):
        """Non-zero S1 values should be within 30% relative error."""
        S1 = np.asarray(ishigami_pce_result.S1)
        analytical = np.array(ishigami.ANALYTICAL_S1)
        for i in range(len(analytical)):
            if analytical[i] > 0.01:
                rel_err = abs(S1[i] - analytical[i]) / analytical[i]
                assert rel_err < 0.30, (
                    f"S1[{i}]: PCE={S1[i]:.4f}, analytical={analytical[i]:.4f}, "
                    f"rel_err={rel_err:.2%}"
                )

    def test_st_within_tolerance(self, ishigami_pce_result):
        """Non-zero ST values should be within 30% relative error."""
        ST = np.asarray(ishigami_pce_result.ST)
        analytical = np.array(ishigami.ANALYTICAL_ST)
        for i in range(len(analytical)):
            if analytical[i] > 0.01:
                rel_err = abs(ST[i] - analytical[i]) / analytical[i]
                assert rel_err < 0.30, (
                    f"ST[{i}]: PCE={ST[i]:.4f}, analytical={analytical[i]:.4f}, "
                    f"rel_err={rel_err:.2%}"
                )

    def test_s1_x3_near_zero(self, ishigami_pce_result):
        """S1 for x3 should be near zero (x3 only appears in interaction)."""
        S1_x3 = float(ishigami_pce_result.S1[2])
        assert abs(S1_x3) < 0.05, f"S1[x3] = {S1_x3}, expected ~0"


# ---------------------------------------------------------------------------
# 3. Emulator round-trip
# ---------------------------------------------------------------------------


class TestEmulatorRoundTrip:
    """emulate() should predict training outputs accurately for a polynomial model."""

    def test_linear_emulator_reproduces_training_data(self, linear_pce_result, linear_pce_data):
        """Emulator predictions on training data should be close to actual outputs."""
        X, Y = linear_pce_data
        Y_pred = pce.emulate(linear_pce_result, X)
        np.testing.assert_allclose(
            np.asarray(Y_pred),
            np.asarray(Y),
            atol=0.05,
            rtol=0.05,
        )

    def test_ishigami_emulator_reasonable(self, ishigami_pce_result, ishigami_pce_data):
        """Emulator R-squared on Ishigami training data should be high."""
        X, Y = ishigami_pce_data
        Y_pred = pce.emulate(ishigami_pce_result, X)
        ss_res = float(jnp.sum((Y - Y_pred) ** 2))
        ss_tot = float(jnp.sum((Y - jnp.mean(Y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot
        assert r_squared > 0.90, f"R^2 = {r_squared:.4f}, expected > 0.90"


# ---------------------------------------------------------------------------
# 4. auto_order reduction
# ---------------------------------------------------------------------------


class TestAutoOrder:
    """Verify that _auto_order reduces the polynomial order for high D / low N."""

    def test_reduces_order_for_large_d(self):
        """Order should be reduced when D is large relative to N."""
        # D=20, N=100, order=5 => C(25, 5) = 53130 >> 50 = 0.5*100
        reduced = _auto_order(D=20, N=100, max_order=5, fit_ratio=0.5)
        assert reduced < 5, f"Expected order < 5, got {reduced}"

    def test_preserves_order_when_affordable(self):
        """Order should remain unchanged when there are enough samples."""
        # D=3, N=500, order=3 => C(6, 3) = 20 << 250 = 0.5*500
        preserved = _auto_order(D=3, N=500, max_order=3, fit_ratio=0.5)
        assert preserved == 3, f"Expected order=3, got {preserved}"

    def test_minimum_order_is_one(self):
        """Order should never drop below 1."""
        result = _auto_order(D=100, N=10, max_order=10, fit_ratio=0.5)
        assert result >= 1, f"Expected order >= 1, got {result}"


# ---------------------------------------------------------------------------
# 5. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """PCE should raise ValueError for invalid inputs."""

    def test_y_ndim_4_raises(self):
        """4-D Y is outside the (N,), (N, K), (N, T, K) contract."""
        problem = linear.PROBLEM
        X = jnp.ones((10, 3))
        Y_4d = jnp.ones((10, 2, 2, 2))
        with pytest.raises(ValueError, match="Y must be 1-D"):
            pce.analyze(problem, X, Y_4d)

    def test_x_column_mismatch_raises(self):
        """X with wrong number of columns should raise ValueError."""
        problem = linear.PROBLEM  # 3 params
        X = jnp.ones((10, 5))  # 5 columns != 3 params
        Y = jnp.ones(10)
        with pytest.raises(ValueError, match="columns"):
            pce.analyze(problem, X, Y)


# ---------------------------------------------------------------------------
# 6. S2 matrix properties
# ---------------------------------------------------------------------------


class TestS2Properties:
    """S2 interaction matrix should have correct structural properties."""

    def test_diagonal_is_nan(self, linear_pce_result):
        """The diagonal of S2 should be NaN."""
        S2 = np.asarray(linear_pce_result.S2)
        assert np.all(np.isnan(np.diag(S2))), f"S2 diagonal should be NaN, got {np.diag(S2)}"

    def test_symmetric(self, linear_pce_result):
        """S2 should be symmetric (upper triangle equals lower triangle)."""
        S2 = np.asarray(linear_pce_result.S2)
        D = S2.shape[0]
        upper = np.triu_indices(D, k=1)
        lower = (upper[1], upper[0])
        np.testing.assert_allclose(S2[upper], S2[lower], atol=1e-10, rtol=1e-10)

    def test_s2_shape(self, linear_pce_result):
        """S2 should be (D, D)."""
        D = linear_pce_result.problem.num_vars
        assert linear_pce_result.S2.shape == (D, D)


# ---------------------------------------------------------------------------
# 7. LOO RMSE
# ---------------------------------------------------------------------------


class TestLooRmse:
    """Leave-one-out RMSE should be finite and small for well-fitted models."""

    def test_loo_finite(self, linear_pce_result):
        """LOO RMSE should be a finite value."""
        assert linear_pce_result.loo_rmse is not None
        assert np.isfinite(float(linear_pce_result.loo_rmse))

    def test_loo_small_for_linear(self, linear_pce_result, linear_pce_data):
        """LOO RMSE should be small relative to the output range for a linear model."""
        _, Y = linear_pce_data
        y_range = float(jnp.max(Y) - jnp.min(Y))
        loo = float(linear_pce_result.loo_rmse)
        assert loo / y_range < 0.05, f"LOO RMSE / range = {loo / y_range:.4f}, expected < 0.05"

    def test_loo_engine_function_matches(self):
        """loo_error called directly should match the result stored in PCEResult."""
        key = jax.random.PRNGKey(7)
        N = 200
        bounds = jnp.array(linear.PROBLEM.bounds)
        X = jax.random.uniform(key, shape=(N, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
        Y = linear.evaluate(X)
        result = pce.analyze(linear.PROBLEM, X, Y, order=2)

        # Reconstruct Phi manually
        from gsax.pce._analyze import _map_to_reference

        X_ref, input_types = _map_to_reference(X, linear.PROBLEM)
        Phi = build_design_matrix(X_ref, result.multi_index, input_types, result.order)
        loo_direct = loo_error(Phi, Y, result.coefficients, ridge=1e-8)

        assert result.loo_rmse is not None
        np.testing.assert_allclose(float(result.loo_rmse), float(loo_direct), rtol=1e-5)


# ---------------------------------------------------------------------------
# 8. Gaussian inputs (Hermite basis)
# ---------------------------------------------------------------------------


class TestGaussianInputs:
    """PCE with Gaussian inputs should use the Hermite basis correctly."""

    def test_gaussian_result_exists(self, gaussian_pce_result):
        """Gaussian-input PCE should produce a valid PCEResult."""
        assert isinstance(gaussian_pce_result, pce.PCEResult)

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
        Y_pred = pce.emulate(gaussian_pce_result, X)
        np.testing.assert_allclose(np.asarray(Y_pred), np.asarray(Y), atol=0.5, rtol=0.1)

    def test_hermite_1d_orthonormality(self):
        """Hermite basis polynomials should be approximately orthonormal under N(0,1)."""
        key = jax.random.PRNGKey(123)
        N = 200000
        x = jax.random.normal(key, shape=(N,))
        H = _hermite_1d(x, 3)
        # Monte Carlo inner product: E[H_m * H_n] ~ delta_mn
        gram = (H.T @ H) / N
        np.testing.assert_allclose(np.asarray(gram), np.eye(4), atol=0.05)

    def test_legendre_1d_orthonormality(self):
        """Legendre basis polynomials should be approximately orthonormal under U[-1,1]."""
        key = jax.random.PRNGKey(456)
        N = 50000
        x = jax.random.uniform(key, shape=(N,), minval=-1.0, maxval=1.0)
        P = _legendre_1d(x, 4)
        # Monte Carlo inner product with weight 1/2: (1/N) * P^T P ~ I
        gram = (P.T @ P) / N
        np.testing.assert_allclose(np.asarray(gram), np.eye(5), atol=0.05)


# ---------------------------------------------------------------------------
# 9. to_dataset()
# ---------------------------------------------------------------------------


class TestToDataset:
    """PCEResult.to_dataset() should produce a well-structured xarray Dataset."""

    def test_dataset_type(self, linear_pce_result):
        """to_dataset() should return an xarray Dataset."""
        ds = linear_pce_result.to_dataset()
        assert isinstance(ds, xr.Dataset)

    def test_dataset_has_required_vars(self, linear_pce_result):
        """Dataset should contain S1, ST, S2, and loo_rmse."""
        ds = linear_pce_result.to_dataset()
        assert "S1" in ds.data_vars
        assert "ST" in ds.data_vars
        assert "S2" in ds.data_vars
        assert "loo_rmse" in ds.data_vars

    def test_dataset_param_coord(self, linear_pce_result):
        """S1 and ST should have a 'param' coordinate matching problem names."""
        ds = linear_pce_result.to_dataset()
        expected_names = list(linear.PROBLEM.names)
        assert list(ds.coords["param"].values) == expected_names

    def test_dataset_s1_dims(self, linear_pce_result):
        """S1 should have dimension 'param'."""
        ds = linear_pce_result.to_dataset()
        assert ds["S1"].dims == ("param",)

    def test_dataset_s2_dims(self, linear_pce_result):
        """S2 should have dimensions ('param_i', 'param_j')."""
        ds = linear_pce_result.to_dataset()
        assert ds["S2"].dims == ("param_i", "param_j")

    def test_dataset_s2_coords(self, linear_pce_result):
        """S2 coordinates should match problem parameter names."""
        ds = linear_pce_result.to_dataset()
        expected_names = list(linear.PROBLEM.names)
        assert list(ds.coords["param_i"].values) == expected_names
        assert list(ds.coords["param_j"].values) == expected_names

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

    def test_build_multi_index_count(self):
        """Multi-index set size should equal C(D+p, p)."""
        from math import comb

        D, p = 3, 4
        mi = build_multi_index(D, p)
        assert mi.shape == (comb(D + p, p), D)

    def test_build_multi_index_first_row_is_zero(self):
        """First row of the multi-index should be the zero vector."""
        mi = build_multi_index(3, 3)
        np.testing.assert_array_equal(mi[0], [0, 0, 0])

    def test_sobol_from_coefficients_single_var(self):
        """With only one active dimension, S1 and ST should be [1]."""
        # D=1, order=2 => mi = [[0], [1], [2]]
        mi = build_multi_index(1, 2)
        coeffs = jnp.array([0.5, 1.0, 0.3])  # constant, linear, quadratic
        S1, ST, S2 = sobol_from_coefficients(coeffs, mi)
        # Only one variable, so S1 = ST = 1.0
        np.testing.assert_allclose(float(S1[0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(float(ST[0]), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 8. Multi-output and time-series support
# ---------------------------------------------------------------------------


class TestMultiOutput:
    """PCE fits every (t, k) output slice against one shared basis."""

    def test_multi_output_shapes(self, linear_pce_data):
        X, Y = linear_pce_data
        Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=-1)
        result = pce.analyze(linear.PROBLEM, X, Y2, order=2)
        n_terms = result.multi_index.shape[0]
        assert result.S1.shape == (2, 3)
        assert result.ST.shape == (2, 3)
        assert result.S2.shape == (2, 3, 3)
        assert result.coefficients.shape == (2, n_terms)
        assert result.loo_rmse is not None and result.loo_rmse.shape == (2,)

    def test_time_series_shapes(self, linear_pce_data):
        X, Y = linear_pce_data
        Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=-1)  # (N, K=2)
        Y3 = jnp.stack([Y2, 0.5 * Y2], axis=1)  # (N, T=2, K=2)
        result = pce.analyze(linear.PROBLEM, X, Y3, order=2)
        n_terms = result.multi_index.shape[0]
        assert result.S1.shape == (2, 2, 3)
        assert result.S2.shape == (2, 2, 3, 3)
        assert result.coefficients.shape == (2, 2, n_terms)
        assert result.loo_rmse is not None and result.loo_rmse.shape == (2, 2)

    def test_slices_match_scalar_run(self, linear_pce_data, linear_pce_result):
        """Each column of a multi-output run equals the standalone scalar run."""
        X, Y = linear_pce_data
        Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=-1)
        multi = pce.analyze(linear.PROBLEM, X, Y2, order=2)
        scalar = linear_pce_result
        # Column 0 is the untouched Y. The multi-RHS matmul reduces in a
        # different order than the scalar matvec, so agreement is to float32
        # noise, not bit-exact.
        np.testing.assert_allclose(
            multi.coefficients[0], scalar.coefficients, rtol=1e-5, atol=1e-6
        )
        assert scalar.loo_rmse is not None and multi.loo_rmse is not None
        # The linear model is exactly representable, so both LOO values are
        # float32 noise around zero — compare absolutely, not relatively.
        np.testing.assert_allclose(float(multi.loo_rmse[0]), float(scalar.loo_rmse), atol=1e-5)
        # Indices are invariant under the affine map of column 1.
        for k in range(2):
            np.testing.assert_allclose(multi.S1[k], scalar.S1, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(multi.ST[k], scalar.ST, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(
                multi.S2[k][~np.eye(3, dtype=bool)],
                np.asarray(scalar.S2)[~np.eye(3, dtype=bool)],
                atol=1e-6,
            )

    def test_emulate_multi_output_round_trip(self, linear_pce_data):
        """emulate_pce mirrors the training layout and reproduces linear Y."""
        X, Y = linear_pce_data
        Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=-1)
        Y3 = jnp.stack([Y2, 0.5 * Y2], axis=1)
        res2 = pce.analyze(linear.PROBLEM, X, Y2, order=2)
        res3 = pce.analyze(linear.PROBLEM, X, Y3, order=2)
        pred2 = pce.emulate(res2, X[:50])
        pred3 = pce.emulate(res3, X[:50])
        assert pred2.shape == (50, 2)
        assert pred3.shape == (50, 2, 2)
        # The linear benchmark is exactly representable at order 2.
        np.testing.assert_allclose(np.asarray(pred2), np.asarray(Y2[:50]), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(pred3), np.asarray(Y3[:50]), rtol=1e-5)

    def test_to_dataset_multi_output(self, linear_pce_data):
        X, Y = linear_pce_data
        Y2 = jnp.stack([Y, 2.0 * Y], axis=-1)
        ds = pce.analyze(linear.PROBLEM, X, Y2, order=2).to_dataset()
        assert ds["S1"].dims == ("output", "param")
        assert ds["S2"].dims == ("output", "param_i", "param_j")
        assert ds["loo_rmse"].dims == ("output",)

    def test_to_dataset_time_series_with_coords(self, linear_pce_data):
        X, Y = linear_pce_data
        Y3 = jnp.stack([jnp.stack([Y, 2.0 * Y], axis=-1)] * 2, axis=1)
        ds = pce.analyze(linear.PROBLEM, X, Y3, order=2).to_dataset(time_coords=[0.0, 0.5])
        assert ds["S1"].dims == ("time", "output", "param")
        assert ds["S2"].dims == ("time", "output", "param_i", "param_j")
        assert ds["loo_rmse"].dims == ("time", "output")
        np.testing.assert_allclose(ds.coords["time"].values, [0.0, 0.5])

    def test_batched_engine_matches_scalar_loop(self):
        """The einsum-based extraction equals per-slice scalar calls."""
        rng = np.random.default_rng(7)
        mi = build_multi_index(3, 3)
        coeffs = jnp.asarray(rng.normal(size=(2, 4, mi.shape[0])))
        S1b, STb, S2b = sobol_from_coefficients(coeffs, mi)
        for t in range(2):
            for k in range(4):
                S1s, STs, S2s = sobol_from_coefficients(coeffs[t, k], mi)
                np.testing.assert_allclose(np.asarray(S1b[t, k]), np.asarray(S1s), rtol=1e-12)
                np.testing.assert_allclose(np.asarray(STb[t, k]), np.asarray(STs), rtol=1e-12)
                np.testing.assert_allclose(np.asarray(S2b[t, k]), np.asarray(S2s), rtol=1e-12)

    def test_s2_pair_mask_matches_dense(self):
        """The upper-triangle pair mask reproduces the dense symmetric einsum."""
        rng = np.random.default_rng(11)
        mi = build_multi_index(4, 3)
        coeffs = np.asarray(rng.normal(size=(2, 3, mi.shape[0])), dtype=np.float64)

        # Dense reference (the pre-optimization construction) in float64.
        c2 = coeffs**2
        active = mi > 0
        active_count = active.sum(axis=1)
        total_var = c2[..., 1:].sum(axis=-1)
        inv_var = np.where(total_var == 0, np.nan, 1.0 / total_var)
        D = mi.shape[1]
        pair = active[:, :, None] & active[:, None, :] & (active_count == 2)[:, None, None]
        pair[:, np.arange(D), np.arange(D)] = False
        s2_dense = np.einsum("...t,tij->...ij", c2, pair) * inv_var[..., None, None]
        s2_dense = np.where(np.eye(D, dtype=bool), np.nan, s2_dense)

        _, _, S2 = sobol_from_coefficients(jnp.asarray(coeffs), mi)
        np.testing.assert_allclose(np.asarray(S2), s2_dense, rtol=1e-5, equal_nan=True)
