"""Tests for the polynomial chaos expansion (PCE) sensitivity analysis module."""

from __future__ import annotations

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
    sobol_from_coefficients,
)
from jaxgsa.problem import GaussianInputSpec, Problem

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

    def test_loo_matches_brute_force_refit(self):
        """Tier T0 (closed form): the reported LOO RMSE equals a refit-per-point one.

        ``loo_error`` uses the hat-matrix shortcut
        ``e_i = (Y_i - Phi_i c) / (1 - H_ii)``. This test never writes that
        expression down. It refits the ridge least-squares problem N times, each
        time leaving one row out, and takes the root mean square of the N held-out
        residuals. That is the definition of leave-one-out error, so the two
        numbers must agree.
        """
        key = jax.random.PRNGKey(7)
        N = 120
        bounds = jnp.array(ishigami.PROBLEM.bounds)
        X = jax.random.uniform(key, shape=(N, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
        Y = ishigami.evaluate(X)
        result = pce.analyze(ishigami.PROBLEM, X, Y, order=3)

        from jaxgsa.pce._analyze import _map_to_reference

        X_ref, input_types = _map_to_reference(X, ishigami.PROBLEM, result.order)
        Phi = np.asarray(
            build_design_matrix(X_ref, result.multi_index, input_types, result.order),
            dtype=np.float64,
        )
        Y_np = np.asarray(Y, dtype=np.float64)
        ridge = 1e-8
        n_terms = Phi.shape[1]

        held_out = np.empty(N, dtype=np.float64)
        for i in range(N):
            rows = np.delete(np.arange(N), i)
            Phi_i, Y_i = Phi[rows], Y_np[rows]
            gram = Phi_i.T @ Phi_i + ridge * np.eye(n_terms)
            coef = np.linalg.solve(gram, Phi_i.T @ Y_i)
            held_out[i] = Y_np[i] - Phi[i] @ coef
        brute_force = float(np.sqrt(np.mean(held_out**2)))

        assert result.loo_rmse is not None
        np.testing.assert_allclose(float(result.loo_rmse), brute_force, rtol=2e-3)

    def test_hat_diagonal_matches_explicit_hat_matrix(self):
        """Tier T0 (closed form): leverage equals the hat matrix's own diagonal.

        ``hat_diagonal`` never forms ``H``. It takes the Cholesky factor ``L``
        of ``Phi^T Phi + ridge*I`` and returns ``|| L^{-1} phi_i ||^2``. This
        test builds the full ``N x N`` hat matrix
        ``H = Phi (Phi^T Phi + ridge*I)^{-1} Phi^T`` in float64 from its
        definition and reads its diagonal. The two must agree, because they
        are the same quantity written two ways.
        """
        key = jax.random.PRNGKey(11)
        N, ridge = 90, 1e-8
        bounds = jnp.array(ishigami.PROBLEM.bounds)
        X = jax.random.uniform(key, shape=(N, 3), minval=bounds[:, 0], maxval=bounds[:, 1])

        from jaxgsa.pce._analyze import _map_to_reference

        X_ref, input_types = _map_to_reference(X, ishigami.PROBLEM, 3)
        Phi = build_design_matrix(X_ref, build_multi_index(3, 3), input_types, 3)

        got = np.asarray(hat_diagonal(Phi, gram_cholesky(Phi, ridge)))

        Phi_np = np.asarray(Phi, dtype=np.float64)
        gram = Phi_np.T @ Phi_np + ridge * np.eye(Phi_np.shape[1])
        H = Phi_np @ np.linalg.solve(gram, Phi_np.T)
        np.testing.assert_allclose(got, np.diag(H), rtol=1e-4, atol=1e-6)


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
        from scipy.stats import norm

        spec = {"dist": "gaussian", "mean": 0.0, "variance": 1.0}
        if q is not None:
            spec = {**spec, "low": float(norm.ppf(q)), "high": float(norm.ppf(1.0 - q))}
        return Problem.from_dict({f"x{i + 1}": dict(spec) for i in range(D)})

    def test_wide_truncation_uses_hermite(self):
        from jaxgsa.pce._analyze import _map_to_reference

        problem = self._problem(2, q=1e-12)  # |z| = 7.03
        X = jax.random.normal(jax.random.PRNGKey(0), (32, 2))
        _, input_types = _map_to_reference(X, problem, 3)
        assert input_types == ("gaussian", "gaussian")

    def test_narrow_truncation_uses_legendre(self):
        from jaxgsa.pce._analyze import _map_to_reference

        problem = self._problem(2, q=1e-3)  # |z| = 3.09
        X = jax.random.normal(jax.random.PRNGKey(0), (32, 2)) * 0.5
        _, input_types = _map_to_reference(X, problem, 3)
        assert input_types == ("uniform", "uniform")

    def test_high_order_falls_back_to_legendre(self):
        """The Hermite Gram defect grows with degree, so a high order drops back."""
        from jaxgsa.pce._analyze import _map_to_reference

        problem = self._problem(2, q=1e-12)
        X = jax.random.normal(jax.random.PRNGKey(0), (32, 2))
        _, input_types = _map_to_reference(X, problem, 10)
        assert input_types == ("uniform", "uniform")

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

    def test_build_multi_index_count(self):
        """Multi-index set size should equal C(D+p, p)."""
        from math import comb

        D, p = 3, 4
        mi = build_multi_index(D, p)
        assert mi.shape == (comb(D + p, p), D)

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
        """The masked-matmul extraction equals per-slice scalar calls.

        Batched and per-slice paths do the same float32 reductions but XLA may
        fuse/order them differently, so compare at float32 precision rather than
        demanding bitwise-identical lowering.
        """
        rng = np.random.default_rng(7)
        mi = build_multi_index(3, 3)
        coeffs = jnp.asarray(rng.normal(size=(2, 4, mi.shape[0])))
        S1b, STb, S2b = sobol_from_coefficients(coeffs, mi)
        for t in range(2):
            for k in range(4):
                S1s, STs, S2s = sobol_from_coefficients(coeffs[t, k], mi)
                np.testing.assert_allclose(np.asarray(S1b[t, k]), np.asarray(S1s), rtol=1e-6)
                np.testing.assert_allclose(np.asarray(STb[t, k]), np.asarray(STs), rtol=1e-6)
                np.testing.assert_allclose(np.asarray(S2b[t, k]), np.asarray(S2s), rtol=1e-6)

    def test_predict_preserves_explicit_time_series_layout(self):
        """Training on (N, T, K) yields (N_new, T, K) predictions."""
        problem = Problem(names=("x0", "x1", "x2"), bounds=((0.0, 1.0),) * 3, output_names=("p",))
        key = jax.random.PRNGKey(4)
        X = jax.random.uniform(key, shape=(500, 3))
        base = jnp.sin(X[:, 0]) + 2.0 * X[:, 1]
        Y = jnp.stack([base, 2.0 * base], axis=-1)[..., None]
        result = pce.analyze(problem, X, Y)
        X_new = jax.random.uniform(jax.random.PRNGKey(5), shape=(30, 3))
        pred = result.predict(X_new)
        assert pred.shape == (30, 2, 1)


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

    def test_drop_matches_a_hand_filtered_fit(self):
        """T4: dropping row i equals never passing row i.

        The point is alignment. If the mask reached Y but not X, every later
        pair would be mismatched and no index could detect it.
        """
        X, Y = _invalid_data()
        Y_bad = Y.copy()
        Y_bad[7] = np.nan
        with pytest.warns(JaxgsaWarning):
            dropped = pce.analyze(linear.PROBLEM, X, Y_bad, on_invalid="drop", order=2)
        keep = np.ones(len(Y), dtype=bool)
        keep[7] = False
        manual = pce.analyze(linear.PROBLEM, X[keep], Y[keep], order=2)
        np.testing.assert_allclose(
            np.asarray(dropped.S1), np.asarray(manual.S1), rtol=1e-5, atol=1e-6
        )

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

    def test_shapley_carries_the_report_through(self):
        """T4: the report survives the analytical Shapley step unchanged."""
        X, Y = _invalid_data()
        Y = Y.copy()
        Y[7] = np.nan
        with pytest.warns(JaxgsaWarning):
            result = pce.analyze(linear.PROBLEM, X, Y, on_invalid="drop", order=2)
        assert result.shapley().invalid == result.invalid
