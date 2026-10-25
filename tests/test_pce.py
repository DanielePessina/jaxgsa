"""Tests for the polynomial chaos expansion (PCE) sensitivity analysis module."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning, pce
from jaxgsa.benchmarks import ishigami, linear
from jaxgsa.pce._engine import (
    _hermite_1d,
    _legendre_1d,
    build_design_matrix,
    build_multi_index,
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
def ishigami_pce_fit(ishigami_pce_data):
    """The order-8 Ishigami fit, with the warnings it raised.

    Order 8 is 165 terms on N=2000. It resolves the model well enough for the
    S2 check, and the same fit serves the explained-variance and the
    healthy-fit checks.
    """
    X, Y = ishigami_pce_data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pce.analyze(ishigami.PROBLEM, X, Y, order=8, verbose=False)
    return result, [w for w in caught if issubclass(w.category, JaxgsaWarning)]


@pytest.fixture(scope="module")
def ishigami_pce_result(ishigami_pce_fit):
    """PCE analysis result for the Ishigami benchmark model."""
    return ishigami_pce_fit[0]


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
# 1. End-to-end checks against closed forms
# ---------------------------------------------------------------------------


def test_linear_model_end_to_end(linear_pce_result, linear_pce_data):
    """Tier T0 (closed form): PCE recovers a linear additive model exactly.

    S1 matches the analytical indices. ST equals S1, because there are no
    interactions, and every off-diagonal S2 is about zero. S2 keeps its
    structure: a NaN diagonal and a symmetric matrix. The emulator reproduces
    the training data, the leave-one-out error is small, and the dataset
    carries the result's values.
    """
    X, Y = linear_pce_data
    result = linear_pce_result
    S1_analytical, _, _ = linear.analytical_indices()
    S1 = np.asarray(result.S1)
    np.testing.assert_allclose(S1, S1_analytical, atol=0.02, rtol=0.05)
    np.testing.assert_allclose(np.asarray(result.ST), S1, atol=0.02, rtol=0.05)

    S2 = np.asarray(result.S2)
    D = S2.shape[0]
    assert np.all(np.isnan(np.diag(S2))), f"S2 diagonal should be NaN, got {np.diag(S2)}"
    upper = np.triu_indices(D, k=1)
    lower = (upper[1], upper[0])
    np.testing.assert_allclose(S2[upper], S2[lower], atol=1e-10, rtol=1e-10)
    assert np.all(np.abs(S2[upper]) < 0.02), f"S2 off-diagonal should be ~0, got {S2[upper]}"

    np.testing.assert_allclose(np.asarray(result.predict(X)), np.asarray(Y), atol=0.05, rtol=0.05)

    y_range = float(jnp.max(Y) - jnp.min(Y))
    loo = float(result.cv_rmse)
    # In-sample RMSE is below the leave-one-out RMSE, and explained_variance
    # is 1 - rmse**2 / Var(Y) by definition.
    assert float(result.rmse) <= loo
    np.testing.assert_allclose(
        float(result.explained_variance),
        1.0 - float(result.rmse) ** 2 / float(jnp.var(Y)),
        atol=1e-5,
    )
    assert loo / y_range < 0.05, f"LOO RMSE / range = {loo / y_range:.4f}, expected < 0.05"

    ds = result.to_dataset()
    np.testing.assert_allclose(ds["S1"].values, S1, rtol=1e-10)
    np.testing.assert_allclose(ds["ST"].values, np.asarray(result.ST), rtol=1e-10)


def test_ishigami_end_to_end(ishigami_pce_result, ishigami_pce_data):
    """Tier T0 (closed form): S1, ST and S2 of the Ishigami function.

    The fixture fits an order 8 expansion, which measures within about 1e-3
    of the closed form on S1, ST and ``S2[x1, x3]``. The other two pairs are
    zero in closed form. Every entry takes the same absolute bound of 0.01.
    """
    result = ishigami_pce_result
    np.testing.assert_allclose(np.asarray(result.S1), ishigami.ANALYTICAL_S1, atol=0.01)
    np.testing.assert_allclose(np.asarray(result.ST), ishigami.ANALYTICAL_ST, atol=0.01)
    S2 = np.asarray(result.S2)
    np.testing.assert_allclose(S2[0, 2], ishigami.ANALYTICAL_S2[0, 2], atol=0.01)
    np.testing.assert_allclose(S2[0, 1], 0.0, atol=0.01)
    np.testing.assert_allclose(S2[1, 2], 0.0, atol=0.01)

    X, Y = ishigami_pce_data
    Y_pred = result.predict(X)
    ss_res = float(jnp.sum((Y - Y_pred) ** 2))
    ss_tot = float(jnp.sum((Y - jnp.mean(Y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot
    assert r_squared > 0.90, f"R^2 = {r_squared:.4f}, expected > 0.90"


def test_hermite_and_legendre_bases_are_orthonormal_by_quadrature():
    """The two one-dimensional bases are exactly orthonormal (Gauss quadrature).

    Ten Gauss points integrate a product of two degree-5 polynomials exactly.
    The Hermite basis uses the standard normal weight. The Legendre basis
    lives on ``[-1, 1]`` with the uniform weight 1/2. No Monte Carlo is used.
    """
    with jax.enable_x64():
        x, w = np.polynomial.hermite_e.hermegauss(10)
        H = np.asarray(_hermite_1d(jnp.asarray(x), 5))
        np.testing.assert_allclose(
            H.T @ (H * (w / np.sqrt(2 * np.pi))[:, None]), np.eye(6), atol=1e-6
        )

        x, w = np.polynomial.legendre.leggauss(10)
        P = np.asarray(_legendre_1d(jnp.asarray(x), 5))
        np.testing.assert_allclose(P.T @ (P * (w / 2)[:, None]), np.eye(6), atol=1e-6)


def test_effective_order_floors_at_one():
    """With 100 inputs and 10 rows, even order 2 has too many terms, so the floor is 1."""
    problem = Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(100)})
    assert pce.effective_order(problem, 10, order=10) == 1


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


# ---------------------------------------------------------------------------
# 7. LOO RMSE
# ---------------------------------------------------------------------------


class TestLooRmse:
    """Leave-one-out RMSE should be finite and small for well-fitted models."""

    def test_interpolating_fit_keeps_loo_finite(self):
        """T4: a fit with as many terms as rows reports a finite LOO RMSE.

        With ``N == C(D + order, order)`` and ``fit_ratio=1.0`` the least
        squares fit interpolates, every leverage is (up to the tiny ridge)
        1, and the LOO shortcut divides by ``1 - h_i``. In float32 the old
        no-op clip let that become residual/0 and an inf/NaN ``cv_rmse``
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
        assert result32.cv_rmse is not None
        assert np.all(np.isfinite(np.asarray(result32.cv_rmse)))

        with jax.enable_x64():
            key = jax.random.PRNGKey(3)
            bounds = jnp.array(ishigami.PROBLEM.bounds)
            X = jax.random.uniform(key, shape=(10, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
            Y = ishigami.evaluate(X)
            fits = []
            for batch_size in (None, 4):
                with pytest.warns(JaxgsaWarning, match="cv_rmse is"):
                    fits.append(
                        pce.analyze(
                            ishigami.PROBLEM,
                            X,
                            Y,
                            order=2,
                            fit_ratio=1.0,
                            batch_size=batch_size,
                            verbose=False,
                        )
                    )
            # Brute-force leave-one-out: the minimum-norm fit on the other
            # nine rows predicts the left-out one. The hat-matrix shortcut
            # must agree; a round-off ridge once made it read 8e-4 here.
            Xr = (
                2.0
                * (np.asarray(X) - np.asarray(bounds[:, 0]))
                / np.asarray(bounds[:, 1] - bounds[:, 0])
                - 1.0
            )
            Phi = np.asarray(
                build_design_matrix(jnp.asarray(Xr), build_multi_index(3, 2), ("uniform",) * 3, 2)
            )
            Yn = np.asarray(Y)
            errors = []
            for i in range(10):
                keep = np.arange(10) != i
                c, *_ = np.linalg.lstsq(Phi[keep], Yn[keep], rcond=None)
                errors.append(Phi[i] @ c - Yn[i])
            brute = float(np.sqrt(np.mean(np.square(errors))))
            for fit in fits:
                assert isinstance(fit, pce.PCEResult)
                assert fit.cv_rmse is not None
                np.testing.assert_allclose(float(fit.cv_rmse), brute, rtol=1e-3)


# ---------------------------------------------------------------------------
# 8. Gaussian inputs (Hermite basis)
# ---------------------------------------------------------------------------


class TestGaussianInputs:
    """PCE with Gaussian inputs should use the Hermite basis correctly."""

    def test_gaussian_linear_model_end_to_end(self, gaussian_pce_result, gaussian_pce_data):
        """S1 for the Gaussian linear model matches analytical values; the emulator fits.

        For f = c1*x1 + c2*x2 with independent inputs:
        S1_j = c_j^2 * Var(x_j) / sum(c_k^2 * Var(x_k)). Here c = [1, 2] and
        Var = [1, 4], so the contributions are [1, 16] and S1 = [1/17, 16/17].
        An error in the Hermite basis would break both checks.
        """
        expected = np.array([1.0 / 17.0, 16.0 / 17.0])
        S1 = np.asarray(gaussian_pce_result.S1)
        np.testing.assert_allclose(S1, expected, atol=0.05, rtol=0.1)

        _, X, Y = gaussian_pce_data
        Y_pred = gaussian_pce_result.predict(X)
        np.testing.assert_allclose(np.asarray(Y_pred), np.asarray(Y), atol=0.5, rtol=0.1)

    def test_basis_keeps_float32_inputs_under_x64(self):
        """A float32 x keeps a float32 PCE basis under x64.

        Both recurrences, and the truncated-normal one, follow the input
        dtype, so every input family gives the same basis dtype and float32
        data is fitted in float32. Integer input is cast to the default float.
        """
        with jax.enable_x64():
            x = jnp.linspace(-0.9, 0.9, 8, dtype=jnp.float32)
            assert _hermite_1d(x, 3).dtype == jnp.float32
            assert _legendre_1d(x, 3).dtype == jnp.float32
            X = jnp.stack([x, x], axis=1)
            Phi = build_design_matrix(X, build_multi_index(2, 2), ("uniform", "gaussian"), 2)
            assert Phi.dtype == jnp.float32
            assert _hermite_1d(jnp.arange(3), 2).dtype == jnp.float64
            assert _legendre_1d(jnp.arange(3), 2).dtype == jnp.float64
        # The NumPy path (VKOGA) always runs in float64.
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

    def test_a_truncated_gaussian_gets_its_own_recurrence_at_every_width_and_order(self):
        from jaxgsa.pce._analyze import _map_to_reference
        from jaxgsa.pce._engine import TruncatedFamily

        for q, scale in ((1e-12, 1.0), (1e-3, 0.5)):  # |z| = 7.03 and 3.09
            problem = self._problem(2, q=q)
            X = jax.random.normal(jax.random.PRNGKey(0), (32, 2)) * scale
            for order in (3, 10):
                _, input_types = _map_to_reference(X, problem, order)
                assert all(isinstance(kind, TruncatedFamily) for kind in input_types)
                assert isinstance(input_types[0], TruncatedFamily)
                assert len(input_types[0].alpha) == order
        # An untruncated Gaussian keeps Hermite.
        _, input_types = _map_to_reference(X, self._problem(2), 10)
        assert input_types == ("gaussian", "gaussian")

    def test_a_sample_beyond_a_truncation_bound_maps_to_a_finite_reference_value(self):
        """A sample at or past a bound must not turn into inf or NaN.

        The affine map is finite for any sample, which the old CDF route
        needed a dtype-aware clip to achieve.
        """
        from jaxgsa.pce._analyze import _map_to_reference

        problem = self._problem(1, q=1e-3)  # bounds at |z| = 3.09
        X = jnp.asarray([[0.0], [3.09], [40.0]], dtype=jnp.float32)
        xi, _ = _map_to_reference(X, problem, 3)
        assert np.all(np.isfinite(np.asarray(xi)))
        assert np.all(np.diff(np.asarray(xi)[:, 0]) > 0)

    @pytest.mark.slow
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
        assert float(wide.cv_rmse) == pytest.approx(float(unbounded.cv_rmse), rel=1e-4)
        assert np.max(np.abs(np.asarray(wide.S1) - analytic)) < 0.003


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

    def test_multi_output_and_time_series_layouts(self, linear_pce_data):
        """Shapes, the predict round trip and the labelled dataset, for (N, K) and (N, T, K)."""
        X, Y = linear_pce_data
        Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=-1)
        res2 = pce.analyze(linear.PROBLEM, X, Y2, order=2)
        n_terms = res2.multi_index.shape[0]
        assert res2.S1.shape == (2, 3)
        assert res2.ST.shape == (2, 3)
        assert res2.S2.shape == (2, 3, 3)
        assert res2.coefficients.shape == (2, n_terms)
        assert res2.cv_rmse is not None and res2.cv_rmse.shape == (2,)

        Y3 = jnp.stack([Y2, 0.5 * Y2], axis=1)  # (N, T=2, K=2)
        res3 = pce.analyze(linear.PROBLEM, X, Y3, order=2)
        assert res3.S1.shape == (2, 2, 3)
        assert res3.S2.shape == (2, 2, 3, 3)
        assert res3.coefficients.shape == (2, 2, n_terms)
        assert res3.cv_rmse is not None and res3.cv_rmse.shape == (2, 2)

        # predict mirrors the training layout. The linear benchmark is
        # exactly representable at order 2.
        pred2 = res2.predict(X[:50])
        pred3 = res3.predict(X[:50])
        assert pred2.shape == (50, 2)
        assert pred3.shape == (50, 2, 2)
        np.testing.assert_allclose(np.asarray(pred2), np.asarray(Y2[:50]), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(pred3), np.asarray(Y3[:50]), rtol=1e-5)

        ds = res3.to_dataset(time_coords=[0.0, 0.5])
        assert ds["S1"].dims == ("time", "output", "param")
        assert ds["S2"].dims == ("time", "output", "param_i", "param_j")
        assert ds["cv_rmse"].dims == ("time", "output")
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
        assert result.cv_rmse is not None
        assert np.all(np.isfinite(np.asarray(result.cv_rmse)))
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

    # One bootstrap size for every test here, so they share compiles.
    R = 6

    @staticmethod
    def _data(n=256, seed=0):
        rng = np.random.default_rng(seed)
        X = jnp.asarray(rng.uniform(-np.pi, np.pi, size=(n, 3)))
        return X, ishigami.evaluate(X)

    def test_interval_brackets_the_estimate_and_is_reproducible(self):
        """T4: one run covers the default, the bracket, the key rules and the layout.

        The plainest call reports no interval and pays for none: each
        replicate refits the expansion, so an on-by-default interval would
        make a routine call an order of magnitude slower. When asked, the
        interval brackets the estimate for all three index arrays, and the
        draw is a function of the key, not of the call. No key means no
        interval, and it is refused rather than seeded, because falling back
        to a constant key would silently correlate every nested or repeated
        bootstrap.
        """
        X, Y = self._data()
        plain = pce.analyze(ishigami.PROBLEM, X, Y, order=2)
        assert plain.ci is None
        assert plain.S1_conf is None
        assert plain.ST_conf is None
        assert plain.S2_conf is None

        kwargs = dict(order=2, n_bootstrap=self.R, key=jax.random.key(0))
        result = pce.analyze(ishigami.PROBLEM, X, Y, **kwargs)
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
            # a small slack is expected on a short run.
            assert (point[finite] >= lower[finite] - 0.2).all()
            assert (point[finite] <= upper[finite] + 0.2).all()

        again = pce.analyze(ishigami.PROBLEM, X, Y, **kwargs)
        np.testing.assert_array_equal(np.asarray(result.S1_conf), np.asarray(again.S1_conf))

        with pytest.raises(ValueError, match="key is required"):
            pce.analyze(ishigami.PROBLEM, X, Y, n_bootstrap=self.R)

    def test_time_series_intervals_record_their_recipe_and_draws(self):
        """T4: the interval keeps the result's layout, plus the [lo, hi] axis.

        It also records its recipe (level, method, size) and optionally
        exposes its draws.
        """
        X, Y = self._data()
        Y3 = jnp.stack([jnp.stack([Y, 2.0 * Y], axis=-1)], axis=1)
        result = pce.analyze(
            ishigami.PROBLEM,
            X,
            Y3,
            order=2,
            n_bootstrap=self.R,
            conf_level=0.8,
            ci_method="gaussian",
            key=jax.random.key(2),
            keep_replicates=True,
        )

        assert result.S1.shape == (1, 2, 3)
        assert result.S1_conf.shape == (2, 1, 2, 3)
        assert result.S2_conf.shape == (2, 1, 2, 3, 3)
        assert result.ci is not None
        assert result.ci.level == 0.8
        assert result.ci.method == "gaussian"
        assert result.ci.n_bootstrap == self.R
        assert result.ci.replicates is not None
        assert result.ci.replicates["S1"].shape == (self.R, 1, 2, 3)
        assert result.ci.replicates["S2"].shape == (self.R, 1, 2, 3, 3)


class TestExplainedVariance:
    """``explained_variance`` is the in-sample R^2, so it stays within [0, 1].

    Regression cover for a fault where the numerator was the sum of squared
    non-constant coefficients, which is the surrogate's variance under the
    *input measure*, while the denominator was the *sample* variance of ``Y``.
    At finite ``N`` the empirical Gram is not the identity, so the two
    disagree, and an order-8 Ishigami fit on ``N=2000`` reported 1.0709 for a
    fit whose true R^2 was 0.9997.
    """

    def test_high_order_ishigami_does_not_exceed_one(self, ishigami_pce_fit):
        """The exact case that used to report 1.0709.

        A well-resolved fit also raises no fit-quality warning. Order 8 on
        N=2000 is 165 terms, a healthy ratio, so a warning here would be a
        false alarm.
        """
        result, warned = ishigami_pce_fit
        ev = float(np.asarray(result.explained_variance))
        assert 0.0 <= ev <= 1.0, f"explained_variance out of range: {ev}"
        assert ev > 0.99, f"an order-8 Ishigami fit should be near-perfect, got {ev}"
        assert warned == []

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
        assert jnp.all(jnp.isnan(result.S1))
        assert jnp.all(jnp.isnan(result.ST))
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
            shape=(500, problem.n_params),
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

    def test_underfit_warns_and_explained_variance_is_r_squared(self, ishigami_pce_data):
        """order=3 on Ishigami explains under half the variance, and says so.

        ``explained_variance`` also equals ``1 - SS_res / SS_tot`` of the
        fitted values.
        """
        X, Y = ishigami_pce_data
        with pytest.warns(JaxgsaWarning, match=r"explained_variance is 0\.\d+"):
            result = pce.analyze(ishigami.PROBLEM, X, Y, order=3, verbose=False)
        ev = float(np.asarray(result.explained_variance))
        assert ev < 0.5
        residual = Y - result.predict(X)
        r2 = 1.0 - float(jnp.sum(residual**2)) / float(jnp.sum((Y - jnp.mean(Y)) ** 2))
        np.testing.assert_allclose(ev, r2, rtol=1e-3, atol=1e-4)

    def test_overfit_warns_on_the_loo_ratio(self, ishigami_pce_data):
        """A fit with too few rows per term is caught out of sample only."""
        X, Y = ishigami_pce_data
        Xs, Ys = X[:64], Y[:64]
        with pytest.warns(JaxgsaWarning, match=r"cv_rmse is \d+\.\d+ times std\(Y\)"):
            result = pce.analyze(ishigami.PROBLEM, Xs, Ys, order=5, fit_ratio=0.9, verbose=False)
        # The in-sample number reports nothing wrong at all, which is the
        # reason the out-of-sample check exists.
        assert float(np.asarray(result.explained_variance)) > 0.9
