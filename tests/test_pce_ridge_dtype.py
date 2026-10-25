"""Regression tests for the 1.3.1 PCE, Shapley and DGSM audit.

- The float64 default ridge is relative to the Gram matrix, so inputs that do
  not fill their declared bounds give exact indices (audit finding 1).
- float32 data under x64 computes and returns float32 (finding 4).
- Public paths that no test ran before (finding 7).
"""

import warnings
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.pce import PCEResult
from jaxgsa.shapley import ShapleyResult

D = 4
UNIT = jaxgsa.Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(D)})
GAUSS = jaxgsa.Problem.from_dict(
    {f"x{i}": {"dist": "gaussian", "mean": 0.0, "variance": 1.0} for i in range(D)}
)
TRUNC = jaxgsa.Problem.from_dict(
    {
        f"x{i}": {"dist": "gaussian", "mean": 0.0, "variance": 1.0, "low": -1.0, "high": 2.0}
        for i in range(D)
    }
)


def _additive_cubic(X):
    """Additive cubic: the exact expansion has ST == S1 and no S2."""
    return X[:, 0] + 2 * X[:, 1] ** 2 + 0.5 * X[:, 2] ** 3 + X[:, 3]


# Exact indices of _additive_cubic on U(0, 1)^4, from a full-range fit.
_EXACT_S1 = np.array([0.15372, 0.65564, 0.03699, 0.15372])


def _narrow(lo, width, n=4096):
    rng = np.random.default_rng(0)
    return lo + rng.uniform(0, 1, (n, D)) * width


# --- finding 1: the float64 default ridge -----------------------------------


@pytest.mark.parametrize(("lo", "width", "order"), [(0.0, 0.3, 6), (0.495, 0.01, 3)])
def test_f64_narrow_inputs_give_exact_indices(lo, width, order):
    """Inputs that fill only part of their bounds: the indices stay exact.

    With the old absolute ridge 1e-8, sum(ST - S1) was 0.73 and 0.47 here, with
    a perfect cv_rmse and no warning.
    """
    with jax.enable_x64():
        X = jnp.asarray(_narrow(lo, width))
        Y = _additive_cubic(X)
        with warnings.catch_warnings():
            warnings.simplefilter("error", JaxgsaWarning)
            r = cast(PCEResult, jaxgsa.pce.analyze(UNIT, X, Y, order=order, verbose=False))
            sh = cast(
                ShapleyResult, jaxgsa.shapley.analyze(UNIT, X, Y, order=order, verbose=False)
            )
        idx = jaxgsa.pce.indices(UNIT, X, Y, order=order)
    S1, ST = np.asarray(r.S1), np.asarray(r.ST)
    assert np.sum(ST - S1) < 1e-5
    np.testing.assert_allclose(S1, _EXACT_S1, atol=1e-3)
    np.testing.assert_allclose(np.asarray(sh.Sh), _EXACT_S1, atol=1e-3)
    np.testing.assert_allclose(np.asarray(idx.ST), ST, rtol=1e-10, atol=1e-12)


def test_f64_explicit_ridge_that_dominates_warns():
    """A caller's ridge that sets the fit along a data direction warns."""
    with jax.enable_x64():
        X = jnp.asarray(_narrow(0.0, 0.3))
        with pytest.warns(JaxgsaWarning, match="indices depend on the ridge"):
            jaxgsa.pce.analyze(UNIT, X, _additive_cubic(X), order=6, ridge=1e-8, verbose=False)


def test_f64_ridge_warning_counts_bootstrap_replicates():
    with jax.enable_x64():
        X = jnp.asarray(_narrow(0.0, 0.3, n=512))
        with pytest.warns(JaxgsaWarning, match="bootstrap replicates, doubling the ridge"):
            jaxgsa.pce.analyze(
                UNIT,
                X,
                _additive_cubic(X),
                order=4,
                ridge=1e-4,
                n_bootstrap=3,
                key=jax.random.key(0),
                verbose=False,
            )


def test_f64_default_ridge_matches_old_ridge_on_full_range_data():
    """On data that fills its bounds the relative default changes nothing."""
    with jax.enable_x64():
        X = jnp.asarray(jaxgsa.sampling.monte_carlo(UNIT, 1024, seed=3))
        Y = jnp.sin(3 * X[:, 0]) + X[:, 1] * X[:, 2] ** 2 + 0.2 * X[:, 3]
        new = jaxgsa.pce.analyze(UNIT, X, Y, order=5, verbose=False)
        old = jaxgsa.pce.analyze(UNIT, X, Y, order=5, ridge=1e-8, verbose=False)
    for field in ("S1", "ST", "S2"):
        a, b = np.asarray(getattr(new, field)), np.asarray(getattr(old, field))
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-12)


def test_f64_rank_warning_through_analyze():
    """An input that copies another up to small noise warns in float64."""
    with jax.enable_x64():
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (2000, D))
        X[:, 1] = np.clip(X[:, 0] + 1e-4 * rng.standard_normal(2000), 0, 1)
        X = jnp.asarray(X)
        Y = X[:, 0] + 0.5 * X[:, 2] ** 3 + X[:, 3]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = cast(PCEResult, jaxgsa.pce.analyze(UNIT, X, Y, order=3, verbose=False))
    messages = [str(w.message) for w in caught]
    assert any("near singular" in m for m in messages)
    # The rank warning covers it; the ridge warning does not repeat it.
    assert not any("depend on the ridge" in m for m in messages)
    assert r.S1.dtype == jnp.float64


# --- finding 4: dtype --------------------------------------------------------


@pytest.mark.parametrize("problem", [UNIT, GAUSS, TRUNC], ids=["uniform", "gaussian", "trunc"])
def test_float32_data_stays_float32_under_x64(problem):
    with jax.enable_x64():
        X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 1024, seed=0), dtype=jnp.float32)
        Y = (X[:, 0] + X[:, 1] * X[:, 2]).astype(jnp.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", JaxgsaWarning)
            r = cast(PCEResult, jaxgsa.pce.analyze(problem, X, Y, order=3, verbose=False))
            sh = r.shapley()
        idx = jaxgsa.pce.indices(problem, X, Y, order=3, batch_size=256)
    for arr in (r.S1, r.ST, r.S2, r.coefficients, r.cv_rmse, sh.Sh, idx.S1, idx.S2):
        assert arr is not None
        assert arr.dtype == jnp.float32


def test_dgsm_float32_bounds_under_x64():
    with jax.enable_x64():
        X = jnp.asarray(jaxgsa.sampling.monte_carlo(UNIT, 256, seed=0), dtype=jnp.float32)
        r = jaxgsa.dgsm.analyze(UNIT, X, fn=lambda x: jnp.sum(x**2), verbose=False)
    for arr in (r.nu, r.mu, r.upper_bound, r.lower_bound):
        assert arr.dtype == jnp.float32


def test_legendre_numpy_path_stays_float64():
    """VKOGA calls the shared recurrence with NumPy arrays: float64 always."""
    from jaxgsa._core.legendre import legendre_orthonormal

    out = legendre_orthonormal(np.linspace(-1, 1, 5, dtype=np.float32), 3)
    assert out.dtype == np.float64


# --- finding 7: paths no test ran -------------------------------------------


def _tol():
    return 1e-10 if jax.config.jax_enable_x64 else 2e-4


@pytest.mark.parametrize("problem", [UNIT, GAUSS, TRUNC], ids=["uniform", "gaussian", "trunc"])
def test_streamed_indices_without_diagnostics_match_single_pass(problem):
    """pce.indices / shapley.indices with batch_size run the streamed fit
    with diagnostics off, on every input family."""
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 1000, seed=1))
    Y = jnp.stack([X[:, 0] + X[:, 1] * X[:, 2], X[:, 3] ** 2 + X[:, 0]], axis=1)
    whole = jaxgsa.pce.indices(problem, X, Y, order=3)
    streamed = jaxgsa.pce.indices(problem, X, Y, order=3, batch_size=128)
    for field in ("S1", "ST", "S2"):
        np.testing.assert_allclose(
            np.asarray(getattr(streamed, field)),
            np.asarray(getattr(whole, field)),
            atol=_tol(),
        )
    sh_whole = jaxgsa.shapley.indices(problem, X, Y, order=3)
    sh_streamed = jaxgsa.shapley.indices(problem, X, Y, order=3, batch_size=128)
    np.testing.assert_allclose(np.asarray(sh_streamed.Sh), np.asarray(sh_whole.Sh), atol=_tol())


def test_shapley_bootstrap_hdmr_backend():
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(UNIT, 512, seed=2))
    Y = X[:, 0] + 2 * X[:, 1] ** 2 + X[:, 0] * X[:, 2]
    r = cast(
        ShapleyResult,
        jaxgsa.shapley.analyze(
            UNIT, X, Y, backend="hdmr", n_bootstrap=8, key=jax.random.key(0), verbose=False
        ),
    )
    assert r.Sh_conf is not None
    assert r.Sh_conf.shape == (2, D)
    lo, hi = np.asarray(r.Sh_conf)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(lo <= hi)
    # The two biggest effects are inside their own intervals.
    sh = np.asarray(r.Sh)
    for i in (0, 1):
        assert lo[i] - 0.05 <= sh[i] <= hi[i] + 0.05


def _nan_model(x):
    return jnp.where(x[0] > 0.9, jnp.nan, jnp.sin(x[0]) + x[1] ** 2 + x[0] * x[2])


def test_dgsm_bootstrap_after_dropped_rows():
    """on_invalid='drop' with a bootstrap re-sweeps the kept rows; batch widths agree."""
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(UNIT, 400, seed=4))
    key = jax.random.key(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole, batched = (
            jaxgsa.dgsm.analyze(
                UNIT,
                X,
                fn=_nan_model,
                on_invalid="drop",
                n_bootstrap=16,
                key=key,
                batch_size=batch_size,
                verbose=False,
            )
            for batch_size in (None, 64)
        )
    kept = X[np.asarray(X[:, 0]) <= 0.9]
    clean = jaxgsa.dgsm.analyze(UNIT, kept, fn=_nan_model, verbose=False)
    tol = 1e-10 if jax.config.jax_enable_x64 else 1e-5
    np.testing.assert_allclose(np.asarray(whole.nu), np.asarray(clean.nu), rtol=tol)
    assert whole.nu_conf is not None and batched.nu_conf is not None
    np.testing.assert_allclose(np.asarray(batched.nu_conf), np.asarray(whole.nu_conf), rtol=tol)
    lo, hi = np.asarray(whole.nu_conf)
    assert np.all(lo <= np.asarray(whole.nu)) and np.all(np.asarray(whole.nu) <= hi)


def test_dgsm_bootstrap_with_user_dfdx_matches_autodiff():
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(UNIT, 300, seed=5))

    def f(x):
        return jnp.sin(x[0]) + x[1] ** 2 + x[0] * x[2]

    Y = jax.vmap(f)(X)
    J = jax.vmap(jax.grad(f))(X)
    key = jax.random.key(2)
    auto = jaxgsa.dgsm.analyze(UNIT, X, fn=f, n_bootstrap=16, key=key, verbose=False)
    given = jaxgsa.dgsm.analyze(UNIT, X, Y, dfdx=J, n_bootstrap=16, key=key, verbose=False)
    batched = jaxgsa.dgsm.analyze(
        UNIT, X, Y, dfdx=J, n_bootstrap=16, key=key, batch_size=64, verbose=False
    )
    tol = 1e-10 if jax.config.jax_enable_x64 else 1e-5
    for name in ("nu_conf", "upper_bound_conf", "lower_bound_conf"):
        a = np.asarray(getattr(auto, name))
        np.testing.assert_allclose(np.asarray(getattr(given, name)), a, rtol=tol, atol=tol)
        np.testing.assert_allclose(np.asarray(getattr(batched, name)), a, rtol=tol, atol=tol)


# --- review follow-ups -------------------------------------------------------


def test_f64_singular_discrete_design_is_minimum_norm_and_batch_stable():
    """A two-level input makes the design singular to round-off.

    The default ridge then switches to the stable relative value, so the null
    directions carry the minimum-norm solution, not noise over round-off, and
    the row batch width does not move the indices.
    """
    from jaxgsa.pce._analyze import _map_to_reference
    from jaxgsa.pce._engine import build_design_matrix, build_multi_index

    problem = jaxgsa.Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(3)})
    with jax.enable_x64():
        rng = np.random.default_rng(0)
        X = np.array([0.0, 1.0])[rng.integers(0, 2, (400, 3))]
        Y = X[:, 0] + 2 * X[:, 1] ** 2 + X[:, 0] * X[:, 2] + 0.1 * rng.standard_normal(400)
        Xr, types = _map_to_reference(jnp.asarray(X), problem, 6)
        Phi = np.asarray(build_design_matrix(Xr, build_multi_index(3, 6), types, 6))
        pinv = np.linalg.pinv(Phi) @ Y
        fits = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", JaxgsaWarning)
            for batch_size in (None, 100, 64):
                r = cast(
                    PCEResult,
                    jaxgsa.pce.analyze(
                        problem,
                        jnp.asarray(X),
                        jnp.asarray(Y),
                        order=6,
                        batch_size=batch_size,
                        verbose=False,
                    ),
                )
                fits.append(r)
    # Measured: 6e-6 and 2e-5. A round-off ridge gave 3e-2 and 7e-2.
    for r in fits:
        np.testing.assert_allclose(np.asarray(r.coefficients), pinv, atol=5e-5)
        np.testing.assert_allclose(np.asarray(r.ST), np.asarray(fits[0].ST), atol=5e-5)


def test_mixed_dtype_fits_in_the_wider_dtype():
    """float32 X with float64 Y under x64 fits in float64 and keeps Y's digits."""
    from jaxgsa.benchmarks import ishigami

    with jax.enable_x64():
        X = jnp.asarray(jaxgsa.sampling.monte_carlo(ishigami.PROBLEM, 3000, seed=0))
        Y = ishigami.evaluate(X) + 1e6
        both = cast(PCEResult, jaxgsa.pce.analyze(ishigami.PROBLEM, X, Y, order=6, verbose=False))
        mixed = cast(
            PCEResult,
            jaxgsa.pce.analyze(ishigami.PROBLEM, X.astype(jnp.float32), Y, order=6, verbose=False),
        )
    assert mixed.ST.dtype == jnp.float64
    # X rounded to float32 moves the inputs by about 1e-7, nothing more.
    np.testing.assert_allclose(np.asarray(mixed.ST), np.asarray(both.ST), atol=1e-6)
    assert mixed.cv_rmse is not None and both.cv_rmse is not None
    np.testing.assert_allclose(float(mixed.cv_rmse), float(both.cv_rmse), rtol=1e-6)


@pytest.mark.parametrize("levels", [4, 5])
def test_f64_default_ridge_is_continuous_in_round_off_jitter(levels):
    """Jittering a discrete design by round-off must not move the indices.

    A hard switch between two ridges at a pivot-share line made ST(x0) jump
    from 0.37 to 0.82 at a jitter of 1e-13 (4 levels). The old absolute 1e-8
    drifted to 0.56 at 1e-11.
    """
    problem = jaxgsa.Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(3)})
    st = []
    with jax.enable_x64():
        for jitter in (0.0, 1e-14, 1e-13, 1e-12, 1e-11):
            rng = np.random.default_rng(0)
            grid = np.linspace(0, 1, levels)
            X = grid[rng.integers(0, levels, (400, 3))] + jitter * rng.uniform(-1, 1, (400, 3))
            X = np.clip(X, 0, 1)
            Y = X[:, 0] + 2 * X[:, 1] ** 2 + X[:, 0] * X[:, 2] + 0.1 * rng.standard_normal(400)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                r = cast(
                    PCEResult,
                    jaxgsa.pce.analyze(
                        problem, jnp.asarray(X), jnp.asarray(Y), order=6, verbose=False
                    ),
                )
            # The rank check still reads the near-singular design.
            assert any("near singular" in str(w.message) for w in caught)
            st.append(np.asarray(r.ST))
    np.testing.assert_allclose(np.stack(st), np.broadcast_to(st[0], (5, 3)), atol=1e-3)


def _gauss_case(n, order):
    from jaxgsa.pce._engine import build_design_matrix, build_multi_index

    problem = jaxgsa.Problem.from_dict(
        {f"x{i}": {"dist": "gaussian", "mean": 0.0, "variance": 1.0} for i in range(2)}
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 2))
    Y = np.sin(X[:, 0]) + 0.3 * X[:, 1] ** 2 + 0.2 * X[:, 0] * X[:, 1]
    Phi = np.asarray(
        build_design_matrix(jnp.asarray(X), build_multi_index(2, order), ("gaussian",) * 2, order)
    )
    Q, R = np.linalg.qr(Phi)
    e = Y - Phi @ np.linalg.solve(R, Q.T @ Y)
    exact = float(np.sqrt(np.mean((e / (1 - np.sum(Q * Q, axis=1))) ** 2)))
    return problem, X, Y, exact


def test_f64_accurate_fit_cv_rmse_matches_exact_loo():
    """cv_rmse of an accurate fit is the exact leave-one-out error.

    A leave-one-out ridge floor applied to every row once inflated it 90
    times here (4.5e-5 against 5.0e-7). The floor now only replaces rows the
    fit interpolates, and never with an explicit ridge.
    """
    with jax.enable_x64():
        problem, X, Y, exact = _gauss_case(3000, 14)
        for ridge in (None, 0.0):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                r = cast(
                    PCEResult,
                    jaxgsa.pce.analyze(
                        problem,
                        jnp.asarray(X),
                        jnp.asarray(Y),
                        order=14,
                        ridge=ridge,
                        verbose=False,
                    ),
                )
            assert r.cv_rmse is not None
            np.testing.assert_allclose(float(r.cv_rmse), exact, rtol=1e-2)
            assert not any("depend on the ridge" in str(w.message) for w in caught)


def test_f64_default_ridge_does_not_warn_on_an_accurate_high_order_fit():
    """Order 20 on Gaussian inputs: the default ridge is 1e-3 of the smallest
    Gram eigenvalue, but the indices do not move with it, so no warning."""
    with jax.enable_x64():
        problem, X, Y, _ = _gauss_case(3000, 20)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jaxgsa.pce.analyze(problem, jnp.asarray(X), jnp.asarray(Y), order=20, verbose=False)
    assert not any("depend on the ridge" in str(w.message) for w in caught)
