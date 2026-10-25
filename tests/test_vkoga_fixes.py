"""Regression tests for float32 VKOGA fits on data with duplicate rows."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import jaxgsa
from jaxgsa.vkoga import VKOGAResult
from jaxgsa.vkoga._engine import _fit_vkoga


def _duplicated_unit_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    rng = np.random.default_rng(0)
    base = rng.random((60, 2)).astype(np.float32)
    rows = np.concatenate([base, base[:30], base[:10]])
    y = (rows[:, 0] + 2 * rows[:, 1] ** 2).astype(np.float32)
    return jnp.asarray(rows), jnp.asarray((y - y.mean())[:, None])


def test_float32_fit_with_duplicate_rows_is_finite():
    U, Y = _duplicated_unit_data()
    for gamma in (1.0, 10.0, 100.0):
        state = _fit_vkoga(U, Y, gamma=gamma, max_centers=100, ridge=1e-6)
        assert np.isfinite(np.asarray(state.coefficients)).all()
        assert np.isfinite(np.asarray(state.rmse)).all()
        # Exact copies must never both be centres.
        centers = np.asarray(state.centers[: int(state.n_centers)])
        assert len(np.unique(centers, axis=0)) == len(centers)


def test_float32_bootstrap_intervals_are_finite():
    R = np.array([[1, 0.6], [0.6, 1]])
    p = jaxgsa.Problem(["a", "b"], [(0, 1), (0, 1)], correlation=R)
    X = jaxgsa.sampling.monte_carlo(p, 256, seed=0)
    X_np = np.asarray(X)
    Y = jnp.asarray(X_np[:, 0] + 2 * X_np[:, 1] ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = jaxgsa.vkoga.analyze(p, jnp.asarray(X), Y, key=0, n_bootstrap=8, verbose=False)
    assert isinstance(r, VKOGAResult)
    assert np.isfinite(np.asarray(r.S_TC)).all()
    assert np.isfinite(np.asarray(r.S_TC_conf)).all()


def test_nonfinite_fit_warns_at_caller(monkeypatch):
    """A singular fit warns once for the point fit and once for the replicates.

    Both warnings must point at this file, not at jaxgsa internals.
    """
    from jaxgsa.vkoga import _engine

    def _nan_solve(eq, ridge):
        return jnp.full_like(eq.VtY, jnp.nan)

    monkeypatch.setattr(_engine, "_solve_ridge", _nan_solve)
    p = jaxgsa.Problem(["a", "b"], [(0, 1), (0, 1)])
    # An odd row count gives fresh jit traces, so the patched solve is used.
    X = jaxgsa.sampling.monte_carlo(p, 97, seed=3)
    X_np = np.asarray(X)
    Y = jnp.asarray(X_np[:, 0] + 2 * X_np[:, 1] ** 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jaxgsa.vkoga.analyze(
            p, jnp.asarray(X), Y, key=0, gamma=1.0, ridge=1e-6, n_bootstrap=2, verbose=False
        )
    coeff = [w for w in caught if "non-finite coefficients" in str(w.message)]
    boot = [w for w in caught if "bootstrap replicates have non-finite" in str(w.message)]
    assert len(coeff) == 1
    assert len(boot) == 1
    assert "2 of 2" in str(boot[0].message)
    for w in coeff + boot:
        assert w.filename == __file__


def test_singular_solve_retries_eager_and_traced():
    """A singular Newton system gives NaN at the default jitter; the retry fixes it.

    ``V^T V = [[1, 1], [1, 1]]`` with ``ridge = 0``: the default jitter
    (1e-8) is below float32 eps, so the first solve divides by an exact zero
    pivot. The retry's ten-eps jitter makes the system solvable.
    """
    import jax

    from jaxgsa.vkoga._engine import _JITTER, _NormalEquations, _solve_ridge

    f32 = jnp.float32
    eq = _NormalEquations(
        active=jnp.ones(2, dtype=bool),
        centers=jnp.zeros((2, 2), dtype=f32),
        L=jnp.eye(2, dtype=f32),
        VtV=jnp.ones((2, 2), dtype=f32),
        VtY=jnp.ones((2, 1), dtype=f32),
        m=jnp.asarray(2, dtype=jnp.int32),
    )
    first = jnp.linalg.solve(eq.VtV + _JITTER * jnp.eye(2, dtype=f32), eq.VtY)
    assert not np.isfinite(np.asarray(first)).all()  # the retry path is exercised
    assert np.isfinite(np.asarray(_solve_ridge(eq, 0.0))).all()
    assert np.isfinite(np.asarray(jax.jit(_solve_ridge)(eq, 0.0))).all()


def test_cross_validation_workers_follow_thread_local_x64(monkeypatch) -> None:
    """Pool workers run under the caller's ``with jax.enable_x64(...)`` value.

    The cross-validation runs most (gamma, fold) pairs on pool threads, which
    do not inherit thread-local JAX config. Without re-entering it, a float64
    analysis inside ``with jax.enable_x64():`` in a float32 process scored
    most of the grid in float32 and picked other hyperparameters.
    """
    import jax

    from jaxgsa.vkoga import _analyze, _engine

    local_x64 = not bool(getattr(jax.config, "jax_enable_x64", False))
    problem = jaxgsa.Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(3)})
    X = np.asarray(jaxgsa.sampling.monte_carlo(problem, 256, seed=0))
    seen: list[str] = []
    real = _engine._cv_fold_sse

    def _spy(*args, **kwargs):
        out = real(*args, **kwargs)
        seen.append(out[0].dtype.name)
        return out

    def _scores(workers: int):
        monkeypatch.setattr(_engine, "_cv_workers", lambda *a, **k: workers)
        with jax.enable_x64(local_x64):
            dtype = jnp.float64 if local_x64 else jnp.float32
            U = jnp.asarray(X, dtype=dtype)
            Y = (jnp.sin(3 * U[:, 0]) + U[:, 1] ** 2 * U[:, 2])[:, None]
            per_slice, pooled = _engine._cross_validate(
                U,
                Y - Y.mean(),
                gammas=_analyze._GAMMA_GRID[:4],
                ridges=_analyze._RIDGE_GRID[:3],
                max_centers=20,
                n_folds=4,
            )
            return np.asarray(per_slice), np.asarray(pooled)

    monkeypatch.setattr(_engine, "_cv_fold_sse", _spy)
    pooled_threads = _scores(4)
    expected = "float64" if local_x64 else "float32"
    assert set(seen) == {expected}
    pooled_serial = _scores(1)
    rtol = 1e-6 if local_x64 else 1e-4
    np.testing.assert_allclose(pooled_threads[0], pooled_serial[0], rtol=rtol)
    np.testing.assert_allclose(pooled_threads[1], pooled_serial[1], rtol=rtol)
    assert np.argmin(pooled_threads[1]) == np.argmin(pooled_serial[1])


def _ishigami_data(n: int, seed: int) -> tuple[jaxgsa.Problem, np.ndarray, np.ndarray]:
    problem = jaxgsa.Problem(["x1", "x2", "x3"], [(-np.pi, np.pi)] * 3)
    X = np.random.default_rng(seed).uniform(-np.pi, np.pi, size=(n, 3))
    Y = np.sin(X[:, 0]) + 7 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])
    return problem, X, Y


_FLOAT_FIELDS = (
    "S_TC",
    "S_TU",
    "S_U",
    "S_C",
    "S_IU",
    "variance",
    "rmse",
    "cv_rmse",
    "explained_variance",
    "S_TC_conf",
    "S_TU_conf",
    "S_U_conf",
    "S_C_conf",
    "S_IU_conf",
)


def test_float32_inputs_under_x64_stay_float32():
    """Float32 inputs give float32 fields, intervals and predictions under x64.

    The index estimators run on the host in float64, and the predictor used to
    evaluate the float32 fit on float64 rows, so the indices came back float64
    while ``rmse`` stayed float32.
    """
    import jax

    problem, X, Y = _ishigami_data(256, seed=0)
    with jax.enable_x64(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = jaxgsa.vkoga.analyze(
            problem,
            jnp.asarray(X, dtype=jnp.float32),
            jnp.asarray(Y, dtype=jnp.float32),
            gamma=3.0,
            max_centers=32,
            n_folds=4,
            n_outer=32,
            n_inner=16,
            n_variance=256,
            n_bootstrap=2,
            key=0,
            verbose=False,
        )
        assert isinstance(r, VKOGAResult)
        for name in _FLOAT_FIELDS:
            assert getattr(r, name).dtype == jnp.float32, name
        assert r.predict(jnp.asarray(X[:8], dtype=jnp.float32)).dtype == jnp.float32
        assert r.predict(jnp.asarray(X[:8])).dtype == jnp.float32


def test_float32_cross_validated_fit_matches_float64():
    """A float32 fit with cross-validated gamma and ridge tracks the float64 one.

    Ishigami, N=600, 60 centres. The coefficient solve used to form
    ``A^T A + ridge * A_mm``, whose condition number (6e8 here) is past
    float32's 1/eps: the float32 fit had explained variance -1.49 against
    0.90 in float64, and S_TU off by up to 0.4. Solved in the Newton basis,
    the two agree to about 2e-4 on every index (same key, so the integration
    draws are shared and only the surrogate differs). The 0.01 tolerance
    leaves room for a different grid point or greedy tie-break, and is still
    far below the old error.
    """
    import jax

    problem, X, Y = _ishigami_data(600, seed=0)
    results = []
    with jax.enable_x64(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dtype in (jnp.float32, jnp.float64):
            r = jaxgsa.vkoga.analyze(
                problem,
                jnp.asarray(X, dtype=dtype),
                jnp.asarray(Y, dtype=dtype),
                max_centers=60,
                n_outer=128,
                n_inner=64,
                n_variance=2048,
                key=0,
                verbose=False,
            )
            assert isinstance(r, VKOGAResult)
            assert r.explained_variance is not None
            results.append(r)
    r32, r64 = results
    ev32 = float(np.asarray(r32.explained_variance))
    ev64 = float(np.asarray(r64.explained_variance))
    assert ev64 > 0.85
    np.testing.assert_allclose(ev32, ev64, atol=0.01)
    for name in ("S_TC", "S_TU", "S_U", "S_C"):
        np.testing.assert_allclose(
            np.asarray(getattr(r32, name), dtype=np.float64),
            np.asarray(getattr(r64, name)),
            atol=0.01,
            err_msg=name,
        )
