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
        return jnp.full_like(eq.AtY, jnp.nan)

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
    """A Gram with a repeated centre is singular in float32; the retry fixes it."""
    import jax

    from jaxgsa.vkoga._engine import _normal_equations, _solve_ridge

    rng = np.random.default_rng(0)
    U = jnp.asarray(rng.random((40, 2)), dtype=jnp.float32)
    Y = jnp.asarray(rng.random((40, 1)), dtype=jnp.float32)
    idx = jnp.asarray([0, 1, 2, 0, 3], dtype=jnp.int32)  # centre 0 twice
    eq = _normal_equations(U, Y, idx, jnp.asarray(5, dtype=jnp.int32), gamma=1.0)
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
