"""Regression tests for float32 VKOGA fits on data with duplicate rows."""

from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.monitoring
import jax.numpy as jnp
import numpy as np
import pytest

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


def _small_kwargs() -> dict[str, Any]:
    return dict(max_centers=40, n_outer=32, n_inner=16, n_variance=512, key=0, verbose=False)


@pytest.mark.parametrize(("x_dtype", "y_dtype"), [("float64", "float32"), ("float32", "float64")])
def test_mixed_input_dtypes_fit_in_the_wider_dtype(x_dtype, y_dtype):
    """Mixed float32/float64 ``X`` and ``Y`` fit in float64, fields and predict alike.

    Float64 ``X`` with float32 ``Y`` used to crash in the greedy loop carry,
    and float32 ``X`` with float64 ``Y`` returned float32 fields but a float64
    prediction.
    """
    import jax

    problem, X, Y = _ishigami_data(256, seed=0)
    with jax.enable_x64(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = jaxgsa.vkoga.analyze(
            problem, jnp.asarray(X, x_dtype), jnp.asarray(Y, y_dtype), **_small_kwargs()
        )
        assert isinstance(r, VKOGAResult)
        for name in ("S_TC", "S_TU", "variance", "rmse", "cv_rmse", "explained_variance"):
            assert getattr(r, name).dtype == jnp.float64, name
        assert r.predict(jnp.asarray(X[:4], x_dtype)).dtype == jnp.float64


@pytest.mark.parametrize("policy", ["propagate", "skip"])
def test_nan_slice_gives_nan_indices_for_that_slice_only(policy):
    """A NaN in one output slice makes that slice NaN; the other slice is fitted.

    Before, the NaN reached the shared cross-validation and every score was
    non-finite, so analyze raised a RuntimeError that blamed float32.
    """
    problem, X, y = _ishigami_data(256, seed=0)
    Y = np.stack([y, X[:, 0] + X[:, 1] ** 2], axis=-1)
    Y[5, 1] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = jaxgsa.vkoga.analyze(
            problem, jnp.asarray(X), jnp.asarray(Y), on_invalid=policy, **_small_kwargs()
        )
    assert isinstance(r, VKOGAResult)
    # "skip" promises silence; "propagate" says what it let through.
    warned = any("non-finite values" in str(w.message) for w in caught)
    assert warned == (policy == "propagate")
    for name in ("S_TC", "S_TU", "S_U", "S_C", "S_IU"):
        values = np.asarray(getattr(r, name))
        assert np.isfinite(values[0]).all(), name
        assert np.isnan(values[1]).all(), name
    assert r.cv_rmse is not None and r.explained_variance is not None
    assert np.isfinite(np.asarray(r.cv_rmse)[0]) and np.isnan(np.asarray(r.cv_rmse)[1])
    assert np.isnan(np.asarray(r.explained_variance)[1])
    pred = np.asarray(r.predict(jnp.asarray(X[:4])))
    assert np.isfinite(pred[:, 0]).all() and np.isnan(pred[:, 1]).all()


def test_all_slices_nan_gives_an_all_nan_result():
    """With nothing finite to fit, propagate returns NaN fields and no surrogate."""
    problem, X, y = _ishigami_data(128, seed=0)
    y = y.copy()
    y[3] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = jaxgsa.vkoga.analyze(
            problem, jnp.asarray(X), jnp.asarray(y), on_invalid="propagate", **_small_kwargs()
        )
    assert isinstance(r, VKOGAResult)
    assert np.isnan(np.asarray(r.S_TC)).all() and r.n_centers == 0
    with pytest.raises(ValueError, match="fitted surrogate"):
        r.predict(jnp.asarray(X[:2]))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("constant", [0.0, 0.0539, 0.1, -3.7, 1e6])
def test_constant_output_slice_has_nan_indices(dtype, constant):
    """A constant slice gives NaN indices, as the zero-variance warning promises.

    In float64 the surrogate variance of a constant slice is round-off (about
    1e-30), not zero, and the indices used to come out small and non-zero
    (about 1e-3) for constants such as 0.0539, 0.1 and -3.7.
    """
    import jax

    n = 500
    problem = jaxgsa.Problem(["x1", "x2", "x3"], [(-np.pi, np.pi)] * 3)
    X = np.random.default_rng(0).uniform(-np.pi, np.pi, size=(n, 3))
    Y = np.stack([np.sin(X[:, 0]) + X[:, 1] ** 2, np.full(n, constant)], axis=-1)
    with jax.enable_x64(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = jaxgsa.vkoga.analyze(
            problem, jnp.asarray(X, dtype), jnp.asarray(Y, dtype), **_small_kwargs()
        )
    assert isinstance(r, VKOGAResult)
    assert any("zero variance" in str(w.message) for w in caught)
    for name in ("S_TC", "S_TU", "S_U", "S_C", "S_IU"):
        values = np.asarray(getattr(r, name))
        assert np.isfinite(values[0]).all(), name
        assert np.isnan(values[1]).all(), name
    assert float(np.asarray(r.variance)[1]) == 0.0


def test_cheap_s_tc_matches_averaged_predictions(monkeypatch):
    """S_TC from the mean kernel row equals S_TC from averaged predictions.

    The two orders agree in exact arithmetic, so float64 results match to
    rounding; the other indices do not use the block means at all.
    """
    import jax

    from jaxgsa.vkoga import _analyze

    problem, X, Y = _ishigami_data(256, seed=0)
    kwargs: dict[str, Any] = dict(gamma=3.0, ridge=1e-6, **_small_kwargs())
    with jax.enable_x64(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cheap = jaxgsa.vkoga.analyze(problem, jnp.asarray(X), jnp.asarray(Y), **kwargs)
        monkeypatch.setattr(_analyze, "_make_group_mean", lambda *a, **k: None)
        plain = jaxgsa.vkoga.analyze(problem, jnp.asarray(X), jnp.asarray(Y), **kwargs)
    assert isinstance(cheap, VKOGAResult) and isinstance(plain, VKOGAResult)
    np.testing.assert_allclose(
        np.asarray(cheap.S_TC), np.asarray(plain.S_TC), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(cheap.S_TU), np.asarray(plain.S_TU), rtol=1e-8, atol=1e-10
    )


def test_greedy_update_incremental_runs_and_validates():
    """``greedy_update="incremental"`` fits a usable surrogate; a bad value raises."""
    import jax

    problem, X, Y = _ishigami_data(600, seed=0)
    kwargs: dict[str, Any] = dict(gamma=3.0, ridge=1e-6, **{**_small_kwargs(), "max_centers": 100})
    with jax.enable_x64(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = jaxgsa.vkoga.analyze(problem, jnp.asarray(X), jnp.asarray(Y), **kwargs)
        incr = jaxgsa.vkoga.analyze(
            problem, jnp.asarray(X), jnp.asarray(Y), greedy_update="incremental", **kwargs
        )
        with pytest.raises(ValueError, match="greedy_update"):
            jaxgsa.vkoga.analyze(
                problem,
                jnp.asarray(X),
                jnp.asarray(Y),
                greedy_update="fast",  # ty: ignore[invalid-argument-type]
                **kwargs,
            )
    assert isinstance(exact, VKOGAResult) and isinstance(incr, VKOGAResult)
    # The same in-sample fit quality as the exact update (0.937 at these settings).
    np.testing.assert_allclose(
        float(np.asarray(incr.explained_variance)),
        float(np.asarray(exact.explained_variance)),
        atol=0.01,
    )
    # Different centres, same model: the indices agree to about their
    # Monte-Carlo error at this small integration budget.
    np.testing.assert_allclose(np.asarray(incr.S_TU), np.asarray(exact.S_TU), atol=0.05)


def test_nonfinite_input_row_names_the_inputs():
    """A NaN in X makes every slice NaN, and the warning blames the inputs."""
    problem, X, y = _ishigami_data(128, seed=0)
    X = X.copy()
    X[3, 0] = np.nan
    Y = np.stack([y, 2 * y], axis=-1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = jaxgsa.vkoga.analyze(
            problem, jnp.asarray(X), jnp.asarray(Y), on_invalid="propagate", **_small_kwargs()
        )
    assert isinstance(r, VKOGAResult)
    assert np.isnan(np.asarray(r.S_TC)).all()
    messages = [str(w.message) for w in caught]
    assert any("inputs X hold non-finite values" in m for m in messages)
    assert not any("output slice(s) hold non-finite" in m for m in messages)


def test_constant_slice_does_not_trigger_the_replicate_warning():
    """A constant slice is NaN by design, not a singular replicate fit."""
    problem, X, y = _ishigami_data(256, seed=0)
    Y = np.stack([y, np.full_like(y, 0.1)], axis=-1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = jaxgsa.vkoga.analyze(
            problem, jnp.asarray(X), jnp.asarray(Y), n_bootstrap=3, **_small_kwargs()
        )
    assert isinstance(r, VKOGAResult) and r.S_TC_conf is not None
    assert not any("bootstrap replicates" in str(w.message) for w in caught)
    conf = np.asarray(r.S_TC_conf)
    assert np.isfinite(conf[:, 0]).all() and np.isnan(conf[:, 1]).all()


def test_float32_cheap_s_tc_matches_a_float64_evaluation(monkeypatch):
    """At a small gamma, float32 S_TC from block means keeps float64 accuracy.

    One float32 surrogate is evaluated three ways: in float64 (the
    reference), by averaging float32 predictions in float64 (plain), and by
    the block-mean path. At gamma 0.1 the coefficients reach about 1e6, and
    a float32 block-mean product shifted S_TC by about 4e-2 against the
    reference, against about 6e-3 for the plain path. The block mean and
    product now run in float64, which matched the plain path (2e-2 against
    2.1e-2 at the larger reviewer budget). The bound allows the cheap path at
    most 1.5 times the plain path's error.
    """
    import jax

    from jaxgsa.vkoga import _analyze

    problem, X, Y = _ishigami_data(1024, seed=0)
    kwargs: dict[str, Any] = dict(
        gamma=0.1,
        ridge=1e-6,
        max_centers=300,
        n_outer=64,
        n_inner=128,
        n_variance=2048,
        key=0,
        verbose=False,
    )
    orig_fit, orig_group = _analyze._fit_vkoga, _analyze._make_group_mean

    def fit64(*args: Any, **kw: Any) -> Any:
        state = orig_fit(*args, **kw)
        return jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            state,
        )

    out = {}
    with jax.enable_x64(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mode in ("ref", "plain", "cheap"):
            monkeypatch.setattr(_analyze, "_fit_vkoga", fit64 if mode == "ref" else orig_fit)
            monkeypatch.setattr(
                _analyze,
                "_make_group_mean",
                (lambda *a, **k: None) if mode == "plain" else orig_group,
            )
            r = jaxgsa.vkoga.analyze(
                problem,
                jnp.asarray(X, jnp.float32),
                jnp.asarray(Y, jnp.float32),
                **kwargs,
            )
            assert isinstance(r, VKOGAResult)
            out[mode] = np.asarray(r.S_TC, dtype=np.float64)
    plain = np.abs(out["plain"] - out["ref"]).mean()
    cheap = np.abs(out["cheap"] - out["ref"]).mean()
    assert cheap <= 1.5 * plain + 1e-4, (cheap, plain)


_VKOGA_COMPILES: list[str] = []


def _count_compile(event: str, duration_secs: float, **_: Any) -> None:
    if event == "/jax/core/compile/backend_compile_duration":
        _VKOGA_COMPILES.append(event)


jax.monitoring.register_event_duration_secs_listener(_count_compile)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_new_data_and_bootstrap_replicates_compile_nothing(dtype):
    """Warm runs on new data, with or without bootstrap replicates, compile nothing.

    The evaluators used to be fresh ``jax.jit`` closures over each fit's
    arrays, and the fitted state was sliced to the exact centre count, so
    every new dataset and every replicate compiled its own programs.
    """
    problem = jaxgsa.Problem(["x1", "x2", "x3"], [(-np.pi, np.pi)] * 3)

    def run(seed: int, n_bootstrap: int) -> None:
        X = np.random.default_rng(seed).uniform(-np.pi, np.pi, (512, 3))
        Y = np.sin(X[:, 0]) + 7 * np.sin(X[:, 1]) ** 2
        r = jaxgsa.vkoga.analyze(
            problem,
            jnp.asarray(X, dtype),
            jnp.asarray(Y, dtype),
            gamma=3.0,
            ridge=1e-6,
            max_centers=64,
            n_outer=32,
            n_inner=16,
            n_variance=512,
            n_bootstrap=n_bootstrap,
            key=seed,
            verbose=False,
        )
        jax.block_until_ready(jax.tree.leaves(r))

    with jax.enable_x64(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run(0, 4)
        _VKOGA_COMPILES.clear()
        run(1, 0)
        run(2, 4)
    assert _VKOGA_COMPILES == [], f"{len(_VKOGA_COMPILES)} programs compiled"
