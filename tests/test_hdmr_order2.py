"""Regression tests for the HDMR second-order fit (backlog #21, #22).

Order 2 used to fit every pair once against the same first-order residual, so
the D-1 pairs that contain one input each counted the same leftover. The sum of
S grew with D and the test error was worse than order 1. The pair basis is now
projected off the first-order space and all pairs are solved jointly.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import hdmr
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.benchmarks import ishigami
from jaxgsa.hdmr import HDMRResult
from jaxgsa.problem import Problem

D_MAX = 23


def _f(X: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * X[:, 0]) + X[:, 1] ** 2 + 0.5 * X[:, 2] + 2 * X[:, 3] * X[:, 4]


@pytest.fixture(scope="module")
def x64():
    """Run the module in float64, the precision the backlog numbers use."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _fit(*args, **kwargs) -> HDMRResult:
    res = hdmr.analyze(*args, verbose=False, **kwargs)
    assert isinstance(res, HDMRResult)
    return res


def _nrmse(pred, y) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - y) ** 2) / np.var(y)))


# N = 1024 at D = 23 has 253 pairs of 16 free coefficients each for 1024 rows.
# Measured: order 2 still beats order 1 there (0.17 against 0.19), by a thin
# margin, so the cell stays in with a looser ratio than the others.
# D = 23 at N = 4096 takes about 20 s per fit (CG over 6325 pair coefficients)
# and was the slowest test in the suite, so it runs with the slow tier: on the
# schedule and before every release.
@pytest.mark.parametrize(
    ("D", "N"),
    [
        *[(D, N) for D in (5, 9, 15) for N in (1024, 4096)],
        (23, 1024),
        pytest.param(23, 4096, marks=pytest.mark.slow),
    ],
)
def test_order2_beats_order1_and_sum_of_s_stays_near_one(x64, D, N):
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(2 * N, D_MAX))[:, :D]
    Y = _f(X)
    problem = Problem(names=[f"x{i}" for i in range(D)], bounds=[(0.0, 1.0)] * D)
    Xj, Yj = jnp.asarray(X), jnp.asarray(Y)
    o1 = _fit(problem, Xj[:N], Yj[:N], order=1, n_intervals=3)
    # Where the pair system is well posed the default call must be silent.
    # Hard cells (D = 15 at N = 1024, D = 23) may warn: the joint solve stops
    # at its iteration budget, or there are more pair coefficients than rows.
    clean = D <= 9 or (D, N) == (15, 4096)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        o2 = _fit(problem, Xj[:N], Yj[:N], order=2, n_intervals=3)
    if clean:
        # The coefficient-count warning is about the sample size, not the
        # solve: at D = 9, n_intervals = 3 there are 945 free coefficients
        # for 1024 rows, so it fires there. Nothing else may.
        msgs = [str(w.message) for w in caught if "jaxgsa.hdmr" in str(w.message)]
        assert [m for m in msgs if "coefficients for" not in m] == []
    if D == 23 and N == 1024:
        assert any("coefficients for" in str(w.message) for w in caught)
    test1 = _nrmse(o1.predict(Xj[N:]), Y[N:])
    test2 = _nrmse(o2.predict(Xj[N:]), Y[N:])
    train2 = _nrmse(o2.predict(Xj[:N]), Y[:N])
    ratio = 0.95 if (D, N) == (23, 1024) else 0.8
    assert test2 < ratio * test1
    # The in-sample error of a least-squares fit cannot exceed the test error
    # by more than noise. Before the fix it was 1.28 against 0.48 at D = 23.
    assert train2 < test2 + 0.02
    # The model is exactly additive plus one pair, so sum S is about the
    # explained variance. It was 1.14 at D = 9 and 3.07 at D = 23 before.
    assert abs(float(np.sum(np.asarray(o2.S))) - 1.0) < 0.05
    # The only true pair x3 x4 has analytic S2 about 0.034. Extra pairs absorb
    # it when N is too small (0.005 at D = 23, N = 1024), so check the rest.
    if (D, N) not in {(15, 1024), (23, 1024)}:
        assert float(np.asarray(o2.S2)[3, 4]) == pytest.approx(0.034, abs=0.01)


def test_ishigami_order2_matches_analytic_indices(x64):
    rng = np.random.default_rng(0)
    N = 4096
    X = rng.uniform(-np.pi, np.pi, size=(N, 3))
    Y = np.asarray(ishigami.evaluate(jnp.asarray(X)))
    res = _fit(ishigami.PROBLEM, jnp.asarray(X), jnp.asarray(Y), order=2, n_intervals=4)
    S1, ST = np.asarray(res.S1), np.asarray(res.ST)
    np.testing.assert_allclose(S1, ishigami.ANALYTICAL_S1, atol=0.05)
    np.testing.assert_allclose(ST, ishigami.ANALYTICAL_ST, atol=0.06)
    # The only interaction is x1 x3 (analytic S13 = ST1 - S1 = 0.244).
    assert float(np.asarray(res.S2)[0, 2]) == pytest.approx(0.244, abs=0.06)


def test_ridge_has_a_measurable_effect_at_any_n(x64):
    problem = Problem(names=["a", "b", "c"], bounds=[(0.0, 1.0)] * 3)
    for N in (512, 4096):
        rng = np.random.default_rng(3)
        X = rng.uniform(size=(N, 3))
        Y = np.sin(2 * np.pi * X[:, 0]) + X[:, 1] + 0.3 * rng.normal(size=N)
        Xj, Yj = jnp.asarray(X), jnp.asarray(Y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits = {
                r: _fit(problem, Xj, Yj, order=1, n_intervals=3, ridge=r) for r in (0.0, 1e-2, 1.0)
            }
        err = {r: _nrmse(f.predict(Xj), Y) for r, f in fits.items()}
        # Same strength means the same shrinkage at both N: ridge is relative
        # to the Gram diagonal, which grows with N.
        assert err[1.0] > err[1e-2] + 0.05
        assert err[1e-2] > err[0.0] + 1e-4


@pytest.mark.parametrize(("batch_size", "slice_chunk_size"), [(None, None), (300, 2)])
def test_dual_pair_solve_matches_dense_solve(x64, monkeypatch, batch_size, slice_chunk_size):
    """The sample-space pair solve gives the exact ridge solution.

    At D = 6 and 9 intervals there are 15 * 144 = 2160 pair coefficients for
    1024 rows, the case the dual solve is for. Measured agreement with the
    dense solve is 4e-11 in S.
    """
    from jaxgsa.hdmr import _fit

    D, N = 6, 1024
    X = np.random.default_rng(0).random((N, D))
    Y = np.stack(
        [
            np.sin(2 * np.pi * X[:, 0])
            + X[:, 1] ** 2
            + 2 * X[:, 2] * X[:, 3]
            + k * X[:, 4] * X[:, 5]
            for k in range(3)
        ],
        axis=1,
    )
    problem = Problem(names=[f"x{i}" for i in range(D)], bounds=[(0.0, 1.0)] * D)
    assert _fit._pair_solver(N, 3, 15, 144, 8) == "dual"
    S = {}
    for kind in ("dense", "dual"):
        monkeypatch.setattr(_fit, "_pair_solver", lambda *_, k=kind: k)
        S[kind] = np.asarray(
            _fit_quiet(problem, X, Y, batch_size, slice_chunk_size).S, dtype=float
        )
    np.testing.assert_allclose(S["dual"], S["dense"], rtol=1e-7, atol=1e-9)


def _fit_quiet(problem, X, Y, batch_size, slice_chunk_size) -> HDMRResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _fit(
            problem,
            X,
            Y,
            order=2,
            n_intervals=9,
            batch_size=batch_size,
            slice_chunk_size=slice_chunk_size,
        )


def _two_solvers(monkeypatch, problem, X, Y, **kwargs) -> dict[str, np.ndarray]:
    from jaxgsa.hdmr import _fit

    S = {}
    for kind in ("dense", "dual"):
        monkeypatch.setattr(_fit, "_pair_solver", lambda *_, k=kind: k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S[kind] = np.asarray(_fit_order2(problem, X, Y, **kwargs).S, dtype=float)
    return S


def _fit_order2(problem, X, Y, **kwargs) -> HDMRResult:
    return _fit(problem, X, Y, order=2, **kwargs)


def _small_case(D: int, N: int = 457, spread: float | None = None):
    rng = np.random.default_rng(1)
    X = rng.random((N, D))
    if spread is not None:
        X[:, -1] = 0.5 + spread * rng.random(N)
    Y = np.stack(
        [np.sin(2 * np.pi * X[:, 0]) + 2 * X[:, 0] * X[:, 1] + k * X[:, 1] for k in range(3)],
        axis=1,
    )
    return Problem(names=[f"x{i}" for i in range(D)], bounds=[(0.0, 1.0)] * D), X, Y


def test_dual_pair_solve_handles_tile_padding_ridge_zero_and_one_pair(x64, monkeypatch):
    """Several kernel tiles with a ragged last one, no ridge, and D = 2."""
    from jaxgsa import config

    problem, X, Y = _small_case(4)
    old = config.get_memory_budget(unit="b")
    config.set_memory_budget(0.25)  # MiB: forces kernel tiles of a few rows
    try:
        S = _two_solvers(monkeypatch, problem, X, Y, n_intervals=6)
    finally:
        config.set_memory_budget(old, unit="b")
    np.testing.assert_allclose(S["dual"], S["dense"], rtol=1e-7, atol=1e-9)

    S = _two_solvers(monkeypatch, problem, X, Y, n_intervals=6, ridge=0.0)
    np.testing.assert_allclose(S["dual"], S["dense"], rtol=1e-6, atol=1e-8)

    problem, X, Y = _small_case(2, N=200)
    S = _two_solvers(monkeypatch, problem, X, Y, n_intervals=9)
    np.testing.assert_allclose(S["dual"], S["dense"], rtol=1e-7, atol=1e-9)


def test_dual_pair_solve_survives_a_near_constant_input(x64):
    """An input with spread 1e-5 used to turn every lane's indices NaN."""
    problem, X, Y = _small_case(4, spread=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = np.asarray(_fit_order2(problem, X, Y, n_intervals=6).S, dtype=float)
    assert np.all(np.isfinite(S))
    np.testing.assert_allclose(S.sum(axis=-1), 1.0, atol=0.15)


def test_float32_never_takes_a_direct_pair_solve():
    from jaxgsa.hdmr import _fit

    assert _fit._pair_solver(500, 4, 10, 1000, 4) == "cg"


def test_all_pair_solvers_drop_a_dead_pair_alike(x64, monkeypatch):
    """Dense, dual and CG agree and stay finite when one input has almost no spread.

    This is a consistency check across the three solvers, not a regression
    test: the old dense and CG paths also stayed finite on every input tried
    (spreads 1e-4 to 1e-12 and an exactly constant column, which is refused).
    The dead-pair guard makes the three treat such a pair alike.
    """
    from jaxgsa.hdmr import _fit as _fit_module

    problem, X, Y = _small_case(4, spread=1e-9)
    S = {}
    for kind in ("dense", "dual", "cg"):
        monkeypatch.setattr(_fit_module, "_pair_solver", lambda *_, k=kind: k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S[kind] = np.asarray(_fit_order2(problem, X, Y, n_intervals=6).S, dtype=float)
        assert np.all(np.isfinite(S[kind])), kind
    np.testing.assert_allclose(S["dense"], S["dual"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(S["cg"], S["dual"], rtol=1e-3, atol=1e-3)


def test_order2_indices_stay_silent():
    """Only ``analyze`` warns about a large order-2 fit, not ``indices``."""
    D, N = 15, 1024
    X_np = np.random.default_rng(0).random((N, D))
    X, Y = jnp.asarray(X_np), jnp.asarray(_f(X_np))
    problem = Problem(names=[f"x{i}" for i in range(D)], bounds=[(0.0, 1.0)] * D)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hdmr.indices(problem, X, Y, order=2)
    with pytest.warns(JaxgsaWarning, match="coefficients for"):
        hdmr.analyze(problem, X, Y, order=2)


def test_unconverged_pair_cg_warns_in_analyze_only(x64, monkeypatch):
    """A joint pair CG cut at ten steps warns in ``analyze`` and not in ``indices``."""
    from jaxgsa.hdmr import _fit

    monkeypatch.setattr(_fit, "_pair_solver", lambda *_: "cg")
    # Measured: the relative residual after ten steps is 5.6e-3 here.
    monkeypatch.setattr(_fit, "_PAIR_WARN_REL", 1e-4)
    problem, X_np, Y_np = _small_case(5, N=700)
    X, Y = jnp.asarray(X_np), jnp.asarray(Y_np)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hdmr.indices(problem, X, Y, order=2, n_intervals=9, max_iter=1, ridge=1e-12)
    with pytest.warns(JaxgsaWarning) as caught:
        hdmr.analyze(problem, X, Y, order=2, n_intervals=9, max_iter=1, ridge=1e-12, verbose=False)
    assert len([w for w in caught if "relative residual" in str(w.message)]) == 1
