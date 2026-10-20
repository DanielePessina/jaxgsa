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


# N = 1024 at D = 23 has 253 pairs of about 25 free coefficients for 1024 rows.
# Measured: order 2 still beats order 1 there (0.17 against 0.19), by a thin
# margin, so the cell stays in with a looser ratio than the others.
@pytest.mark.parametrize("N", [1024, 4096])
@pytest.mark.parametrize("D", [5, 9, 15, 23])
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
        assert [str(w.message) for w in caught if "jaxgsa.hdmr" in str(w.message)] == []
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
