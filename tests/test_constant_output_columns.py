"""A constant output column gets what the zero-variance warning promises.

kucherenko, rank and hdmr divide by the output variance, so the promise is
NaN. A constant float column's variance rounds to a tiny positive value
(its mean rounds), so each method must use the warning's own test,
``_zero_variance_mask``, and not ``variance > 0``. The other columns must be
unaffected, also through a gradient.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning, Problem, hdmr, kucherenko, rank

PI = float(np.pi)
PROBLEM = Problem.from_dict({"x1": (-PI, PI), "x2": (-PI, PI), "x3": (-PI, PI)})
# Constants whose float variance rounds to a nonzero value. Before the fix
# c=0.1 gave kucherenko S1 = ST = 0 in float64 and rank S1 = 1.0.
CONSTANTS = (0.053902783001748715, 0.1, 1e6, -3.7)
X64 = pytest.mark.parametrize("x64", [False, True], ids=["f32", "f64"])


def _varying(X: np.ndarray) -> np.ndarray:
    return np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])


def _two_columns(X: np.ndarray, c: float, dtype) -> jax.Array:
    y = _varying(np.asarray(X, np.float64))
    return jnp.asarray(np.stack([y, np.full(len(y), c)], axis=1), dtype)


def _given(n: int, dtype) -> np.ndarray:
    return np.asarray(np.random.default_rng(0).uniform(-PI, PI, (n, 3)), dtype)


def _all_nan(*arrays) -> bool:
    return all(bool(jnp.all(jnp.isnan(jnp.asarray(a)))) for a in arrays)


@X64
@pytest.mark.parametrize("subsets", ["singletons", "all"])
@pytest.mark.parametrize("c", CONSTANTS)
def test_kucherenko_constant_column_is_nan(c, subsets, x64):
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        s = kucherenko.sample(PROBLEM, base_n=128, subsets=subsets, seed=0, verbose=False)
        Y = _two_columns(np.asarray(s.samples), c, dtype)
        with pytest.warns(JaxgsaWarning, match="zero variance"):
            res = kucherenko.analyze(s, Y, n_bootstrap=10, key=0, verbose=False)
        assert isinstance(res, kucherenko.KucherenkoResult)
        assert res.S1_conf is not None and res.ST_conf is not None
        assert _all_nan(res.S1[1], res.ST[1], res.S1_conf[:, 1], res.ST_conf[:, 1])
        assert bool(jnp.all(jnp.isfinite(res.S1[0]))) and bool(jnp.all(jnp.isfinite(res.ST[0])))


@X64
@pytest.mark.parametrize("c", CONSTANTS)
def test_rank_constant_column_is_nan(c, x64):
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        X = jnp.asarray(_given(1000, np.float64), dtype)
        Y = _two_columns(np.asarray(X), c, dtype)
        with pytest.warns(JaxgsaWarning, match="zero variance"):
            res = rank.analyze(PROBLEM, X, Y, n_bootstrap=10, key=0, verbose=False)
        assert isinstance(res, rank.RankResult)
        ix = rank.indices(PROBLEM, X, Y)
        assert res.S1_conf is not None
        assert _all_nan(res.S1[1], res.S1_conf[:, 1], ix.S1[1])
        alone = rank.indices(PROBLEM, X, Y[:, 0])
        rtol = 1e-10 if x64 else 1e-5
        np.testing.assert_allclose(np.asarray(ix.S1[0]), np.asarray(alone.S1), rtol=rtol)


@X64
def test_rank_constant_column_keeps_other_gradients_finite(x64):
    """jit(grad) of the varying column's index with a constant column beside it."""
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        X = jnp.asarray(_given(500, np.float64), dtype)
        Y = _two_columns(np.asarray(X), 0.1, dtype)

        g = jax.jit(jax.grad(lambda y: rank.indices(PROBLEM, X, y).S1[0].sum()))(Y)
        g_ref = jax.jit(jax.grad(lambda y: rank.indices(PROBLEM, X, y).S1.sum()))(Y[:, 0])
        assert bool(jnp.all(jnp.isfinite(g)))
        np.testing.assert_allclose(np.asarray(g[:, 1]), 0.0, atol=1e-12)
        rtol = 1e-10 if x64 else 1e-4
        np.testing.assert_allclose(np.asarray(g[:, 0]), np.asarray(g_ref), rtol=rtol, atol=rtol)


@X64
@pytest.mark.parametrize("c", CONSTANTS)
def test_hdmr_constant_column_is_nan(c, x64):
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        X = jnp.asarray(_given(600, np.float64), dtype)
        Y = _two_columns(np.asarray(X), c, dtype)
        with pytest.warns(JaxgsaWarning, match="zero variance"):
            res = hdmr.analyze(PROBLEM, X, Y, order=1, n_bootstrap=3, key=0, verbose=False)
        assert isinstance(res, hdmr.HDMRResult)
        ix = hdmr.indices(PROBLEM, X, Y, order=1)
        assert _all_nan(res.S[1], res.ST[1], res.Sa[1], ix.S[1], ix.ST[1])
        assert res.S_conf is not None and _all_nan(res.S_conf[:, 1])
        assert bool(jnp.all(jnp.isfinite(res.ST[0])))


@X64
def test_hdmr_constant_column_keeps_other_tangents_finite(x64):
    """jvp through hdmr.indices: the varying column's tangent is unchanged."""
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        X = jnp.asarray(_given(400, np.float64), dtype)
        Y = _two_columns(np.asarray(X), 0.1, dtype)
        dY = jnp.asarray(np.random.default_rng(1).standard_normal(Y.shape), dtype)

        def st(y):
            return hdmr.indices(PROBLEM, X, y, order=1).ST

        _, t = jax.jvp(st, (Y,), (dY,))
        _, t_ref = jax.jvp(st, (Y[:, 0],), (dY[:, 0],))
        assert bool(jnp.all(jnp.isfinite(t[0])))
        rtol = 1e-8 if x64 else 1e-3
        np.testing.assert_allclose(np.asarray(t[0]), np.asarray(t_ref), rtol=rtol, atol=rtol)


@X64
def test_hdmr_near_constant_column_is_analysed_not_nan(x64):
    """A column that varies is not treated as constant, and nothing warns.

    The old HDMR test, var < eps * max|Y|^2, called a column with relative
    spread below sqrt(eps) constant and returned NaN without a warning. HDMR
    now uses the warning's own test, so such a column gets finite indices.
    """
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        X = jnp.asarray(_given(600, np.float64), dtype)
        rel = 1e-10 if x64 else 1e-5  # below sqrt(eps): 1.5e-8 / 3.4e-4
        y = 0.1 * (1.0 + rel * np.sin(np.asarray(X[:, 0], np.float64)))
        Y = jnp.asarray(np.stack([_varying(np.asarray(X, np.float64)), y], axis=1), dtype)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*zero variance.*")
            res = hdmr.analyze(PROBLEM, X, Y, order=1, verbose=False)
        assert isinstance(res, hdmr.HDMRResult)
        assert bool(jnp.all(jnp.isfinite(res.S[1])))
        # Y depends on x1 only, so x1 carries the variance.
        assert float(res.S[1][0]) > 0.5
