"""Tests for the PAWN 1.2 fixes: centred bootstrap interval and exact KS counts."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

import jaxgsa.pawn as pawn
from jaxgsa.pawn._analyze import _count_cumsum, _ks_from_sorted
from jaxgsa.problem import Problem

PROBLEM = Problem(names=("a", "b", "c"), bounds=((0.0, 1.0),) * 3)


def _data(n: int = 1000):
    X = np.random.default_rng(0).random((n, 3)).astype(np.float32)
    y = (X[:, 0] + 0.3 * X[:, 1]).astype(np.float32)
    return jnp.asarray(X), jnp.asarray(y)


def test_quantile_interval_is_centred_on_the_estimate() -> None:
    X, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = pawn.analyze(
            PROBLEM,
            X,
            y,
            n_bootstrap=100,
            key=jax.random.key(0),
            verbose=False,
            keep_replicates=True,
        )
    assert isinstance(r, pawn.PAWNResult)
    assert r.ci is not None and r.ci.replicates is not None
    draws = np.asarray(r.ci.replicates["pawn"])
    point = np.asarray(r.pawn)
    conf = np.asarray(r.pawn_conf)
    mid_raw = np.quantile(draws, [0.025, 0.975], axis=0).mean(axis=0)
    # The stored draws are raw (their mean is above the point for a null parameter).
    assert draws.mean(0)[2] > point[2]
    # The interval is shifted down by the mean excess.
    shift = draws.mean(0) - point
    np.testing.assert_allclose(
        conf.mean(axis=0) if conf.shape[0] != 2 else conf.mean(0),
        mid_raw - shift,
        rtol=1e-4,
        atol=1e-5,
    )


def test_constant_output_ks_is_exactly_zero() -> None:
    n = 1000
    X = jnp.asarray(np.random.default_rng(0).random((n, 3)).astype(np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = pawn.indices(PROBLEM, X, jnp.ones(n, jnp.float32))
    assert float(jnp.max(jnp.abs(r.pawn))) == 0.0


def test_ks_matches_scipy_to_one_ulp() -> None:
    rng = np.random.default_rng(3)
    n = 503
    y = np.sort(rng.normal(size=n)).astype(np.float32)
    bins = rng.integers(0, 4, size=(n, 1)).astype(np.int32)
    got = np.asarray(_ks_from_sorted(jnp.asarray(y), jnp.asarray(bins), 4))[0]
    for c in range(4):
        ref = stats.ks_2samp(y[bins[:, 0] == c], y).statistic
        assert abs(got[c] - ref) <= 6e-8


def test_count_cumsum_is_int32_and_exact() -> None:
    oh = jnp.asarray((np.random.default_rng(0).random((1000, 3)) > 0.5).astype(np.float32))
    out = _count_cumsum(oh)
    assert out.dtype == jnp.int32
    np.testing.assert_array_equal(
        np.asarray(out), np.cumsum(np.asarray(oh), axis=0).astype(np.int32)
    )
