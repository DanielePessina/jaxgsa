"""Gradients through OT indices with constant outputs, and the joint-mode warnings."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import morris
from jaxgsa._core.validation import _standardize_outputs
from jaxgsa.benchmarks import ishigami
from jaxgsa.optimal_transport import analyze, indices

PROBLEM = jaxgsa.Problem(("a", "b"), ((0.0, 1.0), (0.0, 1.0)))
N, T = 64, 3
TS = jnp.linspace(0.0, 1.0, T)


@pytest.fixture(scope="module")
def X():
    return jnp.asarray(np.random.default_rng(0).uniform(size=(N, 2)), dtype=jnp.float32)


def _traj(X, theta):
    """(N, T, 1) output whose t=0 column is exactly constant."""
    return (X[:, :1] * TS * theta + X[:, 1:2] * TS**2)[..., None]


@pytest.mark.parametrize("mode", ["trajectory", "multivariate"])
@pytest.mark.parametrize("standardize", [True, False])
def test_grad_finite_with_constant_column(X, mode, standardize):
    def f(th):
        Y = _traj(X, th)
        if mode == "multivariate":
            Y = Y[..., 0]
        return indices(
            PROBLEM, X, Y, mode=mode, n_classes=4, standardize_outputs=standardize
        ).ot.sum()

    g = jax.grad(f)(1.3)
    assert np.isfinite(float(g))
    # forward and reverse mode agree
    gf = jax.jacfwd(f)(1.3)
    np.testing.assert_allclose(float(g), float(gf), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("mode", ["trajectory", "multivariate"])
@pytest.mark.parametrize("standardize", [True, False])
def test_grad_wrt_y_finite_with_constant_column(X, mode, standardize):
    Y = _traj(X, 1.3)
    if mode == "multivariate":
        Y = Y[..., 0]

    def f(y):
        return indices(
            PROBLEM, X, y, mode=mode, n_classes=4, standardize_outputs=standardize
        ).ot.sum()

    assert bool(jnp.all(jnp.isfinite(jax.grad(f)(Y))))


@pytest.mark.parametrize("standardize", [True, False])
def test_all_constant_y_has_finite_grad(X, standardize):
    def f(c):
        Y = jnp.broadcast_to(c, (N, 3))
        return indices(
            PROBLEM, X, Y, mode="multivariate", n_classes=4, standardize_outputs=standardize
        ).ot.sum()

    assert np.isfinite(float(jax.grad(f)(0.7)))


def test_standardize_outputs_value_and_grad():
    rng = np.random.default_rng(1)
    Y = jnp.asarray(rng.normal(size=(50, 3)), dtype=jnp.float32)
    _, _, y_std, _ = _standardize_outputs(Y)
    assert np.array_equal(np.asarray(y_std), np.asarray(jnp.std(Y, axis=0)))
    Yc = Y.at[:, 1].set(2.0)
    _, _, y_std_c, scale = _standardize_outputs(Yc)
    assert float(y_std_c[1]) == 0.0 or float(y_std_c[1]) < 1e-6
    g = jax.grad(lambda y: _standardize_outputs(y)[0].sum() + _standardize_outputs(y)[2].sum())(Yc)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(scale[1]) == 1.0


def test_morris_grad_finite_with_constant_column():
    sr = morris.sample(ishigami.PROBLEM, n_trajectories=8, seed=1, verbose=False)
    Y = ishigami.evaluate(jnp.asarray(sr.samples))

    def f(th):
        Ym = jnp.stack([Y * th, jnp.zeros_like(Y)], axis=-1)
        r = morris.indices(sr, Ym, standardize_outputs=True)
        return r.mu_star.sum()

    assert np.isfinite(float(jax.grad(f)(1.5)))


def _zero_col_data():
    rng = np.random.default_rng(1)
    X = jnp.asarray(rng.uniform(-np.pi, np.pi, (100, 2)), dtype=jnp.float32)
    y = jnp.sin(X[:, 0]) + 2 * jnp.sin(X[:, 1]) ** 2
    return X, jnp.stack([jnp.zeros(100), y], 1)


def _warnings(mode, Y, X):
    problem = jaxgsa.Problem(("a", "b"), ((-np.pi, np.pi),) * 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        analyze(problem, X, Y, mode=mode, n_classes=4, verbose=False)
    return [str(x.message) for x in w if "zero variance" in str(x.message)]


def test_joint_zero_variance_warning_does_not_claim_zero_index():
    X, Y = _zero_col_data()
    msgs = _warnings("multivariate", Y, X)
    assert len(msgs) == 1
    assert "indices will be 0" not in msgs[0]
    assert "k=0" in msgs[0]
    assert "joint transport" in msgs[0]
    # trajectory layout labels (t, k)
    msgs_t = _warnings("trajectory", Y[:, None, :], X)
    assert len(msgs_t) == 1 and "(t=0, k=0)" in msgs_t[0]


def test_joint_all_constant_warning_says_zero():
    X, _ = _zero_col_data()
    msgs = _warnings("multivariate", jnp.zeros((100, 2)), X)
    assert len(msgs) == 1
    assert "all indices will be 0" in msgs[0]


def test_univariate_zero_variance_warning_unchanged():
    X, Y = _zero_col_data()
    msgs = _warnings("univariate", Y, X)
    assert len(msgs) == 1 and "indices will be 0" in msgs[0]


def test_non_convergence_warning_mentions_tol_and_x64():
    X, Y = _zero_col_data()
    problem = jaxgsa.Problem(("a", "b"), ((-np.pi, np.pi),) * 2)
    with pytest.warns(jaxgsa.JaxgsaWarning, match="x64"):
        analyze(
            problem,
            X,
            Y[:, 1:].repeat(2, axis=1) + jnp.arange(2.0),
            mode="multivariate",
            n_classes=4,
            max_iter=2,
            tol=1e-12,
            verbose=False,
        )


def test_joint_warning_points_at_caller_for_ragged_outputs(X):
    """A ragged (dict) Y routes through irregular.py, but the warning still names this file."""
    y = np.asarray(X[:, 0] + X[:, 1] ** 2)
    Y = {
        "c1": (np.array([0.0, 1.0]), np.stack([np.zeros(N), y], 1)),
        "c2": (np.array([0.0, 0.5, 1.0]), np.stack([y, 2 * y, 3 * y], 1)),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyze(PROBLEM, X, Y, mode="trajectory", n_classes=4, verbose=False)
    hits = [w for w in caught if "zero variance" in str(w.message)]
    assert hits
    assert all(w.filename == __file__ for w in hits)
