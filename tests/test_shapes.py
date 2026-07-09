import jax
import jax.numpy as jnp

import gsax
from gsax.problem import Problem


def test_2d_input_shape():
    """2D input (n_total, K) should squeeze time dimension."""
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sr = gsax.sample(p, n_samples=256, seed=42, verbose=False)
    K = 3
    Y = jnp.ones((sr.n_total, K))
    # Add some variation so variance isn't zero
    Y = Y + jax.random.normal(jax.random.key(1), (sr.n_total, K))
    result = gsax.analyze(sr, Y)
    assert result.S1.shape == (K, p.num_vars)
    assert result.ST.shape == (K, p.num_vars)
    assert result.S2.shape == (K, p.num_vars, p.num_vars)


def test_3d_input_shape():
    """3D input (n_total, T, K) should preserve both dimensions."""
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sr = gsax.sample(p, n_samples=256, seed=42, verbose=False)
    T, K = 2, 3
    Y = jax.random.normal(jax.random.key(1), (sr.n_total, T, K))
    result = gsax.analyze(sr, Y)
    assert result.S1.shape == (T, K, p.num_vars)
    assert result.ST.shape == (T, K, p.num_vars)
    assert result.S2.shape == (T, K, p.num_vars, p.num_vars)


def test_1d_input_shape():
    """1D input (n_total,) should squeeze both time and output dimensions."""
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sr = gsax.sample(p, n_samples=256, seed=42, verbose=False)
    Y = jax.random.normal(jax.random.key(1), (sr.n_total,))
    result = gsax.analyze(sr, Y)
    assert result.S1.shape == (p.num_vars,)
    assert result.ST.shape == (p.num_vars,)
    assert result.S2.shape == (p.num_vars, p.num_vars)


def test_single_param():
    """Single parameter should produce scalar indices, no S2."""
    p = Problem.from_dict({"x1": (0.0, 1.0)})
    sr = gsax.sample(p, n_samples=256, calc_second_order=False, seed=42, verbose=False)
    Y = jax.random.normal(jax.random.key(1), (sr.n_total,))
    result = gsax.analyze(sr, Y)
    assert result.S1.shape == (p.num_vars,)
    assert result.ST.shape == (p.num_vars,)
    assert result.S2 is None


def test_single_output():
    """Single output (K=1) should still have correct shapes."""
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sr = gsax.sample(p, n_samples=256, seed=42, verbose=False)
    Y = jax.random.normal(jax.random.key(1), (sr.n_total, 1))
    result = gsax.analyze(sr, Y)
    assert result.S1.shape == (1, p.num_vars)
    assert result.ST.shape == (1, p.num_vars)


# --- Bootstrap shape tests ---


def _bootstrap_result(Y_shape_suffix, calc_second_order=True):
    """Helper: sample, generate Y, run bootstrap analyze."""
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sr = gsax.sample(p, n_samples=256, seed=42, calc_second_order=calc_second_order, verbose=False)
    Y = jax.random.normal(jax.random.key(1), (sr.n_total, *Y_shape_suffix))
    return gsax.analyze(sr, Y, num_resamples=50, key=jax.random.key(99)), p


def test_bootstrap_1d_shape():
    """1D Y with bootstrap should have (2, D) conf shapes."""
    result, p = _bootstrap_result(())
    D = p.num_vars
    assert result.S1.shape == (D,)
    assert result.S1_conf.shape == (2, D)
    assert result.ST_conf.shape == (2, D)
    assert result.S2_conf.shape == (2, D, D)


def test_bootstrap_2d_shape():
    """2D Y (n_total, K) with bootstrap should have (2, K, D) conf shapes."""
    result, p = _bootstrap_result((3,))
    D = p.num_vars
    assert result.S1.shape == (3, D)
    assert result.S1_conf.shape == (2, 3, D)
    assert result.ST_conf.shape == (2, 3, D)
    assert result.S2_conf.shape == (2, 3, D, D)


def test_bootstrap_3d_shape():
    """3D Y (n_total, T, K) with bootstrap should have (2, T, K, D) conf shapes."""
    result, p = _bootstrap_result((2, 3))
    D = p.num_vars
    assert result.S1.shape == (2, 3, D)
    assert result.S1_conf.shape == (2, 2, 3, D)
    assert result.ST_conf.shape == (2, 2, 3, D)
    assert result.S2_conf.shape == (2, 2, 3, D, D)


def test_no_bootstrap_conf_is_none():
    """Without bootstrap, _conf fields should be None."""
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sr = gsax.sample(p, n_samples=256, seed=42, verbose=False)
    Y = jax.random.normal(jax.random.key(1), (sr.n_total,))
    result = gsax.analyze(sr, Y)
    assert result.S1_conf is None
    assert result.ST_conf is None
    assert result.S2_conf is None


def test_bootstrap_no_second_order_shape():
    """Bootstrap without second order: S2_conf should be None."""
    result, p = _bootstrap_result((), calc_second_order=False)
    D = p.num_vars
    assert result.S1_conf.shape == (2, D)
    assert result.ST_conf.shape == (2, D)
    assert result.S2 is None
    assert result.S2_conf is None
