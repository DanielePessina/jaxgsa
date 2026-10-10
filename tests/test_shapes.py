"""Tests for result shapes over the ``(N,)`` / ``(N, K)`` / ``(N, T, K)`` layouts.

Tier T4 (behavioural contract). No external oracle exists for our own shape
promotion rules; these tests pin the rank each result comes back at, for the
point estimates and the bootstrap intervals together.
"""

import jax
import pytest

import jaxgsa
from jaxgsa.problem import Problem

N_BOOTSTRAP = 8


@pytest.mark.parametrize(
    ("suffix", "n_bootstrap"),
    [
        ((), N_BOOTSTRAP),
        ((3,), N_BOOTSTRAP),
        ((2, 3), N_BOOTSTRAP),
        ((1,), N_BOOTSTRAP),
        ((2, 3), 0),
    ],
    ids=["1d", "2d", "3d", "k1", "3d-no-bootstrap"],
)
def test_result_ranks_follow_the_output_layout(suffix, n_bootstrap):
    """``(N,)`` squeezes time and output, ``(N, K)`` squeezes time, ``(N, T, K)`` keeps both.

    A single output ``(N, 1)`` keeps its output axis. The interval arrays gain
    a leading axis of two (lower, upper) on the same layout. Without a
    bootstrap the point estimates keep the same ranks and no interval is set.
    """
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    D = p.num_vars
    sr = jaxgsa.sobol.sample(p, n_samples=64, seed=42, verbose=False)
    Y = jax.random.normal(jax.random.key(1), (sr.n_runs, *suffix))
    result = jaxgsa.sobol.analyze(sr, Y, n_bootstrap=n_bootstrap, key=jax.random.key(99))

    assert result.S1.shape == (*suffix, D)
    assert result.ST.shape == (*suffix, D)
    assert result.S2.shape == (*suffix, D, D)
    if n_bootstrap == 0:
        assert result.S1_conf is None
        assert result.ST_conf is None
        assert result.S2_conf is None
        return
    assert result.S1_conf.shape == (2, *suffix, D)
    assert result.ST_conf.shape == (2, *suffix, D)
    assert result.S2_conf.shape == (2, *suffix, D, D)


def test_single_parameter_without_second_order_has_no_s2():
    """One parameter gives vector-free indices and, with no second order, no S2 at all."""
    p = Problem.from_dict({"x1": (0.0, 1.0)})
    sr = jaxgsa.sobol.sample(p, n_samples=64, calc_second_order=False, seed=42, verbose=False)
    Y = jax.random.normal(jax.random.key(1), (sr.n_runs,))
    result = jaxgsa.sobol.analyze(sr, Y, n_bootstrap=N_BOOTSTRAP, key=jax.random.key(99))

    assert result.S1.shape == (p.num_vars,)
    assert result.ST.shape == (p.num_vars,)
    assert result.S1_conf.shape == (2, p.num_vars)
    assert result.ST_conf.shape == (2, p.num_vars)
    assert result.S2 is None
    assert result.S2_conf is None
