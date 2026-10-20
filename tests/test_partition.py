"""Tests for the equal-frequency class layout in ``jaxgsa._core.partition``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import Problem, borgonovo, optimal_transport
from jaxgsa._core import batching
from jaxgsa._core.partition import (
    _class_layout,
    plan_partition,
    replicate_blocks,
)


def test_n60_m22_uses_exact_integer_edges():
    """``N=60, M=22``: edge 11 is ``11 * 60 // 22 = 30``, not 29.

    ``floor(linspace(0, 60, 23))`` gives 29 there, because the float edge
    rounds just below 30. That moved one row between classes 10 and 11.
    """
    _, sizes = _class_layout(60, 22)
    edges = (np.arange(23) * 60) // 22
    np.testing.assert_array_equal(sizes, np.diff(edges))
    assert np.cumsum(sizes)[10] == 30


def test_sizes_are_an_exact_equal_frequency_split():
    """Edges are ``(m * N) // M``; sizes sum to N and differ by at most one."""
    for N in range(1, 301):
        for M in range(1, min(N, 64) + 1):
            take, sizes = _class_layout(N, M)
            edges = (np.arange(M + 1) * N) // M
            np.testing.assert_array_equal(np.concatenate([[0], np.cumsum(sizes)]), edges)
            assert sizes.sum() == N
            assert sizes.max() - sizes.min() <= 1
            assert take.shape == (M, sizes.max())
            np.testing.assert_array_equal(take[:, 0], edges[:-1])


# ---------------------------------------------------------------------------
# Width bucketing and replicate blocks
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_budget():
    """Force several replicate blocks, then restore the budget."""
    old = batching.get_memory_budget()
    yield lambda nbytes: batching._set_memory_budget(nbytes)
    batching._set_memory_budget(old)


def _mixed(N=240, seed=0):
    problem = Problem.from_dict(
        {
            "a": (0.0, 1.0),
            "b": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": [0.7, 0.2, 0.1]},
        }
    )
    rng = np.random.default_rng(seed)
    X = rng.random((N, 3))
    X[:, 2] = rng.choice(3, N, p=[0.7, 0.2, 0.1])
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + 0.5 * X[:, 2]
    return problem, jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)


def test_blocks_are_near_equal_and_cover_every_replicate(tiny_budget):
    problem, X, _ = _mixed()
    all_idx = jnp.concatenate(
        [
            jnp.arange(240, dtype=jnp.int32)[None],
            jax.random.randint(jax.random.key(0), (20, 240), 0, 240, dtype=jnp.int32),
        ]
    )
    dims = ((2, 3),)
    plan = plan_partition(problem, X, all_idx, 8, dims)
    tiny_budget(plan.bytes_per_replicate * 5)
    sizes = [idx.shape[0] for idx, _ in replicate_blocks(plan)]
    assert sum(sizes) == 21
    assert max(sizes) - min(sizes) <= 1
    assert len(sizes) > 1
    # One pad width for every block, so every block is shaped alike.
    widths = {g[1][0].shape[-1] for _, g in replicate_blocks(plan)}
    assert len(widths) == 1


@pytest.mark.parametrize("x64", [False, True])
def test_borgonovo_blocks_match_one_block_bitwise(tiny_budget, x64):
    problem, X, y = _mixed()
    with jax.enable_x64(x64):
        Xd, yd = (
            X.astype(jnp.float64 if x64 else jnp.float32),
            y.astype(jnp.float64 if x64 else jnp.float32),
        )
        kw = dict(
            n_bootstrap=20, key=jax.random.key(1), n_classes=8, slice_chunk_size=1, verbose=False
        )
        ref = borgonovo.analyze(problem, Xd, yd, **kw)
        tiny_budget(40_000)
        blocked = borgonovo.analyze(problem, Xd, yd, **kw)
    np.testing.assert_array_equal(ref.delta, blocked.delta)
    np.testing.assert_array_equal(ref.delta_conf, blocked.delta_conf)
    np.testing.assert_array_equal(ref.S1, blocked.S1)


@pytest.mark.filterwarnings("ignore:.*Sinkhorn solves did not reach")
@pytest.mark.parametrize("mode", ["univariate", "multivariate", "trajectory"])
def test_ot_blocks_match_one_block_bitwise(tiny_budget, mode):
    problem, X, y = _mixed(N=120)
    Y = jnp.stack([y, y**2], axis=1)
    if mode == "trajectory":
        Y = jnp.stack([Y, Y + 1.0, jnp.sin(Y)], axis=1)  # (N, T=3, K=2)
    kw = dict(
        mode=mode,
        n_bootstrap=6,
        key=jax.random.key(2),
        n_classes=5,
        max_iter=50,
        **({"slice_chunk_size": 1} if mode == "univariate" else {}),
        dummy=True,
        verbose=False,
    )
    ref = optimal_transport.analyze(problem, X, Y, **kw)
    tiny_budget(20_000)
    blocked = optimal_transport.analyze(problem, X, Y, **kw)
    np.testing.assert_array_equal(ref.ot, blocked.ot)
    np.testing.assert_array_equal(ref.ot_conf, blocked.ot_conf)
    np.testing.assert_array_equal(ref.advective, blocked.advective)
    np.testing.assert_array_equal(ref.diffusive, blocked.diffusive)
    np.testing.assert_array_equal(ref.ot_dummy, blocked.ot_dummy)
