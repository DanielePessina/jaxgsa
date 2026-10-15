"""Guard the HSIC compile-time fix: the rolled bisection must stay exact."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxgsa.hsic import _analyze as hsic_impl


def _brute_boundaries(y: np.ndarray, t: np.floating) -> tuple[np.ndarray, np.ndarray]:
    """Find the row boundaries by scanning every pair.

    Args:
        y: Sorted values.
        t: Threshold.

    Returns:
        ``(right, left)`` as in ``_pair_boundaries``.
    """
    n = len(y)
    d2 = (y[None, :] - y[:, None]) ** 2
    ok = d2 <= t
    right = np.array([next((j for j in range(i, n) if not ok[i, j]), n) for i in range(n)])
    left = np.array([next(j for j in range(0, i + 1) if ok[i, j]) for i in range(n)])
    return right, left


def test_pair_boundaries_match_brute_force() -> None:
    """The rolled bisection returns the same blocks as a full scan."""
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 64, 100):
        y = np.sort(rng.normal(size=n).astype(np.float32))
        for t in (0.0, 0.01, 0.5, 4.0, 1e6):
            right, left = hsic_impl._pair_boundaries(jnp.asarray(y), jnp.float32(t))
            want_r, want_l = _brute_boundaries(y, np.float32(t))
            np.testing.assert_array_equal(np.asarray(right), want_r)
            np.testing.assert_array_equal(np.asarray(left), want_l)


def test_pairwise_quantile_matches_jnp_quantile() -> None:
    """The selection quantile still equals ``jnp.quantile`` bit for bit."""
    x = jax.random.uniform(jax.random.key(1), (97,))
    q = 0.37
    got = hsic_impl._pairwise_quantile_by_selection(x, q)
    want = jnp.quantile((x[:, None] - x[None, :]) ** 2, q)
    assert np.array_equal(np.asarray(got), np.asarray(want))


def test_input_kernels_match_per_input_build() -> None:
    """The vmapped bandwidth step gives the same kernels as a per-input build."""
    X = jax.random.uniform(jax.random.key(2), (80, 3))
    stack, *_ = jax.jit(hsic_impl._input_setup, static_argnames=("total", "bandwidth"))(
        X, jax.random.key(0), total=2, bandwidth=1.0
    )
    want = jax.jit(
        lambda X: jnp.stack([hsic_impl._build_one_kernel(X[:, d], 1.0) for d in range(3)])
    )(X)
    assert np.array_equal(np.asarray(stack), np.asarray(want))
