"""Array-valued scalar fit arguments must work with the jitted bootstrap."""

from typing import Any

import jax
import jax.numpy as jnp

from jaxgsa import Problem, hdmr, pce, shapley


def test_array_ridge_with_bootstrap() -> None:
    problem = Problem(("a", "b"), ((0.0, 1.0), (0.0, 1.0)))
    X = jax.random.uniform(jax.random.key(0), (256, 2))
    Y = X[:, 0] + 2.0 * X[:, 1] ** 2
    key = jax.random.key(1)
    r: Any = jnp.asarray(1e-6)
    pce.analyze(problem, X, Y, ridge=r, n_bootstrap=2, key=key)
    hdmr.analyze(problem, X, Y, ridge=r, n_bootstrap=2, key=key)
    shapley.analyze(problem, X, Y, ridge=r, n_bootstrap=2, key=key)
