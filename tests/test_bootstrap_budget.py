"""The cached bootstrap program must be rebuilt when the memory budget changes."""

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import pytest

from jaxgsa import config
from jaxgsa._core.bootstrap import bootstrap_draws

_TRACES = {"n": 0}


def _counting_fit(rows: jax.Array, x: jax.Array) -> jax.Array:
    """Count Python-level traces, then return a row-resampled mean."""
    _TRACES["n"] += 1
    return jnp.mean(x[rows])


@pytest.fixture
def restore_budget() -> Iterator[None]:
    old = config.get_memory_budget()
    try:
        yield
    finally:
        config.set_memory_budget(old)


def test_budget_change_retraces_replicate_program(restore_budget: None) -> None:
    x = jnp.arange(32.0)
    key = jax.random.key(0)
    config.set_memory_budget(512)
    bootstrap_draws(key, 4, _counting_fit, (x,))
    after_first = _TRACES["n"]
    bootstrap_draws(key, 4, _counting_fit, (x,))
    assert _TRACES["n"] == after_first  # same budget: cache hit
    config.set_memory_budget(64)
    bootstrap_draws(key, 4, _counting_fit, (x,))
    assert _TRACES["n"] > after_first  # new budget: rebuilt
