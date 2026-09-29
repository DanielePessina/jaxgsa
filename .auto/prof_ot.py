"""Profile the OT kernel internals for the bench cases (untouched method focus)."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

import jaxgsa
from jaxgsa import optimal_transport
from jaxgsa.benchmarks import ishigami, sobol_g

SEED = 20260819


def make_problem(n_dims: int) -> jaxgsa.Problem:
    if n_dims == 3:
        return ishigami.PROBLEM
    return jaxgsa.Problem.from_dict({f"x{i + 1}": (0.0, 1.0) for i in range(n_dims)})


def make_model(n_dims: int):
    if n_dims == 3:
        return ishigami.evaluate
    a = tuple(0.5 * i for i in range(n_dims))
    return lambda x: sobol_g.evaluate(x, a=a)


def widen(y: jax.Array, dims: tuple[int, ...]) -> jax.Array:
    if not dims:
        return y
    n = int(jnp.prod(jnp.asarray(dims)))
    scales = 1.0 + 0.5 * jnp.arange(n, dtype=y.dtype) / n
    offsets = jnp.arange(n, dtype=y.dtype)
    flat = y[:, None] * scales[None, :] + offsets[None, :]
    return flat.reshape((y.shape[0], *dims))


def time_call(fn) -> float:
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    return time.perf_counter() - t0


def bench_case(n_dims: int, n: int, dims: tuple[int, ...], boot: int, tag: str) -> None:
    problem = make_problem(n_dims)
    model = make_model(n_dims)
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, n, seed=SEED))
    y = jnp.asarray(model(X), dtype=jnp.float32)
    y = widen(y, dims)
    jax.block_until_ready((X, y))
    key = jax.random.key(SEED) if boot else None
    call = lambda: optimal_transport.analyze(
        problem, X, y, n_partitions=8, n_bootstrap=boot, dummy=False, key=key, verbose=False
    )
    first = time_call(call)
    best = min(time_call(call) for _ in range(9))
    print(f"{tag}: first={first*1000:.1f}ms best={best*1000:.1f}ms")


if __name__ == "__main__":
    bench_case(3, 1024, (32,), 10, "ot_slices")
    bench_case(3, 1024, (), 50, "ot_boot")
    bench_case(15, 1024, (), 0, "ot_high_d")
    bench_case(3, 8192, (), 0, "ot_high_n")