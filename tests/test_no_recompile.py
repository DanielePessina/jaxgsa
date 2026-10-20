"""A repeated analysis must not compile anything.

``jax.lax.map`` and ``jax.lax.scan`` called outside ``jit`` on a function
defined inside the caller trace and compile again on every call, because
the inner function is a new object each time. The first call is expected to
compile. A second call with the same shapes must reuse every program. This
bug made ``rank.analyze`` take 770 ms instead of 9 ms, and the PAWN
bootstrap 42 ms instead of 4 ms, with no change in the numbers, so only a
compile count can catch it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.monitoring
import jax.numpy as jnp
import pytest

import jaxgsa
from jaxgsa import sampling
from jaxgsa._core.registry import methods
from jaxgsa.benchmarks import ishigami

_COMPILES: list[str] = []


def _cache_size(program: Any) -> int:
    """Number of compiled variants a jitted program holds."""
    return int(program._cache_size())


def _count(event: str, duration_secs: float, **_: Any) -> None:
    if event == "/jax/core/compile/backend_compile_duration":
        _COMPILES.append(event)


jax.monitoring.register_event_duration_secs_listener(_count)

PROBLEM = ishigami.PROBLEM
N = 256


def _given_data() -> tuple[jax.Array, jax.Array]:
    X = jnp.asarray(sampling.monte_carlo(PROBLEM, N, seed=0))
    Y = ishigami.evaluate(X)
    return X, jnp.stack([Y, 2.0 * Y + 1.0], axis=1)


def _design(sample: Any) -> jax.Array:
    Y = ishigami.evaluate(jnp.asarray(sample.samples))
    return jnp.stack([Y, 2.0 * Y + 1.0], axis=1)


def _point(x: jax.Array) -> jax.Array:
    y = ishigami.evaluate(x[None, :])[0]
    return jnp.stack([y, 2.0 * y + 1.0])


def _calls() -> dict[str, Callable[[], Any]]:
    X, Y = _given_data()
    key = jax.random.key(0)
    sob = jaxgsa.sobol.sample(PROBLEM, 256, seed=0, verbose=False)
    mor = jaxgsa.morris.sample(PROBLEM, 8, seed=0, verbose=False)
    efa = jaxgsa.efast.sample(PROBLEM, n_per_curve=129, seed=0)
    kuc = jaxgsa.kucherenko.sample(PROBLEM, base_n=64, seed=0)
    Ys, Ym, Ye, Yk = _design(sob), _design(mor), _design(efa), _design(kuc)
    return {
        "borgonovo": lambda: jaxgsa.borgonovo.analyze(
            PROBLEM, X, Y, n_bootstrap=3, key=key, verbose=False
        ),
        "dgsm": lambda: jaxgsa.dgsm.analyze(PROBLEM, X, fn=_point, verbose=False),
        "efast": lambda: jaxgsa.efast.analyze(efa, Ye, verbose=False),
        "hdmr": lambda: jaxgsa.hdmr.analyze(PROBLEM, X, Y, order=2, max_iter=10, verbose=False),
        "hsic": lambda: jaxgsa.hsic.analyze(PROBLEM, X, Y, n_perms=5, key=key, verbose=False),
        "kucherenko": lambda: jaxgsa.kucherenko.analyze(kuc, Yk, verbose=False),
        "morris": lambda: jaxgsa.morris.analyze(mor, Ym, n_bootstrap=3, key=key, verbose=False),
        "optimal_transport": lambda: jaxgsa.optimal_transport.analyze(
            PROBLEM, X, Y, n_classes=4, n_bootstrap=3, key=key, verbose=False
        ),
        "pawn": lambda: jaxgsa.pawn.analyze(
            PROBLEM, X, Y, n_classes=4, n_bootstrap=3, key=key, verbose=False
        ),
        "pce": lambda: jaxgsa.pce.analyze(PROBLEM, X, Y, order=3, verbose=False),
        "rank": lambda: jaxgsa.rank.analyze(PROBLEM, X, Y, n_bootstrap=3, key=key, verbose=False),
        "shapley": lambda: jaxgsa.shapley.analyze(
            PROBLEM, X, Y, backend="pce", order=3, verbose=False
        ),
        "sobol": lambda: jaxgsa.sobol.analyze(sob, Ys, n_bootstrap=3, key=key, verbose=False),
        "vkoga": lambda: jaxgsa.vkoga.analyze(
            PROBLEM,
            X,
            Y,
            max_centers=16,
            n_folds=3,
            n_outer=16,
            n_inner=8,
            n_variance=64,
            key=key,
            verbose=False,
        ),
    }


def test_every_method_is_covered() -> None:
    assert set(_calls()) == set(methods())


@pytest.mark.parametrize("method", sorted(methods()))
@pytest.mark.filterwarnings("ignore")
def test_second_call_compiles_nothing(method: str) -> None:
    call = _calls()[method]
    jax.block_until_ready(jax.tree.leaves(call()))
    _COMPILES.clear()
    jax.block_until_ready(jax.tree.leaves(call()))
    assert _COMPILES == [], f"{method}: {len(_COMPILES)} programs compiled on a repeated call"


def test_pce_row_counts_in_one_bucket_share_one_compiled_fit() -> None:
    """Three nearby row counts must compile the PCE fit once, not three times.

    The fit pads its rows to a bucket (eight per octave), so 497, 498 and 499
    rows all run the 512-row program. The shared checks in ``_core`` still
    compile per shape, so the PCE programs are counted by their own caches.
    """
    from jaxgsa.pce import _analyze as pce_analyze

    programs = (
        pce_analyze._single_pass_coefficients,
        pce_analyze._single_pass_loo,
        pce_analyze._sobol_from_fit,
        pce_analyze._fit_summary,
    )
    # Another test may already have compiled one of these shapes.
    jax.clear_caches()
    sizes = []
    for n in (497, 498, 499):
        X = jnp.asarray(sampling.monte_carlo(PROBLEM, n, seed=0))
        Y = ishigami.evaluate(X)
        jax.block_until_ready(
            jax.tree.leaves(jaxgsa.pce.analyze(PROBLEM, X, Y, order=3, verbose=False))
        )
        sizes.append([_cache_size(program) for program in programs])
    assert sizes[1] == sizes[0] == sizes[2]
    # And a row count in the next bucket compiles each program once more.
    X = jnp.asarray(sampling.monte_carlo(PROBLEM, 600, seed=0))
    jaxgsa.pce.analyze(PROBLEM, X, ishigami.evaluate(X), order=3, verbose=False)
    assert [_cache_size(program) for program in programs] == [size + 1 for size in sizes[0]]
