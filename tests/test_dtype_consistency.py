"""Every method keeps the dtype of its data, also with x64 on.

With ``jax_enable_x64`` on, a stray float64 constant, NumPy array or
``jnp.zeros`` without ``dtype=`` silently promotes float32 work to float64.
The numbers stay right, so no value test sees it, but the result fields come
back float64 and the work runs at twice the memory. This file pins three
promises for every method and mode:

1. float32 data under x64 gives float32 in every floating field of
   ``analyze``'s result (``*_conf`` included) and of ``indices``' output;
2. those values match a float32 run with x64 off;
3. float32 ``X`` with float64 ``Y`` gives float64, the wider dtype (pce and
   vkoga document this; the other methods follow JAX promotion).

sobol, kucherenko, efast and morris read ``X`` from their own design, which
the samplers return as NumPy float64. There only ``Y`` carries the data
dtype, so the design does not count as data and the result follows ``Y``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import (
    Problem,
    borgonovo,
    dgsm,
    efast,
    hdmr,
    hsic,
    kucherenko,
    morris,
    optimal_transport,
    pawn,
    pce,
    rank,
    shapley,
    sobol,
    vkoga,
)

pytestmark = pytest.mark.filterwarnings("ignore::jaxgsa.JaxgsaWarning")

PI = float(np.pi)
PROBLEM = Problem.from_dict({"x1": (-PI, PI), "x2": (-PI, PI), "x3": (-PI, PI)})
N = 256
N_BOOT = 20

Outputs = dict[str, Any]
Runner = Callable[[Any, Any], Outputs]


def _ishigami(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64)
    return np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])


def _given_X(dtype) -> jax.Array:
    X = np.random.default_rng(0).uniform(-PI, PI, (N, 3))
    return jnp.asarray(X, dtype)


def _y(X: Any, dtype, kind: str = "scalar") -> jax.Array:
    """Model outputs in ``dtype``: ``(N,)``, ``(N, 3)`` or ``(N, 4, 2)``."""
    Xn = np.asarray(X, np.float64)
    y = _ishigami(Xn)
    if kind == "multi":
        y = np.stack([y, np.cos(Xn[:, 1]), y**2], axis=1)
    elif kind == "trajectory":
        t = np.linspace(0.5, 2.0, 4)
        y = np.stack([np.outer(y, t), np.outer(np.cos(Xn[:, 2]), t)], axis=2)
    return jnp.asarray(y, dtype)


# --- runners: one per method or mode --------------------------------------
# Each takes (x_dtype, y_dtype), runs analyze (with a bootstrap where the
# method has one) and indices (where it exists), and returns both.


def _sobol(xdt, ydt) -> Outputs:
    del xdt  # design dtype, see module docstring
    s = sobol.sample(PROBLEM, base_n=128, seed=0, verbose=False)
    Y = _y(s.samples, ydt)
    return {
        "analyze": sobol.analyze(s, Y, n_bootstrap=N_BOOT, key=0, verbose=False),
        "indices": sobol.indices(s, Y),
    }


def _kucherenko(xdt, ydt) -> Outputs:
    del xdt
    s = kucherenko.sample(PROBLEM, base_n=64, seed=0, verbose=False)
    Y = _y(s.samples, ydt)
    return {"analyze": kucherenko.analyze(s, Y, n_bootstrap=N_BOOT, key=0, verbose=False)}


def _efast(xdt, ydt) -> Outputs:
    del xdt
    s = efast.sample(PROBLEM, 129, seed=0, verbose=False)
    Y = _y(s.samples, ydt)
    return {"analyze": efast.analyze(s, Y, verbose=False), "indices": efast.indices(s, Y)}


def _morris(xdt, ydt) -> Outputs:
    del xdt
    s = morris.sample(PROBLEM, 20, seed=0, verbose=False)
    Y = _y(s.samples, ydt)
    return {
        "analyze": morris.analyze(s, Y, n_bootstrap=N_BOOT, key=0, verbose=False),
        "indices": morris.indices(s, Y),
    }


def _dgsm(xdt, ydt) -> Outputs:
    X = _given_X(xdt)

    def fn(x):
        y = jnp.sin(x[0]) + 7.0 * jnp.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * jnp.sin(x[0])
        return y.astype(ydt)

    return {
        "analyze": dgsm.analyze(PROBLEM, X, fn=fn, n_bootstrap=N_BOOT, key=0, verbose=False),
        "indices": dgsm.indices(PROBLEM, X, fn=fn),
    }


def _given(module, kind: str = "scalar", n_boot: int = N_BOOT, **kwargs) -> Runner:
    """Runner for a given-data method: ``analyze`` with a bootstrap, ``indices``."""

    def run(xdt, ydt) -> Outputs:
        X = _given_X(xdt)
        Y = _y(X, ydt, kind)
        return {
            "analyze": module.analyze(
                PROBLEM, X, Y, n_bootstrap=n_boot, key=0, verbose=False, **kwargs
            ),
            "indices": module.indices(PROBLEM, X, Y, **kwargs),
        }

    return run


def _hsic(xdt, ydt) -> Outputs:
    X = _given_X(xdt)
    Y = _y(X, ydt)
    return {
        "analyze": hsic.analyze(PROBLEM, X, Y, n_perms=20, key=0, verbose=False),
        "indices": hsic.indices(PROBLEM, X, Y, n_perms=20, key=0),
    }


def _vkoga(xdt, ydt) -> Outputs:
    X = _given_X(xdt)
    Y = _y(X, ydt)
    res = vkoga.analyze(
        PROBLEM,
        X,
        Y,
        key=0,
        max_centers=40,
        n_folds=4,
        n_outer=64,
        n_inner=32,
        n_variance=512,
        n_bootstrap=4,
        verbose=False,
    )
    return {"analyze": res}


RUNNERS: dict[str, Runner] = {
    "sobol": _sobol,
    "kucherenko": _kucherenko,
    "efast": _efast,
    "morris": _morris,
    "dgsm": _dgsm,
    "borgonovo": _given(borgonovo),
    "pawn": _given(pawn),
    "hsic": _hsic,
    "ot_univariate": _given(optimal_transport, mode="univariate"),
    # Sinkhorn per bootstrap replicate dominates the file's runtime, so the
    # joint modes get fewer replicates and a capped iteration count. Both
    # runs use the same cap, so the comparison is unaffected.
    "ot_multivariate": _given(
        optimal_transport, "multi", n_boot=3, mode="multivariate", n_classes=4, max_iter=200
    ),
    "ot_trajectory": _given(
        optimal_transport, "trajectory", n_boot=3, mode="trajectory", n_classes=4, max_iter=200
    ),
    "rank": _given(rank),
    "pce": _given(pce, order=3),
    "hdmr": _given(hdmr, order=2),
    "shapley_pce": _given(shapley, backend="pce"),
    "shapley_hdmr": _given(shapley, backend="hdmr"),
    "vkoga": _vkoga,
}

# Known leaks, found 2026-10-25: case -> reason. Each is xfail(strict=True)
# in the test it breaks, so a fix turns the xfail into a failure that asks
# for its entry to be removed.
F32_LEAKS: dict[str, str] = {}
VALUE_LEAKS: dict[str, str] = {}
MIXED_LEAKS: dict[str, str] = {}


def _params(leaks: dict[str, str]) -> list[Any]:
    return [
        pytest.param(
            name, marks=pytest.mark.xfail(strict=True, reason=leaks[name]) if name in leaks else ()
        )
        for name in RUNNERS
    ]


@lru_cache(maxsize=None)
def _run(name: str, x64: bool, xdt: str, ydt: str) -> Outputs:
    with jax.enable_x64(x64):
        return RUNNERS[name](jnp.dtype(xdt), jnp.dtype(ydt))


def _floats(obj: Any, path: str, depth: int = 0) -> Iterator[tuple[str, np.dtype, Any]]:
    """Yield ``(path, dtype, value)`` for every floating array inside ``obj``."""
    if depth > 6 or obj is None or isinstance(obj, Problem):
        return
    if isinstance(obj, jax.Array | np.ndarray | np.generic):
        if jnp.issubdtype(obj.dtype, jnp.floating):
            yield path, np.dtype(obj.dtype), obj
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        children = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    elif isinstance(obj, tuple) and hasattr(obj, "_asdict"):
        children = obj._asdict()
    elif isinstance(obj, dict):
        children = {str(k): v for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        children = {str(i): v for i, v in enumerate(obj)}
    else:
        return
    for key, value in children.items():
        yield from _floats(value, f"{path}.{key}", depth + 1)


def _collect(out: Outputs) -> dict[str, tuple[np.dtype, Any]]:
    found = {p: (d, v) for label, obj in out.items() for p, d, v in _floats(obj, label)}
    assert found, "no floating fields found"
    return found


def _wrong_dtypes(out: Outputs, want) -> list[str]:
    return [f"{p}: {d}" for p, (d, _) in _collect(out).items() if d != np.dtype(want)]


@pytest.mark.parametrize("name", _params(F32_LEAKS))
def test_float32_data_stays_float32_under_x64(name):
    out = _run(name, True, "float32", "float32")
    assert not _wrong_dtypes(out, np.float32), _wrong_dtypes(out, np.float32)


# Bootstrap intervals and permutation p-values are quantiles of resampled
# statistics, with the same key in both runs. A float32 rounding change in a
# replicate moves an interval end by rounding only, but it can be amplified
# where a replicate sits near a clip or a tie in the quantile, so these get
# 10x the point tolerance. A changed random stream moves them by ~1e-2 and
# still fails (see VALUE_LEAKS).
_LOOSE = ("_conf", "p_value", "replicates")


@pytest.mark.parametrize("name", _params(VALUE_LEAKS))
def test_float32_values_match_without_x64(name):
    got = _collect(_run(name, True, "float32", "float32"))
    ref = _collect(_run(name, False, "float32", "float32"))
    assert got.keys() == ref.keys()
    for path, (_, value) in got.items():
        loose = any(tag in path for tag in _LOOSE)
        np.testing.assert_allclose(
            np.asarray(value, np.float64),
            np.asarray(ref[path][1], np.float64),
            rtol=1e-3 if loose else 1e-4,
            atol=1e-4 if loose else 1e-5,
            err_msg=path,
        )


@pytest.mark.parametrize("name", _params(MIXED_LEAKS))
def test_float32_X_float64_Y_gives_float64(name):
    out = _run(name, True, "float32", "float64")
    assert not _wrong_dtypes(out, np.float64), _wrong_dtypes(out, np.float64)
