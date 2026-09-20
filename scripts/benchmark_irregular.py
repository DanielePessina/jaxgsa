"""Benchmark the irregular-output path against its regular equivalent.

The irregular path routes one analysis per output channel through the method's
regular kernels, so its cost should be the sum of the per-channel regular
analyses plus a small host-side coercion pass. This script measures that for
:func:`jaxgsa.sobol.analyze` on the reference case (two channels on different,
non-uniform grids):

* ``irregular``: one ``analyze(...)`` call with ragged ``Y``.
* ``manual``: the two channels analyzed separately through the regular path
  (the engine's per-channel work, without the wrapper).

Both run in one process, so the JIT kernels are compiled once by the irregular
call and reused by the manual path; the manual warm time is the reference the
wrapper is measured against. Run it with::

    uv run scripts/benchmark_irregular.py

It prints a table and exits 0. Nothing is written to disk.
"""

from __future__ import annotations

import dataclasses
import time

import jax.numpy as jnp
import numpy as np

import jaxgsa.sobol as sobol
from jaxgsa.problem import Problem

N = 512
D = 8
OUTPUT_NAMES = ("conc", "d43")
PROBLEM = Problem(
    names=tuple(f"x{i}" for i in range(D)),
    bounds=tuple((0.0, 1.0) for _ in range(D)),
    output_names=OUTPUT_NAMES,
)
T_CONC = jnp.array([0.0, 1.0, 2.0, 2.5, 6.0])
T_D43 = jnp.array([0.0, 2.0, 5.5, 6.0])


def _ragged_y(n: int):
    rng = np.random.default_rng(0)
    return [
        (T_CONC, jnp.asarray(rng.normal(size=(n, T_CONC.shape[0])))),
        (T_D43, jnp.asarray(rng.normal(size=(n, T_D43.shape[0])))),
    ]


def _bench(fn, n_reps: int) -> float:
    """Wall-clock a warm call, blocking on every channel's device work."""
    t0 = time.perf_counter()
    for _ in range(n_reps):
        result = fn()
        if hasattr(result, "S1"):
            result.S1.block_until_ready()
        else:
            for channel in result.channels.values():
                channel.S1.block_until_ready()
    return (time.perf_counter() - t0) / n_reps


def main() -> None:
    design = sobol.sample(PROBLEM, N, seed=0, verbose=False)
    ragged = _ragged_y(design.n_runs)

    def irregular():
        return sobol.analyze(design, ragged, verbose=False)

    def manual():
        results = {}
        for name, (times, values) in zip(OUTPUT_NAMES, ragged, strict=True):
            sub = PROBLEM.with_output_names((name,))
            sub_design = dataclasses.replace(design, problem=sub)
            results[name] = sobol.analyze(sub_design, values[:, :, None], verbose=False)
        return results["conc"]

    # Warm-up / cold-compile pass: the reported "first" time is the cold run,
    # everything after is the warm average.
    cold = time.perf_counter()
    irregular()
    irregular_cold = time.perf_counter() - cold

    irregular_warm = _bench(irregular, 20)
    manual_warm = _bench(manual, 20)

    n_cols = 46
    print("irregular-vs-manual benchmark (sobol, second order, D=8, N=512 runs)")
    print("  channels: conc (T=5 on {0,1,2,2.5,6}), d43 (T=4 on {0,2,5.5,6})")
    print("-" * n_cols)
    print(f"{'path':<10}{'first (s)':>10}{'warm (ms)':>11}")
    print("-" * n_cols)
    print(f"{'irregular':<10}{irregular_cold:>10.3f}{irregular_warm * 1e3:>11.3f}")
    print(f"{'manual':<10}{'-':>10}{manual_warm * 1e3:>11.3f}")
    print("-" * n_cols)
    print("first = first call incl. JIT compile; warm = mean of 20 calls.")
    print("manual warm is the same two kernels without the wrapper; both run in")
    print("one process, so the first-call column is measured once, for irregular.")
    extra = (irregular_warm - manual_warm) * 1e6
    if manual_warm > 0:
        ratio = f"{irregular_warm / manual_warm:.2f}x"
    else:
        ratio = "n/a"
    print(f"absolute wrapper overhead: {extra:.0f} us per call ({ratio})")
    if manual_warm > 0 and irregular_warm > max(5 * manual_warm, 5e-3):
        raise SystemExit("irregular wrapper overhead exceeds 5x the per-channel work")


if __name__ == "__main__":
    main()
