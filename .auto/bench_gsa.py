"""Autoresearch benchmark for jaxgsa global-sensitivity-analysis kernels.

Covers the workload axes the user cares about -- small problems, high sample
counts, high parameter counts, high output counts, high timepoint counts and
bootstrap -- across every registered method, and reports both steady-state
analysis time and peak process memory for each case.

Measurement methodology is inherited from ``scripts/benchmark_methods.py``:

* each (method, workload) case runs in its **own subprocess**, so peak memory
  is a true per-case high-water mark (``ru_maxrss``) and the first call is an
  honest cold-compile number (no JIT cache is shared between cases);
* only ``analyze`` is timed -- design generation (``sample``) is excluded
  because that is not where the estimator kernels live;
* every timed call blocks on the result before the clock stops (JAX dispatch
  is asynchronous);
* the first call is reported separately (it carries tracing + XLA compile);
  steady-state is the best of ``repeats`` later calls.

The script writes a human-readable table to ``.auto/bench_latest.md`` (so a
live ``tail -f`` shows the per-case numbers) and prints ``METRIC name=value``
lines on stdout for the autoresearch loop.

Run with::

    uv run .auto/bench_gsa.py [--repeats N] [--out FILE.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import jaxgsa
from jaxgsa import (
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
    shapley,
    sobol,
    vkoga,
)
from jaxgsa.benchmarks import ishigami, sobol_g

SEED = 20260819
DEFAULT_REPEATS = 3

# ---------------------------------------------------------------------------
# Workload matrix. Each row is one (method, workload) case. ``N`` is the
# sample budget, ``D`` the parameter count, ``dims`` the trailing Y shape
# (S = prod(dims) output slices, where a 2-D dims is T x K timepoints x
# outputs), and ``s2``/``boot`` are method-specific toggles. The four "high"
# axes the user named are all present, plus scalar small cases.
# ---------------------------------------------------------------------------

CASES: list[dict[str, Any]] = [
    # ---- sobol (design-based, Saltelli) ----------------------------------
    {"key": "sobol_small", "method": "sobol", "D": 3, "N": 1024,
     "dims": (), "s2": False, "boot": 0},
    {"key": "sobol_high_n", "method": "sobol", "D": 3, "N": 65536,
     "dims": (), "s2": False, "boot": 0},
    {"key": "sobol_high_d", "method": "sobol", "D": 30, "N": 0,
     "base_n": 8192, "dims": (), "s2": True, "boot": 0},
    {"key": "sobol_tk", "method": "sobol", "D": 3, "N": 8192,
     "dims": (50, 6), "s2": False, "boot": 0},
    {"key": "sobol_many_slices", "method": "sobol", "D": 3, "N": 16384,
     "dims": (32, 16), "s2": True, "boot": 0},
    {"key": "sobol_boot", "method": "sobol", "D": 3, "N": 4096,
     "dims": (32,), "s2": False, "boot": 100},
    # ---- morris -----------------------------------------------------------
    {"key": "morris_small", "method": "morris", "D": 3, "N": 128, "dims": (), "boot": 0},
    {"key": "morris_high_d", "method": "morris", "D": 30, "N": 1240, "dims": (), "boot": 0},
    # ---- kucherenko -------------------------------------------------------
    {"key": "kucherenko_small", "method": "kucherenko", "D": 3, "N": 1024, "dims": ()},
    {"key": "kucherenko_slices", "method": "kucherenko", "D": 10, "N": 8192, "dims": (32,)},
    # ---- efast ------------------------------------------------------------
    {"key": "efast_small", "method": "efast", "D": 3, "N": 1025, "dims": ()},
    {"key": "efast_high_d", "method": "efast", "D": 15, "N": 2049, "dims": ()},
    # ---- dgsm (given data, differentiated model) --------------------------
    {"key": "dgsm_small", "method": "dgsm", "D": 3, "N": 4096, "dims": ()},
    {"key": "dgsm_high_d", "method": "dgsm", "D": 15, "N": 4096, "dims": (50, 1)},
    # ---- hdmr / pce / shapley (surrogate, given data) ---------------------
    {"key": "hdmr_small", "method": "hdmr", "D": 3, "N": 1024, "dims": ()},
    {"key": "hdmr_tk", "method": "hdmr", "D": 3, "N": 1024, "dims": (50, 6)},
    {"key": "pce_small", "method": "pce", "D": 3, "N": 1024, "dims": ()},
    {"key": "shapley_small", "method": "shapley", "D": 3, "N": 1024, "dims": ()},
    # ---- moment-independent / distributional (given data) -----------------
    {"key": "borgonovo_small", "method": "borgonovo", "D": 3, "N": 1024, "dims": (), "boot": 10},
    {"key": "pawn_slices", "method": "pawn", "D": 3, "N": 1024, "dims": (32,), "boot": 10},
    {"key": "ot_slices", "method": "optimal_transport", "D": 3, "N": 1024,
     "dims": (32,), "boot": 10},
    # ---- O(N^2) / greedy methods (kept small so the sweep stays bounded) --
    {"key": "hsic_small", "method": "hsic", "D": 3, "N": 512, "dims": ()},
    {"key": "vkoga_small", "method": "vkoga", "D": 3, "N": 512, "dims": ()},
]


# ---------------------------------------------------------------------------
# Model / design helpers
# ---------------------------------------------------------------------------


def make_problem(D: int) -> jaxgsa.Problem:
    """Return a problem with ``D`` independent uniform parameters."""
    if D == 3:
        return ishigami.PROBLEM
    return jaxgsa.Problem.from_dict({f"x{i + 1}": (0.0, 1.0) for i in range(D)})


def make_model(D: int) -> Any:
    """Return a batched model mapping ``(N, D)`` to ``(N,)``."""
    if D == 3:
        return ishigami.evaluate
    a = tuple(0.5 * i for i in range(D))
    return lambda x: sobol_g.evaluate(x, a=a)


def point_model(batched: Any) -> Any:
    """Wrap a batched model into a one-row callable for ``dgsm``."""

    def point(x: jax.Array) -> jax.Array:
        return batched(x[None, :])[0]

    return point


def widen(y: jax.Array, dims: tuple[int, ...]) -> jax.Array:
    """Widen a scalar output to ``(N, *dims)`` so no slice is degenerate."""
    if not dims:
        return y
    n = int(jnp.prod(jnp.asarray(dims)))
    scales = 1.0 + 0.5 * jnp.arange(n, dtype=y.dtype) / n
    offsets = jnp.arange(n, dtype=y.dtype)
    flat = y[:, None] * scales[None, :] + offsets[None, :]
    return flat.reshape((y.shape[0], *dims))


# ---------------------------------------------------------------------------
# Per-method call builders
# ---------------------------------------------------------------------------


def _build_sobol(c: dict[str, Any], X: jax.Array | None, y: jax.Array, sr: Any) -> Any:
    key = jax.random.key(SEED) if c["boot"] else None
    return lambda: sobol.analyze(
        sr, y, n_bootstrap=int(c["boot"]), key=key, verbose=False
    )


def _build_morris(c: dict[str, Any], X: jax.Array | None, y: jax.Array, sr: Any) -> Any:
    key = jax.random.key(SEED) if c["boot"] else None
    return lambda: morris.analyze(
        sr, y, n_bootstrap=int(c["boot"]), key=key, verbose=False
    )


def _build_efast(c: dict[str, Any], X: jax.Array | None, y: jax.Array, sr: Any) -> Any:
    return lambda: efast.analyze(sr, y, verbose=False)


def _build_kucherenko(c: dict[str, Any], X: jax.Array | None, y: jax.Array, sr: Any) -> Any:
    return lambda: kucherenko.analyze(sr, y, verbose=False)


def _build_borgonovo(c: dict[str, Any], X: jax.Array | None, y: jax.Array, sr: Any) -> Any:
    assert X is not None
    return lambda: borgonovo.analyze(
        c["_problem"], X, y, n_bootstrap=int(c["boot"]), key=jax.random.key(SEED)
    )


def _build_pawn(c: dict[str, Any], X: jax.Array | None, y: jax.Array, sr: Any) -> Any:
    assert X is not None
    return lambda: pawn.analyze(
        c["_problem"], X, y, n_bins=8, n_bootstrap=int(c["boot"]), key=jax.random.key(SEED)
    )


def _build_hsic(c: dict[str, Any], X: jax.Array | None, y: jax.Array, sr: Any) -> Any:
    assert X is not None
    return lambda: hsic.analyze(
        c["_problem"], X, y, n_perms=10, key=jax.random.key(SEED)
    )


def _build_ot(c: dict[str, Any], X: jax.Array | None, y: jax.Array, sr: Any) -> Any:
    assert X is not None
    return lambda: optimal_transport.analyze(
        c["_problem"],
        X,
        y,
        n_partitions=8,
        n_bootstrap=int(c["boot"]),
        dummy=False,
        key=jax.random.key(SEED),
    )


def _build_pce(c: dict[str, Any], X: jax.Array | None, y: jax.Array, sr: Any) -> Any:
    assert X is not None
    return lambda: pce.analyze(c["_problem"], X, y, order=3)


def _build_hdmr(c: dict[str, Any], X: jax.Array | None, y: jax.Array, sr: Any) -> Any:
    assert X is not None
    return lambda: hdmr.analyze(c["_problem"], X, y, maxorder=2, maxiter=50)


def _build_shapley(c: dict[str, Any], X: jax.Array | None, y: jax.Array, sr: Any) -> Any:
    assert X is not None
    return lambda: shapley.analyze(c["_problem"], X, y, backend="pce", order=3)


def _build_vkoga(c: dict[str, Any], X: jax.Array | None, y: jax.Array, sr: Any) -> Any:
    assert X is not None
    return lambda: vkoga.analyze(
        c["_problem"],
        X,
        y,
        max_centers=32,
        n_folds=3,
        n_outer=64,
        n_inner=32,
        n_variance=512,
        key=jax.random.key(SEED),
    )


def _build_dgsm(c: dict[str, Any], X: jax.Array | None, y: jax.Array, sr: Any) -> Any:
    assert X is not None
    model = point_model(c["_model"])
    return lambda: dgsm.analyze(c["_problem"], model, X)


BUILDERS = {
    "sobol": _build_sobol,
    "morris": _build_morris,
    "efast": _build_efast,
    "kucherenko": _build_kucherenko,
    "borgonovo": _build_borgonovo,
    "pawn": _build_pawn,
    "hsic": _build_hsic,
    "optimal_transport": _build_ot,
    "pce": _build_pce,
    "hdmr": _build_hdmr,
    "shapley": _build_shapley,
    "vkoga": _build_vkoga,
    "dgsm": _build_dgsm,
}


def build_call(c: dict[str, Any]) -> Any:
    """Prepare inputs and return the zero-argument timed call for a case."""
    D = int(c["D"])
    dims = tuple(c["dims"])
    method = c["method"]
    problem = make_problem(D)
    model = make_model(D)
    c["_problem"] = problem
    c["_model"] = model

    if method in ("sobol", "morris", "efast", "kucherenko"):
        if method == "sobol":
            base_n = c.get("base_n")
            n_samples = 1 if base_n else int(c["N"])
            sr = sobol.sample(
                problem,
                n_samples=n_samples,
                base_n=base_n,
                calc_second_order=bool(c.get("s2", False)),
                seed=SEED,
                verbose=False,
            )
        elif method == "morris":
            n_traj = max(1, int(c["N"]) // (D + 1))
            sr = morris.sample(problem, n_trajectories=n_traj, seed=SEED, verbose=False)
        elif method == "efast":
            sr = efast.sample(problem, n_per_curve=int(c["N"]), seed=SEED, verbose=False)
        else:  # kucherenko
            sr = kucherenko.sample(problem, n_samples=int(c["N"]), seed=SEED, verbose=False)
        y = jnp.asarray(model(jnp.asarray(sr.samples)), dtype=jnp.float32)
        y = widen(y, dims)
        jax.block_until_ready(y)
        return BUILDERS[method](c, None, y, sr)

    # given-data methods: fixed Monte Carlo design
    N = int(c["N"])
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, N, seed=SEED))
    y = jnp.asarray(model(X), dtype=jnp.float32)
    y = widen(y, dims)
    jax.block_until_ready((X, y))
    return BUILDERS[method](c, X, y, None)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _arrays(value: Any, depth: int = 0) -> list[jax.Array]:
    if depth > 6:
        return []
    if isinstance(value, jax.Array):
        return [value]
    if isinstance(value, jaxgsa.Problem | str | bytes):
        return []
    if isinstance(value, dict):
        out: list[jax.Array] = []
        for item in value.values():
            out.extend(_arrays(item, depth + 1))
        return out
    if isinstance(value, list | tuple):
        out2: list[jax.Array] = []
        for item in value:
            out2.extend(_arrays(item, depth + 1))
        return out2
    return []


def timed_call(fn: Any) -> float:
    start = time.perf_counter()
    result = fn()
    jax.block_until_ready(_arrays(result))
    return time.perf_counter() - start


def _maxrss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


def measure_worker(c: dict[str, Any], repeats: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case": c["key"],
        "method": c["method"],
        "D": c["D"],
        "N": c["N"],
        "dims": list(c["dims"]),
    }
    call = build_call(c)
    rss_setup = _maxrss_bytes()
    try:
        first = timed_call(call)
        rss_after_first = _maxrss_bytes()
        later = [timed_call(call) for _ in range(repeats)]
    except (ValueError, TypeError, NotImplementedError) as exc:
        record["status"] = "raised"
        record["exception"] = f"{type(exc).__name__}: {exc}"
        return record
    rss_peak = _maxrss_bytes()
    best = min(later) if later else first
    n_slices = int(np.prod(c["dims"])) if c["dims"] else 1
    record.update(
        {
            "status": "ok",
            "n_slices": n_slices,
            "first_call_s": round(first, 6),
            "run_best_s": round(best, 6),
            "run_median_s": round(float(np.median(later)), 6) if later else round(first, 6),
            "run_worst_s": round(max(later), 6) if later else round(first, 6),
            "run_times_s": [round(t, 6) for t in later],
            "compile_s": round(max(first - best, 0.0), 6),
            "run_per_slice_s": round(best / max(n_slices, 1), 9),
            "peak_rss_bytes": max(rss_peak - rss_setup, 0),
            "compile_rss_bytes": max(rss_after_first - rss_setup, 0),
            "run_extra_rss_bytes": max(rss_peak - rss_after_first, 0),
            "setup_rss_bytes": rss_setup,
        }
    )
    return record


def run_worker(key: str, repeats: int) -> int:
    c = next(cc for cc in CASES if cc["key"] == key)
    sys.stdout.write(json.dumps(measure_worker(c, repeats)))
    return 0


def run_sweep(keys: list[str], repeats: int, timeout: float) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    script = str(Path(__file__).resolve())
    for key in keys:
        cmd = [sys.executable, script, "--worker", key, "--repeats", str(repeats)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            records.append({"case": key, "status": "timeout", "exception": f"> {timeout:.0f}s"})
            continue
        try:
            records.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            records.append(
                {
                    "case": key,
                    "status": "failed",
                    "exception": (proc.stderr or "no output").strip()[-400:],
                }
            )
    return records


def _fmt_time(s: float) -> str:
    return f"{s * 1e3:.1f} ms" if s < 1.0 else f"{s:.2f} s"


def _fmt_bytes(b: int) -> str:
    return "<noise" if b <= 0 else f"{b / 1024**2:.1f} MiB"


def format_table(records: list[dict[str, Any]]) -> str:
    header = (
        "| case | method | D | N | slices | compile | run(best) | spread | "
        "peak RSS | run extra |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for rec in records:
        if rec.get("status") != "ok":
            lines.append(
                f"| {rec['case']} | {rec.get('method', '')} | | | | {rec.get('status')} "
                f"| {str(rec.get('exception', ''))[:60]} | | | |\n"
            )
            continue
        best = float(rec["run_best_s"])
        worst = float(rec["run_worst_s"])
        spread = f"{(worst - best) / best * 100:.0f}%" if best > 0 else "n/a"
        lines.append(
            f"| {rec['case']} | {rec['method']} | {rec['D']} | {rec['N']} | "
            f"{rec['n_slices']} | {_fmt_time(float(rec['compile_s']))} | "
            f"{_fmt_time(best)} | {spread} | "
            f"{_fmt_bytes(int(rec['peak_rss_bytes']))} | "
            f"{_fmt_bytes(int(rec['run_extra_rss_bytes']))} |\n"
        )
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--cases", default="", help="comma-separated subset of case keys")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker is not None:
        return run_worker(args.worker, args.repeats)

    keys = [c["key"] for c in CASES]
    if args.cases:
        wanted = [k.strip() for k in args.cases.split(",") if k.strip()]
        keys = [k for k in keys if k in wanted]

    records = run_sweep(keys, args.repeats, args.timeout)
    table = format_table(records)

    # write the live table for the tmux viewer
    latest = Path(__file__).parent / "bench_latest.md"
    latest.write_text(table)

    ok = [r for r in records if r.get("status") == "ok"]
    times = [float(r["run_best_s"]) for r in ok if float(r["run_best_s"]) > 0]
    rss = [int(r["peak_rss_bytes"]) for r in ok]
    geomean_ms = float(np.exp(np.mean(np.log(np.asarray(times)))) * 1000.0) if times else 0.0
    total_s = float(np.sum(np.asarray(times))) if times else 0.0
    sum_rss_mib = float(np.sum(np.asarray(rss))) / 1024**2 if rss else 0.0
    max_rss_mib = float(np.max(np.asarray(rss))) / 1024**2 if rss else 0.0

    print(table)
    print(f"METRIC geomean_ms={geomean_ms:.4f}")
    print(f"METRIC total_s={total_s:.4f}")
    print(f"METRIC sum_peak_rss_mib={sum_rss_mib:.2f}")
    print(f"METRIC max_peak_rss_mib={max_rss_mib:.2f}")

    if args.out is not None:
        doc = {
            "header": {
                "jax_version": jax.__version__,
                "x64": bool(getattr(jax.config, "jax_enable_x64")),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "seed": SEED,
            },
            "records": records,
        }
        args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
