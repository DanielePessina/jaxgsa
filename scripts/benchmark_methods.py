"""Runtime and peak-memory benchmark for every jaxgsa method.

This is the measurement half of a kernel-optimisation effort. It answers two
questions per method, per output shape: how long does the analysis take, and
how much memory does it peak at. Both matter. The intended kernel shape in
this library is "one atomic kernel for a single output slice, ``vmap`` over a
chunk of slices, loop over chunks". A change that makes a method faster by
mapping over every slice at once, and in doing so materialises the whole
batch, is not an improvement. Timing alone cannot see that, so this script
measures allocation too.

It is not a correctness harness. ``scripts/baseline_dump.py`` pins the
numbers and the root-level ``benchmark_salib.py`` checks parity against
SALib; neither of them times anything but sobol and hdmr.

Run it with::

    uv run scripts/benchmark_methods.py --out results.json

How it works
------------

Each (method, output shape) pair runs in its own subprocess. That is what
makes peak memory measurable at all on this backend -- see "Peak memory"
below -- and it also makes the first-call timing an honest cold-compile
number, because no JIT cache is shared between cases.

Timing
------

The first call is reported separately from the later ones. It carries JIT
tracing and XLA compilation; the later ones do not. A method with no ``jit``
in it shows a first call barely slower than its steady-state call, and a slow
steady state. That contrast is the signal this script exists to expose.

Every timed call blocks on the result before the clock stops. JAX dispatch is
asynchronous, so a timing that does not block measures the time to enqueue
work, which is close to zero and completely meaningless.

Only ``analyze`` is timed. Design generation (``sample``) is excluded: it is
not where the estimator kernels live.

Peak memory
-----------

On this machine (Apple silicon, CPU backend) ``jax.devices()[0].
memory_stats()`` returns ``None``: the CPU client keeps no allocator
statistics. ``jax.profiler.device_memory_profile()`` returns a gzipped pprof
of *live* buffers at one instant, not a high-water mark, so it cannot answer
"how much did this peak at" either. ``tracemalloc`` only sees allocations
made through the CPython allocator, and both XLA buffers and NumPy arrays
bypass it, so it would systematically under-report the thing we care about.

What is left is the operating system's own high-water mark for the process,
``resource.getrusage(RUSAGE_SELF).ru_maxrss``. It is monotonic within a
process, so it cannot separate two cases that run one after the other -- and
that is the reason each case gets a fresh subprocess. The worker records
``ru_maxrss`` once after imports and data preparation, and once at the end.
The difference is reported as the peak attributable to the analysis.

Be clear about what that number is and is not:

* It is resident set size, in bytes, for the whole process. It captures XLA
  scratch buffers, result arrays, NumPy temporaries and any memory the
  compiler itself used.
* It is a high-water mark, not an integral, and the OS may have counted a
  page that the allocator had already freed back to it lazily.
* It cannot go negative and it cannot fall. A case whose analysis allocates
  less than the setup already did reports ``0``, which means "under the
  noise floor", not "allocated nothing".
* Resolution is a page (16 KiB here), and the true floor is a few MiB of
  allocator jitter. Differences below roughly 4 MiB are not meaningful.

The mark is read a third time, straight after the first call, which splits
the total into ``compile_rss_bytes`` (setup to end of the first call, so it
includes whatever XLA compilation itself allocated) and
``run_extra_rss_bytes`` (anything the later calls added on top). Because the
mark only ever rises, ``run_extra_rss_bytes`` is ``0`` whenever the compile
phase already peaked higher -- which is the common case and means "the steady
-state calls stayed inside the compile-time high-water mark", not "the steady
-state calls allocated nothing". A run that starts materialising a whole (T,
K) batch will push past that mark, and that is the regression the split is
there to catch.

Why the dispatch table is written out by hand
---------------------------------------------

The method list comes from ``jaxgsa._core.registry.methods()``, so a method
added later appears here without editing a list, and is reported as
unhandled if nobody wrote a call for it. The *calls* cannot be derived from
the registry, because the signatures genuinely differ:

* design-based methods take a sample result, given-data methods take ``X``
  and ``Y``;
* ``dgsm`` takes a callable that maps one row, not a matrix of outputs;
* every method has its own accuracy knobs (``order``, ``maxorder``,
  ``n_bins``, ``n_perms``, ``max_centers``), and leaving them at their
  defaults would make the timings hostage to a later change of default.

So the registry decides *which* methods are measured and the table below
decides *how* each one is called.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable
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
from jaxgsa._core.registry import methods as registry_methods
from jaxgsa.benchmarks import ishigami

SEED = 20260819
"""One seed everywhere, so repeats differ only by machine noise."""

DEFAULT_REPEATS = 3
"""Timed calls after the first one. Kept small: the spread matters more than
the tail, and the whole sweep has to finish in minutes."""

N_GIVEN = 1024
"""Rows in the given-data design. Large enough that the estimator kernels,
not the fixed overhead, dominate; small enough that the O(N^2) methods
(hsic, vkoga) stay in seconds."""

N_SOBOL = 256
"""Base sample size for the Saltelli column-swap design: 256 * (D + 2) rows
with D = 3, so 1280 model runs, comparable to the given-data budget."""

N_TRAJECTORIES = 32
"""Morris trajectories, giving 32 * (D + 1) = 128 rows."""

N_PER_CURVE = 257
"""Points per eFAST search curve. The method needs at least 4 M^2 (D-1)+1."""


# --------------------------------------------------------------------------
# Output shapes
# --------------------------------------------------------------------------
#
# The sweep varies one axis only: the shape of Y. The problem is Ishigami
# (D = 3, uniform, independent) in every case, so nothing but the number and
# the layout of output slices changes between rows of the table.
#
# The slice counts are 1, 8, 32, 64, 128 and 128. That spans just over two
# orders of magnitude, which is enough to separate a cost that is flat in the
# slice count (one vmapped kernel) from one that is linear in it (a Python
# loop), while keeping the whole sweep to a few minutes. The ceiling is 128
# and not more because the loop-per-slice methods are linear in it: hsic at
# 256 slices already runs past three minutes, and the shape of its curve is
# clear long before that.
#
# (N, 8, 16) and (N, 16, 8) hold the slice count fixed at 128 and swap the
# axes. That catches a method that loops over T but vmaps over K, or the
# other way round: the two rows should agree, and a method where they do not
# is treating the two output axes differently.


@dataclasses.dataclass(frozen=True)
class Shape:
    """One output layout in the sweep.

    Attributes:
        key: Short identifier used on the command line and in the JSON.
        dims: Trailing dimensions of ``Y`` after the sample axis. Empty for
            a scalar output.
    """

    key: str
    dims: tuple[int, ...]

    @property
    def label(self) -> str:
        """The shape as it is written in the documentation, e.g. ``(N, T, K)``."""
        return "(N" + "".join(f", {d}" for d in self.dims) + ")"

    @property
    def n_slices(self) -> int:
        """Number of independent output slices, the product of ``dims``."""
        return int(np.prod(self.dims)) if self.dims else 1


SHAPES: tuple[Shape, ...] = (
    Shape("scalar", ()),
    Shape("k8", (8,)),
    Shape("k32", (32,)),
    Shape("t16k4", (16, 4)),
    Shape("t8k16", (8, 16)),
    Shape("t16k8", (16, 8)),
)

SHAPES_BY_KEY: dict[str, Shape] = {s.key: s for s in SHAPES}


def widen(y: jax.Array, shape: Shape) -> jax.Array:
    """Widen a scalar model output to the requested trailing shape.

    Each slice is the scalar output scaled and offset by a distinct constant,
    so no two slices are identical and no estimator can short-circuit on a
    degenerate column.

    Args:
        y: Scalar model output, shape ``(N,)``.
        shape: Target layout.

    Returns:
        Array of shape ``(N, *shape.dims)``.
    """
    if not shape.dims:
        return y
    n = shape.n_slices
    scales = 1.0 + 0.5 * jnp.arange(n, dtype=y.dtype) / n
    offsets = jnp.arange(n, dtype=y.dtype)
    flat = y[:, None] * scales[None, :] + offsets[None, :]
    return flat.reshape((y.shape[0], *shape.dims))


# --------------------------------------------------------------------------
# The case under measurement
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Case:
    """Everything one worker needs to time one method on one shape.

    Attributes:
        problem: The jaxgsa problem, always Ishigami here.
        shape: The output layout.
        X: Monte Carlo design for the given-data methods, shape ``(N, D)``.
        Y: Model output on ``X``.
    """

    problem: jaxgsa.Problem
    shape: Shape
    X: jax.Array
    Y: jax.Array


def build_case(shape: Shape) -> Case:
    """Build the inputs for one output shape.

    Args:
        shape: The output layout to measure.

    Returns:
        The prepared case, with ``X`` and ``Y`` already materialised on the
        device so their cost is not charged to the method.
    """
    problem = ishigami.PROBLEM
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, N_GIVEN, seed=SEED))
    Y = widen(ishigami.evaluate(X), shape)
    jax.block_until_ready((X, Y))
    return Case(problem=problem, shape=shape, X=X, Y=Y)


def _point_model(x: jax.Array) -> jax.Array:
    """Ishigami on a single row, for ``dgsm``, which differentiates it."""
    return ishigami.evaluate(x[None, :])[0]


def _shaped_point_model(shape: Shape) -> Callable[[jax.Array], jax.Array]:
    """Return a one-row model whose output has the case's trailing shape.

    Args:
        shape: The output layout.

    Returns:
        Callable mapping a ``(D,)`` row to an array of shape ``shape.dims``.
    """

    def point(x: jax.Array) -> jax.Array:
        return widen(_point_model(x)[None], shape)[0]

    return point


# --------------------------------------------------------------------------
# Dispatch: how each registered method is called
# --------------------------------------------------------------------------
#
# Design-based methods sample once outside the timed region -- the design is
# not the estimator -- and the timed call analyses that fixed design.


def _prep_sobol(case: Case) -> Callable[[], Any]:
    sr = sobol.sample(case.problem, n_samples=N_SOBOL, seed=SEED, verbose=False)
    y = widen(ishigami.evaluate(jnp.asarray(sr.samples)), case.shape)
    jax.block_until_ready(y)
    key = jax.random.key(SEED)
    return lambda: sobol.analyze(sr, y, n_bootstrap=20, key=key)


def _prep_morris(case: Case) -> Callable[[], Any]:
    sr = morris.sample(case.problem, n_trajectories=N_TRAJECTORIES, seed=SEED, verbose=False)
    y = widen(ishigami.evaluate(jnp.asarray(sr.samples)), case.shape)
    jax.block_until_ready(y)
    key = jax.random.key(SEED)
    return lambda: morris.analyze(sr, y, n_bootstrap=20, key=key)


def _prep_efast(case: Case) -> Callable[[], Any]:
    sr = efast.sample(case.problem, n_per_curve=N_PER_CURVE, seed=SEED)
    y = widen(ishigami.evaluate(jnp.asarray(sr.samples)), case.shape)
    jax.block_until_ready(y)
    return lambda: efast.analyze(sr, y)


def _prep_kucherenko(case: Case) -> Callable[[], Any]:
    sr = kucherenko.sample(case.problem, n_samples=N_SOBOL, seed=SEED)
    y = widen(ishigami.evaluate(jnp.asarray(sr.samples)), case.shape)
    jax.block_until_ready(y)
    return lambda: kucherenko.analyze(sr, y)


def _prep_borgonovo(case: Case) -> Callable[[], Any]:
    return lambda: borgonovo.analyze(
        case.problem, case.X, case.Y, n_bootstrap=10, key=jax.random.key(SEED)
    )


def _prep_pawn(case: Case) -> Callable[[], Any]:
    return lambda: pawn.analyze(
        case.problem, case.X, case.Y, n_bins=8, n_bootstrap=10, key=jax.random.key(SEED)
    )


def _prep_hsic(case: Case) -> Callable[[], Any]:
    return lambda: hsic.analyze(case.problem, case.X, case.Y, n_perms=10, seed=SEED)


def _prep_optimal_transport(case: Case) -> Callable[[], Any]:
    return lambda: optimal_transport.analyze(
        case.problem,
        case.X,
        case.Y,
        n_partitions=8,
        n_bootstrap=10,
        dummy=False,
        key=jax.random.key(SEED),
    )


def _prep_pce(case: Case) -> Callable[[], Any]:
    return lambda: pce.analyze(case.problem, case.X, case.Y, order=3)


def _prep_hdmr(case: Case) -> Callable[[], Any]:
    return lambda: hdmr.analyze(case.problem, case.X, case.Y, maxorder=2, maxiter=50)


def _prep_shapley(case: Case) -> Callable[[], Any]:
    return lambda: shapley.analyze(case.problem, case.X, case.Y, backend="pce", order=3)


def _prep_vkoga(case: Case) -> Callable[[], Any]:
    return lambda: vkoga.analyze(
        case.problem,
        case.X,
        case.Y,
        max_centers=32,
        n_folds=3,
        n_outer=64,
        n_inner=32,
        n_variance=512,
        seed=SEED,
    )


def _prep_dgsm(case: Case) -> Callable[[], Any]:
    model = _shaped_point_model(case.shape)
    return lambda: dgsm.analyze(case.problem, model, case.X)


PREPARE: dict[str, Callable[[Case], Callable[[], Any]]] = {
    "borgonovo": _prep_borgonovo,
    "dgsm": _prep_dgsm,
    "efast": _prep_efast,
    "hdmr": _prep_hdmr,
    "hsic": _prep_hsic,
    "kucherenko": _prep_kucherenko,
    "morris": _prep_morris,
    "optimal_transport": _prep_optimal_transport,
    "pawn": _prep_pawn,
    "pce": _prep_pce,
    "shapley": _prep_shapley,
    "sobol": _prep_sobol,
    "vkoga": _prep_vkoga,
}
"""How to build the timed call for each registered method.

Keyed by the registry name. :func:`method_names` checks this table against
the registry, so a method added to jaxgsa without a line here is reported,
not silently skipped.
"""


def method_names() -> tuple[list[str], list[str]]:
    """Split the registry into methods this script can call and ones it cannot.

    Returns:
        ``(known, unhandled)``. ``unhandled`` is non-empty only when jaxgsa
        gained a method and :data:`PREPARE` was not updated.
    """
    registered = list(registry_methods())
    known = [name for name in registered if name in PREPARE]
    unhandled = [name for name in registered if name not in PREPARE]
    return known, unhandled


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def _arrays(value: Any, depth: int = 0) -> list[jax.Array]:
    """Collect every JAX array reachable from a result object.

    Args:
        value: A result object, container, or leaf.
        depth: Recursion guard, so a self-referential field cannot hang the
            benchmark.

    Returns:
        Every :class:`jax.Array` found, in traversal order.
    """
    if depth > 6:
        return []
    if isinstance(value, jax.Array):
        return [value]
    if isinstance(value, jaxgsa.Problem | str | bytes):
        return []
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        found: list[jax.Array] = []
        for field in dataclasses.fields(value):
            found.extend(_arrays(getattr(value, field.name, None), depth + 1))
        return found
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


def timed_call(fn: Callable[[], Any]) -> float:
    """Run ``fn`` once and return how long it took, in seconds.

    Every JAX array in the result is blocked on before the clock stops. JAX
    dispatch is asynchronous: without this the measurement is the time to
    enqueue the computation, not to perform it.

    Args:
        fn: The zero-argument call to time.

    Returns:
        Wall-clock seconds.
    """
    start = time.perf_counter()
    result = fn()
    jax.block_until_ready(_arrays(result))
    return time.perf_counter() - start


def _maxrss_bytes() -> int:
    """Return the process high-water resident set size, in bytes.

    ``ru_maxrss`` is bytes on macOS and kibibytes on Linux, so the platform
    decides the multiplier.

    Returns:
        Peak resident set size of this process so far.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


def measure(method: str, shape: Shape, repeats: int) -> dict[str, Any]:
    """Time one method on one output shape and record its peak memory.

    Must run in a fresh process: the memory number is a per-process
    high-water mark and the compile number assumes an empty JIT cache.

    Args:
        method: Registry name of the method.
        shape: The output layout.
        repeats: Timed calls after the first one.

    Returns:
        A JSON-safe record. ``status`` is ``"ok"``, or ``"raised"`` when the
        method refused this input by design.

        Any other exception is left to propagate and kill the worker. That is
        deliberate: the worker is a subprocess, and :func:`run_sweep` already
        turns a worker that produced no JSON into a ``"failed"`` record
        carrying its stderr. Catching it here as well would only hide the
        traceback, and one broken case still cannot take the sweep down.
    """
    record: dict[str, Any] = {
        "method": method,
        "shape": shape.key,
        "shape_label": shape.label,
        "n_slices": shape.n_slices,
    }
    case = build_case(shape)
    call = PREPARE[method](case)

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
    record.update(
        {
            "status": "ok",
            "first_call_s": first,
            "run_best_s": best,
            "run_median_s": float(np.median(later)) if later else first,
            "run_worst_s": max(later) if later else first,
            "run_times_s": later,
            "compile_s": max(first - best, 0.0),
            "run_per_slice_s": best / shape.n_slices,
            "peak_rss_bytes": max(rss_peak - rss_setup, 0),
            "compile_rss_bytes": max(rss_after_first - rss_setup, 0),
            "run_extra_rss_bytes": max(rss_peak - rss_after_first, 0),
            "setup_rss_bytes": rss_setup,
        }
    )
    return record


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_worker(method: str, shape_key: str, repeats: int) -> int:
    """Worker entry point: measure one case and print the record as JSON.

    Args:
        method: Registry name of the method.
        shape_key: Key of the output shape.
        repeats: Timed calls after the first one.

    Returns:
        Process exit code, always ``0``: a refusal or a crash is data.
    """
    record = measure(method, SHAPES_BY_KEY[shape_key], repeats)
    sys.stdout.write(json.dumps(record))
    return 0


def run_sweep(
    method_list: list[str],
    shapes: tuple[Shape, ...],
    repeats: int,
    timeout: float,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Run every case, one subprocess each, and collect the records.

    Args:
        method_list: Registry names to measure.
        shapes: Output layouts to sweep.
        repeats: Timed calls after the first one, per case.
        timeout: Seconds a single case may take before it is abandoned.
        verbose: Print progress to stderr.

    Returns:
        One record per case, in sweep order.
    """
    records: list[dict[str, Any]] = []
    script = str(Path(__file__).resolve())
    for method in method_list:
        for shape in shapes:
            if verbose:
                print(f"  {method:<18} {shape.label}", file=sys.stderr, flush=True)
            cmd = [
                sys.executable,
                script,
                "--worker",
                method,
                "--worker-shape",
                shape.key,
                "--repeats",
                str(repeats),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                records.append(
                    {
                        "method": method,
                        "shape": shape.key,
                        "shape_label": shape.label,
                        "n_slices": shape.n_slices,
                        "status": "timeout",
                        "exception": f"exceeded {timeout:.0f}s",
                    }
                )
                continue
            try:
                records.append(json.loads(proc.stdout.strip().splitlines()[-1]))
            except (ValueError, IndexError):
                records.append(
                    {
                        "method": method,
                        "shape": shape.key,
                        "shape_label": shape.label,
                        "n_slices": shape.n_slices,
                        "status": "failed",
                        "exception": (proc.stderr or "no output").strip()[-400:],
                    }
                )
    return records


def _fmt_time(seconds: float) -> str:
    """Format a duration compactly in milliseconds or seconds."""
    return f"{seconds * 1e3:.1f} ms" if seconds < 1.0 else f"{seconds:.2f} s"


def _fmt_bytes(count: int) -> str:
    """Format a byte count in MiB, or as a floor marker when it is zero."""
    return "<noise" if count <= 0 else f"{count / 1024**2:.1f} MiB"


def format_table(records: list[dict[str, Any]]) -> str:
    """Render the records as a Markdown table.

    Args:
        records: Records from :func:`run_sweep`.

    Returns:
        Markdown text, one row per case.
    """
    header = (
        "| method | shape | slices | compile | run (best) | spread | per slice "
        "| peak RSS | run extra |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for rec in records:
        if rec.get("status") != "ok":
            lines.append(
                f"| {rec['method']} | {rec['shape_label']} | {rec['n_slices']} | "
                f"{rec.get('status')} | {str(rec.get('exception', ''))[:60]} | | | | |\n"
            )
            continue
        best = float(rec["run_best_s"])
        worst = float(rec["run_worst_s"])
        spread = f"{(worst - best) / best * 100:.0f}%" if best > 0 else "n/a"
        lines.append(
            f"| {rec['method']} | {rec['shape_label']} | {rec['n_slices']} | "
            f"{_fmt_time(float(rec['compile_s']))} | {_fmt_time(best)} | {spread} | "
            f"{_fmt_time(float(rec['run_per_slice_s']))} | "
            f"{_fmt_bytes(int(rec['peak_rss_bytes']))} | "
            f"{_fmt_bytes(int(rec['run_extra_rss_bytes']))} |\n"
        )
    return "".join(lines)


def build_header() -> dict[str, Any]:
    """Return the environment header.

    These timings are not portable. A later run is only comparable to this
    one if the machine, the backend and the JAX version all match.

    Returns:
        A JSON-safe mapping describing where the numbers came from.
    """
    try:
        version = importlib.metadata.version("jaxgsa")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "jaxgsa_version": version,
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "jax_default_backend": jax.default_backend(),
        "jax_device_count": jax.device_count(),
        "x64_enabled": bool(getattr(jax.config, "jax_enable_x64")),
        "memory_metric": "ru_maxrss delta (process high-water RSS, bytes)",
        "memory_stats_available": jax.devices()[0].memory_stats() is not None,
        "seed": SEED,
        "n_given": N_GIVEN,
        "n_sobol": N_SOBOL,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the sweep, or one worker case, and write the results.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Benchmark every jaxgsa method.")
    parser.add_argument("--out", type=Path, default=None, help="write results JSON here")
    parser.add_argument("--markdown", type=Path, default=None, help="write the table here")
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS, help="timed calls after the first"
    )
    parser.add_argument("--methods", default="", help="comma-separated subset of methods")
    parser.add_argument("--shapes", default="", help="comma-separated subset of shape keys")
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds allowed per case")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-shape", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker is not None:
        if args.worker_shape is None:
            parser.error("--worker needs --worker-shape")
        return run_worker(args.worker, args.worker_shape, args.repeats)

    known, unhandled = method_names()
    if args.methods:
        wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in PREPARE]
        if unknown:
            parser.error(f"unknown methods: {', '.join(unknown)}")
        known = wanted
    shapes = SHAPES
    if args.shapes:
        keys = [s.strip() for s in args.shapes.split(",") if s.strip()]
        missing = [k for k in keys if k not in SHAPES_BY_KEY]
        if missing:
            parser.error(f"unknown shape keys: {', '.join(missing)}")
        shapes = tuple(SHAPES_BY_KEY[k] for k in keys)

    verbose = not args.quiet
    if unhandled and verbose:
        print(
            f"registry has methods with no call in PREPARE: {', '.join(unhandled)}",
            file=sys.stderr,
        )

    started = time.perf_counter()
    records = run_sweep(known, shapes, args.repeats, args.timeout, verbose)
    elapsed = time.perf_counter() - started

    table = format_table(records)
    print(table)
    if verbose:
        print(f"sweep took {elapsed:.0f} s", file=sys.stderr)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table)
    if args.out is not None:
        doc = {
            "header": build_header() | {"unhandled_methods": unhandled, "sweep_seconds": elapsed},
            "records": records,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        if verbose:
            print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
