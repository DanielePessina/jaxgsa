"""Benchmark: jaxgsa vs SALib — correctness + timing across output shapes.

Correctness: Ishigami function (D=3) with known analytical solutions.
Timing: coupled oscillators model with varying T (timepoints) and K (outputs).
"""

from __future__ import annotations

import sys
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from SALib.analyze import hdmr as salib_hdmr
from SALib.analyze import sobol as salib_sobol
from SALib.analyze.sobol import first_order as salib_first_order
from SALib.analyze.sobol import second_order as salib_second_order
from SALib.analyze.sobol import separate_output_values as salib_separate_output_values
from SALib.analyze.sobol import total_order as salib_total_order

import jaxgsa
from jaxgsa.benchmarks.ishigami import ANALYTICAL_S1 as ISHIGAMI_ANALYTICAL_S1
from jaxgsa.benchmarks.ishigami import ANALYTICAL_ST as ISHIGAMI_ANALYTICAL_ST
from jaxgsa.benchmarks.ishigami import PROBLEM as ISHIGAMI_PROBLEM
from jaxgsa.benchmarks.ishigami import evaluate as ishigami_evaluate

# ---------------------------------------------------------------------------
# Benchmark model: coupled damped oscillators (D=5)
# ---------------------------------------------------------------------------

D_PARAMS = 5

BENCH_PROBLEM = jaxgsa.Problem.from_dict(
    {
        "amplitude": (0.1, 2.0),
        "frequency": (0.5, 5.0),
        "damping": (0.01, 1.0),
        "coupling": (0.1, 3.0),
        "drift": (0.0, 1.0),
    }
)


def coupled_oscillators(X: jax.Array, T: int, K: int) -> jax.Array:
    """Evaluate benchmark model -> shape depends on (T, K).

    Returns:
        (N,)       if T=1, K=1
        (N, K)     if T=1, K>1
        (N, T, K)  if T>1
    """
    x0 = X[:, 0, None]
    x1 = X[:, 1, None]
    x2 = X[:, 2, None]
    x3 = X[:, 3, None]
    x4 = X[:, 4, None]

    t = jnp.linspace(0.1, 5.0, T)[None, :]  # (1, T) — start at 0.1 to avoid t=0 degeneracy

    y0 = x0 * jnp.sin(2 * jnp.pi * x1 * t) * jnp.exp(-x2 * t)
    y1 = x1 * jnp.cos(2 * jnp.pi * x0 * t) + x4 * t**2
    y2 = x3 * jnp.sin(x4 * 10 * t) + x0 * x2
    y3 = (x3 + x4) * jnp.exp(-x0 * t) * jnp.sin(2 * jnp.pi * x1 * t)
    y4 = x0 * x3 * jnp.cos(x1 * t) + x2 * t
    y5 = x4 * jnp.sin(x0 * t) * jnp.exp(-x3 * t)

    all_outputs = [y0, y1, y2, y3, y4, y5]
    Y = jnp.stack(all_outputs[:K], axis=-1)  # (N, T, K)

    if T == 1 and K == 1:
        return Y[:, 0, 0]  # (N,)
    if T == 1:
        return Y[:, 0, :]  # (N, K)
    return Y  # (N, T, K)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def jaxgsa_problem_to_salib(problem: jaxgsa.Problem) -> dict:
    bounds = problem.bounds
    if bounds is None:
        raise ValueError("SALib conversion requires a uniform-only problem with finite bounds")
    return {
        "num_vars": problem.num_vars,
        "names": list(problem.names),
        "bounds": [list(b) for b in bounds],
    }


def expand_sobol_outputs(sr, Y) -> np.ndarray:
    """Expand unique jaxgsa outputs back into Saltelli row order for SALib."""
    Y_np = np.asarray(Y)
    return Y_np[sr.expanded_to_unique]


def salib_sobol_point_estimates(
    salib_problem: dict,
    Y_np: np.ndarray,
    *,
    calc_second_order: bool,
) -> dict[str, np.ndarray]:
    """Compute SALib Sobol point estimates without confidence-interval resampling."""
    D = salib_problem["num_vars"]
    step = 2 * D + 2 if calc_second_order else D + 2
    if Y_np.size % step != 0:
        raise RuntimeError("Incorrect number of samples for SALib Sobol point-estimate path.")

    Y_norm = (Y_np - Y_np.mean()) / Y_np.std()
    N = Y_norm.size // step
    A, B, AB, BA = salib_separate_output_values(Y_norm, D, N, calc_second_order)

    S1 = np.empty(D, dtype=float)
    ST = np.empty(D, dtype=float)
    for j in range(D):
        S1[j] = np.asarray(salib_first_order(A, AB[:, j], B)).item()
        ST[j] = np.asarray(salib_total_order(A, AB[:, j], B)).item()

    result: dict[str, np.ndarray] = {"S1": S1, "ST": ST}
    if calc_second_order:
        assert BA is not None
        S2 = np.full((D, D), np.nan, dtype=float)
        for j in range(D):
            for k in range(j + 1, D):
                value = np.asarray(salib_second_order(A, AB[:, j], AB[:, k], BA[:, j], B)).item()
                S2[j, k] = value
                S2[k, j] = value
        result["S2"] = S2
    return result


# ---------------------------------------------------------------------------
# Correctness (Ishigami)
# ---------------------------------------------------------------------------


def benchmark_correctness() -> bool:
    """Validate jaxgsa analyze and analyze_hdmr against SALib and analytical values."""
    base_n = 1024
    D = ISHIGAMI_PROBLEM.num_vars
    salib_problem = jaxgsa_problem_to_salib(ISHIGAMI_PROBLEM)
    analytical_S1 = np.array(ISHIGAMI_ANALYTICAL_S1)
    analytical_ST = np.array(ISHIGAMI_ANALYTICAL_ST)

    rows: list[tuple[str, str, float, float, float, bool, bool]] = []

    # --- Sobol: jaxgsa vs SALib (shared Saltelli samples) ---
    n_samples = 2**14 * 8
    sr = jaxgsa.sobol.sample(ISHIGAMI_PROBLEM, n_samples, seed=42, calc_second_order=True)
    Y_jax = ishigami_evaluate(jnp.asarray(sr.samples))
    Y_np = expand_sobol_outputs(sr, Y_jax)

    jaxgsa_sobol = jaxgsa.sobol.analyze(sr, Y_jax, verbose=False)
    salib_sobol_result = salib_sobol_point_estimates(
        salib_problem,
        Y_np,
        calc_second_order=True,
    )

    g_S1 = np.asarray(jaxgsa_sobol.S1)
    g_ST = np.asarray(jaxgsa_sobol.ST)
    s_S1 = salib_sobol_result["S1"]
    s_ST = salib_sobol_result["ST"]

    for i in range(D):
        match = bool(np.abs(g_S1[i] - s_S1[i]) < 1e-5)
        rows.append(("analyze (S2)", f"S1[{i}]", g_S1[i], s_S1[i], analytical_S1[i], match, True))
    for i in range(D):
        match = bool(np.abs(g_ST[i] - s_ST[i]) < 1e-5)
        rows.append(("analyze (S2)", f"ST[{i}]", g_ST[i], s_ST[i], analytical_ST[i], match, True))

    # S2 check (upper triangle). jaxgsa averages both S2 triangles to cut
    # Monte Carlo noise, a deliberate departure from SALib's raw upper
    # triangle (see docs/api/sobol.md). So this row is not part of the
    # pass/fail gate; it is printed for context, like the HDMR rows below.
    mask = np.triu(np.ones((D, D), dtype=bool), k=1)
    g_S2 = np.asarray(jaxgsa_sobol.S2)[mask]
    s_S2 = salib_sobol_result["S2"][mask]
    for idx, (g, s) in enumerate(zip(g_S2, s_S2)):
        close = bool(np.abs(g - s) < 5e-2)
        rows.append(("analyze (S2)", f"S2[{idx}]", float(g), float(s), float("nan"), close, False))

    # --- Sobol jaxgsa vs analytical ---
    for i in range(D):
        if analytical_S1[i] == 0.0:
            ok = abs(g_S1[i]) < 0.05
        else:
            ok = abs(g_S1[i] - analytical_S1[i]) / abs(analytical_S1[i]) < 0.15
        rows.append(
            (
                "analyze vs analyt.",
                f"S1[{i}]",
                g_S1[i],
                float("nan"),
                analytical_S1[i],
                bool(ok),
                True,
            )
        )
    for i in range(D):
        ok = abs(g_ST[i] - analytical_ST[i]) / max(abs(analytical_ST[i]), 0.01) < 0.15
        rows.append(
            (
                "analyze vs analyt.",
                f"ST[{i}]",
                g_ST[i],
                float("nan"),
                analytical_ST[i],
                bool(ok),
                True,
            )
        )

    # --- HDMR: jaxgsa vs SALib ---
    rng = np.random.default_rng(42)
    bounds = np.array(ISHIGAMI_PROBLEM.bounds)
    X_np = rng.uniform(bounds[:, 0], bounds[:, 1], size=(base_n, D))
    Y_hdmr_np = np.asarray(ishigami_evaluate(jnp.asarray(X_np)))
    X_jax_hdmr = jnp.asarray(X_np)
    Y_hdmr_jax = jnp.asarray(Y_hdmr_np)

    jaxgsa_hdmr = jaxgsa.hdmr.analyze(
        ISHIGAMI_PROBLEM, X_jax_hdmr, Y_hdmr_jax, maxorder=2, m=2, verbose=False
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        salib_hdmr_result = salib_hdmr.analyze(
            salib_problem,
            X_np,
            Y_hdmr_np,
            maxorder=2,
            maxiter=100,
            print_to_console=False,
        )

    g_hdmr_S1 = np.asarray(jaxgsa_hdmr.S1)
    g_hdmr_ST = np.asarray(jaxgsa_hdmr.ST)
    s_hdmr_S1 = np.array(salib_hdmr_result["Sa"][:D])
    s_hdmr_ST = np.array(salib_hdmr_result["ST"][:D])

    for i in range(D):
        match = bool(np.abs(g_hdmr_S1[i] - s_hdmr_S1[i]) < 0.05)
        rows.append(
            ("analyze_hdmr", f"S1[{i}]", g_hdmr_S1[i], s_hdmr_S1[i], analytical_S1[i], match, True)
        )
    for i in range(D):
        match = bool(np.abs(g_hdmr_ST[i] - s_hdmr_ST[i]) < 0.05)
        rows.append(
            ("analyze_hdmr", f"ST[{i}]", g_hdmr_ST[i], s_hdmr_ST[i], analytical_ST[i], match, True)
        )

    # --- HDMR jaxgsa vs analytical (informational only; RS-HDMR is approximate) ---
    for i in range(D):
        if analytical_S1[i] == 0.0:
            ok = abs(g_hdmr_S1[i]) < 0.1
        else:
            ok = abs(g_hdmr_S1[i] - analytical_S1[i]) / abs(analytical_S1[i]) < 0.30
        rows.append(
            (
                "hdmr vs analyt.",
                f"S1[{i}]",
                g_hdmr_S1[i],
                float("nan"),
                analytical_S1[i],
                bool(ok),
                False,
            )
        )
    for i in range(D):
        ok = abs(g_hdmr_ST[i] - analytical_ST[i]) / max(abs(analytical_ST[i]), 0.01) < 0.25
        rows.append(
            (
                "hdmr vs analyt.",
                f"ST[{i}]",
                g_hdmr_ST[i],
                float("nan"),
                analytical_ST[i],
                bool(ok),
                False,
            )
        )

    # --- Print correctness table ---
    all_pass = True
    print("=" * 78)
    print("CORRECTNESS CHECK  (Ishigami)")
    print("=" * 78)
    print(
        f"{'Method':<22} {'Index':<8} {'jaxgsa':>10} {'SALib':>10} {'analytical':>10} {'match':>6}"
    )
    print("-" * 78)
    print(
        "NOTE: overall PASS/FAIL is driven by jaxgsa-vs-SALib S1/ST agreement "
        "plus Sobol-vs-analytical rows."
    )
    print(
        "      S2 and HDMR-vs-analytical rows are reported for context only (see comments above)."
    )
    print("-" * 78)
    for method, index, g_val, s_val, a_val, ok, gate in rows:
        if gate:
            all_pass &= ok
        g_str = f"{g_val:>10.4f}"
        s_str = f"{s_val:>10.4f}" if not np.isnan(s_val) else f"{'---':>10}"
        a_str = f"{a_val:>10.4f}" if not np.isnan(a_val) else f"{'---':>10}"
        if gate:
            tag = "PASS" if ok else "FAIL"
        else:
            tag = "INFO" if ok else "WARN"
        print(f"{method:<22} {index:<8} {g_str} {s_str} {a_str} {tag:>6}")

    return all_pass


# ---------------------------------------------------------------------------
# Timing benchmark
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("1x1", 1, 1),
    ("1x6", 1, 6),
    ("50x1", 50, 1),
    ("50x6", 50, 6),
]


N_TIMING_ITERS = 5
# SALib's HDMR path costs ~1 s per output slice, so a smaller best-of-N keeps
# the total run bounded; its run-to-run variance is negligible next to the
# multi-second mean, unlike the ms-scale jaxgsa runs that N_TIMING_ITERS de-noises.
N_SALIB_HDMR_ITERS = 2
SOBOL_RESAMPLE_COUNTS = (0, 300)
SOBOL_BOOTSTRAP_SEED = 123


def _block_sobol_result(result) -> None:
    """Synchronize all computed Sobol arrays before stopping the timer."""
    jax.block_until_ready(result.S1)
    jax.block_until_ready(result.ST)
    if result.S2 is not None:
        jax.block_until_ready(result.S2)
    if result.S1_conf is not None:
        jax.block_until_ready(result.S1_conf)
        jax.block_until_ready(result.ST_conf)
        if result.S2_conf is not None:
            jax.block_until_ready(result.S2_conf)


def _block_hdmr_result(result) -> None:
    """Synchronize the arrays materialized by analyze_hdmr."""
    jax.block_until_ready(result.Sa)
    jax.block_until_ready(result.Sb)
    jax.block_until_ready(result.S)
    jax.block_until_ready(result.ST)
    if result.rmse is not None:
        jax.block_until_ready(result.rmse)


def _best_of_n(fn, block_result, iters: int = N_TIMING_ITERS) -> float:
    """Return the best wall time across ``iters`` identical runs.

    Best-of-N reporting de-noises OS-scheduler jitter, which dominates the
    ms-scale jaxgsa runs. Slow (multi-second) paths may pass a smaller ``iters``
    to keep the total benchmark runtime bounded.
    """
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn()
        block_result(result)
        best = min(best, time.perf_counter() - t0)
    return best


def _time_jaxgsa_sobol(sr, Y_jax, *, n_bootstrap: int) -> float:
    """Time jaxgsa.sobol.analyze for either point estimates or bootstrap CIs."""
    key = jax.random.key(SOBOL_BOOTSTRAP_SEED) if n_bootstrap > 0 else None
    return _best_of_n(
        lambda: jaxgsa.sobol.analyze(sr, Y_jax, n_bootstrap=n_bootstrap, key=key, verbose=False),
        _block_sobol_result,
    )


def _salib_sobol_point_estimate_slices(
    salib_problem,
    Y_np,
    calc_second_order: bool,
    T: int,
    K: int,
) -> None:
    """Run SALib Sobol point estimates across all output slices."""
    if T == 1 and K == 1:
        salib_sobol_point_estimates(salib_problem, Y_np, calc_second_order=calc_second_order)
    elif T == 1:
        for k in range(K):
            salib_sobol_point_estimates(
                salib_problem,
                Y_np[:, k],
                calc_second_order=calc_second_order,
            )
    else:
        for t_idx in range(T):
            for k in range(K):
                salib_sobol_point_estimates(
                    salib_problem,
                    Y_np[:, t_idx, k],
                    calc_second_order=calc_second_order,
                )


def _salib_sobol_bootstrap_slices(
    salib_problem,
    Y_np,
    *,
    calc_second_order: bool,
    T: int,
    K: int,
    n_bootstrap: int,
) -> None:
    """Run SALib Sobol analysis with confidence-interval resampling."""
    kwargs = {
        "calc_second_order": calc_second_order,
        "num_resamples": n_bootstrap,
        "print_to_console": False,
        "seed": SOBOL_BOOTSTRAP_SEED,
    }
    if T == 1 and K == 1:
        salib_sobol.analyze(salib_problem, Y_np, **kwargs)
    elif T == 1:
        for k in range(K):
            salib_sobol.analyze(salib_problem, Y_np[:, k], **kwargs)
    else:
        for t_idx in range(T):
            for k in range(K):
                salib_sobol.analyze(salib_problem, Y_np[:, t_idx, k], **kwargs)


def _time_salib_sobol(
    salib_problem,
    Y_np,
    *,
    calc_second_order: bool,
    T: int,
    K: int,
    n_bootstrap: int,
) -> float:
    """Time SALib Sobol analysis with symmetric best-of-N timing."""
    if n_bootstrap == 0:
        return _best_of_n(
            lambda: _salib_sobol_point_estimate_slices(
                salib_problem,
                Y_np,
                calc_second_order,
                T,
                K,
            ),
            lambda _unused: None,
        )
    return _best_of_n(
        lambda: _salib_sobol_bootstrap_slices(
            salib_problem,
            Y_np,
            calc_second_order=calc_second_order,
            T=T,
            K=K,
            n_bootstrap=n_bootstrap,
        ),
        lambda _unused: None,
    )


def _time_jaxgsa_hdmr(problem, X_jax, Y_jax) -> float:
    """Time jaxgsa.analyze_hdmr, best of N_TIMING_ITERS."""
    return _best_of_n(
        lambda: jaxgsa.hdmr.analyze(problem, X_jax, Y_jax, maxorder=2, m=2, verbose=False),
        _block_hdmr_result,
    )


def _salib_hdmr_slices(salib_problem, X_np, Y_np, T: int, K: int) -> None:
    """Run SALib hdmr.analyze across all T*K output slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if T == 1 and K == 1:
            salib_hdmr.analyze(
                salib_problem, X_np, Y_np, maxorder=2, maxiter=100, print_to_console=False
            )
        elif T == 1:
            for k in range(K):
                salib_hdmr.analyze(
                    salib_problem,
                    X_np,
                    Y_np[:, k],
                    maxorder=2,
                    maxiter=100,
                    print_to_console=False,
                )
        else:
            for t_idx in range(T):
                for k_idx in range(K):
                    salib_hdmr.analyze(
                        salib_problem,
                        X_np,
                        Y_np[:, t_idx, k_idx],
                        maxorder=2,
                        maxiter=100,
                        print_to_console=False,
                    )


def _time_salib_hdmr(salib_problem, X_np, Y_np, T: int, K: int) -> float:
    """Time SALib hdmr.analyze with best-of-N timing symmetric to the jaxgsa path.

    SALib is pure NumPy/SciPy (no JIT, so no warmup is needed). Uses the smaller
    ``N_SALIB_HDMR_ITERS`` because each HDMR slice costs ~1 s and its run-to-run
    variance is tiny relative to the mean.
    """
    return _best_of_n(
        lambda: _salib_hdmr_slices(salib_problem, X_np, Y_np, T, K),
        lambda _unused: None,
        iters=N_SALIB_HDMR_ITERS,
    )


def _print_timing_table(title: str, rows, *, method_width: int) -> None:
    """Print one jaxgsa-vs-SALib timing table.

    Args:
        title: Header line printed above the table.
        rows: Iterable of ``(scenario, method, jaxgsa_ms, salib_ms, speedup)``.
        method_width: Column width for the method label (Sobol labels are wider
            than HDMR's). The separator length is derived from the header so the
            two never drift out of sync.
    """
    header = (
        f"{'Scenario (TxK)':<16} {'Method':<{method_width}} "
        f"{'jaxgsa (ms)':>12} {'SALib (ms)':>12} {'speedup':>10}"
    )
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for scenario, method, g_ms, s_ms, sp in rows:
        print(
            f"{scenario:<16} {method:<{method_width}} {g_ms:>10.1f}   {s_ms:>10.1f}   {sp:>8.1f}x"
        )


def benchmark_timing(base_n: int = 1024) -> None:
    """Run timing benchmarks with separate Sobol tables for 0 and 500 bootstraps."""
    D = D_PARAMS
    if base_n < 300:
        raise ValueError(
            "base_n must be >= 300 because analyze_hdmr requires at least 300 samples"
        )
    salib_problem = jaxgsa_problem_to_salib(BENCH_PROBLEM)
    bounds = np.array(BENCH_PROBLEM.bounds)

    print(f"\n{'=' * 78}")
    print("TIMING BENCHMARK — coupled oscillators")
    print(f"  D={D}, base_n={base_n}")
    print("=" * 78)

    scenario_sobol_data: dict[
        tuple[str, int, int, bool],
        tuple[jaxgsa.sobol.SobolSamples, jax.Array, np.ndarray],
    ] = {}
    scenario_hdmr_data: dict[
        tuple[str, int, int],
        tuple[np.ndarray, jax.Array, jax.Array, np.ndarray],
    ] = {}

    for scenario_label, T, K in SCENARIOS:
        for calc_s2 in (False, True):
            step = (2 * D + 2) if calc_s2 else (D + 2)
            sr = jaxgsa.sobol.sample(
                BENCH_PROBLEM,
                base_n * step,
                seed=1,
                calc_second_order=calc_s2,
                verbose=False,
            )
            Y_jax = coupled_oscillators(jnp.asarray(sr.samples), T=T, K=K)
            jax.block_until_ready(Y_jax)
            Y_salib = expand_sobol_outputs(sr, Y_jax)
            scenario_sobol_data[(scenario_label, T, K, calc_s2)] = (sr, Y_jax, Y_salib)

        rng = np.random.default_rng(42)
        X_hdmr_np = rng.uniform(bounds[:, 0], bounds[:, 1], size=(base_n, D))
        X_hdmr_jax = jnp.asarray(X_hdmr_np)
        Y_hdmr_jax = coupled_oscillators(X_hdmr_jax, T=T, K=K)
        jax.block_until_ready(Y_hdmr_jax)
        Y_hdmr_np = np.asarray(Y_hdmr_jax)
        scenario_hdmr_data[(scenario_label, T, K)] = (X_hdmr_np, X_hdmr_jax, Y_hdmr_jax, Y_hdmr_np)

    # --- JIT warmup for jaxgsa (exact timing shapes) ---
    # Pre-compile every kernel the timing loop reuses, so the reported numbers are
    # post-JIT steady-state and the one-off XLA compile is excluded.
    print("\nWarming up jaxgsa JIT ...", end=" ", flush=True)
    for scenario_label, T, K in SCENARIOS:
        for calc_s2 in (False, True):
            sr, Y_jax, _ = scenario_sobol_data[(scenario_label, T, K, calc_s2)]
            for n_bootstrap in SOBOL_RESAMPLE_COUNTS:
                key = jax.random.key(SOBOL_BOOTSTRAP_SEED) if n_bootstrap > 0 else None
                _block_sobol_result(
                    jaxgsa.sobol.analyze(
                        sr, Y_jax, n_bootstrap=n_bootstrap, key=key, verbose=False
                    )
                )

        _, X_hdmr_jax, Y_hdmr_jax, _ = scenario_hdmr_data[(scenario_label, T, K)]
        _block_hdmr_result(
            jaxgsa.hdmr.analyze(
                BENCH_PROBLEM, X_hdmr_jax, Y_hdmr_jax, maxorder=2, m=2, verbose=False
            )
        )
    print("done.")

    # --- Run all scenarios ---
    sobol_rows: dict[int, list[tuple[str, str, float, float, float]]] = {
        n_bootstrap: [] for n_bootstrap in SOBOL_RESAMPLE_COUNTS
    }
    hdmr_rows: list[tuple[str, str, float, float, float]] = []

    for scenario_label, T, K in SCENARIOS:
        print(f"\nScenario {scenario_label} (T={T}, K={K}) ...", flush=True)

        for calc_s2, method_label in [(False, "analyze (no S2)"), (True, "analyze (S2)")]:
            sr, Y_jax, Y_salib = scenario_sobol_data[(scenario_label, T, K, calc_s2)]
            for n_bootstrap in SOBOL_RESAMPLE_COUNTS:
                if n_bootstrap == 0:
                    bootstrap_label = "no bootstrap"
                else:
                    bootstrap_label = f"{n_bootstrap} bootstrap"
                g_time = _time_jaxgsa_sobol(sr, Y_jax, n_bootstrap=n_bootstrap)
                s_time = _time_salib_sobol(
                    salib_problem,
                    Y_salib,
                    calc_second_order=calc_s2,
                    T=T,
                    K=K,
                    n_bootstrap=n_bootstrap,
                )
                speedup = s_time / g_time if g_time > 0 else float("inf")
                sobol_rows[n_bootstrap].append(
                    (
                        scenario_label,
                        f"{method_label}, {bootstrap_label}",
                        g_time * 1e3,
                        s_time * 1e3,
                        speedup,
                    )
                )

        # -- HDMR: shared (X, Y) --
        X_hdmr_np, X_hdmr_jax, Y_hdmr_jax, Y_hdmr_np = scenario_hdmr_data[(scenario_label, T, K)]

        g_time = _time_jaxgsa_hdmr(BENCH_PROBLEM, X_hdmr_jax, Y_hdmr_jax)
        s_time = _time_salib_hdmr(salib_problem, X_hdmr_np, Y_hdmr_np, T, K)
        speedup = s_time / g_time if g_time > 0 else float("inf")
        hdmr_rows.append((scenario_label, "analyze_hdmr", g_time * 1e3, s_time * 1e3, speedup))

    # --- Print timing tables (post-JIT steady-state; one-off compile excluded) ---
    for n_bootstrap in SOBOL_RESAMPLE_COUNTS:
        label = "NO BOOTSTRAP" if n_bootstrap == 0 else f"{n_bootstrap} BOOTSTRAPS"
        _print_timing_table(f"SOBOL TIMING — {label}", sobol_rows[n_bootstrap], method_width=30)

    _print_timing_table("HDMR TIMING", hdmr_rows, method_width=20)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    correct = benchmark_correctness()
    benchmark_timing(base_n=1024)

    print("\n" + "=" * 78)
    if correct:
        print("ALL CORRECTNESS CHECKS PASSED")
        return 0
    else:
        print("SOME CORRECTNESS CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
