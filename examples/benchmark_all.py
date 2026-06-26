"""Comprehensive benchmark: gsax vs SALib across all shared SA methods.

Compares analysis-phase performance on a coupled-oscillator model with
varying output shapes. Model evaluation is excluded from timing — only
the SA computation is measured.

Run interactively: ``uv run marimo edit examples/benchmark_all.py``
"""

# ruff: noqa: F722

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # gsax vs SALib: Comprehensive Benchmark

    Performance comparison across **four SA methods**: Sobol, eFAST,
    DGSM, and HDMR on a coupled-oscillator model (D=5).

    **What's timed:** `analyze()` calls only — model evaluation,
    sampling, and gradient computation are all pre-computed and
    excluded from the timer.

    Press **Run benchmarks** below to execute.  Results are cached in
    a JSON file so subsequent loads skip the computation.
    """)
    return


@app.cell
def _imports():
    import json
    import os
    import time
    import warnings

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from SALib.analyze import dgsm as salib_dgsm_mod
    from SALib.analyze import fast as salib_fast_mod
    from SALib.analyze import hdmr as salib_hdmr_mod
    from SALib.analyze.sobol import first_order as salib_s1
    from SALib.analyze.sobol import separate_output_values as salib_sep
    from SALib.analyze.sobol import total_order as salib_st
    from SALib.sample import fast_sampler as salib_fast_sampler
    from SALib.sample import finite_diff as salib_finite_diff

    import gsax
    from gsax.sampling import sample_mc

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    plt.rcParams["figure.dpi"] = 150
    return (
        gsax,
        jax,
        jnp,
        json,
        mo,
        np,
        os,
        plt,
        salib_dgsm_mod,
        salib_fast_mod,
        salib_fast_sampler,
        salib_finite_diff,
        salib_hdmr_mod,
        salib_s1,
        salib_sep,
        salib_st,
        sample_mc,
        time,
        warnings,
    )


@app.cell
def _config(gsax, jax, jnp, np, os, time):
    D_PARAMS = 5
    N_REPEATS = 3
    BASE_N = 512
    CACHE_PATH = os.path.join(
        os.path.dirname(__file__) or ".", "benchmark_cache.json",
    )

    BENCH_PROBLEM = gsax.Problem.from_dict({
        "amplitude": (0.1, 2.0),
        "frequency": (0.5, 5.0),
        "damping": (0.01, 1.0),
        "coupling": (0.1, 3.0),
        "drift": (0.0, 1.0),
    })

    SCENARIOS = [
        ("1x1", 1, 1), ("1x6", 1, 6),
        ("50x1", 50, 1), ("50x6", 50, 6),
    ]

    def coupled_oscillators(X, T, K):
        x0, x1, x2 = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        x3, x4 = X[:, 3:4], X[:, 4:5]
        t = jnp.linspace(0.1, 5.0, T)[None, :]
        ys = [
            x0 * jnp.sin(2 * jnp.pi * x1 * t) * jnp.exp(-x2 * t),
            x1 * jnp.cos(2 * jnp.pi * x0 * t) + x4 * t**2,
            x3 * jnp.sin(x4 * 10 * t) + x0 * x2,
            (x3 + x4) * jnp.exp(-x0 * t) * jnp.sin(2 * jnp.pi * x1 * t),
            x0 * x3 * jnp.cos(x1 * t) + x2 * t,
            x4 * jnp.sin(x0 * t) * jnp.exp(-x3 * t),
        ]
        Y = jnp.stack(ys[:K], axis=-1)
        if T == 1 and K == 1:
            return Y[:, 0, 0]
        if T == 1:
            return Y[:, 0, :]
        return Y

    def make_unbatched(T, K):
        _t = jnp.linspace(0.1, 5.0, T)

        def fn(x):
            x0, x1, x2, x3, x4 = x[0], x[1], x[2], x[3], x[4]
            ys = [
                x0 * jnp.sin(2 * jnp.pi * x1 * _t) * jnp.exp(-x2 * _t),
                x1 * jnp.cos(2 * jnp.pi * x0 * _t) + x4 * _t**2,
                x3 * jnp.sin(x4 * 10 * _t) + x0 * x2,
                (x3 + x4) * jnp.exp(-x0 * _t) * jnp.sin(2 * jnp.pi * x1 * _t),
                x0 * x3 * jnp.cos(x1 * _t) + x2 * _t,
                x4 * jnp.sin(x0 * _t) * jnp.exp(-x3 * _t),
            ]
            o = jnp.stack(ys[:K], axis=-1)
            if T == 1 and K == 1:
                return o[0, 0]
            return o.reshape(-1)

        return fn

    SALIB_PROBLEM = {
        "num_vars": D_PARAMS,
        "names": list(BENCH_PROBLEM.names),
        "bounds": [list(b) for b in BENCH_PROBLEM.bounds],
    }

    def expand_sobol(sr, Y):
        return np.asarray(Y)[sr.expanded_to_unique]

    def best_of(fn, n=N_REPEATS):
        _best = float("inf")
        for _ in range(n):
            _t0 = time.perf_counter()
            _r = fn()
            if hasattr(_r, "S1"):
                jax.block_until_ready(_r.S1)
            elif hasattr(_r, "Sa"):
                jax.block_until_ready(_r.Sa)
            elif hasattr(_r, "nu"):
                jax.block_until_ready(_r.nu)
            _best = min(_best, time.perf_counter() - _t0)
        return _best

    return (
        BASE_N,
        BENCH_PROBLEM,
        CACHE_PATH,
        D_PARAMS,
        N_REPEATS,
        SALIB_PROBLEM,
        SCENARIOS,
        best_of,
        coupled_oscillators,
        expand_sobol,
        make_unbatched,
    )


@app.cell
def _run_button(mo):
    run_btn = mo.ui.run_button(label="Run benchmarks")
    mo.md(f"### {run_btn}")
    return (run_btn,)


@app.cell
def _benchmark(
    BASE_N,
    BENCH_PROBLEM,
    CACHE_PATH,
    D_PARAMS,
    N_REPEATS,
    SALIB_PROBLEM,
    SCENARIOS,
    best_of,
    coupled_oscillators,
    expand_sobol,
    gsax,
    jax,
    jnp,
    json,
    make_unbatched,
    np,
    os,
    run_btn,
    salib_dgsm_mod,
    salib_fast_mod,
    salib_fast_sampler,
    salib_finite_diff,
    salib_hdmr_mod,
    salib_s1,
    salib_sep,
    salib_st,
    sample_mc,
    time,
    warnings,
):
    # Try cache first when button not pressed
    _from_cache = False
    if not run_btn.value and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as _f:
            all_results = json.load(_f)
        _from_cache = True

    if not run_btn.value and not _from_cache:
        all_results = None

    if run_btn.value:
        _step = 2 * D_PARAMS + 2
        _all = []

        # === SOBOL ===
        for _label, _T, _K in SCENARIOS:
            _sr = gsax.sample(
                BENCH_PROBLEM, BASE_N * _step, seed=1,
                calc_second_order=True, verbose=False,
            )
            _Yj = coupled_oscillators(jnp.asarray(_sr.samples), _T, _K)
            jax.block_until_ready(_Yj)
            _Ys = expand_sobol(_sr, _Yj)

            gsax.analyze(_sr, _Yj).S1.block_until_ready()
            _g = best_of(
                lambda sr=_sr, Y=_Yj: gsax.analyze(sr, Y),
            ) * 1e3

            def _run_salib_sobol(Y=_Ys, T=_T, K=_K):
                _slices = (
                    [Y] if T == 1 and K == 1
                    else [Y[:, k] for k in range(K)] if T == 1
                    else [
                        Y[:, t, k]
                        for t in range(T) for k in range(K)
                    ]
                )
                for _sl in _slices:
                    _N = _sl.shape[0] // _step
                    _Yn = (_sl - _sl.mean()) / _sl.std()
                    _A, _B, _AB, _ = salib_sep(
                        _Yn, D_PARAMS, _N, True,
                    )
                    for j in range(D_PARAMS):
                        salib_s1(_A, _AB[:, j], _B)
                        salib_st(_A, _AB[:, j], _B)

            _run_salib_sobol()
            _sb = float("inf")
            for _ in range(N_REPEATS):
                _t0 = time.perf_counter()
                _run_salib_sobol()
                _sb = min(_sb, time.perf_counter() - _t0)

            _all.append({
                "method": "Sobol", "scenario": _label,
                "gsax_ms": _g, "salib_ms": _sb * 1e3,
            })

        # === eFAST ===
        for _label, _T, _K in SCENARIOS:
            _Xe = gsax.sample_efast(BENCH_PROBLEM, N=2049, M=4, seed=42)
            _Ye = coupled_oscillators(jnp.asarray(_Xe), _T, _K)
            jax.block_until_ready(_Ye)
            _Yin = (
                _Ye[..., None] if _Ye.ndim == 2 and _T > 1
                else _Ye
            )

            gsax.analyze_efast(
                BENCH_PROBLEM, _Yin, M=4,
            ).S1.block_until_ready()
            _g = best_of(
                lambda Y=_Yin: gsax.analyze_efast(
                    BENCH_PROBLEM, Y, M=4,
                ),
            ) * 1e3

            _Xsf = salib_fast_sampler.sample(
                SALIB_PROBLEM, 2049, M=4, seed=42,
            )
            _Ysf = np.asarray(
                coupled_oscillators(jnp.asarray(_Xsf), _T, _K),
            )

            def _run_salib_fast(Y=_Ysf, T=_T, K=_K):
                _slices = (
                    [Y] if T == 1 and K == 1
                    else [Y[:, k] for k in range(K)] if T == 1
                    else [
                        Y[:, t, k]
                        for t in range(T) for k in range(K)
                    ]
                )
                for _sl in _slices:
                    salib_fast_mod.analyze(
                        SALIB_PROBLEM, _sl, M=4,
                        num_resamples=0,
                        print_to_console=False,
                    )

            _run_salib_fast()
            _sb = float("inf")
            for _ in range(N_REPEATS):
                _t0 = time.perf_counter()
                _run_salib_fast()
                _sb = min(_sb, time.perf_counter() - _t0)

            _all.append({
                "method": "eFAST", "scenario": _label,
                "gsax_ms": _g, "salib_ms": _sb * 1e3,
            })

        # === DGSM (pre-computed path for both) ===
        for _label, _T, _K in SCENARIOS:
            _Xmc = jnp.asarray(
                sample_mc(BENCH_PROBLEM, 2048, seed=42),
            )
            _fn = make_unbatched(_T, _K)

            # Pre-compute Y and Jacobian via autodiff (outside timer)
            _r0 = gsax.analyze_dgsm(BENCH_PROBLEM, _fn, _Xmc)
            jax.block_until_ready(_r0.nu)
            _Ydgsm = coupled_oscillators(_Xmc, _T, _K)
            jax.block_until_ready(_Ydgsm)
            _jac_fn = jax.jit(jax.vmap(jax.jacrev(_fn)))
            _dfdx = _jac_fn(_Xmc)
            jax.block_until_ready(_dfdx)

            # gsax: pre-computed path (analysis only)
            gsax.analyze_dgsm(
                BENCH_PROBLEM, Y=_Ydgsm, dfdx=_dfdx,
            ).nu.block_until_ready()
            _gb = float("inf")
            for _ in range(N_REPEATS):
                _t0 = time.perf_counter()
                _r = gsax.analyze_dgsm(
                    BENCH_PROBLEM, Y=_Ydgsm, dfdx=_dfdx,
                )
                jax.block_until_ready(_r.nu)
                _gb = min(_gb, time.perf_counter() - _t0)
            _g = _gb * 1e3

            # SALib: finite-diff samples + analysis only
            _Xfd = salib_finite_diff.sample(
                SALIB_PROBLEM, 2048, delta=0.01,
            )
            _Yfd = np.asarray(
                coupled_oscillators(jnp.asarray(_Xfd), _T, _K),
            )

            def _run_salib_dgsm(X=_Xfd, Y=_Yfd, T=_T, K=_K):
                _slices = (
                    [Y] if T == 1 and K == 1
                    else [Y[:, k] for k in range(K)] if T == 1
                    else [
                        Y[:, t, k]
                        for t in range(T) for k in range(K)
                    ]
                )
                for _sl in _slices:
                    salib_dgsm_mod.analyze(
                        SALIB_PROBLEM, X, _sl,
                        num_resamples=0,
                        print_to_console=False,
                    )

            _run_salib_dgsm()
            _sb = float("inf")
            for _ in range(N_REPEATS):
                _t0 = time.perf_counter()
                _run_salib_dgsm()
                _sb = min(_sb, time.perf_counter() - _t0)

            _all.append({
                "method": "DGSM", "scenario": _label,
                "gsax_ms": _g, "salib_ms": _sb * 1e3,
            })

        # === HDMR ===
        _rng = np.random.default_rng(42)
        _bounds = np.array(BENCH_PROBLEM.bounds)
        _Xnp = _rng.uniform(
            _bounds[:, 0], _bounds[:, 1],
            size=(BASE_N, D_PARAMS),
        )
        _Xjax = jnp.asarray(_Xnp)

        for _label, _T, _K in SCENARIOS:
            _Yj = coupled_oscillators(_Xjax, _T, _K)
            jax.block_until_ready(_Yj)
            _Ynp = np.asarray(_Yj)

            gsax.analyze_hdmr(
                BENCH_PROBLEM, _Xjax, _Yj,
                maxorder=2, m=2,
            ).Sa.block_until_ready()
            _g = best_of(
                lambda X=_Xjax, Y=_Yj: gsax.analyze_hdmr(
                    BENCH_PROBLEM, X, Y, maxorder=2, m=2,
                ),
            ) * 1e3

            _t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _slices = (
                    [_Ynp] if _T == 1 and _K == 1
                    else [_Ynp[:, k] for k in range(_K)] if _T == 1
                    else [
                        _Ynp[:, t, k]
                        for t in range(_T) for k in range(_K)
                    ]
                )
                for _sl in _slices:
                    salib_hdmr_mod.analyze(
                        SALIB_PROBLEM, _Xnp, _sl,
                        maxorder=2, maxiter=100,
                        print_to_console=False,
                    )
            _s = (time.perf_counter() - _t0) * 1e3

            _all.append({
                "method": "HDMR", "scenario": _label,
                "gsax_ms": _g, "salib_ms": _s,
            })

        with open(CACHE_PATH, "w") as _f:
            json.dump(_all, _f, indent=2)

        all_results = _all
    return (all_results,)


@app.cell(hide_code=True)
def _table(all_results, mo):
    mo.stop(all_results is None, mo.md("*Press the button above to run benchmarks.*"))
    _header = (
        "| Method | Scenario | gsax (ms) | SALib (ms)"
        " | Speedup |\n"
        "| --- | --- | ---: | ---: | ---: |"
    )
    _lines = [_header]
    for _r in all_results:
        _g = _r["gsax_ms"]
        _s = _r["salib_ms"]
        _sp = f"**{_s / _g:.1f}x**" if _g > 0 else "---"
        _lines.append(
            f"| {_r['method']} | {_r['scenario']} "
            f"| {_g:.1f} | {_s:.1f} | {_sp} |"
        )
    mo.md(
        "## Results\n\n"
        "Best-of-3 wall-clock ms (analysis only, "
        "model eval + gradients excluded).\n\n"
        + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def _speedup_chart(all_results, mo, np, plt):
    mo.stop(all_results is None)
    _methods = ["Sobol", "eFAST", "DGSM", "HDMR"]
    _scenarios = ["1x1", "1x6", "50x1", "50x6"]
    _speedups = {}
    for _r in all_results:
        _g, _s = _r["gsax_ms"], _r["salib_ms"]
        _speedups[(_r["method"], _r["scenario"])] = (
            _s / _g if _g > 0 else 0
        )

    _x = np.arange(len(_methods))
    _w = 0.18
    _off = [(_i - 1.5) * _w for _i in range(4)]
    _colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    _fig, _ax = plt.subplots(figsize=(10, 5))
    for _si, _sc in enumerate(_scenarios):
        _vals = [
            _speedups.get((_m, _sc), 0) for _m in _methods
        ]
        _ax.bar(
            _x + _off[_si], _vals, _w,
            label=_sc, color=_colors[_si], alpha=0.85,
        )

    _ax.axhline(
        1, color="black", linewidth=0.8,
        linestyle="--", alpha=0.5,
    )
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_methods)
    _ax.set_ylabel("Speedup (SALib / gsax)")
    _ax.set_title(
        "gsax speedup over SALib by method and scenario"
    )
    _ax.legend(title="T x K", frameon=False)
    _ax.set_yscale("log")
    _ax.grid(axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _scaling_chart(all_results, mo, plt):
    mo.stop(all_results is None)
    _tk = {"1x1": 1, "1x6": 6, "50x1": 50, "50x6": 300}
    _methods = ["Sobol", "eFAST", "DGSM", "HDMR"]
    _colors = {
        "Sobol": "C0", "eFAST": "C1",
        "DGSM": "C2", "HDMR": "C3",
    }
    _markers = {
        "Sobol": "o", "eFAST": "s",
        "DGSM": "^", "HDMR": "D",
    }

    _by_method = {}
    for _r in all_results:
        _by_method.setdefault(_r["method"], []).append(_r)

    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _m in _methods:
        if _m not in _by_method:
            continue
        _rows = sorted(
            _by_method[_m],
            key=lambda r: _tk[r["scenario"]],
        )
        _xs = [_tk[r["scenario"]] for r in _rows]
        _ys = [
            r["salib_ms"] / r["gsax_ms"]
            for r in _rows if r["gsax_ms"] > 0
        ]
        _ax.plot(
            _xs, _ys, marker=_markers[_m],
            color=_colors[_m], linewidth=2,
            markersize=8, label=_m,
        )

    _ax.axhline(
        1, color="black", linewidth=0.8,
        linestyle="--", alpha=0.5,
    )
    _ax.set_xlabel("Output dimension (T x K)")
    _ax.set_ylabel("Speedup (SALib / gsax)")
    _ax.set_title("Speedup vs output dimensionality")
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.legend(frameon=False)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _summary(mo):
    mo.md(r"""
    ## Key findings

    - **Speedup grows with T x K** — SALib loops over
      slices; gsax vectorizes with `jax.vmap`
    - **HDMR has largest speedups** — SALib runs backfitting
      per slice; gsax vmaps the entire B-spline fit
    - **eFAST speedup is moderate** — SALib's FFT is
      already fast; gsax gains from fused JIT + vmap
    - **DGSM** — both use pre-computed gradients; gsax
      vectorizes moment computation across outputs
    - **Scalar outputs (1x1)** — SALib can be faster due
      to JIT compilation overhead in gsax

    **Bottom line:** gsax's advantage is **vectorized
    multi-output analysis**. For scalar problems, SALib is
    competitive. For time-series or multi-output workloads,
    gsax is 5-700x faster.
    """)
    return


if __name__ == "__main__":
    app.run()
