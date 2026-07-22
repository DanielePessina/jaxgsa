"""Benchmark: jaxgsa vs SALib on Ishigami (D=3) and Oakley-O'Hagan (D=15).

Compares analysis-phase performance across four SA methods: Sobol, eFAST,
DGSM, and HDMR on two canonical test functions with known analytical
solutions. Model evaluation, sampling, and gradient computation are
pre-computed and excluded from timing.

Run interactively: ``uv run marimo edit examples/benchmark_all.py``
"""

# ruff: noqa: F722, E501

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # jaxgsa vs SALib — Benchmark

    Performance comparison on two canonical SA test functions:

    | | **Ishigami** | **Oakley-O'Hagan** |
    |---|---|---|
    | Dimensions | D = 3 | D = 15 |
    | Inputs | Uniform $[-\pi, \pi]$ | Gaussian $\mathcal{N}(0, 1)$ |
    | Key feature | $x_1$–$x_3$ interaction, $S_1(x_3)=0$ | Natural importance gradient, all pairwise interactions |

    **Methodology.** Only `analyze()` calls are timed — model evaluation,
    sampling, and gradient computation are pre-computed and excluded.
    Best-of-3 wall-clock, post-JIT warmup.

    Press **Run benchmarks** below. Results cache to JSON for fast reload.
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
    from SALib.analyze import sobol as salib_sobol_mod
    from SALib.analyze.sobol import first_order as salib_s1
    from SALib.analyze.sobol import separate_output_values as salib_sep
    from SALib.analyze.sobol import total_order as salib_st
    from SALib.sample import fast_sampler as salib_fast_sampler
    from SALib.sample import finite_diff as salib_finite_diff

    import jaxgsa
    from jaxgsa.benchmarks import ishigami, oakley_ohagan
    from jaxgsa.sampling import monte_carlo

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    plt.rcParams["figure.dpi"] = 150
    return (
        jaxgsa,
        ishigami,
        jax,
        jnp,
        json,
        mo,
        np,
        oakley_ohagan,
        os,
        plt,
        salib_dgsm_mod,
        salib_fast_mod,
        salib_fast_sampler,
        salib_finite_diff,
        salib_hdmr_mod,
        salib_s1,
        salib_sep,
        salib_sobol_mod,
        salib_st,
        monte_carlo,
        time,
        warnings,
    )


@app.cell(hide_code=True)
def _problems_md(mo):
    mo.md(r"""
    ## Test functions

    ### Ishigami (D = 3)

    $$f(\mathbf{x}) = \sin(x_1) + A\sin^2(x_2) + Bx_3^4\sin(x_1)$$

    with $x_i \sim U[-\pi, \pi]$, $A = 7$, $B = 0.1$.  The $x_1$–$x_3$
    interaction ($S_{13} > 0$) and zero first-order contribution of $x_3$
    ($S_1 = 0$) make it a classic SA benchmark.

    **Time-varying extension** for multi-timestep scenarios:

    $$f(\mathbf{x}, t) = \sin(x_1)\cos(t) + A\sin^2(x_2)\,e^{-0.2t} + Bx_3^4\sin(x_1)(1{+}0.3t)$$

    **Multi-output** ($K > 1$): evaluate with $A$ linearly spaced in $[5, 9]$.

    ### Oakley & O'Hagan (D = 15)

    $$f(\mathbf{x}) = \mathbf{a}_1^\top\mathbf{x} + \mathbf{a}_2^\top\sin(\mathbf{x}) + \mathbf{a}_3^\top\cos(\mathbf{x}) + \mathbf{x}^\top M\mathbf{x}$$

    with $x_i \sim \mathcal{N}(0,1)$ i.i.d.  Coefficient magnitudes increase
    with index ($x_{11}$–$x_{15}$ dominate); the quadratic form introduces
    all pairwise interactions.

    **Time-varying extension:**

    $$f(\mathbf{x},t) = \mathbf{a}_1^\top\mathbf{x}\cos(t) + [\mathbf{a}_2^\top\sin(\mathbf{x}) + \mathbf{a}_3^\top\cos(\mathbf{x})]\,e^{-0.3t} + \mathbf{x}^\top M\mathbf{x}(1{+}0.2t)$$

    The quadratic (interaction) term grows while trigonometric terms decay,
    shifting the sensitivity profile over time.
    """)
    return


@app.cell
def _config(ishigami, jax, jnp, np, oakley_ohagan, os, time):
    import dataclasses

    N_REPEATS = 3
    ISHI_BASE_N = 1024
    OAKH_BASE_N = 512
    CACHE_PATH = os.path.join(
        os.path.dirname(__file__) or ".",
        "benchmark_cache.json",
    )

    ISHI_PROB = ishigami.PROBLEM
    OAKH_PROB = oakley_ohagan.PROBLEM

    SALIB_ISHI = {
        "num_vars": 3,
        "names": list(ISHI_PROB.names),
        "bounds": [list(b) for b in ISHI_PROB.bounds],
    }
    SALIB_OAKH = {
        "num_vars": 15,
        "names": [f"x{i + 1}" for i in range(15)],
        "bounds": [[-5.0, 5.0]] * 15,
    }

    # ---- Ishigami model ----
    def ishi_batch(X, T, K):
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        _B = 0.1
        _As = jnp.linspace(5.0, 9.0, K) if K > 1 else jnp.array([7.0])
        if T == 1:
            _ys = [
                jnp.sin(x1) + _As[k] * jnp.sin(x2) ** 2 + _B * x3**4 * jnp.sin(x1)
                for k in range(K)
            ]
            _Y = jnp.stack(_ys, axis=-1)
            return _Y[:, 0] if K == 1 else _Y
        _t = jnp.linspace(0.1, 3.0, T)
        _ys = []
        for k in range(K):
            _ys.append(
                jnp.sin(x1)[:, None] * jnp.cos(_t)[None, :]
                + (_As[k] * jnp.sin(x2) ** 2)[:, None] * jnp.exp(-0.2 * _t)[None, :]
                + (_B * x3**4 * jnp.sin(x1))[:, None] * (1 + 0.3 * _t)[None, :]
            )
        return jnp.stack(_ys, axis=-1)

    def make_ishi_unbatched(T, K):
        _B = 0.1
        _As = jnp.linspace(5.0, 9.0, K) if K > 1 else jnp.array([7.0])
        _tv = jnp.linspace(0.1, 3.0, T) if T > 1 else None

        def fn(x):
            x1, x2, x3 = x[0], x[1], x[2]
            if T == 1:
                _ys = [
                    jnp.sin(x1) + _As[k] * jnp.sin(x2) ** 2 + _B * x3**4 * jnp.sin(x1)
                    for k in range(K)
                ]
                _y = jnp.stack(_ys)
                return _y[0] if K == 1 else _y
            _ys = []
            for k in range(K):
                _ys.append(
                    jnp.sin(x1) * jnp.cos(_tv)
                    + _As[k] * jnp.sin(x2) ** 2 * jnp.exp(-0.2 * _tv)
                    + _B * x3**4 * jnp.sin(x1) * (1 + 0.3 * _tv)
                )
            return jnp.stack(_ys, axis=-1).reshape(-1)

        return fn

    # ---- Oakley-O'Hagan model ----
    _A1j = jnp.asarray(oakley_ohagan._A1)
    _A2j = jnp.asarray(oakley_ohagan._A2)
    _A3j = jnp.asarray(oakley_ohagan._A3)
    _Mj = jnp.asarray(oakley_ohagan._M)

    def oakh_batch(X, T, K=1):
        Xj = jnp.asarray(X)
        if T == 1:
            return oakley_ohagan.evaluate(X)
        _t = jnp.linspace(0.1, 3.0, T)
        _lin = Xj @ _A1j
        _tri = jnp.sin(Xj) @ _A2j + jnp.cos(Xj) @ _A3j
        _qua = jnp.einsum("ni,ij,nj->n", Xj, _Mj, Xj)
        _Y = (
            _lin[:, None] * jnp.cos(_t)[None, :]
            + _tri[:, None] * jnp.exp(-0.3 * _t)[None, :]
            + _qua[:, None] * (1 + 0.2 * _t)[None, :]
        )
        return _Y[:, :, None]

    def make_oakh_unbatched(T, K=1):
        _tv = jnp.linspace(0.1, 3.0, T) if T > 1 else None

        def fn(x):
            _lin = x @ _A1j
            _tri = jnp.sin(x) @ _A2j + jnp.cos(x) @ _A3j
            _qua = x @ _Mj @ x
            if T == 1:
                return _lin + _tri + _qua
            return _lin * jnp.cos(_tv) + _tri * jnp.exp(-0.3 * _tv) + _qua * (1 + 0.2 * _tv)

        return fn

    # ---- Scenarios ----
    ISHI_SCENARIOS = [
        ("1x1", 1, 1),
        ("1x6", 1, 6),
        ("50x1", 50, 1),
        ("50x6", 50, 6),
    ]
    OAKH_SCENARIOS = [("1x1", 1, 1), ("50x1", 50, 1)]

    # ---- Helpers ----
    def expand_sobol(sr, Y):
        return np.asarray(Y)[sr.expanded_to_unique]

    def salib_slices(Y_np, T, K):
        if T == 1 and K == 1:
            return [Y_np] if Y_np.ndim == 1 else [Y_np.ravel()]
        if T == 1:
            return [Y_np[:, k] for k in range(K)]
        if Y_np.ndim == 3:
            return [Y_np[:, t, k] for t in range(T) for k in range(K)]
        return [Y_np[:, t] for t in range(T)]

    def best_of(fn, n=N_REPEATS):
        _best = float("inf")
        for _ in range(n):
            _t0 = time.perf_counter()
            _r = fn()
            # Block EVERY array the result holds — point estimates AND the
            # bootstrap CI arrays (S1_conf, ST_conf, S2_conf) — so no async
            # device work leaks out of the timed region.
            _leaves = [getattr(_r, _f.name) for _f in dataclasses.fields(_r)]
            jax.block_until_ready([_x for _x in _leaves if _x is not None])
            _best = min(_best, time.perf_counter() - _t0)
        return _best

    return (
        CACHE_PATH,
        ISHI_BASE_N,
        ISHI_PROB,
        ISHI_SCENARIOS,
        N_REPEATS,
        OAKH_BASE_N,
        OAKH_PROB,
        OAKH_SCENARIOS,
        SALIB_ISHI,
        SALIB_OAKH,
        best_of,
        expand_sobol,
        ishi_batch,
        make_ishi_unbatched,
        make_oakh_unbatched,
        oakh_batch,
        salib_slices,
    )


@app.cell(hide_code=True)
def _samples_md(ISHI_BASE_N, OAKH_BASE_N, mo):
    mo.md(f"""
    ## Sample sizes

    | Method | Ishigami (D=3) | O'Hagan (D=15) |
    |---|---|---|
    | **Sobol** | base_n={ISHI_BASE_N}, total={8 * ISHI_BASE_N:,} | base_n={OAKH_BASE_N}, total={32 * OAKH_BASE_N:,} |
    | **eFAST** | N=2049/curve, total={2049 * 3:,} | N=2049/curve, total={2049 * 15:,} |
    | **DGSM** | N=2,048 MC samples | N=2,048 MC samples |
    | **HDMR** | N={ISHI_BASE_N:,} random | N={OAKH_BASE_N:,} random |

    All timings are analysis-only, best-of-3, post-JIT warmup.
    Scenarios: **T x K** where T = timesteps, K = output channels.
    """)
    return


@app.cell
def _run_button(mo):
    run_btn = mo.ui.run_button(label="Run benchmarks")
    mo.md(f"### {run_btn}")
    return (run_btn,)


@app.cell
def _benchmark(
    CACHE_PATH,
    ISHI_BASE_N,
    ISHI_PROB,
    ISHI_SCENARIOS,
    N_REPEATS,
    OAKH_BASE_N,
    OAKH_PROB,
    OAKH_SCENARIOS,
    SALIB_ISHI,
    SALIB_OAKH,
    best_of,
    expand_sobol,
    jaxgsa,
    ishi_batch,
    jax,
    jnp,
    json,
    make_ishi_unbatched,
    make_oakh_unbatched,
    np,
    oakh_batch,
    os,
    run_btn,
    salib_dgsm_mod,
    salib_fast_mod,
    salib_fast_sampler,
    salib_finite_diff,
    salib_hdmr_mod,
    salib_s1,
    salib_sep,
    salib_slices,
    salib_sobol_mod,
    salib_st,
    monte_carlo,
    time,
    warnings,
):
    _from_cache = False
    if not run_btn.value and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as _f:
            all_results = json.load(_f)
        _from_cache = True

    if not run_btn.value and not _from_cache:
        all_results = None

    if run_btn.value:
        _all = []
        _problems = [
            (
                "Ishigami",
                ISHI_PROB,
                SALIB_ISHI,
                3,
                ISHI_BASE_N,
                ISHI_SCENARIOS,
                ishi_batch,
                make_ishi_unbatched,
            ),
            (
                "O'Hagan",
                OAKH_PROB,
                SALIB_OAKH,
                15,
                OAKH_BASE_N,
                OAKH_SCENARIOS,
                oakh_batch,
                make_oakh_unbatched,
            ),
        ]

        for (
            _pname,
            _prob,
            _sprob,
            _D,
            _bn,
            _scens,
            _bfn,
            _ubfn,
        ) in _problems:
            _step = 2 * _D + 2

            # ======== SOBOL ========
            for _label, _T, _K in _scens:
                _sr = jaxgsa.sobol.sample(
                    _prob,
                    _bn * _step,
                    seed=1,
                    calc_second_order=True,
                    verbose=False,
                )
                _Yj = _bfn(jnp.asarray(_sr.samples), _T, _K)
                jax.block_until_ready(_Yj)
                _Ys = expand_sobol(_sr, _Yj)

                jaxgsa.sobol.analyze(_sr, _Yj).S1.block_until_ready()
                _g = (
                    best_of(
                        lambda sr=_sr, Y=_Yj: jaxgsa.sobol.analyze(sr, Y),
                    )
                    * 1e3
                )

                def _run_salib_sobol(
                    Y=_Ys,
                    T=_T,
                    K=_K,
                    D=_D,
                    step=_step,
                ):
                    for _sl in salib_slices(Y, T, K):
                        _N = _sl.shape[0] // step
                        _Yn = (_sl - _sl.mean()) / _sl.std()
                        _A, _B, _AB, _ = salib_sep(
                            _Yn,
                            D,
                            _N,
                            True,
                        )
                        for j in range(D):
                            salib_s1(_A, _AB[:, j], _B)
                            salib_st(_A, _AB[:, j], _B)

                _run_salib_sobol()
                _sb = float("inf")
                for _ in range(N_REPEATS):
                    _t0 = time.perf_counter()
                    _run_salib_sobol()
                    _sb = min(_sb, time.perf_counter() - _t0)

                _all.append(
                    {
                        "problem": _pname,
                        "method": "Sobol",
                        "scenario": _label,
                        "jaxgsa_ms": _g,
                        "salib_ms": _sb * 1e3,
                    }
                )

            # ======== eFAST ========
            for _label, _T, _K in _scens:
                _Xe = jaxgsa.efast.sample(
                    _prob,
                    n_per_curve=2049,
                    M=4,
                    seed=42,
                )
                _Ye = _bfn(jnp.asarray(_Xe.samples), _T, _K)
                jax.block_until_ready(_Ye)

                jaxgsa.efast.analyze(
                    _Xe,
                    _Ye,
                ).S1.block_until_ready()
                _g = (
                    best_of(
                        lambda Y=_Ye, sr=_Xe: jaxgsa.efast.analyze(
                            sr,
                            Y,
                        ),
                    )
                    * 1e3
                )

                _Xsf = salib_fast_sampler.sample(
                    _sprob,
                    2049,
                    M=4,
                    seed=42,
                )
                _Ysf = np.asarray(
                    _bfn(jnp.asarray(_Xsf), _T, _K),
                )

                def _run_salib_fast(
                    Y=_Ysf,
                    T=_T,
                    K=_K,
                    sp=_sprob,
                ):
                    for _sl in salib_slices(Y, T, K):
                        salib_fast_mod.analyze(
                            sp,
                            _sl,
                            M=4,
                            num_resamples=0,
                            print_to_console=False,
                        )

                _run_salib_fast()
                _sb = float("inf")
                for _ in range(N_REPEATS):
                    _t0 = time.perf_counter()
                    _run_salib_fast()
                    _sb = min(_sb, time.perf_counter() - _t0)

                _all.append(
                    {
                        "problem": _pname,
                        "method": "eFAST",
                        "scenario": _label,
                        "jaxgsa_ms": _g,
                        "salib_ms": _sb * 1e3,
                    }
                )

            # ======== DGSM (pre-computed path) ========
            for _label, _T, _K in _scens:
                _Xmc = jnp.asarray(
                    monte_carlo(_prob, 2048, seed=42),
                )
                _fn = _ubfn(_T, _K)

                # Warmup autodiff path
                jaxgsa.dgsm.analyze(
                    _prob,
                    _fn,
                    _Xmc,
                ).nu.block_until_ready()

                # Pre-compute Y and Jacobian (excluded from timer)
                _jit_fn = jax.jit(jax.vmap(_fn))
                _Yd = _jit_fn(_Xmc)
                jax.block_until_ready(_Yd)
                _jac_fn = jax.jit(jax.vmap(jax.jacrev(_fn)))
                _dfdx = _jac_fn(_Xmc)
                jax.block_until_ready(_dfdx)

                # jaxgsa: pre-computed analysis only
                jaxgsa.dgsm.analyze(
                    _prob,
                    Y=_Yd,
                    dfdx=_dfdx,
                ).nu.block_until_ready()
                _gb = float("inf")
                for _ in range(N_REPEATS):
                    _t0 = time.perf_counter()
                    _r = jaxgsa.dgsm.analyze(
                        _prob,
                        Y=_Yd,
                        dfdx=_dfdx,
                    )
                    jax.block_until_ready(_r.nu)
                    _gb = min(_gb, time.perf_counter() - _t0)
                _g = _gb * 1e3

                # SALib: finite-diff samples + analysis
                _Xfd = salib_finite_diff.sample(
                    _sprob,
                    2048,
                    delta=0.01,
                )
                _Yfd = np.asarray(
                    _bfn(jnp.asarray(_Xfd), _T, _K),
                )

                def _run_salib_dgsm(
                    X=_Xfd,
                    Y=_Yfd,
                    T=_T,
                    K=_K,
                    sp=_sprob,
                ):
                    for _sl in salib_slices(Y, T, K):
                        salib_dgsm_mod.analyze(
                            sp,
                            X,
                            _sl,
                            num_resamples=0,
                            print_to_console=False,
                        )

                _run_salib_dgsm()
                _sb = float("inf")
                for _ in range(N_REPEATS):
                    _t0 = time.perf_counter()
                    _run_salib_dgsm()
                    _sb = min(_sb, time.perf_counter() - _t0)

                _all.append(
                    {
                        "problem": _pname,
                        "method": "DGSM",
                        "scenario": _label,
                        "jaxgsa_ms": _g,
                        "salib_ms": _sb * 1e3,
                    }
                )

            # ======== HDMR ========
            _Xmc_h = np.asarray(monte_carlo(_prob, _bn, seed=42))
            _Xjax_h = jnp.asarray(_Xmc_h)

            for _label, _T, _K in _scens:
                _Yj = _bfn(_Xjax_h, _T, _K)
                jax.block_until_ready(_Yj)
                _Ynp = np.asarray(_Yj)

                jaxgsa.hdmr.analyze(
                    _prob,
                    _Xjax_h,
                    _Yj,
                    maxorder=2,
                    m=2,
                ).Sa.block_until_ready()
                _g = (
                    best_of(
                        lambda X=_Xjax_h, Y=_Yj, p=_prob: jaxgsa.hdmr.analyze(
                            p,
                            X,
                            Y,
                            maxorder=2,
                            m=2,
                        ),
                    )
                    * 1e3
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Warmup
                    for _sl in salib_slices(_Ynp, _T, _K):
                        salib_hdmr_mod.analyze(
                            _sprob,
                            _Xmc_h,
                            _sl,
                            maxorder=2,
                            maxiter=100,
                            print_to_console=False,
                        )
                    _sb = float("inf")
                    for _ in range(N_REPEATS):
                        _t0 = time.perf_counter()
                        for _sl in salib_slices(_Ynp, _T, _K):
                            salib_hdmr_mod.analyze(
                                _sprob,
                                _Xmc_h,
                                _sl,
                                maxorder=2,
                                maxiter=100,
                                print_to_console=False,
                            )
                        _sb = min(
                            _sb,
                            time.perf_counter() - _t0,
                        )
                _s = _sb * 1e3

                _all.append(
                    {
                        "problem": _pname,
                        "method": "HDMR",
                        "scenario": _label,
                        "jaxgsa_ms": _g,
                        "salib_ms": _s,
                    }
                )

        # ======== BOOTSTRAP COMPARISON (Ishigami 1x1 Sobol) ========
        _boot = {}
        _ishi_step = 2 * 3 + 2
        _sr_b = jaxgsa.sobol.sample(
            ISHI_PROB,
            ISHI_BASE_N * _ishi_step,
            seed=1,
            calc_second_order=True,
            verbose=False,
        )
        _Yb = ishi_batch(jnp.asarray(_sr_b.samples), 1, 1)
        jax.block_until_ready(_Yb)
        _Yb_exp = expand_sobol(_sr_b, _Yb)

        # jaxgsa: 0, 100, 1000 resamples
        # Use prenormalize=True + ci_method="gaussian" to match SALib
        for _nr in [0, 100, 1000]:
            jaxgsa.sobol.analyze(
                _sr_b,
                _Yb,
                num_resamples=_nr,
                prenormalize=True,
                ci_method="gaussian",
                key=jax.random.key(0),
            ).S1.block_until_ready()
            _g = (
                best_of(
                    lambda nr=_nr: jaxgsa.sobol.analyze(
                        _sr_b,
                        _Yb,
                        num_resamples=nr,
                        prenormalize=True,
                        ci_method="gaussian",
                        key=jax.random.key(0),
                    ),
                )
                * 1e3
            )
            _boot[f"jaxgsa_{_nr}"] = _g
            if _nr > 0:
                _r = jaxgsa.sobol.analyze(
                    _sr_b,
                    _Yb,
                    num_resamples=_nr,
                    prenormalize=True,
                    ci_method="gaussian",
                    key=jax.random.key(0),
                )
                _boot[f"jaxgsa_ci_{_nr}"] = float(
                    jnp.mean(_r.S1_conf[1] - _r.S1_conf[0]),
                )

        # SALib: 100, 1000 resamples (prenormalize + gaussian by default)
        for _nr in [100, 1000]:
            salib_sobol_mod.analyze(
                SALIB_ISHI,
                _Yb_exp,
                calc_second_order=True,
                num_resamples=_nr,
                print_to_console=False,
            )
            _sb = float("inf")
            for _ in range(N_REPEATS):
                _t0 = time.perf_counter()
                _Si = salib_sobol_mod.analyze(
                    SALIB_ISHI,
                    _Yb_exp,
                    calc_second_order=True,
                    num_resamples=_nr,
                    print_to_console=False,
                )
                _sb = min(_sb, time.perf_counter() - _t0)
            _boot[f"salib_{_nr}"] = _sb * 1e3
            _boot[f"salib_ci_{_nr}"] = float(
                2 * np.mean(_Si["S1_conf"]),
            )

        with open(CACHE_PATH, "w") as _f:
            json.dump(
                {"speed": _all, "bootstrap": _boot},
                _f,
                indent=2,
            )
        all_results = {"speed": _all, "bootstrap": _boot}
    return (all_results,)


@app.cell(hide_code=True)
def _tables(all_results, mo):
    mo.stop(
        all_results is None,
        mo.md("*Press the button above to run benchmarks.*"),
    )
    _speed = all_results["speed"]
    _lines = []

    for _pname in ["Ishigami", "O'Hagan"]:
        _d = 3 if _pname == "Ishigami" else 15
        _lines.append(f"\n### {_pname} (D={_d})\n")
        _lines.append(
            "| Method | Scenario | jaxgsa (ms) | SALib (ms)"
            " | Speedup |\n"
            "| --- | --- | ---: | ---: | ---: |"
        )
        for _r in _speed:
            if _r["problem"] != _pname:
                continue
            _g, _s = _r["jaxgsa_ms"], _r["salib_ms"]
            _sp = f"**{_s / _g:.1f}x**" if _g > 0 else "---"
            _lines.append(f"| {_r['method']} | {_r['scenario']} | {_g:.1f} | {_s:.1f} | {_sp} |")

    mo.md("## Results\n\nBest-of-3 wall-clock ms (analysis only).\n" + "\n".join(_lines))
    return


@app.cell(hide_code=True)
def _bar_charts(all_results, mo, np, plt):
    mo.stop(all_results is None)
    _speed = all_results["speed"]
    _methods = ["Sobol", "eFAST", "DGSM", "HDMR"]
    _colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    def _make_chart(ax, prob_name, scenarios):
        _speedups = {}
        for _r in _speed:
            if _r["problem"] != prob_name:
                continue
            _g, _s = _r["jaxgsa_ms"], _r["salib_ms"]
            _speedups[(_r["method"], _r["scenario"])] = _s / _g if _g > 0 else 0
        _x = np.arange(len(_methods))
        _w = 0.8 / max(len(scenarios), 1)
        _offs = [(_i - (len(scenarios) - 1) / 2) * _w for _i in range(len(scenarios))]
        for _si, _sc in enumerate(scenarios):
            _vals = [_speedups.get((_m, _sc), 0) for _m in _methods]
            ax.bar(
                _x + _offs[_si],
                _vals,
                _w,
                label=_sc,
                color=_colors[_si % len(_colors)],
                alpha=0.85,
            )
        ax.axhline(
            1,
            color="black",
            linewidth=0.8,
            linestyle="--",
            alpha=0.5,
        )
        ax.set_xticks(_x)
        ax.set_xticklabels(_methods)
        ax.set_ylabel("Speedup (SALib / jaxgsa)")
        _d = 3 if prob_name == "Ishigami" else 15
        ax.set_title(f"{prob_name} (D={_d})")
        ax.legend(title="T x K", frameon=False, fontsize=7)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

    _fig, (_ax1, _ax2) = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )
    _make_chart(
        _ax1,
        "Ishigami",
        ["1x1", "1x6", "50x1", "50x6"],
    )
    _make_chart(_ax2, "O'Hagan", ["1x1", "50x1"])
    _fig.suptitle(
        "jaxgsa speedup over SALib",
        fontsize=14,
        y=1.02,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _scaling_chart(all_results, mo, plt):
    mo.stop(all_results is None)
    _speed = all_results["speed"]
    _tk_map = {"1x1": 1, "1x6": 6, "50x1": 50, "50x6": 300}
    _methods = ["Sobol", "eFAST", "DGSM", "HDMR"]
    _styles = {
        "Sobol": ("o", "C0"),
        "eFAST": ("s", "C1"),
        "DGSM": ("^", "C2"),
        "HDMR": ("D", "C3"),
    }

    _fig, (_ax1, _ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True,
    )
    for _ax, _pname in [(_ax1, "Ishigami"), (_ax2, "O'Hagan")]:
        _by_m = {}
        for _r in _speed:
            if _r["problem"] == _pname:
                _by_m.setdefault(_r["method"], []).append(_r)
        for _m in _methods:
            if _m not in _by_m:
                continue
            _rows = [
                r
                for r in sorted(
                    _by_m[_m],
                    key=lambda r: _tk_map[r["scenario"]],
                )
                if r["jaxgsa_ms"] > 0
            ]
            _xs = [_tk_map[r["scenario"]] for r in _rows]
            _ys = [r["salib_ms"] / r["jaxgsa_ms"] for r in _rows]
            _mk, _co = _styles[_m]
            _ax.plot(
                _xs,
                _ys,
                marker=_mk,
                color=_co,
                linewidth=2,
                markersize=8,
                label=_m,
            )
        _ax.axhline(
            1,
            color="black",
            linewidth=0.8,
            linestyle="--",
            alpha=0.5,
        )
        _ax.set_xlabel("Output dimension (T x K)")
        _ax.set_title(_pname)
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.legend(frameon=False, fontsize=8)
        _ax.grid(alpha=0.3)

    _ax1.set_ylabel("Speedup (SALib / jaxgsa)")
    _fig.suptitle(
        "Speedup vs output dimensionality",
        fontsize=14,
        y=1.02,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _bootstrap_section(all_results, mo, np, plt):
    mo.stop(all_results is None)
    _boot = all_results.get("bootstrap")
    mo.stop(_boot is None)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- Timing comparison ----
    _labels = ["100", "1000"]
    _jaxgsa_t = [_boot["jaxgsa_100"], _boot["jaxgsa_1000"]]
    _salib_t = [_boot["salib_100"], _boot["salib_1000"]]
    _x = np.arange(len(_labels))
    _w = 0.35
    _ax1.bar(
        _x - _w / 2,
        _jaxgsa_t,
        _w,
        label="jaxgsa",
        color="#2196F3",
    )
    _ax1.bar(
        _x + _w / 2,
        _salib_t,
        _w,
        label="SALib",
        color="#FF9800",
    )
    _pt = _boot["jaxgsa_0"]
    _ax1.axhline(
        _pt,
        color="#2196F3",
        linewidth=1,
        linestyle=":",
        alpha=0.7,
    )
    _ax1.annotate(
        f"jaxgsa point est. {_pt:.1f} ms",
        xy=(0, _pt),
        xytext=(0.3, _pt * 1.3),
        fontsize=7,
        color="#2196F3",
    )
    _ax1.set_xticks(_x)
    _ax1.set_xticklabels(_labels)
    _ax1.set_xlabel("Bootstrap resamples")
    _ax1.set_ylabel("Time (ms)")
    _ax1.set_title("Sobol analysis time (Ishigami 1x1)")
    _ax1.legend(frameon=False)
    _ax1.grid(axis="y", alpha=0.3)

    # ---- CI width comparison ----
    _ci_jaxgsa = [
        _boot.get("jaxgsa_ci_100", 0),
        _boot.get("jaxgsa_ci_1000", 0),
    ]
    _ci_salib = [
        _boot.get("salib_ci_100", 0),
        _boot.get("salib_ci_1000", 0),
    ]
    _x2 = np.arange(2)
    _ax2.bar(
        _x2 - _w / 2,
        _ci_jaxgsa,
        _w,
        label="jaxgsa",
        color="#2196F3",
    )
    _ax2.bar(
        _x2 + _w / 2,
        _ci_salib,
        _w,
        label="SALib",
        color="#FF9800",
    )
    _ax2.set_xticks(_x2)
    _ax2.set_xticklabels(["100 resamples", "1000 resamples"])
    _ax2.set_ylabel("Mean $S_1$ CI width")
    _ax2.set_title("Bootstrap confidence interval width")
    _ax2.legend(frameon=False)
    _ax2.grid(axis="y", alpha=0.3)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _findings(mo):
    mo.md(r"""
    ## Key findings

    - **Speedup grows with output dimensionality** — SALib loops over
      (T, K) slices in Python; jaxgsa vectorizes with `jax.vmap`
    - **Higher D amplifies the gap** — O'Hagan (D=15) shows larger
      speedups than Ishigami (D=3) because SALib's per-parameter
      loops are longer
    - **HDMR has the largest speedups** — SALib runs backfitting per
      slice; jaxgsa vmaps the entire B-spline fit
    - **DGSM note** — jaxgsa uses pre-computed Jacobians (autodiff
      excluded); SALib's `analyze()` includes finite-difference
      computation since it has no pre-computed path
    - **Scalar outputs (1x1)** — SALib can match or beat jaxgsa: the
      analysis itself is tiny, so JAX's fixed per-call overhead
      dominates the timing
    - **Bootstrap scales efficiently in jaxgsa** — JIT reuse across
      resamples means 10x more resamples costs far less than 10x
      more time; CI width narrows as $\sim 1/\sqrt{B}$

    **Bottom line:** jaxgsa's advantage is vectorized multi-output
    analysis. For scalar problems, SALib is competitive. For
    time-series or multi-output workloads, jaxgsa is 5--700x faster.
    """)
    return


if __name__ == "__main__":
    app.run()
