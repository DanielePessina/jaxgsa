"""Method comparison: eight GSA approaches on the Ishigami benchmark.

Compares Sobol, eFAST, HDMR, PCE, DGSM, Morris, Shapley effects, and
Borgonovo delta on the Ishigami function, a standard sensitivity
analysis test case with known analytical indices.

Run interactively: ``uv run marimo edit examples/method_comparison.py``
Run as script:     ``uv run python examples/method_comparison.py``
"""

# ruff: noqa: F722

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # GSA Method Comparison

    **Eight methods, one function — which to choose?**

    This notebook runs eight global sensitivity analysis methods on the
    Ishigami function and compares their accuracy, cost, and speed:

    | Method | What it computes |
    | --- | --- |
    | **Sobol** (Saltelli) | S1, ST, S2 via variance decomposition |
    | **eFAST** | S1, ST via Fourier amplitude decomposition |
    | **HDMR** | S1, ST via B-spline surrogate (ANCOVA) |
    | **PCE** | S1, ST, S2 via polynomial chaos expansion (scalar output only) |
    | **DGSM** | Bounds on ST via derivative-based measures |
    | **Morris** | mu*, sigma screening measures (proxy for the ST *ranking*) |
    | **Shapley** | Sh fair variance shares (sum to 1) from a PCE surrogate, plus S1, ST |
    | **Borgonovo delta** | delta moment-independent density shift, plus given-data S1 |
    """)
    return


@app.cell
def _imports():
    import time

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import gsax
    from gsax.benchmarks import ishigami

    plt.rcParams["figure.dpi"] = 150
    return gsax, ishigami, jax, jnp, mo, np, plt, time


@app.cell
def _setup(ishigami, jnp):
    problem = ishigami.PROBLEM
    analytical_s1 = ishigami.ANALYTICAL_S1
    analytical_st = ishigami.ANALYTICAL_ST

    def ishigami_fn(x):
        """Unbatched Ishigami: (3,) -> ()."""
        return jnp.sin(x[0]) + 7.0 * jnp.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * jnp.sin(x[0])

    return analytical_s1, analytical_st, ishigami_fn, problem


@app.cell
def _run_all(gsax, ishigami, ishigami_fn, jax, jnp, problem, time):
    # --- Sobol (Saltelli) ---
    _t0 = time.perf_counter()
    sr = gsax.sample(problem, 4096, seed=42, calc_second_order=True)
    _Y_sobol = ishigami.evaluate(jnp.asarray(sr.samples))
    result_sobol = gsax.analyze(sr, _Y_sobol)
    jax.block_until_ready(result_sobol.S1)
    time_sobol = time.perf_counter() - _t0
    n_evals_sobol = sr.n_total

    # --- eFAST ---
    _t0 = time.perf_counter()
    _X_efast = gsax.sample_efast(problem, N=4096, M=4, seed=42)
    _Y_efast = ishigami.evaluate(jnp.asarray(_X_efast))
    result_efast = gsax.analyze_efast(problem, _Y_efast, M=4)
    jax.block_until_ready(result_efast.S1)
    time_efast = time.perf_counter() - _t0
    n_evals_efast = len(_X_efast)

    # --- HDMR ---
    _key = jax.random.PRNGKey(42)
    _bounds = jnp.array(problem.bounds)
    _X_hdmr = jax.random.uniform(_key, (2000, 3), minval=_bounds[:, 0], maxval=_bounds[:, 1])
    _Y_hdmr = ishigami.evaluate(_X_hdmr)
    _t0 = time.perf_counter()
    result_hdmr = gsax.analyze_hdmr(problem, _X_hdmr, _Y_hdmr, maxorder=2, m=2)
    jax.block_until_ready(result_hdmr.S1)
    time_hdmr = time.perf_counter() - _t0
    n_evals_hdmr = len(_X_hdmr)

    # --- PCE ---
    _t0 = time.perf_counter()
    result_pce = gsax.analyze_pce(problem, _X_hdmr, _Y_hdmr, order=4)
    jax.block_until_ready(result_pce.S1)
    time_pce = time.perf_counter() - _t0
    n_evals_pce = len(_X_hdmr)

    # --- DGSM ---
    _X_dgsm = gsax.sample_mc(problem, N=10_000, seed=42)
    _t0 = time.perf_counter()
    result_dgsm = gsax.analyze_dgsm(problem, ishigami_fn, jnp.asarray(_X_dgsm))
    jax.block_until_ready(result_dgsm.upper_bound)
    time_dgsm = time.perf_counter() - _t0
    n_evals_dgsm = len(_X_dgsm)

    # --- Morris ---
    _t0 = time.perf_counter()
    sr_morris = gsax.sample_morris(problem, 100, seed=42)
    _Y_morris = ishigami.evaluate(jnp.asarray(sr_morris.samples))
    result_morris = gsax.analyze_morris(sr_morris, _Y_morris)
    jax.block_until_ready(result_morris.mu_star)
    time_morris = time.perf_counter() - _t0
    n_evals_morris = sr_morris.n_total

    # --- Shapley (PCE backend, same data and order as the PCE run) ---
    _t0 = time.perf_counter()
    result_shapley = gsax.analyze_shapley(problem, _X_hdmr, _Y_hdmr, order=4)
    jax.block_until_ready(result_shapley.Sh)
    time_shapley = time.perf_counter() - _t0
    n_evals_shapley = len(_X_hdmr)

    # --- Borgonovo delta ---
    _t0 = time.perf_counter()
    result_borgonovo = gsax.analyze_borgonovo(problem, _X_hdmr, _Y_hdmr, seed=42)
    jax.block_until_ready(result_borgonovo.delta)
    time_borgonovo = time.perf_counter() - _t0
    n_evals_borgonovo = len(_X_hdmr)
    return (
        n_evals_borgonovo,
        n_evals_dgsm,
        n_evals_efast,
        n_evals_hdmr,
        n_evals_morris,
        n_evals_pce,
        n_evals_shapley,
        n_evals_sobol,
        result_borgonovo,
        result_dgsm,
        result_efast,
        result_hdmr,
        result_morris,
        result_pce,
        result_shapley,
        result_sobol,
        time_borgonovo,
        time_dgsm,
        time_efast,
        time_hdmr,
        time_morris,
        time_pce,
        time_shapley,
        time_sobol,
    )


@app.cell(hide_code=True)
def _s1_chart(
    analytical_s1,
    np,
    plt,
    problem,
    result_efast,
    result_hdmr,
    result_pce,
    result_sobol,
):
    _names = list(problem.names)
    _x = np.arange(len(_names))
    _n_methods = 4
    _width = 0.15

    _s1_sobol = np.asarray(result_sobol.S1)
    _s1_efast = np.asarray(result_efast.S1)
    _s1_hdmr = np.asarray(result_hdmr.S1)
    _s1_pce = np.asarray(result_pce.S1)
    _s1_ana = np.asarray(analytical_s1)

    _colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    _labels = ["Sobol", "eFAST", "HDMR", "PCE"]
    _data = [_s1_sobol, _s1_efast, _s1_hdmr, _s1_pce]

    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _i, (_d, _c, _l) in enumerate(
        zip(_data, _colors, _labels, strict=True),
    ):
        _offset = (_i - _n_methods / 2 + 0.5) * _width
        _ax.bar(_x + _offset, _d, _width, color=_c, label=_l, alpha=0.85)

    _ax.scatter(_x, _s1_ana, marker="D", color="black", s=60, zorder=5, label="Analytical")

    _ax.set_xticks(_x)
    _ax.set_xticklabels(_names)
    _ax.set_ylabel("S1")
    _ax.set_title("First-Order Indices (S1)")
    _ax.legend(frameon=False, fontsize=8)
    _ax.grid(axis="y", alpha=0.3)
    _ax.set_ylim(bottom=-0.05)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _st_chart(
    analytical_st,
    np,
    plt,
    problem,
    result_dgsm,
    result_efast,
    result_hdmr,
    result_pce,
    result_sobol,
):
    _names = list(problem.names)
    _x = np.arange(len(_names))
    _n_methods = 5
    _width = 0.14

    _st_sobol = np.asarray(result_sobol.ST)
    _st_efast = np.asarray(result_efast.ST)
    _st_hdmr = np.asarray(result_hdmr.ST)
    _st_pce = np.asarray(result_pce.ST)
    _st_dgsm_ub = np.asarray(result_dgsm.upper_bound)
    _st_ana = np.asarray(analytical_st)

    _colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    _labels = ["Sobol", "eFAST", "HDMR", "PCE", "DGSM (upper)"]
    _data = [_st_sobol, _st_efast, _st_hdmr, _st_pce, _st_dgsm_ub]

    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _i, (_d, _c, _l) in enumerate(
        zip(_data, _colors, _labels, strict=True),
    ):
        _offset = (_i - _n_methods / 2 + 0.5) * _width
        _ax.bar(_x + _offset, _d, _width, color=_c, label=_l, alpha=0.85)

    _ax.scatter(_x, _st_ana, marker="D", color="black", s=60, zorder=5, label="Analytical")

    _ax.set_xticks(_x)
    _ax.set_xticklabels(_names)
    _ax.set_ylabel("ST")
    _ax.set_title("Total-Order Indices (ST)")
    _ax.legend(frameon=False, fontsize=8)
    _ax.grid(axis="y", alpha=0.3)
    _ax.set_ylim(bottom=-0.05)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _morris_md(mo):
    mo.md(r"""
    ### Morris screening (different scale)

    Morris $\mu^*$ is the mean absolute elementary effect — a
    **screening** measure on a derivative-like scale, not a variance
    share. It is not directly comparable to $S_1$/$S_T$, so it is left
    out of the index charts above. Normalizing $\mu^*$ to sum to one
    supports a **ranking** check against the analytical $S_T$ shares —
    the ranking ($x_1 > x_2 > x_3$) matches — but the normalized values
    themselves carry no variance meaning.
    """)
    return


@app.cell(hide_code=True)
def _morris_chart(analytical_st, np, plt, problem, result_morris):
    _names = list(problem.names)
    _x = np.arange(len(_names))
    _width = 0.38

    _mu_star = np.asarray(result_morris.mu_star)
    _mu_star_share = _mu_star / _mu_star.sum()
    _st_ana = np.asarray(analytical_st)
    _st_share = _st_ana / _st_ana.sum()

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.bar(
        _x - _width / 2,
        _mu_star_share,
        _width,
        color="#8c564b",
        alpha=0.85,
        label="Morris mu* (normalized)",
    )
    _ax.bar(
        _x + _width / 2,
        _st_share,
        _width,
        color="black",
        alpha=0.35,
        label="Analytical ST (normalized)",
    )
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_names)
    _ax.set_ylabel("Share of total (ranking check only)")
    _ax.set_title("Morris mu* vs analytical ST — normalized for ranking")
    _ax.legend(frameon=False, fontsize=8)
    _ax.grid(axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _shapley_md(mo):
    mo.md(r"""
    ### Shapley effects (fair shares)

    Shapley effects allocate variance **fairly**: each interaction's
    variance is split equally among its participants, so the shares sum
    to exactly 1. The run above reuses the PCE surrogate fit (same data,
    same order), so its $S_1$/$S_T$ match the PCE row exactly — the new
    information is $\mathrm{Sh}$ itself, checked here against the
    analytical Ishigami Shapley effects. A degree-4 polynomial leaves
    the $x_1$ share a little high; see ``examples/shapley_gsa.py`` for
    the order-convergence study.
    """)
    return


@app.cell(hide_code=True)
def _shapley_chart(ishigami, np, plt, problem, result_shapley):
    _names = list(problem.names)
    _x = np.arange(len(_names))
    _width = 0.38

    _sh = np.asarray(result_shapley.Sh)
    _sh_ana = np.asarray(ishigami.ANALYTICAL_SHAPLEY)

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.bar(
        _x - _width / 2,
        _sh,
        _width,
        color="#17becf",
        alpha=0.85,
        label="Shapley Sh (PCE backend)",
    )
    _ax.bar(
        _x + _width / 2,
        _sh_ana,
        _width,
        color="black",
        alpha=0.35,
        label="Analytical Sh",
    )
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_names)
    _ax.set_ylabel("Sh (fair variance share)")
    _ax.set_title("Shapley effects vs analytical (each sums to 1)")
    _ax.legend(frameon=False, fontsize=8)
    _ax.grid(axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _borgonovo_md(mo):
    mo.md(r"""
    ### Borgonovo delta (different scale)

    Borgonovo delta is **moment-independent**: it measures the expected
    L1 shift of the entire output density when an input is fixed, on its
    own [0, 1] scale — not a variance share, so it is left out of the
    index charts above. The same conditioning also yields a given-data
    first-order Sobol estimate for free, plotted against the analytical
    $S_1$ as a sanity check. Note $x_3$: its $S_1$ is near zero, yet its
    delta is clearly positive — fixing it reshapes the output density
    through the $x_1 x_3$ interaction.
    """)
    return


@app.cell(hide_code=True)
def _borgonovo_chart(analytical_s1, np, plt, problem, result_borgonovo):
    _names = list(problem.names)
    _x = np.arange(len(_names))
    _width = 0.38

    _delta = np.asarray(result_borgonovo.delta)
    _s1_given_data = np.asarray(result_borgonovo.S1)
    _s1_ana = np.asarray(analytical_s1)

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.bar(
        _x - _width / 2,
        _delta,
        _width,
        color="#e377c2",
        alpha=0.85,
        label="Borgonovo delta",
    )
    _ax.bar(
        _x + _width / 2,
        _s1_given_data,
        _width,
        color="#7f7f7f",
        alpha=0.85,
        label="Given-data S1",
    )
    _ax.scatter(
        _x,
        _s1_ana,
        marker="D",
        color="black",
        s=60,
        zorder=5,
        label="Analytical S1",
    )
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_names)
    _ax.set_ylabel("index value")
    _ax.set_title("Borgonovo delta and given-data S1")
    _ax.legend(frameon=False, fontsize=8)
    _ax.grid(axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _accuracy_table(
    analytical_s1,
    analytical_st,
    mo,
    n_evals_borgonovo,
    n_evals_dgsm,
    n_evals_efast,
    n_evals_hdmr,
    n_evals_morris,
    n_evals_pce,
    n_evals_shapley,
    n_evals_sobol,
    np,
    result_borgonovo,
    result_dgsm,
    result_efast,
    result_hdmr,
    result_pce,
    result_shapley,
    result_sobol,
    time_borgonovo,
    time_dgsm,
    time_efast,
    time_hdmr,
    time_morris,
    time_pce,
    time_shapley,
    time_sobol,
):
    _s1_ana = np.asarray(analytical_s1)
    _st_ana = np.asarray(analytical_st)

    def _mae(estimate, truth):
        return float(np.mean(np.abs(np.asarray(estimate) - truth)))

    _rows = [
        (
            "Sobol",
            _mae(result_sobol.S1, _s1_ana),
            _mae(result_sobol.ST, _st_ana),
            n_evals_sobol,
            time_sobol,
        ),
        (
            "eFAST",
            _mae(result_efast.S1, _s1_ana),
            _mae(result_efast.ST, _st_ana),
            n_evals_efast,
            time_efast,
        ),
        (
            "HDMR",
            _mae(result_hdmr.S1, _s1_ana),
            _mae(result_hdmr.ST, _st_ana),
            n_evals_hdmr,
            time_hdmr,
        ),
        (
            "PCE",
            _mae(result_pce.S1, _s1_ana),
            _mae(result_pce.ST, _st_ana),
            n_evals_pce,
            time_pce,
        ),
        (
            "DGSM (bound gap)",
            None,
            _mae(result_dgsm.upper_bound, _st_ana),
            n_evals_dgsm,
            time_dgsm,
        ),
        (
            "Morris (screening)",
            None,
            None,
            n_evals_morris,
            time_morris,
        ),
        (
            "Shapley (Sh)",
            _mae(result_shapley.S1, _s1_ana),
            _mae(result_shapley.ST, _st_ana),
            n_evals_shapley,
            time_shapley,
        ),
        (
            "Borgonovo delta",
            _mae(result_borgonovo.S1, _s1_ana),
            None,
            n_evals_borgonovo,
            time_borgonovo,
        ),
    ]

    _header = "| Method | S1 MAE | ST MAE | N evals | Wall time (s) |\n"
    _sep = "| --- | ---: | ---: | ---: | ---: |\n"
    _body = ""
    for _name, _s1e, _ste, _ne, _wt in _rows:
        _s1_str = f"{_s1e:.4f}" if _s1e is not None else "---"
        _st_str = f"{_ste:.4f}" if _ste is not None else "---"
        _body += f"| {_name} | {_s1_str} | {_st_str} | {_ne:,} | {_wt:.2f} |\n"

    mo.md(
        "### Accuracy and cost comparison\n\n" + _header + _sep + _body + "\n"
        "S1 MAE and ST MAE are mean absolute errors versus the known analytical "
        "Ishigami indices. DGSM provides **bounds** on $S_T$ rather than point "
        "estimates, so the S1 column is not applicable and the ST column reports "
        "the bound gap (MAE of the Poincare upper bound vs analytical $S_T$). "
        "Morris $\\mu^*$ is a screening measure on a different scale entirely "
        "(mean absolute elementary effect, not a variance share), so neither MAE "
        "column applies; see the normalized ranking check above. Its N evals is "
        "the **unique** row count after grid deduplication (100 trajectories = "
        "400 expanded rows). Shapley's S1/ST come from the same PCE surrogate as "
        "the PCE row (same data and order), so those MAEs coincide — its headline "
        "index Sh is checked against the analytical Shapley effects in the "
        "fair-shares chart above. Borgonovo's S1 MAE uses its given-data "
        "first-order estimate; delta itself is moment-independent and on its own "
        "scale, so the ST column does not apply.\n\n"
        "**Timing note:** Sobol, eFAST, and Morris times are end-to-end (sample "
        "+ evaluate + analyze). HDMR, PCE, Shapley, and Borgonovo times are "
        "analyze-only (shared pre-computed samples). DGSM time is analyze-only "
        "(internally evaluates via autodiff)."
    )
    return


@app.cell(hide_code=True)
def _cost_scatter(
    analytical_st,
    n_evals_dgsm,
    n_evals_efast,
    n_evals_hdmr,
    n_evals_pce,
    n_evals_sobol,
    np,
    plt,
    result_dgsm,
    result_efast,
    result_hdmr,
    result_pce,
    result_sobol,
):
    _st_ana = np.asarray(analytical_st)

    def _mae(estimate, truth):
        return float(np.mean(np.abs(np.asarray(estimate) - truth)))

    _methods = ["Sobol", "eFAST", "HDMR", "PCE", "DGSM (bound gap)"]
    _n_evals = [
        n_evals_sobol,
        n_evals_efast,
        n_evals_hdmr,
        n_evals_pce,
        n_evals_dgsm,
    ]
    _mae_st = [
        _mae(result_sobol.ST, _st_ana),
        _mae(result_efast.ST, _st_ana),
        _mae(result_hdmr.ST, _st_ana),
        _mae(result_pce.ST, _st_ana),
        _mae(result_dgsm.upper_bound, _st_ana),
    ]
    _colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    _fig, _ax = plt.subplots(figsize=(7, 5))
    for _m, _n, _e, _c in zip(
        _methods,
        _n_evals,
        _mae_st,
        _colors,
        strict=True,
    ):
        _ax.scatter(
            _n,
            _e,
            s=120,
            color=_c,
            zorder=5,
            edgecolors="white",
            linewidth=1.2,
        )
        _ax.annotate(
            _m,
            (_n, _e),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    _ax.set_xlabel("Model evaluations")
    _ax.set_ylabel("MAE of ST vs analytical")
    _ax.set_title("Cost vs accuracy (total-order indices)")
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _summary(mo):
    mo.md(r"""
    ## When to use each method

    | Method | Best for |
    | --- | --- |
    | **Sobol** | Gold standard for **second-order indices** ($S_2$). |
    | **eFAST** | **Screening** at lower cost ($N \times D$ evals). |
    | **DGSM** | **Differentiable models** via JAX autodiff. |
    | **HDMR** | **Arbitrary (X, Y) data**, no structured sampling. |
    | **PCE** | **Emulation** with a reusable polynomial surrogate (scalar output only). |
    | **Morris** | **Cheapest screening** — factor fixing at $r(D+1)$ evals before dedup. |
    | **Shapley** | **Fair variance shares** summing to 1 — interaction attribution. |
    | **Borgonovo delta** | **Moment-independent** influence on the whole output density. |

    - **Sobol** is the gold standard when you can afford
      $N \times (2D+2)$ evaluations and need pairwise interactions.
    - **eFAST** trades second-order detail for a lower sample
      budget ($N \times D$), making it ideal for factor screening.
    - **DGSM** exploits JAX reverse-mode autodiff to bound $S_T$
      without any structured sampling design.
    - **HDMR** works with any pre-existing (X, Y) dataset and
      builds a B-spline surrogate for sensitivity decomposition.
    - **PCE** fits an orthogonal polynomial surrogate that doubles
      as a fast emulator for downstream prediction tasks.
    - **Morris** ranks parameters for the lowest budget of all —
      64 unique evaluations here after grid deduplication. Its $\mu^*$
      proxies the $S_T$ *ranking* (not the variance shares); use it to
      fix unimportant factors before a variance-based follow-up.
    - **Shapley** splits every interaction's variance fairly among its
      participants, giving a single score per parameter that sums to
      exactly 1 — computed analytically from a surrogate fit at zero
      extra model runs.
    - **Borgonovo delta** looks beyond variance entirely: it measures
      how the full output density shifts when an input is fixed, and
      returns a given-data $S_1$ from the same conditioning for free.
    """)
    return


if __name__ == "__main__":
    app.run()
