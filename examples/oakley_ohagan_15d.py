"""Oakley & O'Hagan 15-D benchmark — high-dimensional SA with Gaussian inputs.

Demonstrates three sensitivity analysis methods on a 15-dimensional
benchmark with non-uniform (Gaussian) inputs: eFAST, RS-HDMR, and DGSM.
Compares computed indices against the known analytical values.

Run interactively: ``uv run marimo edit examples/oakley_ohagan_15d.py``
Run as script:     ``uv run python examples/oakley_ohagan_15d.py``
"""

# ruff: noqa: F722

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # High-dimensional SA with Gaussian inputs: Oakley & O'Hagan benchmark

    The Oakley & O'Hagan (2004) function is a **15-dimensional** benchmark
    with **Gaussian inputs** and a rich interaction structure.  It tests
    whether SA methods can identify the few dominant parameters among many,
    and handle non-uniform input distributions.

    The function combines linear, trigonometric, and quadratic terms:

    $$
    f(\mathbf{x}) = \mathbf{a}_1^\top \mathbf{x}
        + \mathbf{a}_2^\top \sin(\mathbf{x})
        + \mathbf{a}_3^\top \cos(\mathbf{x})
        + \mathbf{x}^\top M \mathbf{x}
    $$

    with $x_i \sim \mathcal{N}(0, 1)$ i.i.d. for all 15 inputs.

    We compare three methods that support Gaussian inputs:

    1. **eFAST** — Fourier-based indices via CDF mapping
    2. **RS-HDMR** — B-spline surrogate from random samples
    3. **DGSM** — derivative-based bounds via autodiff
    """)
    return


@app.cell
def _imports():
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import gsax
    from gsax.benchmarks import oakley_ohagan

    plt.rcParams["figure.dpi"] = 150
    return gsax, jax, jnp, mo, np, oakley_ohagan, plt


@app.cell(hide_code=True)
def _problem_md(mo):
    mo.md(r"""
    ## Problem setup

    The benchmark defines 15 Gaussian inputs $x_i \sim \mathcal{N}(0, 1)$.
    The coefficient magnitudes increase with the index, creating a natural
    importance gradient:

    - **x1--x5**: nearly inert (small coefficients)
    - **x6--x10**: intermediate importance
    - **x11--x15**: dominant parameters

    The quadratic form $\mathbf{x}^\top M \mathbf{x}$ introduces pairwise
    interactions, so $S_{T_i} > S_{1_i}$ for most parameters.
    """)
    return


@app.cell
def _problem_setup(np, oakley_ohagan):
    problem = oakley_ohagan.PROBLEM
    s1_ana = np.asarray(oakley_ohagan.ANALYTICAL_S1)
    st_ana = np.asarray(oakley_ohagan.ANALYTICAL_ST)
    names = list(problem.names)

    print(f"Dimensions: {problem.num_vars}")
    print("Distribution: Gaussian N(0, 1) for all inputs")
    print(f"\nAnalytical S1:  {np.array2string(s1_ana, precision=4)}")
    print(f"Analytical ST:  {np.array2string(st_ana, precision=4)}")
    print(f"\nSum S1 = {s1_ana.sum():.4f}  (< 1 indicates interactions)")
    return names, problem, s1_ana, st_ana


@app.cell(hide_code=True)
def _analytical_plot(names, np, plt, s1_ana, st_ana):
    _sort_idx = np.argsort(st_ana)
    _names_sorted = [names[i] for i in _sort_idx]
    _s1_sorted = s1_ana[_sort_idx]
    _st_sorted = st_ana[_sort_idx]

    _y = np.arange(len(_names_sorted))
    _height = 0.35

    # Colour by importance tier
    _colors_st = []
    _colors_s1 = []
    for _val in _st_sorted:
        if _val > 0.08:
            _colors_st.append("#E53935")  # high
            _colors_s1.append("#1E88E5")
        elif _val > 0.02:
            _colors_st.append("#FB8C00")  # medium
            _colors_s1.append("#42A5F5")
        else:
            _colors_st.append("#BDBDBD")  # low
            _colors_s1.append("#90CAF9")

    _fig, _ax = plt.subplots(figsize=(9, 6))
    _ax.barh(
        _y + _height / 2,
        _st_sorted,
        _height,
        color=_colors_st,
        label="$S_T$ (total-order)",
        edgecolor="white",
        linewidth=0.5,
    )
    _ax.barh(
        _y - _height / 2,
        _s1_sorted,
        _height,
        color=_colors_s1,
        label="$S_1$ (first-order)",
        edgecolor="white",
        linewidth=0.5,
    )
    _ax.set_yticks(_y)
    _ax.set_yticklabels(_names_sorted)
    _ax.set_xlabel("Sensitivity index")
    _ax.set_title("Analytical Sobol indices — Oakley & O'Hagan (2004)")
    _ax.legend(frameon=False)
    _ax.grid(axis="x", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _efast_md(mo):
    mo.md(r"""
    ## eFAST analysis

    eFAST maps uniform search-curve samples to the Gaussian input space
    via the inverse CDF, so it works seamlessly with non-uniform
    distributions.  We use $N = 4096$ samples per search curve with
    interference factor $M = 4$.
    """)
    return


@app.cell
def _efast_analysis(gsax, jnp, oakley_ohagan, problem):
    X_ef = gsax.sample_efast(problem, N=4096, M=4, seed=42)
    Y_ef = oakley_ohagan.evaluate(jnp.asarray(X_ef))

    efast_result = gsax.analyze_efast(problem, Y_ef, M=4)
    print(efast_result)
    return (efast_result,)


@app.cell(hide_code=True)
def _efast_plot(efast_result, names, np, plt, s1_ana, st_ana):
    _s1_ef = np.asarray(efast_result.S1)
    _st_ef = np.asarray(efast_result.ST)
    _sort_idx = np.argsort(st_ana)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    _y = np.arange(len(names))
    _height = 0.35

    _names_sorted = [names[i] for i in _sort_idx]

    # S1 panel
    _ax1.barh(
        _y + _height / 2,
        _s1_ef[_sort_idx],
        _height,
        color="#1E88E5",
        label="eFAST",
    )
    _ax1.barh(
        _y - _height / 2,
        s1_ana[_sort_idx],
        _height,
        color="#1E88E5",
        alpha=0.35,
        label="Analytical",
    )
    _ax1.set_yticks(_y)
    _ax1.set_yticklabels(_names_sorted)
    _ax1.set_xlabel("$S_1$")
    _ax1.set_title("First-order indices")
    _ax1.legend(frameon=False, fontsize=8)
    _ax1.grid(axis="x", alpha=0.3)

    # ST panel
    _ax2.barh(
        _y + _height / 2,
        _st_ef[_sort_idx],
        _height,
        color="#E53935",
        label="eFAST",
    )
    _ax2.barh(
        _y - _height / 2,
        st_ana[_sort_idx],
        _height,
        color="#E53935",
        alpha=0.35,
        label="Analytical",
    )
    _ax2.set_xlabel("$S_T$")
    _ax2.set_title("Total-order indices")
    _ax2.legend(frameon=False, fontsize=8)
    _ax2.grid(axis="x", alpha=0.3)

    _fig.suptitle("eFAST vs analytical — Oakley & O'Hagan", y=1.01)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _hdmr_md(mo):
    mo.md(r"""
    ## RS-HDMR analysis

    RS-HDMR works from **arbitrary samples** — no structured design needed.
    We draw 3000 i.i.d. Gaussian samples and fit a B-spline surrogate with
    second-order interactions (`maxorder=2`).
    """)
    return


@app.cell
def _hdmr_analysis(gsax, jax, jnp, oakley_ohagan, problem):
    _key = jax.random.key(0)
    X_hd = jax.random.normal(_key, (3000, 15))
    Y_hd = oakley_ohagan.evaluate(jnp.asarray(X_hd))

    hdmr_result = gsax.analyze_hdmr(problem, X_hd, Y_hd, maxorder=2, m=2)
    print(hdmr_result)
    return (hdmr_result,)


@app.cell(hide_code=True)
def _dgsm_md(mo):
    mo.md(r"""
    ## DGSM analysis

    DGSM uses JAX autodiff to compute partial derivatives and derives
    **upper and lower bounds** on the total Sobol index $S_T$.  For
    Gaussian inputs the Poincare constant is $C_i = \sigma^2 = 1$,
    which gives tighter bounds than uniform distributions.

    We use `sample_mc` to generate i.i.d. samples from the Gaussian
    input distributions.
    """)
    return


@app.cell
def _dgsm_fn(jnp, oakley_ohagan):
    def oakley_fn(x):
        """Unbatched Oakley & O'Hagan: (15,) -> ()."""
        return oakley_ohagan.evaluate(x[None, :])[0]

    # Verify it works on a single point
    _test = oakley_fn(jnp.zeros(15))
    print(f"f(0) = {_test:.4f}")
    return (oakley_fn,)


@app.cell
def _dgsm_analysis(gsax, jnp, oakley_fn, problem):
    X_dg = gsax.sample_mc(problem, N=10000, seed=42)
    dgsm_result = gsax.analyze_dgsm(problem, oakley_fn, jnp.asarray(X_dg))
    print(dgsm_result)
    return (dgsm_result,)


@app.cell(hide_code=True)
def _dgsm_plot(dgsm_result, names, np, plt, st_ana):
    _ub = np.asarray(dgsm_result.upper_bound)
    _lb = np.asarray(dgsm_result.lower_bound)
    _sort_idx = np.argsort(st_ana)
    _names_sorted = [names[i] for i in _sort_idx]
    _y = np.arange(len(names))
    _bw = 0.25

    _fig, _ax = plt.subplots(figsize=(9, 6))
    _ax.barh(
        _y - _bw,
        _lb[_sort_idx],
        _bw,
        label="Lower bound",
        color="#2196F3",
        alpha=0.85,
    )
    _ax.barh(
        _y,
        st_ana[_sort_idx],
        _bw,
        label="Analytical $S_T$",
        color="#4CAF50",
        alpha=0.85,
    )
    _ax.barh(
        _y + _bw,
        _ub[_sort_idx],
        _bw,
        label="Upper bound",
        color="#FF9800",
        alpha=0.85,
    )
    _ax.set_yticks(_y)
    _ax.set_yticklabels(_names_sorted)
    _ax.set_xlabel("Sensitivity index")
    _ax.set_title("DGSM bounds vs analytical $S_T$ — Oakley & O'Hagan")
    _ax.legend(frameon=False)
    _ax.grid(axis="x", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _comparison_md(mo):
    mo.md(r"""
    ## Method comparison

    Side-by-side comparison of first-order ($S_1$) estimates from all
    three methods against the analytical values.  Parameters are sorted
    by importance (analytical $S_T$).
    """)
    return


@app.cell(hide_code=True)
def _comparison_plot(
    efast_result,
    hdmr_result,
    names,
    np,
    plt,
    s1_ana,
    st_ana,
):
    _sort_idx = np.argsort(st_ana)
    _names_sorted = [names[i] for i in _sort_idx]
    _y = np.arange(len(names))
    _height = 0.2

    _s1_ef = np.asarray(efast_result.S1)[_sort_idx]
    _s1_hd = np.asarray(hdmr_result.S1)[_sort_idx]
    _s1_ana = s1_ana[_sort_idx]

    _fig, _ax = plt.subplots(figsize=(10, 7))
    _ax.barh(
        _y - 1.5 * _height,
        _s1_ana,
        _height,
        color="#9E9E9E",
        label="Analytical",
    )
    _ax.barh(
        _y - 0.5 * _height,
        _s1_ef,
        _height,
        color="#1E88E5",
        label="eFAST",
    )
    _ax.barh(
        _y + 0.5 * _height,
        _s1_hd,
        _height,
        color="#43A047",
        label="RS-HDMR",
    )
    # DGSM does not produce S1 estimates — only ST bounds

    _ax.set_yticks(_y)
    _ax.set_yticklabels(_names_sorted)
    _ax.set_xlabel("$S_1$ (first-order index)")
    _ax.set_title("Method comparison — first-order indices")
    _ax.legend(frameon=False)
    _ax.grid(axis="x", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _ranking_md(mo):
    mo.md(r"""
    ## Ranking accuracy

    Do the methods correctly identify the **top 5** most important
    parameters by total-order index?  The analytical ranking places
    x11--x15 at the top.
    """)
    return


@app.cell(hide_code=True)
def _ranking_table(
    dgsm_result,
    efast_result,
    hdmr_result,
    mo,
    names,
    np,
    st_ana,
):
    _top5_ana = set(np.argsort(st_ana)[-5:])

    _st_ef = np.asarray(efast_result.ST)
    _top5_ef = set(np.argsort(_st_ef)[-5:])
    _match_ef = len(_top5_ana & _top5_ef)

    _st_hd = np.asarray(hdmr_result.ST)
    _top5_hd = set(np.argsort(_st_hd)[-5:])
    _match_hd = len(_top5_ana & _top5_hd)

    _ub_dg = np.asarray(dgsm_result.upper_bound)
    _top5_dg = set(np.argsort(_ub_dg)[-5:])
    _match_dg = len(_top5_ana & _top5_dg)

    def _fmt_set(idx_set):
        return ", ".join(names[i] for i in sorted(idx_set))

    _rows = [
        f"| Analytical | {_fmt_set(_top5_ana)} | -- |",
        f"| eFAST ($S_T$) | {_fmt_set(_top5_ef)} | {_match_ef}/5 |",
        f"| RS-HDMR ($S_T$) | {_fmt_set(_top5_hd)} | {_match_hd}/5 |",
        f"| DGSM (upper bound) | {_fmt_set(_top5_dg)} | {_match_dg}/5 |",
    ]
    _table = (
        "| Method | Top-5 parameters | Match |\n"
        "| --- | --- | ---: |\n"
        + "\n".join(_rows)
    )
    mo.md("### Top-5 ranking accuracy\n\n" + _table)
    return


@app.cell(hide_code=True)
def _summary(mo):
    mo.md(r"""
    ## Summary

    All three methods handle the 15-dimensional Gaussian-input benchmark:

    - **eFAST** provides direct $S_1$ and $S_T$ estimates that closely
      match the analytical values.  It requires structured search-curve
      samples ($N \times D$ evaluations) but maps them through the
      Gaussian inverse CDF automatically.

    - **RS-HDMR** works from arbitrary i.i.d. samples, making it the
      most flexible.  With 3000 samples and second-order B-spline
      fitting, it captures the main effects and the dominant interactions.

    - **DGSM** is the cheapest computationally: one reverse-mode autodiff
      pass gives all 15 partial derivatives per sample.  It produces
      **bounds** on $S_T$ rather than exact indices, but for screening
      (identifying which parameters matter) this is often sufficient.
      For Gaussian inputs the Poincare constant $C_i = \sigma^2$ yields
      tighter upper bounds than uniform distributions.

    The Oakley & O'Hagan benchmark demonstrates that gsax's methods
    scale comfortably to moderate dimensionality and handle non-uniform
    input distributions without special configuration.
    """)
    return


if __name__ == "__main__":
    app.run()
