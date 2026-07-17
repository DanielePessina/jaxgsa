"""Shapley-effect sensitivity analysis — gsax tutorial.

Demonstrates the Shapley-effects workflow on the Ishigami benchmark:
Monte Carlo sampling, the default PCE backend, the S1 <= Sh <= ST
bracketing against the analytical solution, surrogate order as a
convergence knob with the explained_variance guardrail, and the HDMR
backend on a multi-output model.

Run interactively: ``uv run marimo edit examples/shapley_gsa.py``
Run as script:     ``uv run python examples/shapley_gsa.py``
"""

# ruff: noqa: F722

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # Shapley effects with **gsax**

    **Shapley effects** (Owen 2014; Song, Nelson & Staum 2016) allocate
    the output variance *fairly*: each interaction's variance is split
    equally among its participants,

    $$
    \mathrm{Sh}_i \;=\; \sum_{u \ni i} \frac{V_u}{|u|}
    $$

    where the $V_u$ are the partial variances of the Hoeffding–Sobol'
    decomposition.  gsax computes them **analytically** from a fitted
    surrogate — PCE (default) or RS-HDMR — with no permutation Monte
    Carlo and no extra model runs.  The result carries four quantities:

    | Quantity | Meaning |
    | --- | --- |
    | $\mathrm{Sh}$ | fair variance share — sums to exactly 1 (Shapley efficiency) |
    | $S_1$ | first-order index from the same surrogate (main effect only) |
    | $S_T$ | total-order index from the same surrogate (interactions in full) |
    | `explained_variance` | $\sum_u V_u / \mathrm{Var}(Y)$ — surrogate fit quality |

    Under independent inputs $S_{1,i} \le \mathrm{Sh}_i \le S_{T,i}$.
    This notebook walks through:

    1. **PCE-backend analysis** on the Ishigami benchmark
    2. The **$S_1 \le \mathrm{Sh} \le S_T$ bracketing** vs the analytical solution
    3. **Surrogate order** as a convergence knob, with the
       `explained_variance` guardrail
    4. The **HDMR backend** on a multi-output model
    """)
    return


@app.cell
def _imports():
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import gsax
    from gsax.benchmarks import ishigami

    plt.rcParams["figure.dpi"] = 150
    return gsax, ishigami, jnp, mo, np, plt


@app.cell(hide_code=True)
def _ishigami_md(mo):
    mo.md(r"""
    ## PCE-backend analysis — Ishigami benchmark

    The Ishigami function $f(x_1, x_2, x_3) = \sin x_1 + 7\sin^2 x_2
    + 0.1\,x_3^4 \sin x_1$ has three inputs uniform on $[-\pi, \pi]$ and
    a single interaction term — the $x_1$–$x_3$ pair.  The workflow is
    the *given-data* one shared by HDMR, PCE, HSIC, PAWN, and Borgonovo
    — it consumes whatever $(X, Y)$ data you already have, with no
    method-specific sampling design:

    1. **Sample** — here `gsax.sampling.monte_carlo` draws plain Monte Carlo
       points, but any sampling scheme would do.
    2. **Evaluate** — run the model on the samples.
    3. **Analyse** — fit PCE, then call `result.shapley()`
       and reads the Shapley allocation off its variance decomposition.

    The default `backend="pce"` is exact *within the fitted
    polynomial*.  Ishigami's sines need a degree-8 polynomial — the
    default `order=3` under-fits here (explained variance
    $\approx 0.47$) and would trip the fit warning shown later.
    """)
    return


@app.cell
def _ishigami_analysis(gsax, ishigami, jnp):
    shapley_problem = ishigami.PROBLEM

    X = jnp.asarray(gsax.sampling.monte_carlo(shapley_problem, n=2000, seed=42))
    Y = ishigami.evaluate(X)

    result = gsax.pce.analyze(shapley_problem, X, Y, order=8).shapley()

    print("Sh:", result.Sh)
    print("S1:", result.S1)
    print("ST:", result.ST)
    print("explained_variance:", float(result.explained_variance))
    print("effective order:", result.order)
    return X, Y, result, shapley_problem


@app.cell
def _efficiency_check(ishigami, np, result):
    _sh = np.asarray(result.Sh)
    _sh_analytical = np.asarray(ishigami.ANALYTICAL_SHAPLEY)

    print(f"sum(Sh) = {_sh.sum():.6f}   (Shapley efficiency: exactly 1)")
    assert np.isclose(_sh.sum(), 1.0, atol=1e-4)

    print("estimated: ", np.round(_sh, 4))
    print("analytical:", np.round(_sh_analytical, 4))
    print(f"max |error| = {np.max(np.abs(_sh - _sh_analytical)):.4f}")
    assert np.allclose(_sh, _sh_analytical, atol=0.01)
    return


@app.cell(hide_code=True)
def _bracketing_md(mo):
    mo.md(r"""
    ## The $S_1 \le \mathrm{Sh} \le S_T$ bracketing

    $S_1$ omits interactions ($\sum_i S_{1,i} \le 1$) while $S_T$ counts
    each interaction once per participant ($\sum_i S_{T,i} \ge 1$); the
    Shapley effect splits every interaction fairly and lands in between.
    Ishigami has exactly one interaction, the $x_1$–$x_3$ variance
    $V_{13}$, shared half-and-half — so $\mathrm{Sh} = (S_1 + S_T)/2$
    holds elementwise here.  Watch $x_3$: its first-order index is
    exactly zero, yet its Shapley effect is clearly positive because it
    owns half of $V_{13}$.
    """)
    return


@app.cell
def _bracketing_plot(ishigami, np, plt, result, shapley_problem):
    _names = list(shapley_problem.names)
    _x = np.arange(len(_names))
    _width = 0.26

    _s1 = np.asarray(result.S1)
    _sh = np.asarray(result.Sh)
    _st = np.asarray(result.ST)

    fig_bracket, ax_bracket = plt.subplots(figsize=(7.5, 4.5))
    ax_bracket.bar(_x - _width, _s1, _width, color="C0", alpha=0.85, label=r"$S_1$")
    ax_bracket.bar(_x, _sh, _width, color="C2", alpha=0.85, label=r"$\mathrm{Sh}$")
    ax_bracket.bar(_x + _width, _st, _width, color="C3", alpha=0.85, label=r"$S_T$")
    ax_bracket.scatter(
        _x,
        np.asarray(ishigami.ANALYTICAL_SHAPLEY),
        marker="D",
        color="black",
        s=60,
        zorder=5,
        label="analytical Sh",
    )
    ax_bracket.set_xticks(_x)
    ax_bracket.set_xticklabels(_names)
    ax_bracket.set_ylabel("index value")
    ax_bracket.set_title(r"Ishigami — $S_1 \leq \mathrm{Sh} \leq S_T$ (PCE backend, order 8)")
    ax_bracket.legend(frameon=False, fontsize=8)
    ax_bracket.grid(axis="y", alpha=0.3)
    fig_bracket.tight_layout()
    fig_bracket
    return


@app.cell(hide_code=True)
def _order_md(mo):
    mo.md(r"""
    ## Surrogate order and the `explained_variance` guardrail

    Interactions the surrogate cannot represent are simply absent from
    the allocation, and $\mathrm{Sh}$ *always* sums to 1 — so the
    fit-quality signal lives in `explained_variance`
    ($\sum_u V_u / \mathrm{Var}(Y)$), reported separately rather than
    silently renormalized away.  gsax emits a `UserWarning` when it
    drops below 0.5 (much of $\mathrm{Var}(Y)$ unexplained — watch the
    low-order runs below) or exceeds 1.3 (an overfit surrogate
    over-counting variance).  Sweeping the polynomial order shows the
    shares converging onto the analytical values as the fit improves.
    """)
    return


@app.cell
def _order_sweep(X, Y, gsax, ishigami, np, shapley_problem):
    orders = (2, 3, 4, 6, 8)
    sweep = {}
    for _order in orders:
        _r = gsax.pce.analyze(shapley_problem, X, Y, order=_order).shapley()
        sweep[_order] = _r
        _err = float(np.max(np.abs(np.asarray(_r.Sh) - ishigami.ANALYTICAL_SHAPLEY)))
        print(
            f"order={_order}  explained_variance={float(_r.explained_variance):.3f}"
            f"  max |Sh - analytical| = {_err:.4f}"
        )
    return orders, sweep


@app.cell
def _order_plot(ishigami, np, orders, plt, shapley_problem, sweep):
    _names = list(shapley_problem.names)
    _sh_by_order = np.stack([np.asarray(sweep[_o].Sh) for _o in orders])
    _analytical = np.asarray(ishigami.ANALYTICAL_SHAPLEY)

    fig_order, ax_order = plt.subplots(figsize=(7.5, 4.5))
    _colors = ("C0", "C2", "C3")
    for _d, (_name, _color) in enumerate(zip(_names, _colors, strict=True)):
        ax_order.plot(
            orders,
            _sh_by_order[:, _d],
            marker="o",
            color=_color,
            label=_name,
        )
        ax_order.axhline(
            _analytical[_d],
            color=_color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )
    ax_order.set_xticks(list(orders))
    ax_order.set_xlabel("PCE order")
    ax_order.set_ylabel(r"$\mathrm{Sh}$")
    ax_order.set_title(r"$\mathrm{Sh}$ convergence with surrogate order (dashed = analytical)")
    ax_order.legend(frameon=False, fontsize=8)
    ax_order.grid(alpha=0.3)
    fig_order.tight_layout()
    fig_order
    return


@app.cell(hide_code=True)
def _hdmr_md(mo):
    mo.md(r"""
    ## HDMR backend — multi-output Y

    Both backends accept multi-output `(N, K)` and time-series
    `(N, T, K)` outputs.  `backend="hdmr"` fits the RS-HDMR B-spline
    surrogate instead and uses its structural (ANCOVA) component
    variances as the $V_u$ — the machinery built for correlated-input
    separation.  The second output below,
    $\sum_i x_i^2$, is purely additive — no interactions to share — so
    its three indices coincide: $S_1 = \mathrm{Sh} = S_T = 1/3$ per
    input.  Each output row of $\mathrm{Sh}$ still sums to exactly 1.
    """)
    return


@app.cell
def _hdmr_multi(X, Y, gsax, jnp, shapley_problem):
    Y_multi = jnp.column_stack([Y, jnp.sum(X**2, axis=1)])  # (N, K=2)

    result_hdmr = gsax.hdmr.analyze(shapley_problem, X, Y_multi).shapley()

    print("Sh shape:", result_hdmr.Sh.shape)  # (K, D) = (2, 3)
    print("row sums:", result_hdmr.Sh.sum(axis=-1))  # each exactly 1
    print("explained_variance:", result_hdmr.explained_variance)  # (K,)
    print(result_hdmr.to_dataset())
    return


@app.cell(hide_code=True)
def _interpretation(mo):
    mo.md(r"""
    ## Interpretation

    - **The shares are fair and complete.**  $\mathrm{Sh} \approx
      (0.44, 0.44, 0.12)$ sums to exactly 1: $x_1$ receives its main
      effect plus half the $x_1$–$x_3$ interaction variance, $x_2$ is
      purely additive ($S_1 = \mathrm{Sh} = S_T$), and $x_3$ receives
      the other half of $V_{13}$.
    - **$x_3$ is invisible to $S_1$ but not to Sh.**  Its first-order
      index is exactly zero — it acts only through the interaction — yet
      its fair share is $\approx 0.12$.  $S_1$ would dismiss it, $S_T$
      double-counts the interaction across $x_1$ and $x_3$; Shapley
      attributes it once, split between the participants.
    - **`explained_variance` is the honest fit signal.**  The shares sum
      to 1 at *every* order, even when a degree-2 polynomial captures a
      fraction of $\mathrm{Var}(Y)$ — the sweep shows the warning firing
      below 0.5 and the shares converging as the fit improves.  Always
      check it before trusting the allocation.
    - **Backend choice.**  Both backends handle scalar, multi-output,
      and time-series `Y`.  `"pce"` (default) is exact within the fitted
      polynomial; `"hdmr"` truncates at `maxorder` and remains the
      choice when you want the B-spline surrogate's ANCOVA
      (correlated-input) separation.  Both assume independent inputs
      (a v1 limitation).
    """)
    return


if __name__ == "__main__":
    app.run()
