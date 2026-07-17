"""eFAST (extended FAST) sensitivity analysis — gsax tutorial.

Demonstrates the eFAST workflow on standard benchmarks: scalar outputs,
multi-output models, and time-series sensitivity evolution.

Run interactively: ``uv run marimo edit examples/efast_gsa.py``
Run as script:     ``uv run python examples/efast_gsa.py``
"""

# ruff: noqa: F722

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # eFAST sensitivity analysis with **gsax**

    The **extended Fourier Amplitude Sensitivity Test** (eFAST) decomposes
    the variance of a model output into contributions from each input
    factor using spectral analysis.  Each input is assigned a characteristic
    frequency along a search curve in the input space; the Fourier power
    spectrum of the model output at that frequency yields the first-order
    index $S_1$, while the complementary spectrum gives the total-order
    index $S_T$.

    Compared to variance-based Sobol methods, eFAST has a lower sample
    cost — $N \times D$ evaluations versus the $N \times (2D + 2)$ of
    Saltelli — at the expense of no second-order indices.  It is a good
    default when pairwise interactions are not needed.

    This notebook walks through four progressively richer use cases:

    1. **Scalar output** — Ishigami benchmark with analytical comparison
    2. **Higher-dimensional** — Sobol G-function with 8 inputs
    3. **Multi-output** — two outputs analysed simultaneously
    4. **Time-series** — sensitivity indices that evolve in time
    """)
    return


@app.cell
def _imports():
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from gsax import Problem, efast
    from gsax.benchmarks import ishigami, sobol_g

    plt.rcParams["figure.dpi"] = 150
    return Problem, efast, ishigami, jnp, mo, np, plt, sobol_g


@app.cell(hide_code=True)
def _ishigami_md(mo):
    mo.md(r"""
    ## Scalar output — Ishigami benchmark

    The Ishigami function $f(x_1, x_2, x_3) = \sin x_1 + 7\sin^2 x_2
    + 0.1\,x_3^4 \sin x_1$ is a standard test case with known analytical
    sensitivity indices.  The three inputs are uniform on $[-\pi, \pi]$.

    The workflow is:

    1. **Sample** — `efast.sample` generates $D$ search curves (one per
       input, each of length `n_per_curve`), stacked into a single
       $(n\_per\_curve \cdot D,\; D)$ matrix carried by an
       `EFASTSamples` object together with `M` and the problem.
    2. **Evaluate** — run the model on all rows of `samples.samples`.
    3. **Analyse** — `efast.analyze(samples, Y)` recovers $S_1$ and $S_T$
       from the Fourier spectrum.

    In the bar chart below, each estimated bar should land on its paler
    analytical twin. Note $x_3$: its $S_1$ is essentially zero yet its
    $S_T$ is not — it influences the output only through its
    interaction with $x_1$, which $S_T$ captures and $S_1$ by
    construction cannot.
    """)
    return


@app.cell
def _ishigami_analysis(efast, ishigami, jnp):
    ishi_problem = ishigami.PROBLEM

    ishi_samples = efast.sample(ishi_problem, n_per_curve=4096, M=4, seed=42)
    Y_ishi = ishigami.evaluate(jnp.asarray(ishi_samples.samples))

    ishi_result = efast.analyze(ishi_samples, Y_ishi)
    print(ishi_result)
    return ishi_problem, ishi_result


@app.cell
def _ishigami_plot(ishi_problem, ishi_result, ishigami, np, plt):
    _names = list(ishi_problem.names)
    _s1 = np.asarray(ishi_result.S1)
    _st = np.asarray(ishi_result.ST)
    _s1_ana = np.asarray(ishigami.ANALYTICAL_S1)
    _st_ana = np.asarray(ishigami.ANALYTICAL_ST)

    _x = np.arange(len(_names))
    _width = 0.2

    fig_ishi, ax_ishi = plt.subplots(figsize=(7.5, 4.5))
    ax_ishi.bar(
        _x - 1.5 * _width,
        _s1,
        _width,
        color="C0",
        label="S1 (eFAST)",
    )
    ax_ishi.bar(
        _x - 0.5 * _width,
        _s1_ana,
        _width,
        color="C0",
        alpha=0.4,
        label="S1 (analytical)",
    )
    ax_ishi.bar(
        _x + 0.5 * _width,
        _st,
        _width,
        color="C3",
        label="ST (eFAST)",
    )
    ax_ishi.bar(
        _x + 1.5 * _width,
        _st_ana,
        _width,
        color="C3",
        alpha=0.4,
        label="ST (analytical)",
    )
    ax_ishi.axhline(0.0, color="black", linewidth=0.5)
    ax_ishi.set_xticks(_x)
    ax_ishi.set_xticklabels(_names)
    ax_ishi.set_ylabel("Sensitivity index")
    ax_ishi.set_title("Ishigami — eFAST vs analytical indices")
    ax_ishi.legend(frameon=False, fontsize=8)
    ax_ishi.grid(axis="y", alpha=0.3)
    fig_ishi.tight_layout()
    fig_ishi
    return


@app.cell(hide_code=True)
def _sobol_g_md(mo):
    mo.md(r"""
    ## Higher-dimensional — Sobol G-function

    The Sobol G-function is an 8-dimensional product whose analytical
    indices are known in closed form.  Inputs with small $a_j$ coefficients
    are important; large $a_j$ suppresses the factor almost entirely.
    With the default coefficients $(0, 1, 4.5, 9, 99, 99, 99, 99)$, only
    the first four inputs carry meaningful sensitivity — the chart below
    should show the last four bars pinned near zero, matching the
    analytical values.
    """)
    return


@app.cell
def _sobol_g_analysis(efast, jnp, sobol_g):
    sg_problem = sobol_g.PROBLEM

    sg_samples = efast.sample(sg_problem, n_per_curve=4096, M=4, seed=123)
    Y_sg = sobol_g.evaluate(jnp.asarray(sg_samples.samples))

    sg_result = efast.analyze(sg_samples, Y_sg)
    print(sg_result)
    return sg_problem, sg_result


@app.cell
def _sobol_g_plot(np, plt, sg_problem, sg_result, sobol_g):
    _names = list(sg_problem.names)
    _s1 = np.asarray(sg_result.S1)
    _st = np.asarray(sg_result.ST)
    _s1_ana = np.asarray(sobol_g.ANALYTICAL_S1)
    _st_ana = np.asarray(sobol_g.ANALYTICAL_ST)

    _x = np.arange(len(_names))
    _width = 0.2

    fig_sg, ax_sg = plt.subplots(figsize=(9.0, 4.5))
    ax_sg.bar(
        _x - 1.5 * _width,
        _s1,
        _width,
        color="C0",
        label="S1 (eFAST)",
    )
    ax_sg.bar(
        _x - 0.5 * _width,
        _s1_ana,
        _width,
        color="C0",
        alpha=0.4,
        label="S1 (analytical)",
    )
    ax_sg.bar(
        _x + 0.5 * _width,
        _st,
        _width,
        color="C3",
        label="ST (eFAST)",
    )
    ax_sg.bar(
        _x + 1.5 * _width,
        _st_ana,
        _width,
        color="C3",
        alpha=0.4,
        label="ST (analytical)",
    )
    ax_sg.axhline(0.0, color="black", linewidth=0.5)
    ax_sg.set_xticks(_x)
    ax_sg.set_xticklabels(_names)
    ax_sg.set_ylabel("Sensitivity index")
    ax_sg.set_title("Sobol G-function — eFAST vs analytical indices")
    ax_sg.legend(frameon=False, fontsize=8)
    ax_sg.grid(axis="y", alpha=0.3)
    fig_sg.tight_layout()
    fig_sg
    return


@app.cell(hide_code=True)
def _multi_output_md(mo):
    mo.md(r"""
    ## Multi-output analysis

    eFAST handles multiple outputs natively.  Pass a 2-D array of shape
    $(N \cdot D,\; K)$ where $K$ is the number of outputs.  Here we stack
    the Ishigami function with a half-amplitude copy to create $K = 2$
    outputs, then analyse both in a single call.

    The result arrays gain a leading output dimension: `S1` and `ST` have
    shape $(K, D)$.
    """)
    return


@app.cell
def _multi_output_analysis(efast, ishigami, jnp):
    multi_problem = ishigami.PROBLEM

    multi_samples = efast.sample(multi_problem, n_per_curve=4096, M=4, seed=7)
    Y_full = ishigami.evaluate(jnp.asarray(multi_samples.samples))
    Y_half = 0.5 * Y_full
    Y_multi = jnp.stack([Y_full, Y_half], axis=-1)  # (n_runs, 2)

    multi_result = efast.analyze(multi_samples, Y_multi)
    print(multi_result)
    return multi_problem, multi_result


@app.cell
def _multi_output_plot(multi_problem, multi_result, np, plt):
    _names = list(multi_problem.names)
    _s1 = np.asarray(multi_result.S1)  # (K, D)
    _st = np.asarray(multi_result.ST)
    _K = _s1.shape[0]

    fig_multi, axes_multi = plt.subplots(
        1,
        _K,
        figsize=(5.5 * _K, 4.5),
        sharey=True,
    )
    _x = np.arange(len(_names))
    _width = 0.38

    _output_labels = ["Ishigami (full)", "Ishigami (half)"]
    for _k in range(_K):
        _ax = axes_multi[_k]
        _ax.bar(
            _x - _width / 2,
            _s1[_k],
            _width,
            color="C0",
            label="S1",
        )
        _ax.bar(
            _x + _width / 2,
            _st[_k],
            _width,
            color="C3",
            label="ST",
        )
        _ax.axhline(0.0, color="black", linewidth=0.5)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_names)
        _ax.set_title(_output_labels[_k])
        _ax.legend(frameon=False, fontsize=8)
        _ax.grid(axis="y", alpha=0.3)

    axes_multi[0].set_ylabel("Sensitivity index")
    fig_multi.suptitle("Multi-output eFAST — per-output bar charts", y=1.02)
    fig_multi.tight_layout()
    fig_multi
    return


@app.cell(hide_code=True)
def _timeseries_md(mo):
    mo.md(r"""
    ## Time-series analysis

    When the model output is a trajectory, eFAST can compute sensitivity
    indices at every time step in a single call.  Pass a 3-D array of
    shape $(N \cdot D,\; T,\; K)$ and the result arrays gain a leading
    time dimension: `S1` and `ST` have shape $(T, K, D)$.

    We use a damped oscillator model
    $$
    y(t) \;=\; A\,e^{-\gamma\,t}\,\sin(\omega\,t)
    $$
    with three uncertain parameters: amplitude $A$, damping $\gamma$,
    and angular frequency $\omega$.  Early in the trajectory, the
    amplitude controls the envelope; at later times, the damping rate
    determines how quickly the signal decays.
    """)
    return


@app.cell
def _timeseries_model(Problem, jnp):
    ts_problem = Problem.from_dict(
        {
            "amplitude": (0.5, 2.0),
            "damping": (0.1, 0.5),
            "frequency": (1.0, 5.0),
        }
    )

    ts_times = jnp.linspace(0.1, 8.0, 50)

    def damped_oscillator(X, ts):
        """Damped oscillator: y(t) = A * exp(-gamma * t) * sin(omega * t)."""
        amplitude = X[:, 0:1]
        gamma = X[:, 1:2]
        omega = X[:, 2:3]
        return amplitude * jnp.exp(-gamma * ts[None, :]) * jnp.sin(omega * ts[None, :])

    return damped_oscillator, ts_problem, ts_times


@app.cell
def _timeseries_analysis(damped_oscillator, efast, jnp, ts_problem, ts_times):
    ts_samples = efast.sample(ts_problem, n_per_curve=4096, M=4, seed=99)
    Y_ts = damped_oscillator(jnp.asarray(ts_samples.samples), ts_times)
    Y_ts = Y_ts[..., None]  # (n_runs, T, 1)

    ts_result = efast.analyze(ts_samples, Y_ts)
    print(ts_result)
    return (ts_result,)


@app.cell
def _timeseries_plot(np, plt, ts_problem, ts_result, ts_times):
    _names = list(ts_problem.names)
    _s1 = np.asarray(ts_result.S1[:, 0, :])  # (T, D)
    _st = np.asarray(ts_result.ST[:, 0, :])
    _t_np = np.asarray(ts_times)

    _colors = ("C0", "C2", "C3")
    _markers = ("o", "s", "^")
    _skip = max(1, len(_t_np) // 10)

    fig_ts, axes_ts = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    for _d, (_name, _color, _marker) in enumerate(
        zip(_names, _colors, _markers, strict=True),
    ):
        axes_ts[0].plot(
            _t_np,
            _s1[:, _d],
            color=_color,
            linewidth=1.6,
            label=_name,
        )
        axes_ts[0].plot(
            _t_np[::_skip],
            _s1[::_skip, _d],
            color=_color,
            marker=_marker,
            linestyle="none",
            markersize=5,
        )
        axes_ts[1].plot(
            _t_np,
            _st[:, _d],
            color=_color,
            linewidth=1.6,
            label=_name,
        )
        axes_ts[1].plot(
            _t_np[::_skip],
            _st[::_skip, _d],
            color=_color,
            marker=_marker,
            linestyle="none",
            markersize=5,
        )

    axes_ts[0].set_title("First-order S1(t)")
    axes_ts[1].set_title("Total-order ST(t)")
    for _ax in axes_ts:
        _ax.set_xlabel("t")
        _ax.set_ylim(-0.05, 1.1)
        _ax.grid(alpha=0.3)
        _ax.legend(frameon=False)
    axes_ts[0].set_ylabel("Sensitivity index")
    fig_ts.suptitle("Damped oscillator — time-resolved eFAST", y=1.02)
    fig_ts.tight_layout()
    fig_ts
    return


@app.cell(hide_code=True)
def _xarray_md(mo):
    mo.md(r"""
    ## xarray export

    Every `EFASTResult` can be exported to an `xarray.Dataset` via
    `to_dataset()`.  The dataset contains `S1` and `ST` as data variables,
    with dimension coordinates inferred from the problem definition.
    For time-series results, pass explicit time coordinates.
    """)
    return


@app.cell
def _xarray_export(ishi_result, np, ts_result, ts_times):
    ds_scalar = ishi_result.to_dataset()
    print("Scalar dataset:\n", ds_scalar, "\n")

    ds_ts = ts_result.to_dataset(time_coords=np.asarray(ts_times))
    print("Time-series dataset:\n", ds_ts)
    return


@app.cell(hide_code=True)
def _outro(mo):
    mo.md(r"""
    ## Summary

    1. **Low sample cost.** eFAST uses `n_per_curve` $\times D$ model
       evaluations versus Saltelli's $N \times (2D + 2)$, making it
       attractive for expensive simulators.
    2. **Same API shape.** `efast.sample` + `efast.analyze` mirror the
       Sobol workflow; switching between the two is a one-line change.
    3. **Multi-output and time-series native.** Passing higher-dimensional
       output arrays adds output or time axes to the result automatically.
    4. **xarray integration.** `to_dataset()` produces labelled, self-
       describing datasets ready for downstream analysis or serialisation.
    """)
    return


if __name__ == "__main__":
    app.run()
