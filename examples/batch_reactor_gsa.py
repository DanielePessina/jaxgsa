"""Batch reactor mechanistic model — Sobol global sensitivity analysis with gsax.

Walks through Sobol global sensitivity analysis on a batch reactor running a
first-order liquid-phase reaction A -> B. The rate constant k(T, pH) combines
a centred Arrhenius temperature dependence with a Hill-type pH saturation
curve, and the inlet concentration Ca0 enters linearly through the mass
balance. The mechanistic model is treated as already fitted: the notebook is
about variance attribution across the operating envelope, not parameter
estimation.

Run interactively: ``uv run marimo edit examples/batch_reactor_gsa.py``
Run as script:     ``uv run python examples/batch_reactor_gsa.py``
"""

# ruff: noqa: F722

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # Batch reactor sensitivity analysis with **gsax**

    A batch reactor running a first-order liquid-phase reaction
    $A \to B$. The rate constant $k(T, \mathrm{pH})$ combines a centred
    Arrhenius temperature dependence with a Hill-type pH saturation
    curve, and the inlet concentration $C_{A,0}$ feeds the mass
    balance directly. We treat the mechanistic model as already
    fitted — the notebook asks the classical global-sensitivity question
    instead: across the operating envelope, how much of the variance in
    the outlet concentration is explained by each input, and which pairs
    interact?

    The batch reactor mass balance for the reactant is

    $$
    \frac{dC_A}{dt} \;=\; \frac{1}{\tau}\,(C_{A,0} - C_A)
        \;-\; k(T, \mathrm{pH})\,C_A,
    $$

    where $\tau = V/q$ is the residence time. Starting from a clean
    reactor at $C_A(0) = 0$, the closed-form solution is

    $$
    C_A(t) \;=\; \frac{C_{A,0}}{1 + k\tau}\,
        \bigl(1 - e^{-(1/\tau + k)\,t}\bigr).
    $$

    The notebook treats $C_A(t)$ as a time-series multi-output, runs
    Sobol with bootstrap resampling **once**, and reads the indices off
    as bar plots, a time-resolved sensitivity profile, and a pairwise
    interaction heatmap.
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

    return gsax, jax, jnp, mo, np, plt


@app.cell(hide_code=True)
def _model_md(mo):
    mo.md(r"""
    ## Mechanistic model

    The kinetics are factored into a temperature term and a pH term and
    multiplied together. Concretely,

    $$
    k(T, \mathrm{pH}) \;=\; k_{\mathrm{sat}}(\mathrm{pH}) \,
        \exp\!\Big(-\tfrac{E_a}{R}\Big(\tfrac{1}{T_K} - \tfrac{1}{T_{\mathrm{ref}}}\Big)\Big),
    \qquad
    k_{\mathrm{sat}}(\mathrm{pH}) \;=\; b + \frac{a}{1 + (\mathrm{pH}/\mathrm{pH}_{50})^n}.
    $$

    Centring at $T_{\mathrm{ref}} = 298.15\,\mathrm{K}$ keeps
    $k(T_{\mathrm{ref}}, \cdot) = k_{\mathrm{sat}}(\cdot)$, so the pH
    curve has a clean physical meaning at room temperature. An
    activation energy of $E_a = 30\,\mathrm{kJ/mol}$ produces roughly a
    two-fold rate change per ten Kelvin around the centring point. The
    residence time is fixed at $\tau = 2$ in dimensionless time units,
    chosen so that the half-life of the start-up transient is
    comparable to the kinetic e-folding scale across the operating box.
    """)
    return


@app.cell
def _model(jnp):
    T_REF = 298.15
    R_GAS = 8.314e-3
    EA = 30.0

    K_BASELINE = 0.14
    K_AMPLITUDE = 1.05
    PH50 = 5.85
    HILL = 5.0

    TAU = 2.0

    def k_rate(temperature_C, pH):
        T_K = temperature_C + 273.15
        arrhenius = jnp.exp(-EA / R_GAS * (1.0 / T_K - 1.0 / T_REF))
        sat = K_BASELINE + K_AMPLITUDE / (1.0 + (pH / PH50) ** HILL)
        return sat * arrhenius

    def batch_reactor_trajectory(Ca0, temperature_C, pH, ts):
        """Closed-form batch reactor concentration starting from Ca(0) = 0."""
        k = k_rate(temperature_C, pH)
        Ca_ss = Ca0 / (1.0 + k * TAU)
        decay = jnp.exp(-(1.0 / TAU + k) * ts)
        return Ca_ss * (1.0 - decay)

    return (batch_reactor_trajectory,)


@app.cell(hide_code=True)
def _doe_md(mo):
    mo.md(r"""
    ## Problem and Saltelli sampling

    The three inputs are uniform on physically reasonable boxes:

    | input | range | units |
    |---|---|---|
    | $C_{A,0}$ | $[0.75,\, 1.5]$ | mol / L |
    | $T$ | $[15,\, 35]$ | °C |
    | $\mathrm{pH}$ | $[4.5,\, 7.5]$ | — |

    `gsax.sample(...)` returns a `SamplingResult` whose `.samples`
    attribute is the deduplicated unique-row sample matrix; gsax
    reconstructs the expanded Saltelli ordering internally inside
    `analyze`. Setting `calc_second_order=True` activates the extra
    cross-matrix needed for $S_{ij}$.
    """)
    return


@app.cell
def _problem(gsax):
    problem = gsax.Problem.from_dict(
        {
            "Ca0": (0.75, 1.5),
            "temperature_C": (15.0, 35.0),
            "pH": (4.5, 7.5),
        },
        output_names=("Ca",),
    )

    sampling_result = gsax.sample(
        problem,
        n_samples=4096,
        seed=0,
        calc_second_order=True,
    )
    print(f"unique Saltelli rows: {sampling_result.samples.shape}")
    return problem, sampling_result


@app.cell
def _evaluate(batch_reactor_trajectory, jnp, np, sampling_result):
    ts = jnp.asarray(np.linspace(0.05, 6.0, 40))

    X = jnp.asarray(sampling_result.samples)
    Ca0 = X[:, 0:1]
    temperature_C = X[:, 1:2]
    pH = X[:, 2:3]

    Y = batch_reactor_trajectory(Ca0, temperature_C, pH, ts[None, :])
    Y = Y[..., None]
    print(f"output shape: {Y.shape}  (N, T, K)")
    return Y, ts


@app.cell(hide_code=True)
def _trajectory_md(mo):
    mo.md(r"""
    ### Trajectory preview

    A handful of trajectories drawn at random from the Saltelli sample.
    The asymptote $C_A(\infty) = C_{A,0}/(1 + k\tau)$ varies by an order
    of magnitude across the input box; the time to reach asymptote is
    set by $1/\tau + k$, so warm and acidic combinations saturate
    fastest while cool and basic ones drag.
    """)
    return


@app.cell
def _trajectory_plot(Y, np, plt, ts):
    rng = np.random.default_rng(0)
    idx = rng.choice(Y.shape[0], size=24, replace=False)
    fig_traj, ax_traj = plt.subplots(figsize=(7.5, 4.0))
    for _i in idx:
        ax_traj.plot(
            np.asarray(ts),
            np.asarray(Y[_i, :, 0]),
            color="C0",
            alpha=0.35,
            linewidth=1.0,
        )
    ax_traj.set_xlabel("t")
    ax_traj.set_ylabel("Ca")
    ax_traj.set_title("Batch reactor concentration trajectories — 24 sampled inputs")
    ax_traj.grid(alpha=0.3)
    fig_traj.tight_layout()
    fig_traj
    return


@app.cell(hide_code=True)
def _analyze_md(mo):
    mo.md(r"""
    ## Sobol analysis with bootstrap

    `gsax.analyze(...)` accepts the `(N, T, K)` output array and
    returns indices with shape `(T, K, D)` for first-order, total-order
    and `(T, K, D, D)` for second-order. Passing `num_resamples > 0`
    together with a PRNG `key` switches on a vectorised non-parametric
    bootstrap and populates the `_conf` arrays with `[lower, upper]`
    endpoints in a leading dimension of size 2.

    `prenormalize=True` matches the SALib output-standardisation
    convention: the cleaned output array is centred and scaled to unit
    variance once, before the bootstrap, not per resample.
    """)
    return


@app.cell
def _analyze(Y, gsax, jax, sampling_result):
    result = gsax.analyze(
        sampling_result,
        Y,
        num_resamples=200,
        conf_level=0.95,
        ci_method="quantile",
        key=jax.random.key(0),
        prenormalize=True,
    )
    print(result)
    return (result,)


@app.cell(hide_code=True)
def _bar_md(mo):
    mo.md(r"""
    ### Steady-state bar plot

    At the final time step the system is essentially at steady state, so
    $C_A(\infty) = C_{A,0}/(1 + k(T,\mathrm{pH})\,\tau)$ depends on all
    three inputs. The bars show first-order $S_1$ and total-order $S_T$
    with the bootstrap 95 % confidence intervals drawn as error bars. A
    visible gap between $S_1$ and $S_T$ for any input signals
    interactions with at least one other input.
    """)
    return


@app.cell
def _bar_plot(np, plt, problem, result):
    _names = list(problem.names)
    _s1 = np.asarray(result.S1[-1, 0, :])
    _st = np.asarray(result.ST[-1, 0, :])
    _s1_lo = np.asarray(result.S1_conf[0, -1, 0, :])
    _s1_hi = np.asarray(result.S1_conf[1, -1, 0, :])
    _st_lo = np.asarray(result.ST_conf[0, -1, 0, :])
    _st_hi = np.asarray(result.ST_conf[1, -1, 0, :])

    _x = np.arange(len(_names))
    _width = 0.38
    fig_bar, ax_bar = plt.subplots(figsize=(7.5, 4.5))
    ax_bar.bar(
        _x - _width / 2,
        _s1,
        _width,
        yerr=np.stack([_s1 - _s1_lo, _s1_hi - _s1]),
        color="C0",
        capsize=4,
        label="S1 (first-order)",
    )
    ax_bar.bar(
        _x + _width / 2,
        _st,
        _width,
        yerr=np.stack([_st - _st_lo, _st_hi - _st]),
        color="C3",
        capsize=4,
        label="ST (total-order)",
    )
    ax_bar.axhline(0.0, color="black", linewidth=0.5)
    ax_bar.set_xticks(_x)
    ax_bar.set_xticklabels(_names)
    ax_bar.set_ylabel("Sobol index")
    ax_bar.set_title("Steady-state sensitivity (Ca at the final time step)")
    ax_bar.legend(frameon=False)
    ax_bar.grid(axis="y", alpha=0.3)
    fig_bar.tight_layout()
    fig_bar
    return


@app.cell(hide_code=True)
def _timeseries_md(mo):
    mo.md(r"""
    ### Time-resolved sensitivity

    Each input's importance evolves through the start-up transient.
    Right after start-up, $C_A(t) \approx (C_{A,0}/\tau)\,t$ and the
    inlet concentration explains essentially all variance. As the
    reactor approaches steady state the kinetic terms — and therefore
    $T$ and $\mathrm{pH}$ — take over a growing share. The shaded bands
    are the bootstrap 95 % envelopes.
    """)
    return


@app.cell
def _timeseries_plot(np, plt, problem, result, ts):
    _names = list(problem.names)
    _s1 = np.asarray(result.S1[:, 0, :])
    _s1_lo = np.asarray(result.S1_conf[0, :, 0, :])
    _s1_hi = np.asarray(result.S1_conf[1, :, 0, :])
    _st = np.asarray(result.ST[:, 0, :])
    _st_lo = np.asarray(result.ST_conf[0, :, 0, :])
    _st_hi = np.asarray(result.ST_conf[1, :, 0, :])

    _t_np = np.asarray(ts)
    _colors = ("C0", "C2", "C3")
    fig_ts, axes_ts = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    for _d, (_name, _color) in enumerate(zip(_names, _colors, strict=True)):
        axes_ts[0].plot(_t_np, _s1[:, _d], color=_color, linewidth=1.6, label=_name)
        axes_ts[0].fill_between(_t_np, _s1_lo[:, _d], _s1_hi[:, _d], color=_color, alpha=0.18)
        axes_ts[1].plot(_t_np, _st[:, _d], color=_color, linewidth=1.6, label=_name)
        axes_ts[1].fill_between(_t_np, _st_lo[:, _d], _st_hi[:, _d], color=_color, alpha=0.18)

    axes_ts[0].set_title("First-order S1(t)")
    axes_ts[1].set_title("Total-order ST(t)")
    for _ax in axes_ts:
        _ax.set_xlabel("t")
        _ax.set_ylim(-0.05, 1.1)
        _ax.grid(alpha=0.3)
        _ax.legend(frameon=False)
    axes_ts[0].set_ylabel("Sobol index")
    fig_ts.tight_layout()
    fig_ts
    return


@app.cell(hide_code=True)
def _interaction_md(mo):
    mo.md(r"""
    ### Pairwise interactions

    The second-order matrix $S_{ij}$ measures the share of variance
    explained jointly by inputs $i$ and $j$ but not by either alone. For
    the steady-state outlet concentration the dominant pair is
    temperature and pH, since both inputs enter only through the rate
    constant and combine multiplicatively inside $k(T, \mathrm{pH})\tau$.
    The diagonal is left blank because $S_{ii}$ is not defined.
    """)
    return


@app.cell
def _interaction_plot(np, plt, problem, result):
    _names = list(problem.names)
    _s2 = np.asarray(result.S2[-1, 0, :, :])

    _finite = _s2[np.isfinite(_s2)]
    _vmin = float(_finite.min()) if _finite.size else 0.0
    _vmax = float(_finite.max()) if _finite.size else 1.0

    fig_s2, ax_s2 = plt.subplots(figsize=(5.5, 4.5))
    _im = ax_s2.imshow(_s2, cmap="viridis", origin="lower", vmin=_vmin, vmax=_vmax)
    _threshold = _vmin + 0.5 * (_vmax - _vmin)
    for _i in range(_s2.shape[0]):
        for _j in range(_s2.shape[1]):
            if np.isfinite(_s2[_i, _j]):
                ax_s2.text(
                    _j,
                    _i,
                    f"{_s2[_i, _j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if _s2[_i, _j] < _threshold else "black",
                    fontsize=9,
                )
    ax_s2.set_xticks(range(len(_names)))
    ax_s2.set_yticks(range(len(_names)))
    ax_s2.set_xticklabels(_names)
    ax_s2.set_yticklabels(_names)
    ax_s2.set_title("Steady-state second-order indices S2[i, j]")
    fig_s2.colorbar(_im, ax=ax_s2, label="S2")
    fig_s2.tight_layout()
    fig_s2
    return


@app.cell(hide_code=True)
def _outro(mo):
    mo.md(r"""
    ## Take-aways

    1. **Inlet concentration dominates the transient, kinetics dominate
       the asymptote.** Right after start-up the variance is almost
       purely a function of $C_{A,0}$ because the reaction has not had
       time to bite. By steady state, all three inputs contribute, and
       the gap between $S_T$ and $S_1$ on $T$ and $\mathrm{pH}$ is wide
       enough to read off without help from the confidence intervals.
    2. **Temperature and pH interact strongly.** Both enter the outlet
       concentration only through the product $k(T, \mathrm{pH})\tau$,
       so their joint variance is inseparable; the heatmap surfaces it
       as the largest off-diagonal $S_{ij}$.
    3. **Bootstrap is one extra argument.** Passing `num_resamples=200`
       and a PRNG key fills the `_conf` arrays at the same broadcast
       `(T, K, D)` shape as the point estimates, so plotting code stays
       the same with or without uncertainty bars.
    """)
    return


if __name__ == "__main__":
    app.run()
