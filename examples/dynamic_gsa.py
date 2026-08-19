"""Dynamic (time-varying) sensitivity analysis with jaxgsa.

Demonstrates how sensitivity indices evolve over time using a coupled damped
oscillator model. Compares Sobol (with bootstrap CIs), eFAST, and DGSM bounds
on total-order indices across the trajectory.

Run interactively: ``uv run marimo edit examples/dynamic_gsa.py``
Run as script:     ``uv run python examples/dynamic_gsa.py``
"""

# ruff: noqa: F722

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # Dynamic sensitivity analysis with jaxgsa

    Dynamic sensitivity analysis reveals how parameter importance evolves
    over time. Early transient behaviour may be dominated by different
    parameters than steady-state behaviour. By computing sensitivity indices
    at every time step we obtain a film rather than a snapshot of the
    variance decomposition. Two Sobol indices carry the story: the
    first-order index $S_1$ is the fraction of output variance a
    parameter explains on its own, and the total-order index $S_T$
    adds every interaction it takes part in.

    This notebook uses a coupled damped oscillator model whose output
    $y(t)$ depends on four physical parameters — amplitude, frequency,
    damping, and coupling strength — and shows:

    1. **Sobol indices** ($S_1$, $S_T$) with bootstrap 95 % confidence bands
    2. **eFAST indices** overlaid for cross-method comparison
    3. **DGSM bounds** (Poincare upper / Kucherenko-Song lower) on $S_T$

    All three methods agree on how the dominant parameter shifts from
    amplitude at early times to damping at late times.
    """)
    return


@app.cell
def _imports():
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import jaxgsa
    from jaxgsa import efast
    from jaxgsa.sampling import monte_carlo

    plt.rcParams["figure.dpi"] = 150
    return efast, jaxgsa, jax, jnp, mo, np, plt, monte_carlo


@app.cell(hide_code=True)
def _model_md(mo):
    mo.md(r"""
    ## Model: coupled damped oscillator

    The output trajectory combines an oscillatory decay with a coupling
    ramp that also decays:

    $$
    y(t) \;=\; A\,\sin(2\pi\omega\,t)\,e^{-\gamma t}
           \;+\; \kappa\,t\,e^{-\gamma t}
    $$

    | parameter | symbol | range |
    |---|---|---|
    | amplitude | $A$ | $[0.5,\;2.0]$ |
    | frequency | $\omega$ | $[1.0,\;5.0]$ |
    | damping | $\gamma$ | $[0.01,\;0.5]$ |
    | coupling | $\kappa$ | $[0.1,\;2.0]$ |

    At short times the amplitude $A$ controls the envelope height; at long
    times the exponential decay $e^{-\gamma t}$ dominates, making the
    damping rate $\gamma$ the most important parameter. The frequency
    $\omega$ matters most during the oscillatory transient.
    """)
    return


@app.cell
def _model(jaxgsa, jnp):
    problem = jaxgsa.Problem.from_dict(
        {
            "amplitude": (0.5, 2.0),
            "frequency": (1.0, 5.0),
            "damping": (0.01, 0.5),
            "coupling": (0.1, 2.0),
        },
    )

    _T = 50
    times = jnp.linspace(0.1, 10.0, _T)

    def oscillator(X):
        """Coupled damped oscillator: (N, D) -> (N, T)."""
        amp = X[:, 0:1]
        freq = X[:, 1:2]
        damping = X[:, 2:3]
        coupling = X[:, 3:4]
        t = times[None, :]
        y = amp * jnp.sin(2 * jnp.pi * freq * t) * jnp.exp(-damping * t) + coupling * t * jnp.exp(
            -damping * t
        )
        return y

    return oscillator, problem, times


@app.cell(hide_code=True)
def _sobol_md(mo):
    mo.md(r"""
    ## Sobol analysis with bootstrap confidence intervals

    We use Saltelli sampling with `calc_second_order=False` (keeping
    $N(2D+2)$ evaluations manageable) and request 200 bootstrap resamples
    for 95 % confidence intervals. The output array has shape
    $(N, T, 1)$ — the trailing singleton marks a single "output channel"
    with $T$ time steps.

    The result shapes are:
    - `S1`, `ST`: $(T, 1, D)$
    - `S1_conf`, `ST_conf`: $(2, T, 1, D)$

    The two plots that follow trace $S_1(t)$ and $S_T(t)$ for each
    parameter, with the bootstrap bands shaded. Watch how the curves
    cross: whichever parameter's curve is on top dominates the output
    variance at that moment.
    """)
    return


@app.cell
def _sobol_analysis(jaxgsa, jax, jnp, oscillator, problem):
    sampling_result = jaxgsa.sobol.sample(
        problem,
        n_samples=4096,
        seed=0,
        calc_second_order=False,
    )

    X_sobol = jnp.asarray(sampling_result.samples)
    Y_sobol = oscillator(X_sobol)
    # Reshape (N, T) -> (N, T, 1) so jaxgsa treats columns as time steps, not outputs
    Y_sobol = Y_sobol[..., None]

    sobol_result = jaxgsa.sobol.analyze(
        sampling_result,
        Y_sobol,
        num_resamples=200,
        conf_level=0.95,
        ci_method="quantile",
        key=jax.random.key(0),
    )
    print(sobol_result)
    return (sobol_result,)


@app.cell(hide_code=True)
def _s1_plot(np, plt, problem, sobol_result, times):
    _names = list(problem.names)
    _s1 = np.asarray(sobol_result.S1[:, 0, :])  # (T, D)
    _s1_lo = np.asarray(sobol_result.S1_conf[0, :, 0, :])
    _s1_hi = np.asarray(sobol_result.S1_conf[1, :, 0, :])
    _t = np.asarray(times)

    _colors = ("C0", "C1", "C2", "C3")
    _fig, _ax = plt.subplots(figsize=(9.0, 5.0))
    for _d, (_name, _color) in enumerate(zip(_names, _colors, strict=True)):
        _ax.plot(_t, _s1[:, _d], color=_color, linewidth=1.6, label=_name)
        _ax.fill_between(_t, _s1_lo[:, _d], _s1_hi[:, _d], color=_color, alpha=0.18)
    _ax.set_xlabel("Time")
    _ax.set_ylabel("$S_1$")
    _ax.set_title("First-Order Sensitivity Over Time (Sobol)")
    _ax.set_ylim(-0.05, 1.1)
    _ax.grid(alpha=0.3)
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _st_plot(np, plt, problem, sobol_result, times):
    _names = list(problem.names)
    _st = np.asarray(sobol_result.ST[:, 0, :])  # (T, D)
    _st_lo = np.asarray(sobol_result.ST_conf[0, :, 0, :])
    _st_hi = np.asarray(sobol_result.ST_conf[1, :, 0, :])
    _t = np.asarray(times)

    _colors = ("C0", "C1", "C2", "C3")
    _fig, _ax = plt.subplots(figsize=(9.0, 5.0))
    for _d, (_name, _color) in enumerate(zip(_names, _colors, strict=True)):
        _ax.plot(_t, _st[:, _d], color=_color, linewidth=1.6, label=_name)
        _ax.fill_between(_t, _st_lo[:, _d], _st_hi[:, _d], color=_color, alpha=0.18)
    _ax.set_xlabel("Time")
    _ax.set_ylabel("$S_T$")
    _ax.set_title("Total-Order Sensitivity Over Time (Sobol)")
    _ax.set_ylim(-0.05, 1.1)
    _ax.grid(alpha=0.3)
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _efast_md(mo):
    mo.md(r"""
    ## eFAST comparison

    We run the same model through eFAST and overlay the time-resolved
    indices with the Sobol estimates. eFAST uses a frequency-based
    variance decomposition that is independent of the Saltelli sampling
    scheme, so it acts as a cross-check. In the plot below, solid lines
    are Sobol and dashed lines eFAST — where the two nearly coincide,
    two unrelated estimators agree on the same indices.
    """)
    return


@app.cell
def _efast_analysis(efast, jnp, oscillator, problem):
    efast_samples = efast.sample(problem, n_per_curve=4096, M=4, seed=42)
    Y_ef = oscillator(jnp.asarray(efast_samples.samples))
    # Reshape (n_runs, T) -> (n_runs, T, 1) so jaxgsa treats columns as time steps, not outputs
    Y_ef = Y_ef[..., None]

    efast_result = efast.analyze(efast_samples, Y_ef)
    print(efast_result)
    return (efast_result,)


@app.cell(hide_code=True)
def _comparison_plot(efast_result, np, plt, problem, sobol_result, times):
    _names = list(problem.names)
    _t = np.asarray(times)
    _colors = ("C0", "C1", "C2", "C3")

    _s1_sob = np.asarray(sobol_result.S1[:, 0, :])
    _st_sob = np.asarray(sobol_result.ST[:, 0, :])
    _s1_ef = np.asarray(efast_result.S1[:, 0, :])
    _st_ef = np.asarray(efast_result.ST[:, 0, :])

    _fig, _axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    for _d, (_name, _color) in enumerate(zip(_names, _colors, strict=True)):
        _axes[0].plot(_t, _s1_sob[:, _d], color=_color, linewidth=1.6, label=f"{_name} (Sobol)")
        _axes[0].plot(
            _t,
            _s1_ef[:, _d],
            color=_color,
            linewidth=1.6,
            linestyle="--",
            label=f"{_name} (eFAST)",
        )
        _axes[1].plot(_t, _st_sob[:, _d], color=_color, linewidth=1.6, label=f"{_name} (Sobol)")
        _axes[1].plot(
            _t,
            _st_ef[:, _d],
            color=_color,
            linewidth=1.6,
            linestyle="--",
            label=f"{_name} (eFAST)",
        )

    _axes[0].set_title("First-order $S_1(t)$")
    _axes[1].set_title("Total-order $S_T(t)$")
    for _ax in _axes:
        _ax.set_xlabel("Time")
        _ax.set_ylim(-0.05, 1.1)
        _ax.grid(alpha=0.3)
        _ax.legend(frameon=False, fontsize=7, ncol=2)
    _axes[0].set_ylabel("Sensitivity index")
    _fig.suptitle("Sobol vs eFAST — time-resolved comparison", y=1.02)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _dgsm_md(mo):
    mo.md(r"""
    ## DGSM bounds over time

    DGSM computes derivative-based bounds on the total Sobol index $S_T$
    from partial derivatives via reverse-mode autodiff. For a time-series
    model, the unbatched function maps $(D,) \to (T,)$, so DGSM treats
    each time step as an independent output ($K = T$).

    The result has `upper_bound` and `lower_bound` with shape $(T, D)$.
    Plotting these as shaded regions shows that the DGSM bracket
    contains the Sobol $S_T$ at every time step — a useful sanity
    check and a way to screen parameters cheaply.
    """)
    return


@app.cell
def _dgsm_fn(jnp, times):
    def oscillator_unbatched(x):
        """Unbatched oscillator: (D,) -> (T,)."""
        amp = x[0]
        freq = x[1]
        damping = x[2]
        coupling = x[3]
        t = times
        y = amp * jnp.sin(2 * jnp.pi * freq * t) * jnp.exp(-damping * t) + coupling * t * jnp.exp(
            -damping * t
        )
        return y

    return (oscillator_unbatched,)


@app.cell
def _dgsm_analysis(jaxgsa, jnp, oscillator_unbatched, problem, monte_carlo):
    X_dgsm = monte_carlo(problem, n=50_000, seed=7)
    dgsm_result = jaxgsa.dgsm.analyze(problem, oscillator_unbatched, jnp.asarray(X_dgsm))
    print(dgsm_result)
    return (dgsm_result,)


@app.cell(hide_code=True)
def _dgsm_plot(dgsm_result, np, plt, problem, sobol_result, times):
    _names = list(problem.names)
    _t = np.asarray(times)
    _colors = ("C0", "C1", "C2", "C3")
    _D = len(_names)

    _st_sob = np.asarray(sobol_result.ST[:, 0, :])  # (T, D)
    _ub = np.asarray(dgsm_result.upper_bound)  # (K=T, D)
    _lb = np.asarray(dgsm_result.lower_bound)  # (K=T, D)

    _fig, _axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)
    _axes = _axes.ravel()
    for _d in range(_D):
        _ax = _axes[_d]
        _ax.fill_between(
            _t,
            _lb[:, _d],
            _ub[:, _d],
            color=_colors[_d],
            alpha=0.20,
            label="DGSM bounds",
        )
        _ax.plot(
            _t,
            _st_sob[:, _d],
            color=_colors[_d],
            linewidth=1.8,
            label="Sobol $S_T$",
        )
        _ax.plot(
            _t,
            _ub[:, _d],
            color=_colors[_d],
            linewidth=0.8,
            linestyle=":",
        )
        _ax.plot(
            _t,
            _lb[:, _d],
            color=_colors[_d],
            linewidth=0.8,
            linestyle=":",
        )
        _ax.set_title(_names[_d])
        _ax.set_ylim(-0.05, None)
        _ax.grid(alpha=0.3)
        _ax.legend(frameon=False, fontsize=8)
    _axes[2].set_xlabel("Time")
    _axes[3].set_xlabel("Time")
    _axes[0].set_ylabel("Sensitivity bound")
    _axes[2].set_ylabel("Sensitivity bound")
    _fig.suptitle("DGSM bounds bracket Sobol $S_T$ at every time step", y=1.01)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _insight(mo):
    mo.md(r"""
    ## Key insight: parameter importance is not static

    The time-resolved analysis reveals a clear shift in dominance
    across the trajectory:

    - **Early times** ($t < 2$): the amplitude $A$ and coupling
      $\kappa$ dominate because the exponential envelope has not yet
      decayed significantly. The output is approximately $A\sin(\cdot) +
      \kappa\,t$, both linear in their respective parameters.
    - **Mid-range** ($2 < t < 5$): the frequency $\omega$ contributes
      most strongly during the oscillatory transient, where the sine
      argument $2\pi\omega t$ sweeps rapidly.
    - **Late times** ($t > 5$): the damping $\gamma$ takes over as
      the dominant parameter. The decay $e^{-\gamma t}$ suppresses
      everything exponentially, so small changes in $\gamma$ produce
      large relative changes in $y$.

    A single steady-state analysis would capture only the late-time
    picture and miss the early dominance of amplitude and coupling
    entirely. Time-resolved indices show when each parameter
    matters — useful when deciding which parameters deserve tighter
    measurement or calibration at each stage of an experiment.
    """)
    return


if __name__ == "__main__":
    app.run()
