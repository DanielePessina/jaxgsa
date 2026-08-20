"""Morris elementary-effects screening — jaxgsa tutorial.

Demonstrates the Morris screening workflow on the Ishigami benchmark:
trajectory sampling, the canonical mu_star-sigma plane, bootstrap
confidence intervals, trajectory vs radial designs, and convergence
checking via downsampling.

Run interactively: ``uv run marimo edit examples/morris_gsa.py``
Run as script:     ``uv run python examples/morris_gsa.py``
"""

# ruff: noqa: F722

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # Morris elementary-effects screening with jaxgsa

    The Morris method is a globalized one-at-a-time screening design.
    It builds $r$ trajectories of $D + 1$ points each; every step along a
    trajectory perturbs a single input, yielding one *elementary effect*

    $$
    EE_i \;=\; \frac{f(x + \Delta\, e_i) - f(x)}{\Delta}
    $$

    per trajectory and parameter.  Reducing the $r$ effects per parameter
    gives three screening measures:

    | Measure | Meaning |
    | --- | --- |
    | $\mu^*$ | mean $\lvert EE \rvert$ — importance; proxy for the total-order ($S_T$) ranking |
    | $\sigma$ | std of $EE$ — large values relative to $\mu^*$ flag nonlinearity or interactions |
    | $\mu$ | signed mean $EE$ — sign cancellation ($\mu \ll \mu^*$) flags non-monotone response |

    At $r \times (D + 1)$ model evaluations (before deduplication), Morris
    is the cheapest method in jaxgsa — ideal for *factor fixing* before a
    full variance-based analysis.  This notebook walks through:

    1. **Trajectory sampling** on the Ishigami benchmark
    2. The canonical **$\mu^*$–$\sigma$ plane**
    3. **Bootstrap confidence intervals** on $\mu^*$
    4. **Trajectory vs radial** designs at the same budget
    5. **Free screening** derived from a Sobol design, at no extra cost
    6. **Downsampling** as a convergence check
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
    from jaxgsa.benchmarks import ishigami

    plt.rcParams["figure.dpi"] = 150
    return jaxgsa, ishigami, jax, jnp, mo, np, plt


@app.cell(hide_code=True)
def _ishigami_md(mo):
    mo.md(r"""
    ## Trajectory sampling — Ishigami benchmark

    The Ishigami function $f(x_1, x_2, x_3) = \sin x_1 + 7\sin^2 x_2
    + 0.1\,x_3^4 \sin x_1$ has three inputs uniform on $[-\pi, \pi]$.
    The workflow mirrors the other jaxgsa methods:

    1. **Sample** — `jaxgsa.morris.sample` builds $r = 100$ trajectories on
       the default `num_levels=4` grid and returns only the unique rows
       to evaluate.
    2. **Evaluate** — run the model on `sr.samples`.
    3. **Analyse** — `jaxgsa.morris.analyze` reconstructs the expanded design
       internally and reduces the elementary effects.

    Watch the verbose summary line printed by the sampler: it reports the
    duplicate-row savings from deduplication.  Trajectory points live
    on a coarse 4-level grid, so in 3-D the 400 expanded rows collapse to
    just 64 unique model evaluations — an 84% saving that jaxgsa banks
    automatically.
    """)
    return


@app.cell
def _ishigami_analysis(jaxgsa, ishigami, jnp):
    morris_problem = ishigami.PROBLEM

    sr_traj = jaxgsa.morris.sample(morris_problem, 100, seed=42)
    Y_traj = ishigami.evaluate(jnp.asarray(sr_traj.samples))

    result_traj = jaxgsa.morris.analyze(sr_traj, Y_traj)
    print(result_traj.to_dataset())
    return Y_traj, morris_problem, result_traj, sr_traj


@app.cell(hide_code=True)
def _plane_md(mo):
    mo.md(r"""
    ## The $\mu^*$–$\sigma$ plane

    The canonical Morris plot places each parameter at
    $(\mu^*, \sigma)$.  Points near the horizontal axis behave almost
    linearly and additively; points above the dashed $\sigma = \mu^*$
    line have effects whose spread exceeds their mean magnitude — the
    classic signature of nonlinearity or interactions.
    """)
    return


@app.cell
def _plane_plot(morris_problem, np, plt, result_traj):
    _names = list(morris_problem.names)
    _mu_star = np.asarray(result_traj.mu_star)
    _sigma = np.asarray(result_traj.sigma)

    fig_plane, ax_plane = plt.subplots(figsize=(7.5, 4.5))
    _colors = ("C0", "C2", "C3")
    for _d, (_name, _color) in enumerate(zip(_names, _colors, strict=True)):
        ax_plane.scatter(
            _mu_star[_d],
            _sigma[_d],
            s=90,
            color=_color,
            zorder=5,
            edgecolors="white",
            linewidth=1.2,
        )
        ax_plane.annotate(
            _name,
            (_mu_star[_d], _sigma[_d]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=10,
        )

    _lim = 1.15 * max(_mu_star.max(), _sigma.max())
    ax_plane.plot(
        [0.0, _lim],
        [0.0, _lim],
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=r"$\sigma = \mu^*$",
    )
    ax_plane.set_xlim(0.0, _lim)
    ax_plane.set_ylim(0.0, _lim)
    ax_plane.set_xlabel(r"$\mu^*$ (mean $|EE|$)")
    ax_plane.set_ylabel(r"$\sigma$ (std of $EE$)")
    ax_plane.set_title(r"Ishigami — Morris $\mu^*$–$\sigma$ plane (trajectory, $r=100$)")
    ax_plane.legend(frameon=False, fontsize=8)
    ax_plane.grid(alpha=0.3)
    fig_plane.tight_layout()
    fig_plane
    return


@app.cell(hide_code=True)
def _bootstrap_md(mo):
    mo.md(r"""
    ## Bootstrap confidence intervals

    Passing `n_bootstrap` and a JAX PRNG key bootstraps the screening
    measures over trajectories (sampling $r$ of them with replacement).
    The result gains `mu_conf`, `mu_star_conf`, and `sigma_conf` arrays of
    shape $(2, D)$ holding the lower and upper bounds.

    One quirk to notice below: the $x_2$ interval collapses to zero width.
    On the 4-level grid every admissible $x_2$ step measures
    $\lvert EE \rvert = 7.875$ exactly (the grid only ever straddles the
    same $\sin^2$ level pair), so resampling trajectories cannot vary the
    mean — a hint that the coarse grid is aliasing $x_2$, which the radial
    design below resolves.
    """)
    return


@app.cell
def _bootstrap(Y_traj, jaxgsa, jax, morris_problem, np, plt, sr_traj):
    result_ci = jaxgsa.morris.analyze(
        sr_traj,
        Y_traj,
        n_bootstrap=500,
        key=jax.random.PRNGKey(0),
    )

    _conf = result_ci.mu_star_conf
    assert _conf is not None
    _mu_star = np.asarray(result_ci.mu_star)
    _lower = np.asarray(_conf[0])
    _upper = np.asarray(_conf[1])
    _yerr = np.stack([np.maximum(_mu_star - _lower, 0.0), np.maximum(_upper - _mu_star, 0.0)])

    _names = list(morris_problem.names)
    _x = np.arange(len(_names))

    fig_ci, ax_ci = plt.subplots(figsize=(7.5, 4.5))
    ax_ci.bar(_x, _mu_star, 0.55, color="C0", alpha=0.85)
    ax_ci.errorbar(
        _x,
        _mu_star,
        yerr=_yerr,
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.4,
    )
    ax_ci.set_xticks(_x)
    ax_ci.set_xticklabels(_names)
    ax_ci.set_ylabel(r"$\mu^*$")
    ax_ci.set_title(r"$\mu^*$ with 95% bootstrap CIs (500 resamples)")
    ax_ci.grid(axis="y", alpha=0.3)
    fig_ci.tight_layout()
    fig_ci
    return


@app.cell(hide_code=True)
def _radial_md(mo):
    mo.md(r"""
    ## Trajectory vs radial designs

    `method="radial"` (Campolongo et al. 2011) replaces grid walks with
    star designs around scrambled-Sobol' base points.  At the same budget
    ($r = 100$, hence the same 400 expanded rows) the two designs differ
    in two ways:

    - **Cost after dedup** — radial points are continuous, so no rows
      coincide: all 400 must be evaluated, versus 64 for the trajectory
      grid (the sampler's verbose line reports `duplicates_removed=0`).
    - **Step sizes** — radial steps $\Delta = b_i - a_i$ vary continuously,
      probing the response at many scales instead of one fixed grid step.

    The designs agree that $x_3$ ranks last, but disagree on $x_1$ vs
    $x_2$: continuous steps expose the steep local slopes of
    $7\sin^2 x_2$ (derivative up to $7\lvert\sin 2x_2\rvert$ per physical
    unit), which the fixed $\Delta = 2/3$ grid step averages away.
    Screening verdicts are design-dependent — when in doubt, keep every
    parameter that either design flags as important.
    """)
    return


@app.cell
def _radial_comparison(jaxgsa, ishigami, jnp, morris_problem, np, plt, result_traj):
    sr_radial = jaxgsa.morris.sample(morris_problem, 100, method="radial", seed=42)
    Y_radial = ishigami.evaluate(jnp.asarray(sr_radial.samples))
    result_radial = jaxgsa.morris.analyze(sr_radial, Y_radial)

    _names = list(morris_problem.names)
    _x = np.arange(len(_names))
    _width = 0.38

    fig_designs, ax_designs = plt.subplots(figsize=(7.5, 4.5))
    ax_designs.bar(
        _x - _width / 2,
        np.asarray(result_traj.mu_star),
        _width,
        color="C0",
        label="trajectory (64 evals)",
    )
    ax_designs.bar(
        _x + _width / 2,
        np.asarray(result_radial.mu_star),
        _width,
        color="C1",
        label="radial (400 evals)",
    )
    ax_designs.set_xticks(_x)
    ax_designs.set_xticklabels(_names)
    ax_designs.set_ylabel(r"$\mu^*$")
    ax_designs.set_title(r"Trajectory vs radial $\mu^*$ at the same budget ($r=100$)")
    ax_designs.legend(frameon=False, fontsize=8)
    ax_designs.grid(axis="y", alpha=0.3)
    fig_designs.tight_layout()
    fig_designs
    return


@app.cell(hide_code=True)
def _from_sobol_md(mo):
    mo.md(r"""
    ## Free screening from a Sobol design

    A Saltelli design is already a radial design: within each base point,
    the row $A$ and each row $A_B^{(j)}$ differ in exactly one parameter.
    Writing $\Delta_j = B_j - A_j$ and
    $EE_j = (f(A_B^{(j)}) - f(A)) / \Delta_j$, Jansen's total-order estimator
    is $\mathbb{E}[\Delta_j^2 EE_j^2] / (2\,\mathrm{Var}\,Y)$ while Morris
    reports $\mu^* = \mathbb{E}|EE_j|$ — the same increments, weighted
    differently.

    So `SobolSamples.to_morris()` hands you $\mu^*$ and $\sigma$ from a design
    you have already paid for, at zero extra model evaluations. Note the
    two measures answer different questions and may rank parameters
    differently: $S_T$ is a variance share, $\mu^*$ a mean absolute
    derivative.
    """)
    return


@app.cell
def _from_sobol(jaxgsa, ishigami, jnp, morris_problem, np, plt):
    sobol_samples = jaxgsa.sobol.sample(morris_problem, 0, base_n=256, seed=42, verbose=False)
    Y_sobol = ishigami.evaluate(jnp.asarray(sobol_samples.samples))

    # Two analyses, one set of model runs.
    derived_samples = sobol_samples.to_morris(verbose=False)
    result_sobol = jaxgsa.sobol.analyze(sobol_samples, Y_sobol)
    result_derived = jaxgsa.morris.analyze(derived_samples, Y_sobol)

    _names = list(morris_problem.names)
    _x = np.arange(len(_names))
    _width = 0.38

    # Normalize each measure by its own maximum: they are on different scales
    # (variance share vs unit-space derivative), so only the profile compares.
    _st = np.asarray(result_sobol.ST)
    _mu = np.asarray(result_derived.mu_star)

    fig_derived, ax_derived = plt.subplots(figsize=(7.5, 4.5))
    ax_derived.bar(_x - _width / 2, _st / _st.max(), _width, color="C2", label=r"$S_T$ (Sobol)")
    ax_derived.bar(
        _x + _width / 2,
        _mu / _mu.max(),
        _width,
        color="C3",
        label=r"$\mu^*$ (Morris, derived)",
    )
    ax_derived.set_xticks(_x)
    ax_derived.set_xticklabels(_names)
    ax_derived.set_ylabel("normalized to own maximum")
    ax_derived.set_title(
        f"Both from one design: {sobol_samples.n_runs} model runs, 0 extra "
        f"({derived_samples.n_trajectories} radial blocks)"
    )
    ax_derived.legend(frameon=False, fontsize=8)
    ax_derived.grid(axis="y", alpha=0.3)
    fig_derived.tight_layout()
    fig_derived
    return


@app.cell(hide_code=True)
def _downsample_md(mo):
    mo.md(r"""
    ## Convergence check via downsampling

    Trajectories are generated sequentially, so the first $m$ trajectories
    of an $r$-trajectory run are identical to drawing $m$ directly with
    the same seed.  `sr.downsample(m, Y)` slices both the design and the
    already-computed outputs — no re-simulation needed.  If the
    $\mu^*$ ranking is stable between $r = 25$ and $r = 100$, the
    screening verdict has converged.
    """)
    return


@app.cell
def _downsample(Y_traj, jaxgsa, jnp, morris_problem, np, plt, result_traj, sr_traj):
    sr_25, Y_25 = sr_traj.downsample(25, np.asarray(Y_traj))
    result_25 = jaxgsa.morris.analyze(sr_25, jnp.asarray(Y_25))

    _names = list(morris_problem.names)
    _mu_star_100 = np.asarray(result_traj.mu_star)
    _mu_star_25 = np.asarray(result_25.mu_star)

    _rank_100 = " > ".join(_names[_i] for _i in np.argsort(-_mu_star_100))
    _rank_25 = " > ".join(_names[_i] for _i in np.argsort(-_mu_star_25))
    print(f"r=100 ranking: {_rank_100}   ({sr_traj.n_runs} unique evals)")
    print(f"r=25  ranking: {_rank_25}   ({sr_25.n_runs} unique evals)")
    print(f"rankings stable: {_rank_100 == _rank_25}")

    _x = np.arange(len(_names))
    _width = 0.38

    fig_ds, ax_ds = plt.subplots(figsize=(7.5, 4.5))
    ax_ds.bar(_x - _width / 2, _mu_star_100, _width, color="C0", label="r = 100")
    ax_ds.bar(_x + _width / 2, _mu_star_25, _width, color="C4", label="r = 25 (downsampled)")
    ax_ds.set_xticks(_x)
    ax_ds.set_xticklabels(_names)
    ax_ds.set_ylabel(r"$\mu^*$")
    ax_ds.set_title(r"$\mu^*$ stability under downsampling (trajectory design)")
    ax_ds.legend(frameon=False, fontsize=8)
    ax_ds.grid(axis="y", alpha=0.3)
    fig_ds.tight_layout()
    fig_ds
    return


@app.cell(hide_code=True)
def _interpretation(mo):
    mo.md(r"""
    ## Interpretation

    - **$x_1$ and $x_2$ are important.**  Both have large $\mu^*$
      ($\approx 8.7$ and $7.9$).  $x_1$ is the only near-monotone input:
      its signed mean $\mu \approx 8.7$ equals its $\mu^*$, so effects
      never change sign.  $x_2$ shows heavy sign cancellation
      ($\mu \approx 0.3$ vs $\mu^* \approx 7.9$) because $7\sin^2 x_2$
      is non-monotone.
    - **$x_3$ acts through interaction.**  Its only pathway into the
      output is the product term $0.1\,x_3^4 \sin x_1$: $\mu \approx 0.5$
      while $\mu^* \approx 7.0$, and its $\sigma/\mu^* \approx 1.3$ is the
      largest of the three — the elementary effects of $x_3$ depend
      strongly on where $x_1$ happens to sit, which is exactly what a
      high-$\sigma$, low-$\mu$ signature means.
    - **$\mu^*$ ranks, it does not apportion.**  The trajectory ranking
      $x_1 > x_2 > x_3$ matches the analytical total-order ordering
      ($S_T = 0.56, 0.44, 0.24$), but $\mu^*$ is a mean absolute
      derivative-like quantity, not a variance share — treat it as a
      proxy for the $S_T$ ranking and follow up with a variance-based
      method (Sobol, eFAST) on the surviving parameters.
    - **Deduplication costs nothing.**  The verbose sampler line showed
      400 expanded rows collapsing to 64 unique evaluations (84% saved)
      on the 4-level grid — screening three parameters cost fewer model
      runs than a single Saltelli block.
    """)
    return


if __name__ == "__main__":
    app.run()
