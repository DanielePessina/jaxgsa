"""Morris elementary-effects screening — gsax tutorial.

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
    # Morris elementary-effects screening with **gsax**

    The **Morris method** is a globalized one-at-a-time screening design.
    It builds $r$ trajectories of $D + 1$ points each; every step along a
    trajectory perturbs a single input, yielding one *elementary effect*

    $$
    EE_i \;=\; \frac{f(x + \Delta\, e_i) - f(x)}{\Delta}
    $$

    per trajectory and parameter.  Reducing the $r$ effects per parameter
    gives three screening measures:

    | Measure | Meaning |
    | --- | --- |
    | $\mu^*$ | mean $\lvert EE \rvert$ — importance; proxy for the total-order ($S_T$) *ranking* |
    | $\sigma$ | std of $EE$ — large values relative to $\mu^*$ flag nonlinearity or interactions |
    | $\mu$ | signed mean $EE$ — sign cancellation ($\mu \ll \mu^*$) flags non-monotone response |

    At $r \times (D + 1)$ model evaluations (before deduplication), Morris
    is the cheapest method in gsax — ideal for *factor fixing* before a
    full variance-based analysis.  This notebook walks through:

    1. **Trajectory sampling** on the Ishigami benchmark
    2. The canonical **$\mu^*$–$\sigma$ plane**
    3. **Bootstrap confidence intervals** on $\mu^*$
    4. **Trajectory vs radial** designs at the same budget
    5. **Downsampling** as a convergence check
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
    from gsax.benchmarks import ishigami

    plt.rcParams["figure.dpi"] = 150
    return gsax, ishigami, jax, jnp, mo, np, plt


@app.cell(hide_code=True)
def _ishigami_md(mo):
    mo.md(r"""
    ## Trajectory sampling — Ishigami benchmark

    The Ishigami function $f(x_1, x_2, x_3) = \sin x_1 + 7\sin^2 x_2
    + 0.1\,x_3^4 \sin x_1$ has three inputs uniform on $[-\pi, \pi]$.
    The workflow mirrors the other gsax methods:

    1. **Sample** — `gsax.sample_morris` builds $r = 100$ trajectories on
       the default `num_levels=4` grid and returns only the **unique** rows
       to evaluate.
    2. **Evaluate** — run the model on `sr.samples`.
    3. **Analyse** — `gsax.analyze_morris` reconstructs the expanded design
       internally and reduces the elementary effects.

    Watch the verbose summary line printed by the sampler: it reports the
    **duplicate-row savings** from deduplication.  Trajectory points live
    on a coarse 4-level grid, so in 3-D the 400 expanded rows collapse to
    just 64 unique model evaluations — an 84% saving that gsax banks
    automatically.
    """)
    return


@app.cell
def _ishigami_analysis(gsax, ishigami, jnp):
    morris_problem = ishigami.PROBLEM

    sr_traj = gsax.sample_morris(morris_problem, 100, seed=42)
    Y_traj = ishigami.evaluate(jnp.asarray(sr_traj.samples))

    result_traj = gsax.analyze_morris(sr_traj, Y_traj)
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
    classic signature of **nonlinearity or interactions**.
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

    Passing `num_resamples` and a JAX PRNG key bootstraps the screening
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
def _bootstrap(Y_traj, gsax, jax, morris_problem, np, plt, sr_traj):
    result_ci = gsax.analyze_morris(
        sr_traj,
        Y_traj,
        num_resamples=500,
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
    parameter that *either* design flags as important.
    """)
    return


@app.cell
def _radial_comparison(gsax, ishigami, jnp, morris_problem, np, plt, result_traj):
    sr_radial = gsax.sample_morris(morris_problem, 100, method="radial", seed=42)
    Y_radial = ishigami.evaluate(jnp.asarray(sr_radial.samples))
    result_radial = gsax.analyze_morris(sr_radial, Y_radial)

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
def _downsample_md(mo):
    mo.md(r"""
    ## Convergence check via downsampling

    Trajectories are generated sequentially, so the first $m$ trajectories
    of an $r$-trajectory run are identical to drawing $m$ directly with
    the same seed.  `sr.downsample(m, Y)` slices both the design and the
    already-computed outputs — **no re-simulation needed**.  If the
    $\mu^*$ ranking is stable between $r = 25$ and $r = 100$, the
    screening verdict has converged.
    """)
    return


@app.cell
def _downsample(Y_traj, gsax, jnp, morris_problem, np, plt, result_traj, sr_traj):
    sr_25, Y_25 = sr_traj.downsample(25, np.asarray(Y_traj))
    result_25 = gsax.analyze_morris(sr_25, jnp.asarray(Y_25))

    _names = list(morris_problem.names)
    _mu_star_100 = np.asarray(result_traj.mu_star)
    _mu_star_25 = np.asarray(result_25.mu_star)

    _rank_100 = " > ".join(_names[_i] for _i in np.argsort(-_mu_star_100))
    _rank_25 = " > ".join(_names[_i] for _i in np.argsort(-_mu_star_25))
    print(f"r=100 ranking: {_rank_100}   ({sr_traj.n_total} unique evals)")
    print(f"r=25  ranking: {_rank_25}   ({sr_25.n_total} unique evals)")
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
      proxy for the $S_T$ *ranking* and follow up with a variance-based
      method (Sobol, eFAST) on the surviving parameters.
    - **Deduplication costs nothing.**  The verbose sampler line showed
      400 expanded rows collapsing to 64 unique evaluations (84% saved)
      on the 4-level grid — screening three parameters cost fewer model
      runs than a single Saltelli block.
    """)
    return


if __name__ == "__main__":
    app.run()
