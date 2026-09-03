"""Morris elementary-effects screening with jaxgsa.

Demonstrates the Morris screening workflow on the Ishigami benchmark:
trajectory sampling, the canonical mu_star-sigma plane, bootstrap
confidence intervals, trajectory vs radial designs, and convergence
checking via downsampling.

Run as script: `uv run python examples/morris_gsa.py`
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxgsa
from jaxgsa.benchmarks import ishigami

plt.rcParams["figure.dpi"] = 150

morris_problem = ishigami.PROBLEM

sr_traj = jaxgsa.morris.sample(morris_problem, 100, seed=42)
Y_traj = ishigami.evaluate(jnp.asarray(sr_traj.samples))

result_traj = jaxgsa.morris.analyze(sr_traj, Y_traj)
print(result_traj.to_dataset())

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
plt.show()

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
plt.show()

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
plt.show()

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
plt.show()

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
plt.show()
