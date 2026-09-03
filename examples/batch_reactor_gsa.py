"""Batch reactor mechanistic model — Sobol global sensitivity analysis with jaxgsa.

First-order liquid-phase reaction A -> B in a CSTR; the rate constant combines
a centred Arrhenius temperature term with a Hill-type pH saturation curve.
Treats the outlet concentration trajectory as a time-series output and reads
off first-, second-, and total-order indices with bootstrap confidence bands.

Run as script: ``uv run python examples/batch_reactor_gsa.py``
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxgsa

T_REF, R_GAS, EA = 298.15, 8.314e-3, 30.0
K_BASELINE, K_AMPLITUDE, PH50, HILL, TAU = 0.14, 1.05, 5.85, 5.0, 2.0


def batch_reactor_trajectory(Ca0, temperature_C, pH, ts):
    """Closed-form batch reactor concentration starting from Ca(0) = 0."""
    T_K = temperature_C + 273.15
    arrhenius = jnp.exp(-EA / R_GAS * (1.0 / T_K - 1.0 / T_REF))
    k = (K_BASELINE + K_AMPLITUDE / (1.0 + (pH / PH50) ** HILL)) * arrhenius
    Ca_ss = Ca0 / (1.0 + k * TAU)
    decay = jnp.exp(-(1.0 / TAU + k) * ts)
    return Ca_ss * (1.0 - decay)


problem = jaxgsa.Problem.from_dict(
    {
        "Ca0": (0.75, 1.5),
        "temperature_C": (15.0, 35.0),
        "pH": (4.5, 7.5),
    },
    output_names=("Ca",),
)

sampling_result = jaxgsa.sobol.sample(
    problem,
    n_samples=4096,
    seed=0,
    calc_second_order=True,
)
print(f"unique Saltelli rows: {sampling_result.samples.shape}")

ts = jnp.asarray(np.linspace(0.05, 6.0, 40))

X = jnp.asarray(sampling_result.samples)
Ca0 = X[:, 0:1]
temperature_C = X[:, 1:2]
pH = X[:, 2:3]

Y = batch_reactor_trajectory(Ca0, temperature_C, pH, ts[None, :])
Y = Y[..., None]
print(f"output shape: {Y.shape}  (N, T, K)")

# Trajectory preview: a handful of runs drawn at random from the sample.
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
plt.show()

result = jaxgsa.sobol.analyze(
    sampling_result,
    Y,
    n_bootstrap=200,
    conf_level=0.95,
    ci_method="quantile",
    key=jax.random.key(0),
)
print(result)

# Steady-state bar plot: S1 and ST at the final time step with bootstrap CIs.
_names = list(problem.names)
_s1 = np.asarray(result.S1[-1, 0, :])
_st = np.asarray(result.ST[-1, 0, :])
assert result.S1_conf is not None and result.ST_conf is not None
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
plt.show()

# Time-resolved sensitivity: S1(t) and ST(t) with bootstrap envelopes.
_s1_t = np.asarray(result.S1[:, 0, :])
_s1_lo_t = np.asarray(result.S1_conf[0, :, 0, :])
_s1_hi_t = np.asarray(result.S1_conf[1, :, 0, :])
_st_t = np.asarray(result.ST[:, 0, :])
_st_lo_t = np.asarray(result.ST_conf[0, :, 0, :])
_st_hi_t = np.asarray(result.ST_conf[1, :, 0, :])

_t_np = np.asarray(ts)
_colors = ("C0", "C2", "C3")
fig_ts, axes_ts = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
for _d, (_name, _color) in enumerate(zip(_names, _colors, strict=True)):
    axes_ts[0].plot(_t_np, _s1_t[:, _d], color=_color, linewidth=1.6, label=_name)
    axes_ts[0].fill_between(_t_np, _s1_lo_t[:, _d], _s1_hi_t[:, _d], color=_color, alpha=0.18)
    axes_ts[1].plot(_t_np, _st_t[:, _d], color=_color, linewidth=1.6, label=_name)
    axes_ts[1].fill_between(_t_np, _st_lo_t[:, _d], _st_hi_t[:, _d], color=_color, alpha=0.18)

axes_ts[0].set_title("First-order S1(t)")
axes_ts[1].set_title("Total-order ST(t)")
for _ax in axes_ts:
    _ax.set_xlabel("t")
    _ax.set_ylim(-0.05, 1.1)
    _ax.grid(alpha=0.3)
    _ax.legend(frameon=False)
axes_ts[0].set_ylabel("Sobol index")
fig_ts.tight_layout()
plt.show()

# Pairwise interaction heatmap at the final time step.
assert result.S2 is not None
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
plt.show()
