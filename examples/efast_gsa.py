"""eFAST (extended FAST) sensitivity analysis with jaxgsa.

Demonstrates the eFAST workflow on standard benchmarks: scalar outputs,
multi-output models, and time-series sensitivity evolution.

Run as script: `uv run python examples/efast_gsa.py`
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxgsa import Problem, efast
from jaxgsa.benchmarks import ishigami, sobol_g

plt.rcParams["figure.dpi"] = 150

ishi_problem = ishigami.PROBLEM

ishi_samples = efast.sample(ishi_problem, n_per_curve=4096, M=4, seed=42)
Y_ishi = ishigami.evaluate(jnp.asarray(ishi_samples.samples))

ishi_result = efast.analyze(ishi_samples, Y_ishi)
print(ishi_result)

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
plt.show()

sg_problem = sobol_g.PROBLEM

sg_samples = efast.sample(sg_problem, n_per_curve=4096, M=4, seed=123)
Y_sg = sobol_g.evaluate(jnp.asarray(sg_samples.samples))

sg_result = efast.analyze(sg_samples, Y_sg)
print(sg_result)

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
plt.show()

multi_problem = ishigami.PROBLEM

multi_samples = efast.sample(multi_problem, n_per_curve=4096, M=4, seed=7)
Y_full = ishigami.evaluate(jnp.asarray(multi_samples.samples))
Y_half = 0.5 * Y_full
Y_multi = jnp.stack([Y_full, Y_half], axis=-1)  # (n_runs, 2)

multi_result = efast.analyze(multi_samples, Y_multi)
print(multi_result)

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
plt.show()

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


ts_samples = efast.sample(ts_problem, n_per_curve=4096, M=4, seed=99)
Y_ts = damped_oscillator(jnp.asarray(ts_samples.samples), ts_times)
Y_ts = Y_ts[..., None]  # (n_runs, T, 1)

ts_result = efast.analyze(ts_samples, Y_ts)
print(ts_result)

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
plt.show()

ds_scalar = ishi_result.to_dataset()
print("Scalar dataset:\n", ds_scalar, "\n")

ds_ts = ts_result.to_dataset(time_coords=np.asarray(ts_times))
print("Time-series dataset:\n", ds_ts)
