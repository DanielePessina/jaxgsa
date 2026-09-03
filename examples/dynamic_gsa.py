"""Dynamic (time-varying) sensitivity analysis with jaxgsa.

Demonstrates how sensitivity indices evolve over time using a coupled damped
oscillator model. Compares Sobol (with bootstrap CIs), eFAST, and DGSM bounds
on total-order indices across the trajectory.

Run as script: `uv run python examples/dynamic_gsa.py`
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxgsa
from jaxgsa import efast
from jaxgsa.sampling import monte_carlo

plt.rcParams["figure.dpi"] = 150

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


sampling_result = jaxgsa.sobol.sample(
    problem,
    n_samples=4096,
    seed=0,
    calc_second_order=False,
)

X_sobol = jnp.asarray(sampling_result.samples)
Y_sobol = oscillator(X_sobol)
Y_sobol = Y_sobol[..., None]

sobol_result = jaxgsa.sobol.analyze(
    sampling_result,
    Y_sobol,
    n_bootstrap=200,
    conf_level=0.95,
    ci_method="quantile",
    key=jax.random.key(0),
)
print(sobol_result)

_names = list(problem.names)
_s1 = np.asarray(sobol_result.S1[:, 0, :])
_s1_conf = sobol_result.S1_conf
assert _s1_conf is not None
_s1_lo = np.asarray(_s1_conf[0, :, 0, :])
_s1_hi = np.asarray(_s1_conf[1, :, 0, :])
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
plt.show()

_names = list(problem.names)
_st = np.asarray(sobol_result.ST[:, 0, :])
_st_conf = sobol_result.ST_conf
assert _st_conf is not None
_st_lo = np.asarray(_st_conf[0, :, 0, :])
_st_hi = np.asarray(_st_conf[1, :, 0, :])
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
plt.show()

efast_samples = efast.sample(problem, n_per_curve=4096, M=4, seed=42)
Y_ef = oscillator(jnp.asarray(efast_samples.samples))
Y_ef = Y_ef[..., None]

efast_result = efast.analyze(efast_samples, Y_ef)
print(efast_result)

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
plt.show()


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


X_dgsm = monte_carlo(problem, n=50_000, seed=7)
dgsm_result = jaxgsa.dgsm.analyze(problem, oscillator_unbatched, jnp.asarray(X_dgsm))
print(dgsm_result)

_names = list(problem.names)
_t = np.asarray(times)
_colors = ("C0", "C1", "C2", "C3")
_D = len(_names)

_st_sob = np.asarray(sobol_result.ST[:, 0, :])
_ub = np.asarray(dgsm_result.upper_bound)
_lb = np.asarray(dgsm_result.lower_bound)

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
plt.show()
