"""DGSM sensitivity analysis on Ishigami and linear benchmarks with jaxgsa.

Computes derivative-based bounds (Poincare upper, Kucherenko-Song lower)
on total Sobol indices for two standard benchmarks and compares them
against the known analytical values.

Run as script: `uv run python examples/dgsm_benchmark.py`
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxgsa
from jaxgsa.benchmarks import ishigami, linear
from jaxgsa.sampling import monte_carlo


def ishigami_fn(x):
    """Unbatched Ishigami: (3,) -> ()."""
    A, B = 7.0, 0.1
    return jnp.sin(x[0]) + A * jnp.sin(x[1]) ** 2 + B * x[2] ** 4 * jnp.sin(x[0])


X_ish = monte_carlo(ishigami.PROBLEM, n=50_000, seed=42)
result_ish = jaxgsa.dgsm.analyze(ishigami.PROBLEM, ishigami_fn, jnp.asarray(X_ish))

_st = np.array(ishigami.ANALYTICAL_ST)
_ub = np.asarray(result_ish.upper_bound)
_lb = np.asarray(result_ish.lower_bound)
_names = list(ishigami.PROBLEM.names)

_fig, _ax = plt.subplots(figsize=(7, 4.5), dpi=150)
_xp = np.arange(len(_names))
_bw = 0.25

_ax.bar(_xp - _bw, _lb, _bw, label="Lower bound", color="#2196F3", alpha=0.85)
_ax.bar(_xp, _st, _bw, label="Analytical $S_T$", color="#4CAF50", alpha=0.85)
_ax.bar(_xp + _bw, _ub, _bw, label="Upper bound", color="#FF9800", alpha=0.85)

_ax.set_xlabel("Parameter")
_ax.set_ylabel("Sensitivity index")
_ax.set_title("Ishigami: DGSM bounds vs analytical $S_T$")
_ax.set_xticks(_xp)
_ax.set_xticklabels(_names)
_ax.legend()
_ax.set_ylim(0, max(_ub) * 1.15)
_fig.tight_layout()
plt.show()

_st = np.array(ishigami.ANALYTICAL_ST)
_nu = np.asarray(result_ish.nu)
_sigma = np.asarray(result_ish.sigma)
_lb = np.asarray(result_ish.lower_bound)
_ub = np.asarray(result_ish.upper_bound)
_names = list(ishigami.PROBLEM.names)

print("| Param | $\\nu_i$ | $\\sigma_i$ | Lower | $S_T$ | Upper |")
print("| --- | ---: | ---: | ---: | ---: | ---: |")
for _i, _name in enumerate(_names):
    print(
        f"| {_name} | {_nu[_i]:.4f} | {_sigma[_i]:+.4f} | "
        f"{_lb[_i]:.4f} | {_st[_i]:.4f} | {_ub[_i]:.4f} |"
    )


def linear_fn(x):
    """Unbatched linear: (3,) -> ()."""
    c = jnp.array([1.0, 2.0, 3.0])
    return jnp.dot(c, x)


X_lin = monte_carlo(linear.PROBLEM, n=10_000, seed=123)
result_lin = jaxgsa.dgsm.analyze(linear.PROBLEM, linear_fn, jnp.asarray(X_lin))

_st = np.array(linear.ANALYTICAL_ST)
_ub = np.asarray(result_lin.upper_bound)
_lb = np.asarray(result_lin.lower_bound)
_names = list(linear.PROBLEM.names)

_fig, _ax = plt.subplots(figsize=(7, 4.5), dpi=150)
_xp = np.arange(len(_names))
_bw = 0.25

_ax.bar(_xp - _bw, _lb, _bw, label="Lower bound", color="#2196F3", alpha=0.85)
_ax.bar(_xp, _st, _bw, label="Analytical $S_T$", color="#4CAF50", alpha=0.85)
_ax.bar(_xp + _bw, _ub, _bw, label="Upper bound", color="#FF9800", alpha=0.85)

_ax.set_xlabel("Parameter")
_ax.set_ylabel("Sensitivity index")
_ax.set_title("Linear model: DGSM bounds collapse to exact $S_T$")
_ax.set_xticks(_xp)
_ax.set_xticklabels(_names)
_ax.legend()
_ax.set_ylim(0, max(_ub) * 1.15)
_fig.tight_layout()
plt.show()
