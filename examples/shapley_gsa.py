"""Shapley-effect sensitivity analysis with jaxgsa.

Demonstrates the Shapley-effects workflow on the Ishigami benchmark:
Monte Carlo sampling, the default PCE backend, the S1 <= Sh <= ST
bracketing against the analytical solution, surrogate order as a
convergence knob with the explained_variance guardrail, and the HDMR
backend on a multi-output model.

Run as script: `uv run python examples/shapley_gsa.py`
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxgsa
from jaxgsa.benchmarks import ishigami

plt.rcParams["figure.dpi"] = 150

shapley_problem = ishigami.PROBLEM

X = jnp.asarray(jaxgsa.sampling.monte_carlo(shapley_problem, n=2000, seed=42))
Y = ishigami.evaluate(X)

result = jaxgsa.pce.analyze(shapley_problem, X, Y, order=8).shapley()

print("Sh:", result.Sh)
print("S1:", result.S1)
print("ST:", result.ST)
print("explained_variance:", float(result.explained_variance))
print("effective order:", result.order)

_sh = np.asarray(result.Sh)
_sh_analytical = np.asarray(ishigami.ANALYTICAL_SHAPLEY)

print(f"sum(Sh) = {_sh.sum():.6f}   (Shapley efficiency: exactly 1)")
assert np.isclose(_sh.sum(), 1.0, atol=1e-4)

print("estimated: ", np.round(_sh, 4))
print("analytical:", np.round(_sh_analytical, 4))
print(f"max |error| = {np.max(np.abs(_sh - _sh_analytical)):.4f}")
assert np.allclose(_sh, _sh_analytical, atol=0.01)

_names = list(shapley_problem.names)
_x = np.arange(len(_names))
_width = 0.26

_s1 = np.asarray(result.S1)
_sh = np.asarray(result.Sh)
_st = np.asarray(result.ST)

fig_bracket, ax_bracket = plt.subplots(figsize=(7.5, 4.5))
ax_bracket.bar(_x - _width, _s1, _width, color="C0", alpha=0.85, label=r"$S_1$")
ax_bracket.bar(_x, _sh, _width, color="C2", alpha=0.85, label=r"$\mathrm{Sh}$")
ax_bracket.bar(_x + _width, _st, _width, color="C3", alpha=0.85, label=r"$S_T$")
ax_bracket.scatter(
    _x,
    np.asarray(ishigami.ANALYTICAL_SHAPLEY),
    marker="D",
    color="black",
    s=60,
    zorder=5,
    label="analytical Sh",
)
ax_bracket.set_xticks(_x)
ax_bracket.set_xticklabels(_names)
ax_bracket.set_ylabel("index value")
ax_bracket.set_title(r"Ishigami — $S_1 \leq \mathrm{Sh} \leq S_T$ (PCE backend, order 8)")
ax_bracket.legend(frameon=False, fontsize=8)
ax_bracket.grid(axis="y", alpha=0.3)
fig_bracket.tight_layout()
plt.show()

orders = (2, 3, 4, 6, 8)
sweep = {}
for _order in orders:
    _r = jaxgsa.pce.analyze(shapley_problem, X, Y, order=_order).shapley()
    sweep[_order] = _r
    _err = float(np.max(np.abs(np.asarray(_r.Sh) - ishigami.ANALYTICAL_SHAPLEY)))
    print(
        f"order={_order}  explained_variance={float(_r.explained_variance):.3f}"
        f"  max |Sh - analytical| = {_err:.4f}"
    )

_names = list(shapley_problem.names)
_sh_by_order = np.stack([np.asarray(sweep[_o].Sh) for _o in orders])
_analytical = np.asarray(ishigami.ANALYTICAL_SHAPLEY)

fig_order, ax_order = plt.subplots(figsize=(7.5, 4.5))
_colors = ("C0", "C2", "C3")
for _d, (_name, _color) in enumerate(zip(_names, _colors, strict=True)):
    ax_order.plot(
        orders,
        _sh_by_order[:, _d],
        marker="o",
        color=_color,
        label=_name,
    )
    ax_order.axhline(
        _analytical[_d],
        color=_color,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
    )
ax_order.set_xticks(list(orders))
ax_order.set_xlabel("PCE order")
ax_order.set_ylabel(r"$\mathrm{Sh}$")
ax_order.set_title(r"$\mathrm{Sh}$ convergence with surrogate order (dashed = analytical)")
ax_order.legend(frameon=False, fontsize=8)
ax_order.grid(alpha=0.3)
fig_order.tight_layout()
plt.show()

Y_multi = jnp.column_stack([Y, jnp.sum(X**2, axis=1)])  # (N, K=2)

result_hdmr = jaxgsa.hdmr.analyze(shapley_problem, X, Y_multi).shapley()

print("Sh shape:", result_hdmr.Sh.shape)  # (K, D) = (2, 3)
print("row sums:", result_hdmr.Sh.sum(axis=-1))  # each exactly 1
print("explained_variance:", result_hdmr.explained_variance)  # (K,)
print(result_hdmr.to_dataset())
