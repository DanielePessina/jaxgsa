"""Method comparison: eight GSA approaches on the Ishigami benchmark.

Compares Sobol, eFAST, HDMR, PCE, DGSM, Morris, Shapley effects, and
Borgonovo delta on the Ishigami function, a standard sensitivity
analysis test case with known analytical indices.

Run as script: ``uv run python examples/method_comparison.py``
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxgsa
from jaxgsa.benchmarks import ishigami

plt.rcParams["figure.dpi"] = 150

# ---- Setup ----
problem = ishigami.PROBLEM
analytical_s1 = ishigami.ANALYTICAL_S1
analytical_st = ishigami.ANALYTICAL_ST


def ishigami_fn(x):
    """Unbatched Ishigami: (3,) -> ()."""
    return jnp.sin(x[0]) + 7.0 * jnp.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * jnp.sin(x[0])


def _mae(estimate, truth):
    """Mean absolute error of an estimate against a reference array."""
    return float(np.mean(np.abs(np.asarray(estimate) - truth)))


# ---- Run all eight methods ----
# --- Sobol (Saltelli) ---
_t0 = time.perf_counter()
sr = jaxgsa.sobol.sample(problem, 4096, seed=42, calc_second_order=True)
_Y_sobol = ishigami.evaluate(jnp.asarray(sr.samples))
result_sobol = jaxgsa.sobol.analyze(sr, _Y_sobol)
jax.block_until_ready(result_sobol.S1)
time_sobol = time.perf_counter() - _t0
n_evals_sobol = sr.n_runs

# --- eFAST ---
_t0 = time.perf_counter()
_efast_samples = jaxgsa.efast.sample(problem, n_per_curve=4096, M=4, seed=42)
_Y_efast = ishigami.evaluate(jnp.asarray(_efast_samples.samples))
result_efast = jaxgsa.efast.analyze(_efast_samples, _Y_efast)
jax.block_until_ready(result_efast.S1)
time_efast = time.perf_counter() - _t0
n_evals_efast = _efast_samples.n_runs

# --- HDMR ---
_key = jax.random.PRNGKey(42)
_bounds = jnp.array(problem.bounds)
_X_hdmr = jax.random.uniform(_key, (2000, 3), minval=_bounds[:, 0], maxval=_bounds[:, 1])
_Y_hdmr = ishigami.evaluate(_X_hdmr)
_t0 = time.perf_counter()
result_hdmr = jaxgsa.hdmr.analyze(problem, _X_hdmr, _Y_hdmr, maxorder=2, m=2)
jax.block_until_ready(result_hdmr.S1)
time_hdmr = time.perf_counter() - _t0
n_evals_hdmr = len(_X_hdmr)

# --- PCE ---
_t0 = time.perf_counter()
result_pce = jaxgsa.pce.analyze(problem, _X_hdmr, _Y_hdmr, order=4)
jax.block_until_ready(result_pce.S1)
time_pce = time.perf_counter() - _t0
n_evals_pce = len(_X_hdmr)

# --- DGSM ---
_X_dgsm = jaxgsa.sampling.monte_carlo(problem, n=10_000, seed=42)
_t0 = time.perf_counter()
result_dgsm = jaxgsa.dgsm.analyze(problem, ishigami_fn, jnp.asarray(_X_dgsm))
jax.block_until_ready(result_dgsm.upper_bound)
time_dgsm = time.perf_counter() - _t0
n_evals_dgsm = len(_X_dgsm)

# --- Morris ---
_t0 = time.perf_counter()
sr_morris = jaxgsa.morris.sample(problem, 100, seed=42)
_Y_morris = ishigami.evaluate(jnp.asarray(sr_morris.samples))
result_morris = jaxgsa.morris.analyze(sr_morris, _Y_morris)
jax.block_until_ready(result_morris.mu_star)
time_morris = time.perf_counter() - _t0
n_evals_morris = sr_morris.n_runs

# --- Shapley (PCE backend, same data and order as the PCE run) ---
_t0 = time.perf_counter()
result_shapley = jaxgsa.pce.analyze(problem, _X_hdmr, _Y_hdmr, order=4).shapley()
jax.block_until_ready(result_shapley.Sh)
time_shapley = time.perf_counter() - _t0
n_evals_shapley = len(_X_hdmr)

# --- Borgonovo delta ---
_t0 = time.perf_counter()
result_borgonovo = jaxgsa.borgonovo.analyze(problem, _X_hdmr, _Y_hdmr, key=jax.random.key(42))
jax.block_until_ready(result_borgonovo.delta)
time_borgonovo = time.perf_counter() - _t0
n_evals_borgonovo = len(_X_hdmr)

# ---- First-order indices (S1) ----
_names = list(problem.names)
_x = np.arange(len(_names))
_n_methods = 4
_width = 0.15

_s1_sobol = np.asarray(result_sobol.S1)
_s1_efast = np.asarray(result_efast.S1)
_s1_hdmr = np.asarray(result_hdmr.S1)
_s1_pce = np.asarray(result_pce.S1)
_s1_ana = np.asarray(analytical_s1)

_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_labels = ["Sobol", "eFAST", "HDMR", "PCE"]
_data = [_s1_sobol, _s1_efast, _s1_hdmr, _s1_pce]

_fig, _ax = plt.subplots(figsize=(8, 5))
for _i, (_d, _c, _l) in enumerate(
    zip(_data, _colors, _labels, strict=True),
):
    _offset = (_i - _n_methods / 2 + 0.5) * _width
    _ax.bar(_x + _offset, _d, _width, color=_c, label=_l, alpha=0.85)

_ax.scatter(_x, _s1_ana, marker="D", color="black", s=60, zorder=5, label="Analytical")

_ax.set_xticks(_x)
_ax.set_xticklabels(_names)
_ax.set_ylabel("S1")
_ax.set_title("First-Order Indices (S1)")
_ax.legend(frameon=False, fontsize=8)
_ax.grid(axis="y", alpha=0.3)
_ax.set_ylim(bottom=-0.05)
_fig.tight_layout()
plt.show()

# ---- Total-order indices (ST) ----
_names = list(problem.names)
_x = np.arange(len(_names))
_n_methods = 5
_width = 0.14

_st_sobol = np.asarray(result_sobol.ST)
_st_efast = np.asarray(result_efast.ST)
_st_hdmr = np.asarray(result_hdmr.ST)
_st_pce = np.asarray(result_pce.ST)
_st_dgsm_ub = np.asarray(result_dgsm.upper_bound)
_st_ana = np.asarray(analytical_st)

_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
_labels = ["Sobol", "eFAST", "HDMR", "PCE", "DGSM (upper)"]
_data = [_st_sobol, _st_efast, _st_hdmr, _st_pce, _st_dgsm_ub]

_fig, _ax = plt.subplots(figsize=(8, 5))
for _i, (_d, _c, _l) in enumerate(
    zip(_data, _colors, _labels, strict=True),
):
    _offset = (_i - _n_methods / 2 + 0.5) * _width
    _ax.bar(_x + _offset, _d, _width, color=_c, label=_l, alpha=0.85)

_ax.scatter(_x, _st_ana, marker="D", color="black", s=60, zorder=5, label="Analytical")

_ax.set_xticks(_x)
_ax.set_xticklabels(_names)
_ax.set_ylabel("ST")
_ax.set_title("Total-Order Indices (ST)")
_ax.legend(frameon=False, fontsize=8)
_ax.grid(axis="y", alpha=0.3)
_ax.set_ylim(bottom=-0.05)
_fig.tight_layout()
plt.show()

# ---- Morris screening (different scale) ----
_names = list(problem.names)
_x = np.arange(len(_names))
_width = 0.38

_mu_star = np.asarray(result_morris.mu_star)
_mu_star_share = _mu_star / _mu_star.sum()
_st_ana = np.asarray(analytical_st)
_st_share = _st_ana / _st_ana.sum()

_fig, _ax = plt.subplots(figsize=(8, 5))
_ax.bar(
    _x - _width / 2,
    _mu_star_share,
    _width,
    color="#8c564b",
    alpha=0.85,
    label="Morris mu* (normalized)",
)
_ax.bar(
    _x + _width / 2,
    _st_share,
    _width,
    color="black",
    alpha=0.35,
    label="Analytical ST (normalized)",
)
_ax.set_xticks(_x)
_ax.set_xticklabels(_names)
_ax.set_ylabel("Share of total (ranking check only)")
_ax.set_title("Morris mu* vs analytical ST — normalized for ranking")
_ax.legend(frameon=False, fontsize=8)
_ax.grid(axis="y", alpha=0.3)
_fig.tight_layout()
plt.show()

# ---- Shapley effects (fair shares) ----
_names = list(problem.names)
_x = np.arange(len(_names))
_width = 0.38

_sh = np.asarray(result_shapley.Sh)
_sh_ana = np.asarray(ishigami.ANALYTICAL_SHAPLEY)

_fig, _ax = plt.subplots(figsize=(8, 5))
_ax.bar(
    _x - _width / 2,
    _sh,
    _width,
    color="#17becf",
    alpha=0.85,
    label="Shapley Sh (PCE backend)",
)
_ax.bar(
    _x + _width / 2,
    _sh_ana,
    _width,
    color="black",
    alpha=0.35,
    label="Analytical Sh",
)
_ax.set_xticks(_x)
_ax.set_xticklabels(_names)
_ax.set_ylabel("Sh (fair variance share)")
_ax.set_title("Shapley effects vs analytical (each sums to 1)")
_ax.legend(frameon=False, fontsize=8)
_ax.grid(axis="y", alpha=0.3)
_fig.tight_layout()
plt.show()

# ---- Borgonovo delta (different scale) ----
_names = list(problem.names)
_x = np.arange(len(_names))
_width = 0.38

_delta = np.asarray(result_borgonovo.delta)
_s1_given_data = np.asarray(result_borgonovo.S1)
_s1_ana = np.asarray(analytical_s1)

_fig, _ax = plt.subplots(figsize=(8, 5))
_ax.bar(
    _x - _width / 2,
    _delta,
    _width,
    color="#e377c2",
    alpha=0.85,
    label="Borgonovo delta",
)
_ax.bar(
    _x + _width / 2,
    _s1_given_data,
    _width,
    color="#7f7f7f",
    alpha=0.85,
    label="Given-data S1",
)
_ax.scatter(
    _x,
    _s1_ana,
    marker="D",
    color="black",
    s=60,
    zorder=5,
    label="Analytical S1",
)
_ax.set_xticks(_x)
_ax.set_xticklabels(_names)
_ax.set_ylabel("index value")
_ax.set_title("Borgonovo delta and given-data S1")
_ax.legend(frameon=False, fontsize=8)
_ax.grid(axis="y", alpha=0.3)
_fig.tight_layout()
plt.show()

# ---- Accuracy and cost comparison ----
_s1_ana = np.asarray(analytical_s1)
_st_ana = np.asarray(analytical_st)

_rows = [
    (
        "Sobol",
        _mae(result_sobol.S1, _s1_ana),
        _mae(result_sobol.ST, _st_ana),
        n_evals_sobol,
        time_sobol,
    ),
    (
        "eFAST",
        _mae(result_efast.S1, _s1_ana),
        _mae(result_efast.ST, _st_ana),
        n_evals_efast,
        time_efast,
    ),
    (
        "HDMR",
        _mae(result_hdmr.S1, _s1_ana),
        _mae(result_hdmr.ST, _st_ana),
        n_evals_hdmr,
        time_hdmr,
    ),
    (
        "PCE",
        _mae(result_pce.S1, _s1_ana),
        _mae(result_pce.ST, _st_ana),
        n_evals_pce,
        time_pce,
    ),
    (
        "DGSM (bound gap)",
        None,
        _mae(result_dgsm.upper_bound, _st_ana),
        n_evals_dgsm,
        time_dgsm,
    ),
    (
        "Morris (screening)",
        None,
        None,
        n_evals_morris,
        time_morris,
    ),
    (
        "Shapley (Sh)",
        _mae(result_shapley.S1, _s1_ana),
        _mae(result_shapley.ST, _st_ana),
        n_evals_shapley,
        time_shapley,
    ),
    (
        "Borgonovo delta",
        _mae(result_borgonovo.S1, _s1_ana),
        None,
        n_evals_borgonovo,
        time_borgonovo,
    ),
]

_header = "| Method | S1 MAE | ST MAE | N evals | Wall time (s) |\n"
_sep = "| --- | ---: | ---: | ---: | ---: |\n"
_body = ""
for _name, _s1e, _ste, _ne, _wt in _rows:
    _s1_str = f"{_s1e:.4f}" if _s1e is not None else "---"
    _st_str = f"{_ste:.4f}" if _ste is not None else "---"
    _body += f"| {_name} | {_s1_str} | {_st_str} | {_ne:,} | {_wt:.2f} |\n"

print("### Accuracy and cost comparison\n\n" + _header + _sep + _body)
print(
    "Timing note: Sobol, eFAST, and Morris times are end-to-end (sample "
    "+ evaluate + analyze). HDMR, PCE, Shapley, and Borgonovo times are "
    "analyze-only (shared pre-computed samples). DGSM time is analyze-only "
    "(internally evaluates via autodiff)."
)

# ---- Cost vs accuracy (total-order indices) ----
_st_ana = np.asarray(analytical_st)

_methods = ["Sobol", "eFAST", "HDMR", "PCE", "DGSM (bound gap)"]
_n_evals = [
    n_evals_sobol,
    n_evals_efast,
    n_evals_hdmr,
    n_evals_pce,
    n_evals_dgsm,
]
_mae_st = [
    _mae(result_sobol.ST, _st_ana),
    _mae(result_efast.ST, _st_ana),
    _mae(result_hdmr.ST, _st_ana),
    _mae(result_pce.ST, _st_ana),
    _mae(result_dgsm.upper_bound, _st_ana),
]
_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

_fig, _ax = plt.subplots(figsize=(7, 5))
for _m, _n, _e, _c in zip(
    _methods,
    _n_evals,
    _mae_st,
    _colors,
    strict=True,
):
    _ax.scatter(
        _n,
        _e,
        s=120,
        color=_c,
        zorder=5,
        edgecolors="white",
        linewidth=1.2,
    )
    _ax.annotate(
        _m,
        (_n, _e),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=9,
    )

_ax.set_xlabel("Model evaluations")
_ax.set_ylabel("MAE of ST vs analytical")
_ax.set_title("Cost vs accuracy (total-order indices)")
_ax.set_xscale("log")
_ax.set_yscale("log")
_ax.grid(True, alpha=0.3)
_fig.tight_layout()
plt.show()
