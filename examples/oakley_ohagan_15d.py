"""Oakley & O'Hagan 15-D benchmark — high-dimensional SA with Gaussian inputs.

Demonstrates three sensitivity analysis methods on a 15-dimensional
benchmark with non-uniform (Gaussian) inputs: eFAST, RS-HDMR, and DGSM.
Compares computed indices against the known analytical values.

Run as script: `uv run python examples/oakley_ohagan_15d.py`
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxgsa
from jaxgsa.benchmarks import oakley_ohagan

plt.rcParams["figure.dpi"] = 150

problem = oakley_ohagan.PROBLEM
s1_ana = np.asarray(oakley_ohagan.ANALYTICAL_S1)
st_ana = np.asarray(oakley_ohagan.ANALYTICAL_ST)
names = list(problem.names)

print(f"Dimensions: {problem.num_vars}")
print("Distribution: Gaussian N(0, 1) for all inputs")
print(f"\nAnalytical S1:  {np.array2string(s1_ana, precision=4)}")
print(f"Analytical ST:  {np.array2string(st_ana, precision=4)}")
print(f"\nSum S1 = {s1_ana.sum():.4f}  (< 1 indicates interactions)")

_sort_idx = np.argsort(st_ana)
_names_sorted = [names[i] for i in _sort_idx]
_s1_sorted = s1_ana[_sort_idx]
_st_sorted = st_ana[_sort_idx]

_y = np.arange(len(_names_sorted))
_height = 0.35

_colors_st = []
_colors_s1 = []
for _val in _st_sorted:
    if _val > 0.08:
        _colors_st.append("#E53935")
        _colors_s1.append("#1E88E5")
    elif _val > 0.02:
        _colors_st.append("#FB8C00")
        _colors_s1.append("#42A5F5")
    else:
        _colors_st.append("#BDBDBD")
        _colors_s1.append("#90CAF9")

_fig, _ax = plt.subplots(figsize=(9, 6))
_ax.barh(
    _y + _height / 2,
    _st_sorted,
    _height,
    color=_colors_st,
    label="$S_T$ (total-order)",
    edgecolor="white",
    linewidth=0.5,
)
_ax.barh(
    _y - _height / 2,
    _s1_sorted,
    _height,
    color=_colors_s1,
    label="$S_1$ (first-order)",
    edgecolor="white",
    linewidth=0.5,
)
_ax.set_yticks(_y)
_ax.set_yticklabels(_names_sorted)
_ax.set_xlabel("Sensitivity index")
_ax.set_title("Analytical Sobol indices — Oakley & O'Hagan (2004)")
_ax.legend(frameon=False)
_ax.grid(axis="x", alpha=0.3)
_fig.tight_layout()
plt.show()

efast_samples = jaxgsa.efast.sample(problem, n_per_curve=4096, M=4, seed=42)
Y_ef = oakley_ohagan.evaluate(jnp.asarray(efast_samples.samples))

efast_result = jaxgsa.efast.analyze(efast_samples, Y_ef)
print(efast_result)

_s1_ef = np.asarray(efast_result.S1)
_st_ef = np.asarray(efast_result.ST)
_sort_idx = np.argsort(st_ana)

_fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
_y = np.arange(len(names))
_height = 0.35

_names_sorted = [names[i] for i in _sort_idx]

_ax1.barh(
    _y + _height / 2,
    _s1_ef[_sort_idx],
    _height,
    color="#1E88E5",
    label="eFAST",
)
_ax1.barh(
    _y - _height / 2,
    s1_ana[_sort_idx],
    _height,
    color="#1E88E5",
    alpha=0.35,
    label="Analytical",
)
_ax1.set_yticks(_y)
_ax1.set_yticklabels(_names_sorted)
_ax1.set_xlabel("$S_1$")
_ax1.set_title("First-order indices")
_ax1.legend(frameon=False, fontsize=8)
_ax1.grid(axis="x", alpha=0.3)

_ax2.barh(
    _y + _height / 2,
    _st_ef[_sort_idx],
    _height,
    color="#E53935",
    label="eFAST",
)
_ax2.barh(
    _y - _height / 2,
    st_ana[_sort_idx],
    _height,
    color="#E53935",
    alpha=0.35,
    label="Analytical",
)
_ax2.set_xlabel("$S_T$")
_ax2.set_title("Total-order indices")
_ax2.legend(frameon=False, fontsize=8)
_ax2.grid(axis="x", alpha=0.3)

_fig.suptitle("eFAST vs analytical — Oakley & O'Hagan", y=1.01)
_fig.tight_layout()
plt.show()

_key = jax.random.key(0)
X_hd = jax.random.normal(_key, (3000, problem.num_vars))
Y_hd = oakley_ohagan.evaluate(jnp.asarray(X_hd))

hdmr_result = jaxgsa.hdmr.analyze(problem, X_hd, Y_hd, maxorder=2, m=2)
print(hdmr_result)


def oakley_fn(x):
    """Unbatched Oakley & O'Hagan: (15,) -> ()."""
    return oakley_ohagan.evaluate(x[None, :])[0]


_test = oakley_fn(jnp.zeros(15))
print(f"f(0) = {_test:.4f}")

X_dg = jaxgsa.sampling.monte_carlo(problem, n=10000, seed=42)
dgsm_result = jaxgsa.dgsm.analyze(problem, oakley_fn, jnp.asarray(X_dg))
print(dgsm_result)

_ub = np.asarray(dgsm_result.upper_bound)
_lb = np.asarray(dgsm_result.lower_bound)
_sort_idx = np.argsort(st_ana)
_names_sorted = [names[i] for i in _sort_idx]
_y = np.arange(len(names))
_bw = 0.25

_fig, _ax = plt.subplots(figsize=(9, 6))
_ax.barh(
    _y - _bw,
    _lb[_sort_idx],
    _bw,
    label="Lower bound",
    color="#2196F3",
    alpha=0.85,
)
_ax.barh(
    _y,
    st_ana[_sort_idx],
    _bw,
    label="Analytical $S_T$",
    color="#4CAF50",
    alpha=0.85,
)
_ax.barh(
    _y + _bw,
    _ub[_sort_idx],
    _bw,
    label="Upper bound",
    color="#FF9800",
    alpha=0.85,
)
_ax.set_yticks(_y)
_ax.set_yticklabels(_names_sorted)
_ax.set_xlabel("Sensitivity index")
_ax.set_title("DGSM bounds vs analytical $S_T$ — Oakley & O'Hagan")
_ax.legend(frameon=False)
_ax.grid(axis="x", alpha=0.3)
_fig.tight_layout()
plt.show()

_sort_idx = np.argsort(st_ana)
_names_sorted = [names[i] for i in _sort_idx]
_y = np.arange(len(names))
_height = 0.2

_s1_ef = np.asarray(efast_result.S1)[_sort_idx]
_s1_hd = np.asarray(hdmr_result.S1)[_sort_idx]
_s1_ana = s1_ana[_sort_idx]

_fig, _ax = plt.subplots(figsize=(10, 7))
_ax.barh(
    _y - 1.5 * _height,
    _s1_ana,
    _height,
    color="#9E9E9E",
    label="Analytical",
)
_ax.barh(
    _y - 0.5 * _height,
    _s1_ef,
    _height,
    color="#1E88E5",
    label="eFAST",
)
_ax.barh(
    _y + 0.5 * _height,
    _s1_hd,
    _height,
    color="#43A047",
    label="RS-HDMR",
)

_ax.set_yticks(_y)
_ax.set_yticklabels(_names_sorted)
_ax.set_xlabel("$S_1$ (first-order index)")
_ax.set_title("Method comparison — first-order indices")
_ax.legend(frameon=False)
_ax.grid(axis="x", alpha=0.3)
_fig.tight_layout()
plt.show()

_top5_ana = set(np.argsort(st_ana)[-5:])

_st_ef = np.asarray(efast_result.ST)
_top5_ef = set(np.argsort(_st_ef)[-5:])
_match_ef = len(_top5_ana & _top5_ef)

_st_hd = np.asarray(hdmr_result.ST)
_top5_hd = set(np.argsort(_st_hd)[-5:])
_match_hd = len(_top5_ana & _top5_hd)

_ub_dg = np.asarray(dgsm_result.upper_bound)
_top5_dg = set(np.argsort(_ub_dg)[-5:])
_match_dg = len(_top5_ana & _top5_dg)


def _fmt_set(idx_set):
    return ", ".join(names[i] for i in sorted(idx_set))


_rows = [
    f"| Analytical | {_fmt_set(_top5_ana)} | -- |",
    f"| eFAST ($S_T$) | {_fmt_set(_top5_ef)} | {_match_ef}/5 |",
    f"| RS-HDMR ($S_T$) | {_fmt_set(_top5_hd)} | {_match_hd}/5 |",
    f"| DGSM (upper bound) | {_fmt_set(_top5_dg)} | {_match_dg}/5 |",
]
print("| Method | Top-5 parameters | Match |")
print("| --- | --- | ---: |")
for _row in _rows:
    print(_row)
