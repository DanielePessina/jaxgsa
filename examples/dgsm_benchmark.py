"""DGSM sensitivity analysis on Ishigami and linear benchmarks with gsax.

Demonstrates derivative-based global sensitivity measures (DGSM) using
JAX autodiff. Computes Poincare upper bounds and Kucherenko-Song lower
bounds on total Sobol indices, then compares against the known analytical
values.

Run interactively: ``uv run marimo edit examples/dgsm_benchmark.py``
Run as script:     ``uv run python examples/dgsm_benchmark.py``
"""

# ruff: noqa: F722

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # Derivative-based Global Sensitivity Measures (DGSM)

    DGSM computes sensitivity information from the **partial derivatives**
    of a model, rather than from variance decomposition. For
    JAX-differentiable models, this is essentially free via reverse-mode
    autodiff (`jax.jacrev`).

    The key quantities are:

    - $\nu_i = \mathbb{E}\!\left[\left(\frac{\partial f}{\partial x_i}\right)^{\!2}\right]$
      — the second moment of the partial derivative (importance measure)
    - $\sigma_i = \mathbb{E}\!\left[\frac{\partial f}{\partial x_i}\right]$
      — the mean partial derivative

    These yield **two-sided bounds** on the total Sobol index $S_{T_i}$:

    $$
    \frac{\mathrm{Var}(X_i) \cdot \sigma_i^2}{\mathrm{Var}(Y)}
    \;\leq\; S_{T_i} \;\leq\;
    \frac{C_i \cdot \nu_i}{\mathrm{Var}(Y)}
    $$

    where $C_i$ is the **Poincare constant** of the $i$-th input's
    marginal distribution.

    This notebook demonstrates DGSM on two standard benchmark functions.
    """)
    return


@app.cell
def _imports():
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import gsax
    from gsax.benchmarks import ishigami, linear
    from gsax.sampling import sample_mc

    return gsax, ishigami, jnp, linear, mo, np, plt, sample_mc


@app.cell(hide_code=True)
def _ishigami_md(mo):
    mo.md(r"""
    ## Ishigami function

    The Ishigami function $f(x_1, x_2, x_3) = \sin(x_1) + 7\sin^2(x_2) + 0.1\,x_3^4\sin(x_1)$
    is a standard sensitivity analysis benchmark with known analytical
    Sobol indices. Parameter $x_3$ has **zero first-order effect** but
    contributes through a higher-order interaction with $x_1$.

    We define an **unbatched** version for autodiff — DGSM needs
    `fn(x) -> (T,)` rather than the batched `evaluate(X) -> (N,)`.
    """)
    return


@app.cell
def _ishigami_fn(jnp):
    def ishigami_fn(x):
        """Unbatched Ishigami: (3,) -> ()."""
        A, B = 7.0, 0.1
        return jnp.sin(x[0]) + A * jnp.sin(x[1]) ** 2 + B * x[2] ** 4 * jnp.sin(x[0])

    return (ishigami_fn,)


@app.cell
def _ishigami_dgsm(gsax, ishigami, ishigami_fn, jnp, sample_mc):
    X_ish = sample_mc(ishigami.PROBLEM, N=50_000, seed=42)
    result_ish = gsax.analyze_dgsm(ishigami.PROBLEM, ishigami_fn, jnp.asarray(X_ish))
    return (result_ish,)


@app.cell(hide_code=True)
def _ishigami_plot(ishigami, np, plt, result_ish):
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
    _fig
    return


@app.cell(hide_code=True)
def _ishigami_table(ishigami, mo, np, result_ish):
    _st = np.array(ishigami.ANALYTICAL_ST)
    _nu = np.asarray(result_ish.nu)
    _sigma = np.asarray(result_ish.sigma)
    _lb = np.asarray(result_ish.lower_bound)
    _ub = np.asarray(result_ish.upper_bound)
    _names = list(ishigami.PROBLEM.names)

    rows = []
    for i, name in enumerate(_names):
        rows.append(
            f"| {name} | {_nu[i]:.4f} | {_sigma[i]:+.4f} | "
            f"{_lb[i]:.4f} | {_st[i]:.4f} | {_ub[i]:.4f} |"
        )
    table = (
        "| Param | $\\nu_i$ | $\\sigma_i$ | Lower | $S_T$ | Upper |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n" + "\n".join(rows)
    )
    mo.md(
        "### Ishigami numerical results\n\n" + table + "\n\n"
        "The bracket correctly contains the analytical $S_T$ for all "
        "three parameters. Note that $x_3$ has a nonzero upper bound "
        "(confirming its interaction effect) despite $\\sigma_3 \\approx 0$ "
        "(its mean derivative is near zero because the effect is symmetric)."
    )
    return


@app.cell(hide_code=True)
def _linear_md(mo):
    mo.md(r"""
    ## Linear model

    For a purely additive linear model $f(\mathbf{x}) = \sum_j c_j x_j$,
    the DGSM bracket **collapses to the exact value**: the partial
    derivatives are constant ($\partial f / \partial x_i = c_i$), so
    $\nu_i = c_i^2$ and $\sigma_i = c_i$ exactly. The lower and upper
    bounds coincide with $S_T$.
    """)
    return


@app.cell
def _linear_fn(jnp):
    def linear_fn(x):
        """Unbatched linear: (3,) -> ()."""
        c = jnp.array([1.0, 2.0, 3.0])
        return jnp.dot(c, x)

    return (linear_fn,)


@app.cell
def _linear_dgsm(gsax, jnp, linear, linear_fn, sample_mc):
    X_lin = sample_mc(linear.PROBLEM, N=10_000, seed=123)
    result_lin = gsax.analyze_dgsm(linear.PROBLEM, linear_fn, jnp.asarray(X_lin))
    return (result_lin,)


@app.cell(hide_code=True)
def _linear_plot(linear, np, plt, result_lin):
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
    _fig
    return


@app.cell(hide_code=True)
def _poincare_md(mo):
    mo.md(r"""
    ## Poincare constants

    The tightness of the DGSM upper bound depends on the **Poincare constant**
    $C_i$ of each input's marginal distribution:

    | Distribution | $C_i$ |
    | --- | --- |
    | Uniform $[a, b]$ | $(b - a)^2 / \pi^2$ |
    | Gaussian $\mathcal{N}(\mu, \sigma^2)$ | $\sigma^2$ |
    | Truncated Normal | Spectral solve (P1 FEM) |

    The upper bound becomes tight when the model response is nearly
    monotone in $x_i$. For strongly nonlinear or non-monotone responses
    (like the Ishigami function), the bound can be loose — this is the
    price of using only derivative information rather than a full
    variance decomposition.
    """)
    return


@app.cell(hide_code=True)
def _outro(mo):
    mo.md(r"""
    ## Summary

    DGSM provides **fast sensitivity screening** for JAX-differentiable
    models. The key advantages:

    1. **Speed**: One reverse-mode autodiff pass gives all $D$ partial
       derivatives simultaneously — no structured sampling design needed.
    2. **Bounds**: The Poincare upper bound and Kucherenko-Song lower
       bound bracket the total Sobol index without computing it exactly.
    3. **Multi-output**: All measures have shape $(K, D)$, handling
       scalar and multi-output models uniformly.

    For exact Sobol indices, use `gsax.analyze()` (Saltelli sampling)
    or `gsax.analyze_pce()` (polynomial chaos expansion).
    """)
    return


if __name__ == "__main__":
    app.run()
