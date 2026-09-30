"""Fit uncertainty ranges to target Sobol indices for an interacting toy model.

Run with ``uv run --locked examples/inverse_sobol_bounds.py``. Outputs are saved
under ``.scratch/inverse-sobol-bounds/`` by default. All model inputs and outputs
are dimensionless. This fits variance shares, rather than minimizing variance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator, NullLocator
from numpy.typing import NDArray
from scipy.optimize import minimize

import jaxgsa

FloatArray = NDArray[np.float64]
INITIAL_WIDTHS = np.array([1.0, 2.0])
TARGET = np.array([0.4, 0.4])
EXACT_WIDTHS = np.full(2, np.sqrt(1.5))


def model(x: jax.Array) -> jax.Array:
    """Evaluate the additive terms and their interaction for a batch of rows."""
    x1, x2 = x[:, 0], x[:, 1]
    return x1 + x2 + x1 * x2


def analytical_indices(widths: FloatArray) -> tuple[FloatArray, FloatArray, float, float]:
    """Return exact S1, ST, interaction share, and output variance.

    Args:
        widths: Positive half-widths of two independent symmetric uniforms.

    Returns:
        First-order indices, total-order indices, S2_12, and Var(Y).
    """
    variances = widths**2 / 3.0
    interaction = float(np.prod(variances))
    variance_y = float(variances.sum()) + interaction
    return (
        variances / variance_y,
        (variances + interaction) / variance_y,
        interaction / variance_y,
        variance_y,
    )


def save_figure(history: FloatArray, report: dict, output_dir: Path) -> None:
    """Plot optimization progress and independently validated indices."""
    style = Path(__file__).parent / "styles" / "paper.mplstyle"
    with plt.style.context(str(style)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
        steps = np.arange(len(history))
        axes[0].semilogy(steps, np.maximum(history[:, 0], np.finfo(float).tiny), marker="o")
        axes[0].set(xlabel="Optimizer iteration", ylabel="Squared target error", title="Fit")
        axes[0].yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=200))
        for i in range(2):
            axes[1].plot(steps, history[:, i + 1], marker="o", label=f"S1, x{i + 1}")
        axes[1].axhline(0.4, color="0.25", linestyle="--", linewidth=1, label="Target")
        axes[1].set(xlabel="Optimizer iteration", ylabel="First-order index", title="Fixed design")
        axes[1].yaxis.set_minor_locator(AutoMinorLocator(2))
        axes[1].legend()
        positions = np.arange(2)
        axes[2].bar(positions - 0.18, report["validation_S1"], 0.36, label="S1", zorder=3)
        axes[2].bar(positions + 0.18, report["validation_ST"], 0.36, label="ST", zorder=3)
        axes[2].axhline(0.4, color="C0", linestyle="--", linewidth=1, label="Exact S1 target")
        axes[2].axhline(0.6, color="C1", linestyle=":", linewidth=1, label="Exact implied ST")
        axes[2].set(xticks=positions, xticklabels=["x1", "x2"], ylim=(0, 1), title="Fresh design")
        axes[2].xaxis.set_minor_locator(NullLocator())
        axes[2].yaxis.set_minor_locator(AutoMinorLocator(2))
        axes[2].legend()
        for ax in axes[:2]:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        for i, ax in enumerate(axes):
            ax.set_axisbelow(True)
            grid_axis = "y" if i == 2 else "both"
            ax.grid(which="major", axis=grid_axis, alpha=0.28, linewidth=0.5)
            ax.grid(which="minor", axis=grid_axis, alpha=0.10, linewidth=0.35)
        fig.savefig(output_dir / "inverse_sobol_bounds.png")
        fig.savefig(output_dir / "inverse_sobol_bounds.svg")
        plt.close(fig)


def main() -> None:
    """Check gradients, fit the widths, and verify against independent evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".scratch/inverse-sobol-bounds"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    problem = jaxgsa.Problem.from_dict({"x1": (-1.0, 1.0), "x2": (-2.0, 2.0)})
    design = jaxgsa.sobol.sample(
        problem, n_samples=1, base_n=4096, calc_second_order=False, seed=42, verbose=False
    )

    def first_order(widths: jax.Array) -> jax.Array:
        theta = {
            name: {"low": -widths[i], "high": widths[i]} for i, name in enumerate(problem.names)
        }
        return jaxgsa.sobol.indices(design, model(design.transform(theta)))[0]

    def loss(log_widths: jax.Array) -> jax.Array:
        return jnp.sum((first_order(jnp.exp(log_widths)) - jnp.asarray(TARGET)) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss))

    def objective(log_widths: FloatArray) -> tuple[float, FloatArray]:
        value, gradient = value_and_grad(jnp.asarray(log_widths))
        return float(value), np.asarray(gradient)

    initial = jnp.asarray(INITIAL_WIDTHS)
    jacobian = np.asarray(jax.jacrev(first_order)(initial))
    step = 1e-5
    finite_difference = np.column_stack(
        [
            np.asarray(
                first_order(initial + step * jnp.eye(2)[i])
                - first_order(initial - step * jnp.eye(2)[i])
            )
            / (2 * step)
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(jacobian, finite_difference, rtol=1e-6, atol=1e-8)

    # Vary only low: upper bounds and the other distribution stay fixed.
    def first_order_from_low(low: jax.Array) -> jax.Array:
        x = design.transform({"x1": {"low": low}})
        return jaxgsa.sobol.indices(design, model(x))[0]

    lower_gradient = np.asarray(jax.jacrev(first_order_from_low)(jnp.asarray(-1.0)))
    history: list[list[float]] = []

    def record(log_widths: FloatArray) -> None:
        indices = np.asarray(first_order(jnp.exp(jnp.asarray(log_widths))))
        history.append([float(np.sum((indices - TARGET) ** 2)), *indices.tolist()])

    record(np.log(INITIAL_WIDTHS))
    optimum = minimize(
        objective,
        np.log(INITIAL_WIDTHS),
        jac=True,
        method="L-BFGS-B",
        bounds=[(np.log(0.1), np.log(4.0))] * 2,
        callback=record,
        options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 100},
    )
    if not optimum.success:
        raise RuntimeError(f"Optimization failed: {optimum.message}")
    fitted = np.exp(optimum.x)
    exact_s1, exact_st, exact_s2, variance_y = analytical_indices(fitted)
    np.testing.assert_allclose(fitted, EXACT_WIDTHS, rtol=0.01)
    np.testing.assert_allclose(exact_s1, TARGET, atol=0.002)

    # New scramble and more points; include second order to check the interaction.
    validation = jaxgsa.sobol.sample(problem, n_samples=1, base_n=32768, seed=123, verbose=False)
    fitted_theta = {
        name: {"low": -fitted[i], "high": fitted[i]} for i, name in enumerate(problem.names)
    }
    result = jaxgsa.sobol.analyze(
        validation, model(validation.transform(fitted_theta)), verbose=False
    )
    assert isinstance(result, jaxgsa.sobol.SobolResult)
    assert result.S2 is not None
    validation_s1, validation_st = np.asarray(result.S1), np.asarray(result.ST)
    validation_s2 = float(result.S2[0, 1])
    np.testing.assert_allclose(validation_s1, exact_s1, atol=0.002)
    np.testing.assert_allclose(validation_st, exact_st, atol=0.002)
    np.testing.assert_allclose(validation_s2, exact_s2, atol=0.002)
    np.testing.assert_allclose(validation_s1, TARGET, atol=0.002)
    np.testing.assert_allclose(validation_st, [0.6, 0.6], atol=0.002)
    np.testing.assert_allclose(validation_s2, 0.2, atol=0.002)

    report = {
        "model": "Y = X1 + X2 + X1*X2; independent Xi ~ Uniform(-wi, wi)",
        "target_S1": TARGET.tolist(),
        "initial_widths": INITIAL_WIDTHS.tolist(),
        "initial_exact_S1": analytical_indices(INITIAL_WIDTHS)[0].tolist(),
        "initial_dS1_dwidth": jacobian.tolist(),
        "initial_dS1_dx1_low": lower_gradient.tolist(),
        "gradient_max_finite_difference_error": float(np.max(abs(jacobian - finite_difference))),
        "fitted_widths": fitted.tolist(),
        "exact_target_widths": EXACT_WIDTHS.tolist(),
        "training_S1": np.asarray(first_order(jnp.asarray(fitted))).tolist(),
        "validation_S1": validation_s1.tolist(),
        "validation_ST": validation_st.tolist(),
        "validation_S2_12": validation_s2,
        "fitted_exact_S1": exact_s1.tolist(),
        "fitted_exact_variance_y": variance_y,
        "initial_exact_variance_y": analytical_indices(INITIAL_WIDTHS)[3],
        "training_base_n": design.base_n,
        "validation_base_n": validation.base_n,
        "iterations": int(optimum.nit),
        "history_columns": ["loss", "S1_x1", "S1_x2"],
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    save_figure(np.asarray(history), report, args.output_dir)
    print(f"Starting widths: {INITIAL_WIDTHS}")
    print(f"dS1/dwidth (rows: indices; columns: widths):\n{jacobian}")
    print(f"dS1/d(x1 lower bound), holding upper bound fixed: {lower_gradient}")
    print(f"Fitted widths: {fitted}; exact: {EXACT_WIDTHS}")
    print(f"Fresh-design S1: {validation_s1}; ST: {validation_st}; S2_12: {validation_s2:.6f}")
    print(
        f"Checks passed in {optimum.nit} optimizer iterations. "
        f"Outputs: {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
