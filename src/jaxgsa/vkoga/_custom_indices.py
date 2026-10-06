"""Existing VKOGA index definitions integrated under custom distributions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from jaxgsa._core.dependence import InputSampler, generate_uniform_points, transform_samples
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa.problem import Problem
from jaxgsa.vkoga._indices import (
    CorrelatedIndices,
    _clip_independent_part,
    _component_basis_kinds,
    _component_value,
    _evaluate_component,
    _fit_component_functions,
)


def checked_unit_samples(X: np.ndarray, problem: Problem) -> np.ndarray:
    """Transform physical rows, rejecting samples outside declared support.

    Args:
        X: Physical input rows.
        problem: Declared marginal distributions, matching the custom sampler.

    Returns:
        Marginal CDF coordinates.
    """
    for i, spec in enumerate(problem.input_specs):
        low = getattr(spec, "low", None)
        high = getattr(spec, "high", None)
        if (low is not None and np.any(X[:, i] < low)) or (
            high is not None and np.any(X[:, i] > high)
        ):
            raise ValueError(f"Custom samples outside declared support for {problem.names[i]!r}")
    return np.asarray(cdf_to_unit_interval(jnp.asarray(X), problem), dtype=np.float64)


def estimate_custom_indices(
    *,
    problem: Problem,
    sampler: InputSampler,
    sampler_dimension: int | None = None,
    sequence: str = "sobol",
    predict: Callable[[np.ndarray], np.ndarray],
    n_outer: int,
    n_inner: int,
    n_variance: int,
    entropy: int,
    param_names: Sequence[str] | None = None,
) -> CorrelatedIndices:
    """Integrate existing VKOGA definitions using independent random replicates.

    Two inner replicates per conditioning row estimate products of means
    without an IID variance correction. Two joint replicates normalize the
    indices. ``n_inner`` and ``n_variance`` count points in each replicate.
    This works with scrambled Sobol and Monte Carlo points. Component fits
    use the declared marginals, just as the built-in copula path does.

    Args:
        problem: Declared marginals.
        sampler: Uniform-point joint and conditional transformation.
        sampler_dimension: Uniform coordinates, defaulting to input dimension.
        sequence: ``sobol`` or ``monte_carlo``.
        predict: Surrogate accepting marginal CDF coordinates.
        n_outer: Number of conditioning rows.
        n_inner: Points in each independent inner replicate.
        n_variance: Points in each joint normalization replicate.
        entropy: Reproducible root entropy.
        param_names: Component diagnostic names.

    Returns:
        All five indices using the built-in definitions.
    """
    D = problem.num_vars
    dimension = D if sampler_dimension is None else sampler_dimension
    streams = np.random.SeedSequence(entropy).spawn(2 + 2 * D)

    def draw(
        n: int,
        stream: np.random.SeedSequence,
        rows: np.ndarray | None = None,
        fixed: tuple[int, ...] = (),
    ) -> np.ndarray:
        points = generate_uniform_points(
            n, dimension, sequence=sequence, rng=np.random.default_rng(stream)
        )
        return transform_samples(sampler, points, D, conditioning_rows=rows, fixed_indices=fixed)

    reference_X = transform_samples(sampler, np.full((1, dimension), 0.5), D)
    reference = np.asarray(predict(checked_unit_samples(reference_X, problem)), dtype=np.float64)
    U_replicates = [
        checked_unit_samples(draw(n_variance, stream), problem) for stream in streams[:2]
    ]
    Y_replicates = [np.asarray(predict(u), dtype=np.float64) - reference for u in U_replicates]
    mean_product = Y_replicates[0].mean(axis=0) * Y_replicates[1].mean(axis=0)
    U_var = np.concatenate(U_replicates)
    Y_var = np.concatenate(Y_replicates)
    variance = _replicate_variance(*Y_replicates)
    safe = np.where(variance > 0, variance, np.nan)
    kinds = _component_basis_kinds(problem)
    Z_var = norm.ppf(np.clip(U_var, 1e-12, 1 - 1e-12))
    coefficients = _fit_component_functions(U_var, Z_var, Y_var, kinds)
    n_slices = Y_var.shape[1]
    TC = np.empty((n_slices, D))
    TU = np.empty_like(TC)
    independent = np.empty_like(TC)
    for i in range(D):
        for first_order in (True, False):
            stream = streams[2 + i + (0 if first_order else D)]
            outer_stream, inner_stream = stream.spawn(2)
            conditioning_rows = draw(n_outer, outer_stream)
            checked_unit_samples(conditioning_rows, problem)
            fixed = (i,) if first_order else tuple(j for j in range(D) if j != i)
            mean_products = np.empty((n_outer, n_slices))
            variances = np.empty_like(mean_products)
            component_variances = np.empty_like(mean_products)
            for row, child in enumerate(inner_stream.spawn(n_outer)):
                rows = np.repeat(conditioning_rows[row : row + 1], n_inner, axis=0)
                outputs = []
                components = []
                for replicate in child.spawn(2):
                    U = checked_unit_samples(draw(n_inner, replicate, rows, fixed), problem)
                    outputs.append(np.asarray(predict(U), dtype=np.float64) - reference)
                    if not first_order:
                        z = norm.ppf(np.clip(U[:, i], 1e-12, 1 - 1e-12))
                        value = _component_value(U[:, i], z, kinds[i])
                        components.append(
                            _evaluate_component(value, coefficients[:, i, :], kinds[i])
                        )
                mean_products[row] = outputs[0].mean(axis=0) * outputs[1].mean(axis=0)
                variances[row] = _replicate_variance(*outputs)
                if not first_order:
                    component_variances[row] = _replicate_variance(*components)
            if first_order:
                TC[:, i] = mean_products.mean(axis=0) - mean_product
            else:
                TU[:, i] = variances.mean(axis=0)
                independent[:, i] = component_variances.mean(axis=0)
    TC /= safe[:, None]
    TU /= safe[:, None]
    independent = _clip_independent_part(independent / safe[:, None], TU, param_names)
    return CorrelatedIndices(TC, TU, independent, TC - independent, TU - independent, variance)


def _replicate_variance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Estimate variance from two independent integration replicates stably."""
    difference = first.mean(axis=0) - second.mean(axis=0)
    return (first.var(axis=0) + second.var(axis=0) + difference**2) / 2
