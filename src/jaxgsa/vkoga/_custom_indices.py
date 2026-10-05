"""IID conditional integration for user-supplied input distributions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from jaxgsa._core.dependence import (
    ConditionalSampler,
    JointSampler,
    draw_conditional_samples,
    draw_joint_samples,
)
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
    joint_sampler: JointSampler,
    conditional_sampler: ConditionalSampler,
    predict: Callable[[np.ndarray], np.ndarray],
    n_outer: int,
    n_inner: int,
    n_variance: int,
    entropy: int,
    param_names: Sequence[str] | None = None,
) -> CorrelatedIndices:
    """Integrate a surrogate with independent conditional Monte Carlo draws.

    Custom callbacks must match the marginals declared in ``problem``. The
    additive component projection uses those marginal polynomial bases, just
    like the built-in Gaussian-copula path. Each outer anchor has its own RNG
    stream; prediction batching therefore cannot change the random sample.

    Args:
        problem: Declared marginals of the target distribution.
        joint_sampler: Joint physical-input sampler.
        conditional_sampler: Conditional physical-input sampler.
        predict: Surrogate accepting marginal CDF coordinates.
        n_outer: Number of outer anchors per conditional estimator.
        n_inner: IID conditional draws per anchor.
        n_variance: Joint draws for variance and additive projection.
        entropy: Reproducible root entropy.
        param_names: Names for component-projection diagnostics.

    Returns:
        All five indices using the same definitions as the built-in path.
    """
    D = problem.num_vars
    streams = np.random.SeedSequence(entropy).spawn(1 + 2 * D)
    X_var = draw_joint_samples(joint_sampler, n_variance, D, np.random.default_rng(streams[0]))
    U_var = checked_unit_samples(X_var, problem)
    Y_var = np.asarray(predict(U_var), dtype=np.float64)
    variance = Y_var.var(axis=0, ddof=1)
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
            stream = streams[1 + i + (0 if first_order else D)]
            outer_stream, inner_stream = stream.spawn(2)
            anchors = draw_joint_samples(
                joint_sampler, n_outer, D, np.random.default_rng(outer_stream)
            )
            checked_unit_samples(anchors, problem)
            fixed = (i,) if first_order else tuple(j for j in range(D) if j != i)
            means = np.empty((n_outer, n_slices))
            variances = np.empty_like(means)
            component_variances = np.empty_like(means)
            # One anchor at a time bounds conditional sampling memory without
            # making callback RNG consumption depend on prediction batch size.
            for row, child in enumerate(inner_stream.spawn(n_outer)):
                conditional = draw_conditional_samples(
                    conditional_sampler,
                    np.repeat(anchors[row : row + 1], n_inner, axis=0),
                    fixed,
                    np.random.default_rng(child),
                )
                U = checked_unit_samples(conditional, problem)
                Y = np.asarray(predict(U), dtype=np.float64)
                means[row] = Y.mean(axis=0)
                variances[row] = Y.var(axis=0, ddof=1)
                if not first_order:
                    z = norm.ppf(np.clip(U[:, i], 1e-12, 1 - 1e-12))
                    value = _component_value(U[:, i], z, kinds[i])
                    f = _evaluate_component(value, coefficients[:, i, :], kinds[i])
                    component_variances[row] = f.var(axis=0, ddof=1)
            if first_order:
                # IID inner error inflates Var(inner mean) by E Var(Y|Xi)/m.
                TC[:, i] = means.var(axis=0, ddof=1) - variances.mean(axis=0) / n_inner
            else:
                TU[:, i] = variances.mean(axis=0)
                # Direct conditional variance is equivalent to the law-of-
                # total-variance subtraction and avoids its inner-mean bias.
                independent[:, i] = component_variances.mean(axis=0)
    TC /= safe[:, None]
    TU /= safe[:, None]
    independent = _clip_independent_part(independent / safe[:, None], TU, param_names)
    return CorrelatedIndices(TC, TU, independent, TC - independent, TU - independent, variance)
