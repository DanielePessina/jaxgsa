"""Variance-based sensitivity indices for correlated inputs (Li et al. 2010).

When inputs are dependent the Sobol' variance decomposition no longer holds,
but the same conditional-variance quantities remain well defined and simply
change connotation (Hilhorst et al. 2024, Figure 1):

===============  ===================================  ==========================
index            definition                           reading
===============  ===================================  ==========================
``S_TC``         ``V(E(Y|X_i)) / V(Y)``               total *correlated*: what
                                                      ``X_i`` explains through
                                                      itself and its correlation
``S_TU``         ``E(V(Y|X_-i)) / V(Y)``              total *uncorrelated*: what
                                                      only ``X_i`` can explain
``S_U``          ``[V(f_i) - V(E(f_i|X_-i))] / V(Y)`` the independent part
``S_C``          ``S_TC - S_U``                       the correlation-borne part
``S_IU``         ``S_TU - S_U``                       independent interactions
===============  ===================================  ==========================

Every expectation is taken under a Gaussian copula, whose conditionals are
closed-form in the latent normal space, and every model evaluation goes through
a fitted surrogate — which is the whole point of the two-stage approach: the
nested conditional sampling would be unaffordable against the original model.

References:
    Li, Rabitz, Yelvington et al. (2010). J. Phys. Chem. A 114:6022-6032.
    Hilhorst, Quicken, van de Vosse & Huberts (2024). Int. J. Numer. Meth.
        Biomed. Engng. 40(2):e3797.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
from scipy.stats import norm

from jaxgsa._core.copula import _ConditionalPlan, latent_normal_sample

# Degree of the marginal Legendre basis used to recover the first-order
# component functions f_i. Six terms capture the smooth univariate shapes a
# kernel surrogate produces without making the joint least squares ill-posed.
_COMPONENT_DEGREE = 6

# Ridge added to the component-function normal equations. The basis is
# marginally orthonormal, so the Gram matrix is well conditioned unless two
# parameters are near-perfectly correlated; this only covers that case.
_COMPONENT_RIDGE = 1e-8


class CorrelatedIndices(NamedTuple):
    """Raw index arrays produced by :func:`estimate_correlated_indices`.

    Every array has shape ``(S, D)`` for ``S`` output slices and ``D``
    parameters, matching the surrogate's flattened output layout.

    Attributes:
        S_TC: Total correlated indices.
        S_TU: Total uncorrelated indices.
        S_U: Uncorrelated (independent) contribution.
        S_C: Correlated contribution, ``S_TC - S_U``.
        S_IU: Independent interaction contribution, ``S_TU - S_U``.
        variance: ``(S,)`` output variance under the correlated input measure.
    """

    S_TC: np.ndarray
    S_TU: np.ndarray
    S_U: np.ndarray
    S_C: np.ndarray
    S_IU: np.ndarray
    variance: np.ndarray


def estimate_correlated_indices(
    *,
    plan: _ConditionalPlan,
    chol_full: np.ndarray,
    predict: Callable[[np.ndarray], np.ndarray],
    n_outer: int,
    n_inner: int,
    n_variance: int,
    seed: int,
) -> CorrelatedIndices:
    """Estimate the five correlated variance-based indices by quasi-Monte-Carlo.

    Everything stays in copula (unit-cube) coordinates: ``u_i = Phi(z_i)`` is
    exactly the marginal CDF value ``F_i(x_i)``, which is the space the
    surrogate is fitted in, so no round-trip through physical units is needed
    for the millions of conditional draws below.

    Args:
        plan: Precomputed Gaussian conditionals from
            :func:`jaxgsa._core.copula.build_conditional_plan`.
        chol_full: ``(D, D)`` Cholesky factor of the copula correlation matrix,
            used for the unconditional variance sample.
        predict: Maps ``(n, D)`` unit-cube samples to ``(n, S)`` outputs. This
            is the fitted surrogate.
        n_outer: Outer (conditioning) sample size per parameter.
        n_inner: Inner (conditional) sample size per outer point.
        n_variance: Sample size for the unconditional output variance and for
            the component-function fit.
        seed: Base seed; each parameter and stage derives a distinct stream.

    Returns:
        A :class:`CorrelatedIndices` with ``(S, D)`` arrays.
    """
    D = chol_full.shape[0]

    # --- Unconditional variance, and the additive component functions --------
    Z_var = latent_normal_sample(n_variance, D, seed=seed) @ chol_full.T
    U_var = norm.cdf(Z_var)
    Y_var = np.asarray(predict(U_var), dtype=np.float64)
    variance = Y_var.var(axis=0)
    # A zero-variance slice makes every index undefined rather than zero; NaN
    # propagates that honestly, matching the convention in hdmr._engine._ancova.
    safe_variance = np.where(variance > 0.0, variance, np.nan)

    coefficients = _fit_component_functions(U_var, Y_var)

    n_slices = Y_var.shape[1]
    S_TC = np.empty((n_slices, D), dtype=np.float64)
    S_TU = np.empty((n_slices, D), dtype=np.float64)
    S_U = np.empty((n_slices, D), dtype=np.float64)

    for i in range(D):
        S_TC[:, i] = _total_correlated(
            plan=plan,
            index=i,
            predict=predict,
            n_outer=n_outer,
            n_inner=n_inner,
            seed=seed + 1 + i,
        )
        total_uncorrelated, conditional_component_var = _total_uncorrelated_and_conditional(
            plan=plan,
            index=i,
            predict=predict,
            component=coefficients[:, i, :],
            n_outer=n_outer,
            n_inner=n_inner,
            seed=seed + 1 + D + i,
        )
        S_TU[:, i] = total_uncorrelated
        # Equation (25): the independent contribution is what f_i explains
        # beyond the part of itself that X_-i already determines through the
        # correlation. V(f_i) is a marginal quantity, so it comes from the
        # joint variance sample rather than from the nested conditional draws.
        f_i = _evaluate_component(U_var[:, i], coefficients[:, i, :])
        S_U[:, i] = np.maximum(f_i.var(axis=0) - conditional_component_var, 0.0)

    S_TC /= safe_variance[:, None]
    S_TU /= safe_variance[:, None]
    S_U /= safe_variance[:, None]
    return CorrelatedIndices(
        S_TC=S_TC,
        S_TU=S_TU,
        S_U=S_U,
        # Identities (24) and (28) of Hilhorst et al.: the correlated part is
        # whatever the total correlated index holds beyond the independent
        # contribution, and likewise for independent interactions.
        S_C=S_TC - S_U,
        S_IU=S_TU - S_U,
        variance=variance,
    )


def _legendre_basis(u: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate shifted Legendre polynomials 1..degree on ``u`` in (0, 1).

    Each ``u_i = F_i(X_i)`` is exactly uniform on (0, 1) by construction of the
    copula, whatever the dependency structure, so shifted Legendre polynomials
    are marginally orthonormal and mean-zero here. That makes the additive
    least squares below well posed without any extra centering.

    Args:
        u: ``(..., )`` values in (0, 1).
        degree: Highest polynomial degree; term 0 (the constant) is dropped.

    Returns:
        ``(..., degree)`` array of basis values.
    """
    x = 2.0 * u - 1.0  # map (0, 1) onto the Legendre domain (-1, 1)
    columns = []
    previous, current = np.ones_like(x), x
    for k in range(1, degree + 1):
        # Normalised so each column has unit variance under a uniform marginal.
        columns.append(current * np.sqrt(2.0 * k + 1.0))
        previous, current = (
            current,
            ((2 * k + 1) * x * current - k * previous) / (k + 1),
        )
    return np.stack(columns, axis=-1)


def _fit_component_functions(U: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fit the additive first-order component functions ``f_i``.

    Least squares of the output onto ``sum_i f_i(X_i)`` under the *correlated*
    input measure is exactly the decomposition Li et al. use: the fitted
    ``f_i`` absorbs both what ``X_i`` explains alone and what it explains
    through its correlation with the rest, which is why the independent part
    has to be extracted afterwards by conditioning.

    Args:
        U: ``(N, D)`` marginal CDF values of the sample.
        Y: ``(N, S)`` surrogate outputs.

    Returns:
        ``(degree, D, S)`` basis coefficients.
    """
    N, D = U.shape
    basis = _legendre_basis(U, _COMPONENT_DEGREE)  # (N, D, degree)
    design = basis.reshape(N, D * _COMPONENT_DEGREE)

    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += _COMPONENT_RIDGE * N
    coefficients = np.linalg.solve(gram, design.T @ (Y - Y.mean(axis=0)))
    return coefficients.reshape(D, _COMPONENT_DEGREE, -1).transpose(1, 0, 2)


def _evaluate_component(u: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate one component function ``f_i`` at ``u``.

    Args:
        u: ``(...,)`` marginal CDF values for parameter ``i``.
        coefficients: ``(degree, S)`` coefficients for that parameter.

    Returns:
        ``(..., S)`` component values.
    """
    return _legendre_basis(u, _COMPONENT_DEGREE) @ coefficients


def _assemble_latent(
    index: int,
    others: np.ndarray,
    z_self: np.ndarray,
    z_rest: np.ndarray,
) -> np.ndarray:
    """Interleave a parameter's latent draw with the other parameters'.

    Args:
        index: Parameter being conditioned on or resampled.
        others: ``(D - 1,)`` indices of the remaining parameters.
        z_self: ``(n,)`` latent values for ``index``.
        z_rest: ``(n, D - 1)`` latent values for the others.

    Returns:
        ``(n, D)`` latent sample in parameter order.
    """
    n = z_self.shape[0]
    Z = np.empty((n, others.shape[0] + 1), dtype=np.float64)
    Z[:, index] = z_self
    Z[:, others] = z_rest
    return Z


def _total_correlated(
    *,
    plan: _ConditionalPlan,
    index: int,
    predict: Callable[[np.ndarray], np.ndarray],
    n_outer: int,
    n_inner: int,
    seed: int,
) -> np.ndarray:
    """Estimate ``V(E(Y|X_i))`` for one parameter.

    Returns:
        ``(S,)`` conditional-expectation variance, not yet normalised.
    """
    others = plan.others[index]
    z_self = latent_normal_sample(n_outer, 1, seed=seed)[:, 0]
    # One inner block, reused across every outer point. Sharing it makes the
    # inner integration rule identical for all conditioning values, which
    # cancels much of its error out of the outer variance.
    inner = latent_normal_sample(n_inner, others.shape[0], seed=seed + 7919)

    # Z_-i | Z_i = beta * z_i + L eps.
    z_rest = (
        plan.beta_rest[index][None, None, :] * z_self[:, None, None]
        + inner @ plan.chol_rest[index].T
    )
    z_self_tiled = np.repeat(z_self, n_inner)
    Z = _assemble_latent(index, others, z_self_tiled, z_rest.reshape(-1, others.shape[0]))

    Y = np.asarray(predict(norm.cdf(Z)), dtype=np.float64)
    Y = Y.reshape(n_outer, n_inner, -1)

    inner_mean = Y.mean(axis=1)
    # The variance of a noisy mean overstates V(E(Y|X_i)) by the inner sampling
    # variance; subtract it rather than leaning on a large n_inner.
    inner_var = Y.var(axis=1, ddof=1)
    return np.maximum(inner_mean.var(axis=0) - inner_var.mean(axis=0) / n_inner, 0.0)


def _total_uncorrelated_and_conditional(
    *,
    plan: _ConditionalPlan,
    index: int,
    predict: Callable[[np.ndarray], np.ndarray],
    component: np.ndarray,
    n_outer: int,
    n_inner: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ``E(V(Y|X_-i))`` and ``V(E(f_i|X_-i))`` for one parameter.

    Both quantities condition on ``X_-i`` and resample ``X_i``, so they share
    one nested sample: the surrogate gives the total uncorrelated index, and
    the fitted component function evaluated on the same draws gives the
    conditional expectation Equation (25) subtracts.

    Returns:
        ``(S,)`` conditional-variance mean and ``(S,)`` variance of
        ``E(f_i|X_-i)``, neither normalised by ``V(Y)``.
    """
    others = plan.others[index]
    z_rest = (
        latent_normal_sample(n_outer, others.shape[0], seed=seed) @ plan.chol_marginal[index].T
    )
    inner = latent_normal_sample(n_inner, 1, seed=seed + 7919)[:, 0]

    # Z_i | Z_-i = beta . z_-i + sigma eps.
    conditional_mean = z_rest @ plan.beta_self[index]
    z_self = conditional_mean[:, None] + plan.std_self[index] * inner[None, :]

    z_rest_tiled = np.repeat(z_rest, n_inner, axis=0)
    Z = _assemble_latent(index, others, z_self.reshape(-1), z_rest_tiled)

    Y = np.asarray(predict(norm.cdf(Z)), dtype=np.float64)
    Y = Y.reshape(n_outer, n_inner, -1)
    total_uncorrelated = Y.var(axis=1, ddof=1).mean(axis=0)

    # E(f_i | X_-i), averaged over the same inner draws of X_i. Its variance
    # over the outer sample is the correlation-explained part of f_i that
    # Equation (25) removes.
    f_hat = _evaluate_component(norm.cdf(z_self), component).mean(axis=1)
    return total_uncorrelated, f_hat.var(axis=0)
