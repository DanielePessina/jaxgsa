"""Variance-based sensitivity indices for correlated inputs (Li et al. 2010).

When inputs are dependent the Sobol' variance decomposition no longer holds.
The same conditional-variance quantities stay well defined, but they change
connotation (Hilhorst et al. 2024, Figure 1):

===============  ===================================  ==========================
index            definition                           reading
===============  ===================================  ==========================
``S_TC``         ``V(E(Y|X_i)) / V(Y)``               total correlated: what
                                                      ``X_i`` explains through
                                                      itself and through its
                                                      correlation. It is a
                                                      first-order quantity in
                                                      form. "Total" names the
                                                      pathways it counts, not
                                                      the interaction order
``S_TU``         ``E(V(Y|X_-i)) / V(Y)``              total uncorrelated: what
                                                      only ``X_i`` can explain
``S_U``          ``E(V(f_i|X_-i)) / V(Y)``            the independent part
``S_C``          ``S_TC - S_U``                       the correlation-borne part
``S_IU``         ``S_TU - S_U``                       independent interactions
===============  ===================================  ==========================

``f_i`` is the additive first-order component of ``Y``. It is fitted by least
squares under the correlated input measure (see
:func:`_fit_component_functions`). ``S_U`` therefore measures the part of
``f_i`` that ``X_-i`` cannot predict. This is the decorrelated first-order
index of Mara & Tarantola (2012), not the structural index
``Var(f_i) / V(Y)`` of Li et al. (2010, Equation 25). The two differ on a
linear-Gaussian model: Li's keeps the correlated share of ``Var(f_i)`` and this
one removes it. The pair ``(S_U, S_C)`` splits ``S_TC`` along exactly the
boundary the decorrelated index draws.

Every expectation is taken under a Gaussian copula, whose conditionals are
closed-form in the latent normal space. Every model evaluation goes through a
fitted surrogate. That is the point of the two-stage approach: the nested
conditional sampling would be unaffordable against the original model.

References:
    Li, Rabitz, Yelvington et al. (2010). J. Phys. Chem. A 114:6022-6032.
    Mara & Tarantola (2012). Reliab. Eng. Syst. Saf. 107:115-121.
    Hilhorst, Quicken, van de Vosse & Huberts (2024). Int. J. Numer. Meth.
        Biomed. Engng. 40(2):e3797.
"""

from __future__ import annotations

import warnings
from typing import Callable, NamedTuple

import numpy as np
from scipy.stats import norm

from jaxgsa._core.batching import resolve_batch_size
from jaxgsa._core.copula import (
    _ConditionalPlan,
    assemble_latent,
    draw_rest_given_self,
    draw_self_given_rest,
    latent_normal_sample,
)
from jaxgsa._core.legendre import legendre_orthonormal

# Degree of the marginal Legendre basis used to recover the first-order
# component functions f_i. Six terms capture the smooth univariate shapes a
# kernel surrogate produces without making the joint least squares ill-posed.
_COMPONENT_DEGREE = 6

# Ridge added to the component-function normal equations. The basis is
# marginally orthonormal, so the Gram matrix is well conditioned unless two
# parameters are near-perfectly correlated; this only covers that case.
_COMPONENT_RIDGE = 1e-8

# How far S_U may exceed S_TU before the clip is reported. Both are fractions
# of V(Y), so this is one percent of the output variance: below it the excess
# is estimator noise, above it the additive projection is genuinely inadequate.
_S_U_CLIP_TOLERANCE = 0.01


class CorrelatedIndices(NamedTuple):
    """Raw index arrays produced by :func:`estimate_correlated_indices`.

    Index arrays have shape ``(S, D)`` for ``S`` output slices and ``D``
    parameters, matching the surrogate's flattened output layout.

    Attributes:
        S_TC: Total correlated indices ``V(E(Y|X_i)) / V(Y)``, shape ``(S, D)``.
        S_TU: Total uncorrelated indices ``E(V(Y|X_-i)) / V(Y)``, shape
            ``(S, D)``.
        S_U: Uncorrelated (independent) contribution, shape ``(S, D)``. Clipped
            to at most ``S_TU``.
        S_C: Correlated contribution ``S_TC - S_U``, shape ``(S, D)``. It may be
            negative.
        S_IU: Independent interaction contribution ``S_TU - S_U``, shape
            ``(S, D)``. Non-negative by construction of the clip.
        variance: Output variance under the correlated input measure, shape
            ``(S,)``.
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

    Everything stays in copula (unit-cube) coordinates. ``u_i = Phi(z_i)`` is
    exactly the marginal CDF value ``F_i(x_i)``, and that is the space the
    surrogate is fitted in. The millions of conditional draws below therefore
    need no round-trip through physical units.

    Args:
        plan: Precomputed Gaussian conditionals from
            :func:`jaxgsa._core.copula.build_conditional_plan`.
        chol_full: Cholesky factor of the copula correlation matrix, shape
            ``(D, D)``. Used for the unconditional variance sample.
        predict: Fitted surrogate. It maps ``(n, D)`` unit-cube samples to
            ``(n, S)`` outputs.
        n_outer: Outer (conditioning) sample size per parameter.
        n_inner: Inner (conditional) sample size per outer point.
        n_variance: Sample size for the unconditional output variance and for
            the component-function fit.
        seed: Base seed. Each parameter and stage derives a distinct stream.

    Returns:
        A :class:`CorrelatedIndices` whose index arrays have shape ``(S, D)``.
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
            n_slices=n_slices,
            seed=seed + 1 + i,
        )
        total_uncorrelated, conditional_component_var = _total_uncorrelated_and_conditional(
            plan=plan,
            index=i,
            predict=predict,
            component=coefficients[:, i, :],
            n_outer=n_outer,
            n_inner=n_inner,
            n_slices=n_slices,
            seed=seed + 1 + D + i,
        )
        S_TU[:, i] = total_uncorrelated
        # The independent contribution is what f_i explains beyond the part of
        # itself that X_-i already determines through the correlation. V(f_i)
        # is a marginal quantity, so it comes from the joint variance sample
        # rather than from the nested conditional draws.
        f_i = _evaluate_component(U_var[:, i], coefficients[:, i, :])
        S_U[:, i] = np.maximum(f_i.var(axis=0) - conditional_component_var, 0.0)

    S_TC /= safe_variance[:, None]
    S_TU /= safe_variance[:, None]
    S_U /= safe_variance[:, None]
    S_U = _clip_independent_part(S_U, S_TU)
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


def _clip_independent_part(S_U: np.ndarray, S_TU: np.ndarray) -> np.ndarray:
    """Hold ``S_U <= S_TU`` so ``S_IU`` cannot go negative, and report the clip.

    ``S_TU`` is everything ``X_i`` alone explains, and ``S_U`` is the part of
    that which the additive component ``f_i`` accounts for. The remainder,
    ``S_IU``, is the independent interaction share, so ``S_U <= S_TU`` is a
    definitional invariant of the split.

    The two indices come from different estimators, and only ``S_TU`` sees the
    surrogate directly. On a model with interactions the degree-6 additive
    projection ``f_i`` absorbs interaction variance that it cannot represent as
    a function of ``X_i`` alone, so under a correlated measure the raw ``S_U``
    can exceed ``S_TU``. Clipping restores the invariant. A clip wider than
    ``_S_U_CLIP_TOLERANCE`` is not noise: it says the additive projection is
    inadequate for this model, and it warns.

    ``S_C = S_TC - S_U`` stays free to go negative. That is a real reading, a
    correlation that opposes the direct effect. It is not an estimator artefact.

    Args:
        S_U: Independent contribution normalised by ``V(Y)``, shape ``(S, D)``.
        S_TU: Total uncorrelated index normalised by ``V(Y)``, shape ``(S, D)``.

    Returns:
        ``S_U`` clipped elementwise to at most ``S_TU``.

    Warns:
        UserWarning: If any entry is clipped by more than
            ``_S_U_CLIP_TOLERANCE`` of the output variance.
    """
    excess = S_U - S_TU
    # NaN slices (zero output variance) compare False and stay out of the report.
    flagged = excess > _S_U_CLIP_TOLERANCE
    if flagged.any():
        params = sorted({int(j) for j in np.argwhere(flagged)[:, 1]})
        warnings.warn(
            f"jaxgsa.vkoga: S_U exceeded S_TU by up to {float(excess[flagged].max()):.3g} "
            f"of the output variance for parameter(s) {params}; S_U was clipped to S_TU so "
            "that S_IU stays non-negative. The additive component functions cannot represent "
            "this model's interactions under the correlated measure, so read S_U, S_C and "
            "S_IU for those parameters as indicative only. S_TC and S_TU are unaffected.",
            # 1 here, 2 estimate_correlated_indices, 3 analyze_vkoga, 4 the caller.
            stacklevel=4,
        )
    return np.minimum(S_U, S_TU)


def _legendre_basis(u: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate shifted Legendre polynomials 1..degree on ``u`` in (0, 1).

    Each ``u_i = F_i(X_i)`` is exactly uniform on (0, 1) by construction of the
    copula, whatever the dependency structure, so shifted Legendre polynomials
    are marginally orthonormal and mean-zero here. That makes the additive
    least squares below well posed without any extra centering.

    Thin wrapper over the shared recurrence in :mod:`jaxgsa._core.legendre`:
    shift ``(0, 1)`` onto ``(-1, 1)`` and drop the constant term.

    Args:
        u: Values in (0, 1), shape ``(...,)``.
        degree: Highest polynomial degree. Term 0 (the constant) is dropped.

    Returns:
        Basis values, shape ``(..., degree)``.
    """
    return legendre_orthonormal(2.0 * u - 1.0, degree)[..., 1:]


def _fit_component_functions(U: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fit the additive first-order component functions ``f_i``.

    Least squares of the output onto ``sum_i f_i(X_i)`` under the correlated
    input measure is exactly the decomposition Li et al. use. The fitted
    ``f_i`` absorbs both what ``X_i`` explains alone and what it explains
    through its correlation with the rest. The independent part therefore has
    to be extracted afterwards by conditioning.

    Args:
        U: Marginal CDF values of the sample, shape ``(N, D)``.
        Y: Surrogate outputs, shape ``(N, S)``.

    Returns:
        Basis coefficients, shape ``(degree, D, S)``.
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
        u: Marginal CDF values for parameter ``i``, shape ``(...,)``.
        coefficients: Coefficients for that parameter, shape ``(degree, S)``.

    Returns:
        Component values, shape ``(..., S)``.
    """
    return _legendre_basis(u, _COMPONENT_DEGREE) @ coefficients


def _outer_chunk(n_outer: int, n_inner: int, n_columns: int) -> int:
    """Return how many outer points fit in the transient-memory budget.

    Each outer point expands into ``n_inner`` rows. Each row holds the
    ``D``-column latent block plus the ``S``-column prediction, in float64.
    Thin wrapper over the shared :func:`resolve_batch_size` with one outer
    point as the "row".

    Args:
        n_outer: Total outer points.
        n_inner: Inner draws per outer point.
        n_columns: Columns held per expanded row (``D + S``).

    Returns:
        Chunk size in ``[1, n_outer]``.
    """
    return resolve_batch_size(8 * n_inner * n_columns, n_outer, None)


def _total_correlated(
    *,
    plan: _ConditionalPlan,
    index: int,
    predict: Callable[[np.ndarray], np.ndarray],
    n_outer: int,
    n_inner: int,
    n_slices: int,
    seed: int,
) -> np.ndarray:
    """Estimate ``V(E(Y|X_i))`` for one parameter.

    Outer points are processed in memory-budgeted chunks. Only the
    ``(n_outer, S)`` inner means stay resident; the full
    ``(n_outer * n_inner, S)`` block is never materialised. All latent draws
    are made up front, so chunking changes evaluation batching only, not the
    sample.

    Note:
        The estimator drops the usual ``inner_var / n_inner`` correction and
        relies on the shared QMC inner block to cancel most of the inner error.
        That argument is empirical, and it holds only while ``n_inner`` is
        large against the outer conditioning signal. Do not lower ``n_inner``
        below the default to buy speed: the residual inner variance then leaks
        into the outer variance and inflates a small ``S_TC``. Raise
        ``n_outer`` instead.

    Args:
        plan: Precomputed Gaussian conditionals for the copula.
        index: Position of the parameter ``X_i`` being conditioned on.
        predict: Fitted surrogate. It maps ``(n, D)`` unit-cube samples to
            ``(n, S)`` outputs.
        n_outer: Outer (conditioning) sample size.
        n_inner: Inner (conditional) sample size per outer point.
        n_slices: Number of output slices ``S``.
        seed: Seed for this parameter's latent draws.

    Returns:
        Conditional-expectation variance, shape ``(S,)``. Not yet normalised.
    """
    others = plan.others[index]
    D_rest = others.shape[0]
    z_self = latent_normal_sample(n_outer, 1, seed=seed)[:, 0]
    # One inner block, reused across every outer point. Sharing it makes the
    # inner integration rule identical for all conditioning values, which
    # cancels much of its error out of the outer variance.
    inner = latent_normal_sample(n_inner, D_rest, seed=seed + 7919)

    inner_mean = np.empty((n_outer, n_slices), dtype=np.float64)
    chunk = _outer_chunk(n_outer, n_inner, D_rest + 1 + n_slices)
    for start in range(0, n_outer, chunk):
        z_chunk = z_self[start : start + chunk]
        # Z_-i | Z_i, broadcast to (chunk, n_inner, D_rest).
        z_rest = draw_rest_given_self(plan, index, z_chunk[:, None], inner)
        z_self_tiled = np.repeat(z_chunk, n_inner)
        Z = assemble_latent(index, others, z_self_tiled, z_rest.reshape(-1, D_rest))
        Y = np.asarray(predict(norm.cdf(Z)), dtype=np.float64)
        inner_mean[start : start + chunk] = Y.reshape(-1, n_inner, n_slices).mean(axis=1)

    # No inner-noise correction. The textbook term ``inner_var / n_inner``
    # assumes iid inner draws. Here the inner block is QMC and shared across
    # outer points, so the inner error is far smaller and mostly constant in
    # the outer variance. Chosen empirically against the linear-Gaussian
    # closed form: the correction clamps small true S_TC to near zero, while
    # the uncorrected estimator is unbiased at equal spread.
    return inner_mean.var(axis=0)


def _total_uncorrelated_and_conditional(
    *,
    plan: _ConditionalPlan,
    index: int,
    predict: Callable[[np.ndarray], np.ndarray],
    component: np.ndarray,
    n_outer: int,
    n_inner: int,
    n_slices: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ``E(V(Y|X_-i))`` and ``V(E(f_i|X_-i))`` for one parameter.

    Both quantities condition on ``X_-i`` and resample ``X_i``, so they share
    one nested sample: the surrogate gives the total uncorrelated index, and
    the fitted component function evaluated on the same draws gives the
    conditional expectation that ``S_U`` subtracts from ``V(f_i)``.

    Outer points are processed in memory-budgeted chunks, as in
    :func:`_total_correlated`. Only ``(n_outer, S)`` accumulators stay
    resident, and chunking does not change the latent sample.

    Args:
        plan: Precomputed Gaussian conditionals for the copula.
        index: Position of the parameter ``X_i`` being resampled.
        predict: Fitted surrogate. It maps ``(n, D)`` unit-cube samples to
            ``(n, S)`` outputs.
        component: Basis coefficients of ``f_i``, shape ``(degree, S)``.
        n_outer: Outer (conditioning) sample size.
        n_inner: Inner (conditional) sample size per outer point.
        n_slices: Number of output slices ``S``.
        seed: Seed for this parameter's latent draws.

    Returns:
        ``(conditional_variance_mean, component_conditional_variance)``, each
        shape ``(S,)``. Neither is normalised by ``V(Y)``.
    """
    others = plan.others[index]
    D_rest = others.shape[0]
    z_rest = latent_normal_sample(n_outer, D_rest, seed=seed) @ plan.chol_marginal[index].T
    inner = latent_normal_sample(n_inner, 1, seed=seed + 7919)[:, 0]

    inner_var = np.empty((n_outer, n_slices), dtype=np.float64)
    f_hat = np.empty((n_outer, n_slices), dtype=np.float64)
    chunk = _outer_chunk(n_outer, n_inner, D_rest + 1 + n_slices)
    for start in range(0, n_outer, chunk):
        stop = start + chunk
        # Z_i | Z_-i, broadcast to (chunk, n_inner) via the batched mean.
        z_self = draw_self_given_rest(plan, index, z_rest[start:stop, None, :], inner[None, :])
        z_rest_tiled = np.repeat(z_rest[start:stop], n_inner, axis=0)
        Z = assemble_latent(index, others, z_self.reshape(-1), z_rest_tiled)

        Y = np.asarray(predict(norm.cdf(Z)), dtype=np.float64)
        inner_var[start:stop] = Y.reshape(-1, n_inner, n_slices).var(axis=1, ddof=1)

        # E(f_i | X_-i), averaged over the same inner draws of X_i. Its
        # variance over the outer sample is the correlation-explained part of
        # f_i that Equation (25) removes. No inner-noise correction here
        # either: the same shared-QMC-block argument as in _total_correlated
        # applies, so both estimators use the same (empirically validated)
        # convention.
        f_hat[start:stop] = _evaluate_component(norm.cdf(z_self), component).mean(axis=1)

    return inner_var.mean(axis=0), f_hat.var(axis=0)
