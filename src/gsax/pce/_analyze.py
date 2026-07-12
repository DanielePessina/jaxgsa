"""PCE analysis and emulation entry points."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
from jax import Array

from gsax._normalization import (
    _prepare_Y,
    _squeeze_output_axes,
    _validate_xy_inputs,
    _warn_zero_variance_slices,
)
from gsax.pce._engine import (
    build_design_matrix,
    build_multi_index,
    loo_error,
    sobol_from_coefficients,
)
from gsax.pce._result import PCEResult
from gsax.problem import Problem


def _map_to_reference(X: Array, problem: Problem) -> tuple[Array, tuple[str, ...]]:
    """Map physical inputs to the orthogonal polynomial reference domain.

    Uniform and truncated-Gaussian inputs use Legendre (mapped to [-1, 1];
    truncated Gaussians go through their CDF first, so the mapped values are
    uniform). Untruncated Gaussian inputs use Hermite (standardized to N(0,1)).

    Args:
        X: (N, D) inputs in physical units.
        problem: Problem whose ``input_specs`` decide the per-dimension map.

    Returns:
        Tuple of the (N, D) reference-domain inputs and a length-D tuple of
        ``"uniform"`` / ``"gaussian"`` tags telling ``build_design_matrix``
        which 1-D polynomial family each dimension needs.
    """
    import numpy as np
    from scipy.stats import truncnorm

    D = problem.num_vars
    cols = []
    input_types: list[str] = []
    for d in range(D):
        dist, first, second, lo, hi = problem.input_specs[d]
        if dist == "uniform":
            cols.append(2.0 * (X[:, d] - first) / (second - first) - 1.0)
            input_types.append("uniform")
        elif lo is not None or hi is not None:
            mean, variance = first, second
            std = float(jnp.sqrt(variance))
            a_std = -np.inf if lo is None else (lo - mean) / std
            b_std = np.inf if hi is None else (hi - mean) / std
            u = jnp.asarray(
                truncnorm.cdf(np.asarray(X[:, d]), a=a_std, b=b_std, loc=mean, scale=std)
            )
            u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
            cols.append(2.0 * u - 1.0)
            input_types.append("uniform")
        else:
            mean, variance = first, second
            cols.append((X[:, d] - mean) / jnp.sqrt(variance))
            input_types.append("gaussian")

    return jnp.column_stack(cols), tuple(input_types)


def _auto_order(D: int, N: int, max_order: int, fit_ratio: float) -> int:
    """Reduce polynomial order so the term count fits within the sample budget."""
    from math import comb

    # Reduce order until C(D+p, p) <= fit_ratio * N to prevent overfitting
    # when the design matrix would have more columns than rows.
    cap = max(1, int(fit_ratio * N))
    order = max_order
    while order >= 1 and comb(D + order, order) > cap:
        order -= 1
    return max(order, 1)


def analyze_pce(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    order: int = 3,
    ridge: float = 1e-8,
    fit_ratio: float = 0.5,
) -> PCEResult:
    """Compute Sobol indices via polynomial chaos expansion (PCE).

    Fits an orthogonal polynomial surrogate to arbitrary (X, Y) pairs -- no
    structured sampling required -- and reads first-, total-, and second-order
    Sobol indices directly off the expansion coefficients (Sudret, 2008), with
    no extra model evaluations. Compared with Monte Carlo Sobol estimators,
    PCE needs far fewer samples when the model response is smooth, but it only
    captures effects the polynomial can represent: check
    ``PCEResult.loo_rmse`` before trusting the indices.

    Args:
        problem: Parameter names and distributions.
        X: (N, D) input samples.
        Y: Model outputs — (N,) scalar, (N, K) multi-output, or (N, T, K)
            time-series. All slices share one basis and are fitted in a
            single multi-right-hand-side solve; indices are computed
            independently per (t, k) slice.
        order: Maximum total polynomial degree. Higher orders capture
            sharper nonlinearity and higher-order interactions, but the
            term count C(D+order, order) grows fast and needs more samples
            to fit. Automatically reduced (with a warning) if the term
            count would exceed ``fit_ratio * N``.
        ridge: Tikhonov regularization for the least-squares fit. The tiny
            default only guards against a singular normal matrix; increase
            it if coefficients look unstable (noisy Y, near-duplicate rows).
        fit_ratio: Maximum ratio of terms to samples before ``order`` is
            reduced. Lower values demand more samples per term (a more
            conservative, less overfit-prone fit).

    Returns:
        PCEResult with S1, ST, S2 (shaped ``(..., D)`` / ``(..., D, D)`` with
        leading output/time dims mirroring ``Y``), the fitted coefficients and
        multi-index (usable with ``emulate_pce``), the effective ``order``,
        and the per-slice leave-one-out RMSE goodness-of-fit diagnostic.

    Raises:
        ValueError: If ``X`` fails validation against ``problem`` or ``Y``'s
            layout cannot be resolved against ``X``'s row count.
    """
    X = jnp.asarray(X)
    Y = _validate_xy_inputs(problem, X, jnp.asarray(Y))
    N, D = X.shape

    # Canonicalize to (N, T, K); the fit and extraction below are batched
    # over the flattened (T, K) slice grid in single vectorized operations.
    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)
    _, T, K = Y_3d.shape

    # A constant output slice makes every index 0/0 = NaN; warn once up front.
    _warn_zero_variance_slices(Y_3d, output_names=problem.output_names)

    # Cap polynomial order so n_terms <= fit_ratio * N (prevents overfitting).
    effective_order = _auto_order(D, N, order, fit_ratio)
    if effective_order < order:
        # The fit is silently coarser than requested; surface it so callers
        # do not mistake a truncated expansion for the order they asked for.
        warnings.warn(
            f"gsax: PCE order reduced from {order} to {effective_order} to keep the "
            f"term count within the sample budget (fit_ratio={fit_ratio}, N={N})",
            stacklevel=2,
        )
    mi = build_multi_index(D, effective_order)

    # Map inputs to reference domain and build the orthonormal design matrix.
    X_ref, input_types = _map_to_reference(X, problem)
    Phi = build_design_matrix(X_ref, mi, input_types, effective_order)

    # Compute Gram factorization once, reused for both fitting and LOO.
    gram = Phi.T @ Phi + ridge * jnp.eye(Phi.shape[1])
    gram_inv_PhiT = jnp.linalg.solve(gram, Phi.T)
    # The basis is identical for every output slice (the effective order
    # depends only on N and D), so fitting all T*K slices is ONE shared solve
    # with multiple right-hand sides -- a single vectorized matmul, no loops.
    Y_flat = Y_3d.reshape(N, T * K)  # column t*K + k is slice (t, k)
    coeffs_flat = gram_inv_PhiT @ Y_flat  # (n_terms, T*K)
    # Terms-last layout, matching HDMR's Sa convention (slices lead).
    coeffs = coeffs_flat.T.reshape(T, K, coeffs_flat.shape[0])

    # Sobol indices are extracted analytically from the coefficients
    # (Sudret 2008), batched over all slices -- no extra sampling, no loops.
    S1, ST, S2 = sobol_from_coefficients(coeffs, mi)  # (T,K,D), (T,K,D,D)

    # Per-slice LOO RMSE as a cheap goodness-of-fit diagnostic; the
    # hat-matrix leverage is shared by every slice.
    loo = loo_error(Phi, Y_flat, coeffs_flat, gram_inv_PhiT=gram_inv_PhiT)
    loo = loo.reshape(T, K)

    # Drop the singleton axes _prepare_Y inserted. S1/ST/coeffs end in
    # (T, K, per-slice) and use the shared helper; S2 (extra trailing D) and
    # loo (no trailing axis) are squeezed positionally.
    S1 = _squeeze_output_axes(S1, squeeze_time, squeeze_output)
    ST = _squeeze_output_axes(ST, squeeze_time, squeeze_output)
    coeffs = _squeeze_output_axes(coeffs, squeeze_time, squeeze_output)
    if squeeze_time and squeeze_output:
        S2, loo = S2[0, 0], loo[0, 0]
    elif squeeze_time:
        S2, loo = S2[0], loo[0]

    return PCEResult(
        S1=S1,
        ST=ST,
        S2=S2,
        problem=problem,
        coefficients=coeffs,
        multi_index=mi,
        order=effective_order,
        loo_rmse=loo,
    )


def emulate_pce(result: PCEResult, X_new: Array) -> Array:
    """Predict at new input points using the fitted PCE surrogate.

    Rebuilds the polynomial basis at ``X_new`` and applies the coefficients
    fitted by ``analyze_pce`` -- no model evaluations are needed. Accuracy
    degrades outside the input region the surrogate was fitted on.

    Args:
        result: PCEResult from ``analyze_pce``.
        X_new: (N_new, D) new input points, in the same physical units as
            the ``X`` passed to ``analyze_pce``.

    Returns:
        Predicted outputs mirroring the training ``Y`` layout: ``(N_new,)``,
        ``(N_new, K)``, or ``(N_new, T, K)``.
    """
    X_new = jnp.asarray(X_new)

    X_ref, input_types = _map_to_reference(X_new, result.problem)
    Phi = build_design_matrix(X_ref, result.multi_index, input_types, result.order)
    # Prediction contracts the term axis (last on coefficients) for every
    # output slice at once: Y = Phi @ c per slice, one einsum.
    return jnp.einsum("nt,...t->n...", Phi, result.coefficients)
