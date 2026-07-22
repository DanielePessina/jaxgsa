"""PCE analysis and emulation entry points."""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.batching import get_memory_budget, resolve_batch_size
from jaxgsa._core.surrogate import _PredictPlan
from jaxgsa._core.validation import (
    _prepare_Y,
    _squeeze_output_axes,
    _validate_xy_inputs,
    _warn_zero_variance_slices,
)
from jaxgsa.pce._engine import (
    build_design_matrix,
    build_multi_index,
    loo_error,
    sobol_from_coefficients,
)
from jaxgsa.pce._result import PCEResult
from jaxgsa.problem import Problem


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


class _PCEFit(NamedTuple):
    """The shared PCE fit: coefficients and everything needed to reuse them.

    PCE analysis derives Sobol indices and the LOO diagnostic from this state;
    :meth:`PCEResult.shapley` reuses its coefficients and multi-index. The
    design matrix and Gram factorization are deliberately NOT carried: the
    streamed path never materializes them at full N, so both fit paths reduce
    to the same coefficient + LOO summary.
    """

    coefficients: Array  # (T, K, n_terms), terms-last
    coeffs_flat: Array  # (n_terms, T*K)
    multi_index: np.ndarray  # (n_terms, D)
    order: int  # effective order after _auto_order
    loo_flat: Array  # (T*K,) per-slice LOO RMSE
    squeeze_time: bool
    squeeze_output: bool


def _fit_pce_streamed(
    X_ref: Array,
    Y_flat: Array,
    multi_index: np.ndarray,
    input_types: tuple[str, ...],
    order: int,
    ridge: float,
    batch_size: int | None,
) -> tuple[Array, Array]:
    """Fit the PCE by streaming row batches: exact normal equations + LOO.

    Two passes over row batches of ``X_ref``/``Y_flat``, never holding more
    than one ``(batch, n_terms)`` design block:

    1. Accumulate ``G = Phi^T Phi`` (n_terms, n_terms) and ``B = Phi^T Y``
       (n_terms, T*K), then solve ``(G + ridge*I) c = B`` once. This is
       mathematically identical to the single-pass normal equations -- only
       the float32 summation order differs.
    2. With ``(G + ridge*I)^{-1}`` known, rebuild each design block to get
       the hat-matrix diagonal ``h_i = phi_i G^{-1} phi_i^T`` and per-row
       residuals, accumulating the exact LOO sum of squares as defined by
       :func:`jaxgsa.pce._engine.loo_error` (same leverage clip, same
       ``mean`` over rows).

    Args:
        X_ref: (N, D) inputs already mapped to the reference domain.
        Y_flat: (N, T*K) flattened output slices.
        multi_index: (n_terms, D) multi-index array.
        input_types: per-dimension ``"uniform"`` / ``"gaussian"`` tags.
        order: effective (post-``_auto_order``) polynomial order.
        ridge: Tikhonov parameter added to the Gram matrix.
        batch_size: rows per batch, or ``None`` to derive one from the
            active memory budget.

    Returns:
        Tuple of ``coeffs_flat`` (n_terms, T*K) and the (T*K,) per-slice
        LOO RMSE.
    """
    N = X_ref.shape[0]
    n_terms = multi_index.shape[0]
    M = Y_flat.shape[1]
    dtype = jnp.result_type(X_ref.dtype, Y_flat.dtype)

    # Per-row transient cost mirrors _single_pass_fit_bytes: 3*n_terms floats
    # for the design-block running product plus 2*M for the residual arrays.
    bytes_per_row = jnp.dtype(dtype).itemsize * (3 * n_terms + 2 * M)
    b = resolve_batch_size(bytes_per_row, N, batch_size)

    # Pass 1: accumulate the Gram matrix and cross-moments batch by batch.
    G = jnp.zeros((n_terms, n_terms), dtype=dtype)
    B = jnp.zeros((n_terms, M), dtype=dtype)
    for i in range(0, N, b):
        Phi_b = build_design_matrix(X_ref[i : i + b], multi_index, input_types, order)
        G = G + Phi_b.T @ Phi_b
        B = B + Phi_b.T @ Y_flat[i : i + b]
    gram = G + ridge * jnp.eye(n_terms, dtype=dtype)
    coeffs_flat = jnp.linalg.solve(gram, B)  # (n_terms, M)

    # Pass 2: exact LOO from the hat-matrix diagonal, streamed. gram is
    # symmetric, so h_i = phi_i gram^{-1} phi_i^T = sum(Phi_b * (Phi_b @
    # gram_inv), axis=1) reproduces loo_error's diag(Phi @ gram_inv_PhiT).
    gram_inv = jnp.linalg.inv(gram)
    sse = jnp.zeros((M,), dtype=dtype)
    for i in range(0, N, b):
        Phi_b = build_design_matrix(X_ref[i : i + b], multi_index, input_types, order)
        residuals = Y_flat[i : i + b] - Phi_b @ coeffs_flat  # (b, M)
        leverage = jnp.sum(Phi_b * (Phi_b @ gram_inv), axis=1)  # (b,)
        # Same interpolation guard as loo_error: a leverage of exactly 1
        # would divide by zero.
        leverage = jnp.clip(leverage, 0.0, 1.0 - 1e-10)
        loo_residuals = residuals / (1.0 - leverage)[:, None]
        sse = sse + jnp.sum(loo_residuals**2, axis=0)
    loo_flat = jnp.sqrt(sse / N)  # == sqrt(mean(loo_residuals^2, axis=0))

    return coeffs_flat, loo_flat


def _single_pass_fit_bytes(N: int, n_terms: int, M: int, itemsize: int) -> int:
    """Estimate the resident bytes of the single-pass PCE fit path.

    Derived from the arrays the single-pass code actually materializes:

    - ``build_design_matrix`` keeps ~3 (N, n_terms) arrays live at its peak
      (running-product accumulator, one gathered factor, multiply output);
      the same count recurs at ``loo_error``'s leverage step, where ``Phi``,
      ``gram_inv_PhiT``, and its transpose coexist -> ``3 * N * n_terms``.
    - ``loo_error`` materializes the (N, M) prediction ``Phi @ coefficients``
      / residuals and the (N, M) ``loo_residuals`` -> ``2 * N * M``.

    ``Y`` itself and the (n_terms, n_terms) Gram matrix are excluded: the
    caller holds Y either way, and the Gram factors are negligible next to
    the N-sized arrays.

    Args:
        N: number of sample rows.
        n_terms: number of expansion terms.
        M: number of flattened output slices (T*K).
        itemsize: bytes per element of the working dtype.

    Returns:
        Estimated peak resident bytes of the single-pass fit + LOO.
    """
    return itemsize * N * (3 * n_terms + 2 * M)


def _fit_pce_core(
    problem: Problem,
    X: Array,
    Y_canonical: Array,
    *,
    order: int,
    ridge: float,
    fit_ratio: float,
    batch_size: int | None = None,
) -> _PCEFit:
    """Fit the shared PCE expansion for every output slice at once.

    Operates on an already-canonical Y (no validation, no zero-variance
    warning, no index extraction). The basis is identical across slices, so
    fitting all ``T*K`` right-hand sides is a single Gram solve.

    When the single-pass residents (design matrix, Gram factorization, LOO
    residual arrays) would exceed the active memory budget -- or when
    ``batch_size`` is an explicit int -- the fit streams over row batches
    instead (see :func:`_fit_pce_streamed`); the streamed fit solves the
    same normal equations and computes the same exact LOO, differing only
    in float32 summation order.
    """
    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y_canonical)
    N, D = X.shape
    _, T, K = Y_3d.shape

    # Cap polynomial order so n_terms <= fit_ratio * N (prevents overfitting).
    effective_order = _auto_order(D, N, order, fit_ratio)
    if effective_order < order:
        # The fit is silently coarser than requested; surface it so callers
        # do not mistake a truncated expansion for the order they asked for.
        warnings.warn(
            f"jaxgsa: PCE order reduced from {order} to {effective_order} to keep the "
            f"term count within the sample budget (fit_ratio={fit_ratio}, N={N})",
            stacklevel=3,
        )
    mi = build_multi_index(D, effective_order)
    n_terms = mi.shape[0]

    # Map inputs to reference domain; (N, D) is small next to (N, n_terms).
    X_ref, input_types = _map_to_reference(X, problem)
    Y_flat = Y_3d.reshape(N, T * K)  # column t*K + k is slice (t, k)

    single_pass_bytes = _single_pass_fit_bytes(N, n_terms, T * K, X_ref.dtype.itemsize)
    if batch_size is None and single_pass_bytes <= get_memory_budget():
        # Single-pass path: build the full design matrix and compute the Gram
        # factorization once, reused for both fitting and LOO.
        Phi = build_design_matrix(X_ref, mi, input_types, effective_order)
        gram = Phi.T @ Phi + ridge * jnp.eye(Phi.shape[1])
        gram_inv_PhiT = jnp.linalg.solve(gram, Phi.T)
        # The basis is identical for every output slice (the effective order
        # depends only on N and D), so fitting all T*K slices is ONE shared
        # solve with multiple right-hand sides -- a single vectorized matmul.
        coeffs_flat = gram_inv_PhiT @ Y_flat  # (n_terms, T*K)
        # Per-slice LOO RMSE from the shared hat-matrix leverage.
        loo_flat = loo_error(Phi, Y_flat, coeffs_flat, gram_inv_PhiT=gram_inv_PhiT)
    else:
        # Streamed path: same normal equations and exact LOO, accumulated
        # over row batches so peak memory stays within the budget.
        coeffs_flat, loo_flat = _fit_pce_streamed(
            X_ref, Y_flat, mi, input_types, effective_order, ridge, batch_size
        )
    # Terms-last layout, matching HDMR's Sa convention (slices lead).
    coeffs = coeffs_flat.T.reshape(T, K, n_terms)

    return _PCEFit(
        coefficients=coeffs,
        coeffs_flat=coeffs_flat,
        multi_index=mi,
        order=effective_order,
        loo_flat=loo_flat,
        squeeze_time=squeeze_time,
        squeeze_output=squeeze_output,
    )


def analyze_pce(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    order: int = 3,
    ridge: float = 1e-8,
    fit_ratio: float = 0.5,
    batch_size: int | None = None,
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
        batch_size: Rows of ``X``/``Y`` processed per batch during the fit
            (the package-wide ``batch_size`` convention). ``None`` (default)
            keeps the single-shot fit unless its estimated resident memory
            (design matrix, Gram factorization, LOO residuals) exceeds the
            active memory budget (~512 MiB, see
            ``jaxgsa.config.set_memory_budget``), in which case the fit
            streams over auto-sized row batches. An explicit int always
            forces the streamed fit with that many rows per batch. Both fit
            paths solve the same normal equations and compute the same
            exact leave-one-out error; results differ only at the level of
            float32 summation order.

    Returns:
        PCEResult with S1, ST, S2 (shaped ``(..., D)`` / ``(..., D, D)`` with
        leading output/time dims mirroring ``Y``), the fitted coefficients and
        multi-index (reused by ``result.predict``), the effective ``order``,
        and the per-slice leave-one-out RMSE goodness-of-fit diagnostic.

    Raises:
        ValueError: If ``X`` fails validation against ``problem``, ``Y``'s
            layout cannot be resolved against ``X``'s row count, or
            ``batch_size`` is given and not a positive integer.
    """
    X = jnp.asarray(X)
    Y = _validate_xy_inputs(problem, X, jnp.asarray(Y))

    # Per-slice output variance, computed once and shared by the zero-variance
    # warning and the explained-variance diagnostic below.
    Y_3d = _prepare_Y(Y)[0]
    total_var = jnp.var(Y_3d, axis=0)  # (T, K)

    # A constant output slice makes every index 0/0 = NaN; warn once up front.
    _warn_zero_variance_slices(Y_3d, output_names=problem.output_names, var_per_slice=total_var)

    fit = _fit_pce_core(
        problem, X, Y, order=order, ridge=ridge, fit_ratio=fit_ratio, batch_size=batch_size
    )
    squeeze_time, squeeze_output = fit.squeeze_time, fit.squeeze_output
    T, K = fit.coefficients.shape[:2]

    # Sobol indices are extracted analytically from the coefficients
    # (Sudret 2008), batched over all slices -- no extra sampling, no loops.
    S1, ST, S2 = sobol_from_coefficients(fit.coefficients, fit.multi_index)  # (T,K,D), (T,K,D,D)

    # Per-slice LOO RMSE as a cheap goodness-of-fit diagnostic, computed by
    # the fit path (single-pass hat-matrix diagonal, or the streamed
    # equivalent); the leverage is shared by every slice.
    loo = fit.loo_flat.reshape(T, K)
    partial_var = jnp.sum(fit.coefficients[..., 1:] ** 2, axis=-1)
    explained_variance = jnp.where(total_var == 0, jnp.nan, partial_var / total_var)

    # Drop the singleton axes _prepare_Y inserted. S1/ST/coeffs end in
    # (T, K, per-slice); S2 carries an extra trailing D and loo has no trailing
    # per-slice axis, so they pass n_trailing=2 and 0 respectively.
    S1 = _squeeze_output_axes(S1, squeeze_time, squeeze_output)
    ST = _squeeze_output_axes(ST, squeeze_time, squeeze_output)
    coeffs = _squeeze_output_axes(fit.coefficients, squeeze_time, squeeze_output)
    S2 = _squeeze_output_axes(S2, squeeze_time, squeeze_output, n_trailing=2)
    loo = _squeeze_output_axes(loo, squeeze_time, squeeze_output, n_trailing=0)
    explained_variance = _squeeze_output_axes(
        explained_variance,
        squeeze_time,
        squeeze_output,
        n_trailing=0,
    )

    return PCEResult(
        S1=S1,
        ST=ST,
        S2=S2,
        problem=problem,
        coefficients=coeffs,
        multi_index=fit.multi_index,
        order=fit.order,
        loo_rmse=loo,
        explained_variance=explained_variance,
    )


def _pce_predict_plan(result: PCEResult, X_new: Array) -> _PredictPlan:
    """Build the prediction plan behind :meth:`PCEResult.predict`.

    Maps the (already validated) inputs to the polynomial reference domain
    and packages the per-row transient cost together with a kernel that
    rebuilds the basis and contracts it with the fitted coefficients; the
    shared template in :class:`jaxgsa._core.surrogate.SurrogateResult` runs
    the kernel in row batches sized against a transient-memory budget.
    """
    X_ref, input_types = _map_to_reference(X_new, result.problem)
    n_terms = result.multi_index.shape[0]
    # Transient footprint per row: build_design_matrix accumulates a running
    # product, so at most the (batch, n_terms) accumulator, one gathered
    # (batch, n_terms) factor, and the multiply output are live at once
    # (~3 * n_terms floats per row), plus the einsum output slices.
    slices = math.prod(result.coefficients.shape[:-1])
    bytes_per_row = X_ref.dtype.itemsize * (3 * n_terms + slices)

    def _predict(X_chunk: Array) -> Array:
        Phi = build_design_matrix(X_chunk, result.multi_index, input_types, result.order)
        # Prediction contracts the term axis (last on coefficients) for every
        # output slice at once: Y = Phi @ c per slice, one einsum.
        return jnp.einsum("nt,...t->n...", Phi, result.coefficients)

    return _PredictPlan(X=X_ref, bytes_per_row=bytes_per_row, kernel=_predict)
