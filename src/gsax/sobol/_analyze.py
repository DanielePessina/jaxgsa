"""Main Sobol sensitivity analysis computation using JAX.

This module implements the Saltelli sampling-based Sobol variance decomposition.
Model outputs Y are split into base matrices A and B and their cross-matrices
AB (and BA for second-order), then first-order (S1), total-order (ST), and
optionally second-order (S2) Sobol indices are computed.

Array shape conventions used throughout:
    N  — number of base Sobol samples (base_n after cleaning)
    D  — number of input parameters
    T  — number of time steps (singleton-squeezed when absent)
    K  — number of output variables (singleton-squeezed when absent)
    R  — number of bootstrap resamples
    step — rows per Saltelli group: 2D+2 (second order) or D+2 (first only)
"""

from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from gsax._bootstrap import _bootstrap_ci_endpoints
from gsax._normalization import (
    _infer_output_layout,
    _prenormalize_outputs,
    _prepare_Y,
    _warn_zero_variance_slices,
)
from gsax.sampling import SamplingResult, _saltelli_step
from gsax.sobol._indices import (
    _fused_first_total,
    _fused_second_order,
)
from gsax.sobol._result import SAResult

# ---------------------------------------------------------------------------
# Cached JIT kernels
# ---------------------------------------------------------------------------


# lru_cache ensures each (calc_second_order,) variant is JIT-compiled exactly
# once across the process lifetime, avoiding repeated tracing overhead.


@lru_cache(maxsize=None)
def _get_scalar_kernel(calc_second_order: bool):
    """Cache JIT-compiled fused kernels for the scalar (T*K=1) path."""
    if calc_second_order:
        return jax.jit(_fused_second_order)
    return jax.jit(_fused_first_total)


@lru_cache(maxsize=None)
def _get_batched_kernel(calc_second_order: bool):
    """Cache JIT-compiled batched kernels for the multi-output path."""
    # vmap over axis 0 maps fused kernels across T*K output slices in parallel
    if calc_second_order:
        return jax.jit(jax.vmap(_fused_second_order, in_axes=(0, 0, 0, 0)))
    return jax.jit(jax.vmap(_fused_first_total, in_axes=(0, 0, 0)))


def _drop_nonfinite(Y: Array, step: int) -> tuple[Array, int]:
    """Drop entire Saltelli groups that contain any non-finite (NaN/Inf) value.

    Each Saltelli group is a contiguous block of ``step`` rows in Y that
    corresponds to one base sample and its D (or 2D) cross-matrix evaluations.
    If any single element in the group is non-finite the whole group is removed,
    because a partial group would corrupt the A/B/AB/BA split.

    Args:
        Y: Model output array, shape (n_total, ...) where n_total = N * step.
            Trailing dimensions are typically (T, K) or absent.
        step: Number of rows per Saltelli group (D+2 or 2D+2).

    Returns:
        (Y_clean, n_dropped) — cleaned output array with shape
        (N_good * step, ...) and the number of groups that were removed.
    """
    n_total = Y.shape[0]
    base_n = n_total // step
    trailing = Y.shape[1:]

    # Each Saltelli group [A_i, AB_{i,0}, …, AB_{i,D-1}, B_i] is an
    # indivisible sampling unit; a single NaN in any row invalidates the
    # whole group, so we mask at the group level, not per row.
    grouped = Y[: base_n * step].reshape(base_n, step, *trailing)
    # Flatten each group to a single row for an efficient all-finite check
    finite_mask = jnp.all(jnp.isfinite(grouped.reshape(base_n, -1)), axis=1)

    n_good = int(jnp.sum(finite_mask))
    n_dropped = base_n - n_good
    if n_dropped > 0:
        import numpy as np

        # Boolean indexing (variable-length output) is not supported by JAX JIT,
        # so we round-trip through NumPy for the filtering step.
        mask_np = np.asarray(finite_mask)
        grouped_clean = jnp.asarray(np.asarray(grouped)[mask_np])
        Y_clean = grouped_clean.reshape(n_good * step, *trailing)
        return Y_clean, n_dropped
    return Y, 0


def _count_nans(
    S1: Array,
    ST: Array,
    S2: Array | None,
) -> dict[str, int]:
    """Count NaN values per index array; callers decide whether to report.

    For S2, only the strict upper triangle is counted — the diagonal is NaN
    by construction (see ``_normalize_s2_matrix``) and would inflate the
    count.
    """
    counts: dict[str, int] = {
        "S1": int(jnp.sum(jnp.isnan(S1))),
        "ST": int(jnp.sum(jnp.isnan(ST))),
    }
    if S2 is not None:
        D = S2.shape[-1]
        upper = jnp.triu(jnp.ones((*S2.shape[:-2], D, D), dtype=bool), k=1)
        counts["S2"] = int(jnp.sum(jnp.isnan(S2) & upper))
    return counts


def _separate_output_values(
    Y: Array, D: int, calc_second_order: bool
) -> tuple[Array, Array, Array, Array | None]:
    """De-interleave flat Saltelli output rows into A, B, AB, BA matrices.

    Args:
        Y: Expanded outputs, shape (N * step, ...) with rows in Saltelli
            group order.
        D: Number of input parameters.
        calc_second_order: Whether the layout includes BA blocks.

    Returns:
        (A, B, AB, BA) with shapes (N, ...), (N, ...), (N, D, ...), and
        (N, D, ...) respectively; BA is None when second order is off.
    """
    step = 2 * D + 2 if calc_second_order else D + 2
    n_total = Y.shape[0]
    base_n = n_total // step
    trailing = Y.shape[1:]

    # Reshape-based extraction: one reshape converts the flat output vector
    # into (base_n, step, ...) groups, then simple slicing pulls A/B/AB/BA.
    # This is much faster than D separate stride-slices + a stack.
    #
    # Saltelli row layout within each group of `step` rows:
    #   [0]=A, [1..D]=AB_j (A with col j from B), [D+1..2D]=BA_j (B with
    #   col j from A), [2D+1]=B.  For first-order only, BA is omitted and
    #   step = D+2 with B at position [D+1].
    grouped = Y.reshape(base_n, step, *trailing)

    A = grouped[:, 0]  # (N, ...)
    B = grouped[:, -1]  # (N, ...)
    AB = grouped[:, 1 : D + 1]  # (N, D, ...)

    BA = None
    if calc_second_order:
        BA = grouped[:, D + 1 : 2 * D + 1]  # (N, D, ...)

    return A, B, AB, BA


def _normalize_s2_matrix(S2: Array) -> Array:
    """Symmetrise the S2 matrix and set diagonal entries to NaN."""
    # The fused kernel computes the full (D,D) matrix, but only the upper
    # triangle S2_{j<k} is meaningful; the lower triangle has a different
    # numerical path and slight floating-point drift.  We canonicalise by
    # keeping the upper triangle and mirroring it.
    D = S2.shape[-1]
    upper = jnp.triu(S2, k=1)
    mirrored = upper + jnp.swapaxes(upper, -1, -2)
    # Diagonal S2_{jj} (self-interaction) is undefined in Sobol ANOVA
    diag_mask = jnp.eye(D, dtype=bool)
    return jnp.where(diag_mask, jnp.nan, mirrored)


def _expand_unique_outputs(sampling_result: SamplingResult, Y: Array) -> Array:
    """Rebuild expanded Saltelli outputs from unique user-evaluated outputs.

    The sampler deduplicates the Saltelli design before returning samples to
    the user, so Y has only unique rows (shape ``(n_total, ...)``). This
    function gathers them back into the full interleaved layout the
    estimators expect, shape ``(expanded_n_total, ...)``.
    """
    if Y.shape[0] != sampling_result.n_total:
        raise ValueError(
            f"Y.shape[0] must match sampling_result.n_total ({sampling_result.n_total}), "
            f"got {Y.shape[0]}"
        )
    # expanded_to_unique[i] gives the unique-row index for expanded position i
    expanded_to_unique = jnp.asarray(sampling_result.expanded_to_unique)
    return jnp.take(Y, expanded_to_unique, axis=0)


def _squeeze_results(
    S1: Array,
    ST: Array,
    S2: Array | None,
    squeeze_time: bool,
    squeeze_output: bool,
    S1_conf: Array | None = None,
    ST_conf: Array | None = None,
    S2_conf: Array | None = None,
) -> tuple[Array, Array, Array | None, Array | None, Array | None, Array | None]:
    """Remove singleton T and/or K dimensions that _prepare_Y inserted.

    _prepare_Y always promotes Y to 3-D (N,T,K) for uniform kernel code.
    After computation, we undo that promotion so results match the user's
    original output shape (e.g. (D,) for scalar, (K,D) for multi-output).
    """
    if squeeze_time and squeeze_output:
        S1 = S1[0, 0]
        ST = ST[0, 0]
        if S2 is not None:
            S2 = S2[0, 0]
        if S1_conf is not None:
            S1_conf = S1_conf[:, 0, 0]
        if ST_conf is not None:
            ST_conf = ST_conf[:, 0, 0]
        if S2_conf is not None:
            S2_conf = S2_conf[:, 0, 0]
    elif squeeze_time:
        S1 = S1[0]
        ST = ST[0]
        if S2 is not None:
            S2 = S2[0]
        if S1_conf is not None:
            S1_conf = S1_conf[:, 0]
        if ST_conf is not None:
            ST_conf = ST_conf[:, 0]
        if S2_conf is not None:
            S2_conf = S2_conf[:, 0]
    return S1, ST, S2, S1_conf, ST_conf, S2_conf


def _analyze_no_bootstrap(
    sampling_result: SamplingResult, Y: Array, *, chunk_size: int
) -> SAResult:
    """Compute Sobol indices with optimized kernel selection.

    For scalar output (T*K=1), uses a direct fused kernel that computes
    variance once. For multi-output, vmaps the fused kernel over T*K batches.
    """
    D = sampling_result.n_params
    calc_second_order = sampling_result.calc_second_order

    # Detect scalar output before _prepare_Y adds singleton T and K dims.
    # The scalar path skips vmap entirely, saving tracing and dispatch cost.
    is_scalar = Y.ndim == 1

    # Promote to uniform 3-D shape (N, T, K) so downstream code is shape-agnostic
    Y, squeeze_time, squeeze_output = _prepare_Y(Y)
    _, T, K = Y.shape

    A, B, AB, BA = _separate_output_values(Y, D, calc_second_order)
    base_n = A.shape[0]

    total = T * K

    if is_scalar:
        # Scalar path (T*K=1): call the fused kernel directly on 1-D arrays.
        # This avoids vmap dispatch overhead and produces a simpler XLA graph.
        # Squeeze trailing (1,1) dims added by _prepare_Y.
        a = A[:, 0, 0]
        b = B[:, 0, 0]
        ab = AB[:, :, 0, 0]

        if calc_second_order:
            assert BA is not None
            ba = BA[:, :, 0, 0]
            kernel = _get_scalar_kernel(True)
            S1_out, ST_out, S2_raw = kernel(a, ab, ba, b)
            S2_out = _normalize_s2_matrix(S2_raw)
        else:
            kernel = _get_scalar_kernel(False)
            S1_out, ST_out = kernel(a, ab, b)
            S2_out = None

        nan_counts = _count_nans(S1_out, ST_out, S2_out)
        return SAResult(
            S1=S1_out,
            ST=ST_out,
            S2=S2_out,
            problem=sampling_result.problem,
            nan_counts=nan_counts,
        )

    # Batched path (T*K > 1): flatten (T, K) into a single batch dimension
    # so vmap processes all output slices in one vectorised call.
    # (N,T,K) -> transpose -> (T,K,N) -> reshape -> (T*K, N)
    # AB is (N,D,T,K) so its transpose puts (T,K) first then (N,D).
    A_flat = A.transpose(1, 2, 0).reshape(T * K, base_n)
    B_flat = B.transpose(1, 2, 0).reshape(T * K, base_n)
    AB_flat = AB.transpose(2, 3, 0, 1).reshape(T * K, base_n, D)

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    # Chunk the T*K batches to cap peak memory when many outputs exist
    cs = min(chunk_size, total)

    if calc_second_order:
        assert BA is not None
        BA_flat = BA.transpose(2, 3, 0, 1).reshape(T * K, base_n, D)

        batched = _get_batched_kernel(True)
        s1_parts, st_parts, s2_parts = [], [], []
        for start in range(0, total, cs):
            end = min(start + cs, total)
            s1, st, s2 = batched(
                A_flat[start:end],
                AB_flat[start:end],
                BA_flat[start:end],
                B_flat[start:end],
            )
            s1_parts.append(s1)
            st_parts.append(st)
            s2_parts.append(s2)

        S1_out = jnp.concatenate(s1_parts).reshape(T, K, D)
        ST_out = jnp.concatenate(st_parts).reshape(T, K, D)
        S2_out = _normalize_s2_matrix(jnp.concatenate(s2_parts).reshape(T, K, D, D))
    else:
        batched = _get_batched_kernel(False)
        s1_parts, st_parts = [], []
        for start in range(0, total, cs):
            end = min(start + cs, total)
            s1, st = batched(
                A_flat[start:end],
                AB_flat[start:end],
                B_flat[start:end],
            )
            s1_parts.append(s1)
            st_parts.append(st)

        S1_out = jnp.concatenate(s1_parts).reshape(T, K, D)
        ST_out = jnp.concatenate(st_parts).reshape(T, K, D)
        S2_out = None

    S1_out, ST_out, S2_out, _, _, _ = _squeeze_results(
        S1_out, ST_out, S2_out, squeeze_time, squeeze_output
    )
    nan_counts = _count_nans(S1_out, ST_out, S2_out)
    return SAResult(
        S1=S1_out,
        ST=ST_out,
        S2=S2_out,
        problem=sampling_result.problem,
        nan_counts=nan_counts,
    )


def _analyze_bootstrap(
    sampling_result: SamplingResult,
    Y: Array,
    *,
    num_resamples: int,
    conf_level: float,
    ci_method: Literal["quantile", "gaussian"],
    key: Array,
    chunk_size: int,
) -> SAResult:
    """Bootstrap path: loop over (T, K) combos, vmap over R resamples.

    Unlike the no-bootstrap path which vmaps over output slices, here we loop
    over (T, K) in Python and vmap over R resamples within each slice.  This
    trades some Python-loop overhead for bounded memory: each vmap call
    materialises R copies of a single (N,) or (N,D) slice, not R * T * K.
    """
    from gsax.sobol._bootstrap import _bootstrap_first_total, _bootstrap_second_order

    Y, squeeze_time, squeeze_output = _prepare_Y(Y)
    D = sampling_result.n_params
    calc_second_order = sampling_result.calc_second_order

    _, T, K = Y.shape
    A, B, AB, BA = _separate_output_values(Y, D, calc_second_order)
    base_n = A.shape[0]

    # Pre-generate all R bootstrap index sets (sampling with replacement).
    # Shared across (T, K) slices so every output sees the same resamples.
    indices = jax.random.randint(key, shape=(num_resamples, base_n), minval=0, maxval=base_n)

    # Reuse the scalar (non-vmapped) kernels for point estimates per slice
    jit_ft = _get_scalar_kernel(False)
    jit_so = _get_scalar_kernel(True)

    S1_list, ST_list = [], []
    S1_lo_list, S1_hi_list = [], []
    ST_lo_list, ST_hi_list = [], []
    S2_list, S2_lo_list, S2_hi_list = [], [], []

    for t in range(T):
        for k in range(K):
            # Extract 1-D slice for this (time, output) combo
            a = A[:, t, k]
            b = B[:, t, k]
            ab = AB[:, :, t, k]

            if calc_second_order:
                assert BA is not None
                ba = BA[:, :, t, k]

                s1, st, s2 = jit_so(a, ab, ba, b)
                S2_list.append(s2)

                s1_boot, st_boot, s2_boot = _bootstrap_second_order(
                    indices, a, ab, ba, b, chunk_size
                )
                s2_lo, s2_hi = _bootstrap_ci_endpoints(
                    s2,
                    s2_boot,
                    conf_level=conf_level,
                    ci_method=ci_method,
                )
                S2_lo_list.append(s2_lo)
                S2_hi_list.append(s2_hi)
            else:
                s1, st = jit_ft(a, ab, b)
                s1_boot, st_boot = _bootstrap_first_total(indices, a, ab, b, chunk_size)

            S1_list.append(s1)
            ST_list.append(st)

            s1_lo, s1_hi = _bootstrap_ci_endpoints(
                s1,
                s1_boot,
                conf_level=conf_level,
                ci_method=ci_method,
            )
            st_lo, st_hi = _bootstrap_ci_endpoints(
                st,
                st_boot,
                conf_level=conf_level,
                ci_method=ci_method,
            )
            S1_lo_list.append(s1_lo)
            S1_hi_list.append(s1_hi)
            ST_lo_list.append(st_lo)
            ST_hi_list.append(st_hi)

    # Reassemble per-slice results into (T, K, D) arrays
    S1_out = jnp.stack(S1_list).reshape(T, K, D)
    ST_out = jnp.stack(ST_list).reshape(T, K, D)

    # Confidence intervals: stack [lower, upper] into leading dim of size 2
    S1_conf = jnp.stack(
        [
            jnp.stack(S1_lo_list).reshape(T, K, D),
            jnp.stack(S1_hi_list).reshape(T, K, D),
        ]
    )
    ST_conf = jnp.stack(
        [
            jnp.stack(ST_lo_list).reshape(T, K, D),
            jnp.stack(ST_hi_list).reshape(T, K, D),
        ]
    )

    if calc_second_order:
        S2_out = _normalize_s2_matrix(jnp.stack(S2_list).reshape(T, K, D, D))
        S2_conf = _normalize_s2_matrix(
            jnp.stack(
                [
                    jnp.stack(S2_lo_list).reshape(T, K, D, D),
                    jnp.stack(S2_hi_list).reshape(T, K, D, D),
                ]
            )
        )
    else:
        S2_out = None
        S2_conf = None

    S1_out, ST_out, S2_out, S1_conf, ST_conf, S2_conf = _squeeze_results(
        S1_out,
        ST_out,
        S2_out,
        squeeze_time,
        squeeze_output,
        S1_conf,
        ST_conf,
        S2_conf,
    )
    nan_counts = _count_nans(S1_out, ST_out, S2_out)
    return SAResult(
        S1=S1_out,
        ST=ST_out,
        S2=S2_out,
        problem=sampling_result.problem,
        S1_conf=S1_conf,
        ST_conf=ST_conf,
        S2_conf=S2_conf,
        nan_counts=nan_counts,
    )


def analyze(
    sampling_result: SamplingResult,
    Y: Array,
    *,
    prenormalize: bool = False,
    num_resamples: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    chunk_size: int = 2048,
) -> SAResult:
    """Compute Sobol sensitivity indices from model outputs using JAX.

    This is the main entry point of the package. Sobol indices apportion the
    variance of a model output among its input parameters: S1 (first-order)
    is the fraction of output variance explained by each parameter alone, ST
    (total-order) additionally includes all of its interactions with other
    parameters, and S2 (second-order) isolates pairwise interactions.

    The function accepts model outputs Y evaluated at the unique rows
    returned by ``gsax.sample()``, reconstructs the expanded Saltelli
    ordering internally, drops sample groups containing non-finite values
    (with a warning), and dispatches to either the fast no-bootstrap path or
    the bootstrap confidence-interval path depending on ``num_resamples``.

    Args:
        sampling_result: Result from ``gsax.sample()`` containing the unique
            sample matrix plus expansion metadata.
        Y: Model outputs evaluated at each unique row of
            ``sampling_result.samples``, in the same row order. Accepted
            shapes:
                (n_total,)       — scalar output, single time step
                (n_total, K)     — K outputs, single time step
                (n_total, T, K)  — K outputs over T time steps
            where ``n_total`` is the unique row count. Indices are computed
            independently for every (t, k) output slice.
        prenormalize: When ``True``, apply SALib-style global output
            standardization over the cleaned expanded sample axis before
            computing Sobol indices. Each output slice is centered to mean 0
            and scaled to unit standard deviation once, not per bootstrap
            resample. Defaults to ``False``.
        num_resamples: R, the number of bootstrap resamples used to estimate
            confidence intervals. Set to 0 (default) to skip the bootstrap
            entirely; a few hundred resamples is typically enough for stable
            intervals.
        conf_level: Two-sided confidence level for bootstrap CIs
            (default 0.95).
        ci_method: Bootstrap CI endpoint method. ``"quantile"`` returns
            percentile lower/upper endpoints from the bootstrap draws.
            ``"gaussian"`` returns symmetric gaussian lower/upper endpoints
            from the bootstrap standard deviation around the point estimate.
            Both methods still return lower/upper bounds, not half-widths.
        key: JAX PRNG key for bootstrap randomness. Required when
            ``num_resamples > 0``.
        chunk_size: Memory/speed trade-off for batched computation. In the
            no-bootstrap path this is the number of (T, K) output slices per
            vmap batch; in the bootstrap path it is the number of resamples
            per batch. Lower it if you hit device out-of-memory errors.
            Defaults to 2048.

    Returns:
        SAResult containing:
            S1 — first-order indices, shape (D,) / (K, D) / (T, K, D)
                 for Y of shape (n,) / (n, K) / (n, T, K) respectively
            ST — total-order indices, same shape as S1
            S2 — second-order indices with shape (..., D, D), or None when
                 the design was drawn with ``calc_second_order=False``
            S1_conf, ST_conf, S2_conf — (2, ...) [lower, upper] CI bounds,
                 or None when ``num_resamples == 0``
    """
    # Resolve the user-supplied layout (sample axis first, labeled output axis
    # last) against the unique design rows, BEFORE any expansion or resampling
    # so every downstream stage sees canonical axes.
    Y = _infer_output_layout(
        Y, sampling_result.problem, int(sampling_result.samples.shape[0])
    )
    # Map user-evaluated unique outputs back to the full Saltelli interleaving
    Y = _expand_unique_outputs(sampling_result, Y)

    D = sampling_result.n_params
    # step = rows per Saltelli group: D+2 (first-order only) or 2D+2 (with S2)
    step = _saltelli_step(D, sampling_result.calc_second_order)

    # Remove entire Saltelli groups containing NaN/Inf before analysis
    Y, n_dropped = _drop_nonfinite(Y, step)
    if n_dropped > 0:
        import warnings

        remaining = Y.shape[0] // step
        total_groups = remaining + n_dropped
        pct = 100.0 * n_dropped / total_groups
        warnings.warn(
            f"gsax: dropped {n_dropped} of {total_groups} sample groups "
            f"({pct:.1f}%) containing non-finite values; "
            f"{remaining} groups remain",
            stacklevel=2,
        )
        if Y.shape[0] == 0:
            raise ValueError("All samples contain non-finite values")
        _MIN_GROUPS = 10
        if remaining < _MIN_GROUPS:
            warnings.warn(
                f"gsax: only {remaining} sample groups remain after dropping "
                f"non-finite values — results may be statistically unreliable "
                f"(recommend >= {_MIN_GROUPS})",
                stacklevel=2,
            )

    if prenormalize:
        Y, _, _, _ = _prenormalize_outputs(Y)

    _warn_zero_variance_slices(Y, output_names=sampling_result.problem.output_names)

    if ci_method not in {"quantile", "gaussian"}:
        raise ValueError("ci_method must be one of {'quantile', 'gaussian'}")

    if num_resamples > 0:
        if key is None:
            raise ValueError("key is required when num_resamples > 0")
        return _analyze_bootstrap(
            sampling_result,
            Y,
            num_resamples=num_resamples,
            conf_level=conf_level,
            ci_method=ci_method,
            key=key,
            chunk_size=chunk_size,
        )

    return _analyze_no_bootstrap(sampling_result, Y, chunk_size=chunk_size)
