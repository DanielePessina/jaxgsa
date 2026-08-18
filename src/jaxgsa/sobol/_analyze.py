"""Main Sobol sensitivity analysis computation using JAX.

This module implements the Saltelli sampling-based Sobol variance
decomposition. It splits the model outputs Y into the base matrices A and B
and their cross-matrices AB (plus BA for second order). It then computes
first-order (S1), total-order (ST), and optionally second-order (S2) Sobol
indices.

The estimator maths lives in exactly one place, ``_indices_from_expanded``.
``analyze`` runs the host-side diagnostics and wraps that core in a result
object; ``indices`` calls the same core with nothing around it, so it stays
traceable by ``jit``, ``vmap`` and ``jacrev``.

Array shape conventions used throughout:
    N: number of base Sobol samples (base_n after cleaning)
    D: number of input parameters
    T: number of time steps (singleton-squeezed when absent)
    K: number of output variables (singleton-squeezed when absent)
    R: number of bootstrap resamples
    step: rows per Saltelli group, 2D+2 (second order) or D+2 (first only)
"""

from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.bootstrap import _bootstrap_ci_endpoints
from jaxgsa._core.entry import at_least, in_open_interval, one_of, prepare
from jaxgsa._core.invalid import InvalidReport, OnInvalid
from jaxgsa._core.result import CIInfo
from jaxgsa._core.validation import (
    _prenormalize_outputs,
    _prepare_Y,
    _squeeze_output_axes,
)
from jaxgsa.sobol._indices import (
    _fused_first_total,
    _fused_second_order,
)
from jaxgsa.sobol._result import SobolResult
from jaxgsa.sobol._sampling import SobolSamples, _saltelli_step

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


def _separate_output_values(
    Y: Array, D: int, calc_second_order: bool
) -> tuple[Array, Array, Array, Array | None]:
    """De-interleave flat Saltelli output rows into A, B, AB, BA matrices.

    Args:
        Y: Expanded outputs, shape ``(N * step, ...)``, with rows in Saltelli
            group order.
        D: Number of input parameters.
        calc_second_order: Whether the layout includes BA blocks.

    Returns:
        Tuple ``(A, B, AB, BA)`` with shapes ``(N, ...)``, ``(N, ...)``,
        ``(N, D, ...)`` and ``(N, D, ...)``. BA is None when second order is
        off.
    """
    step = 2 * D + 2 if calc_second_order else D + 2
    n_rows = Y.shape[0]
    base_n = n_rows // step
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


def _indices_from_expanded(
    Y: Array, D: int, calc_second_order: bool, slice_chunk_size: int
) -> tuple[Array, Array, Array | None]:
    """Compute Sobol indices from expanded-layout outputs, picking the faster kernel.

    This is the single implementation of the estimator maths. Both the public
    :func:`indices` and the ``analyze`` no-bootstrap path go through it, so
    the two can never drift apart.

    It is deliberately free of policy: no diagnostics, no host-side reads of
    array values, no result object. Every branch here is on a shape or a
    Python flag, all of which stay concrete under tracing, so the function is
    ``jit``-, ``vmap``- and ``jacrev``-able.

    For a scalar output (T*K=1) it calls a direct fused kernel that computes
    the variance once. For a multi-output analysis it vmaps the fused kernel
    over the T*K batches.

    Args:
        Y: Model outputs in the expanded Saltelli layout, shape
            ``(base_n * step, ...)``.
        D: Number of input parameters.
        calc_second_order: Whether the layout includes the BA blocks.
        slice_chunk_size: Number of (T, K) output slices per vmap batch.

    Returns:
        ``(S1, ST, S2)``, with ``S2`` ``None`` when second order is off. The
        output axes match ``Y``'s trailing axes, as ``analyze`` documents.

    Raises:
        ValueError: If ``slice_chunk_size`` is below 1.
    """
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

        return S1_out, ST_out, S2_out

    # Batched path (T*K > 1): flatten (T, K) into a single batch dimension
    # so vmap processes all output slices in one vectorised call.
    # (N,T,K) -> transpose -> (T,K,N) -> reshape -> (T*K, N)
    # AB is (N,D,T,K) so its transpose puts (T,K) first then (N,D).
    A_flat = A.transpose(1, 2, 0).reshape(T * K, base_n)
    B_flat = B.transpose(1, 2, 0).reshape(T * K, base_n)
    AB_flat = AB.transpose(2, 3, 0, 1).reshape(T * K, base_n, D)

    if slice_chunk_size < 1:
        raise ValueError(f"slice_chunk_size must be >= 1, got {slice_chunk_size}")
    # Chunk the T*K batches to cap peak memory when many outputs exist
    cs = min(slice_chunk_size, total)

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

    S1_out = _squeeze_output_axes(S1_out, squeeze_time, squeeze_output)
    ST_out = _squeeze_output_axes(ST_out, squeeze_time, squeeze_output)
    if S2_out is not None:
        S2_out = _squeeze_output_axes(S2_out, squeeze_time, squeeze_output, n_trailing=2)
    return S1_out, ST_out, S2_out


def _analyze_no_bootstrap(
    sampling_result: SobolSamples, Y: Array, *, slice_chunk_size: int, invalid: InvalidReport
) -> SobolResult:
    """Wrap the estimator core in a ``SobolResult``, without a bootstrap."""
    S1_out, ST_out, S2_out = _indices_from_expanded(
        Y,
        sampling_result.n_params,
        sampling_result.calc_second_order,
        slice_chunk_size,
    )
    return SobolResult(
        S1=S1_out,
        ST=ST_out,
        S2=S2_out,
        problem=sampling_result.problem,
        invalid=invalid,
    )


def indices(
    sampling_result: SobolSamples, Y: Array, *, slice_chunk_size: int = 2048
) -> tuple[Array, ...]:
    """Compute Sobol indices as plain arrays, with no diagnostics.

    This is the transformable core of :func:`analyze`. It runs the same
    estimators on the same data and returns the same numbers, but it does
    nothing else: no non-finite check, no zero-variance warning, no
    :class:`jaxgsa.sobol.SobolResult`, and no read of any array value on the
    host. So it composes with ``jax.jit``, ``jax.vmap``, ``jax.grad`` and
    ``jax.jacrev``, which :func:`analyze` cannot, because a policy decision
    needs a concrete value and a tracer has none.

    Pair it with :meth:`jaxgsa.sobol.SobolSamples.transform` to differentiate
    an index with respect to the input distribution parameters::

        def s1(theta):
            Y = model(sampling_result.transform(theta))
            return jaxgsa.sobol.indices(sampling_result, Y)[0]

        dS1_dtheta = jax.jacrev(s1)(theta)

    Use :func:`analyze` for ordinary analysis. Nothing here checks the outputs,
    so a single NaN silently turns every index into NaN.

    Tier T4 (behavioural contract): the returned arrays must equal the
    corresponding fields of ``analyze``'s result on clean outputs, and the
    function must survive ``jit``, ``vmap`` and ``jit(jacrev(...))``. Checked
    in ``tests/test_sobol_gradients.py``.

    Args:
        sampling_result: The design from :func:`jaxgsa.sobol.sample`, used
            only for its expansion map and its Saltelli layout flags.
        Y: Model outputs evaluated at each unique row of
            ``sampling_result.samples``, in the same row order. Shapes are
            those :func:`analyze` accepts: ``(n_runs,)``, ``(n_runs, K)`` or
            ``(n_runs, T, K)``.
        slice_chunk_size: Number of (T, K) output slices per vmap batch. Lower
            it if you hit device out-of-memory errors.

    Returns:
        ``(S1, ST)``, or ``(S1, ST, S2)`` when the design was drawn with
        ``calc_second_order=True``. The shapes are those ``analyze`` reports.

    Raises:
        ValueError: If ``Y``'s first axis does not match
            ``sampling_result.n_runs``, or ``slice_chunk_size`` is below 1.
    """
    S1, ST, S2 = _indices_from_expanded(
        sampling_result.expand_outputs(Y),
        sampling_result.n_params,
        sampling_result.calc_second_order,
        slice_chunk_size,
    )
    if S2 is None:
        return S1, ST
    return S1, ST, S2


def _analyze_bootstrap(
    sampling_result: SobolSamples,
    Y: Array,
    *,
    num_resamples: int,
    conf_level: float,
    ci_method: Literal["quantile", "gaussian"],
    key: Array,
    slice_chunk_size: int,
    invalid: InvalidReport,
    keep_replicates: bool,
) -> SobolResult:
    """Compute Sobol indices with bootstrap confidence intervals.

    The no-bootstrap path vmaps over output slices. This path instead loops
    over the (T, K) combinations in Python and vmaps over the R resamples
    inside each slice. That trades some Python-loop overhead for bounded
    memory: each vmap call materialises R copies of a single (N,) or (N, D)
    slice, not R * T * K.
    """
    from jaxgsa.sobol._bootstrap import _bootstrap_first_total, _bootstrap_second_order

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
    # Only filled when the caller asked for the draws. R copies of every index
    # array is the largest thing this function can hold, so it is opt-in.
    S1_draw_list: list[Array] = []
    ST_draw_list: list[Array] = []
    S2_draw_list: list[Array] = []

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
                    indices, a, ab, ba, b, slice_chunk_size
                )
                s2_lo, s2_hi = _bootstrap_ci_endpoints(
                    s2,
                    s2_boot,
                    conf_level=conf_level,
                    ci_method=ci_method,
                )
                S2_lo_list.append(s2_lo)
                S2_hi_list.append(s2_hi)
                if keep_replicates:
                    S2_draw_list.append(s2_boot)
            else:
                s1, st = jit_ft(a, ab, b)
                s1_boot, st_boot = _bootstrap_first_total(indices, a, ab, b, slice_chunk_size)

            S1_list.append(s1)
            ST_list.append(st)
            if keep_replicates:
                S1_draw_list.append(s1_boot)
                ST_draw_list.append(st_boot)

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

    S1_out = _squeeze_output_axes(S1_out, squeeze_time, squeeze_output)
    ST_out = _squeeze_output_axes(ST_out, squeeze_time, squeeze_output)
    S1_conf = _squeeze_output_axes(S1_conf, squeeze_time, squeeze_output)
    ST_conf = _squeeze_output_axes(ST_conf, squeeze_time, squeeze_output)
    if S2_out is not None:
        S2_out = _squeeze_output_axes(S2_out, squeeze_time, squeeze_output, n_trailing=2)
    if S2_conf is not None:
        S2_conf = _squeeze_output_axes(S2_conf, squeeze_time, squeeze_output, n_trailing=2)

    replicates: dict[str, Array] | None = None
    if keep_replicates:

        def _stack_draws(per_slice: list[Array], n_trailing: int) -> Array:
            """Reorder per-slice draws into one array led by the resample axis.

            Each entry is ``(R, ...)`` for one ``(t, k)``. Stacking gives
            ``(T*K, R, ...)``; the resample axis has to move to the front so
            the layout matches the other four methods, and so the squeeze can
            address the inserted T and K axes from the end.
            """
            stacked = jnp.stack(per_slice).reshape(T, K, *per_slice[0].shape)
            moved = jnp.moveaxis(stacked, 2, 0)
            return _squeeze_output_axes(moved, squeeze_time, squeeze_output, n_trailing=n_trailing)

        replicates = {
            "S1": _stack_draws(S1_draw_list, 1),
            "ST": _stack_draws(ST_draw_list, 1),
        }
        if calc_second_order:
            replicates["S2"] = _stack_draws(S2_draw_list, 2)

    return SobolResult(
        S1=S1_out,
        ST=ST_out,
        S2=S2_out,
        problem=sampling_result.problem,
        invalid=invalid,
        S1_conf=S1_conf,
        ST_conf=ST_conf,
        S2_conf=S2_conf,
        ci=CIInfo(
            level=conf_level,
            method=ci_method,
            n_resamples=num_resamples,
            replicates=replicates,
        ),
    )


def analyze(
    sampling_result: SobolSamples,
    Y: Array,
    *,
    prenormalize: bool = False,
    num_resamples: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    slice_chunk_size: int = 2048,
    on_invalid: OnInvalid = "raise",
    keep_replicates: bool = False,
) -> SobolResult:
    """Compute Sobol sensitivity indices from model outputs using JAX.

    This is the main entry point of the package. Sobol indices apportion the
    variance of a model output among its input parameters. S1 (first-order) is
    the fraction of output variance explained by each parameter alone. ST
    (total-order) also includes all of that parameter's interactions with the
    other parameters. S2 (second-order) isolates pairwise interactions.

    The function takes the model outputs Y evaluated at the unique rows that
    ``jaxgsa.sobol.sample()`` returned. It rebuilds the expanded Saltelli
    ordering internally and checks it for non-finite values under the
    ``on_invalid`` policy. It then dispatches on ``num_resamples``: to the
    fast no-bootstrap path, or to the bootstrap confidence-interval path.

    Those checks read array values on the host, so ``analyze`` cannot be
    traced by ``jax.jit`` or differentiated by ``jax.grad``. Use
    :func:`jaxgsa.sobol.indices` when you need that; it runs the same
    estimators and skips every check.

    Args:
        sampling_result: Result from ``jaxgsa.sobol.sample()`` with the unique
            sample matrix plus expansion metadata.
        Y: Model outputs evaluated at each unique row of
            ``sampling_result.samples``, in the same row order, where
            ``n_runs`` is the unique row count. Accepted shapes:
                ``(n_runs,)``: scalar output, single time step
                ``(n_runs, K)``: K outputs, single time step
                ``(n_runs, T, K)``: K outputs over T time steps
            Indices are computed independently for every (t, k) output slice.
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
        slice_chunk_size: Memory/speed trade-off for batched computation.
            In the no-bootstrap path this is the number of (T, K) output
            slices per vmap batch; in the bootstrap path it caps the
            number of bootstrap resamples per batch. Lower it if you hit
            device out-of-memory errors. Defaults to 2048.
        on_invalid: What to do about non-finite model outputs. The unit here
            is one Saltelli group, so a single bad value removes the whole
            group of ``D + 2`` (or ``2D + 2``) rows. ``"raise"`` (the default)
            refuses the sample, ``"propagate"`` lets the value reach the
            indices, and ``"drop"`` analyzes the surviving groups. See
            :mod:`jaxgsa._core.invalid`.

    Returns:
        SobolResult holding:
            S1: first-order indices, shape ``(D,)`` / ``(K, D)`` /
                ``(T, K, D)`` for Y of shape (n,) / (n, K) / (n, T, K)
                respectively
            ST: total-order indices, same shape as S1
            S2: second-order indices with shape ``(..., D, D)``, or None when
                the design was drawn with ``calc_second_order=False``
            S1_conf, ST_conf, S2_conf: ``(2, ...)`` [lower, upper] CI bounds,
                or None when ``num_resamples == 0``
            invalid: What the non-finite check found, and what it did

    Raises:
        ValueError: If ``on_invalid`` is not one of the three policies; if
            ``ci_method`` is not ``"quantile"`` or ``"gaussian"``; if
            ``num_resamples`` is negative; if ``slice_chunk_size`` is below 1;
            if ``conf_level`` is not in ``(0, 1)``; if the sample holds a
            non-finite value under ``on_invalid="raise"``; or if fewer than 2
            Saltelli groups survive a drop.

    Warns:
        JaxgsaWarning: If an output slice has zero variance, which makes its
            indices NaN.
    """
    from jaxgsa.sobol import SPEC

    D = sampling_result.n_params
    # step = rows per Saltelli group: D+2 (first-order only) or 2D+2 (with S2)
    step = _saltelli_step(D, sampling_result.calc_second_order)
    # A Saltelli group [A_i, AB_{i,0}, …, B_i] is one indivisible sampling
    # unit, so the check runs at group granularity: a single bad row condemns
    # the whole block of `step` rows, and a partial group would corrupt the
    # A/B/AB/BA split. The group count comes from the design, not from Y, so
    # it is known before Y is looked at.
    base_n = sampling_result.n_expanded // step

    ctx = prepare(
        SPEC,
        sampling_result.problem,
        Y,
        on_invalid=on_invalid,
        checks=(
            one_of("ci_method", ci_method, ("quantile", "gaussian")),
            at_least("num_resamples", num_resamples, 0),
            at_least("slice_chunk_size", slice_chunk_size, 1),
            in_open_interval("conf_level", conf_level, 0.0, 1.0),
        ),
        n_expected=int(sampling_result.samples.shape[0]),
        expand=sampling_result.expand_outputs,
        n_units=base_n,
        unit_of_row=np.repeat(np.arange(base_n), step),
        min_kept=2,
    )
    # The estimator reads the expanded layout, and its scalar fast path
    # branches on Y's own rank, so this is ctx.Y and not ctx.Y3.
    Y = ctx.Y
    invalid = ctx.invalid
    if not ctx.keep.all():
        trailing = Y.shape[1:]
        # Boolean indexing gives a variable-length result, which JAX cannot
        # trace, so the compaction round-trips through NumPy.
        grouped = np.asarray(Y).reshape(base_n, step, *trailing)[ctx.keep]
        Y = jnp.asarray(grouped.reshape(-1, *trailing))

    if prenormalize:
        Y, _, _, _ = _prenormalize_outputs(Y)

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
            slice_chunk_size=slice_chunk_size,
            invalid=invalid,
            keep_replicates=keep_replicates,
        )

    return _analyze_no_bootstrap(
        sampling_result, Y, slice_chunk_size=slice_chunk_size, invalid=invalid
    )
