"""Optimal-transport index estimators for given data.

This module implements the optimal-transport (OT) sensitivity indices of
Borgonovo, Figalli, Plischke & Savare (2024). For each parameter the
estimator splits the sample into equal-frequency classes by the
parameter's rank. The index is the class-weighted squared 2-Wasserstein
distance between the conditional output distribution of each class and
the unconditional one. Dividing by twice the output variance, the
theoretical maximum of the averaged distance, puts the index in [0, 1].

Every index splits into two parts. The advective component is the
class-averaged squared distance between the conditional and the
unconditional means, so it says how far the parameter relocates the
output distribution. The diffusive remainder covers changes in spread
and shape. The advective numerator is exactly ``Var(E[Y|X_i])``, so the
advective component equals half the given-data first-order Sobol index.
That identity anchors the split to the variance-based indices.

Three modes cover jaxgsa's output shapes:

- ``"univariate"`` scores every output column independently with the
  closed-form 1-D optimal transport (sorted-quantile coupling, no
  solver). The unconditional sample supplies quantiles at the N uniform
  mass points. Each conditional class is evaluated at the same points
  through its empirical quantile function, coupling every mass point to
  the (at most two) class members its interval overlaps and weighting
  each by the exact overlap fraction, so the result is exact for any
  class size, not only one that divides N.
- ``"multivariate"`` and ``"trajectory"`` treat the flattened or
  per-output vector as a point cloud. They transport the unconditional
  cloud onto each class with entropic regularization (log-domain
  Sinkhorn, see :mod:`jaxgsa.optimal_transport._solver`) and report the
  unregularized cost of the entropic plan.

The original sample and any bootstrap resamples run through a single
scanned path. The original sample is replicate 0, gathered through the
identity permutation, so point estimates and confidence intervals share
one code path. The estimator rebuilds the class partitions per replicate
from the resampled parameters, exactly as in :mod:`jaxgsa.borgonovo`.

The estimator follows the paper's published equations. The test suite
validates it numerically against POT and against analytic closed forms.

References:
    Borgonovo, Figalli, Plischke & Savare (2024, online first; 2025 in
    print). Global sensitivity analysis via optimal transport. Management
    Science 71(5):3809-3828. doi:10.1287/mnsc.2023.01796.
"""

from __future__ import annotations

import functools
import warnings
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.batching import resolve_batch_size
from jaxgsa._core.bootstrap import interval
from jaxgsa._core.entry import (
    at_least,
    check_scalars,
    in_open_interval,
    one_of,
    prepare,
    require,
    validate_inputs,
)
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.partition import (
    PartitionGroup,
    _build_class_indices,
    _class_layout,
    _mask_from_counts,
    build_partition_groups,
)
from jaxgsa._core.result import CIInfo
from jaxgsa._core.validation import (
    _prepare_Y,
    _standardize_outputs,
    _validate_output,
)
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.optimal_transport._result import OTResult
from jaxgsa.optimal_transport._solver import _sinkhorn_w2
from jaxgsa.problem import Problem, _categorical_dims

# Live copies of the per-column conditional-quantile tensor to budget for.
# The dominant intermediate of the "univariate" mode is that tensor, of size
# ``chunk_columns * D * M * N`` elements; it is gathered and then reduced, so
# two copies are resident at the peak. The chunk width is sized against this
# multiple of the tensor, in bytes, so the peak stays inside the active
# transient-memory budget at any dtype. Counting elements instead would be
# wrong by the item size, and wrong by a further factor of two under x64.
_CHUNK_LIVE_TENSORS = 2

_MODES = ("univariate", "multivariate", "trajectory")

# Fewest samples that still build two conditioning classes of two points
# each, which is what the default ``M = min(25, N // 2)`` already assumes.
_MIN_KEPT = 4


def _aggregate_normalized(per_class: Array, weights: Array, V: Array) -> Array:
    """Class-weighted average of per-class costs, normalized by ``V = 2 * Var``.

    This is the defining [0, 1] normalization of the OT index. The 1-D
    and joint kernels share it, so the two modes can never drift apart. A
    non-positive ``V`` marks a constant output and yields exactly 0
    instead of NaN.

    A ``NaN`` ``V`` is not a constant output, it is a broken one. ``V > 0``
    is False for it, so without the second branch below it would take the
    constant-output path and report 0, which reads as "no influence" rather
    than "this did not compute". Only ``on_invalid="propagate"`` can reach
    that case: the other two policies remove the non-finite rows or refuse
    the sample.

    Args:
        per_class: Per-class costs, shape ``(D, M)``.
        weights: Class weights ``n_m / N`` that sum to 1 per column, shape
            ``(M,)`` when shared across columns or ``(D, M)`` when per
            column.
        V: Normalizer ``2 * Var`` (scalar). A non-positive value marks a
            constant output and yields exactly 0 instead of NaN.

    Returns:
        Normalized indices, shape ``(D,)``.
    """
    weighted_sum = (weights * per_class).sum(axis=-1)
    V_safe = jnp.where(V > 0, V, 1.0)
    normalized = jnp.where(V > 0, weighted_sum / V_safe, 0.0)
    return jnp.where(jnp.isnan(V), jnp.nan, normalized)


def _quantile_rank_split(counts: Array, N: int) -> tuple[Array, Array, Array]:
    """Exact conditional-quantile coupling for every full-sample mass point.

    Both empirical quantile functions are piecewise constant: the
    unconditional one over ``N`` intervals of width ``1/N``, a class of
    size ``c`` over ``c`` intervals of width ``1/c``. A class is a subset
    of the ``N``-row sample, so ``c <= N`` always, which means the class's
    breakpoints are spaced at least ``1/N`` apart -- at most one of them
    falls strictly inside any one ``1/N`` mass interval. So every mass
    interval couples to at most two class members: ``j_left`` over the
    fraction ``frac_left`` of the interval up to the breakpoint (if any),
    and ``j_right`` over the rest. Weighting each pair by that split
    reproduces the exact squared-2-Wasserstein integral
    ``sum_m (i-th interval) frac * (y - class_member)**2`` with no
    approximation, in particular when ``c`` does not divide ``N``, unlike
    the single nearest-rank lookup this replaced (which is exact only when
    it does).

    The function runs on device, per replicate, from the class counts. A
    precomputed ``(R, Dc, M, N)`` lookup table would be gigabytes for one
    high-cardinality column at large N, while this transient is
    ``(G, M, N)`` per replicate.

    The products ``i * c`` overflow int32 and exceed float32's
    exact-integer range, and float64 is unavailable without the x64 flag.
    The quotient and remainder are therefore formed exactly by schoolbook
    long division over the base-``S`` digits of ``c``, the same technique
    the coarser midpoint-rule lookup used, with more headroom here because
    the multiplier ``i < N`` instead of ``2i + 1 < 2N``. The bound below is
    kept identical to that lookup's for one proven-safe margin.

    Args:
        counts: Integer class sizes, shape ``(..., M)``.
        N: Number of samples.

    Returns:
        ``(j_left, j_right, frac_left)``. ``j_left`` and ``j_right`` are
        int32 lookup indices into each class's sorted members, and
        ``frac_left`` is the fraction of the mass interval's weight that
        goes to ``j_left`` (the rest goes to ``j_right``). All three have
        shape ``(..., M, N)``. ``j_left, j_right < c`` always, so the
        lookup never touches class padding; a zero-size class gets index 0
        and weight 0, discarded by the kernels through the zero class
        weight.
    """
    # int32-exactness bound: the multiplier here is i < N, half the size of
    # the midpoint-rule predecessor's 2i + 1 < 2N, so this bound has strictly
    # more headroom; kept at the same value for one proven-safe margin. A
    # raise, not an assert: asserts vanish under ``python -O`` and a
    # violated bound here is silent wrong numbers, not a debug aid.
    if N > (2**31 + 2) // 6:
        raise ValueError(
            f"jaxgsa.optimal_transport: N={N} exceeds {(2**31 + 2) // 6}, the "
            "largest sample size whose rank arithmetic is int32-exact."
        )
    # q = floor(i * c / N), r = i * c mod N, exact via long division of c's
    # base-S digits (i < N is the multiplier).
    i = jnp.arange(N, dtype=jnp.int32)  # (N,)
    c = counts.astype(jnp.int32)[..., None]  # (..., M, 1)
    shift = max(1, 29 - N.bit_length())  # S = 2**shift keeps 4*N*S <= 2**31
    S = 1 << shift
    n_digits = -(-max(1, N.bit_length()) // shift)  # ceil; c <= N
    q = jnp.zeros_like(c * i)  # broadcast to (..., M, N)
    r = jnp.zeros_like(q)
    for k in reversed(range(n_digits)):
        digit = (c >> (k * shift)) & (S - 1)
        t = r * S + i * digit  # < 4*N*S <= 2**31
        q = q * S + t // N
        r = t % N
    j_left = jnp.minimum(q, jnp.maximum(c - 1, 0)).astype(jnp.int32)
    j_right = jnp.minimum(q + 1, jnp.maximum(c - 1, 0)).astype(jnp.int32)
    width_left = jnp.minimum(N - r, c)  # in units of 1/(N * c), 0 for an empty class
    float_dtype = jnp.result_type(counts.dtype, jnp.float32)
    frac_left = width_left.astype(float_dtype) / jnp.maximum(c, 1).astype(float_dtype)
    return j_left, j_right, frac_left


@functools.partial(jax.jit, static_argnames=("group_levels",))
def _ot_1d_kernel(
    Y_cols: Array,
    all_idx: Array,
    groups: tuple[tuple[Array, Array], ...],
    group_levels: tuple[tuple[int, ...] | None, ...],
) -> tuple[Array, Array, Array, Array]:
    """Per-column 1-D optimal-transport indices for every replicate.

    Uses the closed-form 1-D coupling. Both empirical quantile functions
    are evaluated on the N uniform mass points of the full sample. The
    conditional one couples each mass point to the (at most two) class
    members its interval overlaps, exactly, through
    :func:`_quantile_rank_split`. No transport solver is involved.

    ``groups`` is a tuple of canonical ``(cls_idx, counts)``
    partition-group layouts from
    :func:`jaxgsa._core.partition.build_partition_groups`. The kernel
    processes every group in one call and concatenates the results on the
    parameter axis in group order. It computes the per-replicate column
    statistics (resample gather, sort, mean, variance) once and shares
    them. It derives the validity masks and quantile lookup indices
    in-kernel from the small counts, so it never materializes anything of
    size ``O(R * D * M * N)``. Zero-size classes (empty levels, padded
    level slots) carry zero weight and contribute zero cost.

    Args:
        Y_cols: Output columns, shape ``(N, C)``.
        all_idx: Replicate row indices, shape ``(R, N)``. Row 0 is the
            identity.
        groups: Canonical partition groups; see
            :mod:`jaxgsa._core.partition`.
        group_levels: One entry per group, ``None`` for a continuous group
            (whose class sizes are the same in every replicate) or a tuple
            of declared level counts for a categorical group. Static, so
            the group's quantile lookup table can be shared across
            replicates when it is continuous.

    Returns:
        ``(ot, advective, diffusive, degenerate)``. The three index arrays
        have shape ``(R, C, D)``. ``degenerate`` has shape ``(R, C)`` and
        flags constant (zero-variance) replicate columns, which yield zero
        indices.
    """
    dtype = jnp.result_type(Y_cols.dtype, jnp.float32)
    Y_cols = Y_cols.astype(dtype)
    # Every quantity below is a difference of Y values (the W2^2 gap, the
    # advective mean shift), so a column-wide shift cancels exactly in
    # exact arithmetic. Centering first keeps that cancellation accurate in
    # float32 too: an output offset far from 0 (a large mean, an unshifted
    # physical unit) otherwise leaves the advective component off by up to
    # ~1e-3 at an offset of 1e5, because the shift itself, not the O(1)
    # signal riding on it, was eating the float32 mantissa.
    Y_cols = Y_cols - Y_cols.mean(axis=0, keepdims=True)
    N = Y_cols.shape[0]
    counts_list = tuple(counts for _, counts in groups)
    cls_list = tuple(cls_idx for cls_idx, _ in groups)
    group_is_shared = tuple(levels is None for levels in group_levels)

    def _group_stats(y, y_sorted, mean_r, V, cls_idx, mask_b, counts_b, j_b):
        """Weighted, normalized cost sums for one group's parameters.

        ``mask_b (G, M, P)``, ``counts_b (G, M)``, and ``j_b (G, M, N)``
        broadcast against the group's parameter axis Dg. ``G`` is always Dg.
        """
        weights = counts_b / N  # (G, M)
        safe_counts = jnp.maximum(counts_b, 1.0)
        valid = counts_b > 0

        y_cls = y[cls_idx]  # (Dg, M, P) resampled class members
        # Pads sort to the tail as +inf and are unreachable through j.
        #
        # This sort is the single most expensive operation in the method:
        # it orders every sample of every parameter, once per output
        # column and once per replicate. It asks only for the order
        # statistics, never for which tied element came first, so the
        # stable sort ``jnp.sort`` would emit is wasted work. An unstable
        # sort returns the identical array, because two float32 values
        # that compare equal are the same bits, and it measured about
        # 1.8x faster on the CPU backend.
        y_cls_sorted = jax.lax.sort(
            jnp.where(mask_b, y_cls, jnp.inf), dimension=-1, is_stable=False
        )
        j_left_b, j_right_b, frac_left_b = j_b
        shape = (y_cls.shape[0],) + j_left_b.shape[-2:]
        j_left_full = jnp.broadcast_to(j_left_b, shape)
        j_right_full = jnp.broadcast_to(j_right_b, shape)
        frac_left_full = jnp.broadcast_to(frac_left_b, shape)
        q_left = jnp.take_along_axis(y_cls_sorted, j_left_full, axis=-1)  # (Dg, M, N)
        q_right = jnp.take_along_axis(y_cls_sorted, j_right_full, axis=-1)  # (Dg, M, N)
        # Exact 1-D W2^2: every one of the N mass intervals couples to at
        # most two class members (see _quantile_rank_split), so this is the
        # exact integral, not the midpoint-rule approximation.
        y_b = y_sorted[None, None, :]
        w2 = (
            frac_left_full * (y_b - q_left) ** 2 + (1.0 - frac_left_full) * (y_b - q_right) ** 2
        ).mean(axis=-1)  # (Dg, M)

        cls_mean = (y_cls * mask_b.astype(dtype)).sum(axis=-1) / safe_counts  # (Dg, M)
        adv = (cls_mean - mean_r) ** 2
        # A zero-size class carries zero weight; zero its (infinite) cost
        # so 0 * inf never poisons the weighted sum.
        w2 = jnp.where(valid, w2, 0.0)
        adv = jnp.where(valid, adv, 0.0)
        # W2^2 >= (mean shift)^2 mathematically; the clamp only absorbs
        # float cancellation noise near zero.
        diff = jnp.maximum(w2 - adv, 0.0)

        return (
            _aggregate_normalized(w2, weights, V),
            _aggregate_normalized(adv, weights, V),
            _aggregate_normalized(diff, weights, V),
        )

    def _col_stats(y: Array, r: Array, layouts):
        """OT/advective/diffusive indices for one column and replicate."""
        y_r = y[r]  # resampled column
        # Order statistics only, so an unstable sort is both equivalent
        # and cheaper; see the class sort in _group_stats.
        y_sorted = jax.lax.sort(y_r, dimension=-1, is_stable=False)
        mean_r = y_r.mean()
        V = 2.0 * jnp.var(y_r, ddof=1)
        # Same predicate the aggregation zeroes on, so the CI
        # neutralization can never disagree with the reported zeros.
        degenerate = ~(V > 0)

        outs = [
            _group_stats(y, y_sorted, mean_r, V, cls_idx, mask_b, counts_b, j_b)
            for cls_idx, mask_b, counts_b, j_b in layouts
        ]
        merged = [jnp.concatenate([o[i] for o in outs]) for i in range(3)]
        return merged[0], merged[1], merged[2], degenerate

    # A continuous group's equal-frequency class sizes are the same in
    # every replicate (only the categorical group's class sizes vary with
    # the resample), so its quantile lookup is the same table every time.
    # Build it once here rather than R times inside the scan.
    shared_j = tuple(
        _quantile_rank_split(counts_g[0], N) if is_shared else None
        for counts_g, is_shared in zip(counts_list, group_is_shared, strict=True)
    )

    def _one_replicate(carry, xs):
        i, r, cls_parts = xs
        layouts = []
        for cls_r, counts_g, j_shared in zip(cls_parts, counts_list, shared_j):
            counts_r = counts_g[i]  # (G, M) int
            mask_r = _mask_from_counts(counts_r, cls_r.shape[-1])
            j_r = _quantile_rank_split(counts_r, N) if j_shared is None else j_shared
            layouts.append((cls_r, mask_r, counts_r.astype(dtype), j_r))
        out = jax.vmap(lambda y: _col_stats(y, r, layouts))(Y_cols.T)
        return carry, out

    R = all_idx.shape[0]
    scan_xs = (jnp.arange(R), all_idx, cls_list)
    _, (ot, adv, diff, degen) = jax.lax.scan(_one_replicate, None, scan_xs)
    return ot, adv, diff, degen


@jax.jit
def _joint_kernel(
    Z: Array,
    all_idx: Array,
    groups: tuple[tuple[Array, Array], ...],
    dm_grids: tuple[Array, ...],
    epsilon: Array,
    max_iter: Array,
    tol: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Joint (point-cloud) optimal-transport indices for every replicate.

    For every parameter and class, transports the unconditional output
    cloud onto the class's conditional cloud with entropic
    regularization. It then aggregates the per-class costs into one index
    per parameter. The entropic bias of the regularized cost lands
    entirely in ``diffusive``: ``advective`` comes from exact class means,
    and ``diffusive`` is the remainder ``ot - advective``.

    ``groups`` is a tuple of canonical ``(cls_idx, counts)``
    partition-group layouts from
    :func:`jaxgsa._core.partition.build_partition_groups`. Results
    concatenate on the parameter axis in group order, and the shared
    per-replicate cloud statistics are computed once. ``dm_grids`` gives,
    per group, the flat ``d * Mg + m`` class slots to solve. A categorical
    group's class axis is padded to the largest level count, so the
    statically empty pad slots are excluded from the grid instead of
    running dead Sinkhorn solves. Dynamically empty classes (declared
    levels with no observed samples) still carry zero weight and do not
    count as convergence failures.

    Args:
        Z: Output point cloud, shape ``(N, E)``, already standardized when
            the caller requested it.
        all_idx: Replicate row indices, shape ``(R, N)``. Row 0 is the
            identity.
        groups: Canonical partition groups; see
            :mod:`jaxgsa._core.partition`.
        dm_grids: Per-group int32 arrays of flat class slots to solve.
        epsilon: Entropic regularization strength (scalar).
        max_iter: Sinkhorn iteration cap (scalar).
        tol: Sinkhorn marginal stopping tolerance (scalar).

    Returns:
        ``(ot, advective, diffusive, n_bad, degenerate)``. The three index
        arrays have shape ``(R, D)``. ``n_bad`` has shape ``(R,)`` and
        counts, per replicate, the Sinkhorn solves whose marginal residual
        stayed above ``tol``. ``degenerate`` has shape ``(R,)`` and flags
        constant (zero-variance) replicate clouds, which yield zero
        indices.
    """
    dtype = jnp.result_type(Z.dtype, jnp.float32)
    Z = Z.astype(dtype)
    # See the matching comment in _ot_1d_kernel: the cost and the advective
    # mean shift are both differences of Z values, so centering each output
    # coordinate first keeps that cancellation accurate in float32 when the
    # raw output sits far from 0.
    Z = Z - Z.mean(axis=0, keepdims=True)
    N = Z.shape[0]
    counts_list = tuple(counts for _, counts in groups)
    cls_list = tuple(cls_idx for cls_idx, _ in groups)

    def _group_stats(Z_r, mean_all, sq_full, V, cls_idx, counts_r, dm_grid):
        """Normalized cost sums and failure count for one group's parameters.

        ``counts_r (G, M)`` broadcasts against the group's parameter axis
        Dg. ``G`` is always Dg.
        """
        Dg, M, P = cls_idx.shape[0], cls_idx.shape[1], cls_idx.shape[2]
        mask_r = _mask_from_counts(counts_r, P)  # (G, M, P)
        safe_counts = jnp.maximum(counts_r, 1.0)
        log_b_all = jnp.where(mask_r, -jnp.log(safe_counts)[..., None], -jnp.inf)  # (G, M, P)

        def _one_class(dm: Array):
            """Transport cost of the full cloud onto class dm % M of parameter dm // M."""
            m = dm % M
            g = dm // M  # G is always Dg (per D4), so no clamp is needed
            idx = cls_idx[dm // M, m]  # (P,)
            mask_dm = mask_r[g, m].astype(dtype)  # (P,)
            count_dm = counts_r[g, m]
            Z_c = Z[idx]  # (P, E)
            # Squared Euclidean cost block (N, P). Padded columns carry
            # zero target mass, so zeroing their costs is exact and keeps
            # <P, C> correct regardless of what a pad's raw coordinates are.
            C = sq_full[:, None] + (Z_c**2).sum(axis=-1)[None, :] - 2.0 * (Z_r @ Z_c.T)
            C = jnp.maximum(C, 0.0) * mask_dm[None, :]
            # scale=V, not this class's own max cost: see _sinkhorn_w2's
            # docstring (M3) for why the regularization must be relative to
            # one fixed, class-independent normalizer.
            cost, err = _sinkhorn_w2(C, log_b_all[g, m], epsilon, max_iter, tol, V)
            cls_mean = (Z_c * mask_dm[:, None]).sum(axis=0) / jnp.maximum(count_dm, 1.0)
            adv = ((cls_mean - mean_all) ** 2).sum()
            # An empty class carries zero weight; discard its (junk) solve
            # so it neither poisons the sum nor counts as a failure.
            empty = ~(count_dm > 0)
            zero = jnp.zeros((), dtype)
            return (
                jnp.where(empty, zero, cost),
                jnp.where(empty, zero, adv),
                jnp.where(empty, zero, err),
            )

        # Sequential map keeps peak memory at one (N, P) cost block.
        costs, advs, errs = jax.lax.map(_one_class, dm_grid)
        # Scatter the solved slots back onto the dense (Dg, M) class grid
        # (each slot is unique, so this is a set, not an add). Excluded
        # pad slots stay 0 and carry zero weight, so the weighted
        # aggregation below is unchanged from a full-grid solve.
        d_idx = dm_grid // M
        m_idx = dm_grid % M
        w2 = jnp.zeros((Dg, M), dtype).at[d_idx, m_idx].set(costs)
        adv = jnp.zeros((Dg, M), dtype).at[d_idx, m_idx].set(advs)
        diff = jnp.maximum(w2 - adv, 0.0)
        weights = counts_r / N  # (G, M)
        return (
            _aggregate_normalized(w2, weights, V),
            _aggregate_normalized(adv, weights, V),
            _aggregate_normalized(diff, weights, V),
            (errs > tol).sum(),
        )

    def _one_replicate(carry, xs):
        i, r, cls_parts = xs
        Z_r = Z[r]  # resampled cloud
        mean_all = Z_r.mean(axis=0)
        V = 2.0 * jnp.var(Z_r, axis=0, ddof=1).sum()  # == 2 * Tr(Cov)
        sq_full = (Z_r**2).sum(axis=-1)  # (N,)

        outs = [
            _group_stats(
                Z_r,
                mean_all,
                sq_full,
                V,
                cls_r,
                counts_g[i].astype(dtype),
                dm_grid,
            )
            for cls_r, counts_g, dm_grid in zip(cls_parts, counts_list, dm_grids)
        ]
        merged = [jnp.concatenate([o[i_out] for o in outs]) for i_out in range(3)]
        n_bad = jnp.stack([o[3] for o in outs]).sum()
        return carry, (merged[0], merged[1], merged[2], n_bad, ~(V > 0))

    R = all_idx.shape[0]
    scan_xs = (jnp.arange(R), all_idx, cls_list)
    _, (ot, adv, diff, n_bad, degen) = jax.lax.scan(_one_replicate, None, scan_xs)
    return ot, adv, diff, n_bad, degen


def _static_dm_grid(group: tuple[Array, Array], levels: list[int] | None) -> Array:
    """Flat ``d * Mg + m`` class slots the joint kernel must solve.

    A categorical group's class axis is padded to the largest level count.
    The pad slots are statically empty, because their counts are zero for
    every replicate. They are excluded here instead of running dead
    Sinkhorn solves.

    Args:
        group: One canonical ``(cls_idx, counts)`` partition group.
        levels: Declared level count per column for a categorical group, or
            ``None`` for the shared continuous layout.

    Returns:
        The int32 slot indices to solve.
    """
    D_g, M_g = group[0].shape[1], group[0].shape[2]
    if levels is None:
        return jnp.arange(D_g * M_g, dtype=jnp.int32)
    return jnp.asarray(
        [d * M_g + m for d, n_levels in enumerate(levels) for m in range(n_levels)],
        dtype=jnp.int32,
    )


def _run_univariate(
    Y_cols: Array,
    T: int,
    K: int,
    all_idx: Array,
    groups: list[tuple[Array, Array]],
    group_levels: list[list[int] | None],
    col_order: Array | None,
    slice_chunk_size: int | None,
) -> tuple[Array, Array, Array, Array]:
    """Run :func:`_ot_1d_kernel` over every chunk of output columns.

    This is the whole ``"univariate"`` estimator, and nothing else: it
    branches on shapes and Python scalars only, reads no array value on the
    host, and raises and warns about nothing. :func:`analyze` and
    :func:`indices` both go through it, so the two can never compute
    different numbers.

    Args:
        Y_cols: Output columns, shape ``(N, T*K)``.
        T: Time axis length of the promoted ``Y``.
        K: Output axis length of the promoted ``Y``.
        all_idx: Replicate row indices ``(R, N)``. Row 0 is the identity.
        groups: Canonical partition groups; see
            :mod:`jaxgsa._core.partition`.
        group_levels: Declared level counts per group, ``None`` for the
            continuous group, in the same order as ``groups``.
        col_order: Gather that restores the problem's column order, or
            ``None`` when the groups are already in order.
        slice_chunk_size: Output columns per kernel call, or ``None`` for a
            memory-aware default.

    Returns:
        ``(ot, advective, diffusive, degenerate)`` of shapes
        ``(R, T, K, D)`` three times and ``(R, T, K)``.
    """
    N, total = Y_cols.shape
    R_run = all_idx.shape[0]
    D_run = sum(g[0].shape[1] for g in groups)
    # Peak memory scales with the summed per-group D_g * M_g
    # conditional-quantile tensors, in bytes at the kernel's working dtype.
    layout_elems = sum(g[0].shape[1] * g[0].shape[2] for g in groups)
    itemsize = jnp.dtype(jnp.result_type(Y_cols.dtype, jnp.float32)).itemsize
    bytes_per_column = _CHUNK_LIVE_TENSORS * layout_elems * N * itemsize
    cs = resolve_batch_size(bytes_per_column, total, slice_chunk_size)
    levels_static = tuple(tuple(lv) if lv is not None else None for lv in group_levels)
    parts: tuple[list[Array], list[Array], list[Array], list[Array]] = ([], [], [], [])
    for start in range(0, total, cs):
        chunk = Y_cols[:, start : start + cs]
        n_real = chunk.shape[1]
        if n_real < cs:
            # Pad the ragged trailing chunk back to the full width so the
            # jitted kernel compiles for one shape only. Output columns are
            # scored independently, so a repeated column cannot change a real
            # column's answer, and the padding is sliced off straight away.
            pad = jnp.broadcast_to(chunk[:, :1], (chunk.shape[0], cs - n_real))
            chunk = jnp.concatenate([chunk, pad], axis=1)
        merged = list(_ot_1d_kernel(chunk, all_idx, tuple(groups), levels_static))
        if n_real < cs:
            merged = [arr[:, :n_real] for arr in merged]
        if col_order is not None:
            merged[:3] = [arr[..., col_order] for arr in merged[:3]]
        for part, arr in zip(parts, merged):
            part.append(arr)
    ot, adv, diff, degen = (jnp.concatenate(p, axis=1) for p in parts)
    return (
        ot.reshape(R_run, T, K, D_run),
        adv.reshape(R_run, T, K, D_run),
        diff.reshape(R_run, T, K, D_run),
        degen.reshape(R_run, T, K),
    )


def _build_clouds(Y_3d: Array, mode: str, standardize_outputs: bool) -> list[Array]:
    """Split the promoted output into the point clouds a joint mode scores.

    Args:
        Y_3d: Output promoted to ``(N, T, K)``.
        mode: ``"multivariate"`` (one flattened cloud) or ``"trajectory"``
            (one cloud per output, each output's time course).
        standardize_outputs: Divide each column by its standard deviation first, so
            no single output dominates the joint distance through its units.

    Returns:
        The clouds, each of shape ``(N, E)``.
    """
    N, T, K = Y_3d.shape
    clouds = (
        [Y_3d.reshape(N, T * K)] if mode == "multivariate" else [Y_3d[:, :, k] for k in range(K)]
    )
    if standardize_outputs:
        clouds = [_standardize_outputs(Z)[0] for Z in clouds]
    return clouds


def _run_joint(
    clouds: list[Array],
    mode: str,
    all_idx: Array,
    groups: list[tuple[Array, Array]],
    group_levels: list[list[int] | None],
    col_order: Array | None,
    eps_s: Array,
    max_iter_s: Array,
    tol_s: Array,
) -> tuple[Array, Array, Array, Array, Array, int]:
    """Run :func:`_joint_kernel` over every point cloud.

    This is the whole point-cloud estimator, and nothing else: it branches
    on shapes and Python scalars only, reads no array value on the host, and
    raises and warns about nothing. The Sinkhorn convergence counters come
    back as an array and a Python integer for :func:`analyze` to warn from;
    :func:`indices` drops them without reading either.

    Args:
        clouds: Output point clouds from :func:`_build_clouds`.
        mode: ``"multivariate"`` or ``"trajectory"``. It decides only
            whether the per-cloud results stack onto a new output axis.
        all_idx: Replicate row indices ``(R, N)``. Row 0 is the identity.
        groups: Canonical partition groups; see
            :mod:`jaxgsa._core.partition`.
        group_levels: Declared level counts per group, ``None`` for the
            continuous group, in the same order as ``groups``.
        col_order: Gather that restores the problem's column order, or
            ``None`` when the groups are already in order.
        eps_s: Entropic regularization strength.
        max_iter_s: Sinkhorn iteration cap.
        tol_s: Sinkhorn marginal stopping tolerance.

    Returns:
        ``(ot, advective, diffusive, degenerate, n_bad, n_solves)``. The
        index arrays are ``(R, D)`` in ``"multivariate"`` mode and
        ``(R, K, D)`` in ``"trajectory"`` mode, ``degenerate`` drops the
        parameter axis, ``n_bad`` is a scalar array counting the solves that
        stayed above ``tol``, and ``n_solves`` is how many ran.
    """
    dm_grids = tuple(_static_dm_grid(group, levels) for group, levels in zip(groups, group_levels))
    n_solves = len(clouds) * all_idx.shape[0] * sum(grid.shape[0] for grid in dm_grids)

    def _one_cloud(Z: Array) -> tuple[Array, Array, Array, Array, Array]:
        ot, adv, diff, n_bad, degen = _joint_kernel(
            Z, all_idx, tuple(groups), dm_grids, eps_s, max_iter_s, tol_s
        )
        merged = [ot, adv, diff]
        if col_order is not None:
            merged = [arr[..., col_order] for arr in merged]
        return merged[0], merged[1], merged[2], n_bad.sum(), degen

    cloud_outs = [_one_cloud(Z) for Z in clouds]
    n_bad_total = jnp.stack([o[3] for o in cloud_outs]).sum()
    if mode == "multivariate":
        ot, adv, diff, _, degen = cloud_outs[0]  # (R, D) / (R,)
        return ot, adv, diff, degen, n_bad_total, n_solves
    return (
        jnp.stack([o[0] for o in cloud_outs], axis=1),  # (R, K, D)
        jnp.stack([o[1] for o in cloud_outs], axis=1),
        jnp.stack([o[2] for o in cloud_outs], axis=1),
        jnp.stack([o[4] for o in cloud_outs], axis=1),  # (R, K)
        n_bad_total,
        n_solves,
    )


def _resolve_tol(tol: float | None, dtype: jnp.dtype) -> float:
    """Default the Sinkhorn stopping tolerance to what the dtype can resolve."""
    if tol is not None:
        return tol
    return 1e-9 if dtype == jnp.float64 else 1e-6


def _resolve_n_classes(n_partitions: int | None, N: int, *, needs_M: bool) -> int:
    """Resolve ``M``, the equal-frequency class count, from ``n_partitions``.

    Shared between :func:`analyze` and :func:`indices`: both default
    ``n_partitions`` to ``min(25, N // 2)`` and validate an explicit value
    against ``[2, N // 2]``. Only the default path can skip the too-small
    floor check, and only when nothing in this call actually consumes
    ``M`` -- an all-categorical problem, whose parameters and whose dummy
    floors all use one class per level. :func:`indices` always needs
    ``M``, since it refuses categorical parameters outright.

    Args:
        n_partitions: The caller's ``n_partitions`` argument.
        N: Sample size.
        needs_M: Whether anything in this call consumes ``M``. Only
            relevant when ``n_partitions is None``.

    Returns:
        ``M``, resolved and validated.

    Raises:
        ValueError: If ``n_partitions`` is outside ``[2, N // 2]``, or the
            default is used, ``needs_M`` is true, and ``N`` is too small to
            build two classes.
    """
    if n_partitions is None:
        # The customary 25 classes, clamped so small samples do not raise
        # over a default the user never passed.
        M = min(25, N // 2)
        if needs_M and M < 2:
            raise ValueError(f"building conditioning classes needs N >= 4 samples, got N={N}")
        return M
    M = int(n_partitions)
    if not 2 <= M <= N // 2:
        raise ValueError(f"n_partitions must be in [2, N//2={N // 2}], got {n_partitions}")
    return M


def _matched_dummy_group(key: Array, N: int, sizes: np.ndarray) -> PartitionGroup:
    """Random-membership partition group holding the given class sizes.

    The matched irrelevance floor for one categorical parameter (M4): the
    finite-sample OT bias scales with class size, so the null baseline for
    a categorical column must share its class-size structure, not the
    continuous dummy's equal-frequency M classes. Assigns the N samples to
    classes uniformly at random and independent of the output, holding the
    class sizes fixed at the real column's own observed, non-empty counts.

    Args:
        key: PRNG key for the random assignment.
        N: Number of samples.
        sizes: Observed, non-empty class sizes, shape ``(M,)``, summing to
            ``N``.

    Returns:
        A canonical ``(cls_idx, counts)`` partition group for one column:
        ``cls_idx (1, 1, M, P)``, ``counts (1, 1, M)``.
    """
    edges = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    P = int(sizes.max())
    perm = jax.random.permutation(key, N)
    take = jnp.minimum(jnp.asarray(edges[:-1])[:, None] + jnp.arange(P)[None, :], N - 1)  # (M, P)
    cls_idx = perm[take][None, None]  # (1, 1, M, P)
    return cls_idx, jnp.asarray(sizes)[None, None, :]


def _boot_replicates(vals_all: Array, hat: Array, degen_all: Array) -> Array:
    """The bootstrap draws the interval is taken from.

    A constant bootstrap resample carries no information. It contributes
    the point estimate instead of a spurious zero, so it neither widens
    nor shifts the interval (borgonovo convention).

    Args:
        vals_all: Replicate values, shape ``(R, ..., D)``. Row 0 is the
            point estimate.
        hat: Point estimate, shape ``(..., D)``.
        degen_all: Degeneracy flags, shape ``(R, ...)``.

    Returns:
        The ``(R - 1, ..., D)`` bootstrap draws with degenerate rows
        neutralized.
    """
    degen_boot = degen_all[1:][..., None]
    return jnp.where(degen_boot, hat[None], vals_all[1:])


def analyze(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    mode: Literal["univariate", "multivariate", "trajectory"] = "univariate",
    n_partitions: int | None = None,
    standardize_outputs: bool = True,
    epsilon: float = 0.03,
    max_iter: int = 2000,
    tol: float | None = None,
    dummy: bool = False,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    slice_chunk_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> OTResult:
    """Compute optimal-transport sensitivity indices from given data.

    The optimal-transport (OT) index measures how much knowing a
    parameter's value displaces the whole output distribution. It is the
    class-averaged squared 2-Wasserstein distance between the output
    distribution conditional on the parameter and the unconditional one,
    normalized to [0, 1] by twice the output variance. 0 means the
    parameter leaves the output distribution unchanged. 1 means the
    parameter determines the output distribution fully. Variance-based
    indices do not react to changes in spread, tails and shape, and this
    index does.

    The returned split separates the location-shift part (``advective``)
    from the spread/shape remainder (``diffusive``). The advective part is
    exactly half the given-data first-order Sobol index. Any ``(X, Y)``
    sample works, and no special design is needed.

    Conditioning classes come from the ordinal ranks of the parameters.
    Ranks are invariant under monotone transforms, so the estimator is
    distribution-free in X. It works unchanged for uniform, Gaussian,
    truncated-Gaussian, or mixed marginals, and it applies no CDF
    transform. Categorical parameters work natively. Each one conditions
    on one class per level, and the class sizes are the observed level
    counts, so the index depends only on the level partition and never on
    the arbitrary code order. The estimator drops declared levels with no
    observed samples and warns.

    Correlated parameters are supported. The index stays well-defined
    under dependence, and it then measures each parameter's total
    association with the output, including effects mediated by correlated
    parameters. This is the same reading as the given-data S1, whose half
    remains the advective component. A parameter that does not enter the
    model but correlates with one that does therefore scores non-zero.
    That is the correct reading, not an error.

    Args:
        problem: Problem definition with D parameters.
        X: Parameter sample matrix, shape ``(N, D)``.
        Y: Model output, shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        mode: Output treatment. ``"univariate"`` (default) scores every
            output column independently with exact 1-D optimal transport,
            giving indices of shape ``(T, K, D)``, squeezed.
            ``"multivariate"`` treats the whole flattened output vector as
            one point cloud and gives a single index per parameter over
            the joint output distribution, shape ``(D,)``. It uses
            entropic Sinkhorn transport. ``"trajectory"`` does the same
            per output, with each output's time course as the cloud, shape
            ``(K, D)``. It requires a 3-D ``Y``.
        n_partitions: Number of equal-frequency conditioning classes per
            continuous parameter. More classes localize the conditioning
            and lower the discretization bias, but they leave fewer
            samples per class and raise the estimation noise. About 25 is
            customary for the OT index at N >= 2500. ``None`` (default)
            selects ``min(25, N // 2)``. Categorical parameters ignore it
            and always use one class per level, and so does the dummy
            floor each of them is measured against. A passed value is
            always validated against ``[2, N // 2]``. If every parameter is
            categorical, nothing uses the value and a ``JaxgsaWarning``
            says it is ignored.
        standardize_outputs: Joint modes only. Divide each output column by its
            standard deviation before building the transport cost, so no
            single output dominates the joint distance through its units.
            In ``"trajectory"`` mode a "column" is one time step of the
            cloud, so this standardizes every time step to unit variance on
            its own; it does not preserve the trajectory's own relative
            shape over time, and the default (``True``) never computes a
            plain-units L2 trajectory transport. Pass ``False`` for that.
            Ignored in ``"univariate"`` mode, where each column is
            normalized by its own variance regardless.
        epsilon: Joint modes only. Entropic regularization strength,
            relative to ``V``, the index's own normalizer (``2 * Var`` or
            ``2 * tr(Cov)``). Every parameter and every class share that
            one scale, so the regularization means the same thing for all
            of them. Smaller values approach exact transport at the price
            of more iterations. Measured against POT's exact ``emd2`` on
            Ishigami (N=1000, 10 classes), the default 0.03 reads 7.3 per
            cent high; 0.02 reads 4.8 per cent high and takes half again
            as long; 0.05 reads 12.1 per cent high. The offset is close to
            uniform across parameters, because one ``V`` scales every
            cost, so it shifts the indices together and leaves their
            ranking and the ``above_dummy`` comparison alone.
        max_iter: Joint modes only. Sinkhorn iteration cap per solve.
        tol: Joint modes only. Stopping tolerance on the L1 target-
            marginal violation. ``None`` selects ``1e-9`` in float64 and
            ``1e-6`` in float32, where a tighter value is unresolvable.
            One warning is emitted if any solve fails to converge. In
            float32 the residual can stop falling a little above ``1e-6``
            for a large cloud, because the residual is a sum over ``N``
            rounded terms. The cost itself is converged there, so raising
            ``max_iter`` does not clear the warning. Raise ``tol`` or
            enable float64 if you need it silent.
        dummy: Also push one synthetic parameter through the identical
            pipeline per parameter and report the result as ``ot_dummy``.
            Every synthetic parameter is independent of the output by
            construction. Its index estimates the floor a fully irrelevant
            parameter receives from finite-sample bias and, in the
            point-cloud modes, from entropic bias. All continuous
            parameters share one floor, built from the same equal-frequency
            classes they use themselves. Each categorical parameter gets
            its own floor instead, matched to that column's own observed
            class sizes, because the finite-sample bias scales with class
            size. Parameters not clearly above their own floor are
            indistinguishable from noise.
        n_bootstrap: Number of bootstrap resamples for confidence
            intervals. ``0`` (default) skips them, and the ``*_conf``
            fields are ``None``. Joint modes solve
            ``n_bootstrap * D * n_partitions`` transport problems, so keep
            the value modest there.
        conf_level: Confidence level for the bootstrap intervals.
        ci_method: How the interval endpoints are formed. ``"quantile"``
            (default) reads them off the empirical bootstrap distribution.
            ``"gaussian"`` centres them on the point estimate and takes
            ``+/- z * sd`` of the bootstrap draws, which is smoother for a
            small ``n_bootstrap`` but assumes the draws are normal.
        key: A ``jax.random`` key. It feeds the bootstrap resampling and the
            synthetic ``dummy`` parameter, which are two independent
            consumers, so it is required when either ``n_bootstrap > 0`` or
            ``dummy=True``. Pass ``jax.random.key(0)`` if you have an
            integer seed.
        slice_chunk_size: ``"univariate"`` mode only. Number of flattened
            ``T*K`` output columns processed per kernel call. ``None``
            picks a memory-aware default. The point-cloud modes accept it
            but ignore it, because one ``(N, N/M)`` cost block per solve
            bounds their peak memory.
        on_invalid: What to do about a row of ``X`` or ``Y`` that holds a
            non-finite value. ``"raise"`` (default) refuses the sample,
            ``"drop"`` removes those rows and analyzes the rest, and
            ``"propagate"`` warns and computes anyway. The check reads the
            real ``X`` and ``Y`` the caller passed, not the synthetic
            ``dummy`` column, and it reads them together, so a bad input
            takes its own output with it. See :mod:`jaxgsa._core.invalid`.
        verbose: If ``True`` (default), print a short summary to stdout: the
            problem and the data, the wall-clock timing, and the top
            parameters by ``ot``. Pass ``False`` for a silent run.
        keep_replicates: Keep the per-resample indices on
            ``OTResult.ci.replicates``. Off by default because they are
            large: ``n_bootstrap`` copies of all three index arrays. Turn it
            on to recompute an interval at another level without re-running
            the analysis, which for this method means without re-solving
            every transport problem.

    Returns:
        An :class:`OTResult` with the total, advective and diffusive
        indices, the given-data first-order Sobol index ``S1`` (the
        advective component on borgonovo's ddof=0 Sobol convention),
        optional confidence intervals, the optional dummy baseline together
        with the floor-cleared ``above_dummy``, and the non-finite report in
        ``invalid``. Every index is
        0 for a constant (zero-variance) output
        slice rather than NaN. In the point-cloud modes the entropic and
        finite-sample bias keeps the indices of irrelevant parameters
        strictly positive. Compare those against ``ot_dummy`` rather than
        against 0.

    Raises:
        ValueError: If X is not 2-D, its column count does not match the
            problem, Y is not 1-D/2-D/3-D, X and Y have differing row
            counts, ``mode`` is unknown, ``mode="trajectory"`` is
            used with a non-3-D Y, a passed ``n_partitions`` is not in
            ``[2, N // 2]``, a categorical column of X holds
            values other than its integer level codes, ``epsilon <= 0``,
            ``max_iter < 1``,
            ``tol <= 0``, ``n_bootstrap < 0``, ``ci_method`` is neither
            ``"quantile"`` nor ``"gaussian"``, no ``key`` was given while
            ``n_bootstrap > 0`` or ``dummy=True``, ``conf_level`` is not in
            ``(0, 1)``, ``slice_chunk_size`` is not a positive integer,
            ``on_invalid`` is not one of the three policies, or the
            non-finite policy refuses the sample.
    Warns:
        JaxgsaWarning: If an output slice has zero variance. Every
            conditional distribution then equals the unconditional one, so
            the corresponding indices are an exact 0 rather than an answer.
    """
    from jaxgsa.optimal_transport import SPEC

    # The check runs on the user's own sample, before any partition layout is
    # built and before the dummy column exists. The dummy is synthetic and
    # finite by construction, so it is not part of what is checked here.
    ctx = prepare(
        SPEC,
        problem,
        Y,
        X=X,
        on_invalid=on_invalid,
        checks=(
            require(mode in _MODES, f"mode must be one of {_MODES}, got {mode!r}"),
            # Written as a positive comparison so NaN is rejected too:
            # ``epsilon > 0`` is False for NaN, while ``not epsilon <= 0``
            # was True for NaN and let it flow into log_K, where the
            # Sinkhorn loop exits at once (NaN > tol is False) and returns
            # NaN indices with zero diagnostics.
            require(epsilon > 0, f"epsilon must be > 0, got {epsilon}"),
            at_least("max_iter", max_iter, 1),
            require(tol is None or tol > 0, f"tol must be > 0, got {tol}"),
            at_least("n_bootstrap", n_bootstrap, 0),
            one_of("ci_method", ci_method, ("quantile", "gaussian")),
            in_open_interval("conf_level", conf_level, 0.0, 1.0),
            at_least("slice_chunk_size", slice_chunk_size, 1),
            # The bootstrap and the dummy baseline are the two consumers of
            # ``key``, so it fires before the data is touched, not after.
            require(key is not None or n_bootstrap == 0, "key is required when n_bootstrap > 0"),
            require(key is not None or not dummy, "key is required when dummy=True"),
        ),
        min_kept=_MIN_KEPT,
        # A constant slice leaves every conditional distribution equal to
        # the unconditional one, so the indices come out an exact 0, not
        # the NaN a variance ratio would give.
        zero_variance_outcome="zero",
    )
    assert ctx.X is not None
    X, Y, invalid = ctx.X, ctx.Y, ctx.invalid

    if mode == "trajectory" and Y.ndim != 3:
        raise ValueError(f"mode='trajectory' requires a 3-D (N, T, K) Y, got ndim={Y.ndim}")
    N = X.shape[0]
    # n_partitions applies to the continuous columns only. Categorical
    # columns always get one class per level, and so does the matched dummy
    # each of them is measured against (M4), so an all-categorical problem
    # has no consumer for the value even when it asks for a dummy floor.
    dims_levels = _categorical_dims(problem)
    cat_dims = [d for d, _ in dims_levels]
    cont_dims = [d for d in range(problem.num_vars) if d not in set(cat_dims)]
    M = _resolve_n_classes(n_partitions, N, needs_M=bool(cont_dims))
    if n_partitions is not None and not cont_dims:
        warnings.warn(
            "jaxgsa.optimal_transport: n_partitions is ignored because every parameter is "
            "categorical (one conditioning class per level, and each dummy floor "
            "matches its own parameter's level sizes)",
            stacklevel=2,
            category=JaxgsaWarning,
        )
    Y_3d = ctx.Y3
    _, T, K = Y_3d.shape

    dtype = jnp.result_type(Y_3d.dtype, jnp.float32)
    tol = _resolve_tol(tol, dtype)

    # The bootstrap and the dummy column are independent consumers of
    # randomness, so each gets its own child key rather than sharing one.
    # Both are optional, so a caller who asks for neither needs no key.
    key_boot: Array | None = None
    key_dummy: Array | None = None
    if key is not None:
        key_boot, key_dummy = jax.random.split(key)

    # Replicate 0 is the identity permutation (the original sample); the
    # remaining rows are the bootstrap resamples. Building them together
    # means the point estimate and its interval share one code path.
    identity = jnp.arange(N, dtype=jnp.int32)[None, :]
    if n_bootstrap > 0:
        assert key_boot is not None  # a missing key was refused above
        boot = jax.random.randint(key_boot, (n_bootstrap, N), 0, N, dtype=jnp.int32)
        all_idx = jnp.concatenate([identity, boot], axis=0)
    else:
        all_idx = identity

    # Build one partition layout per column group, shared with borgonovo.
    # Continuous columns share one equal-frequency rank layout, identical
    # for every replicate. Categorical columns get one class per level,
    # with sizes that vary per column and per bootstrap resample, so their
    # layout carries leading (R, Dc) axes. Grouping keeps the padded class
    # tensors rectangular without padding continuous classes up to a
    # categorical level size, or the other way round. Each group is
    # (cls_idx, counts); masks and quantile lookups are derived in-kernel
    # from the counts. build_partition_groups also returns group_levels,
    # the declared level count per group in the same order, so a caller
    # never needs to rebuild it from cont_dims/cat_dims.
    t0 = _verbose.tic()
    groups, group_levels, col_order = build_partition_groups(
        problem, X, all_idx, M, dims_levels, method="jaxgsa.optimal_transport.analyze"
    )

    # `_run` maps replicate indices and partition-layout groups to
    # (ot, advective, diffusive, degenerate) arrays with the parameter axis
    # last, in problem column order, plus a Sinkhorn convergence count (0 in
    # "univariate" mode, which needs no solver). Every caller -- the real
    # run and each dummy baseline below -- appends that count itself, so no
    # closure here needs to carry mutable state of its own.
    _Groups = list[tuple[Array, Array]]
    Y_cols = Y_3d.reshape(N, T * K) if mode == "univariate" else None
    if mode != "univariate":
        eps_s = jnp.asarray(epsilon, dtype)
        max_iter_s = jnp.asarray(max_iter, jnp.int32)
        tol_s = jnp.asarray(tol, dtype)
        clouds = _build_clouds(Y_3d, mode, standardize_outputs)

    def _run(
        idx: Array,
        run_groups: _Groups,
        run_levels: list[list[int] | None],
        order: Array | None,
    ) -> tuple[Array, Array, Array, Array, Array, int]:
        if mode == "univariate":
            assert Y_cols is not None
            ot, adv, diff, degen = _run_univariate(
                Y_cols, T, K, idx, run_groups, run_levels, order, slice_chunk_size
            )
            return ot, adv, diff, degen, jnp.zeros((), jnp.int32), 0
        return _run_joint(
            clouds, mode, idx, run_groups, run_levels, order, eps_s, max_iter_s, tol_s
        )

    n_bad_parts: list[Array] = []
    n_solves_parts: list[int] = []
    ot_all, adv_all, diff_all, degen_all, n_bad0, solves0 = _run(
        all_idx, groups, group_levels, col_order
    )
    n_bad_parts.append(n_bad0)
    n_solves_parts.append(solves0)

    ot_dummy: Array | None = None
    if dummy:
        # A synthetic, per-parameter irrelevance floor (M4). Each floor is
        # a random-membership permutation baseline whose class structure
        # matches the real parameter it stands in for, because the
        # finite-sample OT bias scales with class size, not with the
        # parameter itself. Continuous parameters share one floor, built
        # from the equal-frequency M-class layout every continuous column
        # already uses. Categorical parameters cannot share that floor: a
        # 3-level column's finite-sample bias is nothing like a 25-class
        # continuous column's, so each gets its own matched dummy solve,
        # built from that column's own observed level counts. Every dummy
        # runs through the identical estimator as a single-replicate pass,
        # because the baseline needs no bootstrap interval.
        assert key_dummy is not None  # a missing key was refused above
        dummy_keys = jax.random.split(key_dummy, 1 + len(cat_dims))

        def _dummy_value(group: PartitionGroup) -> Array:
            """Point-estimate OT of one synthetic parameter, slice-shaped."""
            ot, _, _, _, n_bad, solves = _run(identity, [group], [None], None)
            n_bad_parts.append(n_bad)
            n_solves_parts.append(solves)
            return ot[0][..., 0]

        dummy_vals: list[Array] = []
        if cont_dims:
            take_np, sizes_np = _class_layout(N, M)
            dummy_col = jax.random.permutation(dummy_keys[0], N)[:, None]
            dummy_cls_idx = _build_class_indices(dummy_col, identity, jnp.asarray(take_np))
            cont_group = (dummy_cls_idx, jnp.asarray(sizes_np)[None, None, :])  # (1, 1, M, P)
            cont_dummy = _dummy_value(cont_group)
            dummy_vals.extend([cont_dummy] * len(cont_dims))
        if cat_dims:
            # The categorical group is always last (build_partition_groups'
            # order); its counts at replicate 0 give each real column's
            # own observed, non-empty level sizes to match.
            cat_counts0 = np.asarray(groups[-1][1][0])  # (Dc, M)
            for j, key_j in enumerate(dummy_keys[1:]):
                sizes_j = cat_counts0[j]
                sizes_j = sizes_j[sizes_j > 0]
                dummy_vals.append(_dummy_value(_matched_dummy_group(key_j, N, sizes_j)))
        ot_dummy_grouped = jnp.stack(dummy_vals, axis=-1)  # (..., D) in group order
        ot_dummy = ot_dummy_grouped if col_order is None else ot_dummy_grouped[..., col_order]

    # n_solves_parts is plain Python ints (0 in "univariate" mode, which
    # needs no solver), so this check costs no host sync; only the joint
    # modes ever pay for the one host sync int(sum(n_bad_parts)) below.
    n_solves = sum(n_solves_parts)
    if n_solves:
        # One host sync for the whole analysis. The solver itself never
        # raises, because exceptions cannot cross a traced while_loop.
        n_bad = int(sum(n_bad_parts))
        if n_bad:
            warnings.warn(
                f"jaxgsa.optimal_transport: {n_bad} of {n_solves} Sinkhorn solves did not reach "
                f"tol={tol:g} within max_iter={max_iter}; results use the last "
                "iterate (consider raising max_iter or epsilon)",
                stacklevel=2,
                category=JaxgsaWarning,
            )

    # Squeeze the point estimates (and the dummy floor) to the caller's own
    # rank before building the interval, so interval()'s ci.replicates comes
    # back already at that layout and needs no second pass.
    hats_raw = {"ot": ot_all[0], "advective": adv_all[0], "diffusive": diff_all[0]}
    hats = (
        {name: ctx.squeeze(val) for name, val in hats_raw.items()}
        if mode == "univariate"
        else hats_raw
    )
    if mode == "univariate" and ot_dummy is not None:
        ot_dummy = ctx.squeeze(ot_dummy)

    confs: dict[str, Array | None] = dict.fromkeys(hats, None)
    ci_info: CIInfo | None = None
    if n_bootstrap > 0:
        raw_draws = {
            "ot": _boot_replicates(ot_all, hats_raw["ot"], degen_all),
            "advective": _boot_replicates(adv_all, hats_raw["advective"], degen_all),
            "diffusive": _boot_replicates(diff_all, hats_raw["diffusive"], degen_all),
        }
        draws = (
            {name: ctx.squeeze(d) for name, d in raw_draws.items()}
            if mode == "univariate"
            else raw_draws
        )
        endpoints, ci_info = interval(
            hats,
            draws,
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            keep_replicates=keep_replicates,
        )
        confs.update(endpoints)

    # Given-data first-order Sobol index. The advective numerator is exactly
    # Var(E[Y|X_i]) in the population (ddof=0) convention, but the OT
    # normalizer V = 2 * Var(Y) uses ddof=1. Rescaling by N / (N - 1) puts
    # both variances on ddof=0, which is jaxgsa.borgonovo's S1 convention, so
    # the identity "advective = S1 / 2" holds with no ddof caveat. Every
    # bootstrap resample also has N rows, so the same constant rescales the
    # interval exactly.
    ddof_scale = 2.0 * N / (N - 1)
    S1 = ddof_scale * hats["advective"]
    S1_conf = None if confs["advective"] is None else ddof_scale * confs["advective"]

    # The part of the total index that clears the irrelevance floor. The
    # clamp only absorbs sampling noise: an irrelevant parameter's ot
    # fluctuates around the floor, and a negative excess reads as influence.
    # ot_dummy already carries its own trailing parameter axis (M4: each
    # categorical column's floor is matched to its own class structure),
    # so no broadcast is inserted here.
    above_dummy: Array | None = None
    if ot_dummy is not None:
        above_dummy = jnp.maximum(hats["ot"] - ot_dummy, 0.0)

    result = OTResult(
        ot=hats["ot"],
        ot_conf=confs["ot"],
        advective=hats["advective"],
        advective_conf=confs["advective"],
        diffusive=hats["diffusive"],
        diffusive_conf=confs["diffusive"],
        S1=S1,
        S1_conf=S1_conf,
        above_dummy=above_dummy,
        ot_dummy=ot_dummy,
        mode=mode,
        problem=problem,
        invalid=invalid,
        ci=ci_info,
    )

    if verbose:
        elapsed = _verbose.stop(t0, result.ot)
        chunking = (
            f"slice_chunk_size: {slice_chunk_size} (user-set)"
            if slice_chunk_size is not None
            else "slice_chunk_size: auto (resolved from the memory budget)"
        )
        _verbose.analysis_summary(
            method="jaxgsa.optimal_transport.analyze",
            problem=problem,
            n_runs=N,
            T=T,
            K=K,
            invalid=invalid,
            timings=[("estimator (includes compile on the first call)", elapsed)],
            notes=[f"mode: {mode}", f"epsilon: {epsilon}", chunking],
            index_name="ot",
            values=result.ot,
            conf=result.ot_conf,
        )
    return result


def indices(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    mode: Literal["univariate", "multivariate", "trajectory"] = "univariate",
    n_partitions: int | None = None,
    standardize_outputs: bool = True,
    epsilon: float = 0.03,
    max_iter: int = 2000,
    tol: float | None = None,
    slice_chunk_size: int | None = None,
) -> tuple[Array, Array, Array]:
    """Compute optimal-transport indices as plain arrays, with no diagnostics.

    This is the transformable core of :func:`analyze`. It runs the same
    kernels on the same data and returns the same numbers, but it does
    nothing else: no non-finite check, no zero-variance warning, no
    Sinkhorn convergence warning, no ``dummy`` baseline, no
    :class:`jaxgsa.optimal_transport.OTResult`, and no read of any array
    value on the host. So it composes with ``jax.jit``, ``jax.vmap``,
    ``jax.grad`` and ``jax.jacrev``, which :func:`analyze` cannot, because
    a policy decision needs a concrete value and a tracer has none.

    Use :func:`analyze` for ordinary analysis. Nothing here checks the
    outputs, so a single NaN silently turns every index into NaN, and a
    Sinkhorn solve that never converged is reported as if it had.

    ``mode`` stays a branch here, because it is a Python string and so is
    static at trace time: one mode is traced per concrete value, exactly as
    ``analyze`` compiles one. The point-cloud modes go through the
    entropic solver, whose stopping rule is a ``while_loop`` on the
    marginal residual; that is differentiable but the derivative is of the
    iterate the loop stopped at, so tighten ``tol`` before trusting a
    gradient through them.

    There is no ``n_bootstrap``, no ``key`` and no ``dummy``. All three are
    policy: the bootstrap and the dummy baseline draw randomness and return
    a spread rather than an estimate, and both are diagnostics layered on
    the same point estimate this returns.

    Continuous parameters only. A categorical parameter's conditioning
    classes are one class per observed level, and both their number and
    their padded width are read off the data on the host, so that layout
    cannot be built from a tracer. :func:`analyze` handles categorical
    parameters.

    Tier T4 (behavioural contract): the returned arrays must equal the
    ``ot``, ``advective`` and ``diffusive`` fields of ``analyze``'s result
    on clean outputs, and the function must survive ``jit``, ``vmap`` and
    ``jit(jacrev(...))``. Checked in ``tests/test_optimal_transport.py``.

    Args:
        problem: Problem definition with D continuous parameters.
        X: Parameter sample matrix, shape ``(N, D)``.
        Y: Model output, shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        mode: Output treatment, as in :func:`analyze`.
        n_partitions: Number of equal-frequency conditioning classes.
            ``None`` selects ``min(25, N // 2)``, which reads ``N`` only, so
            it stays static under a trace.
        standardize_outputs: Joint modes only. Divide each output column by its
            standard deviation before building the transport cost, as in
            :func:`analyze`. It is arithmetic over the sample axis, not
            policy, so it stays traceable.
        epsilon: Joint modes only. Entropic regularization strength,
            relative to ``V``, as in :func:`analyze`.
        max_iter: Joint modes only. Sinkhorn iteration cap per solve.
        tol: Joint modes only. Marginal stopping tolerance. ``None``
            selects ``1e-9`` in float64 and ``1e-6`` in float32. In
            float32 the residual can floor a little above ``1e-6`` for a
            large cloud without the cost being wrong, as in
            :func:`analyze`.
        slice_chunk_size: ``"univariate"`` mode only. Output columns per
            kernel call, as in :func:`analyze`.

    Returns:
        ``(ot, advective, diffusive)``. In ``"univariate"`` mode the shapes
        are those of the caller's ``Y``: ``(D,)`` for a 1-D ``Y``,
        ``(K, D)`` for 2-D, ``(T, K, D)`` for 3-D. ``"multivariate"``
        returns ``(D,)`` and ``"trajectory"`` returns ``(K, D)``. The shapes
        are those ``analyze`` reports.

    Raises:
        ValueError: If ``X`` is not ``(N, D)`` for the problem's ``D``; if
            ``Y`` is not 1-D, 2-D or 3-D, or its row count does not match
            ``X``; if ``mode`` is unknown; if ``mode="trajectory"`` is used
            with a non-3-D ``Y``; if ``n_partitions`` is not in
            ``[2, N // 2]``; if ``epsilon <= 0``, ``max_iter < 1`` or
            ``tol <= 0``; if ``slice_chunk_size`` is below 1; or if any
            parameter is categorical.
    """
    check_scalars(
        (
            require(mode in _MODES, f"mode must be one of {_MODES}, got {mode!r}"),
            # Written as a positive comparison so NaN is rejected too:
            # ``epsilon > 0`` is False for NaN, while ``not epsilon <= 0``
            # was True for NaN and let it flow into log_K, where the
            # Sinkhorn loop exits at once (NaN > tol is False) and returns
            # NaN indices with zero diagnostics.
            require(epsilon > 0, f"epsilon must be > 0, got {epsilon}"),
            at_least("max_iter", max_iter, 1),
            require(tol is None or tol > 0, f"tol must be > 0, got {tol}"),
            at_least("slice_chunk_size", slice_chunk_size, 1),
        )
    )
    X = validate_inputs(problem, X)
    Y = _validate_output(Y, int(X.shape[0]), problem)
    if mode == "trajectory" and Y.ndim != 3:
        raise ValueError(f"mode='trajectory' requires a 3-D (N, T, K) Y, got ndim={Y.ndim}")
    dims_levels = _categorical_dims(problem)
    if dims_levels:
        names = ", ".join(repr(problem.names[d]) for d, _ in dims_levels)
        raise ValueError(
            "jaxgsa.optimal_transport.indices supports continuous parameters "
            f"only, but {names} is categorical. A categorical parameter "
            "conditions on one class per observed level, and both the class "
            "count and the padded class width are read off the sample on the "
            "host, so the layout cannot be built from a tracer. Use "
            "jaxgsa.optimal_transport.analyze instead; it supports "
            "categorical parameters and is not traceable for this reason."
        )

    N = X.shape[0]
    M = _resolve_n_classes(n_partitions, N, needs_M=True)

    Y_3d, layout = _prepare_Y(Y)
    _, T, K = Y_3d.shape
    dtype = jnp.result_type(Y_3d.dtype, jnp.float32)

    # One replicate, the identity permutation: the original sample. The
    # bootstrap axis exists only for the interval, which is policy.
    all_idx = jnp.arange(N, dtype=jnp.int32)[None, :]
    # Every parameter is continuous here (categorical is refused above), so
    # this always returns exactly one group carrying no level list.
    groups, group_levels, col_order = build_partition_groups(
        problem, X, all_idx, M, dims_levels, method="jaxgsa.optimal_transport.indices"
    )

    if mode == "univariate":
        ot, adv, diff, _ = _run_univariate(
            Y_3d.reshape(N, T * K),
            T,
            K,
            all_idx,
            groups,
            group_levels,
            col_order,
            slice_chunk_size,
        )
        return layout.squeeze(ot[0]), layout.squeeze(adv[0]), layout.squeeze(diff[0])

    tol_v = _resolve_tol(tol, dtype)
    ot, adv, diff, _, _, _ = _run_joint(
        _build_clouds(Y_3d, mode, standardize_outputs),
        mode,
        all_idx,
        groups,
        group_levels,
        col_order,
        jnp.asarray(epsilon, dtype),
        jnp.asarray(max_iter, jnp.int32),
        jnp.asarray(tol_v, dtype),
    )
    return ot[0], adv[0], diff[0]
