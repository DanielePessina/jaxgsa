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
  through its nearest-rank empirical quantile function (midpoint rule
  ``j = floor((i + 0.5) * n_m / N)``, exact whenever the class size
  divides N).
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
    Borgonovo, Figalli, Plischke & Savare (2024). Global sensitivity
    analysis via optimal transport. Management Science.
    doi:10.1287/mnsc.2023.01796.
"""

from __future__ import annotations

import warnings
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from jaxgsa._core.bootstrap import _percentile_ci
from jaxgsa._core.invalid import (
    InvalidUnit,
    OnInvalid,
    check_invalid,
    resolve_policy,
)
from jaxgsa._core.partition import (
    _build_class_indices,
    _class_layout,
    _mask_from_counts,
    _replicate_slice,
    build_partition_groups,
)
from jaxgsa._core.result import CIInfo
from jaxgsa._core.validation import (
    _prenormalize_outputs,
    _prepare_Y,
    _squeeze_output_axes,
    _validate_xy_inputs,
)
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.optimal_transport._result import OTResult
from jaxgsa.optimal_transport._solver import _sinkhorn_w2
from jaxgsa.problem import Problem, _categorical_dims

# Target element budget for the default per-chunk working set of the
# "univariate" mode. The dominant intermediate is the per-column conditional
# quantile tensor of size ``chunk_columns * D * M * N`` elements, so the
# default chunk width keeps it near this many float32 elements (~256 MB).
_CHUNK_ELEM_BUDGET = 1 << 26

_MODES = ("univariate", "multivariate", "trajectory")

# Fewest samples that still build two conditioning classes of two points
# each, which is what the default ``M = min(25, N // 2)`` already assumes.
_MIN_KEPT = 4


def _normalize_by_2var(weighted_sum: Array, V: Array) -> Array:
    """Normalize class-weighted cost sums by ``V = 2 * Var``.

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
    """
    V_safe = jnp.where(V > 0, V, 1.0)
    normalized = jnp.where(V > 0, weighted_sum / V_safe, 0.0)
    return jnp.where(jnp.isnan(V), jnp.nan, normalized)


def _aggregate_normalized(per_class: Array, weights: Array, V: Array) -> Array:
    """Class-weighted average of per-class costs, normalized by ``V``.

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
    return _normalize_by_2var((weights * per_class).sum(axis=-1), V)


def _quantile_rank_indices(counts: Array, N: int) -> Array:
    """Nearest-rank conditional quantile index for every full-sample mass point.

    Applies the midpoint rule ``j = min(floor((i + 0.5) * c / N), c - 1)``
    for every mass point ``i`` and class size ``c``. ``j < c`` always, so
    the lookup never touches class padding. Zero-size classes (empty
    categorical levels) get index 0, and the kernels discard those lookups
    through the zero class weight.

    The function runs on device, per replicate, from the class counts. A
    precomputed ``(R, Dc, M, N)`` lookup table would be gigabytes for one
    high-cardinality column at large N, while this transient is
    ``(G, M, N)`` per replicate.

    The products ``(2i + 1) * c`` overflow int32 and exceed float32's
    exact-integer range, and float64 is unavailable without the x64 flag.
    The quotient is therefore formed exactly by schoolbook long division
    over the base-``S`` digits of ``c``. ``S`` is a static power of two
    chosen so every intermediate stays below ``2**31``. The digit count is
    static, so the Python loop unrolls at trace time. The result is exact
    for any ``N < 2**29`` and bit-identical to a float64 host evaluation
    (tested).

    Args:
        counts: Integer class sizes, shape ``(..., M)``.
        N: Number of samples.

    Returns:
        An int32 lookup index into each class's sorted members, shape
        ``(..., M, N)``.
    """
    # q = floor(a * c / m) with a = 2i + 1 (odd, < 2N) and m = 2N.
    a = 2 * jnp.arange(N, dtype=jnp.int32) + 1  # (N,)
    m = 2 * N
    c = counts.astype(jnp.int32)[..., None]  # (..., M, 1)
    shift = max(1, 29 - N.bit_length())  # S = 2**shift keeps 4*N*S <= 2**31
    S = 1 << shift
    n_digits = -(-max(1, N.bit_length()) // shift)  # ceil; c <= N
    q = jnp.zeros_like(c * a)  # broadcast to (..., M, N)
    r = jnp.zeros_like(q)
    for k in reversed(range(n_digits)):
        digit = (c >> (k * shift)) & (S - 1)
        t = r * S + a * digit  # < 4*N*S <= 2**31
        q = q * S + t // m
        r = t % m
    return jnp.minimum(q, jnp.maximum(c - 1, 0)).astype(jnp.int32)


@jax.jit
def _ot_1d_kernel(
    Y_cols: Array,
    all_idx: Array,
    groups: tuple[tuple[Array, Array], ...],
) -> tuple[Array, Array, Array, Array]:
    """Per-column 1-D optimal-transport indices for every replicate.

    Uses the closed-form 1-D coupling. Both empirical quantile functions
    are evaluated on the N uniform mass points of the full sample. The
    conditional one uses the nearest-rank (midpoint rule) lookup into the
    class's sorted members. No transport solver is involved.

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

    Returns:
        ``(ot, advective, diffusive, degenerate)``. The three index arrays
        have shape ``(R, C, D)``. ``degenerate`` has shape ``(R, C)`` and
        flags constant (zero-variance) replicate columns, which yield zero
        indices.
    """
    dtype = jnp.result_type(Y_cols.dtype, jnp.float32)
    Y_cols = Y_cols.astype(dtype)
    N = Y_cols.shape[0]
    counts_list = tuple(counts for _, counts in groups)
    cls_list = tuple(cls_idx for cls_idx, _ in groups)

    def _group_stats(y, y_sorted, mean_r, V, cls_idx, mask_b, counts_b, j_b):
        """Weighted, normalized cost sums for one group's parameters.

        ``mask_b (G, M, P)``, ``counts_b (G, M)``, and ``j_b (G, M, N)``
        broadcast against the group's parameter axis Dg. ``G`` is 1 or Dg.
        """
        weights = counts_b / N  # (G, M)
        safe_counts = jnp.maximum(counts_b, 1.0)
        valid = counts_b > 0

        y_cls = y[cls_idx]  # (Dg, M, P) resampled class members
        # Pads sort to the tail as +inf and are unreachable through j.
        y_cls_sorted = jnp.sort(jnp.where(mask_b, y_cls, jnp.inf), axis=-1)
        j_full = jnp.broadcast_to(j_b, (y_cls.shape[0],) + j_b.shape[-2:])
        q = jnp.take_along_axis(y_cls_sorted, j_full, axis=-1)  # (Dg, M, N)
        w2 = ((y_sorted[None, None, :] - q) ** 2).mean(axis=-1)  # (Dg, M)

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
        y_sorted = jnp.sort(y_r)
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

    def _one_replicate(carry, xs):
        i, r, cls_parts = xs
        layouts = []
        for cls_r, counts_g in zip(cls_parts, counts_list):
            counts_r = _replicate_slice(counts_g, i)  # (G, M) int
            mask_r = _mask_from_counts(counts_r, cls_r.shape[-1])
            j_r = _quantile_rank_indices(counts_r, N)  # (G, M, N)
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
    per parameter.

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
    N = Z.shape[0]
    counts_list = tuple(counts for _, counts in groups)
    cls_list = tuple(cls_idx for cls_idx, _ in groups)

    def _group_stats(Z_r, mean_all, sq_full, V, cls_idx, counts_r, dm_grid):
        """Normalized cost sums and failure count for one group's parameters.

        ``counts_r (G, M)`` broadcasts against the group's parameter axis
        Dg. ``G`` is 1 or Dg.
        """
        Dg, M, P = cls_idx.shape[0], cls_idx.shape[1], cls_idx.shape[2]
        G = counts_r.shape[0]
        mask_r = _mask_from_counts(counts_r, P)  # (G, M, P)
        safe_counts = jnp.maximum(counts_r, 1.0)
        log_b_all = jnp.where(mask_r, -jnp.log(safe_counts)[..., None], -jnp.inf)  # (G, M, P)

        def _one_class(dm: Array):
            """Transport cost of the full cloud onto class dm % M of parameter dm // M."""
            m = dm % M
            g = jnp.minimum(dm // M, G - 1)  # broadcastable layout axis
            idx = cls_idx[dm // M, m]  # (P,)
            mask_dm = mask_r[g, m].astype(dtype)  # (P,)
            count_dm = counts_r[g, m]
            Z_c = Z[idx]  # (P, E)
            # Squared Euclidean cost block (N, P). Padded columns carry
            # zero target mass, so zeroing their costs is exact. It is also
            # necessary: pads are clamped duplicates of a real sample, and
            # an outlier there would otherwise set the solver's max-cost
            # scale and change the effective regularization per class.
            C = sq_full[:, None] + (Z_c**2).sum(axis=-1)[None, :] - 2.0 * (Z_r @ Z_c.T)
            C = jnp.maximum(C, 0.0) * mask_dm[None, :]
            cost, err = _sinkhorn_w2(C, log_b_all[g, m], epsilon, max_iter, tol)
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
                _replicate_slice(counts_g, i).astype(dtype),
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
    standardize: bool = True,
    epsilon: float = 0.01,
    max_iter: int = 1000,
    tol: float | None = None,
    dummy: bool = False,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    keep_replicates: bool = False,
    seed: int = 0,
    slice_chunk_size: int | None = None,
    on_invalid: OnInvalid = "raise",
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
            and always use one class per level. A passed value is always
            validated against ``[2, N // 2]``. If every parameter is
            categorical and ``dummy`` is false, nothing uses the value and
            a ``JaxgsaWarning`` says it is ignored.
        standardize: Joint modes only. Divide each output column by its
            standard deviation before building the transport cost, so no
            single output dominates the joint distance through its units.
            Ignored in ``"univariate"`` mode, where each column is
            normalized by its own variance regardless.
        epsilon: Joint modes only. Entropic regularization strength,
            relative to the cost matrix scaled to [0, 1]. Smaller values
            approach exact transport at the price of more iterations.
        max_iter: Joint modes only. Sinkhorn iteration cap per solve.
        tol: Joint modes only. Stopping tolerance on the L1 target-
            marginal violation. ``None`` selects ``1e-9`` in float64 and
            ``1e-6`` in float32, where a tighter value is unresolvable.
            One warning is emitted if any solve fails to converge.
        dummy: Also push one synthetic parameter through the identical
            pipeline and report its index as ``ot_dummy``. The synthetic
            parameter is independent of the output by construction. Its
            index estimates the floor a fully irrelevant parameter
            receives from finite-sample bias and, in the point-cloud
            modes, from entropic bias. Parameters not clearly above that
            floor are indistinguishable from noise.
        n_bootstrap: Number of bootstrap resamples for confidence
            intervals. ``0`` (default) skips them, and the ``*_conf``
            fields are ``None``. Joint modes solve
            ``n_bootstrap * D * n_partitions`` transport problems, so keep
            the value modest there.
        conf_level: Confidence level for percentile bootstrap intervals.
        keep_replicates: Keep the per-resample indices on
            ``OTResult.ci.replicates``. Off by default because they are
            large: ``n_bootstrap`` copies of all three index arrays. Turn it
            on to recompute an interval at another level without re-running
            the analysis, which for this method means without re-solving
            every transport problem.
        seed: Random seed for bootstrap resampling and the dummy
            parameter.
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

    Returns:
        An :class:`OTResult` with the total, advective and diffusive
        indices, optional confidence intervals, the optional dummy
        baseline, and the non-finite report in ``invalid``. Every index is
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
            ``tol <= 0``, ``n_bootstrap < 0``, ``conf_level`` is not in
            ``(0, 1)``, ``slice_chunk_size`` is not a positive integer,
            ``on_invalid`` is not one of the three policies, or the
            non-finite policy refuses the sample.
    """
    X = jnp.asarray(X)
    # The OT index measures total, correlation-inclusive influence through
    # rank-based conditioning, so correlated problems are accepted.
    Y = _validate_xy_inputs(
        problem,
        X,
        Y,
        correlation_ok=True,
        categorical_ok=True,
        method="jaxgsa.optimal_transport.analyze",
    )
    # The user's own sample, before any partition layout is built and before
    # the dummy column exists. The dummy is synthetic and finite by
    # construction, so it is not part of what is checked here.
    method = "jaxgsa.optimal_transport.analyze"
    policy = resolve_policy(on_invalid, method=method, unit=InvalidUnit.ROW)
    keep, invalid = check_invalid(
        policy=policy,
        method=method,
        unit=InvalidUnit.ROW,
        n_units=int(X.shape[0]),
        X=X,
        Y=Y,
        min_kept=_MIN_KEPT,
    )
    if not keep.all():
        X = X[keep]
        Y = Y[keep]

    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if mode == "trajectory" and Y.ndim != 3:
        raise ValueError(f"mode='trajectory' requires a 3-D (N, T, K) Y, got ndim={Y.ndim}")
    N = X.shape[0]
    # n_partitions applies to the continuous columns and the dummy
    # parameter. Categorical columns always get one class per level.
    dims_levels = _categorical_dims(problem)
    cat_dims = [d for d, _ in dims_levels]
    cont_dims = [d for d in range(problem.num_vars) if d not in set(cat_dims)]
    if n_partitions is None:
        # The customary 25 classes, clamped so small samples do not raise
        # over a default the user never passed.
        M = min(25, N // 2)
        if (cont_dims or dummy) and M < 2:
            raise ValueError(f"building conditioning classes needs N >= 4 samples, got N={N}")
    else:
        M = int(n_partitions)
        if not 2 <= M <= N // 2:
            # Name the dummy baseline when it is the only consumer, so the
            # error does not read as a complaint about categorical columns.
            scope = (
                " for the dummy baseline (categorical columns use one class per level)"
                if dummy and not cont_dims
                else ""
            )
            raise ValueError(
                f"n_partitions must be in [2, N//2={N // 2}]{scope}, got {n_partitions}"
            )
        if not cont_dims and not dummy:
            warnings.warn(
                "jaxgsa: n_partitions is ignored because every parameter is "
                "categorical (one conditioning class per level) and no dummy "
                "baseline was requested",
                stacklevel=2,
                category=JaxgsaWarning,
            )
    if not epsilon > 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol is not None and not tol > 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be >= 0, got {n_bootstrap}")
    if not 0 < conf_level < 1:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")
    if slice_chunk_size is not None and slice_chunk_size < 1:
        raise ValueError(f"slice_chunk_size must be >= 1, got {slice_chunk_size}")

    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)
    _, T, K = Y_3d.shape

    dtype = jnp.result_type(Y_3d.dtype, jnp.float32)
    if tol is None:
        tol = 1e-9 if dtype == jnp.float64 else 1e-6

    key_boot, key_dummy = jax.random.split(jax.random.PRNGKey(seed))

    # Replicate 0 is the identity permutation (the original sample); the
    # remaining rows are the bootstrap resamples. Building them together
    # means the point estimate and its interval share one code path.
    identity = jnp.arange(N, dtype=jnp.int32)[None, :]
    if n_bootstrap > 0:
        boot = jax.random.randint(key_boot, (n_bootstrap, N), 0, N, dtype=jnp.int32)
        all_idx = jnp.concatenate([identity, boot], axis=0)
    else:
        all_idx = identity

    # Build one partition layout per column group. Continuous columns share
    # one equal-frequency rank layout, identical for every replicate.
    # Categorical columns get one class per level, with sizes that vary per
    # column and per bootstrap resample, so their layout carries leading
    # (R, Dc) axes. Grouping keeps the padded class tensors rectangular
    # without padding continuous classes up to a categorical level size, or
    # the other way round.
    # Canonical partition-group layout, shared with borgonovo. Each group
    # is (cls_idx, counts); masks and quantile lookups are derived
    # in-kernel from the counts.
    groups, _, col_order = build_partition_groups(problem, X, all_idx, M, dims_levels)

    def _static_dm_grid(group: tuple[Array, Array], levels: list[int] | None) -> Array:
        """Flat ``d * Mg + m`` class slots the joint kernel must solve.

        A categorical group's class axis is padded to the largest level
        count. The pad slots are statically empty, because their counts
        are zero for every replicate. They are excluded here instead of
        running dead Sinkhorn solves.
        """
        D_g, M_g = group[0].shape[1], group[0].shape[2]
        if levels is None:
            return jnp.arange(D_g * M_g, dtype=jnp.int32)
        return jnp.asarray(
            [d * M_g + m for d, n_levels in enumerate(levels) for m in range(n_levels)],
            dtype=jnp.int32,
        )

    # Group order mirrors build_partition_groups: continuous (no level
    # list) first when present, then categorical.
    group_levels: list[list[int] | None] = []
    if cont_dims:
        group_levels.append(None)
    if cat_dims:
        group_levels.append([n_levels for _, n_levels in dims_levels])

    # `_run` maps replicate indices and partition-layout groups to
    # (ot, advective, diffusive, degenerate) arrays with the parameter axis
    # last, in problem column order. It also accumulates the Sinkhorn
    # convergence stats for the one end-of-analysis warning. Defining it
    # per mode lets the dummy baseline below reuse the identical estimator.
    n_bad_parts: list[Array] = []
    n_solves = 0
    _Groups = list[tuple[Array, Array]]

    if mode == "univariate":
        Y_cols = Y_3d.reshape(N, T * K)
        total = T * K

        def _run(
            idx: Array,
            run_groups: _Groups,
            run_levels: list[list[int] | None],
            order: Array | None,
        ) -> tuple[Array, Array, Array, Array]:
            R_run = idx.shape[0]
            D_run = sum(g[0].shape[1] for g in run_groups)
            # Peak memory scales with the summed per-group D_g * M_g
            # conditional-quantile tensors.
            layout_elems = sum(g[0].shape[1] * g[0].shape[2] for g in run_groups)
            cs = slice_chunk_size
            if cs is None:
                cs = max(1, _CHUNK_ELEM_BUDGET // (layout_elems * N))
            cs = min(cs, total)
            parts: tuple[list, list, list, list] = ([], [], [], [])
            for start in range(0, total, cs):
                chunk = Y_cols[:, start : start + cs]
                merged = list(_ot_1d_kernel(chunk, idx, tuple(run_groups)))
                if order is not None:
                    merged[:3] = [arr[..., order] for arr in merged[:3]]
                for part, arr in zip(parts, merged):
                    part.append(arr)
            ot, adv, diff, degen = (jnp.concatenate(p, axis=1) for p in parts)
            return (
                ot.reshape(R_run, T, K, D_run),
                adv.reshape(R_run, T, K, D_run),
                diff.reshape(R_run, T, K, D_run),
                degen.reshape(R_run, T, K),
            )
    else:
        eps_s = jnp.asarray(epsilon, dtype)
        max_iter_s = jnp.asarray(max_iter, jnp.int32)
        tol_s = jnp.asarray(tol, dtype)
        # One point cloud for "multivariate" (flattened output), one per
        # output for "trajectory". The runner stacks the per-cloud results
        # on a new output axis only in the "trajectory" case.
        if mode == "multivariate":
            clouds = [Y_3d.reshape(N, T * K)]
        else:
            clouds = [Y_3d[:, :, k] for k in range(K)]
        if standardize:
            clouds = [_prenormalize_outputs(Z)[0] for Z in clouds]

        def _run(
            idx: Array,
            run_groups: _Groups,
            run_levels: list[list[int] | None],
            order: Array | None,
        ) -> tuple[Array, Array, Array, Array]:
            nonlocal n_solves
            dm_grids = tuple(
                _static_dm_grid(group, levels) for group, levels in zip(run_groups, run_levels)
            )
            n_solves += len(clouds) * idx.shape[0] * sum(grid.shape[0] for grid in dm_grids)

            def _one_cloud(Z: Array) -> tuple[Array, Array, Array, Array, Array]:
                ot, adv, diff, n_bad, degen = _joint_kernel(
                    Z, idx, tuple(run_groups), dm_grids, eps_s, max_iter_s, tol_s
                )
                merged = [ot, adv, diff]
                if order is not None:
                    merged = [arr[..., order] for arr in merged]
                return merged[0], merged[1], merged[2], n_bad.sum(), degen

            cloud_outs = [_one_cloud(Z) for Z in clouds]
            n_bad_parts.append(jnp.stack([o[3] for o in cloud_outs]).sum())
            if mode == "multivariate":
                ot, adv, diff, _, degen = cloud_outs[0]  # (R, D) / (R,)
                return ot, adv, diff, degen
            return (
                jnp.stack([o[0] for o in cloud_outs], axis=1),  # (R, K, D)
                jnp.stack([o[1] for o in cloud_outs], axis=1),
                jnp.stack([o[2] for o in cloud_outs], axis=1),
                jnp.stack([o[4] for o in cloud_outs], axis=1),  # (R, K)
            )

    ot_all, adv_all, diff_all, degen_all = _run(all_idx, groups, group_levels, col_order)

    ot_dummy: Array | None = None
    if dummy:
        # A synthetic parameter that is independent of Y by construction.
        # Only its ranks matter, so a permutation of 0..N-1 is enough. It
        # runs through the identical estimator as a single-replicate pass,
        # because the baseline needs no bootstrap interval. The dummy is
        # continuous, so it always uses the shared equal-frequency layout.
        take_np, sizes_np = _class_layout(N, M)
        dummy_col = jax.random.permutation(key_dummy, N)[:, None]
        dummy_cls_idx = _build_class_indices(dummy_col, identity, jnp.asarray(take_np))
        dummy_group = (dummy_cls_idx, jnp.asarray(sizes_np)[None, None, :])  # (1, 1, M, P)
        ot_dummy = _run(identity, [dummy_group], [None], None)[0][0][..., 0]

    if n_bad_parts:
        # One host sync for the whole analysis. The solver itself never
        # raises, because exceptions cannot cross a traced while_loop.
        n_bad = int(sum(n_bad_parts))
        if n_bad:
            warnings.warn(
                f"jaxgsa: {n_bad} of {n_solves} Sinkhorn solves did not reach "
                f"tol={tol:g} within max_iter={max_iter}; results use the last "
                "iterate (consider raising max_iter or epsilon)",
                stacklevel=2,
                category=JaxgsaWarning,
            )

    hats = {"ot": ot_all[0], "advective": adv_all[0], "diffusive": diff_all[0]}
    confs: dict[str, Array | None] = dict.fromkeys(hats, None)
    ci_info: CIInfo | None = None
    if n_bootstrap > 0:
        replicates: dict[str, Array] | None = {} if keep_replicates else None
        for name, vals in (("ot", ot_all), ("advective", adv_all), ("diffusive", diff_all)):
            draws = _boot_replicates(vals, hats[name], degen_all)
            endpoints = _percentile_ci(draws, conf_level)
            if mode == "univariate":
                endpoints = _squeeze_output_axes(endpoints, squeeze_time, squeeze_output)
            confs[name] = endpoints
            if replicates is not None:
                # The leading resample axis survives the squeeze, which
                # addresses the inserted T/K axes from the end.
                if mode == "univariate":
                    draws = _squeeze_output_axes(draws, squeeze_time, squeeze_output)
                replicates[name] = draws
        # This method has one hard-wired endpoint rule, the percentile
        # interval, so "quantile" is what ran and not a guess.
        ci_info = CIInfo(
            level=conf_level,
            method="quantile",
            n_resamples=n_bootstrap,
            replicates=replicates,
        )

    if mode == "univariate":
        hats = {
            name: _squeeze_output_axes(val, squeeze_time, squeeze_output)
            for name, val in hats.items()
        }
        if ot_dummy is not None:
            ot_dummy = _squeeze_output_axes(ot_dummy, squeeze_time, squeeze_output, n_trailing=0)

    return OTResult(
        ot=hats["ot"],
        ot_conf=confs["ot"],
        advective=hats["advective"],
        advective_conf=confs["advective"],
        diffusive=hats["diffusive"],
        diffusive_conf=confs["diffusive"],
        ot_dummy=ot_dummy,
        mode=mode,
        problem=problem,
        invalid=invalid,
        ci=ci_info,
    )
