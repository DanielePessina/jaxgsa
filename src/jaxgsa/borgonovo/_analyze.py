"""Borgonovo delta analysis: moment-independent sensitivity from given data.

This module implements the Plischke, Borgonovo & Smith (2013) given-data
estimator of Borgonovo's (2007) delta index. For each parameter the estimator
splits the sample into equal-frequency classes by the parameter's rank. The
delta index is then the class-weighted L1 distance between the unconditional
output density and each conditional density. Both densities come from a
Gaussian kernel density estimate (KDE) with Silverman bandwidths, evaluated on
a fixed output grid and integrated by the trapezoid rule. The same partition
also yields a given-data first-order Sobol index at negligible cost.

The estimator supports a continuous output distribution only, because a KDE
on a fixed grid cannot represent an atom. ``analyze`` refuses a discrete
output up front and points the caller at ``jaxgsa.optimal_transport``.

The plug-in estimator is biased upward at finite N. By default the central
estimate is therefore bias-corrected with bootstrap resamples, as
``2*d_hat - mean(d_boot)`` (Plischke et al. eqn 30), where ``d_hat`` comes
from the original sample. The same replicates give the percentile confidence
intervals. The original sample and the bootstrap replicates run through one
scanned path, with the original sample as replicate 0 gathered by the identity
permutation. The point estimate and its interval are therefore always computed
under identical conventions.

Two things are computed once and reused. The class-partition indices are built
per parameter for every replicate, then reused across output-column chunks.
The JIT-compiled per-column kernel is cached only on the scalar estimator
settings, that is the grid size, the bandwidth rule, the degenerate-class
floor and the grid tile width, so it captures no sample-sized constants.

The estimator details mirror ``SALib.analyze.delta``: an equal-frequency
ordinal rank partition, the Plischke class-count heuristic, Silverman KDE
factors, and a 100-point output grid. Three differences are deliberate.

1. The central estimate uses the original sample rather than a bootstrap
   resample, so it is deterministic given the data.
2. A constant output column yields ``delta = S1 = 0`` instead of an error.
3. A bootstrap replicate that happens to be constant contributes the point
   estimate rather than a spurious zero, so it neither adds nor removes bias.
   Such a replicate is reachable for rare-event outputs, and SALib raises
   ``LinAlgError`` on such data.

References:
    Borgonovo (2007). A new uncertainty importance measure.
    Reliability Engineering & System Safety 92(6):771-784.

    Plischke, Borgonovo & Smith (2013). Global sensitivity measures from
    given data. European Journal of Operational Research 226(3):536-550.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.batching import resolve_batch_size
from jaxgsa._core.bootstrap import _bootstrap_ci_endpoints
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
    _mask_from_counts,
    _replicate_slice,
    build_partition_groups,
)
from jaxgsa._core.result import CIInfo
from jaxgsa._core.validation import _prepare_Y, _validate_output
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.borgonovo._result import DeltaResult
from jaxgsa.problem import Problem, _categorical_dims

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_MAX_CLASSES = 48
# One-time-per-process flag for the bias_correct=None resolution warning. The
# default is a tri-state, and a silent resolution to True surprised users, so
# the first default call that applies the correction says so once. An explicit
# True or False never warns, and never sets this.
_warned_bias_correct_default = False
# A class whose KDE bandwidth falls below this fraction of the full-sample
# bandwidth counts as degenerate. Two failures live below this line. The
# first is an exactly zero variance, that is a point mass, such as one
# categorical level that maps to one output value. The second is a spread so
# small that the shared output grid cannot resolve the conditional density.
# The second failure is the dangerous one. The grid step is
# (y_max - y_min) / (grid_size - 1). A class whose bandwidth is far below
# that step is sampled at a spacing of many sigma. The trapezoid rule then
# either misses the peak, or lands on it and integrates a spike of height
# ~1/(h*sqrt(2*pi)) over a step-wide interval. Either way the L1 distance,
# and with it delta, explodes far above 1. At the default grid_size = 100 the
# step is about 1/99 of the output range. A Silverman bandwidth of 1e-2 of
# the full-sample bandwidth is already at that scale, so 1e-2 is the
# threshold below which the grid is untrustworthy.
_DEGENERATE_BW_TOL = 1e-2
# Bandwidth given to a degenerate class, as a fraction of the full-sample
# Silverman bandwidth. The applied floor is ``max(fraction * h_full,
# grid_step)``. In the degenerate regime the grid-step term is almost always
# the larger of the two, so the grid sets the width. This fraction only
# binds for a wide output range with a fine grid. The grid-step bound is the
# load-bearing part. A Gaussian sampled at a spacing of at most its own
# sigma has small trapezoid error, while one sampled far more coarsely
# aliases and drives delta above 1. One consequence is that the delta of a
# near-degenerate class depends on the grid resolution and is biased low.
# The knob that moves it is ``grid_size``, not this fraction. See the
# ``analyze`` docstring.
_DEGENERATE_BW_FRACTION = 0.1
# Borgonovo's delta lies in [0, 1] by construction. The default
# bias-corrected estimate can leave that range by a little at small N, so
# only an excursion wider than this counts as an estimator failure.
_DELTA_RANGE_TOL = 0.05
# A discrete output breaks the estimator. The KDE of an atomic density is a
# spike the output grid cannot resolve, so delta is meaningless. An output
# column counts as discrete when it takes at most this many distinct values
# and each value repeats at least _DISCRETE_MIN_MULTIPLICITY times on
# average, that is ``n_distinct * _DISCRETE_MIN_MULTIPLICITY <= N``. Both
# conditions must hold, so the guard does not refuse a continuous output
# rounded to a few decimals, which has many distinct values at any useful N.
# The multiplicity term is the discriminator: a continuous float output
# collides only through rounding, and even coarse rounding keeps the average
# multiplicity near 1, while an atom must repeat to be an atom. Five repeats
# per value on average cannot come from continuity. An earlier version used
# a fraction test, ``n_distinct < 0.01 * N``, which is vacuous at small N:
# a binary (0/1) output passed at N <= 200 (2 < 0.01*150 is false) and a
# 10-level output passed up to N = 1000, so analyze returned a delta that
# describes grid_size rather than the model. The multiplicity rule refuses
# a binary output at N = 150 (2*5 <= 150) and a 10-level output at N = 500
# (10*5 <= 500), still accepts a continuous output at N = 100 (~100
# distinct values, above the cap), and is a strict superset of the old
# refusals (d < 0.01*N implies 5*d < 0.05*N <= N), so no previously refused
# case is now accepted and the large-N behavior is identical. A column
# with a single distinct value is exempt. A constant output needs no
# density, every conditional equals the unconditional, and delta = S1 = 0 is
# the exact answer rather than a failed computation. That contract predates
# this guard and stays.
_DISCRETE_MAX_DISTINCT = 20
_DISCRETE_MIN_MULTIPLICITY = 5
# Live copies of the conditional-KDE tensor to budget for. The tensor is
# built, exponentiated, masked and then reduced, and the XLA CPU runtime was
# measured holding roughly three of those stages resident at once. The slice
# chunk is sized against this multiple so the peak stays inside the
# transient-memory budget rather than three times over it.
_KDE_LIVE_TENSORS = 3
# Target size of one conditional-KDE grid tile, in bytes. This is a working
# set, not a cap: the tile only has to be wide enough to keep the CPU busy,
# and making it wider buys nothing but resident pages. 16 MiB was measured as
# the point where the curve flattens; below it the per-tile dispatch starts to
# show, above it the peak grows with no gain in speed. The real cap is the
# transient-memory budget, which the slice chunk is sized against.
_KDE_TILE_TARGET_BYTES = 16 * 1024**2
# Narrowest conditional-KDE grid tile the sizing rule will pick. Tiling never
# reorders the mathematics: each grid point keeps its own sum over its own
# class members. XLA is still free to reassociate that sum differently for a
# different tile shape, and at a tile of exactly one grid point it does,
# because there is no longer a grid axis to vectorize over; the result then
# moves in the last few bits (measured at 3e-8). Every wider tile returns the
# untiled result exactly, which the test suite pins, so the rule stops at two.
_MIN_GRID_TILE = 2


# Fewest samples that still build two conditioning classes with a density
# each. Below this there is nothing to compare.
_MIN_KEPT = 4


def _plischke_n_classes(n_samples: int) -> int:
    """Sample-size heuristic for the number of conditioning classes.

    Identical to SALib's rule (Plischke et al. 2013): roughly ``N**(2/7)``
    classes, saturating at 48 for large samples.

    Args:
        n_samples: Number of samples N.

    Returns:
        Number of equal-frequency classes M.
    """
    exponent = 2.0 / (7.0 + np.tanh((1500.0 - n_samples) / 500.0))
    return min(int(np.ceil(n_samples**exponent)), _MAX_CLASSES)


def _kde_bandwidths(counts: Array, std: Array, bw_factor: float | None) -> Array:
    """Per-class (or full-sample) KDE bandwidths.

    This is the single definition of the bandwidth rule. The jitted kernel
    and the host-side failure diagnostic both call it, so the two cannot
    drift apart.

    Args:
        counts: Number of samples the bandwidth is computed from. It is the
            per-class count, or ``N`` for the full-sample bandwidth.
        std: Sample standard deviation of the same samples.
        bw_factor: Fixed factor multiplying ``std``, or ``None`` for the
            Silverman rule ``(0.75 * counts) ** (-0.2)``.

    Returns:
        The bandwidth, in the dtype of the inputs.
    """
    factor = (0.75 * counts) ** (-0.2) if bw_factor is None else bw_factor
    return factor * std


@lru_cache(maxsize=32)
def _get_delta_kernel(
    grid_size: int,
    bw_factor: float | None,
    degenerate_tol: float,
    degenerate_bw: float | None,
    grid_chunk: int,
):
    """Return a JIT-compiled delta/S1 kernel for static estimator settings.

    The cache key holds only the scalar settings that change tracing: the grid
    size, the bandwidth branch, and the degenerate-class floor. The kernel
    takes all sample-sized data as runtime arguments, namely the outputs, the
    class indices and the masks. Nothing of size ``O(N)`` is therefore
    captured or baked into the compiled executable.

    Args:
        grid_size: Number of output-grid points for the KDE (static).
        bw_factor: KDE bandwidth factor multiplying the sample standard
            deviation, or ``None`` for the per-class Silverman rule.
        degenerate_tol: A class counts as degenerate when its bandwidth is
            below this fraction of the full-sample bandwidth.
        degenerate_bw: Bandwidth floor for a degenerate class, as a
            fraction of the full-sample bandwidth, applied exactly; or
            ``None`` for the default ``max(_DEGENERATE_BW_FRACTION *
            h_full, grid_step)``.
        grid_chunk: Output-grid points evaluated per conditional-KDE tile
            (static). ``grid_chunk >= grid_size`` evaluates the whole grid
            in one tile. A smaller value evaluates the grid in sequential
            tiles, which bounds the ``(Dg, M, grid_chunk, P)`` kernel
            tensor that dominates this method's peak memory. Every grid
            point keeps its own untouched sum over the class members, so
            tiling reorders nothing; any width of two or more returns the
            untiled result exactly.

    Returns:
        A jitted callable ``(Y_cols (N, C), all_idx (R, N), groups) ->
        (d (R, C, D), s1 (R, C, D), degenerate (R, C), floored (R, C))``.
        ``d`` and ``s1`` are the plug-in estimates for every replicate.
        ``degenerate`` flags each replicate and column whose resample is
        constant. ``floored`` flags each column where a degenerate class
        engaged the bandwidth floor. ``groups`` is a tuple of canonical
        ``(cls_idx, counts)`` partition-group layouts from
        :func:`jaxgsa._core.partition.build_partition_groups`. The kernel
        processes every group in one call, computing the shared
        per-replicate statistics once, and concatenates the results on the
        parameter axis in group order. Zero-size classes carry zero weight.
    """

    def _bandwidths(counts: Array, std: Array) -> Array:
        """Per-class (or full-sample) KDE bandwidths for this kernel's factor."""
        return _kde_bandwidths(counts, std, bw_factor)

    def _over_grid(fn: Callable[[Array], Array], grid: Array) -> Array:
        """Evaluate ``fn`` on tiles of the output grid and rejoin the tiles.

        Every KDE this kernel builds costs one kernel evaluation per
        (grid point, sample) pair, and those tensors are what the method
        peaks at. Splitting the grid axis bounds them: only one
        ``grid_chunk``-wide tile is live at a time, because
        :func:`jax.lax.map` is sequential.

        A grid point's density sums over the samples and over nothing
        else, so the tile width changes no term of any sum and lets no
        sum read a term belonging to another tile. That is an algorithmic
        guarantee, not a bit-level one: the reduction runs over an array
        whose shape carries the tile width, and XLA picks its
        vectorization from the whole shape, so the last bit can still
        move. Measured differences sit at one to two float32 units in the
        last place, and they differ by target, so the tests compare with
        a tolerance rather than exactly. The grid is padded up to a whole
        number of tiles and the padding is dropped again, so only one tile
        shape is ever traced.

        Args:
            fn: Maps a grid tile ``(Gt,)`` to values ``(..., Gt)``.
            grid: The full output grid ``(G,)``.

        Returns:
            ``fn(grid)``, shape ``(..., G)``.
        """
        if grid_chunk >= grid_size:
            return fn(grid)
        n_tiles = -(-grid_size // grid_chunk)
        pad = n_tiles * grid_chunk - grid_size
        padded = jnp.concatenate([grid, jnp.repeat(grid[-1:], pad)]) if pad else grid
        tiles = jax.lax.map(fn, padded.reshape(n_tiles, grid_chunk))
        # (n_tiles, ..., Gt) -> (..., n_tiles * Gt)
        rejoined = jnp.moveaxis(tiles, 0, -2).reshape(*tiles.shape[1:-1], -1)
        return rejoined[..., :grid_size]

    def _kde_full(y: Array, grid: Array, h: Array) -> Array:
        """Gaussian KDE of a full column ``y (N,)`` on ``grid (G,)``."""
        n = y.shape[0]
        safe_h = jnp.where(h > 0, h, 1.0)

        def _tile(grid_tile: Array) -> Array:
            u = (grid_tile[:, None] - y[None, :]) / safe_h
            return jnp.exp(-0.5 * u * u).sum(axis=1) / (n * safe_h * _SQRT_2PI)

        f = _over_grid(_tile, grid)
        # A zero bandwidth means constant data. It drops the density from
        # the integrand, mirroring SALib's degenerate-class treatment.
        return jnp.where(h > 0, f, 0.0)

    def _impl(
        Y_cols: Array,
        all_idx: Array,
        groups: tuple[tuple[Array, Array], ...],
    ):
        dtype = jnp.result_type(Y_cols.dtype, jnp.float32)
        Y_cols = Y_cols.astype(dtype)
        N = Y_cols.shape[0]  # every row belongs to exactly one class per parameter

        # The output grids depend only on the original sample, so every
        # replicate reuses them. SALib does the same.
        y_min = Y_cols.min(axis=0)
        y_max = Y_cols.max(axis=0)
        steps = jnp.linspace(0.0, 1.0, grid_size, dtype=dtype)
        grids = y_min[:, None] + steps[None, :] * (y_max - y_min)[:, None]  # (C, G)

        counts_list = tuple(counts for _, counts in groups)
        cls_list = tuple(cls_idx for cls_idx, _ in groups)

        def _group_stats(y, grid, h_full, fy, y_mean, cls_idx, mask_b, counts_b):
            """Per-parameter delta/S1 numerators and floor flag for one group.

            ``mask_b``, shape ``(G, M, P)``, and ``counts_b``, shape
            ``(G, M)``, are both float and broadcast against the group's
            parameter axis Dg. ``G`` is either 1 or Dg. A zero-size class
            has zero standard deviation and therefore zero bandwidth, so
            its density is dropped. Its zero count also removes it from
            every weighted sum.
            """
            safe_counts = jnp.maximum(counts_b, 1.0)
            y_cls = y[cls_idx]  # (Dg, M, P) resampled class members
            mean = (y_cls * mask_b).sum(axis=-1) / safe_counts  # (Dg, M)
            dev = (y_cls - mean[..., None]) * mask_b
            var = (dev**2).sum(axis=-1) / jnp.maximum(counts_b - 1.0, 1.0)
            h = _bandwidths(safe_counts, jnp.sqrt(var))  # (Dg, M)
            # A degenerate class is one the output grid cannot resolve. A
            # zero-variance class gets bandwidth 0 and a zeroed density,
            # which biases delta far low. A class narrower than the grid
            # step aliases, which pushes delta far above 1. Floor both to a
            # kernel the grid can integrate. See the _DEGENERATE_BW_TOL and
            # _DEGENERATE_BW_FRACTION comments. The predicate keeps
            # resolvable classes bit-identical. It also stays False for a
            # constant column, where h_full == 0, which must keep its
            # delta = 0 contract.
            if degenerate_bw is None:
                floor = jnp.maximum(_DEGENERATE_BW_FRACTION * h_full, grid[1] - grid[0])
            else:
                floor = degenerate_bw * h_full
            floored_cls = (counts_b > 0) & (h < degenerate_tol * h_full)
            h = jnp.where(floored_cls, floor, h)
            safe_h = jnp.where(h > 0, h, 1.0)

            def _density_tile(grid_tile: Array) -> Array:
                """Conditional class densities on one tile of the output grid.

                This is the largest tensor the estimator ever builds:
                ``(Dg, M, len(grid_tile), P)``. Tiling the grid is what
                keeps it, and with it the method's peak memory, bounded.
                """
                u = (grid_tile[None, None, :, None] - y_cls[:, :, None, :]) / safe_h[
                    ..., None, None
                ]
                k = jnp.exp(-0.5 * u * u) * mask_b[..., None, :]
                return k.sum(axis=-1) / (safe_counts[..., None] * safe_h[..., None] * _SQRT_2PI)

            fyc = _over_grid(_density_tile, grid)
            fyc = jnp.where((h > 0)[..., None], fyc, 0.0)  # (Dg, M, G)

            l1 = jnp.trapezoid(jnp.abs(fy[None, None, :] - fyc), grid, axis=-1)
            delta = (counts_b / (2.0 * N) * l1).sum(axis=-1)  # (Dg,)
            Vi = (counts_b / N * (mean - y_mean) ** 2).sum(axis=-1)
            return delta, Vi, floored_cls.any()

        def _col_stats(y: Array, grid: Array, r: Array, layouts):
            """Delta, S1, degeneracy and floor flags for one column/replicate.

            The per-replicate column statistics are the resampled column,
            the full KDE, the mean and the variance. This function computes
            them once and shares them with every partition group. It then
            concatenates the group results on the parameter axis in group
            order.
            """
            y_r = y[r]  # resampled column
            h_full = _bandwidths(jnp.asarray(float(N), dtype=dtype), jnp.std(y_r, ddof=1))
            fy = _kde_full(y_r, grid, h_full)
            degenerate = y_r.max() == y_r.min()
            y_mean = y_r.mean()
            y_var = jnp.var(y_r)
            safe_var = jnp.where(y_var > 0, y_var, 1.0)

            outs = [
                _group_stats(y, grid, h_full, fy, y_mean, cls_idx, mask_b, counts_b)
                for cls_idx, mask_b, counts_b in layouts
            ]
            delta = jnp.concatenate([o[0] for o in outs])
            Vi = jnp.concatenate([o[1] for o in outs])
            s1 = jnp.where(y_var > 0, Vi / safe_var, 0.0)
            floored = jnp.stack([o[2] for o in outs]).any()
            return delta, s1, degenerate, floored

        def _one_replicate(carry, xs):
            i, r, cls_parts = xs
            layouts = []
            for cls_r, counts_g in zip(cls_parts, counts_list):
                counts_r = _replicate_slice(counts_g, i)  # (G, M)
                mask_r = _mask_from_counts(counts_r, cls_r.shape[-1]).astype(dtype)
                layouts.append((cls_r, mask_r, counts_r.astype(dtype)))
            d, s1, degen, floored = jax.vmap(lambda y, grid: _col_stats(y, grid, r, layouts))(
                Y_cols.T, grids
            )
            return carry, (d, s1, degen, floored)

        R = all_idx.shape[0]
        scan_xs = (jnp.arange(R), all_idx, cls_list)
        _, (d_all, s1_all, degen_all, floored_all) = jax.lax.scan(_one_replicate, None, scan_xs)
        return d_all, s1_all, degen_all, floored_all

    return jax.jit(_impl)


def _resolve_slice_chunks(
    groups: list[tuple[Array, Array]],
    Y_cols: Array,
    grid_size: int,
    slice_chunk_size: int | None,
) -> tuple[int, int]:
    """Size the output-column chunk and the output-grid tile.

    Peak memory is the conditional-KDE tensor. It holds one output-grid
    point per padded class slot, so it costs
    ``columns * sum_g(Dg * Mg * Pg) * grid_points`` elements. This sizes it
    from the real per-group layout: assuming that product is ``~ D * N``
    holds for equal-frequency continuous classes, but it under-counts badly
    for an imbalanced categorical column, where every level is padded up to
    the largest one.

    Two knobs bound that tensor, and they answer different questions. The
    slice chunk answers "how much of the output can be in flight at once",
    and the transient-memory budget is its cap. The grid tile answers "how
    much of the KDE has to be resident to keep the CPU busy", and a
    working-set target is its cap; the tile is where nearly all of the
    saving is, because tiling costs no extra passes over the data.

    Both numbers come from shapes and Python scalars only, so this is
    traceable: :func:`indices` and :func:`analyze` share it and therefore
    pick the same chunking for the same call.

    Args:
        groups: Canonical partition groups from
            :func:`jaxgsa._core.partition.build_partition_groups`.
        Y_cols: Output columns ``(N, T*K)``, read for its dtype and width.
        grid_size: Number of output-grid points.
        slice_chunk_size: Caller override, or ``None`` for the budget.

    Returns:
        ``(slice_chunk, grid_chunk)``.
    """
    total = Y_cols.shape[1]
    layout_elems = sum(g[0].shape[1] * g[0].shape[2] * g[0].shape[3] for g in groups)
    itemsize = jnp.dtype(jnp.result_type(Y_cols.dtype, jnp.float32)).itemsize
    bytes_per_column_grid_point = layout_elems * itemsize
    cs = resolve_batch_size(
        bytes_per_column_grid_point * _KDE_LIVE_TENSORS, total, slice_chunk_size
    )
    grid_chunk = min(
        grid_size,
        max(_MIN_GRID_TILE, _KDE_TILE_TARGET_BYTES // (cs * bytes_per_column_grid_point)),
    )
    return cs, grid_chunk


def _run_delta_kernel(
    Y_cols: Array,
    all_idx: Array,
    groups: list[tuple[Array, Array]],
    col_order: Array | None,
    *,
    grid_size: int,
    bw_factor: float | None,
    degenerate_tol: float,
    degenerate_bw: float | None,
    slice_chunk_size: int | None,
) -> tuple[Array, Array, Array, Array]:
    """Run the jitted delta kernel over every chunk of output columns.

    This is the whole estimator, and nothing else: it branches on shapes and
    Python scalars only, reads no array value on the host, and raises and
    warns about nothing. :func:`analyze` and :func:`indices` both go through
    it, so the two can never compute different numbers.

    Args:
        Y_cols: Output columns, shape ``(N, T*K)``.
        all_idx: Replicate row indices ``(R, N)``. Row 0 is the identity, so
            replicate 0 is the original sample.
        groups: Canonical partition groups; see
            :mod:`jaxgsa._core.partition`.
        col_order: Gather that restores the problem's column order, or
            ``None`` when the groups are already in order.
        grid_size: Number of output-grid points for the KDE.
        bw_factor: Fixed KDE bandwidth factor, or ``None`` for Silverman.
        degenerate_tol: Degenerate-class threshold, as a fraction of the
            full-sample bandwidth.
        degenerate_bw: Degenerate-class bandwidth floor, as a fraction of
            the full-sample bandwidth, or ``None`` for ``"auto"``.
        slice_chunk_size: Output columns per kernel call, or ``None`` for
            the transient-memory budget.

    Returns:
        ``(d_all, s1_all, degen_all, floored_cols)``. ``d_all`` and
        ``s1_all`` have shape ``(R, T*K, D)``, ``degen_all`` ``(R, T*K)``,
        and ``floored_cols`` ``(T*K,)``.
    """
    total = Y_cols.shape[1]
    cs, grid_chunk = _resolve_slice_chunks(groups, Y_cols, grid_size, slice_chunk_size)
    kernel = _get_delta_kernel(grid_size, bw_factor, degenerate_tol, degenerate_bw, grid_chunk)
    d_parts, s1_parts, degen_parts, floored_parts = [], [], [], []
    for start in range(0, total, cs):
        chunk = Y_cols[:, start : start + cs]
        n_real = chunk.shape[1]
        if n_real < cs:
            # Pad the ragged trailing chunk back to the full width so the
            # jitted kernel compiles for one shape only. Output columns are
            # analysed independently, so a repeated column cannot change a
            # real column's answer, and the padding is sliced off at once.
            # The chunk width does change how XLA tiles the in-column
            # reductions, so a different slice_chunk_size can move a value
            # by a couple of ulps in float32 — which is why
            # test_slice_chunk_size_invariance compares at rtol=1e-6, not
            # bitwise.
            pad = jnp.broadcast_to(chunk[:, :1], (chunk.shape[0], cs - n_real))
            chunk = jnp.concatenate([chunk, pad], axis=1)
        d, s1, degen, floored = kernel(chunk, all_idx, tuple(groups))
        if n_real < cs:
            d, s1 = d[:, :n_real], s1[:, :n_real]
            degen, floored = degen[:, :n_real], floored[:, :n_real]
        if col_order is not None:
            d = d[..., col_order]
            s1 = s1[..., col_order]
        d_parts.append(d)
        s1_parts.append(s1)
        degen_parts.append(degen)
        # Keep the flag per output column, not per chunk: a failure message
        # has to name the column whose class was floored.
        floored_parts.append(floored.any(axis=0))
    return (
        jnp.concatenate(d_parts, axis=1),
        jnp.concatenate(s1_parts, axis=1),
        jnp.concatenate(degen_parts, axis=1),
        jnp.concatenate(floored_parts),
    )


def _slice_label(flat_index: int, trailing: tuple[int, ...], problem: Problem) -> str:
    """Name one flattened output column for an error message.

    Args:
        flat_index: Index of the column in the flattened ``T*K`` axis.
        trailing: The shape of ``Y`` after the sample axis: ``()``,
            ``(K,)``, or ``(T, K)``.
        problem: Problem definition, which may carry ``output_names``.

    Returns:
        A short label such as ``"k=1 ('flux')"`` or ``"(t=3, k=0)"``. The
        label is empty for a single scalar output, which needs no name.
    """
    if len(trailing) == 0:
        return ""
    K = trailing[-1]
    names = problem.output_names
    t, k = divmod(flat_index, K)

    def _k_label(k_index: int) -> str:
        if names is not None and len(names) == K:
            return f"k={k_index} ('{names[k_index]}')"
        return f"k={k_index}"

    if len(trailing) == 1:
        return _k_label(flat_index)
    return f"(t={t}, {_k_label(k)})"


def _raise_discrete_output(problem: Problem, Y: Array) -> None:
    """Reject a discrete output up front.

    The estimator compares Gaussian kernel density estimates on a shared
    output grid. That construction needs a continuous output. A discrete
    output has atoms, and the density of an atom is a spike no grid
    resolves. The returned delta would then be an artifact of ``grid_size``
    rather than a property of the model. This check runs before any
    expensive work.

    An output column counts as discrete only when it takes at most
    ``_DISCRETE_MAX_DISTINCT`` distinct values and each value repeats at
    least ``_DISCRETE_MIN_MULTIPLICITY`` times on average. Both conditions
    must hold, so a continuous output rounded to a few decimals keeps
    working. A constant column, with one distinct value, is exempt. It
    needs no density, and ``delta = S1 = 0`` is its exact answer.

    Args:
        problem: Problem definition, used to name the offending column.
        Y: Validated output array ``(N,)``, ``(N, K)``, or ``(N, T, K)``.

    Raises:
        ValueError: If any output column is discrete.
    """
    N = Y.shape[0]
    trailing = Y.shape[1:]
    flat = Y.reshape(N, -1)
    if N < 2:
        return
    ordered = jnp.sort(flat, axis=0)
    n_distinct = np.asarray(1 + jnp.sum(ordered[1:] != ordered[:-1], axis=0))
    discrete = (
        (n_distinct > 1)
        & (n_distinct <= _DISCRETE_MAX_DISTINCT)
        & (n_distinct * _DISCRETE_MIN_MULTIPLICITY <= N)
    )
    bad = np.flatnonzero(discrete)
    if bad.size == 0:
        return

    if flat.shape[1] == 1:
        where = f"the output takes only {int(n_distinct[0])} distinct values"
    else:
        detail = ", ".join(
            f"{_slice_label(int(i), trailing, problem)}: {int(n_distinct[i])} distinct values"
            for i in bad[:5]
        )
        extra = f", and {bad.size - 5} more" if bad.size > 5 else ""
        where = f"{bad.size} of {flat.shape[1]} output columns are discrete ({detail}{extra})"

    raise ValueError(
        "jaxgsa.borgonovo.analyze supports a continuous output distribution "
        f"only, but {where} in {N} samples. The delta estimator compares "
        "Gaussian kernel density estimates on a shared output grid; an "
        "atomic density is a spike that no grid resolves, so the index "
        "would report the grid resolution, not the model. Use "
        "jaxgsa.optimal_transport.analyze for a discrete output: it "
        "compares empirical distributions directly and needs no density."
    )


def _bandwidth_advice(
    problem: Problem,
    Y_cols: Array,
    trailing: tuple[int, ...],
    grid_size: int,
    bw_factor: float | None,
    degenerate_tol: float,
    degenerate_bandwidth: float | Literal["auto"],
    degenerate_bw: float | None,
    floored: Array,
) -> str:
    """Explain which knob to turn after a delta estimate leaves ``[0, 1]``.

    An out-of-range delta comes from a conditioning class the output grid
    cannot resolve. Different situations produce it, and they need
    different advice, so this function reads what the kernel actually did
    rather than what the settings asked for.

    If no class was floored, because none fell below ``degenerate_tol``,
    the floor had no effect on the run. The message says so and names
    ``degenerate_tol`` and ``grid_size``.

    If a class was floored under an explicit ``degenerate_bandwidth``, that
    float is the width to change. The message compares it with one grid
    step and gives the fraction that would match one step.

    If a class was floored under ``degenerate_bandwidth="auto"``, the floor
    is already at least one grid step wide, so widening it is not the fix.
    The message names ``grid_size`` only.

    The numbers come from the original sample on the host: the same grid
    construction as the kernel, and the shared :func:`_kde_bandwidths`.
    They are advisory only, so a difference of a few units in the last
    place against the kernel's own values does not matter. Nothing here
    decides whether to raise; :func:`_raise_delta_out_of_range` has already
    decided that from the returned estimate.

    Args:
        problem: Problem definition, used to name the offending column.
        Y_cols: Output columns, shape ``(N, T*K)``, as passed to the kernel.
        trailing: The shape of ``Y`` after the sample axis: ``()``,
            ``(K,)``, or ``(T, K)``.
        grid_size: Number of output-grid points.
        bw_factor: Resolved ``bandwidth`` factor, or ``None`` for Silverman.
        degenerate_tol: The tolerance that decides degeneracy.
        degenerate_bandwidth: The argument as the caller passed it, quoted
            back in the message.
        degenerate_bw: Resolved floor fraction, or ``None`` for ``"auto"``.
        floored: Boolean flags ``(T*K,)`` from the kernel, ``True`` where a
            degenerate class engaged the bandwidth floor in any replicate.

    Returns:
        One or more sentences naming the knob to turn and the value to
        turn it to.
    """
    engaged = np.flatnonzero(np.asarray(floored))
    if engaged.size == 0:
        return (
            "No conditioning class was narrow enough to count as degenerate "
            f"at degenerate_tol={degenerate_tol:.3g}, so the bandwidth floor "
            f"was never applied. Raise grid_size (currently {grid_size}) so "
            "the grid resolves the narrow class. Or raise degenerate_tol so "
            "the class is treated as degenerate and given the wider kernel "
            "that degenerate_bandwidth sets."
        )

    dtype = jnp.result_type(Y_cols.dtype, jnp.float32)
    cols = Y_cols.astype(dtype)
    N = cols.shape[0]
    # Mirror the kernel's grid construction, including the final subtraction.
    y_min = cols.min(axis=0)
    y_max = cols.max(axis=0)
    steps = jnp.linspace(0.0, 1.0, grid_size, dtype=dtype)
    grids = y_min[:, None] + steps[None, :] * (y_max - y_min)[:, None]
    step = np.asarray(grids[:, 1] - grids[:, 0])
    full = np.asarray(
        _kde_bandwidths(
            jnp.asarray(float(N), dtype=dtype), jnp.std(cols, axis=0, ddof=1), bw_factor
        )
    )
    if degenerate_bw is None:
        applied = np.maximum(_DEGENERATE_BW_FRACTION * full, step)
    else:
        applied = degenerate_bw * full
    # Report the floored column whose kernel is narrowest against its own
    # grid. A floored column always has a positive grid step: a constant
    # column has h_full == 0, which no bandwidth falls below.
    ratio = np.where(
        step[engaged] > 0,
        applied[engaged] / np.where(step[engaged] > 0, step[engaged], 1.0),
        np.inf,
    )
    worst = int(engaged[int(np.argmin(ratio))])
    where = "" if cols.shape[1] == 1 else f" in {_slice_label(worst, trailing, problem)}"

    if degenerate_bw is None:
        # Reachable, but only off the usual route. The "auto" floor is
        # never below one grid step, so it cannot alias on its own. It
        # takes a second cause, such as a tiny explicit ``bandwidth``
        # factor that leaves the unconditional density unresolved too.
        # See test_auto_floor_still_aliases_under_a_tiny_bandwidth_factor.
        return (
            f"A conditioning class{where} was too narrow to resolve, so its "
            "kernel was floored at max(0.1 * h_full, grid_step) = "
            f"{float(applied[worst]):.3g}. That floor is already at least "
            "one grid step wide, so widening it further is not the fix. "
            f"Raise grid_size (currently {grid_size}): a shorter step lets "
            "the grid follow the narrow class."
        )
    needed = float(step[worst]) / float(full[worst])
    return (
        f"A conditioning class{where} was too narrow to resolve, so its "
        f"kernel was floored. degenerate_bandwidth={degenerate_bandwidth!r} "
        f"set that floor to {float(applied[worst]):.3g}, against an "
        f"output-grid step of {float(step[worst]):.3g}. Raise "
        f"degenerate_bandwidth to at least {needed:.3g}, which is the "
        "fraction of the full-sample bandwidth that equals one grid step. "
        "Or pass degenerate_bandwidth='auto', the default, which takes "
        "max(0.1 * h_full, grid_step) for you. Raising grid_size (currently "
        f"{grid_size}) shrinks the grid step instead."
    )


def analyze(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_classes: int | None = None,
    grid_size: int = 100,
    bandwidth: float | Literal["silverman"] = "silverman",
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    bias_correct: bool | None = None,
    key: Array | None = None,
    slice_chunk_size: int | None = None,
    degenerate_tol: float = _DEGENERATE_BW_TOL,
    degenerate_bandwidth: float | Literal["auto"] = "auto",
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> DeltaResult:
    """Compute Borgonovo delta and given-data first-order Sobol indices.

    The delta index measures how much knowing a parameter's value shifts the
    whole output density. ``delta_i`` is (half) the expected L1 distance
    between the unconditional output density and the density conditional on
    x_i. It lies in [0, 1]. A value of 0 means the output distribution does
    not change with x_i, and 1 means x_i fully determines it. Delta compares
    whole densities rather than variances, which makes it
    moment-independent. It therefore captures influence on tails and on
    shape that Sobol indices miss. It also needs no special sampling design,
    so any (X, Y) sample works. The same partition returns a given-data
    first-order Sobol index ``S1`` at negligible extra cost.

    The estimator supports correlated inputs. It partitions on the
    parameters' ordinal ranks and compares output densities, so a declared
    ``problem.correlation`` does not invalidate it. Under dependence the
    index measures each parameter's total association with the output. That
    total includes the effects carried by the parameters it correlates with.
    The companion ``S1`` is the given-data first-order Sobol index and reads
    the same way. Neither index separates the direct effect from the
    correlation-borne one. Use :mod:`jaxgsa.vkoga` or
    :mod:`jaxgsa.kucherenko` for that split; both need continuous inputs.

    Args:
        problem: Problem definition with D parameters.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        n_classes: Number of equal-frequency conditioning classes per
            continuous parameter. ``None`` selects the Plischke
            sample-size heuristic, which is identical to SALib's and uses
            at most 48 classes. A categorical parameter ignores this
            argument and always uses one class per level, with class sizes
            equal to the observed level counts. Declared levels with no
            observed samples are dropped with a warning. A passed value is
            always validated against ``[2, N]``. When every parameter is
            categorical, a ``JaxgsaWarning`` says the value is ignored.
        grid_size: Number of points of the output grid the densities are
            compared on. The grid spans ``[Y.min(), Y.max()]`` per column.
            It is also the resolution knob for near-degenerate
            conditioning classes; see the note below.
        bandwidth: KDE bandwidth rule. Use ``"silverman"`` for the
            per-class Silverman factor, or a positive float used directly
            as the factor multiplying the sample standard deviation.
        n_bootstrap: Number of bootstrap resamples for bias correction and
            confidence intervals. ``0`` skips both: the result is the
            plug-in estimate and ``delta_conf`` and ``S1_conf`` are
            ``None``.
        conf_level: Confidence level for the bootstrap intervals.
        ci_method: How the interval endpoints are formed. ``"quantile"``
            (default) reads them off the empirical bootstrap distribution.
            ``"gaussian"`` centres them on the point estimate and takes
            ``+/- z * sd`` of the bootstrap draws, which is smoother for a
            small ``n_bootstrap`` but assumes the draws are normal.
        bias_correct: Apply the Plischke bias reduction
            ``2*d_hat - mean(d_boot)`` to the delta estimate.

            **This default is a tri-state, and what ``None`` does depends on
            ``n_bootstrap``.** ``None`` (the default) applies the correction
            whenever there are replicates to apply it with — that is,
            whenever ``n_bootstrap > 0`` — and does nothing when
            ``n_bootstrap == 0``. So adding ``n_bootstrap=100`` to a default
            call does not only add intervals: it also changes the reported
            ``delta`` from the plug-in estimate to the corrected one. The
            first default call per process that resolves ``None`` to the
            correction says so with a one-time ``JaxgsaWarning``; pass an
            explicit ``True`` or ``False`` to silence it. ``True`` asks for
            the correction explicitly, and warns if ``n_bootstrap`` is ``0``,
            because then it cannot be delivered. ``False`` never applies it.

            The tri-state exists because the correction needs replicates and
            replicates are opt-in: a plain ``bias_correct=True`` default
            beside ``n_bootstrap=0`` would be a contradiction that warns on
            every default call. S1 is never bias-corrected, matching SALib.

            The uncorrected delta is positively biased: a KDE separation is
            a distance, so sampling noise can only push it up. Prefer
            ``n_bootstrap >= 100`` when the value matters, not only the
            ranking.
        key: A ``jax.random`` key for the bootstrap resampling. Required
            when ``n_bootstrap > 0``. Pass ``jax.random.key(0)`` if you have
            an integer seed.
        slice_chunk_size: Number of flattened ``T*K`` output columns
            processed per kernel call. ``None`` derives it from the
            transient-memory budget
            (:func:`jaxgsa.config.set_memory_budget`) and the real class
            layout. Pass a positive integer to override it. One column
            costs the summed padded class layout ``sum_g(Dg * Mg * Pg)``
            per output-grid point. That layout is about ``D * N`` for
            continuous parameters; an imbalanced categorical parameter
            pads every level up to the largest one, so it can be many
            times ``D * N``. The grid is then evaluated in tiles, so peak
            memory follows the tile rather than the whole grid, and a
            narrower chunk is rarely the knob that saves memory here.
        degenerate_tol: A conditioning class counts as degenerate when its
            KDE bandwidth is below this fraction of the full-sample
            bandwidth. Degenerate classes get the floored bandwidth below.
            Lower it to let narrower classes keep their own bandwidth, but
            read the note about grid resolution first. Raising it above the
            floor fraction goes the other way and biases the result: a
            class whose own bandwidth is between the floor and this
            tolerance is then *narrowed* to the floor, which inflates delta
            for the very classes the higher tolerance said to distrust. The
            answer stays a valid computation, so this is a bias to know
            about, not an error.
        degenerate_bandwidth: Bandwidth floor applied to a degenerate
            class. ``"auto"`` uses ``max(0.1 * h_full, grid_step)``, which
            never goes below what the output grid can integrate. A float
            is a fraction of the full-sample bandwidth ``h_full`` and is
            applied exactly, with no grid-step bound. This setting only
            ever reaches a class the estimator already found degenerate,
            so on data with no such class it cannot change the result at
            any value. Where it does apply, a float far below
            ``grid_step / h_full`` risks aliasing on the grid. Whether it
            aliases depends on where the narrow class sits relative to the
            grid points, so ``analyze`` does not refuse the setting up
            front. It checks the returned delta instead, and the error
            message then names this argument and the value that would fix
            it.
        on_invalid: What to do about a row of ``X`` or ``Y`` that holds a
            non-finite value. ``"raise"`` (default) refuses the sample,
            ``"drop"`` removes those rows and analyzes the rest, and
            ``"propagate"`` warns and computes anyway. ``X`` and ``Y`` are
            checked together, so a bad input takes its own output with it.
            The check runs before the KDE, so a failed model run is named
            for what it is instead of surfacing later as a bandwidth
            complaint. See :mod:`jaxgsa._core.invalid`.
        verbose: If ``True`` (default), print a short summary to stdout: the
            problem and the data, the wall-clock timing, and the top
            parameters by ``delta``. Pass ``False`` for a silent run.
        keep_replicates: Keep the per-resample indices on
            ``DeltaResult.ci.replicates``. Off by default because they are
            large: ``n_bootstrap`` copies of both index arrays. Turn it on to
            recompute an interval at another level without re-running the
            analysis. The ``delta`` draws are the ones the interval was taken
            from, so they carry the bias correction when
            ``bias_correct=True``.

    Note:
        This estimator supports a continuous output distribution only. It
        compares Gaussian kernel density estimates on a shared output grid,
        and a discrete output has atoms that no grid resolves. ``analyze``
        checks the output up front and raises ``ValueError`` when a column
        takes at most 20 distinct values and each value repeats at least 5
        times on average. Use :func:`jaxgsa.optimal_transport.analyze` for a
        discrete output: it compares empirical distributions directly and
        needs no density. The check does not refuse a continuous output
        rounded to a few decimals, and it does not refuse a constant
        column, whose exact answer is ``delta = S1 = 0``. A categorical
        parameter is still supported. The restriction is on the output.

    Note:
        A conditioning class can be a point mass or nearly one. That is the
        normal case for a categorical level that maps to one output value.
        For such a class the delta estimate depends on the grid resolution
        and is biased low. ``grid_step`` sets the floored bandwidth, so
        ``grid_size`` is the knob that moves the answer. On a noise-free
        three-atom model with true delta ``2/3``, the estimate goes 0.56 at
        ``grid_size=50``, 0.61 at 100, and 0.61 at 200 and above. The bias
        also does not vanish as N grows, so on atomic conditionals this
        estimator is not consistent. Treat delta on such parameters as a
        ranking signal, not a calibrated number. Parameters with genuine
        conditional spread are unaffected.

    Returns:
        A :class:`DeltaResult` with the delta and ``S1`` indices and the
        optional confidence intervals. The underlying delta index is
        defined on ``[0, 1]`` and the plug-in estimate stays in that range.
        The default bias-corrected estimate and its confidence bounds can
        fall marginally below 0 for weak or near-noninfluential parameters
        at small sample sizes. A constant output column yields
        ``delta = S1 = 0``, where SALib raises an error. A conditioning
        class the output grid cannot resolve gets a floored KDE bandwidth
        instead of its own, together with one ``JaxgsaWarning``; classes with
        genuine spread are unaffected. A confidence bound outside
        ``[0, 1]`` by more than 0.05 raises a ``JaxgsaWarning`` naming the
        parameter and the bound. The point estimate still stands in that
        case, so only the interval is suspect.

    Raises:
        ValueError: If any argument fails validation. The invalid cases are:
            X is not 2-D; the column count of X does not match the problem;
            Y is not 1-D, 2-D or 3-D; X and Y have differing row counts; a
            passed ``n_classes`` is not in ``[2, N]``; a categorical column
            of X holds values other than its integer level codes;
            ``grid_size < 2``; ``bandwidth`` is neither ``"silverman"`` nor
            a positive float; ``n_bootstrap < 0``; ``n_bootstrap > 0`` and no
            ``key`` was given; ``ci_method`` is neither ``"quantile"`` nor
            ``"gaussian"``; ``conf_level`` is not in
            ``(0, 1)``; ``slice_chunk_size`` is not a positive integer;
            ``degenerate_tol`` is not in ``[0, 1)``; or
            ``degenerate_bandwidth`` is neither ``"auto"`` nor a positive
            float; or ``on_invalid`` is not one of the three policies. It
            is also raised when the non-finite policy refuses the sample,
            and when an output column is discrete,
            because the estimator supports a continuous output only; see
            the note above. It is raised again when the returned delta
            leaves ``[0, 1]`` by more than 0.05, which means the
            computation failed rather than returned an estimate.

    Warns:
        JaxgsaWarning: If an output slice has zero variance. Every
            conditional density then equals the unconditional one, so delta
            is an exact 0 rather than an answer.
    """
    from jaxgsa.borgonovo import SPEC

    # A non-finite value has to be settled before the KDE runs. Left alone it
    # would survive the whole estimate and the bootstrap, and then surface as
    # an out-of-range delta, which reads as a bandwidth problem and is not one.
    ctx = prepare(
        SPEC,
        problem,
        Y,
        X=X,
        on_invalid=on_invalid,
        checks=(
            at_least("grid_size", grid_size, 2),
            at_least("n_bootstrap", n_bootstrap, 0),
            one_of("ci_method", ci_method, ("quantile", "gaussian")),
            in_open_interval("conf_level", conf_level, 0.0, 1.0),
            require(
                0 <= float(degenerate_tol) < 1,
                f"degenerate_tol must be in [0, 1), got {degenerate_tol}",
            ),
            at_least("slice_chunk_size", slice_chunk_size, 1),
        ),
        min_kept=_MIN_KEPT,
        # A constant slice leaves every conditional distribution equal to
        # the unconditional one, so the indices come out an exact 0, not
        # the NaN a variance ratio would give.
        zero_variance_outcome="zero",
    )
    X, Y, invalid = ctx.inputs, ctx.Y, ctx.invalid
    # Check the continuous-output contract before any expensive work.
    _raise_discrete_output(problem, Y)

    t0 = _verbose.tic()
    N = X.shape[0]
    # n_classes applies to the continuous columns only. A categorical column
    # always gets one conditioning class per level.
    dims_levels = _categorical_dims(problem)
    cat_dims = [d for d, _ in dims_levels]
    cont_dims = [d for d in range(problem.num_vars) if d not in set(cat_dims)]
    if n_classes is None:
        M = _plischke_n_classes(N)
    else:
        M = int(n_classes)
        # A passed value is always validated, even when nothing uses it.
        # Silently accepting nonsense hides bugs in the caller.
        if not 2 <= M <= N:
            raise ValueError(f"n_classes must be in [2, N={N}], got {n_classes}")
        if not cont_dims:
            warnings.warn(
                "jaxgsa.borgonovo: n_classes is ignored because every parameter is "
                "categorical (one conditioning class per level)",
                stacklevel=2,
                category=JaxgsaWarning,
            )
    bw_factor = _resolve_sentinel_float(bandwidth, "bandwidth", "silverman")
    degenerate_tol = float(degenerate_tol)
    degenerate_bw = _resolve_sentinel_float(degenerate_bandwidth, "degenerate_bandwidth", "auto")

    Y_3d = ctx.Y3
    _, T, K = Y_3d.shape
    D = problem.num_vars
    Y_cols = Y_3d.reshape(N, T * K)

    if slice_chunk_size is not None and slice_chunk_size < 1:
        raise ValueError(f"slice_chunk_size must be >= 1, got {slice_chunk_size}")

    # Replicate 0 is the identity permutation, that is the original sample.
    # The remaining rows are the bootstrap resamples. Building them together
    # means the point estimate and its interval share one code path.
    identity = jnp.arange(N, dtype=jnp.int32)[None, :]
    if bias_correct is True and n_bootstrap == 0:
        warnings.warn(
            "jaxgsa.borgonovo: bias_correct=True needs bootstrap replicates, and "
            "n_bootstrap is 0, so no bias correction was applied. The delta "
            "returned is the raw estimate, which is biased upward because a KDE "
            "separation is a distance and sampling noise can only increase it. "
            "Pass n_bootstrap=100 (and a key) for the corrected value.",
            JaxgsaWarning,
            stacklevel=2,
        )
    apply_bias_correction = bias_correct is not False

    if n_bootstrap > 0:
        if key is None:
            raise ValueError("key is required when n_bootstrap > 0")
        # Warn after the key check, so the one shot lands on a call that
        # actually runs; a failed first call must not burn the warning.
        if bias_correct is None:
            global _warned_bias_correct_default
            if not _warned_bias_correct_default:
                _warned_bias_correct_default = True
                warnings.warn(
                    "jaxgsa.borgonovo: bias_correct was left at its default (None) and "
                    "n_bootstrap > 0, so the Plischke bias correction IS applied: the "
                    "delta reported is 2*d_hat - mean(d_boot), not the plug-in "
                    "estimate. Pass bias_correct=True to keep this and silence this "
                    "warning, or bias_correct=False for the uncorrected delta. "
                    "This warning is shown once per process.",
                    JaxgsaWarning,
                    stacklevel=2,
                )
        boot = jax.random.randint(key, (n_bootstrap, N), 0, N, dtype=jnp.int32)
        all_idx = jnp.concatenate([identity, boot], axis=0)
    else:
        all_idx = identity
    R = all_idx.shape[0]

    # Canonical partition-group layout, shared with optimal_transport.
    groups, _, col_order = build_partition_groups(
        problem, X, all_idx, M, dims_levels, method="jaxgsa.borgonovo.analyze"
    )

    d_all, s1_all, degen_all, floored_cols = _run_delta_kernel(
        Y_cols,
        all_idx,
        groups,
        col_order,
        grid_size=grid_size,
        bw_factor=bw_factor,
        degenerate_tol=degenerate_tol,
        degenerate_bw=degenerate_bw,
        slice_chunk_size=slice_chunk_size,
    )
    if bool(floored_cols.any()):
        warnings.warn(
            "jaxgsa.borgonovo: at least one conditioning class is too narrow for the "
            "output grid to resolve (often a point mass, e.g. a categorical "
            "level that maps to one output value). Its KDE bandwidth was "
            "floored to a kernel the grid can integrate. Delta for such an "
            "input depends on grid_size and is biased low; raise grid_size "
            "if you need a calibrated value",
            stacklevel=2,
            category=JaxgsaWarning,
        )

    d_all = d_all.reshape(R, T, K, D)
    s1_all = s1_all.reshape(R, T, K, D)
    degen_all = degen_all.reshape(R, T, K)

    d_hat = d_all[0]
    S1 = s1_all[0]

    delta_conf: Array | None = None
    S1_conf: Array | None = None
    ci: CIInfo | None = None
    if n_bootstrap > 0:
        # A constant bootstrap resample carries no information. Replace its
        # degenerate zero with the point estimate so it neither inflates nor
        # deflates the bias correction. A whole column that is constant
        # still gives 0.
        degen_boot = degen_all[1:, ..., None]
        d_boot = jnp.where(degen_boot, d_hat[None], d_all[1:])
        s1_boot = jnp.where(degen_boot, S1[None], s1_all[1:])

        if apply_bias_correction:
            # Plischke eqn 30 with d_hat on the original sample.
            d_reps = 2.0 * d_hat[None] - d_boot
            delta = jnp.nanmean(d_reps, axis=0)
        else:
            d_reps = d_boot
            delta = d_hat

        # Stack [lower, upper] into a leading axis of size 2. The Gaussian
        # endpoints are centred on the estimate the method reports, so for
        # delta that is the bias-corrected value the draws were built from.
        delta_conf = ctx.squeeze(
            jnp.stack(
                _bootstrap_ci_endpoints(delta, d_reps, conf_level=conf_level, ci_method=ci_method)
            )
        )
        S1_conf = ctx.squeeze(
            jnp.stack(
                _bootstrap_ci_endpoints(S1, s1_boot, conf_level=conf_level, ci_method=ci_method)
            )
        )
        # The leading resample axis survives the squeeze, which addresses the
        # T/K axes from the end.
        ci = CIInfo(
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            replicates=(
                {
                    "delta": ctx.squeeze(d_reps),
                    "S1": ctx.squeeze(s1_boot),
                }
                if keep_replicates
                else None
            ),
        )
    else:
        delta = d_hat

    # The point estimate is the contract, so an out-of-range value is an
    # error. The interval is a diagnostic, so it only warns. The advice is
    # built from what the kernel did, so it names the knob that actually
    # governs this run.
    #
    # A report that still holds findings here means the policy was
    # "propagate": "raise" would have raised and "drop" would have removed
    # them. The user asked for the non-finite value to reach the indices and
    # was warned that it would, so this check must not answer that with a
    # bandwidth complaint. It still enforces the [0, 1] range on the finite
    # entries.
    propagated = invalid.any_invalid
    _raise_delta_out_of_range(
        problem,
        delta,
        lambda: _bandwidth_advice(
            problem,
            Y_cols,
            Y.shape[1:],
            grid_size,
            bw_factor,
            degenerate_tol,
            degenerate_bandwidth,
            degenerate_bw,
            floored_cols,
        ),
        allow_non_finite=propagated,
    )
    if delta_conf is not None:
        _warn_conf_out_of_range(problem, delta_conf, allow_non_finite=propagated)

    result = DeltaResult(
        delta=ctx.squeeze(delta),
        delta_conf=delta_conf,
        S1=ctx.squeeze(S1),
        S1_conf=S1_conf,
        problem=problem,
        invalid=invalid,
        ci=ci,
    )

    if verbose:
        elapsed = _verbose.stop(t0, result.delta)
        # Same resolver the kernel loop ran, on the same layout: cheap
        # arithmetic, reported rather than re-derived by hand.
        cs, _ = _resolve_slice_chunks(groups, Y_cols, grid_size, slice_chunk_size)
        origin = "user-set" if slice_chunk_size is not None else "resolved from the memory budget"
        _verbose.analysis_summary(
            method="jaxgsa.borgonovo.analyze",
            problem=problem,
            n_runs=int(N),
            T=T,
            K=K,
            invalid=invalid,
            timings=[("estimator (includes compile on the first call)", elapsed)],
            notes=[
                f"slice_chunk_size: {cs} ({origin})",
                f"grid_size: {grid_size}",
                f"bandwidth: {bandwidth}",
            ],
            index_name="delta",
            values=result.delta,
            conf=result.delta_conf,
        )
    return result


def indices(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_classes: int | None = None,
    grid_size: int = 100,
    bandwidth: float | Literal["silverman"] = "silverman",
    slice_chunk_size: int | None = None,
    degenerate_tol: float = _DEGENERATE_BW_TOL,
    degenerate_bandwidth: float | Literal["auto"] = "auto",
) -> tuple[Array, Array]:
    """Compute Borgonovo delta and given-data S1 as plain arrays.

    This is the transformable core of :func:`analyze`. It runs the same
    kernel on the same data and returns the same numbers, but it does
    nothing else: no non-finite check, no discrete-output refusal, no
    zero-variance warning, no bandwidth-floor warning, no ``[0, 1]`` range
    check, no :class:`jaxgsa.borgonovo.DeltaResult`, and no read of any
    array value on the host. So it composes with ``jax.jit``, ``jax.vmap``,
    ``jax.grad`` and ``jax.jacrev``, which :func:`analyze` cannot, because
    a policy decision needs a concrete value and a tracer has none.

    Use :func:`analyze` for ordinary analysis. Nothing here checks the
    outputs, so a single NaN silently turns every index into NaN, and a
    discrete output silently returns a number that describes ``grid_size``
    rather than the model.

    There is no ``n_bootstrap`` and no bias correction. Both are policy.
    The Plischke correction ``2*d_hat - mean(d_boot)`` needs bootstrap
    replicates, which need a PRNG key and a resampling decision, and it
    returns a *different estimand* from the plug-in delta: the correction
    can leave ``[0, 1]``, which is why :func:`analyze` range-checks the
    result afterwards. A core that returned it would carry the replicate
    axis, the key and the range policy into every ``jacrev``. So
    :func:`indices` returns the plug-in point estimate only — exactly what
    ``analyze(..., n_bootstrap=0)`` reports.

    Continuous parameters only. A categorical parameter's conditioning
    classes are one class per observed level, and both their number and
    their padded width are read off the data on the host, so that layout
    cannot be built from a tracer. :func:`analyze` handles categorical
    parameters.

    Tier T4 (behavioural contract): the returned arrays must equal the
    ``delta`` and ``S1`` fields of ``analyze``'s result on clean outputs
    with ``n_bootstrap=0``, and the function must survive ``jit``, ``vmap``
    and ``jit(jacrev(...))``. Checked in ``tests/test_borgonovo.py``.

    Args:
        problem: Problem definition with D continuous parameters.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        n_classes: Number of equal-frequency conditioning classes, as in
            :func:`analyze`. ``None`` selects the Plischke heuristic, which
            reads ``N`` only, so it stays static under a trace.
        grid_size: Number of output-grid points, as in :func:`analyze`.
        bandwidth: KDE bandwidth rule, as in :func:`analyze`.
        slice_chunk_size: Output columns per kernel call, as in
            :func:`analyze`.
        degenerate_tol: Degenerate-class threshold, as in :func:`analyze`.
            The floor it triggers is applied inside the kernel with
            :func:`jax.numpy.where`, so it costs no host branch here; what
            :func:`analyze` adds is only the warning that it engaged.
        degenerate_bandwidth: Degenerate-class bandwidth floor, as in
            :func:`analyze`.

    Returns:
        ``(delta, S1)``, at the rank the caller's ``Y`` had: ``(D,)`` for a
        1-D ``Y``, ``(K, D)`` for 2-D, ``(T, K, D)`` for 3-D. The shapes
        are those ``analyze`` reports.

    Raises:
        ValueError: If ``X`` is not ``(N, D)`` for the problem's ``D``; if
            ``Y`` is not 1-D, 2-D or 3-D, or its row count does not match
            ``X``; if ``grid_size < 2``; if ``n_classes`` is not in
            ``[2, N]``; if ``slice_chunk_size`` is below 1; if
            ``degenerate_tol`` is not in ``[0, 1)``; if ``bandwidth`` or
            ``degenerate_bandwidth`` is neither its sentinel string nor a
            positive float; or if any parameter is categorical.
    """
    check_scalars(
        (
            at_least("grid_size", grid_size, 2),
            require(
                0 <= float(degenerate_tol) < 1,
                f"degenerate_tol must be in [0, 1), got {degenerate_tol}",
            ),
            at_least("slice_chunk_size", slice_chunk_size, 1),
        )
    )
    X = validate_inputs(problem, X)
    Y = _validate_output(Y, int(X.shape[0]), problem)
    dims_levels = _categorical_dims(problem)
    if dims_levels:
        names = ", ".join(repr(problem.names[d]) for d, _ in dims_levels)
        raise ValueError(
            "jaxgsa.borgonovo.indices supports continuous parameters only, "
            f"but {names} is categorical. A categorical parameter conditions "
            "on one class per observed level, and both the class count and "
            "the padded class width are read off the sample on the host, so "
            "the layout cannot be built from a tracer. Use "
            "jaxgsa.borgonovo.analyze instead; it supports categorical "
            "parameters and is not traceable for this reason."
        )

    N = X.shape[0]
    M = _plischke_n_classes(N) if n_classes is None else int(n_classes)
    if not 2 <= M <= N:
        raise ValueError(f"n_classes must be in [2, N={N}], got {n_classes}")
    bw_factor = _resolve_sentinel_float(bandwidth, "bandwidth", "silverman")
    degenerate_bw = _resolve_sentinel_float(degenerate_bandwidth, "degenerate_bandwidth", "auto")

    Y_3d, layout = _prepare_Y(Y)
    _, T, K = Y_3d.shape
    Y_cols = Y_3d.reshape(N, T * K)

    # One replicate, the identity permutation: the original sample. The
    # bootstrap axis exists only for the interval and the bias correction,
    # and both are policy.
    all_idx = jnp.arange(N, dtype=jnp.int32)[None, :]
    groups, _, col_order = build_partition_groups(
        problem, X, all_idx, M, dims_levels, method="jaxgsa.borgonovo.indices"
    )
    d_all, s1_all, _, _ = _run_delta_kernel(
        Y_cols,
        all_idx,
        groups,
        col_order,
        grid_size=grid_size,
        bw_factor=bw_factor,
        degenerate_tol=float(degenerate_tol),
        degenerate_bw=degenerate_bw,
        slice_chunk_size=slice_chunk_size,
    )
    D = problem.num_vars
    delta = layout.squeeze(d_all.reshape(1, T, K, D)[0])
    S1 = layout.squeeze(s1_all.reshape(1, T, K, D)[0])
    return delta, S1


def _out_of_range_columns(
    values: Array, num_vars: int, allow_non_finite: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Find parameter columns whose estimates leave the delta range.

    Args:
        values: Estimates with the parameter axis last ``(..., D)``.
        num_vars: Number of parameters D.
        allow_non_finite: Treat a non-finite estimate as acceptable. Set
            only under ``on_invalid="propagate"``, where the caller asked
            for the bad value to reach the indices and already has a
            warning that says so. Refusing it here would answer a
            deliberate request with an unrelated bandwidth complaint.

    Returns:
        A tuple ``(flat, cols)`` where ``flat`` is the ``(-1, D)`` view of
        ``values`` and ``cols`` holds the indices of the failing columns.
    """
    flat = np.asarray(values).reshape(-1, num_vars)
    # A NaN compares False both ways, so the range tests alone never flag
    # one; an infinity does trip them. Both kinds are handled together.
    finite = np.isfinite(flat)
    bad = (flat < -_DELTA_RANGE_TOL) | (flat > 1.0 + _DELTA_RANGE_TOL)
    bad = (bad & finite) if allow_non_finite else (bad | ~finite)
    return flat, np.flatnonzero(bad.any(axis=0))


def _raise_delta_out_of_range(
    problem: Problem,
    delta: Array,
    advice: Callable[[], str],
    allow_non_finite: bool = False,
) -> None:
    """Reject a delta point estimate that leaves ``[0, 1]``.

    Borgonovo's delta is a half L1 distance between probability densities,
    so it lies in ``[0, 1]`` by construction. A value outside that range is
    a failed computation rather than an estimate, so this function raises
    instead of returning it. The tolerance of 0.05 keeps the documented
    small negative excursion of the bias-corrected form
    ``2*d_hat - d_boot`` legal. The function never clips the value, because
    a clipped value is a plausible-looking wrong answer.

    This is the only place the estimator refuses a bandwidth setting, and
    it refuses on the returned number rather than on the configuration. A
    kernel narrower than one grid step does not by itself break the run:
    whether it does depends on where the narrow class sits relative to the
    grid points, which is a property of the data. See
    :func:`_bandwidth_advice`.

    Args:
        problem: Problem definition (for parameter names in the message).
        delta: Delta point estimates with the parameter axis last
            ``(..., D)``.
        advice: Callable returning the sentences that name the knob to
            turn. It is called only when this function raises, so the
            caller pays for the diagnostic only on a failed run.
        allow_non_finite: Accept a non-finite estimate. Set only under
            ``on_invalid="propagate"``; see :func:`_out_of_range_columns`.

    Raises:
        ValueError: If any estimate is below ``-_DELTA_RANGE_TOL``, above
            ``1 + _DELTA_RANGE_TOL``, or, unless ``allow_non_finite``, not
            finite.
    """
    flat, cols = _out_of_range_columns(delta, problem.num_vars, allow_non_finite)
    if cols.size == 0:
        return
    detail = ", ".join(
        f"{problem.names[d]}: [{np.nanmin(flat[:, d]):.3g}, {np.nanmax(flat[:, d]):.3g}]"
        for d in cols
    )
    raise ValueError(
        "jaxgsa: delta is a half L1 distance between densities, so it lies "
        f"in [0, 1]. The estimate left that range for {detail}. This is a "
        "failed computation, not a result. The cause is a conditioning "
        "class the output grid cannot resolve: a class that is a point "
        f"mass, or one much narrower than the grid step. {advice()} The "
        "value is not clipped, because a clipped value would look plausible "
        "and still be wrong."
    )


def _warn_conf_out_of_range(
    problem: Problem, delta_conf: Array, allow_non_finite: bool = False
) -> None:
    """Warn when a delta confidence bound leaves ``[0, 1]``.

    The point estimate is the contract, and :func:`_raise_delta_out_of_range`
    checks it. The interval is only a diagnostic. An out-of-range bound
    degrades that diagnostic without invalidating the estimate, so this
    function warns instead of raising.

    Args:
        problem: Problem definition (for parameter names in the warning).
        delta_conf: Bootstrap interval ``(2, ..., D)`` with the lower bound
            first.
        allow_non_finite: Accept a non-finite bound. Set only under
            ``on_invalid="propagate"``; see :func:`_out_of_range_columns`.

    Warns:
        JaxgsaWarning: If a bound leaves the range, naming the parameters and
            which bound failed.
    """
    parts = []
    for bound, arr in (("lower", delta_conf[0]), ("upper", delta_conf[1])):
        flat, cols = _out_of_range_columns(arr, problem.num_vars, allow_non_finite)
        if cols.size == 0:
            continue
        detail = ", ".join(
            f"{problem.names[d]}: [{np.nanmin(flat[:, d]):.3g}, {np.nanmax(flat[:, d]):.3g}]"
            for d in cols
        )
        parts.append(f"{bound} bound for {detail}")
    if not parts:
        return
    warnings.warn(
        "jaxgsa.borgonovo: delta is defined on [0, 1] but a bootstrap confidence "
        f"bound left that range — {'; '.join(parts)}. The point estimate is "
        "in range, so the index itself stands; the interval is the part to "
        "distrust. Raise n_bootstrap, or raise grid_size if a conditioning "
        "class is near degenerate.",
        stacklevel=3,
        category=JaxgsaWarning,
    )


def _resolve_sentinel_float(value: float | str, name: str, sentinel: str) -> float | None:
    """Validate a "sentinel string, or positive float" argument.

    Both bandwidth arguments follow the same ladder: one named string
    selects a rule, and any other value must be a positive real number that
    is used directly. This is that ladder, written once.

    Args:
        value: The value the caller passed.
        name: Argument name, used in the error message.
        sentinel: The one string the argument accepts, such as
            ``"silverman"`` or ``"auto"``.

    Returns:
        ``None`` when ``value`` is the sentinel string, otherwise
        ``float(value)``.

    Raises:
        ValueError: If ``value`` is neither the sentinel nor a positive
            real number. Booleans are rejected: ``bool`` is an ``int``
            subclass but is never a meaningful factor.
    """
    message = f"{name} must be {sentinel!r} or a positive float, got {value!r}"
    if isinstance(value, str):
        if value == sentinel:
            return None
        raise ValueError(message)
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        factor = float(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not factor > 0:
        raise ValueError(message)
    return factor
