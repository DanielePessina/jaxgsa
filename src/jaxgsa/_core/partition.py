"""Shared rank-partition helpers for given-data methods.

Both the Borgonovo delta and optimal-transport estimators condition on classes
of each input's value. Continuous inputs use equal-frequency classes of the
ordinal rank. Categorical inputs use one class per level, and the class sizes
are the observed level counts. The helpers here build static, padded gather
indices once, per input and for every bootstrap replicate. The per-column
analysis kernels therefore never re-rank the inputs, and all downstream shapes
stay static under JIT.

:func:`build_partition_groups` is the one entry point the analyzers use.
It returns partition groups in a canonical layout: every group is a
``(cls_idx, counts)`` pair with

- ``cls_idx (R, Dg, Mg, Pg)``: per-replicate global sample indices of
  every class's members (padded entries are clamped duplicates), and
- ``counts (R, Dg, Mg)``: true class sizes, matching the first three axes
  of ``cls_idx``. The continuous layout is the same for every replicate
  and input, so its ``counts`` is a broadcast of one ``(Mg,)`` row; that
  row is about ``1/Pg`` the size of ``cls_idx``, so broadcasting it costs
  little next to the class indices already held at full size.

Kernels index a replicate with plain ``counts[i]`` and derive the
validity mask from the counts with :func:`_mask_from_counts`, so no mask
or per-replicate lookup table is ever materialized over all replicates.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.batching import resolve_batch_size
from jaxgsa._core.warning_types import JaxgsaWarning

if TYPE_CHECKING:
    from jaxgsa.problem import Problem

# One canonical-layout partition group: (cls_idx (R, Dg, Mg, Pg),
# counts (R, Dg, Mg)).
PartitionGroup = tuple[Array, Array]


def _class_layout(N: int, M: int) -> tuple[np.ndarray, np.ndarray]:
    """Build static gather indices for equal-frequency rank classes.

    Class ``j`` holds the samples whose ordinal rank ``r`` (1-based)
    satisfies ``e[j] < r <= e[j+1]`` with the exact integer edges
    ``e[m] = (m * N) // M`` for ``m = 0..M``. This is an exact equal-frequency
    split: the class sizes sum to ``N`` and differ from each other by at most
    one. SALib takes the floor of the float edges ``linspace(0, N, M+1)``
    instead. Float rounding can put one of those edges one row low (for
    example ``N=60, M=22`` gives 29 for edge 11, not 30), so at some
    ``(N, M)`` the classes here can differ from SALib's by one row. Classes
    are padded to the largest size, and the kernels derive the validity mask
    from the sizes (:func:`_mask_from_counts`), so downstream shapes stay
    static.

    Args:
        N: Number of samples.
        M: Number of classes.

    Returns:
        ``(take, sizes)`` where ``take (M, P)`` indexes into a
        rank-sorted array (entries beyond a class's size are clamped) and
        ``sizes (M,)`` holds the true class sizes.
    """
    edges = (np.arange(M + 1, dtype=np.int64) * N) // M
    sizes = np.diff(edges)
    n_pad = int(sizes.max())
    take = edges[:-1, None] + np.arange(n_pad)[None, :]
    take = np.minimum(take, N - 1).astype(np.int32)
    return take, sizes


def _extract_categorical_codes(
    problem: "Problem",
    X: np.ndarray,
    dims_levels: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Validate and extract the integer level codes of categorical columns.

    Args:
        problem: Problem definition (for parameter names in errors).
        X: Input sample matrix ``(N, D)`` on the host.
        dims_levels: ``(dimension index, level count)`` pairs from
            :func:`jaxgsa.problem._categorical_dims`.

    Returns:
        ``codes (N, len(dims_levels))``: the int64 level codes, one column
        per categorical parameter in ``dims_levels`` order.

    Raises:
        ValueError: If a categorical column holds non-integral values or
            codes outside ``[0, L)``.
    """
    codes = np.empty((X.shape[0], len(dims_levels)), dtype=np.int64)
    for j, (d, n_levels) in enumerate(dims_levels):
        col = np.asarray(X[:, d], dtype=np.float64)
        rounded = np.round(col)
        if not (
            np.all(np.isfinite(col))
            and np.all(col == rounded)
            and np.all(rounded >= 0)
            and np.all(rounded < n_levels)
        ):
            raise ValueError(
                f"Column {d} ({problem.names[d]!r}) is categorical with "
                f"{n_levels} levels; X must hold the integer level codes "
                f"0 .. {n_levels - 1} (as floats)"
            )
        codes[:, j] = rounded.astype(np.int64)
    return codes


def _warn_empty_levels(
    problem: "Problem",
    dims_levels: tuple[tuple[int, int], ...],
    counts0: np.ndarray,
    method: str,
) -> None:
    """Warn about declared levels with no observed samples.

    An empty level's conditioning class carries zero weight, so it is
    dropped from the class average; the warning makes the drop visible.

    Args:
        problem: Problem definition (for parameter names in the warning).
        dims_levels: ``(dimension index, level count)`` pairs, aligned with
            ``counts0`` rows.
        counts0: Observed level counts of the original sample ``(Dc, M)``
            (padded level slots beyond a column's own count are ignored).
        method: Fully qualified analyzer name to prefix the warning with.
    """
    for j, (d, n_levels) in enumerate(dims_levels):
        empty = [int(level) for level in range(n_levels) if counts0[j, level] == 0]
        if empty:
            warnings.warn(
                f"{method}: categorical parameter {problem.names[d]!r} has no "
                f"samples at level(s) {empty}; the empty conditioning "
                "class(es) are dropped (zero weight) for this analysis",
                # analyze() -> plan_partition() -> here; point at
                # the caller of analyze().
                stacklevel=4,
                category=JaxgsaWarning,
            )


def _bucket_width(P: int, N: int) -> int:
    """Round a class width up to a coarse grid, so shapes repeat across datasets.

    A categorical class tensor is padded to its largest class. That width
    changes with every dataset, and each new width makes JAX compile the
    kernels again. This rounds the width up to a quarter-octave grid
    ``2 ** (k / 4)``, then up to a multiple of 8, and never above ``N``. The
    padding is at most about 19% plus 7 slots. The padded slots are masked
    through the class counts, so the extra width changes no index beyond
    floating-point summation order.

    Args:
        P: Largest true class size (at least 1).
        N: Number of samples. No class can hold more.

    Returns:
        The bucketed width ``P_b`` with ``P <= P_b <= N``.
    """
    P = max(int(P), 1)
    # The round() keeps an exact power of two on its own grid point despite
    # float error in log2.
    k = math.ceil(round(4.0 * math.log2(P), 9))
    width = math.ceil(round(2.0 ** (k / 4.0), 9))
    width = -(-width // 8) * 8
    return min(N, max(P, width))


def _categorical_counts(codes: np.ndarray, all_idx: np.ndarray, M: int) -> np.ndarray:
    """Count the rows at every level, per replicate and per column.

    Args:
        codes: Integer level codes ``(N, Dc)`` of the original sample.
        all_idx: Replicate row indices ``(R, N)``.
        M: Number of level slots (the largest declared level count).

    Returns:
        ``counts (R, Dc, M)``, zero for empty or padded level slots.
    """
    R = all_idx.shape[0]
    Dc = codes.shape[1]
    counts = np.zeros((R, Dc, M), dtype=np.int64)
    row_offsets = np.arange(R, dtype=np.int64)[:, None] * M
    for j in range(Dc):
        v = codes[:, j][all_idx]  # (R, N) resampled codes
        counts[:, j] = np.bincount((v + row_offsets).ravel(), minlength=R * M).reshape(R, M)
    return counts


def _categorical_class_layout(
    codes: np.ndarray,
    all_idx: np.ndarray,
    n_levels: list[int],
    n_pad: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the per-replicate, per-column class layout of categorical columns.

    Each column gets one class per declared level. A class holds exactly the
    rows observed at that level, so class sizes vary across levels, columns,
    and bootstrap replicates. Members keep their original row order within a
    class, because the sort is stable; that is the ``ties.method="first"``
    rank convention. Everything is padded to one width, so the downstream
    kernel shapes stay static.

    Args:
        codes: Integer level codes ``(N, Dc)`` of the original sample.
        all_idx: Replicate row indices ``(R, N)`` (row 0 is the identity).
        n_levels: Declared level count per column.
        n_pad: Padded class width ``P``. ``None`` uses the largest observed
            class in ``all_idx``. A caller that builds the layout in blocks
            passes one width for all blocks, so every block (and every
            replicate) is padded the same way.

    Returns:
        ``(cls_idx, counts)`` where ``cls_idx (R, Dc, M, P)`` holds
        global sample indices per class (``M = max(n_levels)``, entries
        beyond a class's size are clamped) and ``counts (R, Dc, M)``
        holds the true class sizes (0 for empty or padded level slots);
        the kernels derive the validity mask from the counts
        (:func:`_mask_from_counts`).
    """
    R, N = all_idx.shape
    Dc = codes.shape[1]
    M = max(n_levels)

    counts = _categorical_counts(codes, all_idx, M)

    P = int(counts.max()) if n_pad is None else n_pad
    arange_p = np.arange(P, dtype=np.int64)
    cls_idx = np.empty((R, Dc, M, P), dtype=np.int32)
    for j in range(Dc):
        v = codes[:, j][all_idx]
        # Stable sort groups each level into a contiguous slice whose edges
        # are the cumulative level counts of the resample.
        orders = np.argsort(v, axis=1, kind="stable")  # (R, N)
        sorted_global = np.take_along_axis(all_idx, orders, axis=1)  # (R, N)
        edges = np.concatenate(
            [np.zeros((R, 1), dtype=np.int64), np.cumsum(counts[:, j], axis=1)], axis=1
        )
        take = np.minimum(edges[:, :-1, None] + arange_p[None, None, :], N - 1)  # (R, M, P)
        gathered = np.take_along_axis(sorted_global, take.reshape(R, M * P), axis=1)
        cls_idx[:, j] = gathered.reshape(R, M, P).astype(np.int32)
    return cls_idx, counts


@jax.jit
def _build_class_indices(X: Array, all_idx: Array, take: Array) -> Array:
    """Return the global sample indices of every class, for every replicate.

    Computed once per analysis, never per output-column chunk, so the
    per-column kernel never re-ranks the inputs.

    Args:
        X: Input sample matrix ``(N, D)``.
        all_idx: Replicate row indices ``(R, N)`` (row 0 is the identity).
        take: Static per-class gather indices ``(M, P)`` from
            :func:`_class_layout`.

    Returns:
        Class indices ``(R, D, M, P)`` into the original sample.
    """

    def _one_replicate(r: Array) -> Array:
        # Rank on the ORIGINAL X (never downcast): ordinal ranks == a stable
        # argsort, so each class is a contiguous slice of the rank-sorted
        # resample; gathering ``r[orders]`` yields global indices of the
        # class members.
        orders = jnp.argsort(X[r], axis=0)  # (N, D)
        return r[orders].T[:, take]  # (D, M, P)

    return jax.vmap(_one_replicate)(all_idx)


def _mask_from_counts(counts: Array, n_pad: int) -> Array:
    """Derive the class-validity mask ``(..., M, P)`` from true class sizes.

    Entry ``p`` of a class is valid when ``p < count``. Deriving the mask
    inside the kernel from the small counts array avoids materializing a
    boolean ``(R, D, M, P)`` tensor over all replicates.

    Args:
        counts: ``(..., M)`` true class sizes.
        n_pad: Padded class width ``P``.

    Returns:
        ``(..., M, P)`` boolean mask, ``True`` for a real class member.
    """
    return jnp.arange(n_pad)[(None,) * counts.ndim] < counts[..., None]


@dataclass(frozen=True)
class PartitionPlan:
    """Everything needed to build the class tensors of any block of replicates.

    The class-index tensor ``(R, D, M, P)`` can be far larger than the
    memory budget when ``R`` is a few hundred bootstrap replicates. The plan
    holds the small, replicate-independent parts (counts, the global
    categorical pad width, the column order). :meth:`groups_for` then builds
    the big tensors for a chosen block of replicates only.

    Attributes:
        all_idx: Replicate row indices ``(R, N)`` (row 0 is the identity).
        group_levels: One entry per group, ``None`` for the continuous group
            and the declared level count per column for the categorical one.
        col_order: Gather that restores the problem's column order, or
            ``None`` when one group is already in order.
        cat_counts0: Observed level counts of the original sample
            ``(Dc, M)``, or ``None`` without categorical columns.
        bytes_per_replicate: Estimated bytes one replicate needs while its
            class tensors are built (the tensors plus host transients).
    """

    all_idx: Array
    group_levels: list[list[int] | None]
    col_order: Array | None
    cat_counts0: np.ndarray | None
    bytes_per_replicate: int
    _X_cont: Array | None
    _take: Array | None
    _sizes: np.ndarray | None
    _n_cont: int
    _codes: np.ndarray | None
    _levels: list[int]
    _cat_pad: int
    _all_idx_host: np.ndarray | None

    def groups_for(self, rows: slice) -> list[PartitionGroup]:
        """Build the canonical partition groups for a block of replicates.

        Args:
            rows: Slice of the replicate axis.

        Returns:
            One ``(cls_idx, counts)`` pair per group (continuous first, then
            categorical), with ``R`` replaced by the block length.
        """
        idx = self.all_idx[rows]
        n_rep = idx.shape[0]
        groups: list[PartitionGroup] = []
        if self._take is not None and self._sizes is not None:
            # Rank the inputs once per block (never per output-column chunk);
            # the kernels reuse the class indices across chunks.
            cls_idx = _build_class_indices(self._X_cont, idx, self._take)
            counts = jnp.broadcast_to(
                jnp.asarray(self._sizes), (n_rep, self._n_cont, self._sizes.shape[0])
            )
            groups.append((cls_idx, counts))
        if self._codes is not None and self._all_idx_host is not None:
            cat_cls, cat_counts = _categorical_class_layout(
                self._codes, self._all_idx_host[rows], self._levels, n_pad=self._cat_pad
            )
            groups.append((jnp.asarray(cat_cls), jnp.asarray(cat_counts)))
        return groups


def plan_partition(
    problem: "Problem",
    X: Array,
    all_idx: Array,
    M: int,
    dims_levels: tuple[tuple[int, int], ...],
    method: str = "jaxgsa",
) -> PartitionPlan:
    """Plan the canonical partition layout for a given-data analysis.

    One group per column kind. Continuous columns share one
    equal-frequency rank layout, identical for every replicate and input,
    broadcast to the full ``counts`` shape. Categorical columns get one
    class per level, with sizes that vary per column and per bootstrap
    resample. Grouping keeps the padded class tensors rectangular without
    padding continuous classes up to a categorical level size (or vice
    versa). The categorical pad width comes from the largest class over ALL
    replicates, rounded up by :func:`_bucket_width`, so every block of
    replicates has the same shape and gives bitwise the same result as one
    block.

    Args:
        problem: Problem definition (names for errors/warnings).
        X: Input sample matrix ``(N, D)`` (device array).
        all_idx: Replicate row indices ``(R, N)`` (row 0 is the identity).
        M: Number of equal-frequency classes for the continuous columns.
        dims_levels: ``(dimension index, level count)`` pairs from
            :func:`jaxgsa.problem._categorical_dims`.
        method: Fully qualified analyzer name for the empty-level warning.

    Returns:
        The :class:`PartitionPlan`.

    Raises:
        ValueError: If a categorical column of ``X`` holds values other
            than its integer level codes.

    Warns:
        JaxgsaWarning: If a declared categorical level has no observed
            samples (its class is dropped with zero weight).
    """
    N = X.shape[0]
    cat_dims = [d for d, _ in dims_levels]
    cont_dims = [d for d in range(problem.n_params) if d not in set(cat_dims)]

    group_levels: list[list[int] | None] = []
    group_dims: list[int] = []  # local only: feeds col_order below
    bytes_per_rep = 0
    X_cont = take = sizes = None
    if cont_dims:
        take_np, sizes = _class_layout(N, M)
        take = jnp.asarray(take_np)
        X_cont = X[:, jnp.asarray(cont_dims)] if cat_dims else X
        group_levels.append(None)
        group_dims.extend(cont_dims)
        # cls_idx (int32) plus the per-column argsort and gather transients.
        bytes_per_rep += 4 * len(cont_dims) * int(take_np.shape[0]) * int(
            take_np.shape[1]
        ) + N * len(cont_dims) * (X.dtype.itemsize + 8)
    codes = all_idx_host = cat_counts0 = None
    levels: list[int] = []
    cat_pad = 0
    if cat_dims:
        all_idx_host = np.asarray(all_idx)
        codes = _extract_categorical_codes(problem, np.asarray(X), dims_levels)
        levels = [n_levels for _, n_levels in dims_levels]
        counts_all = _categorical_counts(codes, all_idx_host, max(levels))
        cat_counts0 = counts_all[0]
        _warn_empty_levels(problem, dims_levels, cat_counts0, method)
        cat_pad = _bucket_width(int(counts_all.max()), N)
        # cls_idx (int32) plus the int64 take/gather transients on the host.
        bytes_per_rep += len(cat_dims) * (max(levels) * cat_pad * 20 + N * 24)
        group_levels.append(levels)
        group_dims.extend(cat_dims)
    # Kernel outputs concatenate the groups on the input axis; this gather
    # restores the problem's column order (None when one group is already
    # in order).
    col_order = jnp.asarray(np.argsort(group_dims)) if len(group_levels) > 1 else None
    return PartitionPlan(
        all_idx=all_idx,
        group_levels=group_levels,
        col_order=col_order,
        cat_counts0=cat_counts0,
        bytes_per_replicate=bytes_per_rep,
        _X_cont=X_cont,
        _take=take,
        _sizes=sizes,
        _n_cont=len(cont_dims),
        _codes=codes,
        _levels=levels,
        _cat_pad=cat_pad,
        _all_idx_host=all_idx_host,
    )


def replicate_blocks(plan: PartitionPlan) -> Iterator[tuple[Array, list[PartitionGroup]]]:
    """Yield ``(idx_block, groups_block)`` for near-equal blocks of replicates.

    The block size comes from the memory budget
    (:func:`jaxgsa._core.batching.resolve_batch_size`). The replicates are
    split into near-equal blocks with no padding, so there are at most two
    block shapes. When everything fits the budget, there is one block and
    the result is the same as building all groups at once. The class tensors
    of one block are freed before the next block is built.

    Args:
        plan: Plan from :func:`plan_partition`.

    Yields:
        The row indices ``(Rb, N)`` of the block and its partition groups.
    """
    R = int(plan.all_idx.shape[0])
    rb = resolve_batch_size(plan.bytes_per_replicate, R, None)
    n_blocks = -(-R // rb)
    if n_blocks <= 1:
        yield plan.all_idx, plan.groups_for(slice(None))
        return
    bounds = np.cumsum([0] + [len(c) for c in np.array_split(np.arange(R), n_blocks)])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        rows = slice(int(lo), int(hi))
        yield plan.all_idx[rows], plan.groups_for(rows)


def build_partition_groups(
    problem: "Problem",
    X: Array,
    all_idx: Array,
    M: int,
    dims_levels: tuple[tuple[int, int], ...],
    method: str = "jaxgsa",
) -> tuple[list[PartitionGroup], list[list[int] | None], Array | None]:
    """Build the canonical partition-group layout for all replicates at once.

    Wraps :func:`plan_partition` and builds one block holding every
    replicate. Use :func:`replicate_blocks` instead when ``R`` is large.

    Args:
        problem: Problem definition (names for errors/warnings).
        X: Input sample matrix ``(N, D)`` (device array).
        all_idx: Replicate row indices ``(R, N)`` (row 0 is the identity).
        M: Number of equal-frequency classes for the continuous columns.
        dims_levels: ``(dimension index, level count)`` pairs from
            :func:`jaxgsa.problem._categorical_dims`.
        method: Fully qualified analyzer name for the empty-level warning.

    Returns:
        ``(groups, group_levels, col_order)``. ``groups`` lists the
        canonical ``(cls_idx, counts)`` layouts (continuous first, then
        categorical; either may be absent). ``group_levels`` has one entry
        per group, in the same order: ``None`` for the continuous group,
        the declared level count per column for the categorical group.
        ``col_order`` is the gather that restores the problem's column
        order, or ``None`` when one group is already in order.

    Raises:
        ValueError: If a categorical column of ``X`` holds values other
            than its integer level codes.

    Warns:
        JaxgsaWarning: If a declared categorical level has no observed
            samples (its class is dropped with zero weight).
    """
    plan = plan_partition(problem, X, all_idx, M, dims_levels, method)
    return plan.groups_for(slice(None)), plan.group_levels, plan.col_order
