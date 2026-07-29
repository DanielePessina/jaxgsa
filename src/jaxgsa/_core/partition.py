"""Shared rank-partition helpers for given-data methods.

Both the Borgonovo delta and optimal-transport estimators condition on
classes of each input's value. Continuous inputs use equal-frequency
classes of the ordinal rank; categorical inputs use one class per level
(class sizes are the observed level counts). The helpers here build
static, padded gather indices once (per input, for every bootstrap
replicate) so the per-column analysis kernels never re-rank the inputs
and all downstream shapes stay static under JIT.

:func:`build_partition_groups` is the one entry point the analyzers use.
It returns partition *groups* in a canonical layout: every group is a
``(cls_idx, counts)`` pair with

- ``cls_idx (R, Dg, Mg, Pg)``: per-replicate global sample indices of
  every class's members (padded entries are clamped duplicates), and
- ``counts (R1, G1, Mg)``: true class sizes, where ``R1`` is ``R`` or 1
  and ``G1`` is ``Dg`` or 1. A length-1 leading axis means the layout is
  shared (broadcast) across replicates and/or inputs; it is never tiled,
  so the shared continuous layout stays O(M) rather than O(R * D * M).

Kernels slice one replicate with :func:`_replicate_slice` and derive the
validity mask from the counts with :func:`_mask_from_counts`, so no mask
or per-replicate lookup table is ever materialized over all replicates.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

if TYPE_CHECKING:
    from jaxgsa.problem import Problem

# One canonical-layout partition group: (cls_idx (R, Dg, Mg, Pg),
# counts (R-or-1, Dg-or-1, Mg)).
PartitionGroup = tuple[Array, Array]


def _class_layout(N: int, M: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build static gather indices for equal-frequency rank classes.

    Class ``j`` holds the samples whose ordinal rank ``r`` (1-based)
    satisfies ``m[j] < r <= m[j+1]`` with ``m = linspace(0, N, M+1)`` --
    the same membership rule as SALib. Because floor of the shared float
    edges is used on both sides, class sizes (which differ by at most one)
    match SALib exactly; classes are padded to the largest size with a
    validity mask so downstream shapes stay static.

    Args:
        N: Number of samples.
        M: Number of classes.

    Returns:
        ``(take, mask, sizes)`` where ``take (M, P)`` indexes into a
        rank-sorted array (entries beyond a class's size are clamped),
        ``mask (M, P)`` flags valid entries, and ``sizes (M,)`` holds the
        true class sizes.
    """
    edges = np.floor(np.linspace(0.0, N, M + 1)).astype(np.int64)
    sizes = np.diff(edges)
    n_pad = int(sizes.max())
    take = edges[:-1, None] + np.arange(n_pad)[None, :]
    mask = np.arange(n_pad)[None, :] < sizes[:, None]
    take = np.minimum(take, N - 1).astype(np.int32)
    return take, mask, sizes


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
    """
    for j, (d, n_levels) in enumerate(dims_levels):
        empty = [int(level) for level in range(n_levels) if counts0[j, level] == 0]
        if empty:
            warnings.warn(
                f"jaxgsa: categorical parameter {problem.names[d]!r} has no "
                f"samples at level(s) {empty}; the empty conditioning "
                "class(es) are dropped (zero weight) for this analysis",
                # analyze() -> build_partition_groups() -> here; point at
                # the caller of analyze().
                stacklevel=4,
            )


def _categorical_class_layout(
    codes: np.ndarray,
    all_idx: np.ndarray,
    n_levels: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-replicate, per-column class layout for categorical columns.

    Each column gets one class per declared level; a class holds exactly
    the rows observed at that level, so class sizes vary across levels,
    columns, and bootstrap replicates. Members keep their original row
    order within a class (stable sort — the ``ties.method="first"`` rank
    convention). Everything is padded to the largest observed class so the
    downstream kernel shapes stay static.

    Args:
        codes: Integer level codes ``(N, Dc)`` of the original sample.
        all_idx: Replicate row indices ``(R, N)`` (row 0 is the identity).
        n_levels: Declared level count per column.

    Returns:
        ``(cls_idx, mask, counts)`` where ``cls_idx (R, Dc, M, P)`` holds
        global sample indices per class (``M = max(n_levels)``, entries
        beyond a class's size are clamped), ``mask (R, Dc, M, P)`` flags
        valid entries, and ``counts (R, Dc, M)`` holds the true class
        sizes (0 for empty or padded level slots).
    """
    R, N = all_idx.shape
    Dc = codes.shape[1]
    M = max(n_levels)

    counts = np.zeros((R, Dc, M), dtype=np.int64)
    row_offsets = np.arange(R, dtype=np.int64)[:, None] * M
    for j in range(Dc):
        v = codes[:, j][all_idx]  # (R, N) resampled codes
        counts[:, j] = np.bincount((v + row_offsets).ravel(), minlength=R * M).reshape(R, M)

    P = int(counts.max())
    arange_p = np.arange(P, dtype=np.int64)
    cls_idx = np.empty((R, Dc, M, P), dtype=np.int32)
    mask = arange_p[None, None, None, :] < counts[..., None]
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
    return cls_idx, mask, counts


@jax.jit
def _build_class_indices(X: Array, all_idx: Array, take: Array) -> Array:
    """Global sample indices of every class, for every replicate.

    Computed once (never per output-column chunk) so the per-column kernel
    never re-ranks the inputs.

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


def _replicate_slice(arr: Array, i: Array) -> Array:
    """Select replicate ``i`` from a canonical ``(R-or-1, ...)`` array.

    A length-1 leading axis marks a layout shared by every replicate; the
    static Python branch keeps the shared case free of dynamic gathers.
    """
    return arr[i] if arr.shape[0] > 1 else arr[0]


def _mask_from_counts(counts: Array, n_pad: int) -> Array:
    """Class-validity mask ``(..., M, P)`` derived from true class sizes.

    Entry ``p`` of a class is valid iff ``p < count``. Deriving the mask
    in-kernel from the (small) counts avoids materializing a boolean
    ``(R, D, M, P)`` tensor over all replicates.
    """
    return jnp.arange(n_pad)[(None,) * counts.ndim] < counts[..., None]


def build_partition_groups(
    problem: "Problem",
    X: Array,
    all_idx: Array,
    M: int,
    dims_levels: tuple[tuple[int, int], ...],
) -> tuple[list[PartitionGroup], list[int], Array | None]:
    """Build the canonical partition-group layout for a given-data analysis.

    One group per column kind. Continuous columns share one
    equal-frequency rank layout, identical for every replicate, so its
    ``counts`` carry length-1 leading axes. Categorical columns get one
    class per level, with sizes that vary per column and per bootstrap
    resample, so their ``counts`` carry full ``(R, Dc)`` leading axes.
    Grouping keeps the padded class tensors rectangular without padding
    continuous classes up to a categorical level size (or vice versa).

    Args:
        problem: Problem definition (names for errors/warnings).
        X: Input sample matrix ``(N, D)`` (device array).
        all_idx: Replicate row indices ``(R, N)`` (row 0 is the identity).
        M: Number of equal-frequency classes for the continuous columns.
        dims_levels: ``(dimension index, level count)`` pairs from
            :func:`jaxgsa.problem._categorical_dims`.

    Returns:
        ``(groups, group_dims, col_order)``. ``groups`` lists the
        canonical ``(cls_idx, counts)`` layouts (continuous first, then
        categorical; either may be absent). ``group_dims`` gives the
        problem column index behind each kernel output column (groups
        concatenated). ``col_order`` is the gather that restores the
        problem's column order, or ``None`` when one group is already in
        order.

    Raises:
        ValueError: If a categorical column of ``X`` holds values other
            than its integer level codes.

    Warns:
        UserWarning: If a declared categorical level has no observed
            samples (its class is dropped with zero weight).
    """
    N = X.shape[0]
    cat_dims = [d for d, _ in dims_levels]
    cont_dims = [d for d in range(problem.num_vars) if d not in set(cat_dims)]

    groups: list[PartitionGroup] = []
    group_dims: list[int] = []
    if cont_dims:
        take_np, _, sizes_np = _class_layout(N, M)
        X_cont = X[:, jnp.asarray(cont_dims)] if cat_dims else X
        # Rank the inputs once for every replicate (never per output-column
        # chunk); the kernels reuse the class indices across chunks.
        cls_idx = _build_class_indices(X_cont, all_idx, jnp.asarray(take_np))
        groups.append((cls_idx, jnp.asarray(sizes_np)[None, None, :]))
        group_dims.extend(cont_dims)
    if cat_dims:
        codes = _extract_categorical_codes(problem, np.asarray(X), dims_levels)
        levels = [n_levels for _, n_levels in dims_levels]
        cat_cls_np, _, cat_counts_np = _categorical_class_layout(
            codes, np.asarray(all_idx), levels
        )
        _warn_empty_levels(problem, dims_levels, cat_counts_np[0])
        groups.append((jnp.asarray(cat_cls_np), jnp.asarray(cat_counts_np)))
        group_dims.extend(cat_dims)
    # Kernel outputs concatenate the groups on the input axis; this gather
    # restores the problem's column order (None when one group is already
    # in order).
    col_order = jnp.asarray(np.argsort(group_dims)) if len(groups) > 1 else None
    return groups, group_dims, col_order
