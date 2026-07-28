"""Shared rank-partition helpers for given-data methods.

Both the Borgonovo delta and optimal-transport estimators condition on
classes of each input's value. Continuous inputs use equal-frequency
classes of the ordinal rank; categorical inputs use one class per level
(class sizes are the observed level counts). The helpers here build
static, padded gather indices once (per input, for every bootstrap
replicate) so the per-column analysis kernels never re-rank the inputs
and all downstream shapes stay static under JIT.
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


def _extract_categorical_codes(problem: "Problem", X: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Validate and extract the integer level codes of categorical columns.

    Args:
        problem: Problem definition (may mix marginal kinds).
        X: Input sample matrix ``(N, D)`` on the host.

    Returns:
        ``(cat_dims, codes)`` where ``cat_dims`` lists the categorical
        column indices and ``codes (N, len(cat_dims))`` holds their int64
        level codes.

    Raises:
        ValueError: If a categorical column holds non-integral values or
            codes outside ``[0, L)``.
    """
    from jaxgsa.problem import _categorical_dims

    dims_levels = _categorical_dims(problem)
    cat_dims = [d for d, _ in dims_levels]
    codes = np.empty((X.shape[0], len(cat_dims)), dtype=np.int64)
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
    return cat_dims, codes


def _warn_empty_levels(problem: "Problem", cat_dims: list[int], counts0: np.ndarray) -> None:
    """Warn about declared levels with no observed samples.

    An empty level's conditioning class carries zero weight, so it is
    dropped from the class average; the warning makes the drop visible.

    Args:
        problem: Problem definition.
        cat_dims: Categorical column indices, aligned with ``counts0`` rows.
        counts0: Observed level counts of the original sample ``(Dc, M)``
            (padded level slots beyond a column's own count are ignored).
    """
    from jaxgsa.problem import _categorical_dims

    n_levels = dict(_categorical_dims(problem))
    for j, d in enumerate(cat_dims):
        empty = [int(level) for level in range(n_levels[d]) if counts0[j, level] == 0]
        if empty:
            warnings.warn(
                f"jaxgsa: categorical parameter {problem.names[d]!r} has no "
                f"samples at level(s) {empty}; the empty conditioning "
                "class(es) are dropped (zero weight) for this analysis",
                stacklevel=3,
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
