"""Shared equal-frequency rank-partition helpers for given-data methods.

Both the Borgonovo delta and optimal-transport estimators condition on
equal-frequency classes of each input's ordinal rank. The helpers here
build static, padded gather indices once (per input, for every bootstrap
replicate) so the per-column analysis kernels never re-rank the inputs
and all downstream shapes stay static under JIT.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


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
