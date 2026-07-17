"""Row-batching helpers for surrogate forward prediction.

Surrogate prediction materializes basis
tensors whose size is linear in the number of prediction rows but carries a
large per-row constant (polynomial terms, B-spline tensor products). At large
``N_new`` a single-shot evaluation can exceed available memory, so predictions
are computed in row batches sized against a transient-memory budget.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

# Transient-memory budget (bytes) used to derive the automatic batch size.
# 512 MiB keeps the per-batch basis tensors comfortably resident while making
# per-batch dispatch overhead negligible next to the basis construction cost.
DEFAULT_EMULATE_BUDGET_BYTES: int = 512 * 1024**2


def resolve_batch_size(
    bytes_per_row: int,
    n_rows: int,
    batch_size: int | None,
) -> int:
    """Resolve the number of prediction rows to process per batch.

    Args:
        bytes_per_row: Estimated transient memory needed to emulate a single
            row (basis tensors plus contraction intermediates).
        n_rows: Total number of prediction rows.
        batch_size: User-requested batch size, or ``None`` to derive one from
            ``DEFAULT_EMULATE_BUDGET_BYTES``.

    Returns:
        Batch size in ``[1, n_rows]`` (``n_rows`` means a single-shot call).

    Raises:
        ValueError: If ``batch_size`` is given and not a positive integer.
    """
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        return min(batch_size, n_rows)
    auto = DEFAULT_EMULATE_BUDGET_BYTES // max(bytes_per_row, 1)
    return max(1, min(n_rows, auto))


def apply_batched(fn: Callable[[Array], Array], X: Array, batch_size: int) -> Array:
    """Apply ``fn`` to row batches of ``X``, concatenating along axis 0.

    ``fn`` must be row-independent: its output for a row may not depend on
    which other rows share the batch. ``batch_size >= X.shape[0]`` degrades
    to a plain single-shot call.

    Args:
        fn: Function mapping an ``(n, ...)`` input batch to an ``(n, ...)``
            output batch.
        X: (N, ...) input rows.
        batch_size: Rows per batch, as returned by ``resolve_batch_size``.

    Returns:
        ``fn(X)``, computed at most ``batch_size`` rows at a time.
    """
    n_rows = X.shape[0]
    if batch_size >= n_rows:
        return fn(X)
    return jnp.concatenate(
        [fn(X[i : i + batch_size]) for i in range(0, n_rows, batch_size)], axis=0
    )
