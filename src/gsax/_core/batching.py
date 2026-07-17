"""Row-batching helpers for surrogate forward prediction.

Surrogate prediction materializes basis
tensors whose size is linear in the number of prediction rows but carries a
large per-row constant (polynomial terms, B-spline tensor products). At large
``N_new`` a single-shot evaluation can exceed available memory, so predictions
are computed in row batches sized against a transient-memory budget.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array

# Transient-memory budget (bytes) used to derive the automatic batch size.
# 512 MiB keeps the per-batch basis tensors comfortably resident while making
# per-batch dispatch overhead negligible next to the basis construction cost.
DEFAULT_EMULATE_BUDGET_BYTES: int = 512 * 1024**2

# Module-level override for the transient-memory budget, set through
# ``gsax.config.set_memory_budget``. ``None`` means "use the default"; the
# getter below is the single read path so every budget consumer (surrogate
# predict batching, the HDMR slice-chunk derivation, PCE streaming) honours
# the global override. Per-call parameters (``batch_size``/``slice_chunk_size``)
# always take precedence over this budget.
_memory_budget_bytes: int | None = None


def get_memory_budget() -> int:
    """Return the active transient-memory budget in bytes.

    Returns:
        The budget set via :func:`gsax.config.set_memory_budget`, or
        ``DEFAULT_EMULATE_BUDGET_BYTES`` if none was set.
    """
    return DEFAULT_EMULATE_BUDGET_BYTES if _memory_budget_bytes is None else _memory_budget_bytes


def _set_memory_budget(budget_bytes: int | None) -> None:
    """Set (or with ``None`` reset) the global transient-memory budget.

    Internal state mutator behind :func:`gsax.config.set_memory_budget`,
    which performs the input validation.

    Args:
        budget_bytes: New budget in bytes, or ``None`` to restore the default.
    """
    global _memory_budget_bytes
    _memory_budget_bytes = budget_bytes


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
            the active memory budget (:func:`get_memory_budget`).

    Returns:
        Batch size in ``[1, n_rows]`` (``n_rows`` means a single-shot call).

    Raises:
        ValueError: If ``batch_size`` is given and not a positive integer.
    """
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        return min(batch_size, n_rows)
    auto = get_memory_budget() // max(bytes_per_row, 1)
    return max(1, min(n_rows, auto))


def apply_batched(fn: Callable[[Array], Array], X: Array, batch_size: int) -> Array:
    """Apply ``fn`` to row batches of ``X``, assembling the rows along axis 0.

    ``fn`` must be row-independent: its output for a row may not depend on
    which other rows share the batch. ``batch_size >= X.shape[0]`` degrades
    to a plain single-shot call.

    Each finished batch is staged into a host (NumPy) output buffer, so
    device residency stays at one batch plus the final output — a device-side
    chunk list followed by ``jnp.concatenate`` would instead hold the full
    output twice. Under ``jax.jit`` tracing, host staging is impossible, so
    the traced path falls back to a device concatenate (XLA owns buffer
    lifetimes there anyway).

    Args:
        fn: Function mapping an ``(n, ...)`` input batch to an ``(n, ...)``
            output batch.
        X: (N, ...) input rows.
        batch_size: Rows per batch, as returned by ``resolve_batch_size``.

    Returns:
        ``fn(X)`` as a JAX array, computed at most ``batch_size`` rows at a
        time.
    """
    n_rows = X.shape[0]
    if batch_size >= n_rows:
        return fn(X)
    if isinstance(X, jax.core.Tracer):
        return jnp.concatenate(
            [fn(X[i : i + batch_size]) for i in range(0, n_rows, batch_size)], axis=0
        )
    first = fn(X[:batch_size])
    out = np.empty((n_rows, *first.shape[1:]), dtype=first.dtype)
    out[:batch_size] = np.asarray(first)
    for i in range(batch_size, n_rows, batch_size):
        out[i : i + batch_size] = np.asarray(fn(X[i : i + batch_size]))
    return jnp.asarray(out)
