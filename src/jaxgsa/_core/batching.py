"""Row-batching helpers, sized against the transient-memory budget.

:func:`resolve_batch_size` is the library-wide resolver: every method that
derives a ``None`` batch width from :func:`get_memory_budget` routes through
it or follows its convention — explicit widths win over the budget, ``None``
derives one from it.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array

# Default transient-memory budget, in bytes, for deriving an automatic batch
# size. 512 MiB keeps the per-batch tensors comfortably resident while making
# per-batch dispatch overhead negligible next to the work done per batch.
_DEFAULT_MEMORY_BUDGET_BYTES: int = 512 * 1024**2

# Module-level override for the transient-memory budget, set through
# ``jaxgsa.config.set_memory_budget``. ``None`` means "use the default"; the
# getter below is the single read path so every budget consumer (surrogate
# predict batching, the HDMR and PAWN slice-chunk derivations, PCE and HDMR
# streaming) honours the global override. Per-call parameters
# (``batch_size``/``slice_chunk_size``) always take precedence over this budget.
#
# The budget is always bytes in here. ``jaxgsa.config.set_memory_budget``
# accepts a ``unit`` and converts to bytes before it reaches this module, so
# unit handling lives at the public boundary only.
_memory_budget_bytes: int | None = None


def get_memory_budget() -> int:
    """Return the active transient-memory budget in bytes.

    Returns:
        The budget set via :func:`jaxgsa.config.set_memory_budget`, or
        ``_DEFAULT_MEMORY_BUDGET_BYTES`` if none was set.
    """
    return _DEFAULT_MEMORY_BUDGET_BYTES if _memory_budget_bytes is None else _memory_budget_bytes


def _set_memory_budget(budget_bytes: int) -> None:
    """Set the global transient-memory budget.

    Internal state mutator behind :func:`jaxgsa.config.set_memory_budget`,
    which performs the input validation. To go back to the default, read
    :func:`get_memory_budget` first and set that value again: it resolves
    the unset state to ``_DEFAULT_MEMORY_BUDGET_BYTES``.

    Args:
        budget_bytes: New budget in bytes.
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


def pad_rows(A: Array, n_pad: int) -> Array:
    """Pad ``A`` with ``n_pad`` zero rows on the sample axis.

    Used by a streamed fit (HDMR, PCE) to round the row count up to a whole
    number of batches, so every batch traces at the same shape and the step
    function compiles once. The padded rows carry zero weight in the caller's
    own mask, so they never change the fit.

    Args:
        A: Array to pad, sample axis first.
        n_pad: Zero rows to append. ``0`` returns ``A`` unchanged.

    Returns:
        ``A`` with ``n_pad`` zero rows appended.
    """
    if n_pad == 0:
        return A
    return jnp.concatenate([A, jnp.zeros((n_pad, *A.shape[1:]), dtype=A.dtype)], axis=0)


def apply_batched(fn: Callable[[Array], Array], X: Array, batch_size: int) -> Array:
    """Apply ``fn`` to row batches of ``X``, assembling the rows along axis 0.

    ``fn`` must be row-independent: its output for a row may not depend on
    which other rows share the batch. ``batch_size >= X.shape[0]`` degrades
    to a plain single-shot call.

    Each finished batch is staged into a host (NumPy) output buffer, so device
    residency stays at one batch plus the final output. A device-side chunk
    list followed by ``jnp.concatenate`` would instead hold the full output
    twice. Host staging is impossible under ``jax.jit`` tracing, so the traced
    path falls back to a device concatenate. XLA owns the buffer lifetimes
    there anyway.

    Args:
        fn: Function mapping an ``(n, ...)`` input batch to an ``(n, ...)``
            output batch.
        X: ``(N, ...)`` input rows.
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
