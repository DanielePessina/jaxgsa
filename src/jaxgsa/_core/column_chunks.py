"""Walk the output columns of a traced kernel in fixed-width chunks.

A Python loop over column chunks is fine when every chunk is its own jitted
call: each call frees its buffers before the next one runs. Under an outer
``jax.jit`` the same loop is traced into one program, and XLA keeps every
chunk's working set live at once, so the chunking bounds nothing. A
``jax.lax.map`` over the chunks runs them one after another inside one
program, and the transient memory stays one chunk wide however the caller
transforms it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


def map_column_chunks(fn: Callable[[Array], Any], Y_cols: Array, chunk: int, *, axis: int) -> Any:
    """Apply ``fn`` to fixed-width column chunks of ``Y_cols`` in sequence.

    Call this inside ``jit``. ``fn`` sees ``(N, chunk)`` slices only, so it
    traces once. When ``chunk`` does not divide the column count, the last
    chunk is taken flush with the end, so it overlaps the one before it.
    Only its new columns are kept. No column is padded or copied for that,
    and an output column depends only on its own input column, so the
    overlap changes no value. Unused overlap outputs carry a zero cotangent,
    so the gradient is unchanged too.

    Args:
        fn: Maps an ``(N, chunk)`` column slice to a pytree of arrays whose
            column axis is ``axis``.
        Y_cols: Output columns, shape ``(N, total)``.
        chunk: Columns per step. Values at or above ``total`` run ``fn``
            once on the whole array.
        axis: Position of the column axis in every output leaf of ``fn``.

    Returns:
        The outputs of ``fn`` for all ``total`` columns, in column order.
    """
    total = Y_cols.shape[1]
    if chunk >= total:
        return fn(Y_cols)
    n_chunks = -(-total // chunk)
    tail = total - (n_chunks - 1) * chunk
    starts = jnp.minimum(jnp.arange(n_chunks, dtype=jnp.int32) * chunk, total - chunk)
    outs = jax.lax.map(
        lambda start: fn(jax.lax.dynamic_slice_in_dim(Y_cols, start, chunk, axis=1)), starts
    )

    def _assemble(leaf: Array) -> Array:
        # leaf is (n_chunks, *pre, chunk, *post), with len(pre) == axis.
        body = jnp.moveaxis(leaf[:-1], 0, axis)
        shape = body.shape[:axis] + ((n_chunks - 1) * chunk,) + body.shape[axis + 2 :]
        last = jax.lax.slice_in_dim(leaf[-1], chunk - tail, chunk, axis=axis)
        return jnp.concatenate([body.reshape(shape), last], axis=axis)

    return jax.tree.map(_assemble, outs)
