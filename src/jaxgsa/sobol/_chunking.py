"""Chunk sizing and ragged-chunk padding for the Sobol slice loops.

Both the point-estimate path and the bootstrap path walk the flattened
``T*K`` output slices in chunks and call one jitted kernel per chunk. The two
helpers here are what keeps that loop honest.

**One compilation, not two.** A jitted kernel is traced per input shape, so a
trailing chunk narrower than the rest compiles the same kernel a second time.
The fix is to pad the trailing chunk back to the full width and slice the
answer back to the rows that were asked for. Every slice is independent of
every other under ``vmap``, so the padded lanes cannot reach the real ones and
the sliced answer is bit-for-bit the unpadded one. See ``efast/_analyze.py``,
``morris/_analyze.py`` and ``dgsm/_core.py`` for the same pattern.

**A width from the budget, not a constant.** ``slice_chunk_size=None`` means
"derive one from :func:`jaxgsa._core.batching.get_memory_budget`", which needs
a model of what one slice actually costs. For the point kernels that is the
gathered A, B and AB (and BA) arrays for one slice, which is what
:func:`resolve_point_chunk_size` estimates.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxgsa._core.batching import get_memory_budget

# The estimator holds roughly one working copy of its inputs alive on top of
# the inputs themselves (the centred products and the reduction temporaries),
# so the per-slice estimate carries a factor of two. The bootstrap model in
# ``_bootstrap.py`` uses the same factor, one resample per slice apart.
_POINT_LIVE_COPIES = 2


def pad_slice_axis(a: Array, width: int) -> Array:
    """Zero-pad the leading slice axis of ``a`` up to ``width`` entries.

    Args:
        a: Array whose leading axis indexes output slices.
        width: Target length of the leading axis. Must be at least
            ``a.shape[0]``.

    Returns:
        ``a`` unchanged when it is already ``width`` long, otherwise ``a``
        with zero-filled slices appended. The dtype is preserved, so the
        padded call traces at the same shape *and* the same dtype as a full
        chunk and reuses its compilation.
    """
    n_slices = a.shape[0]
    if n_slices == width:
        return a
    pad = jnp.zeros((width - n_slices, *a.shape[1:]), dtype=a.dtype)
    return jnp.concatenate([a, pad], axis=0)


def resolve_point_chunk_size(
    slice_chunk_size: int | None,
    n_slices: int,
    base_n: int,
    D: int,
    calc_second_order: bool,
    itemsize: int,
) -> int:
    """Resolve how many output slices one point-estimate device call may carry.

    An explicit value is an upper bound the caller chose and is honoured as
    given, capped at the number of slices there are. ``None`` derives a width
    from the active memory budget and the kernel's real working set.

    One slice of the working set is the gathered outputs: ``base_n`` elements
    for each of A and B and ``base_n * D`` for AB, doubled when the design
    also carries BA. That is ``base_n * (D + 2)`` elements, or
    ``base_n * (2D + 2)`` with second order.

    Args:
        slice_chunk_size: Caller's cap on slices per chunk, or ``None`` to
            derive one from :func:`jaxgsa._core.batching.get_memory_budget`.
        n_slices: Total number of flattened (T, K) output slices.
        base_n: N, the number of base samples.
        D: Number of input parameters.
        calc_second_order: Whether the design carries the BA blocks.
        itemsize: Bytes per element of the output dtype.

    Returns:
        A chunk width in ``[1, n_slices]``.

    Raises:
        ValueError: If ``slice_chunk_size`` is given and is below 1.
    """
    if slice_chunk_size is not None:
        if slice_chunk_size < 1:
            raise ValueError(f"slice_chunk_size must be >= 1, got {slice_chunk_size}")
        return max(1, min(slice_chunk_size, n_slices))
    per_slice = base_n * (2 * D + 2 if calc_second_order else D + 2)
    bytes_per_slice = _POINT_LIVE_COPIES * per_slice * itemsize
    budget = get_memory_budget() // max(bytes_per_slice, 1)
    return max(1, min(n_slices, budget))
