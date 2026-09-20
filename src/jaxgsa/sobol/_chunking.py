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

**Widths from the budget, not constants.** ``slice_chunk_size=None`` means
"derive one from :func:`jaxgsa._core.batching.get_memory_budget`", which needs
a model of what one slice actually costs. :func:`slice_elements` is that
model and both paths share it: the gathered A, B and AB (and BA) arrays, plus
a second live working copy. Second order used to be quadratic in ``D``
because every second-order estimator formed an ``(N, D, D)`` outer product;
since ``_estimators.second_order_kernel`` writes the cross-moment as an
explicit matrix product (``BA.T @ AB``), that term no longer materialises and
the per-slice cost is linear in ``D`` for both orders. Bootstrap also derives
a resample width from the same model, so a small output chunk cannot leave
an arbitrarily large ``R`` axis in one device call.

The budget is a transient working-set estimate. Resident input arrays,
already assembled results, and the pre-generated resampling index matrix are
not included, so it is a sizing aid rather than a hard process-memory cap.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxgsa._core.batching import get_memory_budget

# The estimator holds roughly one working copy of its inputs alive on top of
# the inputs themselves (the centred products and the reduction temporaries),
# so the input term of the per-slice estimate carries a factor of two. The
# bootstrap model in ``_bootstrap.py`` uses the same factor, one resample per
# slice apart.
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
    *,
    n_bootstrap: int = 1,
) -> int:
    """Resolve how many output slices one device call may carry.

    An explicit value is an upper bound the caller chose and is honoured as
    given, capped at the number of slices there are. ``None`` derives a width
    from the active memory budget and the kernel's real working set.

    One slice of the working set is the gathered outputs, ``base_n * (D + 2)``
    elements, plus the estimator's own intermediates. See
    :func:`slice_elements` for what that comes to, and why second order is
    not merely twice first order.

    This one resolver serves both the point-estimate path (default
    ``n_bootstrap=1``) and the bootstrap path (``n_bootstrap=R``). A
    bootstrap chunk is the point-estimate working set of one slice, ``R``
    times over: every resample of a slice rides inside the same device call,
    so its working set is ``slice_elements(...) * R``. The two paths used to
    resolve this width through separate functions that could drift apart;
    now there is one model, scaled by the multiplier that tells them apart.

    Args:
        slice_chunk_size: Caller's cap on slices per chunk, honoured as
            given, or ``None`` to derive one from
            :func:`jaxgsa._core.batching.get_memory_budget`. The memory
            budget only sizes the ``None`` default; it never overrides an
            explicit choice.
        n_slices: Total number of flattened (T, K) output slices.
        base_n: N, the number of base samples.
        D: Number of input parameters.
        calc_second_order: Whether the design carries the BA blocks.
        itemsize: Bytes per element of the output dtype.
        n_bootstrap: R, the number of bootstrap resamples riding inside one
            chunk. ``1`` (the default) is the point-estimate case: no
            resample axis, so no extra factor.

    Returns:
        A chunk width in ``[1, n_slices]``.

    Raises:
        ValueError: If ``slice_chunk_size`` is given and is below 1.
    """
    if slice_chunk_size is not None:
        if slice_chunk_size < 1:
            raise ValueError(f"slice_chunk_size must be >= 1, got {slice_chunk_size}")
        return max(1, min(slice_chunk_size, n_slices))
    bytes_per_slice = slice_elements(base_n, D, calc_second_order) * itemsize * n_bootstrap
    budget = max(1, get_memory_budget() // max(bytes_per_slice, 1))
    return max(1, min(n_slices, budget))


def resolve_resample_chunk_size(
    n_bootstrap: int,
    base_n: int,
    D: int,
    calc_second_order: bool,
    itemsize: int,
    slice_chunk_size: int,
) -> int:
    """Resolve bootstrap resamples per device call for one slice chunk.

    The output-slice resolver sizes ``slice_chunk_size`` for the point path
    and, when bootstrapping, may already have accounted for all ``R`` draws.
    That is not enough when a caller explicitly asks for a narrow slice chunk:
    the resampler would still carry every draw for that slice. This resolver
    applies the transient budget to the product of the chosen slice width and
    one resample, then caps the resulting width at ``n_bootstrap``.

    Args:
        n_bootstrap: Total number of bootstrap resamples, ``R``.
        base_n: Number of base sample evaluations, ``N``.
        D: Number of input parameters.
        calc_second_order: Whether the design carries the BA blocks, which
            adds an ``(N, D)`` array to the per-slice input set.
        itemsize: Bytes per output element.
        slice_chunk_size: Number of output slices carried by one call.

    Returns:
        A resample width in ``[1, n_bootstrap]``.
    """
    bytes_per_resample = (
        slice_elements(base_n, D, calc_second_order) * itemsize * max(1, slice_chunk_size)
    )
    budget = max(1, get_memory_budget() // max(bytes_per_resample, 1))
    return max(1, min(n_bootstrap, budget))


def slice_elements(base_n: int, D: int, calc_second_order: bool) -> int:
    """Estimate the live element count of one slice of estimator work.

    Both orders are linear in ``D``. The gathered A, B, AB (and BA for second
    order) are ``base_n * (D + 2)`` or ``base_n * (2D + 2)`` elements, and the
    estimator holds about one working copy of them alive at a time, hence the
    factor ``_POINT_LIVE_COPIES``.

    This used to be quadratic in ``D`` for second order, because the
    joint-variance matrix was formed as an ``(N, D, D)`` outer product
    (``BA[:, :, None] * AB[:, None, :]``) that existed in full before the
    reduction collapsed it to ``(D, D)``. The kernel now writes the same
    quantity as ``BA.T @ AB``, a GEMM that accumulates inside the matmul
    kernel, so no ``(N, D, D)`` intermediate is materialised and second order
    costs no more transient than first order plus its larger input set. The
    output of one slice is at most a ``(D, D)`` matrix, negligible next to
    the inputs.

    The estimate is deliberately conservative: it counts the inputs twice
    and counts no XLA scheduling slack, so the derived chunk width stays
    comfortably inside the budget rather than riding its edge.

    Args:
        base_n: N, the number of base samples.
        D: Number of input parameters.
        calc_second_order: Whether the design carries the BA blocks, which
            adds the ``(N, D)`` BA array to the input set.

    Returns:
        Live elements for one slice, for one resample.
    """
    inputs = base_n * (2 * D + 2 if calc_second_order else D + 2)
    return _POINT_LIVE_COPIES * inputs
