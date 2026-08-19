"""Bootstrap resampling kernels for confidence intervals on Sobol indices.

Bootstrap strategy
------------------
Given N base model evaluations, the code generates R sets of N random indices
into [0, N), which is sampling with replacement. For each of the R resamples it
gathers the matching rows from the model-output arrays (A, B, AB, and
optionally BA) and recomputes the Sobol indices. That gives an empirical
distribution over each index, from which confidence intervals follow.

Dimensions used throughout:
    R = number of bootstrap resamples
    N = number of base-sample evaluations (base_n)
    D = number of input parameters
    S = number of flattened (T, K) output slices

Chunked vmap
~~~~~~~~~~~~
The atomic unit is one estimator call on one output slice and one resample.
The kernels below build that unit up in the shape the rest of the library
uses: the estimator kernel is vmapped over the R resamples of a slice, that
pair is vmapped over a chunk of output slices, and the chunks are looped over
on the host. So one device call covers ``chunk * R`` estimator evaluations,
which is the whole point -- the point-estimate path already batches its
slices, and the bootstrap path now batches the same way.

Chunking is what keeps the peak bounded. A single ``vmap`` over all
``S * R`` pairs would materialise ``S * R`` copies of every (N,) and (N, D)
gather at once. ``analyze`` forwards its ``slice_chunk_size`` as the cap on
slices per chunk, and the caller-independent memory budget can only lower it
further. Every pair is independent of every other, so the chunked answer is
the unchunked one, element for element.

A trailing chunk narrower than the rest would trace the jitted resampler a
second time, so it is padded back to the full width and the answer is sliced
back. The padded slices are separate ``vmap`` lanes, so this too is exact.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import Array

from jaxgsa._core.batching import get_memory_budget
from jaxgsa.sobol._chunking import pad_slice_axis
from jaxgsa.sobol._estimators import first_total_kernel, second_order_kernel


def _resolve_slice_chunk_size(
    slice_chunk_size: int | None,
    n_slices: int,
    n_bootstrap: int,
    base_n: int,
    D: int,
    itemsize: int,
) -> int:
    """Resolve how many output slices one bootstrap device call may carry.

    The working set of a chunk is the gathered outputs, ``chunk * R * N``
    elements for each of A and B and ``chunk * R * N * D`` for AB (twice that
    when the design also carries BA). The estimate below uses the
    first-order-only working set with a factor of two for the estimator's own
    intermediates, which is the same order of magnitude either way.

    Args:
        slice_chunk_size: Caller's cap on slices per chunk, or ``None`` to
            let the budget decide alone. It is an upper bound only: the
            memory budget may lower it, never raise it.
        n_slices: Total number of flattened (T, K) output slices.
        n_bootstrap: R, the number of bootstrap resamples.
        base_n: N, the number of base samples per resample.
        D: Number of input parameters.
        itemsize: Bytes per element of the output dtype.

    Returns:
        A chunk width in ``[1, n_slices]``.
    """
    bytes_per_slice = 2 * n_bootstrap * base_n * (D + 2) * itemsize
    budget = max(1, get_memory_budget() // max(bytes_per_slice, 1))
    if slice_chunk_size is None:
        return max(1, min(budget, n_slices))
    return max(1, min(slice_chunk_size, budget, n_slices))


# lru_cache keys the compiled resampler on the estimator name, the one
# configuration parameter these functions dispatch on. Each named estimator
# is traced once per process, as the point-estimate kernels are.
@lru_cache(maxsize=None)
def _resample_ft(estimator: str):
    """Build the vectorised first/total-order resampler for one estimator.

    Args:
        estimator: The estimator name, as validated by ``analyze``.

    Returns:
        A jitted function ``(idx, A, AB, B) -> (S1, ST)``. ``idx`` has shape
        ``(R, N)``, each row holding N indices in [0, N); ``A`` and ``B`` have
        shape ``(C, N)`` and ``AB`` shape ``(C, N, D)`` for a chunk of C
        output slices; both outputs have shape ``(C, R, D)``.
    """
    kernel = first_total_kernel(estimator)

    @jax.jit
    def resample(idx: Array, A: Array, AB: Array, B: Array):
        def one_slice(a: Array, ab: Array, b: Array):
            # a[rows] gathers N rows with replacement, the core of bootstrap
            # resampling. The inner vmap maps it across the R index sets.
            def one_draw(rows: Array):
                return kernel(a[rows], ab[rows], b[rows])

            return jax.vmap(one_draw)(idx)

        # The outer vmap maps the whole per-slice bootstrap across a chunk of
        # output slices, so one device call covers C * R estimator evaluations.
        return jax.vmap(one_slice)(A, AB, B)

    return resample


@lru_cache(maxsize=None)
def _resample_so(estimator: str):
    """Build the vectorised second-order resampler for one estimator.

    Args:
        estimator: The estimator name, as validated by ``analyze``.

    Returns:
        A jitted function ``(idx, A, AB, BA, B) -> (S1, ST, S2)``. The inputs
        follow :func:`_resample_ft` with BA shaped like AB; ``S1`` and ``ST``
        have shape ``(C, R, D)`` and ``S2`` shape ``(C, R, D, D)``.
    """
    kernel = second_order_kernel(estimator)

    @jax.jit
    def resample(idx: Array, A: Array, AB: Array, BA: Array, B: Array):
        # Same nested-vmap pattern as _resample_ft, extended to include BA
        def one_slice(a: Array, ab: Array, ba: Array, b: Array):
            def one_draw(rows: Array):
                return kernel(a[rows], ab[rows], ba[rows], b[rows])

            return jax.vmap(one_draw)(idx)

        return jax.vmap(one_slice)(A, AB, BA, B)

    return resample


def _bootstrap_first_total(
    indices: Array, A: Array, AB: Array, B: Array, slice_chunk_size: int, estimator: str
) -> tuple[Array, Array]:
    """Bootstrap first-order and total-order Sobol indices over every slice.

    Iterate over the output slices in chunks of ``slice_chunk_size`` and call
    :func:`_resample_ft` on each chunk. Chunking avoids materialising all
    ``S * R`` resampled slices in device memory at once.

    Args:
        indices: Integer array of resampling indices in [0, N), shape
            ``(R, N)``. Shared across slices, so every output sees the same
            resamples.
        A: Model outputs from sample matrix A, shape ``(S, N)``.
        AB: Model outputs from the AB cross-matrices, shape ``(S, N, D)``.
        B: Model outputs from sample matrix B, shape ``(S, N)``.
        slice_chunk_size: Maximum output slices to vmap in a single device call.
        estimator: Which named estimator to resample. The bootstrap must use
            the same formulas as the point estimate, or the interval would
            describe a different quantity from the number at its centre.

    Returns:
        S1_boot: first-order indices per slice and resample, shape
            ``(S, R, D)``.
        ST_boot: total-order indices per slice and resample, same shape.
    """
    n_slices = A.shape[0]
    s1_parts, st_parts = [], []
    resample = _resample_ft(estimator)
    cs = max(1, min(slice_chunk_size, n_slices))
    for start in range(0, n_slices, cs):
        end = min(start + cs, n_slices)
        actual = end - start
        # Process C slices x R resamples per device call; chunking bounds peak
        # device memory. A short trailing chunk is padded back to C so the
        # jitted resampler traces once rather than twice.
        s1, st = resample(
            indices,
            pad_slice_axis(A[start:end], cs),
            pad_slice_axis(AB[start:end], cs),
            pad_slice_axis(B[start:end], cs),
        )
        s1_parts.append(s1[:actual])
        st_parts.append(st[:actual])

    # Concatenate chunks along the slice axis -> (S, R, D)
    return jnp.concatenate(s1_parts), jnp.concatenate(st_parts)


def _bootstrap_second_order(
    indices: Array,
    A: Array,
    AB: Array,
    BA: Array,
    B: Array,
    slice_chunk_size: int,
    estimator: str,
) -> tuple[Array, Array, Array]:
    """Bootstrap first-, total-, and second-order Sobol indices over every slice.

    Same chunked strategy as :func:`_bootstrap_first_total`, extended with the
    BA matrices that second-order index estimation requires.

    Args:
        indices: Integer array of resampling indices in [0, N), shape
            ``(R, N)``.
        A: Model outputs from sample matrix A, shape ``(S, N)``.
        AB: Model outputs from the AB cross-matrices, shape ``(S, N, D)``.
        BA: Model outputs from the BA cross-matrices, shape ``(S, N, D)``.
        B: Model outputs from sample matrix B, shape ``(S, N)``.
        slice_chunk_size: Maximum output slices to vmap in a single device call.
        estimator: Which named estimator to resample, matching the point
            estimate.

    Returns:
        S1_boot: first-order indices, shape ``(S, R, D)``.
        ST_boot: total-order indices, shape ``(S, R, D)``.
        S2_boot: second-order indices, shape ``(S, R, D, D)``.
    """
    n_slices = A.shape[0]
    s1_parts, st_parts, s2_parts = [], [], []
    resample = _resample_so(estimator)
    cs = max(1, min(slice_chunk_size, n_slices))
    for start in range(0, n_slices, cs):
        end = min(start + cs, n_slices)
        actual = end - start
        # Same chunked nested-vmap strategy as _bootstrap_first_total, ragged
        # trailing chunk padded back to C included.
        s1, st, s2 = resample(
            indices,
            pad_slice_axis(A[start:end], cs),
            pad_slice_axis(AB[start:end], cs),
            pad_slice_axis(BA[start:end], cs),
            pad_slice_axis(B[start:end], cs),
        )
        s1_parts.append(s1[:actual])
        st_parts.append(st[:actual])
        s2_parts.append(s2[:actual])

    # Concatenate chunks along the slice axis -> (S, ...) for each output
    return (
        jnp.concatenate(s1_parts),  # (S, R, D)
        jnp.concatenate(st_parts),  # (S, R, D)
        jnp.concatenate(s2_parts),  # (S, R, D, D)
    )
