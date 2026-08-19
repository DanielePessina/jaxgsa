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

Chunked vmap
~~~~~~~~~~~~
``jax.vmap`` over all R resamples at once would materialise R copies of every
(N, D) array at the same time. That exhausts device memory when R is large.
The code instead processes resamples in chunks of ``chunk_size`` rows, vmaps
within each chunk (fully vectorised on-device), and concatenates the results on
the host. ``analyze`` forwards its ``slice_chunk_size`` argument as this
resample cap.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import Array

from jaxgsa.sobol._estimators import first_total_kernel, second_order_kernel


# lru_cache keys the compiled resampler on the estimator name, the one
# configuration parameter these functions dispatch on. Each named estimator
# is traced once per process, as the point-estimate kernels are.
@lru_cache(maxsize=None)
def _resample_ft(estimator: str):
    """Build the vectorised first/total-order resampler for one estimator.

    Args:
        estimator: The estimator name, as validated by ``analyze``.

    Returns:
        A jitted function ``(idx_chunk, A, AB, B) -> (S1, ST)`` where
        ``idx_chunk`` has shape ``(C, N)``, each row holding N indices in
        [0, N), and the two outputs have shape ``(C, D)``.
    """
    kernel = first_total_kernel(estimator)

    @jax.jit
    def resample(idx_chunk: Array, A: Array, AB: Array, B: Array):
        # Closure over A, AB, B lets vmap vary only the index vector per
        # resample. A[idx] gathers N rows with replacement, the core of
        # bootstrap resampling.
        def single(idx):
            return kernel(A[idx], AB[idx], B[idx])

        # vmap maps `single` across C index sets in parallel on the accelerator
        return jax.vmap(single)(idx_chunk)

    return resample


@lru_cache(maxsize=None)
def _resample_so(estimator: str):
    """Build the vectorised second-order resampler for one estimator.

    Args:
        estimator: The estimator name, as validated by ``analyze``.

    Returns:
        A jitted function ``(idx_chunk, A, AB, BA, B) -> (S1, ST, S2)``, with
        the two index arrays of shape ``(C, D)`` and S2 of shape
        ``(C, D, D)``.
    """
    kernel = second_order_kernel(estimator)

    @jax.jit
    def resample(idx_chunk: Array, A: Array, AB: Array, BA: Array, B: Array):
        # Same closure+vmap pattern as _resample_ft, extended to include BA
        def single(idx):
            return kernel(A[idx], AB[idx], BA[idx], B[idx])

        return jax.vmap(single)(idx_chunk)

    return resample


def _bootstrap_first_total(
    indices: Array, A: Array, AB: Array, B: Array, chunk_size: int, estimator: str
) -> tuple[Array, Array]:
    """Bootstrap first-order and total-order Sobol indices over R resamples.

    Iterate over ``indices`` in chunks of ``chunk_size`` rows and call
    ``_resample_ft`` (vectorised via vmap) on each chunk. Chunking avoids
    materialising all R resamples in device memory at once.

    Args:
        indices: Integer array of resampling indices in [0, N), shape
            ``(R, N)``.
        A: Model outputs from sample matrix A, shape ``(N,)``.
        AB: Model outputs from the AB cross-matrices, shape ``(N, D)``.
        B: Model outputs from sample matrix B, shape ``(N,)``.
        chunk_size: Maximum resamples to vmap in a single device call.
        estimator: Which named estimator to resample. The bootstrap must use
            the same formulas as the point estimate, or the interval would
            describe a different quantity from the number at its centre.

    Returns:
        S1_boot: first-order indices for every resample, shape ``(R, D)``.
        ST_boot: total-order indices for every resample, shape ``(R, D)``.
    """
    R = indices.shape[0]
    s1_parts, st_parts = [], []
    resample = _resample_ft(estimator)
    # Clamp chunk_size to R to avoid empty trailing slices
    cs = min(chunk_size, R)
    for start in range(0, R, cs):
        end = min(start + cs, R)
        # Process C resamples via vmap; chunking bounds peak device memory
        s1, st = resample(indices[start:end], A, AB, B)
        s1_parts.append(s1)
        st_parts.append(st)

    # Concatenate chunks along the resample axis -> (R, D)
    return jnp.concatenate(s1_parts), jnp.concatenate(st_parts)


def _bootstrap_second_order(
    indices: Array,
    A: Array,
    AB: Array,
    BA: Array,
    B: Array,
    chunk_size: int,
    estimator: str,
) -> tuple[Array, Array, Array]:
    """Bootstrap first-, total-, and second-order Sobol indices over R resamples.

    Same chunked strategy as ``_bootstrap_first_total``, extended with the BA
    matrices that second-order index estimation requires.

    Args:
        indices: Integer array of resampling indices in [0, N), shape
            ``(R, N)``.
        A: Model outputs from sample matrix A, shape ``(N,)``.
        AB: Model outputs from the AB cross-matrices, shape ``(N, D)``.
        BA: Model outputs from the BA cross-matrices, shape ``(N, D)``.
        B: Model outputs from sample matrix B, shape ``(N,)``.
        chunk_size: Maximum resamples to vmap in a single device call.
        estimator: Which named estimator to resample, matching the point
            estimate.

    Returns:
        S1_boot: first-order indices for every resample, shape ``(R, D)``.
        ST_boot: total-order indices for every resample, shape ``(R, D)``.
        S2_boot: second-order indices for every resample, shape
            ``(R, D, D)``.
    """
    R = indices.shape[0]
    s1_parts, st_parts, s2_parts = [], [], []
    resample = _resample_so(estimator)
    cs = min(chunk_size, R)
    for start in range(0, R, cs):
        end = min(start + cs, R)
        # Same chunked-vmap strategy as _bootstrap_first_total
        s1, st, s2 = resample(indices[start:end], A, AB, BA, B)
        s1_parts.append(s1)
        st_parts.append(st)
        s2_parts.append(s2)

    # Concatenate chunks along the resample axis -> (R, ...) for each output
    return (
        jnp.concatenate(s1_parts),  # (R, D)
        jnp.concatenate(st_parts),  # (R, D)
        jnp.concatenate(s2_parts),  # (R, D, D)
    )
