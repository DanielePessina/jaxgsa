"""Bootstrap resampling kernels for confidence intervals on Sobol indices.

Bootstrap strategy
------------------
Given N base model evaluations, we generate R sets of N random indices into
[0, N) — i.e. sampling *with replacement*.  For each of the R resamples we
gather the corresponding rows from the model-output arrays (A, B, AB, and
optionally BA) and recompute Sobol indices, yielding an empirical distribution
over each index from which confidence intervals can be derived.

Dimensions used throughout:
    R = number of bootstrap resamples
    N = number of base-sample evaluations (base_n)
    D = number of input parameters

Chunked vmap
~~~~~~~~~~~~
``jax.vmap`` over all R resamples at once would materialise R copies of every
(N, D) array simultaneously, easily exhausting device memory when R is large.
Instead we process resamples in chunks of ``chunk_size`` rows, vmap within each
chunk (fully vectorised on-device), and concatenate the results on the host.
(``analyze`` forwards its ``slice_chunk_size`` argument as this resample cap.)
"""

import jax
import jax.numpy as jnp
from jax import Array

from gsax.sobol._indices import _fused_first_total, _fused_second_order

# @jax.jit is applied directly (not via lru_cache) because these functions
# have a fixed signature — no configuration parameter to dispatch on.


@jax.jit
def _resample_ft(idx_chunk: Array, A: Array, AB: Array, B: Array):
    """Vectorised first/total-order Sobol computation for one chunk of resamples.

    Args:
        idx_chunk: (C, N) bootstrap index sets for this chunk, where
            C <= chunk_size and each row contains N indices in [0, N).
        A:  (N,)    base model outputs from sample matrix A.
        AB: (N, D)  model outputs from the AB cross-matrices.
        B:  (N,)    base model outputs from sample matrix B.

    Returns:
        S1: (C, D) first-order indices per resample.
        ST: (C, D) total-order indices per resample.
    """

    # Closure over A, AB, B lets vmap vary only the index vector per resample.
    # A[idx] gathers N rows with replacement — the core of bootstrap resampling.
    def single(idx):
        return _fused_first_total(A[idx], AB[idx], B[idx])

    # vmap maps `single` across C index sets in parallel on the accelerator
    return jax.vmap(single)(idx_chunk)


@jax.jit
def _resample_so(idx_chunk: Array, A: Array, AB: Array, BA: Array, B: Array):
    """Vectorised second-order Sobol computation for one chunk of resamples.

    Args:
        idx_chunk: (C, N) bootstrap index sets for this chunk.
        A:  (N,)    base model outputs from sample matrix A.
        AB: (N, D)  model outputs from the AB cross-matrices.
        BA: (N, D)  model outputs from the BA cross-matrices.
        B:  (N,)    base model outputs from sample matrix B.

    Returns:
        S1: (C, D)    first-order indices per resample.
        ST: (C, D)    total-order indices per resample.
        S2: (C, D, D) second-order indices per resample.
    """

    # Same closure+vmap pattern as _resample_ft, extended to include BA
    def single(idx):
        return _fused_second_order(A[idx], AB[idx], BA[idx], B[idx])

    return jax.vmap(single)(idx_chunk)


def _bootstrap_first_total(
    indices: Array, A: Array, AB: Array, B: Array, chunk_size: int
) -> tuple[Array, Array]:
    """Bootstrap first-order and total-order Sobol indices over R resamples.

    Iterates over ``indices`` in chunks of ``chunk_size`` rows, calling
    ``_resample_ft`` (vectorised via vmap) on each chunk to avoid
    materialising all R resamples in device memory at once.

    Args:
        indices:    (R, N) int array of resampling indices in [0, N).
        A:          (N,)   model outputs from sample matrix A.
        AB:         (N, D) model outputs from the AB cross-matrices.
        B:          (N,)   model outputs from sample matrix B.
        chunk_size: max resamples to vmap in a single device call.

    Returns:
        S1_boot: (R, D) first-order indices for every resample.
        ST_boot: (R, D) total-order indices for every resample.
    """
    R = indices.shape[0]
    s1_parts, st_parts = [], []
    # Clamp chunk_size to R to avoid empty trailing slices
    cs = min(chunk_size, R)
    for start in range(0, R, cs):
        end = min(start + cs, R)
        # Process C resamples via vmap; chunking bounds peak device memory
        s1, st = _resample_ft(indices[start:end], A, AB, B)
        s1_parts.append(s1)
        st_parts.append(st)

    # Concatenate chunks along the resample axis -> (R, D)
    return jnp.concatenate(s1_parts), jnp.concatenate(st_parts)


def _bootstrap_second_order(
    indices: Array, A: Array, AB: Array, BA: Array, B: Array, chunk_size: int
) -> tuple[Array, Array, Array]:
    """Bootstrap first-, total-, and second-order Sobol indices over R resamples.

    Same chunked strategy as ``_bootstrap_first_total``, extended to include
    the BA matrices required for second-order index estimation.

    Args:
        indices:    (R, N) int array of resampling indices in [0, N).
        A:          (N,)   model outputs from sample matrix A.
        AB:         (N, D) model outputs from the AB cross-matrices.
        BA:         (N, D) model outputs from the BA cross-matrices.
        B:          (N,)   model outputs from sample matrix B.
        chunk_size: max resamples to vmap in a single device call.

    Returns:
        S1_boot: (R, D)    first-order indices for every resample.
        ST_boot: (R, D)    total-order indices for every resample.
        S2_boot: (R, D, D) second-order indices for every resample.
    """
    R = indices.shape[0]
    s1_parts, st_parts, s2_parts = [], [], []
    cs = min(chunk_size, R)
    for start in range(0, R, cs):
        end = min(start + cs, R)
        # Same chunked-vmap strategy as _bootstrap_first_total
        s1, st, s2 = _resample_so(indices[start:end], A, AB, BA, B)
        s1_parts.append(s1)
        st_parts.append(st)
        s2_parts.append(s2)

    # Concatenate chunks along the resample axis -> (R, ...) for each output
    return (
        jnp.concatenate(s1_parts),  # (R, D)
        jnp.concatenate(st_parts),  # (R, D)
        jnp.concatenate(s2_parts),  # (R, D, D)
    )
