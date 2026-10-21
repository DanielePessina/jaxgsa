"""Bootstrap resampling kernels for confidence intervals on Sobol indices.

Bootstrap strategy
------------------
Given N base model evaluations, the code draws R resamples of N row indices
in [0, N), which is sampling with replacement. Draw ``r`` comes from its own
key, ``keys[r]``, so no draw depends on the chunk width or the memory budget,
and no ``(R, N)`` index matrix ever exists.

A resample is stored as a count row: how often each base row was drawn. Every
Sobol estimator is a function of sample means, so the indices of a resample
are a function of the count-weighted means of a per-row feature matrix (see
:func:`jaxgsa.sobol._estimators.moment_form`). The features depend on the data
only. One matrix product ``counts @ features`` gives all resamples of a
slice, with no gather of ``N`` rows per resample.

Dimensions used throughout:
    R = number of bootstrap resamples
    N = number of base-sample evaluations (base_n)
    D = number of input parameters
    S = number of flattened (T, K) output slices

Chunking
~~~~~~~~
:func:`_resample_kernel` builds the count rows for one chunk of resamples
once and shares them across every output slice of the chunk. The chunks are
looped over on the host. Both axes are bounded: the slice width follows
``slice_chunk_size`` and the resample width follows the active transient-memory
budget.

When ``keep_replicates=False``, the caller processes one output chunk at a
time and reduces its ``R`` draws to confidence endpoints before moving on.
With ``keep_replicates=True`` the full draw arrays are assembled deliberately
for the result's public ``ci.replicates`` field.

A trailing chunk narrower than the rest would trace the jitted resampler a
second time, so it is padded back to the full width and the answer is sliced
back. The padded lanes are separate, so this too is exact.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import Array

from jaxgsa._core.bootstrap import draw_rows
from jaxgsa.sobol._chunking import pad_slice_axis
from jaxgsa.sobol._estimators import moment_form


@lru_cache(maxsize=None)
def _resample_kernel(estimator: str, calc_second_order: bool, n_params: int):
    """Build the jitted bootstrap resampler for one estimator and design.

    Args:
        estimator: The estimator name, as validated by ``analyze``.
        calc_second_order: Whether the design carries the BA blocks.
        n_params: Number of input parameters, ``D``.

    Returns:
        A jitted function ``(keys, A, AB, BA, B) -> (S1, ST, S2)``. ``keys``
        has shape ``(R_chunk,)``. ``A`` and ``B`` have shape ``(C, N)`` and
        ``AB`` and ``BA`` have shape ``(C, N, D)`` for a chunk of ``C``
        slices. ``BA`` is ``None`` without second order, and ``S2`` is then
        ``None``. Outputs have shape ``(C, R_chunk, D)`` and
        ``(C, R_chunk, D, D)``.
    """
    features, finalize = moment_form(estimator, calc_second_order, n_params)
    highest = jax.lax.Precision.HIGHEST

    def resample(keys: Array, A: Array, AB: Array, BA: Array | None, B: Array):
        n = A.shape[1]
        rows = jax.vmap(lambda k: draw_rows(k, n))(keys)
        counts = jax.vmap(lambda r: jnp.bincount(r, length=n))(rows).astype(A.dtype)
        F = jax.vmap(features)(A, AB, B, BA)
        m = jnp.einsum("rn,snp->srp", counts, F, precision=highest) / n
        if calc_second_order:
            assert BA is not None
            cross = jax.vmap(
                lambda ba, ab: jnp.einsum("rn,nj,nk->rjk", counts, ba, ab, precision=highest) / n
            )(BA, AB)
        else:
            cross = None
        return jax.vmap(finalize)(m, cross)

    return jax.jit(resample)


def _bootstrap_indices(
    keys: Array,
    A: Array,
    AB: Array,
    BA: Array | None,
    B: Array,
    slice_chunk_size: int,
    estimator: str,
    *,
    resample_chunk_size: int | None = None,
) -> tuple[Array, Array, Array | None]:
    """Bootstrap Sobol indices over every output slice.

    Iterates over the output slices in chunks of ``slice_chunk_size`` and the
    resample keys in chunks of ``resample_chunk_size``.

    Args:
        keys: One PRNG key per resample, shape ``(R,)``, from
            :func:`jaxgsa._core.bootstrap.resample_keys`. Shared across
            slices, so every output sees the same resamples.
        A: Model outputs from sample matrix A, shape ``(S, N)``.
        AB: Model outputs from the AB cross-matrices, shape ``(S, N, D)``.
        BA: Model outputs from the BA cross-matrices, shape ``(S, N, D)``, or
            ``None`` to run the first/total-order-only formulas.
        B: Model outputs from sample matrix B, shape ``(S, N)``.
        slice_chunk_size: Maximum output slices in a single device call.
        estimator: Which named estimator to resample. The bootstrap must use
            the same formulas as the point estimate, or the interval would
            describe a different quantity from the number at its centre.
        resample_chunk_size: Maximum bootstrap resamples in one device call.
            ``None`` keeps all keys together for compatibility with direct
            internal callers; the analysis path resolves this width from the
            transient memory budget before calling here.

    Returns:
        ``S1_boot``, ``ST_boot``: first- and total-order indices per slice
        and resample, shape ``(S, R, D)``. ``S2_boot``: second-order indices,
        shape ``(S, R, D, D)``, or ``None`` when ``BA`` is ``None``.
    """
    calc_second_order = BA is not None
    n_slices = A.shape[0]
    n_bootstrap = keys.shape[0]
    cs = max(1, min(slice_chunk_size, n_slices))
    rs = (
        n_bootstrap
        if resample_chunk_size is None
        else max(1, min(resample_chunk_size, n_bootstrap))
    )
    resample = _resample_kernel(estimator, calc_second_order, AB.shape[2])

    s1_parts, st_parts, s2_parts = [], [], []
    for start in range(0, n_slices, cs):
        end = min(start + cs, n_slices)
        actual = end - start
        # A short trailing output chunk is padded back to C so the jitted
        # resampler traces once rather than twice.
        A_chunk = pad_slice_axis(A[start:end], cs)
        AB_chunk = pad_slice_axis(AB[start:end], cs)
        B_chunk = pad_slice_axis(B[start:end], cs)
        BA_chunk = None if BA is None else pad_slice_axis(BA[start:end], cs)
        s1_draws, st_draws, s2_draws = [], [], []
        for resample_start in range(0, n_bootstrap, rs):
            resample_end = min(resample_start + rs, n_bootstrap)
            n_real = resample_end - resample_start
            key_chunk = keys[resample_start:resample_end]
            if n_real < rs:
                # Pad a ragged resample tail back to the compiled width with
                # copies of the first key. The padded draws are sliced away
                # below, and draw r depends on keys[r] alone.
                key_chunk = jnp.concatenate([key_chunk, jnp.repeat(key_chunk[:1], rs - n_real)])
            s1, st, s2 = resample(key_chunk, A_chunk, AB_chunk, BA_chunk, B_chunk)
            s1_draws.append(s1[:, :n_real])
            st_draws.append(st[:, :n_real])
            if s2 is not None:
                s2_draws.append(s2[:, :n_real])
        s1_parts.append(jnp.concatenate(s1_draws, axis=1)[:actual])
        st_parts.append(jnp.concatenate(st_draws, axis=1)[:actual])
        if calc_second_order:
            s2_parts.append(jnp.concatenate(s2_draws, axis=1)[:actual])

    S1 = jnp.concatenate(s1_parts)
    ST = jnp.concatenate(st_parts)
    S2 = jnp.concatenate(s2_parts) if calc_second_order else None
    return S1, ST, S2
