"""Bootstrap resampling kernels for confidence intervals on Sobol indices.

Bootstrap strategy
------------------
Given N base model evaluations, the code generates R sets of N random indices
into [0, N), which is sampling with replacement. For each of the R resamples it
gathers the matching rows from the model-output arrays (A, AB, and optionally
BA, and B) and recomputes the Sobol indices. That gives an empirical
distribution over each index, from which confidence intervals follow.

Dimensions used throughout:
    R = number of bootstrap resamples
    N = number of base-sample evaluations (base_n)
    D = number of input parameters
    S = number of flattened (T, K) output slices

Chunked vmap
~~~~~~~~~~~~
The atomic unit is one estimator call on one output slice and one resample.
:func:`_resample_kernel` builds that unit up in the shape the rest of the
library uses: the estimator kernel is vmapped over a batch of resamples of a
slice, that pair is vmapped over a chunk of output slices, and the chunks are
looped over on the host. A call therefore covers ``slice_chunk *
resample_chunk`` estimator evaluations. Both axes are bounded: the slice
width follows ``slice_chunk_size`` and the resample width follows the active
transient-memory budget. A single ``vmap`` over all ``S * R`` pairs would
materialise all output slices and resamples at once.

When ``keep_replicates=False``, the caller processes one output chunk at a
time and reduces its ``R`` draws to confidence endpoints before moving on.
That keeps the retained bootstrap results proportional to one chunk rather
than to ``S * R``. With ``keep_replicates=True`` the full draw arrays are
assembled deliberately for the result's public ``ci.replicates`` field.

A trailing chunk narrower than the rest would trace the jitted resampler a
second time, so it is padded back to the full width and the answer is sliced
back. The padded slices are separate ``vmap`` lanes, so this too is exact.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import Array

from jaxgsa.sobol._chunking import pad_slice_axis
from jaxgsa.sobol._estimators import first_total_kernel, second_order_kernel

# lru_cache keys the compiled resampler on the estimator name and the two
# axes it varies along, the same three keys _bootstrap_indices dispatches on:
# whether the design carries the BA blocks (calc_second_order), and whether a
# chunk holds one output slice (single) or several. A length-one slice axis
# compiles to a slower gather than no slice axis at all (measured 14.93 ms vs
# 3.74 ms on an Apple M1 Pro, N=256, D=3, R=1000), so the degenerate chunk
# skips the outer vmap rather than mapping over it. Each of the four
# combinations is traced once per process, as the point-estimate kernels are.


@lru_cache(maxsize=None)
def _resample_kernel(estimator: str, calc_second_order: bool, single: bool):
    """Build the jitted bootstrap resampler for one estimator/design/chunk-shape.

    This replaces what used to be four separate functions
    (``_resample_ft``, ``_resample_ft_one``, ``_resample_so``,
    ``_resample_so_one``), which differed only in these two booleans.

    Args:
        estimator: The estimator name, as validated by ``analyze``.
        calc_second_order: Whether the design carries the BA blocks. Selects
            the estimator's ``(A, AB, B) -> (S1, ST)`` formula or its
            ``(A, AB, BA, B) -> (S1, ST, S2)`` one, and whether the returned
            function reads and returns BA/S2 at all.
        single: Whether the chunk this resampler will be called on holds one
            output slice. ``True`` builds a function with no slice axis to
            map over; ``False`` builds one that vmaps a chunk of slices.

    Returns:
        A jitted function. With ``calc_second_order=True`` it is
        ``(idx, a, ab, ba, b) -> (S1, ST, S2)`` when ``single``, or
        ``(idx, A, AB, BA, B) -> (S1, ST, S2)`` otherwise. With
        ``calc_second_order=False`` the same, minus the BA argument and the
        S2 return. ``idx`` has shape ``(R_chunk, N)``. A single chunk's inputs have
        shape ``(N,)`` / ``(N, D)``; a multi-slice chunk's have shape
        ``(C, N)`` / ``(C, N, D)`` for its ``C`` output slices.
    """
    if calc_second_order:
        so_kernel = second_order_kernel(estimator)

        if single:

            def resample(idx: Array, a: Array, ab: Array, ba: Array, b: Array):
                def one_draw(rows: Array):
                    return so_kernel(a[rows], ab[rows], ba[rows], b[rows])

                return jax.vmap(one_draw)(idx)
        else:

            def resample(idx: Array, A: Array, AB: Array, BA: Array, B: Array):
                def one_slice(a: Array, ab: Array, ba: Array, b: Array):
                    def one_draw(rows: Array):
                        return so_kernel(a[rows], ab[rows], ba[rows], b[rows])

                    return jax.vmap(one_draw)(idx)

                return jax.vmap(one_slice)(A, AB, BA, B)
    else:
        ft_kernel = first_total_kernel(estimator)

        if single:

            def resample(idx: Array, a: Array, ab: Array, b: Array):
                def one_draw(rows: Array):
                    return ft_kernel(a[rows], ab[rows], b[rows])

                return jax.vmap(one_draw)(idx)
        else:

            def resample(idx: Array, A: Array, AB: Array, B: Array):
                def one_slice(a: Array, ab: Array, b: Array):
                    def one_draw(rows: Array):
                        return ft_kernel(a[rows], ab[rows], b[rows])

                    return jax.vmap(one_draw)(idx)

                return jax.vmap(one_slice)(A, AB, B)

    return jax.jit(resample)


def _bootstrap_indices(
    indices: Array,
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

    This replaces what used to be two separate functions
    (``_bootstrap_first_total``, ``_bootstrap_second_order``), which differed
    only in whether BA was given. Iterates over the output slices in chunks
    of ``slice_chunk_size`` and the pre-generated resample rows in chunks of
    ``resample_chunk_size``. The latter split is important even when the
    output slice chunk has width one: an arbitrary ``R`` axis would otherwise
    remain live in the jitted gather.

    Args:
        indices: Integer array of resampling indices in [0, N), shape
            ``(R, N)``. Shared across slices, so every output sees the same
            resamples.
        A: Model outputs from sample matrix A, shape ``(S, N)``.
        AB: Model outputs from the AB cross-matrices, shape ``(S, N, D)``.
        BA: Model outputs from the BA cross-matrices, shape ``(S, N, D)``, or
            ``None`` to run the first/total-order-only formulas.
        B: Model outputs from sample matrix B, shape ``(S, N)``.
        slice_chunk_size: Maximum output slices to vmap in a single device
            call.
        estimator: Which named estimator to resample. The bootstrap must use
            the same formulas as the point estimate, or the interval would
            describe a different quantity from the number at its centre.
        resample_chunk_size: Maximum bootstrap resamples to vmap in one
            device call. ``None`` keeps all rows together for compatibility
            with direct internal callers; the analysis path resolves this
            width from the transient memory budget before calling here.

    Returns:
        ``S1_boot``, ``ST_boot``: first- and total-order indices per slice
        and resample, shape ``(S, R, D)``. ``S2_boot``: second-order indices,
        shape ``(S, R, D, D)``, or ``None`` when ``BA`` is ``None``.
    """
    calc_second_order = BA is not None
    n_slices = A.shape[0]
    n_bootstrap = indices.shape[0]
    cs = max(1, min(slice_chunk_size, n_slices))
    rs = (
        n_bootstrap
        if resample_chunk_size is None
        else max(1, min(resample_chunk_size, n_bootstrap))
    )
    single = cs == 1
    resample = _resample_kernel(estimator, calc_second_order, single)

    s1_parts, st_parts, s2_parts = [], [], []
    if single:
        # One slice per chunk leaves nothing to map over, so the resampler
        # built above has no outer slice axis at all. Only ``rs`` resamples
        # are live in one call; concatenate their results along R.
        for start in range(n_slices):
            s1_draws, st_draws, s2_draws = [], [], []
            for resample_start in range(0, n_bootstrap, rs):
                resample_end = min(resample_start + rs, n_bootstrap)
                n_real = resample_end - resample_start
                idx_chunk = indices[resample_start:resample_end]
                if n_real < rs:
                    # Pad a ragged resample tail back to the compiled width;
                    # duplicate rows are discarded immediately after the
                    # call and therefore do not change any random draw.
                    idx_chunk = jnp.concatenate(
                        [
                            idx_chunk,
                            jnp.broadcast_to(idx_chunk[:1], (rs - n_real, idx_chunk.shape[1])),
                        ]
                    )
                if calc_second_order:
                    assert BA is not None
                    s1, st, s2 = resample(idx_chunk, A[start], AB[start], BA[start], B[start])
                    s2_draws.append(s2[:n_real])
                else:
                    s1, st = resample(idx_chunk, A[start], AB[start], B[start])
                s1_draws.append(s1[:n_real])
                st_draws.append(st[:n_real])
            s1_parts.append(jnp.concatenate(s1_draws)[None])
            st_parts.append(jnp.concatenate(st_draws)[None])
            if calc_second_order:
                s2_parts.append(jnp.concatenate(s2_draws)[None])
    else:
        for start in range(0, n_slices, cs):
            end = min(start + cs, n_slices)
            actual = end - start
            # Process C slices x R_chunk resamples per device call. A short
            # trailing output chunk is padded back to C so the jitted
            # resampler traces once rather than twice.
            A_chunk = pad_slice_axis(A[start:end], cs)
            AB_chunk = pad_slice_axis(AB[start:end], cs)
            B_chunk = pad_slice_axis(B[start:end], cs)
            BA_chunk = None if BA is None else pad_slice_axis(BA[start:end], cs)
            s1_draws, st_draws, s2_draws = [], [], []
            for resample_start in range(0, n_bootstrap, rs):
                resample_end = min(resample_start + rs, n_bootstrap)
                n_real = resample_end - resample_start
                idx_chunk = indices[resample_start:resample_end]
                if n_real < rs:
                    # The repeated rows are padding only; slice them away
                    # before concatenating the real resample outputs.
                    idx_chunk = jnp.concatenate(
                        [
                            idx_chunk,
                            jnp.broadcast_to(idx_chunk[:1], (rs - n_real, idx_chunk.shape[1])),
                        ]
                    )
                if calc_second_order:
                    assert BA_chunk is not None
                    s1, st, s2 = resample(idx_chunk, A_chunk, AB_chunk, BA_chunk, B_chunk)
                    s2_draws.append(s2[:, :n_real])
                else:
                    s1, st = resample(idx_chunk, A_chunk, AB_chunk, B_chunk)
                s1_draws.append(s1[:, :n_real])
                st_draws.append(st[:, :n_real])
            s1_parts.append(jnp.concatenate(s1_draws, axis=1)[:actual])
            st_parts.append(jnp.concatenate(st_draws, axis=1)[:actual])
            if calc_second_order:
                s2_parts.append(jnp.concatenate(s2_draws, axis=1)[:actual])

    S1 = jnp.concatenate(s1_parts)
    ST = jnp.concatenate(st_parts)
    S2 = jnp.concatenate(s2_parts) if calc_second_order else None
    return S1, ST, S2
