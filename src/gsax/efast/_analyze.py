"""eFAST analysis: compute S1 and ST from Fourier amplitude decomposition.

For each parameter i the model output along its search curve is
Fourier-transformed. First-order indices are estimated from the power
at harmonics of the focal frequency omega_0, and total-order indices
from the complementary low-frequency content.

Array shape conventions used throughout:
    N  — number of samples per search curve (``n_per_curve``)
    D  — number of input parameters
    T  — number of time steps (singleton-squeezed when absent)
    K  — number of output variables (singleton-squeezed when absent)

References:
    Saltelli, Tarantola & Chan (1999). Technometrics 41(1):39-56.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import Array

from gsax._core.validation import (
    _prenormalize_outputs,
    _prepare_Y,
    _validate_output,
    _warn_zero_variance_slices,
)
from gsax.efast._result import EFASTResult
from gsax.efast._sampling import EFASTSamples


def _compute_indices(Y_curve: Array, N: int, M: int, omega_0: int) -> tuple[Array, Array]:
    """Compute S1 and ST for a single search curve.

    Args:
        Y_curve: (N,) model outputs along one search curve.
        N: Number of samples in the curve.
        M: Interference factor.
        omega_0: Primary frequency.

    Returns:
        (S1, ST) scalar arrays.
    """
    # Discrete Fourier spectrum of the model output along one search curve
    f = jnp.fft.fft(Y_curve)
    # One-sided power spectrum: |F_k|^2/N^2, positive freqs only (skip DC at k=0)
    Sp = jnp.abs(f[1 : (N + 1) // 2]) ** 2 / N**2
    V = 2.0 * jnp.sum(Sp)
    if N % 2 == 0:
        V = V + jnp.abs(f[N // 2]) ** 2 / N**2

    # First-order partial variance: sum power at harmonics p*omega_0, p=1..M.
    # These frequencies carry variance attributable solely to the focal parameter.
    harmonics = jnp.arange(1, M + 1) * omega_0
    D1 = 2.0 * jnp.sum(Sp[harmonics - 1])

    # Complementary variance: power at frequencies 1..omega_0//2 (DC excluded;
    # Sp[k] holds frequency k+1). Everything below omega_0/2 is driven by the
    # complementary (non-focal) parameters' lower frequencies.
    compl_range = jnp.arange(omega_0 // 2)
    Dt = 2.0 * jnp.sum(Sp[compl_range])

    # S1: fraction of total variance from the focal parameter alone
    S1 = jnp.where(V == 0, jnp.nan, D1 / V)
    # ST: 1 - (complementary share) = total effect including all interactions
    ST = jnp.where(V == 0, jnp.nan, 1.0 - Dt / V)
    return S1, ST


# ---------------------------------------------------------------------------
# Cached JIT kernels
# ---------------------------------------------------------------------------

# Closure captures concrete (N, M, omega_0) so jnp.arange() sees Python ints,
# not JAX tracers. vmap only traces the Y_curve argument.


@lru_cache(maxsize=4)
def _get_efast_kernel(N: int, M: int, omega_0: int, batched: bool):
    """Cache JIT-compiled eFAST kernels, optionally vmapped for batched path."""

    def kernel(Y_curve: Array) -> tuple[Array, Array]:
        return _compute_indices(Y_curve, N, M, omega_0)

    if batched:
        return jax.jit(jax.vmap(kernel))
    return jax.jit(kernel)


def analyze(
    samples: EFASTSamples,
    Y: Array,
    *,
    prenormalize: bool = False,
    slice_chunk_size: int = 2048,
) -> EFASTResult:
    """Compute eFAST first- and total-order sensitivity indices.

    eFAST attributes output variance to each parameter from the Fourier
    spectrum of the model output along that parameter's search curve --
    a structured, deterministic alternative to Monte Carlo Sobol estimation
    that yields S1 and ST (but no second-order indices) from
    ``n_per_curve * D`` model runs. ``Y`` must be the model evaluated
    row-by-row on ``samples.samples`` (rows in the same order); the design
    metadata (``n_per_curve``, ``M``, ``problem``) is read from ``samples``
    so it can never be mismatched with the sampling step.

    Args:
        samples: Design returned by ``gsax.efast.sample()``, carrying the
            sample matrix plus ``n_per_curve``, ``M``, and the problem.
        Y: Model outputs evaluated at each row of ``samples.samples``, in
            the same row order. Accepted shapes (``n_runs`` is
            ``samples.n_runs = n_per_curve * D``):
            - ``(n_runs,)`` for scalar output
            - ``(n_runs, K)`` for K output variables
            - ``(n_runs, T, K)`` for K outputs over T time steps
        prenormalize: If True, center and scale each output slice to unit
            variance before computing indices. The indices are ratios, so
            this changes nothing mathematically; it only helps when raw
            output magnitudes risk float overflow/underflow.
        slice_chunk_size: Maximum number of output slices to process in
            one vmapped batch. Caps peak device memory for large
            ``T * K``; smaller values trade speed for memory.

    Returns:
        EFASTResult with S1 and ST, shaped ``(D,)`` / ``(K, D)`` /
        ``(T, K, D)`` to mirror the layout of ``Y``.

    Raises:
        ValueError: If ``Y``'s leading dimension does not equal
            ``samples.n_runs``, or ``Y`` has an invalid rank.
    """
    problem = samples.problem
    M = samples.M
    D = problem.num_vars
    N = samples.n_per_curve

    Y = jnp.asarray(Y)

    if Y.ndim in (1, 2, 3) and Y.shape[0] != samples.n_runs:
        raise ValueError(
            f"Y has {Y.shape[0]} rows but this eFAST design requires "
            f"n_runs = n_per_curve * D = {N} * {D} = {samples.n_runs}; "
            "evaluate the model on every row of samples.samples, in order"
        )

    if not jnp.all(jnp.isfinite(Y)):
        n_bad = int(jnp.sum(~jnp.isfinite(Y)))
        warnings.warn(
            f"eFAST: Y contains {n_bad} non-finite values (NaN/Inf) "
            "which will propagate into indices",
            stacklevel=2,
        )

    Y = _validate_output(Y, samples.n_runs, problem)

    # Detect scalar output before _prepare_Y adds singleton dims
    is_scalar = Y.ndim == 1

    # Promote to canonical (N*D, T, K) shape
    Y, squeeze_time, squeeze_output = _prepare_Y(Y)

    if prenormalize:
        Y, _, _, _ = _prenormalize_outputs(Y)

    _warn_zero_variance_slices(Y, output_names=problem.output_names)

    # Recompute omega_0 from the design's n_per_curve and M — the same
    # formula sample() used, so the harmonics line up exactly.
    omega_0 = (N - 1) // (2 * M)

    _, T, K = Y.shape

    # Split contiguous search curves into (D, N, T, K)
    Y_reshaped = Y.reshape(D, N, T, K)

    # Per-curve zero-variance check: the global _warn_zero_variance_slices above
    # can miss curves where a single parameter has no effect (V=0 on that curve
    # alone), producing silent NaN indices.
    per_curve_var = jnp.var(Y_reshaped, axis=1)  # (D, T, K)
    n_zero_curves = int(jnp.sum(per_curve_var == 0))
    if n_zero_curves > 0:
        warnings.warn(
            f"eFAST: {n_zero_curves} search-curve/output slice(s) have zero "
            "variance — corresponding indices will be NaN",
            stacklevel=2,
        )

    if is_scalar:
        # Scalar path: squeeze trailing singletons, vmap over D curves only
        Y_curves = Y_reshaped[:, :, 0, 0]  # (D, N)
        kernel = _get_efast_kernel(N, M, omega_0, batched=True)
        S1, ST = kernel(Y_curves)  # each (D,)
    else:
        # Batched path: flatten (D, T, K) into a single vmap axis
        Y_batched = Y_reshaped.transpose(0, 2, 3, 1).reshape(D * T * K, N)

        total = D * T * K
        cs = min(slice_chunk_size, total)
        batched = _get_efast_kernel(N, M, omega_0, batched=True)

        s1_parts: list[Array] = []
        st_parts: list[Array] = []
        for start in range(0, total, cs):
            end = min(start + cs, total)
            actual = end - start
            batch = Y_batched[start:end]
            if actual < cs:
                batch = jnp.concatenate([batch, jnp.zeros((cs - actual, N))], axis=0)
            s1_chunk, st_chunk = batched(batch)
            s1_parts.append(s1_chunk[:actual])
            st_parts.append(st_chunk[:actual])

        S1_flat = jnp.concatenate(s1_parts)  # (D*T*K,)
        ST_flat = jnp.concatenate(st_parts)

        # Reshape to (D, T, K) then transpose to (T, K, D) convention
        S1 = S1_flat.reshape(D, T, K).transpose(1, 2, 0)
        ST = ST_flat.reshape(D, T, K).transpose(1, 2, 0)

        # Squeeze singleton dims that _prepare_Y inserted
        if squeeze_time:
            S1 = S1[0]
            ST = ST[0]

    # Indices outside [0,1] indicate the frequency decomposition didn't converge
    if jnp.any((S1 > 1.0) | (S1 < 0.0)) or jnp.any((ST > 1.0) | (ST < 0.0)):
        warnings.warn(
            "eFAST: some indices are outside [0, 1], suggesting "
            "insufficient samples or near-zero output variance",
            stacklevel=2,
        )

    return EFASTResult(
        S1=S1,
        ST=ST,
        problem=problem,
        omega_0=omega_0,
        M=M,
    )
