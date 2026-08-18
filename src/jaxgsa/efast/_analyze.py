"""eFAST analysis: compute S1 and ST from Fourier amplitude decomposition.

For each parameter i the model output along its search curve is
Fourier-transformed. First-order indices are estimated from the power
at harmonics of the focal frequency omega_0, and total-order indices
from the complementary low-frequency content.

Array shape conventions used throughout:
    N: number of samples per search curve (``n_per_curve``)
    D: number of input parameters
    T: number of time steps (singleton-squeezed when absent)
    K: number of output variables (singleton-squeezed when absent)

References:
    Saltelli, Tarantola & Chan (1999). Technometrics 41(1):39-56.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.invalid import (
    InvalidUnit,
    OnInvalid,
    check_invalid,
    resolve_policy,
)
from jaxgsa._core.validation import (
    _prenormalize_outputs,
    _prepare_Y,
    _validate_output,
    _warn_zero_variance_slices,
)
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.efast._result import EFASTResult
from jaxgsa.efast._sampling import EFASTSamples, _frequency_plan


def _compute_indices(
    Y_curve: Array, N: int, M: int, omega_0: int, analysis_max: int
) -> tuple[Array, Array]:
    """Compute S1 and ST for a single search curve.

    Args:
        Y_curve: Model outputs along one search curve, shape ``(N,)``.
        N: Number of samples in the curve.
        M: Interference factor.
        omega_0: Primary frequency, from the design's frequency plan.
        analysis_max: Top of the analysis band, from the same plan. It is
            wider than the band the sampler assigned carriers in; see
            :class:`jaxgsa.efast._sampling._FrequencyPlan`.

    Returns:
        A tuple ``(S1, ST)`` of scalar arrays.
    """
    # Discrete Fourier spectrum of the model output along one search curve.
    f = jnp.fft.fft(Y_curve)
    # One-sided power spectrum |F_k|^2/N^2, positive frequencies only. The DC
    # term at k=0 is skipped.
    Sp = jnp.abs(f[1 : (N + 1) // 2]) ** 2 / N**2
    V = 2.0 * jnp.sum(Sp)
    if N % 2 == 0:
        V = V + jnp.abs(f[N // 2]) ** 2 / N**2

    # First-order partial variance: sum the power at the harmonics p*omega_0
    # for p = 1..M. Those frequencies carry variance attributable to the focal
    # parameter alone.
    harmonics = jnp.arange(1, M + 1) * omega_0
    D1 = 2.0 * jnp.sum(Sp[harmonics - 1])

    # Complementary variance: power at frequencies 1..analysis_max, DC
    # excluded, where Sp[k] holds frequency k+1. Everything in that band comes
    # from the non-focal parameters — the carriers the sampler assigned, and
    # the interference between them, which lands on frequencies nobody was
    # assigned. The band is therefore wider than the assigned band on purpose.
    compl_range = jnp.arange(analysis_max)
    Dt = 2.0 * jnp.sum(Sp[compl_range])

    # S1 is the fraction of total variance from the focal parameter alone.
    S1 = jnp.where(V == 0, jnp.nan, D1 / V)
    # ST is 1 minus the complementary share: the total effect, interactions
    # included.
    ST = jnp.where(V == 0, jnp.nan, 1.0 - Dt / V)
    return S1, ST


# ---------------------------------------------------------------------------
# Cached JIT kernels
# ---------------------------------------------------------------------------

# The closure captures concrete (N, M, omega_0), so jnp.arange() sees Python
# ints rather than JAX tracers. vmap traces the Y_curve argument only.


@lru_cache(maxsize=4)
def _get_efast_kernel(N: int, M: int, omega_0: int, analysis_max: int, batched: bool):
    """Build and cache a JIT-compiled eFAST kernel for one design.

    Every argument is a plain Python int or bool, so it can serve as a cache
    key. The frequency plan itself holds a NumPy array and must not be passed
    here; read its fields instead.

    Args:
        N: Number of samples per search curve.
        M: Interference factor.
        omega_0: Primary frequency, from the design's frequency plan.
        analysis_max: Top of the analysis band, from the same plan.
        batched: If True, vmap the kernel over a leading curve axis.

    Returns:
        A compiled callable mapping curve outputs to ``(S1, ST)``.
    """

    def kernel(Y_curve: Array) -> tuple[Array, Array]:
        return _compute_indices(Y_curve, N, M, omega_0, analysis_max)

    if batched:
        return jax.jit(jax.vmap(kernel))
    return jax.jit(kernel)


def analyze(
    samples: EFASTSamples,
    Y: Array,
    *,
    prenormalize: bool = False,
    slice_chunk_size: int = 2048,
    on_invalid: OnInvalid = "raise",
) -> EFASTResult:
    """Compute eFAST first- and total-order sensitivity indices.

    eFAST attributes output variance to each parameter from the Fourier
    spectrum of the model output along that parameter's search curve. It is a
    structured, deterministic alternative to Monte Carlo Sobol estimation. It
    yields S1 and ST, but no second-order indices, from ``n_per_curve * D``
    model runs.

    ``Y`` must be the model evaluated row by row on ``samples.samples``, with
    the rows in the same order. The design metadata ``n_per_curve``, ``M``,
    and ``problem`` is read from ``samples``, so it can never be mismatched
    with the sampling step.

    Args:
        samples: Design returned by ``jaxgsa.efast.sample()``, carrying the
            sample matrix plus ``n_per_curve``, ``M``, and the problem.
        Y: Model outputs evaluated at each row of ``samples.samples``, in the
            same row order. Accepted shapes, where ``n_runs`` is
            ``samples.n_runs = n_per_curve * D``:
            - ``(n_runs,)`` for scalar output
            - ``(n_runs, K)`` for K output variables
            - ``(n_runs, T, K)`` for K outputs over T time steps
        prenormalize: If True, center and scale each output slice to unit
            variance before computing indices. The indices are ratios, so this
            changes nothing mathematically. It helps only when raw output
            magnitudes risk float overflow or underflow.
        slice_chunk_size: Maximum number of output slices to process in one
            vmapped batch. It caps peak device memory for a large ``T * K``. A
            smaller value trades speed for memory.
        on_invalid: What to do about non-finite model outputs. Only
            ``"raise"`` (the default) and ``"propagate"`` are available.
            ``"drop"`` raises, because a search curve is an ordered sweep read
            by a discrete Fourier transform: removing a point changes what the
            estimator computes instead of shrinking the sample. See
            :mod:`jaxgsa._core.invalid`.

    Returns:
        An ``EFASTResult`` with ``S1`` and ``ST``, shape ``(D,)`` /
        ``(K, D)`` / ``(T, K, D)``, mirroring the layout of ``Y``, plus the
        ``invalid`` report.

    Raises:
        ValueError: If ``Y``'s leading dimension does not equal
            ``samples.n_runs``, or ``Y`` has an invalid rank; if ``on_invalid``
            is not one of the three policies or is ``"drop"``; or if the sample
            holds a non-finite value under ``on_invalid="raise"``.
    """
    method = "jaxgsa.efast.analyze"
    policy = resolve_policy(on_invalid, method=method, unit=InvalidUnit.CURVE, allow_drop=False)
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

    Y = _validate_output(Y, samples.n_runs, problem)

    # The design lays the D search curves out contiguously, N rows each. The
    # check runs after the shape contract, so the row map is known to match.
    # A curve cannot be dropped, so `keep` is always all-True here and is
    # never applied; the report still names the curve to investigate.
    _, invalid = check_invalid(
        policy=policy,
        method=method,
        unit=InvalidUnit.CURVE,
        n_units=D,
        Y=Y,
        unit_of_row=np.repeat(np.arange(D), N),
    )

    # Record scalar output now: _prepare_Y adds singleton dims below.
    is_scalar = Y.ndim == 1

    # Promote to the canonical (N*D, T, K) shape.
    Y, squeeze_time, squeeze_output = _prepare_Y(Y)

    if prenormalize:
        Y, _, _, _ = _prenormalize_outputs(Y)

    _warn_zero_variance_slices(Y, output_names=problem.output_names)

    # Rebuild the frequency plan sample() used, from the design metadata that
    # travels inside EFASTSamples. Both sides read the same numbers, so the
    # harmonics line up by construction rather than by agreement.
    plan = _frequency_plan(D, N, M)
    omega_0 = plan.omega_0

    _, T, K = Y.shape

    # Split the contiguous search curves into (D, N, T, K).
    Y_reshaped = Y.reshape(D, N, T, K)

    # Per-curve zero-variance check. The global _warn_zero_variance_slices
    # above can miss a curve where a single parameter has no effect, giving
    # V=0 on that curve alone and a silent NaN index.
    per_curve_var = jnp.var(Y_reshaped, axis=1)  # (D, T, K)
    n_zero_curves = int(jnp.sum(per_curve_var == 0))
    if n_zero_curves > 0:
        warnings.warn(
            f"eFAST: {n_zero_curves} search-curve/output slice(s) have zero "
            "variance — corresponding indices will be NaN",
            stacklevel=2,
            category=JaxgsaWarning,
        )

    if is_scalar:
        # Scalar path: squeeze the trailing singletons and vmap over the D
        # curves only.
        Y_curves = Y_reshaped[:, :, 0, 0]  # (D, N)
        kernel = _get_efast_kernel(N, M, omega_0, plan.analysis_max, batched=True)
        S1, ST = kernel(Y_curves)  # each (D,)
    else:
        # Batched path: flatten (D, T, K) into a single vmap axis.
        Y_batched = Y_reshaped.transpose(0, 2, 3, 1).reshape(D * T * K, N)

        total = D * T * K
        cs = min(slice_chunk_size, total)
        batched = _get_efast_kernel(N, M, omega_0, plan.analysis_max, batched=True)

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

        # Reshape to (D, T, K), then transpose to the (T, K, D) convention.
        S1 = S1_flat.reshape(D, T, K).transpose(1, 2, 0)
        ST = ST_flat.reshape(D, T, K).transpose(1, 2, 0)

        # Drop the singleton dims that _prepare_Y inserted.
        if squeeze_time:
            S1 = S1[0]
            ST = ST[0]

    # An index outside [0, 1] means the frequency decomposition did not
    # converge.
    if jnp.any((S1 > 1.0) | (S1 < 0.0)) or jnp.any((ST > 1.0) | (ST < 0.0)):
        warnings.warn(
            "eFAST: some indices are outside [0, 1], suggesting "
            "insufficient samples or near-zero output variance",
            stacklevel=2,
            category=JaxgsaWarning,
        )

    return EFASTResult(
        S1=S1,
        ST=ST,
        problem=problem,
        invalid=invalid,
        omega_0=omega_0,
        M=M,
    )
