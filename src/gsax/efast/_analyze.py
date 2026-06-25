"""eFAST analysis: compute S1 and ST from Fourier amplitude decomposition.

For each parameter i the model output along its search curve is
Fourier-transformed. First-order indices are estimated from the power
at harmonics of the focal frequency omega_0, and total-order indices
from the complementary low-frequency content.

References:
    Saltelli, Tarantola & Chan (1999). Technometrics 41(1):39-56.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
from jax import Array

from gsax.efast._result import EFASTResult
from gsax.problem import Problem


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
    f = jnp.fft.fft(Y_curve)
    Sp = jnp.abs(f[1 : (N + 1) // 2]) ** 2 / N**2
    V = 2.0 * jnp.sum(Sp)

    harmonics = jnp.arange(1, M + 1) * omega_0
    D1 = 2.0 * jnp.sum(Sp[harmonics - 1])

    compl_range = jnp.arange(omega_0 // 2)
    Dt = 2.0 * jnp.sum(Sp[compl_range])

    S1 = jnp.where(V == 0, jnp.nan, D1 / V)
    ST = jnp.where(V == 0, jnp.nan, 1.0 - Dt / V)
    return S1, ST


def analyze(
    problem: Problem,
    Y: Array,
    *,
    M: int = 4,
) -> EFASTResult:
    """Compute eFAST sensitivity indices from model outputs.

    Args:
        problem: Problem definition with D parameters.
        Y: (N * D,) model outputs evaluated at eFAST samples. The
            samples must have been generated with the same M.
        M: Interference factor used during sampling. Default 4.

    Returns:
        EFASTResult with S1 and ST indices.
    """
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")

    Y = jnp.asarray(Y)
    if Y.ndim != 1:
        raise ValueError(f"eFAST currently supports scalar output (Y.ndim=1), got {Y.ndim}")

    if not jnp.all(jnp.isfinite(Y)):
        n_bad = int(jnp.sum(~jnp.isfinite(Y)))
        warnings.warn(
            f"eFAST: Y contains {n_bad} non-finite values (NaN/Inf) "
            "which will propagate into indices",
            stacklevel=2,
        )

    D = problem.num_vars
    if Y.size % D != 0:
        raise ValueError(
            f"Y length ({Y.size}) must be a multiple of D ({D})"
        )
    N = Y.size // D

    omega_0 = (N - 1) // (2 * M)

    S1_vals = []
    ST_vals = []

    for i in range(D):
        Y_curve = Y[i * N : (i + 1) * N]
        s1, st = _compute_indices(Y_curve, N, M, omega_0)
        S1_vals.append(s1)
        ST_vals.append(st)

    S1 = jnp.stack(S1_vals)
    ST = jnp.stack(ST_vals)

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
