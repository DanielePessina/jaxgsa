"""eFAST analysis: compute S1 and ST from Fourier amplitude decomposition.

For each parameter i the model output along its search curve is
Fourier-transformed. First-order indices are estimated from the power
at harmonics of the focal frequency omega_0, and total-order indices
from the complementary low-frequency content.

References:
    Saltelli, Tarantola & Chan (1999). Technometrics 41(1):39-56.
"""

from __future__ import annotations

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
    Sp = jnp.abs(f[1 : N // 2 + 1]) ** 2 / N**2
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
    num_resamples: int = 0,
    conf_level: float = 0.95,
    seed: int | None = None,
) -> EFASTResult:
    """Compute eFAST sensitivity indices from model outputs.

    Args:
        problem: Problem definition with D parameters.
        Y: (N * D,) model outputs evaluated at eFAST samples. The
            samples must have been generated with the same M.
        M: Interference factor used during sampling. Default 4.
        num_resamples: Number of bootstrap resamples for confidence
            intervals. Set to 0 (default) to skip.
        conf_level: Confidence level for bootstrap CIs. Default 0.95.
        seed: Random seed for bootstrap reproducibility.

    Returns:
        EFASTResult with S1, ST, and optional confidence intervals.
    """
    Y = jnp.asarray(Y)
    if Y.ndim != 1:
        raise ValueError(f"eFAST currently supports scalar output (Y.ndim=1), got {Y.ndim}")

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

    S1_conf = None
    ST_conf = None
    if num_resamples > 0:
        S1_conf, ST_conf = _bootstrap_ci(
            Y, D, N, M, omega_0, num_resamples, conf_level, seed
        )

    return EFASTResult(
        S1=S1,
        ST=ST,
        problem=problem,
        S1_conf=S1_conf,
        ST_conf=ST_conf,
        omega_0=omega_0,
        M=M,
    )


def _bootstrap_ci(
    Y: Array,
    D: int,
    N: int,
    M: int,
    omega_0: int,
    num_resamples: int,
    conf_level: float,
    seed: int | None,
) -> tuple[Array, Array]:
    """Compute bootstrap confidence half-widths for S1 and ST."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n_sub = max(4 * M**2 + 1, N // 2)

    s1_boot = np.zeros((num_resamples, D))
    st_boot = np.zeros((num_resamples, D))

    for r in range(num_resamples):
        for i in range(D):
            Y_curve = np.asarray(Y[i * N : (i + 1) * N])
            idx = rng.choice(N, size=n_sub, replace=True)
            Y_rs = jnp.asarray(Y_curve[idx])
            omega_rs = (n_sub - 1) // (2 * M)
            s1, st = _compute_indices(Y_rs, n_sub, M, omega_rs)
            s1_boot[r, i] = float(s1)
            st_boot[r, i] = float(st)

    from scipy.stats import norm

    z = norm.ppf(0.5 + conf_level / 2.0)
    S1_conf = jnp.asarray(z * np.std(s1_boot, axis=0, ddof=1))
    ST_conf = jnp.asarray(z * np.std(st_boot, axis=0, ddof=1))
    return S1_conf, ST_conf
