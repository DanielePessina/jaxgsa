"""eFAST (extended FAST) sampling via sinusoidal search curves.

Generates samples along ergodic search curves in the input space using
incommensurate frequencies, one curve per parameter. Each curve assigns
the highest frequency omega_0 to the parameter of interest and lower
complementary frequencies to the remaining parameters.

References:
    Saltelli, Tarantola & Chan (1999). Technometrics 41(1):39-56.
    Cukier et al. (1973). J. Chem. Phys. 59(8):3873-3878.
"""

from __future__ import annotations

import math

import numpy as np

from gsax.problem import Problem
from gsax.sampling import _transform_samples


def _assign_frequencies(D: int, omega_0: int, M: int) -> np.ndarray:
    """Assign complementary frequencies for D-1 non-focal parameters.

    Args:
        D: Number of parameters.
        omega_0: Primary frequency for the focal parameter.
        M: Interference factor.

    Returns:
        (D-1,) array of complementary frequencies.
    """
    if D == 1:
        return np.array([], dtype=np.int64)
    # Spread D-1 complementary frequencies evenly in [1, omega_0/(2M)];
    # if fewer available slots than parameters, wrap cyclically.
    m = omega_0 // (2 * M)
    if m >= D - 1:
        return np.floor(np.linspace(1, m, D - 1)).astype(np.int64)
    return (np.arange(D - 1) % m + 1).astype(np.int64)


def sample(
    problem: Problem,
    N: int,
    *,
    M: int = 4,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Generate eFAST samples along sinusoidal search curves.

    For each of the D parameters, generates N samples along a search
    curve where the focal parameter oscillates at frequency omega_0
    and complementary parameters oscillate at lower frequencies. The
    total output has shape (N * D, D).

    Args:
        problem: Problem definition with parameter distributions.
        N: Number of samples per search curve. Must satisfy N > 4*M^2.
        M: Interference factor (number of harmonics). Default 4.
        seed: Random seed for phase shift reproducibility.

    Returns:
        (N * D, D) sample array in the problem's physical units.
    """
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")

    D = problem.num_vars
    if N <= 4 * M**2:
        raise ValueError(f"N must be > 4*M^2 = {4 * M**2}, got {N}")

    rng = np.random.default_rng(seed)

    # Max integer frequency fitting N samples while keeping M harmonics below Nyquist.
    omega_0 = (N - 1) // (2 * M)
    omega_compl = _assign_frequencies(D, omega_0, M)

    # Parametric variable s in [0, 2pi) — uniform grid along the search curve.
    s = (2 * math.pi / N) * np.arange(N)
    X = np.zeros((N * D, D))

    for i in range(D):
        # Focal param i gets omega_0 (highest freq = most variation = identifiable);
        # remaining params get lower complementary frequencies.
        omega = np.zeros(D, dtype=np.int64)
        omega[i] = omega_0
        idx = [j for j in range(D) if j != i]
        omega[idx] = omega_compl

        # Random phase breaks symmetry so each curve samples a different cross-section.
        phi = 2 * math.pi * rng.random()

        row_slice = slice(i * N, (i + 1) * N)
        for j in range(D):
            # Cukier's transform: arcsin(sin(w*s+phi))/pi + 0.5 maps sinusoidal
            # oscillation to uniform [0,1] marginals (otherwise arcsine-shaped).
            X[row_slice, j] = 0.5 + (1.0 / math.pi) * np.arcsin(np.sin(omega[j] * s + phi))

    # CDF-based transform: map [0,1] samples to the problem's physical parameter space.
    X = _transform_samples(problem, X)

    return X
