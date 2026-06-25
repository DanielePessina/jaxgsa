"""Linear additive model for sensitivity analysis benchmarking.

The simplest possible benchmark: a weighted sum of independent uniform
inputs. Because the model is purely additive, first-order indices equal
total-order indices and all second-order interactions are exactly zero.

Useful for:
    - Verifying that a SA method correctly identifies zero interactions.
    - Sanity-checking that S1 == ST when there are no interactions.
    - Testing convergence rates (exact analytical solution is trivial).
"""

import jax.numpy as jnp
import numpy as np
from jax import Array

from gsax.problem import Problem

DEFAULT_COEFFS = (1.0, 2.0, 3.0)
DEFAULT_BOUNDS = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

PROBLEM = Problem(
    names=tuple(f"x{i + 1}" for i in range(len(DEFAULT_COEFFS))),
    bounds=DEFAULT_BOUNDS,
)


def evaluate(
    X: Array,
    coeffs: tuple[float, ...] = DEFAULT_COEFFS,
) -> Array:
    """Evaluate the linear additive model.

    .. math::
        f(\\mathbf{x}) = \\sum_{j=1}^{D} c_j \\, x_j

    Args:
        X: Input array of shape ``(N, D)``.
        coeffs: Coefficient per dimension.

    Returns:
        Array of shape ``(N,)`` with function values.
    """
    return X @ jnp.asarray(coeffs)


def analytical_indices(
    coeffs: tuple[float, ...] = DEFAULT_COEFFS,
    bounds: tuple[tuple[float, float], ...] = DEFAULT_BOUNDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytical Sobol indices for the linear additive model.

    For ``Y = sum c_j x_j`` with independent inputs:

    .. math::
        V_j = c_j^2 \\operatorname{Var}(x_j), \\quad
        S_{1,j} = S_{T,j} = V_j / V(Y), \\quad
        S_{2,jk} = 0

    Args:
        coeffs: Coefficient per dimension.
        bounds: ``(low, high)`` bounds for each uniform input.

    Returns:
        ``(S1, ST, S2)`` where S1 == ST (no interactions) and S2 is
        all-zero off-diagonal with NaN on the diagonal.
    """
    c = np.asarray(coeffs, dtype=float)
    D = len(c)
    var_x = np.array([(hi - lo) ** 2 / 12.0 for lo, hi in bounds])
    Vi = c**2 * var_x
    VY = Vi.sum()

    S1 = Vi / VY
    ST = S1.copy()

    S2 = np.full((D, D), np.nan)
    for j in range(D):
        for k in range(j + 1, D):
            S2[j, k] = 0.0
            S2[k, j] = 0.0

    return S1, ST, S2


ANALYTICAL_S1, ANALYTICAL_ST, ANALYTICAL_S2 = analytical_indices()
