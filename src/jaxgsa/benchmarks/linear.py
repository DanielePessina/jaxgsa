"""Linear additive model for sensitivity analysis benchmarking.

The model is a weighted sum of independent uniform inputs. It is the
simplest benchmark in this package. The model is purely additive, so the
first-order indices equal the total-order indices and every second-order
interaction is exactly zero.

Useful for:
    - Checking that a sensitivity method reports zero interactions.
    - Checking that S1 == ST when there are no interactions.
    - Measuring convergence rates against a trivial exact solution.
"""

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa.problem import Problem

# Increasing weights (1, 2, 3) test whether the method correctly ranks
# input importance: x3 should have the largest index, x1 the smallest.
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
        X: Input array, shape ``(N, D)``.
        coeffs: Coefficient per dimension.

    Returns:
        Function values, shape ``(N,)``.
    """
    # One matrix-vector product gives the weighted sum for all N samples at once.
    return X @ jnp.asarray(coeffs)


def _additive_sobol_indices(
    coeffs: np.ndarray,
    var_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sobol indices for a purely additive linear model ``Y = sum c_j x_j``.

    The uniform (:mod:`~jaxgsa.benchmarks.linear`) and Gaussian
    (:mod:`~jaxgsa.benchmarks.gaussian_linear`) benchmarks share this helper.
    They differ only in how ``var_x`` comes from the input distributions.
    The indices have a closed form:

    .. math::
        V_j = c_j^2 \\operatorname{Var}(x_j), \\quad
        S_{1,j} = S_{T,j} = V_j / V(Y), \\quad
        S_{2,jk} = 0

    Args:
        coeffs: Coefficient per dimension.
        var_x: Variance of each independent input.

    Returns:
        ``(S1, ST, S2)``. ``S1`` and ``ST`` have shape ``(D,)`` and are equal
        because the model has no interactions. ``S2`` is a ``(D, D)`` matrix
        that is zero off the diagonal and NaN on the diagonal.
    """
    c = np.asarray(coeffs, dtype=float)
    var_x = np.asarray(var_x, dtype=float)
    D = len(c)
    # Variance propagation for linear functions: Vi = c_j^2 * Var(x_j)
    Vi = c**2 * var_x
    VY = Vi.sum()

    # Purely additive model => no interactions => S1 = ST and sum(S1) = 1
    S1 = Vi / VY
    ST = S1.copy()

    # Diagonal is NaN by convention (S2_jj is undefined); off-diagonals are
    # exactly zero because a purely additive model has no interactions.
    S2 = np.zeros((D, D))
    np.fill_diagonal(S2, np.nan)

    return S1, ST, S2


def analytical_indices(
    coeffs: tuple[float, ...] = DEFAULT_COEFFS,
    bounds: tuple[tuple[float, float], ...] = DEFAULT_BOUNDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytical Sobol indices for the linear additive model.

    For ``Y = sum c_j x_j`` with independent inputs, the indices have a
    closed form:

    .. math::
        V_j = c_j^2 \\operatorname{Var}(x_j), \\quad
        S_{1,j} = S_{T,j} = V_j / V(Y), \\quad
        S_{2,jk} = 0

    Args:
        coeffs: Coefficient per dimension.
        bounds: ``(low, high)`` bounds for each uniform input.

    Returns:
        ``(S1, ST, S2)``. ``S1`` and ``ST`` have shape ``(D,)`` and are equal
        because the model has no interactions. ``S2`` is a ``(D, D)`` matrix
        that is zero off the diagonal and NaN on the diagonal.
    """
    c = np.asarray(coeffs, dtype=float)
    # Var(Uniform[lo, hi]) = (hi - lo)^2 / 12
    var_x = np.array([(hi - lo) ** 2 / 12.0 for lo, hi in bounds])
    return _additive_sobol_indices(c, var_x)


def analytical_shapley(
    coeffs: tuple[float, ...] = DEFAULT_COEFFS,
    bounds: tuple[tuple[float, float], ...] = DEFAULT_BOUNDS,
) -> np.ndarray:
    """Compute analytical Shapley effects for the linear additive model.

    For independent inputs the Shapley effect of input j is the average of
    its partial-variance shares over all subsets that contain j (Owen, 2014).
    A purely additive model has no interaction terms. Every share beyond the
    main effect is therefore zero, and the Shapley effects equal the
    first-order indices exactly:

    .. math::
        \\mathrm{Sh}_j = S_{1,j} = c_j^2 \\operatorname{Var}(x_j) / V(Y)

    Args:
        coeffs: Coefficient per dimension.
        bounds: ``(low, high)`` bounds for each uniform input.

    Returns:
        Shapley effects, shape ``(D,)``. They equal ``S1`` and sum to 1.
    """
    # No interactions => Shapley == first-order; reuse the S1 derivation.
    S1, _, _ = analytical_indices(coeffs, bounds)
    return S1


ANALYTICAL_S1, ANALYTICAL_ST, ANALYTICAL_S2 = analytical_indices()
ANALYTICAL_SHAPLEY = analytical_shapley()
