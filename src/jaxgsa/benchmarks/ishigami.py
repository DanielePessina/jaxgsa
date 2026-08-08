"""Ishigami test function for sensitivity analysis benchmarking.

A standard 3-input benchmark with known analytical Sobol indices. It is
nonlinear and non-monotonic. x3 influences the output only through its
interaction with x1, so x3 has S1 = 0 but ST > 0. That makes the function
a sharp test of whether a method separates first-order from total-order
effects.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa.problem import Problem

# All three inputs are uniform on [-pi, pi], the standard domain for
# the trigonometric terms in the Ishigami function.
PROBLEM = Problem.from_dict(
    {
        "x1": (-np.pi, np.pi),
        "x2": (-np.pi, np.pi),
        "x3": (-np.pi, np.pi),
    }
)


def analytical_indices(
    A: float = 7.0, B: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytical first-order, total-order, and second-order Sobol indices.

    For the Ishigami function ``f(x) = sin(x1) + A*sin^2(x2) + B*x3^4*sin(x1)``
    with ``x_i ~ U[-pi, pi]``, the ANOVA decomposition has a closed form. It
    follows from the Gaussian moment identities for uniform trig integrals.

    The main-effect variances are:

    - ``V1 = (1 + B*pi^4/5)^2 / 2``
    - ``V2 = A^2 / 8``
    - ``V3 = 0``

    The only pairwise interaction is between x1 and x3:

    - ``V13 = 8 * B^2 * pi^8 / 225``

    And the total variance:

    - ``V = 1/2 + B*pi^4/5 + B^2*pi^8/18 + A^2/8``

    Args:
        A: Amplitude of the ``sin^2(x2)`` term (the main effect of x2).
        B: Coefficient of the ``x3^4*sin(x1)`` term, which drives the
            x1-x3 interaction.

    Returns:
        ``(S1, ST, S2)``. ``S1`` and ``ST`` have shape ``(3,)``. ``S2`` is a
        symmetric ``(3, 3)`` matrix with NaN on the diagonal.
    """
    pi4 = np.pi**4
    pi8 = np.pi**8

    # Main-effect variances from the ANOVA decomposition.
    V1 = 0.5 * (1.0 + B * pi4 / 5.0) ** 2
    V2 = A**2 / 8.0
    V3 = 0.0

    # x1-x3 interaction variance (the B*x3^4*sin(x1) cross-term).
    V13 = 8.0 * B**2 * pi8 / 225.0

    # Total variance = sum of all partial variances.
    VY = 0.5 + B * pi4 / 5.0 + B**2 * pi8 / 18.0 + A**2 / 8.0

    Vi = np.array([V1, V2, V3])
    S1 = Vi / VY

    # Total-order: main effect + all interactions involving that input.
    ST = np.array([V1 + V13, V2, V13]) / VY

    # Second-order interaction matrix: only (0, 2) and (2, 0) are nonzero.
    S2 = np.full((3, 3), np.nan)
    S2[0, 1] = 0.0
    S2[1, 0] = 0.0
    S2[0, 2] = V13 / VY
    S2[2, 0] = V13 / VY
    S2[1, 2] = 0.0
    S2[2, 1] = 0.0

    return S1, ST, S2


def analytical_shapley(A: float = 7.0, B: float = 0.1) -> np.ndarray:
    """Compute analytical Shapley effects for the Ishigami function.

    For independent inputs the Shapley effect of input j is (Owen, 2014)

    - ``Sh_j = (1/V) * sum over subsets u containing j of V_u / |u|``

    Each interaction variance is therefore split equally among the inputs
    that take part in it. The Ishigami decomposition has a single interaction
    term, the x1-x3 pair, so its variance ``V13`` is shared half-and-half:

    - ``Sh1 = (V1 + V13/2) / V``
    - ``Sh2 = V2 / V``
    - ``Sh3 = (V13/2) / V``

    The only interaction is one 2-way term, so ``Sh = (S1 + ST) / 2`` holds
    elementwise. The effects sum to 1 exactly.

    Args:
        A: Amplitude of the ``sin^2(x2)`` term (the main effect of x2).
        B: Coefficient of the ``x3^4*sin(x1)`` term, which drives the
            x1-x3 interaction.

    Returns:
        Shapley effects for x1, x2, x3, shape ``(3,)``.
    """
    pi4 = np.pi**4
    pi8 = np.pi**8

    # Same partial variances as in analytical_indices.
    V1 = 0.5 * (1.0 + B * pi4 / 5.0) ** 2
    V2 = A**2 / 8.0
    V13 = 8.0 * B**2 * pi8 / 225.0
    VY = 0.5 + B * pi4 / 5.0 + B**2 * pi8 / 18.0 + A**2 / 8.0

    # The {1,3} interaction variance is split equally between x1 and x3.
    return np.array([V1 + V13 / 2.0, V2, V13 / 2.0]) / VY


# Precomputed analytical solutions for A=7, B=0.1.
# x3 has zero first-order effect: it enters only through the B*x3^4*sin(x1)
# interaction. That makes it a good test for methods that must tell S1=0 from ST>0.
ANALYTICAL_S1, ANALYTICAL_ST, ANALYTICAL_S2 = analytical_indices()
# Shapley effects credit x3 with half the x1-x3 interaction, so 0 < Sh3 < ST3.
ANALYTICAL_SHAPLEY = analytical_shapley()


def evaluate(X: Array, A: float = 7.0, B: float = 0.1) -> Array:
    """Evaluate the Ishigami function.

    f(x) = sin(x1) + A*sin^2(x2) + B*x3^4*sin(x1)

    Args:
        X: Input array, shape ``(N, 3)``, with columns x1, x2, x3.
        A: Amplitude of the ``sin^2(x2)`` term (the main effect of x2).
        B: Coefficient of the ``x3^4*sin(x1)`` term, which drives the
            x1-x3 interaction.

    Returns:
        Function values, shape ``(N,)``.
    """
    # sin(x1): first-order effect of x1.
    # A*sin^2(x2): purely additive in x2; the large A=7 makes it the dominant
    #   first-order term.
    # B*x3^4*sin(x1): x1-x3 interaction; the small B=0.1 keeps it a minority
    #   contribution.
    return jnp.sin(X[:, 0]) + A * jnp.sin(X[:, 1]) ** 2 + B * X[:, 2] ** 4 * jnp.sin(X[:, 0])
