"""Ishigami test function for sensitivity analysis benchmarking.

A standard benchmark with 3 input parameters (x1, x2, x3) and known
analytical Sobol indices, commonly used to validate sensitivity analysis methods.
"""

import jax.numpy as jnp
import numpy as np
from jax import Array

from gsax.problem import Problem

# All three inputs are uniform on [-pi, pi], the standard domain for
# the trigonometric terms in the Ishigami function.
PROBLEM = Problem.from_dict(
    {
        "x1": (-np.pi, np.pi),
        "x2": (-np.pi, np.pi),
        "x3": (-np.pi, np.pi),
    }
)

# Analytical solutions for A=7, B=0.1.
# x3 has zero first-order effect (enters only through the B*x3^4*sin(x1) interaction),
# making it a good test for methods that must distinguish S1=0 from ST>0.
ANALYTICAL_S1 = [0.3139, 0.4424, 0.0]
ANALYTICAL_ST = [0.5576, 0.4424, 0.2437]
# Only the x1-x3 pair interacts (via the B*x3^4*sin(x1) term); all other pairs are zero.
ANALYTICAL_S2 = {(0, 2): 0.2437}


def evaluate(X: Array, A: float = 7.0, B: float = 0.1) -> Array:
    """Evaluate the Ishigami function.

    f(x) = sin(x1) + A*sin^2(x2) + B*x3^4*sin(x1)

    Args:
        X: Input array of shape ``(N, 3)`` with columns x1, x2, x3.
        A: Model parameter controlling the second-order term.
        B: Model parameter controlling the higher-order interaction term.

    Returns:
        Array of shape ``(N,)`` with the function output for each sample.
    """
    # sin(x1): first-order effect of x1
    # A*sin^2(x2): purely additive in x2 (large A=7 makes it the dominant first-order term)
    # B*x3^4*sin(x1): x1-x3 interaction (small B=0.1 keeps it a minority contribution)
    return jnp.sin(X[:, 0]) + A * jnp.sin(X[:, 1]) ** 2 + B * X[:, 2] ** 4 * jnp.sin(X[:, 0])
