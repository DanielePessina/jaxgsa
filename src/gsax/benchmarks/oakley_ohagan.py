# ruff: noqa: E501
"""Oakley & O'Hagan (2004) 15-dimensional Gaussian-input benchmark.

A canonical benchmark for variance-based sensitivity analysis with
Gaussian inputs. The function combines linear, trigonometric, and
quadratic terms so that first-order, total-order, and pairwise
interaction Sobol indices are all available in closed form.

.. math::
    f(\\mathbf{x}) = \\mathbf{a}_1^\\top \\mathbf{x}
        + \\mathbf{a}_2^\\top \\sin(\\mathbf{x})
        + \\mathbf{a}_3^\\top \\cos(\\mathbf{x})
        + \\mathbf{x}^\\top M \\mathbf{x}

with :math:`x_i \\sim \\mathcal{N}(0, \\sigma^2)` i.i.d.

The coefficients are the published values from Oakley & O'Hagan (2004,
JRSS-B 66:751-769), embedded as literals so no external data file is
needed.

References:
    Oakley, J. E. and O'Hagan, A. (2004). Probabilistic sensitivity
    analysis of complex models: a Bayesian approach. J. R. Statist.
    Soc. B, 66(3):751-769.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from gsax.problem import Problem

# 15 dimensions — high enough to stress-test scalability and to span
# a wide range of importance levels (some inputs are nearly inert).
D = 15
DEFAULT_SIGMA = 1.0

# Published coefficients from Oakley & O'Hagan (2004, Table 1 / psa_example.txt).
# fmt: off
_M = np.array([
    [-0.0225, -0.185, 0.134, 0.369, 0.172, 0.137, -0.44, -0.0814, 0.713, -0.444, 0.504, -0.0241, -0.0459, 0.217, 0.0559],
    [0.257, 0.0538, 0.258, 0.238, -0.591, -0.0816, -0.287, 0.416, 0.498, 0.0839, -0.111, 0.0332, -0.14, -0.031, -0.223],
    [-0.056, 0.195, 0.0955, -0.286, -0.144, 0.224, 0.145, 0.29, 0.231, -0.319, -0.29, -0.21, 0.431, 0.0244, 0.0449],
    [0.664, 0.431, 0.299, -0.162, -0.315, -0.39, 0.177, 0.058, 0.172, 0.135, -0.353, 0.251, -0.0188, 0.365, -0.325],
    [-0.121, 0.125, 0.107, 0.0466, -0.217, 0.195, -0.0655, 0.0244, -0.0968, 0.194, 0.334, 0.313, -0.0836, -0.253, 0.373],
    [-0.284, -0.328, -0.105, -0.221, -0.137, -0.144, -0.115, 0.224, -0.0304, -0.515, 0.0173, 0.039, 0.361, 0.309, 0.05],
    [-0.0779, 0.00375, 0.887, -0.266, -0.0793, -0.0427, -0.187, -0.356, -0.175, 0.0887, 0.4, -0.056, 0.137, 0.215, -0.0113],
    [-0.0923, 0.592, 0.0313, -0.0331, -0.243, -0.0998, 0.0345, 0.0951, -0.338, 0.00639, -0.612, 0.0813, 0.887, 0.143, 0.148],
    [-0.132, 0.529, 0.127, 0.0451, 0.584, 0.373, 0.114, -0.295, -0.57, 0.463, -0.0941, 0.14, -0.386, -0.449, -0.146],
    [0.0581, -0.323, 0.0931, 0.0724, -0.569, 0.526, 0.237, -0.0118, 0.0718, 0.0783, -0.134, 0.227, 0.144, -0.452, -0.556],
    [0.661, 0.346, 0.141, 0.519, -0.28, -0.16, -0.0684, -0.204, 0.0697, 0.231, -0.0444, -0.165, 0.216, 0.00427, -0.0874],
    [0.316, -0.0276, 0.134, 0.135, 0.054, -0.174, 0.175, 0.0603, -0.179, -0.311, -0.254, 0.0258, -0.43, -0.623, -0.034],
    [-0.29, 0.0341, 0.0349, -0.121, 0.026, -0.335, -0.414, 0.0532, -0.271, -0.0263, 0.41, 0.266, 0.156, -0.187, 0.0199],
    [-0.244, -0.441, 0.0126, 0.249, 0.0711, 0.246, 0.175, 0.00853, 0.251, -0.147, -0.0846, 0.369, -0.3, 0.11, -0.757],
    [0.0415, -0.26, 0.464, -0.361, -0.95, -0.165, 0.00309, 0.0528, 0.225, 0.384, 0.456, -0.186, 0.00823, 0.167, 0.16],
], dtype=float)

# Coefficients for linear, sin, and cos terms respectively.
# Magnitudes increase toward higher indices, creating a natural importance
# gradient: x11-x15 dominate, x1-x5 are nearly inert, x6-x10 intermediate.
_A1 = np.array([0.0118, 0.0456, 0.2297, 0.0393, 0.1177, 0.3865, 0.3897, 0.6061, 0.6159, 0.4005, 1.0741, 1.1474, 0.788, 1.1242, 1.1982], dtype=float)
_A2 = np.array([0.4341, 0.0887, 0.0512, 0.3233, 0.1489, 1.036, 0.9892, 0.9672, 0.8977, 0.8083, 1.8426, 2.4712, 2.3946, 2.0045, 2.2621], dtype=float)
_A3 = np.array([0.1044, 0.2057, 0.0774, 0.273, 0.1253, 0.7526, 0.857, 1.0331, 0.8388, 0.797, 2.2145, 2.0382, 2.4004, 2.0541, 1.9845], dtype=float)
# fmt: on

# Reference S1 values from the original paper; useful for cross-checking
# our analytical derivation against the published table.
PUBLISHED_S1 = np.array([
    0.00156, 0.000186, 0.001307, 0.003045, 0.002905,
    0.023035, 0.024151, 0.026517, 0.046036, 0.014945,
    0.101823, 0.135708, 0.101989, 0.105169, 0.122818,
])

# Gaussian inputs (not uniform) -- one of few SA benchmarks with non-uniform distributions.
PROBLEM = Problem.from_dict({
    f"x{i + 1}": {"dist": "gaussian", "mean": 0.0, "variance": DEFAULT_SIGMA**2}
    for i in range(D)
})


def evaluate(X: Array) -> Array:
    """Evaluate the Oakley & O'Hagan function.

    Args:
        X: Input array of shape ``(N, 15)`` with ``x_i ~ N(0, sigma^2)``.

    Returns:
        Array of shape ``(N,)`` with function values.
    """
    Xj = jnp.asarray(X)
    # Four additive components: linear a1^T x, trigonometric a2^T sin(x) + a3^T cos(x),
    # and the quadratic form x^T M x which introduces all pairwise interactions.
    return (
        Xj @ jnp.asarray(_A1)
        + jnp.sin(Xj) @ jnp.asarray(_A2)
        + jnp.cos(Xj) @ jnp.asarray(_A3)
        # Batched quadratic form: einsum contracts x_i * M_ij * x_j per sample row.
        + jnp.einsum("ni,ij,nj->n", Xj, jnp.asarray(_M), Xj)
    )


def analytical_indices(
    sigma: float = DEFAULT_SIGMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytical first-order, total-order, and second-order Sobol indices.

    Each input enters through the block ``(x, sin x, cos x, x^2)``
    (main effect) plus off-diagonal quadratic cross-terms (pairwise
    interactions). The main-effect variance is ``c_i' Sigma c_i`` where
    ``Sigma`` is the covariance of that block under ``N(0, sigma^2)``,
    and pairwise interaction variance is ``(M_ij + M_ji)^2 * sigma^4``.

    Args:
        sigma: Standard deviation of each input.

    Returns:
        ``(S1, ST, S2)`` where S1 and ST are ``(15,)`` arrays and S2 is
        a ``(15, 15)`` symmetric matrix with NaN on the diagonal.
    """
    s2 = sigma**2
    # Gaussian moment-generating-function identities for X ~ N(0, s2):
    # E[sin X] = 0, E[cos X] = exp(-s2/2), E[sin^2 X] = (1 - exp(-2s2))/2
    es = np.exp(-0.5 * s2)   # = E[cos X], used in cross-covariances
    es2 = np.exp(-2.0 * s2)  # appears in Var[sin X] and Var[cos X]

    # Covariance matrix of the feature vector (x, sin x, cos x, x^2) for X~N(0,s2).
    # Block-diagonal: (x, sin x) decouple from (cos x, x^2) because odd/even symmetry.
    # Each input's main-effect variance is Vi = c_i^T * cov_block * c_i
    # where c_i = [a1_i, a2_i, a3_i, M_ii].
    cov_block = np.array([
        [s2,      s2 * es,          0.0,                  0.0],
        [s2 * es, (1 - es2) / 2,    0.0,                  0.0],
        [0.0,     0.0,              (1 + es2) / 2 - es**2, -s2**2 * es],
        [0.0,     0.0,              -s2**2 * es,           2 * s2**2],
    ])

    # Main-effect variance for each input: quadratic form c_i^T Sigma c_i
    Vi = np.array([
        np.array([_A1[i], _A2[i], _A3[i], _M[i, i]])
        @ cov_block
        @ np.array([_A1[i], _A2[i], _A3[i], _M[i, i]])
        for i in range(D)
    ])

    # Off-diagonal cross-terms x_i*x_k from the quadratic form contribute
    # interaction variance Vpair_ik = ((M_ik + M_ki) * s2)^2 (symmetrized).
    Vpair = ((_M + _M.T) * s2) ** 2
    np.fill_diagonal(Vpair, 0.0)  # diagonal already captured in Vi via M_ii

    # Total variance = sum of all main effects + sum of all unique pairwise interactions.
    total_var = float(Vi.sum() + np.triu(Vpair, 1).sum())
    S1 = Vi / total_var
    # ST_i = main effect + all pairwise interactions involving input i
    ST = (Vi + Vpair.sum(axis=1)) / total_var

    # Second-order interaction matrix: S2_ij = Vpair_ij / total_var.
    S2 = np.full((D, D), np.nan)
    for j in range(D):
        for k in range(j + 1, D):
            val = Vpair[j, k] / total_var
            S2[j, k] = val
            S2[k, j] = val

    return S1, ST, S2


ANALYTICAL_S1, ANALYTICAL_ST, ANALYTICAL_S2 = analytical_indices()
