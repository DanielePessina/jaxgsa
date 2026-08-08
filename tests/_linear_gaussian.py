"""Shared linear-Gaussian closed-form reference for correlated-index tests.

One model serves ``test_vkoga.py``, ``test_kucherenko.py``, and
``test_correlated_agreement.py``: ``Y = a . X`` with ``X ~ N(0, R)`` under a
Gaussian copula, where every correlated variance-based index is closed form:

    V(Y)                = a' R a
    S_TC_i (= S1_i)     = (R a)_i^2 / V(Y)
    S_TU_i (= ST_i)     = a_i^2 (1 - R_i,rest R_rest^-1 R_rest,i) / V(Y)

and the model is additive, so ``S_U = S_TU`` and ``S_IU = 0``.
"""

from __future__ import annotations

import numpy as np

from jaxgsa.problem import Problem

A_COEF = np.array([2.0, 1.0, 0.5])
RHO = 0.6

GAUSS_PROBLEM = Problem.from_dict(
    {
        "x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x2": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x3": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
    }
)

R_GAUSS = np.eye(3)
R_GAUSS[0, 1] = R_GAUSS[1, 0] = RHO

# --- asymmetric-structure regression case ------------------------------------
# R_GAUSS has one non-zero off-diagonal, so its correlation structure is
# symmetric under swapping x1 and x2 and every coefficient magnitude differs.
# That cannot catch a parameter-axis transposition. This D=4 case has six
# distinct off-diagonals of mixed sign and four distinct coefficients, so any
# mix-up of the parameter axis moves the indices.

A_COEF_ASYM = np.array([1.0, -2.0, 0.5, 3.0])

ASYM_PROBLEM = Problem.from_dict(
    {
        f"x{i + 1}": {"dist": "gaussian", "mean": 0.0, "variance": 1.0}
        for i in range(A_COEF_ASYM.shape[0])
    }
)

R_ASYM = np.eye(4)
for (_i, _j), _rho in zip(
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    (0.6, -0.3, 0.1, 0.15, -0.25, 0.45),
):
    R_ASYM[_i, _j] = R_ASYM[_j, _i] = _rho
assert np.linalg.eigvalsh(R_ASYM).min() > 0, "R_ASYM must be positive definite"


def analytic_indices(a: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Closed-form (S_TC/S1, S_TU/ST, V(Y)) for a linear model on N(0, R) inputs."""
    var_y = float(a @ R @ a)
    S_first = (R @ a) ** 2 / var_y
    D = a.shape[0]
    S_total = np.empty(D)
    for i in range(D):
        rest = [j for j in range(D) if j != i]
        r = R[rest, i]
        S_total[i] = a[i] ** 2 * (1.0 - r @ np.linalg.solve(R[np.ix_(rest, rest)], r)) / var_y
    return S_first, S_total, var_y
