"""Shared linear-Gaussian closed-form reference for correlated-index tests.

One model serves ``test_vkoga.py``, ``test_kucherenko.py``, and
``test_correlated_agreement.py``: ``Y = a . X`` with ``X ~ N(0, R)`` under a
Gaussian copula, where every correlated variance-based index is closed form:

    V(Y)   = a' R a
    S_TC_i = (R a)_i^2 / V(Y)
    S_TU_i = a_i^2 (1 - R_i,rest R_rest^-1 R_rest,i) / V(Y)

``S_TC`` is the full (correlated) first-order index ``V(E(Y|X_i)) / V(Y)``.
``S_TU`` is the total-order index ``E(V(Y|X_-i)) / V(Y)`` — this one has no
separate "correlated" and "uncorrelated" version, because the law of total
variance forces ``E(V(Y|X_-i)) = V(Y) - V(E(Y|X_-i))`` for any joint
distribution, correlated or not.

``S_TC`` is *not* bounded by ``S_TU``. The familiar Sobol' property
``S1 <= ST`` comes from the independent-input ANOVA decomposition, where
every interaction term has nonnegative variance. Under correlation that
decomposition does not apply, and ``S_TC`` can exceed ``S_TU`` — this
fixture does, at every nonzero off-diagonal entry of ``R_GAUSS``. Calling
``S_TC`` "S1" and ``S_TU`` "ST" (an earlier version of this module did)
invites exactly that wrong expectation, so this module does not use those
names.

The one first-order/total-order pair that a caller *can* rely on being
ordered is the independent pair, ``S_U`` and ``S_TU``. Because this model is
additive (no interaction terms), the fitted component for ``X_i`` is exactly
``a_i * X_i``, so ``S_U_i = E(V(a_i X_i | X_-i)) / V(Y)`` is the same
expression as ``S_TU_i`` — the two are equal, not merely ordered. That
identity is asserted below for both fixtures, and is what "the model is
additive, so ``S_U = S_TU`` and ``S_IU = 0``" (still true) actually rests on.

**Unit-variance assumption.** ``R`` must be a *correlation* matrix — every
diagonal entry equal to 1 — not a general covariance matrix. Both formulas
above use ``V(X_i) = R_ii = 1`` implicitly (``S_TC``'s numerator would
otherwise need `V(X_i)` in it, and ``S_TU``'s ``a_i**2`` term is really
``a_i**2 * V(X_i)``). Passing a covariance matrix with a non-unit diagonal
silently produces an index outside ``[0, 1]``: for
``Sigma = [[4, 1.2], [1.2, 1]]`` and ``a = [1, 1]``, ``S_TC`` comes back as
``[3.654, 0.654]``.
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
    """Closed-form ``(S_TC, S_TU, V(Y))`` for a linear model on ``N(0, R)`` inputs.

    ``R`` must be a correlation matrix (unit diagonal); see the module
    docstring. ``S_TC`` and ``S_TU`` are not the ``S1``/``ST`` pair a Sobol'
    reader expects to be ordered — see :func:`analytic_independent_first_order`
    for the pair that is.
    """
    var_y = float(a @ R @ a)
    S_TC = (R @ a) ** 2 / var_y
    D = a.shape[0]
    S_TU = np.empty(D)
    for i in range(D):
        rest = [j for j in range(D) if j != i]
        r = R[rest, i]
        S_TU[i] = a[i] ** 2 * (1.0 - r @ np.linalg.solve(R[np.ix_(rest, rest)], r)) / var_y
    return S_TC, S_TU, var_y


def analytic_independent_first_order(a: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Closed-form ``S_U`` (independent first-order index) for this model.

    ``S_U_i`` is the share of ``V(Y)`` explained by the part of ``X_i`` that
    the other inputs cannot predict. Write that part as the linear residual
    ``Z_i = X_i - b' X_-i`` with ``b = R_-i,-i^-1 R_-i,i``. Then

        ``S_U_i = Cov(Y, Z_i)^2 / (V(Z_i) V(Y))``

    which is what this function computes, straight from ``R``. It uses the
    whole ``i``-th column of ``R`` and the whole coefficient vector, so it is
    a different calculation from the ``a_i^2 (1 - r' R^-1 r)`` line in
    :func:`analytic_indices`, not a rename of it. For this additive model the
    two must agree, and :func:`_check_su_le_stu` asserts that they do.

    Args:
        a: Coefficient vector of the linear model.
        R: Correlation matrix of the inputs (unit diagonal).

    Returns:
        ``S_U``, one entry per parameter.
    """
    D = a.shape[0]
    var_y = float(a @ R @ a)
    S_U = np.empty(D)
    for i in range(D):
        rest = [j for j in range(D) if j != i]
        r = R[rest, i]
        b = np.linalg.solve(R[np.ix_(rest, rest)], r)
        # Cov(X_j, Z_i) for every j at once. It is 0 for every j in rest, by
        # construction of the residual, and V(Z_i) at j == i.
        cov_x_z = R[:, i] - R[:, rest] @ b
        var_z = float(R[i, i] - r @ b)
        S_U[i] = float(a @ cov_x_z) ** 2 / (var_z * var_y)
    return S_U


def _check_su_le_stu(a: np.ndarray, R: np.ndarray, label: str) -> None:
    """The one first/total pairing this fixture guarantees is ordered.

    ``S_U <= S_TU`` always holds by construction (``S_IU = S_TU - S_U`` is a
    variance and cannot be negative). For this additive model the two are
    equal, and the two sides come from different algebra: ``S_TU`` from the
    conditional variance of ``X_i``, ``S_U`` from the residual projection
    (see :func:`analytic_independent_first_order`). The check therefore
    fails if either formula drifts on its own. Measured agreement on both
    fixtures is 1e-16.
    """
    S_U = analytic_independent_first_order(a, R)
    _, S_TU, _ = analytic_indices(a, R)
    assert np.all(S_U <= S_TU + 1e-9), f"{label}: S_U must not exceed S_TU"
    np.testing.assert_allclose(
        S_U, S_TU, atol=1e-9, err_msg=f"{label}: additive model needs S_U == S_TU"
    )


_check_su_le_stu(A_COEF, R_GAUSS, "R_GAUSS")
_check_su_le_stu(A_COEF_ASYM, R_ASYM, "R_ASYM")
