"""Shared linear-Gaussian closed-form reference for correlated-index tests.

One model serves ``test_vkoga.py``, ``test_kucherenko.py`` and the
custom-sampler tests: ``Y = a . X`` with ``X ~ N(0, R)`` under a
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

**Covariance inputs.** The formulas above assume a *correlation* matrix
``R`` (unit diagonal). :func:`analytic_indices` also accepts a general
positive-definite covariance matrix ``Sigma``. It then uses the general
forms:

    V(Y)   = a' Sigma a
    S_TC_i = (Sigma a)_i^2 / (Sigma_ii V(Y))
    S_TU_i = a_i^2 (Sigma_ii - Sigma_i,rest Sigma_rest^-1 Sigma_rest,i) / V(Y)

With a unit diagonal these reduce to the correlation forms.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri

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


def gaussian_sampler(R: np.ndarray):
    """Build a custom sampler for ``N(0, R)`` written by hand.

    It follows the ``InputSampler`` contract (``sampler(uniform_points, *,
    conditioning_rows=None, fixed_indices=())``). The joint draw is
    ``z @ chol(R).T``. A conditional draw keeps the fixed columns exactly and
    draws the free ones from the closed-form Gaussian conditional. It does not
    use :mod:`jaxgsa._core.copula`, so it is an independent route to the same
    distribution.

    Args:
        R: Correlation matrix of the inputs. The marginals are ``N(0, 1)``.

    Returns:
        The sampler. Pair it with ``GAUSS_PROBLEM.with_sampler(sampler)``.
    """
    D = R.shape[0]
    chol = np.linalg.cholesky(R)

    def sampler(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        z = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
        if not fixed_indices:
            return z @ chol.T
        assert conditioning_rows is not None
        rows = conditioning_rows.copy()
        fixed = list(fixed_indices)
        free = [j for j in range(D) if j not in fixed]
        if not free:
            return rows
        beta = R[np.ix_(free, fixed)] @ np.linalg.inv(R[np.ix_(fixed, fixed)])
        cov = R[np.ix_(free, free)] - beta @ R[np.ix_(fixed, free)]
        rows[:, free] = rows[:, fixed] @ beta.T + z[:, free] @ np.linalg.cholesky(cov).T
        return rows

    return sampler


def analytic_indices(a: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Closed-form ``(S_TC, S_TU, V(Y))`` for a linear model on ``N(0, R)`` inputs.

    ``S_TC`` and ``S_TU`` are not the ``S1``/``ST`` pair a Sobol' reader
    expects to be ordered. See :func:`analytic_independent_first_order` for
    the pair that is.

    Args:
        a: Model coefficients, shape ``(D,)``.
        R: Input correlation matrix, or any positive-definite covariance
            matrix, shape ``(D, D)``. A unit diagonal gives the correlation
            forms of the module docstring.

    Returns:
        A tuple ``(S_TC, S_TU, var_y)``. ``S_TC`` and ``S_TU`` have shape
        ``(D,)``. ``var_y`` is ``V(Y) = a' R a``.
    """
    var_y = float(a @ R @ a)
    var_x = np.diag(R)
    S_TC = (R @ a) ** 2 / (var_x * var_y)
    D = a.shape[0]
    S_TU = np.empty(D)
    for i in range(D):
        rest = [j for j in range(D) if j != i]
        r = R[rest, i]
        cond_var = var_x[i] - r @ np.linalg.solve(R[np.ix_(rest, rest)], r)
        S_TU[i] = a[i] ** 2 * cond_var / var_y
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


# --- subset costs, Shapley effects and PME from the analytic closed index ------
# These helpers do not use jaxgsa.kucherenko. The Shapley and PME values come
# from the permutation (random-order) definitions, not the subset formulas
# that the estimator code uses.


def analytic_closed(a: np.ndarray, R: np.ndarray, subset: tuple[int, ...]) -> float:
    """Closed index ``V(E(Y|X_u)) / V(Y)`` of a linear model on ``N(0, R)``.

    ``E(Y|X_u) = a' R[:,u] R[u,u]^-1 X_u``, so its variance is
    ``a' R[:,u] R[u,u]^-1 R[u,:] a``.
    """
    var_y = float(a @ R @ a)
    if not subset:
        return 0.0
    u = list(subset)
    cross = R[:, u] @ np.linalg.solve(R[np.ix_(u, u)], R[u, :])
    return float(a @ cross @ a) / var_y


def analytic_costs(a: np.ndarray, R: np.ndarray) -> tuple[dict, dict]:
    """Closed and total cost of every subset, keyed by a frozenset of indices.

    The total index is the dual: ``S^T_u = 1 - S^clos_{D - u}``.
    """
    from itertools import combinations

    D = a.shape[0]
    closed, total = {}, {}
    for size in range(D + 1):
        for u in combinations(range(D), size):
            closed[frozenset(u)] = analytic_closed(a, R, u)
    for u in closed:
        total[u] = 1.0 - closed[frozenset(range(D)) - u]
    return closed, total


def shapley_by_permutations(cost: dict, D: int) -> np.ndarray:
    """Shapley values as the uniform average over orderings (Owen 2014)."""
    from itertools import permutations

    out = np.zeros(D)
    perms = list(permutations(range(D)))
    for pi in perms:
        seen: frozenset = frozenset()
        for i in pi:
            out[i] += cost[seen | {i}] - cost[seen]
            seen = seen | {i}
    return out / len(perms)


def pme_by_permutations(total: dict, D: int) -> np.ndarray:
    """Proportional values of a positive game by Eqs. 10 and 11 of Herin et al. (2024).

    ``p(pi)`` is proportional to ``1 / prod_j v(C_j(pi))``. This needs every
    ``v(C_j(pi)) > 0``, so call it on games with no null coalition.
    """
    from itertools import permutations

    weights, perms = [], list(permutations(range(D)))
    for pi in perms:
        chain = [frozenset(pi[: j + 1]) for j in range(D)]
        weights.append(1.0 / np.prod([total[c] for c in chain]))
    weights_a = np.array(weights) / np.sum(weights)
    out = np.zeros(D)
    for w, pi in zip(weights_a, perms):
        seen: frozenset = frozenset()
        for i in pi:
            out[i] += w * (total[seen | {i}] - total[seen])
            seen = seen | {i}
    return out


# A D=4 correlated Gaussian case with mixed-sign coefficients for subset tests.
R_SUBSET4 = np.array(
    [
        [1.0, 0.5, 0.2, 0.0],
        [0.5, 1.0, 0.3, 0.1],
        [0.2, 0.3, 1.0, -0.4],
        [0.0, 0.1, -0.4, 1.0],
    ]
)
A_SUBSET4 = np.array([1.0, 0.5, -1.0, 2.0])
assert np.linalg.eigvalsh(R_SUBSET4).min() > 0, "R_SUBSET4 must be positive definite"
