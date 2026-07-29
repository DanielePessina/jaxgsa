"""Cross-method agreement on correlated variance-based indices.

jaxgsa offers two estimation routes for the same two conditional-variance
quantities under a Gaussian copula:

- ``jaxgsa.vkoga`` — the surrogate route: fit a kernel surrogate to given
  ``(X, Y)`` data, then nested quasi-Monte-Carlo against the surrogate.
- ``jaxgsa.kucherenko`` — the design route: evaluate the actual model on a
  conditional-copula design, then single-loop estimators.

On the linear-Gaussian closed form (``Y = a . X``, ``X ~ N(0, R)``,
``rho_12 = 0.6``) both must agree with the analytic values and with each
other: ``vkoga.S_TC == kucherenko.S1`` estimates ``V(E(Y|X_i))/V(Y)`` and
``vkoga.S_TU == kucherenko.ST`` estimates ``E(V(Y|X_{~i}))/V(Y)``.

Tolerances: kucherenko's measured error at base_n = 4096 is <= 1.5e-3 over
eight seeds (5e-3 with headroom); vkoga keeps its established 1e-2 / 2e-2
budgets from ``test_vkoga.py`` — they are not loosened here. The route-vs-
route tolerance is the sum of the two per-route budgets.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jaxgsa
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


def _analytic_indices(a: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form (S_TC/S1, S_TU/ST) for a linear model on N(0, R) inputs."""
    var_y = float(a @ R @ a)
    S_TC = (R @ a) ** 2 / var_y
    D = a.shape[0]
    S_TU = np.empty(D)
    for i in range(D):
        rest = [j for j in range(D) if j != i]
        r = R[rest, i]
        S_TU[i] = a[i] ** 2 * (1.0 - r @ np.linalg.solve(R[np.ix_(rest, rest)], r)) / var_y
    return S_TC, S_TU


@pytest.fixture(scope="module")
def three_routes():
    """Analytic, vkoga, and kucherenko indices for the same problem."""
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    analytic = _analytic_indices(A_COEF, R_GAUSS)

    # Surrogate route: independent training design, x64 (the kernel solve
    # squares the condition number; same configuration as test_vkoga.py).
    with jax.enable_x64():
        X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 1536, seed=7)
        Y = X @ A_COEF
        vkoga = jaxgsa.vkoga.analyze(
            problem,
            X,
            Y,
            gamma=2.0,
            ridge=1e-10,
            max_centers=150,
            n_outer=256,
            n_inner=64,
            n_variance=4096,
            seed=0,
        )

    # Design route: the actual model evaluated on the conditional design.
    ks = jaxgsa.kucherenko.sample(problem, 4096, seed=1)
    kucherenko = jaxgsa.kucherenko.analyze(ks, ks.samples @ A_COEF)

    return analytic, vkoga, kucherenko


def test_analytic_vs_kucherenko(three_routes):
    (S_first, S_total), _, kucherenko = three_routes
    np.testing.assert_allclose(np.asarray(kucherenko.S1), S_first, atol=5e-3)
    np.testing.assert_allclose(np.asarray(kucherenko.ST), S_total, atol=5e-3)


def test_analytic_vs_vkoga(three_routes):
    (S_first, S_total), vkoga, _ = three_routes
    # The budgets established in test_vkoga.py, unchanged.
    np.testing.assert_allclose(np.asarray(vkoga.S_TC), S_first, atol=1e-2)
    np.testing.assert_allclose(np.asarray(vkoga.S_TU), S_total, atol=2e-2)


def test_vkoga_vs_kucherenko(three_routes):
    """The surrogate route and the design route agree with each other."""
    _, vkoga, kucherenko = three_routes
    np.testing.assert_allclose(np.asarray(vkoga.S_TC), np.asarray(kucherenko.S1), atol=1e-2 + 5e-3)
    np.testing.assert_allclose(np.asarray(vkoga.S_TU), np.asarray(kucherenko.ST), atol=2e-2 + 5e-3)


def test_routes_agree_on_the_reading(three_routes):
    """Both routes tell the same qualitative story about the coupling."""
    _, vkoga, kucherenko = three_routes
    for first, total in (
        (np.asarray(vkoga.S_TC), np.asarray(vkoga.S_TU)),
        (np.asarray(kucherenko.S1), np.asarray(kucherenko.ST)),
    ):
        # x1 and x2 share correlated variance: large first-order, small total.
        assert first[0] > 0.8 and total[0] < 0.4
        assert first[1] > 0.5 and total[1] < 0.15
        # x3 is uncorrelated and additive: the two coincide.
        assert abs(first[2] - total[2]) < 2e-2
