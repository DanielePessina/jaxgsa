"""Validation gaps in Problem and the correlation entry point."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from jaxgsa._core.copula import build_conditional_plan, canonicalize_correlation
from jaxgsa.problem import CategoricalSpec, Problem


def test_zero_parameter_problem_rejected() -> None:
    with pytest.raises(ValueError, match="at least one parameter"):
        Problem([], [])
    with pytest.raises(ValueError, match="at least one parameter"):
        Problem.from_dict({})


def test_bare_string_labels_rejected() -> None:
    labels: Any = "ab"
    with pytest.raises(TypeError, match="Categorical labels"):
        Problem.from_dict({"c": {"dist": "categorical", "probs": [0.5, 0.5], "labels": labels}})
    with pytest.raises(TypeError, match="Categorical labels"):
        CategoricalSpec((0.5, 0.5), labels=labels)


@pytest.mark.parametrize("bad", [("y", "y"), ("y", 2)])
def test_bad_output_names_rejected(bad: tuple) -> None:
    with pytest.raises(ValueError, match="output names"):
        Problem(["a"], [(0.0, 1.0)], output_names=bad)
    problem = Problem(["a"], [(0.0, 1.0)])
    with pytest.raises(ValueError, match="output names"):
        problem.with_output_names(bad)


def test_complex_correlation_rejected() -> None:
    R = np.array([[1.0, 0.2j], [-0.2j, 1.0]])
    with pytest.raises(TypeError, match="complex"):
        Problem(["a", "b"], [(0.0, 1.0)] * 2, correlation=R)
    with pytest.raises(TypeError, match="complex"):
        canonicalize_correlation(R, 2)


def test_invalid_correlation_type_rejected_without_correlation() -> None:
    bad_type: dict = {"correlation_type": "pearson"}
    with pytest.raises(ValueError, match="correlation_type"):
        Problem(["a"], [(0.0, 1.0)], **bad_type)
    with pytest.raises(ValueError, match="correlation_type"):
        Problem.from_dict({"a": (0.0, 1.0)}, **bad_type)


def test_plan_has_no_marginal_factor() -> None:
    plan = build_conditional_plan(np.full((3, 3), 0.3) + 0.7 * np.eye(3))
    assert not hasattr(plan, "chol_marginal")


def test_plan_matches_direct_conditionals_and_stays_small() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    cov = A @ A.T + 6 * np.eye(6)
    d = np.sqrt(np.diag(cov))
    R = cov / np.outer(d, d)
    plan = build_conditional_plan(R)
    assert not hasattr(plan, "chol_rest")
    for i in range(6):
        rest = np.array([j for j in range(6) if j != i])
        r = R[rest, i]
        np.testing.assert_allclose(
            plan.beta_self[i], np.linalg.solve(R[np.ix_(rest, rest)], r), atol=1e-12
        )
        np.testing.assert_allclose(
            plan.std_self[i] ** 2, 1 - r @ np.linalg.solve(R[np.ix_(rest, rest)], r), atol=1e-12
        )
        np.testing.assert_array_equal(plan.beta_rest(i), r)
        np.testing.assert_allclose(
            plan.rest_factor(i) @ plan.rest_factor(i).T,
            R[np.ix_(rest, rest)] - np.outer(r, r),
            atol=1e-12,
        )


def test_gaussian_draws_plan_is_lazy() -> None:
    from jaxgsa._core.dependence import GaussianDraws

    problem = Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0)}).with_correlation(
        np.array([[1.0, 0.5], [0.5, 1.0]])
    )
    draws = GaussianDraws(problem)
    assert draws._plan is None
    draws.joint(np.full((3, 2), 0.3))
    assert draws._plan is None
    draws.self_given_rest(np.full((3, 1), 0.3), np.zeros((3, 2)), 0)
    assert draws._plan is not None
