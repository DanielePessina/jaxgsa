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
