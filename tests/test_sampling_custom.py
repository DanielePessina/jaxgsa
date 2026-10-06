"""Uniform transformations share validation across ordinary input sampling."""

import numpy as np
import pytest

from jaxgsa import sampling
from jaxgsa._core.dependence import generate_uniform_points, transform_samples
from jaxgsa.problem import Problem

PROBLEM = Problem.from_dict({"a": (0, 1), "b": (0, 1)})


def transform(
    u: np.ndarray,
    *,
    conditioning_rows: np.ndarray | None = None,
    fixed_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Use one uniform to generate two deterministically related inputs."""
    result = np.column_stack((u[:, 0], u[:, 0] ** 2))
    if conditioning_rows is not None:
        result[:, fixed_indices] = conditioning_rows[:, fixed_indices]
    return result


def test_generic_custom_matches_direct_transformation() -> None:
    points = generate_uniform_points(16, 1, sequence="sobol", rng=np.random.default_rng(42))
    samples = sampling.sample(PROBLEM, 16, sampler=transform, sampler_dimension=1, seed=42)
    np.testing.assert_array_equal(samples, transform(points))


def test_generic_builtin_monte_carlo_preserves_existing_samples() -> None:
    np.testing.assert_array_equal(
        sampling.sample(PROBLEM, 19, sequence="monte_carlo", seed=7),
        sampling.monte_carlo(PROBLEM, 19, seed=7),
    )


def test_sobol_exact_count_and_monte_carlo_arbitrary_count() -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        sampling.sample(PROBLEM, 19, sampler=transform)
    assert sampling.sample(PROBLEM, 19, sampler=transform, sequence="monte_carlo").shape == (19, 2)
    for value in (True, 0):
        with pytest.raises(ValueError, match="positive integers"):
            generate_uniform_points(value, 2, sequence="sobol", rng=np.random.default_rng())


def test_callback_mutation_does_not_change_callers_inputs() -> None:
    points = np.full((4, 2), 0.25)
    rows = np.full((4, 2), 0.5)

    def mutate(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        del fixed_indices
        u[:] = 0
        assert conditioning_rows is not None
        result = conditioning_rows.copy()
        conditioning_rows[:] = 0
        return result

    transform_samples(mutate, points, 2, conditioning_rows=rows, fixed_indices=(0, 1))
    np.testing.assert_array_equal(points, np.full((4, 2), 0.25))
    np.testing.assert_array_equal(rows, np.full((4, 2), 0.5))


def test_custom_cannot_accompany_correlation() -> None:
    with pytest.raises(ValueError, match="correlation"):
        sampling.sample(PROBLEM.with_correlation(np.eye(2)), 8, sampler=transform)
