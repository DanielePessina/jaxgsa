"""The shared draw strategies and the sampler contract in ``_core/dependence.py``."""

import numpy as np
import pytest

from jaxgsa._core.dependence import (
    CustomDraws,
    GaussianDraws,
    draws_for,
    transform_samples,
)
from jaxgsa.problem import Problem

INDEPENDENT = Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0)})
GAUSSIAN = Problem.from_dict(
    {"a": (0.0, 1.0), "b": (0.0, 1.0)}, correlation=[[1.0, 0.8], [0.8, 1.0]]
)


def identity_sampler(uniform_points, *, conditioning_rows=None, fixed_indices=()):
    """Transform uniforms while preserving the requested physical columns."""
    result = uniform_points.copy()
    if conditioning_rows is not None:
        result[:, fixed_indices] = conditioning_rows[:, fixed_indices]
    return result


CUSTOM = INDEPENDENT.with_sampler(identity_sampler)


@pytest.mark.parametrize(
    "values, message",
    [
        (np.full((3, 2), np.nan), "finite"),
        (np.zeros((3, 2), dtype=complex), "real numeric"),
        (np.full((3, 2), "1"), "real numeric"),
    ],
)
def test_joint_rejects_malformed_output(values, message):
    def sampler(u, *, conditioning_rows=None, fixed_indices=()):
        return values

    with pytest.raises(ValueError, match=message):
        transform_samples(sampler, np.zeros((3, 2)), 2)


def test_joint_output_is_owned_float64():
    source = np.ones((3, 2), dtype=np.float32)

    def sampler(u, *, conditioning_rows=None, fixed_indices=()):
        return source

    output = transform_samples(sampler, np.zeros((3, 2)), 2)
    output[0, 0] = 9
    assert source[0, 0] == 1
    assert output.dtype == np.float64


def test_conditional_protects_inputs_and_checks_fixed_columns():
    rows = np.arange(6, dtype=float).reshape(3, 2)
    original = rows.copy()
    uniforms = np.full((3, 2), 0.25)

    def sampler(u, *, conditioning_rows=None, fixed_indices=()):
        assert conditioning_rows is not None
        conditioning_rows[:, 1] = u[:, 1]
        u[:] = 0
        return conditioning_rows

    output = transform_samples(sampler, uniforms, 2, conditioning_rows=rows, fixed_indices=(0,))
    np.testing.assert_array_equal(rows, original)
    np.testing.assert_array_equal(uniforms, np.full((3, 2), 0.25))
    np.testing.assert_array_equal(output[:, 0], original[:, 0])
    assert not np.array_equal(output[:, 1], original[:, 1])
    with pytest.raises(ValueError, match="fixed columns"):
        transform_samples(sampler, uniforms, 2, conditioning_rows=rows, fixed_indices=(1,))
    np.testing.assert_array_equal(rows, original)


def test_draws_for_picks_the_strategy():
    assert isinstance(draws_for(INDEPENDENT), GaussianDraws)
    assert isinstance(draws_for(GAUSSIAN), GaussianDraws)
    assert isinstance(draws_for(CUSTOM), CustomDraws)


def test_custom_draws_needs_a_sampler():
    with pytest.raises(ValueError, match="needs a problem with a sampler"):
        CustomDraws(INDEPENDENT)


@pytest.mark.parametrize("problem", [GAUSSIAN, CUSTOM], ids=["gaussian", "custom"])
def test_conditional_redraws_keep_fixed_columns(problem):
    draws = draws_for(problem)
    rng = np.random.default_rng(0)
    rows = draws.joint(rng.random((32, draws.dimension)))
    for index in (0, 1):
        kept = draws.rest_given_self(rng.random((32, draws.rest_dimension)), rows, index)
        np.testing.assert_array_equal(kept[:, index], rows[:, index])
        kept = draws.self_given_rest(rng.random((32, draws.self_dimension)), rows, index)
        rest = 1 - index
        np.testing.assert_array_equal(kept[:, rest], rows[:, rest])


def test_gaussian_draws_on_an_independent_problem_are_independent():
    draws = draws_for(INDEPENDENT)
    rows = draws.joint(np.random.default_rng(1).random((20_000, 2)))
    assert abs(np.corrcoef(rows[:, 0], rows[:, 1])[0, 1]) < 0.03
    noise = np.random.default_rng(2).random((20_000, draws.rest_dimension))
    redrawn = draws.rest_given_self(noise, rows, 0)
    assert abs(np.corrcoef(redrawn[:, 0], redrawn[:, 1])[0, 1]) < 0.03


def test_gaussian_draws_on_a_correlated_problem_are_correlated():
    draws = draws_for(GAUSSIAN)
    rows = draws.joint(np.random.default_rng(1).random((20_000, 2)))
    assert np.corrcoef(rows[:, 0], rows[:, 1])[0, 1] > 0.7
    unit = draws.to_unit(rows)
    assert np.all((unit >= 0) & (unit <= 1))


def test_custom_draws_to_unit_maps_rows_in_support():
    draws = draws_for(CUSTOM)
    rows = np.array([[0.1, 0.9], [0.5, 0.5]])
    np.testing.assert_allclose(draws.to_unit(rows), rows, atol=1e-6)
    np.testing.assert_array_equal(draws.to_physical(rows), rows)
