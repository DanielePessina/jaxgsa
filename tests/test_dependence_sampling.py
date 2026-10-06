"""Validation of the shared custom dependence sampling contract."""

import numpy as np
import pytest

from jaxgsa._core.dependence import (
    generate_uniform_points,
    transform_samples,
    validate_sampler,
)


def identity_sampler(uniform_points, *, conditioning_rows=None, fixed_indices=()):
    """Transform uniforms while preserving the requested physical columns."""
    result = uniform_points.copy()
    if conditioning_rows is not None:
        result[:, fixed_indices] = conditioning_rows[:, fixed_indices]
    return result


def test_sampler_validation():
    assert validate_sampler(None, None, 2, "sobol") is None
    assert validate_sampler(identity_sampler, None, 2, "sobol") == 2
    assert validate_sampler(identity_sampler, 3, 2, "monte_carlo") == 3
    with pytest.raises(ValueError, match="requires sampler"):
        validate_sampler(None, 3, 2, "sobol")
    with pytest.raises(ValueError, match="sequence"):
        validate_sampler(identity_sampler, None, 2, "invalid")


@pytest.mark.parametrize("sampler", [3, "sampler"])
def test_sampler_requires_callable(sampler):
    with pytest.raises(TypeError, match="callable"):
        validate_sampler(sampler, None, 2, "sobol")


@pytest.mark.parametrize("dimension", [0, -1, True, 1.5])
def test_sampler_requires_positive_integer_dimension(dimension):
    with pytest.raises(ValueError, match="positive integer"):
        validate_sampler(identity_sampler, dimension, 2, "sobol")


@pytest.mark.parametrize(
    "values, message",
    [
        (np.zeros((3, 1)), "shape"),
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
    with pytest.raises(ValueError, match="preserve"):
        transform_samples(sampler, uniforms, 2, conditioning_rows=rows, fixed_indices=(1,))
    np.testing.assert_array_equal(rows, original)


@pytest.mark.parametrize("fixed", [(0, 0), (-1,), (2,), (True,), (0.5,), [0]])
def test_conditional_rejects_invalid_fixed_indices(fixed):
    with pytest.raises(ValueError, match="fixed_indices"):
        transform_samples(identity_sampler, np.zeros((3, 2)), 2, fixed_indices=fixed)


@pytest.mark.parametrize("fixed", [(), (0, 1)])
def test_conditional_supports_empty_and_full_subsets(fixed):
    rows = np.ones((3, 2))
    output = transform_samples(
        identity_sampler, np.zeros((3, 2)), 2, conditioning_rows=rows, fixed_indices=fixed
    )
    expected = rows if fixed else np.zeros((3, 2))
    np.testing.assert_array_equal(output, expected)


def test_nonempty_subset_requires_conditioning_rows():
    with pytest.raises(ValueError, match="conditioning_rows is required"):
        transform_samples(identity_sampler, np.zeros((3, 2)), 2, fixed_indices=(0,))


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf])
def test_uniform_points_reject_invalid_values(value):
    with pytest.raises(ValueError, match="finite|lie in"):
        transform_samples(identity_sampler, np.full((3, 2), value), 2)


@pytest.mark.parametrize("shape", [(0, 2), (3, 0), (3,)])
def test_uniform_points_reject_invalid_shape(shape):
    with pytest.raises(ValueError, match="nonempty two-dimensional"):
        transform_samples(identity_sampler, np.zeros(shape), 2)


@pytest.mark.parametrize("sequence, n", [("sobol", 8), ("monte_carlo", 7)])
def test_uniform_generation_is_reproducible_with_exact_count(sequence, n):
    first = generate_uniform_points(n, 3, sequence=sequence, rng=np.random.default_rng(4))
    second = generate_uniform_points(n, 3, sequence=sequence, rng=np.random.default_rng(4))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (n, 3)
    assert np.all((first >= 0) & (first <= 1))


@pytest.mark.parametrize("n, dimension", [(0, 2), (2, 0), (True, 2), (2, 1.5)])
def test_uniform_generation_requires_positive_integer_sizes(n, dimension):
    with pytest.raises(ValueError, match="positive integers"):
        generate_uniform_points(n, dimension, sequence="sobol", rng=np.random.default_rng(0))


def test_sobol_requires_power_of_two_count():
    with pytest.raises(ValueError, match="power-of-two"):
        generate_uniform_points(3, 2, sequence="sobol", rng=np.random.default_rng(0))
