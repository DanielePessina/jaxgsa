"""Validation of the shared custom dependence sampling contract."""

import numpy as np
import pytest

from jaxgsa._core.dependence import (
    draw_conditional_samples,
    draw_joint_samples,
    validate_sampler_pair,
)


def test_sampler_pair_requires_both_callbacks():
    assert not validate_sampler_pair(None, None)

    def joint(n, rng):
        return rng.normal(size=(n, 2))

    def conditional(anchors, fixed, rng):
        return anchors

    assert validate_sampler_pair(joint, conditional)
    with pytest.raises(ValueError, match="together"):
        validate_sampler_pair(joint, None)


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
    with pytest.raises(ValueError, match=message):
        draw_joint_samples(lambda n, rng: values, 3, 2, np.random.default_rng(0))


def test_joint_output_is_owned_float64():
    source = np.ones((3, 2), dtype=np.float32)
    output = draw_joint_samples(lambda n, rng: source, 3, 2, np.random.default_rng(0))
    output[0, 0] = 9
    assert source[0, 0] == 1
    assert output.dtype == np.float64


def test_conditional_protects_anchors_and_checks_fixed_columns():
    anchors = np.arange(6, dtype=float).reshape(3, 2)
    original = anchors.copy()

    def sampler(rows, fixed, rng):
        rows[:, 1] = rng.normal(size=len(rows))
        return rows

    output = draw_conditional_samples(sampler, anchors, (0,), np.random.default_rng(0))
    np.testing.assert_array_equal(anchors, original)
    np.testing.assert_array_equal(output[:, 0], original[:, 0])
    assert not np.array_equal(output[:, 1], original[:, 1])
    with pytest.raises(ValueError, match="preserve"):
        draw_conditional_samples(sampler, anchors, (1,), np.random.default_rng(0))
    np.testing.assert_array_equal(anchors, original)


@pytest.mark.parametrize("fixed", [(0, 0), (-1,), (2,), (True,), (0.5,)])
def test_conditional_rejects_invalid_fixed_indices(fixed):
    with pytest.raises(ValueError, match="fixed_indices"):
        draw_conditional_samples(
            lambda rows, fixed, rng: rows,
            np.zeros((3, 2)),
            fixed,
            np.random.default_rng(0),
        )


@pytest.mark.parametrize("fixed", [(), (0, 1)])
def test_conditional_supports_empty_and_full_subsets(fixed):
    anchors = np.ones((3, 2))
    output = draw_conditional_samples(
        lambda rows, fixed, rng: rows, anchors, fixed, np.random.default_rng(0)
    )
    np.testing.assert_array_equal(output, anchors)
