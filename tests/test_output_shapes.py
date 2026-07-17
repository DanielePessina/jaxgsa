"""Tests for the strict output-shape contract."""

import jax.numpy as jnp
import pytest

import gsax
from gsax._core.validation import _validate_output
from gsax.problem import Problem

N = 40
UNLABELED = Problem.from_dict({"x0": (0.0, 1.0), "x1": (0.0, 1.0)})
TWO_OUTPUTS = Problem.from_dict(
    {"x0": (0.0, 1.0), "x1": (0.0, 1.0)},
    output_names=("a", "b"),
)


@pytest.mark.parametrize("shape", [(N,), (N, 2), (N, 5, 2)])
def test_canonical_shapes_pass(shape):
    assert _validate_output(jnp.ones(shape), N).shape == shape


@pytest.mark.parametrize("shape", [(2, N), (3, N, 2)])
def test_nonleading_sample_axis_is_rejected(shape):
    with pytest.raises(ValueError, match="sample rows"):
        _validate_output(jnp.ones(shape), N)


def test_rank_four_is_rejected():
    with pytest.raises(ValueError, match="Y must be 1-D"):
        _validate_output(jnp.ones((N, 2, 2, 2)), N)


@pytest.mark.parametrize("shape", [(N,), (N, 1), (N, 4, 1), (N, 3)])
def test_output_name_count_must_match_k(shape):
    with pytest.raises(ValueError, match="output_names length"):
        _validate_output(jnp.ones(shape), N, TWO_OUTPUTS)


def test_given_data_methods_do_not_transpose_outputs():
    X = jnp.asarray(gsax.sampling.monte_carlo(UNLABELED, 400, seed=2))
    Y = jnp.stack((X[:, 0], X[:, 1]), axis=1)

    with pytest.raises(ValueError, match="sample rows"):
        gsax.pce.analyze(UNLABELED, X, Y.T)


def test_two_dimensional_output_is_always_n_k():
    X = jnp.asarray(gsax.sampling.monte_carlo(TWO_OUTPUTS, 400, seed=3))
    Y = jnp.stack((X[:, 0], X[:, 1]), axis=1)
    result = gsax.pce.analyze(TWO_OUTPUTS, X, Y, order=1)

    assert result.S1.shape == (2, 2)
    assert result.predict(X[:5]).shape == (5, 2)
