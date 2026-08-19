"""Tests for the strict output-shape contract.

Tier T4 (behavioural contract). No external oracle exists for our own shape
rules; these tests pin which ``Y`` layouts are accepted and how they are read.
"""

import jax.numpy as jnp
import pytest

import jaxgsa
from jaxgsa._core.validation import _validate_output
from jaxgsa.problem import Problem

N = 40
UNLABELED = Problem.from_dict({"x0": (0.0, 1.0), "x1": (0.0, 1.0)})
TWO_OUTPUTS = Problem.from_dict(
    {"x0": (0.0, 1.0), "x1": (0.0, 1.0)},
    output_names=("a", "b"),
)


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
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(UNLABELED, 400, seed=2))
    Y = jnp.stack((X[:, 0], X[:, 1]), axis=1)

    with pytest.raises(ValueError, match="sample rows"):
        jaxgsa.pce.analyze(UNLABELED, X, Y.T)
