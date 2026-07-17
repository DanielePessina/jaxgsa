"""Tests for the NPZ-only Sobol sampling persistence API."""

import numpy as np

import gsax
from gsax.problem import GaussianInputSpec, Problem
from gsax.sobol import SobolSamples


def _assert_equal(left: SobolSamples, right: SobolSamples) -> None:
    np.testing.assert_array_equal(left.samples, right.samples)
    np.testing.assert_array_equal(left.sample_ids, right.sample_ids)
    np.testing.assert_array_equal(left.expanded_to_unique, right.expanded_to_unique)
    assert left.expanded_n_total == right.expanded_n_total
    assert left.base_n == right.base_n
    assert left.n_params == right.n_params
    assert left.calc_second_order == right.calc_second_order
    assert left.problem == right.problem


def test_npz_round_trip(tmp_path):
    problem = Problem.from_dict(
        {
            "uniform": (0.0, 1.0),
            "gaussian": GaussianInputSpec(
                dist="gaussian",
                mean=1.0,
                variance=4.0,
                low=0.0,
                high=3.0,
            ),
        },
        output_names=("response",),
    )
    samples = gsax.sobol.sample(
        problem,
        64,
        calc_second_order=False,
        seed=4,
        verbose=False,
    )

    path = tmp_path / "design"
    samples.save(path)
    loaded = SobolSamples.load(path)

    assert path.with_suffix(".npz").exists()
    _assert_equal(samples, loaded)


def test_explicit_npz_suffix_round_trip(tmp_path):
    samples = gsax.sobol.sample(
        Problem.from_dict({"x": (0.0, 1.0)}),
        16,
        seed=2,
        verbose=False,
    )
    path = tmp_path / "design.npz"

    samples.save(path)

    _assert_equal(samples, SobolSamples.load(path))
