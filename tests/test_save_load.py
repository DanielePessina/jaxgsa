"""Tests for the NPZ-only Sobol sampling persistence API."""

import json

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


def test_dotted_stem_appends_npz_suffix(tmp_path):
    """A dotted stem must gain '.npz' by appending, not by suffix replacement."""
    samples = gsax.sobol.sample(
        Problem.from_dict({"x": (0.0, 1.0)}),
        16,
        seed=2,
        verbose=False,
    )
    path = tmp_path / "design.2026-07"

    samples.save(path)

    assert (tmp_path / "design.2026-07.npz").exists()
    assert not (tmp_path / "design.npz").exists()
    _assert_equal(samples, SobolSamples.load(path))


def test_dotted_stems_do_not_collide(tmp_path):
    """Saving 'run.A' and 'run.B' must produce two distinct files."""
    problem = Problem.from_dict({"x": (0.0, 1.0), "y": (0.0, 1.0)})
    samples_a = gsax.sobol.sample(problem, 32, seed=1, verbose=False)
    samples_b = gsax.sobol.sample(problem, 64, seed=2, verbose=False)

    samples_a.save(tmp_path / "run.A")
    samples_b.save(tmp_path / "run.B")

    assert (tmp_path / "run.A.npz").exists()
    assert (tmp_path / "run.B.npz").exists()
    _assert_equal(samples_a, SobolSamples.load(tmp_path / "run.A"))
    _assert_equal(samples_b, SobolSamples.load(tmp_path / "run.B"))


def test_identity_mapping_skips_index_array(tmp_path):
    """Designs without duplicate rows omit expanded_to_unique from the NPZ."""
    problem = Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(4)})
    samples = gsax.sobol.sample(problem, 64, seed=3, verbose=False)
    assert np.array_equal(samples.expanded_to_unique, np.arange(samples.expanded_n_total)), (
        "test premise: this design should have no duplicate rows"
    )

    path = tmp_path / "identity"
    samples.save(path)

    with np.load(tmp_path / "identity.npz", allow_pickle=False) as data:
        assert "expanded_to_unique" not in data.files
        meta = json.loads(data["metadata"].item())
    assert meta["identity_mapping"] is True
    _assert_equal(samples, SobolSamples.load(path))


def test_duplicate_rows_store_index_array(tmp_path):
    """Designs with duplicate rows still persist the full expansion map."""
    samples = gsax.sobol.sample(
        Problem.from_dict({"x": (0.0, 1.0)}),
        16,
        seed=5,
        verbose=False,
    )
    assert samples.expanded_n_total > samples.n_total, (
        "test premise: 1-D Saltelli designs should contain duplicate rows"
    )

    path = tmp_path / "dupes"
    samples.save(path)

    with np.load(tmp_path / "dupes.npz", allow_pickle=False) as data:
        assert "expanded_to_unique" in data.files
        meta = json.loads(data["metadata"].item())
    assert meta["identity_mapping"] is False
    _assert_equal(samples, SobolSamples.load(path))


def test_metadata_records_gsax_version(tmp_path):
    samples = gsax.sobol.sample(
        Problem.from_dict({"x": (0.0, 1.0)}),
        16,
        seed=2,
        verbose=False,
    )
    samples.save(tmp_path / "versioned")

    with np.load(tmp_path / "versioned.npz", allow_pickle=False) as data:
        meta = json.loads(data["metadata"].item())
    assert isinstance(meta["gsax_version"], str)
    assert meta["gsax_version"] != ""
