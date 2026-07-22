"""Tests for the NPZ-only sampling persistence API (Sobol and Morris)."""

import json

import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa.morris import MorrisSamples
from jaxgsa.problem import GaussianInputSpec, Problem
from jaxgsa.sobol import SobolSamples


def _assert_equal(left: SobolSamples, right: SobolSamples) -> None:
    _assert_array_identical(left.samples, right.samples)
    _assert_array_identical(left.sample_ids, right.sample_ids)
    _assert_array_identical(left.expanded_to_unique, right.expanded_to_unique)
    assert left.n_expanded == right.n_expanded
    assert left.base_n == right.base_n
    assert left.n_params == right.n_params
    assert left.calc_second_order == right.calc_second_order
    assert left.problem == right.problem


def _assert_array_identical(left: np.ndarray, right: np.ndarray) -> None:
    np.testing.assert_array_equal(left, right)
    assert left.dtype == right.dtype


def _assert_morris_equal(left: MorrisSamples, right: MorrisSamples) -> None:
    _assert_array_identical(left.samples, right.samples)
    _assert_array_identical(left.expanded_to_unique, right.expanded_to_unique)
    _assert_array_identical(left.ee_idx_after, right.ee_idx_after)
    _assert_array_identical(left.ee_idx_before, right.ee_idx_before)
    _assert_array_identical(left.ee_delta, right.ee_delta)
    assert left.n_expanded == right.n_expanded
    assert left.n_trajectories == right.n_trajectories
    assert left.num_levels == right.num_levels
    assert left.method == right.method
    assert left.n_params == right.n_params
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
    samples = jaxgsa.sobol.sample(
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
    samples = jaxgsa.sobol.sample(
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
    samples = jaxgsa.sobol.sample(
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
    samples_a = jaxgsa.sobol.sample(problem, 32, seed=1, verbose=False)
    samples_b = jaxgsa.sobol.sample(problem, 64, seed=2, verbose=False)

    samples_a.save(tmp_path / "run.A")
    samples_b.save(tmp_path / "run.B")

    assert (tmp_path / "run.A.npz").exists()
    assert (tmp_path / "run.B.npz").exists()
    _assert_equal(samples_a, SobolSamples.load(tmp_path / "run.A"))
    _assert_equal(samples_b, SobolSamples.load(tmp_path / "run.B"))


def test_identity_mapping_skips_index_array(tmp_path):
    """Designs without duplicate rows omit expanded_to_unique from the NPZ."""
    problem = Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(4)})
    samples = jaxgsa.sobol.sample(problem, 64, seed=3, verbose=False)
    assert np.array_equal(samples.expanded_to_unique, np.arange(samples.n_expanded)), (
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
    samples = jaxgsa.sobol.sample(
        Problem.from_dict({"x": (0.0, 1.0)}),
        16,
        seed=5,
        verbose=False,
    )
    assert samples.n_expanded > samples.n_runs, (
        "test premise: 1-D Saltelli designs should contain duplicate rows"
    )

    path = tmp_path / "dupes"
    samples.save(path)

    with np.load(tmp_path / "dupes.npz", allow_pickle=False) as data:
        assert "expanded_to_unique" in data.files
        meta = json.loads(data["metadata"].item())
    assert meta["identity_mapping"] is False
    _assert_equal(samples, SobolSamples.load(path))


def test_metadata_records_jaxgsa_version(tmp_path):
    samples = jaxgsa.sobol.sample(
        Problem.from_dict({"x": (0.0, 1.0)}),
        16,
        seed=2,
        verbose=False,
    )
    samples.save(tmp_path / "versioned")

    with np.load(tmp_path / "versioned.npz", allow_pickle=False) as data:
        meta = json.loads(data["metadata"].item())
    assert isinstance(meta["jaxgsa_version"], str)
    assert meta["jaxgsa_version"] != ""


# ---------------------------------------------------------------------------
# Morris persistence (via the shared UniqueDesignSamples base)
# ---------------------------------------------------------------------------


def _morris_problem() -> Problem:
    return Problem.from_dict(
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


@pytest.mark.parametrize("method", ["trajectory", "radial"])
def test_morris_npz_round_trip(tmp_path, method):
    samples = jaxgsa.morris.sample(
        _morris_problem(),
        n_trajectories=8,
        method=method,
        seed=4,
        verbose=False,
    )

    path = tmp_path / "morris_design"
    samples.save(path)
    loaded = MorrisSamples.load(path)

    assert path.with_suffix(".npz").exists()
    _assert_morris_equal(samples, loaded)


def test_morris_downsampled_round_trip(tmp_path):
    samples = jaxgsa.morris.sample(
        _morris_problem(),
        n_trajectories=10,
        seed=7,
        verbose=False,
    ).downsample(4)

    path = tmp_path / "morris_small"
    samples.save(path)
    loaded = MorrisSamples.load(path)

    _assert_morris_equal(samples, loaded)
    assert loaded.n_trajectories == 4


def test_morris_expand_outputs_after_load(tmp_path):
    samples = jaxgsa.morris.sample(
        _morris_problem(),
        n_trajectories=6,
        seed=11,
        verbose=False,
    )
    Y = jnp.arange(samples.n_runs, dtype=jnp.float32)

    path = tmp_path / "morris_expand"
    samples.save(path)
    loaded = MorrisSamples.load(path)

    expanded = loaded.expand_outputs(Y)
    assert expanded.shape == (loaded.n_expanded,)
    np.testing.assert_array_equal(np.asarray(expanded), np.asarray(samples.expand_outputs(Y)))


def test_morris_identity_mapping_skips_index_array(tmp_path):
    """Radial designs have no duplicate rows, so the index map is omitted."""
    samples = jaxgsa.morris.sample(
        _morris_problem(),
        n_trajectories=8,
        method="radial",
        seed=3,
        verbose=False,
    )
    assert np.array_equal(samples.expanded_to_unique, np.arange(samples.n_expanded)), (
        "test premise: radial designs should have no duplicate rows"
    )

    path = tmp_path / "morris_identity"
    samples.save(path)

    with np.load(tmp_path / "morris_identity.npz", allow_pickle=False) as data:
        assert "expanded_to_unique" not in data.files
        meta = json.loads(data["metadata"].item())
    assert meta["identity_mapping"] is True
    _assert_morris_equal(samples, MorrisSamples.load(path))


def test_sobol_and_morris_share_metadata_schema(tmp_path):
    """Both NPZ formats carry the same base-owned metadata keys."""
    problem = _morris_problem()
    sobol_samples = jaxgsa.sobol.sample(problem, 32, seed=1, verbose=False)
    morris_samples = jaxgsa.morris.sample(problem, n_trajectories=6, seed=1, verbose=False)

    sobol_samples.save(tmp_path / "schema_sobol")
    morris_samples.save(tmp_path / "schema_morris")

    metas = {}
    for name in ("schema_sobol", "schema_morris"):
        with np.load(tmp_path / f"{name}.npz", allow_pickle=False) as data:
            assert "samples" in data.files
            assert "metadata" in data.files
            metas[name] = json.loads(data["metadata"].item())

    common_keys = {"jaxgsa_version", "problem", "n_expanded", "identity_mapping"}
    for meta in metas.values():
        assert common_keys <= set(meta)
        assert set(meta["problem"]) == {"names", "input_specs", "output_names"}

    # The shared (base-owned) part of the schema is identical across designs.
    assert metas["schema_sobol"]["problem"] == metas["schema_morris"]["problem"]
    assert metas["schema_sobol"]["jaxgsa_version"] == metas["schema_morris"]["jaxgsa_version"]
