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
    assert left.n_blocks_dropped == right.n_blocks_dropped
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
    """Tier T4 (round trip): a downsampled Morris design survives save/load.

    ``downsample`` rebuilds the design from kept trajectories only, so it
    resets ``n_blocks_dropped`` to 0. The npz round trip must carry that
    value through unchanged.
    """
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
    assert samples.n_blocks_dropped == 0
    assert loaded.n_blocks_dropped == 0


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


# ---------------------------------------------------------------------------
# Problem.correlation persistence (JSON metadata + NPZ design files)
# ---------------------------------------------------------------------------


def test_problem_meta_json_round_trip_carries_correlation():
    from jaxgsa._core.samples import _problem_from_meta, _problem_to_meta

    problem = Problem.from_dict(
        {
            "uniform": (0.0, 1.0),
            "gaussian": GaussianInputSpec(dist="gaussian", mean=1.0, variance=4.0),
        },
        correlation=[[1.0, 0.6], [0.6, 1.0]],
    )

    meta = json.loads(json.dumps(_problem_to_meta(problem)))
    restored = _problem_from_meta(meta)

    assert restored == problem
    assert hash(restored) == hash(problem)
    stored = restored.correlation
    assert stored is not None
    np.testing.assert_array_equal(stored, problem.correlation)


def test_problem_meta_load_tolerates_pre_060_files_without_correlation_key():
    from jaxgsa._core.samples import _problem_from_meta, _problem_to_meta

    problem = Problem.from_dict({"x": (0.0, 1.0), "y": (0.0, 1.0)})
    meta = _problem_to_meta(problem)
    del meta["correlation"]  # simulate a design saved by jaxgsa < 0.6
    restored = _problem_from_meta(meta)
    assert restored == problem
    assert restored.correlation is None


def test_npz_round_trip_carries_identity_correlation(tmp_path):
    """End-to-end NPZ persistence with a declared (identity) correlation.

    An identity matrix is 'declared but independent', so the design samplers
    still accept the problem; a non-identity correlation cannot reach a
    Saltelli design at all (the sampler raises), which is why the non-trivial
    matrix is covered through the metadata round-trip tests above.
    """
    problem = Problem.from_dict(
        {"x": (0.0, 1.0), "y": (0.0, 1.0)},
        correlation=np.eye(2),
    )
    samples = jaxgsa.sobol.sample(problem, 32, seed=6, verbose=False)

    path = tmp_path / "correlated_design"
    samples.save(path)
    loaded = SobolSamples.load(path)

    _assert_equal(samples, loaded)
    stored = loaded.problem.correlation
    assert stored is not None
    np.testing.assert_array_equal(stored, np.eye(2))

    with np.load(tmp_path / "correlated_design.npz", allow_pickle=False) as data:
        meta = json.loads(data["metadata"].item())
    assert meta["problem"]["correlation"] == [[1.0, 0.0], [0.0, 1.0]]


def _every_design(problem: Problem):
    """One design object of each class that can be saved, by class name."""
    return {
        "SobolSamples": jaxgsa.sobol.sample(problem, 16, seed=1, verbose=False),
        "MorrisSamples": jaxgsa.morris.sample(problem, 4, seed=1, verbose=False),
        "EFASTSamples": jaxgsa.efast.sample(problem, 65, seed=1, verbose=False),
        "KucherenkoSamples": jaxgsa.kucherenko.sample(problem, 16, seed=1, verbose=False),
    }


def test_save_to_a_missing_directory_names_the_directory(tmp_path):
    """NumPy's own error names a temporary file, not the caller's mistake.

    ``np.savez_compressed`` under a directory that is not there fails deep
    inside NumPy. The check moves the failure to the front, names the missing
    directory, and says what to do about it. It does not make the directory:
    creating one the caller never asked for is a side effect a save should
    not have.
    """
    problem = Problem.from_dict({"x": (0.0, 1.0), "y": (0.0, 1.0)})
    samples = jaxgsa.sobol.sample(problem, 16, seed=1, verbose=False)
    missing = tmp_path / "not_there"

    with pytest.raises(FileNotFoundError) as excinfo:
        samples.save(missing / "design")

    message = str(excinfo.value)
    assert str(missing) in message
    assert "does not exist" in message
    assert "mkdir" in message
    assert not missing.exists()  # the save made nothing


def test_every_design_class_reports_a_missing_directory_the_same_way(tmp_path):
    problem = Problem.from_dict({"x": (0.0, 1.0), "y": (0.0, 1.0)})
    missing = tmp_path / "not_there"

    for name, samples in _every_design(problem).items():
        with pytest.raises(FileNotFoundError) as excinfo:
            samples.save(missing / "design")
        assert "does not exist" in str(excinfo.value), name
    assert not missing.exists()


def test_a_bare_filename_saves_into_the_working_directory(tmp_path, monkeypatch):
    """A path with no directory part must still work: its parent is '.'."""
    monkeypatch.chdir(tmp_path)
    samples = jaxgsa.sobol.sample(Problem.from_dict({"x": (0.0, 1.0)}), 16, seed=2, verbose=False)

    samples.save("design")

    assert (tmp_path / "design.npz").exists()
    _assert_equal(samples, SobolSamples.load("design"))
