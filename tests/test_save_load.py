"""Tests for SamplingResult.save() and gsax.load() round-trip."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import gsax
from gsax.problem import GaussianInputSpec, Problem
from gsax.sampling import SamplingResult, load

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def low_d_problem():
    """1-D problem — Saltelli design will have many duplicate rows."""
    return Problem.from_dict({"x1": (0.0, 1.0)})


@pytest.fixture
def high_d_problem():
    """6-D problem — unlikely to produce duplicate rows."""
    return Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(1, 7)})


@pytest.fixture
def sr_with_duplicates(low_d_problem):
    """SamplingResult where expanded_to_unique != identity."""
    return gsax.sample(low_d_problem, 16, seed=42, verbose=False)


@pytest.fixture
def sr_identity(high_d_problem):
    """SamplingResult where expanded_to_unique IS an identity mapping."""
    sr = gsax.sample(high_d_problem, 16, calc_second_order=False, seed=42, verbose=False)
    # Verify this fixture actually has an identity mapping
    assert np.array_equal(
        sr.expanded_to_unique,
        np.arange(sr.expanded_n_total, dtype=sr.expanded_to_unique.dtype),
    )
    return sr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_sr_equal(a: SamplingResult, b: SamplingResult) -> None:
    """Assert two SamplingResults are semantically equal."""
    np.testing.assert_array_almost_equal(a.samples, b.samples)
    np.testing.assert_array_equal(a.sample_ids, b.sample_ids)
    assert a.expanded_n_total == b.expanded_n_total
    np.testing.assert_array_equal(a.expanded_to_unique, b.expanded_to_unique)
    assert a.base_n == b.base_n
    assert a.n_params == b.n_params
    assert a.calc_second_order == b.calc_second_order
    assert a.problem.names == b.problem.names
    assert a.problem.bounds == b.problem.bounds
    assert a.problem.output_names == b.problem.output_names


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["csv", "txt", "pkl"])
def test_round_trip(sr_with_duplicates, tmp_path, fmt):
    stem = tmp_path / "experiment"
    sr_with_duplicates.save(stem, format=fmt)
    loaded = load(stem, format=fmt)
    _assert_sr_equal(sr_with_duplicates, loaded)


def test_round_trip_identity(sr_identity, tmp_path):
    stem = tmp_path / "experiment"
    sr_identity.save(stem, format="csv")
    loaded = load(stem, format="csv")
    _assert_sr_equal(sr_identity, loaded)


# ---------------------------------------------------------------------------
# Identity mapping optimization
# ---------------------------------------------------------------------------


def test_identity_skips_npz(sr_identity, tmp_path):
    stem = tmp_path / "experiment"
    sr_identity.save(stem, format="csv")
    assert (stem.with_suffix(".csv")).exists()
    assert (stem.with_suffix(".json")).exists()
    assert not (stem.with_suffix(".npz")).exists()


def test_non_identity_writes_npz(sr_with_duplicates, tmp_path):
    stem = tmp_path / "experiment"
    sr_with_duplicates.save(stem, format="csv")
    assert (stem.with_suffix(".npz")).exists()


# ---------------------------------------------------------------------------
# output_names round-trip
# ---------------------------------------------------------------------------


def test_output_names_preserved(tmp_path):
    prob = Problem(
        names=("a", "b"),
        bounds=((0.0, 1.0), (0.0, 1.0)),
        output_names=("y1", "y2"),
    )
    sr = gsax.sample(prob, 16, seed=0, verbose=False)
    stem = tmp_path / "with_outputs"
    sr.save(stem, format="csv")
    loaded = load(stem, format="csv")
    assert loaded.problem.output_names == ("y1", "y2")


def test_mixed_input_specs_round_trip(tmp_path):
    prob = Problem.from_dict(
        {
            "uniform": (0.0, 1.0),
            "gaussian": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
            "truncated": GaussianInputSpec(
                dist="gaussian",
                mean=1.0,
                variance=4.0,
                low=0.0,
                high=3.0,
            ),
        }
    )
    sr = gsax.sample(prob, 32, calc_second_order=False, seed=0, verbose=False)
    stem = tmp_path / "mixed"
    sr.save(stem, format="csv")
    loaded = load(stem, format="csv")
    _assert_sr_equal(sr, loaded)
    assert loaded.problem.bounds is None
    assert loaded.problem.has_non_uniform_inputs is True


def test_load_remains_backward_compatible_with_legacy_bounds_metadata(tmp_path):
    prob = Problem.from_dict({"x1": (0.0, 1.0), "x2": (1.0, 2.0)})
    sr = gsax.sample(prob, 16, calc_second_order=False, seed=0, verbose=False)
    stem = tmp_path / "legacy"
    sr.save(stem, format="csv")

    import json

    json_path = stem.with_suffix(".json")
    meta = json.loads(json_path.read_text())
    del meta["problem"]["input_specs"]
    json_path.write_text(json.dumps(meta))

    loaded = load(stem, format="csv")
    _assert_sr_equal(sr, loaded)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_metadata(tmp_path):
    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        load(tmp_path / "nonexistent", format="csv")


def test_unsupported_format(sr_with_duplicates, tmp_path):
    with pytest.raises(ValueError, match="Unsupported format"):
        sr_with_duplicates.save(tmp_path / "bad", format="hdf5")


def test_xlsx_import_error(sr_with_duplicates, tmp_path):
    stem = tmp_path / "experiment"
    with patch("pandas.DataFrame.to_excel", side_effect=ImportError):
        with pytest.raises(ImportError, match="openpyxl"):
            sr_with_duplicates.save(stem, format="xlsx")


def test_parquet_import_error(sr_with_duplicates, tmp_path):
    stem = tmp_path / "experiment"
    with patch("pandas.DataFrame.to_parquet", side_effect=ImportError):
        with pytest.raises(ImportError, match="pyarrow"):
            sr_with_duplicates.save(stem, format="parquet")


def test_xlsx_read_import_error(sr_with_duplicates, tmp_path):
    stem = tmp_path / "experiment"
    # Save as csv first, then try to load as xlsx with mocked import error
    sr_with_duplicates.save(stem, format="csv")
    # Create a fake json that says xlsx format
    import json

    json_path = stem.with_suffix(".json")
    meta = json.loads(json_path.read_text())
    meta["sample_format"] = "xlsx"
    json_path.write_text(json.dumps(meta))
    # Create a dummy xlsx file
    stem.with_suffix(".xlsx").write_bytes(b"fake")
    with patch("pandas.read_excel", side_effect=ImportError):
        with pytest.raises(ImportError, match="openpyxl"):
            load(stem, format="xlsx")


def test_parquet_read_import_error(sr_with_duplicates, tmp_path):
    stem = tmp_path / "experiment"
    sr_with_duplicates.save(stem, format="csv")
    import json

    json_path = stem.with_suffix(".json")
    meta = json.loads(json_path.read_text())
    meta["sample_format"] = "parquet"
    json_path.write_text(json.dumps(meta))
    stem.with_suffix(".parquet").write_bytes(b"fake")
    with patch("pandas.read_parquet", side_effect=ImportError):
        with pytest.raises(ImportError, match="pyarrow"):
            load(stem, format="parquet")
