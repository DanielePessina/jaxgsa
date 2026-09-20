"""Representative result-schema integration checks.

The schema is generated from declarations, so one scalar fixture per result
class plus a few layout variants is enough. Numerical estimator behavior is
covered by the method integration tests and the analytical fixtures.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import _result_fixtures as fixtures
import jax.numpy as jnp
import pytest

from jaxgsa._core.registry import methods
from jaxgsa._core.result import CIInfo

SNAPSHOT = Path(__file__).parent / "data" / "result_dataset_schema.json"
MAX_REPR_CHARS = 400


@lru_cache(maxsize=None)
def _result(name: str, shape: str) -> Any:
    """Build one representative result once per test session."""
    return fixtures.build(name, shape)


def _snapshot() -> dict[str, dict[str, list[str]]]:
    return json.loads(SNAPSHOT.read_text())


REPRESENTATIVE_CASES = [
    (name, "series" if name == "optimal_transport_trajectory" else "scalar")
    for name in fixtures.BUILDERS
]
REPRESENTATIVE_CASES += [
    ("sobol", "series"),
]


def test_every_registered_result_has_a_compact_repr_and_snapshot_entry() -> None:
    """Every public result prints compactly and appears in the schema snapshot."""
    snapshot_types = {entry["type"] for entry in _snapshot().values()}
    for method, spec in methods().items():
        result = _result(method, "scalar")
        text = repr(result)
        assert text.startswith(f"{type(result).__name__}(")
        assert "\n" not in text
        assert len(text) <= MAX_REPR_CHARS
        assert "Problem(" not in text
        assert spec.result.__name__ in snapshot_types


@pytest.mark.parametrize(("name", "shape"), REPRESENTATIVE_CASES)
def test_representative_dataset_schema_is_stable(name: str, shape: str) -> None:
    """The public xarray layout remains stable for one scalar and time case."""
    expected = _snapshot()[f"{name}@{shape}"]
    actual = fixtures.dataset_schema(_result(name, shape))
    assert actual == {key: value for key, value in expected.items() if key != "type"}


def test_declared_provenance_is_exported_as_plain_attributes() -> None:
    """Every declared provenance field reaches xarray as a serializable scalar."""
    for method in methods():
        result = _result(method, "scalar")
        attrs = result.to_dataset().attrs
        for name in type(result)._schema.meta:
            value = getattr(result, name)
            if value is None:
                assert name not in attrs
            else:
                assert attrs[name] == value
                assert isinstance(attrs[name], (str, bool, int, float, complex))


def test_ciinfo_repr_hides_bootstrap_arrays() -> None:
    """Printing interval metadata never dumps the stored replicate arrays."""
    ci = CIInfo(level=0.9, method="quantile", n_bootstrap=3, replicates={"S1": jnp.zeros((3, 8))})
    assert "0." not in repr(ci).split("replicates=")[1]


def test_second_order_interval_axes_are_labelled_without_a_point_estimate() -> None:
    """An interval-only S2 field still exports parameter-pair coordinates."""
    full = _result("sobol", "scalar")
    partial = type(full)(
        S1=full.S1,
        ST=full.ST,
        S2=None,
        problem=full.problem,
        invalid=full.invalid,
        S2_conf=full.S2_conf,
    )
    ds = partial.to_dataset()
    assert "S2" not in ds.data_vars
    assert "S2_lower" in ds.data_vars
    assert list(ds.coords["param_i"].values) == list(full.problem.names)
