"""Tier T4 (behavioural contract): the declared result schema.

No external oracle applies. Nothing here is a number; it is what a result
promises about how it prints and what it exports. Three things are checked.

``__repr__`` must summarize a result, not dump it. Five result classes used to
have no ``__repr__`` at all, so printing one in a notebook printed every index
array and the whole ``Problem``. The classes are read out of
:func:`jaxgsa._core.registry.methods`, so a fourteenth method is covered the
day it registers itself.

``to_dataset`` must export exactly what it exported before the schema was
declared. The snapshot in ``tests/data/result_dataset_schema.json`` was taken
from the hand-written ``to_dataset`` methods. Deriving them from a declaration
was a refactor, so nothing in that file may move. Numbers are not compared
here: ``scripts/baseline_check.py`` guards those.

``CIInfo`` must record the level and the endpoint rule that actually ran, and
must keep the bootstrap draws only when asked.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import _result_fixtures as fixtures
import jax.numpy as jnp
import pytest

import jaxgsa
from jaxgsa._core.registry import methods
from jaxgsa._core.result import CIInfo, ResultSchema, SchemaResult

SNAPSHOT = Path(__file__).parent / "data" / "result_dataset_schema.json"

MAX_REPR_CHARS = 400
"""A summary that runs past this is dumping data, not summarizing it."""


@lru_cache(maxsize=None)
def _result(name: str, shape: str) -> Any:
    """Build one fixture result, once per session."""
    return fixtures.build(name, shape)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(methods()))
def test_every_registered_result_declares_a_schema(method: str) -> None:
    """Every method's result class derives its repr and export from a schema."""
    cls = methods()[method].result
    assert issubclass(cls, SchemaResult), f"{cls.__name__} does not mix in SchemaResult"
    assert isinstance(cls._schema, ResultSchema), f"{cls.__name__} declares no _schema"
    assert cls._schema.primary in {spec.name for spec in cls._schema.fields}, (
        f"{cls.__name__}._schema.primary names a field the schema does not declare"
    )


@pytest.mark.parametrize("method", sorted(methods()))
def test_every_registered_result_has_a_summary_repr(method: str) -> None:
    """repr() names the class and its field shapes, and nothing bulky."""
    result = _result(method, "scalar")
    text = repr(result)

    assert text.startswith(f"{type(result).__name__}("), text
    assert text.endswith(")"), text
    assert "\n" not in text, "repr must stay on one line"
    assert len(text) <= MAX_REPR_CHARS, f"repr is {len(text)} chars: {text}"

    # The two containers whose own repr is unbounded. A result that embedded
    # either would print the parameter definitions, or the full non-finite
    # report, every time someone echoed it.
    assert "Problem(" not in text
    assert "InvalidReport(" not in text


@pytest.mark.parametrize("method", sorted(methods()))
def test_repr_does_not_grow_with_the_output_size(method: str) -> None:
    """A time-series result summarizes to the same length as a scalar one.

    This is the property that matters. A length bound alone would pass for a
    class that prints small arrays in full; a repr whose size is independent
    of ``T * K * D`` cannot be printing array contents.
    """
    scalar = repr(_result(method, "scalar"))
    series = repr(_result(method, "series"))
    assert len(series) <= len(scalar) + 40, f"scalar: {scalar}\nseries: {series}"


def test_ciinfo_repr_hides_the_stored_draws() -> None:
    """The interval summary names its kept draws without printing them."""
    ci = CIInfo(level=0.9, method="quantile", n_resamples=3, replicates={"S1": jnp.zeros((3, 8))})
    text = repr(ci)
    assert "level=0.9" in text
    assert "method='quantile'" in text
    assert "replicates={S1}" in text
    assert "0." not in text.split("replicates=")[1]


# ---------------------------------------------------------------------------
# to_dataset
# ---------------------------------------------------------------------------


def _snapshot() -> dict[str, dict[str, list[str]]]:
    """Load the stored export schema."""
    return json.loads(SNAPSHOT.read_text())


def test_snapshot_covers_every_result_class() -> None:
    """No registered method's result is missing from the snapshot."""
    covered = {entry["type"] for entry in _snapshot().values()}
    declared = {spec.result.__name__ for spec in methods().values()}
    assert declared <= covered, f"not in the snapshot: {sorted(declared - covered)}"


@pytest.mark.parametrize(("name", "shape"), fixtures.CASES)
def test_dataset_schema_is_unchanged(name: str, shape: str) -> None:
    """to_dataset exports the same dims, coords and variables as before."""
    expected = _snapshot()[f"{name}@{shape}"]
    actual = fixtures.dataset_schema(_result(name, shape))
    assert actual == {k: v for k, v in expected.items() if k != "type"}


def test_hdmr_term_axis_is_not_labelled_param() -> None:
    """HDMR's primary axis is the expansion term, and the export says so."""
    ds = _result("hdmr", "scalar").to_dataset()
    assert ds["Sa"].dims == ("term",)
    assert ds["ST"].dims == ("param",)
    assert ds["select"].dims == ("term",)
    assert len(ds.coords["term"]) > len(ds.coords["param"])


def test_ot_multivariate_shape_comes_from_the_mode() -> None:
    """A multivariate OT result is one index per parameter, whatever Y was."""
    ds = _result("optimal_transport_multivariate", "scalar").to_dataset()
    assert ds["ot"].dims == ("param",)
    assert ds.attrs["mode"] == "multivariate"


def test_ot_trajectory_shape_comes_from_the_mode() -> None:
    """A trajectory OT result is one index per parameter per output."""
    ds = _result("optimal_transport_trajectory", "series").to_dataset()
    assert ds["ot"].dims == ("output", "param")
    assert ds.attrs["mode"] == "trajectory"


def test_second_order_axes_are_labelled_even_without_the_estimate() -> None:
    """A lone ``S2_lower`` still gets its parameter coordinates.

    The coordinates belong to the axes, not to the point estimate. Naming
    them only next to ``S2`` would make ``.sel(param_i=...)`` raise on a
    result that carries the interval but not the estimate.
    """
    full = _result("sobol", "scalar")
    partial = jaxgsa.sobol.SobolResult(
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


# ---------------------------------------------------------------------------
# CIInfo
# ---------------------------------------------------------------------------

_PROBLEM = jaxgsa.Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)})


def _model(X: Any) -> Any:
    """The same model the fixtures use, evaluated on a Morris design."""
    return X[:, 0] + 2.0 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]


@lru_cache(maxsize=None)
def _xy() -> tuple[Any, Any]:
    """A small given-data sample for the bootstrapping methods."""
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(_PROBLEM, 192, seed=3))
    return X, X[:, 0] + 2.0 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]


def _bootstrapped(method: str, *, level: float, keep: bool, n: int = 6) -> Any:
    """Run one bootstrapping method with an explicit level.

    Args:
        method: Namespace name of the method.
        level: The ``conf_level`` to run at, deliberately not the default.
        keep: Whether to ask for the bootstrap draws.
        n: Number of resamples.

    Returns:
        The result the method produced.
    """
    X, Y = _xy()
    if method == "morris":
        import jax

        sr = jaxgsa.morris.sample(_PROBLEM, n_trajectories=8, seed=3, verbose=False)
        Ym = _model(jnp.asarray(sr.samples))
        return jaxgsa.morris.analyze(
            sr,
            Ym,
            num_resamples=n,
            conf_level=level,
            keep_replicates=keep,
            key=jax.random.key(3),
        )
    if method == "pawn":
        return jaxgsa.pawn.analyze(
            _PROBLEM, X, Y, n_bins=4, n_bootstrap=n, conf_level=level, keep_replicates=keep
        )
    if method == "borgonovo":
        return jaxgsa.borgonovo.analyze(
            _PROBLEM, X, Y, n_bootstrap=n, conf_level=level, keep_replicates=keep
        )
    return jaxgsa.optimal_transport.analyze(
        _PROBLEM, X, Y, n_partitions=4, n_bootstrap=n, conf_level=level, keep_replicates=keep
    )


WIRED = ["borgonovo", "morris", "optimal_transport", "pawn"]
"""The bootstrapping methods whose ``analyze`` records a ``CIInfo``."""


@pytest.mark.parametrize("method", WIRED)
def test_ci_records_the_level_that_ran(method: str) -> None:
    """The recorded level is the one the caller asked for, not the default."""
    result = _bootstrapped(method, level=0.8, keep=False)
    assert result.ci is not None
    assert result.ci.level == 0.8
    assert result.ci.n_resamples == 6


@pytest.mark.parametrize("method", WIRED)
def test_ci_records_the_endpoint_rule_that_ran(method: str) -> None:
    """The recorded rule is one this method can actually run."""
    result = _bootstrapped(method, level=0.95, keep=False)
    assert result.ci is not None
    assert result.ci.method in {"quantile", "gaussian"}


def test_morris_records_the_gaussian_rule_when_asked() -> None:
    """Morris offers two rules, and records whichever one ran."""
    import jax

    sr = jaxgsa.morris.sample(_PROBLEM, n_trajectories=8, seed=3, verbose=False)
    Y = _model(jnp.asarray(sr.samples))
    result = jaxgsa.morris.analyze(
        sr, Y, num_resamples=6, ci_method="gaussian", key=jax.random.key(3)
    )
    assert result.ci is not None
    assert result.ci.method == "gaussian"


@pytest.mark.parametrize("method", WIRED)
def test_no_ci_without_a_bootstrap(method: str) -> None:
    """A method that ran no bootstrap reports no interval metadata."""
    X, Y = _xy()
    if method == "borgonovo":
        result = jaxgsa.borgonovo.analyze(_PROBLEM, X, Y, n_bootstrap=0)
    elif method == "pawn":
        result = jaxgsa.pawn.analyze(_PROBLEM, X, Y, n_bins=4)
    elif method == "optimal_transport":
        result = jaxgsa.optimal_transport.analyze(_PROBLEM, X, Y, n_partitions=4)
    else:
        sr = jaxgsa.morris.sample(_PROBLEM, n_trajectories=8, seed=3, verbose=False)
        result = jaxgsa.morris.analyze(sr, _model(jnp.asarray(sr.samples)))
    assert result.ci is None


@pytest.mark.parametrize("method", WIRED)
def test_replicates_are_absent_by_default(method: str) -> None:
    """The bootstrap draws are dropped unless the caller keeps them."""
    result = _bootstrapped(method, level=0.95, keep=False)
    assert result.ci is not None
    assert result.ci.replicates is None


@pytest.mark.parametrize("method", WIRED)
def test_replicates_are_present_when_asked(method: str) -> None:
    """Kept draws lead with the resample axis and match the estimate's shape."""
    result = _bootstrapped(method, level=0.95, keep=True)
    assert result.ci is not None
    assert result.ci.replicates
    for name, draws in result.ci.replicates.items():
        estimate = getattr(result, name)
        assert draws.shape == (result.ci.n_resamples, *estimate.shape), name


def test_morris_physical_units_rescales_the_kept_draws() -> None:
    """Kept draws follow the measures into physical units."""
    unit = _bootstrapped("morris", level=0.95, keep=True)
    physical = unit.to_physical_units()
    assert physical.ci is not None
    assert unit.ci is not None
    bounds = _PROBLEM.bounds
    assert bounds is not None
    ranges = jnp.asarray([high - low for low, high in bounds])
    for name, draws in physical.ci.replicates.items():
        assert jnp.allclose(draws, unit.ci.replicates[name] / ranges)


@pytest.mark.xfail(
    strict=False,
    reason=(
        "sobol.analyze is held by a concurrent branch, so its CIInfo is not "
        "wired yet. SobolResult carries the field; only the constructor call "
        "in sobol/_analyze.py is missing."
    ),
)
def test_sobol_records_its_ci() -> None:
    """Sobol should record its interval metadata like the other four."""
    result = _result("sobol", "scalar")
    assert result.ci is not None
    assert result.ci.method == "quantile"
