"""Tests for ``result.to_pandas()``.

Tier T4 (behavioural contract). The table is a reshape of ``to_dataset()``,
so every check compares the two.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from _result_fixtures import CASES, PROBLEM, build

import jaxgsa
from jaxgsa import kucherenko, sobol
from jaxgsa._core.irregular import IrregularResult

FRAME_DIMS = {"time", "output", "param"}


def _expected_columns(ds: xr.Dataset) -> set[str]:
    """Return the dataset variables that carry one value per input."""
    return {
        str(name)
        for name, var in ds.data_vars.items()
        if "param" in var.dims and set(var.dims) <= FRAME_DIMS
    }


def _check_frame(df: pd.DataFrame, ds: xr.Dataset) -> None:
    """Check that ``df`` holds exactly the per-parameter variables of ``ds``."""
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == _expected_columns(ds)
    dims = list(ds[df.columns[0]].dims)
    assert list(df.index.names) == dims
    assert df.index.names[-1] == "param"
    assert list(df.index.get_level_values("param").unique()) == list(PROBLEM.names)
    assert len(df) == int(np.prod([ds.sizes[d] for d in dims]))
    for column in df.columns:
        expected = ds[column].transpose(*dims).values.ravel()
        np.testing.assert_allclose(df[column].to_numpy(), expected, rtol=1e-6, atol=0)
    assert df.attrs == dict(ds.attrs)


@pytest.mark.parametrize(("name", "shape"), CASES, ids=[f"{n}@{s}" for n, s in CASES])
def test_to_pandas_matches_to_dataset(name: str, shape: str) -> None:
    """Every result class and output shape gives a tidy copy of its dataset."""
    result = build(name, shape)
    df = result.to_pandas()
    ds = result.to_dataset()
    _check_frame(df, ds)
    if shape == "multi":
        assert "output" in df.index.names
    if shape == "series" and name != "optimal_transport_trajectory":
        assert df.index.names[:2] == ["time", "output"]


def _sobol_result(n_bootstrap: int) -> Any:
    design = sobol.sample(PROBLEM, n_samples=128, seed=0, verbose=False)
    X = jnp.asarray(design.samples)
    Y = X[:, 0] + 2.0 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]
    return sobol.analyze(design, Y, n_bootstrap=n_bootstrap, key=jax.random.key(0))


def test_bounds_follow_their_index_when_bootstrapped() -> None:
    """Each index is followed by its lower and upper bound columns."""
    df = _sobol_result(n_bootstrap=4).to_pandas()
    assert list(df.columns) == ["S1", "S1_lower", "S1_upper", "ST", "ST_lower", "ST_upper"]
    assert df.index.name == "param"


def test_no_bound_columns_without_bootstrap() -> None:
    """Without a bootstrap there is nothing to put in a bound column."""
    df = _sobol_result(n_bootstrap=0).to_pandas()
    assert list(df.columns) == ["S1", "ST"]


def test_pair_matrix_stays_in_the_dataset() -> None:
    """``S2`` is not in the table but is still reachable from the dataset."""
    result = _sobol_result(n_bootstrap=0)
    assert "S2" not in result.to_pandas().columns
    s2 = result.to_dataset()["S2"].to_pandas()
    assert s2.shape == (3, 3)
    np.testing.assert_allclose(s2.to_numpy(), np.asarray(result.S2), rtol=1e-6)


def test_time_coords_reach_the_index() -> None:
    """``time_coords`` labels the ``time`` index level."""
    result = build("sobol", "series")
    df = result.to_pandas(time_coords=[0.5, 1.5, 2.5])
    np.testing.assert_allclose(
        np.unique(df.index.get_level_values("time").to_numpy()), [0.5, 1.5, 2.5]
    )


def test_kucherenko_subset_arrays_stay_in_the_dataset() -> None:
    """The ``subset`` arrays of a ``subsets="all"`` design are left out."""
    design = kucherenko.sample(PROBLEM, base_n=16, subsets="all", seed=3, verbose=False)
    X = jnp.asarray(design.samples)
    result = kucherenko.analyze(design, X[:, 0] + 2.0 * X[:, 1], verbose=False)
    ds = result.to_dataset()
    assert "closed" in ds.data_vars
    df = result.to_pandas()
    _check_frame(df, ds)
    assert list(df.columns) == ["S1", "ST"]


def test_irregular_result_stacks_channels() -> None:
    """Ragged channels are stacked under an ``output`` level, each on its own grid."""
    problem = jaxgsa.Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)}, output_names=("a", "b")
    )
    design = sobol.sample(problem, n_samples=128, seed=0, verbose=False)
    rng = np.random.default_rng(0)
    t_a = jnp.array([0.0, 1.0, 2.0])
    t_b = jnp.array([0.0, 3.0])
    ragged = [
        (t_a, jnp.asarray(rng.normal(size=(design.n_runs, 3)))),
        (t_b, jnp.asarray(rng.normal(size=(design.n_runs, 2)))),
    ]
    result = sobol.analyze(design, ragged, verbose=False)
    assert isinstance(result, IrregularResult)
    df = result.to_pandas()
    assert list(df.index.names) == ["output", "time", "param"]
    assert len(df) == (3 + 2) * 3
    for name, times in (("a", t_a), ("b", t_b)):
        part = df.xs(name, level="output")
        np.testing.assert_allclose(
            np.unique(part.index.get_level_values("time").to_numpy()), np.asarray(times)
        )
        ds = result.to_dataset()
        expected = ds[f"{name}_S1"].transpose(f"time_{name}", "param").values.ravel()
        np.testing.assert_allclose(part["S1"].to_numpy(), expected, rtol=1e-6)


def test_param_frame_uses_the_dims_of_every_kept_variable() -> None:
    """A later variable with more dimensions than the first one still fits.

    The index takes every dimension any kept variable has. A variable that
    lacks a dimension is repeated along it.
    """
    from jaxgsa._core.result import param_frame

    names = ["x1", "x2", "x3"]
    ds = xr.Dataset(
        {
            "a": (("param",), np.array([0.1, 0.2, 0.3])),
            "b": (("time", "output", "param"), np.arange(12.0).reshape(2, 2, 3)),
        },
        coords={"param": names, "time": [0.0, 1.0], "output": ["u", "v"]},
    )
    df = param_frame(ds)
    assert list(df.index.names) == ["time", "output", "param"]
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2 * 2 * 3
    np.testing.assert_allclose(df["b"].to_numpy(), np.arange(12.0), rtol=1e-12)
    np.testing.assert_allclose(df["a"].to_numpy(), np.tile([0.1, 0.2, 0.3], 4), rtol=1e-12)
    np.testing.assert_allclose(df.loc[(1.0, "v", "x2"), "b"], 10.0, rtol=1e-12)


def _rank_channel(Y: np.ndarray) -> Any:
    """One regular rank result on three uniform inputs, for a hand-built IrregularResult."""
    from jaxgsa import rank

    problem = jaxgsa.Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)})
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(Y.shape[0], 3))
    Y = Y + X[:, :1].reshape((-1,) + (1,) * (Y.ndim - 1))
    return problem, rank.analyze(problem, jnp.asarray(X), jnp.asarray(Y), verbose=False)


def test_irregular_to_pandas_aligns_a_channel_without_time() -> None:
    """A channel with no ``time`` axis still stacks under the shared levels."""
    rng = np.random.default_rng(0)
    problem, timed = _rank_channel(rng.normal(size=(200, 2, 1)))
    _, flat = _rank_channel(rng.normal(size=(200,)))
    result = IrregularResult(
        channels={"a": timed, "b": flat},
        times={"a": jnp.array([0.0, 1.0]), "b": jnp.array([0.0])},
        problem=problem,
    )
    df = result.to_pandas()
    assert list(df.index.names) == ["output", "time", "param"]
    assert len(df) == 2 * 3 + 3
    assert df.xs("b", level="output").index.get_level_values("time").isna().all()
    np.testing.assert_allclose(
        df.xs("b", level="output")["S1"].to_numpy(), np.asarray(flat.S1), rtol=1e-6
    )


def test_irregular_to_pandas_keeps_several_outputs_of_a_channel() -> None:
    """A channel with more than one output keeps them on a ``channel_output`` level."""
    rng = np.random.default_rng(0)
    problem, single = _rank_channel(rng.normal(size=(200, 2, 1)))
    _, double = _rank_channel(rng.normal(size=(200, 2, 2)))
    result = IrregularResult(
        channels={"a": single, "b": double},
        times={"a": jnp.array([0.0, 1.0]), "b": jnp.array([0.0, 1.0])},
        problem=problem,
    )
    df = result.to_pandas()
    assert list(df.index.names) == ["output", "time", "channel_output", "param"]
    assert len(df) == 2 * 3 + 2 * 2 * 3
    assert df.xs("a", level="output").index.get_level_values("channel_output").isna().all()
    part = df.xs("b", level="output")
    assert part.index.get_level_values("channel_output").nunique() == 2
    expected = np.asarray(double.S1).ravel()  # (T, K, D) in time, output, param order
    np.testing.assert_allclose(part["S1"].to_numpy(), expected, rtol=1e-6)
