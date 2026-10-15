"""Tests for core fixes: netCDF bool attributes, stacklevels, chunk-size types."""

import importlib.util
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import jaxgsa
from jaxgsa._core.result import _attr_value
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

_THIS_FILE = str(Path(__file__).resolve())


def _data(n: int = 400) -> tuple[jnp.ndarray, jnp.ndarray]:
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n, seed=1))
    return X, evaluate(X)


def test_attr_value_bool_becomes_int():
    assert _attr_value(True) == 1
    assert type(_attr_value(True)) is int
    assert _attr_value(np.bool_(False)) == 0
    assert _attr_value(jnp.asarray(True)) == 1
    assert _attr_value("text") == "text"


@pytest.mark.parametrize("method", ["pce", "hdmr", "shapley"])
def test_netcdf_roundtrip_scipy_engine(method, tmp_path):
    X, Y = _data()
    result = getattr(jaxgsa, method).analyze(PROBLEM, X, Y, verbose=False)
    ds = result.to_dataset()
    for value in ds.attrs.values():
        assert not isinstance(value, bool)
    path = tmp_path / f"{method}.nc"
    ds.to_netcdf(path, engine="scipy")
    back = xr.load_dataset(path, engine="scipy")
    for key in ("streamed", "include_correlative"):
        if key in ds.attrs:
            assert int(back.attrs[key]) == int(ds.attrs[key])


@pytest.mark.skipif(importlib.util.find_spec("netCDF4") is None, reason="netCDF4 missing")
@pytest.mark.parametrize("method", ["pce", "hdmr", "shapley"])
def test_netcdf_roundtrip_netcdf4_engine(method, tmp_path):
    X, Y = _data()
    result = getattr(jaxgsa, method).analyze(PROBLEM, X, Y, verbose=False)
    path = tmp_path / f"{method}.nc"
    result.to_dataset().to_netcdf(path, engine="netcdf4")
    xr.load_dataset(path, engine="netcdf4")


def test_netcdf_roundtrip_ot(tmp_path):
    X, Y = _data()
    result = jaxgsa.optimal_transport.analyze(PROBLEM, X, Y, verbose=False)
    path = tmp_path / "ot.nc"
    result.to_dataset().to_netcdf(path, engine="scipy")
    xr.load_dataset(path, engine="scipy")


def _constant_output(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    X, Y = _data(n)
    return X, jnp.stack([Y, jnp.ones_like(Y)], axis=-1)


@pytest.mark.parametrize("method", ["hdmr", "pce", "shapley"])
def test_zero_variance_warning_points_at_caller(method):
    X, Y = _constant_output(400)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(jaxgsa, method).analyze(PROBLEM, X, Y, verbose=False)
    zero = [w for w in caught if "zero variance" in str(w.message) or "constant" in str(w.message)]
    assert zero, [str(w.message) for w in caught]
    assert all(w.filename == _THIS_FILE for w in zero)


def test_efast_constant_slice_warns_once():
    e = jaxgsa.efast.sample(PROBLEM, n_per_curve=129, seed=0, verbose=False)
    Y = jnp.stack([evaluate(jnp.asarray(e.samples)), jnp.ones(len(e.samples))], axis=-1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jaxgsa.efast.analyze(e, Y, verbose=False)
    zero = [w for w in caught if "zero" in str(w.message) and "variance" in str(w.message)]
    assert len(zero) == 1, [str(w.message) for w in zero]


@pytest.mark.parametrize("bad", [2.0, True, 0, -1])
def test_sobol_slice_chunk_size_rejected(bad):
    s = jaxgsa.sobol.sample(PROBLEM, base_n=64, seed=0, verbose=False)
    Y = jnp.stack([evaluate(jnp.asarray(s.samples))] * 2, axis=-1)
    with pytest.raises(ValueError, match="jaxgsa.sobol: slice_chunk_size must be an integer >= 1"):
        jaxgsa.sobol.analyze(s, Y, slice_chunk_size=bad, verbose=False)


@pytest.mark.parametrize("bad", [2.0, True, 0])
def test_efast_slice_chunk_size_rejected(bad):
    e = jaxgsa.efast.sample(PROBLEM, n_per_curve=129, seed=0, verbose=False)
    Y = jnp.stack([evaluate(jnp.asarray(e.samples))] * 2, axis=-1)
    with pytest.raises(ValueError, match="jaxgsa.efast: slice_chunk_size must be an integer"):
        jaxgsa.efast.analyze(e, Y, slice_chunk_size=bad, verbose=False)


@pytest.mark.parametrize("name", ["slice_chunk_size", "batch_size"])
@pytest.mark.parametrize("bad", [2.0, True])
@pytest.mark.parametrize("fn", ["hdmr", "shapley"])
def test_hdmr_shapley_chunk_sizes_rejected(fn, bad, name):
    X, Y = _data()
    kwargs = {name: bad}
    if fn == "shapley":
        kwargs["backend"] = "hdmr"
    with pytest.raises(ValueError, match=f"jaxgsa.{fn}: {name} must be an integer >= 1"):
        getattr(jaxgsa, fn).analyze(PROBLEM, X, Y, verbose=False, **kwargs)


def test_positive_int_accepts_ints():
    from jaxgsa._core.entry import positive_int

    assert positive_int("n", None).ok
    assert positive_int("n", 3).ok
    assert positive_int("n", np.int64(3)).ok


def test_shapley_drop_names_shapley():
    X, Y = _data()
    Y = Y.at[3].set(jnp.nan)
    for backend in ("pce", "hdmr"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jaxgsa.shapley.analyze(
                PROBLEM, X, Y, backend=backend, on_invalid="drop", verbose=False
            )
        dropped = [str(w.message) for w in caught if "dropped" in str(w.message)]
        assert dropped
        assert all(m.startswith("jaxgsa.shapley.analyze") for m in dropped), dropped
