"""Tests for the 1.1.2 HSIC fixes: NaN p-values and the collapsed median warning."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa.hsic as hsic
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.hsic import HSICResult
from jaxgsa.problem import Problem

PROBLEM = Problem(names=("a", "b"), bounds=((0.0, 1.0), (0.0, 1.0)))


def _x(n: int = 300) -> np.ndarray:
    return np.random.default_rng(0).random((n, 2)).astype(np.float32)


def _collapsed(rec: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in rec if "median heuristic failed" in str(w.message)]


def test_zero_variance_output_gives_nan_p_values() -> None:
    X = _x()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = hsic.analyze(
            PROBLEM,
            jnp.asarray(X),
            jnp.ones(len(X), jnp.float32),
            key=0,
            verbose=False,
            n_perms=20,
        )
    assert isinstance(r, HSICResult)
    assert np.all(np.isnan(np.asarray(r.R2)))
    assert np.all(np.isnan(np.asarray(r.T)))
    assert np.all(np.isnan(np.asarray(r.p_values)))


def test_collapsed_median_warns_for_zero_inflated_output() -> None:
    X = _x()
    y = np.maximum(0.0, X[:, 0] - 0.75).astype(np.float32)
    with pytest.warns(JaxgsaWarning, match="median heuristic failed") as rec:
        hsic.analyze(PROBLEM, jnp.asarray(X), jnp.asarray(y), key=0, verbose=False, n_perms=10)
    hit = _collapsed(list(rec))
    assert len(hit) == 1
    assert hit[0].filename == __file__


def test_collapsed_median_warns_for_zero_inflated_input() -> None:
    X = _x()
    X[:, 1] = np.maximum(0.0, X[:, 1] - 0.75)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)
    with pytest.warns(JaxgsaWarning, match="parameter 'b'"):
        hsic.analyze(PROBLEM, jnp.asarray(X), jnp.asarray(y), key=0, verbose=False, n_perms=10)


def test_no_collapsed_median_warning_on_ordinary_data() -> None:
    X = _x()
    y = np.sin(6 * X[:, 0]).astype(np.float32)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        hsic.analyze(PROBLEM, jnp.asarray(X), jnp.asarray(y), key=0, verbose=False, n_perms=10)
    assert not _collapsed(rec)


def test_no_collapsed_median_warning_for_binary_output() -> None:
    X = _x()
    y = (X[:, 0] > 0.3).astype(np.float32)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        hsic.analyze(PROBLEM, jnp.asarray(X), jnp.asarray(y), key=0, verbose=False, n_perms=10)
    assert not _collapsed(rec)


def test_indices_still_jits_and_vmaps_on_collapsed_data() -> None:
    X = jnp.asarray(_x(100))
    y = jnp.asarray(np.maximum(0.0, np.asarray(X[:, 0]) - 0.75).astype(np.float32))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = jax.jit(lambda a, b: hsic.indices(PROBLEM, a, b, key=0, n_perms=5).R2)(X, y)
        batch = jax.vmap(lambda b: hsic.indices(PROBLEM, X, b, key=0, n_perms=5).R2)(
            jnp.stack([y, y + 1.0])
        )
    assert out.shape == (2,)
    assert batch.shape == (2, 2)
    assert not _collapsed(rec)
