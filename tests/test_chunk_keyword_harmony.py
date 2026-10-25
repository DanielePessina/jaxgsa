"""The chunking keywords behave the same way in every method (1.3.1).

Every ``batch_size``, ``slice_chunk_size`` and ``resample_chunk_size`` is
checked up front by the same rule: ``None`` or an integer of at least 1. A
float such as ``2.0`` raises a ``ValueError`` that names the keyword, before
any work runs. A NumPy integer is accepted. The verbose summary prints the
width that ran, as a number, not ``auto``.
"""

import re
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa

PROBLEM = jaxgsa.Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)})
N = 320
KEY = jax.random.key(0)
VKOGA_KWARGS: dict[str, Any] = dict(
    gamma=3.0, ridge=1e-6, max_centers=32, n_outer=32, n_inner=8, n_variance=128, key=KEY
)


def _xy() -> tuple[jax.Array, jax.Array]:
    X = np.random.default_rng(0).uniform(size=(N, 3))
    Y = np.stack([X[:, 0] + X[:, 1] ** 2, X[:, 0] * X[:, 2]], axis=-1)
    return jnp.asarray(X), jnp.asarray(Y)


def _model(x: jax.Array) -> jax.Array:
    return jnp.sin(x[0]) + x[1] ** 2 * x[2]


def _call(method: str, keyword: str, value: object, verbose: bool = False) -> object:
    """Run ``method`` on a small case with ``keyword=value``."""
    X, Y = _xy()
    kw: dict[str, Any] = {keyword: value, "verbose": verbose}
    if method == "sobol":
        s = jaxgsa.sobol.sample(PROBLEM, base_n=64, seed=0, verbose=False)
        Xs = jnp.asarray(s.samples)
        return jaxgsa.sobol.analyze(s, jnp.stack([Xs[:, 0], Xs[:, 1]], -1), **kw)
    if method == "efast":
        e = jaxgsa.efast.sample(PROBLEM, n_per_curve=193, seed=0, verbose=False)
        Xe = jnp.asarray(e.samples)
        return jaxgsa.efast.analyze(e, jnp.stack([Xe[:, 0], Xe[:, 1]], -1), **kw)
    if method == "kucherenko":
        k = jaxgsa.kucherenko.sample(PROBLEM, base_n=64, seed=0, verbose=False)
        Xk = jnp.asarray(k.samples)
        return jaxgsa.kucherenko.analyze(k, jnp.stack([Xk[:, 0], Xk[:, 1]], -1), **kw)
    if method == "morris":
        m = jaxgsa.morris.sample(PROBLEM, n_trajectories=20, seed=0, verbose=False)
        Xm = jnp.asarray(m.samples)
        if keyword == "resample_chunk_size":
            kw.update(n_bootstrap=8, key=KEY)
        return jaxgsa.morris.analyze(m, jnp.stack([Xm[:, 0], Xm[:, 1]], -1), **kw)
    if method == "dgsm":
        return jaxgsa.dgsm.analyze(PROBLEM, X, fn=_model, **kw)
    if method == "vkoga":
        return jaxgsa.vkoga.analyze(PROBLEM, X, Y, **VKOGA_KWARGS, **kw)
    if method == "shapley":
        return jaxgsa.shapley.analyze(PROBLEM, X, Y, backend="hdmr", order=1, **kw)
    if method == "hdmr":
        return jaxgsa.hdmr.analyze(PROBLEM, X, Y, order=1, **kw)
    fn: Callable[..., object] = getattr(jaxgsa, method).analyze
    return fn(PROBLEM, X, Y, **kw)


CASES = [
    ("sobol", "slice_chunk_size"),
    ("efast", "slice_chunk_size"),
    ("borgonovo", "slice_chunk_size"),
    ("pawn", "slice_chunk_size"),
    ("optimal_transport", "slice_chunk_size"),
    ("hdmr", "slice_chunk_size"),
    ("hdmr", "batch_size"),
    ("dgsm", "batch_size"),
    ("vkoga", "batch_size"),
    ("kucherenko", "slice_chunk_size"),
    ("morris", "slice_chunk_size"),
    ("morris", "resample_chunk_size"),
    ("rank", "slice_chunk_size"),
    ("shapley", "slice_chunk_size"),
    ("shapley", "batch_size"),
]


@pytest.mark.parametrize(("method", "keyword"), CASES)
@pytest.mark.parametrize("bad", [2.0, True, 0])
def test_bad_chunk_value_raises_value_error_naming_keyword(method, keyword, bad):
    """A float, a bool or 0 raises ValueError up front, naming the keyword."""
    with pytest.raises(ValueError, match=f"{keyword} must be an integer >= 1"):
        _call(method, keyword, bad)


@pytest.mark.parametrize(
    ("method", "keyword"),
    [
        ("borgonovo", "slice_chunk_size"),
        ("pawn", "slice_chunk_size"),
        ("optimal_transport", "slice_chunk_size"),
        ("dgsm", "batch_size"),
        ("morris", "resample_chunk_size"),
    ],
)
def test_numpy_integer_accepted(method, keyword):
    """A NumPy integer passes the check like a Python int."""
    _call(method, keyword, np.int64(2))


VERBOSE_CASES = [
    ("hdmr", "slice_chunk_size"),
    ("hdmr", "batch_size"),
    ("dgsm", "batch_size"),
    ("vkoga", "batch_size"),
    ("optimal_transport", "slice_chunk_size"),
    ("morris", "resample_chunk_size"),
    ("shapley", "slice_chunk_size"),
    ("shapley", "batch_size"),
]


@pytest.mark.verbose_output
@pytest.mark.parametrize(("method", "keyword"), VERBOSE_CASES)
def test_verbose_prints_resolved_number(method, keyword, capsys):
    """The summary prints the resolved width, with the same suffix everywhere."""
    _call(method, keyword, None, verbose=True)
    out = capsys.readouterr().out
    assert re.search(rf"{keyword}: \d+ \(resolved from the memory budget\)", out), out
    assert "auto (resolved" not in out


@pytest.mark.verbose_output
@pytest.mark.parametrize(("method", "keyword"), VERBOSE_CASES)
def test_verbose_prints_clamped_user_value(method, keyword, capsys):
    """A user value is reported as the width that ran, clamped to its axis."""
    # Two rows per vkoga surrogate call would mean thousands of calls.
    value = 64 if method == "vkoga" else 2
    _call(method, keyword, value, verbose=True)
    out = capsys.readouterr().out
    assert f"{keyword}: {value} (user-set)" in out, out


@pytest.mark.verbose_output
def test_sobol_verbose_has_no_resample_chunk_size_line(capsys):
    """sobol has no resample_chunk_size keyword, so its summary does not print one."""
    s = jaxgsa.sobol.sample(PROBLEM, base_n=64, seed=0, verbose=False)
    Y = jnp.asarray(s.samples)[:, 0]
    jaxgsa.sobol.analyze(s, Y, n_bootstrap=8, key=KEY, verbose=True)
    out = capsys.readouterr().out
    assert "resample_chunk_size" not in out
    assert re.search(r"bootstrap resamples per call: \d+", out), out
