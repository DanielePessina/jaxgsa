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
    if method == "shapley_pce":
        return jaxgsa.shapley.analyze(PROBLEM, X, Y, backend="pce", **kw)
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
    ("pce", "batch_size"),
    ("shapley_pce", "batch_size"),
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
        ("pce", "batch_size"),
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
    ("pce", "batch_size"),
    ("shapley_pce", "batch_size"),
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


# --------------------------------------------------------------------------
# efast: slice_chunk_size counts T*K output slices, each with all D curves
# --------------------------------------------------------------------------

T_E, K_E = 3, 5  # 15 output slices; widths 4 and 7 leave a ragged last chunk


@pytest.fixture(scope="module")
def efast_case() -> tuple[Any, jax.Array]:
    design = jaxgsa.efast.sample(PROBLEM, n_per_curve=257, seed=0, verbose=False)
    X = np.asarray(design.samples)
    Y = np.stack(
        [
            np.stack(
                [(t + 1) * X[:, 0] + k * X[:, 1] ** 2 + X[:, 0] * X[:, 2] for k in range(K_E)], -1
            )
            for t in range(T_E)
        ],
        axis=1,
    )
    return design, jnp.asarray(Y)


def test_efast_widths_agree_with_per_slice_scalar(efast_case):
    """Every width gives the indices of a scalar analysis of each slice."""
    design, Y = efast_case
    ref_S1 = np.empty((T_E, K_E, 3))
    ref_ST = np.empty((T_E, K_E, 3))
    for t in range(T_E):
        for k in range(K_E):
            r = jaxgsa.efast.analyze(design, Y[:, t, k], verbose=False)
            assert isinstance(r, jaxgsa.efast.EFASTResult)
            ref_S1[t, k], ref_ST[t, k] = np.asarray(r.S1), np.asarray(r.ST)
    for width in (1, 4, 7, None):
        res = jaxgsa.efast.analyze(design, Y, slice_chunk_size=width, verbose=False)
        assert isinstance(res, jaxgsa.efast.EFASTResult)
        np.testing.assert_allclose(np.asarray(res.S1), ref_S1, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(res.ST), ref_ST, rtol=1e-5, atol=1e-6)
        idx = jaxgsa.efast.indices(design, Y, slice_chunk_size=width)
        np.testing.assert_allclose(np.asarray(idx.S1), ref_S1, rtol=1e-5, atol=1e-6)


def test_efast_width_counts_output_slices(efast_case, monkeypatch):
    """A user width w gathers w output slices, that is D * w curve rows."""
    from jaxgsa.efast import _analyze as efast_mod

    design, Y = efast_case
    rows: list[int] = []
    real = efast_mod._get_slice_chunk

    def spy(D: int, N: int, width: int):
        fn = real(D, N, width)

        def wrapped(Y_: jax.Array, start: Any) -> jax.Array:
            out = fn(Y_, start)
            rows.append(int(out.shape[0]))
            return out

        return wrapped

    monkeypatch.setattr(efast_mod, "_get_slice_chunk", spy)
    jaxgsa.efast.analyze(design, Y, slice_chunk_size=4, verbose=False)
    assert rows == [3 * 4] * 4  # ceil(15 / 4) chunks of D * 4 rows


@pytest.mark.verbose_output
def test_efast_verbose_prints_output_slices(efast_case, capsys):
    design, Y = efast_case
    jaxgsa.efast.analyze(design, Y, slice_chunk_size=4, verbose=True)
    assert "slice_chunk_size: 4 (user-set)" in capsys.readouterr().out
    jaxgsa.efast.analyze(design, Y, slice_chunk_size=10**6, verbose=True)
    assert "slice_chunk_size: 15 (user-set)" in capsys.readouterr().out
    jaxgsa.efast.analyze(design, Y, verbose=True)
    assert "slice_chunk_size: 15 (resolved from the memory budget)" in capsys.readouterr().out


@pytest.mark.verbose_output
@pytest.mark.parametrize("method", ["pce", "shapley_pce"])
def test_pce_verbose_reports_the_single_pass_width(method, capsys):
    """A user batch_size at or above N runs the single pass, reported as N."""
    _call(method, "batch_size", 10 * N, verbose=True)
    out = capsys.readouterr().out
    assert f"batch_size: {N} (user-set)" in out, out


@pytest.mark.parametrize("bad", [2.0, True, 0])
def test_surrogate_predict_bad_batch_size_raises_value_error(bad):
    """SurrogateResult.predict follows the same rule as the analysis keywords."""
    X, Y = _xy()
    result = jaxgsa.pce.analyze(PROBLEM, X, Y, verbose=False)
    assert isinstance(result, jaxgsa.pce.PCEResult)
    numpy_int: Any = np.int64(32)
    with pytest.raises(ValueError, match="batch_size must be an integer >= 1"):
        result.predict(X, batch_size=bad)
    result.predict(X, batch_size=numpy_int)
