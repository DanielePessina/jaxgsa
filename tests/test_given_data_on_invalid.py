"""The shared ``on_invalid`` policy, checked once for every given-data method.

HSIC, optimal transport, Borgonovo and PAWN all read a given ``(X, Y)`` sample
and all take the same ``on_invalid`` policy: ``"raise"`` (the default),
``"propagate"`` and ``"drop"``. One parametrized test covers the four
behaviours for all four methods.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _fixtures_givendata import invalid_sample

from jaxgsa import JaxgsaWarning, borgonovo, hsic, optimal_transport, pawn

N_ROWS = 200
BAD_Y_ROW = 7
BAD_X_ROW = 4

# name -> (module, extra analyze kwargs, result field holding the main index)
METHODS = {
    "pawn": (pawn, {}, "pawn"),
    "hsic": (hsic, {"n_perms": 5, "key": jax.random.key(0)}, "R2"),
    "borgonovo": (borgonovo, {"n_bootstrap": 0}, "delta"),
    "optimal_transport": (optimal_transport, {}, "ot"),
}

each_method = pytest.mark.parametrize("method", sorted(METHODS))


def _analyze(method: str, problem, X, Y, **kwargs):
    module, defaults, _ = METHODS[method]
    return module.analyze(problem, X, Y, verbose=False, **{**defaults, **kwargs})


def _main_index(method: str, result) -> np.ndarray:
    return np.asarray(getattr(result, METHODS[method][2]))


@each_method
def test_raise_is_the_default_and_names_the_rows(method):
    """A non-finite Y row refuses the analysis and says which row it is."""
    problem, X, Y = invalid_sample(N_ROWS)
    Y = Y.at[BAD_Y_ROW].set(jnp.nan)
    with pytest.raises(ValueError) as exc:
        _analyze(method, problem, X, Y)
    message = str(exc.value)
    assert f"jaxgsa.{method}.analyze" in message
    assert f"1 of {N_ROWS} rows" in message
    assert f"[{BAD_Y_ROW}]" in message
    # The failure is a bad row, not a late bandwidth or grid error.
    assert "half L1 distance" not in message
    assert "degenerate_bandwidth" not in message


@each_method
def test_propagate_warns_and_keeps_the_row(method):
    """'propagate' computes anyway, loudly, and drops nothing.

    HSIC, Borgonovo and OT average over the whole sample, so one NaN reaches
    every index. PAWN reads Y through an ECDF, so a NaN there shifts the index
    instead of poisoning it. For PAWN the row counts as kept when the result
    differs from the result with the row removed.
    """
    problem, X, Y = invalid_sample(N_ROWS)
    Y = Y.at[BAD_Y_ROW].set(jnp.nan)
    with pytest.warns(JaxgsaWarning, match="reaches the indices"):
        result = _analyze(method, problem, X, Y, on_invalid="propagate")
    assert result.invalid.policy == "propagate"
    assert result.invalid.n_invalid == 1
    assert result.invalid.unit_indices == (BAD_Y_ROW,)

    got = _main_index(method, result)
    if method == "pawn":
        keep = np.ones(N_ROWS, dtype=bool)
        keep[BAD_Y_ROW] = False
        without = _analyze(method, problem, X[keep], Y[keep])
        assert not np.array_equal(got, _main_index(method, without))
    else:
        assert not np.all(np.isfinite(got))


@each_method
def test_drop_removes_the_row_and_returns_finite_indices(method):
    """'drop' analyzes the remainder and the indices come back usable."""
    problem, X, Y = invalid_sample(N_ROWS)
    Y = Y.at[BAD_Y_ROW].set(jnp.nan)
    with pytest.warns(JaxgsaWarning, match=f"dropped 1 of {N_ROWS} rows"):
        result = _analyze(method, problem, X, Y, on_invalid="drop")
    assert result.invalid.n_kept == N_ROWS - 1
    assert result.invalid.unit_indices == (BAD_Y_ROW,)
    got = _main_index(method, result)
    assert np.all(np.isfinite(got))
    assert np.all(got >= -1e-6) and np.all(got <= 1.0 + 1e-6)


@each_method
def test_a_bad_x_is_caught_and_named(method):
    """A non-finite input is caught too, and the report says it was X."""
    problem, X, Y = invalid_sample(N_ROWS)
    X = X.at[BAD_X_ROW, 1].set(jnp.inf)
    with pytest.raises(ValueError, match=r"in X\b") as exc:
        _analyze(method, problem, X, Y)
    assert f"[{BAD_X_ROW}]" in str(exc.value)
    with pytest.warns(JaxgsaWarning):
        result = _analyze(method, problem, X, Y, on_invalid="drop")
    assert result.invalid.sources == ("X",)
    assert result.invalid.unit_indices == (BAD_X_ROW,)
