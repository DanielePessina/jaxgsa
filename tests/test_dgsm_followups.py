"""Follow-up fixes for jaxgsa.dgsm and jaxgsa.kucherenko."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import truncnorm

import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate
from jaxgsa.dgsm._poincare import marginal_variance, poincare_constant
from jaxgsa.problem import GaussianSpec

_NARROW = [
    (1.0, 1.0 + 1e-9),
    (-1e-6, 1e-6),
    (1e4, None),
    (40.0, 40.0001),
    (None, -1e4),
]
_ORDINARY = [(None, None), (-3.0, 3.0), (0.0, None), (1.0, None), (-5.0, 2.0), (0.5, 0.9)]


def _spec(low: float | None, high: float | None) -> GaussianSpec:
    return GaussianSpec(mean=0.0, variance=1.0, low=low, high=high)


@pytest.mark.parametrize(("low", "high"), _NARROW)
def test_variance_positive_and_matches_uniform_or_tail_limit(low, high):
    var = marginal_variance(_spec(low, high))
    assert var > 0.0
    if low is not None and high is not None:
        width = high - low
        assert var == pytest.approx(width**2 / 12.0, rel=1e-6)
    elif low is not None:
        # Far upper tail: the density is exponential with rate low.
        assert var == pytest.approx(1.0 / low**2, rel=1e-6)


@pytest.mark.parametrize(("low", "high"), _ORDINARY)
def test_variance_matches_truncnorm_where_well_conditioned(low, high):
    a = -np.inf if low is None else low
    b = np.inf if high is None else high
    assert marginal_variance(_spec(low, high)) == pytest.approx(truncnorm.var(a, b), rel=1e-11)


@pytest.mark.parametrize(("low", "high"), [*_NARROW[:2], *_ORDINARY[1:]])
def test_poincare_constant_at_least_variance(low, high):
    spec = _spec(low, high)
    assert poincare_constant(spec) >= marginal_variance(spec)


def _model(x):
    return jnp.stack([evaluate(x[None, :])[0], jnp.asarray(1.0)])


def test_dgsm_zero_variance_warning_points_at_caller():
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, 60, seed=1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jaxgsa.dgsm.analyze(PROBLEM, X, fn=_model, verbose=False)
    zero = [w for w in caught if "zero variance" in str(w.message)]
    assert len(zero) == 1
    assert zero[0].filename == __file__


def test_kucherenko_zero_variance_warning_points_at_caller_and_has_no_t_label():
    design = jaxgsa.kucherenko.sample(PROBLEM, base_n=64, seed=0, verbose=False)
    y = evaluate(jnp.asarray(design.samples))
    Y = jnp.stack([y, jnp.ones_like(y)], axis=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jaxgsa.kucherenko.analyze(design, Y, verbose=False)
    zero = [w for w in caught if "zero variance" in str(w.message)]
    assert len(zero) == 1
    assert zero[0].filename == __file__
    assert "(k=1)" in str(zero[0].message)
    assert "t=" not in str(zero[0].message)


def test_dgsm_key_error_has_method_prefix():
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, 20, seed=1))
    with pytest.raises(ValueError, match=r"^jaxgsa\.dgsm: .*key is required"):
        jaxgsa.dgsm.analyze(PROBLEM, X, fn=_model, n_bootstrap=10, verbose=False)
