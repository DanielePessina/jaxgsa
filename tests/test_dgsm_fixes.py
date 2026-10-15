"""Regression tests for the 1.1.2 DGSM fixes."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import dgsm
from jaxgsa.dgsm import _core
from jaxgsa.dgsm._poincare import marginal_variance, poincare_constant
from jaxgsa.problem import GaussianSpec

PROBLEM = jaxgsa.Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0)})


def _sample(n: int = 2048) -> jax.Array:
    return jnp.asarray(np.random.default_rng(0).random((n, 2)), dtype=jnp.float32)


def test_bootstrap_ci_is_stable_under_large_output_offset() -> None:
    """The CI must not collapse when Y carries a large constant offset."""
    X = _sample()
    cis = {}
    for off in (0.0, 1e4):
        r = dgsm.analyze(
            PROBLEM,
            X,
            fn=lambda x, off=off: off + x[0] + 0.5 * x[1],
            n_bootstrap=100,
            key=0,
            verbose=False,
        )
        cis[off] = (np.asarray(r.upper_bound), np.asarray(r.upper_bound_conf))
    point, conf = cis[1e4]
    assert np.all(conf[0] <= point)
    assert np.all(point <= conf[1])
    np.testing.assert_allclose(conf, cis[0.0][1], rtol=5e-3)


@dataclass(frozen=True)
class _ArrayModel:
    w: object

    def __call__(self, x):
        return jnp.dot(jnp.asarray(self.w), x)


@pytest.mark.parametrize("w", [jnp.array([1.0, 2.0]), np.array([1.0, 2.0])])
def test_model_with_unhashable_fields_runs(w) -> None:
    """A frozen dataclass with array fields claims Hashable but cannot hash."""
    r = dgsm.analyze(PROBLEM, _sample(256), fn=_ArrayModel(w), verbose=False)
    np.testing.assert_allclose(np.asarray(r.nu), [1.0, 4.0], rtol=1e-5)


def test_is_hashable() -> None:
    assert _core._is_hashable(len)
    assert not _core._is_hashable(_ArrayModel(np.zeros(2)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mean": 100.0, "variance": 1.0, "low": 0.0},
        {"mean": 0.0, "variance": 1.0, "low": -100.0, "high": 100.0},
        {"mean": 0.0, "variance": 1.0, "low": 5.0},
        {"mean": 0.0, "variance": 1.0, "low": 9.0},
        {"mean": 0.0, "variance": 1.0, "high": -9.0},
        {"mean": 0.0, "variance": 1.0, "low": -3.0, "high": 3.0},
        {"mean": 0.0, "variance": 1.0, "low": 0.0},
    ],
)
def test_truncated_gaussian_poincare_bounds_variance(kwargs) -> None:
    """C is finite and at least the variance, even far from the mass."""
    spec = GaussianSpec(**kwargs)
    c = poincare_constant(spec)
    assert np.isfinite(c)
    assert c >= marginal_variance(spec) * (1 - 1e-9)


def test_truncated_gaussian_poincare_known_values() -> None:
    """Mild truncation keeps its value."""
    assert poincare_constant(GaussianSpec(0.0, 1.0, low=0.0)) == pytest.approx(0.50002, abs=1e-5)
    assert poincare_constant(GaussianSpec(0.0, 1.0, low=-8.0, high=8.0)) == pytest.approx(
        1.00008, abs=1e-5
    )


def test_analyze_with_positivity_truncated_far_gaussian() -> None:
    problem = jaxgsa.Problem.from_dict(
        {"k": GaussianSpec(mean=100.0, variance=1.0, low=0.0), "u": (0.0, 1.0)}
    )
    rng = np.random.default_rng(0)
    X = jnp.asarray(np.c_[rng.normal(100, 1, 512), rng.random(512)])
    r = dgsm.analyze(problem, X, fn=lambda x: x[0] + x[1], verbose=False)
    assert np.all(np.isfinite(np.asarray(r.upper_bound)))


def test_bootstrap_is_independent_of_batch_size() -> None:
    """The same key draws the same rows however the sample is batched."""
    X = _sample(1000)
    out = [
        dgsm.analyze(
            PROBLEM,
            X,
            fn=lambda x: jnp.sin(x[0]) + x[1] ** 2,
            n_bootstrap=20,
            key=1,
            batch_size=bs,
            verbose=False,
        )
        for bs in (1000, 300)
    ]
    np.testing.assert_allclose(np.asarray(out[0].nu_conf), np.asarray(out[1].nu_conf), rtol=1e-4)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"mean": 0.0, "variance": 1.0, "low": -1.0}, 0.7203579981925856),
        ({"mean": 0.0, "variance": 1.0, "low": -3.0, "high": 5.0}, 0.988539393225847),
        ({"mean": 2.0, "variance": 4.0, "low": 1.0}, 2.4260346384950227),
        ({"mean": 0.0, "variance": 1.0, "low": -8.0, "high": 8.0}, 1.0000814000776046),
    ],
)
def test_truncated_gaussian_poincare_matches_1_1_1(kwargs, expected) -> None:
    """Ordinary truncations keep the 1.1.1 interval, so the constant is unchanged."""
    assert poincare_constant(GaussianSpec(**kwargs)) == pytest.approx(expected, rel=3e-12)


def test_truncated_gaussian_poincare_mirror_symmetry() -> None:
    """A lower bound at -1 and the mirror upper bound at +1 give the same constant."""
    lower = poincare_constant(GaussianSpec(0.0, 1.0, low=-1.0))
    upper = poincare_constant(GaussianSpec(0.0, 1.0, high=1.0))
    assert lower == pytest.approx(upper, rel=1e-11)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"mean": 0.0, "variance": 1.0, "low": -1.0}, 0.6296862857766055),
        ({"mean": 0.0, "variance": 1.0, "low": 4.0}, 0.04667283839739844),
        ({"mean": 0.0, "variance": 1.0, "low": -3.0, "high": 5.0}, 0.9866593541888516),
        ({"mean": 2.0, "variance": 4.0, "low": 1.0}, 1.9447017427854685),
    ],
)
def test_truncated_gaussian_variance_matches_1_1_1(kwargs, expected) -> None:
    """Ordinary truncations use scipy's closed form, as in 1.1.1."""
    assert marginal_variance(GaussianSpec(**kwargs)) == pytest.approx(expected, rel=1e-14)
