"""End-to-end eFAST checks on analytical fixtures."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks import ishigami, linear
from jaxgsa.efast import analyze, indices, sample
from jaxgsa.problem import Problem


def test_ishigami_matches_analytical_fixture():
    design = sample(ishigami.PROBLEM, n_per_curve=1024, M=4, seed=2, verbose=False)
    result = analyze(design, ishigami.evaluate(jnp.asarray(design.samples)), verbose=False)
    np.testing.assert_allclose(result.S1, ishigami.ANALYTICAL_S1, atol=0.08)
    np.testing.assert_allclose(result.ST, ishigami.ANALYTICAL_ST, atol=0.08)


def test_linear_fixture_has_equal_first_and_total_indices():
    design = sample(linear.PROBLEM, n_per_curve=1024, M=4, seed=3, verbose=False)
    result = analyze(design, linear.evaluate(jnp.asarray(design.samples)), verbose=False)
    np.testing.assert_allclose(result.S1, linear.ANALYTICAL_S1, atol=0.08)
    np.testing.assert_allclose(result.ST, result.S1, atol=0.05)


def test_sampler_is_reproducible_and_respects_bounds():
    problem = Problem(("a", "b"), ((2.0, 5.0), (-1.0, 3.0)))
    first = sample(problem, n_per_curve=257, M=4, seed=4, verbose=False)
    second = sample(problem, n_per_curve=257, M=4, seed=4, verbose=False)
    np.testing.assert_array_equal(first.samples, second.samples)
    assert np.all(np.asarray(first.samples[:, 0]) >= 2.0)
    assert np.all(np.asarray(first.samples[:, 0]) <= 5.0)


def test_multi_output_and_time_series_shapes():
    design = sample(linear.PROBLEM, n_per_curve=257, M=4, seed=5, verbose=False)
    y = linear.evaluate(jnp.asarray(design.samples))
    outputs = jnp.stack([y, 2.0 * y], axis=-1)
    series = jnp.stack([outputs, outputs + 1.0], axis=1)
    result = analyze(design, series, verbose=False)
    assert result.S1.shape == (2, 2, 3)
    assert result.ST.shape == (2, 2, 3)


def test_design_carries_m_into_analysis():
    design = sample(ishigami.PROBLEM, n_per_curve=1024, M=6, seed=6, verbose=False)
    result = analyze(design, ishigami.evaluate(jnp.asarray(design.samples)), verbose=False)
    assert result.M == 6
    assert result.omega_0 > 0


def test_constant_curve_is_reported_as_nan():
    design = sample(ishigami.PROBLEM, n_per_curve=257, M=4, seed=7, verbose=False)
    outputs = ishigami.evaluate(jnp.asarray(design.samples)).at[: design.n_per_curve].set(3.0)
    with pytest.warns(JaxgsaWarning, match="zero variance"):
        result = analyze(design, outputs, verbose=False)
    assert np.isnan(np.asarray(result.S1)[0])
    assert np.isnan(np.asarray(result.ST)[0])
    assert np.all(np.isfinite(np.asarray(result.S1)[1:]))


def test_invalid_row_policy_is_explicit():
    design = sample(linear.PROBLEM, n_per_curve=257, M=4, seed=8, verbose=False)
    outputs = linear.evaluate(jnp.asarray(design.samples)).at[4].set(jnp.nan)
    with pytest.raises(ValueError, match="non-finite"):
        analyze(design, outputs, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=JaxgsaWarning)
        result = analyze(design, outputs, on_invalid="propagate", verbose=False)
    assert result.invalid.n_invalid == 1


def test_traceable_core_matches_public_result():
    design = sample(linear.PROBLEM, n_per_curve=257, M=4, seed=9, verbose=False)
    outputs = linear.evaluate(jnp.asarray(design.samples))
    result = analyze(design, outputs, verbose=False)
    core = indices(design, outputs)
    np.testing.assert_allclose(result.S1, core[0], atol=1e-6)
    np.testing.assert_allclose(result.ST, core[1], atol=1e-6)
