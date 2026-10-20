"""End-to-end eFAST checks on analytical fixtures."""

from __future__ import annotations

import warnings
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks import ishigami, linear
from jaxgsa.efast import EFASTIndices, analyze, indices, sample
from jaxgsa.efast._result import EFASTResult
from jaxgsa.problem import Problem


def _analyze(*args: Any, **kwargs: Any) -> EFASTResult:
    """Run ``analyze`` and narrow the result to ``EFASTResult`` for type checking."""
    return cast(EFASTResult, analyze(*args, **kwargs))


def test_ishigami_matches_analytical_fixture():
    design = sample(ishigami.PROBLEM, n_per_curve=1024, M=4, seed=2, verbose=False)
    result = _analyze(design, ishigami.evaluate(jnp.asarray(design.samples)), verbose=False)
    np.testing.assert_allclose(result.S1, ishigami.ANALYTICAL_S1, atol=0.08)
    np.testing.assert_allclose(result.ST, ishigami.ANALYTICAL_ST, atol=0.08)


def test_linear_fixture_has_equal_first_and_total_indices():
    design = sample(linear.PROBLEM, n_per_curve=1024, M=4, seed=3, verbose=False)
    result = _analyze(design, linear.evaluate(jnp.asarray(design.samples)), verbose=False)
    np.testing.assert_allclose(result.S1, linear.ANALYTICAL_S1, atol=0.08)
    np.testing.assert_allclose(result.ST, result.S1, atol=0.05)


def test_sampler_bounds_multi_output_shapes_and_m():
    """One M=6 design: reproducible, in bounds, multi-output, ``M`` carried into analysis."""
    problem = Problem(("a", "b"), ((2.0, 5.0), (-1.0, 3.0)))
    design = sample(problem, n_per_curve=257, M=6, seed=4, verbose=False)
    again = sample(problem, n_per_curve=257, M=6, seed=4, verbose=False)
    np.testing.assert_array_equal(design.samples, again.samples)
    assert np.all(np.asarray(design.samples[:, 0]) >= 2.0)
    assert np.all(np.asarray(design.samples[:, 0]) <= 5.0)
    assert np.all(np.asarray(design.samples[:, 1]) >= -1.0)
    assert np.all(np.asarray(design.samples[:, 1]) <= 3.0)

    X = jnp.asarray(design.samples)
    y = X[:, 0] + 2.0 * X[:, 1]
    outputs = jnp.stack([y, 2.0 * y], axis=-1)
    series = jnp.stack([outputs, outputs + 1.0], axis=1)
    result = _analyze(design, series, verbose=False)
    assert result.S1.shape == (2, 2, 2)
    assert result.ST.shape == (2, 2, 2)
    assert result.M == 6
    assert result.omega_0 > 0


def test_constant_curve_is_reported_as_nan():
    design = sample(ishigami.PROBLEM, n_per_curve=257, M=4, seed=7, verbose=False)
    outputs = ishigami.evaluate(jnp.asarray(design.samples)).at[: design.n_per_curve].set(3.0)
    with pytest.warns(JaxgsaWarning, match="zero variance"):
        result = _analyze(design, outputs, verbose=False)
    assert np.isnan(np.asarray(result.S1)[0])
    assert np.isnan(np.asarray(result.ST)[0])
    assert np.all(np.isfinite(np.asarray(result.S1)[1:]))


def test_invalid_row_policy_is_explicit():
    design = sample(linear.PROBLEM, n_per_curve=257, M=4, seed=8, verbose=False)
    outputs = linear.evaluate(jnp.asarray(design.samples)).at[4].set(jnp.nan)
    with pytest.raises(ValueError, match="non-finite"):
        _analyze(design, outputs, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=JaxgsaWarning)
        result = _analyze(design, outputs, on_invalid="propagate", verbose=False)
    assert result.invalid.n_invalid == 1


def test_traceable_core_matches_public_result():
    design = sample(linear.PROBLEM, n_per_curve=257, M=4, seed=9, verbose=False)
    outputs = linear.evaluate(jnp.asarray(design.samples))
    result = _analyze(design, outputs, verbose=False)
    core = indices(design, outputs)
    np.testing.assert_allclose(result.S1, core[0], atol=1e-6)
    np.testing.assert_allclose(result.ST, core[1], atol=1e-6)


def test_first_argument_must_be_efast_samples():
    for fn in (analyze, indices):
        with pytest.raises(TypeError, match="takes the EFASTSamples.*not a Problem"):
            fn(cast(Any, linear.PROBLEM), jnp.zeros(8))


def test_indices_named_tuple_ci_and_repr():
    design = sample(linear.PROBLEM, n_per_curve=1024, M=4, seed=3, verbose=False)
    Y = jnp.asarray(design.samples) @ jnp.asarray([1.0, 2.0, 3.0])
    out = indices(design, Y)
    assert isinstance(out, EFASTIndices)
    result = _analyze(design, Y, verbose=False)
    assert result.ci is None
    text = repr(design)
    assert "n_per_curve=1024" in text and "M=4" in text and len(text) < 200


def test_sample_warns_below_resonance_free_n():
    with pytest.warns(JaxgsaWarning, match="1 and 2"):
        sample(ishigami.PROBLEM, n_per_curve=129, M=4, seed=0, verbose=False)


def test_sample_silent_at_resonance_free_n():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sample(ishigami.PROBLEM, n_per_curve=193, M=4, seed=0, verbose=False)
