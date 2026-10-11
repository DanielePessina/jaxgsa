"""Sampling contract of ``jaxgsa.sobol``: budget keywords, repr, and entry-point types."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa.benchmarks import ishigami


def test_sample_needs_exactly_one_of_n_samples_and_base_n():
    with pytest.raises(ValueError, match="n_samples.*base_n"):
        jaxgsa.sobol.sample(ishigami.PROBLEM, verbose=False)
    with pytest.raises(ValueError, match="n_samples.*base_n"):
        jaxgsa.sobol.sample(ishigami.PROBLEM, 1024, base_n=128, verbose=False)


def test_base_n_alone_sets_the_design():
    samples = jaxgsa.sobol.sample(ishigami.PROBLEM, base_n=64, seed=0, verbose=False)
    assert samples.base_n == 64


def test_a_seed_without_scrambling_is_refused():
    with pytest.raises(ValueError, match="seed has no effect with scramble=False"):
        jaxgsa.sobol.sample(ishigami.PROBLEM, 1024, scramble=False, seed=1, verbose=False)


def test_repr_is_compact_and_holds_no_arrays():
    samples = jaxgsa.sobol.sample(ishigami.PROBLEM, 512, seed=0, verbose=False)
    text = repr(samples)
    assert text.startswith("SobolSamples(n_runs=")
    for field in ("n_expanded", "base_n", "n_params", "calc_second_order"):
        assert field in text
    assert len(text) < 200
    assert "array" not in text


@pytest.mark.parametrize("second_order", [True, False])
def test_indices_always_returns_three_named_fields(second_order):
    samples = jaxgsa.sobol.sample(
        ishigami.PROBLEM, base_n=64, calc_second_order=second_order, seed=0, verbose=False
    )
    Y = jnp.asarray(ishigami.evaluate(samples.samples))
    out = jaxgsa.sobol.indices(samples, Y)
    assert isinstance(out, jaxgsa.sobol.SobolIndices)
    S1, ST, S2 = out
    assert out.S1 is S1 and out.ST is ST
    assert (S2 is None) == (not second_order)


def test_entry_points_refuse_a_problem_with_a_pointer_to_given_data_methods():
    Y = np.zeros(8)
    for function in (jaxgsa.sobol.analyze, jaxgsa.sobol.indices):
        with pytest.raises(TypeError, match=r"SobolSamples.*not a Problem.*jaxgsa\.rank"):
            function(ishigami.PROBLEM, Y)  # type: ignore[arg-type]
