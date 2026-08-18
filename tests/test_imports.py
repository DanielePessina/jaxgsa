"""Tests for the namespace-only public API.

The method namespaces are enumerated from the registry, never listed here. An
earlier hand-written list in this file covered eleven of the thirteen methods:
``kucherenko`` and ``vkoga`` were exported in ``jaxgsa.__all__`` and were not
checked, because adding a method meant remembering to edit this list and
nothing said so. ``tests/test_registry.py`` proves the registry is complete
against the source tree, so deriving from it here is safe.
"""

import pytest

import jaxgsa
from jaxgsa._core.registry import methods

ALL_SPECS = sorted(methods().values(), key=lambda s: s.name)
SPEC_IDS = [s.name for s in ALL_SPECS]


def test_root_exports_foundational_types():
    assert isinstance(jaxgsa.Problem, type)
    assert isinstance(jaxgsa.GaussianInputSpec, type)
    assert isinstance(jaxgsa.UniformInputSpec, type)


def test_root_exports_the_support_namespaces():
    assert jaxgsa.config is not None
    assert jaxgsa.sampling is not None
    assert callable(jaxgsa.sampling.monte_carlo)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_every_method_namespace_is_reachable_from_the_root(spec):
    assert getattr(jaxgsa, spec.name) is not None


@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_every_method_exposes_analyze_and_a_result_type(spec):
    namespace = getattr(jaxgsa, spec.name)
    assert callable(namespace.analyze)
    assert isinstance(spec.result, type)
    assert getattr(namespace, spec.result.__name__) is spec.result


@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_a_design_based_method_exposes_sample_and_a_samples_type(spec):
    namespace = getattr(jaxgsa, spec.name)
    if not spec.is_design_based:
        assert not hasattr(namespace, "sample")
        return
    assert callable(namespace.sample)
    samples_types = [n for n in namespace.__all__ if n.endswith("Samples")]
    assert len(samples_types) == 1, f"{spec.name} should export exactly one *Samples type"
    assert isinstance(getattr(namespace, samples_types[0]), type)


def test_removed_root_shortcuts_are_absent():
    for name in (
        "sample",
        "analyze",
        "load",
        "monte_carlo",
        "analyze_pce",
        "emulate_pce",
        "analyze_hdmr",
        "emulate_hdmr",
        "analyze_shapley",
    ):
        assert not hasattr(jaxgsa, name)


def test_prediction_and_shapley_are_result_methods():
    assert callable(jaxgsa.pce.PCEResult.predict)
    assert callable(jaxgsa.pce.PCEResult.shapley)
    assert callable(jaxgsa.hdmr.HDMRResult.predict)
    assert callable(jaxgsa.hdmr.HDMRResult.shapley)
