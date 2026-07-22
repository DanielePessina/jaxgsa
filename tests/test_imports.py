"""Tests for the namespace-only public API."""

import jaxgsa


def test_root_exports_foundational_types_and_namespaces():
    assert isinstance(jaxgsa.Problem, type)
    assert isinstance(jaxgsa.GaussianInputSpec, type)
    assert isinstance(jaxgsa.UniformInputSpec, type)

    for namespace in (
        jaxgsa.borgonovo,
        jaxgsa.config,
        jaxgsa.dgsm,
        jaxgsa.efast,
        jaxgsa.hdmr,
        jaxgsa.hsic,
        jaxgsa.morris,
        jaxgsa.optimal_transport,
        jaxgsa.pawn,
        jaxgsa.pce,
        jaxgsa.sampling,
        jaxgsa.shapley,
        jaxgsa.sobol,
    ):
        assert namespace is not None


def test_method_namespaces_expose_commands_and_results():
    assert callable(jaxgsa.sobol.sample)
    assert callable(jaxgsa.sobol.analyze)
    assert isinstance(jaxgsa.sobol.SobolSamples, type)
    assert isinstance(jaxgsa.sobol.SobolResult, type)

    assert callable(jaxgsa.morris.sample)
    assert callable(jaxgsa.morris.analyze)
    assert isinstance(jaxgsa.morris.MorrisSamples, type)

    assert callable(jaxgsa.pce.analyze)
    assert isinstance(jaxgsa.pce.PCEResult, type)
    assert callable(jaxgsa.hdmr.analyze)
    assert isinstance(jaxgsa.hdmr.HDMRResult, type)

    assert callable(jaxgsa.sampling.monte_carlo)
    assert isinstance(jaxgsa.shapley.ShapleyResult, type)


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
