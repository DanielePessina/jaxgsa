"""Tests for the namespace-only public API."""

import gsax


def test_root_exports_foundational_types_and_namespaces():
    assert isinstance(gsax.Problem, type)
    assert isinstance(gsax.GaussianInputSpec, type)
    assert isinstance(gsax.UniformInputSpec, type)

    for namespace in (
        gsax.borgonovo,
        gsax.config,
        gsax.dgsm,
        gsax.efast,
        gsax.hdmr,
        gsax.hsic,
        gsax.morris,
        gsax.optimal_transport,
        gsax.pawn,
        gsax.pce,
        gsax.sampling,
        gsax.shapley,
        gsax.sobol,
    ):
        assert namespace is not None


def test_method_namespaces_expose_commands_and_results():
    assert callable(gsax.sobol.sample)
    assert callable(gsax.sobol.analyze)
    assert isinstance(gsax.sobol.SobolSamples, type)
    assert isinstance(gsax.sobol.SobolResult, type)

    assert callable(gsax.morris.sample)
    assert callable(gsax.morris.analyze)
    assert isinstance(gsax.morris.MorrisSamples, type)

    assert callable(gsax.pce.analyze)
    assert isinstance(gsax.pce.PCEResult, type)
    assert callable(gsax.hdmr.analyze)
    assert isinstance(gsax.hdmr.HDMRResult, type)

    assert callable(gsax.sampling.monte_carlo)
    assert isinstance(gsax.shapley.ShapleyResult, type)


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
        assert not hasattr(gsax, name)


def test_prediction_and_shapley_are_result_methods():
    assert callable(gsax.pce.PCEResult.predict)
    assert callable(gsax.pce.PCEResult.shapley)
    assert callable(gsax.hdmr.HDMRResult.predict)
    assert callable(gsax.hdmr.HDMRResult.shapley)
