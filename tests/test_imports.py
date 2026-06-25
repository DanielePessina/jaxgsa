"""Tests that verify subpackage import paths and top-level re-exports."""


def test_sobol_subpackage():
    from gsax.sobol import SAResult, analyze

    assert callable(analyze)
    assert isinstance(SAResult, type)


def test_hdmr_subpackage():
    from gsax import hdmr

    assert callable(hdmr.analyze)
    assert callable(hdmr.emulate)


def test_pce_subpackage():
    from gsax import pce

    assert callable(pce.analyze)
    assert callable(pce.emulate)


def test_top_level_re_exports():
    import gsax

    assert callable(gsax.analyze)
    assert callable(gsax.analyze_hdmr)
    assert callable(gsax.analyze_pce)
    assert callable(gsax.emulate_hdmr)
    assert callable(gsax.emulate_pce)
    assert callable(gsax.sample)
    assert callable(gsax.load)


def test_re_exports_are_same_objects():
    import gsax
    from gsax.hdmr import analyze as hdmr_analyze
    from gsax.pce import analyze as pce_analyze
    from gsax.sobol import analyze as sobol_analyze

    assert gsax.analyze is sobol_analyze
    assert gsax.analyze_hdmr is hdmr_analyze
    assert gsax.analyze_pce is pce_analyze
