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


def test_shapley_subpackage():
    from gsax import shapley

    assert callable(shapley.analyze)
    assert isinstance(shapley.ShapleyResult, type)


def test_top_level_re_exports():
    import gsax

    assert callable(gsax.analyze)
    assert callable(gsax.analyze_hdmr)
    assert callable(gsax.analyze_pce)
    assert callable(gsax.analyze_shapley)
    assert callable(gsax.emulate_hdmr)
    assert callable(gsax.emulate_pce)
    assert callable(gsax.sample)
    assert callable(gsax.load)


def test_re_exports_are_same_objects():
    import gsax
    from gsax.hdmr import analyze as hdmr_analyze
    from gsax.pce import analyze as pce_analyze
    from gsax.shapley import analyze as shapley_analyze
    from gsax.sobol import analyze as sobol_analyze

    assert gsax.analyze is sobol_analyze
    assert gsax.analyze_hdmr is hdmr_analyze
    assert gsax.analyze_pce is pce_analyze
    assert gsax.analyze_shapley is shapley_analyze


def test_dgsm_re_exports():
    import gsax

    assert callable(gsax.analyze_dgsm)
    assert isinstance(gsax.DGSMResult, type)


def test_efast_re_exports():
    import gsax

    assert callable(gsax.analyze_efast)
    assert callable(gsax.sample_efast)
    assert isinstance(gsax.EFASTResult, type)


def test_sampling_re_exports():
    import gsax

    assert callable(gsax.sample_mc)


def test_problem_re_exports():
    import gsax

    assert isinstance(gsax.GaussianInputSpec, type)
    assert isinstance(gsax.UniformInputSpec, type)
