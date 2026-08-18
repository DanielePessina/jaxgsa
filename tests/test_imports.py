"""Tests for the namespace-only public API.

What the registry can express — which method namespaces exist, and what each
one exports — is proved generically in ``tests/test_registry.py``. This file
keeps only the claims the registry cannot make: that the support namespaces
are reachable from the root, and that the removed root shortcuts stay removed.
A name that must *not* exist has no registry entry to derive it from.
"""

import jaxgsa


def test_root_exports_the_support_namespaces():
    assert jaxgsa.config is not None
    assert jaxgsa.sampling is not None
    assert callable(jaxgsa.sampling.monte_carlo)


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
