"""Small, registry-driven checks for the public API vocabulary.

These tests protect the package-wide contract without expanding one assertion
into a separate test for every method and every spelling.
"""

from __future__ import annotations

import inspect

import jaxgsa
from jaxgsa._core.registry import methods

RETIRED_NAMES = {
    "num_resamples",
    "seed",
    "chunk_size",
    "n_resamples",
    "num_bootstrap",
    "random_state",
    "rng",
    "standardize",
    "prenormalize",
    "correlation_kind",
    "kind",
}
BATCHING_NAMES = {"batch_size", "slice_chunk_size", "resample_chunk_size"}


def _params(fn) -> dict[str, inspect.Parameter]:
    """Read a postponed-annotation signature by parameter name."""
    return dict(inspect.signature(fn, eval_str=True).parameters)


def test_registry_is_complete_and_exported() -> None:
    """Every registered method has a package-root namespace."""
    registered = set(methods())
    assert len(registered) >= 13
    assert registered <= set(jaxgsa.__all__)
    assert all(getattr(jaxgsa, name, None) is not None for name in registered)


def test_analyze_entrypoints_share_the_basic_contract() -> None:
    """Every analysis entry point exposes the same observability controls."""
    for name, spec in methods().items():
        params = _params(spec.analyze)
        assert params["verbose"].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert params["verbose"].default is True, name
        assert "on_invalid" in params, name


def test_bootstrap_entrypoints_share_the_interval_contract() -> None:
    """Methods offering bootstrap intervals expose all interval controls."""
    for name, spec in methods().items():
        params = _params(spec.analyze)
        if spec.bootstrap is None:
            assert "n_bootstrap" not in params, name
            continue
        assert spec.bootstrap == "n_bootstrap", name
        for required in ("n_bootstrap", "conf_level", "ci_method", "key", "keep_replicates"):
            assert required in params, (name, required)
        assert params["n_bootstrap"].default == 0, name


def test_retired_names_and_unknown_batching_axes_are_absent() -> None:
    """The public vocabulary has one spelling per cross-cutting concept."""
    for name, spec in methods().items():
        params = _params(spec.analyze)
        assert not (set(params) & RETIRED_NAMES), name
        suspicious = {
            parameter
            for parameter in params
            if ("batch" in parameter or "chunk" in parameter) and parameter not in BATCHING_NAMES
        }
        assert not suspicious, (name, sorted(suspicious))


def test_method_entrypoint_shape_is_consistent() -> None:
    """Design builders expose sample(); given-data methods do not."""
    for name, spec in methods().items():
        assert (spec.sample is not None) == spec.is_design_based, name
        first = next(iter(_params(spec.analyze)))
        if name == "dgsm":
            continue
        expected = "sampling_result" if spec.is_design_based else "problem"
        assert first == expected, (name, first, expected)


def test_problem_correlation_surface_uses_one_keyword() -> None:
    """The three Problem constructors agree on the correlation scale name."""
    signatures = (
        inspect.signature(jaxgsa.Problem.__init__),
        inspect.signature(jaxgsa.Problem.from_dict),
        inspect.signature(jaxgsa.Problem.with_correlation),
    )
    assert all("correlation_type" in signature.parameters for signature in signatures)
    assert all("correlation_kind" not in signature.parameters for signature in signatures)
