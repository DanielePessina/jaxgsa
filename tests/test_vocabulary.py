"""The public interface uses one word for one concept.

``CONTEXT.md`` states the vocabulary the 1.0 interface freezes. This module
reads it back off the method registry and off the real signatures, so a
signature that drifts from the vocabulary fails here rather than shipping.

Tier T4 (internal consistency) throughout. There is no external oracle for a
naming convention; what these tests prove is that thirteen methods agree with
each other and with the written specification.

Why the registry and not a hard-coded list of method names: a hard-coded list
silently stops covering a method the day someone adds one. The registry is
what ``jaxgsa`` itself uses to enumerate methods, so a new package is covered
the moment it registers.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal, get_args, get_origin

import pytest

from jaxgsa._core.registry import MethodSpec, methods

# Every batching keyword the vocabulary allows, and the unit each one counts.
# A method takes the axes that apply to it and no others; a keyword that does
# nothing is worse than an absent one.
BATCHING_AXES = {
    "batch_size": "sample rows",
    "slice_chunk_size": "output slices",
    "resample_chunk_size": "bootstrap replicates",
}

# Spellings that lost. Each was in use before the freeze; none may come back.
RETIRED_NAMES = {
    "num_resamples": "n_bootstrap",
    "seed": "key",
    "chunk_size": "slice_chunk_size or resample_chunk_size",
    "n_resamples": "n_bootstrap",
    "num_bootstrap": "n_bootstrap",
    "random_state": "key",
    "rng": "key",
}

# No method bootstraps by default. Borgonovo's bias correction does need
# replicates, but a non-zero default plus a required key would make the
# plainest possible call an error, so it warns instead. Kept as an empty set
# rather than deleted: if a future method wants an on-by-default interval,
# this is where the exception gets argued for in writing.
BOOTSTRAP_ON_BY_DEFAULT: set[str] = set()

# DGSM differentiates a callable, so it cannot take (problem, X, Y) like the
# other given-data methods. The vocabulary documents this; the test allows it
# by name so that a *second* exception cannot appear unnoticed.
POSITIONAL_EXCEPTIONS = {"dgsm"}

# Rules the vocabulary states that the code does not satisfy yet, each with the
# reason and the work that closes it.
#
# These become ``xfail(strict=True)`` markers on the individual parametrised
# cases, so when the work lands the case passes, the strict marker turns that
# into a failure, and the exemption has to be deleted. An imperative
# ``pytest.xfail()`` inside the test body would NOT do this -- it aborts the
# test before the assertion runs, so it can never notice the gap closing, and
# the exemption would outlive the problem.
BUDGET_GAPS: dict[str, str] = {
    # Empty, and kept rather than deleted for the same reason as
    # BOOTSTRAP_ON_BY_DEFAULT: if a future method cannot derive a width from
    # the budget, this is where that exception gets argued for in writing.
    #
    # Both original entries closed in the same round. sobol gained a
    # bytes-per-slice model covering the point and bootstrap paths and both
    # estimator orders; morris's resolver was already a bytes model, so only
    # its 2048 default needed to become None. The strict markers are what
    # forced the deletions: a non-strict xfail would have sat here passing.
}


def _budget_params(specs: list[MethodSpec]) -> list[Any]:
    """Parametrise the budget rule, marking the methods that cannot meet it yet."""
    out: list[Any] = []
    for spec in specs:
        reason = BUDGET_GAPS.get(spec.name)
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        out.append(pytest.param(spec, marks=marks, id=spec.name))
    return out


def _params(spec: MethodSpec) -> dict[str, inspect.Parameter]:
    """Return the analysis entry point's parameters, keyed by name.

    ``eval_str=True`` matters. Most modules here use
    ``from __future__ import annotations``, so their annotations arrive as
    strings and ``get_origin`` on a string returns ``None`` — a check for
    ``Literal`` would fail on the modules that postpone and pass on the ones
    that do not, which tests the import style rather than the signature.
    """
    return dict(inspect.signature(spec.analyze, eval_str=True).parameters)


def _all_methods() -> list[MethodSpec]:
    """Return every registered method, in a stable order."""
    return list(methods().values())


def _ids(specs: list[MethodSpec]) -> list[str]:
    """Return pytest ids matching a list of specs."""
    return [s.name for s in specs]


ALL = _all_methods()


@pytest.mark.parametrize("spec", ALL, ids=_ids(ALL))
def test_no_retired_spelling_survives(spec: MethodSpec) -> None:
    """No method uses a keyword the vocabulary retired.

    This is the test that would have caught the split the freeze exists to
    fix: five methods said ``seed`` while two said ``key``, and two said
    ``num_resamples`` while three said ``n_bootstrap``.
    """
    found = sorted(set(_params(spec)) & set(RETIRED_NAMES))
    assert not found, (
        f"{spec.name}.analyze uses retired keyword(s) {found}. "
        f"Use instead: {', '.join(RETIRED_NAMES[name] for name in found)}. "
        "See CONTEXT.md."
    )


@pytest.mark.parametrize("spec", ALL, ids=_ids(ALL))
def test_registry_bootstrap_keyword_matches_the_signature(spec: MethodSpec) -> None:
    """``MethodSpec.bootstrap`` names a keyword the method really takes.

    The registry is how a caller finds the bootstrap keyword without knowing
    the method. A stale declaration there is worse than none, because it sends
    the caller to a keyword that raises.
    """
    if spec.bootstrap is None:
        assert "n_bootstrap" not in _params(spec), (
            f"{spec.name} declares bootstrap=None but its signature takes "
            "n_bootstrap. One of the two is wrong."
        )
        return

    assert spec.bootstrap == "n_bootstrap", (
        f"{spec.name} declares bootstrap={spec.bootstrap!r}. The vocabulary "
        "allows one spelling, 'n_bootstrap'."
    )
    assert spec.bootstrap in _params(spec), (
        f"{spec.name} declares bootstrap={spec.bootstrap!r} but its signature "
        "has no such parameter."
    )


BOOTSTRAPPERS = [s for s in ALL if s.bootstrap is not None]


@pytest.mark.parametrize("spec", BOOTSTRAPPERS, ids=_ids(BOOTSTRAPPERS))
def test_a_bootstrapper_offers_the_whole_interval_vocabulary(spec: MethodSpec) -> None:
    """Offering ``n_bootstrap`` means offering the four keywords that go with it.

    Half an interval interface is the worse failure: a caller who can ask for
    replicates but cannot choose the endpoint rule, or cannot pass a key, has
    to read the source to find out which method they are holding.
    """
    params = _params(spec)
    for required in ("conf_level", "ci_method", "key", "keep_replicates"):
        assert required in params, (
            f"{spec.name} takes n_bootstrap but not {required!r}. "
            "See the interval table in CONTEXT.md."
        )


@pytest.mark.parametrize("spec", BOOTSTRAPPERS, ids=_ids(BOOTSTRAPPERS))
def test_bootstrap_defaults_are_uniform(spec: MethodSpec) -> None:
    """``n_bootstrap`` defaults to 0, ``conf_level`` to 0.95, ``key`` to None.

    A surrogate-backed method must not bootstrap by default: each replicate
    refits a surrogate, so an on-by-default interval would make a routine call
    an order of magnitude slower for a caller who never asked for one.
    """
    params = _params(spec)

    if spec.name in BOOTSTRAP_ON_BY_DEFAULT:
        assert params["n_bootstrap"].default > 0, (
            f"{spec.name} is listed as bootstrapping by default but defaults "
            "to 0. Update the list or the default."
        )
    else:
        assert params["n_bootstrap"].default == 0, (
            f"{spec.name}.n_bootstrap defaults to "
            f"{params['n_bootstrap'].default!r}, not 0. If that is deliberate, "
            "add it to BOOTSTRAP_ON_BY_DEFAULT with the reason."
        )

    assert params["conf_level"].default == 0.95
    assert params["key"].default is None


@pytest.mark.parametrize("spec", BOOTSTRAPPERS, ids=_ids(BOOTSTRAPPERS))
def test_ci_method_offers_the_same_two_choices_everywhere(spec: MethodSpec) -> None:
    """``ci_method`` is ``Literal["quantile", "gaussian"]``, defaulting to quantile.

    Annotated rather than merely documented, so a caller's type checker
    rejects a third value before it reaches a runtime branch.
    """
    param = _params(spec)["ci_method"]
    assert param.default == "quantile", (
        f"{spec.name}.ci_method defaults to {param.default!r}, not 'quantile'."
    )
    annotation = param.annotation
    assert get_origin(annotation) is Literal, (
        f"{spec.name}.ci_method is annotated {annotation!r}; the vocabulary "
        'requires Literal["quantile", "gaussian"].'
    )
    assert set(get_args(annotation)) == {"quantile", "gaussian"}, (
        f"{spec.name}.ci_method allows {get_args(annotation)}, not exactly "
        "('quantile', 'gaussian')."
    )


@pytest.mark.parametrize("spec", BOOTSTRAPPERS, ids=_ids(BOOTSTRAPPERS))
def test_keep_replicates_is_keyword_only_and_last(spec: MethodSpec) -> None:
    """``keep_replicates`` sits in one place in every signature.

    It was in three different positions before the freeze. Position matters
    even for a keyword-only argument, because the signature is what a reader
    scans and what the rendered docs show.

    "Last" means last of the named parameters. A ``**kwargs`` catch-all is
    excluded because Python requires it to come last syntactically, so a
    method that forwards to a backend cannot put anything after it.
    ``shapley`` is the one such method: it passes its fit arguments through to
    ``pce`` or ``hdmr`` rather than naming them, so that a keyword only one
    backend understands cannot be accepted and silently ignored by the other.
    Reading the rule as "last named parameter" keeps the position fixed for
    every method and stays satisfiable for that one. CONTEXT.md states the
    rule without this qualification.
    """
    params = [p for p in _params(spec).values() if p.kind is not inspect.Parameter.VAR_KEYWORD]
    names = [p.name for p in params]
    assert names[-1] == "keep_replicates", (
        f"{spec.name}.analyze ends with {names[-1]!r}; keep_replicates must be "
        f"the last parameter. Current tail: {names[-3:]}"
    )
    assert params[-1].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[-1].default is False


@pytest.mark.parametrize("spec", ALL, ids=_ids(ALL))
def test_only_the_three_batching_axes_appear(spec: MethodSpec) -> None:
    """A batching keyword is one of exactly three, each counting a stated unit.

    The three are orthogonal: rows, output slices, bootstrap replicates. A
    fourth name means either a duplicate of one of these, or an axis nobody
    wrote down.
    """
    suspicious = {
        name
        for name in _params(spec)
        if ("batch" in name or "chunk" in name) and name not in BATCHING_AXES
    }
    assert not suspicious, (
        f"{spec.name}.analyze takes batching keyword(s) {sorted(suspicious)}, "
        f"which are not in the vocabulary. Allowed: {sorted(BATCHING_AXES)}. "
        "See the three-axis table in CONTEXT.md."
    )


@pytest.mark.parametrize("spec", _budget_params(ALL))
def test_a_batching_axis_accepts_none_to_mean_use_the_budget(spec: MethodSpec) -> None:
    """``None`` means "derive this from the memory budget".

    A hard-coded element budget is a defect: it ignores
    ``jaxgsa.config.set_memory_budget`` and so cannot be tuned for the machine
    the analysis is actually running on.
    """
    params = _params(spec)
    for axis in BATCHING_AXES:
        if axis not in params:
            continue
        assert params[axis].default is None, (
            f"{spec.name}.{axis} defaults to {params[axis].default!r}. The "
            "vocabulary requires None, meaning 'derive one from "
            "get_memory_budget()'."
        )


@pytest.mark.parametrize("spec", ALL, ids=_ids(ALL))
def test_the_first_argument_says_what_kind_of_method_this_is(spec: MethodSpec) -> None:
    """A design-based method takes ``sampling_result``; a given-data one takes ``problem``.

    The first parameter is the fastest way to tell the two families apart, so
    it is worth having it never lie.
    """
    if spec.name in POSITIONAL_EXCEPTIONS:
        pytest.skip(f"{spec.name} differentiates a callable; see CONTEXT.md")

    first = next(iter(_params(spec)))
    expected = "sampling_result" if spec.is_design_based else "problem"
    assert first == expected, (
        f"{spec.name}.analyze starts with {first!r}; a "
        f"{'design-based' if spec.is_design_based else 'given-data'} method "
        f"starts with {expected!r}."
    )


@pytest.mark.parametrize("spec", ALL, ids=_ids(ALL))
def test_the_entry_point_is_called_analyze(spec: MethodSpec) -> None:
    """The function is *defined* as ``analyze``, not merely exported as one.

    Three methods used to define ``analyze_pce``, ``analyze_hdmr`` and
    ``analyze_vkoga`` and alias them on import. The public name was right, but
    ``grep "def analyze("`` under-reported the number of entry points, which
    caused a real miss during an audit of all thirteen.
    """
    assert spec.analyze.__name__ == "analyze", (
        f"{spec.name} exports analyze but the function is defined as "
        f"{spec.analyze.__name__!r}. Rename the def; keep the export."
    )


def test_every_registered_method_is_covered_here() -> None:
    """The registry has all thirteen, so these rules apply to all thirteen.

    Without this, a method that failed to register would be silently exempt
    from every rule above — the parametrisation would just generate one fewer
    case and still pass.
    """
    assert len(ALL) == 13, f"expected 13 registered methods, found {len(ALL)}"
