"""The public interface uses one word for one concept.

This module pins the vocabulary the 1.0 interface freezes: one spelling per
concept across every method, the four keywords that go with ``n_bootstrap``,
``verbose`` on every entry point, the batching axis names, and the first
argument that says which family a method belongs to. It reads the rules back
off the method registry and off the real signatures, so a signature that
drifts from the vocabulary fails here rather than shipping.

Tier T4 (internal consistency) throughout. There is no external oracle for a
naming convention; what these tests prove is that thirteen methods agree with
each other and with the written specification.

Why the registry and not a hard-coded list of method names: a hard-coded list
silently stops covering a method the day someone adds one. The registry is
what ``jaxgsa`` itself uses to enumerate methods, so a new package is covered
the moment it registers.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from collections.abc import Callable
from typing import Any, Literal, get_args, get_origin

import jax
import jax.numpy as jnp
import pytest

import jaxgsa
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

# Spellings the freeze retired from analyze() and sample() alike. Output
# standardization has one spelling: `standardize` (optimal_transport) and
# `prenormalize` (hsic, long removed) lost to `standardize_outputs`. The
# correlation-scale keyword is `correlation_type` on every surface, so
# neither `correlation_kind` nor a bare `kind` may appear as a parameter.
RETIRED_EVERYWHERE = {
    "standardize": "standardize_outputs",
    "prenormalize": "standardize_outputs",
    "correlation_kind": "correlation_type",
    "kind": "correlation_type",
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
        f"Use instead: {', '.join(RETIRED_NAMES[name] for name in found)}."
    )


def _entry_points(spec: MethodSpec) -> list[tuple[str, dict[str, inspect.Parameter]]]:
    """Return the public entry points a spec exposes, with their parameters.

    ``analyze`` always; ``sample`` when the method is design-based. Both are
    read with ``eval_str=True`` for the reason ``_params`` documents.
    """
    out = [("analyze", _params(spec))]
    if spec.sample is not None:
        out.append(("sample", dict(inspect.signature(spec.sample, eval_str=True).parameters)))
    return out


@pytest.mark.parametrize("spec", ALL, ids=_ids(ALL))
def test_no_retired_spelling_in_analyze_or_sample(spec: MethodSpec) -> None:
    """Neither entry point uses a standardization or correlation spelling that lost.

    ``optimal_transport`` said ``standardize`` and ``hsic`` once said
    ``prenormalize`` for the same concept; ``Problem`` said both
    ``correlation_kind`` and ``kind`` for one concept. One spelling each:
    ``standardize_outputs`` and ``correlation_type``.
    """
    for fn_name, params in _entry_points(spec):
        found = sorted(set(params) & set(RETIRED_EVERYWHERE))
        assert not found, (
            f"{spec.name}.{fn_name} uses retired keyword(s) {found}. "
            f"Use instead: {', '.join(RETIRED_EVERYWHERE[name] for name in found)}."
        )


def test_no_retired_spelling_on_the_problem_surfaces() -> None:
    """The Problem correlation surfaces use ``correlation_type``, nothing else.

    The registry walk above never sees ``Problem``, yet its three surfaces
    are the only ones that ever carried ``correlation_kind`` and ``kind``.
    Without this check, re-introducing either there would pass the suite.
    """
    from jaxgsa import Problem

    surfaces = {
        "Problem.__init__": inspect.signature(Problem.__init__),
        "Problem.from_dict": inspect.signature(Problem.from_dict),
        "Problem.with_correlation": inspect.signature(Problem.with_correlation),
    }
    for name, sig in surfaces.items():
        found = sorted(set(sig.parameters) & set(RETIRED_EVERYWHERE))
        assert not found, (
            f"{name} uses retired keyword(s) {found}. "
            f"Use instead: {', '.join(RETIRED_EVERYWHERE[k] for k in found)}."
        )
    assert "correlation_type" in surfaces["Problem.with_correlation"].parameters


@pytest.mark.parametrize("spec", ALL, ids=_ids(ALL))
def test_output_standardization_is_spelled_standardize_outputs(spec: MethodSpec) -> None:
    """A flag that standardizes the outputs is spelled ``standardize_outputs``.

    The rule catches near-misses the retired list cannot enumerate: any
    parameter whose name contains ``standardize`` must be exactly
    ``standardize_outputs``, on ``analyze`` and ``sample`` alike.
    """
    for fn_name, params in _entry_points(spec):
        offenders = sorted(
            name for name in params if "standardize" in name and name != "standardize_outputs"
        )
        assert not offenders, (
            f"{spec.name}.{fn_name} spells an output-standardization flag "
            f"{offenders}; the vocabulary allows one spelling, "
            "'standardize_outputs'."
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
            "A method that offers replicates offers all four of conf_level, "
            "ci_method, key and keep_replicates."
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
    every method and stays satisfiable for that one. The plain rule, that
    ``keep_replicates`` comes last, is stated without this qualification.
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
def test_every_entry_point_takes_verbose_defaulting_to_true(spec: MethodSpec) -> None:
    """Every ``analyze()`` and every ``sample()`` takes ``verbose: bool = True``.

    One keyword toggles the observability of the whole workflow: the problem
    summary, the timings, the top-k results on ``analyze()``, and the design
    narration on ``sample()``. Default ``True`` on all seventeen entry
    points, so a first-time user sees what happened without reading anything,
    and ``verbose=False`` is one spelling everywhere.
    """
    for fn_name, params in _entry_points(spec):
        assert "verbose" in params, (
            f"{spec.name}.{fn_name} takes no 'verbose'. Every entry point "
            "offers the same observability toggle, keyword-only and "
            "defaulting to True."
        )
        param = params["verbose"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{spec.name}.{fn_name}.verbose must be keyword-only, is {param.kind!r}."
        )
        assert param.annotation is bool, (
            f"{spec.name}.{fn_name}.verbose is annotated {param.annotation!r}, not bool."
        )
        assert param.default is True, (
            f"{spec.name}.{fn_name}.verbose defaults to {param.default!r}, not True."
        )


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
        f"which are not in the vocabulary. Allowed: {sorted(BATCHING_AXES)}, "
        "one name per batching axis."
    )


# ---------------------------------------------------------------------------
# The budget rule, made checkable end to end.
#
# For every (method, batching axis) pair, name the resolver whose call is what
# turns None into a width, and a tiny end-to-end run that reaches it. The test
# spies on the resolver in the module that calls it and asserts the axis's
# None default really arrives there as None. A companion test asserts each
# resolver family derives its width from ``get_memory_budget()``. Together
# they make the docstring's claim true, rather than only checking the default
# is spelled None.
# ---------------------------------------------------------------------------

_PROBLEM = jaxgsa.Problem.from_dict({"x1": (-1.0, 1.0), "x2": (-1.0, 1.0), "x3": (-1.0, 1.0)})


def _multi(X: jax.Array) -> jax.Array:
    """Tiny non-additive model with two output columns, so slices exist."""
    y = X[:, 0] + 2.0 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]
    return jnp.stack([y, 2.0 * y + 1.0], axis=-1)


def _given_data(n: int = 64) -> tuple[jax.Array, jax.Array]:
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(_PROBLEM, n, seed=3))
    return X, _multi(X)


def _run_sobol() -> None:
    sr = jaxgsa.sobol.sample(_PROBLEM, n_samples=64, seed=3)
    jaxgsa.sobol.analyze(sr, _multi(jnp.asarray(sr.samples)))


def _run_morris() -> None:
    sr = jaxgsa.morris.sample(_PROBLEM, n_trajectories=10, seed=3)
    # The resample resolver only runs on the bootstrap path.
    jaxgsa.morris.analyze(
        sr, _multi(jnp.asarray(sr.samples)), n_bootstrap=8, key=jax.random.key(3)
    )


def _run_efast() -> None:
    sr = jaxgsa.efast.sample(_PROBLEM, n_per_curve=257, seed=3)
    jaxgsa.efast.analyze(sr, _multi(jnp.asarray(sr.samples)))


def _run_pawn() -> None:
    X, Y = _given_data()
    jaxgsa.pawn.analyze(_PROBLEM, X, Y, n_bins=4)


def _run_borgonovo() -> None:
    X, Y = _given_data()
    jaxgsa.borgonovo.analyze(_PROBLEM, X, Y)


def _run_ot() -> None:
    X, Y = _given_data()
    jaxgsa.optimal_transport.analyze(_PROBLEM, X, Y, n_partitions=4)


def _run_dgsm() -> None:
    X, _ = _given_data()
    jaxgsa.dgsm.analyze(_PROBLEM, lambda x: _multi(x[None, :])[0], X)


def _run_pce() -> None:
    X, Y = _given_data()
    # The PCE row-batch resolver lives on the streamed path, which the None
    # default only takes when the single-pass residents exceed the budget. A
    # tiny budget is exactly the semantic under test: None must follow it.
    jaxgsa.config.set_memory_budget(4, unit="kib")
    jaxgsa.pce.analyze(_PROBLEM, X, Y, order=2)


def _run_hdmr() -> None:
    X, Y = _given_data(n=384)  # RS-HDMR refuses fewer than 300 rows
    jaxgsa.hdmr.analyze(_PROBLEM, X, Y, maxorder=1, maxiter=5)


def _run_vkoga() -> None:
    X, Y = _given_data()
    jaxgsa.vkoga.analyze(
        _PROBLEM,
        X,
        Y,
        max_centers=8,
        n_folds=3,
        n_outer=16,
        n_inner=8,
        n_variance=64,
        key=jax.random.key(3),
    )


# method -> axis -> (module with the resolver binding, resolver attribute,
# the resolver parameter that carries the caller's value). The spy asserts
# that parameter arrives as None when the analyze() keyword is left at its
# default.
BUDGET_RESOLVERS: dict[str, dict[str, tuple[str, str, str]]] = {
    "sobol": {
        "slice_chunk_size": (
            "jaxgsa.sobol._analyze",
            "resolve_point_chunk_size",
            "slice_chunk_size",
        ),
    },
    "morris": {
        "resample_chunk_size": (
            "jaxgsa.morris._analyze",
            "_resolve_resample_chunk_size",
            "resample_chunk_size",
        ),
    },
    "efast": {
        "slice_chunk_size": ("jaxgsa.efast._analyze", "resolve_batch_size", "batch_size"),
    },
    "pawn": {
        "slice_chunk_size": (
            "jaxgsa.pawn._analyze",
            "_resolve_slice_chunk_size",
            "slice_chunk_size",
        ),
    },
    "borgonovo": {
        "slice_chunk_size": ("jaxgsa.borgonovo._analyze", "resolve_batch_size", "batch_size"),
    },
    "optimal_transport": {
        "slice_chunk_size": (
            "jaxgsa.optimal_transport._analyze",
            "resolve_batch_size",
            "batch_size",
        ),
    },
    "dgsm": {
        "batch_size": ("jaxgsa.dgsm._core", "resolve_batch_size", "batch_size"),
    },
    "pce": {
        "batch_size": ("jaxgsa.pce._analyze", "resolve_batch_size", "batch_size"),
    },
    "hdmr": {
        "batch_size": ("jaxgsa.hdmr._fit", "resolve_batch_size", "batch_size"),
        "slice_chunk_size": (
            "jaxgsa.hdmr._fit",
            "_resolve_slice_chunk_size",
            "slice_chunk_size",
        ),
    },
    "vkoga": {
        "batch_size": ("jaxgsa.vkoga._indices", "resolve_batch_size", "batch_size"),
    },
}

BUDGET_RUNNERS: dict[str, Callable[[], None]] = {
    "sobol": _run_sobol,
    "morris": _run_morris,
    "efast": _run_efast,
    "pawn": _run_pawn,
    "borgonovo": _run_borgonovo,
    "optimal_transport": _run_ot,
    "dgsm": _run_dgsm,
    "pce": _run_pce,
    "hdmr": _run_hdmr,
    "vkoga": _run_vkoga,
}


@pytest.fixture
def _restore_memory_budget():
    """Snapshot and restore the process-global memory budget around a test."""
    from jaxgsa._core import batching

    saved = batching._memory_budget_bytes
    yield
    batching._memory_budget_bytes = saved


def test_the_resolver_table_covers_every_batching_axis() -> None:
    """Every (method, batching axis) pair in a signature is in the table.

    Without this, a new method (or a new axis on an old one) would silently
    skip the budget test below: the parametrisation would generate the same
    cases and still pass. A table entry has to be written — with the runner
    that reaches the resolver — before the axis exists.
    """
    for spec in ALL:
        in_signature = {axis for axis in BATCHING_AXES if axis in _params(spec)}
        in_table = set(BUDGET_RESOLVERS.get(spec.name, {}))
        assert in_signature == in_table, (
            f"{spec.name} takes batching axes {sorted(in_signature)} but "
            f"BUDGET_RESOLVERS covers {sorted(in_table)}. Add (or remove) the "
            "entry, with the resolver the axis routes through and a runner "
            "that reaches it."
        )
    for name in BUDGET_RESOLVERS:
        assert name in BUDGET_RUNNERS, f"BUDGET_RESOLVERS names {name} but no runner exists"


@pytest.mark.parametrize("spec", _budget_params(ALL))
def test_a_batching_axis_accepts_none_to_mean_use_the_budget(
    spec: MethodSpec, monkeypatch, _restore_memory_budget
) -> None:
    """``None`` means "derive this from the memory budget" — and really does.

    A hard-coded element budget is a defect: it ignores
    ``jaxgsa.config.set_memory_budget`` and so cannot be tuned for the machine
    the analysis is actually running on.

    Two claims per axis. The default is spelled ``None``, and the default
    actually reaches the method's budget-deriving resolver as ``None``: the
    resolver in ``BUDGET_RESOLVERS`` is spied on in the module that calls it,
    a tiny end-to-end run is made with the axis left at its default, and the
    spy must record a call whose caller-value parameter is ``None`` and whose
    result is a positive int. ``test_each_resolver_family_reads_the_budget``
    closes the loop by checking the resolvers themselves shrink with the
    budget.
    """
    params = _params(spec)
    axes = [axis for axis in BATCHING_AXES if axis in params]
    for axis in axes:
        assert params[axis].default is None, (
            f"{spec.name}.{axis} defaults to {params[axis].default!r}. The "
            "vocabulary requires None, meaning 'derive one from "
            "get_memory_budget()'."
        )
    if not axes:
        return

    calls: dict[str, list[tuple[Any, int]]] = {axis: [] for axis in axes}
    for axis in axes:
        module_path, attr, param_name = BUDGET_RESOLVERS[spec.name][axis]
        module = importlib.import_module(module_path)
        real = getattr(module, attr)
        record = calls[axis]

        def wrapper(*args, _real=real, _param=param_name, _record=record, **kwargs):
            bound = inspect.signature(_real).bind(*args, **kwargs)
            out = _real(*args, **kwargs)
            _record.append((bound.arguments[_param], out))
            return out

        monkeypatch.setattr(module, attr, wrapper)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        BUDGET_RUNNERS[spec.name]()

    for axis in axes:
        assert calls[axis], (
            f"{spec.name}: the run never reached the resolver for {axis}; the "
            "BUDGET_RESOLVERS entry or its runner is stale."
        )
        none_calls = [(req, out) for req, out in calls[axis] if req is None]
        assert none_calls, (
            f"{spec.name}.{axis}: the None default never arrived at the "
            f"resolver as None; recorded caller values "
            f"{[req for req, _ in calls[axis]]}. Somewhere a literal width "
            "replaces the budget-derived one."
        )
        for _, resolved in none_calls:
            assert isinstance(resolved, int) and resolved >= 1


def test_each_resolver_family_reads_the_budget(_restore_memory_budget) -> None:
    """Every resolver the table names shrinks its width when the budget does.

    The spy test above proves ``None`` reaches these functions; this one
    proves the functions answer from ``get_memory_budget()``. Each probe
    resolves a width for ``None`` on a workload big enough that the default
    budget allows more than one unit, then again under a 1-byte budget, and
    the width must shrink to the floor of 1.
    """
    from jaxgsa._core.batching import resolve_batch_size
    from jaxgsa.hdmr import _fit as hdmr_fit
    from jaxgsa.morris import _analyze as morris_analyze
    from jaxgsa.pawn import _analyze as pawn_analyze
    from jaxgsa.sobol import _bootstrap as sobol_bootstrap
    from jaxgsa.sobol import _chunking as sobol_chunking

    probes: dict[str, Callable[[], int]] = {
        "resolve_batch_size": lambda: resolve_batch_size(1024, 10_000, None),
        "sobol.point": lambda: sobol_chunking.resolve_point_chunk_size(None, 64, 256, 3, True, 4),
        "sobol.bootstrap": lambda: sobol_bootstrap._resolve_slice_chunk_size(
            None, 64, 16, 256, 3, True, 4
        ),
        "morris.resample": lambda: morris_analyze._resolve_resample_chunk_size(None, 64, 1024),
        "pawn.slice": lambda: pawn_analyze._resolve_slice_chunk_size(None, 256, 3, 4, 4),
        "hdmr.slice": lambda: hdmr_fit._resolve_slice_chunk_size(None, 64, 128, 1024),
    }
    for name, probe in probes.items():
        jaxgsa.config.set_memory_budget(512, unit="mib")
        wide = probe()
        jaxgsa.config.set_memory_budget(1, unit="b")
        narrow = probe()
        assert narrow == 1 < wide, (
            f"{name}: resolved width did not follow the budget "
            f"(512 MiB -> {wide}, 1 B -> {narrow}). The resolver must derive "
            "its None default from jaxgsa.config.get_memory_budget()."
        )


def test_explicit_widths_win_over_the_budget(_restore_memory_budget) -> None:
    """An explicit width is honoured even when the budget would pick less.

    The vocabulary's third rule: the memory budget only sizes the ``None``
    default; it never caps a width the caller chose. Each probe asks for 8
    under a 1-byte budget (which resolves to 1 when left to the budget, per
    the test above) and must get exactly 8 back.
    """
    from jaxgsa._core.batching import resolve_batch_size
    from jaxgsa.hdmr import _fit as hdmr_fit
    from jaxgsa.morris import _analyze as morris_analyze
    from jaxgsa.pawn import _analyze as pawn_analyze
    from jaxgsa.sobol import _bootstrap as sobol_bootstrap
    from jaxgsa.sobol import _chunking as sobol_chunking

    probes: dict[str, Callable[[], int]] = {
        "resolve_batch_size": lambda: resolve_batch_size(1024, 10_000, 8),
        "sobol.point": lambda: sobol_chunking.resolve_point_chunk_size(8, 64, 256, 3, True, 4),
        "sobol.bootstrap": lambda: sobol_bootstrap._resolve_slice_chunk_size(
            8, 64, 16, 256, 3, True, 4
        ),
        "morris.resample": lambda: morris_analyze._resolve_resample_chunk_size(8, 64, 1024),
        "pawn.slice": lambda: pawn_analyze._resolve_slice_chunk_size(8, 256, 3, 4, 4),
        "hdmr.slice": lambda: hdmr_fit._resolve_slice_chunk_size(8, 64, 128, 1024),
    }
    jaxgsa.config.set_memory_budget(1, unit="b")
    for name, probe in probes.items():
        got = probe()
        assert got == 8, (
            f"{name}: an explicit width of 8 came back as {got} under a "
            "1-byte budget. Explicit chunk widths must win over the budget."
        )


@pytest.mark.parametrize("spec", ALL, ids=_ids(ALL))
def test_the_first_argument_says_what_kind_of_method_this_is(spec: MethodSpec) -> None:
    """A design-based method takes ``sampling_result``; a given-data one takes ``problem``.

    The first parameter is the fastest way to tell the two families apart, so
    it is worth having it never lie.
    """
    if spec.name in POSITIONAL_EXCEPTIONS:
        pytest.skip(f"{spec.name} differentiates a callable, so it takes the model first")

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
