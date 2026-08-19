"""Tests for the method registry in ``jaxgsa._core.registry``.

Two jobs, and the second is the one that matters.

**Completeness.** Every method package on disk must be registered. This is a
filesystem walk, not a list, so a fourteenth method cannot be quietly left
out. The pattern comes from ``tests/test_warning_category.py``, the one
cross-cutting test in this suite that could not drift. Before the registry,
``tests/test_imports.py`` enumerated eleven of the thirteen methods by hand,
having silently missed ``kucherenko`` and ``vkoga``.

**Truthfulness.** Every capability a spec claims is checked against what the
method actually does. A registry that merely restates a hand-written table is
a second copy of that table, and it would drift the same way. These tests make
the declaration answerable to the code.

Tier T4 (behavioural contract). No external oracle: these pin the registry
against the library's own behaviour, not against any numerical result.
"""

from __future__ import annotations

import pathlib
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa._core.registry import MethodSpec, methods

SRC_ROOT = pathlib.Path(jaxgsa.__file__).parent
NOT_A_METHOD = {"_core", "benchmarks", "__pycache__"}

D = 3
PLAIN = jaxgsa.Problem(("a", "b", "c"), ((0.0, 1.0),) * D)
_R = np.eye(D)
_R[0, 1] = _R[1, 0] = 0.5
CORRELATED = PLAIN.with_correlation(_R)
CATEGORICAL = jaxgsa.Problem.from_dict(
    {
        "a": jaxgsa.CategoricalInputSpec(dist="categorical", probs=[0.5, 0.5]),
        "b": jaxgsa.UniformInputSpec(dist="uniform", low=0.0, high=1.0),
        "c": jaxgsa.UniformInputSpec(dist="uniform", low=0.0, high=1.0),
    }
)

# Enough rows for each design-based sampler to get past its own size checks,
# so a refusal we observe is the capability gate and not a size complaint.
# eFAST needs 4 * M**2 * (D - 1) + 1 rows per curve.
DESIGN_SIZE = {"sobol": 64, "morris": 8, "efast": 400, "kucherenko": 64}

# The same idea for the given-data methods: enough rows that a refusal is the
# capability gate and not a size complaint. 64 rows are enough for all but
# hdmr, whose regression needs 300.
GIVEN_DATA_SIZE = {"hdmr": 320}
NEEDS_A_KEY = ("hsic", "vkoga")
"""Methods whose randomness is not optional, so ``key`` has no default.

The gate has to be reached, and a missing required keyword would refuse the
call before it ever got there.
"""
DEFAULT_GIVEN_DATA_SIZE = 64

MIN_METHODS = 13
"""The count at the time this test was written. Methods are only ever added."""


def _discover_method_packages() -> set[str]:
    """Find every method package by walking the source tree.

    A method package is any subpackage of ``jaxgsa`` that is not on the
    not-a-method list. The walk used to look for a file named ``_analyze.py``,
    which made the completeness assertion pass for free for any package that
    named its analysis module something else. Nothing but the directory name
    is read now.

    Returns:
        The package names found on disk.
    """
    return {
        p.name
        for p in SRC_ROOT.iterdir()
        if p.is_dir() and p.name not in NOT_A_METHOD and (p / "__init__.py").exists()
    }


def _point_model(x):
    """One sample in, one scalar out. The dgsm calling convention."""
    return jnp.sin(x[0]) + x[1] ** 2 + 0.1 * x[2]


def _batch_model(X):
    """A whole design in, one output per row. Every other method."""
    return jnp.sin(X[:, 0]) + X[:, 1] ** 2 + 0.1 * X[:, 2]


def _invoke(spec: MethodSpec, problem: jaxgsa.Problem) -> Callable[[], object]:
    """Return a callable that puts ``problem`` through ``spec``'s gate.

    Three calling conventions exist and the registry does not model them, so
    they are spelled out here. That divergence is itself the subject of a
    later change; this test only has to reach the gate.
    """
    sampler = spec.sample
    if sampler is not None:
        n = DESIGN_SIZE[spec.name]
        # efast.sample and kucherenko.sample take no `verbose`.
        kw = {} if spec.name in ("efast", "kucherenko") else {"verbose": False}
        return lambda: sampler(problem, n, **kw)

    n = GIVEN_DATA_SIZE.get(spec.name, DEFAULT_GIVEN_DATA_SIZE)
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, n, seed=0))
    if spec.name == "dgsm":
        return lambda: spec.analyze(problem, _point_model, X)
    extra = {"key": jax.random.key(0)} if spec.name in NEEDS_A_KEY else {}
    return lambda: spec.analyze(problem, X, _batch_model(X), **extra)


_UPDATE_INVOKE = (
    "Update _invoke (and DESIGN_SIZE / GIVEN_DATA_SIZE) in tests/test_registry.py "
    "so the call reaches this method's capability gate."
)


def _gate_fires(spec: MethodSpec, call: Callable[[], object], topic: str) -> bool:
    """Whether the capability gate for ``topic`` refused this problem.

    A call that completes proves an ``"accepts"`` claim. A ``ValueError``
    naming the topic proves a ``"refuses"`` claim. Nothing else is an answer,
    so nothing else is tolerated.

    The earlier version read every other exception as "the gate passed". A
    method with one extra required keyword raised ``TypeError`` before it
    reached any gate, and both truthfulness tests then passed while
    exercising nothing. Only a ``"refuses"`` claim could ever fail.

    Args:
        spec: The method under test, named in the failure message.
        call: A zero-argument callable that runs the method.
        topic: The lowercase substring a refusal must name.

    Returns:
        True if the gate refused, False if the call ran to completion.

    Raises:
        AssertionError: If the call fails for any other reason. The call has
            then proved nothing, so the test must not read it as a verdict.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            call()
        except ValueError as e:
            if topic in str(e).lower():
                return True
            raise AssertionError(
                f"{spec.name}: the call reached no verdict on {topic!r}. It raised "
                f"ValueError({e}), which names no capability. {_UPDATE_INVOKE}"
            ) from e
        except Exception as e:
            raise AssertionError(
                f"{spec.name}: the call reached no verdict on {topic!r}. It raised "
                f"{type(e).__name__}({e}) before any capability gate ran, so an "
                f"'accepts' claim is unproven. {_UPDATE_INVOKE}"
            ) from e
    return False


ALL_SPECS = sorted(methods().values(), key=lambda s: s.name)
SPEC_IDS = [s.name for s in ALL_SPECS]


class TestCompleteness:
    """A fourteenth method cannot be left out of the registry."""

    def test_every_method_package_on_disk_is_registered(self):
        discovered = _discover_method_packages()
        registered = set(methods())
        assert len(discovered) >= MIN_METHODS, (
            f"the walk found only {len(discovered)} method packages under {SRC_ROOT}; "
            "an empty or near-empty walk makes this assertion pass for free"
        )
        assert discovered == registered, (
            f"not registered: {sorted(discovered - registered)}; "
            f"registered but no package: {sorted(registered - discovered)}"
        )

    def test_every_registered_method_is_exported_by_name(self):
        """``MethodSpec.name`` promises the entry in ``jaxgsa.__all__``.

        Only a docs job enforced this before. A method that registers under
        one name and exports under another breaks every message that points a
        caller at ``jaxgsa.<name>``.
        """
        exported = set(jaxgsa.__all__)
        missing = sorted(set(methods()) - exported)
        assert not missing, (
            f"registered but not in jaxgsa.__all__: {missing}. Import the package in "
            "src/jaxgsa/__init__.py and add its name to __all__."
        )
        for name in methods():
            assert getattr(jaxgsa, name, None) is not None, (
                f"jaxgsa.__all__ names {name!r} but jaxgsa has no such attribute"
            )


class TestTheSpecsAreTrue:
    """Every claim is answerable to the code."""

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_the_correlation_claim_matches_behaviour(self, spec):
        fired = _gate_fires(spec, _invoke(spec, CORRELATED), "correlat")
        assert fired == (spec.correlation == "refuses"), (
            f"{spec.name} declares correlation={spec.correlation!r} but the gate "
            f"{'fired' if fired else 'did not fire'}"
        )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_the_categorical_claim_matches_behaviour(self, spec):
        fired = _gate_fires(spec, _invoke(spec, CATEGORICAL), "categorical")
        assert fired == (spec.categorical == "refuses"), (
            f"{spec.name} declares categorical={spec.categorical!r} but the gate "
            f"{'fired' if fired else 'did not fire'}"
        )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_the_bootstrap_keyword_exists_when_claimed(self, spec):
        import inspect

        params = inspect.signature(spec.analyze).parameters
        if spec.bootstrap is None:
            # Only one spelling is legal, so check for that one. Checking a
            # set of two would keep passing for a method that regressed to
            # the retired name, which is the failure this guards against.
            assert "n_bootstrap" not in params, (
                f"{spec.name} declares no bootstrap but its signature offers one"
            )
        else:
            assert spec.bootstrap == "n_bootstrap", (
                f"{spec.name} declares bootstrap={spec.bootstrap!r}; the only "
                "legal spelling is 'n_bootstrap' (see CONTEXT.md)"
            )
            assert spec.bootstrap in params


class TestTheRegistryIsProtected:
    def test_a_duplicate_name_is_refused(self):
        """Two declarations for one name would depend on import order."""
        from jaxgsa._core.registry import register

        existing = methods()["sobol"]
        clash = MethodSpec(
            name="sobol",
            analyze=existing.analyze,
            sample=None,
            result=existing.result,
            correlation="accepts",
            categorical="accepts",
            bootstrap=None,
            invalid_unit=None,
        )
        with pytest.raises(ValueError, match="already registered"):
            register(clash)

    def test_registering_the_same_spec_again_is_harmless(self):
        """Re-importing a module must not explode."""
        from jaxgsa._core.registry import register

        assert register(methods()["sobol"]) is methods()["sobol"]
