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


def _discover_method_packages() -> set[str]:
    """Find every method package by walking the source tree."""
    return {
        p.name
        for p in SRC_ROOT.iterdir()
        if p.is_dir() and p.name not in NOT_A_METHOD and (p / "_analyze.py").exists()
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

    X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 64, seed=0))
    if spec.name == "dgsm":
        return lambda: spec.analyze(problem, _point_model, X)
    return lambda: spec.analyze(problem, X, _batch_model(X))


def _gate_fires(call: Callable[[], object], topic: str) -> bool:
    """Whether the capability gate for ``topic`` refused this problem.

    Any other failure means the gate let the problem through and something
    later objected, which is the answer we want for an "accepts" claim. Only
    a refusal naming the topic counts as the gate firing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            call()
        except ValueError as e:
            return topic in str(e).lower()
        except Exception:  # noqa: BLE001 - anything else means the gate passed
            return False
    return False


ALL_SPECS = sorted(methods().values(), key=lambda s: s.name)
SPEC_IDS = [s.name for s in ALL_SPECS]


class TestCompleteness:
    """A fourteenth method cannot be left out of the registry."""

    def test_every_method_package_on_disk_is_registered(self):
        discovered = _discover_method_packages()
        registered = set(methods())
        assert discovered == registered, (
            f"not registered: {sorted(discovered - registered)}; "
            f"registered but no package: {sorted(registered - discovered)}"
        )


class TestTheSpecsAreTrue:
    """Every claim is answerable to the code."""

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_the_correlation_claim_matches_behaviour(self, spec):
        fired = _gate_fires(_invoke(spec, CORRELATED), "correlat")
        assert fired == (spec.correlation == "refuses"), (
            f"{spec.name} declares correlation={spec.correlation!r} but the gate "
            f"{'fired' if fired else 'did not fire'}"
        )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_the_categorical_claim_matches_behaviour(self, spec):
        fired = _gate_fires(_invoke(spec, CATEGORICAL), "categorical")
        assert fired == (spec.categorical == "refuses"), (
            f"{spec.name} declares categorical={spec.categorical!r} but the gate "
            f"{'fired' if fired else 'did not fire'}"
        )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_the_bootstrap_keyword_exists_when_claimed(self, spec):
        import inspect

        params = inspect.signature(spec.analyze).parameters
        if spec.bootstrap is None:
            assert not ({"num_resamples", "n_bootstrap"} & set(params)), (
                f"{spec.name} declares no bootstrap but its signature offers one"
            )
        else:
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
