"""Tier T4 (internal consistency): every jaxgsa warning carries a category.

Decision D21 gives the package one warning class, ``JaxgsaWarning``. A warning
that does not pass ``category=`` falls back to ``UserWarning``, and a user can
then no longer filter jaxgsa warnings apart from NumPy, SciPy or JAX ones.

These tests walk the source with the ``ast`` module, not a regular expression,
so a new module cannot drift back to an uncategorised warning.
"""

from __future__ import annotations

import ast
import pathlib

import jaxgsa
from jaxgsa import JaxgsaWarning

SRC_ROOT = pathlib.Path(jaxgsa.__file__).parent


def _warn_calls(tree: ast.AST) -> list[ast.Call]:
    """Collect every ``warnings.warn(...)`` call in a parsed module.

    Args:
        tree: The parsed module.

    Returns:
        The ``warnings.warn`` call nodes, in source order.
    """
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "warn"
            and isinstance(func.value, ast.Name)
            and func.value.id == "warnings"
        ):
            calls.append(node)
    return calls


def _uncategorised() -> list[str]:
    """Find every warning in the package that passes no category.

    A category can arrive as the second positional argument or as the
    ``category`` keyword. Anything else leaves the warning as a plain
    ``UserWarning``.

    Returns:
        One ``path:line`` string for each offending call site.
    """
    offenders = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        source = path.read_text()
        if "warnings.warn" not in source:
            continue
        for call in _warn_calls(ast.parse(source)):
            has_keyword = any(kw.arg == "category" for kw in call.keywords)
            has_positional = len(call.args) >= 2
            if not (has_keyword or has_positional):
                rel = path.relative_to(SRC_ROOT.parent)
                offenders.append(f"{rel}:{call.lineno}")
    return offenders


def test_every_warning_passes_a_category():
    """Tier T4 (internal consistency): no warning in the package is uncategorised.

    The check is deliberately permissive: it asks that a category is passed,
    not that the category is ``JaxgsaWarning``. A module may one day raise a
    real ``DeprecationWarning`` or ``RuntimeWarning``, and that is correct
    behaviour, not a defect. What must never happen is a warning with no
    category at all, because that silently becomes ``UserWarning`` and can no
    longer be filtered apart from NumPy, SciPy or JAX warnings.
    """
    offenders = _uncategorised()
    assert offenders == [], (
        "these warnings pass no category, so they fall back to UserWarning: "
        + ", ".join(offenders)
    )


def test_the_walk_finds_the_warnings_it_claims_to_check():
    """Tier T4 (internal consistency): guard the guard.

    An empty walk would pass the test above by accident, so this test pins a
    floor on how many ``warnings.warn`` calls the walk actually finds.
    """
    total = 0
    for path in SRC_ROOT.rglob("*.py"):
        source = path.read_text()
        if "warnings.warn" in source:
            total += len(_warn_calls(ast.parse(source)))
    assert total >= 30


def test_jaxgsa_warning_is_a_user_warning_subclass():
    """Tier T4 (internal consistency): ``JaxgsaWarning`` is a distinct subclass.

    Existing ``pytest.warns(UserWarning)`` assertions must keep passing, and a
    user must still be able to filter jaxgsa warnings on their own.
    """
    assert issubclass(JaxgsaWarning, UserWarning)
    assert JaxgsaWarning is not UserWarning
