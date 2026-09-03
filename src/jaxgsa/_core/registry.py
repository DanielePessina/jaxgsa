"""Where a sensitivity method declares what it is.

Nothing in this module is public API. See :mod:`jaxgsa._core`.

A method declares itself once, in its own package, and registers that record
here: its capabilities (whether it tolerates correlated or categorical
inputs), its exports, and its non-finite unit. Consumers — error messages
that name alternatives, the documentation tables, the test suite — read the
registry instead of keeping their own lists, which is what stops those lists
drifting apart.

Registration happens as an import side effect: importing ``jaxgsa`` imports
all thirteen method packages, and each one calls :func:`register` at module
level. That is why this module imports nothing from those packages — the
dependency runs one way only, which is what makes a central record possible
without an import cycle.

The declarations are not taken on trust. ``tests/test_registry.py`` checks
every package on disk is registered, and checks each claim against what the
method actually does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from jaxgsa._core.invalid import InvalidUnit

Capability = Literal["accepts", "refuses"]
"""Whether a method tolerates a kind of input, or raises on it."""


@dataclass(frozen=True)
class MethodSpec:
    """What one sensitivity method is, as data.

    Attributes:
        name: The namespace name, matching the package directory and the entry
            in ``jaxgsa.__all__``.
        analyze: The public analysis entry point. Every method both exports
            and defines this as ``analyze``, so a grep finds all thirteen;
            reading it from here is still the way to enumerate them without
            knowing the package names.
        sample: The design generator for a design-based method, or ``None``
            for a method that works on data you already have.
        result: The result class the analysis returns.
        correlation: What the method does with a problem that declares a
            correlation structure.
        categorical: What the method does with a problem that has categorical
            parameters.
        bootstrap: The keyword that asks for bootstrap confidence intervals,
            or ``None`` when the method reports no intervals. There is one
            spelling, ``n_bootstrap``; the field stays because it also records
            *whether* a method offers intervals at all, which is not derivable
            from the name.
        pure_core: Whether the package exports a traceable ``indices()``.
            ``True`` for eleven methods. ``False`` for ``kucherenko`` and
            ``vkoga``, which are host NumPy and SciPy end to end and so have
            no traceable core to export. Recorded here rather than left as an
            absence so the exemption is a declaration that gets checked, not
            a gap someone has to notice.
        invalid_unit: The block of data the method treats as indivisible when
            dropping non-finite values, or ``None`` when the method delegates
            to a backend and has no unit of its own. Dropping half a Saltelli
            group would leave the estimator reading misaligned rows, so this
            is a property of the design, not of the caller's rows.
    """

    name: str
    analyze: Callable[..., Any]
    sample: Callable[..., Any] | None
    result: type
    correlation: Capability
    categorical: Capability
    bootstrap: str | None
    invalid_unit: InvalidUnit | None
    pure_core: bool = True

    @property
    def is_design_based(self) -> bool:
        """Whether the method needs its own design rather than given data."""
        return self.sample is not None


_REGISTRY: dict[str, MethodSpec] = {}


def register(spec: MethodSpec) -> MethodSpec:
    """Add one method's declaration to the registry.

    Called at import time from each method package's ``__init__.py``.

    Args:
        spec: The method's declaration.

    Returns:
        The same spec, so the call site can bind it to a module-level name in
        one statement.

    Raises:
        ValueError: If a method of that name is already registered. Two
            packages claiming one name would leave which of them the registry
            describes up to import order.
    """
    if spec.name in _REGISTRY and _REGISTRY[spec.name] is not spec:
        raise ValueError(
            f"jaxgsa._core.registry: {spec.name!r} is already registered. "
            "Two declarations for one name would make the registry depend on "
            "import order."
        )
    _REGISTRY[spec.name] = spec
    return spec


def methods() -> Mapping[str, MethodSpec]:
    """Return every registered method, keyed by namespace name.

    Importing :mod:`jaxgsa` registers all of them, so callers inside the
    package see a complete registry.

    Returns:
        A read-only view, sorted by name. Read-only because a consumer that
        edited it would change what every other consumer sees.
    """
    return MappingProxyType(dict(sorted(_REGISTRY.items())))
