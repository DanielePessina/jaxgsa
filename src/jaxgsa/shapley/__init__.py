"""Shapley-effect sensitivity analysis.

A fitted PCE or HDMR result computes its own Shapley effects:

Example::

    result = pce.analyze(problem, X, Y).shapley()

``analyze`` is a thin convenience that fits the surrogate and calls
``.shapley()`` in one step::

    result = shapley.analyze(problem, X, Y, backend="pce")

``shapley.indices`` is the traceable form: same fit, same allocation, three
plain arrays and no diagnostics.
"""

from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.shapley._analyze import analyze, indices
from jaxgsa.shapley._result import ShapleyResult

__all__ = ["ShapleyResult", "analyze", "indices"]

SPEC = register(
    MethodSpec(
        name="shapley",
        analyze=analyze,
        sample=None,
        result=ShapleyResult,
        # Refused on the default pce backend. backend="hdmr" with
        # include_correlative=True does accept a correlated problem, so this
        # records the default; the backend argument overrides it.
        correlation="refuses",
        categorical="refuses",
        bootstrap="n_bootstrap",
        # No unit of its own: it forwards on_invalid to the backend, which
        # applies its own.
        invalid_unit=None,
    )
)
