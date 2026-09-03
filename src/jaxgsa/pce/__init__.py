"""Polynomial Chaos Expansion (PCE) sensitivity analysis.

The surrogate is an expansion in orthogonal polynomials. Sobol indices come
analytically from the expansion coefficients (Sudret, 2008), so no extra
model evaluations are needed.

Example::

    from jaxgsa import pce

    result = pce.analyze(problem, X, Y)
    Y_pred = result.predict(X_new)

``pce.analyze`` checks its data and reports diagnostics, so it cannot be
traced. ``pce.indices`` runs the same fit and returns the indices as plain
arrays, so it composes with ``jit``, ``vmap`` and ``jacrev``.
``pce.effective_order`` reports the degree a fit will actually use.
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.pce._analyze import analyze, effective_order, indices
from jaxgsa.pce._result import PCEResult

__all__ = ["PCEResult", "analyze", "effective_order", "indices"]

SPEC = register(
    MethodSpec(
        name="pce",
        analyze=analyze,
        sample=None,
        result=PCEResult,
        # The polynomial basis is orthogonal with respect to a product
        # measure, so dependence breaks the decomposition.
        correlation="refuses",
        categorical="refuses",
        bootstrap="n_bootstrap",
        invalid_unit=InvalidUnit.ROW,
    )
)
