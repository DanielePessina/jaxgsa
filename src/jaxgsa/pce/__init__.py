"""Polynomial Chaos Expansion (PCE) sensitivity analysis.

The surrogate is an expansion in orthogonal polynomials. Sobol indices come
analytically from the expansion coefficients (Sudret, 2008), so no extra
model evaluations are needed.

Example::

    from jaxgsa import pce

    result = pce.analyze(problem, X, Y)
    Y_pred = result.predict(X_new)
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.pce._analyze import analyze_pce as analyze
from jaxgsa.pce._result import PCEResult

__all__ = ["PCEResult", "analyze"]

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
        bootstrap=None,
        invalid_unit=InvalidUnit.ROW,
    )
)
