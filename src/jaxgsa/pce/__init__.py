"""Polynomial Chaos Expansion (PCE) sensitivity analysis.

The surrogate is an expansion in orthogonal polynomials. Sobol indices come
analytically from the expansion coefficients (Sudret, 2008), so no extra
model evaluations are needed.

Example::

    from jaxgsa import pce

    result = pce.analyze(problem, X, Y)
    Y_pred = result.predict(X_new)
"""

from jaxgsa.pce._analyze import analyze_pce as analyze
from jaxgsa.pce._result import PCEResult

__all__ = ["PCEResult", "analyze"]
