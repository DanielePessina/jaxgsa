"""Polynomial Chaos Expansion (PCE) sensitivity analysis.

Orthogonal polynomial surrogate with analytical Sobol indices from
the expansion coefficients (Sudret, 2008).

Example::

    from jaxgsa import pce

    result = pce.analyze(problem, X, Y)
    Y_pred = result.predict(X_new)
"""

from jaxgsa.pce._analyze import analyze_pce as analyze
from jaxgsa.pce._result import PCEResult

__all__ = ["PCEResult", "analyze"]
