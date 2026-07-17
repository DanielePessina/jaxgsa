"""Polynomial Chaos Expansion (PCE) sensitivity analysis.

Orthogonal polynomial surrogate with analytical Sobol indices from
the expansion coefficients (Sudret, 2008).

Example::

    from gsax import pce

    result = pce.analyze(problem, X, Y)
    Y_pred = result.predict(X_new)
"""

from gsax.pce._analyze import analyze_pce as analyze
from gsax.pce._result import PCEResult

__all__ = ["PCEResult", "analyze"]
