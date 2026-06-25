"""Polynomial Chaos Expansion (PCE) sensitivity analysis.

Orthogonal polynomial surrogate with analytical Sobol indices from
the expansion coefficients (Sudret, 2008).

Example::

    from gsax.expansions import pce

    result = pce.analyze(problem, X, Y)
    Y_pred = pce.emulate(result, X_new)
"""

from gsax.expansions.pce._analyze import analyze_pce as analyze
from gsax.expansions.pce._analyze import emulate_pce as emulate
from gsax.expansions.pce._result import PCEResult

__all__ = ["PCEResult", "analyze", "emulate"]
