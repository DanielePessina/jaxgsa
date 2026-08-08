"""Shapley-effect sensitivity analysis.

A fitted PCE or HDMR result computes its own Shapley effects:

Example::

    result = pce.analyze(problem, X, Y).shapley()

``analyze`` is a thin convenience that fits the surrogate and calls
``.shapley()`` in one step::

    result = shapley.analyze(problem, X, Y, backend="pce")
"""

from jaxgsa.shapley._analyze import analyze
from jaxgsa.shapley._result import ShapleyResult

__all__ = ["ShapleyResult", "analyze"]
