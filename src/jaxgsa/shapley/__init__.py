"""Shapley-effect sensitivity analysis.

Shapley effects are computed from fitted PCE or HDMR results:

Example::

    result = pce.analyze(problem, X, Y).shapley()

``analyze`` is a thin convenience that fits the surrogate and calls
``.shapley()`` in one step::

    result = shapley.analyze(problem, X, Y, backend="pce")
"""

from jaxgsa.shapley._analyze import analyze
from jaxgsa.shapley._result import ShapleyResult

__all__ = ["ShapleyResult", "analyze"]
