"""Shapley-effect sensitivity analysis.

Shapley effects are computed from fitted PCE or HDMR results:

Example::

    result = pce.analyze(problem, X, Y).shapley()
"""

from gsax.shapley._result import ShapleyResult

__all__ = ["ShapleyResult"]
