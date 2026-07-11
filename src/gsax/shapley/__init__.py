"""Shapley-effect sensitivity analysis.

Global Shapley effects (Owen 2014; Song, Nelson & Staum 2016) computed
analytically from a fitted surrogate's variance decomposition -- RS-HDMR
component functions or PCE coefficients -- with no permutation sampling.
Assumes independent inputs.

Example::

    from gsax import shapley

    result = shapley.analyze(problem, X, Y)
"""

from gsax.shapley._analyze import analyze_shapley as analyze
from gsax.shapley._result import ShapleyResult

__all__ = ["ShapleyResult", "analyze"]
