"""Derivative-based Global Sensitivity Measures (DGSM).

Computes sensitivity measures from partial derivatives of the model
via reverse-mode autodiff, and derives Poincare upper bounds and
Kucherenko-Song lower bounds on total Sobol indices.

Example::

    from gsax import dgsm
    from gsax.sampling import sample_mc

    X = sample_mc(problem, N=10000, seed=42)
    result = dgsm.analyze(problem, fn, jnp.asarray(X))
"""

from gsax.dgsm._analyze import analyze
from gsax.dgsm._poincare import axis_constants, poincare_constant
from gsax.dgsm._result import DGSMResult

__all__ = ["DGSMResult", "analyze", "axis_constants", "poincare_constant"]
