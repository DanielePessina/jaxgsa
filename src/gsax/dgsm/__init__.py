"""Derivative-based Global Sensitivity Measures (DGSM).

DGSM ranks inputs by the mean squared partial derivative
``nu_i = E[(df/dx_i)^2]`` averaged over the input distribution — cheap
to obtain via reverse-mode autodiff when the model is written in JAX.
A Poincare inequality turns ``nu_i`` into an upper bound on the total
Sobol index ``ST_i``, and the Kucherenko-Song inequality turns the mean
derivative ``E[df/dx_i]`` into a lower bound, so one derivative sample
brackets ST at a fraction of the cost of a Sobol design.

Example::

    from gsax import dgsm
    from gsax.sampling import monte_carlo

    X = monte_carlo(problem, n=10000, seed=42)
    result = dgsm.analyze(problem, fn, jnp.asarray(X))
"""

from gsax.dgsm._analyze import analyze
from gsax.dgsm._poincare import axis_constants, poincare_constant
from gsax.dgsm._result import DGSMResult

__all__ = ["DGSMResult", "analyze", "axis_constants", "poincare_constant"]
