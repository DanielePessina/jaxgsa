"""Derivative-based Global Sensitivity Measures (DGSM).

DGSM ranks inputs by the mean squared partial derivative
``nu_i = E[(df/dx_i)^2]``, averaged over the input distribution. Reverse-mode
autodiff gives it cheaply when the model is written in JAX.

Two inequalities turn the derivative moments into a bracket on the total
Sobol index ``ST_i``. A Poincare inequality turns ``nu_i`` into an upper
bound. The Kucherenko-Song inequality turns the mean derivative
``E[df/dx_i]`` into a lower bound. One derivative sample therefore brackets
ST at a fraction of the cost of a Sobol design.

Example::

    from jaxgsa import dgsm
    from jaxgsa.sampling import monte_carlo

    X = monte_carlo(problem, n=10000, seed=42)
    result = dgsm.analyze(problem, fn, jnp.asarray(X))
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.dgsm._analyze import analyze
from jaxgsa.dgsm._poincare import axis_constants, poincare_constant
from jaxgsa.dgsm._result import DGSMResult

__all__ = ["DGSMResult", "analyze", "axis_constants", "poincare_constant"]

SPEC = register(
    MethodSpec(
        name="dgsm",
        analyze=analyze,
        # Not design-based: it takes any X, or a precomputed Jacobian.
        sample=None,
        result=DGSMResult,
        # A derivative with respect to a level code has no meaning, and the
        # Poincare bound assumes a product measure.
        correlation="refuses",
        categorical="refuses",
        bootstrap=None,
        invalid_unit=InvalidUnit.ROW,
    )
)
