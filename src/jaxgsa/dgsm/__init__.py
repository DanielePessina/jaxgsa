"""Derivative-based Global Sensitivity Measures (DGSM).

DGSM ranks inputs by the mean squared partial derivative
``nu_i = E[(df/dx_i)^2]``, averaged over the input distribution. Reverse-mode
autodiff gives it cheaply when the model is written in JAX.

A Poincare inequality turns ``nu_i`` into an upper bound on the total Sobol
index ``ST_i``, for every marginal this package supports. One derivative
sample therefore caps ST at a fraction of the cost of a Sobol design, and a
cap near zero settles the input.

``lower_bound`` reports ``Var(x_i) * E[df/dx_i]^2 / Var(Y)``. Kucherenko &
Song (2016) prove this is a floor under ``ST_i`` for a Gaussian marginal, and
only there. On a uniform or truncated marginal it is an estimate, exact for a
response linear in that input and able to overshoot the true ``ST_i`` for a
curved one. Read the upper bound as a proof and the lower one as a guide.

Example::

    from jaxgsa import dgsm
    from jaxgsa.sampling import monte_carlo

    X = monte_carlo(problem, n=10000, seed=42)
    result = dgsm.analyze(problem, fn, X)
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.dgsm._analyze import analyze, indices
from jaxgsa.dgsm._poincare import axis_constants, poincare_constant
from jaxgsa.dgsm._result import DGSMResult

__all__ = ["DGSMResult", "analyze", "axis_constants", "indices", "poincare_constant"]

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
        bootstrap="n_bootstrap",
        invalid_unit=InvalidUnit.ROW,
    )
)
