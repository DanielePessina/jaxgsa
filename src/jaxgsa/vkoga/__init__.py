"""Correlated variance-based sensitivity analysis via a VKOGA surrogate.

This is a given-data method. It fits a Vectorial Kernel Orthogonal Greedy
Algorithm (VKOGA) surrogate to the ``(X, Y)`` you already have. It then
computes variance-based sensitivity indices that stay meaningful when the
inputs are dependent, using a Gaussian copula for the dependency structure.

Unlike every sampling-based method in this package, the indices here do not
assume independent inputs. They follow Li et al. (2010). The familiar main and
total indices keep their formulas but change connotation, and they split into
correlated and uncorrelated parts.

Gating: ``analyze`` accepts a declared ``problem.correlation``. It refuses
categorical parameters. The isotropic kernel needs a continuous CDF map on
every coordinate, and an unordered level code has none. Use
:mod:`jaxgsa.optimal_transport` or :mod:`jaxgsa.borgonovo` for categorical
inputs.

Example::

    import jax

    import jaxgsa

    problem = jaxgsa.Problem(names=("x1", "x2", "x3"), bounds=[(0, 1)] * 3)
    X = jaxgsa.sampling.monte_carlo(problem, 2048)   # independent, space-filling
    Y = model(X)

    # Declare the dependence on the problem. Fit it from observed data with
    # jaxgsa.sampling.fit_correlation(problem, X_data) when you have no
    # published matrix.
    problem = problem.with_correlation(R)

    result = jaxgsa.vkoga.analyze(problem, X, Y, key=jax.random.key(0))
    result.S_TC   # total correlated: rank parameters to measure
    result.S_TU   # total uncorrelated: rank parameters to fix
    Y_pred = result.predict(X_new)

Use float64. The kernel solve squares the condition number of the cross kernel,
and single precision cannot carry that for small shape parameters. Call
``jax.config.update("jax_enable_x64", True)`` before fitting.

References:
    Hilhorst, Quicken, van de Vosse & Huberts (2024). Int. J. Numer. Meth.
        Biomed. Engng. 40(2):e3797.
    Li, Rabitz, Yelvington et al. (2010). J. Phys. Chem. A 114:6022-6032.
    Wirtz & Haasdonk (2013). Dolomites Res. Notes Approx. 6:83-100.
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.vkoga._analyze import analyze
from jaxgsa.vkoga._result import VKOGAResult

__all__ = ["VKOGAResult", "analyze"]

SPEC = register(
    MethodSpec(
        name="vkoga",
        analyze=analyze,
        sample=None,
        result=VKOGAResult,
        correlation="accepts",
        categorical="refuses",
        bootstrap=None,
        invalid_unit=InvalidUnit.ROW,
    )
)
