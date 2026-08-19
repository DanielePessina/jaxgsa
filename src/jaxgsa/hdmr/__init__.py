"""RS-HDMR (Random Sampling High-Dimensional Model Representation).

B-spline surrogate decomposition with ANCOVA-based sensitivity indices.

Example::

    from jaxgsa import hdmr

    result = hdmr.analyze(problem, X, Y)
    Y_pred = result.predict(X_new)
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.hdmr._analyze import analyze
from jaxgsa.hdmr._result import HDMRResult

__all__ = ["HDMRResult", "analyze"]

SPEC = register(
    MethodSpec(
        name="hdmr",
        analyze=analyze,
        sample=None,
        result=HDMRResult,
        # Accepted, but ST is then the SCSA total rather than a Sobol
        # total-order index, and analyze warns about the reinterpretation.
        correlation="accepts",
        categorical="refuses",
        bootstrap=None,
        invalid_unit=InvalidUnit.ROW,
    )
)
