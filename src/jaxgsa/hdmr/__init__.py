"""RS-HDMR (Random Sampling High-Dimensional Model Representation).

B-spline surrogate decomposition with ANCOVA-based sensitivity indices.

Example::

    from jaxgsa import hdmr

    result = hdmr.analyze(problem, X, Y)
    Y_pred = result.predict(X_new)

``hdmr.analyze`` checks its data and reports diagnostics, so it cannot be
traced. ``hdmr.indices`` runs the same fit and returns ``(Sa, Sb, S, ST)`` as
plain arrays, so it composes with ``jit``, ``vmap`` and ``jacfwd``.
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.hdmr._analyze import analyze, indices
from jaxgsa.hdmr._result import HDMRResult

__all__ = ["HDMRResult", "analyze", "indices"]

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
        bootstrap="n_bootstrap",
        invalid_unit=InvalidUnit.ROW,
    )
)
