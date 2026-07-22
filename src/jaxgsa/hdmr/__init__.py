"""RS-HDMR (Random Sampling High-Dimensional Model Representation).

B-spline surrogate decomposition with ANCOVA-based sensitivity indices.

Example::

    from jaxgsa import hdmr

    result = hdmr.analyze(problem, X, Y)
    Y_pred = result.predict(X_new)
"""

from jaxgsa.hdmr._analyze import analyze_hdmr as analyze
from jaxgsa.hdmr._result import HDMRResult

__all__ = ["HDMRResult", "analyze"]
