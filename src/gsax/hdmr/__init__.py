"""RS-HDMR (Random Sampling High-Dimensional Model Representation).

B-spline surrogate decomposition with ANCOVA-based sensitivity indices.

Example::

    from gsax import hdmr

    result = hdmr.analyze(problem, X, Y)
    Y_pred = hdmr.emulate(result, X_new)
"""

from gsax.hdmr._analyze import analyze_hdmr as analyze
from gsax.hdmr._analyze import emulate_hdmr as emulate
from gsax.hdmr._result import HDMREmulator, HDMRResult

__all__ = ["HDMREmulator", "HDMRResult", "analyze", "emulate"]
