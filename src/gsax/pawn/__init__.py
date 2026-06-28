"""PAWN sensitivity analysis (Pianosi & Wagener, 2015).

Computes distribution-based sensitivity indices using the
Kolmogorov-Smirnov distance between unconditional and conditional
output CDFs.

Example::

    from gsax import pawn
    from gsax.sampling import sample_mc

    X = sample_mc(problem, N=5000, seed=42)
    Y = model(X)
    result = pawn.analyze(problem, X, Y)
"""

from gsax.pawn._analyze import analyze
from gsax.pawn._result import PAWNResult

__all__ = ["PAWNResult", "analyze"]
