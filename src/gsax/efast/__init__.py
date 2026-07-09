"""Extended Fourier Amplitude Sensitivity Test (eFAST).

Frequency-based variance decomposition that computes first-order and
total-order Sobol indices from Fourier amplitudes along sinusoidal
search curves.

Example::

    from gsax import efast

    X = efast.sample(problem, N=4096, seed=42)
    Y = model(X)
    result = efast.analyze(problem, Y)
"""

from gsax.efast._analyze import analyze
from gsax.efast._result import EFASTResult
from gsax.efast._sampling import sample

__all__ = ["EFASTResult", "analyze", "sample"]
