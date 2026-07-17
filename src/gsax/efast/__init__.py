"""Extended Fourier Amplitude Sensitivity Test (eFAST).

Frequency-based variance decomposition that computes first-order and
total-order Sobol indices from Fourier amplitudes along sinusoidal
search curves.

Example::

    from gsax import efast

    samples = efast.sample(problem, n_per_curve=4096, seed=42)
    Y = model(samples.samples)
    result = efast.analyze(samples, Y)
"""

from gsax.efast._analyze import analyze
from gsax.efast._result import EFASTResult
from gsax.efast._sampling import EFASTSamples, sample

__all__ = ["EFASTResult", "EFASTSamples", "analyze", "sample"]
