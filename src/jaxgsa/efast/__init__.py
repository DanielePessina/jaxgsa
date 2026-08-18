"""Extended Fourier Amplitude Sensitivity Test (eFAST).

Frequency-based variance decomposition that computes first-order and
total-order Sobol indices from Fourier amplitudes along sinusoidal
search curves.

Example::

    from jaxgsa import efast

    samples = efast.sample(problem, n_per_curve=4096, seed=42)
    Y = model(samples.samples)
    result = efast.analyze(samples, Y)
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.efast._analyze import analyze
from jaxgsa.efast._result import EFASTResult
from jaxgsa.efast._sampling import EFASTSamples, sample

__all__ = ["EFASTResult", "EFASTSamples", "analyze", "sample"]

SPEC = register(
    MethodSpec(
        name="efast",
        analyze=analyze,
        sample=sample,
        result=EFASTResult,
        correlation="refuses",
        categorical="refuses",
        bootstrap=None,
        # A search curve is an ordered sweep read by a discrete Fourier
        # transform, so it cannot be dropped at all: removing a point does not
        # shrink the sample, it changes what the estimator computes.
        invalid_unit=InvalidUnit.CURVE,
    )
)
