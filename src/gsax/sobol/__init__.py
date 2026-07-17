"""Sobol variance-based sensitivity analysis (Saltelli sampling).

Example::

    from gsax import Problem, sobol

    problem = Problem.from_dict({"x1": (0, 1), "x2": (0, 1)})
    sr = sobol.sample(problem, n_samples=4096, seed=42)
    Y = model(sr.samples)
    result = sobol.analyze(sr, Y)
"""

from gsax.sobol._analyze import analyze
from gsax.sobol._result import SobolResult
from gsax.sobol._sampling import SobolSamples, sample

__all__ = ["SobolResult", "SobolSamples", "analyze", "sample"]
