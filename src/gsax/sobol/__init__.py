"""Sobol variance-based sensitivity analysis (Saltelli sampling).

Example::

    from gsax.sobol import analyze
    from gsax import sample, Problem

    problem = Problem.from_dict({"x1": (0, 1), "x2": (0, 1)})
    sr = sample(problem, n_samples=4096, seed=42)
    Y = model(sr.samples)
    result = analyze(sr, Y)
"""

from gsax.sobol._analyze import analyze
from gsax.sobol._result import SAResult

__all__ = ["SAResult", "analyze"]
