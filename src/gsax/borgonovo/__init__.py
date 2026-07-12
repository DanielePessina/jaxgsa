"""Borgonovo delta sensitivity analysis (Borgonovo, 2007).

Computes moment-independent sensitivity indices from given data using
the Plischke, Borgonovo & Smith (2013) estimator: the class-averaged L1
distance between the unconditional output density and the output density
conditional on each input, plus the given-data first-order Sobol index
from the same partition.

Example::

    from gsax import borgonovo
    from gsax.sampling import sample_mc

    X = sample_mc(problem, N=5000, seed=42)
    Y = model(X)
    result = borgonovo.analyze(problem, X, Y)
"""

from gsax.borgonovo._analyze import analyze
from gsax.borgonovo._result import DeltaResult

__all__ = ["DeltaResult", "analyze"]
