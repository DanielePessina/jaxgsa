"""PAWN sensitivity analysis (Pianosi & Wagener, 2015).

PAWN measures how much fixing an input changes the *entire* output
distribution, not just its variance: the index is the Kolmogorov-Smirnov
distance between the unconditional output CDF and the CDF conditional on
the input, aggregated over conditioning bins. It is a given-data method
(any (X, Y) sample works) and is a good pick when the output is skewed
or multimodal, where variance is a poor summary of uncertainty.

Example::

    from gsax import pawn
    from gsax.sampling import monte_carlo

    X = monte_carlo(problem, n=5000, seed=42)
    Y = model(X)
    result = pawn.analyze(problem, X, Y)
"""

from gsax.pawn._analyze import analyze
from gsax.pawn._result import PAWNResult

__all__ = ["PAWNResult", "analyze"]
