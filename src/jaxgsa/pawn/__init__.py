"""PAWN sensitivity analysis (Pianosi & Wagener, 2015).

PAWN measures how much fixing a parameter changes the whole output
distribution, not just its variance. The index is the Kolmogorov-Smirnov
distance between the unconditional output CDF and the CDF conditional on
the parameter, aggregated over conditioning bins. PAWN is a given-data
method, so any (X, Y) sample works. It is a good choice when the output is
skewed or multimodal, where variance is a poor summary of uncertainty.

Categorical parameters are supported. A categorical parameter gets one
conditioning class per level instead of a bin of its range, so its index
does not depend on the order of the level codes.

Example::

    from jaxgsa import pawn
    from jaxgsa.sampling import monte_carlo

    X = monte_carlo(problem, n=5000, seed=42)
    Y = model(X)
    result = pawn.analyze(problem, X, Y)
"""

from jaxgsa.pawn._analyze import analyze
from jaxgsa.pawn._result import PAWNResult

__all__ = ["PAWNResult", "analyze"]
