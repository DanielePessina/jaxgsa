"""Morris elementary-effects screening method.

Globalized one-at-a-time screening: finite-difference effects sampled across
the whole input domain are reduced to mu_star (importance), mu (signed mean),
and sigma (nonlinearity/interactions) at a cost of ``n_trajectories * (D + 1)``
model evaluations.

Example::

    from gsax import morris

    sr = morris.sample(problem, n_trajectories=30, seed=42)
    Y = model(sr.samples)
    result = morris.analyze(sr, Y)
"""

from gsax.morris._analyze import analyze
from gsax.morris._result import MorrisResult
from gsax.morris._sampling import MorrisSamples, sample

__all__ = ["MorrisResult", "MorrisSamples", "analyze", "sample"]
