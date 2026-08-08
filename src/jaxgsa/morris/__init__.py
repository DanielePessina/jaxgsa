"""Morris elementary-effects screening method.

Morris screening changes one parameter at a time and measures the finite
difference in the output. It calls this difference an elementary effect, and it
samples effects across the whole input domain rather than around one point. The
method reduces the effects to three measures: ``mu_star`` (importance), ``mu``
(signed mean), and ``sigma`` (nonlinearity or interactions). The design costs
``n_trajectories * (D + 1)`` model evaluations.

Example::

    from jaxgsa import morris

    sr = morris.sample(problem, n_trajectories=30, seed=42)
    Y = model(sr.samples)
    result = morris.analyze(sr, Y)
"""

from jaxgsa.morris._analyze import analyze
from jaxgsa.morris._result import MorrisResult
from jaxgsa.morris._sampling import MorrisSamples, sample

__all__ = ["MorrisResult", "MorrisSamples", "analyze", "sample"]
