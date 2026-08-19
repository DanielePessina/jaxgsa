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

``pawn.indices`` is the same estimator with no diagnostics, as a bare tuple of
arrays that ``jit``, ``vmap`` and ``jacrev`` accept. It returns the point
estimate only: a bootstrap draws randomness and forms an interval, which is
policy, so ``n_bootstrap`` stays on ``analyze``.
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.pawn._analyze import analyze, indices
from jaxgsa.pawn._result import PAWNResult

__all__ = ["PAWNResult", "analyze", "indices"]

SPEC = register(
    MethodSpec(
        name="pawn",
        analyze=analyze,
        sample=None,
        result=PAWNResult,
        correlation="accepts",
        categorical="accepts",
        bootstrap="n_bootstrap",
        invalid_unit=InvalidUnit.ROW,
    )
)
