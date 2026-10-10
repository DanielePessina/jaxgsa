"""Rank-based first-order Sobol' indices (Gamboa, Gremaud, Klein & Lagnoux, 2022).

The rank estimator gives all first-order Sobol' indices from one existing
``(X, Y)`` sample. For each input it sorts the rows by that input and pairs
each output with the output of the next-ranked row. The covariance of the
pairs, divided by the output variance, estimates
``S1_i = Var(E[Y | X_i]) / Var(Y)``. There is no binning choice and no tuning
parameter, and the cost is one sort per input.

The index is defined for any joint distribution of the inputs, so correlated
problems are accepted. Categorical inputs are refused, because their tied
level codes make the rank pairing ill defined.

Example::

    from jaxgsa import rank
    from jaxgsa.sampling import monte_carlo

    X = monte_carlo(problem, n=5000, seed=42)
    Y = model(X)
    result = rank.analyze(problem, X, Y)

``rank.indices`` is the same estimator with no diagnostics, as a bare tuple of
arrays that ``jit`` and ``vmap`` accept.
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.rank._analyze import analyze, indices
from jaxgsa.rank._result import RankResult

__all__ = ["RankResult", "analyze", "indices"]

SPEC = register(
    MethodSpec(
        name="rank",
        analyze=analyze,
        sample=None,
        result=RankResult,
        correlation="accepts",
        categorical="refuses",
        bootstrap="n_bootstrap",
        invalid_unit=InvalidUnit.ROW,
    )
)
