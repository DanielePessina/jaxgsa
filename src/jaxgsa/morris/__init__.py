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

from jaxgsa._core.invalid import InvalidUnit as _InvalidUnit
from jaxgsa._core.registry import MethodSpec as _MethodSpec
from jaxgsa._core.registry import register as _register
from jaxgsa.morris._analyze import MorrisIndices, analyze, indices
from jaxgsa.morris._result import MorrisResult
from jaxgsa.morris._sampling import MorrisSamples, sample

__all__ = [
    "MorrisIndices",
    "MorrisResult",
    "MorrisSamples",
    "analyze",
    "indices",
    "sample",
]

_SPEC = _register(
    _MethodSpec(
        name="morris",
        analyze=analyze,
        sample=sample,
        result=MorrisResult,
        correlation="refuses",
        categorical="refuses",
        bootstrap="n_bootstrap",
        # A trajectory is D+1 rows walked one parameter at a time. An
        # elementary effect is a difference between two of its rows.
        invalid_unit=_InvalidUnit.TRAJECTORY,
    )
)
