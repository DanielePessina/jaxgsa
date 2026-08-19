"""Optimal-transport sensitivity analysis (Borgonovo et al., 2024).

The optimal-transport (OT) index measures how much knowing a parameter's
value displaces the whole output distribution. It is the class-averaged
squared 2-Wasserstein distance between the conditional and unconditional
output distributions, normalized to [0, 1] by twice the output variance.
Variance-based indices miss changes in spread, tails and shape. The OT
index reacts to them.

The index splits into two parts. The advective component is the
location shift, and it equals exactly half the given-data first-order
Sobol index. The diffusive component is the remainder, and it covers
spread and shape.

Scalar and per-column analyses use exact 1-D optimal transport and need
no solver. The multivariate and trajectory modes transport output point
clouds with entropic (Sinkhorn) regularization. Conditioning is
rank-based, so any parameter marginals (uniform, Gaussian, mixed) and
correlated parameters are supported.

Example::

    from jaxgsa import optimal_transport
    from jaxgsa.sampling import monte_carlo

    X = monte_carlo(problem, n=5000, seed=42)
    Y = model(X)
    result = optimal_transport.analyze(problem, X, Y)

References:
    Borgonovo, Figalli, Plischke & Savare (2024). Global sensitivity
    analysis via optimal transport. Management Science.
    doi:10.1287/mnsc.2023.01796.
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.optimal_transport._analyze import analyze, indices
from jaxgsa.optimal_transport._result import OTResult

__all__ = ["OTResult", "analyze", "indices"]

SPEC = register(
    MethodSpec(
        name="optimal_transport",
        analyze=analyze,
        sample=None,
        result=OTResult,
        correlation="accepts",
        categorical="accepts",
        bootstrap="n_bootstrap",
        invalid_unit=InvalidUnit.ROW,
    )
)
