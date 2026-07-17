"""Optimal-transport sensitivity analysis (Borgonovo et al., 2024).

The OT index measures how much knowing an input's value displaces the
*entire* output distribution: it is the class-averaged squared
2-Wasserstein distance between the conditional and unconditional output
distributions, normalized to [0, 1] by twice the output variance. It
reacts to changes in spread, tails and shape that variance-based indices
miss, and decomposes into an *advective* (location-shift) component --
exactly half the given-data first-order Sobol index -- plus a
*diffusive* (spread/shape) remainder. Scalar and per-column analyses use
exact 1-D optimal transport (no solver); the multivariate and
trajectory modes transport output point clouds with entropic
(Sinkhorn) regularization.
Conditioning is rank-based, so any input marginals (uniform, Gaussian,
mixed) and correlated inputs are supported.

Example::

    from gsax import optimal_transport
    from gsax.sampling import monte_carlo

    X = monte_carlo(problem, n=5000, seed=42)
    Y = model(X)
    result = optimal_transport.analyze(problem, X, Y)

References:
    Borgonovo, Figalli, Plischke & Savare (2024). Global sensitivity
    analysis via optimal transport. Management Science.
    doi:10.1287/mnsc.2023.01796.
"""

from gsax.optimal_transport._analyze import analyze
from gsax.optimal_transport._result import OTResult

__all__ = ["OTResult", "analyze"]
