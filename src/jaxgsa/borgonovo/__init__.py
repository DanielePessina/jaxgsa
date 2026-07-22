"""Borgonovo delta sensitivity analysis (Borgonovo, 2007).

The delta index measures how much learning an input's value shifts the
*entire* output density — not just its variance — so it captures
influence on tails and shape that variance-based indices miss
("moment-independent"). It is estimated from given data via the
Plischke, Borgonovo & Smith (2013) estimator: the class-averaged L1
distance between the unconditional output density and the output density
conditional on each input, plus the given-data first-order Sobol index
from the same partition.

Example::

    from jaxgsa import borgonovo
    from jaxgsa.sampling import monte_carlo

    X = monte_carlo(problem, n=5000, seed=42)
    Y = model(X)
    result = borgonovo.analyze(problem, X, Y)
"""

from jaxgsa.borgonovo._analyze import analyze
from jaxgsa.borgonovo._result import DeltaResult

__all__ = ["DeltaResult", "analyze"]
