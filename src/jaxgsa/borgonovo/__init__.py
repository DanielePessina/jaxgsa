"""Borgonovo delta sensitivity analysis (Borgonovo, 2007).

The delta index measures how much learning an input's value shifts the
*entire* output density — not just its variance — so it captures
influence on tails and shape that variance-based indices miss
("moment-independent"). It is estimated from given data via the
Plischke, Borgonovo & Smith (2013) estimator: the class-averaged L1
distance between the unconditional output density and the output density
conditional on each input, plus the given-data first-order Sobol index
from the same partition.

The estimator supports a continuous output distribution only. It compares
kernel density estimates on a shared output grid, and a discrete output has
atoms that no grid resolves. ``analyze`` checks the output up front and
raises ``ValueError`` for a discrete one. Use
:func:`jaxgsa.optimal_transport.analyze` in that case: it compares empirical
distributions directly. Categorical *inputs* stay supported — the
restriction applies to the output.

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
