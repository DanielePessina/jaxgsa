"""Borgonovo delta sensitivity analysis (Borgonovo, 2007).

The delta index measures how much learning a parameter's value shifts the
whole output density, not only its variance. It therefore captures influence
on tails and on shape that variance-based indices miss. An index built this
way is called moment-independent. jaxgsa estimates it from given data with
the Plischke, Borgonovo & Smith (2013) estimator. That estimator is the
class-averaged L1 distance between the unconditional output density and the
output density conditional on each parameter. The same partition also yields
the given-data first-order Sobol index.

The estimator supports a continuous output distribution only. It compares
kernel density estimates on a shared output grid, and a discrete output has
atoms that no grid resolves. ``analyze`` checks the output up front and
raises ``ValueError`` for a discrete one. Use
:func:`jaxgsa.optimal_transport.analyze` in that case: it compares empirical
distributions directly. Categorical parameters stay supported. The
restriction applies to the output only.

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
