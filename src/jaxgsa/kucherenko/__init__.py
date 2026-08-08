"""Kucherenko Sobol' indices for dependent inputs (design-based).

Design-based method: generate a conditional-copula design, evaluate your
actual model on it, and estimate ``S1 = V(E(Y|X_i))/V(Y)`` and
``ST = E(V(Y|X_{~i}))/V(Y)`` under the problem's declared dependence
structure (Kucherenko, Tarantola & Annoni 2012). No surrogate is fitted. This
is the design-based counterpart to :mod:`jaxgsa.vkoga`, which estimates the
same two quantities against a kernel surrogate. Under independent inputs both
indices reduce exactly to the Saltelli column-swap estimators of the classic
Sobol' ``S1`` and ``ST``.

Gating: ``sample`` conditions on a declared ``problem.correlation``, so it is
exempt from the correlated-design refusal that ``sobol``, ``morris``, and
``efast`` apply. It refuses categorical parameters. The copula conditionals
need a continuous marginal CDF on every coordinate. Use
:mod:`jaxgsa.optimal_transport` or :mod:`jaxgsa.borgonovo` for categorical
inputs.

Example::

    import jaxgsa

    problem = jaxgsa.Problem(names=("x1", "x2"), bounds=[(0, 1)] * 2)
    problem = problem.with_correlation(R)      # optional dependence

    ks = jaxgsa.kucherenko.sample(problem, 4096, seed=0)
    Y = model(ks.samples)                      # (n_runs,) evaluations
    result = jaxgsa.kucherenko.analyze(ks, Y)
    result.S1   # correlation-inclusive first-order indices
    result.ST   # correlation-exclusive total indices

References:
    Kucherenko, Tarantola & Annoni (2012). Comput. Phys. Commun. 183:937-946.
"""

from jaxgsa.kucherenko._analyze import analyze
from jaxgsa.kucherenko._result import KucherenkoResult
from jaxgsa.kucherenko._sampling import KucherenkoSamples, sample

__all__ = ["KucherenkoResult", "KucherenkoSamples", "analyze", "sample"]
