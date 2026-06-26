"""HSIC (Hilbert-Schmidt Independence Criterion) sensitivity analysis.

Computes kernel-based sensitivity indices from arbitrary (X, Y) sample
pairs. R2-HSIC measures first-order dependence; Total HSIC captures
dependence through interactions.

Example::

    from gsax import hsic
    from gsax.sampling import sample_mc

    X = sample_mc(problem, N=4096, seed=42)
    Y = model(X)
    result = hsic.analyze(problem, jnp.asarray(X), Y)
"""

from gsax.hsic._analyze import analyze
from gsax.hsic._result import HSICResult

__all__ = ["HSICResult", "analyze"]
