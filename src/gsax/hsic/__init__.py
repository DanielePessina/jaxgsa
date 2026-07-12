"""HSIC (Hilbert-Schmidt Independence Criterion) sensitivity analysis.

HSIC measures statistical dependence between each input and the output
via kernel embeddings, so it detects nonlinear and non-monotonic
relationships that correlation-based screening misses. It is a
given-data method: any existing (X, Y) sample works, with no special
sampling design. R2-HSIC is the normalized first-order index; Total
HSIC additionally captures dependence carried through interactions.

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
