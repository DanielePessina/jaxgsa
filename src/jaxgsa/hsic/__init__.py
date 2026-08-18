"""HSIC (Hilbert-Schmidt Independence Criterion) sensitivity analysis.

HSIC uses kernel embeddings to measure the statistical dependence between
each parameter and the output. It therefore finds nonlinear and
non-monotonic relationships that correlation-based screening misses. HSIC
is a given-data method: any existing (X, Y) sample works, and no special
sampling design is needed. R2-HSIC is the normalized first-order index.
Total HSIC also counts dependence carried through interactions.

Example::

    from jaxgsa import hsic
    from jaxgsa.sampling import monte_carlo

    X = monte_carlo(problem, n=4096, seed=42)
    Y = model(X)
    result = hsic.analyze(problem, jnp.asarray(X), Y)
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.hsic._analyze import analyze
from jaxgsa.hsic._result import HSICResult

__all__ = ["HSICResult", "analyze"]

SPEC = register(
    MethodSpec(
        name="hsic",
        analyze=analyze,
        sample=None,
        result=HSICResult,
        correlation="accepts",
        categorical="refuses",
        # Uncertainty is reported as permutation p_values, not as an
        # interval, so there is no bootstrap keyword.
        bootstrap=None,
        invalid_unit=InvalidUnit.ROW,
    )
)
