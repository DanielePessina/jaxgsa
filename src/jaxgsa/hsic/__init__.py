"""HSIC (Hilbert-Schmidt Independence Criterion) sensitivity analysis.

HSIC uses kernel embeddings to measure the statistical dependence between
each parameter and the output. It therefore finds nonlinear and
non-monotonic relationships that correlation-based screening misses. HSIC
is a given-data method: any existing (X, Y) sample works, and no special
sampling design is needed. R2-HSIC is the normalized first-order index.
Total HSIC also counts dependence carried through interactions.

Example::

    import jax

    from jaxgsa import hsic
    from jaxgsa.sampling import monte_carlo

    X = monte_carlo(problem, n_samples=4096, seed=42)
    Y = model(X)
    result = hsic.analyze(problem, X, Y, key=jax.random.key(0))
"""

from jaxgsa._core.invalid import InvalidUnit as _InvalidUnit
from jaxgsa._core.registry import MethodSpec as _MethodSpec
from jaxgsa._core.registry import register as _register
from jaxgsa.hsic._analyze import HSICIndices, analyze, indices
from jaxgsa.hsic._result import HSICResult

__all__ = ["HSICIndices", "HSICResult", "analyze", "indices"]

_SPEC = _register(
    _MethodSpec(
        name="hsic",
        analyze=analyze,
        sample=None,
        result=HSICResult,
        correlation="accepts",
        categorical="refuses",
        # Uncertainty is reported as permutation p_values, not as an
        # interval, so there is no bootstrap keyword.
        bootstrap=None,
        invalid_unit=_InvalidUnit.ROW,
    )
)
