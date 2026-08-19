"""Sobol variance-based sensitivity analysis (Saltelli sampling).

This is the one design-based route that accepts categorical inputs: the
column-swap scheme only moves whole level codes between the A and B
matrices, so it never assumes an order. It refuses a problem with a
declared ``correlation``, because the estimators assume independent
inputs. ``SobolSamples.to_morris`` is the exception on categorical
inputs and refuses them, because an elementary effect needs a step
along an orderable axis.

Example::

    from jaxgsa import Problem, sobol

    problem = Problem.from_dict({"x1": (0, 1), "x2": (0, 1)})
    sr = sobol.sample(problem, n_samples=4096, seed=42)
    Y = model(sr.samples)
    result = sobol.analyze(sr, Y)

There is a second, transformable route through the same estimators.
``SobolSamples.transform(theta)`` rebuilds the design from its unit-cube
points for any input distribution parameters, and ``sobol.indices`` returns
the indices as plain arrays with no diagnostics attached. Both are pure JAX,
so composing them with a JAX model gives a chain that ``jax.grad`` can
differentiate: how an index moves when an input's range or variance moves.
``analyze`` cannot do that, because deciding what to do about a non-finite
output needs a concrete value and a tracer has none. Neither route accepts a
categorical problem for differentiation: a step CDF has no useful derivative.
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.sobol._analyze import analyze, indices
from jaxgsa.sobol._result import SobolResult
from jaxgsa.sobol._sampling import SobolSamples, sample

__all__ = ["SobolResult", "SobolSamples", "analyze", "indices", "sample"]

SPEC = register(
    MethodSpec(
        name="sobol",
        analyze=analyze,
        sample=sample,
        result=SobolResult,
        # The Saltelli design and its estimators assume independent inputs.
        correlation="refuses",
        categorical="accepts",
        bootstrap="n_bootstrap",
        # A Saltelli group is D+2 or 2D+2 rows for one base point. Dropping
        # part of one leaves the estimator reading misaligned rows.
        invalid_unit=InvalidUnit.SALTELLI_GROUP,
    )
)
