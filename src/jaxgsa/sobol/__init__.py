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
"""

from jaxgsa.sobol._analyze import analyze
from jaxgsa.sobol._result import SobolResult
from jaxgsa.sobol._sampling import SobolSamples, sample

__all__ = ["SobolResult", "SobolSamples", "analyze", "sample"]
