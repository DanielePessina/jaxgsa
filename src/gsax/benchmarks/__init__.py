"""Analytical benchmark functions with known Sobol indices.

Each submodule provides:
    - ``PROBLEM``: a :class:`~gsax.Problem` definition.
    - ``evaluate(X)``: the benchmark function (JAX-compatible).
    - ``ANALYTICAL_S1``, ``ANALYTICAL_ST``, ``ANALYTICAL_S2``: precomputed
      indices for the default parameters.
    - ``analytical_indices(...)``: compute indices for custom parameters.

Example::

    from gsax.benchmarks import ishigami
    from gsax import sample, analyze

    sr = sample(ishigami.PROBLEM, 4096)
    Y = ishigami.evaluate(sr.samples)
    result = analyze(sr, Y)
"""

from gsax.benchmarks import ishigami, linear, sobol_g

__all__ = ["ishigami", "linear", "sobol_g"]
