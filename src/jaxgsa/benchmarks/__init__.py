"""Analytical benchmark functions with known Sobol indices.

Use these to validate a sensitivity method or gauge its convergence:
each function's exact indices are available in closed form, so estimates
can be compared against ground truth at any sample size.

Each submodule provides:
    - ``PROBLEM``: a :class:`~jaxgsa.Problem` definition.
    - ``evaluate(X)``: the benchmark function (JAX-compatible).
    - ``ANALYTICAL_S1``, ``ANALYTICAL_ST``, ``ANALYTICAL_S2``: precomputed
      indices for the default parameters.
    - ``analytical_indices(...)``: compute indices for custom parameters.

Some submodules add further ground truths: ``ishigami``, ``linear``, and
``sobol_g`` expose ``ANALYTICAL_SHAPLEY``; ``gaussian_linear`` exposes
``ANALYTICAL_DELTA`` for the Borgonovo delta index.

Example::

    from jaxgsa.benchmarks import ishigami
    from jaxgsa import sample, analyze

    sr = sample(ishigami.PROBLEM, 4096)
    Y = ishigami.evaluate(sr.samples)
    result = analyze(sr, Y)
"""

from jaxgsa.benchmarks import gaussian_linear, ishigami, linear, oakley_ohagan, sobol_g

__all__ = ["gaussian_linear", "ishigami", "linear", "oakley_ohagan", "sobol_g"]
