"""Analytical benchmark functions with known Sobol indices.

Every function here has exact indices in closed form. Use them to check that
a sensitivity method is correct, and to measure how fast its estimates reach
the ground truth as the sample size grows.

Each submodule provides:
    - ``PROBLEM``: a :class:`~jaxgsa.Problem` definition.
    - ``evaluate(X)``: the benchmark function (JAX-compatible).
    - ``ANALYTICAL_S1``, ``ANALYTICAL_ST``, ``ANALYTICAL_S2``: precomputed
      indices for the default parameters.
    - ``analytical_indices(...)``: compute indices for custom parameters.

Some submodules add further ground truths. ``ishigami``, ``linear``, and
``sobol_g`` expose ``ANALYTICAL_SHAPLEY``. ``gaussian_linear`` exposes
``ANALYTICAL_DELTA`` for the Borgonovo delta index.

Example::

    import jax.numpy as jnp

    from jaxgsa import sobol
    from jaxgsa.benchmarks import ishigami

    sr = sobol.sample(ishigami.PROBLEM, n_samples=4096)
    Y = ishigami.evaluate(jnp.asarray(sr.samples))
    result = sobol.analyze(sr, Y)
"""

from jaxgsa.benchmarks import gaussian_linear, ishigami, linear, oakley_ohagan, sobol_g

__all__ = ["gaussian_linear", "ishigami", "linear", "oakley_ohagan", "sobol_g"]
