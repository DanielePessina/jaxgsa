"""Shared input transforms for expansion methods (HDMR, PCE)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.sampling import UNIT_CLIP
from jaxgsa.problem import CategoricalSpec, Problem, UniformSpec


def cdf_to_unit_interval(X: Array, problem: Problem) -> Array:
    """Map each column of ``X`` to ``[0, 1]`` through its marginal CDF.

    Uniform inputs use an affine map. Gaussian inputs, truncated or not, go
    through their normal CDF. The result suits a B-spline basis directly, or
    a further affine map to ``[-1, 1]`` for Legendre polynomials.

    The two families do not treat the interval ends the same way, and the
    difference is deliberate.

    * A Gaussian column is clipped to ``[UNIT_CLIP, 1 - UNIT_CLIP]``. The
      normal CDF saturates to exactly 0 or 1 in the far tails at float
      precision, and the clip keeps those samples off the ends.
    * A uniform column is not clipped. The affine map is exact, so a sample
      inside the declared bounds already lands in ``[0, 1]``, and 0 and 1 are
      legitimate values for a sample that sits on a bound. A sample *outside*
      the declared bounds maps outside ``[0, 1]``, and it must stay there:
      :func:`jaxgsa.pawn._analyze._equal_width_bins` reads that as its
      out-of-range signal and drops the sample. Clipping the column would
      hide the caller's mistake instead.

    Args:
        X: ``(N, D)`` input samples in physical units.
        problem: Problem with per-dimension distribution specs.

    Returns:
        ``(N, D)`` array. Every Gaussian column lies in ``[0, 1]``. A uniform
        column lies in ``[0, 1]`` for every sample within the declared
        bounds.

    Raises:
        ValueError: If any parameter is categorical. Its step CDF collapses
            each level onto one point, which would silently discard the level
            structure. The expansion methods that call this transform already
            reject categorical problems up front, so this is a second,
            layered guard.
    """
    from jax.scipy.stats import norm as jax_norm
    from scipy.stats import truncnorm

    D = problem.num_vars
    cols = []  # one column per dimension, stacked at the end into (N, D)

    # CDF-to-unit-interval mapping: apply F(x) per dimension. For Gaussian
    # inputs, standardised truncation bounds a=(lo-mu)/sigma, b=(hi-mu)/sigma
    # follow scipy's truncnorm convention. Only the Gaussian branches clip; see
    # the docstring for why the uniform branch must not.
    for d in range(D):
        spec = problem.input_specs[d]
        if isinstance(spec, CategoricalSpec):
            raise ValueError(
                f"Parameter {problem.names[d]!r} is categorical; its step CDF is "
                "many-to-one and has no meaningful unit-interval image. Use "
                "jaxgsa.optimal_transport, jaxgsa.borgonovo, or jaxgsa.sobol "
                "for categorical inputs"
            )
        if isinstance(spec, UniformSpec):  # affine map (x - lo)/(hi - lo) is the CDF of U(lo, hi)
            cols.append((X[:, d] - spec.low) / (spec.high - spec.low))
            continue
        std = float(jnp.sqrt(spec.variance))
        # Truncated Gaussian -- need scipy because JAX lacks a truncnorm CDF.
        if spec.low is not None or spec.high is not None:
            a = -np.inf if spec.low is None else (spec.low - spec.mean) / std
            b = np.inf if spec.high is None else (spec.high - spec.mean) / std
            u = jnp.asarray(truncnorm.cdf(np.asarray(X[:, d]), a=a, b=b, loc=spec.mean, scale=std))
        else:  # unbounded Gaussian -- stays on device via jax.scipy
            u = jax_norm.cdf(X[:, d], loc=spec.mean, scale=std)
        cols.append(jnp.clip(u, UNIT_CLIP, 1.0 - UNIT_CLIP))

    return jnp.column_stack(cols)
