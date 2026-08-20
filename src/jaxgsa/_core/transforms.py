"""Shared input transforms for expansion methods (HDMR, PCE)."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import ndtr

from jaxgsa._core.precision import unit_clip_bounds
from jaxgsa._core.sampling import UNIT_CLIP
from jaxgsa.problem import CategoricalSpec, Problem, UniformSpec


def _truncnorm_cdf(z: Array, a: float, b: float, *, clip: bool = True) -> Array:
    """CDF of a standard normal truncated to ``[a, b]``, evaluated at ``z``.

    This exists so the transform stays on device: ``scipy.stats.truncnorm.cdf``
    needs a host read of the array, which raises ``TracerArrayConversionError``
    under ``jit``, ``vmap``, or ``jacrev``.

    The definition is ``(Phi(z) - Phi(a)) / (Phi(b) - Phi(a))``. Written that
    way it cancels catastrophically when ``a`` and ``b``
    both sit far out in the *upper* tail, because ``Phi`` of both is within
    rounding of 1 and the difference is all round-off. The standard remedy,
    which scipy also uses, is to work with the survival function there:
    ``Phi(-x)`` keeps the same values small and well separated, and

        (Phi(-a) - Phi(-z)) / (Phi(-a) - Phi(-b))

    is the same quantity with no cancellation. The lower tail needs no such
    care, since the direct form is already well conditioned there.

    ``a`` and ``b`` are Python floats read off the spec, so the branch between
    the two forms is taken at trace time and never depends on data.

    Args:
        z: Standardised values, any shape.
        a: Lower truncation bound in standard units. May be ``-inf``.
        b: Upper truncation bound in standard units. May be ``+inf``.
        clip: Clamp the result to ``[0, 1]`` when ``True`` (the default). A
            ``z`` outside ``[a, b]`` otherwise computes to a value outside
            ``[0, 1]``, which is what :func:`cdf_to_unit_interval` needs to
            tell an out-of-declared-support sample from an in-range one; pass
            ``False`` there.

    Returns:
        Values in ``[0, 1]`` when ``clip`` is ``True``, the same shape as
        ``z``. Otherwise a value outside ``[0, 1]`` marks ``z`` outside
        ``[a, b]``.
    """
    # Cast the bounds to z's dtype: ndtr rejects integer inputs, and matching
    # the dtype keeps the whole expression in the caller's precision instead
    # of silently promoting a float32 column to float64 under x64.
    lo = jnp.asarray(a, dtype=z.dtype)
    hi = jnp.asarray(b, dtype=z.dtype)
    if a > 0.0:  # interval in the upper tail: use the survival function
        sf_a, sf_b = ndtr(-lo), ndtr(-hi)
        u = (sf_a - ndtr(-z)) / (sf_a - sf_b)
    else:
        cdf_a, cdf_b = ndtr(lo), ndtr(hi)
        u = (ndtr(z) - cdf_a) / (cdf_b - cdf_a)
    return jnp.clip(u, 0.0, 1.0) if clip else u


def cdf_to_unit_interval(X: Array, problem: Problem) -> Array:
    """Map each column of ``X`` to ``[0, 1]`` through its marginal CDF.

    Uniform inputs use an affine map. Gaussian inputs, truncated or not, go
    through their normal CDF. The result suits a B-spline basis directly, or
    a further affine map to ``[-1, 1]`` for Legendre polynomials.

    The two families do not treat the interval ends the same way, and the
    difference is deliberate.

    * A Gaussian column is clipped to ``[UNIT_CLIP, 1 - UNIT_CLIP]``, with the
      upper bound brought inside the column's own dtype — but only when the
      raw CDF value already lies in ``[0, 1]``. The normal CDF saturates to
      exactly 0 or 1 in the far tails at float precision, and the clip keeps
      those in-range samples off the ends. A *truncated* Gaussian sample
      outside its declared ``[low, high]`` computes a value outside ``[0, 1]``
      instead (see :func:`_truncnorm_cdf`), and that value is left alone, for
      the same reason a uniform column is (next bullet).
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
        ``(N, D)`` array. Every column lies in ``[0, 1]`` for every sample
        within its declared support. A sample outside the declared support
        of a uniform or truncated Gaussian marginal maps outside ``[0, 1]``,
        which is the out-of-range signal the callers read.

    Raises:
        ValueError: If any parameter is categorical. Its step CDF collapses
            each level onto one point, which would silently discard the level
            structure. The expansion methods that call this transform already
            reject categorical problems up front, so this is a second,
            layered guard.
    """
    from jax.scipy.stats import norm as jax_norm

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
        # math.sqrt, not float(jnp.sqrt(...)): inside a trace jnp.sqrt returns a
        # tracer even for a Python-float input, so float() on it raises. The
        # variance is a plain float on the spec, so no JAX op is needed at all,
        # and this keeps the whole branch below concrete at trace time.
        std = math.sqrt(spec.variance)
        if spec.low is not None or spec.high is not None:
            a = -np.inf if spec.low is None else (spec.low - spec.mean) / std
            b = np.inf if spec.high is None else (spec.high - spec.mean) / std
            # clip=False: an out-of-declared-bounds sample must compute to a
            # value outside [0, 1], not be silently folded into the nearest
            # edge bin. See the docstring above and _truncnorm_cdf.
            u = _truncnorm_cdf((X[:, d] - spec.mean) / std, a, b, clip=False)
        else:  # unbounded Gaussian -- stays on device via jax.scipy
            u = jax_norm.cdf(X[:, d], loc=spec.mean, scale=std)
        # The upper bound is pulled in to the last float below 1.0 when the
        # dtype cannot tell 1 - UNIT_CLIP from 1. The lower bound is left at
        # UNIT_CLIP, which float32 represents exactly; widening it for
        # symmetry would discard information and move numbers for nothing.
        # See unit_clip_bounds. Only applied to values already in [0, 1]: a
        # truncated sample outside its declared bounds must stay outside.
        clip_lo, clip_hi = unit_clip_bounds(UNIT_CLIP, u.dtype)
        in_range = (u >= 0.0) & (u <= 1.0)
        cols.append(jnp.where(in_range, jnp.clip(u, clip_lo, clip_hi), u))

    return jnp.column_stack(cols)
