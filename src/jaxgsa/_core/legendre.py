"""Shared orthonormal Legendre recurrence.

One implementation serves two callers. The PCE basis evaluates it on
``[-1, 1]`` with JAX arrays. The VKOGA component-function fit evaluates it
with NumPy arrays, shifted to ``(0, 1)``.
"""

from __future__ import annotations

from typing import overload

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


@overload
def legendre_orthonormal(x: Array, max_degree: int) -> Array: ...
@overload
def legendre_orthonormal(x: np.ndarray, max_degree: int) -> np.ndarray: ...
def legendre_orthonormal(x: Array | np.ndarray, max_degree: int) -> Array | np.ndarray:
    """Evaluate orthonormal Legendre polynomials of degree ``0..max_degree``.

    Uses Bonnet's three-term recurrence, which is numerically stable. Degree
    ``k`` is scaled by ``sqrt(2k + 1)``. The result is then orthonormal for
    the uniform measure on ``[-1, 1]`` (weight 1/2).

    The implementation follows the input type. JAX arrays (including tracers)
    stay JAX. NumPy arrays stay NumPy. The recurrence runs in the backend's
    default float dtype (float64 under x64), the same promotion the Hermite
    basis applies, so a float32 ``x`` does not silently downgrade a PCE
    design matrix and mixed uniform/Gaussian problems get one basis dtype.

    Args:
        x: ``(...,)`` points in ``[-1, 1]``.
        max_degree: Highest polynomial degree.

    Returns:
        ``(..., max_degree + 1)`` basis values, in the default float dtype.
    """
    xp = jnp if isinstance(x, jax.Array) else np
    x = x.astype(xp.zeros(()).dtype)
    columns = [xp.ones_like(x)]
    if max_degree >= 1:
        columns.append(x)
    for n in range(1, max_degree):
        # P_{n+1}(x) = ((2n+1) x P_n(x) - n P_{n-1}(x)) / (n+1).
        columns.append(((2 * n + 1) * x * columns[n] - n * columns[n - 1]) / (n + 1))
    basis = xp.stack(columns, axis=-1)
    norms = xp.sqrt(xp.asarray([2.0 * k + 1.0 for k in range(max_degree + 1)], dtype=basis.dtype))
    return basis * norms
