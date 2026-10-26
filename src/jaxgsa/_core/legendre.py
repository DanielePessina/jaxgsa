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

from jaxgsa._core.precision import default_float_dtype


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
    stay JAX. NumPy arrays stay NumPy. A floating JAX ``x`` keeps its dtype,
    as the Hermite basis does, so float32 data gives a float32 design matrix
    under x64 too. A non-floating JAX ``x`` is cast to the backend's default
    float dtype. A NumPy ``x`` always runs in float64 (the VKOGA
    component fit relies on that).

    Args:
        x: ``(...,)`` points in ``[-1, 1]``.
        max_degree: Highest polynomial degree.

    Returns:
        ``(..., max_degree + 1)`` basis values, in the dtype described above.
    """
    on_device = isinstance(x, jax.Array)
    xp = jnp if on_device else np
    if not on_device:
        x = x.astype(np.float64)
    elif not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(default_float_dtype())
    columns = [xp.ones_like(x)]
    if max_degree >= 1:
        columns.append(x)
    for n in range(1, max_degree):
        # P_{n+1}(x) = ((2n+1) x P_n(x) - n P_{n-1}(x)) / (n+1).
        columns.append(((2 * n + 1) * x * columns[n] - n * columns[n - 1]) / (n + 1))
    basis = xp.stack(columns, axis=-1)
    norms = xp.sqrt(xp.asarray([2.0 * k + 1.0 for k in range(max_degree + 1)], dtype=basis.dtype))
    return basis * norms
