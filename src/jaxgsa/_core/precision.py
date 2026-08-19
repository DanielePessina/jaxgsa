"""One place that answers "what float is JAX computing in?".

Nothing in this module is public API. See :mod:`jaxgsa._core`.

jaxgsa computes in whatever precision JAX is configured for and infers the
dtype from the caller's arrays. It does not set ``jax_enable_x64`` and it
ships no wrapper around ``jax.enable_x64()``; see
``docs/adr/0014-float32-default-no-x64-wrapper.md``. The one obligation the
library takes on is the last line of that ADR: **never silently destroy
precision**.

Two idioms for the same question had grown up side by side — reading the flag
off ``jax.config`` and reading the dtype off a throwaway array — which is one
idiom too many for a question the answer to which decides a warning, a clip
bound and a tolerance. :func:`x64_enabled` and :func:`default_float_dtype` are
that one answer.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import jax
import numpy as np

from jaxgsa._core.warning_types import JaxgsaWarning

__all__ = [
    "default_float_dtype",
    "float_eps",
    "unit_clip_bounds",
    "warn_on_float64_downcast",
    "x64_enabled",
]


def x64_enabled() -> bool:
    """Report whether JAX is configured for double precision.

    Read with ``getattr`` rather than ``jax.config.read``: the reader raises
    for a flag that was never explicitly set, which is the common case.

    Returns:
        True when ``jax_enable_x64`` is on.
    """
    return bool(getattr(jax.config, "jax_enable_x64", False))


def default_float_dtype() -> np.dtype[np.floating[Any]]:
    """Return the float dtype JAX will produce for an unannotated literal.

    Returns:
        ``float64`` under x64, ``float32`` otherwise.
    """
    return np.dtype(np.float64) if x64_enabled() else np.dtype(np.float32)


def float_eps() -> float:
    """Machine epsilon of :func:`default_float_dtype`."""
    return float(np.finfo(default_float_dtype()).eps)


def unit_clip_bounds(clip: float, dtype: Any = None) -> tuple[float, float]:
    """Make a unit-interval clip representable in the dtype it will run in.

    The clip exists to keep a unit coordinate off 0 and 1, because ``ppf(0)``
    and ``ppf(1)`` are infinite. ``1 - clip`` only does that if it is a
    *different number* from 1. At ``clip = 1e-12`` it is not: in float32 the
    spacing below 1.0 is about ``6e-8``, so ``1 - 1e-12`` rounds back to
    exactly 1.0 and the upper half of the clip is a silent no-op — while its
    float64 host twin, computing the same design, clips properly.

    The upper bound is therefore pulled in to the largest value below 1.0 that
    the dtype has, and no further. The lower bound needs no such care: ``1e-12``
    is far above the smallest normal float32, so it is kept exactly, and a
    float32 run stays as close to its float64 twin as float32 allows rather
    than being widened for symmetry.

    Args:
        clip: The nominal clip width, normally
            :data:`jaxgsa._core.sampling.UNIT_CLIP`.
        dtype: The dtype the clipped values will carry. ``None`` reads
            :func:`default_float_dtype`.

    Returns:
        ``(low, high)`` as Python floats, with ``high < 1.0`` in ``dtype``.
    """
    resolved = np.dtype(default_float_dtype() if dtype is None else dtype)
    below_one = float(np.nextafter(resolved.type(1.0), resolved.type(0.0)))
    return clip, min(1.0 - clip, below_one)


def warn_on_float64_downcast(
    method: str, arrays: Mapping[str, Any], *, stacklevel: int = 4
) -> None:
    """Say once that a float64 input is about to be truncated to float32.

    ``jnp.asarray`` on a float64 array with x64 off silently returns float32.
    A caller who went to the trouble of producing double-precision outputs
    then loses them with no signal at all, and reads indices computed at a
    precision they did not ask for. This is the "never silently destroy
    precision" clause of ADR 0014, and it is a warning rather than an error
    because the float32 result is degraded, not wrong.

    One warning per analysis, naming every array affected. Per-array warnings
    would repeat the same sentence up to three times for one call.

    Two arrays that reach the preamble are deliberately *not* candidates, and
    the reason is the same for both: their dtype is not the caller's choice,
    so a warning about them names something the caller cannot act on.

    * The input matrix. jaxgsa's own samplers build every design on the host
      in float64, so a call made entirely out of jaxgsa parts would warn on
      every analysis.
    * The companion arrays passed as ``extra``. ``dgsm`` puts a synthetic
      float64 flag column there on its autodiff path — zeros and NaNs, read
      only for finiteness — which has no precision to lose. On its other path
      the array is a real Jacobian, but one the caller almost always built
      alongside ``Y`` and in the same dtype, so ``Y`` speaks for it.

    Args:
        method: Fully qualified analyzer name, for the message.
        arrays: The candidate arrays by the name to use in the message.
        stacklevel: Frames to skip so the warning points at the caller's
            ``analyze()`` call.
    """
    if x64_enabled():
        return
    affected = sorted(name for name, value in arrays.items() if _is_float64(value))
    if not affected:
        return
    listed = " and ".join(affected)
    plural = "s were" if len(affected) > 1 else " was"
    warnings.warn(
        f"{method}: {listed}{plural} passed as float64, but JAX is configured for "
        "float32, so the values are truncated on the way to the device and the extra "
        'digits are lost. Turn float64 on with jax.config.update("jax_enable_x64", True), '
        "or the jax.experimental.enable_x64() context manager, before the analysis.",
        category=JaxgsaWarning,
        stacklevel=stacklevel,
    )


def _is_float64(value: Any) -> bool:
    """Report whether a value already carries a float64 dtype.

    Read off the object rather than through ``np.asarray``: converting to find
    out would copy the sample, and a value with no dtype at all — a nested
    Python list, say — was never in double precision to begin with.
    """
    dtype = getattr(value, "dtype", None)
    return dtype is not None and np.dtype(dtype) == np.float64
