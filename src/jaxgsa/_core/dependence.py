"""Shared contracts and validation for user-supplied input samplers.

Callbacks draw independent Monte Carlo rows in physical input units. Joint
and conditional callbacks must describe the same distribution. Shape checks
cannot establish that statistical consistency.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

JointSampler = Callable[[int, np.random.Generator], npt.ArrayLike]
ConditionalSampler = Callable[[np.ndarray, tuple[int, ...], np.random.Generator], npt.ArrayLike]


def validate_sampler_pair(
    joint_sampler: JointSampler | None,
    conditional_sampler: ConditionalSampler | None,
) -> bool:
    """Check that custom joint and conditional samplers are supplied together.

    Returns:
        Whether a custom sampler pair was supplied.

    Raises:
        ValueError: If exactly one callback is supplied.
        TypeError: If a supplied callback is not callable.
    """
    if (joint_sampler is None) != (conditional_sampler is None):
        raise ValueError("joint_sampler and conditional_sampler must be supplied together")
    if joint_sampler is None:
        return False
    if not callable(joint_sampler) or not callable(conditional_sampler):
        raise TypeError("joint_sampler and conditional_sampler must be callable")
    return True


def _sample_array(values: npt.ArrayLike, shape: tuple[int, int], name: str) -> np.ndarray:
    """Return a validated, independently owned real sample array."""
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{name} must return shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numeric values")
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def draw_joint_samples(
    sampler: JointSampler,
    n: int,
    n_params: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw and validate independent joint rows in physical units.

    Args:
        sampler: Callback taking the row count and RNG positionally.
        n: Positive number of rows.
        n_params: Positive number of input columns.
        rng: Generator supplied to the callback.

    Returns:
        Owned float64 array of shape ``(n, n_params)``.
    """
    if n < 1 or n_params < 1:
        raise ValueError("n and n_params must be positive")
    return _sample_array(sampler(n, rng), (n, n_params), "joint_sampler")


def draw_conditional_samples(
    sampler: ConditionalSampler,
    anchors: np.ndarray,
    fixed_indices: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one independent conditional row per anchor in physical units.

    Args:
        sampler: Callback taking anchor rows, fixed column indices and RNG
            positionally. Redraw the other columns conditionally on the fixed
            values, independently across rows and across callback invocations.
        anchors: Two-dimensional physical input rows.
        fixed_indices: Distinct input column indices to preserve exactly.
        rng: Generator supplied to the callback.

    Returns:
        Owned float64 array matching the anchor shape.

    Raises:
        ValueError: If samples are malformed or fixed values change.
    """
    if anchors.ndim != 2 or min(anchors.shape) < 1:
        raise ValueError("anchors must be a nonempty two-dimensional array")
    shape = (anchors.shape[0], anchors.shape[1])
    original = _sample_array(anchors, shape, "anchors")
    if not isinstance(fixed_indices, tuple) or any(
        isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in fixed_indices
    ):
        raise ValueError("fixed_indices must be a tuple of distinct integer column indices")
    if len(set(fixed_indices)) != len(fixed_indices) or any(
        i < 0 or i >= shape[1] for i in fixed_indices
    ):
        raise ValueError("fixed_indices must be distinct valid column indices")
    result = _sample_array(
        sampler(original.copy(), fixed_indices, rng), shape, "conditional_sampler"
    )
    if not np.array_equal(result[:, fixed_indices], original[:, fixed_indices]):
        raise ValueError("conditional_sampler must preserve fixed columns exactly")
    return result
