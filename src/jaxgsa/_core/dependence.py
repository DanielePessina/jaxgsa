"""Uniform-point transformations for user-supplied joint distributions."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
from scipy.stats import qmc


class InputSampler(Protocol):
    """Transform uniforms into joint or conditional physical input rows."""

    def __call__(
        self,
        uniform_points: np.ndarray,
        /,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> npt.ArrayLike:
        """Return one physical input row per uniform point."""
        ...


def validate_sampler(
    sampler: InputSampler | None,
    sampler_dimension: int | None,
    n_params: int,
    sequence: str,
) -> int | None:
    """Validate sampling options and return the custom uniform dimension."""
    if sequence not in {"sobol", "monte_carlo"}:
        raise ValueError("sequence must be 'sobol' or 'monte_carlo'")
    if sampler is None:
        if sampler_dimension is not None:
            raise ValueError("sampler_dimension requires sampler")
        if sequence != "sobol":
            raise ValueError("sequence='monte_carlo' requires sampler")
        return None
    if not callable(sampler):
        raise TypeError("sampler must be callable")
    dimension = n_params if sampler_dimension is None else sampler_dimension
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, (int, np.integer))
        or dimension < 1
    ):
        raise ValueError("sampler_dimension must be a positive integer")
    return int(dimension)


def generate_uniform_points(
    n: int,
    dimension: int,
    *,
    sequence: str,
    rng: np.random.Generator,
    scramble: bool = True,
) -> np.ndarray:
    """Generate exact-count Monte Carlo or power-of-two Sobol uniform rows."""
    if any(
        isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1
        for v in (n, dimension)
    ):
        raise ValueError("n and dimension must be positive integers")
    n, dimension = int(n), int(dimension)
    if sequence == "monte_carlo":
        return rng.random((n, dimension))
    if sequence != "sobol":
        raise ValueError("sequence must be 'sobol' or 'monte_carlo'")
    if n & (n - 1):
        raise ValueError("Sobol sampling requires a power-of-two sample count")
    return qmc.Sobol(d=dimension, scramble=scramble, seed=rng).random_base2(n.bit_length() - 1)


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


def transform_samples(
    sampler: InputSampler,
    uniform_points: np.ndarray,
    n_params: int,
    *,
    conditioning_rows: np.ndarray | None = None,
    fixed_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Transform joint or conditional rows, protecting callback inputs.

    Args:
        sampler: Distribution transformation receiving uniforms and optional
            conditioning rows. An empty fixed subset requests joint draws.
        uniform_points: Uniform coordinates of shape ``(n, sampler_dimension)``.
        n_params: Number of physical input columns returned by the sampler.
        conditioning_rows: Physical rows containing the values to hold fixed.
        fixed_indices: Distinct physical input columns to preserve exactly.

    Returns:
        Owned finite float64 input rows of shape ``(n, n_params)``.
    """
    u = np.asarray(uniform_points)
    if u.ndim != 2 or min(u.shape) < 1:
        raise ValueError("uniform_points must be a nonempty two-dimensional array")
    u = _sample_array(u, (u.shape[0], u.shape[1]), "uniform_points")
    if np.any((u < 0) | (u > 1)):
        raise ValueError("uniform_points must lie in [0, 1]")
    if not isinstance(fixed_indices, tuple) or any(
        isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in fixed_indices
    ):
        raise ValueError("fixed_indices must be a tuple of distinct integer column indices")
    if len(set(fixed_indices)) != len(fixed_indices) or any(
        i < 0 or i >= n_params for i in fixed_indices
    ):
        raise ValueError("fixed_indices must be distinct valid column indices")
    shape = (len(u), n_params)
    original = None
    if conditioning_rows is not None:
        original = _sample_array(conditioning_rows, shape, "conditioning_rows")
    elif fixed_indices:
        raise ValueError("conditioning_rows is required when fixed_indices is nonempty")
    callback_rows = None if original is None else original.copy()
    result = _sample_array(
        sampler(u.copy(), conditioning_rows=callback_rows, fixed_indices=fixed_indices),
        shape,
        "sampler",
    )
    if original is not None and not np.array_equal(
        result[:, fixed_indices], original[:, fixed_indices]
    ):
        raise ValueError("sampler must preserve fixed columns exactly")
    return result
