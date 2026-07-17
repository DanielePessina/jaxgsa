"""Shared low-level sampling helpers.

Marginal transforms (unit cube to the problem's declared distributions),
power-of-2 utilities for Sobol'-sequence sizing, and stable row
deduplication. Used by the Sobol', Morris, and eFAST samplers as well as
:func:`gsax.sampling.monte_carlo`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm, truncnorm

if TYPE_CHECKING:
    from gsax.problem import Problem


def _is_power_of_2(n: int) -> bool:
    """Check whether *n* is a positive power of 2."""
    return n >= 1 and (n & (n - 1)) == 0


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is >= *n*."""
    if n <= 0:
        return 1
    # Bit-length trick: (n-1).bit_length() gives the position of the highest
    # set bit, so 1 << that yields the smallest power of 2 >= n.
    return 1 << (n - 1).bit_length()


def _power_of_2_error(name: str, value: int, *, reason: str) -> str:
    """Build a ValueError message for a value that must be a power of 2.

    Names the two nearest valid powers of 2 so the user can correct the
    value without guessing.

    Args:
        name: Parameter name to report (e.g. ``"base_n"``).
        value: The offending value.
        reason: Short justification shown in parentheses (e.g.
            ``"Sobol' sequence balance"``).

    Returns:
        The formatted error message.
    """
    if value < 2:
        lower, upper = 1, 2
    else:
        lower = 1 << (value.bit_length() - 1)
        upper = lower * 2
    return (
        f"{name} must be a power of 2 ({reason}); got {value} — nearest valid: {lower} or {upper}"
    )


def _transform_uniform(unit_values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Affine-map unit-interval samples into a finite uniform range."""
    return unit_values * (high - low) + low


def _transform_gaussian(
    unit_values: np.ndarray,
    mean: float,
    variance: float,
    *,
    low: float | None,
    high: float | None,
) -> np.ndarray:
    """Transform unit-interval samples into Gaussian or truncated Gaussian values."""
    # Inverse-CDF sampling (probability integral transform): if U ~ Uniform(0,1)
    # then F^{-1}(U) ~ F.  Clipping to (1e-12, 1-1e-12) prevents ppf (percent-
    # point function = quantile = inverse CDF) from returning +/-inf at boundaries.
    clipped = np.clip(unit_values, 1e-12, 1.0 - 1e-12)
    std = math.sqrt(variance)
    if low is None and high is None:
        return mean + std * norm.ppf(clipped)

    # Standardised truncation bounds a=(lo-mu)/sigma, b=(hi-mu)/sigma follow
    # scipy's truncnorm convention (standard-normal scale).
    a = -np.inf if low is None else (low - mean) / std
    b = np.inf if high is None else (high - mean) / std
    return truncnorm.ppf(clipped, a=a, b=b, loc=mean, scale=std)


def _transform_samples(problem: Problem, samples_unit: np.ndarray) -> np.ndarray:
    """Transform unit-cube samples into the problem's declared marginals."""
    # Pre-allocate output; each column is filled independently by its marginal's inverse CDF
    transformed = np.empty_like(samples_unit, dtype=np.float64)

    for idx, spec in enumerate(problem.input_specs):
        dist, first, second, low, high = spec
        if dist == "uniform":
            transformed[:, idx] = _transform_uniform(samples_unit[:, idx], first, second)
        else:
            transformed[:, idx] = _transform_gaussian(
                samples_unit[:, idx],
                first,
                second,
                low=low,
                high=high,
            )

    return transformed


def _stable_unique_rows(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate rows while preserving first-occurrence order.

    Returns:
        ``(unique_samples, expanded_to_unique)`` where ``expanded_to_unique``
        maps each original row position in ``samples`` back to the retained
        unique row index.
    """
    # Ensure C-contiguous layout so tobytes() gives a consistent byte representation
    samples = np.ascontiguousarray(samples)
    unique_rows: list[np.ndarray] = []
    expanded_to_unique = np.empty(samples.shape[0], dtype=np.int64)
    seen: dict[bytes, int] = {}

    for idx, row in enumerate(samples):
        # ``row.tobytes()`` gives a stable exact-match key for the already
        # scaled floating-point row. Exact deduplication is what we want here:
        # if two rows are bitwise equal, evaluating the model twice is wasteful.
        key = row.tobytes()
        unique_idx = seen.get(key)
        if unique_idx is None:
            unique_idx = len(unique_rows)
            seen[key] = unique_idx
            unique_rows.append(row.copy())
        expanded_to_unique[idx] = unique_idx

    if unique_rows:
        unique_samples = np.vstack(unique_rows)
    else:
        unique_samples = np.empty((0, samples.shape[1]), dtype=samples.dtype)
    return unique_samples, expanded_to_unique
