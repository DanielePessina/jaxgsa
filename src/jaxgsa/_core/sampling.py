"""Shared low-level sampling helpers.

This module holds three groups of helper. Marginal transforms map the unit
cube to the problem's declared distributions and back. Power-of-2 utilities
size a Sobol' sequence. One deduplication routine drops repeated design rows
while keeping their first-occurrence order. The Sobol', Morris, and eFAST
samplers use them, as does :func:`jaxgsa.sampling.monte_carlo`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm, truncnorm

if TYPE_CHECKING:
    from jaxgsa.problem import Problem


# Absolute support bound the library puts on every unbounded marginal.
#
# Unit-cube coordinates are clipped into ``[UNIT_CLIP, 1 - UNIT_CLIP]`` before
# an inverse CDF runs, because ``ppf(0)`` and ``ppf(1)`` return -inf and +inf.
# That clip is what bounds the support: no Gaussian draw can ever leave the
# range +/-7.0345 standard deviations from its mean.
#
# The value stays at 1e-12 on purpose. It is Hermite-safe to about degree 8,
# and the moment error it introduces is about 1e-10, far below any sampling
# error a user will see. A larger clip would visibly truncate the marginal. A
# smaller one runs into the tail resolution of the scipy quantile functions.
UNIT_CLIP = 1e-12


def _is_power_of_2(n: int) -> bool:
    """Check whether ``n`` is a positive power of 2."""
    return n >= 1 and (n & (n - 1)) == 0


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is >= ``n``."""
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
    # then F^{-1}(U) ~ F.  Clipping to (UNIT_CLIP, 1-UNIT_CLIP) prevents ppf
    # (percent-point function = quantile = inverse CDF) from returning +/-inf
    # at the boundaries.
    clipped = np.clip(unit_values, UNIT_CLIP, 1.0 - UNIT_CLIP)
    std = math.sqrt(variance)
    if low is None and high is None:
        return mean + std * norm.ppf(clipped)

    # Standardised truncation bounds a=(lo-mu)/sigma, b=(hi-mu)/sigma follow
    # scipy's truncnorm convention (standard-normal scale).
    a = -np.inf if low is None else (low - mean) / std
    b = np.inf if high is None else (high - mean) / std
    return truncnorm.ppf(clipped, a=a, b=b, loc=mean, scale=std)


def _transform_categorical(unit_values: np.ndarray, probs: tuple[float, ...]) -> np.ndarray:
    """Transform unit-interval samples into integer level codes (as floats).

    Step-function inverse CDF: level ``l`` is selected when the unit value
    falls into the ``l``-th cumulative-probability bin, so level frequencies
    follow ``probs`` exactly in distribution. Codes ``0 .. L-1`` are the
    sample values; labels never enter the sample matrix.
    """
    cum = np.cumsum(np.asarray(probs, dtype=np.float64))
    codes = np.searchsorted(cum, unit_values, side="right")
    # Float cumsum can end slightly below 1.0; clamp the (measure-zero)
    # overflow onto the last level.
    return np.minimum(codes, len(cum) - 1).astype(np.float64)


def _transform_samples(problem: Problem, samples_unit: np.ndarray) -> np.ndarray:
    """Transform unit-cube samples into the problem's declared marginals."""
    # Pre-allocate output; each column is filled independently by its marginal's inverse CDF
    transformed = np.empty_like(samples_unit, dtype=np.float64)

    for idx, spec in enumerate(problem.input_specs):
        dist, first, second, low, high, categorical = spec
        if dist == "uniform":
            transformed[:, idx] = _transform_uniform(samples_unit[:, idx], first, second)
        elif dist == "categorical":
            assert categorical is not None  # categorical specs always carry (probs, labels)
            transformed[:, idx] = _transform_categorical(samples_unit[:, idx], categorical[0])
        else:
            transformed[:, idx] = _transform_gaussian(
                samples_unit[:, idx],
                first,
                second,
                low=low,
                high=high,
            )

    return transformed


def _inverse_transform_uniform(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Affine-map a finite uniform range back onto the unit interval."""
    return (values - low) / (high - low)


def _inverse_transform_gaussian(
    values: np.ndarray,
    mean: float,
    variance: float,
    *,
    low: float | None,
    high: float | None,
) -> np.ndarray:
    """Map Gaussian or truncated-Gaussian values back onto the unit interval."""
    # Forward CDF, the inverse of the ppf used by _transform_gaussian. Values
    # produced by that function were clipped to (UNIT_CLIP, 1-UNIT_CLIP)
    # beforehand, so the round trip cannot land exactly on 0 or 1 here.
    std = math.sqrt(variance)
    if low is None and high is None:
        return np.asarray(norm.cdf((values - mean) / std), dtype=np.float64)

    a = -np.inf if low is None else (low - mean) / std
    b = np.inf if high is None else (high - mean) / std
    return np.asarray(truncnorm.cdf(values, a=a, b=b, loc=mean, scale=std), dtype=np.float64)


def _inverse_transform_samples(problem: Problem, samples_physical: np.ndarray) -> np.ndarray:
    """Recover unit-cube coordinates from samples in the problem's marginals.

    Inverse of :func:`_transform_samples`, computed in float64 throughout.
    Precision matters because the recovered coordinates are differenced to
    form the denominator of an elementary effect. The float32 JAX equivalent
    (:func:`jaxgsa._core.transforms.cdf_to_unit_interval`) loses too many
    digits to divide by safely.

    Args:
        problem: Problem definition whose marginals produced the samples.
        samples_physical: Samples in physical units, shape ``(N, D)``.

    Returns:
        Unit-cube coordinates of shape ``(N, D)``, dtype float64.

    Raises:
        ValueError: If any parameter is categorical. Its step CDF maps a whole
            probability bin onto one code, so no unit-cube coordinate can be
            recovered.
    """
    unit = np.empty_like(samples_physical, dtype=np.float64)

    for idx, spec in enumerate(problem.input_specs):
        dist, first, second, low, high, _ = spec
        if dist == "categorical":
            raise ValueError(
                f"Parameter {problem.names[idx]!r} is categorical; its step CDF is "
                "many-to-one, so physical samples cannot be mapped back onto the "
                "unit cube"
            )
        if dist == "uniform":
            unit[:, idx] = _inverse_transform_uniform(samples_physical[:, idx], first, second)
        else:
            unit[:, idx] = _inverse_transform_gaussian(
                samples_physical[:, idx],
                first,
                second,
                low=low,
                high=high,
            )

    return unit


def _stable_unique_rows(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate rows while preserving first-occurrence order.

    Args:
        samples: ``(N, D)`` design matrix in physical units.

    Returns:
        ``(unique_samples, expanded_to_unique)`` where ``expanded_to_unique``
        maps each original row position in ``samples`` back to the retained
        unique row index.
    """
    # Ensure C-contiguous layout so the byte view below is well defined.
    samples = np.ascontiguousarray(samples)
    n_rows, n_cols = samples.shape

    if n_rows == 0:
        return (
            np.empty((0, n_cols), dtype=samples.dtype),
            np.empty(0, dtype=np.int64),
        )
    if n_cols == 0:
        # Every row holds the same (empty) value, so one row survives.
        return samples[:1].copy(), np.zeros(n_rows, dtype=np.int64)

    # Compare whole rows as raw bytes. This keeps the exact-match semantics of
    # the old ``row.tobytes()`` key: two rows merge only when they are bitwise
    # equal, so 0.0 and -0.0 stay distinct and equal NaN patterns still merge.
    # Exact deduplication is what we want here: if two rows are bitwise equal,
    # evaluating the model twice is wasteful.
    #
    # The void view turns each row into one Python ``bytes`` object, and
    # ``tolist()`` builds all N of them in one C loop. That is the expensive
    # part of the old per-row Python loop, which paid for a fresh array view
    # plus a ``tobytes()`` call on every row. Hashing the keys in a dict then
    # costs far less than sorting them.
    #
    # Measured on an M1 Pro at the hot case (N=2**20, D=20, float64): the old
    # per-row loop took 1099 ms, ``np.unique`` over the same void view took
    # 1128 ms (it falls back to a generic comparison sort with memcmp
    # callbacks, so it loses to hashing once rows are wide), ``np.lexsort``
    # over the columns took 5431 ms, and this form took 325 ms. Do not replace
    # this with a sort-based form without re-measuring.
    keys = samples.view(np.dtype((np.void, samples.dtype.itemsize * n_cols))).ravel().tolist()

    # ``setdefault`` numbers the rows in first-appearance order: the label of a
    # new row is the number of distinct rows seen before it.
    seen: dict[bytes, int] = {}
    setdefault = seen.setdefault
    expanded_to_unique = np.fromiter(
        (setdefault(key, len(seen)) for key in keys),
        dtype=np.int64,
        count=n_rows,
    )

    # A row is a first occurrence exactly when its label is higher than every
    # label before it, because the labels rise by one on each new row.
    is_first = np.empty(n_rows, dtype=bool)
    is_first[0] = True
    np.greater(
        expanded_to_unique[1:],
        np.maximum.accumulate(expanded_to_unique)[:-1],
        out=is_first[1:],
    )
    unique_samples = samples[np.flatnonzero(is_first)]
    return unique_samples, expanded_to_unique
