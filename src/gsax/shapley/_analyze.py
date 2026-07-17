"""Shapley effects derived from fitted PCE and HDMR results."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array

from gsax.shapley._engine import build_membership, shapley_from_variances
from gsax.shapley._result import ShapleyResult

if TYPE_CHECKING:
    from gsax.hdmr import HDMRResult
    from gsax.pce import PCEResult

_POORFIT_THRESHOLD = 0.5
_OVERFIT_THRESHOLD = 1.3


def _normalize_partial_variances(partial: Array, explained_variance: Array) -> Array:
    """Normalize term contributions while preserving degenerate output slices."""
    total = partial.sum(axis=-1, keepdims=True)
    invalid = (
        (total == 0)
        | ~jnp.isfinite(total)
        | ~jnp.isfinite(jnp.asarray(explained_variance)[..., None])
    )
    return jnp.where(invalid, jnp.nan, partial / jnp.where(invalid, 1.0, total))


def _warn_pathological_fit(explained_variance: Array) -> None:
    """Warn when a surrogate captured implausibly little or too much variance."""
    ev = jnp.asarray(explained_variance)
    if bool(jnp.any(ev > _OVERFIT_THRESHOLD)):
        warnings.warn(
            f"gsax: surrogate explained_variance exceeds {_OVERFIT_THRESHOLD}; "
            "Shapley effects may be unreliable",
            stacklevel=3,
        )
    elif bool(jnp.any(ev < _POORFIT_THRESHOLD)):
        warnings.warn(
            f"gsax: surrogate explained_variance is below {_POORFIT_THRESHOLD}; "
            "Shapley effects may be unreliable",
            stacklevel=3,
        )


def _shapley_from_pce(result: "PCEResult") -> ShapleyResult:
    """Compute Shapley effects from fitted orthogonal polynomial coefficients."""
    partial = result.coefficients[..., 1:] ** 2
    membership = np.asarray(result.multi_index[1:] > 0)
    explained = result.explained_variance
    if explained is None:
        raise ValueError("PCEResult does not contain explained-variance diagnostics")
    normalized = _normalize_partial_variances(partial, explained)
    Sh, S1, ST = shapley_from_variances(normalized, membership)
    _warn_pathological_fit(explained)
    return ShapleyResult(
        Sh=Sh,
        S1=S1,
        ST=ST,
        problem=result.problem,
        backend="pce",
        explained_variance=explained,
        order=result.order,
    )


def _shapley_from_hdmr(
    result: "HDMRResult",
    *,
    include_correlative: bool,
) -> ShapleyResult:
    """Compute structural or correlation-aware Shapley effects from HDMR terms."""
    partial = result.Sa + result.Sb if include_correlative else result.Sa
    subsets: list[tuple[int, ...]] = [(i,) for i in range(result.problem.num_vars)]
    subsets.extend(result._c2)
    subsets.extend(result._c3)
    membership = build_membership(subsets, result.problem.num_vars)
    explained = partial.sum(axis=-1)
    normalized = _normalize_partial_variances(partial, explained)
    Sh, S1, ST = shapley_from_variances(normalized, membership)
    _warn_pathological_fit(explained)
    fit = result._fit
    if fit is None:
        raise ValueError("HDMRResult does not contain fitted surrogate state")
    return ShapleyResult(
        Sh=Sh,
        S1=S1,
        ST=ST,
        problem=result.problem,
        backend="hdmr",
        explained_variance=explained,
        order=fit["maxorder"],
        include_correlative=include_correlative,
    )
