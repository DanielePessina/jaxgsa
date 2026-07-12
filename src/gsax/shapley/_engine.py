"""Shapley-effect aggregation from ANOVA subset variances.

Implements the exact Shapley-value allocation of output variance across
inputs (Owen, 2014): given the partial variance ``V_u`` of every modelled
ANOVA subset ``u``, the Shapley effect of input ``i`` is
``Sh_i = sum_{u : i in u} V_u / |u|`` -- each term's variance is split
equally among its participants. The aggregation is a single weighted
matrix product, batched over any leading output dimensions.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array


def build_membership(subsets: Sequence[tuple[int, ...]], D: int) -> np.ndarray:
    """Build the boolean membership matrix of ANOVA terms over parameters.

    Args:
        subsets: One parameter-index tuple per modelled ANOVA term,
            e.g. ``[(0,), (1,), (2,), (0, 1), (0, 2)]``.
        D: Number of parameters.

    Returns:
        ``(n_terms, D)`` boolean array where entry ``[t, d]`` marks whether
        parameter ``d`` participates in term ``t``.
    """
    membership = np.zeros((len(subsets), D), dtype=bool)
    for t, u in enumerate(subsets):
        membership[t, list(u)] = True
    return membership


def shapley_from_variances(
    V: Array,
    membership: np.ndarray,
) -> tuple[Array, Array, Array]:
    """Aggregate per-term variance fractions into Shapley, S1, and ST indices.

    Args:
        V: ``(..., n_terms)`` variance fraction of each modelled ANOVA term,
            already normalized by the total output variance. Leading
            dimensions (output, time) are batched through unchanged.
        membership: ``(n_terms, D)`` boolean membership matrix from
            :func:`build_membership`.

    Returns:
        Tuple ``(Sh, S1, ST)``, each of shape ``(..., D)``:
            Sh: Shapley effects ``sum_{u:i in u} V_u / |u|``.
            S1: First-order indices (singleton terms only).
            ST: Total-order indices ``sum_{u:i in u} V_u``.
    """
    card = membership.sum(axis=1)  # |u| per term
    M = jnp.asarray(membership, dtype=V.dtype)
    # The three indices only differ in how a term's variance is credited to
    # its participants: split evenly (Shapley), singletons only (S1), or in
    # full (ST) -- so all three are weighted variants of the same matmul.
    Sh = (V / jnp.asarray(card, dtype=V.dtype)) @ M
    S1 = V @ jnp.asarray(membership & (card[:, None] == 1), dtype=V.dtype)
    ST = V @ M
    return Sh, S1, ST
