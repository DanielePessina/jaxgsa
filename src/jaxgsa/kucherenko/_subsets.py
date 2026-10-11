"""Allocation rules over subset costs: exact Shapley effects and PME.

A subset cost ``c(u)`` is a number for every subset ``u`` of the ``D`` inputs.
Here it is a closed or a total Sobol' index. A subset is stored as a bitmask:
bit ``j`` is parameter ``j``. A cost array holds the masks ``1 .. 2^D - 1`` on
its last axis, so mask ``m`` sits at position ``m - 1``. The empty set has
cost 0 and is added inside these functions.

Two rules turn a cost into one share per input:

* The Shapley effect (Owen 2014, eq. 2.1). It weights the marginal gain of
  input ``i`` over every subset ``u`` that does not hold ``i``.
* The proportional marginal effect, the proportional value of the total-index
  game (Herin, Il Idrissi, Chabridon & Iooss 2024, Def. 4, Eqs. 12 and 15).

Both functions take any number of leading axes, so a bootstrap replicate axis
passes through unchanged.

References:
    Owen (2014). SIAM/ASA J. Uncertain. Quantif. 2:245-251.
        doi:10.1137/130936233
    Song, Nelson & Staum (2016). SIAM/ASA J. Uncertain. Quantif. 4:1060-1083.
        doi:10.1137/15M1048070
    Herin, Il Idrissi, Chabridon & Iooss (2024). SIAM/ASA J. Uncertain.
        Quantif. 12(2):667-692. doi:10.1137/22M153032X. arXiv:2210.13065.
"""

from __future__ import annotations

from math import factorial

import numpy as np


def subset_matrix(n_params: int) -> np.ndarray:
    """Return the ``(2^D - 1, D)`` boolean membership matrix of the subsets.

    Row ``m - 1`` is the subset with bitmask ``m``, for ``m = 1 .. 2^D - 1``.
    Entry ``[m - 1, j]`` is ``True`` when parameter ``j`` is in the subset.
    """
    masks = np.arange(1, 2**n_params)
    return ((masks[:, None] >> np.arange(n_params)[None, :]) & 1).astype(bool)


def _full_cost(cost: np.ndarray, n_params: int) -> np.ndarray:
    """Flatten the leading axes and prepend the empty set: shape ``(2^D, B)``."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.shape[-1] != 2**n_params - 1:
        raise ValueError(
            f"cost has {cost.shape[-1]} subsets on its last axis, expected {2**n_params - 1}"
        )
    flat = cost.reshape(-1, cost.shape[-1]).T
    return np.concatenate([np.zeros((1, flat.shape[1])), flat], axis=0)


def shapley_from_cost(cost: np.ndarray, n_params: int) -> np.ndarray:
    """Exact Shapley effects of a subset cost.

    Computes ``Sh_i = sum_{u not containing i} |u|! (D-|u|-1)! / D! *
    [c(u + i) - c(u)]`` (Owen 2014, Eq. 2.1) with ``c(empty) = 0``. The
    closed-index cost and the total-index cost give the same effects
    (Song, Nelson & Staum 2016), because the total cost is the dual game
    ``c_tot(u) = c(D) - c_clos(D - u)``, and the Shapley value of a game
    equals that of its dual.

    Args:
        cost: Cost per subset, shape ``(..., 2^D - 1)``. The last entry is the
            full set.
        n_params: Number of parameters ``D``.

    Returns:
        Shapley effects, shape ``(..., D)``. They sum to ``cost[..., -1]``.
    """
    cost = np.asarray(cost, dtype=np.float64)
    lead = cost.shape[:-1]
    full = _full_cost(cost, n_params)
    all_masks = np.arange(2**n_params)
    popcount = np.zeros(2**n_params, dtype=np.int64)
    for j in range(n_params):
        popcount += (all_masks >> j) & 1
    fact = np.array([factorial(k) for k in range(n_params + 1)], dtype=np.float64)
    weight = fact[popcount] * fact[np.maximum(n_params - popcount - 1, 0)] / fact[n_params]
    out = np.empty((n_params, full.shape[1]))
    for i in range(n_params):
        without = all_masks[((all_masks >> i) & 1) == 0]
        gain = full[without | (1 << i)] - full[without]
        out[i] = (weight[without][:, None] * gain).sum(axis=0)
    return out.T.reshape(*lead, n_params)


def _ratio_potential(value: np.ndarray, ground: int, offset: int, n_params: int) -> np.ndarray:
    """Ratio potential ``R(B, v_A)`` of Herin et al. (2024), Eq. 12, for every ``B``.

    ``R(empty) = 1`` and ``R(B) = v(A + B) / sum_{j in B} 1 / R(B - j)``. The
    game is ``v_A(B) = v(A + B)`` on the players of ``ground``. It needs
    ``v(A + B) > 0`` for every non-empty ``B``.

    Args:
        value: Cost per subset mask, shape ``(2^D, B)``, with ``value[0] = 0``.
        ground: Bitmask of the players of the game.
        offset: Bitmask of ``A``, the coalition added to every ``B``.
        n_params: Number of parameters ``D``.

    Returns:
        An array of shape ``(2^D, B)`` whose row ``m`` holds ``R(m)`` for every
        submask ``m`` of ``ground``. Other rows are unset.
    """
    ratio = np.ones_like(value)
    submasks = [m for m in range(1, ground + 1) if m & ~ground == 0]
    for m in submasks:
        total = np.zeros(value.shape[1])
        for j in range(n_params):
            if (m >> j) & 1:
                total += 1.0 / ratio[m ^ (1 << j)]
        ratio[m] = value[offset | m] / total
    return ratio


def _proportional_values(value: np.ndarray, n_params: int) -> np.ndarray:
    """Proportional values of one or more games, shape ``(2^D, B)`` -> ``(D, B)``.

    Columns with no zero-valued non-empty subset use the plain definition,
    ``PV_i = R(D) / R(D - i)`` (Eq. 12), in one vectorised pass. Other columns
    use the continuous extension to nonnegative games (Theorem 1, Eq. 15), one
    vectorised pass per group of columns with the same null coalitions.
    """
    full_mask = 2**n_params - 1
    batch = value.shape[1]
    out = np.full((n_params, batch), np.nan)
    finite = np.isfinite(value).all(axis=0)
    has_null = (value[1:] <= 0.0).any(axis=0)

    plain = finite & ~has_null
    if plain.any():
        sub = value[:, plain]
        ratio = _ratio_potential(sub, full_mask, 0, n_params)
        for i in range(n_params):
            out[i, plain] = ratio[full_mask] / ratio[full_mask ^ (1 << i)]

    # Columns that share the same null coalitions share the same sets in
    # Eq. 15, so each group runs the recursion once, across its columns.
    extended = np.flatnonzero(finite & has_null)
    if extended.size:
        patterns, group = np.unique(value[:, extended] <= 0.0, axis=1, return_inverse=True)
        for g in range(patterns.shape[1]):
            cols = extended[np.ravel(group) == g]
            out[:, cols] = _extended_proportional_value(value[:, cols], n_params)
    return out


def _extended_proportional_value(value: np.ndarray, n_params: int) -> np.ndarray:
    """Eq. 15 of Herin et al. (2024) for games with one null pattern.

    Shape ``(2^D, B)`` -> ``(D, B)``. Every column must have the same null
    coalitions (cost exactly 0).

    ``K`` holds the null coalitions of the largest size ``kmax``. Input ``i``
    gets 0 if it is in every member of ``K``. Otherwise it gets
    ``sum_{A in K, i not in A} 1/R(D - i - A, v_A) / sum_{A in K}
    1/R(D - A, v_A)``.
    """
    full_mask = 2**n_params - 1
    sizes = np.array([bin(m).count("1") for m in range(2**n_params)])
    null = value[:, 0] <= 0.0
    kmax = int(sizes[null].max())  # the empty set is null, so this is >= 0
    members = [m for m in range(2**n_params) if null[m] and sizes[m] == kmax]
    numerator = np.zeros((n_params, value.shape[1]))
    denominator = np.zeros(value.shape[1])
    for a in members:
        ground = full_mask & ~a
        ratio = _ratio_potential(value, ground, a, n_params)
        denominator += 1.0 / ratio[ground]
        for i in range(n_params):
            if not (a >> i) & 1:
                numerator[i] += 1.0 / ratio[ground ^ (1 << i)]
    return numerator / denominator


def pme_from_total(total: np.ndarray, n_params: int) -> np.ndarray:
    """Proportional marginal effects from the total-index cost.

    The PME are the proportional values of the game whose value of subset
    ``A`` is the total index ``S^T_A = E[V(Y | X_{D-A})] / V(Y)`` (Herin et
    al. 2024, Def. 4). That is the dual of the closed-index game. A subset
    with cost exactly 0 is a null coalition, and the continuous extension of
    Theorem 1 (Eq. 15) applies. An input that sits in every largest null
    coalition gets a PME of exactly 0. A cost that is tiny but not exactly 0
    gives a tiny but non-zero effect.

    Args:
        total: Total-index cost per subset, shape ``(..., 2^D - 1)``. The last
            entry is the full set and should be 1.
        n_params: Number of parameters ``D``.

    Returns:
        Effects of shape ``(..., D)``. They are nonnegative and sum to
        ``total[..., -1]``. A slice with a non-finite cost gives ``NaN``.
    """
    total = np.asarray(total, dtype=np.float64)
    lead = total.shape[:-1]
    value = _full_cost(total, n_params)
    return _proportional_values(value, n_params).T.reshape(*lead, n_params)
