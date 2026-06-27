"""PAWN analysis: distribution-based sensitivity via KS distances.

Computes the Kolmogorov-Smirnov distance between the unconditional
output CDF and the conditional CDF when each input is fixed in a bin.
The PAWN index aggregates KS values across bins via median, max, or mean.

References:
    Pianosi & Wagener (2015). A simple and efficient method for global
    sensitivity analysis based on cumulative distribution functions.
    Environmental Modelling & Software 67:1-11.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from gsax._normalization import _prepare_Y
from gsax._transforms import cdf_to_unit_interval
from gsax.pawn._result import PAWNResult
from gsax.problem import Problem


def _ks_stat(y_uncond_sorted: Array, y_cond: Array) -> Array:
    """KS statistic between unconditional and conditional samples.

    Uses a sort-merge approach (like ``scipy.stats.ks_2samp``) that
    evaluates both ECDFs at every sample point from both distributions.

    Args:
        y_uncond_sorted: Sorted unconditional output samples ``(M,)``.
        y_cond: Conditional output samples ``(C,)``.

    Returns:
        Scalar KS statistic.
    """
    n1 = y_uncond_sorted.shape[0]
    n2 = y_cond.shape[0]
    y_cond_sorted = jnp.sort(y_cond)
    all_sorted = jnp.sort(jnp.concatenate([y_uncond_sorted, y_cond_sorted]))
    cdf1 = jnp.searchsorted(y_uncond_sorted, all_sorted, side="right") / n1
    cdf2 = jnp.searchsorted(y_cond_sorted, all_sorted, side="right") / n2
    return jnp.max(jnp.abs(cdf1 - cdf2))


def _pawn_single_output(
    y_uncond_sorted: Array,
    X_u01: Array,
    y: Array,
    n_bins: int,
    statistic: Literal["median", "max", "mean"],
) -> Array:
    """Compute PAWN indices for a single output slice.

    Args:
        y_uncond_sorted: Sorted unconditional outputs ``(M,)``.
        X_u01: Unit-interval inputs ``(N, D)``.
        y: Output samples ``(N,)``.
        n_bins: Number of equal-width bins in [0, 1].
        statistic: Aggregation across bins.

    Returns:
        PAWN indices ``(D,)``.
    """
    D = X_u01.shape[1]
    bin_edges = jnp.linspace(0.0, 1.0, n_bins + 1)

    indices = []
    for d in range(D):
        ks_values = []
        for b in range(n_bins):
            lo = bin_edges[b]
            hi = bin_edges[b + 1]
            if b == n_bins - 1:
                mask = (X_u01[:, d] >= lo) & (X_u01[:, d] <= hi)
            else:
                mask = (X_u01[:, d] >= lo) & (X_u01[:, d] < hi)
            y_bin = y[mask]
            if y_bin.shape[0] < 2:
                continue
            ks = _ks_stat(y_uncond_sorted, y_bin)
            ks_values.append(ks)

        if len(ks_values) == 0:
            indices.append(jnp.float32(0.0))
            continue

        ks_arr = jnp.stack(ks_values)
        if statistic == "median":
            indices.append(jnp.median(ks_arr))
        elif statistic == "max":
            indices.append(jnp.max(ks_arr))
        else:
            indices.append(jnp.mean(ks_arr))

    return jnp.stack(indices)


def _pawn_core(
    problem: Problem,
    X: Array,
    Y_3d: Array,
    n_bins: int,
    statistic: Literal["median", "max", "mean"],
) -> Array:
    """Core PAWN computation over all (T, K) output slices.

    Args:
        problem: Problem definition.
        X: Input samples ``(N, D)`` in physical units.
        Y_3d: Output array promoted to ``(N, T, K)``.
        n_bins: Number of bins per input dimension.
        statistic: Aggregation method across bins.

    Returns:
        PAWN indices ``(T, K, D)``.
    """
    _, T, K = Y_3d.shape
    D = problem.num_vars
    X_u01 = cdf_to_unit_interval(X, problem)

    result = jnp.zeros((T, K, D))

    for t in range(T):
        for k in range(K):
            y = Y_3d[:, t, k]
            y_uncond_sorted = jnp.sort(y)
            pawn_tk = _pawn_single_output(y_uncond_sorted, X_u01, y, n_bins, statistic)
            result = result.at[t, k, :].set(pawn_tk)

    return result


def analyze(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_bins: int = 10,
    statistic: Literal["median", "max", "mean"] = "median",
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    seed: int = 0,
    chunk_size: int = 2048,
) -> PAWNResult:
    """Compute PAWN sensitivity indices.

    Args:
        problem: Problem definition with D parameters.
        X: Input sample matrix ``(N, D)``.
        Y: Model output ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        n_bins: Number of equal-width conditioning bins per input.
        statistic: Aggregation of KS values across bins
            (``"median"``, ``"max"``, or ``"mean"``).
        n_bootstrap: Number of bootstrap resamples for confidence
            intervals. Set to 0 to skip.
        conf_level: Confidence level for bootstrap intervals.
        seed: Random seed for bootstrap resampling.
        chunk_size: Unused, kept for API consistency.

    Returns:
        PAWNResult with PAWN indices and optional confidence intervals.
    """
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (N, D), got ndim={X.ndim}")
    if X.shape[1] != problem.num_vars:
        raise ValueError(
            f"X has {X.shape[1]} columns but problem has {problem.num_vars} parameters"
        )
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but Y has {Y.shape[0]} rows")
    if statistic not in ("median", "max", "mean"):
        raise ValueError(f"statistic must be 'median', 'max', or 'mean', got {statistic!r}")

    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)

    pawn_3d = _pawn_core(problem, X, Y_3d, n_bins, statistic)

    pawn_conf: Array | None = None
    if n_bootstrap > 0:
        key = jax.random.PRNGKey(seed + 1)
        N = X.shape[0]
        boot_draws = []
        for _ in range(n_bootstrap):
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, N, shape=(N,), replace=True)
            X_boot = X[idx]
            Y_boot = Y_3d[idx]
            boot_pawn = _pawn_core(problem, X_boot, Y_boot, n_bins, statistic)
            boot_draws.append(boot_pawn)

        boot_stack = jnp.stack(boot_draws, axis=0)
        alpha = (1.0 - conf_level) / 2.0
        percentiles = jnp.array([alpha * 100, (1.0 - alpha) * 100])
        pawn_conf_3d = jnp.nanpercentile(boot_stack, percentiles, axis=0)

        if squeeze_time and squeeze_output:
            pawn_conf = pawn_conf_3d[:, 0, 0, :]
        elif squeeze_time:
            pawn_conf = pawn_conf_3d[:, 0, :, :]
        else:
            pawn_conf = pawn_conf_3d

    if squeeze_time and squeeze_output:
        pawn_out = pawn_3d[0, 0, :]
    elif squeeze_time:
        pawn_out = pawn_3d[0, :, :]
    else:
        pawn_out = pawn_3d

    return PAWNResult(
        pawn=pawn_out,
        pawn_conf=pawn_conf,
        problem=problem,
    )
