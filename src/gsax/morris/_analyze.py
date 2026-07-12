"""Morris elementary-effects analysis using JAX.

Reconstructs the expanded Morris design from unique model outputs, extracts
one elementary effect per trajectory and parameter with a single
gather-subtract-divide, and reduces them to the screening measures mu,
mu_star, and sigma (optionally with bootstrap confidence intervals over
trajectories).

Array shape conventions used throughout:
    r  — number of trajectories (after cleaning)
    D  — number of input parameters
    T  — number of time steps (singleton-squeezed when absent)
    K  — number of output variables (singleton-squeezed when absent)
    R  — number of bootstrap resamples
"""

import warnings
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from gsax._bootstrap import _bootstrap_ci_endpoints
from gsax._normalization import (
    _infer_output_layout,
    _prenormalize_outputs,
    _prepare_Y,
    _squeeze_output_axes,
)
from gsax.morris._result import MorrisResult
from gsax.morris._sampling import MorrisSamplingResult

# Minimum trajectories for statistically meaningful screening measures;
# only enforced as a warning when non-finite cleaning shrinks the design.
_MIN_TRAJECTORIES = 10

# Peak-memory budget (in array elements) for one bootstrap chunk. Each resample
# gathers a full (r, D, T, K) copy of the elementary effects, so the batch size
# is capped by output volume — not just the resample count — to keep
# multi-output / time-series runs from exhausting device memory.
_BOOTSTRAP_ELEMENT_BUDGET = 64_000_000  # ~256 MB at float32


def _stats_from_ee(ee: Array) -> tuple[Array, Array, Array]:
    """Reduce elementary effects to (mu, mu_star, sigma) over trajectories.

    Args:
        ee: Elementary effects, shape (r, D, T, K).

    Returns:
        Tuple of (T, K, D) arrays: mean, mean absolute value, and sample
        standard deviation (ddof=1) of the r effects per parameter.
    """
    mu = jnp.mean(ee, axis=0)
    mu_star = jnp.mean(jnp.abs(ee), axis=0)
    sigma = jnp.std(ee, axis=0, ddof=1)
    # (D, T, K) -> (T, K, D) to match the package's index-array convention
    return (
        jnp.moveaxis(mu, 0, -1),
        jnp.moveaxis(mu_star, 0, -1),
        jnp.moveaxis(sigma, 0, -1),
    )


# @jax.jit is applied directly (not via lru_cache) because these functions
# have a fixed signature — no configuration parameter to dispatch on.


@jax.jit
def _elementary_effects(Y: Array, idx_after: Array, idx_before: Array, delta: Array) -> Array:
    """Gather elementary effects from expanded outputs in one fused op.

    Args:
        Y: Expanded model outputs, shape (r * (D + 1), T, K).
        idx_after: (r, D) expanded-row indices of the perturbed points.
        idx_before: (r, D) expanded-row indices of the reference points.
        delta: (r, D) signed unit-cube steps.

    Returns:
        Elementary effects, shape (r, D, T, K).
    """
    # Fancy indexing with (r, D) index arrays gathers (r, D, T, K) blocks;
    # the signed delta broadcasts over the trailing output dimensions.
    return (Y[idx_after] - Y[idx_before]) / delta[:, :, None, None]


@jax.jit
def _resample_stats(idx_chunk: Array, ee: Array) -> tuple[Array, Array, Array]:
    """Vectorised Morris statistics for one chunk of bootstrap resamples.

    Args:
        idx_chunk: (C, r) trajectory index sets for this chunk, each row
            containing r indices in [0, r).
        ee: (r, D, T, K) elementary effects.

    Returns:
        Tuple of (C, T, K, D) arrays (mu, mu_star, sigma) per resample.
    """

    # Closure over ee lets vmap vary only the index vector per resample;
    # ee[idx] gathers r trajectories with replacement.
    def single(idx: Array) -> tuple[Array, Array, Array]:
        return _stats_from_ee(ee[idx])

    return jax.vmap(single)(idx_chunk)


def _drop_nonfinite_trajectories(
    Y: Array, sampling_result: MorrisSamplingResult
) -> tuple[Array, Array, Array, Array, int]:
    """Drop entire trajectories that contain any non-finite (NaN/Inf) value.

    A trajectory is an indivisible sampling unit: a single bad row corrupts
    the elementary effects that reference it, so the whole block of D+1 rows
    is removed and the bookkeeping indices are rebuilt against the compacted
    layout.

    Args:
        Y: Expanded model outputs, shape (r * (D + 1), T, K).
        sampling_result: Sampling metadata with elementary-effect bookkeeping.

    Returns:
        ``(Y_clean, idx_after, idx_before, delta, n_dropped)`` where the
        bookkeeping arrays are re-indexed into the cleaned layout.
    """
    r = sampling_result.n_trajectories
    rows_per_traj = sampling_result.n_params + 1
    trailing = Y.shape[1:]

    grouped = Y.reshape(r, rows_per_traj, *trailing)
    finite_mask = jnp.all(jnp.isfinite(grouped.reshape(r, -1)), axis=1)
    n_good = int(jnp.sum(finite_mask))
    n_dropped = r - n_good

    idx_after = jnp.asarray(sampling_result.ee_idx_after)
    idx_before = jnp.asarray(sampling_result.ee_idx_before)
    delta = jnp.asarray(sampling_result.ee_delta)

    if n_dropped == 0:
        return Y, idx_after, idx_before, delta, 0

    # finite_mask is a small (r,) host array; keep the large output tensor on
    # device by gathering the surviving trajectory blocks with jnp.take.
    keep = np.flatnonzero(np.asarray(finite_mask))
    Y_clean = jnp.take(grouped, jnp.asarray(keep), axis=0).reshape(
        n_good * rows_per_traj, *trailing
    )

    # Global indices encode trajectory-block offsets; recompute them for the
    # compacted layout: local offset within the block is invariant.
    old_offsets = (np.asarray(sampling_result.ee_idx_after) // rows_per_traj) * rows_per_traj
    local_after = np.asarray(sampling_result.ee_idx_after) - old_offsets
    local_before = np.asarray(sampling_result.ee_idx_before) - old_offsets
    new_offsets = np.arange(n_good)[:, None] * rows_per_traj
    idx_after = jnp.asarray(local_after[keep] + new_offsets)
    idx_before = jnp.asarray(local_before[keep] + new_offsets)
    delta = jnp.asarray(np.asarray(sampling_result.ee_delta)[keep])

    return Y_clean, idx_after, idx_before, delta, n_dropped


def _expand_unique_outputs(sampling_result: MorrisSamplingResult, Y: Array) -> Array:
    """Rebuild expanded Morris outputs from unique user-evaluated outputs."""
    if Y.shape[0] != sampling_result.n_total:
        raise ValueError(
            f"Y.shape[0] must match sampling_result.n_total ({sampling_result.n_total}), "
            f"got {Y.shape[0]}"
        )
    expanded_to_unique = jnp.asarray(sampling_result.expanded_to_unique)
    return jnp.take(Y, expanded_to_unique, axis=0)


def analyze(
    sampling_result: MorrisSamplingResult,
    Y: Array,
    *,
    prenormalize: bool = False,
    num_resamples: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    chunk_size: int = 2048,
) -> MorrisResult:
    """Compute Morris elementary-effects screening measures using JAX.

    Accepts model outputs Y evaluated at the unique rows returned by
    ``gsax.sample_morris()``, reconstructs the expanded design internally,
    drops trajectories containing non-finite values, and reduces one
    elementary effect per trajectory and parameter to three measures:
    rank parameters by ``mu_star`` (mean absolute effect, the headline
    importance measure); a ``sigma`` (spread of effects) that is large
    relative to ``mu_star`` flags nonlinearity or interactions; ``mu``
    keeps the effect sign but can cancel for non-monotonic responses.

    Elementary effects are computed in unit-cube coordinates, so ``mu_star``
    is directly comparable across parameters regardless of their physical
    ranges; use :meth:`MorrisResult.to_physical_units` for derivative-scale
    values (uniform-marginal problems only — it raises for problems with
    Gaussian marginals, whose inverse-CDF transform is nonlinear).

    Args:
        sampling_result: Result from ``gsax.sample_morris()`` containing the
            unique sample matrix plus elementary-effect bookkeeping.
        Y: Model outputs evaluated at each unique row of
            ``sampling_result.samples``. Accepted shapes:
                (n_total,)       — scalar output, single time step
                (n_total, K)     — K outputs, single time step
                (n_total, T, K)  — K outputs over T time steps
            where ``n_total`` is the unique row count. Any other number of
            dimensions raises ``ValueError``.
        prenormalize: When ``True``, standardize each output slice to mean 0
            and unit standard deviation over the expanded sample axis before
            computing elementary effects, making measures comparable across
            outputs with different magnitudes. Defaults to ``False``.
        num_resamples: R, the number of bootstrap resamples (over
            trajectories, with replacement) for confidence intervals.
            Set to 0 (default) to skip bootstrap.
        conf_level: Confidence level for bootstrap CIs (default 0.95).
        ci_method: Bootstrap CI endpoint method, ``"quantile"`` (empirical
            percentiles) or ``"gaussian"`` (symmetric around the estimate).
        key: JAX PRNG key for bootstrap randomness. Required when
            ``num_resamples > 0``.
        chunk_size: Upper bound on bootstrap resamples processed per vmap
            batch. The effective batch is reduced further for large
            multi-output / time-series outputs so peak memory stays bounded.
            Defaults to 2048.

    Returns:
        MorrisResult containing:
            mu       — mean elementary effect, shape (D,) / (K, D) / (T, K, D)
            mu_star  — mean absolute elementary effect, same shape
            sigma    — standard deviation of elementary effects, same shape
            mu_conf, mu_star_conf, sigma_conf — (2, ...) CI bounds or None

    Raises:
        ValueError: If ``Y`` does not have 1, 2, or 3 dimensions; if ``Y``'s
            first axis does not match ``sampling_result.n_total``; if fewer
            than 2 trajectories survive non-finite cleaning; if ``ci_method``
            is invalid; if ``num_resamples > 0`` but ``key`` is ``None``; or
            if ``chunk_size < 1``.
    """
    if ci_method not in {"quantile", "gaussian"}:
        raise ValueError("ci_method must be one of {'quantile', 'gaussian'}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    Y = jnp.asarray(Y)
    if Y.ndim not in (1, 2, 3):
        raise ValueError(
            "Y must have 1, 2, or 3 dimensions — (n_total,), (n_total, K), or "
            f"(n_total, T, K); got a {Y.ndim}-D array of shape {Y.shape}"
        )
    # Resolve the user-supplied layout (sample axis first, labeled output axis
    # last) against the unique design rows, BEFORE expansion and bootstrap so
    # every downstream stage sees canonical axes.
    Y = _infer_output_layout(
        Y, sampling_result.problem, int(sampling_result.samples.shape[0])
    )
    # Map user-evaluated unique outputs back to the full expanded layout
    Y = _expand_unique_outputs(sampling_result, Y)
    Y, squeeze_time, squeeze_output = _prepare_Y(Y)

    Y, idx_after, idx_before, delta, n_dropped = _drop_nonfinite_trajectories(Y, sampling_result)
    if n_dropped > 0:
        remaining = sampling_result.n_trajectories - n_dropped
        pct = 100.0 * n_dropped / sampling_result.n_trajectories
        warnings.warn(
            f"gsax: dropped {n_dropped} of {sampling_result.n_trajectories} trajectories "
            f"({pct:.1f}%) containing non-finite values; {remaining} trajectories remain",
            stacklevel=2,
        )
        if remaining < 2:
            raise ValueError("Fewer than 2 trajectories remain after dropping non-finite values")
        if remaining < _MIN_TRAJECTORIES:
            warnings.warn(
                f"gsax: only {remaining} trajectories remain after dropping non-finite "
                f"values — results may be statistically unreliable "
                f"(recommend >= {_MIN_TRAJECTORIES})",
                stacklevel=2,
            )

    if prenormalize:
        Y, _, _, _ = _prenormalize_outputs(Y)

    # Constant output slices yield elementary effects of exactly 0 (0/delta),
    # so the screening measures come out 0 — not NaN as in variance-based
    # methods. Warn with the Morris-correct consequence, and report a plain
    # slice count rather than fabricating (t, k) labels for the singleton time
    # axis that _prepare_Y inserts.
    slice_var = jnp.var(Y.reshape(Y.shape[0], -1), axis=0)
    n_zero = int(jnp.sum(slice_var == 0))
    if n_zero > 0:
        which = (
            "output has"
            if slice_var.shape[0] == 1
            else f"{n_zero}/{slice_var.shape[0]} output slice(s) have"
        )
        warnings.warn(
            f"gsax: {which} zero variance — the corresponding screening "
            "measures (mu, mu_star, sigma) will be 0",
            stacklevel=2,
        )

    ee = _elementary_effects(Y, idx_after, idx_before, delta)  # (r, D, T, K)
    mu, mu_star, sigma = _stats_from_ee(ee)  # each (T, K, D)

    mu_conf = mu_star_conf = sigma_conf = None
    if num_resamples > 0:
        if key is None:
            raise ValueError("key is required when num_resamples > 0")
        r = ee.shape[0]
        # Pre-generate all R bootstrap index sets (sampling with replacement)
        indices = jax.random.randint(key, shape=(num_resamples, r), minval=0, maxval=r)

        # Cap the batch by both the user's chunk_size and an element budget:
        # one chunk materialises cs copies of the (r, D, T, K) effects tensor,
        # so large T*K would otherwise blow past device memory at chunk_size.
        per_sample = int(np.prod(ee.shape))  # r * D * T * K
        mem_cap = max(1, _BOOTSTRAP_ELEMENT_BUDGET // max(per_sample, 1))
        cs = max(1, min(chunk_size, num_resamples, mem_cap))
        mu_parts, mu_star_parts, sigma_parts = [], [], []
        for start in range(0, num_resamples, cs):
            end = min(start + cs, num_resamples)
            n_real = end - start
            idx_chunk = indices[start:end]
            if n_real < cs:
                # Pad the final ragged chunk back up to cs so _resample_stats
                # only ever compiles one shape; the padding rows are sliced off.
                pad = jnp.broadcast_to(idx_chunk[:1], (cs - n_real, idx_chunk.shape[1]))
                idx_chunk = jnp.concatenate([idx_chunk, pad], axis=0)
            m, ms, sd = _resample_stats(idx_chunk, ee)
            mu_parts.append(m[:n_real])
            mu_star_parts.append(ms[:n_real])
            sigma_parts.append(sd[:n_real])

        conf_pairs = []
        for estimate, parts in [
            (mu, mu_parts),
            (mu_star, mu_star_parts),
            (sigma, sigma_parts),
        ]:
            draws = jnp.concatenate(parts)  # (R, T, K, D)
            lower, upper = _bootstrap_ci_endpoints(
                estimate, draws, conf_level=conf_level, ci_method=ci_method
            )
            conf_pairs.append(jnp.stack([lower, upper]))
        mu_conf, mu_star_conf, sigma_conf = conf_pairs

    mu = _squeeze_output_axes(mu, squeeze_time, squeeze_output)
    mu_star = _squeeze_output_axes(mu_star, squeeze_time, squeeze_output)
    sigma = _squeeze_output_axes(sigma, squeeze_time, squeeze_output)
    if mu_conf is not None and mu_star_conf is not None and sigma_conf is not None:
        mu_conf = _squeeze_output_axes(mu_conf, squeeze_time, squeeze_output)
        mu_star_conf = _squeeze_output_axes(mu_star_conf, squeeze_time, squeeze_output)
        sigma_conf = _squeeze_output_axes(sigma_conf, squeeze_time, squeeze_output)

    return MorrisResult(
        mu=mu,
        mu_star=mu_star,
        sigma=sigma,
        problem=sampling_result.problem,
        mu_conf=mu_conf,
        mu_star_conf=mu_star_conf,
        sigma_conf=sigma_conf,
        space="unit",
    )
