"""Morris elementary-effects analysis using JAX.

The analysis rebuilds the expanded Morris design from the unique model
outputs. One gather-subtract-divide then extracts one elementary effect per
trajectory and parameter. A final reduction gives the screening measures
``mu``, ``mu_star``, and ``sigma``, with optional bootstrap confidence
intervals resampled over trajectories.

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

from jaxgsa._core.bootstrap import _bootstrap_ci_endpoints
from jaxgsa._core.validation import (
    _prenormalize_outputs,
    _prepare_Y,
    _squeeze_output_axes,
    _validate_output,
)
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.morris._result import MorrisResult
from jaxgsa.morris._sampling import MorrisSamples

# Fewest trajectories that still give statistically meaningful screening
# measures. The analysis reports this floor only when the design lost blocks
# that the user did not give up: blocks with an unmeasurable step, or blocks
# that non-finite cleaning removed. A small design the user asked for is
# deliberate, so it stays silent.
_MIN_TRAJECTORIES = 10

# Peak-memory budget, in array elements, for one bootstrap chunk. Each resample
# gathers a full (r, D, T, K) copy of the elementary effects. The batch size is
# therefore capped by output volume as well as by the resample count, which
# keeps multi-output and time-series runs from exhausting device memory.
_BOOTSTRAP_ELEMENT_BUDGET = 64_000_000  # ~256 MB at float32


def _stats_from_ee(ee: Array) -> tuple[Array, Array, Array]:
    """Reduce elementary effects to mu, mu_star, and sigma over trajectories.

    Args:
        ee: Elementary effects, shape ``(r, D, T, K)``.

    Returns:
        ``(mu, mu_star, sigma)``, each shape ``(T, K, D)``. They hold the mean,
        the mean absolute value, and the sample standard deviation (ddof=1) of
        the r effects of each parameter.
    """
    mu = jnp.mean(ee, axis=0)
    mu_star = jnp.mean(jnp.abs(ee), axis=0)
    sigma = jnp.std(ee, axis=0, ddof=1)
    # Move the parameter axis last, (D, T, K) -> (T, K, D), to match the
    # package's index-array convention.
    return (
        jnp.moveaxis(mu, 0, -1),
        jnp.moveaxis(mu_star, 0, -1),
        jnp.moveaxis(sigma, 0, -1),
    )


# @jax.jit is applied directly, not through lru_cache, because these functions
# have a fixed signature. There is no configuration parameter to dispatch on.


@jax.jit
def _elementary_effects(Y: Array, idx_after: Array, idx_before: Array, delta: Array) -> Array:
    """Gather elementary effects from expanded outputs in one fused op.

    Args:
        Y: Expanded model outputs, shape ``(r * (D + 1), T, K)``.
        idx_after: Expanded-row indices of the perturbed points, shape
            ``(r, D)``.
        idx_before: Expanded-row indices of the reference points, shape
            ``(r, D)``.
        delta: Signed unit-cube steps, shape ``(r, D)``.

    Returns:
        Elementary effects, shape ``(r, D, T, K)``.
    """
    # Fancy indexing with (r, D) index arrays gathers (r, D, T, K) blocks, and
    # the signed delta broadcasts over the trailing output dimensions.
    return (Y[idx_after] - Y[idx_before]) / delta[:, :, None, None]


@jax.jit
def _resample_stats(idx_chunk: Array, ee: Array) -> tuple[Array, Array, Array]:
    """Compute vectorised Morris statistics for one chunk of resamples.

    Args:
        idx_chunk: Trajectory index sets for this chunk, shape ``(C, r)``. Each
            row holds r indices in [0, r).
        ee: Elementary effects, shape ``(r, D, T, K)``.

    Returns:
        ``(mu, mu_star, sigma)``, each shape ``(C, T, K, D)``, one entry per
        resample.
    """

    # The closure over ee lets vmap vary only the index vector per resample.
    # ee[idx] gathers r trajectories with replacement.
    def single(idx: Array) -> tuple[Array, Array, Array]:
        return _stats_from_ee(ee[idx])

    return jax.vmap(single)(idx_chunk)


def _drop_nonfinite_trajectories(
    Y: Array, sampling_result: MorrisSamples
) -> tuple[Array, Array, Array, Array, int]:
    """Drop whole trajectories that hold any non-finite (NaN or Inf) value.

    A trajectory is an indivisible sampling unit. One bad row corrupts every
    elementary effect that references it, so the function removes the whole
    block of D+1 rows. It then rebuilds the bookkeeping indices against the
    compacted layout.

    Args:
        Y: Expanded model outputs, shape ``(r * (D + 1), T, K)``.
        sampling_result: Sampling metadata with elementary-effect bookkeeping.

    Returns:
        ``(Y_clean, idx_after, idx_before, delta, n_dropped)``. The bookkeeping
        arrays are re-indexed into the cleaned layout.
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

    # finite_mask is a small (r,) host array. Gather the surviving trajectory
    # blocks with jnp.take to keep the large output tensor on device.
    keep = np.flatnonzero(np.asarray(finite_mask))
    Y_clean = jnp.take(grouped, jnp.asarray(keep), axis=0).reshape(
        n_good * rows_per_traj, *trailing
    )

    # Global indices encode trajectory-block offsets, so recompute them for the
    # compacted layout. The local offset inside a block does not change.
    old_offsets = (np.asarray(sampling_result.ee_idx_after) // rows_per_traj) * rows_per_traj
    local_after = np.asarray(sampling_result.ee_idx_after) - old_offsets
    local_before = np.asarray(sampling_result.ee_idx_before) - old_offsets
    new_offsets = np.arange(n_good)[:, None] * rows_per_traj
    idx_after = jnp.asarray(local_after[keep] + new_offsets)
    idx_before = jnp.asarray(local_before[keep] + new_offsets)
    delta = jnp.asarray(np.asarray(sampling_result.ee_delta)[keep])

    return Y_clean, idx_after, idx_before, delta, n_dropped


def analyze(
    sampling_result: MorrisSamples,
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

    Pass the model outputs ``Y`` evaluated at the unique rows that
    :func:`jaxgsa.morris.sample` returned. The function rebuilds the expanded
    design internally and drops every trajectory that holds a non-finite
    value. It then reduces one elementary effect per trajectory and parameter
    to three measures. Rank parameters by ``mu_star``, the mean absolute
    effect and the headline importance measure. A ``sigma`` (the spread of the
    effects) that is large next to ``mu_star`` shows nonlinearity or
    interactions. ``mu`` keeps the effect sign, so it can cancel for a
    non-monotonic response.

    The function computes elementary effects in unit-cube coordinates, so
    ``mu_star`` compares directly across parameters whatever their physical
    ranges. Use :meth:`MorrisResult.to_physical_units` for derivative-scale
    values. That method works for uniform marginals only. It raises for a
    problem with Gaussian marginals, because their inverse-CDF transform is
    nonlinear.

    Args:
        sampling_result: Result from :func:`jaxgsa.morris.sample` with the
            unique sample matrix plus elementary-effect bookkeeping.
        Y: Model outputs evaluated at each unique row of
            ``sampling_result.samples``. Accepted shapes:
                (n_runs,)       — scalar output, single time step
                (n_runs, K)     — K outputs, single time step
                (n_runs, T, K)  — K outputs over T time steps
            ``n_runs`` is the unique row count. Any other number of dimensions
            raises ``ValueError``.
        prenormalize: When ``True``, standardize each output slice to mean 0
            and unit standard deviation over the expanded sample axis before
            the elementary effects are computed. This makes the measures
            comparable across outputs of different magnitude. Defaults to
            ``False``.
        num_resamples: R, the number of bootstrap resamples used for the
            confidence intervals. Resampling is over trajectories, with
            replacement. Set to 0 (default) to skip the bootstrap.
        conf_level: Confidence level for the bootstrap intervals (default
            0.95).
        ci_method: Bootstrap endpoint method, ``"quantile"`` (empirical
            percentiles) or ``"gaussian"`` (symmetric around the estimate).
        key: JAX PRNG key for the bootstrap randomness. Required when
            ``num_resamples > 0``.
        chunk_size: Upper bound on the bootstrap resamples processed per vmap
            batch. Large multi-output or time-series outputs reduce the
            effective batch further, which keeps peak memory bounded. Defaults
            to 2048.

    Returns:
        A :class:`MorrisResult` holding:
            mu       — mean elementary effect, shape (D,) / (K, D) / (T, K, D)
            mu_star  — mean absolute elementary effect, same shape
            sigma    — standard deviation of elementary effects, same shape
            mu_conf, mu_star_conf, sigma_conf — (2, ...) CI bounds or None

    Raises:
        ValueError: If ``Y`` does not have 1, 2, or 3 dimensions; if ``Y``'s
            first axis does not match ``sampling_result.n_runs``; if fewer
            than 2 trajectories survive cleaning; if ``ci_method``
            is invalid; if ``num_resamples > 0`` but ``key`` is ``None``; or
            if ``chunk_size < 1``.

    Warns:
        JaxgsaWarning: If the design holds fewer trajectories than the user
            asked for. Non-finite cleaning removes trajectories here, and a
            derived design can already have lost blocks with no measurable
            step. The message names the cause, and it adds a reliability note
            when fewer than 10 trajectories remain. A small design that the
            user asked for is deliberate, so it gives no warning.
        JaxgsaWarning: If an output slice has zero variance.
    """
    if ci_method not in {"quantile", "gaussian"}:
        raise ValueError("ci_method must be one of {'quantile', 'gaussian'}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    Y = _validate_output(
        Y,
        int(sampling_result.samples.shape[0]),
        sampling_result.problem,
    )
    # Map the user-evaluated unique outputs back to the full expanded layout.
    Y = sampling_result.expand_outputs(Y)
    Y, squeeze_time, squeeze_output = _prepare_Y(Y)

    Y, idx_after, idx_before, delta, n_dropped = _drop_nonfinite_trajectories(Y, sampling_result)
    remaining = sampling_result.n_trajectories - n_dropped

    # Warn on the cause of the thinning, not on the surviving count. A small r
    # that the user asked for is a deliberate choice, so it stays silent.
    # Blocks the user never gave up are worth reporting at any count: blocks
    # that SobolSamples.to_morris dropped for having no measurable step, and
    # blocks that non-finite cleaning just removed. The message also carries
    # the reliability floor when the survivors fall below it.
    n_lost_upstream = sampling_result.n_blocks_dropped
    n_lost = n_lost_upstream + n_dropped
    if n_lost > 0:
        requested = remaining + n_lost
        causes = []
        if n_lost_upstream > 0:
            causes.append(
                f"the source design dropped {n_lost_upstream} for having no measurable step"
            )
        if n_dropped > 0:
            pct = 100.0 * n_dropped / sampling_result.n_trajectories
            causes.append(
                f"dropped {n_dropped} of {sampling_result.n_trajectories} ({pct:.1f}%) "
                "containing non-finite values"
            )
        message = (
            f"jaxgsa: {remaining} of the {requested} requested trajectories remain: "
            + "; ".join(causes)
        )
        if remaining < _MIN_TRAJECTORIES:
            message += (
                f" — results may be statistically unreliable (recommend >= {_MIN_TRAJECTORIES})"
            )
        warnings.warn(message, stacklevel=2, category=JaxgsaWarning)

    if remaining < 2:
        raise ValueError("Fewer than 2 trajectories remain after cleaning")

    if prenormalize:
        Y, _, _, _ = _prenormalize_outputs(Y)

    # A constant output slice gives elementary effects of exactly 0 (0/delta),
    # so its screening measures come out 0, not NaN as in the variance-based
    # methods. Warn with that Morris-correct consequence. Report a plain slice
    # count instead of fabricating (t, k) labels for the singleton time axis
    # that _prepare_Y inserts.
    slice_var = jnp.var(Y.reshape(Y.shape[0], -1), axis=0)
    n_zero = int(jnp.sum(slice_var == 0))
    if n_zero > 0:
        which = (
            "output has"
            if slice_var.shape[0] == 1
            else f"{n_zero}/{slice_var.shape[0]} output slice(s) have"
        )
        warnings.warn(
            f"jaxgsa: {which} zero variance — the corresponding screening "
            "measures (mu, mu_star, sigma) will be 0",
            stacklevel=2,
            category=JaxgsaWarning,
        )

    ee = _elementary_effects(Y, idx_after, idx_before, delta)  # (r, D, T, K)
    mu, mu_star, sigma = _stats_from_ee(ee)  # each (T, K, D)

    # One value for all three intervals: they are produced together or not at
    # all, so a single name keeps that fact checkable instead of implied.
    conf_triple: tuple[Array, Array, Array] | None = None
    if num_resamples > 0:
        if key is None:
            raise ValueError("key is required when num_resamples > 0")
        r = ee.shape[0]
        # Pre-generate all R bootstrap index sets, sampling with replacement.
        indices = jax.random.randint(key, shape=(num_resamples, r), minval=0, maxval=r)

        # Cap the batch by the user's chunk_size and by an element budget. One
        # chunk materialises cs copies of the (r, D, T, K) effects tensor, so a
        # large T*K would otherwise exhaust device memory at chunk_size.
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
                # compiles one shape only. The padding rows are sliced off.
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
        conf_triple = (conf_pairs[0], conf_pairs[1], conf_pairs[2])

    mu = _squeeze_output_axes(mu, squeeze_time, squeeze_output)
    mu_star = _squeeze_output_axes(mu_star, squeeze_time, squeeze_output)
    sigma = _squeeze_output_axes(sigma, squeeze_time, squeeze_output)
    if conf_triple is None:
        mu_conf = mu_star_conf = sigma_conf = None
    else:
        mu_conf, mu_star_conf, sigma_conf = (
            _squeeze_output_axes(arr, squeeze_time, squeeze_output) for arr in conf_triple
        )

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
