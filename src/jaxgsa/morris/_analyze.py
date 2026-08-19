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

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.batching import get_memory_budget
from jaxgsa._core.bootstrap import _bootstrap_ci_endpoints
from jaxgsa._core.entry import at_least, in_open_interval, one_of, prepare
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.result import CIInfo
from jaxgsa._core.validation import (
    _prepare_Y,
    _standardize_outputs,
)
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.morris._result import MorrisResult
from jaxgsa.morris._sampling import MorrisSamples

# Fewest trajectories that still give statistically meaningful screening
# measures. The analysis reports this floor only when the design lost blocks
# that the user did not give up, meaning blocks with an unmeasurable step. The
# shared non-finite policy carries its own copy of the floor for the blocks it
# drops. A small design the user asked for is deliberate, so it stays silent.
_MIN_TRAJECTORIES = 10


def _resolve_resample_chunk_size(
    resample_chunk_size: int | None, n_bootstrap: int, bytes_per_replicate: int
) -> int:
    """Resolve how many bootstrap replicates one device call may carry.

    Each replicate gathers a full ``(r, D, T, K)`` copy of the elementary
    effects, so the working set of a chunk grows with output volume as well as
    with the replicate count. An explicit value is an upper bound the caller
    chose and is honoured as given, capped only at the replicate count — the
    same rule as every other batching keyword. The memory budget only sizes
    the ``None`` default, which keeps multi-output and time-series runs from
    exhausting device memory without overriding an explicit choice.

    Args:
        resample_chunk_size: Caller's cap on replicates per chunk, honoured
            as given, or ``None`` to derive one from the memory budget.
        n_bootstrap: R, the number of bootstrap replicates.
        bytes_per_replicate: Transient memory one replicate needs.

    Returns:
        A chunk width in ``[1, n_bootstrap]``.

    Raises:
        ValueError: If ``resample_chunk_size`` is given and is below 1.
    """
    if resample_chunk_size is not None:
        if resample_chunk_size < 1:
            raise ValueError(f"resample_chunk_size must be >= 1, got {resample_chunk_size}")
        return min(resample_chunk_size, n_bootstrap)
    budget = max(1, get_memory_budget() // max(bytes_per_replicate, 1))
    return max(1, min(budget, n_bootstrap))


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


def _ee_bookkeeping(sampling_result: MorrisSamples) -> tuple[Array, Array, Array]:
    """Read the design's elementary-effect bookkeeping onto the device.

    This is the whole traceable half of what :func:`_reindex_after_drop`
    does. The design stores the index maps and the steps as host NumPy, and
    every one of them is a *design* fact rather than a fact about ``Y``, so
    moving them to the device is a constant fold and never a host read of an
    array under trace.

    Args:
        sampling_result: The design from :func:`jaxgsa.morris.sample`.

    Returns:
        ``(idx_after, idx_before, delta)`` as device arrays, in the design's
        own trajectory order and with nothing dropped.
    """
    return (
        jnp.asarray(sampling_result.ee_idx_after),
        jnp.asarray(sampling_result.ee_idx_before),
        jnp.asarray(sampling_result.ee_delta),
    )


def indices(
    sampling_result: MorrisSamples,
    Y: Array,
    *,
    standardize_outputs: bool = False,
) -> tuple[Array, Array, Array]:
    """Compute the Morris screening measures as plain arrays, with no diagnostics.

    This is the transformable core of :func:`analyze`. It runs the same
    gather-subtract-divide on the same data and returns the same numbers, but
    it does nothing else: no non-finite check, no trajectory dropping, no
    zero-variance warning, no thinned-design warning, no bootstrap, no
    :class:`jaxgsa.morris.MorrisResult`, and no read of any array value on
    the host. So it composes with ``jax.jit``, ``jax.vmap``, ``jax.grad`` and
    ``jax.jacrev``, which :func:`analyze` cannot, because a policy decision
    needs a concrete value and a tracer has none.

    The name is ``indices`` for the same reason it is ``indices`` everywhere
    else in jaxgsa: it names the role, the traceable and policy-free core of
    the method. Morris returns screening measures rather than variance
    shares, and the name does not try to say otherwise.

    Point estimates only. Morris's confidence intervals come from a bootstrap
    over trajectories, and which interval to report is a policy choice, so
    ``n_bootstrap``, ``conf_level``, ``ci_method`` and ``key`` stay on
    :func:`analyze`.

    Tier T4 (behavioural contract): the returned arrays must equal ``mu``,
    ``mu_star`` and ``sigma`` of ``analyze``'s result on clean outputs, and
    the function must survive ``jit``, ``vmap`` and ``jit(jacrev(...))``.
    Checked in ``tests/test_morris.py``.

    Use :func:`analyze` for ordinary analysis. Nothing here checks the
    outputs, so a single NaN silently turns whole trajectories of effects
    into NaN.

    Args:
        sampling_result: The design from :func:`jaxgsa.morris.sample`, used
            only for its expansion map and its elementary-effect bookkeeping.
        Y: Model outputs evaluated at each unique row of
            ``sampling_result.samples``, in the same row order. Shapes are
            those :func:`analyze` accepts: ``(n_runs,)``, ``(n_runs, K)`` or
            ``(n_runs, T, K)``.
        standardize_outputs: As in :func:`analyze`. It is kept on the core
            rather than left to the caller for two reasons. It is a pure
            affine rescale over the sample axis, selected by a Python
            ``bool``, so the branch is resolved at trace time and the traced
            graph is the same either way. And the caller cannot reproduce it
            from outside: the standardization reduces over the *expanded*
            sample axis, which this function builds and the caller never
            holds, so a pre-pass on the unique rows would use different means
            and a different scale whenever the design deduplicated a row.

    Returns:
        ``(mu, mu_star, sigma)``, shaped ``(D,)``, ``(K, D)`` or
        ``(T, K, D)`` to match the rank of ``Y``, exactly as ``analyze``
        reports them.

    Raises:
        ValueError: If ``Y``'s first axis does not match
            ``sampling_result.n_runs``.
    """
    Y3, layout = _prepare_Y(sampling_result.expand_outputs(Y))
    if standardize_outputs:
        Y3, _, _, _ = _standardize_outputs(Y3)
    idx_after, idx_before, delta = _ee_bookkeeping(sampling_result)
    ee = _elementary_effects(Y3, idx_after, idx_before, delta)  # (r, D, T, K)
    mu, mu_star, sigma = _stats_from_ee(ee)  # each (T, K, D)
    return layout.squeeze(mu), layout.squeeze(mu_star), layout.squeeze(sigma)


def _reindex_after_drop(
    sampling_result: MorrisSamples, keep: np.ndarray
) -> tuple[Array, Array, Array]:
    """Rebuild the elementary-effect bookkeeping against the compacted layout.

    Removing whole trajectories shifts every later block down, so the global
    row indices stored on the design no longer point at the rows they named.
    The offset of a row *inside* its block does not change, so the fix is to
    strip the old block offset and add the new one.

    Args:
        sampling_result: Sampling metadata with the original bookkeeping.
        keep: Boolean mask of shape ``(r,)``, True for surviving trajectories.

    Returns:
        ``(idx_after, idx_before, delta)``, re-indexed and restricted to the
        surviving trajectories. All three are unchanged when nothing is
        dropped.
    """
    idx_after = np.asarray(sampling_result.ee_idx_after)
    idx_before = np.asarray(sampling_result.ee_idx_before)
    delta = np.asarray(sampling_result.ee_delta)
    if keep.all():
        return _ee_bookkeeping(sampling_result)

    rows_per_traj = sampling_result.n_params + 1
    kept = np.flatnonzero(keep)
    old_offsets = (idx_after // rows_per_traj) * rows_per_traj
    local_after = idx_after - old_offsets
    local_before = idx_before - old_offsets
    new_offsets = np.arange(len(kept))[:, None] * rows_per_traj
    return (
        jnp.asarray(local_after[kept] + new_offsets),
        jnp.asarray(local_before[kept] + new_offsets),
        jnp.asarray(delta[kept]),
    )


def analyze(
    sampling_result: MorrisSamples,
    Y: Array,
    *,
    standardize_outputs: bool = False,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    resample_chunk_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> MorrisResult:
    """Compute Morris elementary-effects screening measures using JAX.

    Pass the model outputs ``Y`` evaluated at the unique rows that
    :func:`jaxgsa.morris.sample` returned. The function rebuilds the expanded
    design internally and checks it for non-finite values under the
    ``on_invalid`` policy. It then reduces one elementary effect per trajectory
    and parameter to three measures. Rank parameters by ``mu_star``, the mean absolute
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

    :func:`jaxgsa.morris.indices` is the transformable core: the same three
    measures as bare arrays, with none of the checks and no bootstrap, so it
    composes with ``jit``, ``vmap`` and ``jacrev``.

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
        standardize_outputs: When ``True``, standardize each output slice to
            mean 0 and unit standard deviation over the expanded sample axis
            before the elementary effects are computed. Morris returns
            dimensional quantities: under ``Y -> a*Y + b`` every one of
            ``mu``, ``mu_star`` and ``sigma`` scales by ``a``. So this keyword
            changes what the numbers mean — they come back in units of the
            output standard deviation, which is what makes slices of
            different magnitude comparable with each other. It does not
            change the ranking within one slice. Defaults to ``False``, which
            reports the measures in the output's own units.
        n_bootstrap: R, the number of bootstrap resamples used for the
            confidence intervals. Resampling is over trajectories, with
            replacement. Set to 0 (default) to skip the bootstrap.
        conf_level: Confidence level for the bootstrap intervals (default
            0.95).
        ci_method: Bootstrap endpoint method, ``"quantile"`` (empirical
            percentiles) or ``"gaussian"`` (symmetric around the estimate).
        key: JAX PRNG key for the bootstrap randomness. Required when
            ``n_bootstrap > 0``.
        resample_chunk_size: Upper bound on the bootstrap replicates processed
            per vmap batch. The unit is bootstrap replicates, not output
            slices. Defaults to ``None``, which takes the width the active
            memory budget (:func:`jaxgsa.config.get_memory_budget`) allows.
            An explicit value is honoured as given, capped only at
            ``n_bootstrap``; the budget never overrides it.
        on_invalid: What to do about non-finite model outputs. The unit here
            is one trajectory, so a single bad value removes the whole block
            of ``D + 1`` rows: the elementary effects are differences between
            neighbours inside the trajectory, and a gap invents an effect that
            was never measured. ``"raise"`` (the default) refuses the sample,
            ``"propagate"`` lets the value reach the measures, and ``"drop"``
            analyzes the surviving trajectories. See
            :mod:`jaxgsa._core.invalid`.
        verbose: If ``True`` (default), print a short summary to stdout: the
            problem and the data, the wall-clock timing, and the top
            parameters by ``mu_star``. Pass ``False`` for a silent run.
        keep_replicates: Keep the per-replicate measures on
            ``MorrisResult.ci.replicates``. Off by default because they are
            large: ``n_bootstrap`` copies of all three measure arrays. Turn
            it on to recompute an interval at another level without
            re-running the analysis.

    Returns:
        A :class:`MorrisResult` holding:
            mu       — mean elementary effect, shape (D,) / (K, D) / (T, K, D)
            mu_star  — mean absolute elementary effect, same shape
            sigma    — standard deviation of elementary effects, same shape
            mu_conf, mu_star_conf, sigma_conf — (2, ...) CI bounds or None
            ci       — how the intervals were made, or None without a
                       bootstrap
            invalid  — what the non-finite check found, and what it did

    Raises:
        ValueError: If ``Y`` does not have 1, 2, or 3 dimensions; if ``Y``'s
            first axis does not match ``sampling_result.n_runs``; if
            ``on_invalid`` is not one of the three policies; if the sample
            holds a non-finite value under ``on_invalid="raise"``; if fewer
            than 2 trajectories survive; if ``ci_method`` is invalid; if
            ``n_bootstrap`` is negative; if ``conf_level`` is not in
            ``(0, 1)``; if ``n_bootstrap > 0`` but ``key`` is ``None``; or
            if ``resample_chunk_size`` is given and below 1.

    Warns:
        JaxgsaWarning: If a derived design already lost blocks with no
            measurable step. The message adds a reliability note when fewer
            than 10 trajectories remain. A small design that the user asked
            for is deliberate, so it gives no warning.
        JaxgsaWarning: If an output slice has zero variance.
    """
    from jaxgsa.morris import SPEC

    # A trajectory is an indivisible sampling unit: it occupies a contiguous
    # block of D+1 expanded rows, and one bad row corrupts every elementary
    # effect that references it.
    r = sampling_result.n_trajectories
    rows_per_traj = sampling_result.n_params + 1

    ctx = prepare(
        SPEC,
        sampling_result.problem,
        Y,
        on_invalid=on_invalid,
        checks=(
            one_of("ci_method", ci_method, ("quantile", "gaussian")),
            at_least("resample_chunk_size", resample_chunk_size, 1),
            at_least("n_bootstrap", n_bootstrap, 0),
            in_open_interval("conf_level", conf_level, 0.0, 1.0),
        ),
        n_expected=int(sampling_result.samples.shape[0]),
        expand=sampling_result.expand_outputs,
        n_units=r,
        unit_of_row=np.repeat(np.arange(r), rows_per_traj),
        # Y is checked expanded, but the caller evaluated one output per
        # unique run. Report the rows they hold, not the expanded ones.
        row_labels=sampling_result.expanded_to_unique,
        min_kept=2,
        # A constant slice gives elementary effects of exactly 0 (0/delta), so
        # the measures come out 0, not NaN as in the variance-based methods.
        zero_variance_outcome="morris",
    )
    Y = ctx.Y3
    keep, invalid = ctx.keep, ctx.invalid

    # Checked here, with the other cheap argument checks, so a missing key is
    # reported before the point estimate is paid for.
    if n_bootstrap > 0 and key is None:
        raise ValueError("key is required when n_bootstrap > 0")

    idx_after, idx_before, delta = _reindex_after_drop(sampling_result, keep)
    if not keep.all():
        trailing = Y.shape[1:]
        kept_rows = np.flatnonzero(np.asarray(keep))
        Y = jnp.take(Y.reshape(r, rows_per_traj, *trailing), jnp.asarray(kept_rows), axis=0)
        Y = Y.reshape(len(kept_rows) * rows_per_traj, *trailing)
    remaining = int(keep.sum())

    # Warn on the cause of the thinning, not on the surviving count. A small r
    # that the user asked for is a deliberate choice, so it stays silent.
    # Blocks the user never gave up are worth reporting: blocks that
    # SobolSamples.to_morris dropped for having no measurable step. The
    # non-finite loss reports itself, through check_invalid. The message also
    # carries the reliability floor when the survivors fall below it.
    n_lost_upstream = sampling_result.n_blocks_dropped
    if n_lost_upstream > 0:
        requested = r + n_lost_upstream
        message = (
            f"jaxgsa.morris: {r} of the {requested} requested trajectories remain: "
            f"the source design dropped {n_lost_upstream} for having no measurable step"
        )
        if remaining < _MIN_TRAJECTORIES:
            message += (
                f" — results may be statistically unreliable (recommend >= {_MIN_TRAJECTORIES})"
            )
        warnings.warn(message, stacklevel=2, category=JaxgsaWarning)

    if remaining < 2:
        raise ValueError("Fewer than 2 trajectories remain after cleaning")

    if standardize_outputs:
        Y, _, _, _ = _standardize_outputs(Y)

    t0 = _verbose.tic()
    ee = _elementary_effects(Y, idx_after, idx_before, delta)  # (r, D, T, K)
    mu, mu_star, sigma = _stats_from_ee(ee)  # each (T, K, D)

    # One value for all three intervals: they are produced together or not at
    # all, so a single name keeps that fact checkable instead of implied.
    conf_triple: tuple[Array, Array, Array] | None = None
    replicates: dict[str, Array] | None = {} if keep_replicates else None
    if n_bootstrap > 0:
        assert key is not None  # a missing key was refused right after prepare
        r = ee.shape[0]
        # Pre-generate all R bootstrap index sets, sampling with replacement.
        # Named resample_idx, not indices: `indices` is this module's public
        # transformable entry point.
        resample_idx = jax.random.randint(key, shape=(n_bootstrap, r), minval=0, maxval=r)

        # One chunk materialises cs copies of the (r, D, T, K) effects tensor,
        # so a large T*K would otherwise exhaust device memory at the caller's
        # width. The memory budget lowers it, never raises it.
        per_replicate = int(np.prod(ee.shape))  # r * D * T * K
        cs = _resolve_resample_chunk_size(
            resample_chunk_size, n_bootstrap, per_replicate * ee.dtype.itemsize
        )
        mu_parts, mu_star_parts, sigma_parts = [], [], []
        for start in range(0, n_bootstrap, cs):
            end = min(start + cs, n_bootstrap)
            n_real = end - start
            idx_chunk = resample_idx[start:end]
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
        for name, estimate, parts in [
            ("mu", mu, mu_parts),
            ("mu_star", mu_star, mu_star_parts),
            ("sigma", sigma, sigma_parts),
        ]:
            draws = jnp.concatenate(parts)  # (R, T, K, D)
            lower, upper = _bootstrap_ci_endpoints(
                estimate, draws, conf_level=conf_level, ci_method=ci_method
            )
            conf_pairs.append(jnp.stack([lower, upper]))
            if replicates is not None:
                # The leading resample axis survives the squeeze, which
                # addresses the inserted T/K axes from the end.
                replicates[name] = ctx.squeeze(draws)
        conf_triple = (conf_pairs[0], conf_pairs[1], conf_pairs[2])

    mu = ctx.squeeze(mu)
    mu_star = ctx.squeeze(mu_star)
    sigma = ctx.squeeze(sigma)
    if conf_triple is None:
        mu_conf = mu_star_conf = sigma_conf = None
    else:
        mu_conf, mu_star_conf, sigma_conf = (ctx.squeeze(arr) for arr in conf_triple)

    ci = (
        CIInfo(
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            replicates=replicates,
        )
        if n_bootstrap > 0
        else None
    )

    result = MorrisResult(
        mu=mu,
        mu_star=mu_star,
        sigma=sigma,
        problem=sampling_result.problem,
        invalid=invalid,
        mu_conf=mu_conf,
        mu_star_conf=mu_star_conf,
        sigma_conf=sigma_conf,
        space="unit",
        ci=ci,
    )

    if verbose:
        elapsed = _verbose.stop(t0, result.mu, result.mu_star, result.sigma)
        _, T, K = ctx.Y3.shape
        # The resample width only exists when bootstrapping ran.
        if n_bootstrap > 0:
            notes = [
                f"resample_chunk_size: {resample_chunk_size} (user-set)"
                if resample_chunk_size is not None
                else "resample_chunk_size: auto (resolved from the memory budget)"
            ]
        else:
            notes = []
        _verbose.analysis_summary(
            method="jaxgsa.morris.analyze",
            problem=sampling_result.problem,
            n_runs=int(Y.shape[0]),
            T=T,
            K=K,
            invalid=invalid,
            timings=[("estimator (includes compile on the first call)", elapsed)],
            notes=notes,
            index_name="mu_star",
            values=result.mu_star,
            conf=result.mu_star_conf,
        )
    return result
