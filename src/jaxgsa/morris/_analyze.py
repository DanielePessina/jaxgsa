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
from collections.abc import Callable
from functools import lru_cache, partial
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.batching import get_memory_budget, resolve_batch_size
from jaxgsa._core.bootstrap import _bootstrap_ci_endpoints, draw_row_matrix
from jaxgsa._core.column_chunks import map_column_chunks
from jaxgsa._core.entry import (
    at_least,
    bootstrap_checks,
    check_scalars,
    positive_int,
    prepare,
)
from jaxgsa._core.invalid import OnInvalid, _unit_of_row_for_policy
from jaxgsa._core.irregular import (
    IrregularResult,
    IrregularY,
    _fwd_kwargs,
    _is_irregular_y,
    analyze_irregular,
)
from jaxgsa._core.result import CIInfo
from jaxgsa._core.sampling import require_design
from jaxgsa._core.validation import (
    YLayout,
    _prepare_Y,
    _standardize_outputs,
    _validate_output,
    as_key,
)
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.morris._result import MorrisResult
from jaxgsa.morris._sampling import MorrisSamples

# Keyword-only parameters the irregular engine forwards unchanged to the
# per-channel recursive analyze() calls. verbose is excluded:
# the engine owns it.
_IRREGULAR_KWARGS = (
    "standardize_outputs",
    "n_bootstrap",
    "conf_level",
    "ci_method",
    "key",
    "resample_chunk_size",
    "slice_chunk_size",
    "on_invalid",
    "keep_replicates",
)

# Fewest trajectories that still give statistically meaningful screening
# measures. The analysis reports this floor only when the design lost blocks
# that the user did not give up, meaning blocks with an unmeasurable step. The
# shared non-finite policy carries its own copy of the floor for the blocks it
# drops. A small design the user asked for is deliberate, so it stays silent.
_MIN_TRAJECTORIES = 10


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
    # the signed delta broadcasts over the trailing output dimensions. The
    # design stores delta in float64; casting it to Y's dtype keeps float32
    # outputs in float32 when x64 is enabled.
    return (Y[idx_after] - Y[idx_before]) / delta.astype(Y.dtype)[:, :, None, None]


def _measures(
    Y: Array,
    idx_after: Array,
    idx_before: Array,
    delta: Array,
    standardize_outputs: bool,
) -> tuple[Array, Array, Array, Array]:
    """Standardize, gather elementary effects, and reduce to the screening measures.

    Shared by :func:`indices` and :func:`analyze`: both run this exact
    standardize-gather-reduce body over the expanded outputs, so the two can
    never drift apart.

    Args:
        Y: Expanded model outputs, shape ``(r * (D + 1), T, K)``.
        idx_after: Expanded-row indices of the perturbed points, shape
            ``(r, D)``.
        idx_before: Expanded-row indices of the reference points, shape
            ``(r, D)``.
        delta: Signed unit-cube steps, shape ``(r, D)``.
        standardize_outputs: Standardize ``Y`` over the expanded sample axis
            before gathering, as in :func:`analyze`.

    Returns:
        ``(ee, mu, mu_star, sigma)``: the elementary effects, shape
        ``(r, D, T, K)``, and the three measures, each shape ``(T, K, D)``.
    """
    if standardize_outputs:
        Y, _, _, _ = _standardize_outputs(Y)
    ee = _elementary_effects(Y, idx_after, idx_before, delta)  # (r, D, T, K)
    mu, mu_star, sigma = _stats_from_ee(ee)  # each (T, K, D)
    return ee, mu, mu_star, sigma


@lru_cache(maxsize=None)
def _get_morris_fused_measures(D: int, standardize_outputs: bool):
    """Build the one-executable take + standardize + measures caller.

    The expansion gather is fused into the estimator when the non-finite
    check was skipped (``on_invalid='skip'``, see
    :func:`jaxgsa._core.entry.prepare`): the expanded array is then read by
    nothing but the estimator, so materialising it as an eager pass wastes a
    write and a read of the whole array. Measured ~1.3x faster than the
    eager pipeline on a half-million-row isotropic design. The fused graph
    takes the unique-row outputs, the design's expansion map and the
    elementary-effect bookkeeping as plain arrays, so the gather stays
    traceable.

    Only the scalar layout (``T * K == 1``) uses this, matching the Sobol
    front half: wide layouts run slower fused (measured there), so they
    expand eagerly instead.

    Args:
        D: Number of input parameters (``n_params + 1`` rows per
            trajectory).
        standardize_outputs: Whether to standardize the expanded outputs
            before gathering, as :func:`analyze` defines it.

    Returns:
        A jitted function of ``(Y_raw, index_map, idx_after, idx_before,
        delta)`` returning the same ``(ee, mu, mu_star, sigma)`` tuple as
        :func:`_measures`, with the inserted ``(T, K)`` axes in place.
    """

    def fn(
        Y_raw: Array,
        index_map: Array,
        idx_after: Array,
        idx_before: Array,
        delta: Array,
    ) -> tuple[Array, Array, Array, Array]:
        Y3 = jnp.take(Y_raw, index_map, axis=0).reshape(-1, 1, 1)
        return _measures(Y3, idx_after, idx_before, delta, standardize_outputs)

    return jax.jit(fn)


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

    # ee[idx] gathers r trajectories with replacement. lax.map runs the
    # resamples one after another, so one gathered copy of ee (plus its |ee|
    # or centred twin, which XLA reuses in place) is live at a time. A vmap
    # would hold C of them at once, measured at 2x the estimate below.
    def single(idx: Array) -> tuple[Array, Array, Array]:
        return _stats_from_ee(ee[idx])

    return jax.lax.map(single, idx_chunk)


def _resample_temp_bytes(ee_shape: tuple[int, ...], itemsize: int, chunk: int) -> int:
    """Estimate the XLA temporary bytes of :func:`_resample_stats`.

    Args:
        ee_shape: Shape ``(r, D, T, K)`` of the elementary effects.
        itemsize: Bytes per element of ``ee``.
        chunk: Resamples per call.

    Returns:
        One gathered copy of ``ee`` (the resamples run sequentially) plus the
        three ``(chunk, T, K, D)`` result arrays.
    """
    n_ee = int(np.prod(ee_shape))
    n_out = int(np.prod(ee_shape[1:]))  # D * T * K, one resample's statistic
    return (n_ee + 3 * chunk * n_out) * itemsize


# Live (n_expanded,)-sized arrays per output slice while one slice chunk is
# turned into screening measures: the standardized column, the two gathered
# rows, the effects, and the |ee| or centred copy the reduction reads. The
# bootstrap adds one gathered copy of the effects and its |ee| or centred twin.
_MORRIS_LIVE_ARRAYS = 5
_MORRIS_BOOT_ARRAYS = 2


def _slice_width(
    n_expanded: int,
    n_slices: int,
    dtype: jnp.dtype,
    slice_chunk_size: int | None,
    *,
    bootstrap: bool = False,
) -> int:
    """Resolve the number of output slices turned into measures per chunk.

    Args:
        n_expanded: Rows of the expanded design, ``r * (D + 1)``.
        n_slices: Flattened output slices, ``T * K``.
        dtype: Dtype of the expanded outputs.
        slice_chunk_size: The caller's value, or ``None`` to derive one from
            the active memory budget.
        bootstrap: Whether the bootstrap runs on each chunk, which keeps more
            arrays live per slice.

    Returns:
        Slices per chunk, in ``[1, n_slices]``.
    """
    live = _MORRIS_LIVE_ARRAYS + (_MORRIS_BOOT_ARRAYS if bootstrap else 0)
    per_slice = live * n_expanded * jnp.dtype(dtype).itemsize
    return resolve_batch_size(per_slice, n_slices, slice_chunk_size)


@partial(jax.jit, static_argnames=("width",))
def _slice_chunk(Y3: Array, start: Array | int, width: int) -> Array:
    """Gather ``width`` flattened output slices of ``Y3`` from ``start`` on.

    ``lax.dynamic_slice`` clamps ``start`` to ``T * K - width``, so the last
    chunk has the same shape as the others and the gather compiles once. Its
    real slices are then the trailing ones, which :func:`_chunked_measures`
    selects.

    Args:
        Y3: Expanded outputs, shape ``(n_expanded, T, K)``.
        start: First flattened slice of the chunk.
        width: Slices per chunk (static).

    Returns:
        The chunk as ``(n_expanded, width, 1)``: each slice becomes one time
        step of a one-output layout.
    """
    flat = Y3.reshape(Y3.shape[0], -1)
    return jax.lax.dynamic_slice_in_dim(flat, start, width, axis=1)[:, :, None]


def _chunked_measures(
    Y3: Array,
    idx_after: Array,
    idx_before: Array,
    delta: Array,
    standardize_outputs: bool,
    width: int,
    on_chunk: Callable[[Array], None] | None = None,
) -> tuple[Array, Array, Array]:
    """Compute the screening measures ``width`` output slices at a time.

    Standardization, the elementary effects and their reduction all act on
    each output slice alone, so a chunk of slices gives the same numbers as
    the whole array. :func:`analyze` runs this eagerly, one set of device
    calls per chunk, so only one chunk of the ``(r, D, T, K)`` effects exists
    at a time. :func:`indices` may be traced and uses
    :func:`_mapped_measures` instead.

    Args:
        Y3: Expanded outputs, shape ``(n_expanded, T, K)``.
        idx_after: Expanded-row indices of the perturbed points, ``(r, D)``.
        idx_before: Expanded-row indices of the reference points, ``(r, D)``.
        delta: Signed unit-cube steps, ``(r, D)``.
        standardize_outputs: As in :func:`analyze`.
        width: Slices per chunk.
        on_chunk: Called with the effects of each chunk's real slices,
            shape ``(r, D, actual, 1)``, in slice order. The bootstrap uses
            it to resample one chunk at a time.

    Returns:
        ``(mu, mu_star, sigma)``, each shape ``(T, K, D)``.
    """
    _, T, K = Y3.shape
    total = T * K
    mu_parts: list[Array] = []
    mu_star_parts: list[Array] = []
    sigma_parts: list[Array] = []
    for start in range(0, total, width):
        actual = min(width, total - start)
        # Where the clamped gather put this chunk's first real slice.
        offset = start - min(start, total - width)
        real = slice(offset, offset + actual)
        Yc = _slice_chunk(Y3, start, width)
        ee, mu, mu_star, sigma = _measures(Yc, idx_after, idx_before, delta, standardize_outputs)
        if on_chunk is not None:
            on_chunk(ee[:, :, real])
        mu_parts.append(mu[real])  # (actual, 1, D)
        mu_star_parts.append(mu_star[real])
        sigma_parts.append(sigma[real])
    return (
        jnp.concatenate(mu_parts).reshape(T, K, -1),
        jnp.concatenate(mu_star_parts).reshape(T, K, -1),
        jnp.concatenate(sigma_parts).reshape(T, K, -1),
    )


@partial(jax.jit, static_argnames=("standardize_outputs", "width"))
def _mapped_measures(
    Y3: Array,
    idx_after: Array,
    idx_before: Array,
    delta: Array,
    *,
    standardize_outputs: bool,
    width: int,
) -> tuple[Array, Array, Array]:
    """Compute the screening measures ``width`` slices at a time, in one program.

    The traceable twin of :func:`_chunked_measures`, for :func:`indices`.
    Under an outer ``jit`` a Python loop over chunks is traced into one
    program that keeps every chunk live, so the chunks run in a
    ``lax.map`` instead (:func:`jaxgsa._core.column_chunks.map_column_chunks`).

    Args:
        Y3: Expanded outputs, shape ``(n_expanded, T, K)``.
        idx_after: Expanded-row indices of the perturbed points, ``(r, D)``.
        idx_before: Expanded-row indices of the reference points, ``(r, D)``.
        delta: Signed unit-cube steps, ``(r, D)``.
        standardize_outputs: As in :func:`analyze` (static).
        width: Slices per chunk (static).

    Returns:
        ``(mu, mu_star, sigma)``, each shape ``(T, K, D)``.
    """
    n_expanded, T, K = Y3.shape

    def one_chunk(cols: Array) -> tuple[Array, Array, Array]:
        _, mu, mu_star, sigma = _measures(
            cols[:, :, None], idx_after, idx_before, delta, standardize_outputs
        )
        return mu[:, 0], mu_star[:, 0], sigma[:, 0]  # each (width, D)

    out = map_column_chunks(one_chunk, Y3.reshape(n_expanded, T * K), width, axis=0)
    mu, mu_star, sigma = (leaf.reshape(T, K, -1) for leaf in out)
    return mu, mu_star, sigma


def _bootstrap_draws(
    ee: Array, resample_idx: Array, resample_chunk_size: int | None
) -> tuple[Array, Array, Array]:
    """Bootstrap the screening measures over trajectories.

    Args:
        ee: Elementary effects, shape ``(r, D, T, K)``.
        resample_idx: Trajectory index sets, shape ``(R, r)``.
        resample_chunk_size: The caller's resamples per call, or ``None`` to
            derive one from the memory budget.

    Returns:
        ``(mu, mu_star, sigma)`` draws, each shape ``(R, T, K, D)``.
    """
    n_bootstrap = resample_idx.shape[0]
    # One chunk holds a single gathered copy of the (r, D, T, K) effects
    # tensor (the resamples run in sequence) plus three (T, K, D) results
    # per resample. The copy is a fixed cost, so the budget left after it
    # sets how many resamples fit in a chunk.
    itemsize = ee.dtype.itemsize
    fixed = _resample_temp_bytes(ee.shape, itemsize, 0)
    per_resample = _resample_temp_bytes(ee.shape, itemsize, 1) - fixed
    if resample_chunk_size is None:
        cs = max(1, min(n_bootstrap, (get_memory_budget() - fixed) // per_resample))
    else:
        cs = resolve_batch_size(per_resample, n_bootstrap, resample_chunk_size)
    mu_parts: list[Array] = []
    mu_star_parts: list[Array] = []
    sigma_parts: list[Array] = []
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
    return (
        jnp.concatenate(mu_parts),
        jnp.concatenate(mu_star_parts),
        jnp.concatenate(sigma_parts),
    )


class MorrisIndices(NamedTuple):
    """The arrays :func:`indices` returns, named like the :class:`MorrisResult` fields.

    Attributes:
        mu: Mean elementary effect, signed.
        mu_star: Mean absolute elementary effect.
        sigma: Standard deviation of the elementary effects (ddof=1).
    """

    mu: Array
    mu_star: Array
    sigma: Array


def indices(
    samples: MorrisSamples,
    Y: Array,
    *,
    standardize_outputs: bool = False,
    slice_chunk_size: int | None = None,
) -> MorrisIndices:
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
        samples: The design from :func:`jaxgsa.morris.sample`, used
            only for its expansion map and its elementary-effect bookkeeping.
        Y: Model outputs evaluated at each unique row of
            ``samples.samples``, in the same row order. Shapes are
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
        slice_chunk_size: As in :func:`analyze`. The value changes only the
            peak memory, never a measure.

    Returns:
        A :class:`MorrisIndices` ``(mu, mu_star, sigma)``, shaped ``(D,)``, ``(K, D)`` or
        ``(T, K, D)`` to match the rank of ``Y``, exactly as ``analyze``
        reports them.

    Raises:
        ValueError: If ``Y`` is not 1-D, 2-D, or 3-D, its first axis does
            not match ``samples.n_runs``, or ``slice_chunk_size`` is given and
            is not an integer of at least 1.
    """
    require_design(samples, MorrisSamples, "jaxgsa.morris.indices", "jaxgsa.morris.sample")
    check_scalars(
        (positive_int("slice_chunk_size", slice_chunk_size),), method="jaxgsa.morris.indices"
    )
    _validate_output(Y, None, None)
    Y3, layout = _prepare_Y(samples.expand_outputs(Y))
    # Design facts, not facts about Y: moving them to the device under trace
    # is a constant fold, never a host read of an array.
    idx_after = jnp.asarray(samples.ee_idx_after)
    idx_before = jnp.asarray(samples.ee_idx_before)
    delta = jnp.asarray(samples.ee_delta)
    n_expanded, T, K = Y3.shape
    width = _slice_width(n_expanded, T * K, Y3.dtype, slice_chunk_size)
    if width >= T * K:
        _, mu, mu_star, sigma = _measures(Y3, idx_after, idx_before, delta, standardize_outputs)
    else:
        mu, mu_star, sigma = _mapped_measures(
            Y3, idx_after, idx_before, delta, standardize_outputs=standardize_outputs, width=width
        )
    return MorrisIndices(layout.squeeze(mu), layout.squeeze(mu_star), layout.squeeze(sigma))


def _reindex_after_drop(samples: MorrisSamples, keep: np.ndarray) -> tuple[Array, Array, Array]:
    """Rebuild the elementary-effect bookkeeping against the compacted layout.

    Removing whole trajectories shifts every later block down, so the global
    row indices stored on the design no longer point at the rows they named.
    The offset of a row *inside* its block does not change, so the fix is to
    strip the old block offset and add the new one.

    Args:
        samples: Sampling metadata with the original bookkeeping.
        keep: Boolean mask of shape ``(r,)``, True for surviving trajectories.

    Returns:
        ``(idx_after, idx_before, delta)``, re-indexed and restricted to the
        surviving trajectories. All three are unchanged when nothing is
        dropped.
    """
    if keep.all():
        return (
            jnp.asarray(samples.ee_idx_after),
            jnp.asarray(samples.ee_idx_before),
            jnp.asarray(samples.ee_delta),
        )

    idx_after = np.asarray(samples.ee_idx_after)
    idx_before = np.asarray(samples.ee_idx_before)
    delta = np.asarray(samples.ee_delta)
    rows_per_traj = samples.n_params + 1
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
    samples: MorrisSamples,
    Y: Array | IrregularY,
    *,
    standardize_outputs: bool = False,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | int | None = None,
    resample_chunk_size: int | None = None,
    slice_chunk_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> MorrisResult | IrregularResult:
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
        samples: Result from :func:`jaxgsa.morris.sample` with the
            unique sample matrix plus elementary-effect bookkeeping.
        Y: Model outputs evaluated at each unique row of
            ``samples.samples``. Accepted shapes:
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
        slice_chunk_size: Maximum number of flattened ``T*K`` output slices
            turned into elementary effects at once. It caps the peak memory
            of the ``(r, D, T, K)`` effects, and of the bootstrap, for a
            large ``T * K``. Every measure is computed per slice, so the
            value never changes a result. ``None`` (default) derives one from
            the active memory budget (:func:`jaxgsa.config.set_memory_budget`).
            An explicit value wins over the budget, is clamped to ``T * K``,
            and must be an integer of at least 1.
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
            first axis does not match ``samples.n_runs``; if
            ``on_invalid`` is not one of the three policies; if the sample
            holds a non-finite value under ``on_invalid="raise"``; if fewer
            than 2 trajectories survive; if ``ci_method`` is invalid; if
            ``n_bootstrap`` is negative; if ``conf_level`` is not in
            ``(0, 1)``; if ``n_bootstrap > 0`` but ``key`` is ``None``; or
            if ``resample_chunk_size`` is given and below 1; or if
            ``slice_chunk_size`` is given and is not an integer of at least 1.

    Warns:
        JaxgsaWarning: If a derived design already lost blocks with no
            measurable step. The message adds a reliability note when fewer
            than 10 trajectories remain. A small design that the user asked
            for is deliberate, so it gives no warning.
        JaxgsaWarning: If an output slice has zero variance.
    """
    require_design(samples, MorrisSamples, "jaxgsa.morris.analyze", "jaxgsa.morris.sample")
    key = as_key(key, "morris")
    if _is_irregular_y(Y):
        return analyze_irregular(
            analyze,
            channel_data=samples,
            problem=samples.problem,
            Y=Y,
            n_expected=int(samples.n_runs),
            design_based=True,
            verbose=verbose,
            kwargs=_fwd_kwargs(locals(), _IRREGULAR_KWARGS),
        )
    Y = cast(Array, Y)

    from jaxgsa.morris import _SPEC

    # A trajectory is an indivisible sampling unit: it occupies a contiguous
    # block of D+1 expanded rows, and one bad row corrupts every elementary
    # effect that references it.
    r = samples.n_trajectories
    rows_per_traj = samples.n_params + 1

    ctx = prepare(
        _SPEC,
        samples.problem,
        Y,
        on_invalid=on_invalid,
        checks=(
            *bootstrap_checks(n_bootstrap, ci_method, conf_level, key),
            at_least("resample_chunk_size", resample_chunk_size, 1),
            positive_int("slice_chunk_size", slice_chunk_size),
        ),
        n_expected=int(samples.samples.shape[0]),
        expand=samples.expand_outputs,
        n_units=r,
        # Under on_invalid='skip' the check never reads the row map
        # (measured ~0.3 ms at r=16384), so it is built only where it is used.
        unit_of_row=_unit_of_row_for_policy(
            on_invalid, lambda: np.repeat(np.arange(r), rows_per_traj)
        ),
        # One trajectory is one unit of rows_per_traj contiguous rows.
        unit_stride=rows_per_traj,
        # Y is checked expanded, but the caller evaluated one output per
        # unique run. Report the rows they hold, not the expanded ones.
        row_labels=samples.expanded_to_unique,
        min_kept=2,
        # A constant slice gives elementary effects of exactly 0 (0/delta), so
        # the measures come out 0, not NaN as in the variance-based methods.
        zero_variance_outcome="morris",
    )
    Y = ctx.Y3
    keep, invalid = ctx.keep, ctx.invalid

    idx_after, idx_before, delta = _reindex_after_drop(samples, keep)
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
    n_lost_upstream = samples.n_blocks_dropped
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

    t0 = _verbose.tic()
    _, T, K = ctx.Y3.shape
    # Sized on the expanded design whichever path runs below, so the width
    # does not depend on whether the expansion was deferred.
    width = _slice_width(
        len(idx_after) * rows_per_traj,
        T * K,
        ctx.Y3.dtype,
        slice_chunk_size,
        bootstrap=n_bootstrap > 0,
    )
    resample_idx = None
    if n_bootstrap > 0:
        assert key is not None  # a missing key was refused right after prepare
        # Pre-generate all R bootstrap index sets, sampling with replacement.
        # Named resample_idx, not indices: `indices` is this module's public
        # transformable entry point. The same sets serve every slice chunk.
        resample_idx = draw_row_matrix(key, n_bootstrap, len(idx_after))

    draws: tuple[Array, Array, Array] | None = None
    if width < T * K:
        # More slices than one chunk holds: never form all of the
        # (r, D, T, K) effects. The bootstrap runs on each chunk's effects
        # with the same resample sets, so its draws equal the unchunked ones.
        if ctx.deferred_expand is not None:
            Y = ctx.deferred_expand(Y)
        boot_parts: list[tuple[Array, Array, Array]] = []

        def _boot_chunk(ee_chunk: Array) -> None:
            if resample_idx is not None:
                boot_parts.append(_bootstrap_draws(ee_chunk, resample_idx, resample_chunk_size))

        mu, mu_star, sigma = _chunked_measures(
            Y, idx_after, idx_before, delta, standardize_outputs, width, _boot_chunk
        )
        if boot_parts:
            # Each part is (R, actual, 1, D); join the slices, then restore (T, K).
            draws = (
                jnp.concatenate([p[0] for p in boot_parts], axis=1).reshape(n_bootstrap, T, K, -1),
                jnp.concatenate([p[1] for p in boot_parts], axis=1).reshape(n_bootstrap, T, K, -1),
                jnp.concatenate([p[2] for p in boot_parts], axis=1).reshape(n_bootstrap, T, K, -1),
            )
    else:
        if (
            ctx.deferred_expand is not None
            and n_bootstrap == 0
            and keep.all()
            and ctx.layout is YLayout.SCALAR
        ):
            # on_invalid='skip' held the expansion back (see
            # Context.deferred_expand). The scalar no-bootstrap path fuses the
            # gather into the estimator -- one executable, measured ~1.3x
            # faster than the eager expansion on a half-million-row design.
            # Wide layouts run slower fused, and the bootstrap resamples from
            # the expanded rows, so both expand eagerly below.
            fused = _get_morris_fused_measures(samples.n_params, standardize_outputs)
            ee, mu, mu_star, sigma = fused(
                ctx.Y,
                samples._expanded_to_unique_jax,
                idx_after,
                idx_before,
                delta,
            )
        else:
            if ctx.deferred_expand is not None:
                # Deferral only happens under on_invalid='skip', where keep
                # is all-True, so the compaction above never ran and Y is
                # still the raw promoted rows; expanding here is exactly what
                # prepare would have done under any other policy.
                Y = ctx.deferred_expand(Y)
            ee, mu, mu_star, sigma = _measures(
                Y, idx_after, idx_before, delta, standardize_outputs
            )
        if resample_idx is not None:
            draws = _bootstrap_draws(ee, resample_idx, resample_chunk_size)

    # One value for all three intervals: they are produced together or not at
    # all, so a single name keeps that fact checkable instead of implied.
    conf_triple: tuple[Array, Array, Array] | None = None
    replicates: dict[str, Array] | None = {} if keep_replicates else None
    if draws is not None:
        conf_pairs = []
        for name, estimate, name_draws in [
            ("mu", mu, draws[0]),
            ("mu_star", mu_star, draws[1]),
            ("sigma", sigma, draws[2]),
        ]:
            lower, upper = _bootstrap_ci_endpoints(
                estimate, name_draws, conf_level=conf_level, ci_method=ci_method
            )
            conf_pairs.append(jnp.stack([lower, upper]))
            if replicates is not None:
                # The leading resample axis survives the squeeze, which
                # addresses the inserted T/K axes from the end.
                replicates[name] = ctx.squeeze(name_draws)
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
        problem=samples.problem,
        invalid=invalid,
        mu_conf=mu_conf,
        mu_star_conf=mu_star_conf,
        sigma_conf=sigma_conf,
        space="unit",
        ci=ci,
    )

    if verbose:
        elapsed = _verbose.stop(t0, result.mu, result.mu_star, result.sigma)
        origin = "user-set" if slice_chunk_size is not None else "resolved from the memory budget"
        notes = [f"slice_chunk_size: {width} ({origin})"]
        # The resample width only exists when bootstrapping ran.
        if n_bootstrap > 0:
            notes.append(
                f"resample_chunk_size: {resample_chunk_size} (user-set)"
                if resample_chunk_size is not None
                else "resample_chunk_size: auto (resolved from the memory budget)"
            )
        _verbose.analysis_summary(
            method="jaxgsa.morris.analyze",
            problem=samples.problem,
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
