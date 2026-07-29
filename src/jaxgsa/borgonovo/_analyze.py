"""Borgonovo delta analysis: moment-independent sensitivity from given data.

Implements the Plischke, Borgonovo & Smith (2013) given-data estimator of
Borgonovo's (2007) delta index. For each input the sample is split into
equal-frequency classes by the input's rank; the delta index is the
class-weighted L1 distance between the unconditional output density and
each conditional density, both estimated by Gaussian KDE with Silverman
bandwidths on a fixed output grid (trapezoidal integration). A given-data
first-order Sobol index falls out of the same partition at negligible cost.

The plug-in estimator is biased upward at finite N, so by default the
central estimate is bias-corrected with bootstrap resamples
(``2*d_hat - mean(d_boot)``, Plischke et al. eqn 30) where ``d_hat`` is
computed on the original sample; percentile confidence intervals come from
the same replicates. The original sample and the bootstrap replicates run
through a single scanned path (the original sample is replicate 0, gathered
via the identity permutation), so the point estimate and its interval are
always computed under identical conventions. Class-partition indices are
built once (per input, for every replicate) and reused across output-column
chunks; the JIT-compiled per-column kernel is cached only on the scalar
estimator settings ``(grid_size, bandwidth)`` so it captures no
sample-sized constants.

Estimator details mirror ``SALib.analyze.delta`` (equal-frequency ordinal
rank partition, Plischke class-count heuristic, Silverman KDE factors,
100-point output grid) with three deliberate differences: the central
estimate uses the original sample rather than a bootstrap resample
(deterministic given the data); a constant output column yields
``delta = S1 = 0`` instead of an error; and a bootstrap replicate that
happens to be constant (reachable for rare-event outputs) contributes the
point estimate rather than a spurious zero, so it neither adds nor removes
bias (SALib raises ``LinAlgError`` on such data).

References:
    Borgonovo (2007). A new uncertainty importance measure.
    Reliability Engineering & System Safety 92(6):771-784.

    Plischke, Borgonovo & Smith (2013). Global sensitivity measures from
    given data. European Journal of Operational Research 226(3):536-550.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.bootstrap import _percentile_ci
from jaxgsa._core.partition import (
    _mask_from_counts,
    _replicate_slice,
    build_partition_groups,
)
from jaxgsa._core.validation import _prepare_Y, _squeeze_output_axes, _validate_xy_inputs
from jaxgsa.borgonovo._result import DeltaResult
from jaxgsa.problem import Problem, _categorical_dims

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_MAX_CLASSES = 48
# A class whose KDE bandwidth falls below this fraction of the full-sample
# bandwidth is numerically degenerate: its variance is zero (a point mass,
# e.g. one categorical level mapping to one output value) or pure float
# noise. Genuine conditional spreads sit orders of magnitude above this.
_DEGENERATE_BW_TOL = 1e-6
# Bandwidth given to a degenerate class, as a fraction of the full-sample
# Silverman bandwidth. The class must stay visibly narrower than the
# unconditional smoothing (so it still reads as a concentrated class), but
# wide enough for the shared output grid to integrate it: a Gaussian
# sampled at a spacing of at most its own sigma has negligible trapezoid
# error, hence the grid-step lower bound. Measured on a noise-free
# three-atom repro (true delta 2/3), 0.1 recovers delta within ~0.07
# across N in [1e3, 1e4]; fractions >= 0.2 over-smooth toward 0.56 and
# fractions <= 0.02 alias on the grid (delta > 1).
_DEGENERATE_BW_FRACTION = 0.1
# Target element budget for the default per-chunk working set. The dominant
# intermediate is the conditional-KDE tensor whose size scales as
# ``chunk_columns * D * N * grid_size`` (class count M times padding P is ~N),
# so the default chunk width is chosen to keep it near this many float32
# elements (~256 MB).
_CHUNK_ELEM_BUDGET = 1 << 26


def _plischke_n_classes(n_samples: int) -> int:
    """Sample-size heuristic for the number of conditioning classes.

    Identical to SALib's rule (Plischke et al. 2013): roughly ``N**(2/7)``
    classes, saturating at 48 for large samples.

    Args:
        n_samples: Number of samples N.

    Returns:
        Number of equal-frequency classes M.
    """
    exponent = 2.0 / (7.0 + np.tanh((1500.0 - n_samples) / 500.0))
    return min(int(np.ceil(n_samples**exponent)), _MAX_CLASSES)


@lru_cache(maxsize=32)
def _get_delta_kernel(grid_size: int, bw_factor: float | None):
    """Return a JIT-compiled delta/S1 kernel for static estimator settings.

    The kernel is cached only on the scalar settings that change tracing
    (grid size and the bandwidth branch); sample-sized data (inputs, class
    indices, masks) are passed as runtime arguments, so nothing of size
    ``O(N)`` is captured or baked into the compiled executable.

    Args:
        grid_size: Number of output-grid points for the KDE (static).
        bw_factor: KDE bandwidth factor multiplying the sample standard
            deviation, or ``None`` for the per-class Silverman rule.

    Returns:
        A jitted callable ``(Y_cols (N, C), all_idx (R, N), groups) ->
        (d (R, C, D), s1 (R, C, D), degenerate (R, C), floored (R, C))``
        giving plug-in estimates for every replicate, a
        per-replicate/per-column flag marking constant resamples, and a
        flag marking columns where a degenerate class engaged the
        bandwidth floor. ``groups`` is a tuple of canonical
        ``(cls_idx, counts)`` partition-group layouts from
        :func:`jaxgsa._core.partition.build_partition_groups`; the kernel
        processes every group in one call (shared per-replicate statistics
        are computed once) and concatenates the results on the input axis
        in group order. Zero-size classes carry zero weight.
    """

    def _bandwidths(counts: Array, std: Array) -> Array:
        """Per-class (or full-sample) KDE bandwidths."""
        factor = (0.75 * counts) ** (-0.2) if bw_factor is None else bw_factor
        return factor * std

    def _kde_full(y: Array, grid: Array, h: Array) -> Array:
        """Gaussian KDE of a full column ``y (N,)`` on ``grid (G,)``."""
        n = y.shape[0]
        safe_h = jnp.where(h > 0, h, 1.0)
        u = (grid[:, None] - y[None, :]) / safe_h
        f = jnp.exp(-0.5 * u * u).sum(axis=1) / (n * safe_h * _SQRT_2PI)
        # Zero bandwidth (constant data) drops the density from the
        # integrand, mirroring SALib's degenerate-class treatment.
        return jnp.where(h > 0, f, 0.0)

    def _impl(
        Y_cols: Array,
        all_idx: Array,
        groups: tuple[tuple[Array, Array], ...],
    ):
        dtype = jnp.result_type(Y_cols.dtype, jnp.float32)
        Y_cols = Y_cols.astype(dtype)
        N = Y_cols.shape[0]  # every row belongs to exactly one class per input

        # Output grids depend only on the original sample and are reused for
        # every replicate (SALib does the same).
        y_min = Y_cols.min(axis=0)
        y_max = Y_cols.max(axis=0)
        steps = jnp.linspace(0.0, 1.0, grid_size, dtype=dtype)
        grids = y_min[:, None] + steps[None, :] * (y_max - y_min)[:, None]  # (C, G)

        counts_list = tuple(counts for _, counts in groups)
        cls_list = tuple(cls_idx for cls_idx, _ in groups)

        def _group_stats(y, grid, h_full, fy, y_mean, cls_idx, mask_b, counts_b):
            """Per-input delta/S1 numerators and floor flag for one group.

            ``mask_b (G, M, P)`` and ``counts_b (G, M)`` (both float)
            broadcast against the group's input axis Dg (``G`` is 1 or
            Dg). A zero-size class has zero std, hence zero bandwidth, so
            its density is dropped; its zero count removes it from every
            weighted sum.
            """
            safe_counts = jnp.maximum(counts_b, 1.0)
            y_cls = y[cls_idx]  # (Dg, M, P) resampled class members
            mean = (y_cls * mask_b).sum(axis=-1) / safe_counts  # (Dg, M)
            dev = (y_cls - mean[..., None]) * mask_b
            var = (dev**2).sum(axis=-1) / jnp.maximum(counts_b - 1.0, 1.0)
            h = _bandwidths(safe_counts, jnp.sqrt(var))  # (Dg, M)
            # A degenerate (zero-variance) class would get bandwidth 0 and a
            # zeroed density, which biases delta far low. Floor it so it
            # becomes a narrow kernel at its value instead; see the
            # _DEGENERATE_BW_FRACTION comment for the width rationale. The
            # predicate keeps non-degenerate classes bit-identical and stays
            # False for a constant column (h_full == 0), which must keep its
            # delta = 0 contract.
            floor = jnp.maximum(_DEGENERATE_BW_FRACTION * h_full, grid[1] - grid[0])
            floored_cls = (counts_b > 0) & (h < _DEGENERATE_BW_TOL * h_full)
            h = jnp.where(floored_cls, floor, h)
            safe_h = jnp.where(h > 0, h, 1.0)

            u = (grid[None, None, :, None] - y_cls[:, :, None, :]) / safe_h[..., None, None]
            k = jnp.exp(-0.5 * u * u) * mask_b[..., None, :]
            fyc = k.sum(axis=-1) / (safe_counts[..., None] * safe_h[..., None] * _SQRT_2PI)
            fyc = jnp.where((h > 0)[..., None], fyc, 0.0)  # (Dg, M, G)

            l1 = jnp.trapezoid(jnp.abs(fy[None, None, :] - fyc), grid, axis=-1)
            delta = (counts_b / (2.0 * N) * l1).sum(axis=-1)  # (Dg,)
            Vi = (counts_b / N * (mean - y_mean) ** 2).sum(axis=-1)
            return delta, Vi, floored_cls.any()

        def _col_stats(y: Array, grid: Array, r: Array, layouts):
            """Delta, S1, degeneracy and floor flags for one column/replicate.

            The per-replicate column statistics (resampled column, full
            KDE, mean, variance) are computed once and shared by every
            partition group; group results are concatenated on the input
            axis in group order.
            """
            y_r = y[r]  # resampled column
            h_full = _bandwidths(jnp.asarray(float(N), dtype=dtype), jnp.std(y_r, ddof=1))
            fy = _kde_full(y_r, grid, h_full)
            degenerate = y_r.max() == y_r.min()
            y_mean = y_r.mean()
            y_var = jnp.var(y_r)
            safe_var = jnp.where(y_var > 0, y_var, 1.0)

            outs = [
                _group_stats(y, grid, h_full, fy, y_mean, cls_idx, mask_b, counts_b)
                for cls_idx, mask_b, counts_b in layouts
            ]
            delta = jnp.concatenate([o[0] for o in outs])
            Vi = jnp.concatenate([o[1] for o in outs])
            s1 = jnp.where(y_var > 0, Vi / safe_var, 0.0)
            floored = jnp.stack([o[2] for o in outs]).any()
            return delta, s1, degenerate, floored

        def _one_replicate(carry, xs):
            i, r, cls_parts = xs
            layouts = []
            for cls_r, counts_g in zip(cls_parts, counts_list):
                counts_r = _replicate_slice(counts_g, i)  # (G, M)
                mask_r = _mask_from_counts(counts_r, cls_r.shape[-1]).astype(dtype)
                layouts.append((cls_r, mask_r, counts_r.astype(dtype)))
            d, s1, degen, floored = jax.vmap(lambda y, grid: _col_stats(y, grid, r, layouts))(
                Y_cols.T, grids
            )
            return carry, (d, s1, degen, floored)

        R = all_idx.shape[0]
        scan_xs = (jnp.arange(R), all_idx, cls_list)
        _, (d_all, s1_all, degen_all, floored_all) = jax.lax.scan(_one_replicate, None, scan_xs)
        return d_all, s1_all, degen_all, floored_all

    return jax.jit(_impl)


def analyze(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_classes: int | None = None,
    grid_size: int = 100,
    bandwidth: float | Literal["silverman"] = "silverman",
    n_bootstrap: int = 100,
    conf_level: float = 0.95,
    bias_correct: bool = True,
    seed: int = 0,
    slice_chunk_size: int | None = None,
) -> DeltaResult:
    """Compute Borgonovo delta and given-data first-order Sobol indices.

    The delta index measures how much knowing an input's value shifts the
    *entire* output density: ``delta_i`` is (half) the expected L1
    distance between the unconditional output density and the density
    conditional on x_i. It lies in [0, 1] — 0 means the output
    distribution is unaffected by x_i, 1 means it is fully determined by
    it. Because delta compares whole densities rather than variances
    ("moment-independent"), it captures influence on tails and shape that
    Sobol indices miss, and it needs no special sampling design: any
    (X, Y) sample works. A given-data first-order Sobol index S1 is
    returned from the same partition at negligible extra cost.

    Args:
        problem: Problem definition with D parameters.
        X: Input sample matrix ``(N, D)``.
        Y: Model output ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        n_classes: Number of equal-frequency conditioning classes per
            *continuous* input. ``None`` selects the Plischke sample-size
            heuristic (SALib-identical, at most 48 classes). Categorical
            inputs ignore it and always use one class per level (class
            sizes are the observed level counts); declared levels with no
            observed samples are dropped with a warning. A passed value
            is always validated against ``[2, N]``; with only categorical
            inputs a ``UserWarning`` says it is ignored.
        grid_size: Number of points of the output grid the densities are
            compared on (spanning ``[Y.min(), Y.max()]`` per column).
        bandwidth: KDE bandwidth rule: ``"silverman"`` for the per-class
            Silverman factor, or a positive float used directly as the
            factor multiplying the sample standard deviation.
        n_bootstrap: Number of bootstrap resamples for bias correction and
            confidence intervals. Set to 0 to skip both (plug-in estimate,
            ``delta_conf``/``S1_conf`` are ``None``).
        conf_level: Confidence level for percentile bootstrap intervals.
        bias_correct: Apply the Plischke bias reduction
            ``2*d_hat - mean(d_boot)`` to the delta estimate (requires
            ``n_bootstrap > 0``; S1 is never bias-corrected, matching
            SALib).
        seed: Random seed for bootstrap resampling.
        slice_chunk_size: Number of flattened ``T*K`` output columns
            processed per kernel call. ``None`` picks a memory-aware
            default from the sample size; pass an explicit positive
            integer to override. Peak memory scales with
            ``slice_chunk_size * D * N * grid_size``.

    Returns:
        DeltaResult with delta and S1 indices and optional confidence
        intervals. The underlying delta index is defined on ``[0, 1]`` and
        the plug-in estimate stays in that range, but the default
        bias-corrected estimate (and its confidence bounds) can fall
        marginally below 0 for weak/near-noninfluential inputs at small
        sample sizes. A constant output column yields ``delta = S1 = 0``
        (SALib raises an error in this case). A conditioning class with
        zero output variance (a point mass) gets a floored KDE bandwidth
        instead of a zeroed density, with one ``UserWarning``; classes
        with genuine spread are unaffected.

    Raises:
        ValueError: If X is not 2-D, its column count does not match the
            problem, Y is not 1-D/2-D/3-D, X and Y have differing row
            counts, a passed ``n_classes`` is not in ``[2, N]``, a
            categorical column of X holds
            values other than its integer level codes, ``grid_size < 2``,
            ``bandwidth`` is neither ``"silverman"`` nor a positive float,
            ``n_bootstrap < 0``, ``conf_level`` is not in ``(0, 1)``, or
            ``slice_chunk_size`` is not a positive integer.
    """
    X = jnp.asarray(X)
    # The delta estimator partitions on rank classes and compares output
    # densities, so a declared input correlation does not invalidate it.
    Y = _validate_xy_inputs(problem, X, Y, correlation_ok=True, categorical_ok=True)

    N = X.shape[0]
    # n_classes applies to the continuous columns only; categorical columns
    # always get one conditioning class per level.
    dims_levels = _categorical_dims(problem)
    cat_dims = [d for d, _ in dims_levels]
    cont_dims = [d for d in range(problem.num_vars) if d not in set(cat_dims)]
    if n_classes is None:
        M = _plischke_n_classes(N)
    else:
        M = int(n_classes)
        # A passed value is always validated, even when nothing uses it;
        # silently accepting nonsense hides bugs in the caller.
        if not 2 <= M <= N:
            raise ValueError(f"n_classes must be in [2, N={N}], got {n_classes}")
        if not cont_dims:
            warnings.warn(
                "jaxgsa: n_classes is ignored because every parameter is "
                "categorical (one conditioning class per level)",
                stacklevel=2,
            )
    if grid_size < 2:
        raise ValueError(f"grid_size must be >= 2, got {grid_size}")
    bw_factor = _resolve_bandwidth(bandwidth)
    if n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be >= 0, got {n_bootstrap}")
    if not 0 < conf_level < 1:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")

    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)
    _, T, K = Y_3d.shape
    D = problem.num_vars
    Y_cols = Y_3d.reshape(N, T * K)

    if slice_chunk_size is None:
        slice_chunk_size = max(1, _CHUNK_ELEM_BUDGET // (D * N * grid_size))
    elif slice_chunk_size < 1:
        raise ValueError(f"slice_chunk_size must be >= 1, got {slice_chunk_size}")

    # Replicate 0 is the identity permutation (the original sample); the
    # remaining rows are the bootstrap resamples. Building them together
    # means the point estimate and its interval share one code path.
    identity = jnp.arange(N, dtype=jnp.int32)[None, :]
    if n_bootstrap > 0:
        boot = jax.random.randint(
            jax.random.PRNGKey(seed), (n_bootstrap, N), 0, N, dtype=jnp.int32
        )
        all_idx = jnp.concatenate([identity, boot], axis=0)
    else:
        all_idx = identity
    R = all_idx.shape[0]

    # Canonical partition-group layout, shared with optimal_transport.
    groups, _, col_order = build_partition_groups(problem, X, all_idx, M, dims_levels)

    kernel = _get_delta_kernel(grid_size, bw_factor)

    total = T * K
    cs = min(slice_chunk_size, total)
    d_parts, s1_parts, degen_parts, floored_parts = [], [], [], []
    for start in range(0, total, cs):
        chunk = Y_cols[:, start : start + cs]
        d, s1, degen, floored = kernel(chunk, all_idx, tuple(groups))
        if col_order is not None:
            d = d[..., col_order]
            s1 = s1[..., col_order]
        d_parts.append(d)
        s1_parts.append(s1)
        degen_parts.append(degen)
        floored_parts.append(floored.any())

    if bool(jnp.stack(floored_parts).any()):
        warnings.warn(
            "jaxgsa: at least one conditioning class has zero sample "
            "variance (a point mass, e.g. a categorical level that maps to "
            "one output value). Its KDE bandwidth was floored to a narrow "
            "kernel; without the floor its density would drop out and bias "
            "delta low",
            stacklevel=2,
        )

    d_all = jnp.concatenate(d_parts, axis=1).reshape(R, T, K, D)
    s1_all = jnp.concatenate(s1_parts, axis=1).reshape(R, T, K, D)
    degen_all = jnp.concatenate(degen_parts, axis=1).reshape(R, T, K)

    d_hat = d_all[0]
    S1 = s1_all[0]

    delta_conf: Array | None = None
    S1_conf: Array | None = None
    if n_bootstrap > 0:
        # A constant bootstrap resample carries no information; replace its
        # degenerate zero with the point estimate so it neither inflates nor
        # deflates the bias correction (constant whole columns still give 0).
        degen_boot = degen_all[1:, ..., None]
        d_boot = jnp.where(degen_boot, d_hat[None], d_all[1:])
        s1_boot = jnp.where(degen_boot, S1[None], s1_all[1:])

        if bias_correct:
            # Plischke eqn 30 with d_hat on the original sample.
            d_reps = 2.0 * d_hat[None] - d_boot
            delta = jnp.nanmean(d_reps, axis=0)
        else:
            d_reps = d_boot
            delta = d_hat

        delta_conf = _squeeze_output_axes(
            _percentile_ci(d_reps, conf_level), squeeze_time, squeeze_output
        )
        S1_conf = _squeeze_output_axes(
            _percentile_ci(s1_boot, conf_level), squeeze_time, squeeze_output
        )
    else:
        delta = d_hat

    return DeltaResult(
        delta=_squeeze_output_axes(delta, squeeze_time, squeeze_output),
        delta_conf=delta_conf,
        S1=_squeeze_output_axes(S1, squeeze_time, squeeze_output),
        S1_conf=S1_conf,
        problem=problem,
    )


def _resolve_bandwidth(bandwidth: float | Literal["silverman"]) -> float | None:
    """Validate the ``bandwidth`` argument and return the KDE factor.

    Args:
        bandwidth: ``"silverman"`` for the per-class Silverman rule, or a
            positive real factor.

    Returns:
        ``None`` for the Silverman rule, otherwise the float factor.

    Raises:
        ValueError: If ``bandwidth`` is not ``"silverman"`` or a positive
            real number (booleans are rejected).
    """
    if isinstance(bandwidth, str):
        if bandwidth == "silverman":
            return None
        raise ValueError(f"bandwidth must be 'silverman' or a positive float, got {bandwidth!r}")
    # bool is an int subclass but is never a meaningful bandwidth factor.
    if isinstance(bandwidth, bool):
        raise ValueError(f"bandwidth must be 'silverman' or a positive float, got {bandwidth!r}")
    try:
        factor = float(bandwidth)
    except (TypeError, ValueError):
        raise ValueError(
            f"bandwidth must be 'silverman' or a positive float, got {bandwidth!r}"
        ) from None
    if not factor > 0:
        raise ValueError(f"bandwidth must be 'silverman' or a positive float, got {bandwidth!r}")
    return factor
