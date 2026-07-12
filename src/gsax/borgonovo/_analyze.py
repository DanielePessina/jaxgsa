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
from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from gsax._bootstrap import _percentile_ci
from gsax._normalization import _prepare_Y, _squeeze_output_axes, _validate_xy_inputs
from gsax.borgonovo._result import DeltaResult
from gsax.problem import Problem

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_MAX_CLASSES = 48
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


def _class_layout(N: int, M: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build static gather indices for equal-frequency rank classes.

    Class ``j`` holds the samples whose ordinal rank ``r`` (1-based)
    satisfies ``m[j] < r <= m[j+1]`` with ``m = linspace(0, N, M+1)`` --
    the same membership rule as SALib. Because floor of the shared float
    edges is used on both sides, class sizes (which differ by at most one)
    match SALib exactly; classes are padded to the largest size with a
    validity mask so downstream shapes stay static.

    Args:
        N: Number of samples.
        M: Number of classes.

    Returns:
        ``(take, mask, sizes)`` where ``take (M, P)`` indexes into a
        rank-sorted array (entries beyond a class's size are clamped),
        ``mask (M, P)`` flags valid entries, and ``sizes (M,)`` holds the
        true class sizes.
    """
    edges = np.floor(np.linspace(0.0, N, M + 1)).astype(np.int64)
    sizes = np.diff(edges)
    n_pad = int(sizes.max())
    take = edges[:-1, None] + np.arange(n_pad)[None, :]
    mask = np.arange(n_pad)[None, :] < sizes[:, None]
    take = np.minimum(take, N - 1).astype(np.int32)
    return take, mask, sizes


@jax.jit
def _build_class_indices(X: Array, all_idx: Array, take: Array) -> Array:
    """Global sample indices of every class, for every replicate.

    Computed once (never per output-column chunk) so the per-column kernel
    never re-ranks the inputs.

    Args:
        X: Input sample matrix ``(N, D)``.
        all_idx: Replicate row indices ``(R, N)`` (row 0 is the identity).
        take: Static per-class gather indices ``(M, P)`` from
            :func:`_class_layout`.

    Returns:
        Class indices ``(R, D, M, P)`` into the original sample.
    """

    def _one_replicate(r: Array) -> Array:
        # Rank on the ORIGINAL X (never downcast): ordinal ranks == a stable
        # argsort, so each class is a contiguous slice of the rank-sorted
        # resample; gathering ``r[orders]`` yields global indices of the
        # class members.
        orders = jnp.argsort(X[r], axis=0)  # (N, D)
        return r[orders].T[:, take]  # (D, M, P)

    return jax.vmap(_one_replicate)(all_idx)


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
        A jitted callable ``(Y_cols (N, C), all_idx (R, N),
        all_cls_idx (R, D, M, P), mask (M, P), counts (M,)) ->
        (d (R, C, D), s1 (R, C, D), degenerate (R, C))`` giving plug-in
        estimates for every replicate and a per-replicate/per-column flag
        marking constant resamples.
    """

    def _bandwidths(counts: Array, std: Array) -> Array:
        """Per-class (or full-sample) KDE bandwidths."""
        factor = (0.75 * counts) ** (-0.2) if bw_factor is None else bw_factor
        return factor * std

    def _kde_full(y: Array, grid: Array) -> Array:
        """Gaussian KDE of a full column ``y (N,)`` on ``grid (G,)``."""
        n = y.shape[0]
        std = jnp.std(y, ddof=1)
        h = _bandwidths(jnp.asarray(float(n), dtype=y.dtype), std)
        safe_h = jnp.where(h > 0, h, 1.0)
        u = (grid[:, None] - y[None, :]) / safe_h
        f = jnp.exp(-0.5 * u * u).sum(axis=1) / (n * safe_h * _SQRT_2PI)
        # Zero bandwidth (constant data) drops the density from the
        # integrand, mirroring SALib's degenerate-class treatment.
        return jnp.where(h > 0, f, 0.0)

    def _impl(
        Y_cols: Array,
        all_idx: Array,
        all_cls_idx: Array,
        mask: Array,
        counts: Array,
    ):
        dtype = jnp.result_type(Y_cols.dtype, jnp.float32)
        Y_cols = Y_cols.astype(dtype)
        mask = mask.astype(dtype)
        counts = counts.astype(dtype)
        n_total = counts.sum()  # == N

        # Output grids depend only on the original sample and are reused for
        # every replicate (SALib does the same).
        y_min = Y_cols.min(axis=0)
        y_max = Y_cols.max(axis=0)
        steps = jnp.linspace(0.0, 1.0, grid_size, dtype=dtype)
        grids = y_min[:, None] + steps[None, :] * (y_max - y_min)[:, None]  # (C, G)

        def _col_stats(y: Array, grid: Array, r: Array, cls_idx: Array):
            """Delta, S1, and a degeneracy flag for one column/replicate."""
            y_r = y[r]  # resampled column
            fy = _kde_full(y_r, grid)
            degenerate = y_r.max() == y_r.min()

            y_cls = y[cls_idx]  # (D, M, P) resampled class members
            mean = (y_cls * mask).sum(axis=-1) / counts  # (D, M)
            dev = (y_cls - mean[..., None]) * mask
            var = (dev**2).sum(axis=-1) / jnp.maximum(counts - 1.0, 1.0)
            h = _bandwidths(counts, jnp.sqrt(var))  # (D, M)
            safe_h = jnp.where(h > 0, h, 1.0)

            u = (grid[None, None, :, None] - y_cls[:, :, None, :]) / safe_h[..., None, None]
            k = jnp.exp(-0.5 * u * u) * mask[None, :, None, :]
            fyc = k.sum(axis=-1) / (counts[:, None] * safe_h[..., None] * _SQRT_2PI)
            fyc = jnp.where((h > 0)[..., None], fyc, 0.0)  # (D, M, G)

            l1 = jnp.trapezoid(jnp.abs(fy[None, None, :] - fyc), grid, axis=-1)
            delta = (counts[None, :] / (2.0 * n_total) * l1).sum(axis=-1)  # (D,)

            y_mean = y_r.mean()
            y_var = jnp.var(y_r)
            safe_var = jnp.where(y_var > 0, y_var, 1.0)
            Vi = (counts[None, :] / n_total * (mean - y_mean) ** 2).sum(axis=-1)
            s1 = jnp.where(y_var > 0, Vi / safe_var, 0.0)
            return delta, s1, degenerate

        def _one_replicate(carry, xs):
            r, cls_idx = xs
            d, s1, degen = jax.vmap(lambda y, grid: _col_stats(y, grid, r, cls_idx))(
                Y_cols.T, grids
            )
            return carry, (d, s1, degen)

        _, (d_all, s1_all, degen_all) = jax.lax.scan(_one_replicate, None, (all_idx, all_cls_idx))
        return d_all, s1_all, degen_all

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
    chunk_size: int | None = None,
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
            input. ``None`` selects the Plischke sample-size heuristic
            (SALib-identical, at most 48 classes).
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
        chunk_size: Number of flattened ``T*K`` output columns processed
            per kernel call. ``None`` picks a memory-aware default from the
            sample size; pass an explicit positive integer to override.
            Peak memory scales with ``chunk_size * D * N * grid_size``.

    Returns:
        DeltaResult with delta and S1 indices and optional confidence
        intervals. The underlying delta index is defined on ``[0, 1]`` and
        the plug-in estimate stays in that range, but the default
        bias-corrected estimate (and its confidence bounds) can fall
        marginally below 0 for weak/near-noninfluential inputs at small
        sample sizes. A constant output column yields ``delta = S1 = 0``
        (SALib raises an error in this case).

    Raises:
        ValueError: If X is not 2-D, its column count does not match the
            problem, Y is not 1-D/2-D/3-D, X and Y have differing row
            counts, ``n_classes`` is not in ``[2, N]``, ``grid_size < 2``,
            ``bandwidth`` is neither ``"silverman"`` nor a positive float,
            ``n_bootstrap < 0``, ``conf_level`` is not in ``(0, 1)``, or
            ``chunk_size`` is not a positive integer.
    """
    X = jnp.asarray(X)
    Y = _validate_xy_inputs(problem, X, Y)

    N = X.shape[0]
    if n_classes is None:
        M = _plischke_n_classes(N)
    else:
        if not 2 <= n_classes <= N:
            raise ValueError(f"n_classes must be in [2, N={N}], got {n_classes}")
        M = int(n_classes)
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

    if chunk_size is None:
        chunk_size = max(1, _CHUNK_ELEM_BUDGET // (D * N * grid_size))
    elif chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

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

    take_np, mask_np, sizes_np = _class_layout(N, M)
    take = jnp.asarray(take_np)
    mask = jnp.asarray(mask_np)
    counts = jnp.asarray(sizes_np)

    # Rank the inputs once for every replicate (never per output-column
    # chunk), then reuse the class indices across chunks.
    all_cls_idx = _build_class_indices(X, all_idx, take)

    kernel = _get_delta_kernel(grid_size, bw_factor)

    total = T * K
    cs = min(chunk_size, total)
    d_parts, s1_parts, degen_parts = [], [], []
    for start in range(0, total, cs):
        d, s1, degen = kernel(Y_cols[:, start : start + cs], all_idx, all_cls_idx, mask, counts)
        d_parts.append(d)
        s1_parts.append(s1)
        degen_parts.append(degen)

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
