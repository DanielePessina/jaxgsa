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
the same replicates. Class-partition constants depend only on static
``(N, M)``, so a single JIT-compiled kernel (one compilation per unique
``(N, M, grid_size, bandwidth)`` and batch shape) is vmapped over the
flattened ``T*K`` output columns and scanned over bootstrap replicates.

Estimator details mirror ``SALib.analyze.delta`` (equal-frequency ordinal
rank partition, Plischke class-count heuristic, Silverman KDE factors,
100-point output grid) except that the central estimate uses the original
sample rather than a bootstrap resample, and a constant output column
yields ``delta = S1 = 0`` instead of an error.

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

from gsax._normalization import _prepare_Y
from gsax.borgonovo._result import DeltaResult
from gsax.problem import Problem

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_MAX_CLASSES = 48


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
    return int(np.round(min(int(np.ceil(n_samples**exponent)), _MAX_CLASSES)))


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


@lru_cache(maxsize=32)
def _get_delta_kernel(N: int, M: int, grid_size: int, bw_factor: float | None):
    """Return a JIT-compiled delta/S1 kernel for static estimator settings.

    Args:
        N: Number of samples (static; fixes the class layout).
        M: Number of equal-frequency classes (static).
        grid_size: Number of output-grid points for the KDE (static).
        bw_factor: KDE bandwidth factor multiplying the sample standard
            deviation, or ``None`` for the per-class Silverman rule.

    Returns:
        A jitted callable ``(X (N, D), Y_cols (N, C), boot_idx (B, N)) ->
        (d_hat (C, D), s1_hat (C, D), d_boot (B, C, D), s1_boot (B, C, D))``
        computing plug-in estimates on the original sample and on each
        bootstrap replicate.
    """
    take_np, mask_np, sizes_np = _class_layout(N, M)
    take = jnp.asarray(take_np)

    def _bandwidths(counts: Array, std: Array) -> Array:
        """Per-class KDE bandwidths (Silverman rule or fixed factor)."""
        if bw_factor is None:
            factor = (0.75 * counts) ** (-0.2)
        else:
            factor = bw_factor
        return factor * std

    def _kde_full(y: Array, grid: Array) -> Array:
        """Gaussian KDE of a full column ``y (N,)`` on ``grid (G,)``."""
        std = jnp.std(y, ddof=1)
        h = _bandwidths(jnp.asarray(float(N), dtype=y.dtype), std)
        safe_h = jnp.where(h > 0, h, 1.0)
        u = (grid[:, None] - y[None, :]) / safe_h
        f = jnp.exp(-0.5 * u * u).sum(axis=1) / (N * safe_h * _SQRT_2PI)
        # Zero-bandwidth (constant data) mirrors SALib's degenerate-class
        # treatment: the density is dropped from the integrand.
        return jnp.where(h > 0, f, 0.0)

    def _impl(X: Array, Y_cols: Array, boot_idx: Array):
        dtype = jnp.result_type(Y_cols.dtype, jnp.float32)
        X = X.astype(dtype)
        Y_cols = Y_cols.astype(dtype)
        mask = jnp.asarray(mask_np, dtype=dtype)
        counts = jnp.asarray(sizes_np, dtype=dtype)

        # Output grids depend only on the original sample and are reused
        # for every bootstrap replicate (SALib does the same).
        y_min = Y_cols.min(axis=0)
        y_max = Y_cols.max(axis=0)
        steps = jnp.linspace(0.0, 1.0, grid_size, dtype=dtype)
        grids = y_min[:, None] + steps[None, :] * (y_max - y_min)[:, None]  # (C, G)

        def _col_stats(
            y: Array,
            grid: Array,
            fy: Array,
            cls_idx: Array,
            y_mean: Array,
            y_var: Array,
        ) -> tuple[Array, Array]:
            """Delta and S1 for one column given class gather indices.

            Args:
                y: Original output column ``(N,)`` (gathered via ``cls_idx``).
                grid: Output grid ``(G,)``.
                fy: Unconditional KDE on the grid ``(G,)``.
                cls_idx: Global sample indices per input and class
                    ``(D, M, P)``.
                y_mean: Mean of the (re)sampled column.
                y_var: Population variance of the (re)sampled column.

            Returns:
                ``(delta, s1)`` plug-in estimates, each ``(D,)``.
            """
            y_cls = y[cls_idx]  # (D, M, P)
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
            delta = (counts[None, :] / (2.0 * N) * l1).sum(axis=-1)  # (D,)

            Vi = (counts[None, :] / N * (mean - y_mean) ** 2).sum(axis=-1)
            s1 = jnp.where(y_var > 0, Vi / jnp.where(y_var > 0, y_var, 1.0), 0.0)
            return delta, s1

        fy0 = jax.vmap(_kde_full)(Y_cols.T, grids)  # (C, G)
        # Ordinal ranks == stable argsort; classes are contiguous slices of
        # the rank-sorted sample, so gathering ``order[take]`` yields global
        # indices of each class's members.
        orders0 = jnp.argsort(X, axis=0)  # (N, D)
        cls_idx0 = orders0.T[:, take]  # (D, M, P)
        d_hat, s1_hat = jax.vmap(
            lambda y, grid, fy: _col_stats(y, grid, fy, cls_idx0, y.mean(), jnp.var(y))
        )(Y_cols.T, grids, fy0)

        def _boot_step(carry: None, r: Array):
            """Plug-in estimates on one bootstrap replicate ``r (N,)``."""
            orders_b = jnp.argsort(X[r], axis=0)
            # Composing the resample with its sort order gives global
            # indices, so columns are gathered without materializing X[r]
            # per column.
            cls_idx_b = r[orders_b].T[:, take]  # (D, M, P)

            def _one_col(y: Array, grid: Array) -> tuple[Array, Array]:
                y_b = y[r]
                fy_b = _kde_full(y_b, grid)
                return _col_stats(y, grid, fy_b, cls_idx_b, y_b.mean(), jnp.var(y_b))

            return carry, jax.vmap(_one_col)(Y_cols.T, grids)

        if boot_idx.shape[0] > 0:
            _, (d_boot, s1_boot) = jax.lax.scan(_boot_step, None, boot_idx)
        else:
            d_boot = jnp.zeros((0,) + d_hat.shape, dtype=dtype)
            s1_boot = jnp.zeros((0,) + s1_hat.shape, dtype=dtype)

        return d_hat, s1_hat, d_boot, s1_boot

    return jax.jit(_impl)


def _squeeze_result(arr: Array, squeeze_time: bool, squeeze_output: bool) -> Array:
    """Remove the singleton T/K axes inserted by ``_prepare_Y``.

    Args:
        arr: Index array shaped ``(..., T, K, D)``.
        squeeze_time: Whether the time axis was inserted.
        squeeze_output: Whether the output axis was inserted.

    Returns:
        Array with the inserted singleton axes removed.
    """
    if squeeze_time and squeeze_output:
        return arr[..., 0, 0, :]
    if squeeze_time:
        return arr[..., 0, :, :]
    return arr


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
    chunk_size: int = 2048,
) -> DeltaResult:
    """Compute Borgonovo delta and given-data first-order Sobol indices.

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
            per kernel call. Peak memory scales with
            ``chunk_size * D * N * grid_size``; lower it for large
            time-series outputs.

    Returns:
        DeltaResult with delta and S1 indices and optional confidence
        intervals. A constant output column yields ``delta = S1 = 0``
        (SALib raises an error in this case).

    Raises:
        ValueError: If X is not 2-D, its column count does not match the
            problem, X and Y have differing row counts, ``n_classes`` is
            not in ``[2, N]``, ``grid_size < 2``, ``bandwidth`` is neither
            ``"silverman"`` nor a positive float, ``n_bootstrap < 0``,
            ``conf_level`` is not in ``(0, 1)``, or ``chunk_size < 1``.
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

    N = X.shape[0]
    if n_classes is None:
        M = _plischke_n_classes(N)
    else:
        if not 2 <= n_classes <= N:
            raise ValueError(f"n_classes must be in [2, N={N}], got {n_classes}")
        M = int(n_classes)
    if grid_size < 2:
        raise ValueError(f"grid_size must be >= 2, got {grid_size}")
    if bandwidth == "silverman":
        bw_factor = None
    else:
        bw_factor = float(bandwidth)
        if not bw_factor > 0:
            raise ValueError(
                f"bandwidth must be 'silverman' or a positive float, got {bandwidth!r}"
            )
    if n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be >= 0, got {n_bootstrap}")
    if not 0 < conf_level < 1:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)
    _, T, K = Y_3d.shape
    Y_cols = Y_3d.reshape(N, T * K)

    if n_bootstrap > 0:
        boot_idx = jax.random.randint(jax.random.PRNGKey(seed), (n_bootstrap, N), 0, N)
    else:
        boot_idx = jnp.zeros((0, N), dtype=jnp.int32)

    kernel = _get_delta_kernel(N, M, grid_size, bw_factor)

    total = T * K
    cs = min(chunk_size, total)
    d_parts, s1_parts, db_parts, s1b_parts = [], [], [], []
    for start in range(0, total, cs):
        d, s1, d_b, s1_b = kernel(X, Y_cols[:, start : start + cs], boot_idx)
        d_parts.append(d)
        s1_parts.append(s1)
        db_parts.append(d_b)
        s1b_parts.append(s1_b)

    D = problem.num_vars
    d_hat = jnp.concatenate(d_parts, axis=0).reshape(T, K, D)
    S1 = jnp.concatenate(s1_parts, axis=0).reshape(T, K, D)

    delta_conf: Array | None = None
    S1_conf: Array | None = None
    if n_bootstrap > 0:
        d_boot = jnp.concatenate(db_parts, axis=1).reshape(n_bootstrap, T, K, D)
        s1_boot = jnp.concatenate(s1b_parts, axis=1).reshape(n_bootstrap, T, K, D)

        if bias_correct:
            # Plischke eqn 30 with d_hat on the original sample: the
            # corrected replicates average to 2*d_hat - mean(d_boot).
            d_reps = 2.0 * d_hat[None] - d_boot
            delta = d_reps.mean(axis=0)
        else:
            d_reps = d_boot
            delta = d_hat

        alpha = (1.0 - conf_level) / 2.0
        percentiles = jnp.array([alpha * 100, (1.0 - alpha) * 100])
        delta_conf = _squeeze_result(
            jnp.percentile(d_reps, percentiles, axis=0), squeeze_time, squeeze_output
        )
        S1_conf = _squeeze_result(
            jnp.percentile(s1_boot, percentiles, axis=0), squeeze_time, squeeze_output
        )
    else:
        delta = d_hat

    return DeltaResult(
        delta=_squeeze_result(delta, squeeze_time, squeeze_output),
        delta_conf=delta_conf,
        S1=_squeeze_result(S1, squeeze_time, squeeze_output),
        S1_conf=S1_conf,
        problem=problem,
    )
