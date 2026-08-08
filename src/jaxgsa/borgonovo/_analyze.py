"""Borgonovo delta analysis: moment-independent sensitivity from given data.

This module implements the Plischke, Borgonovo & Smith (2013) given-data
estimator of Borgonovo's (2007) delta index. For each parameter the estimator
splits the sample into equal-frequency classes by the parameter's rank. The
delta index is then the class-weighted L1 distance between the unconditional
output density and each conditional density. Both densities come from a
Gaussian kernel density estimate (KDE) with Silverman bandwidths, evaluated on
a fixed output grid and integrated by the trapezoid rule. The same partition
also yields a given-data first-order Sobol index at negligible cost.

The estimator supports a continuous output distribution only, because a KDE
on a fixed grid cannot represent an atom. ``analyze`` refuses a discrete
output up front and points the caller at ``jaxgsa.optimal_transport``.

The plug-in estimator is biased upward at finite N. By default the central
estimate is therefore bias-corrected with bootstrap resamples, as
``2*d_hat - mean(d_boot)`` (Plischke et al. eqn 30), where ``d_hat`` comes
from the original sample. The same replicates give the percentile confidence
intervals. The original sample and the bootstrap replicates run through one
scanned path, with the original sample as replicate 0 gathered by the identity
permutation. The point estimate and its interval are therefore always computed
under identical conventions.

Two things are computed once and reused. The class-partition indices are built
per parameter for every replicate, then reused across output-column chunks.
The JIT-compiled per-column kernel is cached only on the scalar estimator
settings ``(grid_size, bandwidth)``, so it captures no sample-sized constants.

The estimator details mirror ``SALib.analyze.delta``: an equal-frequency
ordinal rank partition, the Plischke class-count heuristic, Silverman KDE
factors, and a 100-point output grid. Three differences are deliberate.

1. The central estimate uses the original sample rather than a bootstrap
   resample, so it is deterministic given the data.
2. A constant output column yields ``delta = S1 = 0`` instead of an error.
3. A bootstrap replicate that happens to be constant contributes the point
   estimate rather than a spurious zero, so it neither adds nor removes bias.
   Such a replicate is reachable for rare-event outputs, and SALib raises
   ``LinAlgError`` on such data.

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
# bandwidth counts as degenerate. Two failures live below this line. The
# first is an exactly zero variance, that is a point mass, such as one
# categorical level that maps to one output value. The second is a spread so
# small that the shared output grid cannot resolve the conditional density.
# The second failure is the dangerous one. The grid step is
# (y_max - y_min) / (grid_size - 1). A class whose bandwidth is far below
# that step is sampled at a spacing of many sigma. The trapezoid rule then
# either misses the peak, or lands on it and integrates a spike of height
# ~1/(h*sqrt(2*pi)) over a step-wide interval. Either way the L1 distance,
# and with it delta, explodes far above 1. At the default grid_size = 100 the
# step is about 1/99 of the output range. A Silverman bandwidth of 1e-2 of
# the full-sample bandwidth is already at that scale, so 1e-2 is the
# threshold below which the grid is untrustworthy.
_DEGENERATE_BW_TOL = 1e-2
# Bandwidth given to a degenerate class, as a fraction of the full-sample
# Silverman bandwidth. The applied floor is ``max(fraction * h_full,
# grid_step)``. In the degenerate regime the grid-step term is almost always
# the larger of the two, so the grid sets the width. This fraction only
# binds for a wide output range with a fine grid. The grid-step bound is the
# load-bearing part. A Gaussian sampled at a spacing of at most its own
# sigma has small trapezoid error, while one sampled far more coarsely
# aliases and drives delta above 1. One consequence is that the delta of a
# near-degenerate class depends on the grid resolution and is biased low.
# The knob that moves it is ``grid_size``, not this fraction. See the
# ``analyze`` docstring.
_DEGENERATE_BW_FRACTION = 0.1
# Borgonovo's delta lies in [0, 1] by construction. The default
# bias-corrected estimate can leave that range by a little at small N, so
# only an excursion wider than this counts as an estimator failure.
_DELTA_RANGE_TOL = 0.05
# A discrete output breaks the estimator. The KDE of an atomic density is a
# spike the output grid cannot resolve, so delta is meaningless. An output
# column counts as discrete when it takes at most this many distinct values
# and those values are a vanishing fraction of the sample. Both conditions
# must hold, so the guard does not refuse a continuous output rounded to a
# few decimals, which has many distinct values at any useful N. A column
# with a single distinct value is exempt. A constant output needs no
# density, every conditional equals the unconditional, and delta = S1 = 0 is
# the exact answer rather than a failed computation. That contract predates
# this guard and stays.
_DISCRETE_MAX_DISTINCT = 20
_DISCRETE_DISTINCT_FRACTION = 0.01
# Target element budget for the default per-chunk working set. The dominant
# intermediate is the conditional-KDE tensor. Its size scales as
# ``chunk_columns * sum_g(Dg * Mg * Pg) * grid_size``. The default chunk
# width keeps that tensor near this many float32 elements (~256 MB).
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
def _get_delta_kernel(
    grid_size: int,
    bw_factor: float | None,
    degenerate_tol: float,
    degenerate_bw: float | None,
):
    """Return a JIT-compiled delta/S1 kernel for static estimator settings.

    The cache key holds only the scalar settings that change tracing: the grid
    size, the bandwidth branch, and the degenerate-class floor. The kernel
    takes all sample-sized data as runtime arguments, namely the outputs, the
    class indices and the masks. Nothing of size ``O(N)`` is therefore
    captured or baked into the compiled executable.

    Args:
        grid_size: Number of output-grid points for the KDE (static).
        bw_factor: KDE bandwidth factor multiplying the sample standard
            deviation, or ``None`` for the per-class Silverman rule.
        degenerate_tol: A class counts as degenerate when its bandwidth is
            below this fraction of the full-sample bandwidth.
        degenerate_bw: Bandwidth floor for a degenerate class, as a
            fraction of the full-sample bandwidth, applied exactly; or
            ``None`` for the default ``max(_DEGENERATE_BW_FRACTION *
            h_full, grid_step)``.

    Returns:
        A jitted callable ``(Y_cols (N, C), all_idx (R, N), groups) ->
        (d (R, C, D), s1 (R, C, D), degenerate (R, C), floored (R, C))``.
        ``d`` and ``s1`` are the plug-in estimates for every replicate.
        ``degenerate`` flags each replicate and column whose resample is
        constant. ``floored`` flags each column where a degenerate class
        engaged the bandwidth floor. ``groups`` is a tuple of canonical
        ``(cls_idx, counts)`` partition-group layouts from
        :func:`jaxgsa._core.partition.build_partition_groups`. The kernel
        processes every group in one call, computing the shared
        per-replicate statistics once, and concatenates the results on the
        parameter axis in group order. Zero-size classes carry zero weight.
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
        # A zero bandwidth means constant data. It drops the density from
        # the integrand, mirroring SALib's degenerate-class treatment.
        return jnp.where(h > 0, f, 0.0)

    def _impl(
        Y_cols: Array,
        all_idx: Array,
        groups: tuple[tuple[Array, Array], ...],
    ):
        dtype = jnp.result_type(Y_cols.dtype, jnp.float32)
        Y_cols = Y_cols.astype(dtype)
        N = Y_cols.shape[0]  # every row belongs to exactly one class per parameter

        # The output grids depend only on the original sample, so every
        # replicate reuses them. SALib does the same.
        y_min = Y_cols.min(axis=0)
        y_max = Y_cols.max(axis=0)
        steps = jnp.linspace(0.0, 1.0, grid_size, dtype=dtype)
        grids = y_min[:, None] + steps[None, :] * (y_max - y_min)[:, None]  # (C, G)

        counts_list = tuple(counts for _, counts in groups)
        cls_list = tuple(cls_idx for cls_idx, _ in groups)

        def _group_stats(y, grid, h_full, fy, y_mean, cls_idx, mask_b, counts_b):
            """Per-parameter delta/S1 numerators and floor flag for one group.

            ``mask_b``, shape ``(G, M, P)``, and ``counts_b``, shape
            ``(G, M)``, are both float and broadcast against the group's
            parameter axis Dg. ``G`` is either 1 or Dg. A zero-size class
            has zero standard deviation and therefore zero bandwidth, so
            its density is dropped. Its zero count also removes it from
            every weighted sum.
            """
            safe_counts = jnp.maximum(counts_b, 1.0)
            y_cls = y[cls_idx]  # (Dg, M, P) resampled class members
            mean = (y_cls * mask_b).sum(axis=-1) / safe_counts  # (Dg, M)
            dev = (y_cls - mean[..., None]) * mask_b
            var = (dev**2).sum(axis=-1) / jnp.maximum(counts_b - 1.0, 1.0)
            h = _bandwidths(safe_counts, jnp.sqrt(var))  # (Dg, M)
            # A degenerate class is one the output grid cannot resolve. A
            # zero-variance class gets bandwidth 0 and a zeroed density,
            # which biases delta far low. A class narrower than the grid
            # step aliases, which pushes delta far above 1. Floor both to a
            # kernel the grid can integrate. See the _DEGENERATE_BW_TOL and
            # _DEGENERATE_BW_FRACTION comments. The predicate keeps
            # resolvable classes bit-identical. It also stays False for a
            # constant column, where h_full == 0, which must keep its
            # delta = 0 contract.
            if degenerate_bw is None:
                floor = jnp.maximum(_DEGENERATE_BW_FRACTION * h_full, grid[1] - grid[0])
            else:
                floor = degenerate_bw * h_full
            floored_cls = (counts_b > 0) & (h < degenerate_tol * h_full)
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

            The per-replicate column statistics are the resampled column,
            the full KDE, the mean and the variance. This function computes
            them once and shares them with every partition group. It then
            concatenates the group results on the parameter axis in group
            order.
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


def _slice_label(flat_index: int, trailing: tuple[int, ...], problem: Problem) -> str:
    """Name one flattened output column for an error message.

    Args:
        flat_index: Index of the column in the flattened ``T*K`` axis.
        trailing: The shape of ``Y`` after the sample axis: ``()``,
            ``(K,)``, or ``(T, K)``.
        problem: Problem definition, which may carry ``output_names``.

    Returns:
        A short label such as ``"k=1 ('flux')"`` or ``"(t=3, k=0)"``. The
        label is empty for a single scalar output, which needs no name.
    """
    if len(trailing) == 0:
        return ""
    K = trailing[-1]
    names = problem.output_names
    t, k = divmod(flat_index, K)

    def _k_label(k_index: int) -> str:
        if names is not None and len(names) == K:
            return f"k={k_index} ('{names[k_index]}')"
        return f"k={k_index}"

    if len(trailing) == 1:
        return _k_label(flat_index)
    return f"(t={t}, {_k_label(k)})"


def _raise_discrete_output(problem: Problem, Y: Array) -> None:
    """Reject a discrete output up front.

    The estimator compares Gaussian kernel density estimates on a shared
    output grid. That construction needs a continuous output. A discrete
    output has atoms, and the density of an atom is a spike no grid
    resolves. The returned delta would then be an artifact of ``grid_size``
    rather than a property of the model. This check runs before any
    expensive work.

    An output column counts as discrete only when it takes at most
    ``_DISCRETE_MAX_DISTINCT`` distinct values and those values are fewer
    than ``_DISCRETE_DISTINCT_FRACTION`` of the sample. Both conditions
    must hold, so a continuous output rounded to a few decimals keeps
    working. A constant column, with one distinct value, is exempt. It
    needs no density, and ``delta = S1 = 0`` is its exact answer.

    Args:
        problem: Problem definition, used to name the offending column.
        Y: Validated output array ``(N,)``, ``(N, K)``, or ``(N, T, K)``.

    Raises:
        ValueError: If any output column is discrete.
    """
    N = Y.shape[0]
    trailing = Y.shape[1:]
    flat = Y.reshape(N, -1)
    if N < 2:
        return
    ordered = jnp.sort(flat, axis=0)
    n_distinct = np.asarray(1 + jnp.sum(ordered[1:] != ordered[:-1], axis=0))
    discrete = (
        (n_distinct > 1)
        & (n_distinct <= _DISCRETE_MAX_DISTINCT)
        & (n_distinct < _DISCRETE_DISTINCT_FRACTION * N)
    )
    bad = np.flatnonzero(discrete)
    if bad.size == 0:
        return

    if flat.shape[1] == 1:
        where = f"the output takes only {int(n_distinct[0])} distinct values"
    else:
        detail = ", ".join(
            f"{_slice_label(int(i), trailing, problem)}: {int(n_distinct[i])} distinct values"
            for i in bad[:5]
        )
        extra = f", and {bad.size - 5} more" if bad.size > 5 else ""
        where = f"{bad.size} of {flat.shape[1]} output columns are discrete ({detail}{extra})"

    raise ValueError(
        "jaxgsa.borgonovo.analyze supports a continuous output distribution "
        f"only, but {where} in {N} samples. The delta estimator compares "
        "Gaussian kernel density estimates on a shared output grid; an "
        "atomic density is a spike that no grid resolves, so the index "
        "would report the grid resolution, not the model. Use "
        "jaxgsa.optimal_transport.analyze for a discrete output: it "
        "compares empirical distributions directly and needs no density."
    )


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
    degenerate_tol: float = _DEGENERATE_BW_TOL,
    degenerate_bandwidth: float | Literal["auto"] = "auto",
) -> DeltaResult:
    """Compute Borgonovo delta and given-data first-order Sobol indices.

    The delta index measures how much knowing a parameter's value shifts the
    whole output density. ``delta_i`` is (half) the expected L1 distance
    between the unconditional output density and the density conditional on
    x_i. It lies in [0, 1]. A value of 0 means the output distribution does
    not change with x_i, and 1 means x_i fully determines it. Delta compares
    whole densities rather than variances, which makes it
    moment-independent. It therefore captures influence on tails and on
    shape that Sobol indices miss. It also needs no special sampling design,
    so any (X, Y) sample works. The same partition returns a given-data
    first-order Sobol index ``S1`` at negligible extra cost.

    The estimator supports correlated inputs. It partitions on the
    parameters' ordinal ranks and compares output densities, so a declared
    ``problem.correlation`` does not invalidate it. Under dependence the
    index measures each parameter's total association with the output. That
    total includes the effects carried by the parameters it correlates with.
    The companion ``S1`` is the given-data first-order Sobol index and reads
    the same way. Neither index separates the direct effect from the
    correlation-borne one. Use :mod:`jaxgsa.vkoga` or
    :mod:`jaxgsa.kucherenko` for that split; both need continuous inputs.

    Args:
        problem: Problem definition with D parameters.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        n_classes: Number of equal-frequency conditioning classes per
            continuous parameter. ``None`` selects the Plischke
            sample-size heuristic, which is identical to SALib's and uses
            at most 48 classes. A categorical parameter ignores this
            argument and always uses one class per level, with class sizes
            equal to the observed level counts. Declared levels with no
            observed samples are dropped with a warning. A passed value is
            always validated against ``[2, N]``. When every parameter is
            categorical, a ``UserWarning`` says the value is ignored.
        grid_size: Number of points of the output grid the densities are
            compared on. The grid spans ``[Y.min(), Y.max()]`` per column.
            It is also the resolution knob for near-degenerate
            conditioning classes; see the note below.
        bandwidth: KDE bandwidth rule. Use ``"silverman"`` for the
            per-class Silverman factor, or a positive float used directly
            as the factor multiplying the sample standard deviation.
        n_bootstrap: Number of bootstrap resamples for bias correction and
            confidence intervals. ``0`` skips both: the result is the
            plug-in estimate and ``delta_conf`` and ``S1_conf`` are
            ``None``.
        conf_level: Confidence level for percentile bootstrap intervals.
        bias_correct: Apply the Plischke bias reduction
            ``2*d_hat - mean(d_boot)`` to the delta estimate. It requires
            ``n_bootstrap > 0``. S1 is never bias-corrected, matching
            SALib.
        seed: Random seed for bootstrap resampling.
        slice_chunk_size: Number of flattened ``T*K`` output columns
            processed per kernel call. ``None`` picks a memory-aware
            default from the sample size. Pass a positive integer to
            override it. Peak memory scales with
            ``slice_chunk_size * grid_size`` times the summed padded class
            layout ``sum_g(Dg * Mg * Pg)``. That layout is about ``D * N``
            for continuous parameters. An imbalanced categorical parameter
            pads every level up to the largest one, so the layout can be
            many times ``D * N``. The default accounts for the real
            layout.
        degenerate_tol: A conditioning class counts as degenerate when its
            KDE bandwidth is below this fraction of the full-sample
            bandwidth. Degenerate classes get the floored bandwidth below.
            Lower it to let narrower classes keep their own bandwidth, but
            read the note about grid resolution first.
        degenerate_bandwidth: Bandwidth floor applied to a degenerate
            class. ``"auto"`` uses ``max(0.1 * h_full, grid_step)``, which
            never goes below what the output grid can integrate. A float
            is a fraction of the full-sample bandwidth ``h_full`` and is
            applied exactly, with no grid-step bound. A value far below
            ``grid_step / h_full`` aliases on the grid and returns delta
            far above 1.

    Note:
        This estimator supports a continuous output distribution only. It
        compares Gaussian kernel density estimates on a shared output grid,
        and a discrete output has atoms that no grid resolves. ``analyze``
        checks the output up front and raises ``ValueError`` when a column
        takes at most 20 distinct values and those values are fewer than 1%
        of the sample. Use :func:`jaxgsa.optimal_transport.analyze` for a
        discrete output: it compares empirical distributions directly and
        needs no density. The check does not refuse a continuous output
        rounded to a few decimals, and it does not refuse a constant
        column, whose exact answer is ``delta = S1 = 0``. A categorical
        parameter is still supported. The restriction is on the output.

    Note:
        A conditioning class can be a point mass or nearly one. That is the
        normal case for a categorical level that maps to one output value.
        For such a class the delta estimate depends on the grid resolution
        and is biased low. ``grid_step`` sets the floored bandwidth, so
        ``grid_size`` is the knob that moves the answer. On a noise-free
        three-atom model with true delta ``2/3``, the estimate goes 0.56 at
        ``grid_size=50``, 0.61 at 100, and 0.61 at 200 and above. The bias
        also does not vanish as N grows, so on atomic conditionals this
        estimator is not consistent. Treat delta on such parameters as a
        ranking signal, not a calibrated number. Parameters with genuine
        conditional spread are unaffected.

    Returns:
        A :class:`DeltaResult` with the delta and ``S1`` indices and the
        optional confidence intervals. The underlying delta index is
        defined on ``[0, 1]`` and the plug-in estimate stays in that range.
        The default bias-corrected estimate and its confidence bounds can
        fall marginally below 0 for weak or near-noninfluential parameters
        at small sample sizes. A constant output column yields
        ``delta = S1 = 0``, where SALib raises an error. A conditioning
        class the output grid cannot resolve gets a floored KDE bandwidth
        instead of its own, together with one ``UserWarning``; classes with
        genuine spread are unaffected. A confidence bound outside
        ``[0, 1]`` by more than 0.05 raises a ``UserWarning`` naming the
        parameter and the bound. The point estimate still stands in that
        case, so only the interval is suspect.

    Raises:
        ValueError: If any argument fails validation. The invalid cases are:
            X is not 2-D; the column count of X does not match the problem;
            Y is not 1-D, 2-D or 3-D; X and Y have differing row counts; a
            passed ``n_classes`` is not in ``[2, N]``; a categorical column
            of X holds values other than its integer level codes;
            ``grid_size < 2``; ``bandwidth`` is neither ``"silverman"`` nor
            a positive float; ``n_bootstrap < 0``; ``conf_level`` is not in
            ``(0, 1)``; ``slice_chunk_size`` is not a positive integer;
            ``degenerate_tol`` is not in ``[0, 1)``; or
            ``degenerate_bandwidth`` is neither ``"auto"`` nor a positive
            float. It is also raised when an output column is discrete,
            because the estimator supports a continuous output only; see
            the note above. It is raised again when the returned delta
            leaves ``[0, 1]`` by more than 0.05, which means the
            computation failed rather than returned an estimate.
    """
    X = jnp.asarray(X)
    # The delta estimator partitions on rank classes and compares output
    # densities, so a declared input correlation does not invalidate it.
    Y = _validate_xy_inputs(problem, X, Y, correlation_ok=True, categorical_ok=True)
    # Check the continuous-output contract before any expensive work.
    _raise_discrete_output(problem, Y)

    N = X.shape[0]
    # n_classes applies to the continuous columns only. A categorical column
    # always gets one conditioning class per level.
    dims_levels = _categorical_dims(problem)
    cat_dims = [d for d, _ in dims_levels]
    cont_dims = [d for d in range(problem.num_vars) if d not in set(cat_dims)]
    if n_classes is None:
        M = _plischke_n_classes(N)
    else:
        M = int(n_classes)
        # A passed value is always validated, even when nothing uses it.
        # Silently accepting nonsense hides bugs in the caller.
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
    degenerate_tol = float(degenerate_tol)
    if not 0 <= degenerate_tol < 1:
        raise ValueError(f"degenerate_tol must be in [0, 1), got {degenerate_tol}")
    degenerate_bw = _resolve_degenerate_bandwidth(degenerate_bandwidth)

    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)
    _, T, K = Y_3d.shape
    D = problem.num_vars
    Y_cols = Y_3d.reshape(N, T * K)

    if slice_chunk_size is not None and slice_chunk_size < 1:
        raise ValueError(f"slice_chunk_size must be >= 1, got {slice_chunk_size}")

    # Replicate 0 is the identity permutation, that is the original sample.
    # The remaining rows are the bootstrap resamples. Building them together
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

    kernel = _get_delta_kernel(grid_size, bw_factor, degenerate_tol, degenerate_bw)

    total = T * K
    if slice_chunk_size is None:
        # Peak memory is the conditional-KDE tensor, which holds one grid of
        # length grid_size per padded class slot. Size it from the real
        # per-group layout Dg * Mg * Pg. Assuming that product is ~ D * N
        # holds for equal-frequency continuous classes, but it under-counts
        # badly for an imbalanced categorical column, where every level is
        # padded up to the largest one.
        layout_elems = sum(g[0].shape[1] * g[0].shape[2] * g[0].shape[3] for g in groups)
        slice_chunk_size = max(1, _CHUNK_ELEM_BUDGET // (layout_elems * grid_size))
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
            "jaxgsa: at least one conditioning class is too narrow for the "
            "output grid to resolve (often a point mass, e.g. a categorical "
            "level that maps to one output value). Its KDE bandwidth was "
            "floored to a kernel the grid can integrate. Delta for such an "
            "input depends on grid_size and is biased low; raise grid_size "
            "if you need a calibrated value",
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
        # A constant bootstrap resample carries no information. Replace its
        # degenerate zero with the point estimate so it neither inflates nor
        # deflates the bias correction. A whole column that is constant
        # still gives 0.
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

    # The point estimate is the contract, so an out-of-range value is an
    # error. The interval is a diagnostic, so it only warns.
    _raise_delta_out_of_range(problem, delta, grid_size)
    if delta_conf is not None:
        _warn_conf_out_of_range(problem, delta_conf)

    return DeltaResult(
        delta=_squeeze_output_axes(delta, squeeze_time, squeeze_output),
        delta_conf=delta_conf,
        S1=_squeeze_output_axes(S1, squeeze_time, squeeze_output),
        S1_conf=S1_conf,
        problem=problem,
    )


def _out_of_range_columns(values: Array, num_vars: int) -> tuple[np.ndarray, np.ndarray]:
    """Find parameter columns whose estimates leave the delta range.

    Args:
        values: Estimates with the parameter axis last ``(..., D)``.
        num_vars: Number of parameters D.

    Returns:
        A tuple ``(flat, cols)`` where ``flat`` is the ``(-1, D)`` view of
        ``values`` and ``cols`` holds the indices of the failing columns.
    """
    flat = np.asarray(values).reshape(-1, num_vars)
    bad = ~np.isfinite(flat) | (flat < -_DELTA_RANGE_TOL) | (flat > 1.0 + _DELTA_RANGE_TOL)
    return flat, np.flatnonzero(bad.any(axis=0))


def _raise_delta_out_of_range(problem: Problem, delta: Array, grid_size: int) -> None:
    """Reject a delta point estimate that leaves ``[0, 1]``.

    Borgonovo's delta is a half L1 distance between probability densities,
    so it lies in ``[0, 1]`` by construction. A value outside that range is
    a failed computation rather than an estimate, so this function raises
    instead of returning it. The tolerance of 0.05 keeps the documented
    small negative excursion of the bias-corrected form
    ``2*d_hat - d_boot`` legal. The function never clips the value, because
    a clipped value is a plausible-looking wrong answer.

    Args:
        problem: Problem definition (for parameter names in the message).
        delta: Delta point estimates with the parameter axis last
            ``(..., D)``.
        grid_size: The output-grid size that produced the estimate, named
            in the message so the caller can act on it.

    Raises:
        ValueError: If any estimate is below ``-_DELTA_RANGE_TOL``, above
            ``1 + _DELTA_RANGE_TOL``, or not finite.
    """
    flat, cols = _out_of_range_columns(delta, problem.num_vars)
    if cols.size == 0:
        return
    detail = ", ".join(
        f"{problem.names[d]}: [{np.nanmin(flat[:, d]):.3g}, {np.nanmax(flat[:, d]):.3g}]"
        for d in cols
    )
    raise ValueError(
        "jaxgsa: delta is a half L1 distance between densities, so it lies "
        f"in [0, 1]. The estimate left that range for {detail}. This is a "
        "failed computation, not a result. The cause is a conditioning "
        "class the output grid cannot resolve: a class that is a point "
        "mass, or one much narrower than the grid step. Two knobs fix it. "
        f"Raise grid_size (currently {grid_size}) so the grid resolves the "
        "narrow class. Or raise degenerate_bandwidth, which widens the "
        "kernel given to such a class. The value is not clipped, because a "
        "clipped value would look plausible and still be wrong."
    )


def _warn_conf_out_of_range(problem: Problem, delta_conf: Array) -> None:
    """Warn when a delta confidence bound leaves ``[0, 1]``.

    The point estimate is the contract, and :func:`_raise_delta_out_of_range`
    checks it. The interval is only a diagnostic. An out-of-range bound
    degrades that diagnostic without invalidating the estimate, so this
    function warns instead of raising.

    Args:
        problem: Problem definition (for parameter names in the warning).
        delta_conf: Bootstrap interval ``(2, ..., D)`` with the lower bound
            first.

    Warns:
        UserWarning: If a bound leaves the range, naming the parameters and
            which bound failed.
    """
    parts = []
    for bound, arr in (("lower", delta_conf[0]), ("upper", delta_conf[1])):
        flat, cols = _out_of_range_columns(arr, problem.num_vars)
        if cols.size == 0:
            continue
        detail = ", ".join(
            f"{problem.names[d]}: [{np.nanmin(flat[:, d]):.3g}, {np.nanmax(flat[:, d]):.3g}]"
            for d in cols
        )
        parts.append(f"{bound} bound for {detail}")
    if not parts:
        return
    warnings.warn(
        "jaxgsa: delta is defined on [0, 1] but a bootstrap confidence "
        f"bound left that range — {'; '.join(parts)}. The point estimate is "
        "in range, so the index itself stands; the interval is the part to "
        "distrust. Raise n_bootstrap, or raise grid_size if a conditioning "
        "class is near degenerate.",
        stacklevel=3,
    )


def _resolve_degenerate_bandwidth(degenerate_bandwidth: float | Literal["auto"]) -> float | None:
    """Validate ``degenerate_bandwidth`` and return the floor fraction.

    Args:
        degenerate_bandwidth: ``"auto"`` for the grid-aware default floor,
            or a positive fraction of the full-sample bandwidth.

    Returns:
        ``None`` for the default floor, otherwise the float fraction.

    Raises:
        ValueError: If the value is not ``"auto"`` or a positive real
            number (booleans are rejected).
    """
    if isinstance(degenerate_bandwidth, str):
        if degenerate_bandwidth == "auto":
            return None
        raise ValueError(
            "degenerate_bandwidth must be 'auto' or a positive float, "
            f"got {degenerate_bandwidth!r}"
        )
    if isinstance(degenerate_bandwidth, bool):
        raise ValueError(
            "degenerate_bandwidth must be 'auto' or a positive float, "
            f"got {degenerate_bandwidth!r}"
        )
    try:
        fraction = float(degenerate_bandwidth)
    except (TypeError, ValueError):
        raise ValueError(
            "degenerate_bandwidth must be 'auto' or a positive float, "
            f"got {degenerate_bandwidth!r}"
        ) from None
    if not fraction > 0:
        raise ValueError(
            "degenerate_bandwidth must be 'auto' or a positive float, "
            f"got {degenerate_bandwidth!r}"
        )
    return fraction


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
