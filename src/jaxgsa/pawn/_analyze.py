"""PAWN index estimators for distribution-based sensitivity analysis.

The Kolmogorov-Smirnov (KS) distance is the largest vertical gap between two
empirical CDFs. These estimators take that distance between the
unconditional output CDF and the conditional CDF that holds one parameter
inside a bin of its range. The PAWN index aggregates the KS values across
bins by median, max, or mean.

Array shape conventions used throughout:
    N  — number of samples
    D  — number of parameters
    T  — number of time steps (singleton-squeezed when absent)
    K  — number of output variables (singleton-squeezed when absent)

The KS computation is vectorized. Each output column is sorted once, and the
unconditional and per-bin conditional ECDFs are compared only at distinct
output values, that is at tie-group ends. This makes the statistic equal to
the two-sample KS of ``scipy.stats.ks_2samp`` for tied and discrete outputs,
not only for continuous ones.

Bin assignment (via ``searchsorted``) depends only on the inputs, so it is
computed once and shared across all output columns. One JIT-compiled kernel
is vmapped over the flattened ``T*K`` output columns, with one compilation
per unique ``(N, D, n_bins)`` and output-column count. The columns run in
chunks of at most ``slice_chunk_size``, which bounds the working arrays, so
the column count is the chunk width and a trailing part-chunk adds at most
one more compilation. The ``statistic`` aggregation runs outside JIT so all
three statistics share one compilation.

The kernel's working set is not the ``(chunk, D, n_bins)`` result. The
nested ``vmap`` builds a full ``(N, n_bins)`` ECDF table per (column,
parameter) pair, so the live arrays are ``(chunk, D, N, n_bins)``. That is
``N`` times larger than the result, which is why the default chunk width is
derived from ``N``, ``D`` and ``n_bins`` rather than fixed.

A categorical parameter needs no binning: its level code is already a bin
index. Continuous and categorical columns share one kernel, compiled at
``n_eff = max(n_bins, max_levels)``. The shorter columns leave their
trailing bins empty, the kernel returns ``NaN`` for an empty bin, and the
nan-aware aggregation drops it. A continuous-only problem has
``n_eff == n_bins`` and is bit-for-bit unaffected.

References:
    Pianosi & Wagener (2015). A simple and efficient method for global
    sensitivity analysis based on cumulative distribution functions.
    Environmental Modelling & Software 67:1-11.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.batching import get_memory_budget
from jaxgsa._core.bootstrap import _bootstrap_ci_endpoints
from jaxgsa._core.entry import (
    at_least,
    check_scalars,
    in_open_interval,
    one_of,
    prepare,
    require,
)
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.partition import _extract_categorical_codes
from jaxgsa._core.result import CIInfo
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa._core.validation import _prepare_Y
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.pawn._result import PAWNResult
from jaxgsa.problem import Problem, _categorical_dims

# Fewest samples that still define both an unconditional and a conditional
# ECDF, so that a KS distance is a distance and not an artefact.
_MIN_KEPT = 4


def _equal_width_bins(X_u01: Array, n_bins: int) -> Array:
    """Assign each sample to an equal-width bin in ``[0, 1]``.

    A sample outside ``[0, 1]`` gets the sentinel ``-1``, so it matches no bin
    and drops out of every conditional set. This happens when the caller
    passes inputs beyond the declared bounds of a ``uniform`` parameter. A
    value of exactly ``0.0`` maps to the first bin and ``1.0`` maps to the
    last bin.

    A non-finite input takes the same sentinel. Under the two policies that
    remove or refuse it, ``"raise"`` and ``"drop"``, :func:`analyze` settles
    it before this point, so a failed model run is reported instead of
    quietly leaving every conditional set. Under ``"propagate"`` the value
    does reach here and is silently binned out, which is the behaviour that
    policy asks for -- the caller was told the count and the row numbers
    first.

    Args:
        X_u01: Unit-interval inputs, shape ``(N, D)``.
        n_bins: Number of equal-width bins.

    Returns:
        Integer bin indices, shape ``(N, D)``, in ``[0, n_bins)``, or ``-1``
        for out-of-range or ``NaN`` samples.
    """
    bin_edges = jnp.linspace(0.0, 1.0, n_bins + 1)
    idx = jnp.clip(jnp.searchsorted(bin_edges, X_u01, side="right") - 1, 0, n_bins - 1)
    in_range = (X_u01 >= 0.0) & (X_u01 <= 1.0)
    return jnp.where(in_range, idx, -1)


def _bin_indices(problem: Problem, X: Array, n_bins: int) -> tuple[Array, int]:
    """Assign every parameter column to conditioning bins, categorical or not.

    A continuous column keeps equal-width binning on the CDF-transformed unit
    interval, which is equal-probability under the column's marginal. A
    categorical column needs no binning at all. Its level code ``0..L-1``
    already names the conditioning class, so the code serves as the bin index.

    The two kinds share one kernel, so the kernel is compiled at
    ``n_eff = max(n_bins, max_levels)`` and the shorter columns leave their
    trailing bins empty. An empty bin yields ``NaN`` from the KS kernel and
    the nan-aware aggregation drops it, so the padding changes no index. A
    continuous-only problem has ``n_eff == n_bins`` and is unaffected.

    Nothing here reads an array value on the host, and every branch is on
    ``problem`` metadata, which is a plain Python value and never a tracer. So
    this runs under ``jit``, ``vmap`` and ``jacrev``. Refusing a malformed
    categorical column is policy and needs a concrete value, so it lives in
    :func:`_check_categorical_codes`, which :func:`analyze` calls and
    :func:`indices` does not. A code outside ``[0, L)`` takes the same ``-1``
    sentinel a continuous out-of-range sample takes, so it drops out of every
    conditional set instead of aliasing onto a real level.

    Args:
        problem: Problem definition with D parameters.
        X: Input samples in physical units, shape ``(N, D)``. A categorical
            column holds its integer level codes as floats.
        n_bins: Number of conditioning bins for the continuous columns.

    Returns:
        ``(bin_idx, n_eff)``. ``bin_idx`` holds integer bin indices of shape
        ``(N, D)``, where ``-1`` marks an excluded sample. ``n_eff`` is the
        padded bin count the kernel must be compiled at.
    """
    dims_levels = _categorical_dims(problem)
    if not dims_levels:
        return _equal_width_bins(cdf_to_unit_interval(X, problem), n_bins), n_bins

    n_eff = max(n_bins, max(n_levels for _, n_levels in dims_levels))
    cat_positions = {d: j for j, (d, _) in enumerate(dims_levels)}

    # cdf_to_unit_interval refuses a categorical parameter by design, so the
    # continuous columns are transformed through a continuous-only problem.
    cont_dims = [d for d in range(problem.num_vars) if d not in cat_positions]
    cont_columns: list[Array] = []
    if cont_dims:
        cont_problem = Problem._from_normalized_inputs(
            names=tuple(problem.names[d] for d in cont_dims),
            input_specs=tuple(problem.input_specs[d] for d in cont_dims),
        )
        cont_idx = _equal_width_bins(
            cdf_to_unit_interval(X[:, jnp.asarray(cont_dims)], cont_problem), n_bins
        )
        cont_columns = [cont_idx[:, j].astype(jnp.int32) for j in range(len(cont_dims))]

    cont_iter = iter(cont_columns)
    levels_of_dim = dict(dims_levels)
    columns = [
        _level_codes(X[:, d], levels_of_dim[d]) if d in cat_positions else next(cont_iter)
        for d in range(problem.num_vars)
    ]
    return jnp.stack(columns, axis=1), n_eff


def _level_codes(column: Array, n_levels: int) -> Array:
    """Read one categorical column's level codes as bin indices.

    Args:
        column: One column of ``X``, shape ``(N,)``, holding integer level
            codes stored as floats.
        n_levels: Number of declared levels for that parameter.

    Returns:
        Integer bin indices, shape ``(N,)``, in ``[0, n_levels)``, or ``-1``
        for a value that is not one of the declared codes.
    """
    rounded = jnp.round(column)
    valid = (rounded == column) & (rounded >= 0) & (rounded < n_levels)
    return jnp.where(valid, rounded, -1).astype(jnp.int32)


def _check_categorical_codes(problem: Problem, X: Array) -> None:
    """Refuse a categorical column that does not hold its declared codes.

    This is the host-side half of :func:`_bin_indices`: it needs concrete
    values, so it belongs to :func:`analyze` and not to :func:`indices`.
    Without it a mistyped column would silently take the ``-1`` sentinel and
    empty every conditioning class.

    Args:
        problem: Problem definition with D parameters.
        X: Input samples in physical units, shape ``(N, D)``.

    Raises:
        ValueError: If a categorical column holds non-integral values or
            codes outside ``[0, L)``.
    """
    dims_levels = _categorical_dims(problem)
    if dims_levels:
        _extract_categorical_codes(problem, np.asarray(X), dims_levels)


@lru_cache(maxsize=32)
def _get_pawn_ks(n_bins: int):
    """Return a JIT-compiled KS kernel for the given ``n_bins``.

    The kernel maps precomputed bin indices and a batch of output columns
    to per-bin KS statistics, vmapped over the output columns.

    Args:
        n_bins: Number of equal-width conditioning bins (static).

    Returns:
        A jitted callable that maps ``bin_idx`` of shape ``(N, D)`` and
        ``Y_cols`` of shape ``(N, M)`` to KS statistics of shape
        ``(M, D, n_bins)``, where ``M`` is the number of output columns.
    """

    def _impl(bin_idx: Array, Y_cols: Array) -> Array:
        """Compute per-bin KS statistics for every output column.

        Args:
            bin_idx: Integer bin indices, shape ``(N, D)``, where ``-1``
                marks an excluded sample.
            Y_cols: Output columns, shape ``(N, M)``.

        Returns:
            KS statistics, shape ``(M, D, n_bins)``, with ``NaN`` where a bin
            has fewer than two samples.
        """
        N = bin_idx.shape[0]
        param_bins = jnp.arange(n_bins)

        def _ks_one_col(y: Array) -> Array:
            """Compute KS statistics of shape ``(D, n_bins)`` for one output column."""
            dtype = jnp.result_type(y.dtype, jnp.float32)
            order = jnp.argsort(y)
            y_sorted = y[order]
            # A sorted position is a "group end" if it is the last member of
            # its tie group. Evaluating the ECDFs only there yields the
            # value-based (tie-aware) two-sample KS statistic.
            is_group_end = jnp.concatenate([y_sorted[:-1] != y_sorted[1:], jnp.array([True])])
            bin_idx_sorted = bin_idx[order]
            uncond_cdf = jnp.arange(1, N + 1, dtype=dtype) / N

            def _ks_one_param(bid: Array) -> Array:
                """Compute KS statistics of shape ``(n_bins,)`` for one parameter."""
                oh = (bid[:, None] == param_bins[None, :]).astype(dtype)
                cnt = oh.sum(axis=0)
                cum = jnp.cumsum(oh, axis=0)
                ccdf = cum / jnp.maximum(cnt[None, :], 1.0)
                diff = jnp.abs(uncond_cdf[:, None] - ccdf)
                diff = jnp.where(is_group_end[:, None], diff, -jnp.inf)
                ks = jnp.max(diff, axis=0)
                return jnp.where(cnt >= 2, ks, jnp.nan)

            return jax.vmap(_ks_one_param)(bin_idx_sorted.T)

        return jax.vmap(_ks_one_col, in_axes=1)(Y_cols)

    return jax.jit(_impl)


def _resolve_slice_chunk_size(
    slice_chunk_size: int | None,
    N: int,
    D: int,
    n_eff: int,
    itemsize: int,
) -> int:
    """Resolve the number of output columns per kernel call.

    An explicit value is honoured as given. ``None`` derives one from the
    active memory budget (:func:`jaxgsa._core.batching.get_memory_budget`)
    and the kernel's real working set, which is ``chunk * D * N * n_eff``
    elements, not the ``chunk * D * n_eff`` result. Two such arrays are live
    at once inside the kernel, so the estimate carries a factor of two.

    Args:
        slice_chunk_size: Caller's value, or ``None`` to derive one.
        N: Number of samples.
        D: Number of parameters.
        n_eff: Number of bins the kernel is compiled at.
        itemsize: Bytes per element of the kernel's working dtype.

    Returns:
        A chunk width of at least 1.
    """
    if slice_chunk_size is not None:
        return slice_chunk_size
    bytes_per_column = 2 * D * N * n_eff * itemsize
    return max(1, get_memory_budget() // max(bytes_per_column, 1))


def _aggregate_ks(
    ks: Array,
    statistic: Literal["median", "max", "mean"],
) -> Array:
    """Reduce KS values over the trailing ``n_bins`` axis to PAWN indices.

    All three statistics are nan-aware. A bin with fewer than two samples is
    dropped rather than propagated, whether it is an empty padding bin or a
    genuinely under-filled one. A parameter whose bins are all empty
    therefore yields ``NaN``, which is the honest answer.

    Args:
        ks: KS statistics with bins on the last axis, shape ``(..., n_bins)``.
            The function ignores ``NaN`` entries, which mark bins with fewer
            than two samples.
        statistic: Aggregation across bins.

    Returns:
        PAWN indices with the ``n_bins`` axis removed.
    """
    if statistic == "median":
        return jnp.nanmedian(ks, axis=-1)
    if statistic == "max":
        return jnp.nanmax(ks, axis=-1)
    return jnp.nanmean(ks, axis=-1)


def _pawn_core(
    bin_idx: Array,
    Y_3d: Array,
    n_eff: int,
    statistic: Literal["median", "max", "mean"],
    slice_chunk_size: int | None,
    *,
    warn: bool = True,
) -> Array:
    """Compute PAWN indices over all (T, K) output slices.

    The KS kernel is vmapped over the flattened ``T*K`` output columns. Its
    working set is ``(chunk, D, N, n_eff)``, not the ``(chunk, D, n_eff)``
    result: the inner ``vmap`` builds a full ``(N, n_eff)`` ECDF table per
    (column, parameter) pair. The columns are processed in chunks of at most
    ``slice_chunk_size`` so those arrays stay bounded when ``T*K`` is large.
    Each column is independent of every other, so the chunked result is
    bit-for-bit the unchunked one.

    Args:
        bin_idx: Conditioning-bin indices from :func:`_bin_indices`, shape
            ``(N, D)``, where ``-1`` marks an excluded sample.
        Y_3d: Output array promoted to shape ``(N, T, K)``.
        n_eff: Number of bins the kernel is compiled at.
        statistic: Aggregation method across bins.
        slice_chunk_size: Maximum number of flattened ``T*K`` output columns
            per kernel call, or ``None`` to derive one from the memory
            budget. An explicit value must be at least 1.
        warn: If True, emit one warning per parameter whose bins all hold
            fewer than two samples. Pass False for bootstrap resamples to
            avoid duplicate warnings.

    Returns:
        PAWN indices, shape ``(T, K, D)``.
    """
    N, T, K = Y_3d.shape
    D = bin_idx.shape[1]

    # Bin assignment depends only on the inputs, so it is built once by the
    # caller and every output column runs through the same vmapped kernel.
    Y_cols = Y_3d.reshape(N, T * K)
    total = T * K
    # The kernel promotes its working dtype to at least float32.
    itemsize = jnp.result_type(Y_3d.dtype, jnp.float32).itemsize
    cs = min(_resolve_slice_chunk_size(slice_chunk_size, N, D, n_eff, itemsize), total)
    kernel = _get_pawn_ks(n_eff)

    # The KS values of one output column depend on no other column, so the
    # aggregation runs inside the loop and only the (chunk, D) result is kept.
    pawn_parts: list[Array] = []
    empty_parts: list[Array] = []
    for start in range(0, total, cs):
        ks = kernel(bin_idx, Y_cols[:, start : start + cs])  # (chunk, D, n_eff)
        pawn_parts.append(_aggregate_ks(ks, statistic))  # (chunk, D)
        if warn:
            empty_parts.append(jnp.all(jnp.isnan(ks), axis=(0, 2)))

    if warn:
        # Whether every bin of a parameter is empty depends only on the
        # binning and not on Y, so the answer is identical across output
        # columns and across chunks. One host sync, one warning per affected
        # parameter.
        all_empty = jnp.all(jnp.stack(empty_parts), axis=0).tolist()
        for d, empty in enumerate(all_empty):
            if empty:
                warnings.warn(
                    f"PAWN: all bins empty for parameter {d}, returning NaN",
                    stacklevel=3,
                    category=JaxgsaWarning,
                )

    pawn = jnp.concatenate(pawn_parts, axis=0)  # (T*K, D)
    return pawn.reshape(T, K, D)


def indices(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_bins: int = 10,
    statistic: Literal["median", "max", "mean"] = "median",
    slice_chunk_size: int | None = None,
) -> tuple[Array]:
    """Compute PAWN indices as plain arrays, with no diagnostics.

    This is the transformable core of :func:`analyze`. It runs the same KS
    kernel on the same data and returns the same numbers, but it does nothing
    else: no non-finite check, no zero-variance warning, no empty-bin warning,
    no categorical-code check, no :class:`jaxgsa.pawn.PAWNResult`, and no read
    of any array value on the host. So it composes with ``jax.jit``,
    ``jax.vmap``, ``jax.grad`` and ``jax.jacrev``, which :func:`analyze`
    cannot, because a policy decision needs a concrete value and a tracer has
    none.

    It returns the point estimate only. A bootstrap is policy — it draws
    randomness, needs a ``key``, and forms an interval — so it stays in
    :func:`analyze`, which offers ``n_bootstrap`` for it.

    Use :func:`analyze` for ordinary analysis. Nothing here checks the sample,
    so a non-finite input silently leaves every conditioning class and a
    malformed categorical code silently becomes an excluded sample rather than
    an error.

    Reverse-mode differentiation traces, but be clear about what it gives
    back. A PAWN index reads the *ordering* of the outputs and the bin an
    input falls in, and both are piecewise constant, so the derivative is
    exactly 0 almost everywhere. The value of tracing here is ``jit`` and
    ``vmap``, not a gradient.

    One limit is not ours to lift: a **truncated Gaussian** marginal has no
    CDF in JAX, so :func:`jaxgsa._core.transforms.cdf_to_unit_interval` routes
    that column through SciPy, which reads ``X`` on the host. A problem with
    such a marginal therefore cannot be traced through this function. Every
    other marginal — uniform, unbounded Gaussian, categorical — stays on
    device.

    Tier T4 (behavioural contract): the returned array must equal the ``pawn``
    field of ``analyze``'s result on clean outputs, and the function must
    survive ``jit``, ``vmap`` and ``jit(jacrev(...))``. Checked in
    ``tests/test_pawn.py``.

    Args:
        problem: Problem definition with D parameters.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        n_bins: Number of conditioning bins per continuous parameter, as in
            :func:`analyze`.
        statistic: Aggregation of the KS values across bins, as in
            :func:`analyze`. All three are plain nan-aware reductions, so the
            choice does not affect what ``jit``, ``vmap`` or ``jacrev`` can do
            with this function.
        slice_chunk_size: Number of flattened ``T*K`` output columns per
            kernel call. ``None`` (default) derives one from the active memory
            budget. It changes no index.

    Returns:
        A one-element tuple ``(pawn,)``, shaped ``(D,)``, ``(K, D)`` or
        ``(T, K, D)`` to mirror the layout of ``Y``. A tuple, not a bare
        array, because every method's core returns one.

    Raises:
        ValueError: If ``X`` is not 2-D, if its column count does not match
            the problem, if ``X`` and ``Y`` have differing row counts, if
            ``Y`` has an invalid rank, if ``statistic`` is not one of
            ``"median"``/``"max"``/``"mean"``, if ``n_bins < 2``, or if
            ``slice_chunk_size`` is given and is below 1.
    """
    X_arr, Y_arr = jnp.asarray(X), jnp.asarray(Y)
    rank_ok = Y_arr.ndim in (1, 2, 3)
    check_scalars(
        (
            require(
                statistic in ("median", "max", "mean"),
                f"statistic must be 'median', 'max', or 'mean', got {statistic!r}",
            ),
            at_least("n_bins", n_bins, 2),
            at_least("slice_chunk_size", slice_chunk_size, 1),
            require(X_arr.ndim == 2, f"X must be 2-D (N, D), got shape {X_arr.shape}"),
            require(
                X_arr.ndim != 2 or X_arr.shape[1] == problem.num_vars,
                f"X has {X_arr.shape[-1] if X_arr.ndim else 0} columns but the problem "
                f"declares {problem.num_vars} parameters",
            ),
            require(rank_ok, f"Y must have 1, 2 or 3 dimensions, got {Y_arr.ndim}"),
            require(
                not rank_ok or X_arr.ndim != 2 or Y_arr.shape[0] == X_arr.shape[0],
                f"X has {X_arr.shape[0] if X_arr.ndim else 0} rows but Y has "
                f"{Y_arr.shape[0] if Y_arr.ndim else 0}",
            ),
        )
    )

    Y3, layout = _prepare_Y(Y_arr)
    bin_idx, n_eff = _bin_indices(problem, X_arr, n_bins)
    pawn_3d = _pawn_core(bin_idx, Y3, n_eff, statistic, slice_chunk_size, warn=False)
    return (layout.squeeze(pawn_3d),)


def analyze(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_bins: int = 10,
    statistic: Literal["median", "max", "mean"] = "median",
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    slice_chunk_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    keep_replicates: bool = False,
) -> PAWNResult:
    """Compute PAWN sensitivity indices.

    PAWN measures how much fixing a parameter changes the whole output
    distribution, not just its variance. For each parameter, the samples are
    split into ``n_bins`` conditioning bins. The bins are equal-width on the
    CDF-transformed unit interval, so they are equal-probability under the
    parameter's marginal. Each bin's conditional output CDF is then compared
    with the unconditional CDF by the Kolmogorov-Smirnov (KS) statistic, the
    largest vertical gap between the two CDFs and a number in [0, 1].

    The per-bin KS values are aggregated (median by default) into one index
    per parameter. A value of 0 means the parameter has no effect on the
    output distribution, and larger values mean stronger influence.

    PAWN is a given-data method, so any (X, Y) sample works with no special
    design. It is a good choice when the output is skewed or multimodal, so
    that variance-based indices summarize its uncertainty poorly.

    Use :func:`indices` when you need the same numbers inside ``jit``,
    ``vmap`` or ``jacrev``.

    Correlated parameters are supported. PAWN conditions on bins of one
    parameter and compares output CDFs, so a declared ``problem.correlation``
    does not invalidate the indices. Each index then measures the parameter's
    total influence, which includes influence carried through its correlated
    partners. A parameter that the model ignores can therefore score above 0
    when it correlates with an influential parameter. That reading is
    correct, not an estimation error.

    Args:
        problem: Problem definition with D parameters.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        n_bins: Number of conditioning bins per continuous parameter, which
            are equal-probability under that parameter's marginal. More bins
            condition each parameter more tightly but leave fewer samples per
            bin, roughly ``N / n_bins``, which makes each KS value noisier.
            The default of 10 suits N in the thousands. A categorical
            parameter ignores this and uses one bin per level.
        statistic: Aggregation of KS values across bins. ``"median"``
            (default) is robust to a few noisy bins. ``"max"`` is the
            conservative choice for screening out non-influential
            parameters, because a parameter is negligible only if no bin
            shifts the output. ``"mean"`` weights all bins equally.
        n_bootstrap: Number of bootstrap resamples for confidence intervals.
            ``0`` disables the confidence intervals.
        conf_level: Confidence level for the bootstrap intervals.
        ci_method: How the interval endpoints are formed. ``"quantile"``
            (default) reads them off the empirical bootstrap distribution.
            ``"gaussian"`` centres them on the point estimate and takes
            ``+/- z * sd`` of the bootstrap draws, which is smoother for a
            small ``n_bootstrap`` but assumes the draws are normal.
        key: A ``jax.random`` key for the bootstrap resampling. Required
            when ``n_bootstrap > 0``. Pass ``jax.random.key(0)`` if you have
            an integer seed.
        slice_chunk_size: Number of flattened ``T*K`` output columns
            processed per kernel call. ``None`` (default) derives one from
            the active memory budget, which
            :func:`jaxgsa.config.set_memory_budget` sets. Pass a positive
            integer to override it. Peak memory is dominated by the kernel's
            ECDF tables, about ``2 * slice_chunk_size * D * N * n_bins``
            elements: the inner ``vmap`` holds a full ``(N, n_bins)`` table
            per (column, parameter) pair. The ``slice_chunk_size * D *
            n_bins`` KS result is smaller by a factor of ``N`` and does not
            drive the peak. Lower it if a time-series output runs the device
            out of memory. It changes no index: every output column is
            independent of every other.
        on_invalid: What to do about a row of ``X`` or ``Y`` that holds a
            non-finite value. ``"raise"`` (default) refuses the sample,
            ``"drop"`` removes those rows and analyzes the rest, and
            ``"propagate"`` warns and computes anyway. ``X`` and ``Y`` are
            checked together, so a bad input takes its own output with it.
            See :mod:`jaxgsa._core.invalid`.
        keep_replicates: Keep the per-resample indices on
            ``PAWNResult.ci.replicates``. Off by default because they are
            large: ``n_bootstrap`` copies of the index array. Turn it on to
            recompute an interval at another level without re-running the
            analysis.

    Returns:
        A :class:`PAWNResult` with ``pawn`` shaped ``(D,)``, ``(K, D)``, or
        ``(T, K, D)``, optional confidence intervals in ``pawn_conf``, how
        those intervals were made in ``ci``, and the non-finite report in
        ``invalid``.

    Raises:
        ValueError: If X is not 2-D, its column count does not match the
            problem, X and Y have differing row counts, ``statistic`` is
            not one of ``"median"``/``"max"``/``"mean"``, ``n_bins < 2``,
            ``conf_level`` is not in ``(0, 1)``, ``slice_chunk_size`` is
            given and is not a positive integer, ``on_invalid`` is not one
            of the three policies, ``n_bootstrap < 0``, ``ci_method`` is not
            ``"quantile"`` or ``"gaussian"``, ``n_bootstrap > 0`` and no
            ``key`` was given, or the non-finite policy refuses the sample.
    Warns:
        JaxgsaWarning: If an output slice has zero variance. Every
            conditional distribution then equals the unconditional one, so
            the corresponding indices are an exact 0 rather than an answer.
    """
    from jaxgsa.pawn import SPEC

    # The non-finite check runs before any transform, and before binning above
    # all: a non-finite input would otherwise take the out-of-range sentinel
    # and leave every conditional set without a word.
    ctx = prepare(
        SPEC,
        problem,
        Y,
        X=X,
        on_invalid=on_invalid,
        checks=(
            require(
                statistic in ("median", "max", "mean"),
                f"statistic must be 'median', 'max', or 'mean', got {statistic!r}",
            ),
            at_least("n_bins", n_bins, 2),
            at_least("n_bootstrap", n_bootstrap, 0),
            one_of("ci_method", ci_method, ("quantile", "gaussian")),
            in_open_interval("conf_level", conf_level, 0.0, 1.0),
            at_least("slice_chunk_size", slice_chunk_size, 1),
        ),
        min_kept=_MIN_KEPT,
        # A constant slice leaves every conditional distribution equal to
        # the unconditional one, so the indices come out an exact 0, not
        # the NaN a variance ratio would give.
        zero_variance_outcome="zero",
    )
    if n_bootstrap > 0 and key is None:
        raise ValueError("key is required when n_bootstrap > 0")

    X, invalid = ctx.inputs, ctx.invalid
    Y_3d = ctx.Y3
    # Refusing a malformed categorical column needs concrete values, so it
    # runs here and not in the traceable binning it guards.
    _check_categorical_codes(problem, X)
    bin_idx, n_eff = _bin_indices(problem, X, n_bins)

    pawn_3d = _pawn_core(bin_idx, Y_3d, n_eff, statistic, slice_chunk_size)

    pawn_conf: Array | None = None
    ci: CIInfo | None = None
    if n_bootstrap > 0:
        assert key is not None  # checked above, before the point estimate ran
        boot_key = key
        N = X.shape[0]
        boot_draws = []
        for _ in range(n_bootstrap):
            boot_key, subkey = jax.random.split(boot_key)
            idx = jax.random.choice(subkey, N, shape=(N,), replace=True)
            boot_pawn = _pawn_core(
                bin_idx[idx], Y_3d[idx], n_eff, statistic, slice_chunk_size, warn=False
            )
            boot_draws.append(boot_pawn)

        boot_stack = jnp.stack(boot_draws, axis=0)
        # Stack [lower, upper] into a leading axis of size 2.
        pawn_conf_3d = jnp.stack(
            _bootstrap_ci_endpoints(
                pawn_3d, boot_stack, conf_level=conf_level, ci_method=ci_method
            )
        )

        pawn_conf = ctx.squeeze(pawn_conf_3d)

        # The leading resample axis survives the squeeze, which addresses the
        # T/K axes from the end.
        ci = CIInfo(
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            replicates={"pawn": ctx.squeeze(boot_stack)} if keep_replicates else None,
        )

    pawn_out = ctx.squeeze(pawn_3d)

    return PAWNResult(
        pawn=pawn_out,
        pawn_conf=pawn_conf,
        problem=problem,
        invalid=invalid,
        ci=ci,
    )
