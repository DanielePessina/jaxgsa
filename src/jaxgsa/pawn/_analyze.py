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
per unique ``(N, D, n_classes)`` and output-column count. The columns run in
chunks of at most ``slice_chunk_size``, which bounds the working arrays, so
the column count is the chunk width. A trailing part-chunk is padded back to
that width and the answer sliced back, which keeps the kernel at one
compilation instead of two; the padded columns are separate ``vmap`` lanes, so
a padded column cannot change a real column's answer. For the point
estimate, the ``statistic`` aggregation runs outside JIT so all three
statistics share one compilation. The bootstrap has its own kernel: it draws
each resample, computes the KS statistics and aggregates them inside one JIT,
compiled once per statistic, and runs the resamples one after another.

The kernel's working set is not the ``(chunk, D, n_classes)`` result. The
nested ``vmap`` builds a full ``(N, n_classes)`` ECDF table per (column,
parameter) pair, so the live arrays are ``(chunk, D, N, n_classes)``. That is
``N`` times larger than the result, which is why the default chunk width is
derived from ``N``, ``D`` and ``n_classes`` rather than fixed.

A categorical parameter needs no binning: its level code is already a bin
index. Continuous and categorical columns share one kernel, compiled at
``n_eff = max(n_classes, max_levels)``. The shorter columns leave their
trailing bins empty, the kernel returns ``NaN`` for an empty bin, and the
nan-aware aggregation drops it. A continuous-only problem has
``n_eff == n_classes`` and is bit-for-bit unaffected.

References:
    Pianosi & Wagener (2015). A simple and efficient method for global
    sensitivity analysis based on cumulative distribution functions.
    Environmental Modelling & Software 67:1-11. (Introduces the PAWN
    index and its KS statistic.)
    Pianosi & Wagener (2018). Distribution-based sensitivity analysis from
    a generic input-output sample. Environmental Modelling & Software
    108:197-207. (The estimator implemented here: bins are built on a
    generic (X, Y) sample rather than a dedicated conditional design, and
    the two-sample KS test runs against the whole-sample unconditional
    CDF.)
"""

from __future__ import annotations

import dataclasses
import warnings
from functools import partial
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.batching import resolve_batch_size
from jaxgsa._core.bootstrap import interval
from jaxgsa._core.column_chunks import map_column_chunks
from jaxgsa._core.entry import (
    at_least,
    bootstrap_checks,
    check_scalars,
    positive_int,
    prepare,
    require,
    validate_inputs,
)
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.irregular import (
    IrregularResult,
    IrregularY,
    _fwd_kwargs,
    _is_irregular_y,
    analyze_irregular,
)
from jaxgsa._core.partition import _extract_categorical_codes
from jaxgsa._core.result import CIInfo
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa._core.validation import _prepare_Y, _validate_output, as_key
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.pawn._result import PAWNResult
from jaxgsa.problem import Problem, _categorical_dims

# Keyword-only parameters the irregular engine forwards unchanged to the
# per-channel recursive analyze() calls. verbose is excluded:
# the engine owns it.
_IRREGULAR_KWARGS = (
    "n_classes",
    "statistic",
    "n_bootstrap",
    "conf_level",
    "ci_method",
    "key",
    "slice_chunk_size",
    "on_invalid",
    "keep_replicates",
)

# Fewest samples that still define both an unconditional and a conditional
# ECDF, so that a KS distance is a distance and not an artefact.
_MIN_KEPT = 4

# Warn when fewer than this fraction of a parameter's conditioning bins
# contribute a KS value. The nan-aware aggregation silently drops a bin with
# fewer than two samples, so a heavily depleted parameter reports an index
# that rests on a handful of bins without saying so.
_MIN_VALID_BIN_FRACTION = 0.5


def _count_valid_bins(bin_idx: Array, n_eff: int) -> np.ndarray:
    """Count, per parameter, the conditioning bins that contribute a KS value.

    A bin contributes only when it holds at least two samples; the KS kernel
    returns ``NaN`` for anything smaller and the nan-aware aggregation drops
    it. Bin occupancy depends on the inputs alone (never on ``Y``), so the
    count is computed once per analysis, on the host. The count stays exact
    even under ``on_invalid="propagate"`` with a NaN in ``Y``: the KS kernel
    reads ranks, not values, and ``jnp.argsort`` places a NaN row at the end
    of the sort deterministically, so it still lands in a well-defined group
    and the KS statistic stays finite. A propagated NaN in ``Y`` does not
    make the index NaN.

    Args:
        bin_idx: Conditioning-bin indices from :func:`_bin_indices`, shape
            ``(N, D)``, where ``-1`` marks an excluded sample.
        n_eff: Number of bins the kernel is compiled at.

    Returns:
        Contributing-bin counts per parameter, shape ``(D,)``, int64.
    """
    idx = np.asarray(bin_idx)
    counts = np.stack(
        [np.bincount(col[col >= 0].astype(np.int64), minlength=n_eff) for col in idx.T]
    )
    return (counts >= 2).sum(axis=1)


def _warn_sparse_bins(problem: Problem, n_valid: np.ndarray, n_classes: int) -> None:
    """Warn once when parameters keep fewer than half their bins contributing.

    Args:
        problem: Problem definition, for the parameter names and the level
            counts of categorical parameters.
        n_valid: Contributing-bin counts from :func:`_count_valid_bins`,
            shape ``(D,)``.
        n_classes: Requested bin count for continuous parameters. A categorical
            parameter is measured against its own level count instead.

    Warns:
        JaxgsaWarning: Naming every parameter whose contributing-bin fraction
            is below ``_MIN_VALID_BIN_FRACTION`` (all-empty parameters are
            excluded; they already get their own warning).
    """
    levels_of_dim = dict(_categorical_dims(problem))
    sparse = []
    for d, count in enumerate(n_valid):
        expected = levels_of_dim.get(d, n_classes)
        if 0 < count < _MIN_VALID_BIN_FRACTION * expected:
            sparse.append(f"{problem.names[d]!r} ({int(count)}/{expected})")
    if sparse:
        warnings.warn(
            f"jaxgsa.pawn: parameters {', '.join(sparse)} have fewer than half of "
            "their conditioning bins contributing (a bin needs at least 2 "
            "samples to define a conditional CDF; the rest are dropped). The "
            "reported indices rest on those few bins. Use fewer bins "
            "(lower n_classes) or more samples.",
            stacklevel=3,
            category=JaxgsaWarning,
        )


def _equal_width_bins(X_u01: Array, n_classes: int) -> Array:
    """Assign each sample to an equal-width bin in ``[0, 1]``.

    A sample outside ``[0, 1]`` gets the sentinel ``-1``, so it matches no bin
    and drops out of every conditional set. This happens when the caller
    passes inputs beyond the declared bounds of a ``uniform`` parameter. A
    value of exactly ``0.0`` maps to the first bin and ``1.0`` maps to the
    last bin. Bin ``k`` is ``[k / n_classes, (k + 1) / n_classes)``, so a value
    of exactly ``k / n_classes`` maps to bin ``k``. The edges are the
    correctly rounded quotients ``k / n_classes``, not ``linspace(0, 1, ...)``:
    ``linspace`` can put an interior edge one ulp high, and a sample that sits
    exactly on that edge (a lattice design, for example) then falls into the
    bin below.

    A non-finite input takes the same sentinel. Under the two policies that
    remove or refuse it, ``"raise"`` and ``"drop"``, :func:`analyze` settles
    it before this point, so a failed model run is reported instead of
    quietly leaving every conditional set. Under ``"propagate"`` the value
    does reach here and is silently binned out, which is the behaviour that
    policy asks for -- the caller was told the count and the row numbers
    first.

    Args:
        X_u01: Unit-interval inputs, shape ``(N, D)``.
        n_classes: Number of equal-width bins.

    Returns:
        Integer bin indices, shape ``(N, D)``, in ``[0, n_classes)``, or ``-1``
        for out-of-range or ``NaN`` samples.
    """
    bin_edges = jnp.asarray(np.arange(n_classes + 1) / n_classes, dtype=X_u01.dtype)
    idx = jnp.clip(jnp.searchsorted(bin_edges, X_u01, side="right") - 1, 0, n_classes - 1)
    in_range = (X_u01 >= 0.0) & (X_u01 <= 1.0)
    return jnp.where(in_range, idx, -1)


def _bin_indices(problem: Problem, X: Array, n_classes: int) -> tuple[Array, int]:
    """Assign every parameter column to conditioning bins, categorical or not.

    A continuous column keeps equal-width binning on the CDF-transformed unit
    interval, which is equal-probability under the column's marginal. A
    categorical column needs no binning at all. Its level code ``0..L-1``
    already names the conditioning class, so the code serves as the bin index.

    The two kinds share one kernel, so the kernel is compiled at
    ``n_eff = max(n_classes, max_levels)`` and the shorter columns leave their
    trailing bins empty. An empty bin yields ``NaN`` from the KS kernel and
    the nan-aware aggregation drops it, so the padding changes no index. A
    continuous-only problem has ``n_eff == n_classes`` and is unaffected.

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
        n_classes: Number of conditioning bins for the continuous columns.

    Returns:
        ``(bin_idx, n_eff)``. ``bin_idx`` holds integer bin indices of shape
        ``(N, D)``, where ``-1`` marks an excluded sample. ``n_eff`` is the
        padded bin count the kernel must be compiled at.
    """
    dims_levels = _categorical_dims(problem)
    if not dims_levels:
        return _equal_width_bins(cdf_to_unit_interval(X, problem), n_classes), n_classes

    n_eff = max(n_classes, max(n_levels for _, n_levels in dims_levels))
    levels_of_dim = dict(dims_levels)

    # cdf_to_unit_interval refuses a categorical parameter by design, so the
    # continuous columns are transformed through a continuous-only problem.
    cont_dims = [d for d in range(problem.n_params) if d not in levels_of_dim]
    columns: dict[int, Array] = {}
    if cont_dims:
        cont_problem = Problem._from_normalized_inputs(
            names=tuple(problem.names[d] for d in cont_dims),
            input_specs=tuple(problem.input_specs[d] for d in cont_dims),
        )
        cont_idx = _equal_width_bins(
            cdf_to_unit_interval(X[:, jnp.asarray(cont_dims)], cont_problem), n_classes
        )
        for j, d in enumerate(cont_dims):
            columns[d] = cont_idx[:, j].astype(jnp.int32)
    for d, n_levels in dims_levels:
        columns[d] = _level_codes(X[:, d], n_levels)
    return jnp.stack([columns[d] for d in range(problem.n_params)], axis=1), n_eff


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


# Rows per block in the blocked cumulative sum. A power of two keeps the
# triangular matmul small; the value changes speed, never the result.
_CUMSUM_BLOCK = 64
# Largest N with N*N < 2**32, for the uint32 KS numerator without x64.
_MAX_UINT32_N = 65535


def _count_cumsum(oh: Array) -> Array:
    """Return the cumulative sum of ``oh`` along axis 0, as exact int32 counts.

    ``oh`` holds 0/1 flags. Block sums are at most ``_CUMSUM_BLOCK``, so the
    float matmul inside a block is exact. They are converted to int32 and
    the offsets between blocks are summed in int32, so the counts stay exact
    up to ``N = 2**31`` (float32 would round above ``2**24``). The result
    equals ``jnp.cumsum(oh, axis=0)`` as integers. A lower-triangular matmul
    inside blocks of rows plus a cumulative sum of the block totals runs
    faster on CPU than the XLA reduce-window lowering of ``cumsum``.

    Args:
        oh: 0/1 flags, shape ``(N, ...)``, in a float dtype.

    Returns:
        Cumulative counts, same shape as ``oh``, an integer dtype
        (int32, or int64 when x64 promotes the block-total sum).
    """
    n = oh.shape[0]
    b = min(_CUMSUM_BLOCK, n)
    n_blocks = -(-n // b)
    flat = oh.reshape(n, -1)
    if n_blocks * b != n:
        # Trailing zero rows add nothing to any partial sum; they are cut off below.
        flat = jnp.concatenate(
            [flat, jnp.zeros((n_blocks * b - n, flat.shape[1]), dtype=flat.dtype)], axis=0
        )
    blocks = flat.reshape(n_blocks, b, flat.shape[1])
    tri = jnp.tril(jnp.ones((b, b), dtype=flat.dtype))
    within = jnp.einsum("ij,njk->nik", tri, blocks, precision=jax.lax.Precision.HIGHEST)
    within = within.astype(jnp.int32)
    totals = within[:, -1, :]
    offsets = jnp.cumsum(totals, axis=0) - totals
    out = (within + offsets[:, None, :]).reshape(n_blocks * b, -1)[:n]
    return out.reshape(oh.shape)


def _ks_from_sorted(y_sorted: Array, bin_idx_sorted: Array, n_classes: int) -> Array:
    """Compute KS statistics for one output column already sorted by value.

    A sorted position is a "group end" if it is the last member of its tie
    group. Evaluating the ECDFs only there yields the value-based (tie-aware)
    two-sample KS statistic. The statistic reads only the cumulative bin
    counts at group ends, which include the whole tie group regardless of how
    its members are ordered, so any valid sorted permutation gives the same
    value.

    One vectorized ``(D, n_classes)`` ECDF table replaces ``D`` separate
    per-parameter tables: the one-hot, cumulative sum, normalization and
    max-diff are batched over the parameter axis, which is bit-for-bit
    identical because the per-parameter columns never interact.

    Args:
        y_sorted: Output values in ascending order, shape ``(N,)``.
        bin_idx_sorted: Bin indices in the same row order, shape ``(N, D)``;
            ``-1`` marks an excluded sample.
        n_classes: Number of bins (static).

    Returns:
        KS statistics, shape ``(D, n_classes)``, ``NaN`` where a bin has
        fewer than two samples.
    """
    N = y_sorted.shape[0]
    dtype = jnp.result_type(y_sorted.dtype, jnp.float32)
    param_bins = jnp.arange(n_classes)
    is_group_end = jnp.concatenate([y_sorted[:-1] != y_sorted[1:], jnp.array([True])])
    # 0/1 flags with block sums of at most _CUMSUM_BLOCK, so float32 is exact
    # here at any working dtype, and runs faster than float64 under x64.
    oh = (bin_idx_sorted[:, :, None] == param_bins[None, None, :]).astype(jnp.float32)
    cnt = oh.sum(axis=0, dtype=jnp.int32)
    cum = _count_cumsum(oh)
    pos = jnp.arange(1, N + 1, dtype=jnp.int32)[:, None, None]
    cnt_b = cnt[None, :, :]
    if jax.config.jax_enable_x64:
        int_t = jnp.int64
    elif N <= _MAX_UINT32_N:
        int_t = jnp.uint32
    else:
        int_t = None
    if int_t is not None:
        # |pos/N - cum/cnt| = |pos*cnt - cum*N| / (N*cnt). The numerator is an
        # exact integer, so the gap is exactly 0 when the two ECDFs agree and
        # there is one rounding step. N*N must fit the integer type.
        lhs = pos.astype(int_t) * cnt_b.astype(int_t)
        rhs = cum.astype(int_t) * jnp.asarray(N, dtype=int_t)
        num = jnp.maximum(lhs, rhs) - jnp.minimum(lhs, rhs)
        den = jnp.maximum(cnt_b, 1).astype(int_t) * jnp.asarray(N, dtype=int_t)
        diff = num.astype(dtype) / den.astype(dtype)
    else:
        # N > 65535 without x64: true division. The barrier stops XLA from
        # rewriting x/const as x*(1/const), which is off by 1 ulp.
        d_n = jax.lax.optimization_barrier(jnp.full(cum.shape, N, dtype=dtype))
        d_c = jax.lax.optimization_barrier(
            jnp.broadcast_to(jnp.maximum(cnt_b, 1).astype(dtype), cum.shape)
        )
        diff = jnp.abs(pos.astype(dtype) / d_n - cum.astype(dtype) / d_c)
    diff = jnp.where(is_group_end[:, None, None], diff, -jnp.inf)
    ks = jnp.max(diff, axis=0)
    return jnp.where(cnt >= 2, ks, jnp.nan)


def _ks_columns(bin_idx: Array, Y_cols: Array, n_classes: int) -> Array:
    """Compute per-bin KS statistics for every output column.

    Args:
        bin_idx: Integer bin indices, shape ``(N, D)``, where ``-1`` marks an
            excluded sample.
        Y_cols: Output columns, shape ``(N, M)``.
        n_classes: Number of bins (static).

    Returns:
        KS statistics, shape ``(M, D, n_classes)``, with ``NaN`` where a bin
        has fewer than two samples.
    """

    def _ks_one_col(y: Array) -> Array:
        """Compute KS statistics of shape ``(D, n_classes)`` for one output column."""
        # The sort is deliberately unstable: any valid sorted permutation
        # gives the same statistic (see ``_ks_from_sorted``). Measured
        # ~1.4-1.8x faster than the stable argsort on the CPU backend.
        order = jnp.argsort(y, stable=False)
        return _ks_from_sorted(y[order], bin_idx[order], n_classes)

    return jax.vmap(_ks_one_col, in_axes=1)(Y_cols)


@partial(jax.jit, static_argnames=("n_classes", "statistic", "chunk"))
def _pawn_columns(
    bin_idx: Array,
    Y_3d: Array,
    n_classes: int,
    statistic: Literal["median", "max", "mean"],
    chunk: int,
) -> Array:
    """PAWN indices of every output column, one chunk of columns at a time.

    The chunks run under :func:`jax.lax.map` inside this one program (see
    :func:`jaxgsa._core.column_chunks.map_column_chunks`), so the transient
    memory is one chunk wide in :func:`analyze` and under an outer ``jit``
    alike.

    Args:
        bin_idx: Integer bin indices, shape ``(N, D)``.
        Y_3d: Output array, shape ``(N, T, K)``.
        n_classes: Number of bins (static).
        statistic: Aggregation across bins (static).
        chunk: Output columns per step (static).

    Returns:
        PAWN indices, shape ``(T, K, D)``.
    """
    N, T, K = Y_3d.shape
    pawn = map_column_chunks(
        lambda cols: _aggregate_ks(_ks_columns(bin_idx, cols, n_classes), statistic),
        Y_3d.reshape(N, T * K),
        chunk,
        axis=0,
    )  # (T*K, D)
    return pawn.reshape(T, K, bin_idx.shape[1])


@partial(jax.jit, static_argnames=("n_bootstrap",))
def _bootstrap_subkeys(key: Array, n_bootstrap: int) -> Array:
    """Return the per-replicate keys of the sequential ``split`` chain.

    Replicate ``i`` draws its rows from the second key of the ``i``-th
    ``jax.random.split`` applied to the running key. A ``lax.scan`` builds
    this chain. It is the same chain that
    :func:`jaxgsa._core.bootstrap.bootstrap_draws` uses, but pawn does not
    call that function.

    Args:
        key: PRNG key.
        n_bootstrap: Number of replicates.

    Returns:
        Stacked keys, leading axis of length ``n_bootstrap``.
    """

    # Jitted at module level: a lax.scan outside jit traces and compiles
    # again on every call, because its body is a new function each time.
    def step(k, _):
        k, sub = jax.random.split(k)
        return k, sub

    _, subkeys = jax.lax.scan(step, key, None, length=n_bootstrap)
    return subkeys


def _boot_columns(
    bin_idx: Array,
    Y_cols: Array,
    subkeys: Array,
    n_classes: int,
    statistic: Literal["median", "max", "mean"],
) -> Array:
    """PAWN indices of a chunk of output columns on every bootstrap resample.

    The replicates run one after another under ``lax.map``, so the working
    set is that of a single replicate.

    Each column is sorted once on the full sample. A replicate then sorts the
    integer ranks of its drawn rows instead of re-sorting floats, and reads
    the values back from the full sorted column. That gives a valid sorted
    order of the resample, so the statistic is the same.

    Args:
        bin_idx: Integer bin indices, shape ``(N, D)``.
        Y_cols: Output columns, shape ``(N, M)``.
        subkeys: One PRNG key per replicate, shape ``(R,)``.
        n_classes: Number of bins (static).
        statistic: Aggregation across bins (static).

    Returns:
        PAWN indices, shape ``(R, M, D)``.
    """
    N, M = Y_cols.shape
    order = jnp.argsort(Y_cols, axis=0, stable=False)
    y_full = jnp.take_along_axis(Y_cols, order, axis=0)
    rank = (
        jnp.zeros((N, M), dtype=jnp.int32)
        .at[order, jnp.arange(M)[None, :]]
        .set(jnp.arange(N, dtype=jnp.int32)[:, None])
    )

    def _one_replicate(subkey: Array) -> Array:
        rows = jax.random.choice(subkey, N, shape=(N,), replace=True)
        bins_rep = bin_idx[rows]

        def _ks_one_col(rank_c: Array, y_c: Array) -> Array:
            r = rank_c[rows]
            o = jnp.argsort(r, stable=False)
            return _ks_from_sorted(y_c[r[o]], bins_rep[o], n_classes)

        ks = jax.vmap(_ks_one_col, in_axes=(1, 1))(rank, y_full)  # (M, D, C)
        return _aggregate_ks(ks, statistic)

    return jax.lax.map(_one_replicate, subkeys)


@partial(jax.jit, static_argnames=("n_classes", "statistic", "chunk"))
def _pawn_boot_columns(
    bin_idx: Array,
    Y_3d: Array,
    subkeys: Array,
    n_classes: int,
    statistic: Literal["median", "max", "mean"],
    chunk: int,
) -> Array:
    """Bootstrap PAWN indices of every output column, one chunk at a time.

    Args:
        bin_idx: Integer bin indices, shape ``(N, D)``.
        Y_3d: Output array, shape ``(N, T, K)``.
        subkeys: One PRNG key per replicate, shape ``(R,)``.
        n_classes: Number of bins (static).
        statistic: Aggregation across bins (static).
        chunk: Output columns per step (static).

    Returns:
        PAWN indices per replicate, shape ``(R, T, K, D)``.
    """
    N, T, K = Y_3d.shape
    boot = map_column_chunks(
        lambda cols: _boot_columns(bin_idx, cols, subkeys, n_classes, statistic),
        Y_3d.reshape(N, T * K),
        chunk,
        axis=1,
    )  # (R, T*K, D)
    return boot.reshape(subkeys.shape[0], T, K, bin_idx.shape[1])


def _aggregate_ks(
    ks: Array,
    statistic: Literal["median", "max", "mean"],
) -> Array:
    """Reduce KS values over the trailing ``n_classes`` axis to PAWN indices.

    All three statistics are nan-aware. A bin with fewer than two samples is
    dropped rather than propagated, whether it is an empty padding bin or a
    genuinely under-filled one. A parameter whose bins are all empty
    therefore yields ``NaN``, which is the honest answer.

    Args:
        ks: KS statistics with bins on the last axis, shape ``(..., n_classes)``.
            The function ignores ``NaN`` entries, which mark bins with fewer
            than two samples.
        statistic: Aggregation across bins.

    Returns:
        PAWN indices with the ``n_classes`` axis removed.
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
            per ``lax.map`` step, or ``None`` to derive one from the memory
            budget. An explicit value must be at least 1.

    Returns:
        PAWN indices, shape ``(T, K, D)``.
    """
    N, T, K = Y_3d.shape
    D = bin_idx.shape[1]

    total = T * K
    # The kernel promotes its working dtype to at least float32.
    itemsize = jnp.result_type(Y_3d.dtype, jnp.float32).itemsize
    # The kernel's real working set is `chunk * D * N * n_eff` elements, not
    # the `chunk * D * n_eff` result; two such arrays are live at once inside
    # the kernel, so the per-column byte estimate below carries a factor of
    # two.
    cs = resolve_batch_size(2 * D * N * n_eff * itemsize, total, slice_chunk_size)
    # Bin assignment depends only on the inputs, so it is built once by the
    # caller and every output column runs through the same vmapped kernel.
    # The KS values of one output column depend on no other column, so the
    # aggregation runs per chunk and only the (chunk, D) result is kept.
    return _pawn_columns(bin_idx, Y_3d, n_eff, statistic, cs)


def _pawn_bootstrap(
    bin_idx: Array,
    Y_3d: Array,
    n_eff: int,
    statistic: Literal["median", "max", "mean"],
    slice_chunk_size: int | None,
    key: Array,
    n_bootstrap: int,
) -> Array:
    """Compute PAWN indices on ``n_bootstrap`` row-resamples.

    :func:`_bootstrap_subkeys` makes one key per replicate from ``key``.
    Inside the jitted kernel, ``lax.map`` runs the replicates one after
    another. Each replicate draws ``N`` row indices with replacement by
    ``jax.random.choice``. The same row indices apply to every output column
    of the chunk. One jitted call walks the chunks of output columns under
    ``lax.map`` and runs all replicates per chunk. Chunking and the memory
    estimate match
    :func:`_pawn_core`, because a replicate has the same working set as one
    point-estimate call.

    Args:
        bin_idx: Conditioning-bin indices, shape ``(N, D)``.
        Y_3d: Output array, shape ``(N, T, K)``.
        n_eff: Number of bins the kernel is compiled at.
        statistic: Aggregation method across bins.
        slice_chunk_size: Maximum number of flattened output columns per
            ``lax.map`` step, or ``None`` to derive one from the memory budget.
        key: PRNG key for the row draws.
        n_bootstrap: Number of replicates, at least 1.

    Returns:
        PAWN indices per replicate, shape ``(n_bootstrap, T, K, D)``.
    """
    N, T, K = Y_3d.shape
    D = bin_idx.shape[1]
    total = T * K
    itemsize = jnp.result_type(Y_3d.dtype, jnp.float32).itemsize
    cs = resolve_batch_size(2 * D * N * n_eff * itemsize, total, slice_chunk_size)
    subkeys = _bootstrap_subkeys(key, n_bootstrap)
    return _pawn_boot_columns(bin_idx, Y_3d, subkeys, n_eff, statistic, cs)


def _warn_all_empty(n_valid: np.ndarray) -> None:
    """Warn for each parameter none of whose bins holds two samples.

    Whether every bin of a parameter is empty depends only on the binning,
    never on ``Y``, so it is the same as a zero contributing-bin count.

    Args:
        n_valid: Contributing-bin counts from :func:`_count_valid_bins`,
            shape ``(D,)``.

    Warns:
        JaxgsaWarning: Once per affected parameter.
    """
    for d, count in enumerate(n_valid.tolist()):
        if count == 0:
            warnings.warn(
                f"jaxgsa.pawn: all bins empty for parameter {d}, returning NaN",
                stacklevel=3,
                category=JaxgsaWarning,
            )


class PAWNIndices(NamedTuple):
    """Plain-array PAWN indices, as returned by :func:`indices`.

    Fields are named like the matching result fields. Tuple unpacking
    works as for a plain tuple.

    Attributes:
        pawn: PAWN index per parameter.
    """

    pawn: Array


def indices(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_classes: int = 10,
    statistic: Literal["median", "max", "mean"] = "median",
    slice_chunk_size: int | None = None,
) -> PAWNIndices:
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

    Every marginal traces: uniform, Gaussian, truncated Gaussian and
    categorical. The binning of a continuous column goes through
    :func:`jaxgsa._core.transforms.cdf_to_unit_interval`, which stays on
    device for all of them.

    Tier T4 (behavioural contract): the returned array must equal the ``pawn``
    field of ``analyze``'s result on clean outputs, and the function must
    survive ``jit``, ``vmap`` and ``jit(jacrev(...))``. Checked in
    ``tests/test_pawn.py``.

    Args:
        problem: Problem definition with D parameters.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        n_classes: Number of conditioning bins per continuous parameter, as in
            :func:`analyze`.
        statistic: Aggregation of the KS values across bins, as in
            :func:`analyze`. All three are plain nan-aware reductions, so the
            choice does not affect what ``jit``, ``vmap`` or ``jacrev`` can do
            with this function.
        slice_chunk_size: Number of flattened ``T*K`` output columns per
            step. ``None`` (default) derives one from the active memory
            budget. It changes no index.

    Returns:
        A one-field :class:`PAWNIndices` named tuple ``(pawn,)``, shaped ``(D,)``, ``(K, D)`` or
        ``(T, K, D)`` to mirror the layout of ``Y``. A tuple, not a bare
        array, because every method's core returns one.

    Raises:
        ValueError: If ``X`` is not 2-D, if its column count does not match
            the problem, if ``X`` and ``Y`` have differing row counts, if
            ``Y`` has an invalid rank, if ``statistic`` is not one of
            ``"median"``/``"max"``/``"mean"``, if ``n_classes < 2``, or if
            ``slice_chunk_size`` is given and is below 1.
    """
    check_scalars(
        (
            require(
                statistic in ("median", "max", "mean"),
                f"statistic must be 'median', 'max', or 'mean', got {statistic!r}",
            ),
            at_least("n_classes", n_classes, 2),
            positive_int("slice_chunk_size", slice_chunk_size),
        ),
        method="jaxgsa.pawn.indices",
    )
    X_arr = validate_inputs(problem, X)
    Y_arr = _validate_output(Y, int(X_arr.shape[0]), problem)

    Y3, layout = _prepare_Y(Y_arr)
    bin_idx, n_eff = _bin_indices(problem, X_arr, n_classes)
    pawn_3d = _pawn_core(bin_idx, Y3, n_eff, statistic, slice_chunk_size)
    return PAWNIndices(pawn=layout.squeeze(pawn_3d))


def analyze(
    problem: Problem,
    X: Array,
    Y: Array | IrregularY,
    *,
    n_classes: int = 10,
    statistic: Literal["median", "max", "mean"] = "median",
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | int | None = None,
    slice_chunk_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> PAWNResult | IrregularResult:
    """Compute PAWN sensitivity indices.

    PAWN measures how much fixing a parameter changes the whole output
    distribution, not just its variance. For each parameter, the samples are
    split into ``n_classes`` conditioning bins. The bins are equal-width on the
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
        n_classes: Number of conditioning bins per continuous parameter, which
            are equal-probability under that parameter's marginal. More bins
            condition each parameter more tightly but leave fewer samples per
            bin, roughly ``N / n_classes``, which makes each KS value noisier.
            The default of 10 suits N in the thousands. A categorical
            parameter ignores this and uses one bin per level.
        statistic: Aggregation of KS values across bins. ``"median"``
            (default) is robust to a few noisy bins. ``"max"`` is the
            conservative choice for screening out non-influential
            parameters, because a parameter is negligible only if no bin
            shifts the output. ``"mean"`` weights all bins equally.
        n_bootstrap: Number of bootstrap resamples for confidence intervals.
            ``0`` disables the confidence intervals.
        conf_level: Confidence level for the bootstrap intervals. The KS
            statistic is a maximum of absolute gaps, and the maximum is not
            differentiable at a zero gap. Each replicate adds its own noise
            to a gap that is already noise, so the replicates sit above the
            point estimate on average. This is not caused by duplicated
            rows. For ``ci_method="quantile"`` the draws are therefore
            shifted so that their mean equals the point estimate
            (bias-corrected percentile, Efron and Tibshirani 1993, sec.
            10.6) before the quantiles are read. ``ci.replicates`` keeps
            the raw draws, so quantiles you compute from it differ from the
            interval. The interval describes the sampling noise of the
            estimator around its own expected value. It is NOT a confidence
            interval for the population PAWN index: the estimator
            overestimates that index by about ``1 / sqrt(N / n_classes)``,
            which is larger than its noise. A plain bootstrap is
            inconsistent for a maximum at a boundary (Andrews 2000,
            Econometrica 68:399-405; Fang and Santos 2019, Rev. Econ.
            Stud. 86:377-412). Use a larger ``N`` or a dummy-parameter
            reference to separate weak parameters from noise.
        ci_method: How the interval endpoints are formed. ``"quantile"``
            (default) reads them off the empirical bootstrap distribution.
            ``"gaussian"`` centres them on the point estimate and takes
            ``+/- z * sd`` of the bootstrap draws, which is smoother for a
            small ``n_bootstrap`` but assumes the draws are normal.
        key: A ``jax.random`` key for the bootstrap resampling. Required
            when ``n_bootstrap > 0``. Pass ``jax.random.key(0)`` if you have
            an integer seed.
        slice_chunk_size: Number of flattened ``T*K`` output columns
            processed per step. ``None`` (default) derives one from
            the active memory budget, which
            :func:`jaxgsa.config.set_memory_budget` sets. Pass a positive
            integer to override it. Peak memory is dominated by the kernel's
            ECDF tables, about ``2 * slice_chunk_size * D * N * n_classes``
            elements: the inner ``vmap`` holds a full ``(N, n_classes)`` table
            per (column, parameter) pair. The ``slice_chunk_size * D *
            n_classes`` KS result is smaller by a factor of ``N`` and does not
            drive the peak. Lower it if a time-series output runs the device
            out of memory. It changes no index: every output column is
            independent of every other.
        on_invalid: What to do about a row of ``X`` or ``Y`` that holds a
            non-finite value. ``"raise"`` (default) refuses the sample,
            ``"drop"`` removes those rows and analyzes the rest, and
            ``"propagate"`` warns and computes anyway. ``X`` and ``Y`` are
            checked together, so a bad input takes its own output with it.
            See :mod:`jaxgsa._core.invalid`.
        verbose: If ``True`` (default), print a short summary to stdout: the
            problem and the data, the wall-clock timing, and the top
            parameters by ``PAWN``. Pass ``False`` for a silent run.
        keep_replicates: Keep the per-resample indices on
            ``PAWNResult.ci.replicates``. Off by default because they are
            large: ``n_bootstrap`` copies of the index array. Turn it on to
            recompute an interval at another level without re-running the
            analysis.

    Returns:
        A :class:`PAWNResult` with ``pawn`` shaped ``(D,)``, ``(K, D)``, or
        ``(T, K, D)``, optional confidence intervals in ``pawn_conf``, the
        per-parameter contributing-bin diagnostic in ``n_valid_classes``, how
        those intervals were made in ``ci``, and the non-finite report in
        ``invalid``.

    Raises:
        ValueError: If X is not 2-D, its column count does not match the
            problem, X and Y have differing row counts, ``statistic`` is
            not one of ``"median"``/``"max"``/``"mean"``, ``n_classes < 2``,
            ``conf_level`` is not in ``(0, 1)``, ``slice_chunk_size`` is
            given and is not a positive integer, ``on_invalid`` is not one
            of the three policies, ``n_bootstrap < 0``, ``ci_method`` is not
            ``"quantile"`` or ``"gaussian"``, ``n_bootstrap > 0`` and no
            ``key`` was given, or the non-finite policy refuses the sample.
    Warns:
        JaxgsaWarning: In either of two cases. An output slice has zero
            variance: every conditional distribution then equals the
            unconditional one, so the corresponding indices are an exact 0
            rather than an answer. Or a parameter keeps fewer than half of
            its conditioning bins contributing (a bin needs at least two
            samples): the index then rests on the few bins that survive, and
            the fix is fewer bins or more samples.
    """
    key = as_key(key, "pawn")
    if _is_irregular_y(Y):
        return analyze_irregular(
            analyze,
            channel_data=X,
            problem=problem,
            Y=Y,
            n_expected=int(X.shape[0]),
            design_based=False,
            verbose=verbose,
            kwargs=_fwd_kwargs(locals(), _IRREGULAR_KWARGS),
        )
    Y = cast(Array, Y)

    from jaxgsa.pawn import _SPEC

    # The non-finite check runs before any transform, and before binning above
    # all: a non-finite input would otherwise take the out-of-range sentinel
    # and leave every conditional set without a word.
    ctx = prepare(
        _SPEC,
        problem,
        Y,
        X=X,
        on_invalid=on_invalid,
        checks=(
            require(
                statistic in ("median", "max", "mean"),
                f"statistic must be 'median', 'max', or 'mean', got {statistic!r}",
            ),
            at_least("n_classes", n_classes, 2),
            *bootstrap_checks(n_bootstrap, ci_method, conf_level, key),
            positive_int("slice_chunk_size", slice_chunk_size),
        ),
        min_kept=_MIN_KEPT,
        # A constant slice leaves every conditional distribution equal to
        # the unconditional one, so the indices come out an exact 0, not
        # the NaN a variance ratio would give.
        zero_variance_outcome="zero",
    )

    assert ctx.X is not None  # a given-data method always passes X to prepare()
    X, invalid = ctx.X, ctx.invalid
    Y_3d = ctx.Y3
    t0 = _verbose.tic()
    # Refusing a malformed categorical column needs concrete values, so it
    # runs here and not in the traceable binning it guards.
    _check_categorical_codes(problem, X)
    bin_idx, n_eff = _bin_indices(problem, X, n_classes)

    # The contributing-bin diagnostic depends on the binning alone, so it is
    # computed once here, from the same bin assignment the kernel uses.
    n_valid_classes_per_param = _count_valid_bins(bin_idx, n_eff)
    _warn_sparse_bins(problem, n_valid_classes_per_param, n_classes)

    pawn_3d = _pawn_core(bin_idx, Y_3d, n_eff, statistic, slice_chunk_size)
    _warn_all_empty(n_valid_classes_per_param)

    pawn_conf: Array | None = None
    ci: CIInfo | None = None
    if n_bootstrap > 0:
        assert key is not None  # checked above, before the point estimate ran
        boot_stack = _pawn_bootstrap(
            bin_idx, Y_3d, n_eff, statistic, slice_chunk_size, key, n_bootstrap
        )

        # Squeeze the point estimate and the draws to the caller's own
        # layout first; interval() returns conf/ci already at that layout.
        point_sq = ctx.squeeze(pawn_3d)
        draws_sq = ctx.squeeze(boot_stack)
        if ci_method == "quantile":
            # Mean-centre the draws: the bootstrap excess over the point
            # estimate is bias, not sampling spread. NaN draws are skipped.
            draws_used = draws_sq - (jnp.nanmean(draws_sq, axis=0) - point_sq)
        else:
            draws_used = draws_sq
        confs, ci = interval(
            {"pawn": point_sq},
            {"pawn": draws_used},
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            keep_replicates=keep_replicates,
        )
        if keep_replicates:
            ci = dataclasses.replace(ci, replicates={"pawn": draws_sq})
        pawn_conf = confs["pawn"]

    pawn_out = ctx.squeeze(pawn_3d)

    result = PAWNResult(
        pawn=pawn_out,
        pawn_conf=pawn_conf,
        # Constant across the output axes (bin occupancy never reads Y);
        # broadcast to pawn's shape so the exported dataset aligns the two.
        n_valid_classes=jnp.broadcast_to(jnp.asarray(n_valid_classes_per_param), pawn_out.shape),
        problem=problem,
        invalid=invalid,
        ci=ci,
    )

    if verbose:
        elapsed = _verbose.stop(t0, result.pawn)
        N, T, K = ctx.Y3.shape
        # Same resolver and clamp _pawn_core ran, on the same shapes: cheap
        # arithmetic, reported rather than re-derived by hand.
        itemsize = jnp.result_type(Y_3d.dtype, jnp.float32).itemsize
        cs = resolve_batch_size(
            2 * problem.n_params * N * n_eff * itemsize, T * K, slice_chunk_size
        )
        origin = "user-set" if slice_chunk_size is not None else "resolved from the memory budget"
        _verbose.analysis_summary(
            method="jaxgsa.pawn.analyze",
            problem=problem,
            n_runs=int(ctx.Y.shape[0]),
            T=T,
            K=K,
            invalid=invalid,
            timings=[("estimator (includes compile on the first call)", elapsed)],
            notes=[
                f"slice_chunk_size: {cs} ({origin})",
                f"statistic: {statistic}",
                f"n_classes: {n_classes}",
            ],
            index_name="PAWN",
            values=result.pawn,
            conf=result.pawn_conf,
        )
    return result
