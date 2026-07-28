"""PAWN analysis: distribution-based sensitivity via KS distances.

Computes the Kolmogorov-Smirnov (KS) distance — the largest vertical gap
between two empirical CDFs — between the unconditional output CDF and
the conditional CDF when each input is confined to a bin of its range.
The PAWN index aggregates KS values across bins via median, max, or mean.

Array shape conventions used throughout:
    N  — number of samples
    D  — number of input parameters
    T  — number of time steps (singleton-squeezed when absent)
    K  — number of output variables (singleton-squeezed when absent)

The KS computation is vectorized: each output column is sorted once, and
the unconditional and per-bin conditional ECDFs are compared only at
distinct output values (tie-group ends).  This makes the statistic equal
to the two-sample KS of ``scipy.stats.ks_2samp`` even for tied/discrete
outputs, not just continuous ones.  Bin assignment (via ``searchsorted``)
depends only on the inputs, so it is computed once and shared across all
output columns; a single JIT-compiled kernel (one compilation per unique
``(N, D, n_bins)`` and output-column count ``T*K``) is vmapped over the
flattened ``T*K`` output columns.  The ``statistic`` aggregation runs
outside JIT so all three statistics share one compilation.

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
from jax import Array

from jaxgsa._core.bootstrap import _percentile_ci
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa._core.validation import _prepare_Y, _squeeze_output_axes, _validate_xy_inputs
from jaxgsa.pawn._result import PAWNResult
from jaxgsa.problem import Problem


def _bin_indices(X_u01: Array, n_bins: int) -> Array:
    """Assign each sample to an equal-width bin in ``[0, 1]``.

    Samples falling outside ``[0, 1]`` (possible when the caller passes
    inputs beyond the declared bounds of a ``uniform`` parameter) are
    given the sentinel ``-1`` so they match no bin and are excluded from
    every conditional set.  ``NaN`` inputs are treated the same way.
    Values exactly ``0.0`` map to the first bin and ``1.0`` to the last
    bin.

    Args:
        X_u01: Unit-interval inputs ``(N, D)``.
        n_bins: Number of equal-width bins.

    Returns:
        Integer bin indices ``(N, D)`` in ``[0, n_bins)``, or ``-1`` for
        out-of-range or ``NaN`` samples.
    """
    bin_edges = jnp.linspace(0.0, 1.0, n_bins + 1)
    idx = jnp.clip(jnp.searchsorted(bin_edges, X_u01, side="right") - 1, 0, n_bins - 1)
    in_range = (X_u01 >= 0.0) & (X_u01 <= 1.0)
    return jnp.where(in_range, idx, -1)


@lru_cache(maxsize=32)
def _get_pawn_ks(n_bins: int):
    """Return a JIT-compiled KS kernel for the given ``n_bins``.

    The kernel maps precomputed bin indices and a batch of output columns
    to per-bin KS statistics, vmapped over the output columns.

    Args:
        n_bins: Number of equal-width conditioning bins (static).

    Returns:
        A jitted callable ``(bin_idx (N, D), Y_cols (N, M)) -> ks
        (M, D, n_bins)`` where ``M`` is the number of output columns.
    """

    def _impl(bin_idx: Array, Y_cols: Array) -> Array:
        """Compute per-bin KS statistics for every output column.

        Args:
            bin_idx: Integer bin indices ``(N, D)`` (``-1`` = excluded).
            Y_cols: Output columns ``(N, M)``.

        Returns:
            KS statistics ``(M, D, n_bins)`` with ``NaN`` where a bin has
            fewer than two samples.
        """
        N = bin_idx.shape[0]
        param_bins = jnp.arange(n_bins)

        def _ks_one_col(y: Array) -> Array:
            """KS statistics ``(D, n_bins)`` for a single output column."""
            dtype = jnp.result_type(y.dtype, jnp.float32)
            order = jnp.argsort(y)
            y_sorted = y[order]
            # A sorted position is a "group end" if it is the last member
            # of its tie group; evaluating ECDFs only there yields the
            # value-based (tie-aware) two-sample KS statistic.
            is_group_end = jnp.concatenate([y_sorted[:-1] != y_sorted[1:], jnp.array([True])])
            bin_idx_sorted = bin_idx[order]
            uncond_cdf = jnp.arange(1, N + 1, dtype=dtype) / N

            def _ks_one_param(bid: Array) -> Array:
                """KS statistics ``(n_bins,)`` for one input dimension."""
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


def _aggregate_ks(
    ks: Array,
    statistic: Literal["median", "max", "mean"],
) -> Array:
    """Reduce KS values over the trailing ``n_bins`` axis to PAWN indices.

    Args:
        ks: KS statistics with bins on the last axis, e.g. ``(..., n_bins)``.
            ``NaN`` entries (bins with fewer than two samples) are ignored.
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
    X_u01: Array,
    Y_3d: Array,
    n_bins: int,
    statistic: Literal["median", "max", "mean"],
    *,
    warn: bool = True,
) -> Array:
    """Core PAWN computation over all (T, K) output slices.

    Args:
        X_u01: Unit-interval inputs ``(N, D)``.
        Y_3d: Output array promoted to ``(N, T, K)``.
        n_bins: Number of bins per input dimension.
        statistic: Aggregation method across bins.
        warn: If True, emit one warning per input whose bins are all empty
            (all fewer than two samples).  Set False for bootstrap resamples
            to avoid duplicate warnings.

    Returns:
        PAWN indices ``(T, K, D)``.
    """
    N, T, K = Y_3d.shape
    D = X_u01.shape[1]

    # Bin assignment depends only on the inputs, so build it once and
    # process every output column with a single vmapped kernel.
    bin_idx = _bin_indices(X_u01, n_bins)
    Y_cols = Y_3d.reshape(N, T * K)
    ks = _get_pawn_ks(n_bins)(bin_idx, Y_cols)  # (T*K, D, n_bins)

    if warn:
        # Whether every bin of an input is empty depends only on the input
        # binning (not on Y), so it is identical across output columns:
        # one host sync, one warning per affected input.
        all_empty = jnp.all(jnp.isnan(ks), axis=(0, 2)).tolist()
        for d, empty in enumerate(all_empty):
            if empty:
                warnings.warn(
                    f"PAWN: all bins empty for parameter {d}, returning NaN",
                    stacklevel=3,
                )

    pawn = _aggregate_ks(ks, statistic)  # (T*K, D)
    return pawn.reshape(T, K, D)


def analyze(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_bins: int = 10,
    statistic: Literal["median", "max", "mean"] = "median",
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    seed: int = 0,
    slice_chunk_size: int = 2048,
) -> PAWNResult:
    """Compute PAWN sensitivity indices.

    PAWN measures how much fixing an input changes the whole output
    *distribution*, not just its variance. For each input, the samples
    are split into ``n_bins`` conditioning bins (equal-width on the
    CDF-transformed unit interval, i.e. equal-probability under the
    input's marginal), and each bin's conditional output CDF is compared
    with the unconditional CDF via the Kolmogorov-Smirnov (KS) statistic
    — the largest vertical gap between the two CDFs, a number in [0, 1].
    The per-bin KS values are then aggregated (median by default) into
    one index per input: 0 means the input has no effect on the output
    distribution; larger values mean stronger influence.

    It is a given-data method — any (X, Y) sample works, with no special
    design — and a good pick when the output is skewed or multimodal, so
    that variance-based indices summarize its uncertainty poorly.

    Args:
        problem: Problem definition with D parameters.
        X: Input sample matrix ``(N, D)``.
        Y: Model output ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        n_bins: Number of conditioning bins per input (equal-probability
            under each input's marginal). More bins condition each input
            more tightly but leave fewer samples per bin (roughly
            ``N / n_bins``), making each KS value noisier; the default
            of 10 suits N in the thousands.
        statistic: Aggregation of KS values across bins. ``"median"``
            (default) is robust to a few noisy bins; ``"max"`` is the
            conservative choice for screening out non-influential inputs
            (an input is negligible only if *no* bin shifts the output);
            ``"mean"`` weights all bins equally.
        n_bootstrap: Number of bootstrap resamples for confidence
            intervals. Set to 0 to skip.
        conf_level: Confidence level for bootstrap intervals.
        seed: Random seed for bootstrap resampling.
        slice_chunk_size: Accepted for signature parity with the other
            ``analyze`` functions; PAWN needs no output-slice chunking,
            so it has no effect.

    Returns:
        PAWNResult with PAWN indices and optional confidence intervals.

    Raises:
        ValueError: If X is not 2-D, its column count does not match the
            problem, X and Y have differing row counts, ``statistic`` is
            not one of ``"median"``/``"max"``/``"mean"``, ``n_bins < 2``,
            or ``conf_level`` is not in ``(0, 1)``.
    """
    X = jnp.asarray(X)
    # PAWN conditions on bins of each input and compares output CDFs, so a
    # declared input correlation does not invalidate the indices.
    Y = _validate_xy_inputs(problem, X, Y, correlation_ok=True, method="jaxgsa.pawn.analyze")
    if statistic not in ("median", "max", "mean"):
        raise ValueError(f"statistic must be 'median', 'max', or 'mean', got {statistic!r}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if not 0 < conf_level < 1:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")

    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)
    X_u01 = cdf_to_unit_interval(X, problem)

    pawn_3d = _pawn_core(X_u01, Y_3d, n_bins, statistic)

    pawn_conf: Array | None = None
    if n_bootstrap > 0:
        key = jax.random.PRNGKey(seed + 1)
        N = X.shape[0]
        boot_draws = []
        for _ in range(n_bootstrap):
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, N, shape=(N,), replace=True)
            boot_pawn = _pawn_core(X_u01[idx], Y_3d[idx], n_bins, statistic, warn=False)
            boot_draws.append(boot_pawn)

        boot_stack = jnp.stack(boot_draws, axis=0)
        pawn_conf_3d = _percentile_ci(boot_stack, conf_level)

        pawn_conf = _squeeze_output_axes(pawn_conf_3d, squeeze_time, squeeze_output)

    pawn_out = _squeeze_output_axes(pawn_3d, squeeze_time, squeeze_output)

    return PAWNResult(
        pawn=pawn_out,
        pawn_conf=pawn_conf,
        problem=problem,
    )
