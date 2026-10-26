"""Rank-based first-order Sobol' index estimator.

Sort the sample by one input, pair each output with the output of the
next-ranked row, and read the first-order index off the lag-one
autocovariance of that sorted output sequence. If the output depends
strongly on the input, neighbours in the sort have similar outputs and the
autocovariance is large.

Array shape conventions used throughout:
    N  — number of samples
    D  — number of parameters
    T  — number of time steps (singleton-squeezed when absent)
    K  — number of output variables (singleton-squeezed when absent)

For one parameter, let ``pi(j)`` be the rank of ``X_ij`` among the ``N``
values of that column, and let ``N(j)`` be the row with rank ``pi(j) + 1``.
The row with the largest rank pairs with the row of rank 1. The estimator is::

    S1_i = (mean_j(Y_j * Y_N(j)) - mean(Y)**2) / (mean(Y**2) - mean(Y)**2)

The ranking uses a stable sort, so tied input values keep their row order.
The wrap-around pairing makes ``N(.)`` a permutation of the rows. The mean of
``Y_N(j)`` then equals the mean of ``Y``, so the numerator is the mean
product of the centred outputs. The kernel works on centred outputs, which
avoids the cancellation in ``mean(Y_j * Y_N(j)) - mean(Y)**2`` in float32.

The kernel is one stable ``argsort`` of ``X`` per column, a gather, a roll and
two means. It maps over the parameters with ``lax.map``, so the working set
is ``N * T * K`` elements rather than ``D`` times that. It also maps over
chunks of the flattened ``T * K`` output slices (``slice_chunk_size``), so
the working set is ``N`` times the chunk width.

References:
    Gamboa, Gremaud, Klein & Lagnoux (2022). Global sensitivity analysis: a
    novel generation of mighty estimators based on rank statistics.
    Bernoulli 28(4):2345-2374. doi:10.3150/21-BEJ1421. arXiv:2003.01772,
    Sec. 2.3 eq. (8) (the neighbour map N) and Sec. 3.2 eq. (20) (the
    estimator).
    Chatterjee (2021). A new coefficient of correlation. Journal of the
    American Statistical Association 116(536):2009-2022. (The rank-pairing
    idea that the estimator builds on.)
"""

from __future__ import annotations

from functools import partial
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.batching import resolve_batch_size
from jaxgsa._core.bootstrap import interval
from jaxgsa._core.column_chunks import map_column_chunks
from jaxgsa._core.entry import (
    bootstrap_checks,
    check_scalars,
    gates,
    positive_int,
    prepare,
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
from jaxgsa._core.result import CIInfo
from jaxgsa._core.validation import (
    _prepare_Y,
    _validate_output,
    _zero_variance_mask,
    as_key,
)
from jaxgsa.problem import Problem
from jaxgsa.rank._result import RankResult

# Keyword-only parameters the irregular engine forwards unchanged to the
# per-channel recursive analyze() calls. verbose is excluded: the engine owns it.
_IRREGULAR_KWARGS = (
    "n_bootstrap",
    "conf_level",
    "ci_method",
    "key",
    "slice_chunk_size",
    "on_invalid",
    "keep_replicates",
)

# Fewest samples for which "the next-ranked row" means something.
_MIN_KEPT = 3

# Share of the rows each confidence-interval resample keeps. Same default as
# the R package ``sensitivity`` (``nsample = round(0.8 * nrow(X))``).
_SUBSAMPLE_FRACTION = 0.8


# Live (N,)-sized arrays per output slice in the forward pass: the gathered
# column, its centred copy, the sorted copy, its rolled neighbour, and the
# product the mean reads. The automatic slice chunk is sized against this.
_RANK_LIVE_ARRAYS = 6


def _slice_width(
    n_rows: int, n_slices: int, dtype: jnp.dtype, slice_chunk_size: int | None
) -> int:
    """Resolve the number of output slices the kernel reads per chunk.

    Args:
        n_rows: Number of samples ``N``.
        n_slices: Flattened output slices, ``T * K``.
        dtype: Dtype of the outputs.
        slice_chunk_size: The caller's value, or ``None`` to derive one from
            the active memory budget.

    Returns:
        Slices per chunk, in ``[1, n_slices]``.
    """
    itemsize = jnp.dtype(jnp.result_type(dtype, jnp.float32)).itemsize
    return resolve_batch_size(_RANK_LIVE_ARRAYS * n_rows * itemsize, n_slices, slice_chunk_size)


def _rank_columns(X: Array, Y_cols: Array, width: int, rows: Array | None = None) -> Array:
    """Compute rank-based first-order indices for every output column.

    Every index reads one output column alone, so the columns are taken
    ``width`` at a time (:func:`jaxgsa._core.column_chunks.map_column_chunks`)
    and the working set is ``N * width`` elements, not ``N * T * K``.

    Args:
        X: Input samples, shape ``(N, D)``.
        Y_cols: Outputs with the slices flattened, shape ``(N, T * K)``.
        width: Columns per chunk (static at the jit boundary).
        rows: Rows to keep, shape ``(m,)``, or ``None`` for all. A subsample
            gathers its rows inside each chunk, so no subsampled copy of the
            whole of ``Y`` is formed.

    Returns:
        Indices, shape ``(T * K, D)``. A column with zero variance gives
        ``NaN``.
    """
    if rows is not None:
        X = X[rows]
    # Stable sort: tied input values keep their row order. One sort serves
    # every chunk.
    order = jnp.argsort(X, axis=0, stable=True)  # (N, D)

    def one_chunk(cols: Array) -> Array:
        """Indices of one chunk of columns, shape ``(width, D)``."""
        if rows is not None:
            cols = cols[rows]
        centred = cols - jnp.mean(cols, axis=0, keepdims=True)
        variance = jnp.mean(centred**2, axis=0)  # (width,)
        # The zero-variance warning's own test. A constant float column's
        # mean rounds, so ``centred`` holds one repeated round-off value and
        # the ratio below is that value squared over itself squared: 1.0, or
        # NaN when the round-off happens to be 0. Double where: divide by a
        # safe 1 on a flagged column, then put the NaN back, so the gradient
        # of every other column stays finite.
        zero = _zero_variance_mask(cols)  # (width,)
        safe_variance = jnp.where(zero, jnp.ones_like(variance), variance)

        # Rematerialised in reverse mode: lax.map would otherwise save the
        # gathered and rolled (N, width) copies of every parameter, D of
        # each, for the backward pass (gradient about 20x Y at D = 8). With
        # the checkpoint only ``param_rows`` is saved and the gather is
        # recomputed.
        @jax.checkpoint
        def one_param(param_rows: Array) -> Array:
            """Lag-one autocovariance of the output sorted by one input column."""
            sorted_y = centred[param_rows]  # (N, width)
            # Rank j pairs with rank j + 1. The largest rank wraps to rank 1.
            neighbour_y = jnp.roll(sorted_y, -1, axis=0)
            ratio = jnp.mean(sorted_y * neighbour_y, axis=0) / safe_variance
            return jnp.where(zero, jnp.nan, ratio)

        return jax.lax.map(one_param, order.T).T  # (width, D)

    return map_column_chunks(one_chunk, Y_cols, width, axis=0)


@partial(jax.jit, static_argnames=("width",))
def _rank_core(X: Array, Y_3d: Array, width: int) -> Array:
    """Compute rank-based first-order indices for every output slice.

    Args:
        X: Input samples, shape ``(N, D)``.
        Y_3d: Output array promoted to shape ``(N, T, K)``.
        width: Output slices per chunk.

    Returns:
        Indices, shape ``(T, K, D)``. A slice with zero variance gives ``NaN``.
    """
    n, T, K = Y_3d.shape
    return _rank_columns(X, Y_3d.reshape(n, T * K), width).reshape(T, K, -1)


@partial(jax.jit, static_argnames=("m", "width"))
def _subsample_estimates(keys: Array, X: Array, Y_3d: Array, m: int, width: int) -> Array:
    """Compute the index on one subsample of ``m`` distinct rows per key.

    One compiled program runs every draw. The draws run one after another,
    so the peak memory is one subsample, not all of them.

    Args:
        keys: PRNG keys, one per resample, shape ``(n_resamples,)``.
        X: Input samples, shape ``(N, D)``.
        Y_3d: Outputs, shape ``(N, T, K)``.
        m: Rows kept in each subsample.
        width: Output slices per chunk.

    Returns:
        Indices per resample, shape ``(n_resamples, T, K, D)``.
    """
    n, T, K = Y_3d.shape
    Y_cols = Y_3d.reshape(n, T * K)

    def one_draw(subkey: Array) -> Array:
        rows = jax.random.permutation(subkey, n)[:m]
        return _rank_columns(X, Y_cols, width, rows).reshape(T, K, -1)

    return jax.lax.map(one_draw, keys)


def _subsample_draws(
    key: Array, s1_3d: Array, X: Array, Y_3d: Array, n_resamples: int, width: int
) -> tuple[Array, Array]:
    """Draw resampled indices by subsampling rows without replacement.

    A resample with replacement repeats rows. Repeated rows have equal inputs,
    so they sort next to each other and pair with their own copy. That adds a
    positive bias (about 0.37 for a parameter with a true index of 0, measured
    on Ishigami at N = 4000), so this estimator cannot use the usual bootstrap.

    Each resample keeps ``m = floor(0.8 * N)`` distinct rows instead. Such an
    estimate scatters around the full-sample estimate less than a fresh
    sample would. The variance ratio is the finite-population factor
    ``(N - m) / m``. The deviations of the resamples from their own mean are
    therefore scaled by ``sqrt(m / (N - m))`` (2 for ``m = 0.8 N``).

    The estimator also carries a bias of order ``1/N``: neighbours in a
    smaller sample are farther apart. A smaller resample has a larger bias,
    so the gap between the resample mean and the point estimate measures it:
    the bias estimate is ``(mean - estimate) * m / (N - m)`` (one Richardson
    extrapolation step). The second set of draws is centred on the point
    estimate minus that bias. :func:`analyze` reports the hull of the two
    intervals. The plain interval misses the true value when the bias exceeds
    the spread, for example for ``Y = X_1``, where the estimate is about
    ``1 - 6/N`` and the spread is of order ``N^(-3/2)``. The corrected
    interval alone under-covers when the bias is small, because the bias
    estimate is noisy. In a simulation (300 samples each, 95% level) the hull
    covered the true value in 96 to 100% of samples for Ishigami at
    ``N = 500`` and ``2000`` and for ``Y = X_1`` at ``N = 200`` and ``1000``.
    It is about 10 to 15% wider than the plain interval.

    The scaling is a heuristic, not an exact result for a rank statistic.
    It degenerates for very small ``N`` (for ``N = 4`` every resample has the
    same value).

    Args:
        key: PRNG key.
        s1_3d: Point estimate, shape ``(T, K, D)``.
        X: Input samples, shape ``(N, D)``.
        Y_3d: Outputs, shape ``(N, T, K)``.
        n_resamples: Number of resamples.
        width: Output slices per chunk.

    Returns:
        ``(plain, corrected)`` scaled draws, each of shape
        ``(n_resamples, T, K, D)``. ``plain`` is centred on the point estimate
        and ``corrected`` on the bias-corrected estimate.
    """
    n = int(X.shape[0])
    m = min(max(int(_SUBSAMPLE_FRACTION * n), 2), n - 1)
    scale = (m / (n - m)) ** 0.5
    stacked = _subsample_estimates(jax.random.split(key, n_resamples), X, Y_3d, m, width)
    mean = jnp.mean(stacked, axis=0)
    spread = scale * (stacked - mean)
    # The estimator carries a bias of order 1/N. Under that model the
    # resample mean sits b (1/m - 1/N) away from the full-sample estimate, so
    # the full-sample bias b/N is (mean - estimate) * m / (N - m).
    corrected = s1_3d - (mean - s1_3d) * (m / (n - m))
    return s1_3d + spread, corrected + spread


class RankIndices(NamedTuple):
    """Plain-array rank-based indices, as returned by :func:`indices`.

    Fields are named like the matching result fields. Tuple unpacking
    works as for a plain tuple.

    Attributes:
        S1: Rank-based first-order index per parameter.
    """

    S1: Array


def indices(
    problem: Problem, X: Array, Y: Array, *, slice_chunk_size: int | None = None
) -> RankIndices:
    """Compute rank-based first-order indices as plain arrays, with no diagnostics.

    This is the transformable core of :func:`analyze`. It runs the same kernel
    on the same data and returns the same numbers, but it does nothing else: no
    non-finite check, no zero-variance warning, no
    :class:`jaxgsa.rank.RankResult`, and no read of any array value on the
    host. So it composes with ``jax.jit`` and ``jax.vmap``. The capability gate
    still runs, because it reads only static problem metadata.

    The estimator reads the ranks of ``X``, which are piecewise constant, so
    the gradient with respect to ``X`` is 0 almost everywhere. The gradient
    with respect to ``Y`` is not 0.

    Tier T4 (behavioural contract): the returned array must equal the ``S1``
    field of ``analyze``'s result on clean outputs, and the function must
    survive ``jit`` and ``vmap``. Checked in ``tests/test_rank.py``.

    Args:
        problem: Problem definition with D parameters.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        slice_chunk_size: As in :func:`analyze`. The value changes only the
            peak memory, never an index.

    Returns:
        A one-field :class:`RankIndices` named tuple ``(S1,)``, shaped ``(D,)``, ``(K, D)`` or
        ``(T, K, D)`` to mirror the layout of ``Y``. A tuple, not a bare
        array, because every method's core returns one.

    Raises:
        ValueError: If the problem has categorical parameters, if ``X`` is not
            2-D, if its column count does not match the problem, if ``X`` and
            ``Y`` have differing row counts, if ``Y`` has an invalid rank, or
            if ``slice_chunk_size`` is given and is not an integer of at
            least 1.
    """
    from jaxgsa.rank import _SPEC

    gates(_SPEC, problem, method="jaxgsa.rank.indices")
    check_scalars(
        (positive_int("slice_chunk_size", slice_chunk_size),), method="jaxgsa.rank.indices"
    )
    X_arr = validate_inputs(problem, X)
    Y_arr = _validate_output(Y, int(X_arr.shape[0]), problem)
    Y3, layout = _prepare_Y(Y_arr)
    n, T, K = Y3.shape
    width = _slice_width(n, T * K, Y3.dtype, slice_chunk_size)
    return RankIndices(S1=layout.squeeze(_rank_core(X_arr, Y3, width)))


def analyze(
    problem: Problem,
    X: Array,
    Y: Array | IrregularY,
    *,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | int | None = None,
    slice_chunk_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> RankResult | IrregularResult:
    """Compute first-order Sobol' indices from a given sample by rank statistics.

    For each parameter, the rows are sorted by that parameter. Each output is
    paired with the output of the next-ranked row, and the last rank pairs with
    the first. The index is the covariance of the pairs divided by the output
    variance (Gamboa et al. 2022, eq. 20). It estimates
    ``S1_i = Var(E[Y | X_i]) / Var(Y)`` from one ``(X, Y)`` sample, with no
    special design and no binning choice.

    Any ``(X, Y)`` pair works. The estimator is cheap, since it needs one sort
    per parameter, and it is consistent for the first-order index.

    Correlated parameters are supported. ``Var(E[Y | X_i]) / Var(Y)`` is
    defined for any joint distribution of the inputs, so a declared
    ``problem.correlation`` or a custom sampler does not invalidate the index.
    Under dependence the index measures the parameter's total first-order
    influence, including the influence it carries through correlated partners.
    A parameter that the model ignores can score above 0 when it correlates
    with an influential parameter. That reading is correct, not an estimation
    error.

    Categorical parameters are refused. Their level codes are heavily tied, and
    the rank pairing is ill defined for ties.

    Ties in a continuous parameter are broken by row order (a stable sort). A
    few ties change nothing of substance. Many ties among the inputs bias the
    index, because row order then decides which outputs pair.

    The confidence intervals do not use the usual bootstrap. A resample with
    replacement repeats rows, repeated rows pair with their own copies, and the
    replicates are biased upward. The resamples here keep 80% of the rows,
    drawn without replacement, and the spread is rescaled to the full sample
    size. See ``n_bootstrap``.

    Use :func:`indices` when you need the same numbers inside ``jit`` or
    ``vmap``.

    Args:
        problem: Problem definition with D parameters.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        n_bootstrap: Number of resamples for confidence intervals. ``0``
            disables the confidence intervals. Each resample is a random 80%
            of the rows, drawn without replacement. The deviations from the
            point estimate are scaled by 2 to account for the smaller size.
            The interval is the hull of this interval and the same interval
            centred on a bias-corrected estimate. See the API page.
        conf_level: Confidence level for the intervals.
        ci_method: How the interval endpoints are formed. ``"quantile"``
            (default) reads them off the empirical resample distribution.
            ``"gaussian"`` centres them on the point estimate and takes
            ``+/- z * sd`` of the draws.
        key: A ``jax.random`` key for the resampling. Required when
            ``n_bootstrap > 0``. Pass ``jax.random.key(0)`` if you have an
            integer seed.
        slice_chunk_size: Maximum number of flattened ``T*K`` output slices
            the kernel reads at once. It caps the peak memory, and the memory
            of a gradient, for a large ``T * K``. Every index reads one slice
            alone, so the value never changes a result. ``None`` (default)
            derives one from the active memory budget
            (:func:`jaxgsa.config.set_memory_budget`). An explicit value wins
            over the budget, is clamped to ``T * K``, and must be an integer
            of at least 1.
        on_invalid: What to do about a row of ``X`` or ``Y`` that holds a
            non-finite value. ``"raise"`` (default) refuses the sample,
            ``"drop"`` removes those rows and analyzes the rest, and
            ``"propagate"`` warns and computes anyway. See
            :mod:`jaxgsa._core.invalid`.
        verbose: If ``True`` (default), print a short summary to stdout.
            Pass ``False`` for a silent run.
        keep_replicates: Keep the per-resample indices on
            ``RankResult.ci.replicates``. The kept draws are the plain
            bootstrap draws. Recomputing an interval from them gives the
            plain interval, which can be narrower than ``S1_conf``. That
            field is the hull of the plain and the bias-corrected
            intervals.

    Returns:
        A :class:`RankResult` with ``S1`` shaped ``(D,)``, ``(K, D)``, or
        ``(T, K, D)``, optional confidence intervals in ``S1_conf``, how those
        intervals were made in ``ci``, and the non-finite report in
        ``invalid``.

    Raises:
        ValueError: If the problem has categorical parameters, X is not 2-D,
            its column count does not match the problem, X and Y have differing
            row counts, ``conf_level`` is not in ``(0, 1)``, ``on_invalid`` is
            not one of the policies, ``n_bootstrap < 0``, ``ci_method`` is not
            ``"quantile"`` or ``"gaussian"``, ``n_bootstrap > 0`` and no
            ``key`` was given, ``slice_chunk_size`` is given and is not an
            integer of at least 1, or the non-finite policy refuses the
            sample.

    Warns:
        JaxgsaWarning: If an output slice has zero variance. The index is then
            ``NaN``, because it is a ratio to the output variance.
    """
    key = as_key(key, "rank")
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

    from jaxgsa.rank import _SPEC

    ctx = prepare(
        _SPEC,
        problem,
        Y,
        X=X,
        on_invalid=on_invalid,
        checks=(
            *bootstrap_checks(n_bootstrap, ci_method, conf_level, key),
            positive_int("slice_chunk_size", slice_chunk_size),
        ),
        min_kept=_MIN_KEPT,
    )

    assert ctx.X is not None  # a given-data method always passes X to prepare()
    X, invalid = ctx.X, ctx.invalid
    if X.shape[0] < _MIN_KEPT:
        raise ValueError(f"jaxgsa.rank: analyze needs at least {_MIN_KEPT} rows, got {X.shape[0]}")
    Y_3d = ctx.Y3
    n, T, K = Y_3d.shape
    width = _slice_width(n, T * K, Y_3d.dtype, slice_chunk_size)
    t0 = _verbose.tic()
    s1_3d = _rank_core(X, Y_3d, width)

    s1_conf: Array | None = None
    ci: CIInfo | None = None
    if n_bootstrap > 0:
        assert key is not None  # checked above, before the point estimate ran
        plain, corrected = _subsample_draws(key, s1_3d, X, Y_3d, n_bootstrap, width)
        confs, ci = interval(
            {"S1": ctx.squeeze(s1_3d)},
            {"S1": ctx.squeeze(plain)},
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            keep_replicates=keep_replicates,
        )
        corrected_confs, _ = interval(
            {"S1": ctx.squeeze(s1_3d - jnp.mean(plain - corrected, axis=0))},
            {"S1": ctx.squeeze(corrected)},
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            keep_replicates=False,
        )
        # Report the hull of the plain and the bias-corrected interval. See
        # _subsample_draws for why neither alone is enough.
        s1_conf = jnp.stack(
            [
                jnp.minimum(confs["S1"][0], corrected_confs["S1"][0]),
                jnp.maximum(confs["S1"][1], corrected_confs["S1"][1]),
            ]
        )

    result = RankResult(
        S1=ctx.squeeze(s1_3d),
        S1_conf=s1_conf,
        problem=problem,
        invalid=invalid,
        ci=ci,
    )

    if verbose:
        elapsed = _verbose.stop(t0, result.S1)
        origin = "user-set" if slice_chunk_size is not None else "resolved from the memory budget"
        _verbose.analysis_summary(
            method="jaxgsa.rank.analyze",
            problem=problem,
            n_runs=int(ctx.Y.shape[0]),
            T=T,
            K=K,
            invalid=invalid,
            timings=[("estimator (includes compile on the first call)", elapsed)],
            notes=[f"slice_chunk_size: {width} ({origin})"],
            index_name="S1",
            values=result.S1,
            conf=result.S1_conf,
        )
    return result
