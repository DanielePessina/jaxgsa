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
is ``N * T * K`` elements rather than ``D`` times that.

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

from typing import Literal, cast

import jax
import jax.numpy as jnp
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.bootstrap import interval
from jaxgsa._core.entry import (
    at_least,
    gates,
    in_open_interval,
    one_of,
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
from jaxgsa._core.result import CIInfo
from jaxgsa._core.validation import _prepare_Y, _validate_output
from jaxgsa.problem import Problem
from jaxgsa.rank._result import RankResult

# Keyword-only parameters the irregular engine forwards unchanged to the
# per-channel recursive analyze() calls. verbose is excluded: the engine owns it.
_IRREGULAR_KWARGS = (
    "n_bootstrap",
    "conf_level",
    "ci_method",
    "key",
    "on_invalid",
    "keep_replicates",
)

# Fewest samples for which "the next-ranked row" means something.
_MIN_KEPT = 3

# Share of the rows each confidence-interval resample keeps. Same default as
# the R package ``sensitivity`` (``nsample = round(0.8 * nrow(X))``).
_SUBSAMPLE_FRACTION = 0.8


def _rank_core(X: Array, Y_3d: Array) -> Array:
    """Compute rank-based first-order indices for every output slice.

    Args:
        X: Input samples, shape ``(N, D)``.
        Y_3d: Output array promoted to shape ``(N, T, K)``.

    Returns:
        Indices, shape ``(T, K, D)``. A slice with zero variance gives ``NaN``.
    """
    centred = Y_3d - jnp.mean(Y_3d, axis=0, keepdims=True)
    variance = jnp.mean(centred**2, axis=0)  # (T, K)
    # Stable sort: tied input values keep their row order.
    order = jnp.argsort(X, axis=0, stable=True)  # (N, D)

    def one_param(rows: Array) -> Array:
        """Lag-one autocovariance of the output sorted by one input column."""
        sorted_y = centred[rows]  # (N, T, K)
        # Rank j pairs with rank j + 1. The largest rank wraps to rank 1.
        neighbour_y = jnp.roll(sorted_y, -1, axis=0)
        return jnp.mean(sorted_y * neighbour_y, axis=0) / variance

    s1 = jax.lax.map(one_param, order.T)  # (D, T, K)
    return jnp.moveaxis(s1, 0, -1)


def _subsample_draws(
    key: Array, s1_3d: Array, X: Array, Y_3d: Array, n_resamples: int
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

    Returns:
        ``(plain, corrected)`` scaled draws, each of shape
        ``(n_resamples, T, K, D)``. ``plain`` is centred on the point estimate
        and ``corrected`` on the bias-corrected estimate.
    """
    n = int(X.shape[0])
    m = min(max(int(_SUBSAMPLE_FRACTION * n), 2), n - 1)
    scale = (m / (n - m)) ** 0.5
    raw = []
    for subkey in jax.random.split(key, n_resamples):
        rows = jax.random.permutation(subkey, n)[:m]
        raw.append(_rank_core(X[rows], Y_3d[rows]))
    stacked = jnp.stack(raw)
    mean = jnp.mean(stacked, axis=0)
    spread = scale * (stacked - mean)
    # The estimator carries a bias of order 1/N. Under that model the
    # resample mean sits b (1/m - 1/N) away from the full-sample estimate, so
    # the full-sample bias b/N is (mean - estimate) * m / (N - m).
    corrected = s1_3d - (mean - s1_3d) * (m / (n - m))
    return s1_3d + spread, corrected + spread


def indices(problem: Problem, X: Array, Y: Array) -> tuple[Array]:
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

    Returns:
        A one-element tuple ``(S1,)``, shaped ``(D,)``, ``(K, D)`` or
        ``(T, K, D)`` to mirror the layout of ``Y``. A tuple, not a bare
        array, because every method's core returns one.

    Raises:
        ValueError: If the problem has categorical parameters, if ``X`` is not
            2-D, if its column count does not match the problem, if ``X`` and
            ``Y`` have differing row counts, or if ``Y`` has an invalid rank.
    """
    from jaxgsa.rank import SPEC

    gates(SPEC, problem, method="jaxgsa.rank.indices")
    X_arr = validate_inputs(problem, X)
    Y_arr = _validate_output(Y, int(X_arr.shape[0]), problem)
    Y3, layout = _prepare_Y(Y_arr)
    return (layout.squeeze(_rank_core(X_arr, Y3)),)


def analyze(
    problem: Problem,
    X: Array,
    Y: Array | IrregularY,
    *,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
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
        on_invalid: What to do about a row of ``X`` or ``Y`` that holds a
            non-finite value. ``"raise"`` (default) refuses the sample,
            ``"drop"`` removes those rows and analyzes the rest, and
            ``"propagate"`` warns and computes anyway. See
            :mod:`jaxgsa._core.invalid`.
        verbose: If ``True`` (default), print a short summary to stdout.
            Pass ``False`` for a silent run.
        keep_replicates: Keep the per-resample indices on
            ``RankResult.ci.replicates``. The kept draws are the rescaled
            ones that the interval is built from.

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
            ``key`` was given, or the non-finite policy refuses the sample.

    Warns:
        JaxgsaWarning: If an output slice has zero variance. The index is then
            ``NaN``, because it is a ratio to the output variance.
    """
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

    from jaxgsa.rank import SPEC

    ctx = prepare(
        SPEC,
        problem,
        Y,
        X=X,
        on_invalid=on_invalid,
        checks=(
            at_least("n_bootstrap", n_bootstrap, 0),
            one_of("ci_method", ci_method, ("quantile", "gaussian")),
            in_open_interval("conf_level", conf_level, 0.0, 1.0),
            require(
                n_bootstrap == 0 or key is not None,
                "key is required when n_bootstrap > 0",
            ),
        ),
        min_kept=_MIN_KEPT,
    )

    assert ctx.X is not None  # a given-data method always passes X to prepare()
    X, invalid = ctx.X, ctx.invalid
    if X.shape[0] < _MIN_KEPT:
        raise ValueError(f"jaxgsa.rank.analyze needs at least {_MIN_KEPT} rows, got {X.shape[0]}")
    Y_3d = ctx.Y3
    t0 = _verbose.tic()
    s1_3d = _rank_core(X, Y_3d)

    s1_conf: Array | None = None
    ci: CIInfo | None = None
    if n_bootstrap > 0:
        assert key is not None  # checked above, before the point estimate ran
        plain, corrected = _subsample_draws(key, s1_3d, X, Y_3d, n_bootstrap)
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
        _, T, K = Y_3d.shape
        _verbose.analysis_summary(
            method="jaxgsa.rank.analyze",
            problem=problem,
            n_runs=int(ctx.Y.shape[0]),
            T=T,
            K=K,
            invalid=invalid,
            timings=[("estimator (includes compile on the first call)", elapsed)],
            index_name="S1",
            values=result.S1,
            conf=result.S1_conf,
        )
    return result
