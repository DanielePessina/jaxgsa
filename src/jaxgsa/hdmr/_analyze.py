"""RS-HDMR (Random Sampling High-Dimensional Model Representation) analysis.

Computes ANCOVA-based sensitivity indices from arbitrary ``(X, Y)`` pairs
using B-spline surrogate modelling. The result keeps the fitted surrogate, so
it can also predict at new points and give Shapley effects.
"""

import itertools
import math
from collections.abc import Callable
from functools import lru_cache
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.bootstrap import _bootstrap_ci_endpoints
from jaxgsa._core.entry import (
    ScalarCheck,
    at_least,
    check_scalars,
    in_open_interval,
    one_of,
    prepare,
)
from jaxgsa._core.invalid import InvalidReport, OnInvalid
from jaxgsa._core.result import CIInfo
from jaxgsa._core.surrogate import _PredictPlan
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa._core.validation import (
    YLayout,
    _prepare_Y,
)
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.hdmr._engine import (
    _build_B1,
    _build_B2,
    _build_B3,
    _compute_f_crits,
)
from jaxgsa.hdmr._fit import _fit_hdmr
from jaxgsa.hdmr._result import HDMRResult, _HDMRFit
from jaxgsa.problem import Problem

# Fewest rows the B-spline backfitting fit can run on. It is the module's own
# documented minimum, re-used as the floor for ``on_invalid="drop"`` so that
# dropping past it refuses in the policy's words rather than in the generic
# "Need at least 300 samples" check further down.
_MIN_ROWS = 300


def _fit_scalar_checks(maxiter: int, m: int, lambdax: float) -> tuple[ScalarCheck, ...]:
    """Build the scalar checks the HDMR fit arguments must pass.

    One definition serves both entry points: :func:`analyze` hands these to
    ``prepare`` and :func:`_hdmr_core` (which :func:`indices` routes through)
    re-runs them with ``check_scalars``, so the two cannot drift apart.

    The bounds are where the fit degenerates silently rather than loudly:

    * ``m >= 1``: the cubic B-spline basis is scaled by ``m**3``
      (see ``_engine._bspline_basis``), so ``m = 0`` makes every basis
      function exactly zero and every index silently 0. ``m = 1`` is the
      smallest valid basis — one knot interval, ``m + 3 = 4`` functions.
    * ``maxiter >= 1``: zero iterations skips backfitting entirely and
      returns the unfitted coefficients without a word.
    * ``lambdax >= 0``: a negative value subtracts from the Gram diagonal,
      which turns the regularized solve into NaN coefficients.

    Args:
        maxiter: Maximum backfitting iterations.
        m: B-spline intervals per dimension.
        lambdax: Tikhonov regularization strength.

    Returns:
        The checks, ready for ``prepare(checks=...)`` or ``check_scalars``.
    """
    return (
        at_least("maxiter", maxiter, 1),
        at_least("m", m, 1),
        at_least("lambdax", lambdax, 0.0),
    )


class _HDMRStaticData(NamedTuple):
    """Term metadata and basis index tables for one ``(D, maxorder, m)``.

    Every field is host-side and shared by all output slices. The names
    matter: the values used to travel as a bare twelve-tuple unpacked
    positionally at three call sites, two of them with blind placeholders, so
    reordering the fields produced silently wrong indices instead of an error.
    Every consumer now reads by attribute name, and the one construction site
    passes keywords, so the field order carries no meaning of its own.

    Attributes:
        c1: First-order parameter indices, one per dimension.
        c2: Second-order parameter index pairs, empty below ``maxorder=2``.
        c3: Third-order parameter index triples, empty below ``maxorder=3``.
        n1: Number of first-order terms (``= D``).
        n2: Number of second-order terms.
        n3: Number of third-order terms.
        n: Total number of terms, ``n1 + n2 + n3``.
        m1: Number of B-spline basis functions per dimension, ``m + 3``.
        m2: Tensor-product basis size of a second-order term, ``m1 ** 2``.
        m3: Tensor-product basis size of a third-order term, ``m1 ** 3``.
        beta2: Flat index -> ``(i, j)`` lookup for the order-2 tensor
            product, shape ``(m2, 2)``.
        beta3: Flat index -> ``(i, j, k)`` lookup for the order-3 tensor
            product, shape ``(m3, 3)``.
    """

    c1: tuple[int, ...]
    c2: tuple[tuple[int, int], ...]
    c3: tuple[tuple[int, int, int], ...]
    n1: int
    n2: int
    n3: int
    n: int
    m1: int
    m2: int
    m3: int
    beta2: np.ndarray
    beta3: np.ndarray


@lru_cache(maxsize=None)
def _get_hdmr_static_data(D: int, maxorder: int, m: int) -> _HDMRStaticData:
    """Build and cache the HDMR term metadata and basis index tables (host side)."""
    # Enumerate all parameter index combinations up to maxorder.
    # c1: single dimensions, c2: pairs, c3: triples.
    c1 = tuple(range(D))
    c2 = tuple(itertools.combinations(range(D), 2)) if maxorder >= 2 else tuple()
    c3 = tuple(itertools.combinations(range(D), 3)) if maxorder >= 3 else tuple()
    n1 = D
    n2 = len(c2)
    n3 = len(c3)
    # Total number of HDMR component functions across all active orders.
    n = n1 + n2 + n3
    # m1 = basis functions per dimension (m intervals + 3 for cubic B-spline
    # boundary support); m2, m3 = tensor-product basis sizes for orders 2, 3.
    m1 = m + 3
    m2 = m1**2
    m3 = m1**3
    # beta tables enumerate all multi-index pairs/triples into the 1-D basis.
    # They turn the tensor product into a flat index -> (i, j[, k]) lookup
    # so the basis can be built via gather + elementwise multiply.
    beta2 = (
        np.asarray(list(itertools.product(range(m1), repeat=2)), dtype=np.int32)
        if n2 > 0
        else np.zeros((0, 2), dtype=np.int32)
    )
    beta3 = (
        np.asarray(list(itertools.product(range(m1), repeat=3)), dtype=np.int32)
        if n3 > 0
        else np.zeros((0, 3), dtype=np.int32)
    )
    return _HDMRStaticData(
        c1=c1,
        c2=c2,
        c3=c3,
        n1=n1,
        n2=n2,
        n3=n3,
        n=n,
        m1=m1,
        m2=m2,
        m3=m3,
        beta2=beta2,
        beta3=beta3,
    )


def _build_term_labels(
    problem: Problem,
    c1: tuple[int, ...],
    c2: tuple[tuple[int, int], ...],
    c3: tuple[tuple[int, int, int], ...],
) -> tuple[str, ...]:
    """Build one human-readable label per HDMR term, in term order."""
    names = problem.names
    labels = [names[i] for i in c1]
    labels += ["/".join(names[i] for i in combo) for combo in c2]
    labels += ["/".join(names[i] for i in combo) for combo in c3]
    return tuple(labels)


def _compute_ST(
    S: Array,
    c2: Array,
    c3: Array,
    n1: int,
) -> Array:
    """Sum S over every term that contains a parameter, giving its total index."""
    # ST_j = S_j + sum_{i<j or j<i} S_{ij} + sum_{i<j<k, j in {i,j,k}} S_{ijk}
    # i.e. the total for parameter j is its first-order term plus every
    # interaction term (2nd and 3rd order) that contains j.
    ST = S[..., :n1]  # First-order terms map 1:1 to parameters.

    # Scatter-add each 2nd-order term S_{ab} to both parameters a and b.
    n2 = c2.shape[0]
    S2 = S[..., n1 : n1 + n2]
    ST = ST.at[..., c2[:, 0]].add(S2)
    ST = ST.at[..., c2[:, 1]].add(S2)

    # Scatter-add each 3rd-order term S_{abc} to all three participating params.
    S3 = S[..., n1 + n2 :]
    ST = ST.at[..., c3[:, 0]].add(S3)
    ST = ST.at[..., c3[:, 1]].add(S3)
    ST = ST.at[..., c3[:, 2]].add(S3)

    return ST


def _reshape_emulator_value(value: Array, T: int, K_out: int, layout: YLayout) -> Array:
    """Reshape flattened per-output emulator state back to the analyzed layout."""
    value = value.reshape((T, K_out) + value.shape[1:])
    return layout.squeeze(value, n_trailing=value.ndim - 2)


def _warn_correlated_index_reading(problem: Problem) -> None:
    """Warn once that ST and S1 carry the SCSA reading on a correlated problem.

    HDMR accepts correlated inputs, and the ANCOVA split stays valid. The two
    aggregate fields are the risk. ``ST`` is the SCSA total of Li et al.
    (2010) Section 2.2.3, not a Sobol total. ``S1`` is a structural share
    only. The paper itself invites the confusion: it reuses the symbol
    ``S_Ti`` for both this term-membership sum and the classical
    conditional-variance total of its Eq. (4). Both are valid published
    quantities that are easy to misread, so this warns rather than raises.
    Independent problems stay silent.

    Args:
        problem: Problem definition for the analysis; nothing is emitted
            unless it declares a dependence structure.
    """
    if not problem.has_correlated_inputs:
        return
    import warnings

    warnings.warn(
        "jaxgsa.hdmr: problem.correlation declares dependent inputs. HDMRResult.ST "
        "is then the SCSA total, sum over every term u containing i of "
        "(Sa_u + Sb_u) (Li et al. 2010, Section 2.2.3; the SALib HDMR "
        "convention). It is not a Sobol total-order index: it can be "
        "negative, it is not bounded in [0, 1], and it does not measure the "
        "expected variance reduction from fixing a parameter, so do not use "
        "it to decide a parameter can be fixed. Li et al. also require the "
        "per-term S values to sum to about 1 for the totals to be reliable "
        "(their Eq. 24) -- check result.S.sum(). HDMRResult.S1 is likewise "
        "the structural share only, not the Sobol first-order index. For a "
        "conditional-variance total under dependence use jaxgsa.kucherenko "
        "(ST) or jaxgsa.vkoga (S_TU). The per-term Sa/Sb/S fields are "
        "unaffected.",
        stacklevel=3,
        category=JaxgsaWarning,
    )


def analyze(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    maxorder: int = 2,
    maxiter: int = 100,
    m: int = 2,
    lambdax: float = 0.01,
    slice_chunk_size: int | None = None,
    batch_size: int | None = None,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> HDMRResult:
    """Compute sensitivity indices via RS-HDMR (public entry point).

    Validates ``(X, Y)``, warns once about any zero-variance output slice, then
    delegates to :func:`_analyze_hdmr_core`. See that function for the full
    parameter and return documentation.

    Correlated inputs are supported, so a declared ``problem.correlation`` is
    welcome. The ANCOVA decomposition splits each term's variance share into a
    structural part (``Sa``) and a correlation-induced part (``Sb``). ``Sb``
    doubles as a correlation diagnostic, and
    ``result.shapley(include_correlative=True)`` folds it into the Shapley
    allocation.

    Args:
        n_bootstrap: Number of bootstrap resamples for confidence intervals
            on ``Sa``, ``Sb``, ``S`` and ``ST``. ``0`` (default) disables
            them. The resampling unit is one row of ``(X, Y)``.

            This is the expensive kind of bootstrap. Every replicate refits
            the whole expansion — fresh B-spline bases, a fresh backfitting
            solve, a fresh F-test — so the cost is an order of magnitude
            above the row resample of a method such as ``jaxgsa.pawn``, which
            only re-reduces numbers it already has. Twenty replicates cost
            about twenty fits. That is why the default is ``0``.

            The interval measures the sampling variability of ``(X, Y)``
            propagated through the fit. It does not measure the surrogate's
            truncation error: every replicate uses the same basis and the
            same maximum order, so a systematic misfit stays inside every
            interval. ``rmse`` and ``S.sum()`` are what report that.
        conf_level: Two-sided confidence level for the intervals.
        ci_method: How the interval endpoints are formed. ``"quantile"``
            (default) reads them off the empirical bootstrap distribution.
            ``"gaussian"`` centres them on the point estimate and takes
            ``+/- z * sd`` of the draws.
        key: A ``jax.random`` key for the bootstrap resampling. Required when
            ``n_bootstrap > 0``. Pass ``jax.random.key(0)`` if you have an
            integer seed.
        on_invalid: What to do about non-finite values in ``X`` or ``Y``. One
            row is one unit here, so ``"drop"`` removes the affected
            ``(X, Y)`` pairs and fits on the rest. See
            :mod:`jaxgsa._core.invalid`. Every other argument is documented on
            :func:`_analyze_hdmr_core`.
        verbose: If ``True`` (default), print a short summary to stdout: the
            problem and the data, the wall-clock timing, and the top
            parameters by ``ST``. Pass ``False`` for a silent run.
        keep_replicates: Retain the per-replicate index arrays on
            ``result.ci``. Off by default: the draws are large.

    Raises:
        ValueError: If ``X`` or ``Y`` violates the shared shape contract, or
            ``problem`` has categorical parameters. The B-spline component
            functions need an orderable axis, which an unordered level code
            does not give. Also if ``on_invalid`` is not one of the three
            policies, ``on_invalid="raise"`` (the default) and ``X`` or
            ``Y`` holds a non-finite value, or ``n_bootstrap > 0`` without a
            ``key``. :func:`_analyze_hdmr_core` raises for the remaining
            argument checks.
    """
    from jaxgsa.hdmr import SPEC

    # The preamble runs in this public wrapper only. Callers routing through
    # `_analyze_hdmr_core` (Shapley) must not have the policy applied, or the
    # zero-variance warning issued, a second time.
    ctx = prepare(
        SPEC,
        problem,
        Y,
        X=X,
        on_invalid=on_invalid,
        checks=(
            *_fit_scalar_checks(maxiter, m, lambdax),
            at_least("n_bootstrap", n_bootstrap, 0),
            one_of("ci_method", ci_method, ("quantile", "gaussian")),
            in_open_interval("conf_level", conf_level, 0.0, 1.0),
        ),
        min_kept=_MIN_ROWS,
    )
    if n_bootstrap > 0 and key is None:
        raise ValueError("key is required when n_bootstrap > 0")
    X, Y, invalid = ctx.inputs, ctx.Y, ctx.invalid
    # ST and S1 change meaning under dependence; say so once per analyze call,
    # in the public wrapper only, so Shapley routing through the core stays
    # quiet.
    _warn_correlated_index_reading(problem)
    t0 = _verbose.tic()
    result = _analyze_hdmr_core(
        problem,
        X,
        Y,
        maxorder=maxorder,
        maxiter=maxiter,
        m=m,
        lambdax=lambdax,
        slice_chunk_size=slice_chunk_size,
        batch_size=batch_size,
        invalid=invalid,
        n_bootstrap=n_bootstrap,
        conf_level=conf_level,
        ci_method=ci_method,
        key=key,
        keep_replicates=keep_replicates,
    )

    if verbose:
        elapsed = _verbose.stop(t0, result.ST)
        _, T, K = ctx.Y3.shape
        slice_note = (
            f"slice_chunk_size: {slice_chunk_size} (user-set)"
            if slice_chunk_size is not None
            else "slice_chunk_size: auto (resolved from the memory budget)"
        )
        batch_note = (
            f"batch_size: {batch_size} (user-set)"
            if batch_size is not None
            else "batch_size: auto (resolved from the memory budget)"
        )
        _verbose.analysis_summary(
            method="jaxgsa.hdmr.analyze",
            problem=problem,
            n_runs=int(ctx.Y.shape[0]),
            T=T,
            K=K,
            invalid=invalid,
            timings=[("fit + estimator (first call, includes compile)", elapsed)],
            notes=[f"maxorder: {maxorder}", slice_note, batch_note],
            index_name="ST",
            values=result.ST,
            conf=result.ST_conf,
        )
    return result


class _HDMRCore(NamedTuple):
    """One HDMR fit, as arrays, before any result is built.

    The index fields are in canonical ``(T, K, ...)`` layout: the caller
    squeezes them back to its own rank. The emulator fields are still in the
    flat ``(T*K, ...)`` layout the kernel produced, because only a caller
    that wants a surrogate pays for reshaping them.

    Attributes:
        Sa: Structural variance fraction per term, shape ``(T, K, n_terms)``.
        Sb: Correlative variance fraction per term, same shape.
        S: ``Sa + Sb``, same shape.
        ST: SCSA total per parameter, shape ``(T, K, D)``.
        select: F-test significance count per term, shape ``(n_terms,)``.
        rmse: Per-slice fit RMSE, shape ``(T*K,)``.
        C1: First-order component coefficients, flat slice axis leading.
        C2: Second-order coefficients, or ``None`` below ``maxorder=2``.
        C3: Third-order coefficients, or ``None`` below ``maxorder=3``.
        f0: Grand mean per slice.
        maxorder: The expansion order actually fitted, clamped to ``D``.
        streamed: Whether the sample axis was split into row batches.
        c2: Parameter index pairs, in term order.
        c3: Parameter index triples, in term order.
        layout: Which rank the caller's ``Y`` had, for the squeeze back.
    """

    Sa: Array
    Sb: Array
    S: Array
    ST: Array
    select: Array
    rmse: Array
    C1: Array
    C2: Array | None
    C3: Array | None
    f0: Array
    maxorder: int
    streamed: bool
    c2: tuple[tuple[int, int], ...]
    c3: tuple[tuple[int, int, int], ...]
    layout: YLayout


def _hdmr_core(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    maxorder: int = 2,
    maxiter: int = 100,
    m: int = 2,
    lambdax: float = 0.01,
    slice_chunk_size: int | None = None,
    batch_size: int | None = None,
) -> _HDMRCore:
    """Fit RS-HDMR and return the arrays, with no result and no diagnostics.

    This is the whole estimator: the CDF transform, the B-spline bases, the
    backfitting solve, the F-test and the ANCOVA split. What it does not do
    is anything that needs the host — no non-finite check, no zero-variance
    warning, no clamp warning, no term labels, no result class. Every branch
    it takes is on a shape or on a Python scalar, so it traces.

    There is one fit path, and it chunks both axes: row batches over the
    sample axis, slice chunks over the output-slice axis. ``batch_size`` and
    ``slice_chunk_size`` size them, and both compose. The old "in-memory"
    fit is now the degenerate corner ``batch_size = N`` and
    ``slice_chunk_size = T*K``, which is what the automatic sizes resolve to
    whenever the whole fit fits the memory budget. Both sizes are Python
    values, so the loop bounds are fixed at trace time and nothing reads a
    sample value on the host. See :mod:`jaxgsa.hdmr._fit`.

    Args:
        problem: Parameter names and distributions.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, at any of the three accepted ranks.
        maxorder: Requested maximum expansion order; clamped to ``D``.
        maxiter: Maximum backfitting iterations.
        m: B-spline intervals per dimension.
        lambdax: Tikhonov regularization strength.
        slice_chunk_size: Output slices per vmap batch, or ``None``.
        batch_size: Rows per fit batch, or ``None``.

    Returns:
        The :class:`_HDMRCore` bundle.

    Raises:
        ValueError: If ``N < 300``, ``maxorder`` is not 1/2/3, ``maxiter``
            or ``m`` is below 1, ``lambdax`` is negative, or a chunk size is
            below 1. All are decided on Python scalars. See
            :func:`_fit_scalar_checks` for why those bounds.
    """
    N, D = X.shape
    # B-spline regression with backfitting needs a reasonable sample size
    # to avoid overfitting; 300 is a practical lower bound.
    if N < 300:
        raise ValueError(f"Need at least 300 samples, got {N}")
    if maxorder not in (1, 2, 3):
        raise ValueError(f"maxorder must be 1, 2, or 3, got {maxorder}")
    check_scalars(_fit_scalar_checks(maxiter, m, lambdax))
    # Clamped silently: there is no more expansion order available than there
    # are parameters. `analyze` warns about it; this is the traceable half,
    # and the clamped value comes back on `_HDMRCore.maxorder`.
    maxorder = min(maxorder, D)
    if slice_chunk_size is not None and slice_chunk_size < 1:
        raise ValueError(f"slice_chunk_size must be >= 1, got {slice_chunk_size}")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    lambdax = float(lambdax)

    # Build term metadata (cached on host; only computed once per D/maxorder/m).
    static = _get_hdmr_static_data(D, maxorder, m)
    c2, c3 = static.c2, static.c3
    n1, n2, n3, n = static.n1, static.n2, static.n3, static.n
    m1, m2, m3 = static.m1, static.m2, static.m3
    beta2_host, beta3_host = static.beta2, static.beta3
    # Transfer index tables to device; empty arrays for inactive orders.
    c2_idx = jnp.asarray(c2, dtype=int) if n2 > 0 else jnp.zeros((0, 2), dtype=int)
    c3_idx = jnp.asarray(c3, dtype=int) if n3 > 0 else jnp.zeros((0, 3), dtype=int)
    beta2 = jnp.asarray(beta2_host, dtype=int)
    beta3 = jnp.asarray(beta3_host, dtype=int)

    # CDF transform: maps each dimension's marginal to U[0,1] so the B-spline
    # basis operates on a uniform domain regardless of the original distribution.
    X_n = cdf_to_unit_interval(X, problem)

    # F critical values at alpha=0.95, precomputed outside JIT to avoid
    # re-running the bisection solver on every vmap lane.
    f_crits = _compute_f_crits(0.95, m1, m2, m3, N)

    # Promote Y to canonical (N, T, K) layout; track which dims were singleton
    # so the output arrays can be squeezed back to the user's original shape.
    Y_3d, layout = _prepare_Y(Y)
    _, T, K_out = Y_3d.shape

    total = T * K_out

    # (N, T, K) -> (N, T*K): keep the sample axis leading so row batches are
    # contiguous slices; column t*K + k is output slice (t, k).
    Y_nl = Y_3d.reshape(N, total)
    (
        sa_cat,
        sb_cat,
        s_cat,
        select_lanes,
        rmse_cat,
        c1_cat,
        c2_cat,
        c3_cat,
        f0_cat,
        streamed,
    ) = _fit_hdmr(
        X_n,
        Y_nl,
        m=m,
        maxorder=maxorder,
        maxiter=maxiter,
        lambdax=lambdax,
        f_crits=f_crits,
        c2_idx=c2_idx,
        c3_idx=c3_idx,
        beta2=beta2,
        beta3=beta3,
        n1=n1,
        n2=n2,
        n3=n3,
        n=n,
        m1=m1,
        m2=m2,
        m3=m3,
        batch_size=batch_size,
        slice_chunk_size=slice_chunk_size,
    )
    select_sum = jnp.sum(select_lanes, axis=0)

    # Reassemble into the full (T, K, n_terms) index arrays.
    Sa_out = sa_cat.reshape(T, K_out, n)
    Sb_out = sb_cat.reshape(T, K_out, n)
    S_out = s_cat.reshape(T, K_out, n)
    # Aggregate per-term S into per-parameter total-order indices.
    ST_out = _compute_ST(S_out, c2_idx, c3_idx, n1)

    return _HDMRCore(
        Sa=Sa_out,
        Sb=Sb_out,
        S=S_out,
        ST=ST_out,
        select=select_sum,
        rmse=rmse_cat,
        C1=c1_cat,
        C2=c2_cat,
        C3=c3_cat,
        f0=f0_cat,
        maxorder=maxorder,
        streamed=streamed,
        c2=tuple(c2),
        c3=tuple(c3),
        layout=layout,
    )


_INDEX_FIELDS = ("Sa", "Sb", "S", "ST")
"""The index arrays an interval is reported for, in ``_HDMRCore`` order."""


def indices(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    maxorder: int = 2,
    maxiter: int = 100,
    m: int = 2,
    lambdax: float = 0.01,
    slice_chunk_size: int | None = None,
    batch_size: int | None = None,
) -> tuple[Array, ...]:
    """Compute HDMR indices as plain arrays, with no diagnostics.

    This is the transformable core of :func:`analyze`. It fits the same
    B-spline expansion on the same data and returns the same numbers, but it
    does nothing else: no non-finite check, no zero-variance warning, no
    clamp warning, no SCSA reinterpretation warning, no term labels, no
    F-test counts, no fitted surrogate, no :class:`jaxgsa.hdmr.HDMRResult`,
    and no read of any array value on the host. So it composes with
    ``jax.jit``, ``jax.vmap`` and ``jax.jacfwd``, which :func:`analyze`
    cannot, because a policy decision needs a concrete value and a tracer has
    none.

    Use :func:`analyze` for ordinary analysis. Nothing here checks the
    inputs, so a single NaN reaches the regression and turns every index into
    NaN without a word, and nothing here says that ``ST`` and ``Sa[:D]``
    change meaning under a declared correlation. That reading is in
    :func:`analyze`'s warning and in :class:`jaxgsa.hdmr.HDMRResult`.

    Tier T4 (behavioural contract): the returned arrays must equal the
    corresponding fields of ``analyze``'s result on clean outputs, and the
    function must survive ``jit``, ``vmap`` and ``jit(jacfwd(...))``. Checked
    in ``tests/test_hdmr_gradients.py``.

    Note:
        **Reverse mode does not work, forward mode does.** The backfitting
        solve stops early once the coefficients settle, and it expresses that
        as a ``lax.while_loop``, which JAX cannot differentiate in reverse.
        ``jacfwd`` works and is the right tool anyway here: the Jacobian this
        core has is tall, one row per index against ``N`` outputs. Making
        ``jacrev`` work means replacing the loop with a fixed-trip
        ``lax.scan`` of ``maxiter`` iterations, which costs every fit the
        iterations the early stop currently saves. That is a real trade, not
        an oversight, so it is not made here.

    Args:
        problem: Parameter names and distributions.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)`` or ``(N, T, K)``, as
            :func:`analyze` accepts them.
        maxorder: Maximum expansion order, clamped to ``D`` silently rather
            than with the warning :func:`analyze` gives.
        maxiter: Maximum backfitting iterations.
        m: B-spline intervals per dimension.
        lambdax: Tikhonov regularization strength.
        slice_chunk_size: Output slices per vmap batch, or ``None``.
        batch_size: Rows per fit batch, or ``None``.

    Returns:
        ``(Sa, Sb, S, ST)``, at the shapes ``analyze`` reports for this
        ``Y``. ``Sa``/``Sb``/``S`` are per term and ``ST`` is per parameter.

    Raises:
        ValueError: If ``N < 300``, ``maxorder`` is not 1/2/3, ``maxiter``
            or ``m`` is below 1, ``lambdax`` is negative, a chunk size is
            below 1, or ``problem`` has a categorical parameter, whose step
            CDF has no unit-interval image.
    """
    core = _hdmr_core(
        problem,
        X,
        Y,
        maxorder=maxorder,
        maxiter=maxiter,
        m=m,
        lambdax=lambdax,
        slice_chunk_size=slice_chunk_size,
        batch_size=batch_size,
    )
    return tuple(core.layout.squeeze(getattr(core, name)) for name in _INDEX_FIELDS)


def _bootstrap_indices(
    problem: Problem,
    X: Array,
    Y: Array,
    key: Array,
    *,
    n_bootstrap: int,
    maxorder: int,
    maxiter: int,
    m: int,
    lambdax: float,
    slice_chunk_size: int | None,
    batch_size: int | None,
) -> dict[str, Array]:
    """Refit the expansion on ``n_bootstrap`` row resamples.

    One replicate is a full refit: the rows are drawn with replacement, the
    B-spline bases are rebuilt on them, the backfitting runs again and the
    F-test is redone. No part of an HDMR index survives a resample without
    that, so the replicates run as a Python loop over independent fits and
    the cost is ``n_bootstrap`` fits — an order of magnitude above a method
    that only re-reduces per-row numbers it already holds.

    The row draw is shared by every output slice, exactly as the point
    estimate shares one basis across slices.

    Args:
        problem: Parameter names and distributions.
        X: Input samples with invalid rows already removed, shape ``(N, D)``.
        Y: Outputs at the caller's own rank.
        key: A ``jax.random`` key, split once per replicate.
        n_bootstrap: Number of replicates.
        maxorder: Requested maximum expansion order.
        maxiter: Maximum backfitting iterations.
        m: B-spline intervals per dimension.
        lambdax: Tikhonov regularization strength.
        slice_chunk_size: Output slices per vmap batch, or ``None``.
        batch_size: Rows per fit batch, or ``None``.

    Returns:
        One stacked draw array per index field, each with a leading axis of
        length ``n_bootstrap`` followed by the canonical ``(T, K, ...)``
        shape.
    """
    N = X.shape[0]
    draws: dict[str, list[Array]] = {name: [] for name in _INDEX_FIELDS}
    for _ in range(n_bootstrap):
        key, subkey = jax.random.split(key)
        rows = jax.random.choice(subkey, N, shape=(N,), replace=True)
        replicate = _hdmr_core(
            problem,
            X[rows],
            Y[rows],
            maxorder=maxorder,
            maxiter=maxiter,
            m=m,
            lambdax=lambdax,
            slice_chunk_size=slice_chunk_size,
            batch_size=batch_size,
        )
        for name in _INDEX_FIELDS:
            draws[name].append(getattr(replicate, name))
    return {name: jnp.stack(parts) for name, parts in draws.items()}


def _analyze_hdmr_core(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    maxorder: int = 2,
    maxiter: int = 100,
    m: int = 2,
    lambdax: float = 0.01,
    slice_chunk_size: int | None = None,
    batch_size: int | None = None,
    invalid: InvalidReport,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    keep_replicates: bool = False,
) -> HDMRResult:
    """Fit RS-HDMR on an already-canonical Y (no re-validation, no warn).

    Compute sensitivity indices via RS-HDMR with B-spline surrogate modelling.

    Works with any set of (X, Y) pairs. No structured sampling is required, so
    it suits existing datasets and expensive models where Sobol/eFAST sampling
    schemes are unaffordable. B-spline regression decomposes the input-output
    relationship into hierarchical component functions: one per parameter, one
    per parameter pair, and so on. The ANCOVA-based sensitivity indices then
    come from the fitted components. Unlike pure Sobol
    estimators, the ANCOVA split into structural (Sa) and correlative (Sb)
    parts remains meaningful when inputs are correlated.

    Args:
        problem: Parameter names and distributions.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)`` / ``(N, K)`` / ``(N, T, K)``. A 2-D
            array is read as ``(N, K)`` unless ``problem.output_names`` has
            exactly one entry. In that case the columns are T timepoints of
            that single output.
        maxorder: Maximum HDMR expansion order (1, 2, or 3), which is the
            largest interaction size modelled. Order 2 (default) captures
            pairwise interactions. Order 3 adds triples, but the term count
            and the fit cost grow combinatorially. Clamped to D (with a
            warning) when D < maxorder.
        maxiter: Maximum backfitting iterations for the first-order terms.
            The default rarely needs raising, because iteration stops early
            once the coefficients stop changing.
        m: Number of B-spline intervals per dimension (basis size m + 3).
            Larger m resolves sharper features of the component functions.
            It also multiplies the coefficient count (the per-term basis grows
            as (m+3)^order) and needs more samples to avoid overfitting.
        lambdax: Tikhonov regularization strength. Increase it for noisy Y or
            small N, which gives smoother and more stable components.
            Decrease it if genuine sharp features are being oversmoothed.
        slice_chunk_size: Maximum number of (T, K) output slices fitted per
            vmap. It caps peak device memory for large T*K, and a smaller
            value trades speed for memory. ``None`` (default) derives the
            chunk size from the active memory budget
            (``jaxgsa.config.set_memory_budget``), because the per-slice
            transient scales with ``batch_size * n_terms``. It composes with
            ``batch_size``: there is one fit path and it honours both.
        batch_size: Rows of ``X``/``Y`` processed per batch during the fit
            (the package-wide ``batch_size`` convention). ``None`` (default)
            derives one from the active memory budget, which resolves to a
            single batch whenever the whole fit fits; an explicit int forces
            that many rows per batch. Row batching is exact: it accumulates
            the same Gram matrices and moments, solves the same regressions
            and runs the same F-test, so results differ only at the level of
            float32 summation order.
        invalid: The non-finite report the public :func:`analyze` wrapper
            produced, carried onto the result. It is a required argument
            rather than a default so that no caller can construct an
            ``HDMRResult`` that silently claims the check ran. The check
            itself is deliberately not repeated here: applying it twice would
            warn twice for one user call.
        n_bootstrap: Number of row resamples behind the confidence intervals,
            or ``0`` for none. Each one is a full refit; see :func:`analyze`.
        conf_level: Two-sided confidence level for the intervals.
        ci_method: Endpoint rule, ``"quantile"`` or ``"gaussian"``.
        key: A ``jax.random`` key. :func:`analyze` has already refused a
            missing one when ``n_bootstrap > 0``.
        keep_replicates: Retain the per-replicate arrays on ``result.ci``.

    Returns:
        HDMRResult with per-term indices Sa, Sb, S, per-parameter ST,
        human-readable term labels, F-test selection counts, the fitted
        surrogate state and its RMSE.

        ``ST`` is the SCSA total: ``ST_i = sum over u containing i of
        (Sa_u + Sb_u)``. Section 2.2.3 of Li et al. (2010) defines it from
        the per-term indices of their Eqs. (19)-(22). It is the same
        convention as SALib's HDMR. With independent inputs the
        correlative shares vanish and it reduces to the ordinary Sobol
        total-order index.

    Warning:
        With correlated inputs, ``ST`` is not a Sobol total-order index. It
        can be negative. It is not bounded in ``[0, 1]``. It does not measure
        the expected reduction of output variance from fixing a parameter. Do
        not use it as a fixing criterion. It is also not comparable with the
        ``ST`` of ``jaxgsa.kucherenko`` or the ``S_TU`` of ``jaxgsa.vkoga``.
        Use one of those when you need a genuine total under dependence.
        ``S1`` is the structural share only, not the Sobol first-order index.
        The per-term ``Sa``, ``Sb`` and ``S`` fields keep their ANCOVA
        meaning. A correlated problem emits one ``JaxgsaWarning`` that says
        this.

    Raises:
        ValueError: If ``N < 300``, ``maxorder`` is not 1/2/3, an explicit
            ``slice_chunk_size < 1`` or ``batch_size < 1`` is passed, or
            ``problem`` has categorical parameters. The B-spline component
            functions need an orderable axis, which an unordered level code
            does not give. The gate on categorical parameters runs in the
            public :func:`analyze` wrapper.

    Note:
        The fit materializes B-spline basis tensors of roughly
        ``batch_size * (m+3)^2 * n2`` floats at order 2 and
        ``batch_size * (m+3)^3 * n3`` floats at order 3 (``n2``/``n3`` =
        number of parameter pairs/triples). That footprint is what
        ``batch_size`` bounds; ``slice_chunk_size`` bounds the per-slice
        statistics transient, ``batch_size * slice_chunk_size * n_terms``.
        Both default to the whole axis when the memory budget allows it.
    """
    core = _hdmr_core(
        problem,
        X,
        Y,
        maxorder=maxorder,
        maxiter=maxiter,
        m=m,
        lambdax=lambdax,
        slice_chunk_size=slice_chunk_size,
        batch_size=batch_size,
    )
    if core.maxorder < maxorder:
        import warnings

        warnings.warn(
            f"jaxgsa.hdmr: maxorder clamped to {core.maxorder} "
            f"(need D >= maxorder, got D={X.shape[1]})",
            stacklevel=2,
            category=JaxgsaWarning,
        )
    layout = core.layout
    T, K_out = core.Sa.shape[:2]
    term_labels = _build_term_labels(problem, tuple(range(problem.num_vars)), core.c2, core.c3)

    # Bundle all fitted state needed to reconstruct predictions at new points.
    fit_state: _HDMRFit = {
        "C1": _reshape_emulator_value(core.C1, T, K_out, layout),
        "C2": None if core.C2 is None else _reshape_emulator_value(core.C2, T, K_out, layout),
        "C3": None if core.C3 is None else _reshape_emulator_value(core.C3, T, K_out, layout),
        "f0": _reshape_emulator_value(core.f0, T, K_out, layout),
        "m": m,
        "maxorder": core.maxorder,
    }

    conf: dict[str, Array | None] = {f"{name}_conf": None for name in _INDEX_FIELDS}
    ci: CIInfo | None = None
    if n_bootstrap > 0:
        assert key is not None  # checked in analyze, before the fit ran
        draws = _bootstrap_indices(
            problem,
            X,
            Y,
            key,
            n_bootstrap=n_bootstrap,
            maxorder=maxorder,
            maxiter=maxiter,
            m=m,
            lambdax=lambdax,
            slice_chunk_size=slice_chunk_size,
            batch_size=batch_size,
        )
        for name in _INDEX_FIELDS:
            conf[f"{name}_conf"] = layout.squeeze(
                jnp.stack(
                    _bootstrap_ci_endpoints(
                        getattr(core, name),
                        draws[name],
                        conf_level=conf_level,
                        ci_method=ci_method,
                    )
                )
            )
        ci = CIInfo(
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            # The leading replicate axis survives the squeeze, which addresses
            # the T/K axes from the end.
            replicates={name: layout.squeeze(draw) for name, draw in draws.items()}
            if keep_replicates
            else None,
        )

    Sa_out, Sb_out, S_out, ST_out = (
        layout.squeeze(a) for a in (core.Sa, core.Sb, core.S, core.ST)
    )

    return HDMRResult(
        Sa=Sa_out,
        Sb=Sb_out,
        S=S_out,
        ST=ST_out,
        problem=problem,
        terms=term_labels,
        invalid=invalid,
        _fit=fit_state,
        select=core.select,
        rmse=_reshape_emulator_value(core.rmse, T, K_out, layout),
        streamed=core.streamed,
        _c2=core.c2,
        _c3=core.c3,
        Sa_conf=conf["Sa_conf"],
        Sb_conf=conf["Sb_conf"],
        S_conf=conf["S_conf"],
        ST_conf=conf["ST_conf"],
        ci=ci,
    )


def _emulator_contract(B: Array, C: Array) -> Array:
    """Contract basis B (N, m, j) with coefficients C, summing over terms.

    Dispatches on C.ndim to handle scalar (m, j), multi-output (K, m, j),
    and time-series (T, K, m, j) coefficient layouts.
    """
    # Dispatch on C.ndim to handle different output layouts:
    #   2D (m, j):      scalar output   -> einsum("rmj,mj->rj")   -> sum over j terms
    #   3D (K, m, j):   multi-output    -> einsum("rmj,kmj->rkj") -> sum over j terms
    #   4D (T, K, m, j): time-series    -> einsum("rmj,tkmj->rtkj") -> sum over j terms
    if C.ndim == 2:
        return jnp.sum(jnp.einsum("rmj,mj->rj", B, C), axis=1)
    if C.ndim == 3:
        return jnp.sum(jnp.einsum("rmj,kmj->rkj", B, C), axis=2)
    return jnp.sum(jnp.einsum("rmj,tkmj->rtkj", B, C), axis=3)


def _hdmr_predict_plan(result: HDMRResult, X_new: Array) -> _PredictPlan:
    """Build the prediction plan behind :meth:`HDMRResult.predict`.

    Applies the same CDF -> [0,1] transform used during fitting to the
    (already validated) inputs. It then packages the per-row transient cost
    together with a kernel that reconstructs ``f0 + sum of component
    functions``. The shared template in
    :class:`jaxgsa._core.surrogate.SurrogateResult` runs that kernel in row
    batches sized against a transient-memory budget. The fit runs on the
    output scale the caller passed in, so predictions land on that scale with
    no inverse transform.

    Note: The kernel is not JIT-compatible because ``HDMRResult`` is not a
    registered JAX pytree type.

    Raises:
        ValueError: If ``result`` carries no fitted surrogate state.
    """
    em = result._fit
    if em is None:
        raise ValueError(
            "HDMRResult carries no fitted surrogate state; "
            "obtain results from jaxgsa.hdmr.analyze()"
        )

    maxorder = em["maxorder"]
    C1 = em["C1"]
    f0 = em["f0"]

    # Apply the same CDF -> [0,1] transform used during fitting.
    X_n = cdf_to_unit_interval(X_new, result.problem)

    # Hoist the static basis index tables out of the per-batch path. Each
    # higher_orders entry is (basis builder, term index table, tensor-product
    # index table, coefficients) for one interaction order. The loop also
    # tracks the transient footprint per prediction row: each basis tensor,
    # its build temporaries (2 gathered factors for B2, 3 for B3), and the
    # pre-sum einsum intermediate of _emulator_contract, (slices, n_terms).
    m1 = C1.shape[-2]
    D = C1.shape[-1]
    slices = math.prod(C1.shape[:-2])
    higher_orders: list[tuple[Callable[[Array, Array, Array], Array], Array, Array, Array]] = []
    elems_per_row = m1 * D + slices * D
    if maxorder >= 2:
        static = _get_hdmr_static_data(result.problem.num_vars, maxorder, em["m"])
        beta2_host, beta3_host = static.beta2, static.beta3
        C2 = em["C2"]
        if C2 is not None:
            c2 = jnp.asarray(result._c2, dtype=int)
            higher_orders.append((_build_B2, c2, jnp.asarray(beta2_host, dtype=int), C2))
            elems_per_row += (3 * m1**2 + slices) * len(result._c2)
        C3 = em["C3"]
        if maxorder >= 3 and C3 is not None:
            c3 = jnp.asarray(result._c3, dtype=int)
            higher_orders.append((_build_B3, c3, jnp.asarray(beta3_host, dtype=int), C3))
            elems_per_row += (4 * m1**3 + slices) * len(result._c3)

    def _predict(X_chunk: Array) -> Array:
        # Reconstruct prediction as f0 + sum of component functions.
        # Start with first-order: sum_j B1_j @ C1_j.
        B1 = _build_B1(X_chunk, em["m"])  # (batch, m1, D)
        Y_total = _emulator_contract(B1, C1)
        for build, c_idx, beta, C in higher_orders:
            Y_total = Y_total + _emulator_contract(build(B1, c_idx, beta), C)
        # Add the grand mean to recover the full surrogate prediction. It is
        # elementwise per row, so doing it per batch matches single-shot.
        return Y_total + f0

    return _PredictPlan(X=X_n, bytes_per_row=X_n.dtype.itemsize * elems_per_row, kernel=_predict)
