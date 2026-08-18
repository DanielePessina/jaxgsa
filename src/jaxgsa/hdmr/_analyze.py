"""RS-HDMR (Random Sampling High-Dimensional Model Representation) analysis.

Computes ANCOVA-based sensitivity indices from arbitrary ``(X, Y)`` pairs
using B-spline surrogate modelling. The result keeps the fitted surrogate, so
it can also predict at new points and give Shapley effects.
"""

import itertools
import math
from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.batching import get_memory_budget
from jaxgsa._core.invalid import (
    InvalidReport,
    InvalidUnit,
    OnInvalid,
    check_invalid,
    resolve_policy,
)
from jaxgsa._core.surrogate import _PredictPlan
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa._core.validation import (
    _prenormalize_outputs,
    _prepare_Y,
    _squeeze_output_axes,
    _validate_xy_inputs,
    _warn_zero_variance_slices,
)
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.hdmr._engine import (
    _build_B1,
    _build_B2,
    _build_B3,
    _compute_f_crits,
    _make_hdmr_kernel,
)
from jaxgsa.hdmr._result import HDMRResult, _HDMRFit
from jaxgsa.hdmr._stream import _fit_hdmr_streamed, _full_fit_bytes
from jaxgsa.problem import Problem

# Fewest rows the B-spline backfitting fit can run on. It is the module's own
# documented minimum, re-used as the floor for ``on_invalid="drop"`` so that
# dropping past it refuses in the policy's words rather than in the generic
# "Need at least 300 samples" check further down.
_MIN_ROWS = 300


class _HDMRStaticData(NamedTuple):
    """Term metadata and basis index tables for one ``(D, maxorder, m)``.

    Every field is host-side and shared by all output slices. The names
    matter: the values used to travel as a bare twelve-tuple unpacked
    positionally at three call sites, two of them with blind placeholders, so
    reordering the fields produced silently wrong indices instead of an error.

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
    return _HDMRStaticData(c1, c2, c3, n1, n2, n3, n, m1, m2, m3, beta2, beta3)


@lru_cache(maxsize=None)
def _get_batched_hdmr_kernel(
    D: int,
    maxorder: int,
    m: int,
    maxiter: int,
    lambdax: float,
    N: int,
):
    """Build and cache the jitted, vmapped HDMR fitting kernel."""
    # Caching by (D, maxorder, m, maxiter, lambdax, N) avoids re-tracing the
    # JIT+vmap wrapper when analyze_hdmr is called repeatedly with the same
    # structural parameters but different data.
    sd = _get_hdmr_static_data(D, maxorder, m)
    kernel = _make_hdmr_kernel(
        maxorder,
        sd.m1,
        sd.n1,
        maxiter,
        sd.m2,
        sd.m3,
        sd.n2,
        sd.n3,
        sd.n,
        lambdax,
        N,
    )
    # in_axes=(None, None, None, 0, None): B1/B2/B3 basis matrices and f_crits
    # are shared across all output slices, while Y is batched along axis 0 so
    # each (T*K) output slice gets its own response vector.
    return jax.jit(jax.vmap(kernel, in_axes=(None, None, None, 0, None)))


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


def _squeeze_hdmr(
    Sa: Array,
    Sb: Array,
    S: Array,
    ST: Array,
    squeeze_time: bool,
    squeeze_output: bool,
) -> tuple:
    """Remove singleton T/K dims from HDMR result arrays."""
    return tuple(_squeeze_output_axes(a, squeeze_time, squeeze_output) for a in (Sa, Sb, S, ST))


def _reshape_emulator_value(
    value: Array,
    T: int,
    K_out: int,
    squeeze_time: bool,
    squeeze_output: bool,
) -> Array:
    """Reshape flattened per-output emulator state back to the analyzed layout."""
    value = value.reshape((T, K_out) + value.shape[1:])
    return _squeeze_output_axes(value, squeeze_time, squeeze_output, n_trailing=value.ndim - 2)


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
        "jaxgsa: problem.correlation declares dependent inputs. HDMRResult.ST "
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


def analyze_hdmr(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    prenormalize: bool = False,
    maxorder: int = 2,
    maxiter: int = 100,
    m: int = 2,
    lambdax: float = 0.01,
    slice_chunk_size: int | None = None,
    batch_size: int | None = None,
    on_invalid: OnInvalid = "raise",
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
        on_invalid: What to do about non-finite values in ``X`` or ``Y``. One
            row is one unit here, so ``"drop"`` removes the affected
            ``(X, Y)`` pairs and fits on the rest. See
            :mod:`jaxgsa._core.invalid`. Every other argument is documented on
            :func:`_analyze_hdmr_core`.

    Raises:
        ValueError: If ``X`` or ``Y`` violates the shared shape contract, or
            ``problem`` has categorical parameters. The B-spline component
            functions need an orderable axis, which an unordered level code
            does not give. Also if ``on_invalid`` is not one of the three
            policies, or ``on_invalid="raise"`` (the default) and ``X`` or
            ``Y`` holds a non-finite value.
            :func:`_analyze_hdmr_core` raises for the remaining argument
            checks.
    """
    method = "jaxgsa.hdmr.analyze"
    policy = resolve_policy(on_invalid, method=method, unit=InvalidUnit.ROW)
    X = jnp.asarray(X)
    # HDMR's ANCOVA decomposition separates structural (Sa) from
    # correlation-induced (Sb) variance, so correlated problems are welcome.
    Y = _validate_xy_inputs(problem, X, jnp.asarray(Y), correlation_ok=True, method=method)
    # The non-finite check lives in this public wrapper only, next to the two
    # warnings below and for the same reason: callers routing through
    # `_analyze_hdmr_core` must not have the policy applied a second time.
    # X and Y are checked jointly, so a bad input takes its own output with it
    # and the rows stay aligned.
    keep, invalid = check_invalid(
        policy=policy,
        method=method,
        unit=InvalidUnit.ROW,
        n_units=int(X.shape[0]),
        X=X,
        Y=Y,
        min_kept=_MIN_ROWS,
    )
    if not keep.all():
        mask = jnp.asarray(keep)
        X = X[mask]
        Y = Y[mask]
    # A constant output slice makes every index 0/0 = NaN; warn once up front,
    # in the public wrapper only, so callers routing through the core (Shapley)
    # do not double-warn.
    _warn_zero_variance_slices(_prepare_Y(Y)[0], output_names=problem.output_names)
    # ST and S1 change meaning under dependence; say so once per analyze call,
    # in the public wrapper only, so Shapley routing through the core stays
    # quiet.
    _warn_correlated_index_reading(problem)
    return _analyze_hdmr_core(
        problem,
        X,
        Y,
        prenormalize=prenormalize,
        maxorder=maxorder,
        maxiter=maxiter,
        m=m,
        lambdax=lambdax,
        slice_chunk_size=slice_chunk_size,
        batch_size=batch_size,
        invalid=invalid,
    )


def _analyze_hdmr_core(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    prenormalize: bool = False,
    maxorder: int = 2,
    maxiter: int = 100,
    m: int = 2,
    lambdax: float = 0.01,
    slice_chunk_size: int | None = None,
    batch_size: int | None = None,
    invalid: InvalidReport,
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
        prenormalize: When ``True``, standardize each output slice over the
            sample axis (subtract mean, divide by standard deviation) before
            fitting. That puts disparate output magnitudes on an equal
            numerical footing. The indices are ratios, so they are
            unaffected. Predictions from the returned emulator are still on
            the original output scale. Defaults to ``False``.
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
            vmap batch on the in-memory (non-streamed) path. It caps peak
            device memory for large T*K, and a smaller value trades speed for
            memory. ``None`` (default) derives the chunk size from the active
            memory budget (``jaxgsa.config.set_memory_budget``), because the
            per-slice cost inside the fitting kernel scales with
            ``N * n_terms``. Ignored when the fit streams over rows (see
            ``batch_size``).
        batch_size: Rows of ``X``/``Y`` processed per batch during the fit
            (the package-wide ``batch_size`` convention). ``None`` (default)
            keeps the in-memory fit unless its estimated resident memory
            exceeds the active memory budget, in which case the fit streams
            over auto-sized row batches. That estimate covers the full-N
            B-spline bases plus the per-slice kernel transients: see
            :func:`jaxgsa.hdmr._stream._full_fit_bytes`. An explicit int
            always forces the streamed fit with that many rows per batch.
            Both fit paths solve the same regressions and the same F-test.
            Results differ only at the level of float32 summation order.
        invalid: The non-finite report the public :func:`analyze_hdmr` wrapper
            produced, carried onto the result. It is a required argument
            rather than a default so that no caller can construct an
            ``HDMRResult`` that silently claims the check ran. The check
            itself is deliberately not repeated here: applying it twice would
            warn twice for one user call.

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
            public :func:`analyze_hdmr` wrapper.

    Note:
        On the in-memory path, the fit materializes full-N B-spline basis
        tensors of roughly ``N * (m+3)^2 * n2`` floats at order 2 and
        ``N * (m+3)^3 * n3`` floats at order 3 (``n2``/``n3`` = number of
        parameter pairs/triples). ``slice_chunk_size`` does not change that
        footprint. When it exceeds the memory budget, the fit switches to the
        row-streamed path automatically.
    """
    N, D = X.shape
    # B-spline regression with backfitting needs a reasonable sample size
    # to avoid overfitting; 300 is a practical lower bound.
    if N < 300:
        raise ValueError(f"Need at least 300 samples, got {N}")
    if maxorder not in (1, 2, 3):
        raise ValueError(f"maxorder must be 1, 2, or 3, got {maxorder}")
    if D < maxorder:
        import warnings

        maxorder = min(maxorder, D)
        warnings.warn(
            f"jaxgsa: maxorder clamped to {maxorder} (need D >= maxorder, got D={D})",
            stacklevel=2,
            category=JaxgsaWarning,
        )
    if slice_chunk_size is not None and slice_chunk_size < 1:
        raise ValueError(f"slice_chunk_size must be >= 1, got {slice_chunk_size}")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    lambdax = float(lambdax)

    # Build term metadata (cached on host; only computed once per D/maxorder/m).
    static = _get_hdmr_static_data(D, maxorder, m)
    c1, c2, c3 = static.c1, static.c2, static.c3
    n1, n2, n3, n = static.n1, static.n2, static.n3, static.n
    m1, m2, m3 = static.m1, static.m2, static.m3
    beta2_host, beta3_host = static.beta2, static.beta3
    term_labels = _build_term_labels(problem, c1, c2, c3)
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
    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)
    _, T, K_out = Y_3d.shape
    if prenormalize:
        # Standardize each (t, k) slice to zero-mean, unit-variance before
        # fitting. The scale factors are stored so prediction can invert it.
        Y_3d, y_mean, y_std, _ = _prenormalize_outputs(Y_3d)
    else:
        y_mean = jnp.zeros(Y_3d.shape[1:], dtype=Y_3d.dtype)
        y_std = jnp.ones(Y_3d.shape[1:], dtype=Y_3d.dtype)

    total = T * K_out

    # Streamed path: engage when the user forces it with an explicit
    # batch_size, or when the in-memory fit's resident footprint (full-N
    # bases + one lane's kernel transients) would exceed the memory budget.
    full_bytes = _full_fit_bytes(N, D, m1, m2, m3, n1, n2, n3, n, Y_3d.dtype.itemsize)
    streamed = batch_size is not None or full_bytes > get_memory_budget()
    if streamed:
        # (N, T, K) -> (N, T*K): keep the sample axis leading so row batches
        # are contiguous slices; column t*K + k is output slice (t, k).
        Y_nl = Y_3d.reshape(N, total)
        sa_cat, sb_cat, s_cat, select_lanes, rmse_cat, c1_cat, c2_cat, c3_cat, f0_cat = (
            _fit_hdmr_streamed(
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
            )
        )
        select_sum = jnp.sum(select_lanes, axis=0)
    else:
        # In-memory path: build the full-N B-spline bases once (shared across
        # output slices; only Y changes) and vmap the fitting kernel over
        # chunks of output slices.
        B1 = _build_B1(X_n, m)  # (N, m1, D)
        B2 = _build_B2(B1, c2_idx, beta2) if n2 > 0 else jnp.zeros((N, 1, 1))
        B3 = _build_B3(B1, c3_idx, beta3) if n3 > 0 else jnp.zeros((N, 1, 1))

        # (N,T,K) -> (T,K,N) -> (T*K, N): move sample axis last, then flatten
        # time x output into a single batch dim so each row is an independent
        # response vector for the vmapped kernel.
        Y_flat = Y_3d.transpose(1, 2, 0).reshape(T * K_out, N)
        if slice_chunk_size is None:
            # Peak per-lane transient inside the vmapped kernel is dominated by
            # ~5 arrays of shape (chunk, N, n_terms) (Y_em, its centered copy,
            # the leave-one-out sums Y0_minus, their centered copy, and slack),
            # so derive the chunk count from a fixed bytes budget.
            itemsize = Y_flat.dtype.itemsize
            slice_chunk_size = max(1, get_memory_budget() // (5 * N * n * itemsize))
        cs = min(slice_chunk_size, total)

        # Accumulate chunk results; select_sum aggregates F-test pass counts
        # across chunks to report how many output slices found each term
        # significant.
        sa_parts, sb_parts, s_parts, rmse_parts = [], [], [], []
        c1_parts, c2_parts, c3_parts, f0_parts = [], [], [], []
        select_sum = jnp.zeros(n)

        batched_kernel = _get_batched_hdmr_kernel(
            D,
            maxorder,
            m,
            maxiter,
            lambdax,
            N,
        )
        # Process output slices in chunks to bound peak device memory when T*K
        # is large. Each chunk is a vmap batch of independent HDMR fits.
        for start in range(0, total, cs):
            end = min(start + cs, total)
            sa, sb, s, sel, rmse_val, c1_coef, c2_coef, c3_coef, f0_val = batched_kernel(
                B1,
                B2,
                B3,
                Y_flat[start:end],
                f_crits,
            )
            sa_parts.append(sa)
            sb_parts.append(sb)
            s_parts.append(s)
            rmse_parts.append(rmse_val)
            c1_parts.append(c1_coef)
            f0_parts.append(f0_val)
            select_sum = select_sum + jnp.sum(sel, axis=0)
            if n2 > 0:
                c2_parts.append(c2_coef)
            if n3 > 0:
                c3_parts.append(c3_coef)

        sa_cat = jnp.concatenate(sa_parts)
        sb_cat = jnp.concatenate(sb_parts)
        s_cat = jnp.concatenate(s_parts)
        rmse_cat = jnp.concatenate(rmse_parts)
        c1_cat = jnp.concatenate(c1_parts)
        c2_cat = jnp.concatenate(c2_parts) if n2 > 0 else None
        c3_cat = jnp.concatenate(c3_parts) if n3 > 0 else None
        f0_cat = jnp.concatenate(f0_parts)

    # Reassemble into the full (T, K, n_terms) index arrays.
    Sa_out = sa_cat.reshape(T, K_out, n)
    Sb_out = sb_cat.reshape(T, K_out, n)
    S_out = s_cat.reshape(T, K_out, n)
    # Aggregate per-term S into per-parameter total-order indices.
    ST_out = _compute_ST(S_out, c2_idx, c3_idx, n1)

    # Squeeze
    Sa_out, Sb_out, S_out, ST_out = _squeeze_hdmr(
        Sa_out,
        Sb_out,
        S_out,
        ST_out,
        squeeze_time,
        squeeze_output,
    )

    C1_out = _reshape_emulator_value(
        c1_cat,
        T,
        K_out,
        squeeze_time,
        squeeze_output,
    )
    C2_out = None
    C3_out = None
    if c2_cat is not None:
        C2_out = _reshape_emulator_value(
            c2_cat,
            T,
            K_out,
            squeeze_time,
            squeeze_output,
        )
    if c3_cat is not None:
        C3_out = _reshape_emulator_value(
            c3_cat,
            T,
            K_out,
            squeeze_time,
            squeeze_output,
        )
    f0_out = _reshape_emulator_value(
        f0_cat,
        T,
        K_out,
        squeeze_time,
        squeeze_output,
    )
    y_mean_out = _squeeze_output_axes(y_mean, squeeze_time, squeeze_output, n_trailing=0)
    y_std_out = _squeeze_output_axes(y_std, squeeze_time, squeeze_output, n_trailing=0)

    # Bundle all fitted state needed to reconstruct predictions at new points.
    fit_state: _HDMRFit = {
        "C1": C1_out,
        "C2": C2_out,
        "C3": C3_out,
        "f0": f0_out,
        "prenormalize": prenormalize,
        "y_mean": y_mean_out,
        "y_std": y_std_out,
        "m": m,
        "maxorder": maxorder,
    }

    return HDMRResult(
        Sa=Sa_out,
        Sb=Sb_out,
        S=S_out,
        ST=ST_out,
        problem=problem,
        terms=term_labels,
        invalid=invalid,
        _fit=fit_state,
        select=select_sum,
        # RMSE is computed on the standardized scale inside the kernel;
        # multiply by y_std to report it on the original output scale.
        rmse=_reshape_emulator_value(rmse_cat, T, K_out, squeeze_time, squeeze_output) * y_std_out,
        streamed=streamed,
        _c2=tuple(c2),
        _c3=tuple(c3),
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
    batches sized against a transient-memory budget. The kernel also inverts
    the output standardization when the fit used ``prenormalize=True``, so
    batched predictions land on the original output scale.

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
    prenormalize = em["prenormalize"]
    y_mean = em["y_mean"]
    y_std = em["y_std"]

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
        # Add grand mean to recover the full surrogate prediction, then undo
        # the standardization applied during fitting, if any. Both are
        # elementwise per row, so doing them per batch matches single-shot.
        Y_pred = Y_total + f0
        if prenormalize:
            Y_pred = Y_pred * y_std + y_mean
        return Y_pred

    return _PredictPlan(X=X_n, bytes_per_row=X_n.dtype.itemsize * elems_per_row, kernel=_predict)
