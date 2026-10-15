"""PCE analysis and emulation entry points."""

from __future__ import annotations

import math
import warnings
from functools import lru_cache, partial
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.batching import get_memory_budget, pad_rows, resolve_batch_size
from jaxgsa._core.bootstrap import bootstrap_draws, interval
from jaxgsa._core.entry import (
    at_least,
    bootstrap_checks,
    check_scalars,
    gates,
    prepare,
    require,
)
from jaxgsa._core.fit_quality import check_fit_finite, warn_cv_overfit, warn_variance_fit
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.irregular import (
    IrregularResult,
    IrregularY,
    _fwd_kwargs,
    _is_irregular_y,
    analyze_irregular,
)
from jaxgsa._core.precision import unit_clip_bounds
from jaxgsa._core.result import CIInfo
from jaxgsa._core.sampling import UNIT_CLIP
from jaxgsa._core.surrogate import _PredictPlan
from jaxgsa._core.transforms import GROUPED_MIN_D, _in_column_order, _truncnorm_cdf
from jaxgsa._core.validation import (
    _is_constant_slice,
    _prepare_Y,
    as_key,
)
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.pce._engine import (
    build_design_matrix,
    build_multi_index,
    hat_diagonal,
    loo_error,
    sobol_from_coefficients,
)
from jaxgsa.pce._result import PCEResult
from jaxgsa.problem import CategoricalSpec, Problem, UniformSpec

# Keyword-only parameters the irregular engine forwards unchanged to the
# per-channel recursive analyze() calls. verbose is excluded:
# the engine owns it.
_IRREGULAR_KWARGS = (
    "order",
    "ridge",
    "fit_ratio",
    "batch_size",
    "n_bootstrap",
    "conf_level",
    "ci_method",
    "key",
    "on_invalid",
    "keep_replicates",
)

# A truncated Gaussian is treated as effectively unbounded, and so kept on the
# Hermite basis, when both of its bounds sit at least this many standard
# deviations from the mean. Below the threshold the truncated measure differs
# enough from the standard normal that Hermite polynomials are no longer
# orthogonal against it.
#
# M12: this gate and the order cap below must be calibrated at the same
# width. Measured max |G - I| for orthonormalised He_0..He_P under a
# truncated N(0, 1) at |z| = 5.0: 1.5e-3 (P=3), 3.1e-2 (P=5), 2.0e-1 (P=7) --
# large enough that a response with its variance concentrated in a degree-7
# mode returned S1 that missed the true split by 0.05 with no warning. At
# |z| = 7.03, the width used below, the same defects are 4.0e-8 (P=3),
# 2.8e-5 (P=6), 9.5e-3 (P=10): negligible up to the order cap this module
# enforces.
_WIDE_TRUNCATION_Z = 7.0

# Above this polynomial degree the Hermite Gram defect against even a very wide
# truncated normal stops being negligible, so Legendre is the safer basis.
# Measured max |G - I| for orthonormalised He_0..He_P against the truncated
# normal at |z| = 7.03: 4.0e-8 (P=3), 2.8e-5 (P=6), 9.5e-3 (P=10).
_MAX_HERMITE_ORDER_UNDER_TRUNCATION = 7

# Fewest rows the fit can still run on after ``on_invalid="drop"`` removed
# some. Two is the floor of the arithmetic itself: a sample variance, which
# every index divides by, is undefined below it. `_auto_order` never drops
# below the order-1 expansion (D + 1 terms), so a row count between this
# floor and that expansion's term budget still fails, with a clear message
# (M2), rather than fitting an underdetermined system silently.
_MIN_ROWS = 2


def _truncated_gaussian_is_wide(
    lo: float | None, hi: float | None, mean: float, std: float
) -> bool:
    """Decide whether a truncated Gaussian is wide enough to stay on Hermite.

    The test uses the standardized bound ``|z| = |bound - mean| / std``. Both
    declared sides must sit at least ``_WIDE_TRUNCATION_Z`` standard deviations
    out. A one-sided truncation is judged by its bounded side alone.

    Args:
        lo: Lower truncation bound, or ``None`` when the side is open.
        hi: Upper truncation bound, or ``None`` when the side is open.
        mean: Marginal mean.
        std: Marginal standard deviation.

    Returns:
        ``True`` when every declared bound is at least ``_WIDE_TRUNCATION_Z``
        standardized units from the mean.
    """
    z = [abs(bound - mean) / std for bound in (lo, hi) if bound is not None]
    return bool(z) and min(z) >= _WIDE_TRUNCATION_Z


def _order_forced_legendre(problem: Problem, order: int) -> list[str]:
    """Name the inputs that a high order moves from Hermite to Legendre.

    A wide truncated Gaussian stays on Hermite while the order is at most
    ``_MAX_HERMITE_ORDER_UNDER_TRUNCATION``. Above that cap it goes through
    its CDF onto Legendre, which has to approximate the steep inverse normal
    CDF and fits worse. Only the order causes this switch, which is why it
    deserves a warning. A narrow truncation moves to Legendre at any order
    and is not reported.

    Args:
        problem: Problem whose ``input_specs`` decide the per-dimension map.
        order: Effective maximum polynomial degree of the expansion.

    Returns:
        Names of the inputs that the order alone moved to Legendre.
    """
    if order <= _MAX_HERMITE_ORDER_UNDER_TRUNCATION:
        return []
    names = []
    for name, spec in zip(problem.names, problem.input_specs, strict=True):
        if isinstance(spec, CategoricalSpec | UniformSpec):
            continue
        if spec.low is None and spec.high is None:
            continue
        std = math.sqrt(float(spec.variance))
        if _truncated_gaussian_is_wide(spec.low, spec.high, spec.mean, std):
            names.append(name)
    return names


def _map_to_reference(X: Array, problem: Problem, order: int) -> tuple[Array, tuple[str, ...]]:
    """Map physical inputs to the orthogonal polynomial reference domain.

    This is the Wiener-Askey rule. Uniform inputs use Legendre, mapped to
    [-1, 1]. Untruncated Gaussian inputs use Hermite, standardized to
    N(0, 1).

    A truncated Gaussian picks its basis from how tight the truncation is.
    A narrow truncation is a genuinely different measure, so the input goes
    through its truncated CDF and onto Legendre. A wide truncation keeps
    Hermite. Wide means every declared bound sits at least
    ``_WIDE_TRUNCATION_Z`` standard deviations from the mean, which is almost
    the standard normal. Hermite is both cheaper and far more accurate there,
    because the Legendre route has to approximate the steep inverse normal
    CDF with low-order polynomials, and that measurably degrades the fit.
    Requesting an order above ``_MAX_HERMITE_ORDER_UNDER_TRUNCATION`` forces
    Legendre again, because the Hermite Gram defect against the truncated
    measure grows with degree.

    Args:
        X: Inputs in physical units, shape ``(N, D)``.
        problem: Problem whose ``input_specs`` decide the per-dimension map.
        order: Effective maximum polynomial degree of the expansion.

    Returns:
        Tuple of the reference-domain inputs, shape ``(N, D)``, and a
        length-``D`` tuple of ``"uniform"`` / ``"gaussian"`` tags telling
        ``build_design_matrix`` which 1-D polynomial family each dimension
        needs.
    """
    if problem.n_params < GROUPED_MIN_D:
        return _map_by_column(X, problem, order)
    return _map_by_group(X, problem, order)


def _map_by_column(X: Array, problem: Problem, order: int) -> tuple[Array, tuple[str, ...]]:
    """Map physical inputs to the reference domain, one column at a time.

    See :func:`_map_to_reference` for the rule and why this loop is kept for
    small ``D``.

    Args:
        X: Inputs in physical units, shape ``(N, D)``.
        problem: Problem whose ``input_specs`` decide the per-dimension map.
        order: Effective maximum polynomial degree of the expansion.

    Returns:
        The reference-domain inputs and the per-dimension family tags.
    """
    D = problem.n_params
    cols = []
    input_types: list[str] = []
    hermite_safe_order = order <= _MAX_HERMITE_ORDER_UNDER_TRUNCATION
    for d in range(D):
        spec = problem.input_specs[d]
        if isinstance(spec, CategoricalSpec):
            # Layered guard: pce.analyze rejects categorical problems up front.
            raise ValueError(
                f"Parameter {problem.names[d]!r} is categorical; no orthogonal "
                "polynomial family exists for an unordered marginal"
            )
        if isinstance(spec, UniformSpec):
            cols.append(2.0 * (X[:, d] - spec.low) / (spec.high - spec.low) - 1.0)
            input_types.append("uniform")
            continue

        mean, lo, hi = spec.mean, spec.low, spec.high
        # math, not jnp: the standard deviation is a property of the declared
        # marginal, so it must stay a host float. Under `jit` a `jnp` call on
        # it returns a tracer, and `float()` on a tracer raises -- which is
        # what made the reference map untraceable for a Gaussian input.
        std = math.sqrt(float(spec.variance))
        truncated = lo is not None or hi is not None
        if truncated and not (
            hermite_safe_order and _truncated_gaussian_is_wide(lo, hi, mean, std)
        ):
            a = -np.inf if lo is None else (lo - mean) / std
            b = np.inf if hi is None else (hi - mean) / std
            u = _truncnorm_cdf((X[:, d] - mean) / std, a, b)
            # The clip keeps u strictly inside (0, 1), so a sample at or
            # beyond a truncation bound stays distinguishable from the bound
            # itself. In float32 its upper half did nothing at all --
            # ``1 - UNIT_CLIP`` is exactly 1.0 there -- so the bound is
            # brought inside u's own dtype instead. The lower bound stays at
            # UNIT_CLIP, which float32 represents exactly; widening it for
            # symmetry would discard information and move numbers for
            # nothing. See unit_clip_bounds.
            u = jnp.clip(u, *unit_clip_bounds(UNIT_CLIP, u.dtype))
            cols.append(2.0 * u - 1.0)
            input_types.append("uniform")
        else:
            cols.append((X[:, d] - mean) / std)
            input_types.append("gaussian")

    return jnp.column_stack(cols), tuple(input_types)


def _map_by_group(X: Array, problem: Problem, order: int) -> tuple[Array, tuple[str, ...]]:
    """Map physical inputs to the reference domain, one op per map family.

    Args:
        X: Inputs in physical units, shape ``(N, D)``.
        problem: Problem whose ``input_specs`` decide the per-dimension map.
        order: Effective maximum polynomial degree of the expansion.

    Returns:
        The reference-domain inputs and the per-dimension family tags.
    """
    D = problem.n_params
    X = jnp.asarray(X)
    dtype = X.dtype if jnp.issubdtype(X.dtype, jnp.floating) else jnp.result_type(X.dtype, float)
    X = X.astype(dtype)
    hermite_safe_order = order <= _MAX_HERMITE_ORDER_UNDER_TRUNCATION

    # Columns are grouped by map and each group goes through one broadcast op,
    # so the number of eager dispatches does not grow with D. Per-column
    # constants are computed on the host in double precision and cast to the
    # column dtype, which is what a Python-float operand becomes inside a
    # single-column op, so, run eagerly, the result is bit-identical to a column loop.
    input_types: list[str] = ["gaussian"] * D
    uniform: list[int] = []
    standard: list[int] = []  # Hermite: (x - mean) / std
    lower: list[int] = []  # truncated CDF -> Legendre, interval not in the upper tail
    upper: list[int] = []  # truncated CDF -> Legendre, interval in the upper tail
    consts: dict[int, tuple[float, ...]] = {}
    for d in range(D):
        spec = problem.input_specs[d]
        if isinstance(spec, CategoricalSpec):
            # Layered guard: pce.analyze rejects categorical problems up front.
            raise ValueError(
                f"Parameter {problem.names[d]!r} is categorical; no orthogonal "
                "polynomial family exists for an unordered marginal"
            )
        if isinstance(spec, UniformSpec):
            uniform.append(d)
            input_types[d] = "uniform"
            consts[d] = (spec.low, spec.high - spec.low)
            continue

        mean, lo, hi = spec.mean, spec.low, spec.high
        # math, not jnp: the standard deviation is a property of the declared
        # marginal, so it must stay a host float. Under `jit` a `jnp` call on
        # it returns a tracer, and `float()` on a tracer raises -- which is
        # what made the reference map untraceable for a Gaussian input.
        std = math.sqrt(float(spec.variance))
        truncated = lo is not None or hi is not None
        if truncated and not (
            hermite_safe_order and _truncated_gaussian_is_wide(lo, hi, mean, std)
        ):
            a = -np.inf if lo is None else (lo - mean) / std
            b = np.inf if hi is None else (hi - mean) / std
            (upper if a > 0.0 else lower).append(d)
            input_types[d] = "uniform"
            consts[d] = (mean, std, a, b)
        else:
            standard.append(d)
            consts[d] = (mean, std)

    def column_consts(group: list[int], n: int) -> list[Array]:
        values = np.asarray([consts[d][:n] for d in group], dtype=np.float64)
        return [jnp.asarray(values[:, j], dtype=dtype) for j in range(n)]

    parts: list[tuple[list[int], Array]] = []
    if uniform:
        low, width = column_consts(uniform, 2)
        parts.append((uniform, 2.0 * (X[:, uniform] - low) / width - 1.0))
    if standard:
        mean, std = column_consts(standard, 2)
        parts.append((standard, (X[:, standard] - mean) / std))
    for group in (lower, upper):
        if group:
            mean, std = column_consts(group, 2)
            a_host = np.asarray([consts[d][2] for d in group], dtype=np.float64)
            b_host = np.asarray([consts[d][3] for d in group], dtype=np.float64)
            u = _truncnorm_cdf((X[:, group] - mean) / std, a_host, b_host)
            # The clip keeps u strictly inside (0, 1), so a sample at or
            # beyond a truncation bound stays distinguishable from the bound
            # itself. In float32 its upper half did nothing at all --
            # ``1 - UNIT_CLIP`` is exactly 1.0 there -- so the bound is
            # brought inside u's own dtype instead. The lower bound stays at
            # UNIT_CLIP, which float32 represents exactly; widening it for
            # symmetry would discard information and move numbers for
            # nothing. See unit_clip_bounds.
            u = jnp.clip(u, *unit_clip_bounds(UNIT_CLIP, u.dtype))
            parts.append((group, 2.0 * u - 1.0))

    return _in_column_order(parts, D), tuple(input_types)


def _auto_order(D: int, N: int, max_order: int, fit_ratio: float) -> int:
    """Reduce polynomial order so the term count fits within the sample budget."""
    from math import comb

    # Reduce order until C(D+p, p) <= fit_ratio * N to prevent overfitting
    # when the design matrix would have more columns than rows.
    cap = max(1, int(fit_ratio * N))
    order = max_order
    while order >= 1 and comb(D + order, order) > cap:
        order -= 1
    return max(order, 1)


def effective_order(
    problem: Problem, n_samples: int, *, order: int = 3, fit_ratio: float = 0.5
) -> int:
    """Report the polynomial degree a fit would actually use.

    A PCE with ``C(D + order, order)`` terms cannot be fitted on fewer rows
    than terms, so :func:`analyze` and :func:`indices` reduce ``order`` until
    the term count fits within ``fit_ratio * n_samples``. That reduction is
    real information: an expansion fitted at degree 2 does not represent the
    cubic effects a caller who asked for degree 3 expects to see.

    :func:`analyze` reports it twice, as a warning and as ``result.order``.
    :func:`indices` reports it neither way — it is traceable, so it has no
    result object to carry it and no place to put a side effect. This
    function is where an ``indices`` caller reads it instead. It needs only
    the sample size, so it answers before the model has been run.

    Args:
        problem: Parameter names and distributions. Only ``n_params`` is
            read.
        n_samples: Number of rows the fit will run on, after any invalid
            rows have been dropped.
        order: Maximum total polynomial degree requested.
        fit_ratio: Maximum ratio of terms to samples, as in :func:`analyze`.

    Returns:
        The degree a fit with these settings would use. Equal to ``order``
        when the sample budget allows it, and lower otherwise, never below 1.
    """
    return _auto_order(problem.n_params, n_samples, order, fit_ratio)


class PCEIndices(NamedTuple):
    """PCE Sobol indices as plain arrays, as :func:`indices` returns them.

    Tuple unpacking works: ``S1, ST, S2 = indices(...)``.

    Attributes:
        S1: First-order Sobol indices, shape ``(D,)`` / ``(K, D)`` /
            ``(T, K, D)``.
        ST: Total-order Sobol indices, same shape as ``S1``.
        S2: Second-order interaction indices, shape ``(D, D)`` /
            ``(K, D, D)`` / ``(T, K, D, D)``, NaN on the diagonal.
    """

    S1: Array
    ST: Array
    S2: Array


class _PCEFit(NamedTuple):
    """The shared PCE fit: coefficients and everything needed to reuse them.

    PCE analysis derives Sobol indices and the leave-one-out (LOO) diagnostic
    from this state. :meth:`PCEResult.shapley` reuses its coefficients and
    multi-index. The design matrix and Gram factorization are deliberately
    NOT carried: the streamed path never materializes them at full ``N``, so
    both fit paths reduce to the same coefficient, LOO and fitted-variance
    summary.
    """

    coefficients: Array  # (T, K, n_terms), terms-last
    multi_index: np.ndarray  # (n_terms, D)
    order: int  # effective order after _auto_order
    loo_flat: Array | None  # (T*K,) per-slice LOO RMSE, or None (diagnostics=False)
    rmse_flat: Array | None  # (T*K,) per-slice in-sample RMSE, or None (diagnostics=False)
    streamed: bool  # True when the row-streamed fit path ran


@lru_cache(maxsize=32)
def _get_streamed_steps(input_types: tuple[str, ...], order: int):
    """Build and cache the two jitted per-batch steps of the streamed fit.

    The streamed fit used to rebuild the design block and run its
    accumulation matmuls as eagerly dispatched ops, once per batch — the
    same per-batch dispatch overhead the HDMR fit unification measured at
    1.2-1.67x and fixed with jitted, padded per-batch steps (see
    ``jaxgsa/hdmr/_fit.py``). This is the same house pattern: each step is
    one jitted call, and the caller pads the ragged trailing batch so each
    step compiles once per shape.

    Masking. A padded row of ``X_ref`` is all zeros, and the constant basis
    term evaluates to 1 there, so the raw design block is NOT zero on padded
    rows. Both steps therefore multiply the design block by the 0/1 row mask
    ``w``. A zeroed design row contributes a hard zero to ``G = Phi^T Phi``
    and to ``B = Phi^T Y``, and in the LOO pass it makes the padded row's
    leverage 0 and its residual ``Y_pad - 0 = 0`` (``Y`` is zero-padded), so
    the accumulated sum of squares gains exactly nothing.

    The multi-index is rebuilt from ``(D, order)`` inside the cache rather
    than passed in: :func:`build_multi_index` is deterministic in those two
    scalars, and an array argument would not be hashable as a cache key.

    Args:
        input_types: Per-dimension ``"uniform"`` / ``"gaussian"`` tags. Its
            length is ``D``.
        order: Effective (post-``_auto_order``) polynomial order.

    Returns:
        ``(acc_normal, acc_loo)``. ``acc_normal(X_b, Y_b, w, G, B)`` folds
        one batch into the running normal equations; ``acc_loo(X_b, Y_b, w,
        coeffs_flat, gram_chol, sse)`` folds one batch into the running
        ``(LOO, in-sample)`` sums of squares.
    """
    multi_index = build_multi_index(len(input_types), order)

    @jax.jit
    def acc_normal(X_b: Array, Y_b: Array, w: Array, G: Array, B: Array) -> tuple[Array, Array]:
        Phi_b = build_design_matrix(X_b, multi_index, input_types, order) * w[:, None]
        return G + Phi_b.T @ Phi_b, B + Phi_b.T @ Y_b

    @jax.jit
    def acc_loo(
        X_b: Array,
        Y_b: Array,
        w: Array,
        coeffs_flat: Array,
        gram_chol: Array,
        sse: tuple[Array, Array],
    ) -> tuple[Array, Array]:
        Phi_b = build_design_matrix(X_b, multi_index, input_types, order) * w[:, None]
        residuals = Y_b - Phi_b @ coeffs_flat  # (b, M); zero on padded rows
        loo_residuals = residuals / (1.0 - hat_diagonal(Phi_b, gram_chol))[:, None]
        sse_loo, sse_fit = sse
        return sse_loo + jnp.sum(loo_residuals**2, axis=0), sse_fit + jnp.sum(residuals**2, axis=0)

    return acc_normal, acc_loo


def _fit_pce_streamed(
    X_ref: Array,
    Y_flat: Array,
    multi_index: np.ndarray,
    input_types: tuple[str, ...],
    order: int,
    ridge: float,
    batch_size: int | None,
    diagnostics: bool,
) -> tuple[Array, Array | None, Array | None]:
    """Fit the PCE by streaming row batches.

    The normal-equation pass always runs. A second pass computes LOO and the
    in-sample RMSE (from the same residuals) only when ``diagnostics`` is true.
    Neither pass holds more than one
    ``(batch, n_terms)`` design block:

    1. Accumulate ``G = Phi^T Phi`` (n_terms, n_terms) and ``B = Phi^T Y``
       (n_terms, T*K), then solve ``(G + ridge*I) c = B`` once. This is
       mathematically identical to the single-pass normal equations. Only
       the float32 summation order differs.
    2. When diagnostics are enabled, with the Cholesky factor of ``G + ridge*I``
       known, rebuild each design block, get its hat-matrix diagonal from
       :func:`jaxgsa.pce._engine.hat_diagonal` and its per-row residuals, and
       accumulate the exact LOO sum of squares as defined by
       :func:`jaxgsa.pce._engine.loo_error`. Both paths call the same leverage
       function, so they cannot disagree about the clip or the formula; only
       the ``mean`` over rows is re-derived here, from the accumulated sum.

    Each per-batch step is one jitted call from :func:`_get_streamed_steps`,
    and the ragged trailing batch is zero-padded back to the full width with
    a row mask, so each step compiles once instead of once per batch shape.

    Args:
        X_ref: Inputs already mapped to the reference domain, shape
            ``(N, D)``.
        Y_flat: Flattened output slices, shape ``(N, T*K)``.
        multi_index: Multi-index array, shape ``(n_terms, D)``.
        input_types: Per-dimension ``"uniform"`` / ``"gaussian"`` tags.
        order: Effective (post-``_auto_order``) polynomial order.
        ridge: Tikhonov parameter added to the Gram matrix.
        batch_size: Rows per batch, or ``None`` to derive one from the
            active memory budget.
        diagnostics: Whether to run the streamed LOO pass.

    Returns:
        Tuple of ``coeffs_flat``, shape ``(n_terms, T*K)``, the per-slice LOO
        RMSE, shape ``(T*K,)``, and the per-slice in-sample RMSE, same shape.
        Both are ``None`` when diagnostics are disabled.
    """
    N = X_ref.shape[0]
    n_terms = multi_index.shape[0]
    M = Y_flat.shape[1]
    dtype = jnp.result_type(X_ref.dtype, Y_flat.dtype)

    # Per-row transient cost of one batch: 3*n_terms floats for the
    # design-block running product plus 2*M for the residual arrays.
    bytes_per_row = jnp.dtype(dtype).itemsize * (3 * n_terms + 2 * M)
    b = resolve_batch_size(bytes_per_row, N, batch_size)

    # Pad the sample axis so every batch traces at the same shape. The padded
    # rows are masked to zero inside the jitted steps, so they contribute
    # nothing to either pass.
    n_batches = math.ceil(N / b)
    n_pad = n_batches * b - N
    X_p = pad_rows(X_ref, n_pad)
    Y_p = pad_rows(Y_flat, n_pad)
    w = jnp.concatenate([jnp.ones((N,), dtype=dtype), jnp.zeros((n_pad,), dtype=dtype)])
    rows = [(i * b, i * b + b) for i in range(n_batches)]

    acc_normal, acc_loo = _get_streamed_steps(input_types, order)

    # Pass 1: accumulate the Gram matrix and cross-moments batch by batch.
    G = jnp.zeros((n_terms, n_terms), dtype=dtype)
    B = jnp.zeros((n_terms, M), dtype=dtype)
    for lo, hi in rows:
        G, B = acc_normal(X_p[lo:hi], Y_p[lo:hi], w[lo:hi], G, B)
    gram = G + ridge * jnp.eye(n_terms, dtype=dtype)
    coeffs_flat = jnp.linalg.solve(gram, B)  # (n_terms, M)

    if diagnostics:
        # Pass 2: exact LOO from the hat-matrix diagonal, streamed. The
        # leverage comes from the shared hat_diagonal helper, which only needs
        # the small (n_terms, n_terms) Cholesky factor, so this path holds
        # nothing that scales with N and applies the same clip as the
        # single-pass path.
        gram_chol = jnp.linalg.cholesky(gram)
        sse = (jnp.zeros((M,), dtype=dtype), jnp.zeros((M,), dtype=dtype))
        for lo, hi in rows:
            sse = acc_loo(X_p[lo:hi], Y_p[lo:hi], w[lo:hi], coeffs_flat, gram_chol, sse)
        loo_flat = jnp.sqrt(sse[0] / N)  # == sqrt(mean(loo_residuals^2, axis=0))
        rmse_flat = jnp.sqrt(sse[1] / N)  # in-sample, from the same residuals
    else:
        loo_flat = None
        rmse_flat = None

    return coeffs_flat, loo_flat, rmse_flat


def _single_pass_fit_bytes(N: int, n_terms: int, M: int, itemsize: int) -> int:
    """Estimate the resident bytes of the single-pass PCE fit path.

    Derived from the arrays the single-pass code actually materializes. At the
    peak, which is inside the leave-one-out step, two (N, n_terms) arrays are
    resident at once:

    - ``Phi``, the design matrix.
    - the triangular-solve result inside
      :func:`jaxgsa.pce._engine.hat_diagonal`.

    The coefficient solve reads ``Phi.T @ Y_flat`` (an ``(n_terms, M)``
    product, not an ``(n_terms, N)`` one) straight into ``jnp.linalg.solve``,
    matching the streamed path: there is no third ``(n_terms, N)`` array held
    across the leave-one-out step.

    Plus the (N, M) prediction and residual arrays ``loo_error`` materializes,
    for ``2 * N * M``.

    ``build_design_matrix`` also peaks at ~3 (N, n_terms) transients in an
    earlier phase, but that phase ends before ``Y`` is touched, so it does not
    add to this one.

    ``Y`` itself and the (n_terms, n_terms) Gram matrix are excluded: the
    caller holds Y either way, and the Gram factors are negligible next to
    the N-sized arrays.

    Args:
        N: Number of sample rows.
        n_terms: Number of expansion terms.
        M: Number of flattened output slices (T*K).
        itemsize: Bytes per element of the working dtype.

    Returns:
        Estimated peak resident bytes of the single-pass fit + LOO.
    """
    return itemsize * N * (2 * n_terms + 2 * M)


@partial(jax.jit, static_argnames=("problem", "order"))
def _single_pass_coefficients(
    problem: Problem, X: Array, Y_3d: Array, ridge: float, *, order: int
) -> tuple[Array, Array, Array]:
    """Fit the expansion in one pass, as a single compiled computation.

    Run eagerly, every op in this body compiles on its own on the first call
    (dozens of tiny XLA programs). Jitted, they compile once together. The
    problem and the effective order are static, so the multi-index and the
    basis family per dimension are trace-time constants.

    The coefficients come from this one program whether or not the caller
    wants diagnostics, so :func:`analyze` and :func:`indices` fit bit for bit
    the same expansion. The diagnostics run in :func:`_single_pass_loo`.

    Args:
        problem: Parameter names and distributions (static).
        X: Inputs in physical units, shape ``(N, D)``.
        Y_3d: Outputs in canonical layout, shape ``(N, T, K)``.
        ridge: Tikhonov parameter added to the Gram matrix.
        order: Effective (post-``_auto_order``) polynomial order.

    Returns:
        ``(coefficients, Phi, gram)``: coefficients in terms-last layout, shape
        ``(T, K, n_terms)``, the design matrix and the regularized Gram
        matrix. A caller that skips the diagnostics drops the last two.
    """
    N = X.shape[0]
    _, T, K = Y_3d.shape
    mi = build_multi_index(problem.n_params, order)
    X_ref, input_types = _map_to_reference(X, problem, order)
    Y_flat = Y_3d.reshape(N, T * K)  # column t*K + k is slice (t, k)
    Phi = build_design_matrix(X_ref, mi, input_types, order)
    gram = Phi.T @ Phi + ridge * jnp.eye(Phi.shape[1])
    # The basis is identical for every output slice (the effective order
    # depends only on N and D), so fitting all T*K slices is ONE shared
    # solve with multiple right-hand sides. Solving against
    # ``Phi.T @ Y_flat`` (an (n_terms, M) product) rather than building
    # ``solve(gram, Phi.T)`` (an (n_terms, N) product held across the LOO
    # step) matches the streamed path and drops a third N-sized array.
    coeffs_flat = jnp.linalg.solve(gram, Phi.T @ Y_flat)  # (n_terms, T*K)
    # Terms-last layout, matching HDMR's Sa convention (slices lead).
    return coeffs_flat.T.reshape(T, K, mi.shape[0]), Phi, gram


@jax.jit
def _single_pass_loo(
    Phi: Array, gram: Array, Y_3d: Array, coefficients: Array
) -> tuple[Array, Array]:
    """Per-slice LOO and in-sample RMSE of a single-pass fit.

    Args:
        Phi: Design matrix, shape ``(N, n_terms)``.
        gram: Regularized Gram matrix, shape ``(n_terms, n_terms)``.
        Y_3d: Outputs in canonical layout, shape ``(N, T, K)``.
        coefficients: Fitted coefficients, shape ``(T, K, n_terms)``.

    Returns:
        ``(loo_flat, rmse_flat)``, each of shape ``(T*K,)``.
    """
    N = Phi.shape[0]
    Y_flat = Y_3d.reshape(N, -1)
    coeffs_flat = coefficients.reshape(-1, coefficients.shape[-1]).T  # (n_terms, T*K)
    # Pass the small Cholesky factor of the Gram matrix, so loo_error reuses
    # the factorization without carrying anything that scales with N.
    loo_flat = loo_error(Phi, Y_flat, coeffs_flat, gram_chol=jnp.linalg.cholesky(gram))
    rmse_flat = jnp.sqrt(jnp.mean((Y_flat - Phi @ coeffs_flat) ** 2, axis=0))
    return loo_flat, rmse_flat


@partial(jax.jit, static_argnames=("D", "order", "budget"))
def _sobol_from_fit(
    coefficients: Array, Y_3d: Array, *, D: int, order: int, budget: int
) -> tuple[Array, Array, Array]:
    """Read the Sobol indices off a fitted expansion, as one compiled call.

    ``budget`` is the memory budget in bytes. It is static only so that a
    changed budget retraces :func:`sobol_from_coefficients`, which sizes its
    pair chunks from it at trace time.

    Args:
        coefficients: Fitted coefficients, shape ``(T, K, n_terms)``.
        Y_3d: Outputs in canonical layout, shape ``(N, T, K)``.
        D: Number of parameters.
        order: Effective polynomial order that built the multi-index.
        budget: Active memory budget in bytes.

    Returns:
        ``(S1, ST, S2)`` in canonical ``(T, K, ...)`` layout.
    """
    _ = budget
    N = Y_3d.shape[0]
    # is_constant reads the raw Y column, not the fit: a constant slice's
    # fitted non-constant coefficients are rarely bit-exact zero in float32
    # (M1), so the fit-only guard inside sobol_from_coefficients would miss
    # most of them.
    is_constant = _is_constant_slice(Y_3d.reshape(N, -1)).reshape(Y_3d.shape[1:])
    return sobol_from_coefficients(coefficients, build_multi_index(D, order), is_constant)


@jax.jit
def _fit_summary(
    loo_flat: Array, rmse_flat: Array, Y_3d: Array
) -> tuple[Array, Array, Array, Array]:
    """Shape the fit diagnostics and derive the explained variance.

    Args:
        loo_flat: Per-slice LOO RMSE, shape ``(T*K,)``.
        rmse_flat: Per-slice in-sample RMSE, shape ``(T*K,)``.
        Y_3d: Outputs in canonical layout, shape ``(N, T, K)``.

    Returns:
        ``(loo, rmse, explained_variance, std)``, each of shape ``(T, K)``,
        where ``std`` is ``sqrt(Var(Y))`` per slice.
    """
    T, K = Y_3d.shape[1:]
    total_var = jnp.var(Y_3d, axis=0)  # (T, K)
    is_constant = _is_constant_slice(Y_3d.reshape(Y_3d.shape[0], -1)).reshape(T, K)
    loo = loo_flat.reshape(T, K)
    rmse = rmse_flat.reshape(T, K)
    # 1 - in-sample MSE / sample variance of Y, on the same rows. The residual
    # comes straight from the fit, so a good fit cannot be pushed above 1 by
    # a mismatch between numerator and denominator.
    explained_variance = jnp.where(is_constant, jnp.nan, 1.0 - rmse**2 / total_var)
    return loo, rmse, explained_variance, jnp.sqrt(total_var)


def _fit_pce_core(
    problem: Problem,
    X: Array,
    Y_canonical: Array,
    *,
    # The defaults mirror `analyze`'s, so a caller that forwards only the
    # arguments it was given -- `jaxgsa.shapley.indices` does -- fits the same
    # expansion `analyze` would. `tests/test_shapley.py` compares the two.
    order: int = 3,
    ridge: float = 1e-8,
    fit_ratio: float = 0.5,
    batch_size: int | None = None,
    diagnostics: bool = True,
) -> _PCEFit:
    """Fit the shared PCE expansion for every output slice at once.

    Operates on an already-canonical Y (no validation, no zero-variance
    warning, no index extraction). The basis is identical across slices, so
    fitting all ``T*K`` right-hand sides is a single Gram solve.

    The fit streams over row batches instead (see :func:`_fit_pce_streamed`)
    in two cases: when an explicit ``batch_size`` is smaller than ``N``, so
    it actually splits the sample, or when ``batch_size`` is ``None`` and
    the single-pass residents (design matrix, Gram factorization, LOO
    residual arrays) would exceed the active memory budget. An explicit
    width always wins over the budget: ``batch_size >= N`` degenerates to
    the single-pass path, as one full row block, even when the single pass
    is over budget. The streamed fit solves the
    same normal equations and computes the same exact LOO. It differs only
    in float32 summation order.

    Nothing here warns. The order it actually fitted at comes back on
    ``_PCEFit.order``, and :func:`analyze` compares that against the request
    and warns. A warning inside the fit would put a host-side side effect on
    the path :func:`indices` traces, for information the caller can also get
    from :func:`effective_order` without running a fit at all.

    Args:
        diagnostics: Compute ``loo_flat`` and ``rmse_flat``. Off for
            :func:`indices` and every bootstrap replicate, neither of which
            reads them: on the single-pass path this skips a Cholesky
            factor, an ``(n_terms, N)`` triangular solve and an ``(N, T*K)``
            residual array that the Sobol-index extraction never touches.
            ``rmse_flat`` is then ``None``. So is ``loo_flat`` on either fit
            path.
    """
    Y_3d, _ = _prepare_Y(Y_canonical)
    N, D = X.shape
    _, T, K = Y_3d.shape

    # Cap polynomial order so n_terms <= fit_ratio * N (prevents overfitting).
    effective_order = _auto_order(D, N, order, fit_ratio)
    mi = build_multi_index(D, effective_order)
    n_terms = mi.shape[0]

    # Streamed iff the caller's explicit batch width actually splits the
    # sample, or no width was given and the single-pass residents exceed the
    # budget. An explicit ``batch_size`` always wins over the budget, so
    # ``batch_size >= N`` runs the single-pass path even on an over-budget
    # problem — the library-wide semantics: ``batch_size`` sizes row blocks,
    # clamped to N, and never selects a different algorithm.
    itemsize = jnp.dtype(jnp.result_type(X.dtype, float)).itemsize
    single_pass_bytes = _single_pass_fit_bytes(N, n_terms, T * K, itemsize)
    if batch_size is not None:
        streamed = batch_size < N
    else:
        streamed = single_pass_bytes > get_memory_budget()
    if not streamed:
        # Single-pass path: one jitted call maps the inputs, builds the design
        # matrix, solves the normal equations and (optionally) the exact LOO.
        coeffs, Phi, gram = _single_pass_coefficients(
            problem, X, Y_3d, ridge, order=effective_order
        )
        loo_flat = None
        rmse_flat = None
        if diagnostics:
            loo_flat, rmse_flat = _single_pass_loo(Phi, gram, Y_3d, coeffs)
    else:
        # Streamed path: same normal equations and exact LOO, accumulated
        # over row batches so peak memory stays within the budget.
        # Map inputs to reference domain; (N, D) is small next to (N, n_terms).
        X_ref, input_types = _map_to_reference(X, problem, effective_order)
        Y_flat = Y_3d.reshape(N, T * K)  # column t*K + k is slice (t, k)
        coeffs_flat, loo_flat, rmse_flat = _fit_pce_streamed(
            X_ref, Y_flat, mi, input_types, effective_order, ridge, batch_size, diagnostics
        )
        # Terms-last layout, matching HDMR's Sa convention (slices lead).
        coeffs = coeffs_flat.T.reshape(T, K, n_terms)

    return _PCEFit(
        coefficients=coeffs,
        multi_index=mi,
        order=effective_order,
        loo_flat=loo_flat,
        rmse_flat=rmse_flat,
        streamed=streamed,
    )


def _indices_3d(
    problem: Problem,
    X: Array,
    Y_3d: Array,
    *,
    order: int,
    ridge: float,
    fit_ratio: float,
    batch_size: int | None,
) -> tuple[Array, Array, Array]:
    """Fit the expansion and read the indices off it, in canonical layout.

    The shared body of :func:`indices` and of one bootstrap replicate. It
    takes ``Y`` already promoted to ``(N, T, K)`` and returns ``(T, K, ...)``
    arrays, so neither caller squeezes twice. Neither caller reads
    ``loo_flat`` or ``rmse_flat``, so the fit skips them
    (``diagnostics=False``); a bootstrap run is ``n_bootstrap`` of these.
    """
    fit = _fit_pce_core(
        problem,
        X,
        Y_3d,
        order=order,
        ridge=ridge,
        fit_ratio=fit_ratio,
        batch_size=batch_size,
        diagnostics=False,
    )
    return _sobol_from_fit(
        fit.coefficients,
        Y_3d,
        D=problem.n_params,
        order=fit.order,
        budget=get_memory_budget(),
    )


def _bootstrap_indices(
    problem: Problem,
    X: Array,
    Y_3d: Array,
    key: Array,
    *,
    n_bootstrap: int,
    order: int,
    ridge: float,
    fit_ratio: float,
    batch_size: int | None,
) -> tuple[Array, Array, Array]:
    """Refit the expansion on ``n_bootstrap`` row resamples.

    One replicate is a full refit: the rows are drawn with replacement, the
    design matrix is rebuilt on them and the normal equations are solved
    again. Nothing about a PCE index survives a resample without that — the
    indices are functions of the coefficients, not of any per-row statistic
    the point estimate already computed. So the replicates run as a Python
    loop over independent fits rather than as a vmapped reduction, and the
    cost is ``n_bootstrap`` fits.

    The row draw is shared by every output slice, exactly as the point
    estimate shares one basis across slices.

    Args:
        problem: Parameter names and distributions.
        X: Input samples with invalid rows already removed, shape ``(N, D)``.
        Y_3d: Outputs in canonical ``(N, T, K)`` layout.
        key: A ``jax.random`` key, split once per replicate.
        n_bootstrap: Number of replicates.
        order: Requested maximum polynomial degree.
        ridge: Tikhonov parameter.
        fit_ratio: Terms-to-samples cap.
        batch_size: Rows per fit batch, or ``None``.

    Returns:
        ``(S1, ST, S2)`` draws, each with a leading axis of length
        ``n_bootstrap`` followed by the canonical ``(T, K, ...)`` shape.
    """

    def _replicate(rows: Array) -> tuple[Array, Array, Array]:
        return _indices_3d(
            problem,
            X[rows],
            Y_3d[rows],
            order=order,
            ridge=ridge,
            fit_ratio=fit_ratio,
            batch_size=batch_size,
        )

    return bootstrap_draws(key, X.shape[0], n_bootstrap, _replicate)


def indices(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    order: int = 3,
    ridge: float = 1e-8,
    fit_ratio: float = 0.5,
    batch_size: int | None = None,
) -> PCEIndices:
    """Compute PCE Sobol indices as plain arrays, with no diagnostics.

    This is the transformable core of :func:`analyze`. It fits the same
    expansion on the same data and returns the same numbers, but it does
    nothing else: no non-finite check, no zero-variance warning, no
    reduced-order warning, no leave-one-out diagnostic, no
    :class:`jaxgsa.pce.PCEResult`, and no read of any array value on the
    host. So it composes with ``jax.jit``, ``jax.vmap``, ``jax.grad`` and
    ``jax.jacrev``, which :func:`analyze` cannot, because a policy decision
    needs a concrete value and a tracer has none.

    Use :func:`analyze` for ordinary analysis. Nothing here checks the
    inputs, so a single NaN reaches the normal equations and turns every
    index into NaN without a word. Nothing here reports the fit quality
    either: ``cv_rmse`` and ``explained_variance`` are what tell you whether
    a PCE index means anything, and only :func:`analyze` computes them.

    The polynomial degree may still be reduced to fit the sample budget, as
    in :func:`analyze`, and here that happens silently. Call
    :func:`effective_order` to find out what degree a fit will use.

    Tier T4 (behavioural contract): the returned arrays must equal the
    corresponding fields of ``analyze``'s result on clean outputs, and the
    function must survive ``jit``, ``vmap`` and ``jit(jacrev(...))``. Checked
    in ``tests/test_pce_gradients.py``.

    Args:
        problem: Parameter names and distributions.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)`` or ``(N, T, K)``, as
            :func:`analyze` accepts them.
        order: Maximum total polynomial degree, as in :func:`analyze`.
        ridge: Tikhonov regularization for the least-squares fit.
        fit_ratio: Maximum ratio of terms to samples before ``order`` is
            reduced. Must be at most 1, and must leave room for the order-1
            expansion: ``D + 1 <= int(fit_ratio * N)``.
        batch_size: Rows per batch during the fit, clamped to ``N``, or
            ``None`` to derive one from the memory budget, as in
            :func:`analyze`. Both paths trace; they differ only in float32
            summation order.

    Returns:
        A :class:`PCEIndices` ``(S1, ST, S2)``, at the shapes ``analyze``
        reports for this ``Y``.

    Raises:
        ValueError: If ``order``, ``ridge``, ``fit_ratio`` or ``batch_size``
            is out of range, ``fit_ratio`` is above 1, ``X`` has too few rows
            for even the order-1 expansion at that ``fit_ratio``
            (``D + 1 > int(fit_ratio * N)``, which would make the fit
            underdetermined), ``problem.correlation`` declares a dependence
            structure (the basis is not orthogonal under the dependent
            measure, so the indices would be silently wrong, exactly as in
            :func:`analyze`), or ``problem`` has a categorical parameter,
            which no orthogonal polynomial family covers.
    """
    from jaxgsa.pce import _SPEC

    D = problem.n_params
    N = X.shape[0]
    min_terms = D + 1  # term count of the order-1 expansion
    check_scalars(
        (
            at_least("order", order, 1),
            at_least("ridge", ridge, 0.0),
            at_least("fit_ratio", fit_ratio, 0.0),
            require(fit_ratio <= 1.0, f"fit_ratio must be <= 1, got {fit_ratio}"),
            at_least("batch_size", batch_size, 1),
            require(
                min_terms <= int(fit_ratio * N),
                f"fit_ratio={fit_ratio} and N={N} rows cannot fit even the "
                f"order-1 expansion ({min_terms} terms for D={D} parameters); "
                "raise fit_ratio or provide more rows",
            ),
        ),
        method="jaxgsa.pce.indices",
    )
    # The same capability gate analyze applies through prepare().
    # problem.dependence is static host-side metadata, so the check
    # runs at trace time and the core still composes with jit/vmap/grad.
    gates(_SPEC, problem, method="jaxgsa.pce.indices")
    Y_3d, layout = _prepare_Y(Y)
    S1, ST, S2 = _indices_3d(
        problem,
        jnp.asarray(X),
        Y_3d,
        order=order,
        ridge=ridge,
        fit_ratio=fit_ratio,
        batch_size=batch_size,
    )
    return PCEIndices(
        S1=layout.squeeze(S1), ST=layout.squeeze(ST), S2=layout.squeeze(S2, n_trailing=2)
    )


def analyze(
    problem: Problem,
    X: Array,
    Y: Array | IrregularY,
    *,
    order: int = 3,
    ridge: float = 1e-8,
    fit_ratio: float = 0.5,
    batch_size: int | None = None,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> PCEResult | IrregularResult:
    """Compute Sobol indices via polynomial chaos expansion (PCE).

    Fits an orthogonal polynomial surrogate to arbitrary (X, Y) pairs, so no
    structured sampling is required. First-, total-, and second-order Sobol
    indices are then read directly off the expansion coefficients
    (Sudret, 2008), with no extra model evaluations. Compared with Monte
    Carlo Sobol estimators, PCE needs far fewer samples when the model
    response is smooth. It only captures effects the polynomial can
    represent, so check ``PCEResult.cv_rmse`` before trusting the indices.

    Args:
        problem: Parameter names and distributions.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)`` scalar, ``(N, K)`` multi-output, or
            ``(N, T, K)`` time-series. All slices share one basis and are
            fitted in a single multi-right-hand-side solve. Indices are
            computed independently per (t, k) slice.
        order: Maximum total polynomial degree. Higher orders capture
            sharper nonlinearity and higher-order interactions, but the
            term count C(D+order, order) grows fast and needs more samples
            to fit. Automatically reduced (with a warning) if the term
            count would exceed ``fit_ratio * N``.
        ridge: Tikhonov regularization for the least-squares fit. The tiny
            default only guards against a singular normal matrix. Increase
            it if coefficients look unstable (noisy Y, near-duplicate rows).
        fit_ratio: Maximum ratio of terms to samples before ``order`` is
            reduced. Lower values demand more samples per term (a more
            conservative, less overfit-prone fit). Must be at most 1, and
            must leave room for the order-1 expansion:
            ``D + 1 <= int(fit_ratio * N)``.
        batch_size: Rows of ``X``/``Y`` processed per batch during the fit
            (the package-wide ``batch_size`` convention): it sizes row
            blocks, clamped to ``N``. ``None`` (default) keeps the
            single-shot fit unless its estimated resident memory (design
            matrix, Gram factorization, LOO residuals) exceeds the active
            memory budget (~512 MiB, see
            ``jaxgsa.config.set_memory_budget``), in which case the fit
            streams over auto-sized row batches. An explicit
            ``batch_size < N`` streams over row blocks of that many rows;
            ``batch_size >= N`` is one full block, which is the single-pass
            fit. Both fit paths solve the same normal equations and compute
            the same exact leave-one-out error; results differ only at the
            level of float32 summation order.
        n_bootstrap: Number of bootstrap resamples for confidence intervals.
            ``0`` (default) disables them. The resampling unit is one row of
            ``(X, Y)``.

            This is the expensive kind of bootstrap. Every replicate refits
            the whole expansion — a fresh design matrix, a fresh Gram solve —
            so the cost is an order of magnitude above the row resample of a
            method such as ``jaxgsa.pawn``, which only re-reduces numbers it
            already has. Twenty replicates cost about twenty fits. That is
            why the default is ``0``, and why a small ``n_bootstrap`` is the
            usual choice here.

            The interval measures the sampling variability of ``(X, Y)``,
            propagated through the fit. It does not measure truncation
            error: an expansion too coarse to represent the model gives a
            biased index, and every replicate inherits the same bias, so the
            interval stays narrow and wrong. ``cv_rmse`` and
            ``explained_variance`` are what report that.
        conf_level: Two-sided confidence level for the intervals.
        ci_method: How the interval endpoints are formed. ``"quantile"``
            (default) reads them off the empirical bootstrap distribution.
            ``"gaussian"`` centres them on the point estimate and takes
            ``+/- z * sd`` of the draws, which is smoother for a small
            ``n_bootstrap`` but assumes the draws are normal.
        key: A ``jax.random`` key for the bootstrap resampling. Required when
            ``n_bootstrap > 0``. Pass ``jax.random.key(0)`` if you have an
            integer seed.
        on_invalid: What to do about non-finite values in ``X`` or ``Y``.
            One row is one unit here, so ``"drop"`` removes the affected
            ``(X, Y)`` pairs and fits on the rest. See
            :mod:`jaxgsa._core.invalid`. The check matters for PCE: a single
            NaN reaches the normal equations and poisons every coefficient,
            every leave-one-out RMSE and every index.
        verbose: If ``True`` (default), print a short summary to stdout: the
            problem and the data, the wall-clock timing, and the top
            parameters by ``ST``. Pass ``False`` for a silent run.
        keep_replicates: Retain the per-replicate index arrays on
            ``result.ci``. Off by default: the draws are large.

    Returns:
        PCEResult holding S1 and ST, shape ``(D,)`` / ``(K, D)`` /
        ``(T, K, D)``, and S2, shape ``(D, D)`` / ``(K, D, D)`` /
        ``(T, K, D, D)``, with leading output and time dims mirroring ``Y``.
        It also carries the fitted coefficients and multi-index (reused by
        ``result.predict``), the effective ``order``, the per-slice
        leave-one-out RMSE goodness-of-fit diagnostic, and the confidence
        intervals when ``n_bootstrap > 0``.

    Raises:
        ValueError: If ``X`` fails validation against ``problem``, ``Y``'s
            layout cannot be resolved against ``X``'s row count,
            ``batch_size`` is given and not a positive integer,
            ``problem.correlation`` declares a dependence structure (the
            Wiener-Askey basis is orthogonal only under independent
            inputs), or ``problem`` has categorical parameters (a
            polynomial in an unordered level code has no meaning),
            ``on_invalid`` is not one of the three policies,
            ``on_invalid="raise"`` (the default) and ``X`` or ``Y`` holds a
            non-finite value, ``n_bootstrap > 0`` without a ``key``,
            ``fit_ratio`` is above 1, or too few rows survive
            ``on_invalid`` for even the order-1 expansion
            (``D + 1 > int(fit_ratio * N)``, which would make the fit
            underdetermined).
    """
    key = as_key(key, "pce")
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

    from jaxgsa.pce import _SPEC

    # The non-finite check runs here, in the public entry point only.
    # Everything downstream (`_fit_pce_core`, and Shapley routing through
    # `PCEResult.shapley`) then works on data the policy has already passed,
    # so the policy is applied exactly once per user call.
    ctx = prepare(
        _SPEC,
        problem,
        Y,
        X=X,
        on_invalid=on_invalid,
        checks=(
            at_least("order", order, 1),
            at_least("ridge", ridge, 0.0),
            at_least("fit_ratio", fit_ratio, 0.0),
            require(fit_ratio <= 1.0, f"fit_ratio must be <= 1, got {fit_ratio}"),
            at_least("batch_size", batch_size, 1),
            *bootstrap_checks(n_bootstrap, ci_method, conf_level, key),
        ),
        min_kept=_MIN_ROWS,
    )
    assert ctx.X is not None
    X, Y, invalid = ctx.X, ctx.Y, ctx.invalid

    # M2: an underdetermined fit (fewer rows than even the order-1 expansion
    # has terms) used to pass silently and hand back a plausible-looking but
    # meaningless index. N is the row count after `on_invalid` has run, so
    # this reads the fit's real sample budget.
    _D = problem.n_params
    _N = X.shape[0]
    _min_terms = _D + 1
    if _min_terms > int(fit_ratio * _N):
        raise ValueError(
            f"fit_ratio={fit_ratio} and N={_N} rows (after invalid-row handling) "
            f"cannot fit even the order-1 expansion ({_min_terms} terms for "
            f"D={_D} parameters); raise fit_ratio or provide more rows"
        )

    t0 = _verbose.tic()
    fit = _fit_pce_core(
        problem, X, Y, order=order, ridge=ridge, fit_ratio=fit_ratio, batch_size=batch_size
    )
    T, K = fit.coefficients.shape[:2]
    if fit.order < order:
        # The fit is coarser than requested; surface it so callers do not
        # mistake a truncated expansion for the order they asked for. The
        # warning lives here rather than in the fit, so that `indices` stays
        # free of host-side side effects. `effective_order` answers the same
        # question without a fit.
        warnings.warn(
            f"jaxgsa.pce: PCE order reduced from {order} to {fit.order} to keep the "
            f"term count within the sample budget (fit_ratio={fit_ratio}, N={X.shape[0]})",
            stacklevel=2,
            category=JaxgsaWarning,
        )

    switched = _order_forced_legendre(problem, fit.order)
    if switched:
        warnings.warn(
            f"jaxgsa.pce: order {fit.order} is above {_MAX_HERMITE_ORDER_UNDER_TRUNCATION}, "
            f"so the truncated Gaussian input(s) {switched} use a Legendre basis "
            "through the truncated CDF instead of Hermite. That basis fits a "
            "Gaussian-shaped input less accurately and can degrade the indices. "
            f"Use order <= {_MAX_HERMITE_ORDER_UNDER_TRUNCATION} to keep Hermite.",
            stacklevel=2,
            category=JaxgsaWarning,
        )

    # Sobol indices are extracted analytically from the coefficients
    # (Sudret 2008), batched over all slices: no extra sampling, no loops.
    # Per-slice LOO RMSE (cv_rmse) and in-sample RMSE are goodness-of-fit
    # diagnostics, computed by the fit path (single-pass hat-matrix diagonal,
    # or the streamed equivalent); the leverage is shared by every slice.
    # _fit_pce_core ran with its default diagnostics=True here, so both
    # fields are populated. The extraction is two jitted calls; the warnings
    # below then read their outputs on the host.
    assert fit.loo_flat is not None
    assert fit.rmse_flat is not None
    S1, ST, S2 = _sobol_from_fit(
        fit.coefficients,
        ctx.Y3,
        D=problem.n_params,
        order=fit.order,
        budget=get_memory_budget(),
    )  # S1/ST (T,K,D), S2 (T,K,D,D)
    loo, rmse, explained_variance, y_std = _fit_summary(fit.loo_flat, fit.rmse_flat, ctx.Y3)

    # Non-finite numbers come first: the two checks below skip them, so a
    # failed Cholesky would otherwise pass as a clean fit.
    # Skipped when non-finite rows were let through (`on_invalid="propagate"`):
    # NaN coefficients are then the documented outcome, already warned about,
    # and not a sign of a singular Gram matrix.
    if bool(jnp.all(jnp.isfinite(X)) & jnp.all(jnp.isfinite(ctx.Y3))):
        check_fit_finite(
            "jaxgsa.pce",
            fit.coefficients,
            loo,
            rmse,
            y_std,
            advice=(
                "Enable jax x64 (jax.config.update('jax_enable_x64', True)), lower "
                "order, or remove duplicated or constant inputs."
            ),
        )

    # Both fit-quality checks run here, in the public entry point only, so
    # that `indices` stays free of host-side side effects and one analyze call
    # speaks once. They are the difference between an index that is wrong and
    # an index that says it is wrong: `order` defaults to 3, which does not
    # resolve a strong nonlinearity, and the indices of a cubic that is not
    # this model look exactly like the indices of this model.
    warn_variance_fit(
        "jaxgsa.pce",
        explained_variance,
        quantity="explained_variance",
        advice_low=(
            "Raise order until cv_rmse stops falling, or use jaxgsa.hdmr for a "
            "response with kinks a polynomial cannot follow."
        ),
        # Unreachable for PCE, whose diagnostic is a true in-sample R-squared
        # and so cannot pass 1. Passed for the shared signature.
        advice_high="Check the output for a near-constant slice.",
    )
    warn_cv_overfit(
        "jaxgsa.pce",
        loo,
        y_std,
        explained_variance,
        advice="Refit with a lower order or more samples.",
    )

    # Drop the singleton axes _prepare_Y inserted. S1/ST/coeffs end in
    # (T, K, per-slice); S2 carries an extra trailing D and loo has no trailing
    # per-slice axis, so they pass n_trailing=2 and 0 respectively.
    S1_conf: Array | None = None
    ST_conf: Array | None = None
    S2_conf: Array | None = None
    ci: CIInfo | None = None
    if n_bootstrap > 0:
        assert key is not None  # checked above, before the fit ran
        S1_draws, ST_draws, S2_draws = _bootstrap_indices(
            problem,
            X,
            ctx.Y3,
            key,
            n_bootstrap=n_bootstrap,
            order=order,
            ridge=ridge,
            fit_ratio=fit_ratio,
            batch_size=batch_size,
        )
        # Squeeze before calling interval(): it reads no shape of its own, so
        # points and draws must already be at the layout the result reports
        # (the leading replicate axis on the draws survives the squeeze,
        # which addresses the T/K axes from the end).
        points = {
            "S1": ctx.squeeze(S1),
            "ST": ctx.squeeze(ST),
            "S2": ctx.squeeze(S2, n_trailing=2),
        }
        draws_sq = {
            "S1": ctx.squeeze(S1_draws),
            "ST": ctx.squeeze(ST_draws),
            "S2": ctx.squeeze(S2_draws, n_trailing=2),
        }
        field_conf, ci = interval(
            points,
            draws_sq,
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            keep_replicates=keep_replicates,
        )
        S1_conf, ST_conf, S2_conf = field_conf["S1"], field_conf["ST"], field_conf["S2"]

    S1 = ctx.squeeze(S1)
    ST = ctx.squeeze(ST)
    coeffs = ctx.squeeze(fit.coefficients)
    S2 = ctx.squeeze(S2, n_trailing=2)
    loo = ctx.squeeze(loo, n_trailing=0)
    rmse = ctx.squeeze(rmse, n_trailing=0)
    explained_variance = ctx.squeeze(explained_variance, n_trailing=0)

    result = PCEResult(
        S1=S1,
        ST=ST,
        S2=S2,
        problem=problem,
        coefficients=coeffs,
        multi_index=fit.multi_index,
        order=fit.order,
        rmse=rmse,
        cv_rmse=loo,
        explained_variance=explained_variance,
        streamed=fit.streamed,
        invalid=invalid,
        S1_conf=S1_conf,
        ST_conf=ST_conf,
        S2_conf=S2_conf,
        ci=ci,
    )

    if verbose:
        elapsed = _verbose.stop(t0, result.ST)
        batch_note = (
            f"batch_size: {batch_size} (user-set)"
            if batch_size is not None
            else "batch_size: auto (resolved from the memory budget)"
        )
        _verbose.analysis_summary(
            method="jaxgsa.pce.analyze",
            problem=problem,
            n_runs=int(ctx.Y.shape[0]),
            T=T,
            K=K,
            invalid=invalid,
            timings=[("fit + estimator (includes compile on the first call)", elapsed)],
            notes=[
                f"order: {order}",
                f"fit: {'streamed' if result.streamed else 'single-pass'}",
                batch_note,
            ],
            index_name="ST",
            values=result.ST,
            conf=result.ST_conf,
        )
    return result


def _pce_predict_plan(result: PCEResult, X_new: Array) -> _PredictPlan:
    """Build the prediction plan behind :meth:`PCEResult.predict`.

    Maps the (already validated) inputs to the polynomial reference domain.
    It then packages the per-row transient cost together with a kernel that
    rebuilds the basis and contracts it with the fitted coefficients. The
    shared template in :class:`jaxgsa._core.surrogate.SurrogateResult` runs
    the kernel in row batches sized against a transient-memory budget.
    """
    X_ref, input_types = _map_to_reference(X_new, result.problem, result.order)
    n_terms = result.multi_index.shape[0]
    # Transient footprint per row: build_design_matrix accumulates a running
    # product, so at most the (batch, n_terms) accumulator, one gathered
    # (batch, n_terms) factor, and the multiply output are live at once
    # (~3 * n_terms floats per row), plus the einsum output slices.
    slices = math.prod(result.coefficients.shape[:-1])
    bytes_per_row = X_ref.dtype.itemsize * (3 * n_terms + slices)

    def _predict(X_chunk: Array) -> Array:
        Phi = build_design_matrix(X_chunk, result.multi_index, input_types, result.order)
        # Prediction contracts the term axis (last on coefficients) for every
        # output slice at once: Y = Phi @ c per slice, one einsum.
        return jnp.einsum("nt,...t->n...", Phi, result.coefficients)

    return _PredictPlan(X=X_ref, bytes_per_row=bytes_per_row, kernel=_predict)
