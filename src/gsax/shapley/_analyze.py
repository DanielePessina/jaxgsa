"""Shapley-effect analysis entry point.

Computes global Shapley effects (Owen 2014; Song, Nelson & Staum 2016)
analytically from a fitted surrogate's variance decomposition -- either
RS-HDMR component-function variances or PCE coefficients -- with no
permutation Monte Carlo. Assumes independent inputs.
"""

from __future__ import annotations

import warnings
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax import Array

from gsax._normalization import (
    _prepare_Y,
    _squeeze_output_axes,
    _validate_xy_inputs,
    _warn_zero_variance_slices,
)
from gsax.hdmr._analyze import _analyze_hdmr_core
from gsax.pce._analyze import _fit_pce_core
from gsax.problem import Problem
from gsax.shapley._engine import build_membership, shapley_from_variances
from gsax.shapley._result import ShapleyResult

# explained_variance outside [_POORFIT, _OVERFIT] flags an untrustworthy fit:
# below -> much of Var(Y) is unexplained (truncation/poor fit); above -> the
# surrogate's partial variances over-count Var(Y) (typically overfitting).
# The Shapley effects still sum to 1 either way.
_POORFIT_THRESHOLD = 0.5
_OVERFIT_THRESHOLD = 1.3

# Backend defaults, applied when the corresponding kwarg is left as None.
# They mirror the signatures of analyze_hdmr / analyze_pce exactly.
_HDMR_DEFAULTS: dict = {
    "prenormalize": False,
    "maxorder": 2,
    "maxiter": 100,
    "m": 2,
    "lambdax": 0.01,
    "chunk_size": 2048,
}
_PCE_DEFAULTS: dict = {"order": 3, "ridge": 1e-8, "fit_ratio": 0.5}


def _resolve_backend_kwargs(
    backend: str,
    hdmr_kwargs: dict,
    pce_kwargs: dict,
) -> dict:
    """Validate per-backend kwargs and fill in the backend's defaults.

    Args:
        backend: Selected backend, ``"hdmr"`` or ``"pce"``.
        hdmr_kwargs: HDMR-only kwargs as passed by the user (None = unset).
        pce_kwargs: PCE-only kwargs as passed by the user (None = unset).

    Returns:
        The selected backend's kwargs with None entries replaced by defaults.

    Raises:
        ValueError: If ``backend`` is unknown, or a kwarg belonging to the
            non-selected backend was explicitly set.
    """
    if backend == "hdmr":
        selected, foreign, defaults = hdmr_kwargs, pce_kwargs, _HDMR_DEFAULTS
        foreign_backend = "pce"
    elif backend == "pce":
        selected, foreign, defaults = pce_kwargs, hdmr_kwargs, _PCE_DEFAULTS
        foreign_backend = "hdmr"
    else:
        raise ValueError(f"backend must be 'hdmr' or 'pce', got {backend!r}")

    bad = sorted(k for k, v in foreign.items() if v is not None)
    if bad:
        raise ValueError(
            f"{bad} only apply to backend='{foreign_backend}', but backend='{backend}' "
            "was selected"
        )

    return {k: (defaults[k] if v is None else v) for k, v in selected.items()}


def analyze_shapley(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    backend: Literal["hdmr", "pce"] = "pce",
    include_correlative: bool = False,
    prenormalize: bool | None = None,
    maxorder: int | None = None,
    maxiter: int | None = None,
    m: int | None = None,
    lambdax: float | None = None,
    chunk_size: int | None = None,
    order: int | None = None,
    ridge: float | None = None,
    fit_ratio: float | None = None,
) -> ShapleyResult:
    """Compute Shapley effects analytically from a fitted surrogate.

    Fits an RS-HDMR or PCE surrogate to arbitrary (X, Y) pairs -- no
    structured sampling required -- and allocates the output variance
    across inputs via the exact Shapley formula
    ``Sh_i = sum_{u : i in u} V_u / |u|``, where ``V_u`` are the
    surrogate's per-subset partial variances. No permutation sampling
    is involved.

    By default inputs are assumed independent. Under independence the Shapley
    effects satisfy ``S1_i <= Sh_i <= ST_i`` and split each interaction's
    variance equally among its participants. Indices are normalized by the
    surrogate's total decomposed variance ``sum_u V_u``, so ``Sh`` sums to
    exactly 1 (the efficiency property; Owen 2014). How much of ``Var(Y)``
    the surrogate captured is reported separately as
    ``ShapleyResult.explained_variance`` (close to 1 for a good fit, below 1
    when truncation leaves variance unexplained, above 1 when an overfit
    surrogate over-counts shared variance); a ``UserWarning`` is emitted when
    it is far from 1. Interactions beyond the surrogate's truncation order
    (``maxorder`` / ``order``) are absent from the allocation.

    Pass ``include_correlative=True`` (HDMR backend only) to additionally
    credit the correlative ANCOVA variance ``Sb = Cov(f_j, sum_{k!=j} f_k) /
    Var(Y)`` (Li et al. 2010), so the allocation reflects correlation *present
    in the supplied ``X`` samples* -- no joint distribution is specified; it is
    read empirically from the data. Two consequences follow: individual indices
    may then be **negative** and the ordering ``S1 <= Sh <= ST`` need not hold
    (both expected under dependence, not bugs), while efficiency (``Sh`` sums to
    1) is preserved; and ``explained_variance`` becomes the true emulator
    ``Var(Y_hat)/Var(Y)`` rather than the structural sum ``sum_j Sa_j``, which
    over-counts under correlation. This is the given-data ANCOVA notion, not the
    conditional-variance Shapley estimator (Song, Nelson & Staum 2016); the two
    differ once inputs are correlated, so read the correlated indices as a
    variance attribution, not as conditional-expectation Shapley values.

    For ``backend="pce"`` this normalization coincides with ``analyze_pce``,
    so ``S1``/``ST`` match it exactly. For ``backend="hdmr"`` the indices
    differ from ``analyze_hdmr``'s (which normalize by ``Var(Y)``) by a
    factor of ``explained_variance``; the total-order ``ST`` is built from the
    structural ANCOVA terms only and excludes the correlative ``Sb`` part
    (zero under the independence assumption).

    The ``"hdmr"`` backend inherits ``analyze_hdmr``'s input contract: it
    requires at least 300 samples and ``maxorder`` in ``{1, 2, 3}`` (clamped
    with a warning when ``D < maxorder``).

    Args:
        problem: Parameter names and distributions.
        X: (N, D) input samples.
        Y: Model outputs — (N,) scalar, (N, K) multi-output, or (N, T, K)
            time-series; both backends accept all three, and indices are
            computed independently per (t, k) slice.
        backend: Surrogate providing the variance decomposition.
            ``"pce"`` (default) reads subset variances off orthonormal
            polynomial coefficients (Sudret, 2008) — exact for the fitted
            polynomial; ``"hdmr"`` fits B-spline component functions and
            additionally separates correlation-induced variance.
        include_correlative: When ``True`` (requires ``backend="hdmr"``), fold
            the correlative ANCOVA variance ``Sb`` into the allocation so the
            indices reflect correlation in the supplied ``X`` (see above).
            Defaults to False (independent-input allocation).
        prenormalize: HDMR-only; see ``analyze_hdmr``. Defaults to False.
        maxorder: HDMR-only; maximum expansion order (1-3). Defaults to 2.
        maxiter: HDMR-only; backfitting iterations. Defaults to 100.
        m: HDMR-only; B-spline intervals per dimension. Defaults to 2.
        lambdax: HDMR-only; Tikhonov regularization. Defaults to 0.01.
        chunk_size: HDMR-only; output slices per vmap batch. Defaults to 2048.
        order: PCE-only; maximum total polynomial degree. Defaults to 3.
        ridge: PCE-only; Tikhonov regularization. Defaults to 1e-8.
        fit_ratio: PCE-only; maximum terms-to-samples ratio. Defaults to 0.5.

    Returns:
        ShapleyResult with ``Sh`` (summing to 1), the bracketing ``S1``/``ST``
        from the same decomposition, ``explained_variance``, the effective
        surrogate ``order``, the problem, and the backend name.

    Raises:
        ValueError: If ``backend`` is unknown, a kwarg belonging to the
            non-selected backend is explicitly set, ``include_correlative`` is
            set with a non-HDMR backend, or ``Y``'s layout cannot be resolved
            against ``X``'s row count.
    """
    if include_correlative and backend != "hdmr":
        raise ValueError(
            "include_correlative=True requires backend='hdmr'; the PCE backend "
            "assumes independent inputs and has no correlative (Sb) term."
        )

    resolved = _resolve_backend_kwargs(
        backend,
        hdmr_kwargs={
            "prenormalize": prenormalize,
            "maxorder": maxorder,
            "maxiter": maxiter,
            "m": m,
            "lambdax": lambdax,
            "chunk_size": chunk_size,
        },
        pce_kwargs={"order": order, "ridge": ridge, "fit_ratio": fit_ratio},
    )

    # Resolve Y to the canonical layout ONCE, here, so total_var below agrees
    # with the fit cores (which we call directly, skipping their re-validation).
    Y, _ = _validate_xy_inputs(problem, jnp.asarray(X), jnp.asarray(Y))

    # Per-output-slice variance of the raw outputs, used to normalize the PCE
    # explained fraction and to flag constant (zero-variance) slices uniformly.
    total_var = jnp.var(Y, axis=0)
    # Warn once here for both backends; the cores below do not warn.
    _warn_zero_variance_slices(_prepare_Y(Y)[0], output_names=problem.output_names)

    if backend == "hdmr":
        result = _analyze_hdmr_core(problem, jnp.asarray(X), Y, **resolved)
        emulator = result.emulator
        if emulator is None:  # pragma: no cover - the core always sets it
            raise RuntimeError("HDMR core returned no emulator")
        # HDMR terms enumerate [singles, pairs, triples]; the emulator holds
        # the exact pair/triple index tuples used (post maxorder clamping).
        subsets: list[tuple[int, ...]] = [(i,) for i in range(problem.num_vars)]
        subsets += [tuple(u) for u in emulator["c2"]]
        subsets += [tuple(u) for u in emulator["c3"]]
        membership = build_membership(subsets, problem.num_vars)
        # Sa_j = Var(f_j)/Var(Y): partial variances already divided by Var(Y),
        # so their sum over terms is the explained-variance fraction directly.
        # No zero-variance guard needed here: _ancova already emits NaN Sa for
        # constant output slices, which propagates through the sum.
        partial = result.Sa
        if include_correlative:
            # Fold in the correlative ANCOVA share Sb (per term, possibly
            # negative). Sa_j + Sb_j = Cov(f_j, Y_hat)/Var(Y), so the sum over
            # terms telescopes to Var(Y_hat)/Var(Y) -- the true emulator R2,
            # which is the correct explained fraction under correlation (the
            # structural sum sum_j Sa_j double-counts). Sb aligns term-for-term
            # with Sa (same _ancova call, same emulator columns).
            partial = partial + result.Sb
        explained_variance = partial.sum(axis=-1)
        effective_order = emulator["maxorder"]
    else:
        # Fit-only core: skips the S2 einsum and LOO diagnostic that analyze_pce
        # would compute and Shapley would discard. Squeeze the coefficients with
        # the fit's flags so their leading dims match total_var.
        fit = _fit_pce_core(problem, jnp.asarray(X), Y, **resolved)
        coeffs = _squeeze_output_axes(fit.coefficients, fit.squeeze_time, fit.squeeze_output)
        # Orthonormality makes each squared coefficient a partial variance;
        # the constant term (index 0 on the trailing term axis) carries none.
        # multi_index[1:] > 0 IS the membership matrix, so no tuple
        # round-trip is needed. coefficients are terms-last, so this and the
        # shared normalization below are batched over any leading slice dims.
        partial = coeffs[..., 1:] ** 2
        membership = np.asarray(fit.multi_index[1:] > 0)
        explained_variance = jnp.where(total_var == 0, jnp.nan, partial.sum(axis=-1) / total_var)
        effective_order = fit.order

    # Normalize the partial variances to sum to 1 so Sh satisfies the Shapley
    # efficiency property (Owen 2014); explained_variance carries the
    # fit-quality signal separately. A constant output slice (total_var == 0,
    # which a ridge-regularized PCE fit leaves tiny-but-nonzero) or a degenerate
    # decomposition (sum 0 / NaN) yields NaN indices, matching the backends.
    v_total = partial.sum(axis=-1, keepdims=True)
    degenerate = (v_total == 0) | ~jnp.isfinite(v_total) | (total_var[..., None] == 0)
    V = jnp.where(degenerate, jnp.nan, partial / jnp.where(degenerate, 1.0, v_total))

    Sh, S1, ST = shapley_from_variances(V, membership)
    _warn_pathological_fit(explained_variance)

    return ShapleyResult(
        Sh=Sh,
        S1=S1,
        ST=ST,
        problem=problem,
        backend=backend,
        explained_variance=explained_variance,
        order=int(effective_order),
        include_correlative=include_correlative,
    )


def _warn_pathological_fit(explained_variance: Array) -> None:
    """Warn when the surrogate fit is too poor or overfit to trust the indices.

    Args:
        explained_variance: ``sum_u V_u / Var(Y)`` per output slice. NaN slices
            (constant output) already warned via the backend and do not
            re-trigger here, since NaN comparisons are False.
    """
    ev = jnp.asarray(explained_variance)
    if bool(jnp.any(ev > _OVERFIT_THRESHOLD)):
        warnings.warn(
            f"gsax: surrogate explained_variance exceeds {_OVERFIT_THRESHOLD} "
            "(partial variances over-count Var(Y), typically overfitting); "
            "Shapley effects still sum to 1 but may be unreliable.",
            stacklevel=3,
        )
    elif bool(jnp.any(ev < _POORFIT_THRESHOLD)):
        warnings.warn(
            f"gsax: surrogate explained_variance is below {_POORFIT_THRESHOLD} "
            "(much of Var(Y) is unexplained); Shapley effects may be unreliable.",
            stacklevel=3,
        )
