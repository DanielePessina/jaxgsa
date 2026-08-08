"""Shared Shapley pipeline tail and the convenience ``analyze`` wrapper.

``PCEResult.shapley`` and ``HDMRResult.shapley`` each supply their own
variance decomposition (per-term partial variances plus term membership) and
delegate the common tail -- normalization, Shapley allocation, fit-quality
warning, result construction -- to :func:`_shapley_result_from_variances`.
:func:`analyze` is a thin convenience over those result methods.
"""

from __future__ import annotations

import inspect
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.validation import _raise_categorical_analysis
from jaxgsa.shapley._engine import shapley_from_variances
from jaxgsa.shapley._result import ShapleyResult

if TYPE_CHECKING:
    from jaxgsa.problem import Problem

_POORFIT_THRESHOLD = 0.5
_OVERFIT_THRESHOLD = 1.3


def _external_stacklevel(default: int = 2) -> int:
    """``warnings`` stacklevel of the first caller frame outside jaxgsa.

    The Shapley fit-quality warning is emitted from a shared tail reached by
    two call chains of different depth (``result.shapley()`` directly vs. via
    the ``jaxgsa.shapley.analyze`` wrapper, which adds a frame), so no fixed
    ``stacklevel`` points at the user's frame in both. Walking out of the
    package locates it regardless of chain length.

    Args:
        default: Fallback level for interpreters without frame introspection.

    Returns:
        The ``stacklevel`` that attributes a warning issued by the immediate
        caller to the nearest frame outside the ``jaxgsa`` package.
    """
    import jaxgsa

    frame = inspect.currentframe()
    if frame is None:  # pragma: no cover - non-CPython fallback
        return default
    pkg_root = os.path.dirname(jaxgsa.__file__)
    frame = frame.f_back  # the frame that will call warnings.warn
    level = 1
    while frame is not None:
        if not frame.f_code.co_filename.startswith(pkg_root):
            return level
        frame = frame.f_back
        level += 1
    return level


def _normalize_partial_variances(
    partial: Array,
    explained_variance: Array,
    total: Array,
) -> Array:
    """Normalize term contributions while preserving degenerate output slices.

    Args:
        partial: Per-term variance contributions, shape ``(..., n_terms)``.
        explained_variance: Per-slice fit diagnostic, shape ``(...,)``;
            non-finite slices are propagated as NaN.
        total: Pre-computed ``partial.sum(axis=-1)``, shape ``(...,)``.
            Passed in so callers that already hold the sum do not trigger a
            recomputation.

    Returns:
        ``partial / total`` with degenerate slices (zero or non-finite total,
        non-finite explained variance) set to NaN.
    """
    total = jnp.asarray(total)[..., None]
    invalid = (
        (total == 0)
        | ~jnp.isfinite(total)
        | ~jnp.isfinite(jnp.asarray(explained_variance)[..., None])
    )
    return jnp.where(invalid, jnp.nan, partial / jnp.where(invalid, 1.0, total))


def _warn_pathological_fit(explained_variance: Array) -> None:
    """Warn when a surrogate captured implausibly little or too much variance.

    Reached from two call chains of different depth (``result.shapley()`` and
    the ``jaxgsa.shapley.analyze`` wrapper), so the warning's ``stacklevel`` is
    resolved dynamically by :func:`_external_stacklevel` to attribute it to the
    user's frame in either case.
    """
    ev = jnp.asarray(explained_variance)
    if bool(jnp.any(ev > _OVERFIT_THRESHOLD)):
        warnings.warn(
            f"jaxgsa: surrogate explained_variance exceeds {_OVERFIT_THRESHOLD}; "
            "Shapley effects may be unreliable",
            stacklevel=_external_stacklevel(),
        )
    elif bool(jnp.any(ev < _POORFIT_THRESHOLD)):
        warnings.warn(
            f"jaxgsa: surrogate explained_variance is below {_POORFIT_THRESHOLD}; "
            "Shapley effects may be unreliable",
            stacklevel=_external_stacklevel(),
        )


def _shapley_result_from_variances(
    partial: Array,
    membership: np.ndarray,
    explained: Array,
    *,
    total: Array,
    problem: "Problem",
    backend: Literal["hdmr", "pce"],
    order: int,
    include_correlative: bool = False,
) -> ShapleyResult:
    """Finish a Shapley analysis from a surrogate's variance decomposition.

    The single shared tail of both ``shapley()`` result methods: normalize
    the partial variances so the modelled terms sum to 1, allocate each
    term's share equally among its participants, warn on a pathological fit,
    and assemble the result. Callers perform their fitted-state precheck and
    decomposition before delegating here.

    Args:
        partial: Per-term variance contributions, shape ``(..., n_terms)``.
        membership: ``(n_terms, D)`` boolean matrix marking which parameters
            participate in each term.
        explained: Per-slice explained-variance diagnostic, shape ``(...,)``.
        total: Pre-computed ``partial.sum(axis=-1)`` (for HDMR this is the
            same array as ``explained``, so passing it avoids a recompute).
        problem: Problem definition carried onto the result.
        backend: Surrogate backend label, ``"hdmr"`` or ``"pce"``.
        order: Effective surrogate order actually used.
        include_correlative: Whether the correlative ANCOVA part was folded
            into ``partial`` (HDMR only).

    Returns:
        ShapleyResult with ``Sh``/``S1``/``ST``, the diagnostic, and
        provenance fields.

    Warns:
        UserWarning: If ``explained`` flags a pathological fit (well below
            1, or above 1 -- overfit).
    """
    normalized = _normalize_partial_variances(partial, explained, total)
    Sh, S1, ST = shapley_from_variances(normalized, membership)
    _warn_pathological_fit(explained)
    return ShapleyResult(
        Sh=Sh,
        S1=S1,
        ST=ST,
        problem=problem,
        backend=backend,
        explained_variance=explained,
        order=order,
        include_correlative=include_correlative,
    )


def analyze(
    problem: "Problem",
    X: Array,
    Y: Array,
    *,
    backend: Literal["pce", "hdmr"] = "pce",
    include_correlative: bool = False,
    **backend_kwargs: Any,
) -> ShapleyResult:
    """Fit a surrogate and return its Shapley effects (convenience wrapper).

    This is literally ``jaxgsa.pce.analyze(problem, X, Y, **kw).shapley()`` /
    ``jaxgsa.hdmr.analyze(problem, X, Y, **kw).shapley(include_correlative=...)``
    depending on ``backend`` -- there is no separate Shapley pipeline. Prefer
    the two-step form when you also want the fitted result (Sobol indices,
    ``predict``, fit diagnostics); use this wrapper when only the Shapley
    effects are needed.

    Args:
        problem: Parameter names and distributions.
        X: (N, D) input samples.
        Y: Model outputs -- (N,) scalar, (N, K) multi-output, or (N, T, K)
            time-series.
        backend: Surrogate providing the variance decomposition. ``"pce"``
            (default) reads subset variances off orthonormal polynomial
            coefficients; ``"hdmr"`` fits B-spline component functions and
            additionally separates correlation-induced variance.
        include_correlative: HDMR-only; fold the correlative ANCOVA part
            (``Sb``) into the allocation. See ``HDMRResult.shapley``.
        **backend_kwargs: Passed through unchanged to the selected backend's
            ``analyze`` (e.g. ``order``/``ridge``/``fit_ratio`` for PCE,
            ``maxorder``/``m``/``lambdax`` for HDMR).

    Returns:
        ShapleyResult, exactly as returned by the corresponding result
        method.

    Raises:
        ValueError: If ``backend`` is unknown, ``include_correlative`` is
            requested with the PCE backend, ``problem.correlation`` declares
            a dependence structure with the PCE backend (use
            ``backend="hdmr"`` with ``include_correlative=True``),
            ``problem`` has categorical parameters (both backends fit a
            smooth surrogate over the inputs, which is undefined for
            unordered level codes), or the underlying ``analyze`` rejects
            its inputs.
        TypeError: If ``backend_kwargs`` contains a keyword the selected
            backend's ``analyze`` does not accept.
    """
    # Both backends fit smooth surrogates over the inputs, which is undefined
    # for unordered level codes; reject with the Shapley-specific name.
    _raise_categorical_analysis(problem, "jaxgsa.shapley.analyze")
    if backend == "pce":
        if include_correlative:
            raise ValueError("include_correlative requires backend='hdmr'")
        if problem.has_correlated_inputs:
            # The delegated pce.analyze would reject the problem anyway; this
            # guard names the Shapley-specific alternative in the message.
            raise ValueError(
                "jaxgsa.shapley.analyze with backend='pce' computes a variance "
                "allocation that assumes independent inputs, but "
                "problem.correlation declares a dependence structure. Use "
                "backend='hdmr' with include_correlative=True, which allocates "
                "the ANCOVA (structural + correlative) decomposition instead — "
                "an ANCOVA-based attribution, not conditional-variance Shapley "
                "effects — or a correlation-tolerant method (jaxgsa.vkoga and "
                "jaxgsa.kucherenko for variance-based indices, "
                "jaxgsa.optimal_transport, jaxgsa.borgonovo, jaxgsa.hsic, "
                "jaxgsa.pawn). Those methods do not return Shapley effects."
            )
        from jaxgsa.pce import analyze as analyze_pce

        return analyze_pce(problem, X, Y, **backend_kwargs).shapley()
    if backend == "hdmr":
        from jaxgsa.hdmr import analyze as analyze_hdmr

        return analyze_hdmr(problem, X, Y, **backend_kwargs).shapley(
            include_correlative=include_correlative
        )
    raise ValueError(f"backend must be 'pce' or 'hdmr', got {backend!r}")
