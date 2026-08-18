"""Shared Shapley pipeline tail and the convenience ``analyze`` wrapper.

``PCEResult.shapley`` and ``HDMRResult.shapley`` each supply their own
variance decomposition, then delegate the common tail to
:func:`jaxgsa.shapley._engine._shapley_result_from_variances`. That tail lives
in the engine, not here, so a sibling method package importing it does not have
to reach into another package's ``_analyze``. :func:`analyze` is a thin
convenience over those result methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from jax import Array

from jaxgsa._core.entry import gates
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.validation import _correlation_tolerant_methods
from jaxgsa.shapley._result import ShapleyResult

if TYPE_CHECKING:
    from jaxgsa.problem import Problem


def analyze(
    problem: "Problem",
    X: Array,
    Y: Array,
    *,
    backend: Literal["pce", "hdmr"] = "pce",
    include_correlative: bool = False,
    on_invalid: OnInvalid = "raise",
    **backend_kwargs: Any,
) -> ShapleyResult:
    """Fit a surrogate and return its Shapley effects (convenience wrapper).

    This is literally ``jaxgsa.pce.analyze(problem, X, Y, **kw).shapley()`` /
    ``jaxgsa.hdmr.analyze(problem, X, Y, **kw).shapley(include_correlative=...)``
    depending on ``backend``. There is no separate Shapley pipeline. Prefer
    the two-step form when you also want the fitted result (Sobol indices,
    ``predict``, fit diagnostics). Use this wrapper when only the Shapley
    effects are needed.

    Args:
        problem: Parameter names and distributions.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)`` scalar, ``(N, K)`` multi-output, or
            ``(N, T, K)`` time-series.
        backend: Surrogate providing the variance decomposition. ``"pce"``
            (default) reads subset variances off orthonormal polynomial
            coefficients. ``"hdmr"`` fits B-spline component functions and
            additionally separates correlation-induced variance.
        include_correlative: HDMR-only flag. Set it to fold the correlative
            ANCOVA part (``Sb``) into the allocation. See
            ``HDMRResult.shapley``.
        on_invalid: What to do about non-finite values in ``X`` or ``Y``. One
            row is one unit for both backends, so ``"drop"`` removes the
            affected ``(X, Y)`` pairs. See :mod:`jaxgsa._core.invalid`.

            This is a named parameter rather than one of ``backend_kwargs``
            on purpose. Naming it here forwards it to exactly one backend
            ``analyze``, which applies the policy exactly once. The returned
            ``ShapleyResult.invalid`` is that backend's report.
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
    from jaxgsa.shapley import SPEC

    # Both backends fit smooth surrogates over the inputs, which is undefined
    # for unordered level codes; reject with the Shapley-specific name. Only
    # the categorical half is gated from the record: whether a correlation is
    # tolerated depends on the backend, so it is settled below instead.
    gates(SPEC, problem, method="jaxgsa.shapley.analyze", check=("categorical",))
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
                "effects — or one of the correlation-tolerant methods: "
                f"{_correlation_tolerant_methods()}. Those methods do not "
                "return Shapley effects."
            )
        from jaxgsa.pce import analyze as analyze_pce

        return analyze_pce(problem, X, Y, on_invalid=on_invalid, **backend_kwargs).shapley()
    if backend == "hdmr":
        from jaxgsa.hdmr import analyze as analyze_hdmr

        return analyze_hdmr(problem, X, Y, on_invalid=on_invalid, **backend_kwargs).shapley(
            include_correlative=include_correlative
        )
    raise ValueError(f"backend must be 'pce' or 'hdmr', got {backend!r}")
