"""Shared Shapley pipeline tail and the convenience ``analyze`` wrapper.

``PCEResult.shapley`` and ``HDMRResult.shapley`` each supply their own
variance decomposition, then delegate the common tail to
:func:`jaxgsa.shapley._engine._shapley_result_from_variances`. That tail lives
in the engine, not here, so a sibling method package importing it does not have
to reach into another package's ``_analyze``. :func:`analyze` is a thin
convenience over those result methods.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.bootstrap import bootstrap_draws, interval
from jaxgsa._core.entry import (
    bootstrap_checks,
    check_scalars,
    gates,
    method_alias,
    one_of,
    positive_int,
    require,
)
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.irregular import (
    IrregularResult,
    IrregularY,
    _fwd_kwargs,
    _is_irregular_y,
    analyze_irregular,
)
from jaxgsa._core.validation import (
    _dependence_tolerant_methods,
    _is_constant_slice,
    _prepare_Y,
    as_key,
)
from jaxgsa.shapley._engine import (
    _allocate_variances,
    _normalize_partial_variances,
    build_membership,
    shapley_from_variances,
)
from jaxgsa.shapley._result import ShapleyResult

# Keyword-only parameters the irregular engine forwards unchanged to the
# per-channel recursive analyze() calls. verbose is excluded:
# the engine owns it.
_IRREGULAR_KWARGS = (
    "backend",
    "include_correlative",
    "n_bootstrap",
    "conf_level",
    "ci_method",
    "key",
    "on_invalid",
    "keep_replicates",
)

if TYPE_CHECKING:
    from jaxgsa.problem import Problem

_INDEX_FIELDS = ("Sh", "S1", "ST")
"""The index arrays an interval is reported for, in ``indices`` order."""


class ShapleyIndices(NamedTuple):
    """Shapley indices as plain arrays, as :func:`indices` returns them.

    Tuple unpacking works: ``Sh, S1, ST = indices(...)``.

    Attributes:
        Sh: Shapley effects, shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``.
        S1: First-order lower bound, same shape as ``Sh``.
        ST: Total-order upper bound, same shape as ``Sh``.
    """

    Sh: Array
    S1: Array
    ST: Array


def _raise_correlated_pce_backend(problem: "Problem", method: str) -> None:
    """Reject the PCE backend for a problem with correlated inputs.

    The refusal is per backend, not per method: the HDMR backend allocates
    the ANCOVA decomposition and tolerates a declared correlation, so the
    registry record cannot answer for both. :func:`analyze` and
    :func:`indices` share this helper so their refusals stay word-for-word
    identical apart from the entry-point name.

    Args:
        problem: Problem the allocation was requested for.
        method: Fully qualified entry-point name for the error message.

    Raises:
        ValueError: If ``problem`` declares a non-identity correlation. The
            message names the HDMR alternative and the dependence-tolerant
            methods.
    """
    if problem.dependence == "independent":
        return
    # The delegated pce core would compute silently wrong numbers; this
    # guard names the Shapley-specific alternative in the message.
    raise ValueError(
        f"{method} with backend='pce' computes a variance "
        "allocation that assumes independent inputs, but this problem "
        "declares dependent inputs (problem.correlation or problem.sampler). Use "
        "backend='hdmr' with include_correlative=True, which allocates "
        "the ANCOVA (structural + correlative) decomposition instead — "
        "an ANCOVA-based attribution, not conditional-variance Shapley "
        "effects — or one of the dependence-tolerant methods: "
        f"{_dependence_tolerant_methods()}. Those methods do not "
        "return Shapley effects."
    )


@jax.jit
def _pce_allocation(
    coefficients: Array, Y_3d: Array, membership: Array | np.ndarray
) -> tuple[Array, Array, Array]:
    """Allocate PCE partial variances to ``(Sh, S1, ST)`` in one jitted call.

    Everything after the fit is one compiled computation: the partial
    variances, the constant-slice gate, the explained variance and the
    allocation.

    Args:
        coefficients: PCE coefficients, shape ``(T, K, n_terms)``.
        Y_3d: Outputs, shape ``(N, T, K)``.
        membership: Boolean membership matrix, shape ``(n_terms - 1, D)``.

    Returns:
        Tuple ``(Sh, S1, ST)``, each of shape ``(T, K, D)``.
    """
    partial = coefficients[..., 1:] ** 2
    total = partial.sum(axis=-1)
    total_var = jnp.var(Y_3d, axis=0)
    # A constant slice's sample variance is almost never bit-exact zero
    # in float32 (the mean itself rounds), so gate on the slice being
    # numerically constant instead of on var == 0.
    is_constant = _is_constant_slice(Y_3d.reshape(Y_3d.shape[0], -1)).reshape(Y_3d.shape[1:])
    explained = jnp.where(is_constant, jnp.nan, total / total_var)
    normalized = _normalize_partial_variances(partial, explained)
    return shapley_from_variances(normalized, membership)


def _indices_3d(
    problem: "Problem",
    X: Array,
    Y_3d: Array,
    *,
    backend: Literal["pce", "hdmr"],
    include_correlative: bool,
    backend_kwargs: dict[str, Any],
) -> tuple[Array, Array, Array]:
    """Allocate the backend's variance decomposition, in canonical layout.

    The shared body of :func:`indices` and of one bootstrap replicate. It
    fits the surrogate the caller chose and runs the same allocation
    :meth:`jaxgsa.pce.PCEResult.shapley` and
    :meth:`jaxgsa.hdmr.HDMRResult.shapley` run, minus the fit-quality warning
    and the result class. Which backend it goes through is a Python string,
    so the choice is made at trace time.

    Args:
        problem: Parameter names and distributions.
        X: Input samples, shape ``(N, D)``.
        Y_3d: Outputs in canonical ``(N, T, K)`` layout.
        backend: ``"pce"`` or ``"hdmr"``.
        include_correlative: Fold the correlative ANCOVA part into the
            allocation. HDMR only.
        backend_kwargs: Fit arguments for the chosen backend's core.

    Returns:
        ``(Sh, S1, ST)`` in ``(T, K, D)`` layout.
    """
    if backend == "pce":
        from jaxgsa.pce._analyze import _fit_pce_core

        # indices() and the bootstrap replicates read only the coefficients
        # (the variance allocation below), never the LOO RMSE or the
        # fitted-value variance. The pce core defaults diagnostics=True, so
        # pass False explicitly to skip the Cholesky factor, the
        # (n_terms, N) triangular solve and the (N, T*K) residual array the
        # coefficient extraction never touches. See _fit_pce_core's
        # diagnostics docstring; the coefficients are bit-identical either
        # way (verified on the benchmark shape).
        fit = _fit_pce_core(problem, X, Y_3d, diagnostics=False, **backend_kwargs)
        # Orthonormality makes each squared non-constant coefficient a
        # partial variance; multi_index[1:] > 0 IS the membership matrix.
        membership = np.asarray(fit.multi_index[1:] > 0)
        return _pce_allocation(fit.coefficients, Y_3d, membership)
    else:
        from jaxgsa.hdmr._analyze import _hdmr_core

        core = _hdmr_core(problem, X, Y_3d, **backend_kwargs)
        partial = core.Sa + core.Sb if include_correlative else core.Sa
        subsets: list[tuple[int, ...]] = [(i,) for i in range(problem.n_params)]
        subsets.extend(core.c2)
        subsets.extend(core.c3)
        membership = build_membership(subsets, problem.n_params)
        # For HDMR the per-term sum doubles as the explained-variance
        # diagnostic, so it is both the normalizer and the diagnostic.
        total = partial.sum(axis=-1)
        explained = total

    return _allocate_variances(partial, explained, membership)


def indices(
    problem: "Problem",
    X: Array,
    Y: Array,
    *,
    backend: Literal["pce", "hdmr"] = "pce",
    include_correlative: bool = False,
    **backend_kwargs: Any,
) -> ShapleyIndices:
    """Compute Shapley effects as plain arrays, with no diagnostics.

    This is the transformable core of :func:`analyze`. It fits the same
    surrogate on the same data and allocates the same variance, and it
    applies the same capability gates (categorical inputs for both backends,
    correlated inputs for the PCE backend; ``problem`` is static host-side
    metadata, so the gates run at trace time). But it does nothing else: no
    non-finite check, no zero-variance warning, no pathological-fit warning,
    no ``explained_variance`` diagnostic, no
    :class:`jaxgsa.shapley.ShapleyResult`, and no read of any array value on
    the host. So it composes with ``jax.jit``, ``jax.vmap``
    and reverse- or forward-mode differentiation, which :func:`analyze`
    cannot, because a policy decision needs a concrete value and a tracer has
    none.

    Use :func:`analyze` for ordinary analysis. The warning this drops is the
    one that matters most for Shapley effects: an allocation of a surrogate's
    variance is only as good as the surrogate, and nothing here tells you the
    fit was poor.

    Tier T4 (behavioural contract): the returned arrays must equal the
    corresponding fields of ``analyze``'s result on clean outputs, and the
    function must survive ``jit``, ``vmap`` and, on the PCE backend,
    ``jit(jacrev(...))``. Checked in ``tests/test_shapley.py``.

    Note:
        The HDMR backend inherits the one limit of
        :func:`jaxgsa.hdmr.indices`: reverse-mode differentiation is blocked
        by the early-stopping backfitting loop, so use ``jacfwd`` there. The
        PCE backend differentiates either way.

    Args:
        problem: Parameter names and distributions.
        X: Input samples, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)`` or ``(N, T, K)``.
        backend: Surrogate providing the variance decomposition, ``"pce"``
            (default) or ``"hdmr"``.
        include_correlative: HDMR-only flag, as in :func:`analyze`.
        **backend_kwargs: Fit arguments passed to the chosen backend's core:
            ``order``/``ridge``/``fit_ratio``/``batch_size`` for PCE,
            ``order``/``max_iter``/``n_intervals``/``ridge``/
            ``slice_chunk_size``/``batch_size`` for HDMR.

    Returns:
        A :class:`ShapleyIndices` ``(Sh, S1, ST)``, at the shapes ``analyze``
        reports for this ``Y``.

    Raises:
        ValueError: If ``backend`` is unknown, ``include_correlative`` is
            asked for with the PCE backend, ``problem`` has a categorical
            parameter (either backend), or ``problem.correlation`` declares a
            dependence structure with the PCE backend — exactly the refusals
            :func:`analyze` makes.
    """
    from jaxgsa.shapley import _SPEC

    # The same gates analyze applies, in the same order: categorical for both
    # backends from the registry record, correlation per backend below.
    gates(_SPEC, problem, method="jaxgsa.shapley.indices", check=("categorical",))
    check_scalars(
        (
            one_of("backend", backend, ("pce", "hdmr")),
            require(
                not (include_correlative and backend == "pce"),
                "include_correlative requires backend='hdmr'",
            ),
        ),
        method="jaxgsa.shapley.indices",
    )
    if backend == "pce":
        _raise_correlated_pce_backend(problem, "jaxgsa.shapley.indices")
    Y_3d, layout = _prepare_Y(Y)
    Sh, S1, ST = _indices_3d(
        problem,
        jnp.asarray(X),
        Y_3d,
        backend=backend,
        include_correlative=include_correlative,
        backend_kwargs=backend_kwargs,
    )
    return ShapleyIndices(Sh=layout.squeeze(Sh), S1=layout.squeeze(S1), ST=layout.squeeze(ST))


def _shapley_replicate(
    rows: Array,
    X: Array,
    Y_3d: Array,
    *,
    problem: "Problem",
    backend: Literal["pce", "hdmr"],
    include_correlative: bool,
    backend_kwargs: tuple[tuple[str, Any], ...],
) -> tuple[Array, Array, Array]:
    """Refit the backend on the rows ``rows`` and return ``(Sh, S1, ST)``."""
    return _indices_3d(
        problem,
        X[rows],
        Y_3d[rows],
        backend=backend,
        include_correlative=include_correlative,
        backend_kwargs=dict(backend_kwargs),
    )


def _bootstrap_indices(
    problem: "Problem",
    X: Array,
    Y_3d: Array,
    key: Array,
    *,
    n_bootstrap: int,
    backend: Literal["pce", "hdmr"],
    include_correlative: bool,
    backend_kwargs: dict[str, Any],
) -> dict[str, Array]:
    """Refit the backend surrogate on ``n_bootstrap`` row resamples.

    Each replicate goes through the same backend the point estimate did, and
    refits it: a Shapley effect is an allocation of fitted variances, so
    there is nothing cheaper to resample. The cost is ``n_bootstrap``
    surrogate fits, an order of magnitude above a row resample of an
    estimator that only re-reduces per-row numbers.

    Args:
        problem: Parameter names and distributions.
        X: Input samples with invalid rows already removed, shape ``(N, D)``.
        Y_3d: Outputs in canonical ``(N, T, K)`` layout.
        key: A ``jax.random`` key, split once per replicate.
        n_bootstrap: Number of replicates.
        backend: Which surrogate to refit.
        include_correlative: HDMR-only allocation flag.
        backend_kwargs: Fit arguments for the backend's core.

    Returns:
        One stacked draw array per index field, each with a leading axis of
        length ``n_bootstrap`` followed by the canonical ``(T, K, D)`` shape.
    """
    draws = bootstrap_draws(
        key,
        n_bootstrap,
        _shapley_replicate,
        (X, Y_3d),
        static=(
            ("problem", problem),
            ("backend", backend),
            ("include_correlative", include_correlative),
            ("backend_kwargs", tuple(sorted(backend_kwargs.items()))),
        ),
    )
    return dict(zip(_INDEX_FIELDS, draws, strict=True))


def analyze(
    problem: "Problem",
    X: Array,
    Y: Array | IrregularY,
    *,
    backend: Literal["pce", "hdmr"] = "pce",
    include_correlative: bool = False,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
    **backend_kwargs: Any,
) -> ShapleyResult | IrregularResult:
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
        n_bootstrap: Number of bootstrap resamples for confidence intervals
            on ``Sh``, ``S1`` and ``ST``. ``0`` (default) disables them. The
            resampling unit is one row of ``(X, Y)``, and each replicate goes
            through the same backend the point estimate used.

            This is the expensive kind of bootstrap, and the most expensive
            of the three surrogate methods: a Shapley effect is an allocation
            of *fitted* variances, so a replicate has to refit the whole
            surrogate. The cost is an order of magnitude above the row
            resample of a method such as ``jaxgsa.pawn``, which only
            re-reduces numbers it already has. That is why the default
            is ``0``.

            The interval measures the sampling variability of ``(X, Y)``
            propagated through the fit. It does not measure how well the
            surrogate represents the model: every replicate shares the same
            basis and the same order, so a systematic misfit sits inside
            every interval. ``explained_variance`` is what reports that.
        conf_level: Two-sided confidence level for the intervals.
        ci_method: How the interval endpoints are formed. ``"quantile"``
            (default) reads them off the empirical bootstrap distribution.
            ``"gaussian"`` centres them on the point estimate and takes
            ``+/- z * sd`` of the draws.
        key: A ``jax.random`` key for the bootstrap resampling. Required when
            ``n_bootstrap > 0``. Pass ``jax.random.key(0)`` if you have an
            integer seed.
        verbose: If ``True`` (default), print a short summary to stdout: the
            problem and the data, the wall-clock timing, and the top
            parameters by ``Sh``. Pass ``False`` for a silent run. The
            backend's own summary is always suppressed, so one call prints
            one summary.
        keep_replicates: Retain the per-replicate index arrays on
            ``result.ci``. Off by default: the draws are large.
        **backend_kwargs: Passed through unchanged to the selected backend's
            ``analyze`` (e.g. ``order``/``ridge``/``fit_ratio`` for PCE,
            ``order``/``n_intervals``/``ridge`` for HDMR). Python requires
            ``**kwargs`` to come last in a signature, which is why
            ``keep_replicates`` is not the final parameter here as it is on
            every other method.

    Returns:
        ShapleyResult, exactly as returned by the corresponding result
        method, plus the confidence intervals when ``n_bootstrap > 0``.

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

    Warns:
        JaxgsaWarning: If the surrogate fit looks pathological. Both backends
            warn when ``explained_variance`` sits well below 1, and the HDMR
            backend also warns when it sits above 1. On the PCE backend
            ``explained_variance`` is a true in-sample R-squared, which an
            overfit expansion drives *towards* 1, so overfitting is caught
            out of sample instead: the fit's ``cv_rmse`` approaching
            ``std(Y)`` means the expansion predicts no better than the output
            mean.
        JaxgsaWarning: If ``backend="hdmr"``, ``problem`` declares a
            correlation, and ``include_correlative=False``. That combination
            allocates the structural share ``Sa`` only, renormalized to sum
            to 1, and drops the correlative share the correlation carries.
            The warning comes from :meth:`HDMRResult.shapley`, so the direct
            route raises it too.
    """
    key = as_key(key, "shapley")
    if _is_irregular_y(Y):
        return analyze_irregular(
            analyze,
            channel_data=X,
            problem=problem,
            Y=Y,
            n_expected=int(X.shape[0]),
            design_based=False,
            verbose=verbose,
            # The backend keyword arguments are re-expanded by the recursive
            # call's own ``**backend_kwargs``, so they must be flattened here
            # rather than forwarded as one nested ``backend_kwargs`` dict.
            kwargs={**_fwd_kwargs(locals(), _IRREGULAR_KWARGS), **backend_kwargs},
        )
    Y = cast(Array, Y)

    from jaxgsa.shapley import _SPEC

    # Both backends fit smooth surrogates over the inputs, which is undefined
    # for unordered level codes; reject with the Shapley-specific name. Only
    # the categorical half is gated from the record: whether a correlation is
    # tolerated depends on the backend, so it is settled below instead.
    gates(_SPEC, problem, method="jaxgsa.shapley.analyze", check=("categorical",))
    check_scalars(
        (
            *bootstrap_checks(n_bootstrap, ci_method, conf_level, key),
            positive_int("slice_chunk_size", backend_kwargs.get("slice_chunk_size")),
            positive_int("batch_size", backend_kwargs.get("batch_size")),
        ),
        method="jaxgsa.shapley.analyze",
    )
    # The backend prints no summary of its own: one shapley call prints one
    # block. A caller-supplied verbose inside backend_kwargs is stripped, not
    # refused, so the explicit verbose=False below always wins.
    backend_kwargs.pop("verbose", None)
    t0 = _verbose.tic()
    if backend == "pce":
        if include_correlative:
            raise ValueError("include_correlative requires backend='hdmr'")
        _raise_correlated_pce_backend(problem, "jaxgsa.shapley.analyze")
        from jaxgsa.pce import analyze as analyze_pce

        # The alias makes the shared preamble name shapley, not pce, in the
        # warnings and errors it raises for the user's call.
        with method_alias("jaxgsa.shapley.analyze"):
            pce_result = analyze_pce(
                problem, X, Y, on_invalid=on_invalid, verbose=False, **backend_kwargs
            )
        # Unreachable: the irregular path returned at the top of this
        # function, so Y is a regular array by the time the backend runs.
        # The guard keeps the union return type honest without a cast.
        if isinstance(pce_result, IrregularResult):
            raise AssertionError("shapley pce backend returned an irregular result")
        # No fit-quality warning here. jaxgsa.pce.analyze, one frame below,
        # already read both diagnostics off this fit and warned about them.
        result = pce_result.shapley()
    elif backend == "hdmr":
        from jaxgsa.hdmr import analyze as analyze_hdmr

        # The warning for a correlated problem with include_correlative=False
        # lives on HDMRResult.shapley, which is the call below and is also
        # reachable on its own. Warning here as well would report it twice.
        with method_alias("jaxgsa.shapley.analyze"):
            hdmr_result = analyze_hdmr(
                problem, X, Y, on_invalid=on_invalid, verbose=False, **backend_kwargs
            )
        if isinstance(hdmr_result, IrregularResult):
            raise AssertionError("shapley hdmr backend returned an irregular result")
        result = hdmr_result.shapley(include_correlative=include_correlative)
    else:
        raise ValueError(f"backend must be 'pce' or 'hdmr', got {backend!r}")

    if n_bootstrap > 0:
        assert key is not None  # checked above, before the surrogate was fitted
        result = _with_intervals(
            result,
            problem,
            X,
            Y,
            key,
            n_bootstrap=n_bootstrap,
            conf_level=conf_level,
            ci_method=ci_method,
            keep_replicates=keep_replicates,
            backend=backend,
            include_correlative=include_correlative,
            backend_kwargs=backend_kwargs,
        )

    if verbose:
        elapsed = _verbose.stop(t0, result.Sh)
        # No prepare() context here: the backend validated everything. The
        # promoted layout is read off the shapes at hand instead — Sh is
        # (D,), (K, D) or (T, K, D) matching Y (N,), (N, K), (N, T, K).
        # "After any drop", as the summary contract says: the backend's
        # report counts what the fit actually used.
        if result.invalid.policy == "drop" and result.invalid.any_invalid:
            n_runs = int(result.invalid.n_kept)
        else:
            n_runs = int(np.shape(Y)[0])
        sh_shape = np.shape(result.Sh)
        if len(sh_shape) == 3:
            T, K = int(sh_shape[0]), int(sh_shape[1])
        elif len(sh_shape) == 2:
            T, K = 1, int(sh_shape[0])
        else:
            T, K = 1, 1
        _verbose.analysis_summary(
            method="jaxgsa.shapley.analyze",
            problem=problem,
            n_runs=n_runs,
            T=T,
            K=K,
            invalid=result.invalid,
            timings=[("backend fit + Shapley (includes compile on the first call)", elapsed)],
            notes=[f"backend: {backend}", f"order: {result.order}"],
            index_name="Sh",
            values=result.Sh,
            conf=result.Sh_conf,
        )
    return result


def _with_intervals(
    result: ShapleyResult,
    problem: "Problem",
    X: Array,
    Y: Array,
    key: Array,
    *,
    n_bootstrap: int,
    conf_level: float,
    ci_method: Literal["quantile", "gaussian"],
    keep_replicates: bool,
    backend: Literal["pce", "hdmr"],
    include_correlative: bool,
    backend_kwargs: dict[str, Any],
) -> ShapleyResult:
    """Attach bootstrap intervals to a point-estimate Shapley result.

    The replicates resample the rows the backend actually fitted on, which is
    not always the rows the caller passed: under ``on_invalid="drop"`` the
    backend removed some, and ``result.invalid.row_indices`` says which, in
    the caller's own numbering. Resampling the full array instead would draw
    the dropped rows back in and hand every replicate a NaN fit.

    Args:
        result: The point-estimate result, returned with intervals attached.
        problem: Parameter names and distributions.
        X: The caller's input samples, before any drop.
        Y: The caller's outputs, before any drop.
        key: A ``jax.random`` key.
        n_bootstrap: Number of replicates.
        conf_level: Two-sided confidence level.
        ci_method: Endpoint rule.
        keep_replicates: Whether to retain the draws on the result.
        backend: Which surrogate to refit per replicate.
        include_correlative: HDMR-only allocation flag.
        backend_kwargs: Fit arguments for the backend.

    Returns:
        A copy of ``result`` carrying ``Sh_conf``, ``S1_conf``, ``ST_conf``
        and ``ci``.
    """
    X_kept, Y_kept = jnp.asarray(X), jnp.asarray(Y)
    if result.invalid.policy == "drop" and result.invalid.any_invalid:
        dropped = np.asarray(result.invalid.row_indices, dtype=int)
        keep = np.setdiff1d(np.arange(X_kept.shape[0]), dropped)
        X_kept, Y_kept = X_kept[keep], Y_kept[keep]

    Y_3d, layout = _prepare_Y(Y_kept)
    draws = _bootstrap_indices(
        problem,
        X_kept,
        Y_3d,
        key,
        n_bootstrap=n_bootstrap,
        backend=backend,
        include_correlative=include_correlative,
        backend_kwargs=backend_kwargs,
    )
    # The leading replicate axis survives the squeeze, which addresses the
    # T/K axes from the end.
    points = {name: getattr(result, name) for name in _INDEX_FIELDS}
    squeezed_draws = {name: layout.squeeze(draws[name]) for name in _INDEX_FIELDS}
    confs, ci = interval(
        points,
        squeezed_draws,
        level=conf_level,
        method=ci_method,
        n_bootstrap=n_bootstrap,
        keep_replicates=keep_replicates,
    )
    return dataclasses.replace(
        result,
        **{f"{name}_conf": confs[name] for name in _INDEX_FIELDS},
        ci=ci,
    )
