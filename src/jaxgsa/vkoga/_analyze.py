"""Two-stage correlated sensitivity analysis: VKOGA surrogate, then indices.

Implements the surrogate-based sensitivity analysis (SSA) of Hilhorst et al.
(2024). Stage one fits a greedy kernel surrogate to the given ``(X, Y)`` data.
Stage two computes the correlated variance-based indices of Li et al. (2010)
against that surrogate under the problem's input distribution: a Gaussian
copula, or a custom sampler declared on the problem. The split is what makes
the method affordable. The indices need nested conditional sampling, which is
unaffordable against an expensive model but cheap against a kernel expansion.

**This module has no pure core.** Stage two is a host NumPy/SciPy quasi-Monte
-Carlo loop end to end, so there is no ``indices()`` here that returns bare
arrays and survives ``jit``, ``vmap`` and ``jacrev``. That is a declared
exemption (see ``jaxgsa._core.registry.MethodRecord.pure_core``), not a gap:
no traceable core exists to export.

References:
    Hilhorst, Quicken, van de Vosse & Huberts (2024). Int. J. Numer. Meth.
        Biomed. Engng. 40(2):e3797.
    Li, Rabitz, Yelvington et al. (2010). J. Phys. Chem. A 114:6022-6032.
    Wirtz & Haasdonk (2013). Dolomites Res. Notes Approx. 6:83-100.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.batching import resolve_batch_size
from jaxgsa._core.bootstrap import interval
from jaxgsa._core.dependence import Draws, draws_for
from jaxgsa._core.entry import (
    _METHOD_ALIAS,
    _user_stacklevel,
    at_least,
    bootstrap_checks,
    prepare,
    require,
)
from jaxgsa._core.fit_quality import check_degenerate_inputs
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.irregular import (
    IrregularResult,
    IrregularY,
    _fwd_kwargs,
    _is_irregular_y,
    analyze_irregular,
)
from jaxgsa._core.precision import x64_enabled
from jaxgsa._core.result import CIInfo
from jaxgsa._core.sampling import _next_power_of_2
from jaxgsa._core.surrogate import _PredictPlan
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa._core.validation import _zero_variance_mask, as_key
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.problem import Problem
from jaxgsa.vkoga._engine import (
    GreedyUpdate,
    _content_digest,
    _cross_validate,
    _fit_vkoga,
    _gaussian_kernel,
    _predict_vkoga,
)
from jaxgsa.vkoga._indices import estimate_correlated_indices
from jaxgsa.vkoga._result import VKOGAResult

# Keyword-only parameters the irregular engine forwards unchanged to the
# per-channel recursive analyze() calls. verbose is excluded:
# the engine owns it.
_IRREGULAR_KWARGS = (
    "gamma",
    "ridge",
    "max_centers",
    "greedy_update",
    "n_folds",
    "n_outer",
    "n_inner",
    "n_variance",
    "n_bootstrap",
    "conf_level",
    "ci_method",
    "key",
    "batch_size",
    "on_invalid",
    "keep_replicates",
)

# Memo of the compiled surrogate predictor, keyed on the fitted state and
# output mean it freezes as closure constants (see
# :func:`_make_unit_predictor`). Bounded so frozen fit state cannot accumulate.
_PREDICTOR_CACHE_MAX = 4
_PREDICTOR_CACHE: dict[tuple[Any, ...], Callable[[np.ndarray], np.ndarray]] = {}
# The same memo for the block-mean evaluator S_TC uses (see _make_group_mean).
_GROUP_MEAN_CACHE: dict[tuple[Any, ...], Callable[[np.ndarray, int], np.ndarray]] = {}

# Centre count granularity of the fitted state handed to the evaluators (see
# _fit_and_estimate). Up to 31 padding centres with zero coefficients cost
# little next to one compile per distinct centre count.
_CENTER_BUCKET = 32

# The surrogate evaluation inside the index estimators sizes its batches
# against 1 / share of the memory budget; the estimators' queue and draws
# hold the rest (see vkoga._indices._QUEUE_BUDGET_SHARE).
_PREDICT_BUDGET_SHARE = 2

# Hyperparameter search grid, following Hilhorst et al. Section 2.4.1: ten
# log-spaced values each, cross-validated as a 10x10 product.
_GAMMA_GRID = np.logspace(-2, np.log10(50.0), 10)
# Ridge candidates for a float32 fit: ten log-spaced values from 1e-6.
# Candidates below 1e-6 were measured on six problems (Ishigami N=600/2048,
# correlated Ishigami, G-function, linear, a (T, K) model; two seeds) and
# never helped in float32: the held-out error moved by -3% to +1.6%.
_RIDGE_GRID = np.logspace(-6, -2, 10)
# Ridge candidates for a float64 fit: the same spacing continued down four
# more steps, to about 1.7e-8. On the same twelve float64 fits the held-out
# error fell on ten (6% to 54%) and never rose. The lowest candidates are at
# the scale of _engine._JITTER (an extra ridge of about 3e-8 to 2e-7 at
# n ~ 1024), so VKOGAResult.ridge can undercount the regularisation in force
# there.
_RIDGE_GRID_F64 = np.concatenate([10.0 ** (-6 - 4 / 9 * np.arange(4, 0, -1)), _RIDGE_GRID])

# Cap on kernel centres when the caller does not choose one. Greedy selection
# cost is O(max_centers * n), and the marginal accuracy of further centres
# falls off quickly once the power function has collapsed.
_DEFAULT_MAX_CENTERS = 300

# Largest cross-validated RMSE, as a fraction of the output standard deviation,
# that still gives usable indices. At 0.5 the surrogate leaves a quarter of the
# output variance unexplained; past that the measured indices stop tracking the
# true ones, and on oscillatory models they can even invert the ranking.
_CV_RMSE_WARN_FRACTION = 0.5

# The index fields a bootstrap reports an interval for. The diagnostics
# (variance, rmse, cv_rmse, n_centers, gamma, ridge) get none: they describe
# the fit, not the sensitivity of the model, and a caller reading them wants
# the value the reported surrogate actually has.
_INTERVAL_FIELDS = ("S_TC", "S_TU", "S_U", "S_C", "S_IU")


def analyze(
    problem: Problem,
    X: Array,
    Y: Array | IrregularY,
    *,
    gamma: float | None = None,
    ridge: float | None = None,
    max_centers: int | None = None,
    greedy_update: GreedyUpdate = "exact",
    n_folds: int = 10,
    n_outer: int = 512,
    n_inner: int = 128,
    n_variance: int = 8192,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | int,
    batch_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> VKOGAResult | IrregularResult:
    """Correlated variance-based sensitivity indices via a VKOGA surrogate.

    Fits a Vectorial Kernel Orthogonal Greedy Algorithm surrogate to given
    ``(X, Y)`` data, then estimates the five correlated indices of Li et al.
    (2010) against it under the problem's input distribution: independent,
    a Gaussian copula (``problem.correlation``), or a custom sampler
    (``problem.with_sampler(...)``). A custom sampler's marginals must match
    the declarations in ``problem``.

    The surrogate must be accurate where the analysis distribution puts its
    mass. Both nested estimators draw from conditionals of the joint
    distribution, never from a whole marginal with the rest held fixed, so
    every surrogate evaluation lies where the joint distribution puts mass.
    Training rows drawn from the analysis distribution itself therefore work.
    An independent, space-filling design is a safe default when the
    correlation is not fixed yet, or when one fit is reused under several
    correlations. Avoid training data that misses regions the analysis
    covers.

    Args:
        problem: Problem defining the parameters and their marginals.
        X: Inputs in physical units, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        gamma: RBF shape parameter, finite and positive when given. ``None``
            cross-validates over a grid.
        ridge: Kernel regularisation, finite and positive when given. ``None``
            cross-validates over a grid.
        max_centers: Maximum kernel centres the greedy may select, at least 1.
            Defaults to 300, capped at ``N``.
        greedy_update: How the greedy updates the power function after each
            centre. ``"exact"`` (default) recomputes it from the whole Newton
            basis. ``"incremental"`` subtracts the new basis column only,
            which makes the greedy loop faster. Its rounding differs, so it
            often selects different centres, and the indices move by about
            their Monte-Carlo error. They are not less accurate.
        n_folds: Folds for hyperparameter cross-validation, at least 2.
        n_outer: Outer (conditioning) sample size per parameter, at least 2.
            Rounded up to the next power of two (Sobol' balance).
        n_inner: Inner (conditional) sample size per outer point, at least 2.
            Rounded up to the next power of two. ``S_TC`` reuses one inner
            block across every outer point and drops the iid inner-noise
            correction, so do not lower ``n_inner`` below the default to buy
            speed there: a small ``n_inner`` inflates a small ``S_TC``, and
            raising ``n_outer`` does not fix it, because the same shared
            block still drives every outer point. ``S_TU`` draws a fresh
            inner block per outer point, so its sampling noise falls as
            ``n_outer`` grows; raise ``n_outer`` when a noisy ``S_TU`` is the
            concern.
        n_variance: Sample size for the output variance and the component-
            function fit, at least 2. Rounded up to the next power of two.
        n_bootstrap: Number of bootstrap resamples for confidence intervals.
            ``0`` (default) computes no intervals, and leaving it there is
            the right call for a routine analysis: **each replicate refits
            the kernel surrogate and re-runs the nested conditional
            integration**, so ``n_bootstrap=100`` costs about a hundred times
            what the analysis costs without it.

            The interval measures **total uncertainty in the reported index
            given this training sample**: the training rows are resampled
            with replacement, a new surrogate is fitted to each resample, and
            a fresh quasi-random stream integrates against it. So it folds
            two sources together, the sampling error of the ``(X, Y)`` data
            and the Monte-Carlo error of the integration, and it is the
            interval to quote when asking "how much would this index move if
            I had drawn a different training set". It is *not* an
            integration-only interval: fitting once and resampling only the
            integration would measure how much ``n_outer``/``n_inner`` cost
            you, which is the part a caller can shrink for free by raising
            them, and it would understate the real uncertainty by leaving out
            the surrogate.

            Two things the interval holds fixed, so it is a conditional
            statement rather than a full one. ``gamma`` and ``ridge`` are the
            values cross-validated on the full sample and are not re-selected
            per replicate, because a per-replicate grid search would multiply
            the cost again for a nuisance parameter. And the interval is
            about the *surrogate's* indices: it cannot see error the
            surrogate makes against the true model, which is what ``cv_rmse``
            is for.

            Intervals are reported for the five index fields. The
            diagnostics — ``variance``, ``rmse``, ``cv_rmse``, ``n_centers``
            — get none: they describe the reported fit itself, so an interval
            over other fits would not be about the thing they name.
        conf_level: Confidence level for the bootstrap intervals.
        ci_method: How the interval endpoints are formed. ``"quantile"``
            (default) reads them off the empirical bootstrap distribution.
            ``"gaussian"`` centres them on the point estimate and takes
            ``+/- z * sd`` of the bootstrap draws, which is smoother for a
            small ``n_bootstrap`` but assumes the draws are normal.
        key: JAX PRNG key that drives the quasi-random index integration.
            Required: the indices are a Monte-Carlo estimate, so there is no
            sensible default. Write ``jax.random.key(0)`` if you only want
            reproducibility. A key rather than an integer seed because a key
            can be split, so repeated or nested analyses draw independent
            streams instead of silently correlated ones.
        batch_size: Sample rows per device call, at least 1. It bounds every
            row-wise step of the method: the surrogate evaluations behind
            :meth:`VKOGAResult.predict` and the nested conditional draws the
            index estimators make. ``None`` derives one from the memory
            budget.
        on_invalid: What to do about non-finite values in ``X`` or ``Y``. One
            training row is one unit here, so ``"drop"`` removes the affected
            ``(X, Y)`` pairs and fits the surrogate on the rest. See
            :mod:`jaxgsa._core.invalid`. The check runs before the
            hyperparameter search on purpose: a single non-finite ``Y`` makes
            every cross-validation score non-finite, and the caller would
            otherwise meet a ``RuntimeError`` about failed kernel solves that
            names the wrong cause. ``"drop"`` needs at least ``n_folds`` rows
            to survive, because a fold with no rows in it scores nothing.
        verbose: If ``True`` (default), print a short summary to stdout: the
            problem and the data, the wall-clock timing, and the top
            parameters by ``S_TC``. Pass ``False`` for a silent run.
    Returns:
        A :class:`VKOGAResult` with ``S_TC``, ``S_TU``, ``S_U``, ``S_C``, and
        ``S_IU`` shaped ``(D,)``, ``(K, D)``, or ``(T, K, D)`` to mirror ``Y``.

    Raises:
        ValueError: If ``X``/``Y`` violate the output contract, if the problem
            has fewer than two parameters or any categorical parameter, if
            the custom sampler returns rows outside a declared bounded
            support, if ``gamma`` or
            ``ridge`` is not finite and positive, or if a size argument is out
            of range, if ``key`` is ``None``, if ``on_invalid`` is not one of
            the three policies, or
            if ``on_invalid="raise"`` (the default) and ``X`` or ``Y`` holds a
            non-finite value, or if an input column of ``X`` is constant,
            equal to another column, or a linear combination of the columns
            before it to within rounding error. Duplicate rows are allowed.
        RuntimeError: If every cross-validation score is non-finite. Non-finite
            training data no longer reaches this check; it is now a genuine
            solver failure.

    Warns:
        JaxgsaWarning: In any of four cases. An output slice has zero variance.
            The surrogate is fitted in single precision (float32 inputs, or
            JAX without x64), where the greedy centre selection stops early
            for small ``gamma`` (see :mod:`jaxgsa.vkoga`). The cross-validated
            surrogate error is a large fraction of the output standard
            deviation, which makes every index untrustworthy. ``S_U`` had to be
            clipped to ``S_TU`` by a wide margin, which says the additive
            component functions cannot represent the model's interactions.
    """
    key = as_key(key, "vkoga")
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

    from jaxgsa.vkoga import _SPEC

    D = problem.n_params
    # Every scalar argument is checked before any expensive work
    # (cross-validation, fitting, index estimation). The preamble runs them
    # first, so that ordering is structural rather than a convention.
    ctx = prepare(
        _SPEC,
        problem,
        Y,
        X=X,
        on_invalid=on_invalid,
        checks=(
            at_least("max_centers", max_centers, 1),
            require(
                greedy_update in ("exact", "incremental"),
                f'greedy_update must be "exact" or "incremental", got {greedy_update!r}',
            ),
            require(
                n_folds >= 2,
                f"n_folds must be >= 2 for cross-validation, got {n_folds}",
            ),
            *(
                require(
                    value is None or bool(np.isfinite(value) and value > 0),
                    f"{name} must be a finite positive number, got {value}",
                )
                for name, value in (("gamma", gamma), ("ridge", ridge))
            ),
            *(
                at_least(name, value, 2)
                for name, value in (
                    ("n_outer", n_outer),
                    ("n_inner", n_inner),
                    ("n_variance", n_variance),
                )
            ),
            at_least("batch_size", batch_size, 1),
            *bootstrap_checks(n_bootstrap, ci_method, conf_level, key),
            require(
                D >= 2,
                f"Correlated sensitivity indices need at least 2 parameters, got {D}",
            ),
        ),
        min_kept=n_folds,
    )
    # The index estimate needs a key even with no bootstrap. This check runs
    # after the preamble, so with n_bootstrap=0 a problem this method refuses
    # outright (categorical parameters) still reports that, not a missing key.
    if key is None:
        raise ValueError("key is required for the Monte-Carlo index estimate")
    seed = _seed_from_key(key)
    assert ctx.X is not None  # a given-data method always passes X to prepare()
    X, Y, invalid = ctx.X, ctx.Y, ctx.invalid
    # A constant, duplicated or linearly dependent input column is off the
    # manifold the surrogate is asked to extrapolate to, so its indices come
    # out wrong with no error. Skipped when non-finite rows were let through
    # (`on_invalid="propagate"`): the check reads the inputs and needs them finite.
    if bool(jnp.all(jnp.isfinite(X))):
        check_degenerate_inputs(_METHOD_ALIAS.get() or "jaxgsa.vkoga", X, problem.names)
    # The clock starts before the hyperparameter search and the greedy fit:
    # for this method the fit is the expensive work, not just the indices.
    t0 = _verbose.tic()

    if max_centers is None:
        max_centers = _DEFAULT_MAX_CENTERS
    # The uniform noise comes from Sobol' sequences, which need power-of-two
    # sizes to keep their balance guarantees (scipy warns otherwise). The
    # defaults are already powers of two, so they pass through unchanged.
    n_outer = _next_power_of_2(n_outer)
    n_inner = _next_power_of_2(n_inner)
    n_variance = _next_power_of_2(n_variance)

    Y_canonical = ctx.Y3
    n_time, n_out = Y_canonical.shape[1], Y_canonical.shape[2]
    # One fit dtype for the inputs and the outputs. Mixed float32/float64
    # inputs would otherwise meet in the greedy loop's carry and fail there.
    fit_dtype = jnp.result_type(X.dtype, Y_canonical.dtype)
    Y_flat = Y_canonical.reshape(Y_canonical.shape[0], n_time * n_out).astype(fit_dtype)

    # The RBF kernel is isotropic, so every column must share a scale; the
    # marginal CDF map is the same transform HDMR uses for its basis.
    U = cdf_to_unit_interval(X, problem).astype(fit_dtype)
    _warn_single_precision(fit_dtype)
    # The input draws are built once and shared by the point estimate and every
    # bootstrap replicate. The dependence comes from the problem alone, so the
    # result's ``dependence`` always equals ``problem.dependence``.
    input_draws = draws_for(problem)

    # VKOGA carries no constant term, so a non-zero output mean would have to
    # be reconstructed by the kernel expansion itself. Centring removes that
    # burden and measurably improves the fit.
    y_mean = Y_flat.mean(axis=0)
    Y_centered = Y_flat - y_mean

    # Non-finite data reaches this point only under on_invalid="propagate"
    # or "skip". All slices share one fit, so a NaN in one slice would poison
    # the whole fit. Fit the finite slices on their own and give the others
    # NaN, which is what those policies promise: the NaN flows into the
    # indices of the slice it is in. A non-finite input row touches every
    # slice.
    fit_slices = np.asarray(jnp.all(jnp.isfinite(Y_flat), axis=0)) & bool(jnp.all(jnp.isfinite(U)))
    # "skip" promises silence, so the slices are only masked there.
    if not fit_slices.all() and on_invalid != "skip":
        _warn_nonfinite_slices(
            fit_slices,
            on_invalid,
            inputs=not bool(jnp.all(jnp.isfinite(U))),
            stacklevel=_user_stacklevel(),
        )
    if not fit_slices.any():
        return _all_nan_result(
            problem,
            ctx,
            n_time=n_time,
            n_out=n_out,
            dtype=fit_dtype,
            gamma=gamma,
            ridge=ridge,
            invalid=invalid,
            y_mean=y_mean,
        )
    fit_cols = np.flatnonzero(fit_slices)
    n_slices = int(fit_slices.shape[0])

    def _expand(a: Any, axis: int) -> np.ndarray:
        """Put per-fitted-slice values back on the full slice axis, NaN elsewhere."""
        a = np.asarray(a)
        axis = axis % a.ndim
        shape = list(a.shape)
        shape[axis] = n_slices
        full = np.full(shape, np.nan, dtype=a.dtype)
        index: list[Any] = [slice(None)] * a.ndim
        index[axis] = fit_cols
        full[tuple(index)] = a
        return full

    resolved_centers = min(int(max_centers), int(U.shape[0]))
    gamma_value, ridge_value, cv_rmse = _resolve_hyperparameters(
        U,
        Y_centered[:, fit_cols],
        gamma=gamma,
        ridge=ridge,
        max_centers=resolved_centers,
        greedy_update=greedy_update,
        n_folds=n_folds,
        seed=seed,
    )
    _warn_poor_surrogate(cv_rmse, Y_centered[:, fit_cols])
    state, indices = _fit_and_estimate(
        U,
        Y_flat[:, fit_cols],
        gamma=gamma_value,
        ridge=ridge_value,
        max_centers=resolved_centers,
        greedy_update=greedy_update,
        input_draws=input_draws,
        problem=problem,
        n_outer=n_outer,
        n_inner=n_inner,
        n_variance=n_variance,
        entropy=seed,
        batch_size=batch_size,
        param_names=problem.names,
    )

    _warn_nonfinite_coefficients(state.coefficients, stacklevel=_user_stacklevel())
    if cv_rmse is not None:
        cv_rmse = _expand(cv_rmse, -1)
    if not fit_slices.all():
        # The skipped slices predict NaN, like their indices.
        state = state._replace(
            coefficients=jnp.asarray(_expand(state.coefficients, -1)),
            rmse=jnp.asarray(_expand(state.rmse, -1)),
        )
        indices = indices._replace(
            **{name: _expand(getattr(indices, name), -2) for name in _INTERVAL_FIELDS},
            variance=_expand(indices.variance, -1),
        )
    # A constant training slice has no variance to decompose. Its surrogate
    # variance is round-off (about 1e-30 in float64), which divided out would
    # give small non-zero indices. Use the zero-variance warning's own test,
    # so the indices are NaN as that warning promises.
    constant = np.asarray(_zero_variance_mask(Y_flat))
    if constant.any():
        indices = indices._replace(
            **{
                name: np.where(constant[:, None], np.nan, getattr(indices, name))
                for name in _INTERVAL_FIELDS
            },
            variance=np.where(constant, 0.0, indices.variance),
        )

    # The index estimators run on the host in float64. Every float field is
    # returned in the dtype the surrogate was fitted in, so float32 inputs
    # give float32 results under x64 too.
    work_dtype = state.centers.dtype

    def _shape_index(flat: np.ndarray) -> Array:
        """Reshape an ``(..., S, D)`` index block to the output contract."""
        arr = jnp.asarray(flat.reshape(*flat.shape[:-2], n_time, n_out, D), dtype=work_dtype)
        return ctx.squeeze(arr, n_trailing=1)

    def _shape_slice(flat: np.ndarray) -> Array:
        """Reshape an ``(S,)`` per-slice diagnostic to the output contract."""
        arr = jnp.asarray(flat.reshape(n_time, n_out), dtype=work_dtype)
        return ctx.squeeze(arr, n_trailing=0)

    confs: dict[str, Array | None] = dict.fromkeys(_INTERVAL_FIELDS)
    ci: CIInfo | None = None
    if n_bootstrap > 0:
        replicates = _bootstrap_replicates(
            U,
            Y_flat[:, fit_cols],
            gamma=gamma_value,
            ridge=ridge_value,
            max_centers=resolved_centers,
            greedy_update=greedy_update,
            input_draws=input_draws,
            problem=problem,
            n_outer=n_outer,
            n_inner=n_inner,
            n_variance=n_variance,
            key=key,
            n_bootstrap=n_bootstrap,
            batch_size=batch_size,
            stacklevel=_user_stacklevel(),
            nan_by_design=constant[fit_cols],
        )
        # _shape_index already squeezes to the caller's own rank; interval()
        # reads a shape only to broadcast [lower, upper], so it works the
        # same on this squeezed layout as on the raw (T, K, D) one.
        points = {name: _shape_index(getattr(indices, name)) for name in _INTERVAL_FIELDS}
        draws = {
            name: _shape_index(np.where(constant[:, None], np.nan, _expand(replicates[name], -2)))
            for name in _INTERVAL_FIELDS
        }
        interval_confs, ci = interval(
            points,
            draws,
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            keep_replicates=keep_replicates,
        )
        confs.update(interval_confs)

    # 1 - in-sample MSE / output variance, from the training rows.
    with np.errstate(divide="ignore", invalid="ignore"):
        explained = 1.0 - np.asarray(state.rmse) ** 2 / np.asarray(Y_flat).var(axis=0)
    # A constant slice has no variance to explain; float32 var is ~1e-16, not 0.
    explained = np.where(constant, np.nan, explained)
    output_shape = ctx.squeeze(jnp.zeros((n_time, n_out)), n_trailing=0).shape
    result = VKOGAResult(
        S_TC=_shape_index(indices.S_TC),
        S_TU=_shape_index(indices.S_TU),
        S_U=_shape_index(indices.S_U),
        S_C=_shape_index(indices.S_C),
        S_IU=_shape_index(indices.S_IU),
        problem=problem,
        variance=_shape_slice(indices.variance),
        n_centers=int(state.n_centers),
        gamma=float(gamma_value),
        ridge=float(ridge_value),
        invalid=invalid,
        rmse=_shape_slice(np.asarray(state.rmse)),
        cv_rmse=None if cv_rmse is None else _shape_slice(cv_rmse),
        explained_variance=_shape_slice(explained),
        _fit=state,
        _y_mean=y_mean,
        _output_shape=output_shape,
        S_TC_conf=confs["S_TC"],
        S_TU_conf=confs["S_TU"],
        S_U_conf=confs["S_U"],
        S_C_conf=confs["S_C"],
        S_IU_conf=confs["S_IU"],
        ci=ci,
    )

    if verbose:
        # Host NumPy/SciPy end to end, so there is no compile to mention and
        # stop() has nothing to block on.
        elapsed = _verbose.stop(t0, result.S_TC)
        batching = (
            f"batch_size: {batch_size} (user-set)"
            if batch_size is not None
            else "batch_size: auto (resolved from the memory budget)"
        )
        _verbose.analysis_summary(
            method="jaxgsa.vkoga.analyze",
            dependence=problem.dependence,
            problem=problem,
            n_runs=int(ctx.Y.shape[0]),
            T=n_time,
            K=n_out,
            invalid=invalid,
            timings=[("fit + compute", elapsed)],
            notes=[
                f"n_centers: {result.n_centers}",
                f"gamma: {_verbose.format_value(result.gamma)}",
                f"ridge: {_verbose.format_value(result.ridge)}",
                batching,
            ],
            index_name="S_TC",
            values=result.S_TC,
            conf=result.S_TC_conf,
        )
    return result


def _fit_and_estimate(
    U: Array,
    Y_flat: Array,
    *,
    gamma: float,
    ridge: float,
    max_centers: int,
    greedy_update: GreedyUpdate = "exact",
    input_draws: Draws,
    problem: Problem,
    n_outer: int,
    n_inner: int,
    n_variance: int,
    entropy: int,
    batch_size: int | None,
    param_names: Sequence[str] | None = None,
):
    """Fit one surrogate to one training set and read the indices off it.

    The two stages of the method, in one function, so that a bootstrap
    replicate runs exactly what the reported analysis ran.

    Args:
        U: Training inputs in unit-cube coordinates, shape ``(N, D)``.
        Y_flat: Training outputs, shape ``(N, S)``, uncentred.
        gamma: RBF shape parameter to fit with.
        ridge: Kernel regularisation to fit with.
        greedy_update: ``"exact"`` or ``"incremental"``; see
            :func:`jaxgsa.vkoga.analyze`.
        max_centers: Maximum kernel centres the greedy may select.
        input_draws: Joint and conditional input draws for the index integration.
        problem: Problem whose declared marginals pick each parameter's
            component-fit basis.
        n_outer: Outer (conditioning) sample size per parameter.
        n_inner: Inner (conditional) sample size per outer point.
        n_variance: Sample size for the variance and the component fit.
        entropy: Root entropy for the host-side quasi-random draws.
        batch_size: Sample rows per device call, or ``None`` for the budget.
        param_names: Parameter names for the clip warning, or ``None`` on a
            bootstrap replicate, where the warning is suppressed anyway.

    Returns:
        ``(state, indices)``: the fitted surrogate, sliced down to the centres
        the greedy actually selected, and its
        :class:`~jaxgsa.vkoga._indices.CorrelatedIndices`.
    """
    # VKOGA carries no constant term, so a non-zero output mean would have to
    # be reconstructed by the kernel expansion itself. Centring removes that
    # burden and measurably improves the fit.
    y_mean = Y_flat.mean(axis=0)
    state = _fit_vkoga(
        U,
        Y_flat - y_mean,
        gamma=gamma,
        max_centers=max_centers,
        greedy_update=greedy_update,
        ridge=ridge,
    )
    # The fitted state is padded to the static max_centers size (rows past
    # n_centers hold zero coefficients). Slice it down so predict and the
    # index estimators pay only for about the centres the greedy selected.
    # The width is rounded up to a multiple of _CENTER_BUCKET: the
    # evaluators compile per width, and the greedy's stopping count varies
    # from fit to fit (often in float32), so an exact slice would compile
    # anew for every dataset and bootstrap replicate. The slice runs on the
    # host for the same reason: an eager device slice compiles per width.
    width = min(
        int(state.centers.shape[0]),
        -(-int(state.n_centers) // _CENTER_BUCKET) * _CENTER_BUCKET,
    )
    state = state._replace(
        centers=jnp.asarray(np.asarray(state.centers)[:width]),
        coefficients=jnp.asarray(np.asarray(state.coefficients)[:width]),
    )
    indices = estimate_correlated_indices(
        problem=problem,
        draws=input_draws,
        predict=_make_unit_predictor(state, y_mean, batch_size),
        n_outer=n_outer,
        n_inner=n_inner,
        predict_group_mean=_make_group_mean(state, y_mean, batch_size),
        n_variance=n_variance,
        entropy=entropy,
        batch_size=batch_size,
        param_names=param_names,
    )
    return state, indices


def _bootstrap_replicates(
    U: Array,
    Y_flat: Array,
    *,
    gamma: float,
    ridge: float,
    max_centers: int,
    greedy_update: GreedyUpdate = "exact",
    input_draws: Draws,
    problem: Problem,
    n_outer: int,
    n_inner: int,
    n_variance: int,
    key: Array,
    n_bootstrap: int,
    batch_size: int | None,
    stacklevel: int = 2,
    nan_by_design: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Refit the surrogate on resampled training rows, once per replicate.

    This is the expensive path the ``n_bootstrap`` docstring warns about: one
    greedy fit and one full nested integration per replicate. What it buys is
    an interval that includes the surrogate's own sampling error, which an
    integration-only resample would leave out entirely.

    Each replicate gets its own quasi-random stream, split off the caller's
    key with :func:`jax.random.fold_in` rather than by adding to a seed, so
    the streams are independent rather than merely different.

    The point estimate has already warned about a poor fit or a wide ``S_U``
    clip. A replicate warning about the same thing tells the caller nothing
    new and would arrive ``n_bootstrap`` times, so warnings raised inside the
    loop are suppressed.

    Args:
        U: Training inputs in unit-cube coordinates, shape ``(N, D)``.
        Y_flat: Training outputs, shape ``(N, S)``, uncentred.
        gamma: RBF shape parameter, held at the cross-validated value.
        ridge: Kernel regularisation, held at the cross-validated value.
        greedy_update: ``"exact"`` or ``"incremental"``; see
            :func:`jaxgsa.vkoga.analyze`.
        max_centers: Maximum kernel centres the greedy may select.
        input_draws: Joint and conditional input draws for the index integration.
        problem: Problem whose declared marginals pick each parameter's
            component-fit basis.
        n_outer: Outer (conditioning) sample size per parameter.
        n_inner: Inner (conditional) sample size per outer point.
        n_variance: Sample size for the variance and the component fit.
        key: PRNG key the per-replicate streams are folded out of.
        n_bootstrap: Number of replicates.
        batch_size: Sample rows per device call, or ``None`` for the budget.
        stacklevel: ``stacklevel`` for the non-finite replicate warning, from
            :func:`jaxgsa._core.entry._user_stacklevel`.
        nan_by_design: Boolean mask over the slices, ``True`` for a slice
            whose indices are NaN anyway (a constant output). Those slices do
            not count towards the non-finite replicate warning.

    Returns:
        Mapping from index name to an ``(n_bootstrap, S, D)`` array.
    """
    n_rows = int(U.shape[0])
    draws: dict[str, list[np.ndarray]] = {name: [] for name in _INTERVAL_FIELDS}
    for replicate in range(n_bootstrap):
        stream = jax.random.fold_in(key, replicate)
        row_key, index_key = jax.random.split(stream)
        rows = jax.random.randint(row_key, shape=(n_rows,), minval=0, maxval=n_rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", JaxgsaWarning)
            _, indices = _fit_and_estimate(
                U[rows],
                Y_flat[rows],
                gamma=gamma,
                ridge=ridge,
                max_centers=max_centers,
                greedy_update=greedy_update,
                input_draws=input_draws,
                problem=problem,
                n_outer=n_outer,
                n_inner=n_inner,
                n_variance=n_variance,
                entropy=_seed_from_key(index_key),
                batch_size=batch_size,
            )
        for name in _INTERVAL_FIELDS:
            draws[name].append(np.asarray(getattr(indices, name)))
    stacked = {name: np.stack(values) for name, values in draws.items()}
    keep = np.ones(Y_flat.shape[1], dtype=bool) if nan_by_design is None else ~nan_by_design
    n_bad = int(
        np.sum(
            ~np.all(
                [
                    np.isfinite(v[:, keep]).all(axis=tuple(range(1, v.ndim)))
                    for v in stacked.values()
                ],
                axis=0,
            )
        )
    )
    if n_bad:
        hint = "a larger ridge" if jnp.finfo(U.dtype).bits == 64 else "float64 or a larger ridge"
        warnings.warn(
            f"{n_bad} of {n_bootstrap} bootstrap replicates have non-finite "
            "indices (a singular kernel fit). The intervals "
            f"skip them. Try {hint}.",
            category=JaxgsaWarning,
            stacklevel=stacklevel,
        )
    return stacked


def _warn_nonfinite_coefficients(coefficients: Array, *, stacklevel: int) -> None:
    """Warn when the fitted surrogate has NaN or infinite coefficients.

    Args:
        coefficients: The fitted expansion coefficients.
        stacklevel: ``stacklevel`` for the warning, from
            :func:`jaxgsa._core.entry._user_stacklevel`.
    """
    if not np.isfinite(np.asarray(coefficients)).all():
        warnings.warn(
            "The VKOGA surrogate has non-finite coefficients, so the indices "
            "are NaN. The kernel system is numerically singular. Try a "
            "smaller gamma, a larger ridge, or float64 "
            "(jax.config.update('jax_enable_x64', True)).",
            category=JaxgsaWarning,
            stacklevel=stacklevel,
        )


def _warn_nonfinite_slices(
    fit_slices: np.ndarray, on_invalid: str, *, inputs: bool, stacklevel: int
) -> None:
    """Warn that output slices with non-finite data get NaN indices.

    Args:
        fit_slices: Boolean mask over the flattened output slices, ``True``
            where the slice is finite and is fitted.
        on_invalid: The policy that let the non-finite data through.
        inputs: ``True`` when the inputs hold the non-finite value, which
            touches every slice.
        stacklevel: ``stacklevel`` for the warning, from
            :func:`jaxgsa._core.entry._user_stacklevel`.
    """
    if inputs:
        message = (
            f'jaxgsa.vkoga: the inputs X hold non-finite values (on_invalid="{on_invalid}"), '
            "so nothing can be fitted and every index is NaN."
        )
    else:
        n_bad = int((~fit_slices).sum())
        message = (
            f"jaxgsa.vkoga: {n_bad} of {fit_slices.size} output slice(s) hold non-finite values "
            f'(on_invalid="{on_invalid}"); their indices are NaN. The other slices are fitted '
            "without them."
        )
    warnings.warn(
        message,
        category=JaxgsaWarning,
        stacklevel=stacklevel,
    )


def _all_nan_result(
    problem: Problem,
    ctx: Any,
    *,
    n_time: int,
    n_out: int,
    dtype: Any,
    gamma: float | None,
    ridge: float | None,
    invalid: Any,
    y_mean: Array,
) -> VKOGAResult:
    """Build the result for data where every output slice is non-finite.

    Nothing can be fitted, so every index and diagnostic is NaN and the
    result carries no surrogate (``predict`` raises).

    Args:
        problem: Problem of the analysis.
        ctx: The preamble context, for the output-rank squeeze.
        n_time: Number of time points ``T``.
        n_out: Number of outputs ``K``.
        dtype: Fit dtype of the result fields.
        gamma: The caller's ``gamma``, or ``None``.
        ridge: The caller's ``ridge``, or ``None``.
        invalid: The non-finite report from the preamble.
        y_mean: Output means, shape ``(S,)``.

    Returns:
        A :class:`VKOGAResult` of NaN fields.
    """
    D = problem.n_params
    index = ctx.squeeze(jnp.full((n_time, n_out, D), jnp.nan, dtype=dtype), n_trailing=1)
    per_slice = ctx.squeeze(jnp.full((n_time, n_out), jnp.nan, dtype=dtype), n_trailing=0)
    return VKOGAResult(
        S_TC=index,
        S_TU=index,
        S_U=index,
        S_C=index,
        S_IU=index,
        problem=problem,
        variance=per_slice,
        n_centers=0,
        gamma=float("nan") if gamma is None else float(gamma),
        ridge=float("nan") if ridge is None else float(ridge),
        invalid=invalid,
        rmse=per_slice,
        cv_rmse=None if gamma is not None and ridge is not None else per_slice,
        explained_variance=per_slice,
        _fit=None,
        _y_mean=y_mean,
        _output_shape=per_slice.shape,
    )


def _seed_from_key(key: Array) -> int:
    """Fold a PRNG key down to the integer seed the QMC engines take.

    The index estimators are host-side scipy loops: ``qmc.Sobol`` scrambling
    and the cross-validation fold permutation both want a plain integer, and
    neither has a JAX PRNG interface. The public contract is still a key,
    because that is what the caller can split; this is the one place where the
    key is turned into what the host-side engines accept.

    The fold is an exclusive-or over the key's words. It is stable across the
    key implementations JAX ships, and for ``jax.random.key(s)`` with a seed
    below ``2**32`` it returns ``s`` itself, so an integer the caller wrapped
    reaches scipy unchanged.

    Args:
        key: A JAX PRNG key, typed or in the legacy ``uint32`` form.

    Returns:
        A non-negative integer seed below ``2**32``.

    Raises:
        TypeError: If ``key`` is not PRNG key data — an integer, for example.
    """
    words = np.asarray(jax.random.key_data(key), dtype=np.uint32).ravel()
    return int(np.bitwise_xor.reduce(words))


def _warn_single_precision(dtype: Any) -> None:
    """Warn that a float32 fit limits the surrogate's accuracy.

    The coefficient solve is stable in float32 (it runs in the Newton basis).
    Two limits remain, both at small ``gamma``. First, the greedy's power
    function falls to float32 rounding level after fewer centres than float64
    would select, and the greedy stops there. Second, prediction sums kernel
    translates with coefficients ``alpha = L^{-T} beta`` that grow large (up
    to about 1e6 at ``gamma = 0.1``), so float32 cancellation in that sum
    costs accuracy. Predicting through the Newton basis instead was measured
    and did not help (Ishigami, N=2048: test RMSE 1.433 against 1.443 at
    ``gamma = 0.5``, float64 1.23) at 5-7 times the cost, so the greedy stop
    is the larger of the two. Cross-validation scores in the same arithmetic,
    so it steers away from that corner of the grid, but an explicit small
    ``gamma`` can still land in it.

    Args:
        dtype: Fit dtype. The fit runs in this dtype whatever the x64 flag
            says, so float32 inputs under x64 warn too.
    """
    if jnp.finfo(dtype).bits < 64:
        warnings.warn(
            "jaxgsa.vkoga: the surrogate is fitted in single precision. For small gamma the "
            "greedy centre selection stops early and the kernel sum in prediction loses digits "
            "to cancellation, so the surrogate may be inaccurate. Enable float64 with "
            'jax.config.update("jax_enable_x64", True) and pass float64 inputs.',
            stacklevel=3,
            category=JaxgsaWarning,
        )


def _resolve_hyperparameters(
    U: Array,
    Y: Array,
    *,
    gamma: float | None,
    ridge: float | None,
    max_centers: int,
    greedy_update: GreedyUpdate = "exact",
    n_folds: int,
    seed: int,
) -> tuple[float, float, np.ndarray | None]:
    """Return ``(gamma, ridge, cv_rmse)``, cross-validating what was not given.

    A partially specified pair is handled by searching a one-element grid for
    the fixed member. That keeps one code path instead of four.

    Args:
        U: Training inputs in unit-cube coordinates, shape ``(N, D)``.
        Y: Centred training outputs, shape ``(N, S)``.
        gamma: Fixed RBF shape parameter, or ``None`` to cross-validate.
        ridge: Fixed kernel regularisation, or ``None`` to cross-validate.
        greedy_update: ``"exact"`` or ``"incremental"``; see
            :func:`jaxgsa.vkoga.analyze`.
        max_centers: Maximum kernel centres the greedy may select.
        n_folds: Folds for the cross-validation.
        seed: Seed for the fold assignment.

    Returns:
        The chosen ``gamma`` and ``ridge``, plus the out-of-sample RMSE of the
        winning grid point, one value per output slice, shape ``(S,)``. The
        grid point is chosen on the RMSE pooled over the slices, and the
        per-slice values come from the same fold residuals. The score is
        ``None`` when the caller fixed both hyperparameters, because no
        cross-validation ran.

    Raises:
        RuntimeError: If every cross-validation score is non-finite. Argmin
            over such a grid would silently pick flat index 0, the most
            ill-conditioned corner.
    """
    if gamma is not None and ridge is not None:
        return float(gamma), float(ridge), None

    gammas = np.asarray([gamma]) if gamma is not None else _GAMMA_GRID
    default_ridges = _RIDGE_GRID_F64 if jnp.finfo(U.dtype).bits == 64 else _RIDGE_GRID
    ridges = np.asarray([ridge]) if ridge is not None else default_ridges
    per_slice_jax, pooled_jax = _cross_validate(
        U,
        Y,
        gammas=gammas,
        ridges=ridges,
        max_centers=max_centers,
        greedy_update=greedy_update,
        n_folds=n_folds,
        seed=seed,
    )
    per_slice = np.asarray(per_slice_jax)
    # The grid choice uses the pooled score computed on device.
    scores = np.asarray(pooled_jax)
    finite = np.isfinite(scores)
    if not finite.any():
        hint = (
            "Increase ridge or pass gamma and ridge explicitly."
            if jnp.finfo(U.dtype).bits == 64
            else "Enable float64 with jax.config.update('jax_enable_x64', True) and pass "
            "float64 inputs, increase ridge, or pass gamma and ridge explicitly."
        )
        raise RuntimeError(
            "Every cross-validation score is non-finite; the kernel solves failed on the whole "
            f"hyperparameter grid. {hint}"
        )
    # A fold can return a non-finite score when the solve blows up; treat those
    # as the worst possible rather than letting argmin pick a NaN.
    scores = np.where(finite, scores, np.inf)
    best = np.unravel_index(int(np.argmin(scores)), scores.shape)
    return float(gammas[best[0]]), float(ridges[best[1]]), per_slice[best]


def _warn_poor_surrogate(cv_rmse: np.ndarray | None, Y_centered: Array) -> None:
    """Warn when the cross-validated fit is too weak to carry the indices.

    Every index is measured against the surrogate, not against the model, so a
    surrogate that misses most of the output variance produces indices that
    describe the surrogate alone. The failure is quiet and it can invert the
    ranking, so the honest out-of-sample score is checked here rather than left
    for the user to notice.

    Args:
        cv_rmse: Per-slice out-of-sample RMSE of the chosen hyperparameters,
            shape ``(S,)``, or ``None`` when the caller fixed both and no cross-validation ran.
        Y_centered: Centred training outputs, shape ``(N, S)``. They set the
            scale to compare against.

    Warns:
        JaxgsaWarning: If ``cv_rmse`` is above ``_CV_RMSE_WARN_FRACTION`` of the
            pooled output standard deviation.
    """
    if cv_rmse is None:
        return
    scale = float(jnp.sqrt(jnp.mean(Y_centered**2)))
    pooled = float(np.sqrt(np.mean(np.asarray(cv_rmse) ** 2)))
    if scale <= 0.0 or not np.isfinite(pooled):
        return
    relative = pooled / scale
    if relative <= _CV_RMSE_WARN_FRACTION:
        return
    warnings.warn(
        f"jaxgsa.vkoga: the cross-validated surrogate error is {relative:.2f} of the output "
        "standard deviation, so the surrogate misses most of the output variation. Every index "
        "is computed against the surrogate, so the reported values — including the ranking — "
        "are not trustworthy. This happens on high-frequency or oscillatory responses, which a "
        "greedy Gaussian kernel cannot resolve. Add training points, or use a method that does "
        "not need a surrogate (jaxgsa.kucherenko on a conditional design).",
        stacklevel=3,
        category=JaxgsaWarning,
    )


def _unit_evaluator(state, y_mean: Array):
    """Build the shared unit-cube evaluation kernel and its memory cost.

    One place defines how a fitted state is evaluated: the kernel expansion
    plus the training mean. It also defines what one row of that evaluation
    costs. The ``(n, n_centers)`` kernel block dominates the transient memory,
    and each prediction adds one output row. Both the estimator-facing
    predictor and the public ``predict`` plan build on it.

    Args:
        state: Fitted (sliced) surrogate state.
        y_mean: Training output mean to restore, shape ``(S,)``.

    Returns:
        ``(kernel, bytes_per_row)``. The kernel maps ``(n, D)`` unit-cube rows
        to ``(n, S)`` outputs. The second item is the per-row transient-memory
        estimate.
    """
    n_centers = int(state.centers.shape[0])
    n_slices = int(state.coefficients.shape[1])
    itemsize = int(state.centers.dtype.itemsize)

    def kernel(U: Array) -> Array:
        return _predict_vkoga(state, U) + y_mean

    return kernel, itemsize * (n_centers + n_slices)


@jax.jit
def _eval_rows(
    U: Array, centers: Array, coefficients: Array, gamma: Array, y_mean: Array
) -> Array:
    """Evaluate the surrogate at ``(n, D)`` unit-cube rows.

    The fitted arrays are arguments, not closure constants, so one compiled
    program serves every fit of the same shape and dtype: a new dataset or a
    bootstrap replicate compiles nothing.

    Args:
        U: Unit-cube rows, shape ``(n, D)``.
        centers: Kernel centres, shape ``(m, D)``.
        coefficients: Expansion coefficients, shape ``(m, S)``.
        gamma: Gaussian shape parameter.
        y_mean: Training output mean, shape ``(S,)``.

    Returns:
        Predictions, shape ``(n, S)``.
    """
    return _gaussian_kernel(U, centers, gamma) @ coefficients + y_mean


@functools.partial(jax.jit, static_argnums=5)
def _eval_group_means(
    U: Array, centers: Array, coefficients: Array, gamma: Array, y_mean: Array, group: int
) -> Array:
    """Mean surrogate value over consecutive blocks of ``group`` rows.

    The kernel is evaluated in the dtype of ``U`` and ``centers``; the block
    mean and the product with the coefficients run in the dtype of
    ``coefficients``, which the caller may widen to float64.

    Args:
        U: Unit-cube rows, shape ``(n, D)``, ``n`` a multiple of ``group``.
        centers: Kernel centres, shape ``(m, D)``.
        coefficients: Expansion coefficients, shape ``(m, S)``.
        gamma: Gaussian shape parameter.
        y_mean: Training output mean, shape ``(S,)``.
        group: Block size.

    Returns:
        Block means, shape ``(n / group, S)``.
    """
    K = _gaussian_kernel(U, centers, gamma).astype(coefficients.dtype)
    return K.reshape(-1, group, K.shape[1]).mean(axis=1) @ coefficients + y_mean


def _make_unit_predictor(state, y_mean: Array, batch_size: int | None):
    """Build the ``(n, D) unit-cube -> (n, S)`` callable the estimators use.

    The index estimators are host-side quasi-Monte-Carlo loops, so the returned
    function is plain NumPy-in and NumPy-out. Batching keeps the kernel matrix
    within the configured memory budget for the millions of conditional draws.

    The jitted kernel is memoised per distinct ``(state, y_mean)`` content
    (see :data:`_PREDICTOR_CACHE`). Building the predictor wraps the surrogate
    kernel in a fresh ``jax.jit`` whose closure freezes the fitted state; a
    fresh wrapper on every :func:`_fit_and_estimate` call re-traces and
    recompiles it inside the estimator's first prediction (~30 ms at
    ``n_centers=32``), even though every steady-state analysis on the same
    data fits the identical state. Memoising on the content reuses the exact
    compiled kernel for repeated analyses (bootstrap / parameter sweep /
    interactive re-runs), while a genuinely new fit -- new centres,
    coefficients or output mean -- still builds and compiles its own
    predictor, exactly as before.

    Args:
        state: Fitted (sliced) surrogate state.
        y_mean: Training output mean to restore, shape ``(S,)``.
        batch_size: Rows per batch, or ``None`` to derive one from the memory
            budget.

    Returns:
        A callable mapping ``(n, D)`` unit-cube rows to ``(n, S)`` outputs.
    """
    # The kernel reads centres, coefficients and gamma, and adds ``y_mean``;
    # those four arrays decide the compiled executable, so the memo key is a
    # digest of their full content plus the (runtime) batch policy.
    key = (
        _content_digest(state.centers, state.coefficients, state.gamma, y_mean),
        batch_size,
    )
    pred = _PREDICTOR_CACHE.get(key)
    if pred is not None:
        return pred

    _, bytes_per_row = _unit_evaluator(state, y_mean)
    centers, coefficients, gamma = state.centers, state.coefficients, state.gamma

    def predict(U: np.ndarray) -> np.ndarray:
        n_rows = U.shape[0]
        # Batches use half the budget: the index estimators' queue and draws
        # hold the other half (see vkoga._indices._QUEUE_BUDGET_SHARE). The
        # queue flushes blocks of many sizes, so each batch is padded to a
        # power of two or to the budgeted width. The kernel then compiles for
        # a few shapes, not once per block size.
        batch = resolve_batch_size(
            _PREDICT_BUDGET_SHARE * bytes_per_row, _next_power_of_2(n_rows), batch_size
        )
        # The batch width can change the rounding of the kernel products
        # (about 3e-8 relative in float32 at a one-row batch), so results
        # agree across memory budgets to rounding, not bit for bit.
        # Float64 on the host, which is what the estimators read, so the
        # queue does not need a second, converted copy.
        out = np.empty((n_rows, int(state.coefficients.shape[1])), dtype=np.float64)
        for start in range(0, n_rows, batch):
            rows = U[start : start + batch]
            n_valid = rows.shape[0]
            if n_valid < batch:
                rows = np.concatenate(
                    [rows, np.zeros((batch - n_valid, rows.shape[1]), dtype=rows.dtype)]
                )
            # The estimators hand over float64 host rows. Evaluate them in the
            # fit's own dtype: a float32 fit under x64 would otherwise run a
            # float64 kernel matrix, about 2.5 times slower for no accuracy.
            batch_rows = jnp.asarray(rows, dtype=state.centers.dtype)
            pred = _eval_rows(batch_rows, centers, coefficients, gamma, y_mean)
            out[start : start + n_valid] = np.asarray(pred)[:n_valid]
        return out

    if len(_PREDICTOR_CACHE) >= _PREDICTOR_CACHE_MAX:
        _PREDICTOR_CACHE.pop(next(iter(_PREDICTOR_CACHE)))
    _PREDICTOR_CACHE[key] = predict
    return predict


def _make_group_mean(
    state, y_mean: Array, batch_size: int | None
) -> Callable[[np.ndarray, int], np.ndarray] | None:
    """Build the ``(n, D) -> (n / g, S)`` evaluator of inner-block means.

    ``S_TC`` needs only the mean prediction over each inner block. The
    surrogate is linear in its coefficients, so that mean is the mean kernel
    row times the coefficients. This replaces ``g`` products with the
    ``(m, S)`` coefficients by one, which is most of the prediction cost when
    ``S`` is large. The two orders agree in exact arithmetic.

    In floating point they do not, for a float32 fit. The coefficients of a
    small-``gamma`` fit are large (up to about 1e6), so a float32 mean kernel
    row times them rounds once with a large relative error, while the plain
    path rounds each prediction and averages in float64. Measured on a fixed
    float32 surrogate against a float64 evaluation of it: a float32 product
    shifted ``S_TC`` by up to 1e-2 (G-function, ``gamma = 0.066``) and 5e-2
    (Ishigami, ``gamma = 0.1``). So for a float32 fit the block mean and the
    product run in float64, which needs x64; without x64 this returns
    ``None`` and the caller averages the predictions instead. The kernel
    itself stays in the fit's dtype. Memoised like
    :func:`_make_unit_predictor`, and the compiled program takes the fitted
    arrays as arguments, so a new fit of the same shape compiles nothing.

    Args:
        state: Fitted (sliced) surrogate state.
        y_mean: Training output mean to restore, shape ``(S,)``.
        batch_size: Rows per batch, or ``None`` to derive one from the memory
            budget.

    Returns:
        A callable ``(U, g)`` mapping ``(n, D)`` unit-cube rows, ``n`` a
        multiple of ``g``, to the ``(n / g, S)`` float64 block means, or
        ``None`` for a float32 fit without x64.
    """
    single = jnp.finfo(state.centers.dtype).bits < 64
    if single and not x64_enabled():
        return None
    key = (
        _content_digest(state.centers, state.coefficients, state.gamma, y_mean),
        batch_size,
    )
    cached = _GROUP_MEAN_CACHE.get(key)
    if cached is not None:
        return cached

    _, bytes_per_row = _unit_evaluator(state, y_mean)
    centers, gamma = state.centers, state.gamma
    # Accumulate a float32 fit's block means and product in float64.
    coefficients = state.coefficients.astype(jnp.float64) if single else state.coefficients
    shift = jnp.asarray(y_mean, dtype=coefficients.dtype)

    def group_mean(U: np.ndarray, group: int) -> np.ndarray:
        n_rows = U.shape[0]
        if n_rows % group:
            raise ValueError(f"{n_rows} rows do not split into blocks of {group}")
        batch = resolve_batch_size(
            _PREDICT_BUDGET_SHARE * bytes_per_row, _next_power_of_2(n_rows), batch_size
        )
        # Whole blocks per batch; the last batch is padded with whole blocks.
        batch = max(group, batch // group * group)
        out = np.empty((n_rows // group, int(coefficients.shape[1])), dtype=np.float64)
        for start in range(0, n_rows, batch):
            rows = U[start : start + batch]
            n_valid = rows.shape[0]
            if n_valid < batch:
                rows = np.concatenate(
                    [rows, np.zeros((batch - n_valid, rows.shape[1]), dtype=rows.dtype)]
                )
            # Evaluate in the fit's dtype, as _make_unit_predictor does.
            batch_rows = jnp.asarray(rows, dtype=centers.dtype)
            means = _eval_group_means(batch_rows, centers, coefficients, gamma, shift, group)
            means = np.asarray(means)[: n_valid // group]
            out[start // group : (start + n_valid) // group] = means
        return out

    if len(_GROUP_MEAN_CACHE) >= _PREDICTOR_CACHE_MAX:
        _GROUP_MEAN_CACHE.pop(next(iter(_GROUP_MEAN_CACHE)))
    _GROUP_MEAN_CACHE[key] = group_mean
    return group_mean


def _vkoga_predict_plan(result: VKOGAResult, X_new: Array) -> _PredictPlan:
    """Plan a batched surrogate evaluation for :meth:`VKOGAResult.predict`.

    Args:
        result: Fitted result carrying the kernel state.
        X_new: Inputs in physical units, shape ``(N_new, D)``.

    Returns:
        A ``_PredictPlan`` whose kernel returns predictions shaped to the
        result's output contract.

    Raises:
        ValueError: If ``result`` carries no fitted surrogate state.
    """
    state = result._fit
    if state is None or result._y_mean is None:
        raise ValueError("VKOGAResult does not contain a fitted surrogate")

    U = cdf_to_unit_interval(X_new, result.problem).astype(state.centers.dtype)
    unit_kernel, bytes_per_row = _unit_evaluator(state, result._y_mean)
    output_shape = result._output_shape

    def kernel(U_batch: Array) -> Array:
        return unit_kernel(U_batch).reshape(U_batch.shape[0], *output_shape)

    return _PredictPlan(X=U, bytes_per_row=bytes_per_row, kernel=kernel)
