"""Two-stage correlated sensitivity analysis: VKOGA surrogate, then indices.

Implements the surrogate-based sensitivity analysis (SSA) of Hilhorst et al.
(2024). Stage one fits a greedy kernel surrogate to the given ``(X, Y)`` data.
Stage two computes the correlated variance-based indices of Li et al. (2010)
against that surrogate under a Gaussian copula. The split is what makes the
method affordable. The indices need nested conditional sampling, which is
unaffordable against an expensive model but cheap against a kernel expansion.

**This module has no pure core.** Stage two is a host NumPy/SciPy quasi-Monte
-Carlo loop end to end, so there is no ``indices()`` here that returns bare
arrays and survives ``jit``, ``vmap`` and ``jacrev``. See
``docs/adr/0015-pure-core-exemptions.md``.

References:
    Hilhorst, Quicken, van de Vosse & Huberts (2024). Int. J. Numer. Meth.
        Biomed. Engng. 40(2):e3797.
    Li, Rabitz, Yelvington et al. (2010). J. Phys. Chem. A 114:6022-6032.
    Wirtz & Haasdonk (2013). Dolomites Res. Notes Approx. 6:83-100.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.batching import apply_batched, resolve_batch_size
from jaxgsa._core.bootstrap import _bootstrap_ci_endpoints
from jaxgsa._core.copula import (
    _ConditionalPlan,
    build_conditional_plan,
    canonicalize_correlation,
    fit_gaussian_copula,
    independent_correlation,
    is_independent,
)
from jaxgsa._core.entry import at_least, in_open_interval, one_of, prepare, require
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.precision import x64_enabled
from jaxgsa._core.result import CIInfo
from jaxgsa._core.sampling import _next_power_of_2
from jaxgsa._core.surrogate import _PredictPlan
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.problem import Problem
from jaxgsa.vkoga._engine import _cross_validate, _fit_vkoga, _predict_vkoga
from jaxgsa.vkoga._indices import estimate_correlated_indices
from jaxgsa.vkoga._result import VKOGAResult

# Hyperparameter search grid, following Hilhorst et al. Section 2.4.1: ten
# log-spaced values each, cross-validated as a 10x10 product.
_GAMMA_GRID = np.logspace(-2, np.log10(50.0), 10)
_RIDGE_GRID = np.logspace(-16, -2, 10)

# Cap on kernel centres when the caller does not choose one. Greedy selection
# cost is O(max_centers * n), and the marginal accuracy of further centres
# falls off quickly once the power function has collapsed.
_DEFAULT_MAX_CENTERS = 300

# Largest cross-validated RMSE, as a fraction of the output standard deviation,
# that still gives usable indices. At 0.5 the surrogate leaves a quarter of the
# output variance unexplained; past that the measured indices stop tracking the
# true ones, and on oscillatory models they can even invert the ranking.
_CV_RMSE_WARN_FRACTION = 0.5

# Largest fitted off-diagonal latent correlation the training X may show
# before the design is called correlated. Spearman rank-correlation noise on
# an independent design of n rows has standard deviation about 1/sqrt(n)
# (~0.18 at n=30, ~0.07 at n=200), and the check takes a max over D(D-1)/2
# entries, which inflates the expected extreme further. So at very small n
# with many parameters this can fire on an independent design; at the
# training sizes the surrogate needs to be any good (hundreds of rows) the
# noise sits well below 0.3. A training set actually drawn from the analysis
# copula, the mistake this check catches, fits near the declared entries,
# which sit well above 0.3 whenever the correlation matters at all. The
# consequence of a false positive is one warning, not a wrong number.
_TRAINING_CORRELATION_WARN = 0.3

# The index fields a bootstrap reports an interval for. The diagnostics
# (variance, rmse, cv_rmse, n_centers, gamma, ridge) get none: they describe
# the fit, not the sensitivity of the model, and a caller reading them wants
# the value the reported surrogate actually has.
_INTERVAL_FIELDS = ("S_TC", "S_TU", "S_U", "S_C", "S_IU")


def analyze(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    correlation: Array | np.ndarray | None = None,
    gamma: float | None = None,
    ridge: float | None = None,
    max_centers: int | None = None,
    n_folds: int = 10,
    n_outer: int = 512,
    n_inner: int = 128,
    n_variance: int = 8192,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    batch_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> VKOGAResult:
    """Correlated variance-based sensitivity indices via a VKOGA surrogate.

    Fits a Vectorial Kernel Orthogonal Greedy Algorithm surrogate to given
    ``(X, Y)`` data, then estimates the five correlated indices of Li et al.
    (2010) against it under a Gaussian copula.

    The training design must be independent and space-filling even when the
    analysis is correlated. The correlated measure concentrates on a ridge, but
    ``S_TU`` conditions on the other parameters and then resamples ``X_i``
    across its whole marginal. A surrogate trained only on correlated data
    would be extrapolating for exactly those draws.

    Args:
        problem: Problem defining the parameters and their marginals.
        X: Inputs in physical units, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        correlation: Gaussian-copula dependency structure. ``None`` (default)
            reads ``problem.correlation`` and falls back to independent inputs
            when the problem declares none. A ``(D, D)`` matrix overrides the
            problem's declaration for this call. To fit a matrix from data,
            use :func:`jaxgsa.sampling.fit_correlation` and attach it with
            ``problem.with_correlation(...)``.
        gamma: RBF shape parameter, finite and positive when given. ``None``
            cross-validates over a grid.
        ridge: Kernel regularisation, finite and positive when given. ``None``
            cross-validates over a grid.
        max_centers: Maximum kernel centres the greedy may select, at least 1.
            Defaults to 300, capped at ``N``.
        n_folds: Folds for hyperparameter cross-validation, at least 2.
        n_outer: Outer (conditioning) sample size per parameter, at least 2.
            Rounded up to the next power of two (Sobol' balance).
        n_inner: Inner (conditional) sample size per outer point, at least 2.
            Rounded up to the next power of two. Do not lower it below the
            default to buy speed: the estimators drop the iid inner-noise
            correction and depend on a large shared inner block, so a small
            ``n_inner`` inflates a small ``S_TC``. Raise ``n_outer`` instead.
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
            ``correlation`` is not ``None`` or a valid matrix, if ``gamma`` or
            ``ridge`` is not finite and positive, or if a size argument is out
            of range, if ``key`` is ``None``, if ``on_invalid`` is not one of
            the three policies, or
            if ``on_invalid="raise"`` (the default) and ``X`` or ``Y`` holds a
            non-finite value.
        RuntimeError: If every cross-validation score is non-finite. Non-finite
            training data no longer reaches this check; it is now a genuine
            solver failure.

    Warns:
        JaxgsaWarning: In any of five cases. An output slice has zero variance.
            JAX is in single precision, where the kernel solve loses accuracy
            for small ``gamma`` (see :mod:`jaxgsa.vkoga`). The cross-validated
            surrogate error is a large fraction of the output standard
            deviation, which makes every index untrustworthy. ``S_U`` had to be
            clipped to ``S_TU`` by a wide margin, which says the additive
            component functions cannot represent the model's interactions.
            The analysis is correlated but the training ``X`` is itself
            correlated, so the surrogate extrapolates on the conditional
            draws; train on an independent design and declare the dependence
            in ``problem.correlation`` instead.
    """
    from jaxgsa.vkoga import SPEC

    D = problem.num_vars
    # Every scalar argument is checked before any expensive work
    # (cross-validation, fitting, index estimation). The preamble runs them
    # first, so that ordering is structural rather than a convention.
    ctx = prepare(
        SPEC,
        problem,
        Y,
        X=X,
        on_invalid=on_invalid,
        checks=(
            at_least("max_centers", max_centers, 1),
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
            at_least("n_bootstrap", n_bootstrap, 0),
            one_of("ci_method", ci_method, ("quantile", "gaussian")),
            in_open_interval("conf_level", conf_level, 0.0, 1.0),
            require(
                D >= 2,
                f"Correlated sensitivity indices need at least 2 parameters, got {D}",
            ),
        ),
        min_kept=n_folds,
    )
    # Checked after the preamble so a problem this method refuses outright
    # (categorical parameters) still reports that, and not a missing key.
    if key is None:
        raise ValueError("key is required for the Monte-Carlo index estimate")
    seed = _seed_from_key(key)
    X, Y, invalid = ctx.inputs, ctx.Y, ctx.invalid
    # The clock starts before the hyperparameter search and the greedy fit:
    # for this method the fit is the expensive work, not just the indices.
    t0 = _verbose.tic()

    if max_centers is None:
        max_centers = _DEFAULT_MAX_CENTERS
    # The latent draws come from Sobol' sequences, which need power-of-two
    # sizes to keep their balance guarantees (scipy warns otherwise). The
    # defaults are already powers of two, so they pass through unchanged.
    n_outer = _next_power_of_2(n_outer)
    n_inner = _next_power_of_2(n_inner)
    n_variance = _next_power_of_2(n_variance)

    Y_canonical = ctx.Y3
    n_time, n_out = Y_canonical.shape[1], Y_canonical.shape[2]
    Y_flat = Y_canonical.reshape(Y_canonical.shape[0], n_time * n_out)

    _warn_single_precision()
    R = _resolve_correlation(problem, correlation)
    _warn_correlated_training_design(problem, X, R)

    # The RBF kernel is isotropic, so every column must share a scale; the
    # marginal CDF map is the same transform HDMR uses for its basis.
    U = cdf_to_unit_interval(X, problem)
    # VKOGA carries no constant term, so a non-zero output mean would have to
    # be reconstructed by the kernel expansion itself. Centring removes that
    # burden and measurably improves the fit.
    y_mean = Y_flat.mean(axis=0)
    Y_centered = Y_flat - y_mean

    resolved_centers = min(int(max_centers), int(U.shape[0]))
    gamma_value, ridge_value, cv_rmse = _resolve_hyperparameters(
        U,
        Y_centered,
        gamma=gamma,
        ridge=ridge,
        max_centers=resolved_centers,
        n_folds=n_folds,
        seed=seed,
    )
    _warn_poor_surrogate(cv_rmse, Y_centered)
    plan = build_conditional_plan(R)
    state, indices = _fit_and_estimate(
        U,
        Y_flat,
        gamma=gamma_value,
        ridge=ridge_value,
        max_centers=resolved_centers,
        plan=plan,
        n_outer=n_outer,
        n_inner=n_inner,
        n_variance=n_variance,
        entropy=seed,
        batch_size=batch_size,
        param_names=problem.names,
    )

    def _shape_index(flat: np.ndarray) -> Array:
        """Reshape an ``(..., S, D)`` index block to the output contract."""
        arr = jnp.asarray(flat.reshape(*flat.shape[:-2], n_time, n_out, D))
        return ctx.squeeze(arr, n_trailing=1)

    def _shape_slice(flat: np.ndarray) -> Array:
        """Reshape an ``(S,)`` per-slice diagnostic to the output contract."""
        arr = jnp.asarray(flat.reshape(n_time, n_out))
        return ctx.squeeze(arr, n_trailing=0)

    confs: dict[str, Array | None] = dict.fromkeys(_INTERVAL_FIELDS)
    ci: CIInfo | None = None
    if n_bootstrap > 0:
        replicates = _bootstrap_replicates(
            U,
            Y_flat,
            gamma=gamma_value,
            ridge=ridge_value,
            max_centers=resolved_centers,
            plan=plan,
            n_outer=n_outer,
            n_inner=n_inner,
            n_variance=n_variance,
            key=key,
            n_bootstrap=n_bootstrap,
            batch_size=batch_size,
        )
        for name in _INTERVAL_FIELDS:
            point = jnp.asarray(getattr(indices, name).reshape(n_time, n_out, D))
            draws = jnp.asarray(replicates[name].reshape(-1, n_time, n_out, D))
            lo, hi = _bootstrap_ci_endpoints(
                point, draws, conf_level=conf_level, ci_method=ci_method
            )
            # Stack [lower, upper] into a leading axis of size 2. n_trailing=1
            # says one axis (D) follows the promoted (T, K) pair; the squeeze
            # addresses those axes from the end, so the leading 2 survives.
            confs[name] = ctx.squeeze(jnp.stack((lo, hi)), n_trailing=1)
        ci = CIInfo(
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            replicates=(
                {name: _shape_index(replicates[name]) for name in _INTERVAL_FIELDS}
                if keep_replicates
                else None
            ),
        )

    output_shape = ctx.squeeze(jnp.zeros((n_time, n_out)), n_trailing=0).shape
    result = VKOGAResult(
        S_TC=_shape_index(indices.S_TC),
        S_TU=_shape_index(indices.S_TU),
        S_U=_shape_index(indices.S_U),
        S_C=_shape_index(indices.S_C),
        S_IU=_shape_index(indices.S_IU),
        problem=problem,
        correlation=R,
        variance=_shape_slice(indices.variance),
        n_centers=int(state.n_centers),
        gamma=float(gamma_value),
        ridge=float(ridge_value),
        invalid=invalid,
        rmse=_shape_slice(np.asarray(state.rmse)),
        cv_rmse=cv_rmse,
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
    plan: _ConditionalPlan,
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
        max_centers: Maximum kernel centres the greedy may select.
        plan: Precomputed Gaussian conditionals for the index integration.
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
        ridge=ridge,
    )
    # The fitted state is padded to the static max_centers size (rows past
    # n_centers hold zero coefficients). Slice it down host-side so predict
    # and the index estimators pay only for the centres the greedy selected.
    n_selected = int(state.n_centers)
    state = state._replace(
        centers=state.centers[:n_selected],
        coefficients=state.coefficients[:n_selected],
    )
    indices = estimate_correlated_indices(
        plan=plan,
        predict=_make_unit_predictor(state, y_mean, batch_size),
        n_outer=n_outer,
        n_inner=n_inner,
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
    plan: _ConditionalPlan,
    n_outer: int,
    n_inner: int,
    n_variance: int,
    key: Array,
    n_bootstrap: int,
    batch_size: int | None,
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
        max_centers: Maximum kernel centres the greedy may select.
        plan: Precomputed Gaussian conditionals for the index integration.
        n_outer: Outer (conditioning) sample size per parameter.
        n_inner: Inner (conditional) sample size per outer point.
        n_variance: Sample size for the variance and the component fit.
        key: PRNG key the per-replicate streams are folded out of.
        n_bootstrap: Number of replicates.
        batch_size: Sample rows per device call, or ``None`` for the budget.

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
                plan=plan,
                n_outer=n_outer,
                n_inner=n_inner,
                n_variance=n_variance,
                entropy=_seed_from_key(index_key),
                batch_size=batch_size,
            )
        for name in _INTERVAL_FIELDS:
            draws[name].append(np.asarray(getattr(indices, name)))
    return {name: np.stack(values) for name, values in draws.items()}


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


def _warn_single_precision() -> None:
    """Warn that float32 limits the kernel solve's accuracy.

    The regularised normal equations square the condition number of the cross
    kernel, which for small ``gamma`` exceeds what float32 can carry. Cross
    validation partly self-corrects by scoring in the same arithmetic and so
    avoiding the blown-up corner of the grid, but the ceiling is real.
    """
    if not x64_enabled():
        warnings.warn(
            "jaxgsa.vkoga: JAX is in single precision; the kernel solve is ill-conditioned for "
            "small gamma and the surrogate may be inaccurate. Enable float64 with "
            'jax.config.update("jax_enable_x64", True) before fitting.',
            stacklevel=3,
            category=JaxgsaWarning,
        )


def _warn_correlated_training_design(
    problem: Problem,
    X: Array,
    R: np.ndarray,
) -> None:
    """Warn when a correlated analysis is fed a correlated training design.

    The estimator requires an independent, space-filling training ``X`` even
    when the analysis correlation is not the identity. The correlated measure
    concentrates on a ridge, but ``S_TU`` conditions on the other parameters
    and then resamples ``X_i`` across its whole marginal, so the surrogate is
    evaluated far off that ridge. A surrogate trained only on correlated data
    extrapolates for exactly those draws, and every index quietly inherits
    the extrapolation error.

    The check fits the training design's own empirical latent correlation
    (rank-based, so it is invariant to the marginals) and compares its
    largest off-diagonal entry against ``_TRAINING_CORRELATION_WARN``. It
    runs only when the analysis correlation is not the identity: an
    independent analysis never conditions, so a structured training design is
    then the user's own business. The fit's data-quality warnings are
    suppressed here — this is a diagnostic on a sample the invalid check has
    already vetted, not a fit the user asked for.

    Args:
        problem: Problem the training sample belongs to.
        X: Training inputs in physical units, shape ``(N, D)``, after the
            non-finite policy ran.
        R: Resolved analysis correlation matrix, shape ``(D, D)``.

    Warns:
        JaxgsaWarning: If the training design's fitted latent correlation has
            an off-diagonal entry above ``_TRAINING_CORRELATION_WARN``.
    """
    if is_independent(R):
        return
    with warnings.catch_warnings():
        # fit_gaussian_copula warns about degenerate columns and material
        # repairs. Those describe a fit the user asked for; this internal
        # diagnostic only reads the magnitude.
        warnings.simplefilter("ignore", JaxgsaWarning)
        R_fit = fit_gaussian_copula(problem, np.asarray(X))
    off_diag = np.abs(R_fit - np.diag(np.diag(R_fit)))
    largest = float(off_diag.max())
    if largest <= _TRAINING_CORRELATION_WARN:
        return
    i, j = np.unravel_index(int(off_diag.argmax()), off_diag.shape)
    warnings.warn(
        f"jaxgsa.vkoga: the training X is itself correlated (fitted latent "
        f"correlation between {problem.names[int(i)]!r} and "
        f"{problem.names[int(j)]!r} is {largest:.2f}). The estimator needs an "
        "independent, space-filling training design even for a correlated "
        "analysis: S_TU conditions on the other parameters and then resamples "
        "each X_i across its whole marginal, so a surrogate trained only on "
        "correlated data extrapolates for exactly those conditional draws. "
        "Train on an independent design; the dependence structure belongs in "
        "problem.correlation (or the correlation= argument), not in the "
        "training data.",
        stacklevel=3,
        category=JaxgsaWarning,
    )


def _resolve_correlation(
    problem: Problem,
    correlation: Array | np.ndarray | None,
) -> np.ndarray:
    """Turn the ``correlation`` argument into a validated latent matrix.

    ``None`` reads ``problem.correlation``, and falls back to independent
    inputs when the problem declares none. A matrix is an explicit per-call
    override and is canonicalized like a constructor argument. Strings are
    rejected. The one workflow for fitting a matrix from data is
    ``problem.with_correlation(jaxgsa.sampling.fit_correlation(problem, X))``,
    which states in the call which sample the copula comes from.

    Args:
        problem: Problem whose declared correlation is the fallback.
        correlation: Per-call override, shape ``(D, D)``, or ``None``.

    Returns:
        A canonicalized latent correlation matrix, shape ``(D, D)``.

    Raises:
        ValueError: If ``correlation`` is a string, or not a valid matrix.
    """
    if correlation is None:
        declared = problem.correlation
        if declared is None:
            return independent_correlation(problem.num_vars)
        # Problem construction already canonicalized the declared matrix.
        return declared
    if isinstance(correlation, str):
        raise ValueError(
            f"correlation must be None or a (D, D) matrix, got {correlation!r}. To fit a "
            "matrix from observed data, use jaxgsa.sampling.fit_correlation(problem, X_data) "
            "and attach it with problem.with_correlation(...)."
        )
    # A per-call override is user-declared, so it takes the strict repair
    # policy: a matrix that has to move materially is rejected, not repaired.
    return canonicalize_correlation(
        correlation, problem.num_vars, policy="declared", method="jaxgsa.vkoga.analyze"
    )


def _resolve_hyperparameters(
    U: Array,
    Y: Array,
    *,
    gamma: float | None,
    ridge: float | None,
    max_centers: int,
    n_folds: int,
    seed: int,
) -> tuple[float, float, float | None]:
    """Return ``(gamma, ridge, cv_rmse)``, cross-validating what was not given.

    A partially specified pair is handled by searching a one-element grid for
    the fixed member. That keeps one code path instead of four.

    Args:
        U: Training inputs in unit-cube coordinates, shape ``(N, D)``.
        Y: Centred training outputs, shape ``(N, S)``.
        gamma: Fixed RBF shape parameter, or ``None`` to cross-validate.
        ridge: Fixed kernel regularisation, or ``None`` to cross-validate.
        max_centers: Maximum kernel centres the greedy may select.
        n_folds: Folds for the cross-validation.
        seed: Seed for the fold assignment.

    Returns:
        The chosen ``gamma`` and ``ridge``, plus the pooled out-of-sample RMSE
        of the winning grid point. The score is ``None`` when the caller fixed
        both hyperparameters, because no cross-validation ran.

    Raises:
        RuntimeError: If every cross-validation score is non-finite. Argmin
            over such a grid would silently pick flat index 0, the most
            ill-conditioned corner.
    """
    if gamma is not None and ridge is not None:
        return float(gamma), float(ridge), None

    gammas = np.asarray([gamma]) if gamma is not None else _GAMMA_GRID
    ridges = np.asarray([ridge]) if ridge is not None else _RIDGE_GRID
    scores = np.asarray(
        _cross_validate(
            U,
            Y,
            gammas=gammas,
            ridges=ridges,
            max_centers=max_centers,
            n_folds=n_folds,
            seed=seed,
        )
    )
    finite = np.isfinite(scores)
    if not finite.any():
        raise RuntimeError(
            "Every cross-validation score is non-finite; the kernel solves failed on the whole "
            "hyperparameter grid. Enable float64 with jax.config.update('jax_enable_x64', True), "
            "increase ridge, or pass gamma and ridge explicitly."
        )
    # A fold can return a non-finite score when the solve blows up; treat those
    # as the worst possible rather than letting argmin pick a NaN.
    scores = np.where(finite, scores, np.inf)
    best = np.unravel_index(int(np.argmin(scores)), scores.shape)
    return float(gammas[best[0]]), float(ridges[best[1]]), float(scores[best])


def _warn_poor_surrogate(cv_rmse: float | None, Y_centered: Array) -> None:
    """Warn when the cross-validated fit is too weak to carry the indices.

    Every index is measured against the surrogate, not against the model, so a
    surrogate that misses most of the output variance produces indices that
    describe the surrogate alone. The failure is quiet and it can invert the
    ranking, so the honest out-of-sample score is checked here rather than left
    for the user to notice.

    Args:
        cv_rmse: Pooled out-of-sample RMSE of the chosen hyperparameters, or
            ``None`` when the caller fixed both and no cross-validation ran.
        Y_centered: Centred training outputs, shape ``(N, S)``. They set the
            scale to compare against.

    Warns:
        JaxgsaWarning: If ``cv_rmse`` is above ``_CV_RMSE_WARN_FRACTION`` of the
            pooled output standard deviation.
    """
    if cv_rmse is None:
        return
    scale = float(jnp.sqrt(jnp.mean(Y_centered**2)))
    if scale <= 0.0 or not np.isfinite(cv_rmse):
        return
    relative = cv_rmse / scale
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
    itemsize = int(jnp.zeros(()).itemsize)

    def kernel(U: Array) -> Array:
        return _predict_vkoga(state, U) + y_mean

    return kernel, itemsize * (n_centers + n_slices)


def _make_unit_predictor(state, y_mean: Array, batch_size: int | None):
    """Build the ``(n, D) unit-cube -> (n, S)`` callable the estimators use.

    The index estimators are host-side quasi-Monte-Carlo loops, so the returned
    function is plain NumPy-in and NumPy-out. Batching keeps the kernel matrix
    within the configured memory budget for the millions of conditional draws.

    Args:
        state: Fitted (sliced) surrogate state.
        y_mean: Training output mean to restore, shape ``(S,)``.
        batch_size: Rows per batch, or ``None`` to derive one from the memory
            budget.

    Returns:
        A callable mapping ``(n, D)`` unit-cube rows to ``(n, S)`` outputs.
    """
    kernel, bytes_per_row = _unit_evaluator(state, y_mean)
    compiled = jax.jit(kernel)

    def predict(U: np.ndarray) -> np.ndarray:
        U_device = jnp.asarray(U)
        batch = resolve_batch_size(bytes_per_row, U_device.shape[0], batch_size)
        return np.asarray(apply_batched(compiled, U_device, batch))

    return predict


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

    U = cdf_to_unit_interval(X_new, result.problem)
    unit_kernel, bytes_per_row = _unit_evaluator(state, result._y_mean)
    output_shape = result._output_shape

    def kernel(U_batch: Array) -> Array:
        return unit_kernel(U_batch).reshape(U_batch.shape[0], *output_shape)

    return _PredictPlan(X=U, bytes_per_row=bytes_per_row, kernel=kernel)
