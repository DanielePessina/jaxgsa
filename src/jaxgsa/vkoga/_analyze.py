"""Two-stage correlated sensitivity analysis: VKOGA surrogate, then indices.

Implements the surrogate-based sensitivity analysis (SSA) of Hilhorst et al.
(2024). Stage one fits a greedy kernel surrogate to the given ``(X, Y)`` data.
Stage two computes the correlated variance-based indices of Li et al. (2010)
against that surrogate under a Gaussian copula. The split is what makes the
method affordable. The indices need nested conditional sampling, which is
unaffordable against an expensive model but cheap against a kernel expansion.

References:
    Hilhorst, Quicken, van de Vosse & Huberts (2024). Int. J. Numer. Meth.
        Biomed. Engng. 40(2):e3797.
    Li, Rabitz, Yelvington et al. (2010). J. Phys. Chem. A 114:6022-6032.
    Wirtz & Haasdonk (2013). Dolomites Res. Notes Approx. 6:83-100.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.batching import apply_batched, resolve_batch_size
from jaxgsa._core.copula import (
    build_conditional_plan,
    canonicalize_correlation,
    independent_correlation,
)
from jaxgsa._core.invalid import InvalidUnit, OnInvalid, check_invalid, resolve_policy
from jaxgsa._core.sampling import _next_power_of_2
from jaxgsa._core.surrogate import _PredictPlan
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa._core.validation import (
    _prepare_Y,
    _squeeze_output_axes,
    _validate_xy_inputs,
    _warn_zero_variance_slices,
)
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


def analyze_vkoga(
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
    seed: int = 0,
    batch_size: int | None = None,
    on_invalid: OnInvalid = "raise",
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
        seed: Base seed for the quasi-random draws.
        batch_size: Rows per batch when evaluating the surrogate, at least 1.
            ``None`` derives one from the memory budget.
        on_invalid: What to do about non-finite values in ``X`` or ``Y``. One
            training row is one unit here, so ``"drop"`` removes the affected
            ``(X, Y)`` pairs and fits the surrogate on the rest. See
            :mod:`jaxgsa._core.invalid`. The check runs before the
            hyperparameter search on purpose: a single non-finite ``Y`` makes
            every cross-validation score non-finite, and the caller would
            otherwise meet a ``RuntimeError`` about failed kernel solves that
            names the wrong cause. ``"drop"`` needs at least ``n_folds`` rows
            to survive, because a fold with no rows in it scores nothing.

    Returns:
        A :class:`VKOGAResult` with ``S_TC``, ``S_TU``, ``S_U``, ``S_C``, and
        ``S_IU`` shaped ``(D,)``, ``(K, D)``, or ``(T, K, D)`` to mirror ``Y``.

    Raises:
        ValueError: If ``X``/``Y`` violate the output contract, if the problem
            has fewer than two parameters or any categorical parameter, if
            ``correlation`` is not ``None`` or a valid matrix, if ``gamma`` or
            ``ridge`` is not finite and positive, or if a size argument is out
            of range, if ``on_invalid`` is not one of the three policies, or
            if ``on_invalid="raise"`` (the default) and ``X`` or ``Y`` holds a
            non-finite value.
        RuntimeError: If every cross-validation score is non-finite. Non-finite
            training data no longer reaches this check; it is now a genuine
            solver failure.

    Warns:
        JaxgsaWarning: In any of four cases. An output slice has zero variance.
            JAX is in single precision, where the kernel solve loses accuracy
            for small ``gamma`` (see :mod:`jaxgsa.vkoga`). The cross-validated
            surrogate error is a large fraction of the output standard
            deviation, which makes every index untrustworthy. ``S_U`` had to be
            clipped to ``S_TU`` by a wide margin, which says the additive
            component functions cannot represent the model's interactions.
    """
    # Raise-early validation: every scalar argument is checked before any
    # expensive work (cross-validation, fitting, index estimation).
    method = "jaxgsa.vkoga.analyze"
    policy = resolve_policy(on_invalid, method=method, unit=InvalidUnit.ROW)
    if max_centers is None:
        max_centers = _DEFAULT_MAX_CENTERS
    elif max_centers < 1:
        raise ValueError(f"max_centers must be >= 1, got {max_centers}")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 for cross-validation, got {n_folds}")
    for name, value in (("gamma", gamma), ("ridge", ridge)):
        if value is not None and not (np.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be a finite positive number, got {value}")
    for name, value in (("n_outer", n_outer), ("n_inner", n_inner), ("n_variance", n_variance)):
        if value < 2:
            raise ValueError(f"{name} must be >= 2, got {value}")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    # The latent draws come from Sobol' sequences, which need power-of-two
    # sizes to keep their balance guarantees (scipy warns otherwise). The
    # defaults are already powers of two, so they pass through unchanged.
    n_outer = _next_power_of_2(n_outer)
    n_inner = _next_power_of_2(n_inner)
    n_variance = _next_power_of_2(n_variance)

    X = jnp.asarray(X)
    # The Gaussian copula is the method's dependence model, so a declared
    # problem.correlation is welcome. Categorical parameters are not: the
    # isotropic RBF needs a continuous CDF map per coordinate, and a step-CDF
    # coordinate breaks both the kernel metric and the copula conditionals.
    Y = _validate_xy_inputs(
        problem,
        X,
        jnp.asarray(Y),
        correlation_ok=True,
        categorical_ok=False,
        method=method,
    )
    # Applied here, in the only public entry point, before any fitting. The
    # joint X/Y check keeps the pairs aligned when rows are dropped.
    keep, invalid = check_invalid(
        policy=policy,
        method=method,
        unit=InvalidUnit.ROW,
        n_units=int(X.shape[0]),
        X=X,
        Y=Y,
        min_kept=n_folds,
    )
    if not keep.all():
        mask = jnp.asarray(keep)
        X = X[mask]
        Y = Y[mask]
    D = problem.num_vars
    if D < 2:
        raise ValueError(f"Correlated sensitivity indices need at least 2 parameters, got {D}")

    Y_canonical, squeeze_time, squeeze_output = _prepare_Y(Y)
    _warn_zero_variance_slices(Y_canonical, output_names=problem.output_names)
    n_time, n_out = Y_canonical.shape[1], Y_canonical.shape[2]
    Y_flat = Y_canonical.reshape(Y_canonical.shape[0], n_time * n_out)

    _warn_single_precision()
    R = _resolve_correlation(problem, correlation)

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
    state = _fit_vkoga(
        U,
        Y_centered,
        gamma=gamma_value,
        max_centers=resolved_centers,
        ridge=ridge_value,
    )
    # The fitted state is padded to the static max_centers size (rows past
    # n_centers hold zero coefficients). Slice it down host-side so predict
    # and the index estimators pay only for the centres the greedy selected.
    n_selected = int(state.n_centers)
    state = state._replace(
        centers=state.centers[:n_selected],
        coefficients=state.coefficients[:n_selected],
    )

    predict = _make_unit_predictor(state, y_mean, batch_size)
    indices = estimate_correlated_indices(
        plan=build_conditional_plan(R),
        predict=predict,
        n_outer=n_outer,
        n_inner=n_inner,
        n_variance=n_variance,
        seed=seed,
    )

    def _shape_index(flat: np.ndarray) -> Array:
        """Reshape an ``(S, D)`` index block to the output contract."""
        arr = jnp.asarray(flat.reshape(n_time, n_out, D))
        return _squeeze_output_axes(arr, squeeze_time, squeeze_output, n_trailing=1)

    def _shape_slice(flat: np.ndarray) -> Array:
        """Reshape an ``(S,)`` per-slice diagnostic to the output contract."""
        arr = jnp.asarray(flat.reshape(n_time, n_out))
        return _squeeze_output_axes(arr, squeeze_time, squeeze_output, n_trailing=0)

    output_shape = _squeeze_output_axes(
        jnp.zeros((n_time, n_out)), squeeze_time, squeeze_output, n_trailing=0
    ).shape
    return VKOGAResult(
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
    )


def _warn_single_precision() -> None:
    """Warn that float32 limits the kernel solve's accuracy.

    The regularised normal equations square the condition number of the cross
    kernel, which for small ``gamma`` exceeds what float32 can carry. Cross
    validation partly self-corrects by scoring in the same arithmetic and so
    avoiding the blown-up corner of the grid, but the ceiling is real.
    """
    # Read the flag off the config object directly; config.read() raises for
    # flags that were never explicitly set.
    if not getattr(jax.config, "jax_enable_x64", False):
        warnings.warn(
            "jaxgsa.vkoga: JAX is in single precision; the kernel solve is ill-conditioned for "
            "small gamma and the surrogate may be inaccurate. Enable float64 with "
            'jax.config.update("jax_enable_x64", True) before fitting.',
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
    return canonicalize_correlation(correlation, problem.num_vars, policy="declared")


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
