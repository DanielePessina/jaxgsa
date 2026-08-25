"""DGSM analysis: compute derivative-based sensitivity measures and bounds.

Computes the DGSM moments (nu, sigma) from a JAX-differentiable function
via autodiff — forward mode when the output slices outnumber the inputs,
reverse mode otherwise — then derives Poincare upper bounds and
Kucherenko-Song lower bounds on the total Sobol index ST.

References:
    Sobol' & Kucherenko (2009). Math. Comp. Sim. 79:3009-3017.
    Kucherenko & Song (2016). In: Monte Carlo and Quasi-Monte Carlo Methods
        2014 (MCQMC 2014), Springer Proceedings in Mathematics & Statistics
        163, pp. 455-469. doi: 10.1007/978-3-319-33507-0_23.
    Lamboni et al. (2013). Math. Comp. Sim. 87:45-54.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.batching import resolve_batch_size
from jaxgsa._core.bootstrap import _bootstrap_ci_endpoints
from jaxgsa._core.entry import (
    at_least,
    check_scalars,
    gates,
    in_open_interval,
    one_of,
    prepare,
    validate_inputs,
)
from jaxgsa._core.invalid import InvalidUnit, OnInvalid, resolve_policy
from jaxgsa._core.result import CIInfo
from jaxgsa._core.validation import _prepare_Y, _warn_zero_variance_slices
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.dgsm._core import (
    _jac_bytes_per_row,
    bounds_from_moments,
    jac_batches,
    jacobian_mode,
    jacobian_sums,
    moment_sums,
    promote_jac,
    promote_moments,
    resample_moment_sums,
    slice_count,
)
from jaxgsa.dgsm._result import DGSMResult
from jaxgsa.problem import GaussianSpec, InputSpec, Problem

# Both bounds divide by Var(Y), so a single surviving row leaves nothing to
# divide by. Two is the fewest rows that still define a variance.
_MIN_KEPT = 2

# What the report calls the two arrays it checked. DGSM puts the model
# output and its derivative into one slot, because on the autodiff path the
# derivative is model output too. Saying "Y" alone would send a reader to an
# output array that is finite everywhere.
_SOURCE_NAMES = ("X", "Y or its derivative")

_METHOD = "jaxgsa.dgsm.analyze"

# The fields a bootstrap reports an interval for. ``var_y`` is left out on
# purpose: it is the denominator of both bounds, a property of the output
# sample rather than a sensitivity measure, and its uncertainty is already
# carried inside the two bound intervals.
_INTERVAL_FIELDS = ("nu", "sigma", "upper_bound", "lower_bound")

# How many offending parameter names the lower-bound warning prints before it
# stops and counts the rest. A 50-parameter uniform problem must not paste 50
# names into one warning line.
_MAX_NAMED = 8


def _meets_lower_bound_condition(spec: InputSpec) -> bool:
    """Report whether a marginal satisfies the Kucherenko-Song condition.

    ``lower_bound`` is ``Var(x_i) * sigma_i^2 / Var(Y)``. Kucherenko & Song
    (2016), Theorem 6 (Section 4.1, eq. 31), prove it is a lower bound on
    ``ST_i`` through Stein's identity ``Cov(f, x_i) = E[tau(x_i) * df/dx_i]``.
    The kernel ``tau`` is the constant ``Var(x_i)`` only for an untruncated
    Gaussian. Truncating the
    Gaussian bends ``tau`` back to zero at each finite edge, so a truncated
    marginal fails the condition just as a uniform one does.

    Args:
        spec: Marginal spec of one parameter.

    Returns:
        True only for an untruncated Gaussian marginal.
    """
    return isinstance(spec, GaussianSpec) and spec.low is None and spec.high is None


def _warn_lower_bound_condition(problem: Problem) -> None:
    """Warn once when ``lower_bound`` is reported outside its proven case.

    The docstrings state the condition, but a result object read on its own
    carries no docstring, and the uniform marginal — the common case — is
    exactly where the number can mislead. So ``analyze`` says it at runtime
    too, once per call rather than once per parameter.

    ``upper_bound`` is deliberately left out. The Poincare inequality holds for
    every marginal this package supports, so that bound is a certificate
    whatever the distribution is.

    Args:
        problem: Problem definition whose marginals are checked.
    """
    offenders = [
        name
        for name, spec in zip(problem.names, problem.input_specs, strict=True)
        if not _meets_lower_bound_condition(spec)
    ]
    if not offenders:
        return
    shown = ", ".join(offenders[:_MAX_NAMED])
    if len(offenders) > _MAX_NAMED:
        shown += f", and {len(offenders) - _MAX_NAMED} more"
    warnings.warn(
        "jaxgsa.dgsm: lower_bound is a valid lower bound on the total Sobol index "
        "only for untruncated Gaussian marginals (Kucherenko & Song 2016, "
        "Theorem 6, Section 4.1, eq. 31). These marginals do not meet that "
        f"condition: {shown}. For them lower_bound is an estimate, not a bound: "
        "it is exact when the response is linear in that input, and it can "
        "exceed the true total index when the response is curved. Confirm "
        "anything that rests on it "
        "with jaxgsa.sobol. upper_bound is unaffected: the Poincare bound holds "
        "for every supported marginal.",
        stacklevel=3,
        category=JaxgsaWarning,
    )


def _warn_vacuous_upper_bounds(upper: Array) -> None:
    """Warn when the Poincare bound exceeds 1 for every parameter in a slice.

    A total Sobol index is a variance fraction, so it is at most 1 by
    definition. An upper bound above 1 therefore rules nothing out, and a
    slice where every parameter is above 1 rules nothing out at all: the
    bounds cannot rank the parameters and cannot say any of them is
    negligible. That state is common rather than exotic. On Ishigami at
    ``N=1024`` the bounds come out ``[2.35, 7.38, 3.11]``, and raising ``N``
    does not bring them down, because the Poincare constant, not the sample
    size, is what sets the slack.

    The bounds are still correct. They are simply not usable, and the numbers
    alone do not say so.

    Args:
        upper: Poincare upper bounds, shape ``(T, K, D)``.

    Warns:
        JaxgsaWarning: If some output slice has every bound above 1.
    """
    finite = jnp.isfinite(upper)
    # Per slice: every parameter finite and above 1. A slice holding a NaN is
    # not evidence either way, so it is not counted.
    vacuous = jnp.all(finite & (upper > 1.0), axis=-1)
    if not bool(jnp.any(vacuous)):
        return
    worst = float(jnp.min(jnp.where(vacuous[..., None], upper, jnp.inf)))
    warnings.warn(
        "jaxgsa.dgsm: every upper_bound exceeds 1 on at least one output slice "
        f"(the smallest is {worst:.2f}), and a total Sobol index is at most 1 by "
        "definition, so the bound excludes nothing there. The Poincare constant "
        "sets the slack, not the sample size, so more samples will not tighten "
        "it. Screen on nu for parameters that do nothing, and rank the rest with "
        "jaxgsa.sobol.",
        stacklevel=3,
        category=JaxgsaWarning,
    )


def _resolve_call_style(
    fn: Callable | None,
    X: Array | None,
    Y: Array | None,
    dfdx: Array | None,
) -> bool:
    """Pick one calling convention, and reject an ambiguous or incomplete call.

    ``analyze`` accepts two argument groups: ``(fn, X)`` for the autodiff path
    and ``(Y, dfdx)`` for the pre-computed path. Exactly one group must be
    complete. Resolving this once, before any computation, stops an
    over-specified call such as ``analyze(problem, X=X, Y=Y, dfdx=J)`` from
    silently dropping ``X`` and the bounds check that goes with it.

    Args:
        fn: The model function, or None.
        X: The sample matrix, or None.
        Y: Pre-computed outputs, or None.
        dfdx: Pre-computed Jacobian, or None.

    Returns:
        True for the autodiff path ``(fn, X)``, False for the pre-computed
        path ``(Y, dfdx)``.

    Raises:
        ValueError: If arguments from both groups are given, if neither group
            is given, or if one group is only partly filled.
    """
    autodiff_given = [name for name, v in (("fn", fn), ("X", X)) if v is not None]
    precomputed_given = [name for name, v in (("Y", Y), ("dfdx", dfdx)) if v is not None]

    if autodiff_given and precomputed_given:
        raise ValueError(
            f"Provide either (fn, X) or (Y, dfdx), not both: got "
            f"{', '.join(autodiff_given)} from the autodiff path and "
            f"{', '.join(precomputed_given)} from the pre-computed path. "
            f"Drop {', '.join(precomputed_given)} to differentiate the model, or drop "
            f"{', '.join(autodiff_given)} to use the values you already have."
        )

    if autodiff_given:
        if len(autodiff_given) == 2:
            return True
        missing = "X" if fn is not None else "fn"
        raise ValueError(
            f"The autodiff path needs both fn and X: {autodiff_given[0]} was given, "
            f"{missing} was not."
        )

    if precomputed_given:
        if len(precomputed_given) == 2:
            return False
        missing = "dfdx" if Y is not None else "Y"
        raise ValueError(
            f"The pre-computed path needs both Y and dfdx: {precomputed_given[0]} was "
            f"given, {missing} was not."
        )

    raise ValueError("Provide either (fn, X) or (Y, dfdx)")


def _looks_like_missing_axis(exc: BaseException) -> bool:
    """Report whether a trace failure says ``fn`` wanted an axis the row lacks.

    A batch callable fails on a one-row trace in one of two recognisable
    ways: it indexes a sample axis (``x[:, 0]``), which raises an
    ``IndexError`` about too many indices, or it reduces over one
    (``sum(x, axis=1)``), which raises an out-of-bounds-axis error. Both say
    the function wanted a second axis that a ``(D,)`` row does not have, so
    the "wrap the batch model" advice is right for them.

    Everything else stays out. A broadcasting mismatch, for one, is far more
    often an ordinary bug inside the model than a batch-convention error, and
    pointing that caller at a wrapper sends them the wrong way.

    Args:
        exc: The exception raised while tracing ``fn`` on one row.

    Returns:
        True if the failure matches a missing-axis signature.
    """
    message = str(exc).lower()
    if isinstance(exc, IndexError) and "too many indices" in message:
        return True
    return "out of bounds for array of dimension 1" in message


def _check_point_callable(fn: Callable, X: Array) -> int:
    """Reject a batch callable before it reaches the autodiff machinery.

    ``analyze`` differentiates a **one-sample** function: it maps one row of
    shape ``(D,)`` to ``()``, ``(K,)``, or ``(T, K)``. Every other module in
    this package takes the whole ``(N, D)`` matrix, so a batch callable is the
    natural thing to try, and it used to fail with an ``IndexError`` raised
    deep inside ``jax.jacrev``.

    The check uses :func:`jax.eval_shape`, which traces ``fn`` on an abstract
    row and never runs it. An expensive model therefore pays nothing: no
    forward evaluation, and no compilation. ``analyze`` already requires a
    traceable function, because the autodiff has to differentiate it, so the
    check demands nothing the method did not already demand.

    That same trace also answers ``T*K``, the number of output slices the
    autodiff mode is picked from, so the caller reads it off here instead of
    tracing the row a second time through :func:`jaxgsa.dgsm._core.n_output_slices`.

    A trace failure is reported with its original error and the expected
    signature. The "wrap the batch model" advice is added only where it is
    indicated: an output with more than two axes, or a trace failure that
    :func:`_looks_like_missing_axis` recognises. A generic failure gets no
    guess at its cause.

    Args:
        fn: The candidate one-sample function.
        X: The validated sample matrix, shape ``(N, D)``. Only its column
            count and dtype are used.

    Returns:
        ``T*K``, the number of scalar output slices ``fn`` produces.

    Raises:
        ValueError: If ``fn`` cannot be traced on a single row, or if it
            returns an array with more than two axes.
    """
    D = int(X.shape[1])
    row = jax.ShapeDtypeStruct((D,), X.dtype)
    expected = (
        "jaxgsa.dgsm.analyze differentiates a one-sample function: it maps one "
        f"row of shape ({D},) to a scalar (), a (K,) vector, or a (T, K) array."
    )
    wrap_hint = "Wrap a batch model that takes (N, D) as `lambda x: model(x[None, :])[0]`."
    try:
        out = jax.eval_shape(fn, row)
    except Exception as exc:  # noqa: BLE001 - re-raised with the real cause named
        # A trace failure has many causes, and a batch callable is only one of
        # them. Report the real error and state the expected signature, but
        # add the wrapper advice only when the failure actually looks like a
        # missing sample axis. Telling a user whose model has an unrelated
        # internal shape bug to wrap an already-correct function sends them
        # the wrong way.
        advice = f" {wrap_hint}" if _looks_like_missing_axis(exc) else ""
        raise ValueError(
            f"fn could not be evaluated on a single sample row: "
            f"{type(exc).__name__}: {exc}\n{expected}{advice}"
        ) from exc

    for leaf in jax.tree.leaves(out):
        if getattr(leaf, "ndim", 0) > 2:
            # An extra leading axis is what a batch callable returns, so here
            # the wrap hint is indicated.
            raise ValueError(
                f"fn returned an output of shape {tuple(leaf.shape)} for one sample "
                f"row, which has more than two axes. {expected} {wrap_hint}"
            )
    return slice_count(out)


def _finite_flag(row_finite: Array | npt.NDArray[np.bool_]) -> npt.NDArray[np.floating]:
    """Turn a per-row verdict back into a one-column array the check reads.

    This runs only from ``analyze``, which is eager, so converting a JAX
    array to NumPy here reads a value that already exists on the host.

    Args:
        row_finite: True where the row is clean, shape ``(N,)``.

    Returns:
        A ``(N, 1)`` array holding ``0.0`` on the clean rows and ``NaN`` on
        the rest.
    """
    return np.where(np.asarray(row_finite), 0.0, np.nan)[:, None]


def _check_dfdx(dfdx: Array, Y: Array, D: int) -> Array:
    """Check a pre-computed Jacobian against ``Y``, and promote it to 4-D.

    The rules read against the validated ``Y``, so they run after any row
    removal. Every one of them compares shapes, which is why the pure core can
    run them as well.

    Args:
        dfdx: The caller's Jacobian.
        Y: The validated model output, at the caller's own rank.
        D: The problem's parameter count.

    Returns:
        The Jacobian as ``(N, T, K, D)``.

    Raises:
        ValueError: If ``dfdx`` has the wrong rank, the wrong trailing width,
            or a leading shape that does not mirror ``Y``.
    """
    if dfdx.ndim != Y.ndim + 1:
        raise ValueError("dfdx ndim must equal Y.ndim + 1 and end with the derivative axis")
    if dfdx.shape[-1] != D:
        raise ValueError(
            f"dfdx last dimension ({dfdx.shape[-1]}) must match problem.num_vars ({D})"
        )
    if dfdx.shape[:-1] != Y.shape:
        raise ValueError(
            f"dfdx shape {dfdx.shape} does not match Y shape {Y.shape} with trailing D={D}"
        )
    return promote_jac(dfdx)


def _check_moment_layout(sigma: Array, Y_3d: Array) -> None:
    """Check the moments' slice axes against the canonicalized ``Y``.

    The moments were realigned in lockstep with ``Y``'s canonicalization, so
    their slice axes must already match. A mismatch means ``dfdx`` did not
    mirror ``Y``'s layout: wrong ndim, or a transposed Jacobian that inference
    could not recover.

    Args:
        sigma: The mean derivative, shape ``(T, K, D)``.
        Y_3d: The canonicalized output, shape ``(N, T, K)``.

    Raises:
        ValueError: If the two slice shapes disagree.
    """
    if sigma.shape[:2] != Y_3d.shape[1:3]:
        raise ValueError(
            f"dfdx ndim/shape is incompatible with Y: derivative slice dims "
            f"{sigma.shape[:2]} do not match Y's output dims {Y_3d.shape[1:3]}; "
            "dfdx must mirror Y's layout with one extra trailing (D,) axis"
        )


def indices(
    problem: Problem,
    fn: Callable | None = None,
    X: Array | None = None,
    *,
    Y: Array | None = None,
    dfdx: Array | None = None,
    standardize_outputs: bool = False,
    batch_size: int | None = None,
) -> tuple[Array, ...]:
    """Compute the DGSM moments and bounds as plain arrays, with no diagnostics.

    This is the transformable core of :func:`analyze`. It differentiates the
    same model, runs the same estimator on the same data, and returns the same
    numbers, but it does nothing else: no non-finite check, no zero-variance
    warning, no bound-ordering warning, no warning that ``lower_bound`` is
    outside its proven Gaussian case, no :class:`jaxgsa.dgsm.DGSMResult`,
    and no read of any array value on the host. So it composes with
    ``jax.jit``, ``jax.vmap``, ``jax.grad`` and ``jax.jacrev``, which
    :func:`analyze` cannot, because a policy decision needs a concrete value
    and a tracer has none.

    Differentiating a bound with respect to the sample is one line::

        def upper(X):
            return jaxgsa.dgsm.indices(problem, fn, X)[2]

        d_upper = jax.jacrev(upper)(X)

    That nests one differentiation inside another: the Jacobian of ``fn`` is
    itself the thing being differentiated, so the outer pass needs ``fn`` to
    be twice differentiable.

    Use :func:`analyze` for ordinary analysis. Nothing here checks the model
    output or its derivative, so a single NaN silently turns every returned
    number into NaN.

    Tier T4 (behavioural contract): the returned arrays must equal the
    corresponding fields of ``analyze``'s result on clean outputs, and the
    function must survive ``jit``, ``vmap`` and ``jit(jacrev(...))``. Checked
    in ``tests/test_dgsm.py``.

    Args:
        problem: Problem definition with D parameters. Read for the Poincare
            constants and the marginal variances only, and both are static, so
            passing one never turns a marginal parameter into a tracer.
        fn: JAX-differentiable **one-sample** function, as in :func:`analyze`.
        X: Sample matrix in the problem's physical units, shape ``(N, D)``.
        Y: Forward model outputs for the pre-computed path, shape ``(N,)`` /
            ``(N, K)`` / ``(N, T, K)``.
        dfdx: Pre-computed Jacobian mirroring ``Y``'s layout with one extra
            trailing ``(D,)`` axis.
        standardize_outputs: As in :func:`analyze`.
        batch_size: Sample rows per batch on the autodiff path, clamped to
            ``N``, or ``None`` to derive one from the active memory budget,
            as in :func:`analyze`.

    Returns:
        ``(nu, sigma, upper_bound, lower_bound, var_y)``. The first four have
        the shapes ``analyze`` reports for them, and ``var_y`` is the
        per-slice output variance.

    Raises:
        ValueError: If the calling convention is ambiguous or incomplete, if
            ``fn`` cannot be traced on one sample row, if ``dfdx`` does not
            mirror ``Y``, if ``problem.correlation`` declares a dependence
            structure (both Sobol-index bounds assume independent inputs, so
            they would be silently wrong, exactly as in :func:`analyze`), or
            if ``problem`` has a categorical parameter.
    """
    from jaxgsa.dgsm import SPEC

    # The same capability gates analyze applies through prepare().
    # problem.has_correlated_inputs is static host-side metadata, so the check
    # runs at trace time and the core still composes with jit/vmap/grad.
    gates(SPEC, problem, method="jaxgsa.dgsm.indices")
    use_autodiff = _resolve_call_style(fn, X, Y, dfdx)
    D = problem.num_vars

    if use_autodiff:
        assert fn is not None and X is not None  # guaranteed by _resolve_call_style
        n_outputs = _check_point_callable(fn, X)
        sum_jac, sum_jac2, Y_raw = moment_sums(fn, X, batch_size, n_outputs=n_outputs)
        n_rows = X.shape[0]
        sigma = promote_moments(sum_jac / n_rows)
        nu = promote_moments(sum_jac2 / n_rows)
        Y_3d, layout = _prepare_Y(jnp.asarray(Y_raw))
    else:
        assert Y is not None and dfdx is not None  # guaranteed by _resolve_call_style
        Y_arr = jnp.asarray(Y)
        dfdx_arr = _check_dfdx(jnp.asarray(dfdx), Y_arr, D)
        sigma = jnp.mean(dfdx_arr, axis=0)
        nu = jnp.mean(dfdx_arr**2, axis=0)
        Y_3d, layout = _prepare_Y(Y_arr)

    _check_moment_layout(sigma, Y_3d)
    nu, sigma, upper, lower, var_y = bounds_from_moments(
        problem,
        nu=nu,
        sigma=sigma,
        var_y=jnp.var(Y_3d, axis=0),
        standardize_outputs=standardize_outputs,
    )
    return (
        layout.squeeze(nu),
        layout.squeeze(sigma),
        layout.squeeze(upper),
        layout.squeeze(lower),
        layout.squeeze(var_y, n_trailing=0),
    )


def analyze(
    problem: Problem,
    fn: Callable | None = None,
    X: Array | None = None,
    *,
    Y: Array | None = None,
    dfdx: Array | None = None,
    standardize_outputs: bool = False,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    batch_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> DGSMResult:
    """Compute DGSM sensitivity indices and Sobol index bounds.

    DGSM ranks inputs by how strongly the output reacts to them on average.
    The measure is ``nu_i = E[(df/dx_i)^2]``, the mean squared partial
    derivative over the input distribution. It is the natural pick when the
    model is JAX-differentiable: one autodiff sweep over an ordinary Monte
    Carlo sample replaces a dedicated Sobol design.

    The moments then convert into two numbers that frame the total Sobol
    index:

    - **Upper bound** (Poincare / Sobol-Kucherenko inequality):
      ``ST_i <= C_i * nu_i / Var(Y)``. ``C_i`` is the Poincare constant of
      input i's marginal distribution, the sharpest factor for which the
      inequality holds (see :mod:`jaxgsa.dgsm._poincare`). This holds for
      every marginal this package supports, so an input whose upper bound is
      near zero is provably negligible.
    - **Lower bound**: ``Var(x_i) * sigma_i^2 / Var(Y)``, with
      ``sigma_i = E[df/dx_i]``, the mean (signed) derivative. Kucherenko &
      Song (2016), Theorem 6 (Section 4.1, eq. 31), prove ``ST_i >=`` this
      expression when input i's marginal is an **untruncated Gaussian**. That
      condition is not decoration: the proof needs Stein's identity, which
      holds with a
      constant kernel only for the Gaussian. On a uniform or truncated
      marginal the expression is exact for a response linear in that input
      and near-exact for a nearly linear one, but a strongly curved response
      can push it above the true ``ST_i``. ``f(p) = 1/p`` on
      ``p ~ U(0.1, 0.4)`` returns 1.29 for the only input of a one-input
      model, whose ``ST`` is 1 by definition.

    So ``upper_bound`` is a proof and, off the Gaussian case,
    ``lower_bound`` is a guide. Confirm anything that rests on the latter
    with :mod:`jaxgsa.sobol`. :class:`jaxgsa.dgsm.DGSMResult` carries the
    full statement.

    There are two calling conventions, and you must use exactly one of them:

    - **Autodiff path** (primary): pass ``fn`` and ``X``. JAX differentiates
      the function, and one pass returns both the Jacobian and the forward
      outputs. The autodiff mode is selected from the shapes: ``jax.jacfwd``
      when the output slices outnumber the inputs (``T*K > D``), ``jax.jacrev``
      otherwise. The two modes compute the same Jacobian; only float
      arithmetic order differs.
    - **Pre-computed path**: pass ``Y`` and ``dfdx``. Use it when the model is
      not JAX-differentiable, or when the Jacobian comes from elsewhere.

    Arguments from both groups, or one group only partly filled, raise a
    ``ValueError``. Nothing is dropped silently.

    ``fn`` takes **one sample**, not a batch. Unlike the other methods in this
    package, which call the model on the whole ``(N, D)`` matrix, ``fn`` maps a
    single row of shape ``(D,)`` to ``()``, ``(K,)``, or ``(T, K)``. Wrap a
    batch model as ``lambda x: model(x[None, :])[0]``.

    Args:
        problem: Problem definition with D parameters.
        fn: JAX-differentiable **one-sample** function: ``(D,) -> ()``,
            ``(D,) -> (K,)``, or ``(D,) -> (T, K)`` for time-series outputs.
            It is called on one row at a time; ``analyze`` does the batching.
        X: Sample matrix in the problem's physical units, shape ``(N, D)``.
        Y: Forward model outputs, shape ``(N,)`` / ``(N, K)`` / ``(N, T, K)``.
        dfdx: Pre-computed Jacobian, mirroring ``Y``'s layout with one extra
            trailing ``(D,)`` axis: ``(N, D)`` for ``(N,)`` Y, ``(N, K, D)``
            for ``(N, K)``, and ``(N, T, K, D)`` for ``(N, T, K)``.
        standardize_outputs: When ``True``, report ``nu``, ``sigma`` and
            ``var_y`` for the standardized output ``(Y - mean) / std``, one
            mean and one standard deviation per output slice. DGSM returns
            dimensional quantities: under ``Y -> a*Y + b``, ``sigma`` scales
            by ``a`` and ``nu`` by ``a^2``. Dividing them out puts every
            output slice in units of its own standard deviation, so slices of
            different magnitude compare directly. ``upper_bound`` and
            ``lower_bound`` are ratios and do not move. Defaults to
            ``False``, which reports the moments in the output's own units.
        n_bootstrap: Number of bootstrap resamples for confidence intervals.
            ``0`` (default) computes no intervals. The resampling unit is one
            row of the sample. ``nu`` and ``sigma`` are plain i.i.d. means
            over rows, so a row bootstrap is exactly the right resample for
            them, and both bounds are the plug-in ratios of those means to
            ``Var(Y)``, recomputed on the same resampled rows. Intervals are
            reported for all four: ``nu``, ``sigma``, ``upper_bound`` and
            ``lower_bound``. ``var_y`` gets none. It is the denominator, a
            property of the output sample rather than a sensitivity measure,
            and its uncertainty is already inside the two bound intervals.
            The cost is one extra sweep of the Jacobian over the sample, not
            one sweep per replicate: a replicate is a weighted row sum, so
            the resample is a matrix product against the same batches.
        conf_level: Confidence level for the bootstrap intervals.
        ci_method: How the interval endpoints are formed. ``"quantile"``
            (default) reads them off the empirical bootstrap distribution.
            ``"gaussian"`` centres them on the point estimate and takes
            ``+/- z * sd`` of the bootstrap draws, which is smoother for a
            small ``n_bootstrap`` but assumes the draws are normal.
        key: A ``jax.random`` key for the bootstrap resampling. Required when
            ``n_bootstrap > 0``. Pass ``jax.random.key(0)`` if you have an
            integer seed.
        batch_size: Number of N sample rows per batch on the autodiff path,
            clamped to ``N``. The Jacobian accumulates in batches of this
            many samples, which bounds peak memory. None (default) derives a
            width from the active memory budget (see
            ``jaxgsa.config.set_memory_budget``), pricing each row at a few
            Jacobian-sized transients (``T*K*D`` floats per row times a
            small live factor).
        on_invalid: What to do about a sample row that holds a non-finite
            value. ``"raise"`` (default) refuses the sample, ``"drop"``
            removes those rows and analyzes the rest, and ``"propagate"``
            warns and computes anyway. The check covers the **derivative** as
            well as the output, on both calling conventions: a derivative that
            blows up poisons ``nu`` even where the output itself is finite.
            The derivative is checked in the output slot, so the report names
            ``"Y or its derivative"`` for a bad derivative. On the autodiff path ``X`` is
            checked too, and the rows are masked before the batch reduction,
            so ``"drop"`` gives the same moments as re-running on the smaller
            sample. See :mod:`jaxgsa._core.invalid`.
        verbose: If ``True`` (default), print a short summary to stdout: the
            problem and the data, the wall-clock timing, and the top
            parameters by ``nu``. Pass ``False`` for a silent run.
        keep_replicates: Keep the per-resample moments and bounds on
            ``DGSMResult.ci.replicates``. Off by default because they are
            large: ``n_bootstrap`` copies of four index arrays.

    Returns:
        A ``DGSMResult`` with ``nu``, ``sigma``, ``upper_bound``, and
        ``lower_bound``, each of shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``
        mirroring the output layout, plus ``var_y``, the optional ``*_conf``
        intervals with the ``ci`` record that describes them, and the
        non-finite report in ``invalid``.

    Raises:
        ValueError: In any of these cases. ``on_invalid`` is not one of the
            three policies, or the non-finite policy refuses the sample.
            Arguments from both ``(fn, X)`` and
            ``(Y, dfdx)`` were given. Neither pair was given. One pair was only
            partly filled, such as ``fn`` without ``X``. ``fn`` takes a batch
            ``(N, D)`` instead of one ``(D,)`` row, or cannot be traced on one
            row. ``dfdx`` has an unexpected shape, or does
            not match ``Y`` or the problem dimension. ``problem.correlation``
            declares a dependence structure, and the Poincare-inequality bounds
            assume independent inputs. ``problem`` has categorical parameters,
            and a derivative along an unordered level code has no meaning.
            ``n_bootstrap`` is negative, ``conf_level`` is not in ``(0, 1)``,
            ``ci_method`` is not ``"quantile"`` or ``"gaussian"``, or
            ``n_bootstrap > 0`` and no ``key`` was given.
    Warns:
        JaxgsaWarning: If an output slice has zero variance, which makes both
            bounds for that slice NaN. Also, once per call, if any marginal is
            not an untruncated Gaussian, because ``lower_bound`` is a proven
            bound only there; the warning names the marginals that fail the
            condition. ``upper_bound`` is never in question.
    """
    from jaxgsa.dgsm import SPEC

    # DGSM cannot call the one-shot prepare(), because on the autodiff path
    # the model output is what the preamble would validate and it does not
    # exist yet. It runs the same preamble in its two halves instead. The
    # scalar half settles the policy and batch_size before anything expensive
    # runs — a misspelled on_invalid must not cost a differentiation sweep —
    # and applies the capability gates from the registry record: the
    # Poincare-inequality bound on ST assumes independent inputs, so both
    # calling conventions are refused for a correlated problem.
    checks = (
        at_least("batch_size", batch_size, 1),
        at_least("n_bootstrap", n_bootstrap, 0),
        one_of("ci_method", ci_method, ("quantile", "gaussian")),
        in_open_interval("conf_level", conf_level, 0.0, 1.0),
    )
    unit = SPEC.invalid_unit
    assert unit is not None  # dgsm declares InvalidUnit.ROW
    resolve_policy(
        on_invalid,
        method=_METHOD,
        unit=unit,
        allow_drop=unit is not InvalidUnit.CURVE,
    )
    check_scalars(checks)
    gates(SPEC, problem, method=_METHOD)
    if n_bootstrap > 0 and key is None:
        raise ValueError("key is required when n_bootstrap > 0")
    # Say once, before the expensive work, that lower_bound is only a proven
    # bound for untruncated Gaussian marginals. The gates above already
    # rejected categorical parameters, so the check only separates untruncated
    # Gaussians from uniform and truncated ones.
    _warn_lower_bound_condition(problem)
    # The clock starts before the model is differentiated or evaluated: on
    # the autodiff path the Jacobian sweep is the expensive work, so it
    # belongs inside the timed span.
    t0 = _verbose.tic()
    D = problem.num_vars

    # Resolve the calling convention once, before any computation, so that an
    # over-specified call cannot pick one branch and drop the other group's
    # arguments unchecked.
    use_autodiff = _resolve_call_style(fn, X, Y, dfdx)

    if use_autodiff:
        assert fn is not None and X is not None  # guaranteed by _resolve_call_style
        # X is checked here rather than in the data half, because the model is
        # about to be differentiated on it.
        X = validate_inputs(problem, X)
        # Free shape-only trace: catches a batch (N, D) callable here, instead
        # of as an IndexError from inside the autodiff.
        n_outputs = _check_point_callable(fn, X)
        moments = jacobian_sums(fn, X, batch_size, n_outputs=n_outputs)
        N = int(X.shape[0])
        # The Jacobian never leaves the device whole, so it reaches the check
        # as a per-row verdict rather than as its own array. The verdict is
        # the same one the batch loop masked with. It rides in as an extra
        # array, which puts it on the model side of the check with Y.
        ctx = prepare(
            SPEC,
            problem,
            moments.Y,
            X=X,
            on_invalid=on_invalid,
            checks=checks,
            method=_METHOD,
            extra={"derivative": _finite_flag(moments.jac_finite)},
            min_kept=_MIN_KEPT,
            source_names=_SOURCE_NAMES,
            # The bounds need Var(Y) anyway, so DGSM warns with the variance
            # it already has, further down.
            warn_zero_variance=False,
        )
        keep, invalid = ctx.keep, ctx.invalid
        Y_valid = ctx.Y
        if keep.all():
            # No row was removed, so the totals over every row are the ones
            # to divide. Under "propagate" this is the branch that lets the
            # non-finite value through, which is what it is for.
            sigma = moments.sum_jac / N
            nu = moments.sum_jac2 / N
        else:
            # The batch loop masked with its own copy of this verdict, built
            # from the same three arrays. If the two ever disagreed, the
            # moments below would come from a different set of rows than the
            # ones the report names. Raised, not asserted: the check is
            # load-bearing and must survive python -O.
            if not np.array_equal(keep, moments.row_ok):
                raise ValueError(
                    "jaxgsa.dgsm.analyze: internal error: the non-finite check kept "
                    f"{int(keep.sum())} rows but the batched moments kept "
                    f"{int(moments.row_ok.sum())}; the reported rows would not be the "
                    "rows the moments were computed from. Please report this bug."
                )
            n_kept = int(keep.sum())
            sigma = moments.sum_jac_kept / n_kept
            nu = moments.sum_jac2_kept / n_kept
        sigma = promote_moments(sigma)
        nu = promote_moments(nu)
    else:
        assert Y is not None and dfdx is not None  # guaranteed by _resolve_call_style
        # Pre-computed path: the caller supplies the Jacobian and the forward
        # outputs directly.
        dfdx_arr = jnp.asarray(dfdx)
        # Both arrays are in hand here, so the derivative needs no stand-in:
        # it goes into the check whole, on the model side with Y, and comes
        # back compacted by the same mask.
        ctx = prepare(
            SPEC,
            problem,
            Y,
            on_invalid=on_invalid,
            checks=checks,
            method=_METHOD,
            extra={"derivative": dfdx_arr},
            n_expected=int(dfdx_arr.shape[0]),
            min_kept=_MIN_KEPT,
            source_names=_SOURCE_NAMES,
            warn_zero_variance=False,
        )
        invalid = ctx.invalid
        Y_valid = ctx.Y
        # The mirror rules on dfdx are read against the validated Y, so they
        # come after the data half. Row counts are settled before this point:
        # prepare() was told to expect one Y row per Jacobian row.
        dfdx_arr = _check_dfdx(ctx.extra["derivative"], Y_valid, D)
        # One vectorized reduction over N covers every (t, k) slice at once.
        sigma = jnp.mean(dfdx_arr, axis=0)  # E[df/dx_i], (T, K, D)
        nu = jnp.mean(dfdx_arr**2, axis=0)  # E[(df/dx_i)^2]

    # Canonicalize Y to (N, T, K), then check the moments line up with it.
    Y_3d = ctx.Y3
    _check_moment_layout(sigma, Y_3d)

    # Var(Y) per (t, k) output slice, denominator of both bounds. A constant
    # slice makes both bounds NaN, so say so rather than returning it silently.
    var_y = jnp.var(Y_3d, axis=0)  # (T, K)
    _warn_zero_variance_slices(Y_valid, output_names=problem.output_names, method=_METHOD)

    nu, sigma, upper, lower, var_y = bounds_from_moments(
        problem,
        nu=nu,
        sigma=sigma,
        var_y=var_y,
        standardize_outputs=standardize_outputs,
    )

    # The upper bound must sit at or above the lower bound. Check it within a
    # numerical tolerance and warn if it does not hold.
    if jnp.any(jnp.isfinite(upper) & jnp.isfinite(lower) & (upper < lower * 0.9)):
        warnings.warn(
            "jaxgsa.dgsm: some upper bounds are below lower bounds, suggesting "
            "insufficient samples or numerical issues",
            stacklevel=2,
            category=JaxgsaWarning,
        )
    _warn_vacuous_upper_bounds(upper)

    confs: dict[str, Array | None] = dict.fromkeys(_INTERVAL_FIELDS)
    ci: CIInfo | None = None
    if n_bootstrap > 0:
        assert key is not None  # checked before the point estimate ran
        batches: Iterator[tuple[Array, Array]]
        if use_autodiff:
            assert fn is not None
            # ctx.X is X with the dropped rows already removed, so the
            # resample draws from the same rows the point estimate averaged.
            assert ctx.X is not None  # the autodiff path always passes X to prepare()
            batches = ((jac, y) for jac, y, _ in jac_batches(fn, ctx.X, batch_size))
        else:
            batches = _array_batches(dfdx_arr, Y_3d, batch_size)
        draws = _bootstrap_draws(
            problem,
            batches=batches,
            n_rows=int(Y_3d.shape[0]),
            slice_shape=(int(Y_3d.shape[1]), int(Y_3d.shape[2])),
            n_params=D,
            key=key,
            n_bootstrap=n_bootstrap,
            standardize_outputs=standardize_outputs,
        )
        point = {"nu": nu, "sigma": sigma, "upper_bound": upper, "lower_bound": lower}
        for name in _INTERVAL_FIELDS:
            lo, hi = _bootstrap_ci_endpoints(
                point[name], draws[name], conf_level=conf_level, ci_method=ci_method
            )
            # Stack [lower, upper] into a leading axis of size 2. The squeeze
            # addresses the (T, K) axes from the end, so that axis survives.
            confs[name] = ctx.squeeze(jnp.stack((lo, hi)))
        ci = CIInfo(
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            replicates=(
                {name: ctx.squeeze(draws[name]) for name in _INTERVAL_FIELDS}
                if keep_replicates
                else None
            ),
        )

    # Drop the singleton axes the preamble inserted; var_y has no trailing
    # param axis, so it passes n_trailing=0.
    sigma = ctx.squeeze(sigma)
    nu = ctx.squeeze(nu)
    upper = ctx.squeeze(upper)
    lower = ctx.squeeze(lower)
    var_y = ctx.squeeze(var_y, n_trailing=0)

    result = DGSMResult(
        nu=nu,
        sigma=sigma,
        upper_bound=upper,
        lower_bound=lower,
        var_y=var_y,
        problem=problem,
        invalid=invalid,
        nu_conf=confs["nu"],
        sigma_conf=confs["sigma"],
        upper_bound_conf=confs["upper_bound"],
        lower_bound_conf=confs["lower_bound"],
        ci=ci,
    )

    if verbose:
        elapsed = _verbose.stop(t0, result.nu)
        _, T, K = ctx.Y3.shape
        # On the autodiff path the timed span includes the Jacobian sweep;
        # on the pre-computed path only the estimator ran.
        label = (
            "model sweep + estimator (includes compile on the first call)"
            if use_autodiff
            else "estimator (includes compile on the first call)"
        )
        if use_autodiff:
            mode = jacobian_mode(n_inputs=problem.num_vars, n_outputs=T * K)
            gradients = f"gradients: {mode}-mode autodiff (T*K={T * K}, D={problem.num_vars})"
        else:
            gradients = "gradients: user-supplied dfdx"
        batching = (
            f"batch_size: {batch_size} (user-set)"
            if batch_size is not None
            else "batch_size: auto (resolved from the memory budget)"
        )
        _verbose.analysis_summary(
            method="jaxgsa.dgsm.analyze",
            problem=problem,
            n_runs=int(ctx.Y.shape[0]),
            T=T,
            K=K,
            invalid=ctx.invalid,
            timings=[(label, elapsed)],
            notes=[gradients, batching],
            index_name="nu",
            values=result.nu,
            conf=result.nu_conf,
        )
    return result


def _array_batches(
    dfdx: Array,
    Y_3d: Array,
    batch_size: int | None,
) -> Iterator[tuple[Array, Array]]:
    """Walk a pre-computed Jacobian and its output in batches of sample rows.

    The pre-computed path holds both arrays already, so there is nothing to
    differentiate. This only re-uses the batch protocol the autodiff path
    yields, so one resampler serves both conventions.

    Args:
        dfdx: The validated Jacobian, shape ``(N, T, K, D)``.
        Y_3d: The validated output, shape ``(N, T, K)``.
        batch_size: Rows per batch, or ``None`` to derive one from the active
            memory budget with the same Jacobian bytes model the autodiff
            path uses. The Jacobian already exists whole here, so the batch
            width only bounds the bootstrap's per-batch transients (the
            weight matrix and the squared block), but the two calling
            conventions read one keyword and must give it one meaning.

    Yields:
        ``(jac, Y)`` per batch, in row order.
    """
    N = int(Y_3d.shape[0])
    _, T, K, D = dfdx.shape
    step = resolve_batch_size(
        _jac_bytes_per_row(T * K, D, jnp.dtype(dfdx.dtype).itemsize), N, batch_size
    )
    for start in range(0, N, step):
        end = min(start + step, N)
        yield dfdx[start:end], Y_3d[start:end]


def _bootstrap_draws(
    problem: Problem,
    *,
    batches: Iterator[tuple[Array, Array]],
    n_rows: int,
    slice_shape: tuple[int, int],
    n_params: int,
    key: Array,
    n_bootstrap: int,
    standardize_outputs: bool,
) -> dict[str, Array]:
    """Recompute the four reported fields on ``n_bootstrap`` row resamples.

    Rows are drawn with replacement, and every quantity DGSM reports is a
    weighted sum over rows, so the whole bootstrap is one weighted pass over
    the same batches. ``Var(Y)`` is resampled along with the moments, which is
    what makes each bound an honest plug-in ratio rather than a numerator
    interval over a frozen denominator.

    Args:
        problem: Problem definition, for the bound constants.
        batches: ``(jac, Y)`` per batch, over the surviving rows only.
        n_rows: Number of surviving rows, which is also the resample size.
        slice_shape: ``(T, K)``, the canonical output slice axes.
        n_params: ``D``.
        key: PRNG key for the row draws.
        n_bootstrap: Number of replicates.
        standardize_outputs: As in :func:`analyze`.

    Returns:
        Mapping from field name to a ``(n_bootstrap, T, K, D)`` array of
        replicate values.
    """
    idx = jax.random.randint(key, shape=(n_bootstrap, n_rows), minval=0, maxval=n_rows)
    sum_jac, sum_jac2, sum_y, sum_y2 = resample_moment_sums(batches, idx)

    T, K = slice_shape
    sigma = sum_jac.reshape(n_bootstrap, T, K, n_params) / n_rows
    nu = sum_jac2.reshape(n_bootstrap, T, K, n_params) / n_rows
    mean_y = sum_y.reshape(n_bootstrap, T, K) / n_rows
    # Var(Y) from the two raw moments, because a replicate never materialises
    # the rows it drew. Clipped at zero: the raw-moment form can go a rounding
    # step negative on a near-constant slice, and a negative variance would
    # turn into a NaN standard deviation.
    var_y = jnp.maximum(sum_y2.reshape(n_bootstrap, T, K) / n_rows - mean_y**2, 0.0)

    # Every step below broadcasts over the leading replicate axis, so a
    # replicate runs through exactly the arithmetic the point estimate did.
    nu, sigma, upper, lower, _ = bounds_from_moments(
        problem,
        nu=nu,
        sigma=sigma,
        var_y=var_y,
        standardize_outputs=standardize_outputs,
    )
    return {"nu": nu, "sigma": sigma, "upper_bound": upper, "lower_bound": lower}
