"""DGSM analysis: compute derivative-based sensitivity measures and bounds.

Computes the DGSM moments (nu, sigma) from a JAX-differentiable function
via reverse-mode autodiff, then derives Poincare upper bounds and
Kucherenko-Song lower bounds on the total Sobol index ST.

References:
    Sobol' & Kucherenko (2009). Math. Comp. Sim. 79:3009-3017.
    Kucherenko & Song (2016). Rel. Eng. Sys. Safety 148:81-95.
    Lamboni et al. (2013). Math. Comp. Sim. 87:44-54.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from jaxgsa._core.bootstrap import _bootstrap_ci_endpoints
from jaxgsa._core.entry import (
    at_least,
    in_open_interval,
    one_of,
    prepare_scalars,
    validate_inputs,
)
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.result import CIInfo
from jaxgsa._core.validation import _prepare_Y, _warn_zero_variance_slices
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.dgsm._core import (
    bounds_from_moments,
    jac_batches,
    moment_sums,
    promote_jac,
    promote_moments,
    resample_moment_sums,
)
from jaxgsa.dgsm._result import DGSMResult
from jaxgsa.problem import Problem

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


def _check_point_callable(fn: Callable, X: Array) -> None:
    """Reject a batch callable before it reaches the autodiff machinery.

    ``analyze`` differentiates a **one-sample** function: it maps one row of
    shape ``(D,)`` to ``()``, ``(K,)``, or ``(T, K)``. Every other module in
    this package takes the whole ``(N, D)`` matrix, so a batch callable is the
    natural thing to try, and it used to fail with an ``IndexError`` raised
    deep inside ``jax.jacrev``.

    The check uses :func:`jax.eval_shape`, which traces ``fn`` on an abstract
    row and never runs it. An expensive model therefore pays nothing: no
    forward evaluation, and no compilation. ``analyze`` already requires a
    traceable function, because ``jax.jacrev`` has to differentiate it, so the
    check demands nothing the method did not already demand.

    A trace failure is reported with its original error and the expected
    signature. The "wrap the batch model" advice is added only where it is
    indicated: an output with more than two axes, or a trace failure that
    :func:`_looks_like_missing_axis` recognises. A generic failure gets no
    guess at its cause.

    Args:
        fn: The candidate one-sample function.
        X: The validated sample matrix, shape ``(N, D)``. Only its column
            count and dtype are used.

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


@dataclass(frozen=True)
class _Moments:
    """Unreduced derivative sums from the autodiff path, in two versions.

    The autodiff path makes ``Y`` itself, inside a jitted ``vmap`` of
    ``jax.jacrev``, and reduces each batch of Jacobian rows to a running total
    as soon as it has it. A row that is already inside that total cannot be
    taken out again: ``NaN`` plus anything is ``NaN``. So the batch loop keeps
    two totals side by side, one over every row and one over the clean rows
    only, and ``analyze`` picks the one its policy calls for. The masking
    happens where the row still exists, which is inside the loop.

    Keeping both costs one extra pair of sums per batch. The Jacobian itself
    is what the batch loop exists to bound, and neither total holds a sample
    axis, so peak memory is unchanged.

    Attributes:
        Y: Stacked raw ``fn`` output, shape ``(N,)`` / ``(N, K)`` /
            ``(N, T, K)``.
        jac_finite: For each row, whether every entry of its Jacobian block is
            finite, shape ``(N,)``. Reported separately from ``Y`` because a
            derivative can blow up where the output does not, and ``nu`` is
            built from the derivative.
        row_ok: For each row, whether ``X``, ``Y`` and the Jacobian are all
            finite, shape ``(N,)``. This is the mask the ``*_kept`` totals
            used.
        sum_jac: Sum of the Jacobian over every row, shape ``(D,)`` /
            ``(K, D)`` / ``(T, K, D)``.
        sum_jac2: Sum of the squared Jacobian over every row, same shape.
        sum_jac_kept: Sum of the Jacobian over the rows in ``row_ok``.
        sum_jac2_kept: Sum of the squared Jacobian over those same rows.
    """

    Y: Array
    jac_finite: npt.NDArray[np.bool_]
    row_ok: npt.NDArray[np.bool_]
    sum_jac: Array
    sum_jac2: Array
    sum_jac_kept: Array
    sum_jac2_kept: Array


def _rows_finite(array: Array) -> Array:
    """Return a ``(N,)`` mask that is True where a whole row is finite.

    Args:
        array: Any array whose leading axis is the sample axis.

    Returns:
        A boolean array of shape ``(N,)``.
    """
    return jnp.all(jnp.isfinite(array.reshape(array.shape[0], -1)), axis=1)


def _mask_rows(array: Array, row_ok: Array) -> Array:
    """Zero out the rows a mask rejects, without arithmetic on their values.

    ``jnp.where`` is required here rather than a multiply by ``0.0``: a
    multiply leaves ``NaN * 0 == NaN``, which is exactly the value the mask
    exists to remove.

    Args:
        array: Array whose leading axis is the sample axis.
        row_ok: Boolean mask of shape ``(N,)``.

    Returns:
        ``array`` with the rejected rows replaced by zeros.
    """
    return jnp.where(row_ok.reshape((-1,) + (1,) * (array.ndim - 1)), array, 0.0)


def _finite_flag(row_finite: npt.NDArray[np.bool_]) -> npt.NDArray[np.floating]:
    """Turn a per-row verdict back into a one-column array the check reads.

    Args:
        row_finite: True where the row is clean, shape ``(N,)``.

    Returns:
        A ``(N, 1)`` array holding ``0.0`` on the clean rows and ``NaN`` on
        the rest.
    """
    return np.where(row_finite, 0.0, np.nan)[:, None]


def _compute_moments(
    fn: Callable,
    X: Array,
    batch_size: int | None = None,
) -> _Moments:
    """Compute DGSM derivative sums and forward outputs by autodiff.

    The reduction over the sample axis is one vectorized operation that covers
    every (t, k) output slice at once. The per-slice kernel needs no explicit
    vmap, because the closed form already batches it. Which differentiation
    mode runs is decided in :func:`jaxgsa.dgsm._core.jacobian_of`, from the
    output and input widths.

    Every batch is reduced twice: once over all of its rows, and once over the
    rows in which ``X``, the output and the Jacobian are all finite. See
    :class:`_Moments` for why both are needed. ``analyze`` divides by the
    matching row count.

    This is the policy-aware twin of :func:`jaxgsa.dgsm._core.moment_sums`,
    which the pure core uses and which masks nothing. Both walk the sample
    axis through the same batch iterator, so there is one copy of the padding
    and of the sweep.

    Args:
        fn: JAX-differentiable function ``(D,) -> ()``, ``(D,) -> (K,)``, or
            ``(D,) -> (T, K)``.
        X: Sample matrix, shape ``(N, D)``.
        batch_size: Number of N sample rows per batch, to limit memory. None
            processes all N rows at once.

    Returns:
        A :class:`_Moments`. Its sums carry the ``fn`` output's own slice
        axes, shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``. ``analyze`` applies
        the layout canonicalization (promotion and label-driven axis moves)
        once the semantic axes are known.

    Raises:
        ValueError: If ``X`` has no rows, so there is nothing to average.
    """
    sum_jac: Array | None = None
    sum_jac2: Array | None = None
    sum_jac_kept: Array | None = None
    sum_jac2_kept: Array | None = None
    Y_parts: list[Array] = []
    jac_finite_parts: list[Array] = []
    row_ok_parts: list[Array] = []

    for jac, Y_chunk, X_chunk in jac_batches(fn, X, batch_size):
        Y_parts.append(Y_chunk)
        # The mask is built and applied here, before the sums below fold this
        # batch into the running totals. After that point the batch is gone.
        chunk_jac_finite = _rows_finite(jac)
        chunk_row_ok = chunk_jac_finite & _rows_finite(Y_chunk) & _rows_finite(X_chunk)
        jac_finite_parts.append(chunk_jac_finite)
        row_ok_parts.append(chunk_row_ok)
        jac_kept = _mask_rows(jac, chunk_row_ok)

        sj = jnp.sum(jac, axis=0)
        sj2 = jnp.sum(jac**2, axis=0)
        sjk = jnp.sum(jac_kept, axis=0)
        sjk2 = jnp.sum(jac_kept**2, axis=0)
        sum_jac = sj if sum_jac is None else sum_jac + sj
        sum_jac2 = sj2 if sum_jac2 is None else sum_jac2 + sj2
        sum_jac_kept = sjk if sum_jac_kept is None else sum_jac_kept + sjk
        sum_jac2_kept = sjk2 if sum_jac2_kept is None else sum_jac2_kept + sjk2

    if sum_jac is None or sum_jac2 is None or sum_jac_kept is None or sum_jac2_kept is None:
        raise ValueError("X has no sample rows, so there is no derivative to average")

    return _Moments(
        Y=_join(Y_parts),
        jac_finite=np.asarray(_join(jac_finite_parts)),
        row_ok=np.asarray(_join(row_ok_parts)),
        sum_jac=sum_jac,
        sum_jac2=sum_jac2,
        sum_jac_kept=sum_jac_kept,
        sum_jac2_kept=sum_jac2_kept,
    )


def _join(parts: list[Array]) -> Array:
    """Concatenate batch pieces back into one sample-axis array.

    Args:
        parts: One array per batch, all sharing their trailing shape.

    Returns:
        The pieces stacked along the sample axis. A single piece is returned
        unchanged, so the unbatched path copies nothing.
    """
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)


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
    warning, no bound-ordering warning, no :class:`jaxgsa.dgsm.DGSMResult`,
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
        batch_size: Sample rows per batch on the autodiff path, or ``None``
            for one batch.

    Returns:
        ``(nu, sigma, upper_bound, lower_bound, var_y)``. The first four have
        the shapes ``analyze`` reports for them, and ``var_y`` is the
        per-slice output variance.

    Raises:
        ValueError: If the calling convention is ambiguous or incomplete, if
            ``fn`` cannot be traced on one sample row, or if ``dfdx`` does not
            mirror ``Y``.
    """
    use_autodiff = _resolve_call_style(fn, X, Y, dfdx)
    D = problem.num_vars

    if use_autodiff:
        assert fn is not None and X is not None  # guaranteed by _resolve_call_style
        _check_point_callable(fn, X)
        sum_jac, sum_jac2, Y_raw = moment_sums(fn, X, batch_size)
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
    keep_replicates: bool = False,
) -> DGSMResult:
    """Compute DGSM sensitivity indices and Sobol index bounds.

    DGSM ranks inputs by how strongly the output reacts to them on average.
    The measure is ``nu_i = E[(df/dx_i)^2]``, the mean squared partial
    derivative over the input distribution. It is the natural pick when the
    model is JAX-differentiable: one autodiff sweep over an ordinary Monte
    Carlo sample replaces a dedicated Sobol design.

    Two inequalities convert the moments into a bracket on the total Sobol
    index:

    - **Upper bound** (Poincare / Sobol-Kucherenko inequality):
      ``ST_i <= C_i * nu_i / Var(Y)``. ``C_i`` is the Poincare constant of
      input i's marginal distribution, the sharpest factor for which the
      inequality holds (see :mod:`jaxgsa.dgsm._poincare`).
    - **Lower bound** (Kucherenko-Song):
      ``ST_i >= Var(x_i) * sigma_i^2 / Var(Y)``, with
      ``sigma_i = E[df/dx_i]``, the mean (signed) derivative.

    An input whose upper bound is near zero is provably negligible.

    There are two calling conventions, and you must use exactly one of them:

    - **Autodiff path** (primary): pass ``fn`` and ``X``. ``jax.jacrev``
      differentiates the function, and one pass returns both the Jacobian and
      the forward outputs.
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
        batch_size: Number of N sample rows per batch on the autodiff path.
            The Jacobian accumulates in batches of this many samples, which
            bounds peak memory. None (default) processes all N samples at once.
        on_invalid: What to do about a sample row that holds a non-finite
            value. ``"raise"`` (default) refuses the sample, ``"drop"``
            removes those rows and analyzes the rest, and ``"propagate"``
            warns and computes anyway. The check covers the **derivative** as
            well as the output, on both calling conventions: a derivative that
            blows up poisons ``nu`` even where the output itself is finite.
            The derivative is checked in the ``"Y"`` slot, so the report names
            ``"Y"`` for a bad derivative. On the autodiff path ``X`` is
            checked too, and the rows are masked before the batch reduction,
            so ``"drop"`` gives the same moments as re-running on the smaller
            sample. See :mod:`jaxgsa._core.invalid`.
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
            bounds for that slice NaN.
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
    preamble = prepare_scalars(
        SPEC,
        problem,
        on_invalid=on_invalid,
        checks=(
            at_least("batch_size", batch_size, 1),
            at_least("n_bootstrap", n_bootstrap, 0),
            one_of("ci_method", ci_method, ("quantile", "gaussian")),
            in_open_interval("conf_level", conf_level, 0.0, 1.0),
        ),
        method=_METHOD,
    )
    if n_bootstrap > 0 and key is None:
        raise ValueError("key is required when n_bootstrap > 0")
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
        # of as an IndexError from inside jax.jacrev.
        _check_point_callable(fn, X)
        moments = _compute_moments(fn, X, batch_size=batch_size)
        N = int(X.shape[0])
        # The Jacobian never leaves the device whole, so it reaches the check
        # as a per-row verdict rather than as its own array. The verdict is
        # the same one the batch loop masked with. It rides in as an extra
        # array, which puts it on the model side of the check with Y.
        ctx = preamble.with_data(
            moments.Y,
            X=X,
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
            # ones the report names.
            assert np.array_equal(keep, moments.row_ok)
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
        ctx = preamble.with_data(
            Y,
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
        # with_data was told to expect one Y row per Jacobian row.
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
    _warn_zero_variance_slices(Y_valid, output_names=problem.output_names, var_per_slice=var_y)

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
            "DGSM: some upper bounds are below lower bounds, suggesting "
            "insufficient samples or numerical issues",
            stacklevel=2,
            category=JaxgsaWarning,
        )

    confs: dict[str, Array | None] = dict.fromkeys(_INTERVAL_FIELDS)
    ci: CIInfo | None = None
    if n_bootstrap > 0:
        assert key is not None  # checked before the point estimate ran
        batches: Iterator[tuple[Array, Array]]
        if use_autodiff:
            assert fn is not None
            # ctx.inputs is X with the dropped rows already removed, so the
            # resample draws from the same rows the point estimate averaged.
            batches = ((jac, y) for jac, y, _ in jac_batches(fn, ctx.inputs, batch_size))
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

    return DGSMResult(
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
        batch_size: Rows per batch, or ``None`` for one batch.

    Yields:
        ``(jac, Y)`` per batch, in row order.
    """
    N = int(Y_3d.shape[0])
    step = N if not batch_size or batch_size <= 0 else batch_size
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
