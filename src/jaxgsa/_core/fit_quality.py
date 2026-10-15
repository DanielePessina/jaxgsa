"""Fit-quality warnings shared by the surrogate-backed methods.

Every index a surrogate method reports is read off the fitted surrogate, so a
surrogate that does not represent the model returns indices that are exactly
right about the wrong function. Nothing in the index arrays says so: they carry
the same shapes, the same [0, 1] range and the same plausible ranking either
way. The two diagnostics that do say so are an in-sample variance fraction and
an out-of-sample error, and both live here because ``pce``, ``hdmr`` and the
Shapley effects derived from them read the same numbers on the same scale.

The thresholds are shared for the same reason. One line at "half the output
variance is unaccounted for" means a warning from ``jaxgsa.pce`` and one from
``jaxgsa.hdmr`` report comparable severity.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import Array

from jaxgsa._core.warning_types import JaxgsaWarning

# The directory this package's own frames live under. A warning raised with it
# points at the first frame outside jaxgsa, whichever call chain arrived:
# `pce.analyze` called directly, `shapley.analyze` calling it one frame deeper,
# or `result.shapley()` from a fit the caller already holds. A hand-counted
# stacklevel cannot be right for all three at once.
PACKAGE_DIR = str(Path(__file__).resolve().parent.parent) + os.sep

# A surrogate that leaves more than half the output variance unaccounted for.
POORFIT_THRESHOLD = 0.5

# A decomposition that accounts for more variance than the output has. Only
# reachable for a diagnostic that is not a true R-squared: HDMR's per-term sum
# can pass 1 when the terms overlap, while PCE's in-sample fraction cannot.
OVERFIT_THRESHOLD = 1.3

# The out-of-sample twin of POORFIT_THRESHOLD, for a surrogate that reports a
# leave-one-out error.
#
# An in-sample fraction cannot see an overfit: an expansion with as many terms
# as it has rows scores near 1 on the rows it was fitted to, by construction.
# The signal that does move is the leave-one-out error turning back up towards
# the spread of the data itself. Predicting every row by the output mean
# already scores ``cv_rmse == std(Y)``, so a ratio anywhere near 1 says the
# surrogate carries no usable predictive information, whatever its in-sample
# fit looks like.
#
# The line sits where the leave-one-out R-squared falls to POORFIT_THRESHOLD,
# that is ``1 - ratio**2 < 0.5``. Both warnings then fire at "half the variance
# is unaccounted for" and only the sample they measure on differs. Measured
# cases sit well clear of it on both sides: an order-5 PCE on 64 Ishigami rows
# (fit_ratio=0.9) gives cv_rmse 9.59 against std(Y) 3.59, a ratio of 2.67,
# while its in-sample fraction reads 0.98 and reports nothing wrong at all. An
# order-8 fit on 2000 rows gives cv_rmse 0.076 against std(Y) 3.75, a ratio
# of 0.020.
CV_RATIO_THRESHOLD = math.sqrt(1.0 - POORFIT_THRESHOLD)

# Limits on the pivot share of a Gram matrix, per dtype.
#
# The pivot share of column j is the squared Cholesky pivot ``L[j, j]**2``
# over the Gram diagonal ``G[j, j]``: the fraction of the column's variance
# that is left after the columns before it explain it. A rank-deficient
# design has a share of zero in exact arithmetic. In practice the share stops
# at a floor: the ridge sets it in float64 (about ``ridge / N``), and
# rounding sets it in float32 (it grows with N, about 1.8e-5 at N = 10000).
#
# float64 refuses a share below 1e-8. Measured singular designs, on 2000 rows
# at order 3, sit at or below 1.1e-10: a duplicated input gives 4.9e-12, a
# constant input 8.8e-12, a three-level input 3.5e-12, an input that is the
# square of another 1.1e-10. The lowest well-posed shares measured are 5.6e-7
# (D = 10, order 4, fit_ratio = 1) and 1.6e-6 (three Gaussian inputs, order
# 10, N = 600), and those fits are accurate.
#
# float32 cannot resolve the same range. Its rounding floor overlaps the
# shares of singular designs, so a three-level input on 2000 rows gives
# 1.6e-5 in float32 and 3.5e-12 in float64. Below 1e-5 float32 refuses the
# fit, and NaN (a failed factorisation) counts as below. From 1e-5 to 1e-4 it
# warns, because the design may be singular and float32 cannot tell. The
# lowest well-posed float32 share measured is 1.5e-4 (D = 10, order 4,
# fit_ratio = 1).
SINGULAR_PIVOT_THRESHOLD_F64 = 1e-8
SINGULAR_PIVOT_THRESHOLD_F32 = 1e-5
AMBIGUOUS_PIVOT_THRESHOLD_F32 = 1e-4


@jax.jit
def _finite_range(values: Array) -> tuple[Array, Array, Array]:
    """Count the finite entries and find their smallest and largest.

    One compiled program replaces a boolean-mask gather and two reductions,
    each of which would compile on its first eager call. The masked min and
    max return the same elements the gather would have picked.

    Args:
        values: Any array.

    Returns:
        ``(count, lowest, highest)``. The extremes are meaningless when the
        count is zero.
    """
    values = jnp.asarray(values)
    ok = jnp.isfinite(values)
    lowest = jnp.min(jnp.where(ok, values, jnp.inf))
    highest = jnp.max(jnp.where(ok, values, -jnp.inf))
    return jnp.sum(ok), lowest, highest


@jax.jit
def _cv_ratio_stats(
    cv_rmse: Array, std_y: Array, explained: Array | None
) -> tuple[Array, Array, Array, Array]:
    """Everything :func:`warn_cv_overfit` reads, from one compiled program.

    Args:
        cv_rmse: Leave-one-out RMSE per output slice.
        std_y: Standard deviation of ``Y`` per output slice.
        explained: The in-sample variance fraction, or ``None``.

    Returns:
        ``(n_finite, worst, n_healthy, lowest_in_sample)``: the count of
        finite ratios, the largest of them, and the count and smallest of the
        finite in-sample fractions (zero and ``inf`` when ``explained`` is
        ``None``).
    """
    ratio = jnp.asarray(cv_rmse) / jnp.asarray(std_y)
    ok = jnp.isfinite(ratio)
    worst = jnp.max(jnp.where(ok, ratio, -jnp.inf))
    if explained is None:
        return jnp.sum(ok), worst, jnp.zeros((), jnp.int32), jnp.asarray(jnp.inf, ratio.dtype)
    in_sample = jnp.asarray(explained)
    healthy = jnp.isfinite(in_sample)
    lowest = jnp.min(jnp.where(healthy, in_sample, jnp.inf))
    return jnp.sum(ok), worst, jnp.sum(healthy), lowest


def warn_variance_fit(
    namespace: str,
    explained: Array | None,
    *,
    quantity: str,
    advice_low: str,
    advice_high: str,
) -> None:
    """Warn when a surrogate's in-sample variance fraction is implausible.

    The check is per output slice, and one warning names the worst slice
    rather than one warning per slice. Slices whose value is not finite (a
    constant output has no variance to divide by) are skipped.

    Args:
        namespace: The public function the warning speaks for, e.g.
            ``"jaxgsa.pce"``. It opens the message.
        explained: The fraction of output variance the fit accounts for, per
            output slice. ``None`` silences the check.
        quantity: What to call that number in the message, as the caller's
            own users would read it, e.g. ``"explained_variance"``.
        advice_low: What to do about a fit that explains too little. Appended
            to the message.
        advice_high: What to do about a fit that accounts for more variance
            than the output has.

    Warns:
        JaxgsaWarning: If the worst slice falls below
            :data:`POORFIT_THRESHOLD` or rises above :data:`OVERFIT_THRESHOLD`.
    """
    if explained is None:
        return
    n_finite, lowest_dev, highest_dev = _finite_range(explained)
    if int(n_finite) == 0:
        return
    lowest = float(lowest_dev)
    highest = float(highest_dev)
    if lowest < POORFIT_THRESHOLD:
        warnings.warn(
            f"{namespace}: {quantity} is {lowest:.2f} on at least one output slice, "
            f"below {POORFIT_THRESHOLD}, so the surrogate leaves more than half of "
            "the output variance unexplained. Every index is computed from the fit, "
            f"so they describe the surrogate rather than your model. {advice_low}",
            skip_file_prefixes=(PACKAGE_DIR,),
            category=JaxgsaWarning,
        )
    elif highest > OVERFIT_THRESHOLD:
        warnings.warn(
            f"{namespace}: {quantity} is {highest:.2f} on at least one output slice, "
            f"above {OVERFIT_THRESHOLD}, so the fitted terms account for more variance "
            f"than the output has. {advice_high}",
            skip_file_prefixes=(PACKAGE_DIR,),
            category=JaxgsaWarning,
        )


def warn_cv_overfit(
    namespace: str,
    cv_rmse: Array | None,
    std_y: Array,
    explained: Array | None = None,
    *,
    advice: str,
) -> None:
    """Warn when a surrogate's leave-one-out error approaches ``std(Y)``.

    This is the out-of-sample half of the pair. :func:`warn_variance_fit`
    catches a surrogate that missed variance on the rows it was fitted to.
    An overfit surrogate passes that check by construction, and only a
    leave-one-out number sees it.

    The check is per output slice, and one warning names the worst slice.
    Slices whose ratio is not finite are skipped.

    Args:
        namespace: The public function the warning speaks for.
        cv_rmse: Cross-validated (leave-one-out) RMSE per output slice, in the units of ``Y``,
            or ``None`` when the fit reported none, which silences the check.
        std_y: Sample standard deviation of ``Y`` per output slice, over the
            rows the surrogate was fitted on. Same shape as ``cv_rmse``.
        explained: The in-sample variance fraction, when the caller has it.
            A fit that already failed :func:`warn_variance_fit` is
            underfitting, and calling it overfit as well would be two
            warnings for one problem and the wrong name for it. A real
            overfit scores near 1 in sample, so this gate costs no
            sensitivity.
        advice: What to do about it. Appended to the message.

    Warns:
        JaxgsaWarning: If the worst slice's ratio passes
            :data:`CV_RATIO_THRESHOLD`.
    """
    if cv_rmse is None:
        return
    n_finite, worst_dev, n_healthy, lowest_dev = _cv_ratio_stats(cv_rmse, std_y, explained)
    if int(n_finite) == 0:
        return
    worst = float(worst_dev)
    if worst <= CV_RATIO_THRESHOLD:
        return
    if explained is not None and int(n_healthy) and float(lowest_dev) < POORFIT_THRESHOLD:
        return
    warnings.warn(
        f"{namespace}: cv_rmse is {worst:.2f} times std(Y) on at least one output "
        "slice, so the surrogate predicts less than half of the output variance out "
        "of sample. A leave-one-out error approaching std(Y) is the signature of an "
        f"overfit, which a high in-sample fit will not show. {advice}",
        skip_file_prefixes=(PACKAGE_DIR,),
        category=JaxgsaWarning,
    )


@jax.jit
def min_pivot_share(gram: Array) -> Array:
    """The smallest unexplained share of any column of a Gram matrix.

    Args:
        gram: Symmetric Gram matrix, shape ``(n_terms, n_terms)``, with any
            ridge already added.

    Returns:
        ``min_j L[j, j]**2 / G[j, j]``, a scalar in ``[0, 1]``. NaN when the
        Cholesky factorisation fails or a column is all zeros.
    """
    chol = jnp.linalg.cholesky(gram)
    return jnp.min(jnp.diagonal(chol) ** 2 / jnp.diagonal(gram))


def check_gram_rank(namespace: str, pivot_share: Array, *, advice: str) -> None:
    """Refuse a fit whose Gram matrix is numerically rank deficient.

    :func:`check_fit_finite` sees a singular Gram matrix only when it turns
    into NaN. That happens in float32. In float64 the factorisation succeeds,
    the ridge keeps the solve finite, and the coefficients split one effect
    across collinear columns at will. Every number stays finite and the
    indices are wrong: a duplicated input shares the true input's index, and
    a constant input gets a large one. This check reads the conditioning
    directly. The limits depend on the dtype of ``pivot_share``; see
    :data:`SINGULAR_PIVOT_THRESHOLD_F64`.

    Args:
        namespace: The public function the message speaks for.
        pivot_share: The output of :func:`min_pivot_share`.
        advice: What to do about it. Appended to the message.

    Raises:
        ValueError: If ``pivot_share`` is not finite, or is below
            :data:`SINGULAR_PIVOT_THRESHOLD_F64` in float64 or
            :data:`SINGULAR_PIVOT_THRESHOLD_F32` in float32.

    Warns:
        JaxgsaWarning: In float32, if ``pivot_share`` is below
            :data:`AMBIGUOUS_PIVOT_THRESHOLD_F32` but not refused.
    """
    is_f64 = jnp.asarray(pivot_share).dtype == jnp.float64
    limit = SINGULAR_PIVOT_THRESHOLD_F64 if is_f64 else SINGULAR_PIVOT_THRESHOLD_F32
    share = float(pivot_share)
    if not math.isfinite(share):
        raise ValueError(
            f"{namespace}: the Gram matrix of the design is numerically singular: "
            "its Cholesky factorisation failed. The coefficients are not "
            f"identified, so the indices are undefined. {advice}"
        )
    if share < limit:
        raise ValueError(
            f"{namespace}: the Gram matrix of the design is numerically singular: "
            f"one basis column has only {share:.1e} of its variance left after the "
            f"earlier columns explain it, below {limit:.0e}. The coefficients are "
            f"not identified, so the indices are undefined. {advice}"
        )
    if not is_f64 and share < AMBIGUOUS_PIVOT_THRESHOLD_F32:
        warnings.warn(
            f"{namespace}: the Gram matrix of the design may be numerically "
            f"singular: one basis column has only {share:.1e} of its variance left "
            "after the earlier columns explain it. In float32 this cannot be told "
            "apart from a singular design, so the indices may be undefined. Enable "
            "jax x64 (jax.config.update('jax_enable_x64', True)) to resolve it. "
            f"{advice}",
            skip_file_prefixes=(PACKAGE_DIR,),
            category=JaxgsaWarning,
        )


@jax.jit
def _non_finite_stats(
    coefficients: Array, cv_rmse: Array, rmse: Array, std_y: Array
) -> tuple[Array, Array]:
    """Count the non-finite fit entries that a constant output cannot explain.

    A constant output slice has ``std(Y) == 0`` and legitimately reports
    undefined ratios. A slice with spread has no such excuse, so a non-finite
    leave-one-out or in-sample error there means the fit itself failed.

    Args:
        coefficients: Fitted coefficients, any shape.
        cv_rmse: Leave-one-out RMSE per output slice.
        rmse: In-sample RMSE per output slice, same shape as ``cv_rmse``.
        std_y: Standard deviation of ``Y`` per slice, same shape as
            ``cv_rmse``.

    Returns:
        ``(n_bad_coefficients, n_bad_slices)``.
    """
    spread = jnp.asarray(std_y) > 0
    bad_slice = spread & ~(jnp.isfinite(cv_rmse) & jnp.isfinite(rmse))
    return jnp.sum(~jnp.isfinite(coefficients)), jnp.sum(bad_slice)


def check_fit_finite(
    namespace: str,
    coefficients: Array,
    cv_rmse: Array | None,
    rmse: Array | None,
    std_y: Array,
    *,
    advice: str,
) -> None:
    """Refuse or warn when a fit produced non-finite numbers.

    The other checks in this module skip non-finite entries, because a
    constant output has none to report. That also hides a real failure: when
    the Gram matrix is numerically singular, the Cholesky factor behind the
    leave-one-out error is NaN, the coefficients can overflow, and every
    index read from them is meaningless, yet the arrays keep their shapes.

    Non-finite coefficients make the indices undefined, so the call raises.
    A non-finite error on a slice with spread leaves the indices defined but
    unverified, so the call warns.

    Args:
        namespace: The public function the message speaks for.
        coefficients: Fitted coefficients, any shape.
        cv_rmse: Leave-one-out RMSE per output slice, or ``None`` when the
            fit reported none (the error check is then skipped).
        rmse: In-sample RMSE per output slice, or ``None``.
        std_y: Standard deviation of ``Y`` per output slice.
        advice: What to do about it. Appended to the message.

    Raises:
        ValueError: If any coefficient is not finite.

    Warns:
        JaxgsaWarning: If ``cv_rmse`` or ``rmse`` is not finite on a slice
            whose output is not constant.
    """
    check_errors = cv_rmse is not None and rmse is not None
    n_coef, n_slices = _non_finite_stats(
        coefficients,
        cv_rmse if cv_rmse is not None else jnp.zeros(()),
        rmse if rmse is not None else jnp.zeros(()),
        std_y if check_errors else jnp.ones(()),
    )
    if int(n_coef):
        raise ValueError(
            f"{namespace}: the fitted coefficients are not finite, because the Gram "
            "matrix of the design is numerically singular. The indices are "
            f"undefined. {advice}"
        )
    if check_errors and int(n_slices):
        warnings.warn(
            f"{namespace}: cv_rmse or rmse is not finite on {int(n_slices)} output "
            "slice(s) whose output is not constant. The Gram matrix of the design "
            "is numerically singular, so the leave-one-out factorisation failed. "
            f"The indices are unreliable. {advice}",
            skip_file_prefixes=(PACKAGE_DIR,),
            category=JaxgsaWarning,
        )
