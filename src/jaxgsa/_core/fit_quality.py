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
# already scores ``loo_rmse == std(Y)``, so a ratio anywhere near 1 says the
# surrogate carries no usable predictive information, whatever its in-sample
# fit looks like.
#
# The line sits where the leave-one-out R-squared falls to POORFIT_THRESHOLD,
# that is ``1 - ratio**2 < 0.5``. Both warnings then fire at "half the variance
# is unaccounted for" and only the sample they measure on differs. Measured
# cases sit well clear of it on both sides: an order-5 PCE on 64 Ishigami rows
# (fit_ratio=0.9) gives loo_rmse 9.59 against std(Y) 3.59, a ratio of 2.67,
# while its in-sample fraction reads 0.98 and reports nothing wrong at all. An
# order-8 fit on 2000 rows gives loo_rmse 0.076 against std(Y) 3.75, a ratio
# of 0.020.
LOO_RATIO_THRESHOLD = math.sqrt(1.0 - POORFIT_THRESHOLD)


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
    values = jnp.asarray(explained)
    finite = values[jnp.isfinite(values)]
    if finite.size == 0:
        return
    lowest = float(jnp.min(finite))
    highest = float(jnp.max(finite))
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


def warn_loo_overfit(
    namespace: str,
    loo_rmse: Array | None,
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
        loo_rmse: Leave-one-out RMSE per output slice, in the units of ``Y``,
            or ``None`` when the fit reported none, which silences the check.
        std_y: Sample standard deviation of ``Y`` per output slice, over the
            rows the surrogate was fitted on. Same shape as ``loo_rmse``.
        explained: The in-sample variance fraction, when the caller has it.
            A fit that already failed :func:`warn_variance_fit` is
            underfitting, and calling it overfit as well would be two
            warnings for one problem and the wrong name for it. A real
            overfit scores near 1 in sample, so this gate costs no
            sensitivity.
        advice: What to do about it. Appended to the message.

    Warns:
        JaxgsaWarning: If the worst slice's ratio passes
            :data:`LOO_RATIO_THRESHOLD`.
    """
    if loo_rmse is None:
        return
    ratio = jnp.asarray(loo_rmse) / jnp.asarray(std_y)
    finite = ratio[jnp.isfinite(ratio)]
    if finite.size == 0:
        return
    worst = float(jnp.max(finite))
    if worst <= LOO_RATIO_THRESHOLD:
        return
    if explained is not None:
        in_sample = jnp.asarray(explained)
        healthy = in_sample[jnp.isfinite(in_sample)]
        if healthy.size and float(jnp.min(healthy)) < POORFIT_THRESHOLD:
            return
    warnings.warn(
        f"{namespace}: loo_rmse is {worst:.2f} times std(Y) on at least one output "
        "slice, so the surrogate predicts less than half of the output variance out "
        "of sample. A leave-one-out error approaching std(Y) is the signature of an "
        f"overfit, which a high in-sample fit will not show. {advice}",
        skip_file_prefixes=(PACKAGE_DIR,),
        category=JaxgsaWarning,
    )
