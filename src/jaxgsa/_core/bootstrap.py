"""Shared bootstrap confidence-interval helpers.

Every method that reports bootstrap confidence intervals (Sobol, Morris, PAWN,
Borgonovo) goes through these helpers. One percentile convention therefore
holds across the package: two-tailed, ``alpha/2`` per tail, and tolerant of
NaN draws.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array


def _percentile_ci(bootstrap_draws: Array, conf_level: float) -> Array:
    """Stack the two-tailed percentile CI endpoints of bootstrap draws.

    Args:
        bootstrap_draws: ``(n_bootstrap, ...)`` bootstrap replicates.
        conf_level: Two-sided confidence level, e.g. ``0.95``.

    Returns:
        ``(2, ...)`` array holding the ``[lower, upper]`` endpoints.
    """
    # Split the two-tailed CI: e.g. conf_level=0.95 -> alpha=0.025 per tail
    alpha = (1.0 - conf_level) / 2.0
    percentiles = jnp.array([alpha * 100, (1.0 - alpha) * 100])
    return jnp.nanpercentile(bootstrap_draws, percentiles, axis=0)


def _bootstrap_ci_endpoints(
    estimate: Array,
    bootstrap_draws: Array,
    *,
    conf_level: float,
    ci_method: Literal["quantile", "gaussian"],
) -> tuple[Array, Array]:
    """Convert bootstrap draws into lower and upper endpoint arrays.

    Args:
        estimate: Point estimate the interval is centered on. Used by the
            ``"gaussian"`` method only.
        bootstrap_draws: ``(n_bootstrap, ...)`` bootstrap replicates.
        conf_level: Two-sided confidence level, e.g. ``0.95``.
        ci_method: ``"quantile"`` for the empirical percentile interval, or
            ``"gaussian"`` for the normal-approximation interval.

    Returns:
        A tuple ``(lower, upper)``, each shaped like ``estimate``.
    """
    if ci_method == "quantile":
        # Non-parametric: read endpoints directly from the empirical bootstrap
        # distribution.  No normality assumption, but needs enough resamples.
        endpoints = _percentile_ci(bootstrap_draws, conf_level)
        return endpoints[0], endpoints[1]

    # Parametric (Gaussian) CI: assumes the bootstrap distribution is normal.
    # ndtri is the inverse normal CDF (quantile function): z = Phi^-1(1-alpha/2).
    # CI = estimate +/- z * sigma_boot.  nanstd tolerates degenerate bootstrap
    # resamples that collapse to a single unique value and produce NaN.
    alpha = (1.0 - conf_level) / 2.0
    z_score = jax.scipy.special.ndtri(1.0 - alpha)
    bootstrap_sd = jnp.nanstd(bootstrap_draws, axis=0, ddof=1)
    half_width = z_score * bootstrap_sd
    return estimate - half_width, estimate + half_width
