"""Shared bootstrap confidence-interval helpers.

Every method that reports bootstrap confidence intervals goes through these
helpers, so one convention holds across the package: two-tailed, ``alpha/2``
per tail, a choice between the empirical-quantile and normal-approximation
endpoint rules, and tolerance of NaN draws.

There are two entry points, and a bootstrapping ``analyze()`` normally uses
both:

* :func:`bootstrap_draws` runs the row-resample loop — split the key, draw
  row indices, call the caller's fit, collect the result — once, so the four
  copies of that loop (pawn, pce, hdmr, shapley) become one.
* :func:`interval` turns the point estimate(s) and the draws
  :func:`bootstrap_draws` produced into ``*_conf`` endpoint arrays and the
  one :class:`~jaxgsa._core.result.CIInfo` record a result carries, so the
  eleven hand-built ``CIInfo(...)`` constructions become one call each.

See each function's docstring for the contract, and
:mod:`jaxgsa._core.entry` for the matching convention on the ``key is None``
check: it belongs in ``prepare``'s ``checks=`` tuple, not after ``prepare``
returns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, TypeVar

import jax
import jax.numpy as jnp
from jax import Array

from jaxgsa._core.result import CIInfo

_T = TypeVar("_T")


def bootstrap_draws(
    key: Array,
    N: int,
    n_bootstrap: int,
    replicate_fn: Callable[[Array], _T],
) -> _T:
    """Run ``replicate_fn`` on ``n_bootstrap`` row-resamples and stack the results.

    The loop this replaces is written out identically in four modules (pawn,
    pce, hdmr, shapley): split the key, draw ``N`` row indices with
    replacement, call the per-replicate fit, append the result, and
    ``jnp.stack`` every collected list at the end. ``replicate_fn`` returns
    whatever pytree the per-replicate fit already returns — a single array, a
    tuple, or a ``dict`` of named arrays — and this function stacks every
    leaf of it the same way, so the caller keeps writing its fit function
    exactly as it did with the hand-rolled loop; only the loop itself moves
    here.

    Args:
        key: PRNG key. Not otherwise used by the caller: an internal
            ``n_bootstrap``-long split sequence draws the row resamples, so
            passing the same key twice reproduces the same draws.
        N: Row count of the pool ``replicate_fn`` resamples from.
        n_bootstrap: Number of bootstrap replicates to draw. Must be at
            least 1. Every caller already guards its bootstrap branch with
            ``if n_bootstrap > 0``, so this function does not repeat the
            check; there is no empty pytree to return for 0.
        replicate_fn: Called once per replicate with that replicate's row
            indices, shape ``(N,)`` (drawn with replacement). Must return the
            same pytree structure every call — the same set of dict keys, the
            same tuple length — because the leaves are stacked position by
            position.

    Returns:
        The same pytree ``replicate_fn`` returns, with every leaf array
        gaining a new leading axis of length ``n_bootstrap``.
    """
    parts = []
    for _ in range(n_bootstrap):
        key, subkey = jax.random.split(key)
        rows = jax.random.choice(subkey, N, shape=(N,), replace=True)
        parts.append(replicate_fn(rows))
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *parts)


def _bootstrap_ci_endpoints(
    estimate: Array,
    draws: Array,
    *,
    conf_level: float,
    ci_method: Literal["quantile", "gaussian"],
) -> tuple[Array, Array]:
    """Convert bootstrap draws into lower and upper endpoint arrays.

    Args:
        estimate: Point estimate the interval is centered on. Used by the
            ``"gaussian"`` method only.
        draws: ``(n_bootstrap, ...)`` bootstrap replicates.
        conf_level: Two-sided confidence level, e.g. ``0.95``.
        ci_method: ``"quantile"`` for the empirical percentile interval, or
            ``"gaussian"`` for the normal-approximation interval.

    Returns:
        A tuple ``(lower, upper)``, each shaped like ``estimate``.
    """
    # Split the two-tailed CI: e.g. conf_level=0.95 -> alpha=0.025 per tail.
    alpha = (1.0 - conf_level) / 2.0

    if ci_method == "quantile":
        # Non-parametric: read endpoints directly from the empirical bootstrap
        # distribution.  No normality assumption, but needs enough resamples.
        percentiles = jnp.array([alpha * 100, (1.0 - alpha) * 100])
        endpoints = jnp.nanpercentile(draws, percentiles, axis=0)
        return endpoints[0], endpoints[1]

    # Parametric (Gaussian) CI: assumes the bootstrap distribution is normal.
    # ndtri is the inverse normal CDF (quantile function): z = Phi^-1(1-alpha/2).
    # CI = estimate +/- z * sigma_boot.  nanstd tolerates degenerate bootstrap
    # resamples that collapse to a single unique value and produce NaN.
    z_score = jax.scipy.special.ndtri(1.0 - alpha)
    bootstrap_sd = jnp.nanstd(draws, axis=0, ddof=1)
    half_width = z_score * bootstrap_sd
    return estimate - half_width, estimate + half_width


def interval(
    points: Mapping[str, Array],
    draws: Mapping[str, Array],
    *,
    level: float,
    method: Literal["quantile", "gaussian"],
    n_bootstrap: int,
    keep_replicates: bool,
) -> tuple[dict[str, Array], CIInfo]:
    """Turn named point estimates and bootstrap draws into ``*_conf`` arrays and a ``CIInfo``.

    Every bootstrapping ``analyze()`` ends its bootstrap branch the same way:
    call :func:`_bootstrap_ci_endpoints` once per index field, ``jnp.stack``
    each ``[lower, upper]`` pair, and hand-build a
    :class:`~jaxgsa._core.result.CIInfo` that repeats ``level``, ``method``
    and ``n_bootstrap`` and conditionally carries the draws. This function is
    that ending, written once.

    Contract:

    * ``points`` and ``draws`` must share the same keys — one key per index
      field the result reports an interval for (``"S1"``, ``"mu_star"``,
      ``"pawn"``, ...). ``draws[name]`` must have a leading axis of length
      ``n_bootstrap`` and otherwise the shape of ``points[name]``.
    * Call it on whatever layout ``points``/``draws`` already carry — the
      caller's own squeezed layout, or the ``(T, K, ...)``-promoted one.
      Nothing here reads a shape or applies :meth:`Context.squeeze
      <jaxgsa._core.entry.Context.squeeze>`; do that to ``points``/``draws``
      *before* calling, exactly as before, and the returned ``conf`` and
      ``ci.replicates`` come back already at that layout. Squeezing after the
      call would need a second pass over ``ci.replicates``, which is a frozen
      dataclass and cannot be edited in place.
    * ``n_bootstrap`` is taken from the caller, not read off ``draws``'
      leading axis, so a caller that drew fewer usable replicates than it
      asked for (a degenerate resample dropped, say) still reports the count
      it asked for, matching every existing ``CIInfo(n_bootstrap=n_bootstrap,
      ...)`` construction.
    * ``ci.replicates`` is ``dict(draws)`` when ``keep_replicates`` is
      ``True``, ``None`` otherwise — the same ``if keep_replicates else
      None`` every call site wrote by hand.

    Args:
        points: Point estimate per index field, keyed by field name.
        draws: Bootstrap draws per index field, same keys as ``points``, each
            with a leading axis of length ``n_bootstrap``.
        level: Two-sided confidence level the interval is drawn at.
        method: Endpoint rule; ``"quantile"`` or ``"gaussian"``.
        n_bootstrap: Number of bootstrap resamples drawn, as recorded on the
            returned ``CIInfo`` (see the contract above).
        keep_replicates: Whether ``ci.replicates`` carries ``draws``.

    Returns:
        ``(conf, ci)``. ``conf`` maps each name in ``points`` to a
        ``(2, ...)`` ``[lower, upper]`` array, ready to assign to the
        result's ``<name>_conf`` field. ``ci`` is the one
        :class:`~jaxgsa._core.result.CIInfo` to store on the result.
    """
    conf = {
        name: jnp.stack(
            _bootstrap_ci_endpoints(points[name], draws[name], conf_level=level, ci_method=method)
        )
        for name in points
    }
    ci = CIInfo(
        level=level,
        method=method,
        n_bootstrap=n_bootstrap,
        replicates=dict(draws) if keep_replicates else None,
    )
    return conf, ci
