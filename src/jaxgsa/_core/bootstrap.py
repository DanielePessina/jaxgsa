"""Shared bootstrap confidence-interval helpers.

Every method that reports bootstrap confidence intervals goes through these
helpers, so one convention holds across the package: two-tailed, ``alpha/2``
per tail, a choice between the empirical-quantile and normal-approximation
endpoint rules, and tolerance of NaN draws.

There are two entry points. pce, hdmr and shapley use both. The other bootstrapping
``analyze()`` functions draw their own resamples and use :func:`interval` or
the lower-level ``_bootstrap_ci_endpoints``:

* :func:`bootstrap_draws` runs the row-resample loop: split the key, draw
  row indices, call the caller's fit, collect the result. pce and shapley
  call it. hdmr calls it for a small fit and :func:`bootstrap_draws_host`,
  which makes the same draws in a host loop of eager fits, for a large one.
  pawn keeps its own loop, which draws the same rows from the same key. The
  other methods use their own resamplers.
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
from functools import partial
from typing import Any, Literal, TypeVar

import jax
import jax.numpy as jnp
from jax import Array

from jaxgsa._core.batching import get_memory_budget
from jaxgsa._core.result import CIInfo

_T = TypeVar("_T")


def resample_keys(key: Array, n_bootstrap: int) -> Array:
    """Return one PRNG key per bootstrap draw.

    Draw ``r`` depends only on ``keys[r]``. Changing the chunk width or the
    memory budget therefore changes no draw. The array of keys costs almost
    nothing, unlike a pre-generated ``(R, N)`` index matrix.

    Args:
        key: The caller's PRNG key.
        n_bootstrap: Number of draws, ``R``.

    Returns:
        A key array of shape ``(R,)``.
    """
    return jax.random.split(key, n_bootstrap)


def draw_rows(key: Array, n: int) -> Array:
    """Draw ``n`` row indices in ``[0, n)`` with replacement, from one draw key.

    The dtype is ``int32`` on purpose. The default is ``int64`` under x64,
    which doubles the memory of the draw.

    Args:
        key: One key from :func:`resample_keys`.
        n: Number of rows, ``N``.

    Returns:
        An ``int32`` array of shape ``(n,)``.
    """
    return jax.random.randint(key, (n,), 0, n, dtype=jnp.int32)


@partial(jax.jit, static_argnames=("n_bootstrap", "n"))
def draw_row_matrix(key: Array, n_bootstrap: int, n: int) -> Array:
    """Draw all bootstrap resamples as an ``(R, n)`` ``int32`` index matrix.

    Row ``r`` is :func:`draw_rows` of ``resample_keys(key, R)[r]``. The rows
    are drawn one at a time with ``lax.map``, so the transient of the random
    bit generator is one row, not the whole matrix (measured 6x the matrix
    for a single batched ``randint``). Use this where the method needs the
    full index matrix. Methods that can chunk should call
    :func:`resample_keys` and :func:`draw_rows` per chunk.

    Args:
        key: The caller's PRNG key.
        n_bootstrap: Number of resamples, ``R``.
        n: Number of rows to draw from, which is also the resample size.

    Returns:
        An ``int32`` array of shape ``(R, n)``.
    """
    return jax.lax.map(lambda k: draw_rows(k, n), resample_keys(key, n_bootstrap))


@partial(jax.jit, static_argnames=("replicate_fn", "static", "n_bootstrap", "budget"))
def _mapped_draws(
    key: Array,
    operands: tuple[Array, ...],
    *,
    replicate_fn: Callable[..., _T],
    static: tuple[tuple[str, Any], ...],
    n_bootstrap: int,
    budget: int | float,
) -> _T:
    """Run the replicates under one cached ``jit`` with a sequential ``lax.map``.

    Args:
        key: PRNG key the split chain starts from.
        operands: Arrays passed to ``replicate_fn``. The first one sets the
            row count.
        replicate_fn: Module-level per-replicate fit.
        static: Hashable keyword arguments for ``replicate_fn``.
        n_bootstrap: Number of replicates.
        budget: Memory budget in bytes at call time. The fits read it while
            tracing (streamed-versus-single-pass, batch widths), so it is part
            of the cache key and a budget change rebuilds the program. The
            body does not use it.

    Returns:
        The stacked replicate pytree.
    """
    del budget
    N = operands[0].shape[0]
    kwargs = dict(static)
    subkeys = _split_chain(key, n_bootstrap)

    def _one(sub: Array) -> _T:
        return replicate_fn(_replicate_rows(sub, N), *operands, **kwargs)

    return jax.lax.map(_one, subkeys)


def _split_chain(key: Array, n_bootstrap: int) -> Array:
    """Return the replicate keys: ``key`` split once per replicate, in order.

    Args:
        key: PRNG key the split chain starts from.
        n_bootstrap: Number of replicates.

    Returns:
        A key array of shape ``(n_bootstrap,)``.
    """

    def _next(k: Array, _: None) -> tuple[Array, Array]:
        k, sub = jax.random.split(k)
        return k, sub

    _, subkeys = jax.lax.scan(_next, key, None, length=n_bootstrap)
    return subkeys


def _replicate_rows(sub: Array, n: int) -> Array:
    """Draw one replicate's ``n`` row indices with replacement.

    Args:
        sub: The replicate's key from :func:`_split_chain`.
        n: Number of rows, ``N``.

    Returns:
        An integer array of shape ``(n,)``.
    """
    return jax.random.choice(sub, n, shape=(n,), replace=True)


_split_chain_jit = jax.jit(_split_chain, static_argnames=("n_bootstrap",))
_replicate_rows_jit = jax.jit(_replicate_rows, static_argnames=("n",))


def bootstrap_draws(
    key: Array,
    n_bootstrap: int,
    replicate_fn: Callable[..., _T],
    operands: tuple[Array, ...],
    *,
    static: tuple[tuple[str, Any], ...] = (),
) -> _T:
    """Run ``replicate_fn`` on ``n_bootstrap`` row-resamples and stack the results.

    pce, shapley and a small hdmr fit use this loop. One cached ``jax.jit`` holds a
    sequential ``lax.map`` over the replicates, so the program compiles once
    and a repeat call compiles nothing. Each replicate draws ``N`` row
    indices with replacement from its own key, split from ``key`` in order.
    Only one replicate runs at a time. Peak memory is still higher than in a
    Python loop, because the streamed fit's row batches unroll under one jit.
    :func:`bootstrap_draws_host` runs the same draws as a host loop.

    Args:
        key: PRNG key. The same key gives the same row draws.
        n_bootstrap: Number of replicates. Must be at least 1.
        replicate_fn: Called as ``replicate_fn(rows, *operands, **static)``
            with the row indices, shape ``(N,)``. It must be a module-level
            function, so the jit cache hits, and it must return the same
            pytree every call.
        operands: Arrays the replicate fit reads. The first one has ``N``
            rows.
        static: Hashable keyword arguments for ``replicate_fn``, as a tuple
            of ``(name, value)`` pairs, for example ``tuple(sorted(d.items()))``.

    Returns:
        The pytree ``replicate_fn`` returns, with every leaf array gaining a
        new leading axis of length ``n_bootstrap``.
    """
    return _mapped_draws(
        key,
        tuple(operands),
        replicate_fn=replicate_fn,
        static=_hashable_static(static),
        n_bootstrap=n_bootstrap,
        budget=get_memory_budget(),
    )


def bootstrap_draws_host(
    key: Array,
    n_bootstrap: int,
    replicate_fn: Callable[..., _T],
    operands: tuple[Array, ...],
    *,
    static: tuple[tuple[str, Any], ...] = (),
) -> _T:
    """Run the replicates as a host loop of eager fits, one at a time.

    The draws are the ones :func:`bootstrap_draws` makes: the same key chain
    and the same row indices per replicate. Only the execution differs.
    ``replicate_fn`` runs eagerly, so a fit made of several cached jitted
    steps runs each step on its own, exactly as the point estimate does.
    Each replicate finishes before the next one starts. The peak is then the
    point fit's peak plus one replicate's resampled operands, whatever
    ``n_bootstrap`` is, and the loop compiles nothing the point fit did not
    already compile.

    Use this when the replicate is a multi-pass streamed fit. Traced as one
    program, such a fit unrolls its Python row-batch loops and XLA keeps the
    buffers of every pass live together, so the program's peak is far above
    the eager fit's and its compile cost grows with the number of row
    batches. hdmr uses this loop for that reason, except for a small fit,
    where the jitted loop is faster and its extra memory is small.

    Args:
        key: PRNG key. The same key gives the same row draws as
            :func:`bootstrap_draws`.
        n_bootstrap: Number of replicates. Must be at least 1.
        replicate_fn: Called as ``replicate_fn(rows, *operands, **static)``
            with the row indices, shape ``(N,)``. It must return the same
            pytree every call.
        operands: Arrays the replicate fit reads. The first one has ``N``
            rows.
        static: Keyword arguments for ``replicate_fn``, as a tuple of
            ``(name, value)`` pairs.

    Returns:
        The pytree ``replicate_fn`` returns, with every leaf array gaining a
        new leading axis of length ``n_bootstrap``.
    """
    N = operands[0].shape[0]
    kwargs = dict(static)
    subkeys = _split_chain_jit(key, n_bootstrap=n_bootstrap)
    outs = []
    for r in range(n_bootstrap):
        out = replicate_fn(_replicate_rows_jit(subkeys[r], n=N), *operands, **kwargs)
        # Eager dispatch is asynchronous: without this wait the host would
        # queue the next fits and their buffers would all be live at once.
        outs.append(jax.block_until_ready(out))
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *outs)


def _hashable_static(static: tuple[tuple[str, Any], ...]) -> tuple[tuple[str, Any], ...]:
    """Turn scalar arrays in a static tuple into Python scalars.

    A jit static argument must be hashable. A 0-d JAX or NumPy array, for
    example ``ridge=jnp.asarray(1e-6)``, is not, so it becomes a Python scalar.

    Args:
        static: ``(name, value)`` pairs, possibly nested in tuples.

    Returns:
        The same pairs with every 0-d array replaced by its Python value.
    """

    def fix(value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(fix(v) for v in value)
        if hasattr(value, "ndim") and hasattr(value, "item") and value.ndim == 0:
            return value.item()
        return value

    return fix(static)


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
