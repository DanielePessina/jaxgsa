"""Opt-in runtime configuration helpers for jaxgsa.

These helpers wrap two kinds of process-global setting worth tuning for
sensitivity-analysis workloads: JAX runtime flags, and jaxgsa's own
transient-memory budget. Nothing here is applied on import. Each setting
mutates global state the host application may also depend on, so it changes
behavior only when you call the helper yourself.
"""

import math as _math
from pathlib import Path as _Path

import jax as _jax
import numpy as _np

from jaxgsa._core import batching as _batching

__all__ = ["enable_compilation_cache", "get_memory_budget", "set_memory_budget"]

# Accepted ``unit`` spellings and the number of bytes each one means.
#
# All multipliers are binary (powers of 1024), so ``mb`` is 1024**2 and not
# 1000**2. Memory is conventionally counted in binary units, and this choice
# makes ``set_memory_budget(512)`` resolve to exactly the historical default of
# ``512 * 1024**2`` bytes. The explicit binary spellings (``kib``, ``mib`` and
# so on) are exact synonyms of their short forms; they are accepted so callers
# who prefer to be unambiguous can be.
_UNIT_BYTES: dict[str, int] = {
    "b": 1,
    "bytes": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024**2,
    "mib": 1024**2,
    "gb": 1024**3,
    "gib": 1024**3,
    "tb": 1024**4,
    "tib": 1024**4,
}


def enable_compilation_cache(
    path: str | _Path,
    *,
    min_compile_time_secs: float = 1.0,
    min_entry_size_bytes: int = 0,
) -> str:
    """Enable JAX's persistent, on-disk compilation cache.

    JAX caches compiled XLA executables in memory for the lifetime of a process.
    The persistent cache stores them on disk as well. Repeated runs of the same
    analysis across process restarts (parameter sweeps, CI jobs, HPC batches)
    then skip the cold XLA compile. Call this once, before the first ``analyze``
    call (``jaxgsa.sobol.analyze``, for example), so the cache is active when the
    first compilation happens.

    Args:
        path: Directory used to store compiled executables. A leading ``~`` is
            expanded and the result is resolved to an absolute path, so the cache
            location does not depend on the process's working directory. Created
            lazily by JAX on the first cache write.
        min_compile_time_secs: Only cache executables whose compilation took at
            least this many seconds, so trivially cheap kernels are not persisted.
        min_entry_size_bytes: Minimum serialized executable size, in bytes, to
            cache. ``0`` allows a filesystem-specific default. Coerced to ``int``.

    Returns:
        The absolute cache directory path that was configured.

    Warning:
        The cache directory is effectively executable: anyone who can write to it
        can make this process load and run arbitrary compiled code. Never point it
        at a world-writable or shared, untrusted location.
    """
    # expanduser() handles ``~``; absolute() pins the cache to a fixed location
    # independent of cwd (JAX would otherwise resolve a relative path at each write).
    cache_dir = str(_Path(path).expanduser().absolute())
    _jax.config.update("jax_compilation_cache_dir", cache_dir)
    _jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)
    # This flag is strict-int in JAX; coerce so a float like 1e6 does not raise
    # partway through after the two updates above have already taken effect.
    _jax.config.update("jax_persistent_cache_min_entry_size_bytes", int(min_entry_size_bytes))
    return cache_dir


def _resolve_unit(unit: str) -> int:
    """Return the byte multiplier for a unit name.

    The name is harmonised before it is looked up: surrounding whitespace is
    stripped and the rest is lowercased, so ``" MB "``, ``"Mb"`` and ``"mb"``
    are the same unit.

    Args:
        unit: Unit name, such as ``"mb"`` or ``"GiB"``.

    Returns:
        Number of bytes in one of that unit.

    Raises:
        ValueError: If the name is not one of the accepted units.
    """
    if not isinstance(unit, str):
        raise ValueError(f"unit must be a string, got {unit!r}")
    key = unit.strip().lower()
    if key not in _UNIT_BYTES:
        accepted = ", ".join(sorted(_UNIT_BYTES))
        raise ValueError(f"unknown memory unit {unit!r}; accepted units are: {accepted}")
    return _UNIT_BYTES[key]


def set_memory_budget(budget: int | float, *, unit: str | None = None) -> None:
    """Set the global transient-memory budget used for automatic batching.

    jaxgsa bounds peak memory in several places by processing data in batches
    sized against this budget: surrogate ``predict`` (PCE, HDMR), the
    output-slice chunking of HDMR, PAWN, Borgonovo, Sobol and eFAST, the
    Sobol bootstrap and Morris resample chunking, the DGSM Jacobian batching,
    the PCE S2 pair-mask chunking, and the PCE and HDMR streaming fits that
    engage when the single-pass design matrix would not fit. All of them
    derive their automatic batch/chunk sizes from this budget (default:
    512 MiB); an explicit ``batch_size``/``*_chunk_size`` always wins over
    the budget.

    Borgonovo also tiles its output grid, which is what bounds its peak in
    practice, but the tile width comes from a fixed working-set target rather
    than from this budget. Lowering the budget narrows the slice chunk; it
    does not narrow the tile.

    The budget is read in megabytes unless you say otherwise, so
    ``set_memory_budget(512)`` restates the default. Every unit is binary:
    ``mb`` means ``1024**2`` bytes, not ``1000**2``, and ``kb``/``gb``/``tb``
    follow the same rule. ``mib``, ``kib``, ``gib`` and ``tib`` are exact
    synonyms of the short forms. Unit names are case-insensitive and
    surrounding whitespace is ignored.

    This is an opt-in process-global setting. Nothing changes until you call
    it, and the new budget applies only to jaxgsa calls made after it.
    Analyses already running keep the budget they started with. Explicit
    per-call parameters (``batch_size``, ``slice_chunk_size``) always take
    precedence over this budget.

    Args:
        budget: New budget, counted in ``unit``. May be a Python ``int`` or
            ``float``, or a NumPy integer or floating scalar
            (``set_memory_budget(1.5, unit="gb")`` and
            ``set_memory_budget(np.int64(512))`` are both fine). It must be
            positive and finite. ``bool`` is rejected. The resolved byte
            count is rounded to the nearest whole byte and must come to at
            least one byte.
        unit: Unit of ``budget``. One of ``"b"``, ``"bytes"``, ``"kb"``,
            ``"kib"``, ``"mb"``, ``"mib"``, ``"gb"``, ``"gib"``, ``"tb"`` or
            ``"tib"``. Defaults to megabytes.

    Raises:
        ValueError: If ``budget`` is not a positive, finite number, if it is a
            ``bool``, if it rounds to less than one byte, if ``unit`` is not an
            accepted name.
    """
    if isinstance(budget, bool | _np.bool_) or not isinstance(
        budget, int | float | _np.integer | _np.floating
    ):
        raise ValueError(f"memory budget must be a positive, finite number, got {budget!r}")
    if not _math.isfinite(budget) or budget <= 0:
        raise ValueError(f"memory budget must be a positive, finite number, got {budget!r}")

    multiplier = _resolve_unit("mb" if unit is None else unit)
    budget_bytes = round(budget * multiplier)
    if budget_bytes < 1:
        raise ValueError(
            f"memory budget rounds to {budget_bytes} bytes; it must be at least 1 byte"
        )
    _batching._set_memory_budget(budget_bytes)


def get_memory_budget(*, unit: str = "mb") -> int | float:
    """Return the active transient-memory budget.

    The default unit is megabytes, the same as :func:`set_memory_budget`, so
    ``set_memory_budget(get_memory_budget())`` restores the current budget.

    Args:
        unit: Unit of the returned value. Accepts the same names as
            :func:`set_memory_budget`. Defaults to megabytes.

    Returns:
        The budget set by the most recent :func:`set_memory_budget` call, or
        the built-in default (512 MiB) if it was never called. An exact ``int``
        of bytes for ``"b"``/``"bytes"``; a ``float`` for every other unit,
        because the budget need not be a whole number of them.

    Raises:
        ValueError: If ``unit`` is not an accepted name.
    """
    multiplier = _resolve_unit(unit)
    budget_bytes = _batching.get_memory_budget()
    if multiplier == 1:
        return budget_bytes
    return budget_bytes / multiplier
