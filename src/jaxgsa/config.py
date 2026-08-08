"""Opt-in runtime configuration helpers for jaxgsa.

These helpers wrap two kinds of process-global setting worth tuning for
sensitivity-analysis workloads: JAX runtime flags, and jaxgsa's own
transient-memory budget. Nothing here is applied on import. Each setting
mutates global state the host application may also depend on, so it changes
behavior only when you call the helper yourself.
"""

from pathlib import Path

import jax

from jaxgsa._core import batching as _batching

__all__ = ["enable_compilation_cache", "get_memory_budget", "set_memory_budget"]


def enable_compilation_cache(
    path: str | Path,
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
    cache_dir = str(Path(path).expanduser().absolute())
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)
    # This flag is strict-int in JAX; coerce so a float like 1e6 does not raise
    # partway through after the two updates above have already taken effect.
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", int(min_entry_size_bytes))
    return cache_dir


def set_memory_budget(budget_bytes: int) -> None:
    """Set the global transient-memory budget used for automatic batching.

    jaxgsa bounds peak memory in several places by processing data in batches
    sized against a bytes budget: surrogate ``predict`` (PCE, HDMR), the HDMR
    output-slice chunking, and the PCE streaming fit that engages when the
    single-pass design matrix would not fit. All of them derive their
    automatic batch/chunk sizes from this budget (default: 512 MiB).

    This is an opt-in process-global setting. Nothing changes until you call
    it, and the new budget applies only to jaxgsa calls made after it.
    Analyses already running keep the budget they started with. Explicit
    per-call parameters (``batch_size``, ``slice_chunk_size``) always take
    precedence over this budget.

    Args:
        budget_bytes: New budget in bytes; must be a positive integer
            (e.g. ``256 * 1024**2`` for 256 MiB).

    Raises:
        ValueError: If ``budget_bytes`` is not a positive integer.
    """
    if not isinstance(budget_bytes, int) or isinstance(budget_bytes, bool) or budget_bytes <= 0:
        raise ValueError(f"memory budget must be a positive int of bytes, got {budget_bytes!r}")
    _batching._set_memory_budget(budget_bytes)


def get_memory_budget() -> int:
    """Return the active transient-memory budget in bytes.

    Returns:
        The budget set by the most recent :func:`set_memory_budget` call, or
        the built-in default (512 MiB) if it was never called.
    """
    return _batching.get_memory_budget()
