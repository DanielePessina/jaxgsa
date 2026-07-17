"""Opt-in JAX runtime configuration helpers for gsax.

These helpers wrap ``jax.config`` settings worth tuning for sensitivity-analysis
workloads. They are deliberately never applied on import, because they mutate
global JAX state that the host application may also depend on.
"""

from pathlib import Path

import jax

__all__ = ["enable_compilation_cache"]


def enable_compilation_cache(
    path: str | Path,
    *,
    min_compile_time_secs: float = 1.0,
    min_entry_size_bytes: int = 0,
) -> str:
    """Enable JAX's persistent, on-disk compilation cache.

    JAX caches compiled XLA executables in memory for the lifetime of a process.
    Enabling the *persistent* cache additionally stores them on disk, so repeated
    runs of the same analysis across process restarts (parameter sweeps, CI jobs,
    HPC batches) skip the cold XLA compile. This is an opt-in convenience: call it
    once, before the first ``analyze`` call (e.g. ``gsax.sobol.analyze``), so the
    cache is active when the first compilation happens.

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
