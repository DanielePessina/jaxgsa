"""Tests for jaxgsa runtime configuration helpers."""

import math

import jax
import pytest

import jaxgsa
from jaxgsa._core import batching as _batching

# JAX config is process-global; enable_compilation_cache mutates it. Snapshot the
# three flags this module touches and restore them after each test so the cache
# is not left enabled at a deleted tmp_path (which would leak into other tests).
_CACHE_FLAGS = (
    "jax_compilation_cache_dir",
    "jax_persistent_cache_min_compile_time_secs",
    "jax_persistent_cache_min_entry_size_bytes",
)


@pytest.fixture(autouse=True)
def _restore_jax_cache_config():
    saved = {flag: getattr(jax.config, flag) for flag in _CACHE_FLAGS}
    yield
    for flag, value in saved.items():
        jax.config.update(flag, value)


def test_enable_compilation_cache_sets_config_and_expands_user(tmp_path, monkeypatch):
    """enable_compilation_cache wires up the JAX persistent-cache config flags.

    ``jax.config.update`` raises on an unknown flag name, so a successful call
    also confirms the three flag names are valid; the value read-backs confirm
    each was applied. A leading ``~`` in the path is expanded to the home
    directory.
    """
    returned = jaxgsa.config.enable_compilation_cache(
        tmp_path, min_compile_time_secs=2.5, min_entry_size_bytes=128
    )
    assert returned == str(tmp_path)
    # These flags are registered dynamically on jax.config, so read them back
    # with getattr: attribute access is the documented reader (jax.config.read
    # rejects contextmanager-backed flags), and getattr keeps it type-checkable.
    assert getattr(jax.config, "jax_compilation_cache_dir") == str(tmp_path)
    assert getattr(jax.config, "jax_persistent_cache_min_compile_time_secs") == 2.5
    assert getattr(jax.config, "jax_persistent_cache_min_entry_size_bytes") == 128

    monkeypatch.setenv("HOME", str(tmp_path))
    returned = jaxgsa.config.enable_compilation_cache("~/jaxgsa-cache")
    assert returned == str(tmp_path / "jaxgsa-cache")


@pytest.fixture
def _restore_memory_budget():
    """Restore the process-global memory budget after a test mutates it."""
    saved = _batching._memory_budget_bytes
    yield
    # Assign the module global directly: ``None`` (never set) must be
    # restorable as ``None``, which the int-typed internal setter cannot do.
    _batching._memory_budget_bytes = saved


def test_default_budget_is_512_mib_and_512_restates_it(_restore_memory_budget):
    """T4: the unit change moved only how the budget is written, never its value.

    536870912 is written out rather than as ``512 * 1024**2`` so that the
    assertion cannot follow the source if the multiplier ever changes. The
    claim the unit change rests on is that the new spelling of the default
    resolves to the old one.
    """
    assert jaxgsa.config.get_memory_budget() == 536870912
    jaxgsa.config.set_memory_budget(512)
    assert jaxgsa.config.get_memory_budget() == 536870912


def test_each_unit_resolves_to_its_exact_byte_count(_restore_memory_budget):
    """T4: every accepted unit is binary, and its byte value is exact.

    Oracle: the definition itself. ``kb``/``mb``/``gb``/``tb`` are powers of
    1024, and the ``*ib`` spellings are exact synonyms. Unit names are
    harmonised by stripping and lowercasing, and an unknown unit names the
    input it rejected.
    """
    for unit, factor in (("b", 1), ("kb", 1024), ("kib", 1024)):
        jaxgsa.config.set_memory_budget(3, unit=unit)
        assert jaxgsa.config.get_memory_budget() == 3 * factor
    jaxgsa.config.set_memory_budget(7, unit="\tMiB\n")
    assert jaxgsa.config.get_memory_budget() == 7 * 1024**2

    with pytest.raises(ValueError, match="unknown memory unit") as exc:
        jaxgsa.config.set_memory_budget(1, unit="megabytes")
    assert repr("megabytes") in str(exc.value)
    with pytest.raises(ValueError, match="unknown memory unit"):
        jaxgsa.config.get_memory_budget(unit="megabytes")


def test_values_convert_to_bytes(_restore_memory_budget):
    """T4: a bare number is megabytes, a float is legal, a fractional byte rounds."""
    jaxgsa.config.set_memory_budget(256)
    assert jaxgsa.config.get_memory_budget() == 256 * 1024**2

    jaxgsa.config.set_memory_budget(1.5, unit="gb")
    assert jaxgsa.config.get_memory_budget() == 1536 * 1024**2

    jaxgsa.config.set_memory_budget(1.6, unit="b")
    budget = jaxgsa.config.get_memory_budget()
    assert budget == 2
    assert isinstance(budget, int)


def test_invalid_values_are_rejected():
    """T4: bool is an int subclass in Python, and must not slip through."""
    for bad in (True, False, 0, -0.5, math.inf, -math.inf, math.nan, "512MiB", None, [512]):
        with pytest.raises(ValueError, match="positive, finite number"):
            jaxgsa.config.set_memory_budget(bad)
    # A value that rounds away to zero bytes is not a usable budget either.
    with pytest.raises(ValueError, match="at least 1 byte"):
        jaxgsa.config.set_memory_budget(0.4, unit="b")


def test_bytes_shaped_guard(_restore_memory_budget):
    """T4: pre-0.9 byte counts without a unit raise instead of meaning TB.

    ``512 * 1024**2`` used to mean 512 MiB. Under the MB default it would mean
    512 TB. The guard turns that silent reinterpretation into an error, and
    leaves large but real MB figures (64000 MB = 62.5 GiB) alone.
    """
    for legacy in (512 * 1024**2, 1024**2, 2**30, 1e9):
        with pytest.raises(ValueError, match="megabytes by default") as exc:
            jaxgsa.config.set_memory_budget(legacy)
        message = str(exc.value)
        assert "unit='b'" in message
        assert f"set_memory_budget({legacy / 1024**2:g})" in message

    for plausible in (512, 64000, 1024**2 - 1):
        jaxgsa.config.set_memory_budget(plausible)
        assert jaxgsa.config.get_memory_budget() == plausible * 1024**2


def test_get_memory_budget_returns_int_bytes_by_default(_restore_memory_budget):
    """T4: the no-argument return value is bytes, as an int."""
    jaxgsa.config.set_memory_budget(256)
    budget = jaxgsa.config.get_memory_budget()
    assert budget == 268435456
    assert isinstance(budget, int)
    jaxgsa.config.set_memory_budget(1536, unit="mb")
    assert jaxgsa.config.get_memory_budget(unit="mb") == 1536.0
    assert jaxgsa.config.get_memory_budget(unit="gb") == 1.5
