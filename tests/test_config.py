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


def test_enable_compilation_cache_sets_config(tmp_path):
    """enable_compilation_cache wires up the JAX persistent-cache config flags.

    ``jax.config.update`` raises on an unknown flag name, so a successful call
    also confirms the three flag names are valid; the value read-backs confirm
    each was applied.
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


def test_enable_compilation_cache_expands_user(tmp_path, monkeypatch):
    """A leading ``~`` in the path is expanded to the home directory."""
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


class TestMemoryBudgetUnits:
    """T4: the unit keyword on set_memory_budget / get_memory_budget."""

    def test_each_unit_resolves_to_its_exact_byte_count(self, _restore_memory_budget):
        """T4: every accepted unit is binary, and its byte value is exact.

        Oracle: the definition itself. ``kb``/``mb``/``gb``/``tb`` are powers of
        1024, and the ``*ib`` spellings are exact synonyms.
        """
        for unit, factor in (("b", 1), ("kb", 1024), ("kib", 1024)):
            jaxgsa.config.set_memory_budget(3, unit=unit)
            assert jaxgsa.config.get_memory_budget() == 3 * factor

    def test_unit_ignores_case_and_whitespace(self, _restore_memory_budget):
        """T4: unit names are harmonised by stripping and lowercasing."""
        jaxgsa.config.set_memory_budget(7, unit="\tMiB\n")
        assert jaxgsa.config.get_memory_budget() == 7 * 1024**2

    def test_unknown_unit_raises_on_set_and_get(self):
        """T4: an unknown unit names the input it rejected."""
        with pytest.raises(ValueError, match="unknown memory unit") as exc:
            jaxgsa.config.set_memory_budget(1, unit="megabytes")
        assert repr("megabytes") in str(exc.value)
        with pytest.raises(ValueError, match="unknown memory unit"):
            jaxgsa.config.get_memory_budget(unit="megabytes")


class TestMemoryBudgetValue:
    """T4: value validation and conversion in set_memory_budget."""

    def test_default_unit_is_mb(self, _restore_memory_budget):
        """T4: a bare number is read as megabytes."""
        jaxgsa.config.set_memory_budget(256)
        assert jaxgsa.config.get_memory_budget() == 256 * 1024**2

    def test_default_budget_is_unchanged_in_bytes(self, _restore_memory_budget):
        """T4: the shipped default is still 512 MiB, and 512 restates it.

        The unit change must move only how you write the budget, never its
        value. ``set_memory_budget(512)`` therefore has to land exactly on the
        untouched default.

        536870912 is written out rather than as ``512 * 1024**2`` so that the
        assertion cannot follow the source if the multiplier ever changes.
        """
        assert jaxgsa.config.get_memory_budget() == 536870912

        # The claim the unit change rests on: the new spelling of the default
        # resolves to the old one. Asserting the module constant instead would
        # only re-check the line above, since the getter returns it.
        jaxgsa.config.set_memory_budget(512)
        assert jaxgsa.config.get_memory_budget() == 536870912

    def test_float_value_accepted(self, _restore_memory_budget):
        """T4: a float value is legal, so 1.5 GB can be expressed directly."""
        jaxgsa.config.set_memory_budget(1.5, unit="gb")
        assert jaxgsa.config.get_memory_budget() == 1536 * 1024**2

    def test_float_bytes_are_rounded_to_nearest(self, _restore_memory_budget):
        """T4: a fractional byte count rounds, and the result stays an int."""
        jaxgsa.config.set_memory_budget(1.6, unit="b")
        budget = jaxgsa.config.get_memory_budget()
        assert budget == 2
        assert isinstance(budget, int)

    def test_value_rounding_below_one_byte_raises(self):
        """T4: a value that rounds away to zero bytes is not a usable budget."""
        with pytest.raises(ValueError, match="at least 1 byte"):
            jaxgsa.config.set_memory_budget(0.4, unit="b")

    def test_invalid_values_rejected(self):
        """T4: bool is an int subclass in Python, and must not slip through."""
        for bad in (True, False, 0, -0.5, math.inf, -math.inf, math.nan, "512MiB", None, [512]):
            with pytest.raises(ValueError, match="positive, finite number"):
                jaxgsa.config.set_memory_budget(bad)


class TestBytesShapedGuard:
    """T4: the guard against pre-0.9 byte counts passed without a unit."""

    def test_guard_fires_without_unit(self):
        """T4: a unit-less bytes-shaped value raises instead of meaning TB.

        ``512 * 1024**2`` used to mean 512 MiB. Under the MB default it would
        mean 512 TB. The guard turns that silent reinterpretation into an
        error.
        """
        for legacy in (512 * 1024**2, 1024**2, 2**30, 1e9):
            with pytest.raises(ValueError, match="megabytes by default") as exc:
                jaxgsa.config.set_memory_budget(legacy)
            message = str(exc.value)
            assert "unit='b'" in message
            assert f"set_memory_budget({legacy / 1024**2:g})" in message

    def test_guard_does_not_reject_plausible_mb_figures(self, _restore_memory_budget):
        """T4: a large but real MB budget (64000 MB = 62.5 GiB) still works."""
        for plausible in (512, 64000, 1024**2 - 1):
            jaxgsa.config.set_memory_budget(plausible)
            assert jaxgsa.config.get_memory_budget() == plausible * 1024**2


class TestGetMemoryBudget:
    """T4: the getter keeps returning bytes by default."""

    def test_default_and_explicit_units_return_expected_values(self, _restore_memory_budget):
        """T4: the no-argument return value is unchanged by this work."""
        jaxgsa.config.set_memory_budget(256)
        budget = jaxgsa.config.get_memory_budget()
        assert budget == 268435456
        assert isinstance(budget, int)
        jaxgsa.config.set_memory_budget(1536, unit="mb")
        assert jaxgsa.config.get_memory_budget(unit="mb") == 1536.0
        assert jaxgsa.config.get_memory_budget(unit="gb") == 1.5
