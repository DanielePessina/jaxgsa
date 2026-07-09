"""Tests for gsax runtime configuration helpers."""

import jax
import pytest

import gsax

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
    returned = gsax.enable_compilation_cache(
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
    returned = gsax.enable_compilation_cache("~/gsax-cache")
    assert returned == str(tmp_path / "gsax-cache")
