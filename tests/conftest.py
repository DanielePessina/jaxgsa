"""Shared test fixtures and helpers for jaxgsa tests."""

import contextlib
import os
import warnings

# Under pytest-xdist every worker is its own process. XLA's CPU backend would
# otherwise size its thread pool to every core in each of them, and the
# workers would fight over the same cores. XLA reads the pool size from NPROC,
# so one thread per worker lets `-n auto` scale. (The often-quoted
# `--xla_cpu_multi_thread_eigen=false` flag does not limit the pool; measured.)
# It must be set before JAX starts its backend, so it sits above the import.
if os.environ.get("PYTEST_XDIST_WORKER"):
    os.environ.setdefault("NPROC", "1")

# Keep compiled XLA programs on disk between runs, so a rerun loads them
# instead of compiling again. Most of a test's time is compilation, not
# arithmetic. Set JAX_COMPILATION_CACHE_DIR="" to turn it off.
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".pytest_cache", "jax")),
)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")

import jax  # noqa: E402
import pytest

from jaxgsa import JaxgsaWarning
from jaxgsa._core import verbose as _verbose

# Shared fixtures, one module per test group. Loaded as plugins so each
# session-scoped fixture is built once per process, whatever file asks for it.
pytest_plugins = [
    "_fixtures_variance",
    "_fixtures_dependence",
    "_fixtures_givendata",
    "_fixtures_crosscutting",
]


@pytest.fixture(autouse=True, scope="module")
def _free_compiled_programs():
    """Drop JAX's in-memory compiled programs after each test file.

    Every jitted kernel a test compiles stays in JAX's caches for the life
    of the worker. Over the whole suite this grew to about 10 GB on a CI
    runner (16 GB) and ran it out of memory with jax 0.11.0. A later file
    that needs a program again loads it from the disk cache above instead
    of compiling it.
    """
    yield
    jax.clear_caches()


@pytest.fixture(autouse=True)
def _silence_verbose(request, monkeypatch):
    """Silence the verbose seam for the whole suite.

    ``verbose`` defaults to ``True`` on every ``analyze()`` and ``sample()``,
    so an unsilenced suite would bury real test output under thousands of
    summaries. Every verbose line goes through one seam,
    :func:`jaxgsa._core.verbose.emit`, so replacing that one function is
    enough — and only the printing is skipped: the summary code still runs
    and builds its lines, so a bug in the formatting still fails whichever
    test triggers it. The tests that must see the real output carry the
    ``verbose_output`` marker (tests/test_verbose.py), and for them the seam
    stays live.
    """
    if request.node.get_closest_marker("verbose_output"):
        return
    monkeypatch.setattr(_verbose, "emit", lambda text: None)


@contextlib.contextmanager
def single_precision_warning():
    """Assert the float32 warning fires in float32, and is absent under x64.

    ``vkoga.analyze`` and ``hsic.analyze`` warn about single precision only
    when JAX is actually in float32, which is right. Written as a bare
    ``pytest.warns``, such a test fails with ``DID NOT WARN`` the moment the
    suite runs with ``JAX_ENABLE_X64=1`` — the test would be precision-blind,
    not the source. The flag is read here rather than at import so a caller
    that switches precision with a context manager gets the treatment that
    matches the precision actually in force.
    """
    if bool(getattr(jax.config, "jax_enable_x64", False)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
        assert not [w for w in caught if "single precision" in str(w.message)]
    else:
        with pytest.warns(JaxgsaWarning, match="single precision"):
            yield
