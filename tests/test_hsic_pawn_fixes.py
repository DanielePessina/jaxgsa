"""Tests for the 1.1.2 HSIC fixes: NaN p-values and the near-collapse median warning."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa.hsic as hsic
import jaxgsa.hsic._analyze as hsic_impl
from jaxgsa._core.batching import _set_memory_budget as _set_budget
from jaxgsa._core.batching import get_memory_budget
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.hsic import HSICResult
from jaxgsa.problem import Problem

PROBLEM = Problem(names=("a", "b"), bounds=((0.0, 1.0), (0.0, 1.0)))


def _x(n: int = 300) -> np.ndarray:
    return np.random.default_rng(0).random((n, 2)).astype(np.float32)


def _collapsed(rec: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in rec if "median heuristic failed" in str(w.message)]


def test_zero_variance_output_gives_nan_p_values() -> None:
    X = _x()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = hsic.analyze(
            PROBLEM,
            jnp.asarray(X),
            jnp.ones(len(X), jnp.float32),
            key=0,
            verbose=False,
            n_perms=20,
        )
    assert isinstance(r, HSICResult)
    assert np.all(np.isnan(np.asarray(r.R2)))
    assert np.all(np.isnan(np.asarray(r.T)))
    assert np.all(np.isnan(np.asarray(r.p_values)))


def test_collapsed_median_warns_for_near_atom_output() -> None:
    X = _x()
    y = (np.maximum(0.0, X[:, 0] - 0.75) + 1e-6 * X[:, 1]).astype(np.float32)
    with pytest.warns(JaxgsaWarning, match="median heuristic failed") as rec:
        hsic.analyze(PROBLEM, jnp.asarray(X), jnp.asarray(y), key=0, verbose=False, n_perms=10)
    hit = _collapsed(list(rec))
    assert len(hit) == 1
    assert hit[0].filename == __file__


def test_collapsed_median_warns_for_near_atom_input() -> None:
    X = _x()
    X[:, 1] = np.maximum(0.0, X[:, 1] - 0.75) + 1e-6 * X[:, 0]
    y = (X[:, 0] + X[:, 1]).astype(np.float32)
    with pytest.warns(JaxgsaWarning, match="parameter 'b'"):
        hsic.analyze(PROBLEM, jnp.asarray(X), jnp.asarray(y), key=0, verbose=False, n_perms=10)


def test_exact_atoms_use_nonzero_median_and_do_not_warn() -> None:
    from jaxgsa.hsic._analyze import _median_bandwidth_sq

    X = _x()
    y = np.maximum(0.0, X[:, 0] - 0.75).astype(np.float32)
    assert np.mean(y == 0) > 0.7
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = float(_median_bandwidth_sq(jnp.asarray(y)))
        hsic.analyze(PROBLEM, jnp.asarray(X), jnp.asarray(y), key=0, verbose=False, n_perms=10)
    assert not _collapsed(rec)
    d2 = (y[:, None].astype(np.float64) - y[None, :]) ** 2
    expected = np.median(d2[np.triu_indices(len(y), 1)][d2[np.triu_indices(len(y), 1)] > 0])
    np.testing.assert_allclose(got, expected, rtol=1e-5)
    assert got > 1e-6


def test_nonzero_median_is_unchanged_on_tie_free_data() -> None:
    from jaxgsa.hsic._analyze import _median_bandwidth_sq

    x = jnp.asarray(np.random.default_rng(1).random(257).astype(np.float32))
    d2 = (x[:, None] - x[None, :]) ** 2
    n = x.shape[0]
    ref = jnp.median(d2[jnp.triu_indices(n, 1)])
    np.testing.assert_allclose(float(_median_bandwidth_sq(x)), float(ref), rtol=1e-6)


def test_no_collapsed_median_warning_on_ordinary_data() -> None:
    X = _x()
    y = np.sin(6 * X[:, 0]).astype(np.float32)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        hsic.analyze(PROBLEM, jnp.asarray(X), jnp.asarray(y), key=0, verbose=False, n_perms=10)
    assert not _collapsed(rec)


def test_no_collapsed_median_warning_for_binary_output() -> None:
    X = _x()
    y = (X[:, 0] > 0.3).astype(np.float32)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        hsic.analyze(PROBLEM, jnp.asarray(X), jnp.asarray(y), key=0, verbose=False, n_perms=10)
    assert not _collapsed(rec)


def test_indices_still_jits_and_vmaps_on_collapsed_data() -> None:
    X = jnp.asarray(_x(100))
    y = jnp.asarray(np.maximum(0.0, np.asarray(X[:, 0]) - 0.75).astype(np.float32))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = jax.jit(lambda a, b: hsic.indices(PROBLEM, a, b, key=0, n_perms=5).R2)(X, y)
        batch = jax.vmap(lambda b: hsic.indices(PROBLEM, X, b, key=0, n_perms=5).R2)(
            jnp.stack([y, y + 1.0])
        )
    assert out.shape == (2,)
    assert batch.shape == (2, 2)
    assert not _collapsed(rec)


def test_binary_output_gets_a_real_bandwidth() -> None:
    from jaxgsa.hsic._analyze import _median_bandwidth_sq

    y = jnp.asarray((np.random.default_rng(2).random(200) > 0.3).astype(np.float32))
    assert float(_median_bandwidth_sq(y)) == pytest.approx(1.0)


def test_hsic_setup_and_slice_kernel_temp_memory_is_bounded() -> None:
    """Compile-only: transient memory stays a small multiple of one (N, N) kernel.

    Before the fix the setup held ~D extra (N, N) matrices (list-then-stack and
    prefix/suffix lists) and the per-slice kernel broadcast the output kernel to
    (D, N, N). Nothing is executed, so the check is cheap.
    """
    impl = hsic_impl

    n, d, t = 1024, 16, 8
    mat = n * n * 4
    sds = jax.ShapeDtypeStruct
    setup = jax.jit(impl._input_setup, static_argnames=("total", "bandwidth"))
    temp = (
        setup.lower(sds((n, d), jnp.float32), jax.random.key(0), total=t, bandwidth=1.0)
        .compile()
        .memory_analysis()
        .temp_size_in_bytes
    )
    assert temp <= 6 * mat

    old_budget = get_memory_budget()
    try:
        _set_budget(1)  # force the sequential path
        kern = impl._get_hsic_kernel(4, 1.0, impl._fits_vectorised(d, n, 4))
        temp = (
            kern.lower(
                sds((d, n, n), jnp.float32),
                sds((d, n, n), jnp.float32),
                sds((n, n), jnp.float32),
                sds((d,), jnp.float32),
                sds((n, t), jnp.float32),
                jax.random.split(jax.random.key(0), t),
            )
            .compile()
            .memory_analysis()
            .temp_size_in_bytes
        )
    finally:
        _set_budget(old_budget)
    assert temp <= 8 * mat


def test_hsic_budget_change_reaches_the_cached_kernel() -> None:
    """A new memory budget picks a new kernel variant, not a stale cached one."""
    n, d = 64, 3
    old_budget = get_memory_budget()
    try:
        _set_budget(1)
        small = hsic_impl._fits_vectorised(d, n, 4)
        _set_budget(1 << 40)
        large = hsic_impl._fits_vectorised(d, n, 4)
    finally:
        _set_budget(old_budget)
    assert (small, large) == (False, True)
    assert hsic_impl._get_hsic_kernel(4, 1.0, small) is not hsic_impl._get_hsic_kernel(
        4, 1.0, large
    )


def test_hsic_gradient_matches_listed_complements() -> None:
    """The custom VJP on the complement kernels equals autodiff of the list form."""
    Ks = jax.random.uniform(jax.random.key(3), (4, 12, 12))
    Ks = 0.5 * (Ks + Ks.transpose(0, 2, 1))

    def loss(f):
        def g(K):
            full, comps = f(K)
            return jnp.sum(full**2) + jnp.sum(jnp.sin(comps))

        return g

    got = jax.grad(loss(hsic_impl._complement_kernels))(Ks)
    want = jax.grad(loss(hsic_impl._complement_kernels_listed))(Ks)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-5)
