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


def test_hsic_pass_temp_memory_is_bounded_by_the_chunk() -> None:
    """Compile-only: each pass holds its chunk plus a few (N, N) matrices, whatever D is.

    Before the fix the estimator kept all D input kernels, their D complement
    products and the full product resident, ``(2D + 1)`` matrices, whatever
    the memory budget. Nothing is executed, so the check is cheap.
    """
    n, d, t, chunk = 1024, 16, 8, 2
    mat = n * n * 4
    sds = jax.ShapeDtypeStruct
    X, Y = sds((n, d), jnp.float32), sds((n, t), jnp.float32)
    xm, ym = sds((d,), jnp.float32), sds((t,), jnp.float32)
    keys = jax.random.split(jax.random.key(0), t)
    first = hsic_impl._first_order_pass.lower(
        X, Y, xm, ym, keys, n_perms=4, bandwidth=1.0, chunk=chunk
    )
    total = hsic_impl._total_pass.lower(X, Y, xm, ym, bandwidth=1.0, chunk=chunk)
    passes = ((first, hsic_impl._R_FIXED_MATRICES), (total, hsic_impl._T_FIXED_MATRICES))
    for lowered, fixed in passes:
        stats = lowered.compile().memory_analysis()
        assert stats is not None
        temp = stats.temp_size_in_bytes
        assert temp <= (chunk + fixed) * mat


def test_hsic_plan_follows_the_memory_budget() -> None:
    """The chunk widths come from the global budget, at least 1, at most D."""
    n, d = 64, 5
    mat = n * n * 4
    old_budget = get_memory_budget()
    try:
        _set_budget(1)
        small = hsic_impl._plan(d, n, 4)
        _set_budget((2 + hsic_impl._T_FIXED_MATRICES) * mat)
        mid = hsic_impl._plan(d, n, 4)
        _set_budget(1 << 40)
        large = hsic_impl._plan(d, n, 4)
    finally:
        _set_budget(old_budget)
    assert small == (1, 1)
    assert mid.total_chunk == 2
    assert mid.input_chunk == 2 + hsic_impl._T_FIXED_MATRICES - hsic_impl._R_FIXED_MATRICES
    assert large == (d, d)


@pytest.mark.parametrize("budget_mats", [1, 7, None], ids=["one-wide", "padded", "whole"])
def test_hsic_chunking_does_not_change_the_indices(budget_mats: int | None) -> None:
    """Every chunk width, padded or not, gives the indices of the unchunked run."""
    n, d = 96, 5
    X = jax.random.uniform(jax.random.key(4), (n, d))
    Y = jnp.stack([jnp.sin(6 * X[:, 0]) + X[:, 1] * X[:, 2], X[:, 3] ** 2], axis=1)
    problem = Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(d)})
    old_budget = get_memory_budget()
    try:
        _set_budget(1 << 40)
        want = hsic.indices(problem, X, Y, n_perms=30, key=jax.random.key(1))
        if budget_mats is not None:
            _set_budget(budget_mats * n * n * 4)
        got = hsic.indices(problem, X, Y, n_perms=30, key=jax.random.key(1))
    finally:
        _set_budget(old_budget)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("chunk", [1, 2, 3, 4])
def test_hsic_total_pass_matches_listed_products(chunk: int) -> None:
    """The streamed complement products equal the products formed in full."""
    n, d = 40, 4
    X = jax.random.uniform(jax.random.key(5), (n, d))
    Y = (X[:, 0] * X[:, 1] + X[:, 2])[:, None]
    xm, ym = hsic_impl._medians(X, Y)
    got_c, got_f = hsic_impl._total_pass(X, Y, xm, ym, bandwidth=1.0, chunk=chunk)

    raw = [hsic_impl._build_one_kernel(X[:, e], 1.0) for e in range(d)]
    aug = [1.0 + hsic_impl._center_kernel(K) for K in raw]
    L = hsic_impl._build_one_kernel(Y[:, 0], 1.0)
    full = aug[0] * aug[1] * aug[2] * aug[3]
    want_c = [
        hsic_impl._hsic_v(jnp.prod(jnp.stack([aug[e] for e in range(d) if e != k]), axis=0), L)
        for k in range(d)
    ]
    np.testing.assert_allclose(np.asarray(got_c[0]), np.asarray(want_c), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(got_f[0]), float(hsic_impl._hsic_v(full, L)), rtol=1e-4)


@pytest.mark.parametrize("chunk", [1, 3, 4])
def test_hsic_total_gradient_matches_listed_complements(chunk: int) -> None:
    """The gradient of the streamed total pass equals autodiff of explicit complements."""
    n, d = 32, 4
    X = jax.random.uniform(jax.random.key(6), (n, d))
    Y = jnp.stack([X[:, 0] * X[:, 1] + X[:, 2], jnp.sin(5 * X[:, 3])], axis=1)
    xm, ym = hsic_impl._medians(X, Y)
    w = jax.random.normal(jax.random.key(7), (2, d))

    def streamed(X, Y):
        hc, hf = hsic_impl._total_pass(X, Y, xm, ym, bandwidth=1.0, chunk=chunk)
        return jnp.sum(w * hc) + jnp.sum(jnp.cos(hf))

    def explicit(X, Y):
        aug = [
            1.0 + hsic_impl._center_kernel(hsic_impl._kernel_from_median(X[:, e], xm[e], 1.0))
            for e in range(d)
        ]
        Ls = [hsic_impl._kernel_from_median(Y[:, t], ym[t], 1.0) for t in range(2)]
        full = aug[0] * aug[1] * aug[2] * aug[3]
        hc = jnp.stack(
            [
                jnp.stack(
                    [
                        hsic_impl._hsic_v(
                            jnp.prod(jnp.stack([aug[e] for e in range(d) if e != k]), axis=0), L
                        )
                        for k in range(d)
                    ]
                )
                for L in Ls
            ]
        )
        hf = jnp.stack([hsic_impl._hsic_v(full, L) for L in Ls])
        return jnp.sum(w * hc) + jnp.sum(jnp.cos(hf))

    got = jax.grad(streamed, argnums=(0, 1))(X, Y)
    want = jax.grad(explicit, argnums=(0, 1))(X, Y)
    for g, e in zip(got, want, strict=True):
        np.testing.assert_allclose(np.asarray(g), np.asarray(e), rtol=1e-3, atol=1e-5)


def test_hsic_gradient_memory_does_not_grow_with_chunking() -> None:
    """Compile-only: reverse mode wrt X stays a few D matrices, whatever the budget.

    Differentiating the streamed loops saved the running product of every
    step of every chunk, so a budget of one matrix cost about 100 * D
    matrices in the backward pass (measured 2474 at D = 24). The custom VJP
    differentiates the stored-kernel form instead.
    """
    n, d = 128, 24
    problem = Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(d)})
    X = jax.random.uniform(jax.random.key(8), (n, d))
    # One (N, N) matrix in the working dtype (float64 when x64 is on).
    mat = n * n * X.dtype.itemsize
    Y = jnp.sin(6 * X[:, 0]) + X[:, 1] * X[:, 2]
    old_budget = get_memory_budget()
    temps = []
    try:
        for budget in (1, 1 << 40):
            _set_budget(budget)
            loss = jax.grad(
                lambda X: hsic.indices(problem, X, Y, n_perms=4, key=jax.random.key(0)).T.sum()
            )
            stats = jax.jit(loss).lower(X).compile().memory_analysis()
            assert stats is not None
            temps.append(stats.temp_size_in_bytes / mat)
    finally:
        _set_budget(old_budget)
    assert max(temps) <= 4 * d
