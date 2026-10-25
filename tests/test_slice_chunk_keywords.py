"""The chunking keywords added in 1.3.1: kucherenko, morris and shapley.

Each keyword only bounds peak memory, so the indices must agree for
``slice_chunk_size=1``, an odd width that leaves a ragged last chunk, and
``None``. Each keyword is validated like the others (an integer of at least
1), and the verbose summary prints the resolved width.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import jaxgsa
from jaxgsa.kucherenko import KucherenkoResult, KucherenkoSamples
from jaxgsa.morris import MorrisResult, MorrisSamples
from jaxgsa.shapley import ShapleyResult

PROBLEM = jaxgsa.Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)})
T, K = 3, 5  # 15 output slices; a width of 4 leaves a last chunk of 3
WIDTHS: list[int | None] = [1, 4, None]
RTOL, ATOL = 1e-5, 1e-6


def _close(actual: Any, desired: Any, rtol: float = RTOL, atol: float = ATOL) -> None:
    """Compare two index arrays (or two ``None``) to float tolerance."""
    assert (actual is None) == (desired is None)
    if actual is not None:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(desired), rtol=rtol, atol=atol)


def _outputs(X: np.ndarray) -> Array:
    """A (N, T, K) output with a different mix of the inputs in every slice."""
    cols = [
        [(t + 1) * X[:, 0] + k * X[:, 1] ** 2 + X[:, 0] * X[:, 2] for k in range(K)]
        for t in range(T)
    ]
    return jnp.asarray(np.stack([np.stack(c, axis=-1) for c in cols], axis=1))


def _kucherenko(samples: KucherenkoSamples, Y: Array, **kw: Any) -> KucherenkoResult:
    kw.setdefault("verbose", False)
    res = jaxgsa.kucherenko.analyze(samples, Y, **kw)
    assert isinstance(res, KucherenkoResult)
    return res


def _morris(samples: MorrisSamples, Y: Array, **kw: Any) -> MorrisResult:
    kw.setdefault("verbose", False)
    res = jaxgsa.morris.analyze(samples, Y, **kw)
    assert isinstance(res, MorrisResult)
    return res


def _shapley(X: Array, Y: Array, **kw: Any) -> ShapleyResult:
    kw.setdefault("verbose", False)
    res = jaxgsa.shapley.analyze(PROBLEM, X, Y, **kw)
    assert isinstance(res, ShapleyResult)
    return res


# --------------------------------------------------------------------------
# kucherenko.analyze
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kucherenko_case() -> tuple[KucherenkoSamples, Array]:
    samples = jaxgsa.kucherenko.sample(PROBLEM, base_n=128, seed=0, verbose=False)
    return samples, _outputs(np.asarray(samples.samples))


@pytest.mark.parametrize("n_bootstrap", [0, 20])
def test_kucherenko_slice_chunk_size_agrees(kucherenko_case, n_bootstrap):
    samples, Y = kucherenko_case
    results = [
        _kucherenko(samples, Y, slice_chunk_size=w, n_bootstrap=n_bootstrap, key=jax.random.key(0))
        for w in WIDTHS
    ]
    ref = results[-1]
    for res in results[:-1]:
        _close(res.S1, ref.S1)
        _close(res.ST, ref.ST)
        _close(res.variance, ref.variance)
        _close(res.S1_conf, ref.S1_conf)
        _close(res.ST_conf, ref.ST_conf)


def test_kucherenko_slice_chunk_size_subsets_all():
    samples = jaxgsa.kucherenko.sample(PROBLEM, base_n=64, subsets="all", seed=0, verbose=False)
    Y = _outputs(np.asarray(samples.samples))
    ref = _kucherenko(samples, Y)
    res = _kucherenko(samples, Y, slice_chunk_size=4)
    _close(res.closed, ref.closed)
    _close(res.subset_total, ref.subset_total)


@pytest.mark.parametrize("bad", [0, -1, 2.0, True])
def test_kucherenko_slice_chunk_size_rejected(kucherenko_case, bad):
    samples, Y = kucherenko_case
    with pytest.raises(ValueError, match="slice_chunk_size must be an integer >= 1"):
        _kucherenko(samples, Y, slice_chunk_size=bad)


@pytest.mark.verbose_output
def test_kucherenko_verbose_prints_width(kucherenko_case, capsys):
    samples, Y = kucherenko_case
    _kucherenko(samples, Y, slice_chunk_size=4, verbose=True)
    assert "slice_chunk_size: 4 (user-set)" in capsys.readouterr().out
    _kucherenko(samples, Y, slice_chunk_size=10**6, verbose=True)
    assert "slice_chunk_size: 15 (user-set)" in capsys.readouterr().out
    _kucherenko(samples, Y, verbose=True)
    assert "slice_chunk_size: 15 (resolved from the memory budget)" in capsys.readouterr().out


# --------------------------------------------------------------------------
# morris.analyze / morris.indices
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def morris_case() -> tuple[MorrisSamples, Array]:
    samples = jaxgsa.morris.sample(PROBLEM, n_trajectories=20, seed=0, verbose=False)
    return samples, _outputs(np.asarray(samples.samples))


@pytest.mark.parametrize("standardize", [False, True])
def test_morris_slice_chunk_size_agrees(morris_case, standardize):
    samples, Y = morris_case
    results = [
        _morris(
            samples,
            Y,
            slice_chunk_size=w,
            n_bootstrap=30,
            key=jax.random.key(0),
            keep_replicates=True,
            standardize_outputs=standardize,
        )
        for w in WIDTHS
    ]
    ref = results[-1]
    for res in results[:-1]:
        for name in ("mu", "mu_star", "sigma", "mu_conf", "mu_star_conf", "sigma_conf"):
            _close(getattr(res, name), getattr(ref, name))
        assert res.ci is not None and ref.ci is not None
        assert res.ci.replicates is not None and ref.ci.replicates is not None
        for name, draws in ref.ci.replicates.items():
            _close(res.ci.replicates[name], draws)


@pytest.mark.parametrize("width", [1, 4])
def test_morris_indices_slice_chunk_size_agrees(morris_case, width):
    samples, Y = morris_case
    ref = _morris(samples, Y, standardize_outputs=True)

    def core(y: Array) -> jaxgsa.morris.MorrisIndices:
        return jaxgsa.morris.indices(samples, y, standardize_outputs=True, slice_chunk_size=width)

    for got in (core(Y), jax.jit(core)(Y)):
        _close(got.mu, ref.mu)
        _close(got.mu_star, ref.mu_star)
        _close(got.sigma, ref.sigma)


def test_morris_slice_chunk_size_under_skip_and_drop(morris_case):
    samples, Y = morris_case
    ref = _morris(samples, Y, on_invalid="skip")
    res = _morris(samples, Y, on_invalid="skip", slice_chunk_size=4)
    _close(res.mu_star, ref.mu_star)
    Y_bad = Y.at[5, 0, 0].set(jnp.nan)
    ref = _morris(samples, Y_bad, on_invalid="drop")
    res = _morris(samples, Y_bad, on_invalid="drop", slice_chunk_size=4)
    _close(res.sigma, ref.sigma)


@pytest.mark.parametrize("bad", [0, -2, 1.5])
def test_morris_slice_chunk_size_rejected(morris_case, bad):
    samples, Y = morris_case
    with pytest.raises(ValueError, match="slice_chunk_size must be an integer >= 1"):
        _morris(samples, Y, slice_chunk_size=bad)
    with pytest.raises(ValueError, match="slice_chunk_size must be an integer >= 1"):
        jaxgsa.morris.indices(samples, Y, slice_chunk_size=bad)


@pytest.mark.verbose_output
def test_morris_verbose_prints_width(morris_case, capsys):
    samples, Y = morris_case
    _morris(samples, Y, slice_chunk_size=4, verbose=True)
    assert "slice_chunk_size: 4 (user-set)" in capsys.readouterr().out
    _morris(samples, Y, verbose=True)
    assert "slice_chunk_size: 15 (resolved from the memory budget)" in capsys.readouterr().out


# --------------------------------------------------------------------------
# shapley.analyze / shapley.indices
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shapley_case() -> tuple[Array, Array]:
    X = np.random.default_rng(0).uniform(size=(320, 3))
    return jnp.asarray(X), _outputs(X)


@pytest.mark.parametrize(
    ("backend", "chunk_kw"),
    [
        ("pce", {"batch_size": 1}),
        ("pce", {"batch_size": 37}),
        ("hdmr", {"slice_chunk_size": 1}),
        ("hdmr", {"slice_chunk_size": 4, "batch_size": 37}),
    ],
)
def test_shapley_chunk_keywords_agree(shapley_case, backend, chunk_kw):
    X, Y = shapley_case
    ref = _shapley(X, Y, backend=backend, order=2)
    res = _shapley(X, Y, backend=backend, order=2, **chunk_kw)
    # The streamed fits accumulate the Gram matrix batch by batch, so allow
    # a little more than round-off.
    _close(res.Sh, ref.Sh, rtol=1e-4, atol=1e-5)
    _close(res.ST, ref.ST, rtol=1e-4, atol=1e-5)
    out = jaxgsa.shapley.indices(PROBLEM, X, Y, backend=backend, order=2, **chunk_kw)
    _close(out.Sh, ref.Sh, rtol=1e-4, atol=1e-5)


def test_shapley_slice_chunk_size_refused_for_pce(shapley_case):
    X, Y = shapley_case
    with pytest.raises(ValueError, match="slice_chunk_size applies to backend='hdmr' only"):
        _shapley(X, Y, slice_chunk_size=2)
    with pytest.raises(ValueError, match="slice_chunk_size applies to backend='hdmr' only"):
        jaxgsa.shapley.indices(PROBLEM, X, Y, slice_chunk_size=2)


@pytest.mark.parametrize("name", ["batch_size", "slice_chunk_size"])
@pytest.mark.parametrize("bad", [0, 2.0])
def test_shapley_chunk_keywords_rejected(shapley_case, name, bad):
    X, Y = shapley_case
    match = f"{name} must be an integer >= 1"
    with pytest.raises(ValueError, match=match):
        _shapley(X, Y, backend="hdmr", **{name: bad})
    with pytest.raises(ValueError, match=match):
        jaxgsa.shapley.indices(PROBLEM, X, Y, backend="hdmr", **{name: bad})


@pytest.mark.verbose_output
def test_shapley_verbose_prints_chunk_keywords(shapley_case, capsys):
    X, Y = shapley_case
    _shapley(X, Y, backend="hdmr", order=1, slice_chunk_size=4, verbose=True)
    out = capsys.readouterr().out
    assert "slice_chunk_size: 4 (user-set)" in out
    assert "batch_size: auto (resolved from the memory budget)" in out
    _shapley(X, Y, order=2, batch_size=64, verbose=True)
    out = capsys.readouterr().out
    assert "batch_size: 64 (user-set)" in out
    assert "slice_chunk_size" not in out


def test_shapley_hdmr_bootstrap_runs_on_host_like_hdmr(monkeypatch):
    """A streamed hdmr-backend bootstrap uses the host loop and hdmr's draws.

    At order 1 every HDMR term is one input, so a replicate's Shapley effect
    is its ``Sa`` divided by the sum of ``Sa``. The two analyses draw the
    same rows from the same key, so the draws must agree replicate by
    replicate.
    """
    import jaxgsa._core.bootstrap as boot
    import jaxgsa.shapley._analyze as shapley_mod

    calls: list[str] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append("host")
        return boot.bootstrap_draws_host(*args, **kwargs)

    monkeypatch.setattr(shapley_mod, "bootstrap_draws_host", spy)
    rng = np.random.default_rng(1)
    X = jnp.asarray(rng.uniform(size=(400, 3)))
    Y = X[:, 0] + 2.0 * X[:, 1] ** 2 + 0.5 * X[:, 2]
    # batch_size below N streams the fit, which hdmr runs on the host.
    kw: dict[str, Any] = {
        "order": 1,
        "batch_size": 128,
        "n_bootstrap": 4,
        "key": jax.random.key(3),
        "keep_replicates": True,
        "verbose": False,
    }
    sh = _shapley(X, Y, backend="hdmr", **kw)
    hd = jaxgsa.hdmr.analyze(PROBLEM, X, Y, **kw)
    assert isinstance(hd, jaxgsa.hdmr.HDMRResult)
    assert calls == ["host"]
    assert sh.ci is not None and sh.ci.replicates is not None
    assert hd.ci is not None and hd.ci.replicates is not None
    Sa = np.asarray(hd.ci.replicates["Sa"])
    _close(sh.ci.replicates["Sh"], Sa / Sa.sum(axis=-1, keepdims=True), rtol=1e-5, atol=1e-7)


def test_shapley_pce_bootstrap_stays_jitted(shapley_case, monkeypatch):
    import jaxgsa.shapley._analyze as shapley_mod

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the PCE backend must not use the host loop")

    monkeypatch.setattr(shapley_mod, "bootstrap_draws_host", fail)
    X, Y = shapley_case
    res = _shapley(X, Y, order=2, batch_size=64, n_bootstrap=3, key=jax.random.key(0))
    assert res.Sh_conf is not None


# --------------------------------------------------------------------------
# rank.analyze / rank.indices
# --------------------------------------------------------------------------


def _rank(X: Array, Y: Array, **kw: Any) -> jaxgsa.rank.RankResult:
    kw.setdefault("verbose", False)
    res = jaxgsa.rank.analyze(PROBLEM, X, Y, **kw)
    assert isinstance(res, jaxgsa.rank.RankResult)
    return res


def test_rank_slice_chunk_size_agrees(shapley_case):
    X, Y = shapley_case
    results = [
        _rank(
            X,
            Y,
            slice_chunk_size=w,
            n_bootstrap=20,
            key=jax.random.key(0),
            keep_replicates=True,
        )
        for w in WIDTHS
    ]
    ref = results[-1]
    for res in results[:-1]:
        _close(res.S1, ref.S1)
        _close(res.S1_conf, ref.S1_conf)
        assert res.ci is not None and ref.ci is not None
        assert res.ci.replicates is not None and ref.ci.replicates is not None
        _close(res.ci.replicates["S1"], ref.ci.replicates["S1"])


@pytest.mark.parametrize("width", [1, 4])
def test_rank_indices_slice_chunk_size_agrees(shapley_case, width):
    X, Y = shapley_case
    ref = _rank(X, Y)

    def core(y: Array) -> Array:
        return jaxgsa.rank.indices(PROBLEM, X, y, slice_chunk_size=width).S1

    _close(core(Y), ref.S1)
    _close(jax.jit(core)(Y), ref.S1)
    # The gradient with respect to Y goes through the chunk map unchanged.
    grad_chunked = jax.grad(lambda y: core(y).sum())(Y)
    grad_whole = jax.grad(lambda y: jaxgsa.rank.indices(PROBLEM, X, y).S1.sum())(Y)
    _close(grad_chunked, grad_whole)


@pytest.mark.parametrize("bad", [0, 3.0])
def test_rank_slice_chunk_size_rejected(shapley_case, bad):
    X, Y = shapley_case
    with pytest.raises(ValueError, match="slice_chunk_size must be an integer >= 1"):
        _rank(X, Y, slice_chunk_size=bad)
    with pytest.raises(ValueError, match="slice_chunk_size must be an integer >= 1"):
        jaxgsa.rank.indices(PROBLEM, X, Y, slice_chunk_size=bad)


@pytest.mark.verbose_output
def test_rank_verbose_prints_width(shapley_case, capsys):
    X, Y = shapley_case
    _rank(X, Y, slice_chunk_size=4, verbose=True)
    assert "slice_chunk_size: 4 (user-set)" in capsys.readouterr().out
    _rank(X, Y, verbose=True)
    assert "slice_chunk_size: 15 (resolved from the memory budget)" in capsys.readouterr().out


@pytest.mark.parametrize("width", [1, 4])
def test_morris_indices_gradient_with_chunks(morris_case, width):
    samples, Y = morris_case

    def loss(y: Array, w: int | None) -> Array:
        return jaxgsa.morris.indices(samples, y, slice_chunk_size=w).mu_star.sum()

    _close(jax.grad(loss)(Y, width), jax.grad(loss)(Y, None))
