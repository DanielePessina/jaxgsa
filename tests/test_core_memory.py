"""Copy-free paths in ``_core``: Y promotion, identity expansion, design dedup.

These pin the behaviour behind the 1.3.1 memory fixes: the results must not
change, and the large copies must not come back.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

import jaxgsa
from jaxgsa._core import sampling as core_sampling
from jaxgsa._core.copula import latent_to_physical
from jaxgsa._core.sampling import _stable_unique_rows, _transform_samples
from jaxgsa._core.validation import YLayout, _prepare_Y, _promote_with_zero_stats

ON_CPU = jax.default_backend() == "cpu"


def _computed(shape: tuple[int, ...]) -> jax.Array:
    """A device array produced by a computation, as model outputs are."""
    return jax.block_until_ready(jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) * 2)


@pytest.mark.parametrize(
    ("shape", "promoted", "layout"),
    [
        ((7,), (7, 1, 1), YLayout.SCALAR),
        ((7, 3), (7, 1, 3), YLayout.MULTI_OUTPUT),
        ((7, 2, 3), (7, 2, 3), YLayout.TIME_SERIES),
    ],
)
def test_prepare_Y_values_and_layout(shape, promoted, layout):
    """The promotion keeps every value and records the caller's rank."""
    Y = _computed(shape)
    Y3, got_layout = _prepare_Y(Y)
    assert Y3.shape == promoted
    assert got_layout is layout
    assert Y3.dtype == Y.dtype
    np.testing.assert_array_equal(np.asarray(Y3).reshape(shape), np.asarray(Y))


@pytest.mark.skipif(not ON_CPU, reason="buffer sharing is a CPU-backend path")
@pytest.mark.parametrize("shape", [(9,), (9, 4)])
def test_prepare_Y_shares_the_buffer_on_cpu(shape):
    """A rank-1 or rank-2 Y is promoted without a second copy on CPU."""
    Y = _computed(shape)
    Y3, _ = _prepare_Y(Y)
    assert Y3.unsafe_buffer_pointer() == Y.unsafe_buffer_pointer()
    # The original stays usable and unchanged after the promoted view dies.
    del Y3
    np.testing.assert_array_equal(np.asarray(Y), np.asarray(_computed(shape)))


@pytest.mark.skipif(not ON_CPU, reason="buffer sharing is a CPU-backend path")
def test_promote_with_zero_stats_shares_the_buffer():
    """The fused promotion returns the shared view, and the right stats."""
    Y = _computed((16, 3)).at[:, 1].set(5.0)
    Y3, mask, count = _promote_with_zero_stats(Y)
    assert Y3 is not None
    assert Y3.unsafe_buffer_pointer() == Y.unsafe_buffer_pointer()
    np.testing.assert_array_equal(np.asarray(mask), [False, True, False])
    assert int(count) == 1
    rank3 = _promote_with_zero_stats(_computed((16, 2, 3)))
    assert rank3[0] is None


def test_prepare_Y_under_jit_and_grad():
    """Inside a trace the plain reshape runs, so transforms still work."""

    def total(Y):
        Y3, _ = _prepare_Y(Y)
        return jnp.sum(Y3**2)

    Y = _computed((5, 2))
    np.testing.assert_allclose(jax.jit(total)(Y), jnp.sum(Y**2), rtol=1e-6)
    np.testing.assert_allclose(jax.grad(total)(Y), 2 * Y, rtol=1e-6)


def test_expand_outputs_identity_skips_the_gather():
    """A design with no repeated row hands Y back instead of copying it."""
    problem = jaxgsa.Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0)})
    with pytest.warns(jaxgsa.JaxgsaWarning, match="declares no dependence"):
        design = jaxgsa.kucherenko.sample(problem, base_n=16, seed=0, verbose=False)
    assert design.n_runs == design.n_expanded
    Y = _computed((design.n_runs, 3))
    out = design.expand_outputs(Y)
    assert out is Y
    with pytest.raises(ValueError, match="must match samples.n_runs"):
        design.expand_outputs(Y[:-1])


def test_expand_outputs_non_identity_still_gathers():
    """A design with repeated rows still expands them."""
    problem = jaxgsa.Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0)})
    design = jaxgsa.sobol.sample(problem, base_n=16, scramble=False, verbose=False)
    assert design.n_runs < design.n_expanded
    Y = _computed((design.n_runs,))
    out = design.expand_outputs(Y)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(Y)[design.expanded_to_unique])


def _bytes_reference(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First-occurrence dedup on raw row bytes, written the obvious way."""
    samples = np.ascontiguousarray(samples)
    seen: dict[bytes, int] = {}
    labels = np.array([seen.setdefault(r.tobytes(), len(seen)) for r in samples], dtype=np.int64)
    first = sorted({lab: i for i, lab in reversed(list(enumerate(labels)))}.values())
    return samples[first], labels


def _dedup_cases() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    signed = np.zeros((6, 2))
    signed[1, 0] = -0.0
    signed[2:4] = np.nan
    return [
        rng.integers(0, 3, size=(400, 4)).astype(np.float64),
        rng.integers(0, 3, size=(400, 3)).astype(np.float32),  # 12-byte rows
        rng.integers(0, 3, size=(400, 6)).astype(np.float64)[:, ::2],
        rng.random((300, 5)),
        signed,
        np.ones((1, 3)),
    ]


@pytest.mark.parametrize("case", range(6))
def test_stable_unique_rows_matches_bytes_keys(case):
    """Fingerprint dedup equals an exact bytes-key dedup, row for row."""
    samples = _dedup_cases()[case]
    want_rows, want_map = _bytes_reference(samples)
    got_rows, got_map = _stable_unique_rows(samples)
    np.testing.assert_array_equal(got_map, want_map)
    assert got_rows.tobytes() == want_rows.tobytes()


def test_stable_unique_rows_survives_fingerprint_collisions(monkeypatch):
    """When every row collides, verification rejects it and keys decide."""
    monkeypatch.setattr(
        core_sampling, "_row_fingerprints", lambda w: np.zeros(w.shape[0], dtype=np.uint64)
    )
    for samples in _dedup_cases():
        want_rows, want_map = _bytes_reference(samples)
        got_rows, got_map = _stable_unique_rows(samples)
        np.testing.assert_array_equal(got_map, want_map)
        assert got_rows.tobytes() == want_rows.tobytes()


def test_stable_unique_rows_all_unique_returns_the_input():
    """No repeated row: the input is the unique array, not a copy of it."""
    samples = np.random.default_rng(1).random((50, 4))
    rows, mapping = _stable_unique_rows(samples)
    assert rows is samples
    np.testing.assert_array_equal(mapping, np.arange(50))


def test_transform_samples_in_place_matches():
    """Writing the transform over its own input gives the same values."""
    problem = jaxgsa.Problem.from_dict(
        {
            "u": (1.0, 3.0),
            "g": {"dist": "gaussian", "mean": 0.5, "variance": 2.0},
            "c": {"dist": "categorical", "probs": [0.3, 0.7]},
        }
    )
    unit = np.random.default_rng(2).random((200, 3))
    want = _transform_samples(problem, unit)
    buf = unit.copy()
    got = _transform_samples(problem, buf, out=buf)
    assert got is buf
    assert got.tobytes() == want.tobytes()


def test_latent_to_physical_matches_norm_cdf():
    """The ndtr path equals the norm.cdf formula it replaced, bit for bit."""
    problem = jaxgsa.Problem.from_dict(
        {
            "u": (0.0, 2.0),
            "ln": {"dist": "lognormal", "mu": 0.1, "sigma": 0.5},
        }
    )
    Z = np.random.default_rng(3).standard_normal((300, 2)) * 3
    Z_before = Z.copy()
    unit = np.clip(norm.cdf(Z), 1e-12, 1.0 - 1e-12)
    want = _transform_samples(problem, unit)
    got = latent_to_physical(problem, Z)
    assert got.tobytes() == want.tobytes()
    assert Z.tobytes() == Z_before.tobytes()  # the caller's latent rows are untouched


def test_prepare_Y_of_a_closed_over_array_inside_jit():
    """A concrete Y captured by a jitted function takes the plain reshape."""
    Y = _computed((6, 2))

    @jax.jit
    def total(scale):
        Y3, _ = _prepare_Y(Y)
        return jnp.sum(Y3) * scale

    np.testing.assert_allclose(total(2.0), 2 * jnp.sum(Y), rtol=1e-6)


def _jvp(f, x, t, *, jit):
    """``jax.jvp`` of ``f`` at ``x`` along ``t``, optionally under ``jit``."""

    def run(x, t):
        return jax.jvp(f, (x,), (t,))

    return (jax.jit(run) if jit else run)(x, t)


def _sum_sq_promoted(Y):
    Y3, _ = _prepare_Y(Y)
    return jnp.sum(Y3**2)


@pytest.mark.parametrize("shape", [(6,), (6, 2), (6, 2, 3)])
@pytest.mark.parametrize(
    "transform",
    ["jvp", "jit_jvp", "jacfwd", "jacrev", "vmap", "jit_vmap", "linearize", "jit_grad"],
)
def test_prepare_Y_through_every_tracer_kind(shape, transform):
    """Y as the traced argument: every transform sees the plain promotion."""
    Y = _computed(shape)
    f = _sum_sq_promoted
    want_grad = 2 * Y
    if transform in ("jvp", "jit_jvp"):
        primal, tangent = _jvp(f, Y, jnp.ones_like(Y), jit=transform == "jit_jvp")
        np.testing.assert_allclose(primal, jnp.sum(Y**2), rtol=1e-6)
        np.testing.assert_allclose(tangent, jnp.sum(want_grad), rtol=1e-6)
    elif transform in ("jacfwd", "jacrev", "jit_grad"):
        op = {"jacfwd": jax.jacfwd, "jacrev": jax.jacrev, "jit_grad": jax.grad}[transform]
        g = jax.jit(op(f)) if transform == "jit_grad" else op(f)
        np.testing.assert_allclose(g(Y), want_grad, rtol=1e-6)
    elif transform in ("vmap", "jit_vmap"):
        batch = jnp.stack([Y, 2 * Y])
        v = jax.vmap(f) if transform == "vmap" else jax.jit(jax.vmap(f))
        np.testing.assert_allclose(v(batch), jnp.array([1.0, 4.0]) * jnp.sum(Y**2), rtol=1e-6)
    else:
        primal, lin = jax.linearize(f, Y)
        np.testing.assert_allclose(primal, jnp.sum(Y**2), rtol=1e-6)
        np.testing.assert_allclose(lin(jnp.ones_like(Y)), jnp.sum(want_grad), rtol=1e-6)


@pytest.mark.parametrize("shape", [(6,), (6, 2)])
@pytest.mark.parametrize("transform", ["jvp", "jit_jvp", "jacfwd", "vmap", "grad"])
def test_prepare_Y_closed_over_under_every_tracer_kind(shape, transform):
    """A concrete Y captured inside a transform still promotes correctly."""
    Y = _computed(shape)

    def f(s):
        Y3, _ = _prepare_Y(Y)
        return jnp.sum(Y3) * s

    total = jnp.sum(Y)
    if transform in ("jvp", "jit_jvp"):
        out = _jvp(f, jnp.asarray(2.0), jnp.asarray(1.0), jit=transform == "jit_jvp")
        np.testing.assert_allclose(out[0], 2 * total, rtol=1e-6)
        np.testing.assert_allclose(out[1], total, rtol=1e-6)
    elif transform == "jacfwd":
        np.testing.assert_allclose(jax.jacfwd(f)(2.0), total, rtol=1e-6)
    elif transform == "grad":
        np.testing.assert_allclose(jax.grad(f)(2.0), total, rtol=1e-6)
    else:
        out = jax.vmap(f)(jnp.array([1.0, 3.0]))
        np.testing.assert_allclose(out, jnp.array([1.0, 3.0]) * total, rtol=1e-6)
