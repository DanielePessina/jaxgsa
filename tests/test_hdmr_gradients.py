"""The HDMR pure core: ``jaxgsa.hdmr.indices``.

The claim is that ``hdmr.indices`` computes the same numbers as
``hdmr.analyze`` and survives ``jit``, ``vmap`` and ``jit(jacfwd(...))``,
which ``analyze`` cannot.

Forward mode, not reverse. The backfitting solve stops early once the
coefficients settle, and expresses that as a ``lax.while_loop``, which JAX
refuses to differentiate in reverse. The test below pins that refusal, so the
day the loop becomes a fixed-trip ``scan`` the exemption fails and has to be
removed rather than quietly outliving the problem.

Tier T4 (internal consistency) throughout: what a transformability contract
can be checked against is the untransformed call.

Every case runs on a truncated Gaussian as well as a uniform problem. The
truncated marginal is the one that used to break, through the shared CDF
transform; a uniform-only test would not see it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import hdmr
from jaxgsa.benchmarks import ishigami
from jaxgsa.hdmr._engine import _compute_f_crits
from jaxgsa.problem import Problem

# HDMR refuses to fit below this, so every case here is at least this large.
N_ROWS = 400

TRUNCATED_PROBLEM = Problem.from_dict(
    {
        "x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0, "low": -1.5, "high": 2.0},
        "x2": (0.0, 1.0),
        "x3": {"dist": "gaussian", "mean": 1.0, "variance": 0.25},
    }
)


def _uniform_case(n: int = N_ROWS, seed: int = 0):
    """Return ``(problem, X, Y)`` for the Ishigami function on uniforms."""
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.uniform(-np.pi, np.pi, size=(n, 3)))
    return ishigami.PROBLEM, X, ishigami.evaluate(X)


def _truncated_case(n: int = N_ROWS, seed: int = 1):
    """Return ``(problem, X, Y)`` for a smooth model on a truncated Gaussian."""
    rng = np.random.default_rng(seed)
    X = jnp.asarray(
        np.column_stack(
            [
                rng.uniform(-1.4, 1.9, size=n),
                rng.uniform(0.0, 1.0, size=n),
                rng.normal(1.0, 0.5, size=n),
            ]
        )
    )
    Y = X[:, 0] ** 2 + 2.0 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]
    return TRUNCATED_PROBLEM, X, Y


CASES = {"uniform": _uniform_case, "truncated_gaussian": _truncated_case}


@pytest.fixture(params=sorted(CASES), ids=sorted(CASES))
def case(request):
    """One ``(problem, X, Y)`` triple per marginal family under test."""
    return CASES[request.param]()


# ---------------------------------------------------------------------------
# 1. The core returns what analyze reports
# ---------------------------------------------------------------------------


def test_indices_matches_analyze(case):
    """Tier T4: ``indices`` returns exactly what ``analyze`` reports."""
    problem, X, Y = case
    result = hdmr.analyze(problem, X, Y, maxorder=2, maxiter=50)
    Sa, Sb, S, ST = hdmr.indices(problem, X, Y, maxorder=2, maxiter=50)

    np.testing.assert_array_equal(np.asarray(Sa), np.asarray(result.Sa))
    np.testing.assert_array_equal(np.asarray(Sb), np.asarray(result.Sb))
    np.testing.assert_array_equal(np.asarray(S), np.asarray(result.S))
    np.testing.assert_array_equal(np.asarray(ST), np.asarray(result.ST))


def test_indices_matches_analyze_for_a_time_series(case):
    """Tier T4: the ``(N, T, K)`` layout comes back at the same shapes."""
    problem, X, Y = case
    Y3 = jnp.stack([jnp.stack([Y, 2.0 * Y], axis=-1), jnp.stack([-Y, 0.5 * Y], axis=-1)], axis=1)

    result = hdmr.analyze(problem, X, Y3, maxorder=2, maxiter=50)
    Sa, _, _, ST = hdmr.indices(problem, X, Y3, maxorder=2, maxiter=50)

    assert Sa.shape == result.Sa.shape == (2, 2, 6)
    assert ST.shape == result.ST.shape == (2, 2, 3)
    np.testing.assert_array_equal(np.asarray(Sa), np.asarray(result.Sa))
    np.testing.assert_array_equal(np.asarray(ST), np.asarray(result.ST))


def test_indices_returns_a_bare_tuple(case):
    """Tier T4: four arrays, no result object, no term labels, no F counts."""
    problem, X, Y = case
    out = hdmr.indices(problem, X, Y, maxorder=2, maxiter=50)

    assert type(out) is tuple
    assert len(out) == 4
    assert all(isinstance(a, jax.Array) for a in out)


def test_indices_clamps_maxorder_without_warning():
    """Tier T4: the core takes the clamp silently; ``analyze`` warns about it."""
    rng = np.random.default_rng(3)
    problem = Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0)})
    X = jnp.asarray(rng.uniform(0.0, 1.0, size=(N_ROWS, 2)))
    Y = X[:, 0] + X[:, 1] ** 2

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, _, ST = hdmr.indices(problem, X, Y, maxorder=3, maxiter=50)

    with pytest.warns(Warning, match="maxorder clamped"):
        result = hdmr.analyze(problem, X, Y, maxorder=3, maxiter=50)
    np.testing.assert_array_equal(np.asarray(ST), np.asarray(result.ST))


# ---------------------------------------------------------------------------
# 2. Transformability
# ---------------------------------------------------------------------------


def test_indices_is_jittable(case):
    """Tier T4: ``jit`` traces ``indices`` and returns the eager values.

    The tolerance is float32 reassociation only. The backfitting solve is a
    long chain of matmuls, and XLA fuses it differently under ``jit``; the
    measured gap is around 1e-5 on an index in [0, 1].
    """
    problem, X, Y = case
    jitted = jax.jit(lambda outputs: hdmr.indices(problem, X, outputs, maxorder=2, maxiter=50))

    ST_jit = jitted(Y)[3]
    ST = hdmr.indices(problem, X, Y, maxorder=2, maxiter=50)[3]

    np.testing.assert_allclose(np.asarray(ST_jit), np.asarray(ST), rtol=1e-3, atol=1e-4)


def test_jit_then_eager_shares_the_f_crit_cache(case):
    """Tier T4: a jitted call must not poison the eager path (H1 regression).

    ``_compute_f_crits`` is ``lru_cache``\\ d on ``(alpha, m1, m2, m3, N)``. A
    call made from inside ``jax.jit`` traces this function once for that key.
    If the cached return value were a JAX array, it would be a tracer from
    that trace, and every later caller with the same key -- including the
    plain eager call right below -- would inherit a dead tracer and raise
    ``UnexpectedTracerError``. This test must run the jitted call first so a
    regression cannot hide behind test order the way it did before the fix:
    the full suite always warmed the cache eagerly first, and only running
    this test in isolation showed the crash.

    Clearing the cache first is what makes the order hold. Another test in
    this file has already run the eager path on the same key, so without the
    clear the jitted call below is a cache hit and the poisoning never
    happens: the test passes against the broken code in a full-file run, and
    fails only in isolation. That is the exact gap this test exists to close.
    """
    problem, X, Y = case
    _compute_f_crits.cache_clear()

    jitted = jax.jit(lambda outputs: hdmr.indices(problem, X, outputs, maxorder=2, maxiter=50))
    ST_jit = jitted(Y)[3]

    result = hdmr.analyze(problem, X, Y, maxorder=2, maxiter=50)

    np.testing.assert_allclose(np.asarray(ST_jit), np.asarray(result.ST), rtol=1e-3, atol=1e-4)


def test_indices_is_vmappable(case):
    """Tier T4: ``vmap`` maps ``indices`` over a batch of output vectors."""
    problem, X, Y = case
    batch = jnp.stack([Y, 2.0 * Y + 1.0])

    ST_batch = jax.vmap(lambda outputs: hdmr.indices(problem, X, outputs, maxorder=2, maxiter=50))(
        batch
    )[3]

    assert ST_batch.shape == (2, problem.num_vars)
    # An affine rescaling of Y leaves every variance fraction unchanged, so
    # the two lanes must agree. That is a real check on the batching, not a
    # restatement of it: a mis-mapped axis would mix the two.
    np.testing.assert_allclose(
        np.asarray(ST_batch[0]), np.asarray(ST_batch[1]), rtol=1e-3, atol=1e-4
    )


def test_jit_of_jacfwd_of_the_inputs(case):
    """Tier T4: ``jit(jacfwd(...))`` differentiates an index w.r.t. ``X``."""
    problem, X, Y = case

    def total_st(inputs):
        return hdmr.indices(problem, inputs, Y, maxorder=2, maxiter=20)[3].sum()

    jac = jax.jit(jax.jacfwd(total_st))(X)

    assert jac.shape == X.shape
    assert np.isfinite(np.asarray(jac)).all()
    assert np.abs(np.asarray(jac)).max() > 0.0

    # Finite and non-zero is not the same as correct: spot check a handful of
    # entries against a central finite difference, which re-runs the whole
    # backfitting fit at X +/- h and so exercises none of the autodiff path.
    # A full-Jacobian check would refit the whole design 2 * N * D times,
    # which is too slow here; a fixed, seeded sample of entries is cheap and
    # deterministic. The F-test term selection makes this comparison
    # genuinely discontinuous in float32 near a threshold crossing, so this
    # step -- like every other gradient check in this file family -- runs
    # under x64. The review measured 2e-5 in float32; under x64 the sampled
    # entries agree to 3.6e-9 against Jacobian entries of order 1e-3, so the
    # tolerance is set at 1e-7. That is still 30x the measured error, and it
    # is tight enough to catch a wrong scale factor, which 2e-5 (2% of a
    # typical entry) would not.
    step = 1e-4
    rng = np.random.default_rng(0)
    n_rows, n_cols = X.shape
    sample_rows = rng.choice(n_rows, size=min(4, n_rows), replace=False)
    with jax.enable_x64():
        X64 = jnp.asarray(X, dtype=jnp.float64)
        jac64 = np.asarray(jax.jacfwd(total_st)(X64))
        for row in sample_rows:
            for col in range(n_cols):
                delta = jnp.zeros_like(X64).at[row, col].set(step)
                plus = float(total_st(X64 + delta))
                minus = float(total_st(X64 - delta))
                fd = (plus - minus) / (2.0 * step)
                np.testing.assert_allclose(jac64[row, col], fd, rtol=0, atol=1e-7)


def test_reverse_mode_is_refused_by_the_backfitting_loop():
    """Tier T4: ``jacrev`` fails, and it fails for the documented reason.

    This is an exemption with an expiry. The early-stopping ``while_loop`` in
    the backfitting is what blocks reverse mode; replacing it with a
    fixed-trip ``scan`` would make this test fail, which is the signal to
    delete it and promote the contract to ``jacrev``.
    """
    problem, X, Y = _uniform_case()

    with pytest.raises(ValueError, match="while_loop"):
        jax.jacrev(lambda outputs: hdmr.indices(problem, X, outputs, maxorder=2, maxiter=20)[3])(Y)


def test_analyze_is_not_jittable(case):
    """Tier T4: the split is real -- ``analyze`` cannot be traced."""
    problem, X, Y = case

    with pytest.raises(Exception, match="[Tt]racer"):
        jax.jit(lambda outputs: hdmr.analyze(problem, X, outputs).ST)(Y)
