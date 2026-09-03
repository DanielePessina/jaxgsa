"""The PCE pure core: ``jaxgsa.pce.indices``.

Two claims are checked here.

1. ``pce.indices`` computes the same numbers as ``pce.analyze``, and it
   survives ``jit``, ``vmap`` and ``jit(jacrev(...))``, which ``analyze``
   cannot.
2. The polynomial degree the fit actually used is still reachable without a
   warning, through ``pce.effective_order``.

Tier T4 (internal consistency) throughout, except where noted: what a
transformability contract can be checked against is the untransformed call,
and there is no external oracle for "this function traces".

Every case runs on a truncated Gaussian as well as a uniform problem. The
truncated marginal is the one that used to break: its CDF went through scipy
on the host, and the standard deviation went through ``jnp`` inside a trace.
A uniform-only test sees neither.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import pce
from jaxgsa.benchmarks import ishigami
from jaxgsa.problem import Problem

# A truncated Gaussian narrow enough to take the Legendre route (its bounds
# sit well inside _WIDE_TRUNCATION_Z), so the truncated CDF really runs.
TRUNCATED_PROBLEM = Problem.from_dict(
    {
        "x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0, "low": -1.5, "high": 2.0},
        "x2": (0.0, 1.0),
        "x3": {"dist": "gaussian", "mean": 1.0, "variance": 0.25},
    }
)


def _uniform_case(n: int = 512, seed: int = 0):
    """Return ``(problem, X, Y)`` for the Ishigami function on uniforms."""
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.uniform(-np.pi, np.pi, size=(n, 3)))
    return ishigami.PROBLEM, X, ishigami.evaluate(X)


def _truncated_case(n: int = 512, seed: int = 1):
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
    """Tier T4: ``indices`` returns exactly what ``analyze`` reports.

    ``analyze`` fits through the same core, so this guards the wiring rather
    than the estimator maths.
    """
    problem, X, Y = case
    result = pce.analyze(problem, X, Y, order=3)
    S1, ST, S2 = pce.indices(problem, X, Y, order=3)

    np.testing.assert_array_equal(np.asarray(S1), np.asarray(result.S1))
    np.testing.assert_array_equal(np.asarray(ST), np.asarray(result.ST))
    np.testing.assert_array_equal(np.asarray(S2), np.asarray(result.S2))


def test_indices_matches_analyze_for_a_time_series(case):
    """Tier T4: the ``(N, T, K)`` layout comes back at the same shapes."""
    problem, X, Y = case
    Y3 = jnp.stack([jnp.stack([Y, 2.0 * Y], axis=-1), jnp.stack([Y**2, -Y], axis=-1)], axis=1)
    assert Y3.shape == (X.shape[0], 2, 2)

    result = pce.analyze(problem, X, Y3, order=2)
    S1, ST, S2 = pce.indices(problem, X, Y3, order=2)

    assert S1.shape == result.S1.shape == (2, 2, 3)
    assert S2.shape == result.S2.shape == (2, 2, 3, 3)
    np.testing.assert_array_equal(np.asarray(S1), np.asarray(result.S1))
    np.testing.assert_array_equal(np.asarray(ST), np.asarray(result.ST))


def test_indices_returns_a_bare_tuple(case):
    """Tier T4: three arrays, no result object, nothing to unwrap."""
    problem, X, Y = case
    out = pce.indices(problem, X, Y, order=2)

    assert type(out) is tuple
    assert len(out) == 3
    assert all(isinstance(a, jax.Array) for a in out)


# ---------------------------------------------------------------------------
# 2. Transformability
# ---------------------------------------------------------------------------


def test_indices_is_jittable(case):
    """Tier T4: ``jit`` traces ``indices`` and returns the eager values.

    The tolerance is float32 reassociation only: XLA fuses the Gram solve
    differently under ``jit`` than the eager path does.
    """
    problem, X, Y = case
    jitted = jax.jit(lambda outputs: pce.indices(problem, X, outputs, order=3))

    S1_jit, ST_jit, _ = jitted(Y)
    S1, ST, _ = pce.indices(problem, X, Y, order=3)

    np.testing.assert_allclose(np.asarray(S1_jit), np.asarray(S1), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(ST_jit), np.asarray(ST), rtol=1e-5, atol=1e-6)


def test_indices_is_vmappable(case):
    """Tier T4: ``vmap`` maps ``indices`` over a batch of output vectors."""
    problem, X, Y = case
    batch = jnp.stack([Y, 2.0 * Y + 1.0, Y**2])

    S1_batch, _, _ = jax.vmap(lambda outputs: pce.indices(problem, X, outputs, order=2))(batch)

    assert S1_batch.shape == (3, problem.num_vars)
    for row, outputs in enumerate(batch):
        S1, _, _ = pce.indices(problem, X, outputs, order=2)
        np.testing.assert_allclose(np.asarray(S1_batch[row]), np.asarray(S1), rtol=1e-4, atol=1e-6)


def test_jit_of_jacrev_of_the_inputs(case):
    """Tier T4: ``jit(jacrev(...))`` differentiates an index w.r.t. ``X``.

    Differentiating through the sample is the harder direction: it runs the
    marginal transform, the design matrix and the Gram solve in reverse. A
    finite Jacobian with a non-zero entry is what says the chain closed.
    """
    problem, X, Y = case

    def total_s1(inputs):
        return pce.indices(problem, inputs, Y, order=2)[0].sum()

    with jax.enable_x64():
        jac = jax.jit(jax.jacrev(total_s1))(jnp.asarray(X, dtype=jnp.float64))

    assert jac.shape == X.shape
    assert np.isfinite(np.asarray(jac)).all()
    assert np.abs(np.asarray(jac)).max() > 0.0

    # A finite, non-zero Jacobian is not evidence it is the *right* Jacobian:
    # a constant offset or a wrong sign would pass both checks above. Spot
    # check a handful of entries against a central finite difference, which
    # re-runs the whole chain (design matrix, Gram solve, index) at X +/- h
    # and so exercises none of the autodiff path. Full-size X makes checking
    # every entry too slow; a fixed, seeded sample of entries is cheap and
    # deterministic. Measured worst-case agreement on this chain is 3e-9.
    step = 1e-5
    rng = np.random.default_rng(0)
    n_rows, n_cols = X.shape
    sample_rows = rng.choice(n_rows, size=min(6, n_rows), replace=False)
    with jax.enable_x64():
        X64 = jnp.asarray(X, dtype=jnp.float64)
        jac64 = np.asarray(jax.jacrev(total_s1)(X64))
        for row in sample_rows:
            for col in range(n_cols):
                delta = jnp.zeros_like(X64).at[row, col].set(step)
                plus = float(total_s1(X64 + delta))
                minus = float(total_s1(X64 - delta))
                fd = (plus - minus) / (2.0 * step)
                np.testing.assert_allclose(jac64[row, col], fd, rtol=0, atol=3e-9)


def test_analyze_is_not_jittable(case):
    """Tier T4: the split is real -- ``analyze`` cannot be traced.

    If this ever passes, ``analyze`` has stopped reading its data on the host
    and the two functions no longer differ in the way the docstrings claim.
    """
    problem, X, Y = case

    with pytest.raises(Exception, match="[Tt]racer"):
        jax.jit(lambda outputs: pce.analyze(problem, X, outputs).S1)(Y)


# ---------------------------------------------------------------------------
# 3. The reduced order is still reachable
# ---------------------------------------------------------------------------


def test_effective_order_predicts_the_order_analyze_used():
    """Tier T4: the helper answers what the fit reports, without fitting."""
    problem, X, Y = _uniform_case(n=16)

    with pytest.warns(Warning, match="order reduced"):
        result = pce.analyze(problem, X, Y, order=5)

    assert pce.effective_order(problem, X.shape[0], order=5) == result.order
    assert result.order < 5


def test_effective_order_leaves_a_fitted_order_alone():
    """Tier T4: with rows to spare the requested degree is what runs."""
    problem, X, Y = _uniform_case(n=512)

    assert pce.effective_order(problem, X.shape[0], order=3) == 3
    assert pce.analyze(problem, X, Y, order=3).order == 3


def test_indices_reduces_the_order_silently():
    """Tier T4: the core takes the same reduction and says nothing.

    A warning is a host-side side effect. The point of the core is that it has
    none, which is why ``effective_order`` exists to answer the same question.
    """
    problem, X, Y = _uniform_case(n=16)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        S1, _, _ = pce.indices(problem, X, Y, order=5)

    with pytest.warns(Warning, match="order reduced"):
        result = pce.analyze(problem, X, Y, order=5)
    np.testing.assert_array_equal(np.asarray(S1), np.asarray(result.S1))


class TestIndicesSharesTheCapabilityGates:
    """``pce.indices`` refuses exactly the problems ``pce.analyze`` refuses.

    Tier T4 (behavioural contract). The Wiener-Askey basis is orthogonal
    under the independent product measure only. Before the gate, the core
    silently returned Sobol indices from a basis that is not orthogonal under
    the dependent measure ``problem.correlation`` declares.
    """

    @staticmethod
    def _correlated_case(n: int = 64):
        problem, X, Y = _uniform_case(n=n)
        R = np.eye(3)
        R[0, 1] = R[1, 0] = 0.8
        return problem.with_correlation(R), X, Y

    def test_a_correlated_problem_raises_the_analyze_message(self):
        """The refusal is word-for-word analyze's, apart from the entry name."""
        problem, X, Y = self._correlated_case()

        with pytest.raises(ValueError, match="assume independent inputs") as e_core:
            pce.indices(problem, X, Y)
        with pytest.raises(ValueError, match="assume independent inputs") as e_analyze:
            pce.analyze(problem, X, Y)

        assert "jaxgsa.pce.indices" in str(e_core.value)
        assert str(e_core.value).replace("jaxgsa.pce.indices", "jaxgsa.pce.analyze") == str(
            e_analyze.value
        )

    def test_an_identity_correlation_passes_under_jit(self):
        """An explicit identity matrix declares independence, and the gate
        reads only static problem metadata, so it must not break tracing."""
        problem, X, Y = _uniform_case(n=256)
        identity = problem.with_correlation(np.eye(3))

        eager = pce.indices(identity, X, Y, order=2)
        jitted = jax.jit(lambda Y_: pce.indices(identity, X, Y_, order=2))(Y)

        np.testing.assert_allclose(
            np.asarray(jitted[0]), np.asarray(eager[0]), rtol=1e-5, atol=1e-7
        )
