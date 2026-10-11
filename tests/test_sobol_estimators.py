"""The ``estimator=`` menu on ``sobol.analyze`` and ``sobol.indices``.

Verification tiers, per the project's oracle policy.

* **T0, closed form.** Ishigami and Sobol-G have analytical ``S1`` and
  ``ST``. Every estimator must converge to them, because that is the only
  claim they all share. ``test_ishigami_end_to_end`` and ``test_sobol_g``.
* **Structural.** Two of the claims in the estimator docstrings are exact,
  not statistical: Jansen's total order can never be negative, and
  Azzini-Rosati can never report ``S1 > ST``. Those are checked on samples
  small enough that the estimators are noisy, which is where a false claim
  would show.

None of these tests restates an estimator's own formula. A test that did
would pass for a wrong formula written twice.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa.benchmarks import ishigami, sobol_g
from jaxgsa.sobol._estimators import DEFAULT_ESTIMATOR, ESTIMATORS, Estimator

# Estimators that run on the cheaper N(D+2) design.
FIRST_ORDER_DESIGN_ESTIMATORS: list[Estimator] = [
    "saltelli-jansen",
    "jansen",
    "janon-monod",
    "martinez",
    "mauntz-kucherenko",
]


def _analyze_with_bad_name(samples, Y, name: str):
    """Call ``analyze`` with a name the type checker would never allow.

    The cast is the point. A bad ``estimator=`` is a runtime concern, and a
    static type checker would refuse the literal before the library ever saw
    it, so these tests would prove nothing about the library.
    """
    return jaxgsa.sobol.analyze(samples, Y, estimator=cast("Estimator", name))


def _indices_with_bad_name(samples, Y, name: str):
    """Call ``indices`` with a name the type checker would never allow."""
    return jaxgsa.sobol.indices(samples, Y, estimator=cast("Estimator", name))


@pytest.fixture(scope="module")
def ishigami_design():
    """A second-order Ishigami design and its outputs, shared by the module."""
    sr = jaxgsa.sobol.sample(ishigami.PROBLEM, base_n=4096, seed=11, verbose=False)
    return sr, ishigami.evaluate(jnp.asarray(sr.samples))


@pytest.fixture(scope="module")
def sobol_g_design():
    """A second-order Sobol-G design and its outputs, shared by the module."""
    sr = jaxgsa.sobol.sample(sobol_g.PROBLEM, base_n=4096, seed=13, verbose=False)
    return sr, sobol_g.evaluate(jnp.asarray(sr.samples))


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_ishigami_end_to_end(ishigami_design, estimator):
    """T0. Every estimator lands on the closed-form Ishigami indices.

    One ``analyze`` call per estimator carries all the checks. S2 must find
    the x1-x3 pair whichever first-order estimator it subtracts. The result
    names its estimator, because six estimators disagree at finite N and a
    stored result would otherwise be ambiguous. ``indices`` agrees with
    ``analyze`` (T4). The vmapped (T, K) path agrees with the scalar one,
    which is a separate kernel; scaling an output leaves every index
    unchanged, so the answer is known.
    """
    sr, Y = ishigami_design
    result = jaxgsa.sobol.analyze(sr, Y, estimator=estimator)
    np.testing.assert_allclose(np.asarray(result.S1), ishigami.ANALYTICAL_S1, atol=0.02)
    np.testing.assert_allclose(np.asarray(result.ST), ishigami.ANALYTICAL_ST, atol=0.02)
    S2 = np.asarray(result.S2)
    np.testing.assert_allclose(S2[0, 2], ishigami.ANALYTICAL_S2[0, 2], atol=0.03)
    np.testing.assert_allclose(S2[0, 1], 0.0, atol=0.03)
    np.testing.assert_allclose(S2[1, 2], 0.0, atol=0.03)

    assert result.estimator == estimator
    assert f"estimator='{estimator}'" in repr(result)

    S1, ST, S2_core = jaxgsa.sobol.indices(sr, Y, estimator=estimator)
    np.testing.assert_array_equal(np.asarray(S1), np.asarray(result.S1))
    np.testing.assert_array_equal(np.asarray(ST), np.asarray(result.ST))
    np.testing.assert_array_equal(np.asarray(S2_core), np.asarray(result.S2))

    stacked = jnp.stack([Y, 2.0 * Y], axis=-1)
    batched = jaxgsa.sobol.analyze(sr, stacked, estimator=estimator)
    for k in range(2):
        np.testing.assert_allclose(
            np.asarray(batched.S1)[k], np.asarray(result.S1), rtol=1e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(batched.ST)[k], np.asarray(result.ST), rtol=1e-4, atol=1e-6
        )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_sobol_g(sobol_g_design, estimator):
    """T0. Every estimator lands on the closed-form Sobol-G indices.

    Sobol-G is the harder case: eight parameters, four of them nearly
    inert, so the near-zero regime is well covered.
    """
    sr, Y = sobol_g_design
    result = jaxgsa.sobol.analyze(sr, Y, estimator=estimator)
    np.testing.assert_allclose(np.asarray(result.S1), sobol_g.ANALYTICAL_S1, atol=0.03)
    np.testing.assert_allclose(np.asarray(result.ST), sobol_g.ANALYTICAL_ST, atol=0.03)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_bootstrap_keeps_the_point_estimate_and_both_kernels_agree(ishigami_design, estimator):
    """Tier T4. One bootstrap run checks the centre and the single-slice kernel.

    The bootstrap resamples with the chosen estimator too, so turning it on
    must not move the point estimate. A chunk holding one output slice drops
    the outer ``vmap``, because mapping a length-one axis makes every gather
    batched on two axes and XLA compiles that to a slower gather. The two
    kernels must still agree bit for bit, or the shortcut would buy speed
    with numbers. Feeding the same output as ``(N,)`` and as ``(N, 1, 1)``
    runs the single-slice kernel and the mapped one on identical data.

    Only the interval endpoints are compared across kernels. The point
    estimate comes from a different pair of kernels, and two estimators
    already disagree there between the scalar and 3-D paths by up to 1e-6 in
    float32, which predates this test.
    """
    sr, Y = ishigami_design
    key = jax.random.key(5)
    plain = jaxgsa.sobol.analyze(sr, Y, estimator=estimator)
    boot = jaxgsa.sobol.analyze(sr, Y, estimator=estimator, n_bootstrap=32, key=key)
    np.testing.assert_array_equal(np.asarray(plain.S1), np.asarray(boot.S1))
    np.testing.assert_array_equal(np.asarray(plain.ST), np.asarray(boot.ST))
    lower, upper = np.asarray(boot.S1_conf)
    assert np.all(lower <= upper)

    mapped = jaxgsa.sobol.analyze(
        sr, Y[:, None, None], estimator=estimator, n_bootstrap=32, key=key
    )
    for name in ("S1_conf", "ST_conf", "S2_conf"):
        got, want = getattr(boot, name), getattr(mapped, name)
        np.testing.assert_array_equal(
            np.asarray(got), np.asarray(want).reshape(np.shape(got)), err_msg=name
        )


class TestStructuralProperties:
    """Claims that hold on every sample, not just in the limit."""

    def test_azzini_rosati_never_reports_s1_above_st(self):
        """``S1 <= ST`` holds sample by sample, which no other scheme gives.

        The design here is deliberately tiny, so the estimators are noisy.
        Under that noise every other scheme crosses the bound somewhere in
        this sweep, and Azzini-Rosati never does.
        """
        crossed_elsewhere = False
        for seed in range(12):
            sr = jaxgsa.sobol.sample(sobol_g.PROBLEM, base_n=32, seed=seed, verbose=False)
            Y = sobol_g.evaluate(jnp.asarray(sr.samples))
            S1, ST, _ = jaxgsa.sobol.indices(sr, Y, estimator="azzini-rosati")
            assert np.all(np.asarray(S1) <= np.asarray(ST) + 1e-6)
            for other in FIRST_ORDER_DESIGN_ESTIMATORS:
                S1, ST, _ = jaxgsa.sobol.indices(sr, Y, estimator=other)
                crossed_elsewhere |= bool(np.any(np.asarray(S1) > np.asarray(ST)))
        assert crossed_elsewhere, "the comparison estimators were not noisy enough to be a control"

    @pytest.mark.parametrize("estimator", ["saltelli-jansen", "jansen"])
    def test_jansen_total_order_is_never_negative(self, estimator):
        """A mean of squares cannot be negative, however small the sample."""
        for seed in range(12):
            sr = jaxgsa.sobol.sample(sobol_g.PROBLEM, base_n=16, seed=seed, verbose=False)
            Y = sobol_g.evaluate(jnp.asarray(sr.samples))
            _, ST, _ = jaxgsa.sobol.indices(sr, Y, estimator=estimator)
            assert np.all(np.asarray(ST) >= 0.0)

    @pytest.mark.slow
    def test_negative_first_order_estimates_survive_a_large_sample(self):
        """A negative ``S1`` is sampling error, not under-sampling.

        Ishigami's third parameter has ``S1`` exactly 0, so its estimate is
        symmetric noise about zero and falls below it about half the time,
        however large N is. This pins the documented behaviour: the package
        does not clip, and the negatives do not go away with more samples.

        Sobol-G is the wrong benchmark for this. Its four "inert" parameters
        have ``S1`` near 7.2e-5, not 0, and once the outputs are standardized
        the estimator resolves that at ``base_n = 8192``, so the estimates
        stay positive. They only used to cross zero because the uncentred
        estimator added an error term proportional to the output mean.
        """
        seen_negative = 0
        for seed in range(8):
            sr = jaxgsa.sobol.sample(ishigami.PROBLEM, base_n=8192, seed=seed, verbose=False)
            Y = ishigami.evaluate(jnp.asarray(sr.samples))
            result = jaxgsa.sobol.analyze(sr, Y, estimator=DEFAULT_ESTIMATOR)
            seen_negative += int(np.asarray(result.S1)[2] < 0.0)
        assert seen_negative > 0


def test_the_default_estimator_is_named_and_shares_a_first_order_formula(ishigami_design):
    """The default is bit-for-bit the named default, and it reports its name.

    Saltelli-Jansen and Mauntz-Kucherenko share a first-order formula and
    differ in total order. This is documented, and it is easy to break by
    editing one of the two without the other.
    """
    sr, Y = ishigami_design
    implicit = jaxgsa.sobol.analyze(sr, Y)
    explicit = jaxgsa.sobol.analyze(sr, Y, estimator=DEFAULT_ESTIMATOR)
    assert implicit.estimator == DEFAULT_ESTIMATOR
    for field in ("S1", "ST", "S2"):
        np.testing.assert_array_equal(
            np.asarray(getattr(implicit, field)), np.asarray(getattr(explicit, field))
        )
    other = jaxgsa.sobol.analyze(sr, Y, estimator="mauntz-kucherenko")
    np.testing.assert_allclose(np.asarray(implicit.S1), np.asarray(other.S1), rtol=1e-6)
    assert not np.allclose(np.asarray(implicit.ST), np.asarray(other.ST), rtol=1e-9)


class TestValidation:
    """A bad ``estimator=`` is refused before any array is touched."""

    def test_unknown_name_is_refused_everywhere(self, ishigami_design):
        """The message names the argument; both entry points check; the name wins over Y."""
        sr, Y = ishigami_design
        with pytest.raises(ValueError, match="estimator must be one of"):
            _analyze_with_bad_name(sr, Y, "saltelli")
        with pytest.raises(ValueError, match="estimator must be one of"):
            _indices_with_bad_name(sr, Y, "nonsense")
        with pytest.raises(ValueError, match="estimator must be one of"):
            _analyze_with_bad_name(sr, jnp.zeros(7), "nope")

    def test_azzini_rosati_refuses_a_first_order_only_design(self):
        """It reads the BA blocks, and an N(D+2) design has none."""
        sr = jaxgsa.sobol.sample(
            ishigami.PROBLEM,
            base_n=64,
            seed=3,
            verbose=False,
            calc_second_order=False,
        )
        Y = ishigami.evaluate(jnp.asarray(sr.samples))
        with pytest.raises(ValueError, match="calc_second_order=True"):
            jaxgsa.sobol.analyze(sr, Y, estimator="azzini-rosati")
        with pytest.raises(ValueError, match="calc_second_order=True"):
            jaxgsa.sobol.indices(sr, Y, estimator="azzini-rosati")


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_indices_stays_transformable(estimator):
    """Each estimator survives ``jit``, ``jit(jacrev(...))`` and ``vmap``.

    Every formula in the menu is plain arithmetic on the output vectors, so
    none of them costs the property. If one ever did, this is where it
    would show.
    """
    problem = ishigami.PROBLEM
    sr = jaxgsa.sobol.sample(problem, base_n=64, seed=5, verbose=False)
    theta = {name: {"low": float(-np.pi), "high": float(np.pi)} for name in problem.names}

    def first_index(theta):
        X = sr.transform(theta)
        Y = ishigami.evaluate(X)
        return jaxgsa.sobol.indices(sr, Y, estimator=estimator)[0][0]

    value = jax.jit(first_index)(theta)
    assert np.isfinite(np.asarray(value))
    jacobian = jax.jit(jax.jacrev(first_index))(theta)
    assert np.all(np.isfinite(np.asarray(jacobian["x1"]["low"])))

    np.testing.assert_allclose(
        np.asarray(jax.jit(first_index)(theta)),
        np.asarray(first_index(theta)),
        rtol=1e-4,
        atol=1e-6,
    )

    X = jnp.asarray(sr.samples)
    batch = jnp.stack([ishigami.evaluate(X), X[:, 0], X[:, 1] ** 2])

    def s1(Y_one):
        return jaxgsa.sobol.indices(sr, Y_one, estimator=estimator)[0]

    mapped = jax.vmap(s1)(batch)
    for row in range(batch.shape[0]):
        np.testing.assert_allclose(
            np.asarray(mapped[row]), np.asarray(s1(batch[row])), rtol=1e-4, atol=1e-6
        )


def test_default_estimator_is_saltelli_jansen():
    assert DEFAULT_ESTIMATOR == "saltelli-jansen"


def test_first_order_design_list_covers_every_non_ba_estimator():
    """The hand-written list above must not drift from ``ESTIMATORS``."""
    assert set(FIRST_ORDER_DESIGN_ESTIMATORS) == set(ESTIMATORS) - {"azzini-rosati"}
