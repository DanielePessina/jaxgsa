"""The ``estimator=`` menu on ``sobol.analyze`` and ``sobol.indices``.

Verification tiers, per the project's oracle policy.

* **T0, closed form.** Ishigami and Sobol-G have analytical ``S1`` and
  ``ST``. Every estimator must converge to them, because that is the only
  claim they all share. ``TestConvergesToAnalytic``.
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


def _analyze_with_bad_name(sampling_result, Y, name: str):
    """Call ``analyze`` with a name the type checker would never allow.

    The cast is the point. A bad ``estimator=`` is a runtime concern, and a
    static type checker would refuse the literal before the library ever saw
    it, so these tests would prove nothing about the library.
    """
    return jaxgsa.sobol.analyze(sampling_result, Y, estimator=cast("Estimator", name))


def _indices_with_bad_name(sampling_result, Y, name: str):
    """Call ``indices`` with a name the type checker would never allow."""
    return jaxgsa.sobol.indices(sampling_result, Y, estimator=cast("Estimator", name))


@pytest.fixture(scope="module")
def ishigami_design():
    """A second-order Ishigami design and its outputs, shared by the module."""
    sr = jaxgsa.sobol.sample(ishigami.PROBLEM, n_samples=1, base_n=4096, seed=11, verbose=False)
    return sr, ishigami.evaluate(jnp.asarray(sr.samples))


@pytest.fixture(scope="module")
def sobol_g_design():
    """A second-order Sobol-G design and its outputs, shared by the module."""
    sr = jaxgsa.sobol.sample(sobol_g.PROBLEM, n_samples=1, base_n=4096, seed=13, verbose=False)
    return sr, sobol_g.evaluate(jnp.asarray(sr.samples))


class TestConvergesToAnalytic:
    """T0. Every estimator must land on the closed-form indices."""

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_ishigami(self, ishigami_design, estimator):
        """All six estimators reproduce the analytical Ishigami indices."""
        sr, Y = ishigami_design
        result = jaxgsa.sobol.analyze(sr, Y, estimator=estimator)
        np.testing.assert_allclose(np.asarray(result.S1), ishigami.ANALYTICAL_S1, atol=0.02)
        np.testing.assert_allclose(np.asarray(result.ST), ishigami.ANALYTICAL_ST, atol=0.02)

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_sobol_g(self, sobol_g_design, estimator):
        """All six estimators reproduce the analytical Sobol-G indices.

        Sobol-G is the harder case: eight parameters, four of them nearly
        inert, so the near-zero regime is well covered.
        """
        sr, Y = sobol_g_design
        result = jaxgsa.sobol.analyze(sr, Y, estimator=estimator)
        np.testing.assert_allclose(np.asarray(result.S1), sobol_g.ANALYTICAL_S1, atol=0.03)
        np.testing.assert_allclose(np.asarray(result.ST), sobol_g.ANALYTICAL_ST, atol=0.03)

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_second_order_still_lands(self, ishigami_design, estimator):
        """S2 stays correct whichever first-order estimator it subtracts.

        Only the x1-x3 pair interacts in Ishigami, and every scheme has to
        find that pair and leave the other two near zero.
        """
        sr, Y = ishigami_design
        result = jaxgsa.sobol.analyze(sr, Y, estimator=estimator)
        S2 = np.asarray(result.S2)
        assert S2 is not None
        np.testing.assert_allclose(S2[0, 2], ishigami.ANALYTICAL_S2[0, 2], atol=0.03)
        np.testing.assert_allclose(S2[0, 1], 0.0, atol=0.03)
        np.testing.assert_allclose(S2[1, 2], 0.0, atol=0.03)


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
            sr = jaxgsa.sobol.sample(
                sobol_g.PROBLEM, n_samples=1, base_n=32, seed=seed, verbose=False
            )
            Y = sobol_g.evaluate(jnp.asarray(sr.samples))
            azzini = jaxgsa.sobol.analyze(sr, Y, estimator="azzini-rosati")
            assert np.all(np.asarray(azzini.S1) <= np.asarray(azzini.ST) + 1e-6)
            for other in FIRST_ORDER_DESIGN_ESTIMATORS:
                result = jaxgsa.sobol.analyze(sr, Y, estimator=other)
                crossed_elsewhere |= bool(np.any(np.asarray(result.S1) > np.asarray(result.ST)))
        assert crossed_elsewhere, "the comparison estimators were not noisy enough to be a control"

    @pytest.mark.parametrize("estimator", ["saltelli-jansen", "jansen"])
    def test_jansen_total_order_is_never_negative(self, estimator):
        """A mean of squares cannot be negative, however small the sample."""
        for seed in range(12):
            sr = jaxgsa.sobol.sample(
                sobol_g.PROBLEM, n_samples=1, base_n=16, seed=seed, verbose=False
            )
            Y = sobol_g.evaluate(jnp.asarray(sr.samples))
            result = jaxgsa.sobol.analyze(sr, Y, estimator=estimator)
            assert np.all(np.asarray(result.ST) >= 0.0)

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
            sr = jaxgsa.sobol.sample(
                ishigami.PROBLEM, n_samples=1, base_n=8192, seed=seed, verbose=False
            )
            Y = ishigami.evaluate(jnp.asarray(sr.samples))
            result = jaxgsa.sobol.analyze(sr, Y, estimator=DEFAULT_ESTIMATOR)
            seen_negative += int(np.asarray(result.S1)[2] < 0.0)
        assert seen_negative > 0


class TestDefaultIsUnchanged:
    """The default must stay bit-for-bit what jaxgsa has always computed."""

    def test_named_default_equals_no_argument(self, ishigami_design):
        """Passing the default by name changes nothing at all."""
        sr, Y = ishigami_design
        implicit = jaxgsa.sobol.analyze(sr, Y)
        explicit = jaxgsa.sobol.analyze(sr, Y, estimator=DEFAULT_ESTIMATOR)
        for field in ("S1", "ST", "S2"):
            np.testing.assert_array_equal(
                np.asarray(getattr(implicit, field)),
                np.asarray(getattr(explicit, field)),
            )

    def test_default_name_is_saltelli_jansen(self):
        """A rename is a breaking change; this pins the name itself."""
        assert DEFAULT_ESTIMATOR == "saltelli-jansen"

    def test_first_order_of_the_default_is_the_mauntz_kucherenko_one(self, ishigami_design):
        """The two schemes share a first-order formula, and differ in total order.

        This is documented, and it is easy to break by editing one of the two
        without the other.
        """
        sr, Y = ishigami_design
        default = jaxgsa.sobol.analyze(sr, Y, estimator="saltelli-jansen")
        other = jaxgsa.sobol.analyze(sr, Y, estimator="mauntz-kucherenko")
        np.testing.assert_allclose(np.asarray(default.S1), np.asarray(other.S1), rtol=1e-6)
        assert not np.allclose(np.asarray(default.ST), np.asarray(other.ST), rtol=1e-9)


class TestValidation:
    """A bad ``estimator=`` is refused before any array is touched."""

    def test_unknown_name_is_refused(self, ishigami_design):
        """The message names the argument and lists what is accepted."""
        sr, Y = ishigami_design
        with pytest.raises(ValueError, match="estimator must be one of"):
            _analyze_with_bad_name(sr, Y, "saltelli")

    def test_unknown_name_is_refused_by_indices_too(self, ishigami_design):
        """The transformable entry point validates the same way."""
        sr, Y = ishigami_design
        with pytest.raises(ValueError, match="estimator must be one of"):
            _indices_with_bad_name(sr, Y, "nonsense")

    def test_the_name_is_checked_before_the_outputs(self, ishigami_design):
        """A bad name wins over a bad Y, which is what the preamble promises."""
        sr, _ = ishigami_design
        wrong_length = jnp.zeros(7)
        with pytest.raises(ValueError, match="estimator must be one of"):
            _analyze_with_bad_name(sr, wrong_length, "nope")

    def test_azzini_rosati_refuses_a_first_order_only_design(self):
        """It reads the BA blocks, and an N(D+2) design has none."""
        sr = jaxgsa.sobol.sample(
            ishigami.PROBLEM,
            n_samples=1,
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


class TestPathsAgree:
    """Every code path must give one estimator one answer."""

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_indices_matches_analyze(self, ishigami_design, estimator):
        """The transformable entry point and the diagnosed one agree."""
        sr, Y = ishigami_design
        result = jaxgsa.sobol.analyze(sr, Y, estimator=estimator)
        S1, ST, S2 = jaxgsa.sobol.indices(sr, Y, estimator=estimator)
        np.testing.assert_array_equal(np.asarray(S1), np.asarray(result.S1))
        np.testing.assert_array_equal(np.asarray(ST), np.asarray(result.ST))
        np.testing.assert_array_equal(np.asarray(S2), np.asarray(result.S2))

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_multi_output_matches_the_scalar_path(self, ishigami_design, estimator):
        """The vmapped path over (T, K) slices agrees with the scalar path.

        The two are separate kernels, and only a shared estimator keeps them
        equal. Stacking the same output twice makes the answer known.
        """
        sr, Y = ishigami_design
        scalar = jaxgsa.sobol.analyze(sr, Y, estimator=estimator)
        stacked = jnp.stack([Y, 2.0 * Y], axis=-1)
        batched = jaxgsa.sobol.analyze(sr, stacked, estimator=estimator)
        for k in range(2):
            # Scaling an output by a constant leaves every index unchanged.
            np.testing.assert_allclose(
                np.asarray(batched.S1)[k], np.asarray(scalar.S1), rtol=1e-4, atol=1e-6
            )
            np.testing.assert_allclose(
                np.asarray(batched.ST)[k], np.asarray(scalar.ST), rtol=1e-4, atol=1e-6
            )

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_bootstrap_centres_on_the_same_point_estimate(self, ishigami_design, estimator):
        """Turning the bootstrap on must not move the point estimate.

        The bootstrap path resamples with the chosen estimator too, so the
        interval describes the same quantity as the number at its centre.
        """
        sr, Y = ishigami_design
        plain = jaxgsa.sobol.analyze(sr, Y, estimator=estimator)
        boot = jaxgsa.sobol.analyze(
            sr, Y, estimator=estimator, n_bootstrap=40, key=jax.random.key(0)
        )
        np.testing.assert_array_equal(np.asarray(plain.S1), np.asarray(boot.S1))
        np.testing.assert_array_equal(np.asarray(plain.ST), np.asarray(boot.ST))
        lower, upper = np.asarray(boot.S1_conf)
        assert np.all(lower <= upper)

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_single_slice_bootstrap_matches_the_mapped_one(self, ishigami_design, estimator):
        """Tier T4 (internal consistency): one slice takes a different kernel.

        A chunk holding one output slice drops the outer ``vmap``, because
        mapping a length-one axis makes every gather batched on two axes and
        XLA compiles that to a slower gather. The two kernels must still
        agree bit for bit, or the shortcut would buy speed with numbers.
        Feeding the same output as ``(N,)`` and as ``(N, 1, 1)`` runs the
        single-slice kernel and the mapped one on identical data.

        Only the interval endpoints are compared. The point estimate comes
        from a different pair of kernels that this shortcut does not touch,
        and two estimators already disagree there between the scalar and 3-D
        paths by up to 1e-6 in float32, which predates this test.
        """
        sr, Y = ishigami_design
        key = jax.random.key(5)
        one = jaxgsa.sobol.analyze(sr, Y, estimator=estimator, n_bootstrap=32, key=key)
        mapped = jaxgsa.sobol.analyze(
            sr, Y[:, None, None], estimator=estimator, n_bootstrap=32, key=key
        )
        for name in ("S1_conf", "ST_conf", "S2_conf"):
            got, want = getattr(one, name), getattr(mapped, name)
            np.testing.assert_array_equal(
                np.asarray(got), np.asarray(want).reshape(np.shape(got)), err_msg=name
            )


class TestStaysTransformable:
    """``indices`` is the differentiable entry point, for every estimator."""

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_jit_and_jacrev(self, estimator):
        """Each estimator survives ``jit`` and ``jit(jacrev(...))``.

        Every formula in the menu is plain arithmetic on the output vectors,
        so none of them costs the property. If one ever did, this is where it
        would show.
        """
        problem = ishigami.PROBLEM
        sr = jaxgsa.sobol.sample(problem, n_samples=1, base_n=64, seed=5, verbose=False)
        theta = {
            name: {"low": float(low), "high": float(high)}
            for name, (low, high) in zip(
                problem.names,
                [(-np.pi, np.pi)] * 3,
                strict=True,
            )
        }

        def first_index(theta):
            X = sr.transform(theta)
            Y = ishigami.evaluate(X)
            return jaxgsa.sobol.indices(sr, Y, estimator=estimator)[0][0]

        value = jax.jit(first_index)(theta)
        assert np.isfinite(np.asarray(value))
        jacobian = jax.jit(jax.jacrev(first_index))(theta)
        assert np.all(np.isfinite(np.asarray(jacobian["x1"]["low"])))

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_vmap_over_outputs(self, ishigami_design, estimator):
        """``vmap`` over a batch of output vectors gives the per-item answer."""
        sr, Y = ishigami_design

        def s1(Y_one):
            return jaxgsa.sobol.indices(sr, Y_one, estimator=estimator)[0]

        batch = jnp.stack([Y, 3.0 * Y])
        mapped = jax.vmap(s1)(batch)
        np.testing.assert_allclose(
            np.asarray(mapped[0]), np.asarray(mapped[1]), rtol=1e-4, atol=1e-6
        )


def test_first_order_design_list_covers_every_non_ba_estimator():
    """The hand-written list above must not drift from ``ESTIMATORS``."""
    assert set(FIRST_ORDER_DESIGN_ESTIMATORS) == set(ESTIMATORS) - {"azzini-rosati"}


class TestTheResultSaysWhichEstimatorRanIt:
    """T4: six estimators disagree at finite N, so a result must name its own.

    Without this a stored result is ambiguous. The same design and the same
    outputs give six different index vectors, and nothing on the result would
    say which one is in front of you.
    """

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_each_estimator_is_recorded_on_the_result(self, ishigami_design, estimator):
        """The name the caller passed is the name the result reports."""
        sr, Y = ishigami_design
        assert jaxgsa.sobol.analyze(sr, Y, estimator=estimator).estimator == estimator

    def test_the_default_names_itself_rather_than_staying_blank(self, ishigami_design):
        """A caller who passes nothing still learns what ran."""
        sr, Y = ishigami_design
        assert jaxgsa.sobol.analyze(sr, Y).estimator == DEFAULT_ESTIMATOR

    def test_the_repr_carries_it(self, ishigami_design):
        """Provenance a user can read without knowing the field name."""
        sr, Y = ishigami_design
        assert "estimator='martinez'" in repr(jaxgsa.sobol.analyze(sr, Y, estimator="martinez"))
