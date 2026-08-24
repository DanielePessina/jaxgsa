"""Tests for the shared analysis preamble in ``jaxgsa._core.entry``.

Tier T4 (behavioural contract). No external oracle: these pin the entry
points against the library's own stated rules, not against any numerical
result.

Every ``analyze()`` used to hand-write the same ten-step preamble. Thirteen
copies drifted, and this module holds one test per thing the drift broke:

1. ``conf_level`` went unchecked in two of the five bootstrapping methods, so
   a level outside ``(0, 1)`` came back as an inverted interval.
2. ``sobol`` checked ``slice_chunk_size`` on one of its three paths.
3. A negative resample count silently returned no interval.
4. Six methods returned NaN or 0 for a constant output slice without a word.
5. Three methods validated their scalar arguments after the host-side data
   check, so a typo cost a full pass over the sample before it was reported.

``TestEveryMethodWarnsOnAConstantOutput`` reads its method list off the
registry, so a fourteenth method is covered the moment it registers.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import JaxgsaWarning
from jaxgsa._core.entry import (
    at_least,
    check_scalars,
    gates,
    in_open_interval,
    one_of,
    prepare,
    require,
)
from jaxgsa._core.invalid import InvalidUnit, resolve_policy
from jaxgsa._core.precision import _loses_precision, unit_clip_bounds
from jaxgsa._core.registry import methods
from jaxgsa._core.sampling import UNIT_CLIP
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa._core.validation import _correlation_tolerant_methods

D = 3
PROBLEM = jaxgsa.Problem(("x1", "x2", "x3"), ((0.0, 1.0),) * D)
N = 128


def _xy(*, constant: bool = False, n: int = N):
    """A clean given-data sample, optionally with a constant output."""
    X = np.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n, seed=0))
    # 0.1, not a value like 3.0 whose float32 mean is exact: a constant whose
    # mean rounds leaves var ~1e-16 rather than 0, which is the case the
    # exact-constancy guard exists for. A test using 3.0 passes either way.
    Y = np.full(n, 0.1) if constant else X[:, 0] + 2.0 * X[:, 1] ** 2
    return jnp.asarray(X), jnp.asarray(Y)


def _sobol_design(n=32):
    return jaxgsa.sobol.sample(PROBLEM, n, seed=0, verbose=False)


def _morris_design(n=8):
    return jaxgsa.morris.sample(PROBLEM, n, seed=0, verbose=False)


class TestScalarCheckHelpers:
    """The declarative checks say what they mean."""

    def test_at_least_lets_none_through(self):
        check_scalars((at_least("chunk", None, 1),))

    def test_at_least_reports_the_name_and_the_value(self):
        with pytest.raises(ValueError, match=r"chunk must be >= 1, got 0"):
            check_scalars((at_least("chunk", 0, 1),))

    def test_in_open_interval_rejects_both_endpoints(self):
        for bad in (0.0, 1.0, -0.2, 1.5):
            with pytest.raises(ValueError, match="conf_level must be in"):
                check_scalars((in_open_interval("conf_level", bad, 0.0, 1.0),))

    def test_one_of_names_the_options(self):
        with pytest.raises(ValueError, match="ci_method must be one of"):
            check_scalars((one_of("ci_method", "nope", ("quantile", "gaussian")),))

    def test_the_first_failure_wins(self):
        checks = (require(False, "first"), require(False, "second"))
        with pytest.raises(ValueError, match="^first$"):
            check_scalars(checks)


class TestConfLevelIsValidatedEverywhere:
    """Defect 1: sobol and morris accepted any conf_level at all.

    Before this change ``sobol.analyze(..., conf_level=-0.2)`` returned an
    inverted interval, lower above upper, with no error and no warning.
    """

    def test_sobol_refuses_a_conf_level_outside_the_unit_interval(self):
        sr = _sobol_design()
        Y = jnp.asarray(np.asarray(sr.samples)[:, 0])
        with pytest.raises(ValueError, match="conf_level must be in"):
            jaxgsa.sobol.analyze(sr, Y, n_bootstrap=4, key=jax.random.key(0), conf_level=-0.2)

    def test_morris_refuses_a_conf_level_outside_the_unit_interval(self):
        sr = _morris_design()
        Y = jnp.asarray(np.asarray(sr.samples)[:, 0])
        with pytest.raises(ValueError, match="conf_level must be in"):
            jaxgsa.morris.analyze(sr, Y, n_bootstrap=4, key=jax.random.key(0), conf_level=1.5)


class TestChunkSizeIsValidatedOnEveryPath:
    """Defect 2: sobol checked it inside one branch of one of three paths."""

    def test_the_scalar_no_bootstrap_path_refuses_it(self):
        sr = _sobol_design()
        Y = jnp.asarray(np.asarray(sr.samples)[:, 0])
        with pytest.raises(ValueError, match="slice_chunk_size must be >= 1"):
            jaxgsa.sobol.analyze(sr, Y, slice_chunk_size=0)

    def test_the_bootstrap_path_refuses_it(self):
        sr = _sobol_design()
        Y = jnp.asarray(np.asarray(sr.samples)[:, 0])
        with pytest.raises(ValueError, match="slice_chunk_size must be >= 1"):
            jaxgsa.sobol.analyze(sr, Y, n_bootstrap=4, key=jax.random.key(0), slice_chunk_size=0)

    def test_the_multi_output_path_refuses_it(self):
        sr = _sobol_design()
        base = np.asarray(sr.samples)[:, 0]
        Y = jnp.asarray(np.stack([base, 2.0 * base], axis=-1))
        with pytest.raises(ValueError, match="slice_chunk_size must be >= 1"):
            jaxgsa.sobol.analyze(sr, Y, slice_chunk_size=0)


class TestNegativeResampleCountsAreRefused:
    """Defect 3: a negative count silently disabled the intervals.

    ``n_bootstrap=-1`` failed the ``> 0`` test, so the no-bootstrap path
    ran and the result came back with ``S1_conf=None``, as though the caller
    had asked for no interval.
    """

    def test_sobol(self):
        sr = _sobol_design()
        Y = jnp.asarray(np.asarray(sr.samples)[:, 0])
        with pytest.raises(ValueError, match="n_bootstrap must be >= 0"):
            jaxgsa.sobol.analyze(sr, Y, n_bootstrap=-1)

    def test_morris(self):
        sr = _morris_design()
        Y = jnp.asarray(np.asarray(sr.samples)[:, 0])
        with pytest.raises(ValueError, match="n_bootstrap must be >= 0"):
            jaxgsa.morris.analyze(sr, Y, n_bootstrap=-1)

    def test_pawn(self):
        X, Y = _xy()
        with pytest.raises(ValueError, match="n_bootstrap must be >= 0"):
            jaxgsa.pawn.analyze(PROBLEM, X, Y, n_bootstrap=-1)

    def test_optimal_transport(self):
        X, Y = _xy()
        with pytest.raises(ValueError, match="n_bootstrap must be >= 0"):
            jaxgsa.optimal_transport.analyze(PROBLEM, X, Y, n_bootstrap=-1)

    def test_borgonovo(self):
        X, Y = _xy()
        with pytest.raises(ValueError, match="n_bootstrap must be >= 0"):
            jaxgsa.borgonovo.analyze(PROBLEM, X, Y, n_bootstrap=-1)


def _constant_output_call(name):
    """Return a callable that analyzes a constant output with ``name``.

    Every method has to be reachable, so the three calling conventions are
    spelled out. Sizes are the smallest each method accepts, because the
    point of the call is the warning, not the number.
    """
    # HDMR's B-spline backfitting refuses a sample under 300 rows.
    X, Y = _xy(constant=True, n=320 if name == "hdmr" else N)
    if name == "sobol":
        sobol_design = _sobol_design()
        return lambda: jaxgsa.sobol.analyze(sobol_design, jnp.full(sobol_design.n_runs, 0.1))
    if name == "morris":
        morris_design = _morris_design()
        return lambda: jaxgsa.morris.analyze(morris_design, jnp.full(morris_design.n_runs, 0.1))
    if name == "efast":
        efast_design = jaxgsa.efast.sample(PROBLEM, 400, seed=0)
        return lambda: jaxgsa.efast.analyze(efast_design, jnp.full(efast_design.n_runs, 0.1))
    if name == "kucherenko":
        kuch_design = jaxgsa.kucherenko.sample(PROBLEM, 64, seed=0)
        return lambda: jaxgsa.kucherenko.analyze(kuch_design, jnp.full(kuch_design.n_runs, 0.1))
    if name == "dgsm":
        return lambda: jaxgsa.dgsm.analyze(PROBLEM, lambda x: jnp.asarray(0.1), X)
    if name == "vkoga":
        return lambda: jaxgsa.vkoga.analyze(
            PROBLEM,
            X,
            Y,
            max_centers=8,
            n_outer=4,
            n_inner=4,
            n_variance=4,
            n_folds=2,
            key=jax.random.key(0),
        )
    if name == "hsic":
        return lambda: jaxgsa.hsic.analyze(PROBLEM, X, Y, key=jax.random.key(0))
    if name == "shapley":
        return lambda: jaxgsa.shapley.analyze(PROBLEM, X, Y, order=2)
    # Borgonovo is the one method whose bootstrap is on by default, so it is
    # the one that needs a key to run at all.
    if name == "borgonovo":
        return lambda: jaxgsa.borgonovo.analyze(PROBLEM, X, Y, key=jax.random.key(0))
    return lambda: getattr(jaxgsa, name).analyze(PROBLEM, X, Y)


ZERO_VARIANCE_IDS = sorted(methods())


class TestEveryMethodWarnsOnAConstantOutput:
    """Defect 4: only seven of the thirteen said anything.

    A constant output slice gives NaN in the variance-based methods and an
    exact 0 in the distribution-comparison ones. Either way the number is not
    an answer to the question the caller asked, so it is never returned in
    silence.
    """

    @pytest.mark.parametrize("name", ZERO_VARIANCE_IDS)
    def test_it_warns(self, name):
        call = _constant_output_call(name)
        with pytest.warns(JaxgsaWarning, match="zero variance"):
            call()


class TestScalarsAreCheckedBeforeTheData:
    """Defect 5: three methods validated arguments after the data check.

    A sample with a non-finite value would raise the (expensive, host-side)
    non-finite error even when a scalar argument was already unusable.
    """

    def _poisoned(self):
        X, Y = _xy()
        Y = Y.at[3].set(jnp.nan)
        return X, Y

    def test_pawn_reports_the_bad_argument_not_the_bad_sample(self):
        X, Y = self._poisoned()
        with pytest.raises(ValueError, match="n_bins must be >= 2"):
            jaxgsa.pawn.analyze(PROBLEM, X, Y, n_bins=1)

    def test_borgonovo_reports_the_bad_argument_not_the_bad_sample(self):
        X, Y = self._poisoned()
        with pytest.raises(ValueError, match="grid_size must be >= 2"):
            jaxgsa.borgonovo.analyze(PROBLEM, X, Y, grid_size=1)

    def test_optimal_transport_reports_the_bad_argument_not_the_bad_sample(self):
        X, Y = self._poisoned()
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            jaxgsa.optimal_transport.analyze(PROBLEM, X, Y, max_iter=0)

    def test_sobol_reports_the_bad_argument_not_the_bad_sample(self):
        sr = _sobol_design()
        Y = np.asarray(sr.samples)[:, 0].copy()
        Y[2] = np.nan
        with pytest.raises(ValueError, match="ci_method must be one of"):
            jaxgsa.sobol.analyze(sr, jnp.asarray(Y), ci_method=cast(Any, "nope"))

    def test_the_policy_itself_is_still_settled_first(self):
        """A misspelled on_invalid is reported before anything else."""
        X, Y = self._poisoned()
        with pytest.raises(ValueError, match="on_invalid must be one of"):
            jaxgsa.pawn.analyze(PROBLEM, X, Y, n_bins=1, on_invalid=cast(Any, "nope"))


class TestTheGatesComeFromTheRegistry:
    """The recommendation lists are generated, so they cannot drift."""

    def test_a_refusing_method_quotes_the_generated_list(self):
        R = np.eye(D)
        R[0, 1] = R[1, 0] = 0.4
        correlated = PROBLEM.with_correlation(R)
        X = jnp.asarray(jaxgsa.sampling.monte_carlo(correlated, 64, seed=0))
        Y = X[:, 0]
        with pytest.raises(ValueError) as excinfo:
            jaxgsa.pce.analyze(correlated, X, Y)
        assert _correlation_tolerant_methods() in str(excinfo.value)

    def test_the_shapley_pce_message_quotes_it_too(self):
        """The one hand-written list is now generated like the others."""
        R = np.eye(D)
        R[0, 1] = R[1, 0] = 0.4
        correlated = PROBLEM.with_correlation(R)
        X = jnp.asarray(jaxgsa.sampling.monte_carlo(correlated, 64, seed=0))
        Y = X[:, 0]
        with pytest.raises(ValueError) as excinfo:
            jaxgsa.shapley.analyze(correlated, X, Y, backend="pce")
        assert _correlation_tolerant_methods() in str(excinfo.value)


class TestACleanSampleStaysSilent:
    """The new warnings must not fire on data that is fine.

    Six methods gained the zero-variance check. A sample with a varying
    output must be as quiet as it was before.
    """

    @pytest.mark.parametrize(
        "name", ["borgonovo", "dgsm", "pawn", "optimal_transport", "hsic", "pce"]
    )
    def test_no_jaxgsa_warning_on_a_varying_output(self, name):
        X, Y = _xy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if name == "dgsm":
                jaxgsa.dgsm.analyze(PROBLEM, lambda x: jnp.sum(x**2), X)
            elif name == "hsic":
                jaxgsa.hsic.analyze(PROBLEM, X, Y, key=jax.random.key(0))
            else:
                getattr(jaxgsa, name).analyze(PROBLEM, X, Y)
        # Two warnings here are not about the data, and both fire on a clean
        # sample by design. The single-precision one is about the arithmetic,
        # and hsic and vkoga both raise it. The dgsm lower-bound one is about
        # the problem definition: PROBLEM has uniform marginals, so
        # lower_bound is outside the case Kucherenko & Song prove, whatever
        # the sample looks like.
        exempt = ("single precision", "untruncated Gaussian")
        left = [
            w
            for w in caught
            if issubclass(w.category, JaxgsaWarning)
            and not any(phrase in str(w.message) for phrase in exempt)
        ]
        assert left == []


class TestDgsmsEarlyGateSettlesWhatPrepareSettles:
    """``dgsm`` runs prepare()'s own early steps itself, then calls prepare().

    ``dgsm`` cannot use the one-call form directly: on its autodiff path the
    model output is what ``prepare`` would validate, and it does not exist
    until the model has been differentiated. It runs ``resolve_policy``,
    ``check_scalars`` and ``gates`` -- the same three calls ``prepare`` makes
    internally, before it touches any array -- and then calls ``prepare``
    once, in full, with the output in hand. These pin that running the three
    pieces first and then the whole settles the same things a single
    ``prepare`` call would.
    """

    def _spec(self):
        return methods()["borgonovo"]

    def _early_gate(self, spec, problem, *, on_invalid="raise"):
        """The three pieces a two-phase caller runs before it has ``Y``."""
        unit = spec.invalid_unit
        resolve_policy(
            on_invalid, method="test", unit=unit, allow_drop=unit is not InvalidUnit.CURVE
        )
        check_scalars(())
        if not spec.is_design_based:
            gates(spec, problem, method="test")

    def test_the_early_gate_then_prepare_settles_what_one_call_settles(self):
        X, Y = _xy()
        spec = self._spec()
        whole = prepare(spec, PROBLEM, Y, X=X)
        self._early_gate(spec, PROBLEM)
        two_phase = prepare(spec, PROBLEM, Y, X=X)
        assert two_phase.method == whole.method
        assert two_phase.policy == whole.policy
        assert two_phase.Y3.shape == whole.Y3.shape
        np.testing.assert_array_equal(np.asarray(two_phase.Y), np.asarray(whole.Y))
        np.testing.assert_array_equal(two_phase.keep, whole.keep)

    def test_the_early_gate_settles_the_policy_before_any_data_exists(self):
        with pytest.raises(ValueError, match="on_invalid must be one of"):
            self._early_gate(self._spec(), PROBLEM, on_invalid=cast(Any, "nope"))

    def test_the_early_gate_applies_the_gates(self):
        correlated = jaxgsa.Problem(
            ("x1", "x2", "x3"),
            ((0.0, 1.0),) * D,
            correlation=np.eye(D) + np.eye(D, k=1) * 0.4 + np.eye(D, k=-1) * 0.4,
        )
        with pytest.raises(ValueError, match="correlation"):
            self._early_gate(methods()["dgsm"], correlated)


class TestExtraArraysRideWithY:
    """A third sample-axis array is checked with ``Y`` and dropped with it.

    ``dgsm`` passes its Jacobian this way. Keeping the drop in one place is
    the point: a method that compacted its own third array could silently
    disagree with the rows the report names.
    """

    def _prepare(self, Y, **kwargs):
        return prepare(methods()["borgonovo"], PROBLEM, Y, on_invalid=cast(Any, "drop"), **kwargs)

    def test_a_non_finite_extra_condemns_its_row(self):
        X, Y = _xy()
        jac = np.zeros((N, D))
        jac[7, 1] = np.nan
        with pytest.warns(JaxgsaWarning, match="dropped"):
            ctx = self._prepare(Y, X=X, extra={"jac": jnp.asarray(jac)})
        assert ctx.invalid.unit_indices == (7,)
        assert ctx.Y.shape[0] == N - 1
        assert ctx.extra["jac"].shape == (N - 1, D)

    def test_the_extra_is_compacted_with_the_same_mask(self):
        X, Y = _xy()
        Y = np.asarray(Y).copy()
        Y[3] = np.nan
        jac = np.arange(N * D, dtype=float).reshape(N, D)
        with pytest.warns(JaxgsaWarning, match="dropped"):
            ctx = self._prepare(jnp.asarray(Y), X=X, extra={"jac": jnp.asarray(jac)})
        np.testing.assert_allclose(np.asarray(ctx.extra["jac"]), np.delete(jac, 3, axis=0))

    def test_a_context_with_no_extra_carries_an_empty_mapping(self):
        X, Y = _xy()
        assert prepare(methods()["borgonovo"], PROBLEM, Y, X=X).extra == {}


class TestAFloat64OutputIsNotTruncatedInSilence:
    """Truncation to float32 is announced only when it really costs something.

    ``jnp.asarray`` on a float64 array returns float32 when ``jax_enable_x64``
    is off, with no signal of any kind. jaxgsa keeps float32 as the default
    and takes on exactly one obligation in exchange: never destroy precision
    silently.

    A float64 dtype on its own does not mean precision is being destroyed.
    NumPy makes float64 arrays by default, so an ordinary ``Y = model(X)``
    written from a documentation example is float64 without anyone asking for
    double precision, and its values fit in float32 with room to spare. The
    warning therefore fires on the round trip, not on the dtype: it speaks
    only when casting to float32 and back changes the numbers by more than
    float32's own resolution.
    """

    def _double(self):
        X = np.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, N, seed=0), dtype=np.float64)
        return X, np.asarray(X[:, 0] + 2.0 * X[:, 1] ** 2, dtype=np.float64)

    def _downcast_warnings(self, Y, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prepare(methods()["borgonovo"], PROBLEM, Y, **kwargs)
        # Match the category and the fix the message names, not its prose.
        # Filtering on a phrase means a reworded warning silently stops
        # being captured and every test here passes for the wrong reason.
        return [
            w
            for w in caught
            if issubclass(w.category, JaxgsaWarning) and "jax_enable_x64" in str(w.message)
        ]

    def test_an_ordinary_float64_output_stays_quiet(self):
        """The path someone takes when they copy an example from the docs.

        NumPy hands them float64 and every value fits in float32. Nothing is
        lost that float32 could have held, so there is nothing to say.
        """
        X, Y = self._double()
        assert Y.dtype == np.float64  # the premise of the test
        assert self._downcast_warnings(Y, X=X) == []

    def test_an_output_too_small_for_float32_warns_naming_the_fix(self):
        X, Y = self._double()
        tiny = Y * 1e-300  # below the float32 range: narrowing gives zero
        caught = self._downcast_warnings(tiny, X=X)
        if bool(getattr(jax.config, "jax_enable_x64", False)):
            assert caught == []  # nothing is lost, so there is nothing to say
            return
        assert len(caught) == 1
        message = str(caught[0].message)
        assert issubclass(caught[0].category, JaxgsaWarning)
        assert "Y" in message and "jax_enable_x64" in message

    def test_the_round_trip_rule_reads_the_values_not_the_dtype(self):
        """Both directions of the rule, on the check itself.

        Overflow is tested here rather than through ``analyze()`` because a
        float64 value above the float32 maximum becomes ``inf`` on the device,
        and the non-finite check raises before the run gets any further.
        """
        if bool(getattr(jax.config, "jax_enable_x64", False)):
            pytest.skip("nothing is downcast under x64")
        ordinary = np.linspace(-3.0, 3.0, 64, dtype=np.float64)
        assert not _loses_precision(ordinary)
        assert not _loses_precision(np.asarray(ordinary, dtype=np.float32))
        assert not _loses_precision([1.0, 2.0, 3.0])  # no dtype at all
        assert _loses_precision(ordinary * 1e39)  # overflows to inf
        assert _loses_precision(ordinary * 1e-300)  # underflows to zero

    def test_measuring_the_loss_raises_no_numpy_warning_of_its_own(self):
        """The overflow is the thing being measured, not a fault to report."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _loses_precision(np.full(8, 1e300, dtype=np.float64))
        assert [w for w in caught if "overflow" in str(w.message)] == []

    def test_a_float32_output_stays_quiet(self):
        X, Y = self._double()
        assert self._downcast_warnings(np.asarray(Y, dtype=np.float32), X=X) == []

    def test_one_warning_per_analysis(self):
        """One sentence for the call, not one per array it looked at."""
        if bool(getattr(jax.config, "jax_enable_x64", False)):
            pytest.skip("nothing is downcast under x64")
        X, Y = self._double()
        caught = self._downcast_warnings(Y * 1e-300, X=X, extra={"jac": np.zeros((N, D))})
        assert len(caught) == 1

    def test_the_input_matrix_is_not_a_candidate(self):
        """X is float64 because jaxgsa's own samplers build it that way.

        Warning about it would fire on essentially every call and name a
        downcast the caller cannot avoid without turning x64 on for the
        design as well. The same goes for the companion arrays: dgsm's
        autodiff path puts a synthetic float64 flag column in ``extra``.
        """
        X, Y = self._double()
        Y32 = np.asarray(Y, dtype=np.float32)
        # Values that would fail the round trip on their own, and stay inside
        # the problem bounds so nothing else in the preamble objects.
        lossy_X = np.array(X, dtype=np.float64)
        lossy_X[0, 0] = 1e-300
        lossy_jac = np.full((N, D), 1e-300, dtype=np.float64)
        assert _loses_precision(lossy_X) and _loses_precision(lossy_jac)
        assert self._downcast_warnings(Y32, X=lossy_X) == []
        assert self._downcast_warnings(Y32, X=lossy_X, extra={"jac": lossy_jac}) == []

    def test_the_output_keeps_the_caller_s_dtype(self):
        """Whatever JAX is configured for, the preamble does not widen it."""
        X, Y = self._double()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", JaxgsaWarning)
            ctx = prepare(methods()["borgonovo"], PROBLEM, np.asarray(Y, dtype=np.float32), X=X)
        assert ctx.Y.dtype == jnp.float32
        assert ctx.Y3.dtype == jnp.float32


class TestTheUnitClipSurvivesFloat32:
    """``1 - UNIT_CLIP`` has to be a number below 1 in the dtype it runs in.

    The clip keeps a unit coordinate off 0 and 1 so an inverse CDF stays
    finite. At ``UNIT_CLIP = 1e-12`` the upper half of it was a no-op in
    float32 — ``1 - 1e-12`` rounds straight back to 1.0 there — while the
    float64 host twin of the same transform clipped properly.
    """

    def test_the_upper_bound_is_below_one_in_both_dtypes(self):
        for dtype in (np.float32, np.float64):
            low, high = unit_clip_bounds(UNIT_CLIP, dtype)
            assert low == UNIT_CLIP
            assert dtype(high) < dtype(1.0)

    def test_float64_keeps_the_nominal_bound(self):
        """float64 can represent it, so nothing is given away."""
        assert unit_clip_bounds(UNIT_CLIP, np.float64) == (UNIT_CLIP, 1.0 - UNIT_CLIP)

    def test_a_saturated_normal_cdf_does_not_become_infinite(self):
        """The reason the clip exists, at the dtype that used to skip it."""
        problem = jaxgsa.Problem.from_dict(
            {"g": jaxgsa.GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0)}
        )
        X = jnp.asarray([[40.0], [-40.0]], dtype=jnp.float32)
        u = cdf_to_unit_interval(X, problem)
        assert np.all(np.asarray(u) < 1.0)
        assert np.all(np.isfinite(np.asarray(jax.scipy.special.ndtri(u))))
