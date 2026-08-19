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
    in_open_interval,
    one_of,
    prepare,
    prepare_scalars,
    require,
)
from jaxgsa._core.registry import methods
from jaxgsa._core.validation import _correlation_tolerant_methods

D = 3
PROBLEM = jaxgsa.Problem(("x1", "x2", "x3"), ((0.0, 1.0),) * D)
N = 128


def _xy(*, constant: bool = False, n: int = N):
    """A clean given-data sample, optionally with a constant output."""
    X = np.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n, seed=0))
    Y = np.full(n, 3.0) if constant else X[:, 0] + 2.0 * X[:, 1] ** 2
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
        return lambda: jaxgsa.sobol.analyze(sobol_design, jnp.full(sobol_design.n_runs, 3.0))
    if name == "morris":
        morris_design = _morris_design()
        return lambda: jaxgsa.morris.analyze(morris_design, jnp.full(morris_design.n_runs, 3.0))
    if name == "efast":
        efast_design = jaxgsa.efast.sample(PROBLEM, 400, seed=0)
        return lambda: jaxgsa.efast.analyze(efast_design, jnp.full(efast_design.n_runs, 3.0))
    if name == "kucherenko":
        kuch_design = jaxgsa.kucherenko.sample(PROBLEM, 64, seed=0)
        return lambda: jaxgsa.kucherenko.analyze(kuch_design, jnp.full(kuch_design.n_runs, 3.0))
    if name == "dgsm":
        return lambda: jaxgsa.dgsm.analyze(PROBLEM, lambda x: jnp.asarray(3.0), X)
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
        # The single-precision warning is about the arithmetic, not the data:
        # it fires on a clean sample too, and hsic and vkoga both raise it.
        left = [
            w
            for w in caught
            if issubclass(w.category, JaxgsaWarning) and "single precision" not in str(w.message)
        ]
        assert left == []


class TestTheTwoHalvesAreTheOnePreamble:
    """``prepare`` is ``prepare_scalars`` plus ``with_data``, not a second copy.

    ``dgsm`` cannot use the one-call form: on its autodiff path the model
    output is what the preamble would validate, and it does not exist until
    the model has been differentiated. It calls the halves instead, and these
    pin that the halves do what the whole does.
    """

    def _spec(self):
        return methods()["borgonovo"]

    def test_the_halves_settle_what_the_one_call_form_settles(self):
        X, Y = _xy()
        spec = self._spec()
        whole = prepare(spec, PROBLEM, Y, X=X)
        halves = prepare_scalars(spec, PROBLEM).with_data(Y, X=X)
        assert halves.method == whole.method
        assert halves.policy == whole.policy
        assert halves.Y3.shape == whole.Y3.shape
        np.testing.assert_array_equal(np.asarray(halves.Y), np.asarray(whole.Y))
        np.testing.assert_array_equal(halves.keep, whole.keep)

    def test_the_scalar_half_settles_the_policy_before_any_data_exists(self):
        with pytest.raises(ValueError, match="on_invalid must be one of"):
            prepare_scalars(self._spec(), PROBLEM, on_invalid=cast(Any, "nope"))

    def test_the_scalar_half_applies_the_gates(self):
        correlated = jaxgsa.Problem(
            ("x1", "x2", "x3"),
            ((0.0, 1.0),) * D,
            correlation=np.eye(D) + np.eye(D, k=1) * 0.4 + np.eye(D, k=-1) * 0.4,
        )
        with pytest.raises(ValueError, match="correlation"):
            prepare_scalars(methods()["dgsm"], correlated)


class TestExtraArraysRideWithY:
    """A third sample-axis array is checked with ``Y`` and dropped with it.

    ``dgsm`` passes its Jacobian this way. Keeping the drop in one place is
    the point: a method that compacted its own third array could silently
    disagree with the rows the report names.
    """

    def _preamble(self):
        return prepare_scalars(methods()["borgonovo"], PROBLEM, on_invalid=cast(Any, "drop"))

    def test_a_non_finite_extra_condemns_its_row(self):
        X, Y = _xy()
        jac = np.zeros((N, D))
        jac[7, 1] = np.nan
        with pytest.warns(JaxgsaWarning, match="dropped"):
            ctx = self._preamble().with_data(Y, X=X, extra={"jac": jnp.asarray(jac)})
        assert ctx.invalid.unit_indices == (7,)
        assert ctx.Y.shape[0] == N - 1
        assert ctx.extra["jac"].shape == (N - 1, D)

    def test_the_extra_is_compacted_with_the_same_mask(self):
        X, Y = _xy()
        Y = np.asarray(Y).copy()
        Y[3] = np.nan
        jac = np.arange(N * D, dtype=float).reshape(N, D)
        with pytest.warns(JaxgsaWarning, match="dropped"):
            ctx = self._preamble().with_data(jnp.asarray(Y), X=X, extra={"jac": jnp.asarray(jac)})
        np.testing.assert_allclose(np.asarray(ctx.extra["jac"]), np.delete(jac, 3, axis=0))

    def test_a_context_with_no_extra_carries_an_empty_mapping(self):
        X, Y = _xy()
        assert prepare(methods()["borgonovo"], PROBLEM, Y, X=X).extra == {}
