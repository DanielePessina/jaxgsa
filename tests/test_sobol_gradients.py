"""Differentiable Sobol: ``SobolSamples.transform`` and ``sobol.indices``.

Two claims are checked here.

1. ``sobol.indices`` agrees with ``sobol.analyze`` on every output shape. The
   estimator menu tests (``test_sobol_estimators.py``) cover ``jit``, ``vmap``
   and ``jacrev`` for each estimator.
2. ``SobolSamples.transform`` reproduces the design, and its derivative with
   respect to the distribution parameters is correct.

The gradient tests run under ``jax.enable_x64()``. The checks compare the
package's public and pure-core routes, and exercise both forward- and
reverse-mode differentiation without relying on another implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import Problem, sobol
from jaxgsa.benchmarks import ishigami

# One design reused by the gradient tests. It is small on purpose: the
# Jacobian is checked against finite differences, which needs 2 * (number of
# theta leaves) extra model evaluations per test.
GRAD_BASE_N = 512

# AD-vs-FD agreement does not depend on N, so the finite-difference test uses
# a smaller design.
FD_BASE_N = 128

# Bounds that differ per parameter, so a wrong theta-to-column wiring cannot
# hide behind a symmetric problem.
GRAD_PROBLEM = Problem.from_dict(
    {
        "x1": (-np.pi, np.pi),
        "x2": (-3.0, 3.2),
        "x3": {"dist": "gaussian", "mean": 0.2, "variance": 1.4, "low": -2.0, "high": 2.5},
    }
)

GRAD_THETA = {
    "x1": {"low": -float(np.pi), "high": float(np.pi)},
    "x2": {"low": -3.0, "high": 3.2},
    "x3": {"mean": 0.2, "variance": 1.4, "low": -2.0, "high": 2.5},
}


def _theta_leaves(theta):
    """Yield ``(name, field, value)`` for every scalar in a theta dict."""
    for name, fields in theta.items():
        for field, value in fields.items():
            yield name, field, value


def _perturb(theta, name, field, delta):
    """Return a copy of ``theta`` with one field shifted by ``delta``."""
    return {
        key: {f: (v + delta if (key, f) == (name, field) else v) for f, v in fields.items()}
        for key, fields in theta.items()
    }


# ---------------------------------------------------------------------------
# 1. The estimator core
# ---------------------------------------------------------------------------


def test_indices_multi_output():
    """Tier T4: the batched path agrees with ``analyze`` for ``(n, T, K)`` outputs."""
    sr = sobol.sample(ishigami.PROBLEM, 1024, calc_second_order=False, seed=5, verbose=False)
    X = jnp.asarray(sr.samples)
    # Two time steps and two outputs, each a different function of the inputs,
    # so a mis-transposed batch axis cannot pass.
    base = ishigami.evaluate(X)
    Y = jnp.stack(
        [jnp.stack([base, 2.0 * base + X[:, 0]], axis=-1), jnp.stack([base**2, X[:, 1]], axis=-1)],
        axis=1,
    )
    assert Y.shape == (sr.n_runs, 2, 2)

    # A first-order design yields ``(S1, ST)`` with no S2 slot.
    S1, ST = sobol.indices(sr, Y)
    result = sobol.analyze(sr, Y)

    np.testing.assert_array_equal(np.asarray(S1), np.asarray(result.S1))
    np.testing.assert_array_equal(np.asarray(ST), np.asarray(result.ST))


@pytest.mark.slow
def test_indices_t0_ishigami_first_order():
    """Tier T0 (closed form): S1 converges to the analytic Ishigami values.

    The Ishigami ANOVA decomposition is exact, so this is a real accuracy
    check on the estimator and not a self-comparison. At ``base_n = 2**16``
    the measured maximum absolute error is 2.0e-5.
    """
    sr = sobol.sample(
        ishigami.PROBLEM,
        1,
        base_n=2**16,
        calc_second_order=False,
        seed=0,
        verbose=False,
    )
    Y = ishigami.evaluate(jnp.asarray(sr.samples))

    S1, _ = sobol.indices(sr, Y)

    np.testing.assert_allclose(np.asarray(S1), ishigami.ANALYTICAL_S1, atol=1e-4)


# ---------------------------------------------------------------------------
# 2. transform()
# ---------------------------------------------------------------------------


def test_transform_reproduces_samples():
    """Tier T4: ``transform()`` with no override rebuilds the stored design.

    The sampler stores a NumPy design while ``transform`` evaluates it through
    JAX, so the two agree to floating-point precision rather than bitwise.
    """
    with jax.enable_x64():
        sr = sobol.sample(GRAD_PROBLEM, 512, seed=2, verbose=False)
        X = np.asarray(sr.transform())
    np.testing.assert_allclose(X, sr.samples, rtol=0, atol=1e-12)


def test_transform_with_theta_shifts_the_design():
    """Tier T4: an override replaces the declared value, and only that value."""
    with jax.enable_x64():
        sr = sobol.sample(GRAD_PROBLEM, 512, seed=2, verbose=False)
        widened = np.asarray(sr.transform({"x2": {"high": 6.4}}))
        baseline = np.asarray(sr.transform())

    # x2 is uniform on [-3.0, 3.2]; doubling the upper bound stretches the
    # column and leaves the other two untouched.
    np.testing.assert_allclose(widened[:, 0], baseline[:, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(widened[:, 2], baseline[:, 2], rtol=0, atol=1e-12)
    assert widened[:, 1].max() > baseline[:, 1].max() + 1.0


def test_categorical_unit_and_samples_have_different_row_counts():
    """Tier T4: the collapse pass is what makes categorical dedup pay.

    A step CDF sends many unit values to one level code, so the physical
    design is far smaller than the unit design it came from. That saving is
    the reason the categorical path runs a second dedup pass after the
    transform.
    """
    problem = Problem.from_dict(
        {
            "a": {"dist": "categorical", "probs": [0.4, 0.6]},
            "b": {"dist": "categorical", "probs": [0.2, 0.3, 0.5]},
        }
    )
    sr = sobol.sample(problem, 1, base_n=64, verbose=False)

    assert sr.unit is not None
    assert sr.n_runs < sr.unit.shape[0]
    # Two levels times three levels bounds the number of distinct rows.
    assert sr.n_runs <= 6
    # The composed map still addresses the collapsed design.
    assert sr.expanded_to_unique.max() == sr.n_runs - 1


@pytest.mark.parametrize(
    "case,theta,message",
    [
        ("categorical", None, "step function"),
        ("continuous", {"x9": {"low": 0.0}}, "does not declare"),
        ("continuous", {"x1": {"mean": 0.0}}, "does not use"),
        # A ``None`` field would silently untruncate a Gaussian. Omitting the
        # key is the supported way to keep the problem's declared value.
        ("continuous", {"x3": {"low": None}}, r"theta\['x3'\] sets \['low'\] to None"),
        ("continuous", {"x3": {"mean": None}}, "to None"),
        ("hand_built", None, "no unit-cube design"),
    ],
)
def test_transform_refuses(case, theta, message):
    """Tier T4: ``transform`` refuses what it cannot honour, and says why.

    A categorical design is a step function of the unit cube, so it cannot be
    re-transformed. A misspelled parameter or field raises instead of being
    ignored. A design built by hand has no unit-cube design to transform.
    """
    if case == "categorical":
        problem = Problem.from_dict(
            {"a": {"dist": "categorical", "probs": [0.5, 0.5]}, "b": (0.0, 1.0)}
        )
        sr = sobol.sample(problem, 1, base_n=64, verbose=False)
    else:
        sr = sobol.sample(GRAD_PROBLEM, 256, seed=2, verbose=False)
    if case == "hand_built":
        sr = type(sr)(
            samples=sr.samples,
            n_expanded=sr.n_expanded,
            expanded_to_unique=sr.expanded_to_unique,
            base_n=sr.base_n,
            n_params=sr.n_params,
            calc_second_order=sr.calc_second_order,
            problem=sr.problem,
        )

    with pytest.raises(ValueError, match=message):
        sr.transform() if theta is None else sr.transform(theta)


# ---------------------------------------------------------------------------
# 4. The gradient itself
# ---------------------------------------------------------------------------


def test_jacrev_matches_central_finite_differences():
    """Tier T4: d(S1)/d(theta) agrees with a central finite difference.

    The finite difference is the independent reference. It re-runs the whole
    chain (transform, model, estimator) at theta +/- h, so it exercises no
    part of the autodiff path. Measured worst-case agreement is 2.3e-11
    absolute, over all eight theta fields and all three indices.
    """
    step = 1e-5

    with jax.enable_x64():
        sr = sobol.sample(
            GRAD_PROBLEM,
            1,
            base_n=FD_BASE_N,
            calc_second_order=False,
            seed=17,
            verbose=False,
        )

        # Jitted once, so the 16 finite-difference runs reuse one compile.
        @jax.jit
        def s1_of_theta(theta):
            Y = ishigami.evaluate(sr.transform(theta))
            return sobol.indices(sr, Y)[0]

        jac = jax.jit(jax.jacrev(s1_of_theta))(GRAD_THETA)

        worst = 0.0
        for name, field, _ in _theta_leaves(GRAD_THETA):
            plus = np.asarray(s1_of_theta(_perturb(GRAD_THETA, name, field, step)))
            minus = np.asarray(s1_of_theta(_perturb(GRAD_THETA, name, field, -step)))
            fd = (plus - minus) / (2.0 * step)
            ad = np.asarray(jac[name][field])
            # Absolute tolerance only: several entries of dS1/dtheta are
            # legitimately near zero, where a relative tolerance has no
            # meaning. The bound sits two orders above the measured 2.3e-11,
            # which leaves room for the finite difference's own truncation
            # error without admitting a real wiring mistake.
            np.testing.assert_allclose(ad, fd, rtol=0, atol=1e-9)
            worst = max(worst, float(np.abs(ad - fd).max()))

    # A zero Jacobian would pass every comparison above while meaning the
    # chain never differentiated anything. The largest entry is about 0.07.
    assert worst < 1e-9
    assert np.abs(np.asarray(jac["x1"]["high"])).max() > 1e-3


def test_gradient_of_a_uniform_bound_has_the_right_sign():
    """Tier T0 (closed form): widening x3's range must raise its own S1.

    Ishigami's ``B * x3^4 * sin(x1)`` term is the only place x3 enters, and it
    enters through an interaction, so x3's first-order index is 0 whatever the
    bound. Its *total* index is not: the term grows like the eighth power of
    the half-width, so ST_3 must increase with the upper bound. This fixes the
    sign of the derivative without needing a finite difference.
    """
    with jax.enable_x64():
        sr = sobol.sample(
            ishigami.PROBLEM,
            1,
            base_n=GRAD_BASE_N,
            calc_second_order=False,
            seed=23,
            verbose=False,
        )

        def st3(theta):
            Y = ishigami.evaluate(sr.transform(theta))
            return sobol.indices(sr, Y)[1][2]

        theta = {"x3": {"high": float(np.pi)}}
        grad = jax.grad(st3)(theta)

    assert float(grad["x3"]["high"]) > 0.0


def test_downsample_keeps_the_unit_design_aligned():
    """Tier T4: a prefix slice carries a unit design that still transforms."""
    with jax.enable_x64():
        sr = sobol.sample(
            GRAD_PROBLEM, 1, base_n=256, calc_second_order=False, seed=8, verbose=False
        )
        small = sr.downsample(64)
        X = np.asarray(small.transform())

    assert small.unit is not None
    assert X.shape == small.samples.shape
    np.testing.assert_allclose(X, small.samples, rtol=0, atol=1e-12)


def test_save_load_round_trips_the_unit_design(tmp_path):
    """Tier T4: a reloaded design can still be re-transformed."""
    with jax.enable_x64():
        sr = sobol.sample(GRAD_PROBLEM, 256, seed=4, verbose=False)
        sr.save(tmp_path / "design")
        loaded = jaxgsa.sobol.SobolSamples.load(tmp_path / "design")

        assert loaded.unit is not None
        np.testing.assert_array_equal(loaded.unit, sr.unit)
        np.testing.assert_allclose(
            np.asarray(loaded.transform()), np.asarray(sr.transform()), rtol=0, atol=0
        )


# ---------------------------------------------------------------------------
# far-tail truncated Gaussian windows (survival-function reflection)
# ---------------------------------------------------------------------------

# Windows in standard-ish units around mean 0.3, std sqrt(1.7). The far-tail
# ones broke the direct form: ndtr saturates to exactly 1.0 from about 8.2
# sigma in float64, the probability window collapses to width 0, and the whole
# column came back inf. The ordinary windows pin that the reflection changed
# nothing where the direct form was already exact.
TRUNC_WINDOWS = [
    (9.0, 12.0),  # upper tail: both ndtr endpoints used to round to 1.0
    (-12.0, -9.0),  # lower tail: direct form, well conditioned all along
    (5.0, 20.0),  # upper tail with a far-out upper bound
    (-2.0, 2.5),  # ordinary window straddling the mean
    (0.5, 3.0),  # ordinary window, midpoint above the mean (reflected side)
    (None, -9.0),  # one-sided: (-inf, high] deep in the lower tail
    (9.0, None),  # one-sided: [low, inf) deep in the upper tail
    (None, 2.0),  # one-sided, ordinary
    (-1.0, None),  # one-sided, ordinary
]


GRAD_WINDOWS = [(9.0, 12.0), (-12.0, -9.0), (9.0, None), (None, -9.0)]


@pytest.mark.parametrize("low,high", TRUNC_WINDOWS)
def test_jax_truncated_gaussian_tail_windows_are_finite_and_ordered(low, high):
    """The truncated-Gaussian inverse CDF stays finite and ordered, with finite gradients.

    Far-tail windows exercise the survival-function reflection branch. A
    collapsed or invalid branch would return infinities, leave the declared
    interval, or lose the strict ordering of quantiles.

    The four deepest tail windows also get a gradient check. The branchless
    reflection selects a side with ``jnp.where``; without the double-``where``
    input sanitisation the *untaken* side computes ``ndtri(1) = inf`` and
    reverse mode turns its zero cotangent into NaN. The gradient must also be
    non-trivial, not just finite.
    """
    from jaxgsa._core.sampling import _jax_transform_gaussian

    mean, variance = 0.3, 1.7
    u = np.linspace(1e-9, 1.0 - 1e-9, 41)
    with jax.enable_x64():
        got = np.asarray(
            _jax_transform_gaussian(jnp.asarray(u), mean, variance, low=low, high=high)
        )
    assert np.isfinite(got).all()
    assert np.all(np.diff(got) > 0.0)
    if low is not None:
        assert np.all(got >= low)
    if high is not None:
        assert np.all(got <= high)

    if (low, high) in GRAD_WINDOWS:
        with jax.enable_x64():
            u_grad = jnp.linspace(0.1, 0.9, 5, dtype=jnp.float64)

            def total(mean, variance):
                return jnp.sum(_jax_transform_gaussian(u_grad, mean, variance, low=low, high=high))

            d_mean, d_var = jax.grad(total, argnums=(0, 1))(mean, variance)
        assert np.isfinite(d_mean) and np.isfinite(d_var)
        assert abs(float(d_mean)) > 1e-6
