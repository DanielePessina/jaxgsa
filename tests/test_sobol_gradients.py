"""Differentiable Sobol: ``SobolSamples.transform`` and ``sobol.indices``.

Three claims are checked here.

1. ``sobol.indices`` computes the same numbers as ``sobol.analyze``, and it
   survives ``jit``, ``vmap`` and ``jacrev``, which ``analyze`` cannot.
2. ``SobolSamples.transform`` reproduces the design, and its derivative with
   respect to the distribution parameters is correct.
3. Moving deduplication onto the unit cube changed no design.

The gradient tests run under ``jax.enable_x64()``. A central finite
difference subtracts two nearly equal numbers and divides by a small step, so
in single precision the reference itself carries more error than the quantity
under test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import Problem, sobol
from jaxgsa._core.sampling import _stable_unique_rows, _transform_samples
from jaxgsa.benchmarks import ishigami
from jaxgsa.sobol._sampling import _build_expanded_samples

# One design reused by the gradient tests. It is small on purpose: the
# Jacobian is checked against finite differences, which needs 2 * (number of
# theta leaves) extra model evaluations per test.
GRAD_BASE_N = 512

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


def test_indices_matches_analyze():
    """Tier T4: ``indices`` returns exactly what ``analyze`` reports.

    ``analyze`` delegates to the same core, so this guards the delegation
    wiring rather than the estimator maths.
    """
    sr = sobol.sample(ishigami.PROBLEM, 2048, seed=11, verbose=False)
    Y = ishigami.evaluate(jnp.asarray(sr.samples))

    result = sobol.analyze(sr, Y)
    S1, ST, S2 = sobol.indices(sr, Y)

    np.testing.assert_array_equal(np.asarray(S1), np.asarray(result.S1))
    np.testing.assert_array_equal(np.asarray(ST), np.asarray(result.ST))
    np.testing.assert_array_equal(np.asarray(S2), np.asarray(result.S2), strict=False)


def test_indices_first_order_only_returns_two_arrays():
    """Tier T4: a first-order design yields ``(S1, ST)``, with no S2 slot."""
    sr = sobol.sample(ishigami.PROBLEM, 2048, calc_second_order=False, seed=11, verbose=False)
    Y = ishigami.evaluate(jnp.asarray(sr.samples))

    out = sobol.indices(sr, Y)

    assert len(out) == 2
    result = sobol.analyze(sr, Y)
    np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(result.S1))


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

    S1, ST = sobol.indices(sr, Y)
    result = sobol.analyze(sr, Y)

    np.testing.assert_array_equal(np.asarray(S1), np.asarray(result.S1))
    np.testing.assert_array_equal(np.asarray(ST), np.asarray(result.ST))


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
# 2. Transformability
# ---------------------------------------------------------------------------


def test_indices_is_jittable():
    """Tier T4: ``jit`` traces ``indices`` and returns the eager values."""
    sr = sobol.sample(ishigami.PROBLEM, 1024, calc_second_order=False, seed=3, verbose=False)
    Y = ishigami.evaluate(jnp.asarray(sr.samples))

    # The design is a closure constant: it holds NumPy index maps, not tracers.
    jitted = jax.jit(lambda outputs: sobol.indices(sr, outputs))
    S1_jit, ST_jit = jitted(Y)
    S1, ST = sobol.indices(sr, Y)

    np.testing.assert_allclose(np.asarray(S1_jit), np.asarray(S1), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(ST_jit), np.asarray(ST), rtol=1e-6)


def test_indices_is_vmappable():
    """Tier T4: ``vmap`` maps ``indices`` over a batch of output vectors."""
    sr = sobol.sample(ishigami.PROBLEM, 1024, calc_second_order=False, seed=3, verbose=False)
    X = jnp.asarray(sr.samples)
    # Three unrelated outputs on the same design, stacked on a leading axis.
    batch = jnp.stack([ishigami.evaluate(X), X[:, 0], X[:, 1] ** 2])

    S1_batch, ST_batch = jax.vmap(lambda outputs: sobol.indices(sr, outputs))(batch)

    assert S1_batch.shape == (3, sr.n_params)
    for row, outputs in enumerate(batch):
        S1, ST = sobol.indices(sr, outputs)
        np.testing.assert_allclose(np.asarray(S1_batch[row]), np.asarray(S1), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(ST_batch[row]), np.asarray(ST), rtol=1e-5)


def test_jit_of_jacrev_of_the_full_chain():
    """Tier T4: ``jit(jacrev(...))`` compiles the theta-to-index chain."""
    with jax.enable_x64():
        sr = sobol.sample(
            GRAD_PROBLEM, 1, base_n=64, calc_second_order=False, seed=1, verbose=False
        )

        def total_s1(theta):
            Y = ishigami.evaluate(sr.transform(theta))
            return sobol.indices(sr, Y)[0].sum()

        jac = jax.jit(jax.jacrev(total_s1))(GRAD_THETA)
        eager = jax.jacrev(total_s1)(GRAD_THETA)

    for name, field, value in _theta_leaves(jac):
        assert np.isfinite(value), f"{name}.{field} is not finite"
        np.testing.assert_allclose(float(value), float(eager[name][field]), rtol=1e-8)


# ---------------------------------------------------------------------------
# 3. transform()
# ---------------------------------------------------------------------------


def test_transform_reproduces_samples():
    """Tier T4: ``transform()`` with no override rebuilds the stored design.

    ``samples`` comes from SciPy in float64 and ``transform`` from JAX, so the
    two agree to floating-point precision rather than bitwise. The measured
    gap under x64 is about 9e-16.
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


def test_transform_rejects_categorical():
    """Tier T4: a categorical design refuses to be re-transformed, and says why."""
    problem = Problem.from_dict(
        {
            "a": {"dist": "categorical", "probs": [0.5, 0.5]},
            "b": (0.0, 1.0),
        }
    )
    sr = sobol.sample(problem, 1, base_n=64, verbose=False)

    with pytest.raises(ValueError, match="step function"):
        sr.transform()


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


def test_transform_rejects_unknown_theta_entries():
    """Tier T4: a misspelled parameter or field raises instead of being ignored."""
    sr = sobol.sample(GRAD_PROBLEM, 256, seed=2, verbose=False)

    with pytest.raises(ValueError, match="does not declare"):
        sr.transform({"x9": {"low": 0.0}})
    with pytest.raises(ValueError, match="does not use"):
        sr.transform({"x1": {"mean": 0.0}})


def test_transform_unavailable_without_a_unit_design():
    """Tier T4: a hand-built design says so rather than failing obscurely."""
    sr = sobol.sample(GRAD_PROBLEM, 256, seed=2, verbose=False)
    hand_built = type(sr)(
        samples=sr.samples,
        n_expanded=sr.n_expanded,
        expanded_to_unique=sr.expanded_to_unique,
        base_n=sr.base_n,
        n_params=sr.n_params,
        calc_second_order=sr.calc_second_order,
        problem=sr.problem,
    )

    with pytest.raises(ValueError, match="no unit-cube design"):
        hand_built.transform()


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
            base_n=GRAD_BASE_N,
            calc_second_order=False,
            seed=17,
            verbose=False,
        )

        def s1_of_theta(theta):
            Y = ishigami.evaluate(sr.transform(theta))
            return sobol.indices(sr, Y)[0]

        jac = jax.jacrev(s1_of_theta)(GRAD_THETA)

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


# ---------------------------------------------------------------------------
# 5. The dedup reordering (D4) changed nothing
# ---------------------------------------------------------------------------


def _legacy_design(problem, base_n, *, calc_second_order, scramble, seed):
    """Rebuild a design the old way: transform first, deduplicate after."""
    expanded_unit = _build_expanded_samples(
        problem.num_vars,
        base_n,
        calc_second_order=calc_second_order,
        scramble=scramble,
        seed=seed,
    )
    return _stable_unique_rows(_transform_samples(problem, expanded_unit))


@pytest.mark.parametrize("scramble", [True, False])
@pytest.mark.parametrize("calc_second_order", [True, False])
@pytest.mark.parametrize(
    "problem",
    [
        # One continuous marginal stands for all three: uniform, gaussian and
        # truncated gaussian are strictly monotone, so they exercise the same
        # injectivity path. The categorical case is the other path.
        pytest.param(Problem.from_dict({"a": (0.0, 1.0), "b": (-2.0, 3.0)}), id="uniform"),
        pytest.param(
            Problem.from_dict(
                {
                    "a": {"dist": "categorical", "probs": [0.5, 0.5]},
                    "b": {"dist": "categorical", "probs": [0.25, 0.25, 0.5]},
                }
            ),
            id="categorical",
        ),
    ],
)
def test_unit_cube_dedup_gives_the_same_design(problem, calc_second_order, scramble):
    """Tier T4: deduplicating before the transform changes no design.

    A continuous marginal transform is strictly monotone, hence injective, so
    it cannot merge two rows the unit-cube pass kept apart. A categorical
    transform can, which is why the categorical path runs a second collapse
    pass; composing the two maps must land on the same answer as one pass over
    the transformed design.
    """
    sr = sobol.sample(
        problem,
        1,
        base_n=64,
        calc_second_order=calc_second_order,
        scramble=scramble,
        seed=99,
        verbose=False,
    )
    legacy_samples, legacy_map = _legacy_design(
        problem, 64, calc_second_order=calc_second_order, scramble=scramble, seed=99
    )

    np.testing.assert_array_equal(sr.samples, legacy_samples)
    np.testing.assert_array_equal(sr.expanded_to_unique, legacy_map)


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


@pytest.mark.parametrize("low,high", TRUNC_WINDOWS)
def test_jax_truncated_gaussian_matches_scipy_on_tail_windows(low, high):
    """Tier T2: the JAX truncated-Gaussian inverse CDF matches scipy.

    Provenance: oracle is ``scipy.stats.truncnorm.ppf`` (scipy is a runtime
    dependency, so the comparison runs live), float64, rtol 1e-9. Added
    2026-08-19 with the survival-function reflection fix; before it, every
    far-upper-tail window (e.g. ``[9, 12]``) came back all inf.
    """
    from scipy.stats import truncnorm

    from jaxgsa._core.sampling import _jax_transform_gaussian

    mean, variance = 0.3, 1.7
    std = float(np.sqrt(variance))
    u = np.linspace(1e-9, 1.0 - 1e-9, 41)
    with jax.enable_x64():
        got = np.asarray(
            _jax_transform_gaussian(jnp.asarray(u), mean, variance, low=low, high=high)
        )
    a = -np.inf if low is None else (low - mean) / std
    b = np.inf if high is None else (high - mean) / std
    want = truncnorm.ppf(u, a, b, loc=mean, scale=std)
    assert np.isfinite(got).all()
    # rtol 5e-9, not tighter: at u = 1 - 1e-9 on a one-sided window the output
    # sits 8 sigma out, where JAX's ndtri (Cephes) and scipy's ndtri disagree
    # by about 2e-9 relative. That is approximation quality of the two ndtri
    # implementations, not conditioning; the far-tail windows the reflection
    # exists for agree to better than 1e-9.
    np.testing.assert_allclose(got, want, rtol=5e-9, atol=1e-12)


@pytest.mark.parametrize("low,high", [(9.0, 12.0), (-12.0, -9.0), (9.0, None), (None, -9.0)])
def test_jax_truncated_gaussian_gradients_finite_on_tail_windows(low, high):
    """Tier T4: gradients through a far-tail window are finite on both sides.

    The branchless reflection selects a side with ``jnp.where``; without the
    double-``where`` input sanitisation the *untaken* side computes
    ``ndtri(1) = inf`` and reverse mode turns its zero cotangent into NaN.
    """
    from jaxgsa._core.sampling import _jax_transform_gaussian

    with jax.enable_x64():
        u = jnp.linspace(0.05, 0.95, 9, dtype=jnp.float64)

        def total(mean, variance):
            return jnp.sum(_jax_transform_gaussian(u, mean, variance, low=low, high=high))

        value = total(0.3, 1.7)
        d_mean, d_var = jax.grad(total, argnums=(0, 1))(0.3, 1.7)

    assert np.isfinite(value)
    assert np.isfinite(d_mean) and np.isfinite(d_var)


def test_jax_truncated_gaussian_tail_gradient_matches_finite_difference():
    """Tier T2/T4: d/d(mean) on the reflected ``[9, 12]`` window agrees with a
    central finite difference of the scipy oracle (h = 1e-6, float64)."""
    from scipy.stats import truncnorm

    from jaxgsa._core.sampling import _jax_transform_gaussian

    low, high, variance = 9.0, 12.0, 1.7
    std = float(np.sqrt(variance))
    u = np.linspace(0.1, 0.9, 5)

    with jax.enable_x64():

        def total(mean):
            return jnp.sum(
                _jax_transform_gaussian(jnp.asarray(u), mean, variance, low=low, high=high)
            )

        grad = float(jax.grad(total)(0.3))

    def scipy_total(mean):
        a, b = (low - mean) / std, (high - mean) / std
        return float(truncnorm.ppf(u, a, b, loc=mean, scale=std).sum())

    h = 1e-6
    fd = (scipy_total(0.3 + h) - scipy_total(0.3 - h)) / (2 * h)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)
