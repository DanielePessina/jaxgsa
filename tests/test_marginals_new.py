"""Tests for the lognormal, Weibull, triangular and beta marginals.

Tier T0 (closed form) for the inverse CDFs, CDFs, moments and the product
model Sobol indices: every reference value comes from ``scipy.stats`` or from
the closed-form product-model formulas (research-marginals-maths.md, Q9b).
Tier T4 (internal consistency) for the JAX transform, dict form and
save/load round trips.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import jaxgsa
from jaxgsa import BetaSpec, LognormalSpec, Problem, TriangularSpec, WeibullSpec
from jaxgsa._core import marginals
from jaxgsa._core.sampling import _inverse_transform_samples, _transform_samples
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa.pce import PCEResult
from jaxgsa.problem import ContinuousFamilySpec
from jaxgsa.sobol import SobolResult, SobolSamples

# (spec, the same law in scipy.stats, the dict form, a valid theta override)
CASES: dict[str, tuple[ContinuousFamilySpec, Any, Any, dict[str, float]]] = {
    "lognormal": (
        LognormalSpec(0.3, 0.5),
        stats.lognorm(s=0.5, scale=math.exp(0.3)),
        {"dist": "lognormal", "mu": 0.3, "sigma": 0.5},
        {"mu": 0.3, "sigma": 0.5},
    ),
    "weibull": (
        WeibullSpec(1.5, 2.0),
        stats.weibull_min(c=1.5, scale=2.0),
        {"dist": "weibull", "shape": 1.5, "scale": 2.0},
        {"shape": 1.5, "scale": 2.0},
    ),
    "triangular": (
        TriangularSpec(1.0, 1.6, 3.0),
        stats.triang(c=0.3, loc=1.0, scale=2.0),
        {"dist": "triangular", "low": 1.0, "mode": 1.6, "high": 3.0},
        {"low": 1.0, "mode": 1.6, "high": 3.0},
    ),
    "beta": (
        BetaSpec(2.0, 5.0, -1.0, 2.0),
        stats.beta(2.0, 5.0, loc=-1.0, scale=3.0),
        {"dist": "beta", "a": 2.0, "b": 5.0, "low": -1.0, "high": 2.0},
        {"low": -1.0, "high": 2.0},
    ),
}
FAMILIES = list(CASES)

# Unit grid including the clipped ends and points near 0 and 1.
U_GRID = np.concatenate(
    [
        [1e-12, 1e-10, 1e-6, 1e-3],
        np.linspace(0.01, 0.99, 99),
        [1 - 1e-3, 1 - 1e-6, 1 - 1e-10, 1 - 1e-12],
    ]
)


def _problem(name: str, n: int = 1) -> Problem:
    spec = CASES[name][0]
    return Problem.from_dict({f"x{i}": spec for i in range(n)})


@pytest.mark.parametrize("name", FAMILIES)
def test_host_ppf_and_cdf_match_scipy(name):
    spec, dist, _, _ = CASES[name]
    x = marginals.host_ppf(spec, U_GRID)
    np.testing.assert_allclose(x, dist.ppf(U_GRID), rtol=1e-10)
    np.testing.assert_allclose(marginals.host_cdf(spec, x), dist.cdf(x), rtol=1e-10, atol=1e-15)
    # The problem-level transform and its inverse round-trip the grid.
    problem = _problem(name)
    xs = _transform_samples(problem, U_GRID[:, None])
    np.testing.assert_allclose(xs[:, 0], dist.ppf(U_GRID), rtol=1e-10)
    back = _inverse_transform_samples(problem, xs)
    np.testing.assert_allclose(back[:, 0], U_GRID, rtol=1e-8, atol=1e-15)


@pytest.mark.parametrize("name", FAMILIES)
def test_jax_cdf_matches_scipy_and_flags_out_of_support(name):
    spec, dist, _, _ = CASES[name]
    x = dist.ppf(U_GRID[3:-3])
    with jax.enable_x64():
        for d in (1, 20):  # per-column path and grouped path
            problem = _problem(name, d)
            X = jnp.asarray(np.repeat(x[:, None], d, axis=1))
            U = np.asarray(cdf_to_unit_interval(X, problem))
            np.testing.assert_allclose(U, np.repeat(dist.cdf(x)[:, None], d, axis=1), rtol=1e-10)
        low, high = spec.support()
        below = low - 1.0
        U_out = np.asarray(cdf_to_unit_interval(jnp.asarray([[below]]), _problem(name)))
        assert U_out[0, 0] < 0.0
        if math.isfinite(high):
            U_out = np.asarray(cdf_to_unit_interval(jnp.asarray([[high + 1.0]]), _problem(name)))
            assert U_out[0, 0] > 1.0


@pytest.mark.parametrize("name", FAMILIES)
def test_jax_transform_matches_host_and_has_finite_gradients(name):
    _, _, _, theta_fields = CASES[name]
    problem = _problem(name)
    samples = jaxgsa.sobol.sample(problem, base_n=64, seed=1, verbose=False)
    # float32 default; atol covers cancellation in low + width * z near 0
    np.testing.assert_allclose(
        np.asarray(samples.transform()), samples.samples, rtol=1e-5, atol=1e-6
    )
    with jax.enable_x64():
        np.testing.assert_allclose(np.asarray(samples.transform()), samples.samples, rtol=1e-10)

        def total(theta):
            return jnp.sum(samples.transform({"x0": theta}))

        grads = jax.grad(total)({k: jnp.asarray(v) for k, v in theta_fields.items()})
        for field, value in grads.items():
            assert np.isfinite(float(value)), field
            assert float(value) != 0.0, field


def test_jax_transform_gradient_matches_finite_difference():
    problem = Problem.from_dict(
        {
            "a": LognormalSpec(0.3, 0.5),
            "b": WeibullSpec(1.5, 2.0),
            "c": TriangularSpec(1.0, 1.6, 3.0),
            "d": BetaSpec(2.0, 5.0, -1.0, 2.0),
        }
    )
    samples = jaxgsa.sobol.sample(problem, base_n=32, seed=2, verbose=False)
    theta0 = {name: CASES[fam][3] for name, fam in zip("abcd", FAMILIES)}
    with jax.enable_x64():

        def total(theta):
            return jnp.sum(samples.transform(theta) ** 2)

        grads = jax.grad(total)(jax.tree.map(jnp.asarray, theta0))
        h = 1e-6
        for name, fields in theta0.items():
            for field, value in fields.items():
                plus = {n: dict(f) for n, f in theta0.items()}
                minus = {n: dict(f) for n, f in theta0.items()}
                plus[name][field] = value + h
                minus[name][field] = value - h
                fd = (float(total(plus)) - float(total(minus))) / (2 * h)
                np.testing.assert_allclose(float(grads[name][field]), fd, rtol=1e-5)


def test_beta_transform_refuses_shape_parameters():
    samples = jaxgsa.sobol.sample(_problem("beta"), base_n=16, seed=0, verbose=False)
    for field in ("a", "b"):
        with pytest.raises(ValueError, match="beta shape parameter"):
            samples.transform({"x0": {field: 2.0}})


@pytest.mark.parametrize("name", FAMILIES)
def test_sampled_moments_match_scipy(name):
    _, dist, _, _ = CASES[name]
    samples = jaxgsa.sobol.sample(
        _problem(name), base_n=2**12, calc_second_order=False, seed=3, verbose=False
    )
    x = samples.samples[:, 0]
    mean, var = dist.stats(moments="mv")
    np.testing.assert_allclose(x.mean(), float(mean), rtol=5e-3)
    np.testing.assert_allclose(x.var(), float(var), rtol=3e-2)


def _product_indices(dists: list) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form S1 and ST of Y = prod X_i with independent inputs."""
    m1 = np.array([float(d.mean()) for d in dists])
    m2 = np.array([float(d.moment(2)) for d in dists])
    var = m2 - m1**2
    total_var = np.prod(m2) - np.prod(m1**2)
    s1 = np.array([var[i] * np.prod(np.delete(m1**2, i)) / total_var for i in range(len(dists))])
    st = np.array([var[i] * np.prod(np.delete(m2, i)) / total_var for i in range(len(dists))])
    return s1, st


PRODUCT_CASES = {
    **{name: [CASES[name][0]] * 3 for name in FAMILIES},
    "mixed": [CASES["lognormal"][0], CASES["weibull"][0], CASES["triangular"][0]],
}


@pytest.mark.parametrize("case", list(PRODUCT_CASES))
def test_product_model_sobol_indices_match_closed_form(case):
    specs = PRODUCT_CASES[case]
    lookup = {type(spec): CASES[name][1] for name, (spec, *_rest) in CASES.items()}
    dists = [lookup[type(spec)] for spec in specs]
    problem = Problem.from_dict({f"x{i}": spec for i, spec in enumerate(specs)})
    samples = jaxgsa.sobol.sample(
        problem, base_n=2**14, calc_second_order=False, seed=5, verbose=False
    )
    Y = np.prod(samples.samples, axis=1)
    result = jaxgsa.sobol.analyze(samples, jnp.asarray(Y), n_bootstrap=0, verbose=False)
    assert isinstance(result, SobolResult)
    s1, st = _product_indices(dists)
    np.testing.assert_allclose(np.asarray(result.S1).ravel(), s1, atol=0.03)
    np.testing.assert_allclose(np.asarray(result.ST).ravel(), st, atol=0.03)


@pytest.mark.parametrize("name", FAMILIES)
def test_dict_form_and_save_load_round_trip(name, tmp_path):
    spec, _, as_dict, _ = CASES[name]
    problem = Problem.from_dict({"x0": as_dict, "y": (0.0, 1.0)})
    assert problem.input_specs[0] == spec
    assert problem.bounds is None  # bounds stay uniform-only
    samples = jaxgsa.sobol.sample(problem, base_n=16, seed=0, verbose=False)
    path = tmp_path / "design"
    samples.save(path)
    loaded = SobolSamples.load(path)
    assert loaded.problem == problem
    assert loaded.problem.input_specs[0] == spec
    np.testing.assert_allclose(loaded.samples, samples.samples, rtol=1e-12)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: LognormalSpec(0.0, 0.0), "sigma > 0"),
        (lambda: LognormalSpec(float("nan"), 1.0), "finite mu"),
        (lambda: WeibullSpec(-1.0, 1.0), "shape > 0"),
        (lambda: WeibullSpec(1.0, 0.0), "scale > 0"),
        (lambda: TriangularSpec(1.0, 0.5, 2.0), "low <= mode <= high"),
        (lambda: TriangularSpec(1.0, 1.0, 1.0), "low < high"),
        (lambda: BetaSpec(0.0, 1.0), "a > 0"),
        (lambda: BetaSpec(1.0, -2.0), "b > 0"),
        (lambda: BetaSpec(1.0, 1.0, 2.0, 1.0), "low < high"),
        (lambda: BetaSpec(1.0, 1.0, 0.0, float("inf")), "finite high"),
    ],
)
def test_constructor_validation(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_degenerate_triangular_modes_are_accepted():
    for mode, c in ((0.0, 0.0), (2.0, 1.0)):
        spec = TriangularSpec(0.0, mode, 2.0)
        dist = stats.triang(c=c, loc=0.0, scale=2.0)
        np.testing.assert_allclose(marginals.host_ppf(spec, U_GRID), dist.ppf(U_GRID), rtol=1e-10)
        samples = jaxgsa.sobol.sample(Problem.from_dict({"t": spec}), base_n=16, verbose=False)
        with jax.enable_x64():
            grads = jax.grad(
                lambda m, samples=samples: jnp.sum(samples.transform({"t": {"mode": m}}))
            )(jnp.asarray(mode))
        assert np.isfinite(float(grads))


def test_dict_form_rejects_unknown_keys():
    bad_key: Any = {"dist": "lognormal", "mu": 0.0, "sigma": 1.0, "mean": 1.0}
    missing: Any = {"dist": "triangular", "low": 0.0, "high": 1.0}
    with pytest.raises(ValueError, match="unknown key"):
        Problem.from_dict({"x": bad_key})
    with pytest.raises(ValueError, match="missing 'mode'"):
        Problem.from_dict({"x": missing})


def test_custom_sampler_support_check_names_family():
    from jaxgsa._core.dependence import _check_support

    rows = np.array([[0.5, -1.0]])
    with pytest.raises(ValueError, match="lognormal"):
        _check_support(rows, [TriangularSpec(0.0, 0.5, 1.0), LognormalSpec(0.0, 1.0)], ["t", "l"])
    rows = np.array([[1.5, 1.0]])
    with pytest.raises(ValueError, match="triangular"):
        _check_support(rows, [TriangularSpec(0.0, 0.5, 1.0), LognormalSpec(0.0, 1.0)], ["t", "l"])


@pytest.mark.parametrize("name", FAMILIES)
def test_pce_recovers_product_model_indices(name):
    """PCE maps the new families through their CDF, then uses Legendre."""
    spec, dist, _, _ = CASES[name]
    problem = Problem.from_dict({f"x{i}": spec for i in range(2)})
    X = jaxgsa.sampling.monte_carlo(problem, 2048, seed=0)
    Y = jnp.asarray(np.prod(np.asarray(X), axis=1))
    result = jaxgsa.pce.analyze(problem, jnp.asarray(X), Y, order=6, verbose=False)
    assert isinstance(result, PCEResult)
    s1, st = _product_indices([dist, dist])
    np.testing.assert_allclose(np.asarray(result.S1).ravel(), s1, atol=0.03)
    np.testing.assert_allclose(np.asarray(result.ST).ravel(), st, atol=0.03)


def test_morris_refuses_open_tailed_families_for_now():
    with pytest.raises(TypeError, match="unsupported marginal LognormalSpec"):
        jaxgsa.morris.sample(_problem("lognormal", 2), 4, seed=0, verbose=False)
