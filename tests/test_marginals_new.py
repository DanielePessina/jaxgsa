"""Tests for the lognormal, Weibull, triangular and beta marginals.

Tier T0 (closed form) for the inverse CDFs, CDFs, moments and the product
model Sobol indices: every reference value comes from ``scipy.stats`` or from
the closed-form product-model formulas (research-marginals-maths.md, Q9b).
Tier T4 (internal consistency) for the JAX transform, dict form and
save/load round trips.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import jaxgsa
from jaxgsa import BetaSpec, LognormalSpec, Problem, TriangularSpec, WeibullSpec
from jaxgsa._core.sampling import _inverse_transform_samples, _transform_samples
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa.efast import EFASTResult
from jaxgsa.morris import MorrisResult
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
    x = spec.ppf(U_GRID)
    np.testing.assert_allclose(x, dist.ppf(U_GRID), rtol=1e-10)
    np.testing.assert_allclose(spec.cdf(x), dist.cdf(x), rtol=1e-10, atol=1e-15)
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


def test_beta_transform_computes_its_quantiles_once_and_jits(monkeypatch):
    """The host betaincinv runs once per design, not once per transform call."""
    from scipy import special

    problem = Problem.from_dict({"x0": BetaSpec(2.0, 5.0, -1.0, 2.0), "x1": (0.0, 1.0)})
    samples = jaxgsa.sobol.sample(problem, base_n=32, seed=4, verbose=False)
    calls = {"n": 0}
    real = special.betaincinv

    def counted(*args):
        calls["n"] += 1
        return real(*args)

    monkeypatch.setattr(special, "betaincinv", counted)
    with jax.enable_x64():
        eager = samples.transform()
        jitted = jax.jit(lambda th: samples.transform({"x0": th}))
        theta = {"low": jnp.asarray(-1.0), "high": jnp.asarray(2.0)}
        np.testing.assert_allclose(np.asarray(jitted(theta)), samples.samples, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(eager), samples.samples, rtol=1e-10)

        # d/d(high) of sum(x0) is sum(z) and d/d(low) is sum(1 - z), with z
        # the standard-beta quantile of the column.
        def total(th):
            return jnp.sum(samples.transform({"x0": th})[:, 0])

        grads = jax.jit(jax.grad(total))(theta)
        z = (samples.samples[:, 0] + 1.0) / 3.0
        np.testing.assert_allclose(float(grads["high"]), z.sum(), rtol=1e-8)
        np.testing.assert_allclose(float(grads["low"]), (1.0 - z).sum(), rtol=1e-8)
    assert calls["n"] == 1


def test_beta_dict_form_defaults_its_bounds_like_the_class():
    from_dict = Problem.from_dict({"x": {"dist": "beta", "a": 2, "b": 5}}).input_specs[0]
    assert from_dict == BetaSpec(2, 5)
    assert from_dict.support() == (0.0, 1.0)
    no_b: Any = {"dist": "beta", "a": 2}
    with pytest.raises(ValueError, match="missing 'b'"):
        Problem.from_dict({"x": no_b})


def test_dgsm_marginal_refusal_matches_the_problem_refusal():
    """poincare_constant and dgsm.analyze state the same reason and alternatives."""
    spec = CASES["lognormal"][0]
    with pytest.raises(ValueError) as single:
        jaxgsa.dgsm.poincare_constant(spec)
    problem = _mixed("lognormal")
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 32, seed=0))
    with pytest.raises(ValueError) as whole:
        jaxgsa.dgsm.analyze(problem, X, fn=_point_model, verbose=False)
    tail = str(whole.value).split("It does not support these marginals: ")[1].split(". ", 1)[1]
    assert str(single.value).endswith(tail)
    assert str(single.value).startswith("jaxgsa.dgsm.poincare_constant")


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
        np.testing.assert_allclose(spec.ppf(U_GRID), dist.ppf(U_GRID), rtol=1e-10)
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


# ---------------------------------------------------------------------------
# Method capability gates (ticket 03). Tier T4: behavioural contract.
# ---------------------------------------------------------------------------


def _mixed(name: str) -> Problem:
    """One parameter of the family under test, then two uniforms."""
    return Problem.from_dict({"x0": CASES[name][0], "x1": (0.0, 1.0), "x2": (0.0, 1.0)})


def _point_model(x):
    return jnp.sin(x[0]) + x[1] ** 2 + 0.1 * x[2]


@pytest.mark.parametrize("name", FAMILIES)
def test_dgsm_refuses_each_new_family_with_the_registry_message(name):
    problem = _mixed(name)
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 32, seed=0))
    cls = type(CASES[name][0]).__name__
    for call in (
        lambda: jaxgsa.dgsm.analyze(problem, X, fn=_point_model, verbose=False),
        lambda: jaxgsa.dgsm.indices(problem, X, fn=_point_model),
    ):
        # Refused before the lower_bound warning: no warning may escape first.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=cls) as info:
                call()
        message = str(info.value)
        assert "['x0']" in message
        assert "jaxgsa.sobol" in message  # names an accepting method
        assert "jaxgsa.dgsm," not in message  # never names itself


@pytest.mark.parametrize("name", FAMILIES)
def test_dgsm_poincare_constant_refuses_each_new_family(name):
    with pytest.raises(ValueError, match="does not support"):
        jaxgsa.dgsm.poincare_constant(CASES[name][0])


@pytest.mark.parametrize("name", ["lognormal", "weibull"])
def test_efast_refuses_heavy_tailed_families(name):
    with pytest.raises(ValueError, match=type(CASES[name][0]).__name__) as info:
        jaxgsa.efast.sample(_mixed(name), 400)
    message = str(info.value)
    assert "biased low" in message
    assert "jaxgsa.efast" not in message.split("instead:")[1]


@pytest.mark.parametrize("name", ["triangular", "beta"])
def test_efast_accepts_bounded_families(name):
    problem = _mixed(name)
    samples = jaxgsa.efast.sample(problem, 400)
    X = np.asarray(samples.samples)
    low, high = CASES[name][0].support()
    assert X[:, 0].min() >= low and X[:, 0].max() <= high
    Y = jnp.asarray(X[:, 0] + X[:, 1] + X[:, 2])
    result = jaxgsa.efast.analyze(samples, Y, verbose=False)
    assert isinstance(result, EFASTResult)
    assert np.all(np.isfinite(np.asarray(result.S1)))


def _hdmr_entry_points(problem: Problem) -> dict[str, Any]:
    """Every public route that runs the HDMR fit, on cheap settings."""
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 320, seed=0))
    Y = X[:, 0] + X[:, 1] + X[:, 2]
    return {
        "jaxgsa.hdmr.analyze": lambda: jaxgsa.hdmr.analyze(
            problem, X, Y, order=1, max_iter=3, verbose=False
        ),
        "jaxgsa.hdmr.indices": lambda: jaxgsa.hdmr.indices(problem, X, Y, order=1, max_iter=3),
        "jaxgsa.shapley.analyze with backend='hdmr'": lambda: jaxgsa.shapley.analyze(
            problem, X, Y, backend="hdmr", order=1, max_iter=3, verbose=False
        ),
        "jaxgsa.shapley.indices with backend='hdmr'": lambda: jaxgsa.shapley.indices(
            problem, X, Y, backend="hdmr", order=1, max_iter=3
        ),
    }


@pytest.mark.parametrize("name", ["lognormal", "weibull"])
def test_hdmr_refuses_heavy_tailed_families(name):
    """HDMR and the HDMR-backed Shapley route refuse an open-tailed input."""
    for method, call in _hdmr_entry_points(_mixed(name)).items():
        with pytest.raises(ValueError, match=type(CASES[name][0]).__name__) as info:
            call()
        message = str(info.value)
        assert message.startswith(method), (method, message)
        assert "biased low" in message
        accepting = message.split("instead:")[1]
        assert "jaxgsa.hdmr" not in accepting
        assert "jaxgsa.pce" in accepting
        if "backend='hdmr'" in method:
            assert message.endswith("For jaxgsa.shapley, pass backend='pce'.")


def test_transform_of_a_traced_beta_design_gives_the_host_error():
    """A traced design with a beta input names the host inverse CDF."""
    samples = jaxgsa.sobol.sample(
        _problem("beta"), base_n=16, calc_second_order=False, seed=0, verbose=False
    )

    def physical(unit: jax.Array) -> jax.Array:
        return dataclasses.replace(samples, unit=unit).transform()

    with pytest.raises(ValueError, match="concrete array, not a traced one"):
        jax.jit(physical)(jnp.asarray(samples.unit))


@pytest.mark.parametrize("name", ["lognormal", "weibull"])
def test_shapley_pce_backend_still_accepts_heavy_tailed_families(name):
    problem = _mixed(name)
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 256, seed=0))
    result = jaxgsa.shapley.analyze(problem, X, X[:, 0] + X[:, 1], order=2, verbose=False)
    assert isinstance(result, jaxgsa.shapley.ShapleyResult)
    assert np.all(np.isfinite(np.asarray(result.Sh)))


@pytest.mark.parametrize("name", ["triangular", "beta"])
def test_hdmr_accepts_bounded_families(name):
    for method, call in _hdmr_entry_points(_mixed(name)).items():
        # Every route returns an object with a per-parameter ST.
        assert np.all(np.isfinite(np.asarray(call().ST))), method


@pytest.mark.parametrize("name", ["lognormal", "weibull"])
def test_morris_squashes_only_the_open_upper_side(name):
    spec, dist, _, _ = CASES[name]
    problem = _mixed(name)
    q = 0.01
    sr = jaxgsa.morris.sample(problem, 20, seed=1, truncation_quantile=q, verbose=False)
    x0 = np.asarray(sr.samples)[:, 0]
    assert np.all(np.isfinite(x0))
    # The top level sits at the 1 - q quantile, the bottom one at the clip.
    np.testing.assert_allclose(x0.max(), dist.ppf(1.0 - q), rtol=1e-6)
    assert x0.min() < dist.ppf(1e-6)


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("method", ["trajectory", "radial"])
def test_morris_effects_use_the_squashed_step(name, method):
    """A model linear in each unit coordinate recovers its exact coefficients.

    ``F(x0)`` is the unit coordinate of ``x0``, so ``Y = 3 x1 + 5 F(x0)`` has
    elementary effects of exactly 5 on ``x0`` and 3 on ``x1`` once
    ``ee_delta`` follows the squash.
    """
    problem = _mixed(name)
    spec = CASES[name][0]
    sr = jaxgsa.morris.sample(
        problem, 30, seed=2, truncation_quantile=0.05, method=method, verbose=False
    )
    X = np.asarray(sr.samples)
    Y = jnp.asarray(5.0 * spec.cdf(X[:, 0]) + 3.0 * X[:, 1])
    res = jaxgsa.morris.analyze(sr, Y, verbose=False)
    assert isinstance(res, MorrisResult)
    np.testing.assert_allclose(np.asarray(res.mu_star)[:2], [5.0, 3.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(res.mu_star)[2], 0.0, atol=1e-9)
    with pytest.raises(ValueError, match="finite uniform bounds"):
        res.to_physical_units()


def test_morris_does_not_squash_bounded_families():
    """A large q would pull the ends to the 0.1 and 0.9 quantiles if applied."""
    for name in ("triangular", "beta"):
        dist = CASES[name][1]
        sr = jaxgsa.morris.sample(_mixed(name), 20, seed=1, truncation_quantile=0.1, verbose=False)
        x0 = np.asarray(sr.samples)[:, 0]
        # The ends are the unit clip's quantiles, not the squash's.
        np.testing.assert_allclose(
            [x0.min(), x0.max()], dist.ppf([1e-12, 1 - 1e-12]), rtol=1e-6, atol=1e-9
        )


@pytest.mark.parametrize("name", FAMILIES)
def test_sobol_to_morris_warns_for_open_tails_only(name):
    samples = jaxgsa.sobol.sample(_mixed(name), 256, verbose=False)
    open_tail = name in ("lognormal", "weibull")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        samples.to_morris(verbose=False)
    hits = [w for w in caught if "unbounded marginals" in str(w.message)]
    assert len(hits) == int(open_tail)
    if open_tail:
        assert f"({name})" in str(hits[0].message)
        assert "truncate_gaussians" not in str(hits[0].message)
