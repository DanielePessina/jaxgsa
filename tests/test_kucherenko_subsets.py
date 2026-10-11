"""Exact Shapley effects and PME on the ``subsets="all"`` Kucherenko design."""

from __future__ import annotations

import warnings
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _linear_gaussian import (
    A_COEF,
    A_SUBSET4,
    GAUSS_PROBLEM,
    R_GAUSS,
    R_SUBSET4,
    analytic_closed,
    analytic_costs,
    gaussian_sampler,
    pme_by_permutations,
    shapley_by_permutations,
)

import jaxgsa
from jaxgsa import kucherenko
from jaxgsa.kucherenko._subsets import (
    pme_from_total,
    shapley_from_cost,
    subset_matrix,
)
from jaxgsa.problem import Problem
from jaxgsa.shapley import ShapleyResult

N = 8192


def _gauss_problem(D: int, R: np.ndarray) -> Problem:
    base = Problem.from_dict(
        {f"x{i + 1}": {"dist": "gaussian", "mean": 0.0, "variance": 1.0} for i in range(D)}
    )
    return base.with_correlation(R)


PROBLEM4 = _gauss_problem(4, R_SUBSET4)
PROBLEM3 = GAUSS_PROBLEM.with_correlation(R_GAUSS)


def _cost_arrays(a: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytic closed and total cost on the bitmask layout ``1 .. 2^D - 1``."""
    D = a.shape[0]
    closed, total = analytic_costs(a, R)
    members = subset_matrix(D)
    key = [frozenset(np.flatnonzero(row).tolist()) for row in members]
    return np.array([closed[k] for k in key]), np.array([total[k] for k in key])


@pytest.fixture(scope="module")
def design4():
    return kucherenko.sample(PROBLEM4, base_n=N, subsets="all", seed=3, verbose=False)


@pytest.fixture(scope="module")
def result4(design4):
    Y = jnp.asarray(design4.samples @ A_SUBSET4)
    return kucherenko.analyze(design4, Y, verbose=False)


# --- design shape and refusals ---------------------------------------------


def test_design_has_two_to_the_d_blocks(design4):
    assert design4.subsets == "all"
    assert design4.n_blocks == 16
    assert design4.n_runs == N * 2**4
    assert design4.samples.shape == (N * 16, 4)
    assert "subsets='all'" in repr(design4)


def test_blocks_keep_the_subset_columns_from_block_a(design4):
    a = design4.samples[:N]
    for mask in (1, 5, 6, 14):
        block = design4.samples[N * (1 + mask) : N * (2 + mask)]
        kept = [j for j in range(4) if (mask >> j) & 1]
        free = [j for j in range(4) if not (mask >> j) & 1]
        np.testing.assert_array_equal(block[:, kept], a[:, kept])
        assert not np.allclose(block[:, free], a[:, free])


def test_block_b_is_an_independent_joint_draw(design4):
    a, b = design4.samples[:N], design4.samples[N : 2 * N]
    np.testing.assert_allclose(np.corrcoef(a.T, b.T)[:4, 4:].diagonal(), 0.0, atol=0.03)
    np.testing.assert_allclose(np.corrcoef(b.T), R_SUBSET4, atol=0.04)


def test_unknown_subsets_mode_is_refused():
    with pytest.raises(ValueError, match="subsets"):
        kucherenko.sample(PROBLEM3, base_n=64, subsets="some", verbose=False)  # type: ignore[arg-type]


def test_more_than_twelve_parameters_are_refused():
    problem = Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(13)})
    with pytest.raises(ValueError, match=r"2\^13") as excinfo:
        kucherenko.sample(problem, base_n=1024, subsets="all", verbose=False)
    message = str(excinfo.value)
    assert str(1024 * 2**13) in message
    assert "given data" in message


def test_one_parameter_is_refused():
    problem = Problem.from_dict({"x": (0.0, 1.0)})
    with pytest.raises(ValueError, match="at least 2"):
        kucherenko.sample(problem, base_n=64, subsets="all", verbose=False)


def test_singleton_design_refuses_shapley_and_pme():
    design = kucherenko.sample(PROBLEM3, base_n=256, seed=1, verbose=False)
    res = kucherenko.analyze(design, jnp.asarray(design.samples @ A_COEF), verbose=False)
    assert res.closed is None and res.subset_total is None and res.subsets is None
    for call in (res.shapley, res.pme):
        with pytest.raises(ValueError, match="subsets='all'"):
            call()


def test_default_design_is_unchanged():
    design = kucherenko.sample(PROBLEM3, base_n=256, seed=1, verbose=False)
    assert design.subsets == "singletons"
    assert design.n_blocks == 7
    assert design.n_runs == 256 * 7


def test_default_estimator_matches_a_plain_numpy_formula():
    """S1 and ST on the default design equal the textbook single-loop sums."""
    design = kucherenko.sample(PROBLEM3, base_n=1024, seed=2, verbose=False)
    y = design.samples @ A_COEF
    res = kucherenko.analyze(design, jnp.asarray(y), verbose=False)
    n, D = 1024, 3
    f = y.reshape(2 * D + 1, n)
    var = f[0].var()
    for i in range(D):
        s1 = (np.mean(f[0] * f[1 + i]) - f[0].mean() * f[1 + i].mean()) / var
        st = 0.5 * np.mean((f[0] - f[1 + D + i]) ** 2) / var
        np.testing.assert_allclose(float(res.S1[i]), s1, atol=2e-6)
        np.testing.assert_allclose(float(res.ST[i]), st, atol=2e-6)


# --- T0: linear Gaussian closed forms --------------------------------------


def test_subset_indices_match_the_closed_form(result4):
    closed, total = _cost_arrays(A_SUBSET4, R_SUBSET4)
    assert result4.closed.shape == (15,)
    np.testing.assert_allclose(np.asarray(result4.closed), closed, atol=0.015)
    np.testing.assert_allclose(np.asarray(result4.subset_total), total, atol=0.015)
    # The full set is exactly 1.
    assert float(result4.closed[-1]) == pytest.approx(1.0)
    assert float(result4.subset_total[-1]) == pytest.approx(1.0)


def test_subsets_matrix_is_ordered_by_bitmask(result4):
    members = np.asarray(result4.subsets)
    assert members.shape == (15, 4) and members.dtype == bool
    np.testing.assert_array_equal(members[0], [True, False, False, False])  # mask 1
    np.testing.assert_array_equal(members[4], [True, False, True, False])  # mask 5
    np.testing.assert_array_equal(members[-1], [True] * 4)
    # Position m - 1 of closed is the subset with bitmask m.
    expected = analytic_closed(A_SUBSET4, R_SUBSET4, (0, 2))
    assert float(result4.closed[4]) == pytest.approx(expected, abs=0.015)


def test_singleton_indices_come_from_the_singleton_blocks(result4):
    for i in range(4):
        assert float(result4.S1[i]) == float(result4.closed[(1 << i) - 1])
        assert float(result4.ST[i]) == float(result4.subset_total[(1 << i) - 1])


def test_shapley_matches_the_closed_form(result4):
    _, total = analytic_costs(A_SUBSET4, R_SUBSET4)
    exact = shapley_by_permutations(total, 4)
    sh = result4.shapley()
    assert isinstance(sh, ShapleyResult)
    np.testing.assert_allclose(np.asarray(sh.Sh), exact, atol=0.015)
    np.testing.assert_allclose(float(sh.Sh.sum()), 1.0, atol=1e-5)
    assert sh.backend == "exact"
    np.testing.assert_allclose(np.asarray(sh.S1), np.asarray(result4.S1))
    # The ordering S1 <= Sh does not hold under correlation, but Sh <= 1 does.
    assert (np.asarray(sh.Sh) > -0.01).all()


def test_closed_and_total_costs_give_the_same_shapley(result4):
    total_cost = np.asarray(result4.shapley().Sh)
    closed_cost = np.asarray(result4.shapley(cost="closed").Sh)
    np.testing.assert_allclose(closed_cost, total_cost, atol=0.01)
    with pytest.raises(ValueError, match="cost"):
        result4.shapley(cost="bogus")  # type: ignore[arg-type]


def test_pme_matches_the_closed_form(result4):
    _, total = analytic_costs(A_SUBSET4, R_SUBSET4)
    exact = pme_by_permutations(total, 4)
    pme = result4.pme()
    np.testing.assert_allclose(np.asarray(pme.Sh), exact, atol=0.015)
    np.testing.assert_allclose(float(pme.Sh.sum()), 1.0, atol=1e-5)
    assert (np.asarray(pme.Sh) >= 0).all()
    # PME differs from Shapley here, so the test would catch a swapped rule.
    assert np.abs(exact - shapley_by_permutations(total, 4)).max() > 0.01


def test_d3_design_and_shapley():
    design = kucherenko.sample(PROBLEM3, base_n=N, subsets="all", seed=4, verbose=False)
    res = kucherenko.analyze(design, jnp.asarray(design.samples @ A_COEF), verbose=False)
    _, total = analytic_costs(A_COEF, R_GAUSS)
    np.testing.assert_allclose(
        np.asarray(res.shapley().Sh), shapley_by_permutations(total, 3), atol=0.01
    )
    np.testing.assert_allclose(np.asarray(res.pme().Sh), pme_by_permutations(total, 3), atol=0.01)


def test_all_design_first_order_and_total_match_the_closed_form():
    from _linear_gaussian import analytic_indices

    design = kucherenko.sample(PROBLEM3, base_n=N, subsets="all", seed=4, verbose=False)
    res = kucherenko.analyze(design, jnp.asarray(design.samples @ A_COEF), verbose=False)
    S_TC, S_TU, _ = analytic_indices(A_COEF, R_GAUSS)
    np.testing.assert_allclose(np.asarray(res.S1), S_TC, atol=0.01)
    np.testing.assert_allclose(np.asarray(res.ST), S_TU, atol=0.01)


# --- exogenous inputs -------------------------------------------------------


def test_exogenous_input_gets_zero_pme_and_positive_shapley():
    rho = 0.8
    R = np.array([[1.0, rho], [rho, 1.0]])
    problem = _gauss_problem(2, R)
    design = kucherenko.sample(problem, base_n=4096, subsets="all", seed=5, verbose=False)
    Y = jnp.asarray(design.samples[:, 0])  # Y = X1; X2 is exogenous
    res = kucherenko.analyze(design, Y, verbose=False)
    pme = np.asarray(res.pme().Sh)
    sh = np.asarray(res.shapley().Sh)
    assert pme[1] == 0.0
    assert pme[0] == pytest.approx(1.0)
    # Shapley's joke: Sh_2 = rho^2 / 2, Sh_1 = 1 - rho^2 / 2.
    np.testing.assert_allclose(sh, [1 - rho**2 / 2, rho**2 / 2], atol=0.01)
    assert sh[1] > 0.25


def test_exogenous_input_among_three():
    """X3 is independent and unused, X2 is correlated with X1 and unused."""
    R = np.array([[1.0, 0.7, 0.0], [0.7, 1.0, 0.0], [0.0, 0.0, 1.0]])
    design = kucherenko.sample(
        _gauss_problem(3, R), base_n=2048, subsets="all", seed=6, verbose=False
    )
    res = kucherenko.analyze(design, jnp.asarray(design.samples[:, 0] + 0.0), verbose=False)
    pme = np.asarray(res.pme().Sh)
    np.testing.assert_array_equal(pme[1:], [0.0, 0.0])
    assert pme[0] == pytest.approx(1.0)


# --- unit tests of the allocation rules ------------------------------------


def _random_game(D: int, seed: int) -> np.ndarray:
    """A random monotone positive game on masks 1 .. 2^D - 1 with v(D) = 1."""
    rng = np.random.default_rng(seed)
    v = np.zeros(2**D)
    for m in range(1, 2**D - 1):
        subs = [m & ~(1 << j) for j in range(D) if (m >> j) & 1]
        v[m] = max(v[s] for s in subs) + rng.uniform(0.02, 0.4) * (1 - max(v[s] for s in subs))
    v[-1] = 1.0
    return v[1:]


def _as_dict(v: np.ndarray, D: int) -> dict:
    out = {frozenset(): 0.0}
    for m in range(1, 2**D):
        out[frozenset(j for j in range(D) if (m >> j) & 1)] = float(v[m - 1])
    return out


@pytest.mark.parametrize("D", [2, 3, 4, 5])
def test_shapley_rule_matches_the_permutation_average(D):
    v = _random_game(D, seed=D)
    got = shapley_from_cost(v, D)
    np.testing.assert_allclose(got, shapley_by_permutations(_as_dict(v, D), D), atol=1e-12)
    assert got.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("D", [2, 3, 4, 5])
def test_pme_rule_matches_the_permutation_definition(D):
    v = _random_game(D, seed=10 + D)
    got = pme_from_total(v, D)
    np.testing.assert_allclose(got, pme_by_permutations(_as_dict(v, D), D), atol=1e-12)
    assert got.sum() == pytest.approx(1.0)
    assert (got >= 0).all()


def test_two_player_proportional_value_is_eq_14a():
    # v1 = 0.2, v2 = 0.5, v(D) = 1 -> PV_i = v_i + v_i/(v1+v2) (1 - v1 - v2).
    v = np.array([0.2, 0.5, 1.0])
    expected = [0.2 + 0.2 / 0.7 * 0.3, 0.5 + 0.5 / 0.7 * 0.3]
    np.testing.assert_allclose(pme_from_total(v, 2), expected, atol=1e-14)


def test_pme_null_coalition_is_the_limit_of_a_small_cost():
    """Eq. 15 equals the limit of the positive game with 0 replaced by eps."""
    D = 4
    v = _random_game(D, seed=3)
    for null in (0b0100, 0b0101):  # {2} and {0, 2}; the second needs {0}, {2} null too
        v2 = v.copy()
        nulls = [m for m in range(1, 2**D - 1) if m & ~null == 0]
        v2[np.array(nulls) - 1] = 0.0
        eps = 1e-10
        limit = np.where(v2 == 0.0, eps, v2)
        extended = pme_from_total(v2, D)
        np.testing.assert_allclose(extended, pme_from_total(limit, D), atol=1e-6)
        assert extended.sum() == pytest.approx(1.0)
        members = [j for j in range(D) if (null >> j) & 1]
        np.testing.assert_array_equal(extended[members], 0.0)


def test_batched_rules_keep_leading_axes():
    v = np.stack([_random_game(3, seed=s) for s in range(6)]).reshape(2, 3, 7)
    sh = shapley_from_cost(v, 3)
    pme = pme_from_total(v, 3)
    assert sh.shape == pme.shape == (2, 3, 3)
    np.testing.assert_allclose(sh[1, 2], shapley_from_cost(v[1, 2], 3))
    np.testing.assert_allclose(pme[1, 2], pme_from_total(v[1, 2], 3))


def test_nonfinite_cost_gives_nan():
    v = _random_game(3, seed=1)
    v[2] = np.nan
    assert np.isnan(pme_from_total(v, 3)).all()
    assert np.isnan(shapley_from_cost(v, 3)).all()


# --- intervals, shapes, aggregation -----------------------------------------


def test_shapley_interval_needs_kept_replicates():
    design = kucherenko.sample(PROBLEM3, base_n=1024, subsets="all", seed=7, verbose=False)
    Y = jnp.asarray(design.samples @ A_COEF)
    plain = kucherenko.analyze(design, Y, n_bootstrap=20, key=jax.random.key(0), verbose=False)
    assert plain.closed_conf is not None and plain.closed_conf.shape == (2, 7)
    assert plain.S1_conf is not None
    assert np.isnan(np.asarray(plain.shapley().Sh_conf)).all()

    kept = kucherenko.analyze(
        design, Y, n_bootstrap=60, key=jax.random.key(0), keep_replicates=True, verbose=False
    )
    for res in (kept.shapley(), kept.pme()):
        lo, hi = np.asarray(res.Sh_conf)
        assert np.isfinite(lo).all() and np.isfinite(hi).all()
        assert (lo <= np.asarray(res.Sh) + 1e-6).all() and (np.asarray(res.Sh) <= hi + 1e-6).all()
        assert ((hi - lo) < 0.15).all()
        assert res.ci is not None and res.ci.n_bootstrap == 60
    _, total = analytic_costs(A_COEF, R_GAUSS)
    exact = shapley_by_permutations(total, 3)
    lo, hi = np.asarray(kept.shapley().Sh_conf)
    assert ((lo - 0.02 <= exact) & (exact <= hi + 0.02)).all()


def test_multi_output_shapes_and_aggregate():
    design = kucherenko.sample(PROBLEM3, base_n=2048, subsets="all", seed=8, verbose=False)
    a2 = np.array([0.3, -1.0, 2.0])
    Y = jnp.asarray(np.stack([design.samples @ A_COEF, design.samples @ a2], axis=1))
    res = kucherenko.analyze(design, Y, verbose=False)
    assert res.closed.shape == (2, 7) and res.subset_total.shape == (2, 7)
    sh = res.shapley()
    assert sh.Sh.shape == (2, 3)
    pme = res.pme()
    assert pme.Sh.shape == (2, 3)
    agg = res.aggregate(weights="equal")
    assert agg.closed.shape == (1, 7)
    np.testing.assert_allclose(
        np.asarray(agg.shapley().Sh)[0], np.asarray(sh.Sh).mean(axis=0), atol=1e-6
    )
    var = np.asarray(res.variance)
    w = var / var.sum()
    agg_var = res.aggregate()
    np.testing.assert_allclose(
        np.asarray(agg_var.shapley().Sh)[0], (w[:, None] * np.asarray(sh.Sh)).sum(0), atol=1e-6
    )
    # Each output slice agrees with a single-output analysis of that column.
    solo = kucherenko.analyze(design, Y[:, 1], verbose=False)
    np.testing.assert_allclose(np.asarray(solo.shapley().Sh), np.asarray(sh.Sh)[1], atol=1e-6)


def test_aggregate_without_subsets_is_unchanged():
    design = kucherenko.sample(PROBLEM3, base_n=512, seed=1, verbose=False)
    Y = jnp.asarray(np.stack([design.samples @ A_COEF, design.samples[:, 0]], axis=1))
    agg = kucherenko.analyze(design, Y, verbose=False).aggregate()
    assert agg.closed is None and agg.S1.shape == (1, 3)


def test_invalid_rows_drop_whole_base_points():
    design = kucherenko.sample(PROBLEM3, base_n=256, subsets="all", seed=1, verbose=False)
    y = np.asarray(design.samples @ A_COEF)
    y[300] = np.nan  # a row of block 1 (block B), base point 44
    with pytest.raises(ValueError):
        kucherenko.analyze(design, jnp.asarray(y), verbose=False)
    with pytest.warns(jaxgsa.JaxgsaWarning, match="dropped 1 of 256"):
        res = kucherenko.analyze(design, jnp.asarray(y), on_invalid="drop", verbose=False)
    assert res.invalid.n_invalid == 1
    assert np.isfinite(np.asarray(res.shapley().Sh)).all()


# --- save / load ------------------------------------------------------------


def test_subset_design_round_trips(tmp_path):
    design = kucherenko.sample(PROBLEM3, base_n=64, subsets="all", seed=1, verbose=False)
    design.save(tmp_path / "design")
    back = kucherenko.KucherenkoSamples.load(tmp_path / "design.npz")
    assert back.subsets == "all"
    assert back.n_blocks == 8
    np.testing.assert_array_equal(back.samples, design.samples)
    res = kucherenko.analyze(back, jnp.asarray(back.samples @ A_COEF), verbose=False)
    assert res.shapley().Sh.shape == (3,)


def test_singleton_design_round_trip_keeps_the_default(tmp_path):
    design = kucherenko.sample(PROBLEM3, base_n=64, seed=1, verbose=False)
    design.save(tmp_path / "design")
    assert kucherenko.KucherenkoSamples.load(tmp_path / "design.npz").subsets == "singletons"


def test_old_saved_design_without_subsets_loads_as_singletons():
    from jaxgsa.kucherenko._sampling import _parse_subsets

    assert _parse_subsets(None) == "singletons"
    with pytest.raises(ValueError, match="Unknown subsets"):
        _parse_subsets("some")


def test_result_with_subsets_round_trips_through_netcdf(result4, tmp_path):
    xr = pytest.importorskip("xarray")
    ds = result4.to_dataset()
    assert ds["closed"].dims == ("subset",)
    assert ds["subset_members"].dims == ("subset", "param")
    assert ds["subset"].values[0] == "x1" and ds["subset"].values[4] == "x1+x3"
    path = tmp_path / "res.nc"
    ds.to_netcdf(path)
    back = xr.open_dataset(path)
    np.testing.assert_allclose(back["closed"].values, ds["closed"].values)
    np.testing.assert_allclose(back["subset_total"].values, ds["subset_total"].values)
    np.testing.assert_array_equal(back["subset_members"].values.astype(bool), result4.subsets)
    assert "closed=(15,)" in repr(result4)


def test_result_without_subsets_exports_as_before():
    design = kucherenko.sample(PROBLEM3, base_n=64, seed=1, verbose=False)
    res = kucherenko.analyze(design, jnp.asarray(design.samples @ A_COEF), verbose=False)
    assert "closed" not in res.to_dataset()
    assert "closed" not in repr(res)


# --- custom samplers --------------------------------------------------------


def test_custom_gaussian_sampler_gives_the_same_shapley():
    sampler = gaussian_sampler(R_SUBSET4)
    custom = PROBLEM4.with_correlation(None).with_sampler(sampler, sampler_dimension=4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        design = kucherenko.sample(custom, base_n=N, subsets="all", seed=9, verbose=False)
    assert design.dependence == "custom"
    res = kucherenko.analyze(design, jnp.asarray(design.samples @ A_SUBSET4), verbose=False)
    _, total = analytic_costs(A_SUBSET4, R_SUBSET4)
    np.testing.assert_allclose(
        np.asarray(res.shapley().Sh), shapley_by_permutations(total, 4), atol=0.015
    )
    np.testing.assert_allclose(np.asarray(res.pme().Sh), pme_by_permutations(total, 4), atol=0.015)


def test_custom_and_copula_designs_agree_within_error(result4):
    sampler = gaussian_sampler(R_SUBSET4)
    custom = PROBLEM4.with_correlation(None).with_sampler(sampler, sampler_dimension=4)
    design = kucherenko.sample(custom, base_n=N, subsets="all", seed=12, verbose=False)
    res = kucherenko.analyze(design, jnp.asarray(design.samples @ A_SUBSET4), verbose=False)
    np.testing.assert_allclose(
        np.asarray(res.shapley().Sh), np.asarray(result4.shapley().Sh), atol=0.02
    )


def test_permutation_helper_uses_every_ordering():
    # Guards the test oracle itself: 4! orderings, each adds up to v(D) = 1.
    _, total = analytic_costs(A_SUBSET4, R_SUBSET4)
    assert len(list(permutations(range(4)))) == 24
    assert shapley_by_permutations(total, 4).sum() == pytest.approx(1.0)
    assert pme_by_permutations(total, 4).sum() == pytest.approx(1.0)


# --- T3: R 'sensitivity' package as a black box ------------------------------


@pytest.mark.slow
def test_shapley_matches_r_shapley_linear_gaussian(result4):
    from _r_oracle import run_r

    out = run_r(
        "sh <- as.numeric(shapleyLinearGaussian(Beta = as.numeric(Beta), Sigma = Sigma))",
        arrays={"Beta": A_SUBSET4, "Sigma": R_SUBSET4},
        packages=("sensitivity", "igraph"),
        outputs=("sh",),
    )
    np.testing.assert_allclose(np.asarray(result4.shapley().Sh), out["sh"], atol=0.015)
    np.testing.assert_allclose(
        np.asarray(result4.shapley(cost="closed").Sh), out["sh"], atol=0.015
    )


@pytest.mark.slow
def test_pme_matches_r_pme_knn_loosely():
    from _r_oracle import run_r

    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(R_GAUSS)
    X = rng.standard_normal((3000, 3)) @ L.T
    Y = X @ A_COEF
    out = run_r(
        """
        x <- pme_knn(model = NULL, X = as.data.frame(X), method = "knn", n.knn = 3)
        tell(x, as.numeric(Y))
        pme <- as.numeric(x$PME)
        """,
        arrays={"X": X, "Y": Y},
        packages=("sensitivity", "gtools", "RANN"),
        outputs=("pme",),
    )
    design = kucherenko.sample(PROBLEM3, base_n=8192, subsets="all", seed=2, verbose=False)
    res = kucherenko.analyze(design, jnp.asarray(design.samples @ A_COEF), verbose=False)
    np.testing.assert_allclose(np.asarray(res.pme().Sh), out["pme"].ravel(), atol=0.08)
