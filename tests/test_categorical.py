"""Tests for categorical (unordered discrete) input support."""

import json

import numpy as np
import pytest

import jaxgsa
from jaxgsa import (
    borgonovo,
    dgsm,
    efast,
    hdmr,
    hsic,
    morris,
    optimal_transport,
    pawn,
    shapley,
    sobol,
)
from jaxgsa import pce as pce_mod
from jaxgsa._core.partition import _categorical_class_layout, _extract_categorical_codes
from jaxgsa._core.sampling import _inverse_transform_samples
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa.problem import CategoricalInputSpec, Problem, _categorical_dims

PROBS = [0.5, 0.3, 0.2]
OFFSETS = np.array([0.0, 2.0, -1.0])


def _mixed_problem():
    """One uniform and one 3-level categorical parameter."""
    return Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": PROBS},
        }
    )


def _mixed_data(n=4000, seed=1, noise=0.1):
    """Sample the mixed problem and evaluate the per-level-offset model."""
    problem = _mixed_problem()
    X = jaxgsa.sampling.monte_carlo(problem, n, seed=seed)
    codes = X[:, 1].astype(int)
    rng = np.random.default_rng(seed)
    Y = X[:, 0] + OFFSETS[codes] + noise * rng.standard_normal(n)
    return problem, X, Y


def _analytic_s1():
    """Closed-form S1 of (x1, c) for the per-level-offset model."""
    pr = np.array(PROBS)
    m = (pr * OFFSETS).sum()
    v_cat = (pr * (OFFSETS - m) ** 2).sum()
    v_x1 = 1.0 / 12.0
    v_y = v_cat + v_x1 + 0.01  # + noise variance
    return v_x1 / v_y, v_cat / v_y


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_categorical_spec_normalizes_and_stores_probs_and_labels():
    p = Problem.from_dict({"c": {"dist": "categorical", "probs": [0.25, 0.25, 0.5]}})
    spec = p.input_specs[0]
    assert spec[0] == "categorical"
    assert spec[5] is not None
    probs, labels = spec[5]
    assert probs == (0.25, 0.25, 0.5)
    assert labels == ("0", "1", "2")
    assert p.has_categorical_inputs is True
    assert p.has_non_uniform_inputs is True
    assert p.bounds is None
    assert _categorical_dims(p) == ((0, 3),)


def test_categorical_spec_renormalizes_small_prob_error():
    p = Problem.from_dict({"c": {"dist": "categorical", "probs": [1 / 3, 1 / 3, 1 / 3]}})
    probs = p.input_specs[0][5][0]
    assert sum(probs) == pytest.approx(1.0, abs=1e-15)


def test_categorical_spec_labels_stored_as_strings():
    p = Problem.from_dict(
        {"c": {"dist": "categorical", "probs": [0.5, 0.5], "labels": ["red", 7]}}
    )
    assert p.categorical_labels == {"c": ("red", "7")}


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"dist": "categorical", "probs": [1.0]}, "at least 2 levels"),
        ({"dist": "categorical", "probs": [0.5, -0.5, 1.0]}, "positive"),
        ({"dist": "categorical", "probs": [0.5, 0.0, 0.5]}, "positive"),
        ({"dist": "categorical", "probs": [0.5, np.nan]}, "positive"),
        ({"dist": "categorical", "probs": [0.3, 0.3]}, "sum to 1"),
        ({"dist": "categorical", "probs": [0.5, 0.5], "labels": ["a"]}, "labels length"),
        ({"dist": "categorical", "probs": [0.5, 0.5], "labels": ["a", "a"]}, "unique"),
    ],
)
def test_categorical_spec_rejects_invalid_input(spec, match):
    with pytest.raises(ValueError, match=match):
        Problem.from_dict({"c": spec})


def test_categorical_labels_empty_without_categorical_params():
    p = Problem.from_dict({"x": (0.0, 1.0)})
    assert p.categorical_labels == {}
    assert p.has_categorical_inputs is False


def test_categorical_problem_is_hashable_and_comparable():
    p1 = _mixed_problem()
    p2 = _mixed_problem()
    assert p1 == p2
    assert hash(p1) == hash(p2)


def test_categorical_input_spec_typed_dict_accepted():
    spec = CategoricalInputSpec(dist="categorical", probs=[0.5, 0.5], labels=["lo", "hi"])
    p = Problem.from_dict({"c": spec})
    assert p.categorical_labels["c"] == ("lo", "hi")


# ---------------------------------------------------------------------------
# Correlation x categorical
# ---------------------------------------------------------------------------


def test_correlation_touching_categorical_raises():
    with pytest.raises(ValueError, match="polychoric"):
        Problem.from_dict(
            {"x": (0.0, 1.0), "c": {"dist": "categorical", "probs": [0.5, 0.5]}},
            correlation=[[1.0, 0.3], [0.3, 1.0]],
        )


def test_correlation_with_identity_categorical_rows_passes():
    p = Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "x2": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": [0.5, 0.5]},
        },
        correlation=[[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert p.has_correlated_inputs is True
    X = jaxgsa.sampling.monte_carlo(p, 20_000, seed=0)
    freqs = np.bincount(X[:, 2].astype(int), minlength=2) / X.shape[0]
    np.testing.assert_allclose(freqs, [0.5, 0.5], atol=0.02)
    assert abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1]) > 0.3


def test_with_correlation_touching_categorical_raises():
    p = _mixed_problem()
    with pytest.raises(ValueError, match="'c'"):
        p.with_correlation([[1.0, -0.2], [-0.2, 1.0]])


def test_identity_categorical_rows_survive_psd_repair():
    """The PSD repair's float noise must not read as a categorical coupling."""
    p = Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "x2": (0.0, 1.0),
            "x3": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": [0.5, 0.5]},
        }
    )
    # Non-PSD continuous block; the categorical row is exact identity.
    R = np.array(
        [
            [1.0, 0.9, 0.9, 0.0],
            [0.9, 1.0, -0.9, 0.0],
            [0.9, -0.9, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.linalg.eigvalsh(R).min() < 0  # the repair must engage
    with pytest.warns(UserWarning, match="not positive definite"):
        p2 = p.with_correlation(R)
    stored = p2.correlation
    assert stored is not None
    # The stored matrix carries exact identity categorical rows and columns.
    assert np.all(stored[3, :3] == 0.0)
    assert np.all(stored[:3, 3] == 0.0)
    assert stored[3, 3] == 1.0
    # A genuine coupling in the same problem still raises.
    R_bad = np.eye(4)
    R_bad[0, 3] = R_bad[3, 0] = 0.3
    with pytest.raises(ValueError, match="polychoric"):
        p.with_correlation(R_bad)


def test_fit_correlation_roundtrip_on_mixed_problem():
    """with_correlation(fit_correlation(...)) must work end to end."""
    p = Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "x2": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": PROBS},
        },
        correlation=[[1.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    X = jaxgsa.sampling.monte_carlo(p, 2000, seed=0)
    with pytest.warns(UserWarning, match="categorical"):
        fitted = jaxgsa.sampling.fit_correlation(p, X)
    p2 = p.with_correlation(fitted)  # must not raise
    R2 = p2.correlation
    assert R2 is not None
    assert abs(R2[0, 1] - 0.6) < 0.1
    assert np.all(R2[2, :2] == 0.0) and np.all(R2[:2, 2] == 0.0)


def test_fit_correlation_categorical_warning_and_identity_rows():
    problem, X, _ = _mixed_data(n=500)
    with pytest.warns(UserWarning, match="'c'"):
        fitted = jaxgsa.sampling.fit_correlation(problem, X)
    assert fitted[0, 1] == 0.0 and fitted[1, 0] == 0.0
    assert fitted[1, 1] == 1.0


def test_fit_correlation_invariant_under_level_relabeling():
    problem, X, _ = _mixed_data(n=800)
    codes = X[:, 1].astype(int)
    perm = np.array([2, 0, 1])
    X_perm = X.copy()
    X_perm[:, 1] = perm[codes]
    with pytest.warns(UserWarning, match="categorical"):
        fitted = jaxgsa.sampling.fit_correlation(problem, X)
    with pytest.warns(UserWarning, match="categorical"):
        fitted_perm = jaxgsa.sampling.fit_correlation(problem, X_perm)
    np.testing.assert_array_equal(fitted, fitted_perm)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_monte_carlo_level_frequencies_match_probs():
    p = Problem.from_dict({"c": {"dist": "categorical", "probs": PROBS}})
    X = jaxgsa.sampling.monte_carlo(p, 50_000, seed=0)
    codes = X[:, 0]
    assert np.all(codes == np.round(codes))
    freqs = np.bincount(codes.astype(int), minlength=3) / X.shape[0]
    np.testing.assert_allclose(freqs, PROBS, atol=0.01)


def test_monte_carlo_mixed_preserves_both_marginals():
    p = _mixed_problem()
    X = jaxgsa.sampling.monte_carlo(p, 50_000, seed=2)
    assert X[:, 0].min() >= 0.0 and X[:, 0].max() <= 1.0
    np.testing.assert_allclose(X[:, 0].mean(), 0.5, atol=0.01)
    freqs = np.bincount(X[:, 1].astype(int), minlength=3) / X.shape[0]
    np.testing.assert_allclose(freqs, PROBS, atol=0.01)


def test_inverse_transform_raises_for_categorical():
    p = _mixed_problem()
    X = jaxgsa.sampling.monte_carlo(p, 16, seed=0)
    with pytest.raises(ValueError, match="'c'"):
        _inverse_transform_samples(p, X)


def test_cdf_to_unit_interval_raises_for_categorical():
    p = _mixed_problem()
    X = jaxgsa.sampling.monte_carlo(p, 16, seed=0)
    with pytest.raises(ValueError, match="'c'"):
        cdf_to_unit_interval(X, p)


# ---------------------------------------------------------------------------
# Persistence round-trips
# ---------------------------------------------------------------------------


def test_problem_meta_json_round_trip_carries_probs_and_labels():
    from jaxgsa._core.samples import _problem_from_meta, _problem_to_meta

    p = Problem.from_dict(
        {
            "x": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": PROBS, "labels": ["a", "b", "c"]},
        }
    )
    meta = json.loads(json.dumps(_problem_to_meta(p)))  # force a real JSON round-trip
    restored = _problem_from_meta(meta)
    assert restored == p
    assert restored.categorical_labels == {"c": ("a", "b", "c")}


def test_sobol_npz_round_trip_carries_categorical_problem(tmp_path):
    p = Problem.from_dict(
        {
            "x": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": PROBS, "labels": ["lo", "mid", "hi"]},
        }
    )
    sr = sobol.sample(p, 64, seed=0, verbose=False)
    path = tmp_path / "design.npz"
    sr.save(path)
    loaded = sobol.SobolSamples.load(path)
    assert loaded.problem == p
    assert loaded.problem.categorical_labels == {"c": ("lo", "mid", "hi")}
    np.testing.assert_array_equal(loaded.samples, sr.samples)


# ---------------------------------------------------------------------------
# Partition layer
# ---------------------------------------------------------------------------


def test_extract_categorical_codes_rejects_non_code_values():
    p = _mixed_problem()
    dims_levels = _categorical_dims(p)
    X = np.asarray(jaxgsa.sampling.monte_carlo(p, 32, seed=0))
    bad = X.copy()
    bad[0, 1] = 0.5
    with pytest.raises(ValueError, match="'c'"):
        _extract_categorical_codes(p, bad, dims_levels)
    bad = X.copy()
    bad[0, 1] = 3.0  # out of range for 3 levels
    with pytest.raises(ValueError, match="'c'"):
        _extract_categorical_codes(p, bad, dims_levels)


def test_categorical_class_layout_partitions_by_level():
    codes = np.array([[0], [1], [0], [2], [1], [0]], dtype=np.int64)
    all_idx = np.arange(6, dtype=np.int64)[None, :]
    cls_idx, mask, counts = _categorical_class_layout(codes, all_idx, [3])
    np.testing.assert_array_equal(counts[0, 0], [3, 2, 1])
    # Class members are the rows at each level, in original row order.
    np.testing.assert_array_equal(cls_idx[0, 0, 0][mask[0, 0, 0]], [0, 2, 5])
    np.testing.assert_array_equal(cls_idx[0, 0, 1][mask[0, 0, 1]], [1, 4])
    np.testing.assert_array_equal(cls_idx[0, 0, 2][mask[0, 0, 2]], [3])


def test_categorical_class_layout_recounts_bootstrap_resamples():
    codes = np.array([[0], [0], [1], [1]], dtype=np.int64)
    all_idx = np.array([[0, 1, 2, 3], [0, 0, 0, 2]], dtype=np.int64)
    cls_idx, mask, counts = _categorical_class_layout(codes, all_idx, [2])
    np.testing.assert_array_equal(counts[0, 0], [2, 2])
    np.testing.assert_array_equal(counts[1, 0], [3, 1])
    np.testing.assert_array_equal(cls_idx[1, 0, 0][mask[1, 0, 0]], [0, 0, 0])
    np.testing.assert_array_equal(cls_idx[1, 0, 1][mask[1, 0, 1]], [2])


def test_empty_declared_level_warns_and_completes():
    p = Problem.from_dict(
        {"c": {"dist": "categorical", "probs": [0.5, 0.4, 0.1]}},
    )
    rng = np.random.default_rng(0)
    n = 400
    codes = rng.integers(0, 2, size=n)  # level 2 never observed
    X = codes[:, None].astype(np.float64)
    Y = 1.0 * codes + 0.1 * rng.standard_normal(n)
    with pytest.warns(UserWarning, match="no\\s+samples at level"):
        res = borgonovo.analyze(p, X, Y, n_bootstrap=4, seed=0)
    assert np.all(np.isfinite(np.asarray(res.delta)))
    assert float(res.S1[0]) > 0.8


# ---------------------------------------------------------------------------
# OT + Borgonovo estimates
# ---------------------------------------------------------------------------


def test_borgonovo_degenerate_class_recovers_delta():
    """A noise-free per-level output must not bias delta low.

    Y = offsets[code] with three balanced levels has true delta = 2/3.
    Without the bandwidth floor the degenerate (zero-variance) classes drop
    out of the integrand and delta collapses to ~0.33. With the floor the
    estimate lands at 0.60-0.62 for N in [1e3, 1e4] (measured error <= 0.07;
    asserted with headroom).
    """
    p = Problem.from_dict({"c": {"dist": "categorical", "probs": [1 / 3, 1 / 3, 1 / 3]}})
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 3, 3000)
    X = codes[:, None].astype(np.float64)
    Y = OFFSETS[codes]
    with pytest.warns(UserWarning, match="bandwidth"):
        res = borgonovo.analyze(p, X, Y, seed=0)
    assert abs(float(res.delta[0]) - 2.0 / 3.0) < 0.11
    assert float(res.S1[0]) == pytest.approx(1.0, abs=1e-6)


def test_borgonovo_s1_matches_analytic_reference():
    problem, X, Y = _mixed_data(n=8000)
    res = borgonovo.analyze(problem, X, Y, seed=0)
    s1_x1, s1_cat = _analytic_s1()
    np.testing.assert_allclose(np.asarray(res.S1), [s1_x1, s1_cat], atol=0.03)
    assert np.all(np.asarray(res.delta) > 0)


def test_ot_advective_matches_analytic_reference():
    problem, X, Y = _mixed_data(n=8000)
    res = optimal_transport.analyze(problem, X, Y)
    s1_x1, s1_cat = _analytic_s1()
    # The advective component equals half the given-data S1.
    np.testing.assert_allclose(2.0 * np.asarray(res.advective), [s1_x1, s1_cat], atol=0.03)
    assert float(res.ot[1]) > float(res.ot[0])


def test_pure_noise_categorical_index_is_near_zero():
    problem, X, _ = _mixed_data(n=8000)
    rng = np.random.default_rng(7)
    Y = X[:, 0] + 0.1 * rng.standard_normal(X.shape[0])
    res_b = borgonovo.analyze(problem, X, Y, seed=0)
    res_o = optimal_transport.analyze(problem, X, Y)
    assert float(res_b.S1[1]) < 0.01
    assert float(res_o.ot[1]) < 0.02


@pytest.mark.parametrize("method", ["borgonovo", "ot"])
def test_level_permutation_invariance(method):
    """The acid test: relabeling levels must not change the indices."""
    problem, X, Y = _mixed_data(n=4000)
    codes = X[:, 1].astype(int)

    perm = np.array([2, 0, 1])  # old code -> new code
    inv = np.argsort(perm)
    X_perm = X.copy()
    X_perm[:, 1] = perm[codes]
    probs_perm = [PROBS[inv[level]] for level in range(3)]
    problem_perm = Problem.from_dict(
        {"x1": (0.0, 1.0), "c": {"dist": "categorical", "probs": probs_perm}}
    )

    if method == "borgonovo":
        res = borgonovo.analyze(problem, X, Y, n_bootstrap=16, seed=0)
        res_perm = borgonovo.analyze(problem_perm, X_perm, Y, n_bootstrap=16, seed=0)
        pairs = [(res.delta, res_perm.delta), (res.S1, res_perm.S1)]
    else:
        res = optimal_transport.analyze(problem, X, Y, n_bootstrap=16, seed=0)
        res_perm = optimal_transport.analyze(problem_perm, X_perm, Y, n_bootstrap=16, seed=0)
        pairs = [(res.ot, res_perm.ot), (res.advective, res_perm.advective)]

    for a, b in pairs:
        # Same seed, same rows per class (only the class order changes):
        # the estimates agree to float noise.
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_ot_multivariate_mode_supports_categorical():
    problem, X, Y = _mixed_data(n=2000)
    rng = np.random.default_rng(3)
    Y2 = np.stack([Y, -Y + 0.05 * rng.standard_normal(Y.shape[0])], axis=1)
    res = optimal_transport.analyze(problem, X, Y2, mode="multivariate", max_iter=5000)
    assert res.ot.shape == (2,)
    assert float(res.ot[1]) > float(res.ot[0])


def test_ot_dummy_baseline_works_with_categorical():
    problem, X, Y = _mixed_data(n=2000)
    res = optimal_transport.analyze(problem, X, Y, dummy=True)
    assert res.ot_dummy is not None
    assert float(res.ot_dummy) < float(res.ot[1])


def _all_categorical_data(n=2000, seed=2):
    p = Problem.from_dict(
        {
            "c1": {"dist": "categorical", "probs": [0.5, 0.5]},
            "c2": {"dist": "categorical", "probs": [0.25, 0.25, 0.5]},
        }
    )
    X = jaxgsa.sampling.monte_carlo(p, n, seed=seed)
    rng = np.random.default_rng(seed)
    Y = 3.0 * X[:, 0] + 0.5 * rng.standard_normal(n)
    return p, X, Y


def test_all_categorical_problem_validates_passed_n_partitions():
    p, X, Y = _all_categorical_data()
    # A passed value is validated even when nothing uses it.
    with pytest.raises(ValueError, match="n_partitions"):
        optimal_transport.analyze(p, X, Y, n_partitions=10**9)
    with pytest.raises(ValueError, match="n_classes"):
        borgonovo.analyze(p, X, Y, n_classes=10**9, n_bootstrap=0)


def test_all_categorical_problem_warns_ignored_n_partitions():
    p, X, Y = _all_categorical_data()
    # A valid value that nothing uses draws an "ignored" warning.
    with pytest.warns(UserWarning, match="n_partitions is ignored"):
        res = optimal_transport.analyze(p, X, Y, n_partitions=10)
    assert float(res.ot[0]) > 0.5
    assert float(res.ot[1]) < 0.05
    with pytest.warns(UserWarning, match="n_classes is ignored"):
        res_b = borgonovo.analyze(p, X, Y, n_classes=10, n_bootstrap=0)
    assert float(res_b.S1[0]) > 0.7


def test_all_categorical_problem_default_n_partitions_is_silent():
    p, X, Y = _all_categorical_data()
    res = optimal_transport.analyze(p, X, Y)
    assert float(res.ot[0]) > 0.5
    # With a dummy the passed value is used; out of range names the dummy.
    with pytest.raises(ValueError, match="dummy baseline"):
        optimal_transport.analyze(p, X, Y, n_partitions=10**9, dummy=True)
    # The dummy default adapts to small N instead of raising over a bare 25.
    p_small, X_small, Y_small = _all_categorical_data(n=40, seed=5)
    res_small = optimal_transport.analyze(p_small, X_small, Y_small, dummy=True)
    assert res_small.ot_dummy is not None


def test_mixed_partition_bootstrap_cis_are_finite_and_ordered():
    problem, X, Y = _mixed_data(n=2000)
    res = borgonovo.analyze(problem, X, Y, n_bootstrap=32, seed=0)
    assert res.delta_conf is not None
    lo, hi = np.asarray(res.delta_conf)
    assert np.all(lo <= hi)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    res_o = optimal_transport.analyze(problem, X, Y, n_bootstrap=32, seed=0)
    assert res_o.ot_conf is not None
    lo_o, hi_o = np.asarray(res_o.ot_conf)
    assert np.all(lo_o <= hi_o)


# ---------------------------------------------------------------------------
# Sobol design + analysis
# ---------------------------------------------------------------------------


def test_sobol_saltelli_matches_analytic_reference():
    problem = _mixed_problem()
    sr = sobol.sample(problem, 2**13, seed=0, verbose=False)
    codes = sr.samples[:, 1].astype(int)
    assert np.array_equal(sr.samples[:, 1], codes)  # codes, never physical values
    Y = sr.samples[:, 0] + OFFSETS[codes]
    res = sobol.analyze(sr, Y)

    pr = np.array(PROBS)
    m = (pr * OFFSETS).sum()
    v_cat = (pr * (OFFSETS - m) ** 2).sum()
    v_y = v_cat + 1.0 / 12.0
    expected = np.array([(1.0 / 12.0) / v_y, v_cat / v_y])
    np.testing.assert_allclose(np.asarray(res.S1), expected, atol=5e-2)
    np.testing.assert_allclose(np.asarray(res.ST), expected, atol=5e-2)


def test_sobol_design_inflation_guard_warns_and_completes():
    p = Problem.from_dict(
        {
            "c1": {"dist": "categorical", "probs": [0.5, 0.5]},
            "c2": {"dist": "categorical", "probs": [0.3, 0.7]},
        }
    )
    with pytest.warns(UserWarning, match="possible distinct rows"):
        sr = sobol.sample(p, 1000, seed=0, verbose=False)
    assert sr.n_runs <= 4  # at most prod(L_d) distinct rows
    Y = sr.samples[:, 0] + 0.5 * sr.samples[:, 1]
    res = sobol.analyze(sr, Y)
    # V1 = Var(Bernoulli(0.5)) = 0.25; V2 = 0.25 * Var(Bernoulli(0.7)).
    v1, v2 = 0.25, 0.25 * 0.21
    np.testing.assert_allclose(np.asarray(res.S1), [v1 / (v1 + v2), v2 / (v1 + v2)], atol=5e-2)


def test_sobol_inflation_guard_detects_flat_doubling(monkeypatch):
    """Generic saturation: a doubling that adds no unique rows stops the loop.

    Simulates a future finite-support marginal with no known distinct-row
    bound by disabling the bound; the flat-doubling predicate must catch
    the saturation instead.
    """
    from jaxgsa.sobol import _sampling as sobol_sampling

    p = Problem.from_dict(
        {
            "c1": {"dist": "categorical", "probs": [0.5, 0.5]},
            "c2": {"dist": "categorical", "probs": [0.3, 0.7]},
        }
    )
    monkeypatch.setattr(sobol_sampling, "_max_distinct_rows", lambda problem: None)
    with pytest.warns(UserWarning, match="added no new unique rows"):
        sr = sobol.sample(p, 1000, seed=0, verbose=False)
    assert sr.n_runs <= 4


def test_sobol_expand_outputs_round_trips_duplicate_rows():
    p = Problem.from_dict(
        {
            "c1": {"dist": "categorical", "probs": [0.5, 0.5]},
            "c2": {"dist": "categorical", "probs": [0.3, 0.7]},
        }
    )
    with pytest.warns(UserWarning, match="possible distinct rows"):
        sr = sobol.sample(p, 1000, seed=0, verbose=False)
    Y = sr.samples[:, 0]
    expanded = np.asarray(sr.expand_outputs(Y))
    assert expanded.shape[0] == sr.n_expanded
    # Every expanded row's output matches its unique row's output.
    np.testing.assert_array_equal(expanded, Y[sr.expanded_to_unique])


# ---------------------------------------------------------------------------
# Method gating
# ---------------------------------------------------------------------------


def _gated_calls(problem, X, Y):
    return [
        (lambda: morris.sample(problem, 8)),
        (lambda: efast.sample(problem, 65)),
        (lambda: dgsm.analyze(problem, fn=lambda x: x[:, 0], X=X)),
        (lambda: pce_mod.analyze(problem, X, Y)),
        (lambda: hdmr.analyze(problem, X, Y)),
        (lambda: hsic.analyze(problem, X, Y)),
        (lambda: pawn.analyze(problem, X, Y)),
        (lambda: shapley.analyze(problem, X, Y)),
    ]


@pytest.mark.parametrize("call_idx", range(8))
def test_gated_methods_raise_naming_the_categorical_parameter(call_idx):
    problem, X, Y = _mixed_data(n=256)
    call = _gated_calls(problem, X, Y)[call_idx]
    with pytest.raises(ValueError, match="'c'"):
        call()


def test_to_morris_raises_for_categorical_design():
    problem = _mixed_problem()
    sr = sobol.sample(problem, 128, seed=0, verbose=False)
    with pytest.raises(ValueError, match="'c'"):
        sr.to_morris(verbose=False)
