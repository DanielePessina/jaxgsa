"""Tests for jaxgsa.kucherenko — design-based Sobol' indices for dependent inputs.

The decisive check is the linear-Gaussian closed form: for ``Y = a . X`` with
``X ~ N(0, R)``,

    V(Y)   = a' R a
    S1_i   = (R a)_i^2 / V(Y)
    ST_i   = a_i^2 (1 - R_i,rest R_rest^-1 R_rest,i) / V(Y)

Tolerances are set from the measured estimator error at the test sample sizes
(max ~1.5e-3 over eight seeds at base_n = 4096) with more than 3x headroom.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jaxgsa
from jaxgsa.benchmarks import ishigami
from jaxgsa.problem import Problem

# Closed-form linear-Gaussian reference, shared with test_vkoga.py and
# test_correlated_agreement.py.
from _linear_gaussian import (  # isort: skip
    A_COEF,
    A_COEF_ASYM,
    ASYM_PROBLEM,
    GAUSS_PROBLEM,
    R_ASYM,
    R_GAUSS,
    RHO,
    analytic_indices,
)


@pytest.fixture(scope="module")
def correlated_result():
    """Kucherenko analysis of the correlated linear-Gaussian model."""
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, 4096, seed=1)
    Y = ks.samples @ A_COEF
    return jaxgsa.kucherenko.analyze(ks, Y)


# --- closed-form validation --------------------------------------------------


def test_closed_form_linear_gaussian(correlated_result):
    S1_true, ST_true, var_y = analytic_indices(A_COEF, R_GAUSS)
    np.testing.assert_allclose(np.asarray(correlated_result.S1), S1_true, atol=5e-3)
    np.testing.assert_allclose(np.asarray(correlated_result.ST), ST_true, atol=5e-3)
    np.testing.assert_allclose(float(np.asarray(correlated_result.variance)), var_y, rtol=2e-2)
    assert correlated_result.is_correlated
    # Under correlation ST < S1 is the expected picture for coupled inputs.
    assert float(correlated_result.ST[0]) < float(correlated_result.S1[0])


def test_asymmetric_correlation_structure():
    """A D=4 case with six distinct off-diagonals of mixed sign.

    R_GAUSS has one non-zero off-diagonal, so the parameter axis could be
    transposed inside it without changing the result. This structure cannot be
    permuted onto itself, so any index landing on the wrong parameter shows.
    """
    S1_true, ST_true, var_y = analytic_indices(A_COEF_ASYM, R_ASYM)
    ks = jaxgsa.kucherenko.sample(ASYM_PROBLEM.with_correlation(R_ASYM), 4096, seed=1)
    result = jaxgsa.kucherenko.analyze(ks, ks.samples @ A_COEF_ASYM)
    # Measured errors are 7e-4 (S1) and 2e-5 (ST); the budgets keep headroom.
    np.testing.assert_allclose(np.asarray(result.S1), S1_true, atol=5e-3)
    np.testing.assert_allclose(np.asarray(result.ST), ST_true, atol=5e-3)
    np.testing.assert_allclose(float(np.asarray(result.variance)), var_y, rtol=2e-2)


def test_parameter_permutation_equivariance():
    """Relabelling the parameters permutes the indices, and nothing else."""
    perm = np.array([2, 0, 3, 1])
    ks = jaxgsa.kucherenko.sample(ASYM_PROBLEM.with_correlation(R_ASYM), 4096, seed=1)
    base = jaxgsa.kucherenko.analyze(ks, ks.samples @ A_COEF_ASYM)

    # The same model with its parameters in a different order: permute the
    # correlation matrix and the coefficients together.
    ks_perm = jaxgsa.kucherenko.sample(
        ASYM_PROBLEM.with_correlation(R_ASYM[np.ix_(perm, perm)]), 4096, seed=1
    )
    permuted = jaxgsa.kucherenko.analyze(ks_perm, ks_perm.samples @ A_COEF_ASYM[perm])

    np.testing.assert_allclose(np.asarray(permuted.S1), np.asarray(base.S1)[perm], atol=5e-3)
    np.testing.assert_allclose(np.asarray(permuted.ST), np.asarray(base.ST)[perm], atol=5e-3)


def test_independent_ishigami_matches_sobol():
    """Identity correlation reproduces the classic Saltelli estimates."""
    ks = jaxgsa.kucherenko.sample(ishigami.PROBLEM, 4096, seed=3)
    Y = ishigami.evaluate(np.asarray(ks.samples))
    result = jaxgsa.kucherenko.analyze(ks, Y)
    assert not result.is_correlated

    sr = jaxgsa.sobol.sample(ishigami.PROBLEM, n_samples=2**15, seed=42, verbose=False)
    sob = jaxgsa.sobol.analyze(sr, ishigami.evaluate(np.asarray(sr.samples)))

    # Both estimators against the analytic values, and against each other,
    # within Monte-Carlo error (measured max ~5e-3 at this size).
    np.testing.assert_allclose(np.asarray(result.S1), ishigami.ANALYTICAL_S1, atol=2e-2)
    np.testing.assert_allclose(np.asarray(result.ST), ishigami.ANALYTICAL_ST, atol=2e-2)
    np.testing.assert_allclose(np.asarray(result.S1), np.asarray(sob.S1), atol=2e-2)
    np.testing.assert_allclose(np.asarray(result.ST), np.asarray(sob.ST), atol=2e-2)


def test_large_output_mean_does_not_corrupt_the_indices():
    """A +1e8 output offset must leave every index unchanged.

    The S1 estimator subtracts one shared shift from both product factors,
    so the covariance never competes with O(mean^2) terms in rounding. The
    uncentered product form lost it: at this configuration the offset moved
    S1 by up to 0.11 (and zeroed small indices). Measured drift after the
    fix is < 3e-11; 1e-8 leaves ~300x headroom. Runs under x64 so the f32
    representation of Y + 1e8 does not mask the estimator behaviour.
    """
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, 2048, seed=1)
    Y = np.asarray(ks.samples @ A_COEF, dtype=np.float64)
    with jax.enable_x64():
        base = jaxgsa.kucherenko.analyze(ks, Y)
        shifted = jaxgsa.kucherenko.analyze(ks, Y + 1e8)
    np.testing.assert_allclose(np.asarray(shifted.S1), np.asarray(base.S1), atol=1e-8)
    np.testing.assert_allclose(np.asarray(shifted.ST), np.asarray(base.ST), atol=1e-8)
    np.testing.assert_allclose(float(shifted.variance), float(base.variance), rtol=1e-9)


# --- design contracts ---------------------------------------------------------


def test_design_block_structure():
    """The joint block is shared; each conditional block keeps its fixed part."""
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, 256, seed=0)
    n, D = ks.base_n, ks.n_params
    assert ks.n_runs == n * (2 * D + 1)
    assert ks.n_expanded == ks.n_runs
    np.testing.assert_array_equal(ks.expanded_to_unique, np.arange(ks.n_runs))

    X = ks.samples
    A = X[:n]
    for i in range(D):
        first = X[n * (1 + i) : n * (2 + i)]
        total = X[n * (1 + D + i) : n * (2 + D + i)]
        others = [j for j in range(D) if j != i]
        # First-order block: column i is copied from the joint block.
        np.testing.assert_allclose(first[:, i], A[:, i], atol=1e-12)
        assert not np.allclose(first[:, others], A[:, others])
        # Total block: the other columns are copied from the joint block.
        np.testing.assert_allclose(total[:, others], A[:, others], atol=1e-12)
        assert not np.allclose(total[:, i], A[:, i])


def test_conditional_blocks_follow_the_copula():
    """Redrawn columns must reproduce the declared conditional correlation."""
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, 8192, seed=5)
    n = ks.base_n
    X = ks.samples
    # In the first-order block for x1, x2 is redrawn given x1: the pair
    # correlation must match rho, and x3 must stay uncorrelated.
    first_x1 = X[n : 2 * n]
    corr = np.corrcoef(first_x1, rowvar=False)
    assert abs(corr[0, 1] - RHO) < 0.05
    assert abs(corr[0, 2]) < 0.05
    # Marginals survive the conditional construction.
    np.testing.assert_allclose(first_x1.mean(axis=0), 0.0, atol=0.05)
    np.testing.assert_allclose(first_x1.std(axis=0), 1.0, atol=0.05)


def test_sample_size_rounds_up_to_power_of_two():
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 1000, seed=0)
    assert ks.base_n == 1024


def test_correlated_problem_is_accepted():
    """kucherenko.sample is exempt from the correlated-design guard."""
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, 64, seed=0)
    assert ks.problem.has_correlated_inputs
    # The guarded samplers refuse the same problem.
    with pytest.raises(ValueError, match="independent"):
        jaxgsa.sobol.sample(problem, 64, verbose=False)


def test_sampler_validation_errors():
    with pytest.raises(ValueError, match="at least 2 parameters"):
        jaxgsa.kucherenko.sample(Problem(names=("x",), bounds=((0.0, 1.0),)), 64)
    with pytest.raises(ValueError, match="n_samples"):
        jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 1)


def test_categorical_problem_raises():
    problem = Problem.from_dict(
        {
            "u1": {"dist": "uniform", "low": 0.0, "high": 1.0},
            "c1": {"dist": "categorical", "probs": [0.5, 0.5]},
        }
    )
    with pytest.raises(ValueError, match="categorical"):
        jaxgsa.kucherenko.sample(problem, 64)


# --- analyze contracts --------------------------------------------------------


def test_output_shape_contract():
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 128, seed=0)
    y0 = ks.samples @ A_COEF
    Y2 = np.stack([y0, 2.0 * y0, 1.0 - y0], axis=-1)  # (N, K=3)
    Y3 = np.stack([Y2, Y2 + 0.5], axis=1)  # (N, T=2, K=3)
    D = 3
    for Y, expected in ((y0, (D,)), (Y2, (3, D)), (Y3, (2, 3, D))):
        result = jaxgsa.kucherenko.analyze(ks, Y)
        assert result.S1.shape == expected
        assert result.ST.shape == expected
        assert result.variance.shape == expected[:-1]


def test_row_count_mismatch_raises():
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 128, seed=0)
    with pytest.raises(ValueError):
        jaxgsa.kucherenko.analyze(ks, np.zeros(ks.n_runs - 1))


def test_zero_variance_slice_warns_and_yields_nan():
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 128, seed=0)
    Y = np.stack([ks.samples @ A_COEF, np.ones(ks.n_runs)], axis=-1)
    with pytest.warns(UserWarning, match="zero variance"):
        result = jaxgsa.kucherenko.analyze(ks, Y)
    assert np.all(np.isfinite(np.asarray(result.S1)[0]))
    assert np.all(np.isnan(np.asarray(result.S1)[1]))
    assert np.all(np.isnan(np.asarray(result.ST)[1]))


def test_nonfinite_outputs_drop_base_points_with_warning():
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 512, seed=0)
    Y = np.asarray(ks.samples @ A_COEF)
    Y[3] = np.nan  # poisons base point 3 in the joint block
    with pytest.warns(UserWarning, match="non-finite"):
        result = jaxgsa.kucherenko.analyze(ks, Y)
    assert np.all(np.isfinite(np.asarray(result.S1)))
    # The estimate barely moves: one base point out of 512.
    clean = jaxgsa.kucherenko.analyze(ks, np.asarray(ks.samples @ A_COEF))
    np.testing.assert_allclose(np.asarray(result.S1), np.asarray(clean.S1), atol=2e-2)


# --- persistence and reporting ------------------------------------------------


def test_save_load_round_trip(tmp_path):
    problem = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    ks = jaxgsa.kucherenko.sample(problem, 128, seed=0)
    path = tmp_path / "design.npz"
    ks.save(path)
    loaded = jaxgsa.kucherenko.KucherenkoSamples.load(path)
    np.testing.assert_array_equal(loaded.samples, ks.samples)
    assert loaded.base_n == ks.base_n
    assert loaded.n_params == ks.n_params
    np.testing.assert_allclose(loaded.problem.correlation, R_GAUSS, atol=1e-12)
    # The loaded design analyzes identically.
    Y = ks.samples @ A_COEF
    a = jaxgsa.kucherenko.analyze(ks, Y)
    b = jaxgsa.kucherenko.analyze(loaded, Y)
    np.testing.assert_array_equal(np.asarray(a.S1), np.asarray(b.S1))


def test_to_dataset_schema(correlated_result):
    ds = correlated_result.to_dataset()
    assert set(ds.data_vars) == {"S1", "ST", "variance"}
    assert list(ds.coords["param"].values) == ["x1", "x2", "x3"]
    assert ds.attrs["method"] == "kucherenko"
    assert ds.attrs["correlated"] is True


def test_repr(correlated_result):
    text = repr(correlated_result)
    assert "KucherenkoResult" in text
    assert "correlated=True" in text
    ks = jaxgsa.kucherenko.sample(GAUSS_PROBLEM, 64, seed=0)
    assert "correlated=False" in repr(ks)
