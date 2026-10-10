"""Verify Sobol indices against analytical benchmark values.

Tier T0 (closed form) throughout. Each test compares jaxgsa-computed indices
to the closed-form analytical solutions for the Ishigami, linear additive,
Sobol G-function, and Oakley-O'Hagan benchmarks.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa.benchmarks import ishigami, linear, oakley_ohagan, sobol_g

# The big closed-form checks sit behind the ``slow`` marker. The default tier
# keeps one small Ishigami run and the cheap oracle checks.

# ---------------------------------------------------------------------------
# Ishigami
# ---------------------------------------------------------------------------


def _run(problem, benchmark, n_samples, seed, **kwargs):
    sr = jaxgsa.sobol.sample(problem, n_samples=n_samples, seed=seed, verbose=False, **kwargs)
    return jaxgsa.sobol.analyze(sr, benchmark.evaluate(jnp.asarray(sr.samples)))


# Sobol estimation error scales as O(1/sqrt(N)). Higher-dimensional or more
# complex models need proportionally larger sample sizes for convergence.
@pytest.fixture(scope="module")
def ishigami_sobol_result():
    return _run(ishigami.PROBLEM, ishigami, 2**14 * 8, 42)


@pytest.fixture(scope="module")
def sobol_g_result():
    return _run(sobol_g.PROBLEM, sobol_g, 2**14 * 10, 456, calc_second_order=False)


# Oakley-O'Hagan has 15 Gaussian inputs with correlated interactions,
# requiring ~2^14 * 32 samples for stable convergence of S1 and ST.
@pytest.fixture(scope="module")
def oakley_sobol_result():
    return _run(oakley_ohagan.PROBLEM, oakley_ohagan, 2**14 * 32, 789, calc_second_order=False)


def test_ishigami_small_run_lands_on_the_closed_form():
    """A small Ishigami run lands near the closed-form S1, ST and S2 (cheap tier).

    Tier T0 (closed form). The tight-tolerance versions are the ``slow`` tests
    below. Measured worst errors over 8 seeds at this size are 0.0022 on S1,
    0.0015 on ST and 0.0022 on S2; the bounds leave a wide margin.
    """
    result = _run(ishigami.PROBLEM, ishigami, 2**12 * 8, 42)
    np.testing.assert_allclose(result.S1, ishigami.ANALYTICAL_S1, atol=0.02)
    np.testing.assert_allclose(result.ST, ishigami.ANALYTICAL_ST, atol=0.02)
    S2 = np.asarray(result.S2)
    assert np.all(np.isnan(np.diag(S2)))
    np.testing.assert_allclose(S2, S2.T, equal_nan=True)
    assert abs(S2[0, 2] - ishigami.ANALYTICAL_S2[0, 2]) < 0.03


@pytest.mark.slow
class TestIshigami:
    def test_s1(self, ishigami_sobol_result):
        S1 = np.asarray(ishigami_sobol_result.S1)
        for i, expected in enumerate(ishigami.ANALYTICAL_S1):
            if expected == 0.0:
                assert abs(S1[i]) < 0.05, f"S1[{i}]={S1[i]:.4f}, expected ~0"
            else:
                rel = abs(S1[i] - expected) / expected
                assert rel < 0.05, f"S1[{i}]={S1[i]:.4f}, expected {expected}, rel={rel:.3f}"

    def test_st(self, ishigami_sobol_result):
        ST = np.asarray(ishigami_sobol_result.ST)
        for i, expected in enumerate(ishigami.ANALYTICAL_ST):
            rel = abs(ST[i] - expected) / expected
            assert rel < 0.05, f"ST[{i}]={ST[i]:.4f}, expected {expected}, rel={rel:.3f}"

    def test_s2(self, ishigami_sobol_result):
        S2 = np.asarray(ishigami_sobol_result.S2)
        assert np.all(np.isnan(np.diag(S2)))
        assert np.allclose(
            S2[np.triu_indices_from(S2, k=1)],
            S2[(np.triu_indices_from(S2, k=1)[1], np.triu_indices_from(S2, k=1)[0])],
        )
        expected_02 = ishigami.ANALYTICAL_S2[0, 2]
        rel = abs(S2[0, 2] - expected_02) / expected_02
        assert rel < 0.10, f"S2[0,2]={S2[0, 2]:.4f}, expected {expected_02}"


@pytest.mark.slow
@pytest.mark.parametrize("n_samples", [1024, 4096, 16384])
def test_ishigami_converges_to_the_analytic_indices(n_samples):
    """Sobol converges on the closed form as N grows.

    Tier T0 (closed form): the target is
    ``jaxgsa.benchmarks.ishigami.ANALYTICAL_S1`` / ``ANALYTICAL_ST``, not a
    recorded number, so this test cannot certify the behaviour it is meant to
    catch.

    The tolerance tightens with N, which is the point: the error has to be
    sampling error, so it has to shrink. (The output-offset case lives in
    ``test_analyze.py``.)
    """
    tolerance = {1024: 0.15, 4096: 0.05, 16384: 0.02}[n_samples]
    sr = jaxgsa.sobol.sample(
        ishigami.PROBLEM, n_samples=n_samples, seed=7, calc_second_order=False, verbose=False
    )
    Y = ishigami.evaluate(jnp.asarray(sr.samples))
    result = jaxgsa.sobol.analyze(sr, Y)

    assert np.abs(np.asarray(result.S1) - np.asarray(ishigami.ANALYTICAL_S1)).max() < tolerance
    assert np.abs(np.asarray(result.ST) - np.asarray(ishigami.ANALYTICAL_ST)).max() < tolerance


# ---------------------------------------------------------------------------
# Linear additive
# ---------------------------------------------------------------------------


def test_linear_analytical_values():
    """Verify the analytical formulas directly."""
    S1, ST, S2 = linear.analytical_indices()
    np.testing.assert_allclose(S1, ST)
    # Linear model f = c . x with c = [1,2,3] and uniform inputs (Var = 1/12).
    # S1_j = c_j^2 * Var(x_j) / sum(c_k^2 * Var(x_k)) = c_j^2 / sum(c_k^2) = c_j^2 / 14.
    np.testing.assert_allclose(S1, [1.0 / 14, 4.0 / 14, 9.0 / 14])
    off_diag = S2[np.triu_indices_from(S2, k=1)]
    np.testing.assert_allclose(off_diag, 0.0)


# ---------------------------------------------------------------------------
# Sobol G-function
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSobolG:
    def test_s1(self, sobol_g_result):
        S1 = np.asarray(sobol_g_result.S1)
        for i, expected in enumerate(sobol_g.ANALYTICAL_S1):
            if expected < 0.001:
                assert abs(S1[i]) < 0.02, f"S1[{i}]={S1[i]:.4f}, expected ~0"
            else:
                rel = abs(S1[i] - expected) / expected
                assert rel < 0.10, f"S1[{i}]={S1[i]:.4f}, expected {expected:.4f}"

    def test_st(self, sobol_g_result):
        ST = np.asarray(sobol_g_result.ST)
        for i, expected in enumerate(sobol_g.ANALYTICAL_ST):
            if expected < 0.001:
                assert abs(ST[i]) < 0.02, f"ST[{i}]={ST[i]:.4f}, expected ~0"
            else:
                rel = abs(ST[i] - expected) / expected
                assert rel < 0.10, f"ST[{i}]={ST[i]:.4f}, expected {expected:.4f}"


# ---------------------------------------------------------------------------
# Oakley & O'Hagan
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestOakleyOHagan:
    def test_s1(self, oakley_sobol_result):
        S1 = np.asarray(oakley_sobol_result.S1)
        for i, expected in enumerate(oakley_ohagan.ANALYTICAL_S1):
            if expected < 0.005:
                assert abs(S1[i]) < 0.02, f"S1[{i}]={S1[i]:.4f}, expected ~{expected:.4f}"
            else:
                rel = abs(S1[i] - expected) / expected
                assert rel < 0.15, f"S1[{i}]={S1[i]:.4f}, expected {expected:.4f}, rel={rel:.3f}"

    def test_st(self, oakley_sobol_result):
        ST = np.asarray(oakley_sobol_result.ST)
        for i, expected in enumerate(oakley_ohagan.ANALYTICAL_ST):
            if expected < 0.005:
                assert abs(ST[i]) < 0.02, f"ST[{i}]={ST[i]:.4f}, expected ~{expected:.4f}"
            else:
                rel = abs(ST[i] - expected) / expected
                assert rel < 0.15, f"ST[{i}]={ST[i]:.4f}, expected {expected:.4f}, rel={rel:.3f}"

    def test_ranking_preserved(self, oakley_sobol_result):
        """Most/least important parameters should match analytical ranking."""
        S1 = np.asarray(oakley_sobol_result.S1)
        analytical_rank = np.argsort(oakley_ohagan.ANALYTICAL_S1)
        computed_rank = np.argsort(S1)
        top3_analytical = set(analytical_rank[-3:])
        top3_computed = set(computed_rank[-3:])
        assert top3_analytical == top3_computed, (
            f"Top-3 mismatch: analytical={top3_analytical}, computed={top3_computed}"
        )


def test_sobol_g_analytical_degenerate():
    """When all a=0, each factor is equally important."""
    # All a_j = 0 makes the G-function symmetric in all inputs, so S1_j = 1/D.
    S1, ST, _ = sobol_g.analytical_indices(a=(0.0, 0.0, 0.0))
    np.testing.assert_allclose(S1, S1[0], atol=1e-10)
    np.testing.assert_allclose(ST, ST[0], atol=1e-10)


def test_oakley_ohagan_analytical_matches_published():
    """Analytical closed-form S1 matches Oakley & O'Hagan (2004) Table."""
    S1, _, S2 = oakley_ohagan.analytical_indices(sigma=1.0)
    np.testing.assert_allclose(S1, oakley_ohagan.PUBLISHED_S1, atol=2e-5)
    # S2 should be a (15, 15) matrix with NaN diagonal
    assert S2.shape == (15, 15)
    assert np.all(np.isnan(np.diag(S2)))
