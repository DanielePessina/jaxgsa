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

# ---------------------------------------------------------------------------
# Ishigami
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("offset", [0.0, 1e4])
@pytest.mark.parametrize("n_samples", [1024, 4096, 16384])
def test_ishigami_converges_to_the_analytic_indices_at_any_output_offset(n_samples, offset):
    """Sobol converges on the closed form, and an output offset does not stop it.

    Tier T0 (closed form): the target is
    ``jaxgsa.benchmarks.ishigami.ANALYTICAL_S1`` / ``ANALYTICAL_ST``, not a
    recorded number, so this test cannot certify the behaviour it is meant to
    catch.

    The tolerance tightens with N, which is the point: the error has to be
    sampling error, so it has to shrink. It also has to be the *same* error
    with and without the offset. Before the output standardization became
    unconditional, an offset of 1e4 put S1 near ``[6.26, 0.434, 1.71]`` at
    N=4096 — in float64 as well, so it was estimator bias and not rounding.
    """
    tolerance = {1024: 0.15, 4096: 0.05, 16384: 0.02}[n_samples]
    sr = jaxgsa.sobol.sample(
        ishigami.PROBLEM, n_samples=n_samples, seed=7, calc_second_order=False, verbose=False
    )
    Y = ishigami.evaluate(jnp.asarray(sr.samples)) + offset
    result = jaxgsa.sobol.analyze(sr, Y)

    assert np.abs(np.asarray(result.S1) - np.asarray(ishigami.ANALYTICAL_S1)).max() < tolerance
    assert np.abs(np.asarray(result.ST) - np.asarray(ishigami.ANALYTICAL_ST)).max() < tolerance


# ---------------------------------------------------------------------------
# Linear additive
# ---------------------------------------------------------------------------


class TestLinear:
    def test_s1(self, linear_sobol_result):
        S1 = np.asarray(linear_sobol_result.S1)
        for i, expected in enumerate(linear.ANALYTICAL_S1):
            rel = abs(S1[i] - expected) / expected
            assert rel < 0.05, f"S1[{i}]={S1[i]:.4f}, expected {expected:.4f}"

    def test_st_equals_s1(self, linear_sobol_result):
        """No interactions: ST should equal S1."""
        S1 = np.asarray(linear_sobol_result.S1)
        ST = np.asarray(linear_sobol_result.ST)
        np.testing.assert_allclose(ST, S1, atol=0.03)

    def test_s2_zero(self, linear_sobol_result):
        """No interactions: all S2 off-diagonal entries should be ~0."""
        S2 = np.asarray(linear_sobol_result.S2)
        off_diag = S2[np.triu_indices_from(S2, k=1)]
        np.testing.assert_allclose(off_diag, 0.0, atol=0.03)

    def test_analytical_values(self):
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

    def test_analytical_degenerate(self):
        """When all a=0, each factor is equally important."""
        # All a_j = 0 makes the G-function symmetric in all inputs, so S1_j = 1/D.
        S1, ST, _ = sobol_g.analytical_indices(a=(0.0, 0.0, 0.0))
        np.testing.assert_allclose(S1, S1[0], atol=1e-10)
        np.testing.assert_allclose(ST, ST[0], atol=1e-10)


# ---------------------------------------------------------------------------
# Oakley & O'Hagan
# ---------------------------------------------------------------------------


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

    def test_analytical_matches_published(self):
        """Analytical closed-form S1 matches Oakley & O'Hagan (2004) Table."""
        S1, _, S2 = oakley_ohagan.analytical_indices(sigma=1.0)
        np.testing.assert_allclose(S1, oakley_ohagan.PUBLISHED_S1, atol=2e-5)
        # S2 should be a (15, 15) matrix with NaN diagonal
        assert S2.shape == (15, 15)
        assert np.all(np.isnan(np.diag(S2)))

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
