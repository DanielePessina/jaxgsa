"""Verify Sobol indices against analytical benchmark values.

Each test compares gsax-computed indices to the closed-form analytical
solutions for the Ishigami, linear additive, and Sobol G-function
benchmarks.
"""

import numpy as np

from gsax.benchmarks import ishigami, linear, sobol_g

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
        expected_02 = ishigami.ANALYTICAL_S2[(0, 2)]
        rel = abs(S2[0, 2] - expected_02) / expected_02
        assert rel < 0.10, f"S2[0,2]={S2[0, 2]:.4f}, expected {expected_02}"

    def test_s1_sum_leq_1(self, ishigami_sobol_result):
        assert np.sum(np.asarray(ishigami_sobol_result.S1)) <= 1.0 + 0.05

    def test_st_geq_s1(self, ishigami_sobol_result):
        S1 = np.asarray(ishigami_sobol_result.S1)
        ST = np.asarray(ishigami_sobol_result.ST)
        assert np.all(ST >= S1 - 0.02)


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

    def test_s1_sums_to_1(self, linear_sobol_result):
        """Additive model: S1 values should sum to 1."""
        total = float(np.sum(np.asarray(linear_sobol_result.S1)))
        assert abs(total - 1.0) < 0.05, f"sum(S1) = {total:.4f}"

    def test_analytical_values(self):
        """Verify the analytical formulas directly."""
        S1, ST, S2 = linear.analytical_indices()
        np.testing.assert_allclose(S1, ST)
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

    def test_st_geq_s1(self, sobol_g_result):
        S1 = np.asarray(sobol_g_result.S1)
        ST = np.asarray(sobol_g_result.ST)
        assert np.all(ST >= S1 - 0.02)

    def test_analytical_indices_consistency(self):
        """Verify analytical formulas: S1 sums to < 1, ST sums to > 1."""
        S1, ST, S2 = sobol_g.analytical_indices()
        assert S1.sum() < 1.0
        assert ST.sum() > 1.0
        assert np.all(ST >= S1)
        off_diag = S2[np.triu_indices_from(S2, k=1)]
        assert np.all(off_diag >= 0)

    def test_analytical_degenerate(self):
        """When all a=0, each factor is equally important."""
        S1, ST, _ = sobol_g.analytical_indices(a=(0.0, 0.0, 0.0))
        np.testing.assert_allclose(S1, S1[0], atol=1e-10)
        np.testing.assert_allclose(ST, ST[0], atol=1e-10)
