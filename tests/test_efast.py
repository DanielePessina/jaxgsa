"""Tests for eFAST (extended FAST) sensitivity analysis."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa.benchmarks import ishigami, linear, sobol_g
from jaxgsa.efast import EFASTSamples, analyze, sample
from jaxgsa.efast._analyze import _compute_indices
from jaxgsa.efast._sampling import _assign_frequencies
from jaxgsa.problem import GaussianInputSpec, Problem


@pytest.fixture(scope="module")
def ishigami_efast_result():
    """eFAST result for Ishigami benchmark."""
    sr = sample(ishigami.PROBLEM, n_per_curve=4096, M=4, seed=42)
    Y = ishigami.evaluate(jnp.asarray(sr.samples))
    return analyze(sr, jnp.asarray(Y))


@pytest.fixture(scope="module")
def linear_efast_result():
    """eFAST result for linear benchmark."""
    sr = sample(linear.PROBLEM, n_per_curve=2048, M=4, seed=123)
    Y = linear.evaluate(jnp.asarray(sr.samples))
    return analyze(sr, jnp.asarray(Y))


class TestIshigamiAccuracy:
    def test_s1(self, ishigami_efast_result):
        S1 = np.asarray(ishigami_efast_result.S1)
        for i, expected in enumerate(ishigami.ANALYTICAL_S1):
            if expected == 0.0:
                assert abs(S1[i]) < 0.02, f"S1[{i}]={S1[i]:.4f}, expected ~0"
            else:
                rel = abs(S1[i] - expected) / expected
                assert rel < 0.10, f"S1[{i}]={S1[i]:.4f}, expected {expected}"

    def test_st(self, ishigami_efast_result):
        ST = np.asarray(ishigami_efast_result.ST)
        for i, expected in enumerate(ishigami.ANALYTICAL_ST):
            rel = abs(ST[i] - expected) / expected
            assert rel < 0.10, f"ST[{i}]={ST[i]:.4f}, expected {expected}"

    def test_st_geq_s1(self, ishigami_efast_result):
        S1 = np.asarray(ishigami_efast_result.S1)
        ST = np.asarray(ishigami_efast_result.ST)
        assert np.all(ST >= S1 - 0.02)


class TestLinearAccuracy:
    def test_s1(self, linear_efast_result):
        S1 = np.asarray(linear_efast_result.S1)
        for i, expected in enumerate(linear.ANALYTICAL_S1):
            rel = abs(S1[i] - expected) / expected
            assert rel < 0.10, f"S1[{i}]={S1[i]:.4f}, expected {expected:.4f}"

    def test_st_equals_s1(self, linear_efast_result):
        S1 = np.asarray(linear_efast_result.S1)
        ST = np.asarray(linear_efast_result.ST)
        np.testing.assert_allclose(ST, S1, atol=0.05)

    def test_s1_sums_to_1(self, linear_efast_result):
        total = float(np.sum(np.asarray(linear_efast_result.S1)))
        assert abs(total - 1.0) < 0.10, f"sum(S1) = {total:.4f}"


class TestSampling:
    def test_returns_efast_samples(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        sr = sample(problem, n_per_curve=257, M=4, seed=1)
        assert isinstance(sr, EFASTSamples)
        assert sr.n_per_curve == 257
        assert sr.M == 4
        assert sr.problem is problem

    def test_shape(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        sr = sample(problem, n_per_curve=257, M=4, seed=1)
        assert sr.samples.shape == (257 * 2, 2)
        assert sr.n_runs == 257 * 2

    def test_within_bounds(self):
        problem = Problem(names=("a", "b"), bounds=((2.0, 5.0), (-1.0, 3.0)))
        X = sample(problem, n_per_curve=257, M=4, seed=2).samples
        assert np.all(X[:, 0] >= 2.0 - 1e-10)
        assert np.all(X[:, 0] <= 5.0 + 1e-10)
        assert np.all(X[:, 1] >= -1.0 - 1e-10)
        assert np.all(X[:, 1] <= 3.0 + 1e-10)

    def test_n_too_small_raises(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="4.*M.*2"):
            sample(problem, n_per_curve=64, M=4)

    def test_invalid_m_raises(self):
        problem = Problem(names=("x",), bounds=((0, 1),))
        with pytest.raises(ValueError, match="M must be >= 1"):
            sample(problem, n_per_curve=257, M=0)

    def test_n_below_dimension_bound_raises(self):
        """n_per_curve above 4*M^2 is still too small once D grows."""
        problem = Problem(names=tuple(f"x{i}" for i in range(6)), bounds=((0, 1),) * 6)
        # 4*M^2 = 64 passes the old bound; the D-aware bound is 4*M^2*5 + 1 = 321.
        with pytest.raises(ValueError, match=r"4\*M\^2\*\(D-1\) \+ 1 = 321"):
            sample(problem, n_per_curve=257, M=4)
        sample(problem, n_per_curve=321, M=4, seed=0)

    def test_complementary_frequencies_are_distinct(self):
        """Every parameter on a curve oscillates at its own frequency."""
        for D in (2, 5, 9):
            for M in (2, 4, 6):
                n = 4 * M**2 * (D - 1) + 1
                omega_0 = (n - 1) // (2 * M)
                omega_compl = _assign_frequencies(D, omega_0, M)
                assert len(set(omega_compl.tolist())) == D - 1, (D, M)
                assert omega_compl.min() >= 1
                assert omega_compl.max() <= omega_0 // (2 * M)

    def test_gaussian_inputs(self):
        problem = Problem.from_dict(
            {
                "x1": (0.0, 1.0),
                "x2": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
            }
        )
        X = sample(problem, n_per_curve=257, M=4, seed=3).samples
        assert X.shape == (257 * 2, 2)
        assert np.all(X[:, 0] >= 0.0 - 1e-10)
        assert np.all(X[:, 0] <= 1.0 + 1e-10)

    def test_reproducible(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        X1 = sample(problem, n_per_curve=257, M=4, seed=99).samples
        X2 = sample(problem, n_per_curve=257, M=4, seed=99).samples
        np.testing.assert_array_equal(X1, X2)


class TestEFASTSamplesValidation:
    def test_inconsistent_shape_raises(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        with pytest.raises(ValueError, match="n_per_curve"):
            EFASTSamples(samples=np.zeros((100, 2)), n_per_curve=257, M=4, problem=problem)

    def test_invalid_m_raises(self):
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        with pytest.raises(ValueError, match="M must be >= 1"):
            EFASTSamples(samples=np.zeros((514, 2)), n_per_curve=257, M=0, problem=problem)

    def test_n_below_dimension_bound_raises(self):
        """A hand-built design too short for its dimension is rejected."""
        problem = Problem(names=tuple(f"x{i}" for i in range(6)), bounds=((0, 1),) * 6)
        with pytest.raises(ValueError, match=r"4\*M\^2\*\(D-1\) \+ 1 = 321"):
            EFASTSamples(samples=np.zeros((257 * 6, 6)), n_per_curve=257, M=4, problem=problem)


class TestAnalysis:
    def test_result_shapes(self, ishigami_efast_result):
        assert ishigami_efast_result.S1.shape == (3,)
        assert ishigami_efast_result.ST.shape == (3,)

    def test_no_s2(self, ishigami_efast_result):
        assert set(vars(ishigami_efast_result).keys()) == {"S1", "ST", "problem", "omega_0", "M"}

    def test_omega_and_m(self, ishigami_efast_result):
        assert ishigami_efast_result.M == 4
        assert ishigami_efast_result.omega_0 > 0

    def test_wrong_y_row_count_raises_with_both_numbers(self):
        """A wrong Y row count names the given and required counts."""
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        sr = sample(problem, n_per_curve=257, M=4, seed=1)
        with pytest.raises(ValueError, match=r"7 rows.*514"):
            analyze(sr, jnp.ones(7))

    def test_transposed_y_raises(self):
        """Y transposed to (K, n_runs) must be rejected, not reinterpreted."""
        problem = Problem(names=("x1", "x2"), bounds=((0, 1), (0, 1)))
        sr = sample(problem, n_per_curve=257, M=4, seed=1)
        Y = jnp.ones((sr.n_runs, 3))
        with pytest.raises(ValueError, match=r"3 rows.*514"):
            analyze(sr, Y.T)


class TestMConsistency:
    """M travels inside EFASTSamples, so sample/analyze can never disagree."""

    def test_analyze_uses_samples_m(self):
        """A design built with M=6 is analyzed with 6 harmonics."""
        sr = sample(ishigami.PROBLEM, n_per_curve=4096, M=6, seed=42)
        Y = ishigami.evaluate(jnp.asarray(sr.samples))
        result = analyze(sr, jnp.asarray(Y))

        # Metadata threads through from the design, not from any analyze arg.
        assert result.M == 6
        assert result.omega_0 == (4096 - 1) // (2 * 6)

        # The indices must equal a manual M=6 computation on curve 0 —
        # proof analyze summed 6 harmonics of the M=6 omega_0.
        n = sr.n_per_curve
        omega_0 = (n - 1) // (2 * 6)
        s1_manual, st_manual = _compute_indices(jnp.asarray(Y[:n]), n, 6, omega_0)
        np.testing.assert_allclose(float(result.S1[0]), float(s1_manual), rtol=1e-6)
        np.testing.assert_allclose(float(result.ST[0]), float(st_manual), rtol=1e-6)

    def test_m6_still_accurate(self):
        """With M carried in the design, M=6 indices stay accurate."""
        sr = sample(ishigami.PROBLEM, n_per_curve=4096, M=6, seed=42)
        Y = ishigami.evaluate(jnp.asarray(sr.samples))
        result = analyze(sr, jnp.asarray(Y))
        S1 = np.asarray(result.S1)
        for i, expected in enumerate(ishigami.ANALYTICAL_S1):
            if expected == 0.0:
                assert abs(S1[i]) < 0.03, f"S1[{i}]={S1[i]:.4f}, expected ~0"
            else:
                rel = abs(S1[i] - expected) / expected
                assert rel < 0.10, f"S1[{i}]={S1[i]:.4f}, expected {expected}"


class TestFrequencyAssignment:
    def test_d1(self):
        freqs = _assign_frequencies(1, 100, 4)
        assert len(freqs) == 0

    def test_d2(self):
        freqs = _assign_frequencies(2, 100, 4)
        assert len(freqs) == 1
        assert freqs[0] >= 1

    def test_high_omega(self):
        freqs = _assign_frequencies(4, 200, 4)
        assert len(freqs) == 3
        assert np.all(freqs >= 1)
        assert len(np.unique(freqs)) == 3

    def test_low_omega_raises_instead_of_wrapping(self):
        """omega_0 = 15, M = 4 leaves one slot — three parameters cannot share it."""
        with pytest.raises(ValueError, match="distinct complementary frequencies"):
            _assign_frequencies(4, 15, 4)


class TestComputeIndices:
    def test_constant_output(self):
        """Constant Y has no meaningful variance; indices should be near zero or NaN."""
        Y = jnp.ones(257)
        s1, st = _compute_indices(Y, 257, 4, 32)
        assert jnp.isnan(s1) or abs(float(s1)) < 0.1
        assert jnp.isnan(st) or float(st) < 1.0

    def test_single_frequency(self):
        N = 257
        omega = 32
        s = (2 * np.pi / N) * np.arange(N)
        Y = jnp.asarray(np.sin(omega * s))
        s1, st = _compute_indices(Y, N, 4, omega)
        assert float(s1) > 0.9


class TestMultiOutputShapes:
    """Tests for multi-output and time-series Y shapes."""

    @pytest.fixture(scope="class")
    def ishigami_multi(self):
        """Ishigami design, scalar result, and multi-output Y (K=3)."""
        sr = sample(ishigami.PROBLEM, n_per_curve=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(sr.samples))
        scalar_result = analyze(sr, jnp.asarray(Y))
        Y_multi = jnp.stack([Y, 2 * Y, 0.5 * Y], axis=-1)
        return sr, Y, Y_multi, scalar_result

    def test_2d_multi_output(self, ishigami_multi):
        """Y shape (N*D, K) produces S1/ST shape (K, D)."""
        sr, _, Y_multi, _ = ishigami_multi
        result = analyze(sr, Y_multi)
        D = ishigami.PROBLEM.num_vars
        K = 3
        assert result.S1.shape == (K, D)
        assert result.ST.shape == (K, D)

    def test_3d_time_series(self, ishigami_multi):
        """Y shape (N*D, T, K) produces S1/ST shape (T, K, D)."""
        sr, Y_scalar, _, _ = ishigami_multi
        T, K = 5, 2
        slices = []
        for t in range(T):
            scale_0 = float(t + 1)
            scale_1 = float(t + 1) * 0.5
            slices.append(jnp.stack([scale_0 * Y_scalar, scale_1 * Y_scalar], axis=-1))
        Y_3d = jnp.stack(slices, axis=1)  # (N*D, T, K)
        result = analyze(sr, Y_3d)
        D = ishigami.PROBLEM.num_vars
        assert result.S1.shape == (T, K, D)
        assert result.ST.shape == (T, K, D)

    def test_scalar_unchanged(self, ishigami_multi):
        """1-D Y still produces (D,) indices."""
        _, _, _, scalar_result = ishigami_multi
        D = ishigami.PROBLEM.num_vars
        assert scalar_result.S1.shape == (D,)
        assert scalar_result.ST.shape == (D,)


class TestMultiOutputAccuracy:
    """Scaled copies of Ishigami must produce the same indices as scalar."""

    def test_multi_output_accuracy(self):
        sr = sample(ishigami.PROBLEM, n_per_curve=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(sr.samples))
        scalar_result = analyze(sr, jnp.asarray(Y))
        Y_multi = jnp.stack([Y, 2 * Y, 0.5 * Y], axis=-1)
        multi_result = analyze(sr, Y_multi)
        np.testing.assert_allclose(multi_result.S1[0], scalar_result.S1, atol=1e-5)
        np.testing.assert_allclose(multi_result.ST[0], scalar_result.ST, atol=1e-5)


class TestPrenormalize:
    """Prenormalize should handle large constant offsets."""

    def test_prenormalize(self):
        sr = sample(ishigami.PROBLEM, n_per_curve=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(sr.samples))
        baseline = analyze(sr, jnp.asarray(Y))
        Y_shifted = Y + 1e6
        result = analyze(sr, jnp.asarray(Y_shifted), prenormalize=True)
        np.testing.assert_allclose(np.asarray(result.S1), np.asarray(baseline.S1), atol=0.02)
        np.testing.assert_allclose(np.asarray(result.ST), np.asarray(baseline.ST), atol=0.02)


class TestSliceChunkSize:
    """Slice chunk size should not affect results."""

    def test_slice_chunk_size(self):
        sr = sample(ishigami.PROBLEM, n_per_curve=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(sr.samples))
        Y_multi = jnp.stack([Y, 2 * Y, 0.5 * Y], axis=-1)
        default_result = analyze(sr, Y_multi)
        chunked_result = analyze(sr, Y_multi, slice_chunk_size=1)
        np.testing.assert_allclose(
            np.asarray(chunked_result.S1),
            np.asarray(default_result.S1),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(chunked_result.ST),
            np.asarray(default_result.ST),
            atol=1e-6,
        )


class TestToDatasetMultiOutput:
    """Tests for to_dataset with multi-output and time-series results."""

    def test_2d_dataset(self):
        """Multi-output result exports with ('output', 'param') dims."""
        sr = sample(ishigami.PROBLEM, n_per_curve=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(sr.samples))
        Y_multi = jnp.stack([Y, 2 * Y, 0.5 * Y], axis=-1)
        result = analyze(sr, Y_multi)
        ds = result.to_dataset()
        assert set(ds["S1"].dims) == {"output", "param"}
        assert list(ds.coords["param"].values) == list(ishigami.PROBLEM.names)
        assert len(ds.coords["output"]) == 3

    def test_3d_dataset_with_time_coords(self):
        """Time-series result exports with ('time', 'output', 'param') dims."""
        sr = sample(ishigami.PROBLEM, n_per_curve=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(sr.samples))
        T = 5
        slices = []
        for t in range(T):
            slices.append(jnp.stack([float(t + 1) * Y, float(t + 1) * 0.5 * Y], axis=-1))
        Y_3d = jnp.stack(slices, axis=1)
        result = analyze(sr, Y_3d)
        time_coords = [0.0, 0.1, 0.2, 0.3, 0.4]
        ds = result.to_dataset(time_coords=time_coords)
        assert set(ds["S1"].dims) == {"time", "output", "param"}
        assert list(ds.coords["time"].values) == time_coords
        assert list(ds.coords["param"].values) == list(ishigami.PROBLEM.names)

    def test_repr(self):
        """repr() includes shape info."""
        sr = sample(ishigami.PROBLEM, n_per_curve=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(sr.samples))
        Y_multi = jnp.stack([Y, 2 * Y], axis=-1)
        result = analyze(sr, Y_multi)
        r = repr(result)
        assert "EFASTResult" in r
        assert "S1" in r


class TestToDataset:
    def test_conversion(self, ishigami_efast_result):
        ds = ishigami_efast_result.to_dataset()
        assert "S1" in ds.data_vars
        assert "ST" in ds.data_vars
        assert list(ds.coords["param"].values) == list(ishigami.PROBLEM.names)


class TestSobolGAccuracy:
    """Validate eFAST against the Sobol G-function benchmark (8-D)."""

    @pytest.fixture(scope="module")
    def efast_sobol_g_result(self):
        """eFAST result for the Sobol G-function benchmark.

        Named to distinguish it from the session fixture `sobol_g_result` in
        conftest, which holds a Sobol result for the same model.
        """
        sr = sample(sobol_g.PROBLEM, n_per_curve=4096, M=4, seed=42)
        Y = sobol_g.evaluate(jnp.asarray(sr.samples))
        return analyze(sr, jnp.asarray(Y))

    def test_s1(self, efast_sobol_g_result):
        S1 = np.asarray(efast_sobol_g_result.S1)
        analytical = np.asarray(sobol_g.ANALYTICAL_S1)
        for i in range(len(analytical)):
            if analytical[i] > 0.01:
                rel = abs(S1[i] - analytical[i]) / analytical[i]
                assert rel < 0.20, (
                    f"S1[{i}]={S1[i]:.4f}, expected {analytical[i]:.4f}, rel error {rel:.2%}"
                )
            else:
                assert abs(S1[i]) < 0.02, f"S1[{i}]={S1[i]:.4f}, expected ~0"

    def test_st(self, efast_sobol_g_result):
        """Tier T0 (closed form): every analytical ST entry of the Sobol G-function.

        Large entries hold to 20% relative error. The four near-zero entries
        (analytical 1.05e-4) get an absolute bound instead. The reason is not
        that no relative bound holds: the estimate is about 1.8e-4 against
        1.05e-4, a relative error near 77%, so ``rel < 1.0`` would hold. It is
        that a relative bound on a near-zero true value measures nothing useful.
        The eFAST total index sums all spectral power outside a parameter's own
        harmonics, so aliasing and interpolation noise leave a positive floor
        whose size is set by the design, not by the analytical value. What the
        test needs to catch is that floor growing, and an absolute bound says
        that directly. 1e-3 keeps about 5x headroom over the 1.8e-4 seen here.
        """
        ST = np.asarray(efast_sobol_g_result.ST)
        analytical = np.asarray(sobol_g.ANALYTICAL_ST)
        for i in range(len(analytical)):
            if analytical[i] > 0.01:
                rel = abs(ST[i] - analytical[i]) / analytical[i]
                assert rel < 0.20, (
                    f"ST[{i}]={ST[i]:.4f}, expected {analytical[i]:.4f}, rel error {rel:.2%}"
                )
            else:
                assert abs(ST[i]) < 1e-3, f"ST[{i}]={ST[i]:.6f}, expected ~0"

    def test_ranking(self, efast_sobol_g_result):
        """First three params should be ordered by importance (a_j = 0, 1, 4.5)."""
        S1 = np.asarray(efast_sobol_g_result.S1)
        assert S1[0] > S1[1], f"S1[0]={S1[0]:.4f} should be > S1[1]={S1[1]:.4f}"
        assert S1[1] > S1[2], f"S1[1]={S1[1]:.4f} should be > S1[2]={S1[2]:.4f}"


def test_single_param():
    """D=1 edge case: single parameter problem."""
    problem = Problem(names=("x",), bounds=((0.0, 1.0),))
    sr = sample(problem, n_per_curve=257, M=4, seed=1)
    Y = jnp.sin(jnp.asarray(sr.samples[:, 0]))
    result = analyze(sr, Y)
    assert result.S1.shape == (1,)
    assert result.ST.shape == (1,)
    assert float(result.S1[0]) > 0.5
    assert float(result.ST[0]) > 0.5
