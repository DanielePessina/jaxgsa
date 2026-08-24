"""Tests for eFAST (extended FAST) sensitivity analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning
from jaxgsa._core.invalid import InvalidUnit
from jaxgsa.benchmarks import ishigami, linear, sobol_g
from jaxgsa.efast import EFASTSamples, analyze, indices, sample
from jaxgsa.efast._analyze import _compute_indices
from jaxgsa.efast._sampling import _frequency_plan
from jaxgsa.problem import Problem


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


class TestLargeNPerCurve:
    """Regression test for H2: N**2 overflowed int32 at n_per_curve >= 46341.

    ``jnp.abs(f) ** 2 / N**2`` traced the Python int ``N**2`` as int32, and
    46341**2 exceeds ``2**31 - 1``. The fix divides by ``N`` before squaring,
    so ``N`` never appears squared as a Python int.
    """

    def test_n_per_curve_above_int32_sqrt_bound(self):
        """T0: a linear model still gives the closed-form S1 at N = 65537."""
        sr = sample(linear.PROBLEM, n_per_curve=65537, M=4, seed=1)
        Y = linear.evaluate(jnp.asarray(sr.samples))

        result = analyze(sr, Y, verbose=False)

        S1 = np.asarray(result.S1)
        for i, expected in enumerate(linear.ANALYTICAL_S1):
            rel = abs(S1[i] - expected) / expected
            assert rel < 0.10, f"S1[{i}]={S1[i]:.4f}, expected {expected:.4f}"


class TestSampling:
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

    def test_n_below_dimension_bound_raises(self):
        """n_per_curve above 4*M^2 is still too small once D grows."""
        problem = Problem(names=tuple(f"x{i}" for i in range(6)), bounds=((0, 1),) * 6)
        # 4*M^2 = 64 passes the old bound; the D-aware bound is 4*M^2*5 + 1 = 321.
        with pytest.raises(ValueError, match=r"4\*M\^2\*\(D-1\) \+ 1 = 321"):
            sample(problem, n_per_curve=257, M=4)
        sample(problem, n_per_curve=321, M=4, seed=0)

    def test_complementary_frequencies_are_distinct(self):
        """Every parameter on a curve oscillates at its own frequency.

        The upper bound is written out here rather than read from
        ``plan.assigned_max``. Carriers must stay clear of the focal
        parameter's ``M`` harmonics, which puts the highest usable carrier at
        ``omega_0 / (2*M)``. Comparing the plan against its own field would
        pass for any band the plan happened to choose.
        """
        for D in (2, 5, 9):
            for M in (2, 4, 6):
                n = 4 * M**2 * (D - 1) + 1
                plan = _frequency_plan(D, n, M)
                omega_compl = plan.omega_compl
                assert len(set(omega_compl.tolist())) == D - 1, (D, M)
                assert omega_compl.min() >= 1
                assert omega_compl.max() <= plan.omega_0 // (2 * M), (D, M)

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
        plan = _frequency_plan(1, 4 * 4**2 + 1, 4)
        assert len(plan.omega_compl) == 0

    def test_too_few_slots_raises_instead_of_wrapping(self):
        """A curve too short to hold D-1 distinct carriers is rejected up front.

        Tier T4 (internal consistency): three parameters cannot share one
        assigned slot, so the plan refuses to build rather than emitting
        duplicate frequencies.
        """
        with pytest.raises(ValueError, match=r"4\*M\^2\*\(D-1\) \+ 1"):
            _frequency_plan(4, 129, 4)


def _column(sr: EFASTSamples, curve: int, param: int) -> np.ndarray:
    """One parameter column of one search curve, mean removed.

    Args:
        sr: An evaluated eFAST design.
        curve: Index of the search curve (the focal parameter of the block).
        param: Index of the parameter column to pull out.

    Returns:
        The column, shape ``(n_per_curve,)``, centred so that the DFT's DC bin
        is empty and ``argmax`` finds the carrier.
    """
    n = sr.n_per_curve
    col = np.asarray(sr.samples)[curve * n : (curve + 1) * n, param]
    return col - col.mean()


def _tone(f: int, n: int) -> np.ndarray:
    """A unit cosine at integer frequency ``f`` over ``n`` curve points.

    The search-curve parameter ``s`` runs over ``2*pi*k/n`` for
    ``k = 0..n-1``, so ``cos(f*s)`` completes exactly ``f`` cycles across the
    curve. Its DFT therefore puts all of its power in bin ``f`` and nowhere
    else. This is the definition of the discrete Fourier transform, not a
    restatement of anything in ``jaxgsa.efast``.

    Args:
        f: Frequency in cycles per curve.
        n: Number of samples along the curve.

    Returns:
        The sampled cosine, shape ``(n,)``.
    """
    return np.cos(2.0 * np.pi * f * np.arange(n) / n)


def _st_for_tone(sr: EFASTSamples, f: int) -> float:
    """Run ``analyze`` on a design whose every curve carries one pure tone.

    Every search curve gets the same cosine at frequency ``f``. All of the
    output variance sits in bin ``f``, so ``ST = 1 - Dt/V`` is 0 when the
    analyzer's complementary band contains ``f`` and 1 when it does not. The
    value is therefore a direct read-out of the band the analyzer integrates
    over.

    Args:
        sr: An eFAST design; only its shape metadata is used.
        f: Frequency of the injected tone, in cycles per curve.

    Returns:
        The ST of the first parameter, which is the same for all of them.
    """
    n = sr.n_per_curve
    Y = jnp.asarray(np.tile(_tone(f, n), sr.problem.num_vars))
    return float(np.asarray(analyze(sr, Y).ST).reshape(-1)[0])


class TestAnalysisBandEdge:
    """Where the analyzer's complementary band actually stops.

    Every assertion here is driven by a signal built from the DFT definition,
    so none of it can agree with a wrong band by reading that band's own value.
    """

    @pytest.fixture(scope="class")
    def design(self):
        """A four-parameter design with omega_0 = 128, analysis band [1, 64]."""
        problem = Problem(names=tuple(f"x{i}" for i in range(4)), bounds=((0, 1),) * 4)
        return sample(problem, n_per_curve=1025, M=4, seed=7)

    def test_kernel_counts_the_top_of_the_band_and_nothing_above(self):
        """Tier T0 (analytic): ``analysis_max`` means what its name says.

        Feed ``_compute_indices`` a cosine at a chosen frequency and read ST
        back. A tone inside the complementary band gives ``Dt = V`` and so
        ``ST = 0``; a tone above the band gives ``Dt = 0`` and so ``ST = 1``.
        The expectation comes from Parseval and the DFT of a cosine, so this
        pins the exact meaning of the ``analysis_max`` argument: the top
        frequency counted, inclusive.
        """
        n, m, omega_0 = 1025, 4, 128
        for analysis_max in (17, 40, 64):
            inside = _compute_indices(
                jnp.asarray(_tone(analysis_max, n)), n, m, omega_0, analysis_max
            )[1]
            above = _compute_indices(
                jnp.asarray(_tone(analysis_max + 1, n)), n, m, omega_0, analysis_max
            )[1]
            assert float(inside) == pytest.approx(0.0, abs=1e-6), analysis_max
            assert float(above) == pytest.approx(1.0, abs=1e-6), analysis_max

    def test_analyzer_band_is_the_plan_band(self, design):
        """Tier T0/T4: the band ``analyze`` integrates over, measured.

        Sweep a pure tone across the frequency axis and record which
        frequencies ``analyze`` charges to the complementary variance. The
        counted set must be exactly ``[1, analysis_max]`` — contiguous, and
        stopping at the plan's edge. A band that is one bin short, or eight
        bins too wide, moves the measured edge and fails here, even though
        neither changes any value the plan itself reports.
        """
        plan = _frequency_plan(design.problem.num_vars, design.n_per_curve, design.M)
        counted = [f for f in range(1, plan.omega_0) if _st_for_tone(design, f) < 0.5]
        assert counted == list(range(1, plan.analysis_max + 1))

    def test_assigned_carriers_lie_in_the_measured_band(self, design):
        """Tier T4 (behavioural): the sampler puts power where analyze looks.

        Recover the complementary carriers from the sampled design by FFT,
        not from the plan, and recover the analyzer's band edge by the tone
        sweep, not from the plan. Every carrier must sit at or below that
        edge. If the sampler and the analyzer ever drift apart in either
        direction, carrier power lands outside the band that is summed and
        this fails.
        """
        n = design.n_per_curve
        omega_0 = int(np.argmax(np.abs(np.fft.rfft(_column(design, 0, 0)))))
        edge = max(f for f in range(1, omega_0) if _st_for_tone(design, f) < 0.5)

        carriers = set()
        for i in range(design.problem.num_vars):
            for j in range(design.problem.num_vars):
                if j == i:
                    continue
                col = _column(design, i, j)
                carriers.add(int(np.argmax(np.abs(np.fft.rfft(col)))))
        assert carriers, "design has no non-focal columns to recover"
        assert min(carriers) >= 1
        assert max(carriers) <= edge, (sorted(carriers), edge, n)

    def test_focal_frequency_is_recoverable_from_the_design(self, design):
        """Tier T4 (behavioural): column i of curve i really runs at omega_0.

        The sampler laid the design out on the plan's frequencies. Recover the
        focal frequency from the sample matrix itself: on curve ``i``, column
        ``i`` oscillates at ``omega_0``, so its spectrum peaks at that bin.
        """
        plan = _frequency_plan(design.problem.num_vars, design.n_per_curve, design.M)
        for i in range(design.problem.num_vars):
            spectrum = np.abs(np.fft.rfft(_column(design, i, i)))
            assert int(np.argmax(spectrum)) == plan.omega_0, i


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


class TestAffineInvariance:
    """``Y -> a*Y + b`` must not move an eFAST index.

    Tier T0 (closed form): the indices are ratios of Fourier power in
    different frequency bands, and the shift lands entirely in the excluded
    DC term while the scale cancels between numerator and denominator. No
    keyword is involved: eFAST has no output-standardization switch, because
    there is nothing for one to do.
    """

    def test_indices_are_affine_invariant(self):
        sr = sample(ishigami.PROBLEM, n_per_curve=4096, M=4, seed=42)
        Y = ishigami.evaluate(jnp.asarray(sr.samples))
        baseline = analyze(sr, jnp.asarray(Y))
        result = analyze(sr, jnp.asarray(3.0 * Y + 1e4))
        np.testing.assert_allclose(np.asarray(result.S1), np.asarray(baseline.S1), atol=1e-4)
        np.testing.assert_allclose(np.asarray(result.ST), np.asarray(baseline.ST), atol=1e-4)


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

    def test_trailing_chunk_pad_keeps_dtype_under_x64(self):
        """T4: the padded trailing chunk stays in the outputs' own dtype.

        Under x64 an unannotated ``jnp.zeros`` pad is float64, so float32
        outputs promoted on the trailing chunk only: the kernel re-traced at
        a second dtype, the trailing slices were computed at a different
        precision, and the concatenated indices came back float64. The pad
        now carries the batch's dtype, so a chunk size that does not divide
        the work must give the same numbers and the same dtype as the
        un-chunked run.
        """
        with jax.enable_x64():
            sr = sample(ishigami.PROBLEM, n_per_curve=257, M=4, seed=42)
            X32 = jnp.asarray(sr.samples, dtype=jnp.float32)
            Y = ishigami.evaluate(X32)
            Y_multi = jnp.stack([Y, 2.0 * Y], axis=-1)  # (n_runs, 2), float32
            assert Y_multi.dtype == jnp.float32

            # D=3, K=2 -> 6 slices; chunks of 4 leave a padded trailing 2.
            S1_full, ST_full = indices(sr, Y_multi)
            S1_chunk, ST_chunk = indices(sr, Y_multi, slice_chunk_size=4)

            assert S1_chunk.dtype == S1_full.dtype
            assert ST_chunk.dtype == ST_full.dtype
            np.testing.assert_allclose(
                np.asarray(S1_chunk), np.asarray(S1_full), rtol=1e-6, atol=1e-7
            )
            np.testing.assert_allclose(
                np.asarray(ST_chunk), np.asarray(ST_full), rtol=1e-6, atol=1e-7
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
        """The ``time_coords=`` kwarg lands on the time coordinate.

        The dims and data-var names are pinned for every method and Y-rank by
        ``tests/data/result_dataset_schema.json``, so only the kwarg — a
        coordinate *value* the caller supplied — is asserted here.
        """
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
        assert list(ds.coords["time"].values) == time_coords


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


class TestOnInvalidPolicy:
    """T4 (behavioural): efast.analyze applies the shared non-finite policy.

    eFAST is the one method where dropping is undefined. A search curve is an
    ordered sweep read by a discrete Fourier transform, so removing a point
    does not shrink the sample: it changes the sequence the transform reads,
    and every harmonic with it.
    """

    N = 257

    def _design_and_Y(self):
        """An Ishigami eFAST design plus its finite outputs, one row per run."""
        sr = sample(ishigami.PROBLEM, n_per_curve=self.N, M=4, seed=3)
        Y = np.asarray(ishigami.evaluate(jnp.asarray(sr.samples)), dtype=np.float64)
        return sr, Y

    def _poison(self, sr, Y: np.ndarray, curve: int) -> tuple[np.ndarray, int]:
        """Set one NaN inside one search curve, and return the row it hit."""
        row = curve * self.N + 7
        poisoned = Y.copy()
        poisoned[row] = np.nan
        return poisoned, row

    def test_drop_is_refused_and_the_message_names_the_search_curve(self):
        """T4: 'drop' raises, because a curve cannot lose a point and stay valid."""
        sr, Y = self._design_and_Y()
        with pytest.raises(ValueError, match="not available for this method") as exc:
            analyze(sr, jnp.asarray(Y), on_invalid="drop")
        message = str(exc.value)
        assert "jaxgsa.efast.analyze" in message
        assert "search curve" in message
        assert "'raise'" in message and "'propagate'" in message

    def test_raise_is_the_default_and_names_the_curve_and_its_rows(self):
        """T4: the default refuses, and locates the failure in the design.

        The reported rows are the whole curve, not the single bad row: the
        curve is the unit, and every one of its rows feeds the transform that
        the failure invalidated.
        """
        sr, Y = self._design_and_Y()
        bad, row = self._poison(sr, Y, curve=1)
        with pytest.raises(ValueError, match="1 of 3 search curves") as exc:
            analyze(sr, jnp.asarray(bad))
        message = str(exc.value)
        assert "jaxgsa.efast.analyze" in message
        assert "They condemn search curves [1]" in message
        assert f"which covers {self.N} rows" in message
        assert row // self.N == 1

    def test_propagate_warns_and_lets_the_value_reach_the_indices(self):
        """T4: nothing is removed, so the transform returns non-finite indices."""
        sr, Y = self._design_and_Y()
        bad, _ = self._poison(sr, Y, curve=1)
        with pytest.warns(JaxgsaWarning, match="reaches the indices"):
            result = analyze(sr, jnp.asarray(bad), on_invalid="propagate")
        assert not np.all(np.isfinite(np.asarray(result.S1)))
        assert result.invalid.policy == "propagate"
        assert result.invalid.unit is InvalidUnit.CURVE

    @pytest.mark.parametrize("policy", ["raise", "propagate"])
    def test_a_clean_sample_reports_nothing_and_stays_silent(self, policy, recwarn):
        """T4: a clean run gives an empty report under both usable policies."""
        sr, Y = self._design_and_Y()
        result = analyze(sr, jnp.asarray(Y), on_invalid=policy)
        assert result.invalid.n_invalid == 0
        assert result.invalid.n_units == 3
        assert result.invalid.unit is InvalidUnit.CURVE
        assert [w for w in recwarn if issubclass(w.category, JaxgsaWarning)] == []


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


class TestPureCore:
    """The transformable core ``efast.indices``.

    Tier T4 throughout (internal consistency and transformability): the
    accuracy of the estimator itself is checked against the closed-form
    Ishigami and linear values elsewhere in this file, and ``indices`` runs
    the same kernel ``analyze`` runs.
    """

    @staticmethod
    def _design_and_Y():
        """Return an Ishigami eFAST design and its scalar outputs."""
        sr = sample(ishigami.PROBLEM, n_per_curve=257, M=4, seed=7)
        return sr, ishigami.evaluate(jnp.asarray(sr.samples))

    def test_matches_analyze_scalar(self):
        """T4: ``indices`` returns exactly what ``analyze`` reports."""
        sr, Y = self._design_and_Y()

        result = analyze(sr, Y)
        S1, ST = indices(sr, Y)

        np.testing.assert_array_equal(np.asarray(S1), np.asarray(result.S1))
        np.testing.assert_array_equal(np.asarray(ST), np.asarray(result.ST))

    def test_matches_analyze_time_series(self):
        """T4: the batched path agrees with ``analyze`` for ``(n, T, K)`` outputs."""
        sr, base = self._design_and_Y()
        X = jnp.asarray(sr.samples)
        # Four different functions of the inputs, so a mis-transposed batch
        # axis cannot pass.
        Y = jnp.stack(
            [
                jnp.stack([base, 2.0 * base + X[:, 0]], axis=-1),
                jnp.stack([base**2, X[:, 1]], axis=-1),
            ],
            axis=1,
        )
        assert Y.shape == (sr.n_runs, 2, 2)

        S1, ST = indices(sr, Y)
        result = analyze(sr, Y)

        assert S1.shape == (2, 2, 3)
        np.testing.assert_array_equal(np.asarray(S1), np.asarray(result.S1))
        np.testing.assert_array_equal(np.asarray(ST), np.asarray(result.ST))

    def test_returns_a_bare_tuple(self):
        """T4: the core hands back arrays, with no result object attached."""
        sr, Y = self._design_and_Y()

        out = indices(sr, Y)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert all(isinstance(a, jax.Array) for a in out)

    def test_is_jittable(self):
        """T4: ``jit`` traces ``indices`` and returns the eager values."""
        sr, Y = self._design_and_Y()

        # The design is a closure constant: it holds NumPy arrays, not tracers.
        jitted = jax.jit(lambda outputs: indices(sr, outputs))
        S1_jit, ST_jit = jitted(Y)
        S1, ST = indices(sr, Y)

        np.testing.assert_allclose(np.asarray(S1_jit), np.asarray(S1), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(ST_jit), np.asarray(ST), rtol=1e-6)

    def test_is_vmappable(self):
        """T4: ``vmap`` maps ``indices`` over a batch of output vectors."""
        sr, base = self._design_and_Y()
        X = jnp.asarray(sr.samples)
        batch = jnp.stack([base, X[:, 0], X[:, 1] ** 2])

        S1_batch, ST_batch = jax.vmap(lambda outputs: indices(sr, outputs))(batch)

        assert S1_batch.shape == (3, 3)
        for row, outputs in enumerate(batch):
            S1, ST = indices(sr, outputs)
            # vmap reassociates the reductions, so float32 agreement is to a
            # few ULP of the index scale rather than bitwise.
            np.testing.assert_allclose(
                np.asarray(S1_batch[row]), np.asarray(S1), rtol=1e-5, atol=1e-6
            )
            np.testing.assert_allclose(
                np.asarray(ST_batch[row]), np.asarray(ST), rtol=1e-5, atol=1e-6
            )

    def test_jit_of_jacrev(self):
        """T4: ``jit(jacrev(...))`` compiles a scalar-to-index chain.

        The differentiated quantity is a scale on the design, which changes
        the model outputs and therefore the Fourier amplitudes. The derivative
        is compared against a central finite difference under x64, so this
        checks a number rather than only that the trace succeeded.
        """
        with jax.enable_x64():
            sr = sample(ishigami.PROBLEM, n_per_curve=257, M=4, seed=7)
            X = jnp.asarray(sr.samples, dtype=jnp.float64)

            def total_s1(scale):
                return indices(sr, ishigami.evaluate(X * scale))[0].sum()

            jac = float(jax.jit(jax.jacrev(total_s1))(1.0))
            h = 1e-5
            fd = (float(total_s1(1.0 + h)) - float(total_s1(1.0 - h))) / (2 * h)

        assert np.isfinite(jac)
        np.testing.assert_allclose(jac, fd, rtol=1e-5)

    def test_zero_variance_curve_is_nan_without_a_warning(self, recwarn):
        """T4: the core handles a constant curve in the kernel, not on the host.

        ``analyze`` warns about the same curve. The core cannot, because a
        warning needs a concrete value, so it reports NaN from inside
        ``jnp.where`` and says nothing.
        """
        sr, Y = self._design_and_Y()
        # Exactly zero, so the spectrum is exactly zero and V == 0 holds in
        # floating point. A constant non-zero curve leaves FFT round-off
        # behind and is a different test.
        Y = Y.at[: sr.n_per_curve].set(0.0)

        S1, ST = indices(sr, Y)

        assert np.isnan(np.asarray(S1)[0])
        assert np.isnan(np.asarray(ST)[0])
        assert [w for w in recwarn if issubclass(w.category, JaxgsaWarning)] == []

    def test_rejects_a_row_count_the_design_cannot_hold(self):
        """T4: a shape check still raises; only data-driven policy is dropped."""
        sr, Y = self._design_and_Y()

        with pytest.raises(ValueError, match="rows"):
            indices(sr, Y[:-1])


def test_analyze_offers_no_interval_and_says_why():
    """T4: eFAST has no ``n_bootstrap``, and the docs give the design reason.

    A curve cannot be resampled and there is one curve per parameter, so an
    interval needs replicated designs rather than a keyword. That reason is
    part of the public contract, so it is asserted rather than left to review.
    """
    import inspect

    import jaxgsa.efast as efast_pkg

    assert "n_bootstrap" not in inspect.signature(analyze).parameters
    for text in (analyze.__doc__, efast_pkg.__doc__, efast_pkg._analyze.__doc__):
        assert text is not None
        assert "phase" in text


class TestEFASTSaveLoad:
    """Tier T4 (internal consistency): NPZ persistence round-trips the design."""

    @staticmethod
    def _mixed_problem() -> Problem:
        from jaxgsa.problem import GaussianInputSpec

        return Problem.from_dict(
            {
                "u": (0.0, 2.0),
                "g": GaussianInputSpec(dist="gaussian", mean=1.0, variance=4.0),
            }
        )

    def test_round_trip_preserves_every_field(self, tmp_path):
        problem = self._mixed_problem()
        sr = sample(problem, n_per_curve=128, M=4, seed=7)
        sr.save(tmp_path / "design")

        loaded = EFASTSamples.load(tmp_path / "design")
        np.testing.assert_array_equal(loaded.samples, sr.samples)
        assert loaded.samples.dtype == sr.samples.dtype
        assert loaded.n_per_curve == sr.n_per_curve
        assert loaded.M == sr.M
        assert loaded.problem == sr.problem

    def test_npz_suffix_is_appended_like_the_other_designs(self, tmp_path):
        sr = sample(self._mixed_problem(), n_per_curve=80, M=4, seed=0)
        sr.save(tmp_path / "run.A")
        assert (tmp_path / "run.A.npz").exists()
        loaded = EFASTSamples.load(tmp_path / "run.A")
        np.testing.assert_array_equal(loaded.samples, sr.samples)

    def test_metadata_matches_the_shared_format_family(self, tmp_path):
        """The metadata blob carries the same common keys as the base class."""
        import json

        sr = sample(self._mixed_problem(), n_per_curve=80, M=4, seed=0)
        sr.save(tmp_path / "design")
        with np.load(tmp_path / "design.npz", allow_pickle=False) as data:
            meta = json.loads(data["metadata"].item())
        assert set(data.files) == {"samples", "metadata"}
        assert "jaxgsa_version" in meta
        assert meta["problem"]["names"] == ["u", "g"]
        assert meta["n_per_curve"] == 80
        assert meta["M"] == 4

    def test_analyze_after_load_equals_analyze_before(self, tmp_path):
        problem = self._mixed_problem()
        sr = sample(problem, n_per_curve=256, M=4, seed=11)
        Y = jnp.asarray(np.sin(sr.samples[:, 0]) + sr.samples[:, 1] ** 2)
        sr.save(tmp_path / "design")

        loaded = EFASTSamples.load(tmp_path / "design")
        before = analyze(sr, Y)
        after = analyze(loaded, Y)
        np.testing.assert_array_equal(np.asarray(before.S1), np.asarray(after.S1))
        np.testing.assert_array_equal(np.asarray(before.ST), np.asarray(after.ST))
        assert after.problem == before.problem

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EFASTSamples.load(tmp_path / "nope")
