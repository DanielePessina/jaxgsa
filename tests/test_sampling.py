import numpy as np
import pytest
from scipy.stats import truncnorm

from gsax.problem import GaussianInputSpec, InputSpecValue, Problem, UniformInputSpec
from gsax.sobol import sample
from gsax.sobol._sampling import _next_power_of_2, _saltelli_step


def test_next_power_of_2():
    assert _next_power_of_2(1) == 1
    assert _next_power_of_2(2) == 2
    assert _next_power_of_2(3) == 4
    assert _next_power_of_2(5) == 8
    assert _next_power_of_2(1024) == 1024
    assert _next_power_of_2(1025) == 2048


def test_sample_returns_unique_rows():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)})
    result = sample(p, n_samples=100, seed=42, verbose=False)
    assert result.n_total >= 100
    assert result.samples.shape == (result.n_total, p.num_vars)
    assert np.unique(result.samples, axis=0).shape[0] == result.n_total
    assert result.sample_ids.tolist() == list(range(result.n_total))
    assert result.expanded_n_total == result.base_n * _saltelli_step(p.num_vars, True)
    assert result.expanded_to_unique.shape == (result.expanded_n_total,)
    assert result.expanded_to_unique.max() < result.n_total
    assert result.n_params == p.num_vars
    assert result.calc_second_order is True


def test_sample_within_bounds():
    p = Problem.from_dict({"x1": (-5.0, 5.0), "x2": (0.0, 10.0)})
    result = sample(p, n_samples=200, seed=42, verbose=False)
    assert np.all(result.samples[:, 0] >= -5.0)
    assert np.all(result.samples[:, 0] <= 5.0)
    assert np.all(result.samples[:, 1] >= 0.0)
    assert np.all(result.samples[:, 1] <= 10.0)


def test_power_of_2_enforcement():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=100, seed=42, verbose=False)
    assert result.base_n & (result.base_n - 1) == 0


def test_no_second_order_expanded_count():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=100, calc_second_order=False, seed=42, verbose=False)
    assert result.calc_second_order is False
    assert result.expanded_n_total == result.base_n * _saltelli_step(p.num_vars, False)


def test_single_parameter_mapping_collapses_duplicates():
    p = Problem.from_dict({"x1": (0.0, 1.0)})

    first_only = sample(p, n_samples=16, calc_second_order=False, seed=42, verbose=False)
    step = _saltelli_step(p.num_vars, False)
    for i in range(first_only.base_n):
        group = first_only.expanded_to_unique[i * step : (i + 1) * step]
        assert group[1] == group[2]
        assert group[0] != group[1]

    second_order = sample(p, n_samples=16, calc_second_order=True, seed=42, verbose=False)
    step = _saltelli_step(p.num_vars, True)
    for i in range(second_order.base_n):
        group = second_order.expanded_to_unique[i * step : (i + 1) * step]
        assert group[0] == group[2]
        assert group[1] == group[3]
        assert group[0] != group[1]


def test_two_parameter_second_order_mapping_collapses_cross_duplicates():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=32, calc_second_order=True, seed=42, verbose=False)
    step = _saltelli_step(p.num_vars, True)

    for i in range(result.base_n):
        group = result.expanded_to_unique[i * step : (i + 1) * step]
        assert group[1] == group[4]
        assert group[2] == group[3]
        assert len(set(group.tolist())) == 4


def test_reconstructing_expanded_samples_matches_mapping():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=64, seed=42, verbose=False)
    reconstructed = result.samples[result.expanded_to_unique]
    assert reconstructed.shape == (result.expanded_n_total, p.num_vars)
    assert np.unique(reconstructed, axis=0).shape[0] == result.n_total


def test_sample_verbose_prints_summary(capsys):
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sample(p, n_samples=32, seed=42, verbose=True)
    out = capsys.readouterr().out
    assert "gsax.sobol.sample:" in out
    assert "requested_unique>=" in out
    assert "returned_unique=" in out
    assert "duplicates_removed=" in out


def test_sample_verbose_false_is_silent(capsys):
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sample(p, n_samples=32, seed=42, verbose=False)
    out = capsys.readouterr().out
    assert out == ""


def test_mixed_distributions_preserve_sampling_metadata():
    p = Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "x2": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
            "x3": GaussianInputSpec(dist="gaussian", mean=1.0, variance=4.0, low=-2.0),
        }
    )

    result = sample(p, n_samples=128, calc_second_order=False, seed=42, verbose=False)
    assert result.n_total >= 128
    assert result.samples.shape == (result.n_total, p.num_vars)
    assert result.expanded_n_total == result.base_n * _saltelli_step(p.num_vars, False)
    assert result.expanded_to_unique.shape == (result.expanded_n_total,)
    assert result.problem.has_non_uniform_inputs is True


def test_uniform_columns_stay_within_bounds_for_mixed_problem():
    p = Problem.from_dict(
        {
            "uniform": UniformInputSpec(dist="uniform", low=-3.0, high=2.0),
            "gaussian": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
        }
    )

    result = sample(p, n_samples=256, seed=1, verbose=False)
    assert np.all(result.samples[:, 0] >= -3.0)
    assert np.all(result.samples[:, 0] <= 2.0)


def test_gaussian_column_matches_target_mean_and_variance():
    p = Problem.from_dict(
        {
            "x1": GaussianInputSpec(dist="gaussian", mean=1.5, variance=2.25),
            "x2": (0.0, 1.0),
        }
    )

    result = sample(p, n_samples=4096, calc_second_order=False, seed=123, verbose=False)
    gaussian = result.samples[:, 0]

    assert abs(np.mean(gaussian) - 1.5) < 0.05
    assert abs(np.var(gaussian) - 2.25) < 0.08


def test_truncated_gaussian_columns_respect_one_sided_and_two_sided_bounds():
    p = Problem.from_dict(
        {
            "lower_only": GaussianInputSpec(
                dist="gaussian",
                mean=0.0,
                variance=1.0,
                low=-0.25,
            ),
            "upper_only": GaussianInputSpec(
                dist="gaussian",
                mean=0.0,
                variance=1.0,
                high=0.5,
            ),
            "two_sided": GaussianInputSpec(
                dist="gaussian",
                mean=0.0,
                variance=1.0,
                low=-1.0,
                high=1.0,
            ),
        }
    )

    result = sample(p, n_samples=512, calc_second_order=False, seed=7, verbose=False)
    assert np.all(result.samples[:, 0] >= -0.25)
    assert np.all(result.samples[:, 1] <= 0.5)
    assert np.all(result.samples[:, 2] >= -1.0)
    assert np.all(result.samples[:, 2] <= 1.0)


def test_two_sided_truncated_gaussian_matches_target_variance_formula():
    p = Problem.from_dict(
        {
            "x": GaussianInputSpec(
                dist="gaussian",
                mean=0.5,
                variance=1.44,
                low=-0.5,
                high=1.5,
            )
        }
    )

    result = sample(p, n_samples=4096, calc_second_order=False, seed=99, verbose=False)
    observed = np.var(result.samples[:, 0])
    std = np.sqrt(1.44)
    a = (-0.5 - 0.5) / std
    b = (1.5 - 0.5) / std
    expected = truncnorm.var(a, b, loc=0.5, scale=std)
    assert abs(observed - expected) < 0.03


# ---------------------------------------------------------------------------
# Prefix downsampling tests
# ---------------------------------------------------------------------------


class TestSamplingResultDownsample:
    """Tests for SobolSamples.downsample()."""

    def _make_sr(self, D: int = 3, base_n: int = 32, second_order: bool = True, seed: int = 42):
        names: dict[str, InputSpecValue] = {f"x{i}": (0.0, 1.0) for i in range(D)}
        p = Problem.from_dict(names)
        return sample(
            p, n_samples=1, base_n=base_n, calc_second_order=second_order, seed=seed, verbose=False
        )

    def test_identity_when_same_base_n(self):
        sr = self._make_sr(base_n=16)
        assert sr.downsample(16) is sr

    def test_samples_are_prefix(self):
        sr_full = self._make_sr(base_n=64)
        sr_small = sr_full.downsample(16)
        assert np.array_equal(sr_small.samples, sr_full.samples[: sr_small.n_total])

    def test_expanded_n_total_matches_step(self):
        sr_full = self._make_sr(D=4, base_n=32, second_order=True)
        sr_small = sr_full.downsample(8)
        step = _saltelli_step(4, True)
        assert sr_small.expanded_n_total == 8 * step

    def test_expanded_to_unique_is_consistent(self):
        sr_full = self._make_sr(base_n=64)
        sr_small = sr_full.downsample(16)
        assert sr_small.expanded_to_unique.shape == (sr_small.expanded_n_total,)
        assert sr_small.expanded_to_unique.max() < sr_small.n_total

    def test_base_n_stored(self):
        sr_full = self._make_sr(base_n=32)
        sr_small = sr_full.downsample(8)
        assert sr_small.base_n == 8

    def test_multiple_rungs_are_nested(self):
        sr_full = self._make_sr(base_n=64)
        sr_32 = sr_full.downsample(32)
        sr_16 = sr_full.downsample(16)
        sr_8 = sr_full.downsample(8)
        assert sr_8.n_total <= sr_16.n_total <= sr_32.n_total <= sr_full.n_total
        assert np.array_equal(sr_8.samples, sr_16.samples[: sr_8.n_total])
        assert np.array_equal(sr_16.samples, sr_32.samples[: sr_16.n_total])

    def test_first_order_only(self):
        sr_full = self._make_sr(base_n=32, second_order=False)
        sr_small = sr_full.downsample(8)
        step = _saltelli_step(3, False)
        assert sr_small.expanded_n_total == 8 * step
        assert sr_small.calc_second_order is False

    def test_upsample_raises(self):
        sr = self._make_sr(base_n=16)
        with pytest.raises(ValueError, match="Cannot upsample"):
            sr.downsample(32)

    def test_non_power_of_two_raises(self):
        sr = self._make_sr(base_n=16)
        with pytest.raises(ValueError, match="power of 2"):
            sr.downsample(12)

    def test_single_param_with_duplicates(self):
        sr_full = self._make_sr(D=1, base_n=32, second_order=True)
        sr_small = sr_full.downsample(8)
        reconstructed = sr_small.samples[sr_small.expanded_to_unique]
        assert reconstructed.shape == (sr_small.expanded_n_total, 1)

    def test_problem_preserved(self):
        sr_full = self._make_sr()
        sr_small = sr_full.downsample(8)
        assert sr_small.problem is sr_full.problem
        assert sr_small.n_params == sr_full.n_params

    def test_with_Y_returns_tuple(self):
        sr_full = self._make_sr(base_n=32)
        Y = np.arange(sr_full.n_total * 4, dtype=np.float64).reshape(sr_full.n_total, 4)
        sr_small, Y_small = sr_full.downsample(8, Y)
        assert Y_small.shape == (sr_small.n_total, 4)
        assert np.array_equal(Y_small, Y[: sr_small.n_total])

    def test_with_Y_identity_returns_same_Y(self):
        sr_full = self._make_sr(base_n=16)
        Y = np.ones((sr_full.n_total, 3))
        sr_same, Y_same = sr_full.downsample(16, Y)
        assert sr_same is sr_full
        assert Y_same is Y

    def test_with_Y_misaligned_raises(self):
        sr_full = self._make_sr(base_n=32)
        Y_wrong = np.zeros((sr_full.n_total + 5, 3))
        with pytest.raises(ValueError, match="does not match n_total"):
            sr_full.downsample(8, Y_wrong)

    def test_samples_do_not_share_memory(self):
        sr_full = self._make_sr(base_n=32)
        sr_small = sr_full.downsample(8)
        assert not np.shares_memory(sr_full.samples, sr_small.samples)

    def test_Y_does_not_share_memory(self):
        sr_full = self._make_sr(base_n=32)
        Y = np.ones((sr_full.n_total, 3))
        _, Y_small = sr_full.downsample(8, Y)
        assert not np.shares_memory(Y, Y_small)

    def test_downsample_is_bit_identical_to_direct_draw(self):
        """Prefix property: downsampling to K equals drawing K base points directly.

        This backs the ``downsample`` docstring claim that the first K base
        points of a draw with N > K base points are bit-identical to drawing
        K base points with the same seed and scramble.
        """
        p = Problem.from_dict(
            {
                "uniform": UniformInputSpec(dist="uniform", low=-2.0, high=3.0),
                "gaussian": GaussianInputSpec(
                    dist="gaussian", mean=1.0, variance=4.0, low=-1.0, high=4.0
                ),
            }
        )
        N, K, seed = 64, 16, 1234

        sr_small = sample(p, n_samples=1, base_n=N, seed=seed, verbose=False).downsample(K)
        sr_direct = sample(p, n_samples=1, base_n=K, seed=seed, verbose=False)

        np.testing.assert_array_equal(sr_small.samples, sr_direct.samples)
        np.testing.assert_array_equal(sr_small.expanded_to_unique, sr_direct.expanded_to_unique)
        np.testing.assert_array_equal(sr_small.sample_ids, sr_direct.sample_ids)
        assert sr_small.expanded_n_total == sr_direct.expanded_n_total
        assert sr_small.base_n == sr_direct.base_n == K
