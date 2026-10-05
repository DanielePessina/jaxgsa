"""Custom conditional integration and VKOGA forwarding checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import vkoga
from jaxgsa.problem import Problem
from jaxgsa.vkoga._custom_indices import checked_unit_samples, estimate_custom_indices

PROBLEM = Problem(names=("a", "b"), bounds=((0.0, 1.0),) * 2)


def joint(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw independent uniform inputs."""
    return rng.random((n, 2))


def conditional(
    anchors: np.ndarray, fixed: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """Resample unfixed columns independently."""
    result = joint(len(anchors), rng)
    result[:, fixed] = anchors[:, fixed]
    return result


def test_inner_noise_corrected_for_irrelevant_input() -> None:
    """Small IID inner budgets must not systematically inflate first order."""
    result = estimate_custom_indices(
        problem=PROBLEM,
        joint_sampler=joint,
        conditional_sampler=conditional,
        predict=lambda u: u[:, :1],
        n_outer=4000,
        n_inner=2,
        n_variance=16000,
        entropy=3,
    )
    assert abs(result.S_TC[0, 1]) < 0.025
    np.testing.assert_allclose(result.S_TU[0], [1, 0], atol=0.035)
    np.testing.assert_allclose(result.S_U[0], [1, 0], atol=0.035)


def test_correlated_nonlinear_distribution() -> None:
    """A sign mixture has Gaussian marginals but non-Gaussian dependence."""
    problem = Problem.from_dict(
        {
            "a": {"dist": "gaussian", "mean": 0, "variance": 1},
            "b": {"dist": "gaussian", "mean": 0, "variance": 1},
        }
    )

    def draw(n: int, rng: np.random.Generator) -> np.ndarray:
        a = rng.normal(size=n)
        return np.column_stack((a, a * rng.choice([-1, 1], size=n)))

    def given(x: np.ndarray, fixed: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        y = x.copy()
        i = 1 - fixed[0]
        y[:, i] = x[:, fixed[0]] * rng.choice([-1, 1], size=len(x))
        return y

    from scipy.stats import norm

    with jax.enable_x64():
        result = estimate_custom_indices(
            problem=problem,
            joint_sampler=draw,
            conditional_sampler=given,
            predict=lambda u: norm.ppf(u[:, :1]) ** 2,
            n_outer=5000,
            n_inner=2,
            n_variance=50000,
            entropy=8,
        )
    np.testing.assert_allclose(result.S_TC, [[1, 1]], atol=0.15)
    np.testing.assert_allclose(result.S_TU, [[0, 0]], atol=1e-12)


def test_analyze_reproducible_batched_bootstrap_and_counts() -> None:
    """Custom callbacks reach integration and bootstrap without QMC rounding."""
    counts: list[int] = []

    def draw(n: int, rng: np.random.Generator) -> np.ndarray:
        counts.append(n)
        return joint(n, rng)

    with jax.enable_x64():
        x = jnp.asarray(joint(70, np.random.default_rng(1)))

        def run(batch_size: int) -> vkoga.VKOGAResult:
            value = vkoga.analyze(
                PROBLEM,
                x,
                x[:, 0],
                batch_size=batch_size,
                joint_sampler=draw,
                conditional_sampler=conditional,
                gamma=2.0,
                ridge=1e-6,
                max_centers=15,
                n_outer=9,
                n_inner=3,
                n_variance=31,
                n_bootstrap=2,
                key=jax.random.key(2),
                verbose=False,
            )
            assert isinstance(value, vkoga.VKOGAResult)
            return value

        result = run(3)
        other = run(11)
    assert set(counts) == {9, 31}
    assert result.correlation is None
    assert result.is_correlated is None
    assert result.dependence_source == "custom"
    assert result.S_TC_conf is not None
    assert other.S_TC_conf is not None
    np.testing.assert_allclose(result.S_TC, other.S_TC, atol=1e-12)
    np.testing.assert_allclose(result.S_TC_conf, other.S_TC_conf, atol=1e-12)


def test_support_rejected() -> None:
    """Bound violations cannot be hidden by marginal CDF clipping."""
    with pytest.raises(ValueError, match="outside declared support"):
        checked_unit_samples(np.array([[1.01, 0.2]]), PROBLEM)


def test_conflicting_correlation_rejected_before_fit() -> None:
    """The custom path must have one explicit distribution source."""
    with pytest.raises(ValueError, match="cannot be combined"):
        vkoga.analyze(
            PROBLEM,
            jnp.ones((4, 2)),
            jnp.ones(4),
            joint_sampler=joint,
            conditional_sampler=conditional,
            correlation=np.eye(2),
        )


def test_correlated_gaussian_conditional_indices() -> None:
    """Check both conditional directions against a linear Gaussian reference."""
    from scipy.stats import norm

    problem = Problem.from_dict(
        {
            "a": {"dist": "gaussian", "mean": 0, "variance": 1},
            "b": {"dist": "gaussian", "mean": 0, "variance": 1},
        }
    )
    rho = 0.6

    def draw(n: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.normal(size=(n, 2))
        z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
        return z

    def given(x: np.ndarray, fixed: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        y = x.copy()
        i = 1 - fixed[0]
        y[:, i] = rho * x[:, fixed[0]] + np.sqrt(1 - rho**2) * rng.normal(size=len(x))
        return y

    with jax.enable_x64():
        result = estimate_custom_indices(
            problem=problem,
            joint_sampler=draw,
            conditional_sampler=given,
            predict=lambda u: (norm.ppf(u) @ np.array([1.0, 2.0]))[:, None],
            n_outer=2000,
            n_inner=8,
            n_variance=30000,
            entropy=19,
        )
    variance = 5 + 4 * rho
    np.testing.assert_allclose(
        result.S_TC[0], np.array([1 + 2 * rho, 2 + rho]) ** 2 / variance, atol=0.065
    )
    expected_total = np.array([1.0, 4.0]) * (1 - rho**2) / variance
    np.testing.assert_allclose(result.S_TU[0], expected_total, atol=0.03)
    np.testing.assert_allclose(result.S_U[0], expected_total, atol=0.03)


def test_irregular_channels_forward_custom_samplers() -> None:
    """Each irregular channel integrates the explicitly supplied distribution."""
    from jaxgsa._core.irregular import IrregularResult

    with jax.enable_x64():
        x = jnp.asarray(joint(45, np.random.default_rng(4)))
        y = {
            "first": (jnp.array([0.0, 1.0]), jnp.stack((x[:, 0], x[:, 0] + 1), axis=1)),
            "second": (jnp.array([0.0]), x[:, 1:2]),
        }
        result = vkoga.analyze(
            PROBLEM,
            x,
            y,
            joint_sampler=joint,
            conditional_sampler=conditional,
            gamma=2.0,
            ridge=1e-6,
            max_centers=10,
            n_outer=5,
            n_inner=2,
            n_variance=25,
            key=jax.random.key(6),
            verbose=False,
        )
    assert isinstance(result, IrregularResult)
    assert set(result.channels) == {"first", "second"}
    for channel in result.channels.values():
        assert isinstance(channel, vkoga.VKOGAResult)
        assert channel.dependence_source == "custom"
        assert channel.correlation is None


@pytest.mark.verbose_output
def test_time_multi_output_export_and_receipt(capsys: pytest.CaptureFixture[str]) -> None:
    """Custom dependence stays explicit in the receipt and exported metadata."""
    with jax.enable_x64():
        x = jnp.asarray(joint(45, np.random.default_rng(4)))
        outputs = jnp.stack((x[:, 0], x[:, 1]), axis=1)
        y = jnp.stack((outputs, 2 * outputs), axis=1)
        result = vkoga.analyze(
            PROBLEM,
            x,
            y,
            joint_sampler=joint,
            conditional_sampler=conditional,
            gamma=2.0,
            ridge=1e-6,
            max_centers=10,
            n_outer=5,
            n_inner=2,
            n_variance=25,
            key=jax.random.key(6),
        )
    assert isinstance(result, vkoga.VKOGAResult)
    assert result.S_TC.shape == (2, 2, 2)
    dataset = result.to_dataset()
    assert dataset.attrs["dependence_source"] == "custom"
    assert "correlation" not in dataset
    assert "custom (user-supplied conditionals)" in capsys.readouterr().out
