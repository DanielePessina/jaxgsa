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


def sampler(
    u: np.ndarray,
    *,
    conditioning_rows: np.ndarray | None = None,
    fixed_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Transform uniforms and preserve any conditioning columns."""
    result = u.copy()
    if fixed_indices:
        assert conditioning_rows is not None
        result[:, fixed_indices] = conditioning_rows[:, fixed_indices]
    return result


def test_inner_noise_corrected_for_irrelevant_input() -> None:
    """Small IID inner budgets must not systematically inflate first order."""
    result = estimate_custom_indices(
        problem=PROBLEM,
        sampler=sampler,
        sequence="monte_carlo",
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

    def draw(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        from scipy.special import ndtri

        signs = np.where(u[:, 1] < 0.5, -1, 1)
        if not fixed_indices:
            a = ndtri(u[:, 0])
            return np.column_stack((a, a * signs))
        assert conditioning_rows is not None
        y = conditioning_rows.copy()
        i = 1 - fixed_indices[0]
        y[:, i] = y[:, fixed_indices[0]] * signs
        return y

    from scipy.stats import norm

    with jax.enable_x64():
        result = estimate_custom_indices(
            problem=problem,
            sampler=draw,
            sequence="sobol",
            predict=lambda u: norm.ppf(u[:, :1]) ** 2,
            n_outer=2048,
            n_inner=2,
            n_variance=16384,
            entropy=8,
        )
    np.testing.assert_allclose(result.S_TC, [[1, 1]], atol=0.15)
    np.testing.assert_allclose(result.S_TU, [[0, 0]], atol=1e-12)


def test_analyze_reproducible_batched_bootstrap_and_counts() -> None:
    """Custom callbacks reach integration and bootstrap without QMC rounding."""
    counts: list[int] = []

    def draw(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        counts.append(len(u))
        return sampler(u, conditioning_rows=conditioning_rows, fixed_indices=fixed_indices)

    with jax.enable_x64():
        x = jnp.asarray(joint(70, np.random.default_rng(1)))

        def run(batch_size: int) -> vkoga.VKOGAResult:
            value = vkoga.analyze(
                PROBLEM,
                x,
                x[:, 0],
                batch_size=batch_size,
                sampler=draw,
                sequence="monte_carlo",
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
    assert set(counts) == {1, 3, 9, 31}
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
            sampler=sampler,
            sequence="monte_carlo",
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

    def draw(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        z = norm.ppf(u)
        if not fixed_indices:
            z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
            return z
        assert conditioning_rows is not None
        y = conditioning_rows.copy()
        i = 1 - fixed_indices[0]
        y[:, i] = rho * y[:, fixed_indices[0]] + np.sqrt(1 - rho**2) * z[:, i]
        return y

    with jax.enable_x64():
        result = estimate_custom_indices(
            problem=problem,
            sampler=draw,
            sequence="monte_carlo",
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
            sampler=sampler,
            sequence="monte_carlo",
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
            sampler=sampler,
            sequence="monte_carlo",
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


@pytest.mark.parametrize("sequence", ["sobol", "monte_carlo"])
def test_offset_and_small_inner_replicates(sequence: str) -> None:
    """Large constants and tiny inner blocks must not inflate irrelevant indices."""
    with jax.enable_x64():
        results = [
            estimate_custom_indices(
                problem=PROBLEM,
                sampler=sampler,
                sequence=sequence,
                predict=lambda u, offset=offset: u[:, :1] + offset,
                n_outer=256,
                n_inner=2,
                n_variance=4096,
                entropy=31,
            )
            for offset in (0.0, 1e8)
        ]
    if sequence == "sobol":
        for result in results:
            np.testing.assert_allclose(result.S_TC, [[1, 0]], atol=0.06)
            np.testing.assert_allclose(result.S_TU, [[1, 0]], atol=0.06)
    np.testing.assert_allclose(results[0].S_TC, results[1].S_TC, atol=1e-7)
    np.testing.assert_allclose(results[0].S_TU, results[1].S_TU, atol=1e-7)


def test_custom_sobol_counts_are_not_rounded() -> None:
    """Reject an unbalanced Sobol budget before fitting the surrogate."""
    with pytest.raises(ValueError, match="n_outer must be a power of two"):
        vkoga.analyze(
            PROBLEM,
            jnp.ones((4, 2)),
            jnp.ones(4),
            sampler=sampler,
            n_outer=9,
            key=jax.random.key(0),
        )
