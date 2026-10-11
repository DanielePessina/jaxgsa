"""Custom conditional sampling preserves nonlinear dependence and design rows."""

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtri

from jaxgsa import kucherenko
from jaxgsa.problem import Problem

BASE_PROBLEM = Problem.from_dict({"x": (-1, 1), "square": (0, 1)})


def sampler(
    u: np.ndarray,
    *,
    conditioning_rows: np.ndarray | None = None,
    fixed_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Transform uniforms into a nonlinear pair or its conditional draws."""
    if not fixed_indices:
        x = 2 * u[:, 0] - 1
        return np.column_stack((x, x**2))
    assert conditioning_rows is not None
    result = conditioning_rows.copy()
    if fixed_indices == (1,):
        result[:, 0] = np.where(u[:, 0] < 0.5, -1, 1) * np.sqrt(result[:, 1])
    return result


PROBLEM = BASE_PROBLEM.with_sampler(sampler, sampler_dimension=1)


@pytest.mark.verbose_output
def test_nonlinear_dependence_and_repeated_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        design = kucherenko.sample(PROBLEM, base_n=1024, seed=31)
    assert not caught
    assert design.base_n == 1024
    assert design.n_runs == 5 * 1024
    assert design.dependence == "custom"
    assert "custom" in repr(design)
    np.testing.assert_array_equal(design.samples[:1024], design.samples[1024:2048])
    result = kucherenko.analyze(
        design, jnp.asarray(design.samples[:, 1]), n_bootstrap=8, key=jax.random.key(7)
    )
    assert isinstance(result, kucherenko.KucherenkoResult)
    np.testing.assert_allclose(result.S1, [1, 1], atol=0.002)
    np.testing.assert_array_equal(result.ST, [0, 0])
    assert result.dependence == "custom"
    receipt = capsys.readouterr().out
    assert "custom" in receipt
    assert "correlation: independent" not in receipt
    target = tmp_path / "custom.npz"
    design.save(target)
    loaded = kucherenko.KucherenkoSamples.load(target)
    # The file cannot store the sampler, but the loaded problem stays custom:
    # the independence checks still refuse it, and drawing new rows raises.
    assert loaded.problem.dependence == "custom"
    assert loaded.problem.sampler_dimension == PROBLEM.sampler_dimension
    assert loaded.dependence == "custom"
    with pytest.raises(ValueError, match="does not store the custom sampler"):
        kucherenko.sample(loaded.problem, base_n=4, seed=0, verbose=False)
    restored = kucherenko.analyze(loaded, jnp.asarray(loaded.samples[:, 1]), verbose=False)
    assert isinstance(restored, kucherenko.KucherenkoResult)
    assert restored.dependence == "custom"
    np.testing.assert_array_equal(restored.S1, result.S1)
    repeat = kucherenko.sample(PROBLEM, base_n=1024, seed=31, verbose=False)
    np.testing.assert_array_equal(repeat.samples, design.samples)


def test_custom_sample_size_rounds_up_to_power_of_two() -> None:
    design = kucherenko.sample(PROBLEM, 1025 * 5, seed=3, verbose=False)
    assert design.base_n == 2048
    assert design.n_runs == 5 * 2048


def test_conditional_output_must_preserve_fixed_columns() -> None:
    def bad(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        del fixed_indices
        return sampler(u) if conditioning_rows is None else conditioning_rows + 1

    with pytest.raises(ValueError, match="fixed columns"):
        kucherenko.sample(
            BASE_PROBLEM.with_sampler(bad, sampler_dimension=1), base_n=8, verbose=False
        )


def _gaussian_sampler(rho: float):
    scale = np.sqrt(1 - rho**2)

    def gaussian(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        z = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
        if not fixed_indices:
            return np.column_stack((z[:, 0], rho * z[:, 0] + scale * z[:, 1]))
        assert conditioning_rows is not None
        result = conditioning_rows.copy()
        if len(fixed_indices) == 1:
            held = fixed_indices[0]
            result[:, 1 - held] = rho * result[:, held] + scale * z[:, 1 - held]
        else:
            free = ({0, 1} - set(fixed_indices)).pop()
            result[:, free] = rho * result[:, 1 - free] + scale * z[:, free]
        return result

    return gaussian


def test_gaussian_custom_conditionals_match_analytic_indices() -> None:
    rho = 0.7
    problem = Problem.from_dict({"x": (-10, 10), "y": (-10, 10)}).with_sampler(
        _gaussian_sampler(rho), sampler_dimension=2
    )
    # Measured error at base 4096 over eight seeds is under 3e-4, against the
    # tolerances below.
    design = kucherenko.sample(problem, base_n=4096, seed=19, verbose=False)
    result = kucherenko.analyze(design, jnp.asarray(design.samples.sum(axis=1)), verbose=False)
    assert isinstance(result, kucherenko.KucherenkoResult)
    np.testing.assert_allclose(result.S1, [(1 + rho) / 2] * 2, atol=0.015)
    np.testing.assert_allclose(result.ST, [(1 - rho) / 2] * 2, atol=0.01)


def test_independent_custom_sampler_matches_builtin_independent_path() -> None:
    """A custom sampler that draws independently reproduces the built-in path."""
    bounds = (-3.0, 3.0)

    def independent(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        x = bounds[0] + (bounds[1] - bounds[0]) * u
        if not fixed_indices:
            return x[:, :3]
        assert conditioning_rows is not None
        result = conditioning_rows.copy()
        for j in range(3):
            if j not in fixed_indices:
                result[:, j] = x[:, j]
        return result

    def model(x: np.ndarray) -> np.ndarray:
        return x[:, 0] + 0.5 * x[:, 1] ** 2 + 0.3 * x[:, 0] * x[:, 2]

    base = Problem.from_dict({"a": bounds, "b": bounds, "c": bounds})
    custom = kucherenko.sample(
        base.with_sampler(independent, sampler_dimension=3), base_n=2**12, seed=4
    )
    with pytest.warns(Warning, match="declares no dependence"):
        builtin = kucherenko.sample(base, base_n=2**12, seed=5, verbose=False)
    r_custom = kucherenko.analyze(custom, jnp.asarray(model(custom.samples)), verbose=False)
    r_builtin = kucherenko.analyze(builtin, jnp.asarray(model(builtin.samples)), verbose=False)
    assert isinstance(r_custom, kucherenko.KucherenkoResult)
    assert isinstance(r_builtin, kucherenko.KucherenkoResult)
    assert r_custom.dependence == "custom"
    assert r_builtin.dependence == "independent"
    np.testing.assert_allclose(r_custom.S1, r_builtin.S1, atol=0.02)
    np.testing.assert_allclose(r_custom.ST, r_builtin.ST, atol=0.02)


def test_custom_time_resolved_and_irregular_outputs_keep_dependence_metadata() -> None:
    from jaxgsa._core.irregular import IrregularResult

    problem = Problem.from_dict(
        {"x": (-1, 1), "square": (0, 1)}, output_names=("a", "b")
    ).with_sampler(sampler, sampler_dimension=1)
    design = kucherenko.sample(problem, base_n=32, seed=5, verbose=False)
    y = jnp.asarray(design.samples[:, 1])
    values = jnp.stack((jnp.stack((y, 2 * y), axis=1), jnp.stack((3 * y, 4 * y), axis=1)), axis=2)
    regular = kucherenko.analyze(design, values, verbose=False)
    assert isinstance(regular, kucherenko.KucherenkoResult)
    assert regular.S1.shape == (2, 2, 2)
    dataset = regular.to_dataset()
    assert dataset.attrs["dependence"] == "custom"
    assert "is_correlated" not in dataset.attrs
    irregular = kucherenko.analyze(
        design,
        [(jnp.array([0.0, 1.0]), values[:, :, 0]), (jnp.array([3.0]), values[:, :1, 1])],
        verbose=False,
    )
    assert isinstance(irregular, IrregularResult)
    for channel in irregular.channels.values():
        assert isinstance(channel, kucherenko.KucherenkoResult)
        assert channel.dependence == "custom"
    np.testing.assert_allclose(irregular.channels["a"].S1, regular.S1[:, :1], atol=1e-6)
