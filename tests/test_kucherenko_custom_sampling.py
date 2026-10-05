"""Custom conditional sampling preserves nonlinear dependence and design rows."""

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import kucherenko
from jaxgsa.problem import Problem

PROBLEM = Problem.from_dict({"x": (-1, 1), "square": (0, 1)})


def joint(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a deterministically dependent pair with zero Pearson correlation."""
    x = rng.uniform(-1, 1, n)
    return np.column_stack((x, x**2))


def conditional(
    anchors: np.ndarray, fixed: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """Condition on either x or its square, retaining the physical columns."""
    result = anchors.copy()
    if fixed == (1,):
        result[:, 0] = rng.choice([-1.0, 1.0], len(anchors)) * np.sqrt(anchors[:, 1])
    elif fixed != (0,):
        raise ValueError("Unexpected fixed columns")
    return result


@pytest.mark.verbose_output
def test_nonlinear_dependence_and_repeated_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        design = kucherenko.sample(
            PROBLEM,
            1025,
            seed=31,
            scramble=False,
            joint_sampler=joint,
            conditional_sampler=conditional,
        )
    assert not caught
    assert design.base_n == 1025
    assert design.n_runs == 5125
    assert design.dependence_source == "custom"
    assert "custom" in repr(design)
    np.testing.assert_array_equal(design.samples[:1025], design.samples[1025:2050])
    result = kucherenko.analyze(
        design, jnp.asarray(design.samples[:, 1]), n_bootstrap=8, key=jax.random.key(7)
    )
    assert isinstance(result, kucherenko.KucherenkoResult)
    np.testing.assert_allclose(result.S1, [1, 1], atol=0.002)
    np.testing.assert_array_equal(result.ST, [0, 0])
    assert result.dependence_source == "custom"
    assert result.is_correlated is None
    receipt = capsys.readouterr().out
    assert "custom-conditional" in receipt
    assert "correlation: independent" not in receipt
    target = tmp_path / "custom.npz"
    design.save(target)
    loaded = kucherenko.KucherenkoSamples.load(target)
    assert loaded.dependence_source == "custom"
    restored = kucherenko.analyze(loaded, jnp.asarray(loaded.samples[:, 1]), verbose=False)
    assert isinstance(restored, kucherenko.KucherenkoResult)
    np.testing.assert_array_equal(restored.S1, result.S1)
    repeat = kucherenko.sample(
        PROBLEM, 1025, seed=31, joint_sampler=joint, conditional_sampler=conditional, verbose=False
    )
    np.testing.assert_array_equal(repeat.samples, design.samples)


def test_callback_pair_and_conflicting_correlation() -> None:
    with pytest.raises(ValueError):
        kucherenko.sample(PROBLEM, 8, joint_sampler=joint)
    with pytest.raises(ValueError, match="correlation"):
        kucherenko.sample(
            PROBLEM.with_correlation(np.eye(2)),
            8,
            joint_sampler=joint,
            conditional_sampler=conditional,
        )


def test_conditional_output_must_preserve_fixed_columns() -> None:
    def bad(anchors: np.ndarray, fixed: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        del fixed, rng
        return anchors + 1

    with pytest.raises(ValueError):
        kucherenko.sample(PROBLEM, 8, joint_sampler=joint, conditional_sampler=bad)


def test_gaussian_custom_conditionals_match_analytic_indices() -> None:
    rho = 0.7

    def draw(n: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.normal(size=(n, 2))
        z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
        return z

    def condition(
        anchors: np.ndarray, fixed: tuple[int, ...], rng: np.random.Generator
    ) -> np.ndarray:
        result = anchors.copy()
        held = fixed[0]
        result[:, 1 - held] = rho * anchors[:, held] + np.sqrt(1 - rho**2) * rng.normal(
            size=len(anchors)
        )
        return result

    problem = Problem.from_dict({"x": (-10, 10), "y": (-10, 10)})
    design = kucherenko.sample(
        problem, 30000, seed=19, joint_sampler=draw, conditional_sampler=condition, verbose=False
    )
    result = kucherenko.analyze(design, jnp.asarray(design.samples.sum(axis=1)), verbose=False)
    assert isinstance(result, kucherenko.KucherenkoResult)
    np.testing.assert_allclose(result.S1, [(1 + rho) / 2] * 2, atol=0.015)
    np.testing.assert_allclose(result.ST, [(1 - rho) / 2] * 2, atol=0.01)


def test_old_design_metadata_defaults_to_gaussian() -> None:
    design = kucherenko.sample(
        PROBLEM, 8, joint_sampler=joint, conditional_sampler=conditional, verbose=False
    )
    loaded = kucherenko.KucherenkoSamples._from_payload(
        samples=design.samples,
        n_expanded=design.n_expanded,
        expanded_to_unique=design.expanded_to_unique,
        problem=PROBLEM,
        arrays={},
        meta={"base_n": 8},
    )
    assert loaded.dependence_source == "gaussian"


def test_custom_time_resolved_and_irregular_outputs_keep_dependence_metadata() -> None:
    from jaxgsa._core.irregular import IrregularResult

    problem = Problem.from_dict({"x": (-1, 1), "square": (0, 1)}, output_names=("a", "b"))
    design = kucherenko.sample(
        problem, 33, seed=5, joint_sampler=joint, conditional_sampler=conditional, verbose=False
    )
    y = jnp.asarray(design.samples[:, 1])
    values = jnp.stack((jnp.stack((y, 2 * y), axis=1), jnp.stack((3 * y, 4 * y), axis=1)), axis=2)
    regular = kucherenko.analyze(design, values, verbose=False)
    assert isinstance(regular, kucherenko.KucherenkoResult)
    assert regular.S1.shape == (2, 2, 2)
    dataset = regular.to_dataset()
    assert dataset.attrs["dependence_source"] == "custom"
    assert "is_correlated" not in dataset.attrs
    irregular = kucherenko.analyze(
        design,
        [(jnp.array([0.0, 1.0]), values[:, :, 0]), (jnp.array([3.0]), values[:, :1, 1])],
        verbose=False,
    )
    assert isinstance(irregular, IrregularResult)
    for channel in irregular.channels.values():
        assert isinstance(channel, kucherenko.KucherenkoResult)
        assert channel.dependence_source == "custom"
        assert channel.is_correlated is None
    np.testing.assert_allclose(irregular.channels["a"].S1, regular.S1[:, :1], atol=1e-6)
