"""A custom ``Problem.sampler`` drives ordinary input sampling."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import sampling
from jaxgsa._core.dependence import sobol_uniforms, transform_samples
from jaxgsa.problem import Problem

PROBLEM = Problem.from_dict({"a": (0, 1), "b": (0, 1)})


def transform(
    u: np.ndarray,
    *,
    conditioning_rows: np.ndarray | None = None,
    fixed_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Use one uniform to generate two deterministically related inputs."""
    result = np.column_stack((u[:, 0], u[:, 0] ** 2))
    if conditioning_rows is not None:
        result[:, fixed_indices] = conditioning_rows[:, fixed_indices]
    return result


CUSTOM = PROBLEM.with_sampler(transform, dimension=1)


def test_monte_carlo_applies_the_sampler_to_uniform_points() -> None:
    samples = sampling.monte_carlo(CUSTOM, 64, seed=3)
    assert samples.shape == (64, 2)
    np.testing.assert_allclose(samples[:, 1], samples[:, 0] ** 2)
    np.testing.assert_array_equal(samples, sampling.monte_carlo(CUSTOM, 64, seed=3))


def test_monte_carlo_without_sampler_is_unchanged() -> None:
    samples = sampling.monte_carlo(PROBLEM, 19, seed=7)
    assert samples.shape == (19, 2)
    assert not np.allclose(samples[:, 1], samples[:, 0] ** 2)


def test_sobol_uniforms_are_reproducible_and_validated() -> None:
    first = sobol_uniforms(8, 3, seed=4)
    np.testing.assert_array_equal(first, sobol_uniforms(8, 3, seed=4))
    assert first.shape == (8, 3)
    assert np.all((first >= 0) & (first <= 1))
    unscrambled = sobol_uniforms(8, 3, seed=4, scramble=False)
    assert np.all(unscrambled > 0)


def test_sampler_returning_wrong_shape_is_rejected() -> None:
    bad = PROBLEM.with_sampler(lambda u, **_: u[:, :1], dimension=2)
    with pytest.raises(ValueError, match="shape"):
        sampling.monte_carlo(bad, 8, seed=0)


def test_callback_mutation_does_not_change_callers_inputs() -> None:
    points = np.full((4, 2), 0.25)
    rows = np.full((4, 2), 0.5)

    def mutate(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        del fixed_indices
        u[:] = 0
        assert conditioning_rows is not None
        result = conditioning_rows.copy()
        conditioning_rows[:] = 0
        return result

    transform_samples(mutate, points, 2, conditioning_rows=rows, fixed_indices=(0, 1))
    np.testing.assert_array_equal(points, np.full((4, 2), 0.25))
    np.testing.assert_array_equal(rows, np.full((4, 2), 0.5))


def test_custom_cannot_accompany_correlation() -> None:
    with pytest.raises(ValueError, match="correlation"):
        CUSTOM.with_correlation(np.eye(2))


def test_structured_samplers_and_refusing_analyzers_reject_a_custom_sampler() -> None:
    with pytest.raises(ValueError, match="problem.sampler"):
        jaxgsa.sobol.sample(CUSTOM, 8, verbose=False)
    with pytest.raises(ValueError, match="problem.sampler"):
        jaxgsa.morris.sample(CUSTOM, 4, verbose=False)
    with pytest.raises(ValueError, match="problem.sampler"):
        jaxgsa.efast.sample(CUSTOM, 65, verbose=False)
    X = sampling.monte_carlo(CUSTOM, 64, seed=0)
    Y = jnp.asarray(X[:, 0] + X[:, 1])
    with pytest.raises(ValueError, match="problem.sampler"):
        jaxgsa.dgsm.indices(CUSTOM, X=jnp.asarray(X), Y=Y, dfdx=jnp.ones((64, 2)))
