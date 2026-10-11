"""A custom ``Problem.sampler`` drives ordinary input sampling."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import sampling
from jaxgsa._core.dependence import CustomDraws, sobol_uniforms
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


CUSTOM = PROBLEM.with_sampler(transform, sampler_dimension=1)


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
    bad = PROBLEM.with_sampler(lambda u, **_: u[:, :1], sampler_dimension=2)
    with pytest.raises(ValueError, match="shape"):
        sampling.monte_carlo(bad, 8, seed=0)


@pytest.mark.parametrize("order", ["correlation-then-sampler", "sampler-then-correlation"])
def test_custom_sampler_and_correlation_conflict(order: str) -> None:
    """A problem has one dependence source, whichever way round it is declared."""
    with pytest.raises(ValueError, match="correlation"):
        if order == "correlation-then-sampler":
            PROBLEM.with_correlation(np.eye(2)).with_sampler(transform, sampler_dimension=1)
        else:
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
        jaxgsa.dgsm.indices(CUSTOM, jnp.asarray(X), Y, dfdx=jnp.ones((64, 2)))


def _support_problem() -> Problem:
    return Problem.from_dict(
        {
            "u": (0, 1),
            "t": {"dist": "gaussian", "mean": 0.0, "variance": 1.0, "low": -1.0, "high": 1.0},
            "g": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
            "c": {"dist": "categorical", "probs": [0.2, 0.3, 0.5]},
        }
    )


def _rows(u: np.ndarray) -> np.ndarray:
    return np.column_stack((u[:, 0], 2 * u[:, 0] - 1, 50 * (u[:, 0] - 0.5), np.floor(3 * u[:, 0])))


def _draw(problem: Problem) -> None:
    CustomDraws(problem).joint(np.linspace(0.0, 0.99, 100).reshape(-1, 1))


def test_sampler_inside_every_support_is_accepted() -> None:
    problem = _support_problem().with_sampler(lambda u, **_: _rows(u), sampler_dimension=1)
    _draw(problem)


@pytest.mark.parametrize(
    ("column", "bad", "message"),
    [
        (
            0,
            1.5,
            r'"?\'u\'"? \(uniform on \[0, 1\]\): 1 rows, first at row 7 of this sampler call, '
            r"value 1.5",
        ),
        (1, -1.25, r"\'t\' \(truncated Gaussian on \[-1, 1\]\): 1 rows, first at row 7"),
        (3, 3.0, r"\'c\' \(categorical, integer codes 0 to 2\): 1 rows, first at row 7"),
        (
            3,
            1.5,
            r"\'c\' \(categorical, integer codes 0 to 2\): 1 rows, "
            r"first at row 7 of this sampler call, value 1.5",
        ),
        (3, -1.0, r"\'c\' \(categorical"),
    ],
)
def test_sampler_outside_support_is_rejected(column: int, bad: float, message: str) -> None:
    def sampler(u: np.ndarray, **_: object) -> np.ndarray:
        rows = _rows(u)
        rows[7, column] = bad
        return rows

    problem = _support_problem().with_sampler(sampler, sampler_dimension=1)
    with pytest.raises(ValueError, match="outside the support of.*" + message):
        _draw(problem)


def test_unbounded_gaussian_accepts_any_finite_value() -> None:
    problem = _support_problem().with_sampler(
        lambda u, **_: _rows(u) * np.array([1, 1, 1e6, 1]), sampler_dimension=1
    )
    _draw(problem)


def test_support_error_counts_rows() -> None:
    problem = PROBLEM.with_sampler(
        lambda u, **_: np.column_stack((u[:, 0] * 5, u[:, 0])), sampler_dimension=1
    )
    from jaxgsa._core.dependence import CustomDraws

    with pytest.raises(
        ValueError, match=r'"?\'a\'"? \(uniform on \[0, 1\]\): \d+ rows, first at row'
    ):
        CustomDraws(problem).joint(np.linspace(0.0, 0.99, 100).reshape(-1, 1))


def test_monte_carlo_checks_the_sampler_support() -> None:
    problem = PROBLEM.with_sampler(
        lambda u, **_: np.column_stack((u[:, 0] * 5, u[:, 0])), sampler_dimension=1
    )
    with pytest.raises(ValueError, match="outside the support of 'a'"):
        sampling.monte_carlo(problem, 64, seed=0)


def test_float32_overshoot_at_a_bound_is_clipped() -> None:
    problem = Problem.from_dict({"a": (0.0, 0.3), "b": (0.7, 1.0)})

    def single_precision(u: np.ndarray, **_: object) -> np.ndarray:
        rows = np.column_stack((0.3 * u[:, 0], 0.7 + 0.3 * u[:, 0]))
        rows[0] = (0.3, 0.7)
        return rows.astype(np.float32)

    samples = sampling.monte_carlo(problem.with_sampler(single_precision, sampler_dimension=1), 8)
    assert samples[0, 0] == 0.3
    assert samples[0, 1] == 0.7
    assert np.all((samples[:, 0] >= 0.0) & (samples[:, 0] <= 0.3))


def test_sampler_with_zero_rows_is_accepted() -> None:
    rows = CustomDraws(CUSTOM).joint(np.empty((0, 1)))
    assert rows.shape == (0, 2)
