"""VKOGA under a custom sampler declared on the problem.

There is one index estimator. A custom sampler only changes where its joint
and conditional draws come from (:class:`jaxgsa._core.dependence.CustomDraws`
instead of :class:`~jaxgsa._core.dependence.GaussianDraws`). The checks here
therefore compare the custom path against the built-in path on the same
distribution, and against closed forms, with an exact predictor wherever the
surrogate would only add noise.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from jaxgsa import vkoga
from jaxgsa._core.dependence import CustomDraws, GaussianDraws
from jaxgsa._core.irregular import IrregularResult
from jaxgsa.problem import Problem
from jaxgsa.vkoga._indices import estimate_correlated_indices

PROBLEM = Problem(names=("a", "b"), bounds=((0.0, 1.0),) * 2)

# Ishigami on [-pi, pi]^3, evaluated in marginal CDF coordinates.
ISHIGAMI = Problem(names=("x1", "x2", "x3"), bounds=((-np.pi, np.pi),) * 3)
_LOW, _HIGH = -np.pi, np.pi


def _ishigami_unit(U: np.ndarray) -> np.ndarray:
    x = _LOW + (_HIGH - _LOW) * U
    return (np.sin(x[:, 0]) + 7 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0]))[
        :, None
    ]


def independent_sampler(
    u: np.ndarray,
    *,
    conditioning_rows: np.ndarray | None = None,
    fixed_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Independent uniforms on ``[0, 1]``, keeping any fixed columns."""
    rows = u.copy()
    if fixed_indices:
        assert conditioning_rows is not None
        rows[:, fixed_indices] = conditioning_rows[:, fixed_indices]
    return rows


def ishigami_independent_sampler(
    u: np.ndarray,
    *,
    conditioning_rows: np.ndarray | None = None,
    fixed_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Independent uniforms on ``[-pi, pi]``, keeping any fixed columns."""
    rows = _LOW + (_HIGH - _LOW) * u
    if fixed_indices:
        assert conditioning_rows is not None
        rows[:, fixed_indices] = conditioning_rows[:, fixed_indices]
    return rows


def gaussian_copula_sampler(R: np.ndarray, low: float, high: float):
    """Build a custom sampler that implements a Gaussian copula by hand.

    The marginals are uniform on ``[low, high]``. Every conditional is the
    closed-form Gaussian one in the latent space, written independently of
    :mod:`jaxgsa._core.copula`.
    """
    D = R.shape[0]

    def sampler(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        noise = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
        if not fixed_indices:
            return low + (high - low) * ndtr(noise @ np.linalg.cholesky(R).T)
        assert conditioning_rows is not None
        fixed = list(fixed_indices)
        free = [j for j in range(D) if j not in fixed]
        unit = np.clip((conditioning_rows[:, fixed] - low) / (high - low), 1e-12, 1 - 1e-12)
        z_fixed = ndtri(unit)
        beta = R[np.ix_(free, fixed)] @ np.linalg.inv(R[np.ix_(fixed, fixed)])
        cov = R[np.ix_(free, free)] - beta @ R[np.ix_(fixed, free)]
        z_free = z_fixed @ beta.T + noise[:, free] @ np.linalg.cholesky(cov).T
        rows = conditioning_rows.copy()
        rows[:, free] = low + (high - low) * ndtr(z_free)
        return rows

    return sampler


_R3 = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, -0.4], [0.0, -0.4, 1.0]])


def _estimate(problem: Problem, draws, predict, entropy: int):
    """Run the shared estimator at the default analysis budget."""
    return estimate_correlated_indices(
        problem=problem,
        draws=draws,
        predict=predict,
        n_outer=512,
        n_inner=128,
        n_variance=8192,
        entropy=entropy,
        batch_size=None,
    )


def _index_spread(problem: Problem, draws, n_seeds: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Mean and spread of (S_TC, S_TU, S_U) over seeds with the exact predictor."""
    runs = []
    for seed in range(n_seeds):
        result = _estimate(problem, draws, _ishigami_unit, 100 + seed)
        runs.append(np.stack([result.S_TC[0], result.S_TU[0], result.S_U[0]]))
    stacked = np.array(runs)
    return stacked.mean(axis=0), stacked.std(axis=0)


def test_custom_independent_sampler_matches_the_built_in_path() -> None:
    """A custom sampler of independent inputs gives the classical Ishigami indices.

    Same estimator, same budget: the custom path must agree with the built-in
    independent path within Monte-Carlo noise, and both with the closed form
    (``S_TC = S1``, ``S_TU = ST``, ``S_U = S1``).
    """
    custom = CustomDraws(ISHIGAMI.with_sampler(ishigami_independent_sampler))
    built_in = GaussianDraws(ISHIGAMI)
    custom_mean, custom_sd = _index_spread(ISHIGAMI, custom)
    built_in_mean, built_in_sd = _index_spread(ISHIGAMI, built_in)

    tolerance = 4 * np.hypot(custom_sd, built_in_sd) + 2e-3
    assert np.all(np.abs(custom_mean - built_in_mean) <= tolerance)
    S1 = np.array([0.3139, 0.4424, 0.0])
    ST = np.array([0.5576, 0.4424, 0.2437])
    # The ddof=1 inner variance over a stratified QMC block reads S_TU about
    # n_inner / (n_inner - 1) high, under 1 percent at n_inner=128.
    np.testing.assert_allclose(custom_mean[0], S1, atol=0.01)
    np.testing.assert_allclose(custom_mean[1], ST, atol=0.01)
    np.testing.assert_allclose(custom_mean[2], S1, atol=0.01)


def test_custom_gaussian_copula_matches_the_built_in_copula() -> None:
    """A hand-written Gaussian-copula sampler agrees with the built-in copula.

    The two paths share the estimator but not the conditional algebra, so this
    is a T1 cross-check of :class:`CustomDraws` against
    :class:`GaussianDraws` on a correlated, non-additive model.
    """
    custom = CustomDraws(ISHIGAMI.with_sampler(gaussian_copula_sampler(_R3, _LOW, _HIGH)))
    built_in = GaussianDraws(ISHIGAMI.with_correlation(_R3))
    custom_mean, custom_sd = _index_spread(ISHIGAMI, custom)
    built_in_mean, built_in_sd = _index_spread(ISHIGAMI, built_in)
    tolerance = 4 * np.hypot(custom_sd, built_in_sd) + 2e-3
    assert np.all(np.abs(custom_mean - built_in_mean) <= tolerance), (
        custom_mean,
        built_in_mean,
    )
    # The correlation must actually matter, or the check above is vacuous.
    independent_mean, _ = _index_spread(ISHIGAMI, GaussianDraws(ISHIGAMI))
    assert np.max(np.abs(custom_mean - independent_mean)) > 0.1


def test_correlated_gaussian_conditional_indices() -> None:
    """Both conditional directions against a linear-Gaussian closed form."""
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
        z = ndtri(u)
        if not fixed_indices:
            z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
            return z
        assert conditioning_rows is not None
        rows = conditioning_rows.copy()
        free = 1 - fixed_indices[0]
        rows[:, free] = rho * rows[:, fixed_indices[0]] + np.sqrt(1 - rho**2) * z[:, free]
        return rows

    result = _estimate(
        problem,
        CustomDraws(problem.with_sampler(draw)),
        lambda u: (ndtri(np.clip(u, 1e-12, 1 - 1e-12)) @ np.array([1.0, 2.0]))[:, None],
        19,
    )
    variance = 5 + 4 * rho
    np.testing.assert_allclose(
        result.S_TC[0], np.array([1 + 2 * rho, 2 + rho]) ** 2 / variance, atol=0.02
    )
    expected_total = np.array([1.0, 4.0]) * (1 - rho**2) / variance
    np.testing.assert_allclose(result.S_TU[0], expected_total, atol=0.01)
    np.testing.assert_allclose(result.S_U[0], expected_total, atol=0.01)


def test_one_sampler_call_per_chunk() -> None:
    """The custom path is batched: one sampler call per chunk, never per outer point.

    With the auto budget every stage fits in one chunk, so the sampler sees
    one joint call for the variance sample and, per parameter and stage, one
    joint call for the outer points and one conditional call for all
    ``n_outer * n_inner`` inner rows.
    """
    calls: list[tuple[int, tuple[int, ...]]] = []

    def counting(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        calls.append((u.shape[0], fixed_indices))
        return independent_sampler(
            u, conditioning_rows=conditioning_rows, fixed_indices=fixed_indices
        )

    n_outer, n_inner = 64, 16
    estimate_correlated_indices(
        problem=PROBLEM,
        draws=CustomDraws(PROBLEM.with_sampler(counting)),
        predict=lambda u: u[:, :1] + u[:, 1:2] ** 2,
        n_outer=n_outer,
        n_inner=n_inner,
        n_variance=256,
        entropy=0,
        batch_size=None,
    )
    assert len(calls) == 1 + 2 * 2 * 2
    assert sorted(size for size, fixed in calls if fixed) == [n_outer * n_inner] * 4


def test_analyze_rounds_custom_sizes_and_labels_the_result() -> None:
    """Custom sample sizes round up like the built-in ones, and the result says custom."""
    sizes: set[int] = set()

    def recording(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        sizes.add(u.shape[0])
        return independent_sampler(
            u, conditioning_rows=conditioning_rows, fixed_indices=fixed_indices
        )

    with jax.enable_x64():
        x = jnp.asarray(np.random.default_rng(1).random((70, 2)))
        result = vkoga.analyze(
            PROBLEM.with_sampler(recording),
            x,
            x[:, 0],
            gamma=2.0,
            ridge=1e-6,
            max_centers=15,
            n_outer=9,
            n_inner=3,
            n_variance=31,
            key=jax.random.key(2),
            verbose=False,
        )
    assert isinstance(result, vkoga.VKOGAResult)
    # n_outer 9 -> 16, n_inner 3 -> 4 (16 * 4 = 64 inner rows), n_variance 31 -> 32.
    assert sizes == {16, 32, 64}
    assert result.dependence == "custom"
    assert result.correlation is None


def test_batch_size_does_not_move_a_custom_estimate() -> None:
    """Chunking is a memory decision for the custom path too."""
    with jax.enable_x64():
        x = jnp.asarray(np.random.default_rng(1).random((70, 2)))

        def run(batch_size: int) -> vkoga.VKOGAResult:
            value = vkoga.analyze(
                PROBLEM.with_sampler(independent_sampler),
                x,
                x[:, 0],
                batch_size=batch_size,
                gamma=2.0,
                ridge=1e-6,
                max_centers=15,
                n_outer=16,
                n_inner=4,
                n_variance=32,
                n_bootstrap=2,
                key=jax.random.key(2),
                verbose=False,
            )
            assert isinstance(value, vkoga.VKOGAResult)
            return value

        result, other = run(4), run(1024)
    assert result.S_TC_conf is not None
    assert other.S_TC_conf is not None
    np.testing.assert_allclose(result.S_TC, other.S_TC, atol=1e-12)
    np.testing.assert_allclose(result.S_TU, other.S_TU, atol=1e-12)
    np.testing.assert_allclose(result.S_TC_conf, other.S_TC_conf, atol=1e-12)


def test_sampler_outside_the_declared_support_is_rejected() -> None:
    """Rows outside a bounded marginal cannot hide behind CDF clipping."""

    def drifting(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        rows = independent_sampler(
            u, conditioning_rows=conditioning_rows, fixed_indices=fixed_indices
        )
        return rows + 0.5

    with jax.enable_x64(), pytest.raises(ValueError, match="outside the declared support"):
        x = jnp.asarray(np.random.default_rng(1).random((40, 2)))
        vkoga.analyze(
            PROBLEM.with_sampler(drifting),
            x,
            x[:, 0],
            gamma=2.0,
            ridge=1e-6,
            max_centers=10,
            n_outer=8,
            n_inner=4,
            n_variance=32,
            key=jax.random.key(0),
            verbose=False,
        )


def test_correlation_override_is_refused_for_a_custom_problem() -> None:
    """A problem with a sampler has one dependence source; correlation= cannot replace it."""
    with pytest.raises(ValueError, match="custom sampler"):
        vkoga.analyze(
            PROBLEM.with_sampler(independent_sampler),
            jnp.ones((4, 2)),
            jnp.ones(4),
            correlation=np.eye(2),
            key=jax.random.key(0),
        )


def test_irregular_channels_inherit_the_custom_sampler() -> None:
    """Each irregular channel integrates the problem's custom distribution."""
    with jax.enable_x64():
        x = jnp.asarray(np.random.default_rng(4).random((45, 2)))
        y = {
            "first": (jnp.array([0.0, 1.0]), jnp.stack((x[:, 0], x[:, 0] + 1), axis=1)),
            "second": (jnp.array([0.0]), x[:, 1:2]),
        }
        result = vkoga.analyze(
            PROBLEM.with_sampler(independent_sampler),
            x,
            y,
            gamma=2.0,
            ridge=1e-6,
            max_centers=10,
            n_outer=8,
            n_inner=4,
            n_variance=32,
            key=jax.random.key(6),
            verbose=False,
        )
    assert isinstance(result, IrregularResult)
    assert set(result.channels) == {"first", "second"}
    for channel in result.channels.values():
        assert isinstance(channel, vkoga.VKOGAResult)
        assert channel.dependence == "custom"
        assert channel.correlation is None


@pytest.mark.verbose_output
def test_time_multi_output_export_and_receipt(capsys: pytest.CaptureFixture[str]) -> None:
    """Custom dependence stays explicit in the receipt and exported metadata."""
    with jax.enable_x64():
        x = jnp.asarray(np.random.default_rng(4).random((45, 2)))
        outputs = jnp.stack((x[:, 0], x[:, 1]), axis=1)
        y = jnp.stack((outputs, 2 * outputs), axis=1)
        result = vkoga.analyze(
            PROBLEM.with_sampler(independent_sampler),
            x,
            y,
            gamma=2.0,
            ridge=1e-6,
            max_centers=10,
            n_outer=8,
            n_inner=4,
            n_variance=32,
            key=jax.random.key(6),
        )
    assert isinstance(result, vkoga.VKOGAResult)
    assert result.S_TC.shape == (2, 2, 2)
    dataset = result.to_dataset()
    assert dataset.attrs["dependence"] == "custom"
    assert "correlation" not in dataset
    assert "custom" in capsys.readouterr().out
