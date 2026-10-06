"""Conditional estimators retain dependence that every input pair misses."""

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning, kucherenko, sampling
from jaxgsa._core.dependence import transform_samples
from jaxgsa.problem import Problem
from jaxgsa.vkoga._custom_indices import estimate_custom_indices

PROBLEM = Problem(names=("u", "v", "w"), bounds=((0.0, 1.0),) * 3)


def sampler(
    u: np.ndarray,
    *,
    conditioning_rows: np.ndarray | None = None,
    fixed_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Transform uniforms under the modular-sum joint and its conditionals."""
    if not fixed_indices:
        return np.column_stack((u[:, :2], u[:, :2].sum(axis=1) % 1.0))
    assert conditioning_rows is not None
    rows = conditioning_rows.copy()
    held = set(fixed_indices)
    if len(held) == 3:
        return rows
    if len(held) == 1:
        free = 1 if 0 in held else 0
        rows[:, free] = u[:, 0]
        held.add(free)
    if 2 not in held:
        rows[:, 2] = (rows[:, 0] + rows[:, 1]) % 1.0
    elif 1 not in held:
        rows[:, 1] = (rows[:, 2] - rows[:, 0]) % 1.0
    else:
        rows[:, 0] = (rows[:, 2] - rows[:, 1]) % 1.0
    return rows


def test_conditionals_preserve_the_joint_constraint_for_every_subset() -> None:
    """Pairwise independence must not turn conditional redraws independent."""
    rng = np.random.default_rng(4)
    conditioning_rows = sampler(rng.random((1024, 2)))
    for size in range(4):
        for fixed in combinations(range(3), size):
            rows = transform_samples(
                sampler,
                rng.random((1024, 2)),
                3,
                conditioning_rows=conditioning_rows,
                fixed_indices=fixed,
            )
            np.testing.assert_allclose(rows[:, 2], (rows[:, 0] + rows[:, 1]) % 1.0, atol=1e-15)
            np.testing.assert_array_equal(rows[:, fixed], conditioning_rows[:, fixed])


def test_kucherenko_retains_higher_order_dependence() -> None:
    """For Y=W, first order is (0,0,1) and every conditional total is zero."""
    with jax.enable_x64():
        design = kucherenko.sample(
            PROBLEM,
            32768,
            sampler=sampler,
            sampler_dimension=2,
            seed=7,
            verbose=False,
        )
        result = kucherenko.analyze(design, jnp.asarray(design.samples[:, 2]), verbose=False)
    assert isinstance(result, kucherenko.KucherenkoResult)
    np.testing.assert_allclose(result.S1, [0, 0, 1], atol=0.025)
    np.testing.assert_allclose(result.ST, [0, 0, 0], atol=1e-25)
    base = design.samples[: design.base_n]
    latent = sampling.fit_correlation(PROBLEM, base)
    np.testing.assert_allclose(latent, np.eye(3), atol=0.025)


def test_identity_copula_loses_the_triple_constraint() -> None:
    """An independent joint model gives W a total index of one instead."""
    with jax.enable_x64():
        with pytest.warns(JaxgsaWarning, match="plain independent draws"):
            design = kucherenko.sample(PROBLEM, 8192, seed=7, verbose=False)
        result = kucherenko.analyze(design, jnp.asarray(design.samples[:, 2]), verbose=False)
    assert isinstance(result, kucherenko.KucherenkoResult)
    np.testing.assert_allclose(result.ST, [0, 0, 1], atol=0.015)


def test_vkoga_conditional_integration_retains_higher_order_dependence() -> None:
    """Use an exact predictor to isolate integration from surrogate error."""
    with jax.enable_x64():
        result = estimate_custom_indices(
            problem=PROBLEM,
            sampler=sampler,
            sampler_dimension=2,
            predict=lambda unit: unit[:, 2:3],
            n_outer=2048,
            n_inner=8,
            n_variance=16384,
            entropy=7,
        )
    np.testing.assert_allclose(result.S_TC, [[0, 0, 1]], atol=0.07)
    np.testing.assert_allclose(result.S_TU, [[0, 0, 0]], atol=1e-25)
