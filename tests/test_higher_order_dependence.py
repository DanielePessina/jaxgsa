"""Conditional estimators retain dependence that every input pair misses."""

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning, kucherenko, sampling
from jaxgsa._core.dependence import draw_conditional_samples
from jaxgsa.problem import Problem
from jaxgsa.vkoga._custom_indices import estimate_custom_indices

PROBLEM = Problem(names=("u", "v", "w"), bounds=((0.0, 1.0),) * 3)


def joint(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw pairwise-independent uniforms with a deterministic triple relation."""
    uv = rng.random((n, 2))
    return np.column_stack((uv, uv.sum(axis=1) % 1.0))


def conditional(
    anchors: np.ndarray, fixed: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """Draw from the modular-sum joint conditional on any fixed subset."""
    if not fixed:
        return joint(len(anchors), rng)
    rows = anchors.copy()
    held = set(fixed)
    if len(held) == 3:
        return rows
    if len(held) == 1:
        free = 1 if 0 in held else 0
        rows[:, free] = rng.random(len(rows))
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
    anchors = joint(1024, rng)
    for size in range(4):
        for fixed in combinations(range(3), size):
            rows = draw_conditional_samples(conditional, anchors, fixed, rng)
            np.testing.assert_allclose(rows[:, 2], (rows[:, 0] + rows[:, 1]) % 1.0, atol=1e-15)
            np.testing.assert_array_equal(rows[:, fixed], anchors[:, fixed])


def test_kucherenko_retains_higher_order_dependence() -> None:
    """For Y=W, first order is (0,0,1) and every conditional total is zero."""
    with jax.enable_x64():
        design = kucherenko.sample(
            PROBLEM,
            32768,
            joint_sampler=joint,
            conditional_sampler=conditional,
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
            joint_sampler=joint,
            conditional_sampler=conditional,
            predict=lambda unit: unit[:, 2:3],
            n_outer=2048,
            n_inner=8,
            n_variance=16384,
            entropy=7,
        )
    np.testing.assert_allclose(result.S_TC, [[0, 0, 1]], atol=0.07)
    np.testing.assert_allclose(result.S_TU, [[0, 0, 0]], atol=1e-25)
