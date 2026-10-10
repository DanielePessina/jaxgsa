"""Conditional estimators retain dependence that every input pair misses."""

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning, kucherenko, sampling
from jaxgsa._core.dependence import CustomDraws, transform_samples
from jaxgsa.problem import Problem
from jaxgsa.vkoga._indices import estimate_correlated_indices

BASE = Problem(names=("u", "v", "w"), bounds=((0.0, 1.0),) * 3)


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


# W = (U + V) mod 1: every pair is independent, but the triple is degenerate.
PROBLEM = BASE.with_sampler(sampler, dimension=2)


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
        design = kucherenko.sample(PROBLEM, 8192, seed=7, verbose=False)
        result = kucherenko.analyze(design, jnp.asarray(design.samples[:, 2]), verbose=False)
    assert isinstance(result, kucherenko.KucherenkoResult)
    # Worst S1 error over eight seeds at base 8192 is 0.016.
    np.testing.assert_allclose(result.S1, [0, 0, 1], atol=0.05)
    np.testing.assert_allclose(result.ST, [0, 0, 0], atol=1e-25)
    # The constraint is invisible to every pairwise statistic.
    base = design.samples[: design.base_n]
    latent = sampling.fit_correlation(BASE, base)
    np.testing.assert_allclose(latent, np.eye(3), atol=0.025)


def test_identity_copula_loses_the_triple_constraint() -> None:
    """An independent joint model gives W a total index of one instead."""
    with jax.enable_x64():
        with pytest.warns(JaxgsaWarning, match="plain independent draws"):
            design = kucherenko.sample(BASE, 1024, seed=7, verbose=False)
        result = kucherenko.analyze(design, jnp.asarray(design.samples[:, 2]), verbose=False)
    assert isinstance(result, kucherenko.KucherenkoResult)
    # The error is 6e-6 even at base 1024; the tolerance is far above it.
    np.testing.assert_allclose(result.ST, [0, 0, 1], atol=0.015)


def test_vkoga_conditional_integration_retains_higher_order_dependence() -> None:
    """Use an exact predictor to isolate integration from surrogate error.

    ``S_TC`` of W is one, of U and V zero. ``S_TU`` is exactly zero for every
    input, because fixing any two inputs fixes the third.
    """
    result = estimate_correlated_indices(
        problem=PROBLEM,
        draws=CustomDraws(PROBLEM),
        predict=lambda unit: unit[:, 2:3],
        n_outer=512,
        n_inner=64,
        n_variance=4096,
        entropy=7,
        batch_size=None,
    )
    # Worst error over eight entropies at this budget is 0.008.
    np.testing.assert_allclose(result.S_TC, [[0, 0, 1]], atol=0.04)
    np.testing.assert_allclose(result.S_TU, [[0, 0, 0]], atol=1e-25)
