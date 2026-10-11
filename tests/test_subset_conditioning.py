"""Conditioning on an arbitrary subset of inputs: Gaussian and custom draws."""

import itertools
from typing import Any

import numpy as np
import pytest

from jaxgsa._core import dependence
from jaxgsa._core.dependence import CustomDraws, GaussianDraws
from jaxgsa.problem import Problem

R4 = np.array(
    [
        [1.0, 0.5, -0.2, 0.2],
        [0.5, 1.0, 0.3, -0.1],
        [-0.2, 0.3, 1.0, 0.5],
        [0.2, -0.1, 0.5, 1.0],
    ]
)
assert np.linalg.eigvalsh(R4).min() > 0.05
PROBLEM = Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(4)}, correlation=R4)
# The problem stores the latent matrix, which is not the declared Spearman one.
RL = np.asarray(PROBLEM.correlation)
SUBSETS = [(0,), (2,), (0, 1), (1, 3), (0, 2, 3), (1, 2, 3)]
N = 200_000


def _schur(R: np.ndarray, fixed: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    free = [j for j in range(R.shape[0]) if j not in fixed]
    R_fk = R[np.ix_(free, fixed)]
    R_kk = R[np.ix_(fixed, fixed)]
    beta = R_fk @ np.linalg.inv(R_kk)
    return np.array(free), beta, R[np.ix_(free, free)] - beta @ R_fk.T


@pytest.mark.parametrize("fixed", SUBSETS)
def test_gaussian_subset_matches_schur_complement(fixed: tuple[int, ...]) -> None:
    """Residuals of the free columns have the analytic conditional moments."""
    draws = GaussianDraws(PROBLEM)
    rng = np.random.default_rng(1)
    rows = rng.standard_normal((N, 4))
    u = rng.random((N, draws.free_dimension(fixed)))
    out = draws.free_given_fixed(u, rows, fixed)
    free, beta, cov = _schur(RL, fixed)
    resid = out[:, free] - rows[:, list(fixed)] @ beta.T
    np.testing.assert_allclose(resid.mean(axis=0), 0.0, atol=1.5e-2)
    np.testing.assert_allclose(np.cov(resid.T), cov, atol=1.5e-2)
    # Free columns are independent of the fixed ones given the regression.
    cross = (resid - resid.mean(0)).T @ (rows[:, list(fixed)] - rows[:, list(fixed)].mean(0)) / N
    np.testing.assert_allclose(cross, 0.0, atol=1.5e-2)


@pytest.mark.parametrize("fixed", SUBSETS)
def test_gaussian_fixed_columns_are_bit_exact(fixed: tuple[int, ...]) -> None:
    """Fixed columns come back unchanged and the input is not modified."""
    draws = GaussianDraws(PROBLEM)
    rng = np.random.default_rng(2)
    rows = rng.standard_normal((50, 4))
    before = rows.copy()
    out = draws.free_given_fixed(rng.random((50, draws.free_dimension(fixed))), rows, fixed)
    assert np.array_equal(out[:, list(fixed)], rows[:, list(fixed)])
    assert np.array_equal(rows, before)


def test_empty_and_full_fixed_sets() -> None:
    """``()`` is a joint draw and all columns fixed returns a copy."""
    draws = GaussianDraws(PROBLEM)
    rng = np.random.default_rng(3)
    u = rng.random((20, 4))
    np.testing.assert_array_equal(draws.free_given_fixed(u, np.empty((20, 4)), ()), draws.joint(u))
    assert draws.free_dimension(()) == 4
    rows = rng.standard_normal((20, 4))
    out = draws.free_given_fixed(np.empty((20, 0)), rows, (0, 1, 2, 3))
    assert draws.free_dimension((0, 1, 2, 3)) == 0
    np.testing.assert_array_equal(out, rows)
    assert out is not rows


def test_matches_rest_given_self_and_self_given_rest() -> None:
    """The one-fixed and all-but-one subsets agree with the existing paths."""
    draws = GaussianDraws(PROBLEM)
    rng = np.random.default_rng(4)
    rows = rng.standard_normal((N, 4))
    for i in range(4):
        u_rest = rng.random((N, 3))
        a = draws.rest_given_self(u_rest, rows, i)
        b = draws.free_given_fixed(u_rest, rows, (i,))
        others = [j for j in range(4) if j != i]
        # Same law: compare residual covariances (the noise factor may differ).
        beta = RL[others, i]
        ra, rb = (
            a[:, others] - np.outer(rows[:, i], beta),
            b[:, others] - np.outer(rows[:, i], beta),
        )
        np.testing.assert_allclose(np.cov(ra.T), np.cov(rb.T), atol=2e-2)
        np.testing.assert_allclose(rb.mean(0), ra.mean(0), atol=2e-2)

        u_self = rng.random((N, 1))
        c = draws.self_given_rest(u_self, rows, i)
        d = draws.free_given_fixed(u_self, rows, tuple(others))
        np.testing.assert_allclose(d[:, i], c[:, i], atol=1e-10)


def test_plan_cache_builds_each_subset_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A design that revisits subsets builds each plan once."""
    calls: list[tuple[int, ...]] = []
    real = dependence.build_subset_plan

    def counting(R: np.ndarray, fixed: tuple[int, ...]) -> Any:
        calls.append(fixed)
        return real(R, fixed)

    monkeypatch.setattr(dependence, "build_subset_plan", counting)
    draws = GaussianDraws(PROBLEM)
    rng = np.random.default_rng(5)
    rows = rng.standard_normal((8, 4))
    for _ in range(3):
        for k in range(1, 4):
            for fixed in itertools.combinations(range(4), k):
                draws.free_given_fixed(rng.random((8, 4 - k)), rows, fixed)
    assert len(calls) == 14
    assert len(set(calls)) == 14


def test_uniform_points_give_coupled_draws() -> None:
    """The same uniforms under one subset are a deterministic map."""
    draws = GaussianDraws(PROBLEM)
    rng = np.random.default_rng(6)
    rows, u = rng.standard_normal((10, 4)), rng.random((10, 2))
    a = draws.free_given_fixed(u, rows, (1, 3))
    b = draws.free_given_fixed(u, rows, (1, 3))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("bad", [(1, 0), (1, 1), (4,), (-1,), (0.5,)])
def test_validation_errors(bad: Any) -> None:
    """Unsorted, repeated, out-of-range or non-integer indices raise."""
    draws = GaussianDraws(PROBLEM)
    with pytest.raises(ValueError):
        draws.free_dimension(bad)
    with pytest.raises(ValueError):
        draws.free_given_fixed(np.empty((3, 1)), np.zeros((3, 4)), bad)


def test_shape_errors() -> None:
    """Wrong noise width or row shape raises."""
    draws = GaussianDraws(PROBLEM)
    with pytest.raises(ValueError, match="uniform points"):
        draws.free_given_fixed(np.zeros((3, 2)), np.zeros((3, 4)), (0,))
    with pytest.raises(ValueError, match="rows of shape"):
        draws.free_given_fixed(np.zeros((3, 3)), np.zeros((3, 3)), (0,))


def test_degenerate_conditional_warns_once() -> None:
    """A near-deterministic conditional warns when its plan is built."""
    R = np.array([[1.0, 0.999], [0.999, 1.0]])
    problem = Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0)}, correlation=R)
    draws = GaussianDraws(problem)
    rows = np.zeros((4, 2))
    with pytest.warns(UserWarning, match="almost deterministic"):
        draws.free_given_fixed(np.full((4, 1), 0.5), rows, (0,))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        draws.free_given_fixed(np.full((4, 1), 0.5), rows, (0,))


def test_custom_forwards_exact_tuple() -> None:
    """The custom path hands ``fixed`` to the sampler unchanged."""
    seen: list[tuple[tuple[int, ...], bool]] = []

    def sampler(
        u: np.ndarray,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        seen.append((fixed_indices, conditioning_rows is None))
        out = np.tile(u[:, :1], (1, 4)) if conditioning_rows is None else conditioning_rows.copy()
        if conditioning_rows is not None:
            free = [j for j in range(4) if j not in fixed_indices]
            out[:, free] = u[:, :1]
        return out

    problem = Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(4)}).with_sampler(
        sampler, sampler_dimension=2
    )
    draws = CustomDraws(problem)
    u, rows = np.full((5, 2), 0.25), np.full((5, 4), 0.9)
    draws.free_given_fixed(u, rows, ())
    out = draws.free_given_fixed(u, rows, (0, 2))
    draws.free_given_fixed(u, rows, (0, 1, 2, 3))
    assert seen == [((), True), ((0, 2), False), ((0, 1, 2, 3), False)]
    np.testing.assert_array_equal(out[:, [0, 2]], 0.9)
    np.testing.assert_array_equal(out[:, [1, 3]], 0.25)
    assert draws.free_dimension((0, 2)) == 2
    with pytest.raises(ValueError):
        draws.free_given_fixed(u, rows, (2, 0))
    with pytest.raises(ValueError):
        draws.free_dimension((7,))
