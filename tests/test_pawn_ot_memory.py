"""Compile-time memory of pawn and optimal_transport under jit and grad.

Every check here lowers and compiles a program and reads XLA's
``temp_size_in_bytes``. Nothing is executed, so the checks are cheap. They
guard three regressions found in the 1.3.1 audit:

- ``indices()`` under an outer ``jax.jit`` traced the Python chunk loop into
  one program, and XLA kept every chunk's buffers live (pawn: 10.8 GiB at
  4096 columns against a 512 MiB budget).
- Reverse mode through the OT point-cloud kernel saved the ``(N, P)`` cost
  blocks of every class, so gradient memory grew as ``N**2`` (3.6 GiB at
  N=4096 against 8 MiB for the forward pass).
- Reverse mode through OT univariate grew by about 1 MiB per output column.

The dummy-floor tests at the end check that folding the synthetic
parameters into the real run gives the same numbers as running them apart.
"""

from __future__ import annotations

import warnings
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa.optimal_transport import OTResult

MIB = 2**20


def _temp_mib(fn, *args) -> float:
    """Compiled XLA temp memory of ``jax.jit(fn)(*args)``, in MiB."""
    stats = jax.jit(fn).lower(*args).compile().memory_analysis()
    assert stats is not None
    return stats.temp_size_in_bytes / MIB


def _problem(d: int) -> jaxgsa.Problem:
    return jaxgsa.Problem.from_dict({f"x{i}": (0.0, 1.0) for i in range(d)})


@pytest.fixture(scope="module")
def xs():
    n, d = 1024, 4
    X = jnp.asarray(np.random.default_rng(0).random((n, d)), jnp.float32)
    return _problem(d), X


def test_pawn_jit_indices_temp_does_not_grow_with_columns(xs) -> None:
    """Under an outer jit the chunks run one after another, not all at once."""
    problem, X = xs
    n = X.shape[0]

    def f(y):
        return jaxgsa.pawn.indices(problem, X, y, slice_chunk_size=8).pawn

    small = _temp_mib(f, jax.ShapeDtypeStruct((n, 32), jnp.float32))
    large = _temp_mib(f, jax.ShapeDtypeStruct((n, 512), jnp.float32))
    # Before the fix the temp grew with the number of chunks (16x here).
    assert large <= 1.5 * small + 1.0, (small, large)


def test_ot_univariate_jit_indices_temp_does_not_grow_with_columns(xs) -> None:
    """Same contract for OT univariate."""
    problem, X = xs
    n = X.shape[0]

    def f(y):
        return jaxgsa.optimal_transport.indices(problem, X, y, slice_chunk_size=8).ot

    small = _temp_mib(f, jax.ShapeDtypeStruct((n, 32), jnp.float32))
    large = _temp_mib(f, jax.ShapeDtypeStruct((n, 512), jnp.float32))
    assert large <= 1.5 * small + 1.0, (small, large)


def test_ot_univariate_grad_temp_does_not_grow_with_columns(xs) -> None:
    """Reverse mode keeps one chunk's intermediates, not every column's."""
    problem, X = xs
    n = X.shape[0]

    def loss(y):
        return jaxgsa.optimal_transport.indices(problem, X, y, slice_chunk_size=8).ot.sum()

    small = _temp_mib(jax.grad(loss), jax.ShapeDtypeStruct((n, 32), jnp.float32))
    large = _temp_mib(jax.grad(loss), jax.ShapeDtypeStruct((n, 512), jnp.float32))
    # Before the fix: about 0.3 MiB per column at this N, so ~150 MiB at 512.
    # What is left grows only with the saved chunk inputs (N * C floats).
    assert large <= 2.0 * small + 4.0, (small, large)


@pytest.mark.parametrize("mode", ["multivariate", "trajectory"])
def test_ot_point_cloud_grad_temp_stays_near_the_forward_pass(mode) -> None:
    """Reverse mode keeps one class's cost blocks, not every class's (was N**2)."""
    n, d = 1024, 2
    problem = _problem(d)
    X = jnp.asarray(np.random.default_rng(1).random((n, d)), jnp.float32)
    shape = (n, 2) if mode == "multivariate" else (n, 2, 1)
    Y = jax.ShapeDtypeStruct(shape, jnp.float32)

    def fwd(y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return jaxgsa.optimal_transport.indices(problem, X, y, mode=mode).ot.sum()

    fwd_temp = _temp_mib(fwd, Y)
    grad_temp = _temp_mib(jax.grad(fwd), Y)
    # Before the fix: 0.6 MiB forward and 227 MiB for the gradient at N=1024.
    assert grad_temp <= 4.0 * fwd_temp + 4.0, (fwd_temp, grad_temp)


def test_ot_point_cloud_grad_matches_finite_differences() -> None:
    """The checkpointed class cost keeps the implicit derivative exact."""
    with jax.enable_x64():
        n, d = 60, 2
        problem = _problem(d)
        rng = np.random.default_rng(3)
        X = jnp.asarray(rng.random((n, d)))
        Y = jnp.stack([jnp.sin(4 * X[:, 0]) + 0.1 * X[:, 1], X[:, 0] * X[:, 1]], axis=1)
        direction = jnp.asarray(rng.normal(size=Y.shape))

        def loss(y):
            return jaxgsa.optimal_transport.indices(
                problem, X, y, mode="multivariate", n_classes=3, epsilon=0.1, tol=1e-12
            ).ot.sum()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = jax.grad(loss)(Y)
            h = 1e-5
            fd = (loss(Y + h * direction) - loss(Y - h * direction)) / (2 * h)
        along = float((g * direction).sum())
    np.testing.assert_allclose(along, float(fd), rtol=1e-5)


def _cat_problem() -> jaxgsa.Problem:
    return jaxgsa.Problem.from_dict(
        {
            "a": (0.0, 1.0),
            "b": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": [0.5, 0.3, 0.2]},
        }
    )


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("mode", ["univariate", "multivariate"])
def test_dummy_floor_is_the_same_folded_or_apart(
    categorical: bool, mode: Literal["univariate", "multivariate"]
) -> None:
    """The dummy groups fold into the real run without a bootstrap.

    With a bootstrap they run apart, on the original sample. The dummy key
    does not depend on ``n_bootstrap``, so both paths must give the same
    floor and the same point estimate.
    """
    n = 400
    rng = np.random.default_rng(4)
    X = rng.random((n, 3))
    if categorical:
        problem = _cat_problem()
        X[:, 2] = rng.integers(0, 3, n)
    else:
        problem = _problem(3)
    X = jnp.asarray(X, jnp.float32)
    Y = jnp.stack([jnp.sin(6 * X[:, 0]) + X[:, 2], X[:, 1] * X[:, 0]], axis=1)
    key = jax.random.key(7)
    max_iter = 2000 if mode == "univariate" else 300

    def run(n_bootstrap: int) -> OTResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = jaxgsa.optimal_transport.analyze(
                problem,
                X,
                Y,
                mode=mode,
                dummy=True,
                key=key,
                max_iter=max_iter,
                n_bootstrap=n_bootstrap,
                verbose=False,
            )
        assert isinstance(out, OTResult)
        return out

    folded, apart = run(0), run(2)
    assert folded.ot_dummy is not None and apart.ot_dummy is not None
    assert folded.ot_dummy.shape == folded.ot.shape
    np.testing.assert_allclose(
        np.asarray(folded.ot_dummy), np.asarray(apart.ot_dummy), rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(np.asarray(folded.ot), np.asarray(apart.ot), rtol=1e-5, atol=1e-7)
    if not categorical:
        # Every continuous parameter shares the one continuous floor.
        floor = np.asarray(folded.ot_dummy)
        np.testing.assert_allclose(floor, floor[..., :1] * np.ones_like(floor), rtol=1e-6)
