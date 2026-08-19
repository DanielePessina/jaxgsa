"""One small result object per result class, for schema regression tests.

``tests/test_result_schema.py`` compares the ``to_dataset()`` schema of every
result against the stored snapshot ``tests/data/result_dataset_schema.json``.
The snapshot is only meaningful if the results that produce it are fixed, so
they are built here rather than inside the test. ``scripts/dump_result_schema.py``
writes the snapshot from these same builders, so a deliberate schema change is
regenerated rather than hand-edited.

The problems are deliberately tiny. Nothing here checks a number; it checks
which dimensions, coordinates and variables a result exports.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

import jaxgsa
from jaxgsa import (
    borgonovo,
    dgsm,
    efast,
    hdmr,
    hsic,
    kucherenko,
    morris,
    optimal_transport,
    pawn,
    pce,
    shapley,
    sobol,
    vkoga,
)

SEED = 7
"""Fixed seed, so the snapshot is reproducible."""

PROBLEM = jaxgsa.Problem.from_dict({"x1": (-1.0, 1.0), "x2": (-1.0, 1.0), "x3": (-1.0, 1.0)})
"""Three uniform inputs. Three is enough for a second-order term to exist."""


def _scalar(X: jax.Array) -> jax.Array:
    """Smooth non-additive model with a scalar output, shape ``(N,)``."""
    return X[:, 0] + 2.0 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]


def _multi(X: jax.Array) -> jax.Array:
    """The same model widened to two outputs, shape ``(N, 2)``."""
    y = _scalar(X)
    return jnp.stack([y, 2.0 * y + 1.0], axis=-1)


def _series(X: jax.Array) -> jax.Array:
    """The same model widened to a time series, shape ``(N, 3, 2)``."""
    y = _scalar(X)
    factors = jnp.asarray([1.0, 2.0, 3.0])
    offsets = jnp.asarray([0.0, 1.0])
    return y[:, None, None] * factors[None, :, None] + offsets[None, None, :]


MODELS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "scalar": _scalar,
    "multi": _multi,
    "series": _series,
}


def _given_data(shape: str, n: int = 256) -> tuple[jax.Array, jax.Array]:
    """Return a Monte Carlo ``(X, Y)`` pair for one output shape."""
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n, seed=SEED))
    return X, MODELS[shape](X)


def _point_model(shape: str) -> Callable[[jax.Array], jax.Array]:
    """Adapt a batch model to the one-sample convention ``dgsm.analyze`` wants."""
    model = MODELS[shape]

    def point(x: jax.Array) -> jax.Array:
        return model(x[None, :])[0]

    return point


def _sobol(shape: str) -> Any:
    sr = sobol.sample(PROBLEM, n_samples=128, seed=SEED, verbose=False)
    Y = MODELS[shape](jnp.asarray(sr.samples))
    return sobol.analyze(sr, Y, n_bootstrap=8, key=jax.random.key(SEED))


def _morris(shape: str) -> Any:
    sr = morris.sample(PROBLEM, n_trajectories=8, seed=SEED, verbose=False)
    Y = MODELS[shape](jnp.asarray(sr.samples))
    return morris.analyze(sr, Y, n_bootstrap=8, key=jax.random.key(SEED))


def _efast(shape: str) -> Any:
    sr = efast.sample(PROBLEM, n_per_curve=257, seed=SEED)
    Y = MODELS[shape](jnp.asarray(sr.samples))
    return efast.analyze(sr, Y)


def _kucherenko(shape: str) -> Any:
    sr = kucherenko.sample(PROBLEM, n_samples=128, seed=SEED)
    Y = MODELS[shape](jnp.asarray(sr.samples))
    return kucherenko.analyze(sr, Y)


def _dgsm(shape: str) -> Any:
    X, _ = _given_data(shape)
    return dgsm.analyze(PROBLEM, _point_model(shape), X)


def _borgonovo(shape: str) -> Any:
    X, Y = _given_data(shape)
    return borgonovo.analyze(PROBLEM, X, Y, n_bootstrap=8, key=jax.random.key(SEED))


def _pawn(shape: str) -> Any:
    X, Y = _given_data(shape)
    return pawn.analyze(PROBLEM, X, Y, n_bins=4, n_bootstrap=8, key=jax.random.key(SEED))


def _hsic(shape: str) -> Any:
    X, Y = _given_data(shape)
    return hsic.analyze(PROBLEM, X, Y, n_perms=20, key=jax.random.key(SEED))


def _ot(shape: str) -> Any:
    X, Y = _given_data(shape)
    return optimal_transport.analyze(
        PROBLEM, X, Y, n_partitions=4, n_bootstrap=4, dummy=True, key=jax.random.key(SEED)
    )


def _ot_multivariate(shape: str) -> Any:
    X, Y = _given_data(shape)
    return optimal_transport.analyze(
        PROBLEM, X, Y, mode="multivariate", n_partitions=4, n_bootstrap=4, key=jax.random.key(SEED)
    )


def _ot_trajectory(shape: str) -> Any:
    X, Y = _given_data(shape)
    return optimal_transport.analyze(
        PROBLEM, X, Y, mode="trajectory", n_partitions=4, n_bootstrap=4, key=jax.random.key(SEED)
    )


def _pce(shape: str) -> Any:
    X, Y = _given_data(shape)
    return pce.analyze(PROBLEM, X, Y, order=3)


def _hdmr(shape: str) -> Any:
    # RS-HDMR refuses fewer than 300 rows: the B-spline fit needs them.
    X, Y = _given_data(shape, n=384)
    return hdmr.analyze(PROBLEM, X, Y, maxorder=2, maxiter=20)


def _hdmr_third_order(shape: str) -> Any:
    X, Y = _given_data(shape, n=384)
    return hdmr.analyze(PROBLEM, X, Y, maxorder=3, maxiter=20)


def _shapley(shape: str) -> Any:
    X, Y = _given_data(shape)
    return shapley.analyze(PROBLEM, X, Y, backend="pce", order=3)


def _vkoga(shape: str) -> Any:
    X, Y = _given_data(shape)
    return vkoga.analyze(
        PROBLEM,
        X,
        Y,
        max_centers=16,
        n_folds=3,
        n_outer=32,
        n_inner=16,
        n_variance=128,
        key=jax.random.key(SEED),
    )


BUILDERS: dict[str, Callable[[str], Any]] = {
    "borgonovo": _borgonovo,
    "dgsm": _dgsm,
    "efast": _efast,
    "hdmr": _hdmr,
    "hdmr_third_order": _hdmr_third_order,
    "hsic": _hsic,
    "kucherenko": _kucherenko,
    "morris": _morris,
    "optimal_transport": _ot,
    "optimal_transport_multivariate": _ot_multivariate,
    "optimal_transport_trajectory": _ot_trajectory,
    "pawn": _pawn,
    "pce": _pce,
    "shapley": _shapley,
    "sobol": _sobol,
    "vkoga": _vkoga,
}
"""One builder per exported schema variant, keyed by snapshot name."""

# Two OT modes fix their own output layout, so extra output shapes would add
# rows without adding a schema. "multivariate" always reduces to one index per
# parameter, and "trajectory" only accepts a 3-D Y.
_SHAPES: dict[str, list[str]] = {
    "optimal_transport_multivariate": ["scalar"],
    "optimal_transport_trajectory": ["series"],
}

CASES: list[tuple[str, str]] = sorted(
    (name, shape) for name in BUILDERS for shape in _SHAPES.get(name, list(MODELS))
)
"""Every ``(builder, output shape)`` pair the snapshot covers."""


def build(name: str, shape: str) -> Any:
    """Build one result.

    Args:
        name: Key into :data:`BUILDERS`.
        shape: Key into :data:`MODELS`.

    Returns:
        The result object that method returns for that output shape.
    """
    return BUILDERS[name](shape)


def dataset_schema(result: Any) -> dict[str, list[str]]:
    """Describe the exported schema of one result.

    Args:
        result: Any jaxgsa result with a ``to_dataset`` method.

    Returns:
        Sorted dimension names, coordinate names and data-variable names.
        Only names are recorded: the snapshot guards the export schema, and
        ``scripts/baseline_check.py`` guards the numbers.
    """
    ds = result.to_dataset()
    return {
        "dims": sorted(str(d) for d in ds.sizes),
        "coords": sorted(str(c) for c in ds.coords),
        "data_vars": sorted(str(v) for v in ds.data_vars),
        "var_dims": sorted(f"{v}:{','.join(str(d) for d in ds[v].dims)}" for v in ds.data_vars),
        "attrs": sorted(str(a) for a in ds.attrs),
    }
