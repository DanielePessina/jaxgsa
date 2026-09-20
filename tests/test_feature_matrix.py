"""Small package-only end-to-end coverage for every sensitivity method.

The ordinary method tests go deep on individual kernels and edge cases. This
matrix answers a different question: can a user take a deterministic problem,
evaluate a model, and complete one supported analysis through every public
method? It deliberately compares no result with another implementation. The
assertions are package contracts — result type, output shape, finite primary
indices, and the metadata for the feature being exercised.

The cases are intentionally small. A method-specific lower bound is retained
where the algorithm needs it (RS-HDMR needs 300 rows); all other rows use a
few dozen deterministic samples. The matrix is the default CI gate, while the
larger method checks remain useful for focused local work.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pytest

import jaxgsa
from jaxgsa._core.registry import methods

SEED = 17
"""Seed shared by every deterministic fixture."""

CONTINUOUS = jaxgsa.Problem.from_dict({"x1": (-1.0, 1.0), "x2": (-1.0, 1.0), "x3": (-1.0, 1.0)})
"""Three independent continuous inputs for the smooth-model cases."""

CORRELATED = jaxgsa.Problem.from_dict(
    {"x1": (-1.0, 1.0), "x2": (-1.0, 1.0), "x3": (-1.0, 1.0)},
    correlation=((1.0, 0.5, 0.2), (0.5, 1.0, 0.1), (0.2, 0.1, 1.0)),
)
"""A continuous problem whose given-data methods can read dependence."""

CATEGORICAL = jaxgsa.Problem.from_dict(
    {"u": (0.0, 1.0), "kind": {"dist": "categorical", "probs": [0.5, 0.5]}}
)
"""Mixed continuous/categorical problem for native categorical methods."""

REQUIRED_FEATURES = frozenset(
    {
        "scalar-output",
        "multi-output",
        "time-series",
        "given-data",
        "own-design",
        "bootstrap",
        "second-order",
        "categorical",
        "correlated",
        "gradient",
        "permutation",
        "surrogate",
        "dummy",
        "ot-multivariate",
        "ot-trajectory",
    }
)
"""Features that the matrix must exercise at least once."""


def _multi_model(X: jax.Array) -> jax.Array:
    """Evaluate a smooth two-output model on a batch of rows."""
    first = X[:, 0] + 2.0 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]
    second = 0.5 * X[:, 1] - X[:, 2] ** 2 + 0.2 * X[:, 0]
    return jnp.stack([first, second], axis=-1)


def _series_model(X: jax.Array) -> jax.Array:
    """Widen the two-output model to two time steps."""
    outputs = _multi_model(X)
    return jnp.stack([outputs, 0.5 * outputs + 0.2], axis=1)


def _given_data(problem: jaxgsa.Problem, n: int = 32) -> tuple[jax.Array, jax.Array]:
    """Build deterministic independent Monte Carlo data for a problem."""
    X = jaxgsa.sampling.monte_carlo(problem, n, seed=SEED)
    return jnp.asarray(X), _multi_model(jnp.asarray(X))


def _point_model(model: Callable[[jax.Array], jax.Array]) -> Callable[[jax.Array], jax.Array]:
    """Adapt a batch model to DGSM's one-row calling convention."""

    def evaluate_one(x: jax.Array) -> jax.Array:
        return model(x[None, :])[0]

    return evaluate_one


def _categorical_data() -> tuple[jax.Array, jax.Array]:
    """Build balanced mixed data without a second implementation."""
    unit = jnp.linspace(0.01, 0.99, 64)
    levels = jnp.tile(jnp.asarray([0.0, 1.0]), 32)
    X = jnp.stack([unit, levels], axis=1)
    Y = unit + 0.3 * levels + 0.1 * jnp.sin(7.0 * unit)
    return X, Y


@dataclass(frozen=True)
class FeatureCase:
    """One end-to-end row in the package-only feature matrix."""

    name: str
    method: str
    expected_shape: tuple[int, ...]
    features: frozenset[str]
    run: Callable[[], Any]


def _run_borgonovo() -> Any:
    X, Y = _categorical_data()
    return jaxgsa.borgonovo.analyze(CATEGORICAL, X, Y, n_classes=4, grid_size=24, verbose=False)


def _run_dgsm() -> Any:
    X, _ = _given_data(CONTINUOUS)
    return jaxgsa.dgsm.analyze(
        CONTINUOUS,
        _point_model(_series_model),
        X,
        standardize_outputs=True,
        verbose=False,
    )


def _run_efast() -> Any:
    design = jaxgsa.efast.sample(CONTINUOUS, n_per_curve=129, seed=SEED, verbose=False)
    Y = _multi_model(jnp.asarray(design.samples))
    return jaxgsa.efast.analyze(design, Y, verbose=False)


def _run_hdmr() -> Any:
    X, Y = _given_data(CONTINUOUS, n=300)
    return jaxgsa.hdmr.analyze(CONTINUOUS, X, Y[:, 0], maxorder=2, maxiter=3, verbose=False)


def _run_hsic() -> Any:
    X, _ = _given_data(CORRELATED, n=40)
    return jaxgsa.hsic.analyze(
        CORRELATED,
        X,
        _series_model(X),
        n_perms=3,
        key=jax.random.key(SEED),
        verbose=False,
    )


def _run_kucherenko() -> Any:
    design = jaxgsa.kucherenko.sample(CONTINUOUS, n_samples=16, seed=SEED, verbose=False)
    Y = _series_model(jnp.asarray(design.samples))
    return jaxgsa.kucherenko.analyze(
        design,
        Y,
        n_bootstrap=2,
        key=jax.random.key(SEED),
        verbose=False,
    )


def _run_morris() -> Any:
    design = jaxgsa.morris.sample(CONTINUOUS, n_trajectories=4, seed=SEED, verbose=False)
    Y = _multi_model(jnp.asarray(design.samples))
    return jaxgsa.morris.analyze(
        design,
        Y,
        n_bootstrap=2,
        key=jax.random.key(SEED),
        verbose=False,
    )


def _run_ot_categorical() -> Any:
    X, Y = _categorical_data()
    return jaxgsa.optimal_transport.analyze(
        CATEGORICAL,
        X,
        Y,
        n_partitions=4,
        dummy=True,
        n_bootstrap=2,
        key=jax.random.key(SEED),
        verbose=False,
    )


def _run_ot_multivariate() -> Any:
    X, Y = _given_data(CONTINUOUS)
    return jaxgsa.optimal_transport.analyze(
        CONTINUOUS,
        X,
        Y,
        mode="multivariate",
        n_partitions=4,
        epsilon=0.05,
        max_iter=100,
        tol=1e-4,
        verbose=False,
    )


def _run_ot_trajectory() -> Any:
    X, _ = _given_data(CONTINUOUS)
    return jaxgsa.optimal_transport.analyze(
        CONTINUOUS,
        X,
        _series_model(X),
        mode="trajectory",
        n_partitions=4,
        epsilon=0.05,
        max_iter=100,
        tol=1e-4,
        verbose=False,
    )


def _run_pawn() -> Any:
    X, Y = _categorical_data()
    return jaxgsa.pawn.analyze(
        CATEGORICAL,
        X,
        jnp.stack([Y, Y**2], axis=1),
        n_bins=3,
        statistic="median",
        verbose=False,
    )


def _run_pce() -> Any:
    X, Y = _given_data(CONTINUOUS)
    return jaxgsa.pce.analyze(CONTINUOUS, X, Y, order=2, verbose=False)


def _run_shapley() -> Any:
    X, Y = _given_data(CONTINUOUS)
    return jaxgsa.shapley.analyze(CONTINUOUS, X, Y[:, 0], backend="pce", order=2, verbose=False)


def _run_sobol() -> Any:
    design = jaxgsa.sobol.sample(CONTINUOUS, n_samples=64, seed=SEED, verbose=False)
    return jaxgsa.sobol.analyze(
        design,
        _series_model(jnp.asarray(design.samples)),
        n_bootstrap=2,
        key=jax.random.key(SEED),
        verbose=False,
    )


def _run_vkoga() -> Any:
    # Train on independent rows while declaring the correlated target measure.
    X, Y = _given_data(CONTINUOUS)
    return jaxgsa.vkoga.analyze(
        CORRELATED,
        X,
        Y,
        max_centers=4,
        n_folds=2,
        n_outer=4,
        n_inner=4,
        n_variance=16,
        key=jax.random.key(SEED),
        verbose=False,
    )


MATRIX = (
    FeatureCase(
        "borgonovo-categorical",
        "borgonovo",
        (2,),
        frozenset({"given-data", "scalar-output", "categorical"}),
        _run_borgonovo,
    ),
    FeatureCase(
        "dgsm-series",
        "dgsm",
        (2, 2, 3),
        frozenset({"given-data", "time-series", "gradient"}),
        _run_dgsm,
    ),
    FeatureCase(
        "efast-multi-output",
        "efast",
        (2, 3),
        frozenset({"own-design", "multi-output"}),
        _run_efast,
    ),
    FeatureCase(
        "hdmr-scalar-surrogate",
        "hdmr",
        (6,),
        frozenset({"given-data", "scalar-output", "surrogate"}),
        _run_hdmr,
    ),
    FeatureCase(
        "hsic-correlated-series",
        "hsic",
        (2, 2, 3),
        frozenset({"given-data", "time-series", "multi-output", "correlated", "permutation"}),
        _run_hsic,
    ),
    FeatureCase(
        "kucherenko-series-bootstrap",
        "kucherenko",
        (2, 2, 3),
        frozenset({"own-design", "time-series", "bootstrap"}),
        _run_kucherenko,
    ),
    FeatureCase(
        "morris-multi-bootstrap",
        "morris",
        (2, 3),
        frozenset({"own-design", "multi-output", "bootstrap"}),
        _run_morris,
    ),
    FeatureCase(
        "ot-categorical-dummy-bootstrap",
        "optimal_transport",
        (2,),
        frozenset(
            {"given-data", "scalar-output", "categorical", "dummy", "bootstrap", "ot-scalar"}
        ),
        _run_ot_categorical,
    ),
    FeatureCase(
        "ot-multivariate",
        "optimal_transport",
        (3,),
        frozenset({"given-data", "multi-output", "ot-multivariate"}),
        _run_ot_multivariate,
    ),
    FeatureCase(
        "ot-trajectory",
        "optimal_transport",
        (2, 3),
        frozenset({"given-data", "time-series", "ot-trajectory"}),
        _run_ot_trajectory,
    ),
    FeatureCase(
        "pawn-categorical-multi-output",
        "pawn",
        (2, 2),
        frozenset({"given-data", "categorical", "multi-output"}),
        _run_pawn,
    ),
    FeatureCase(
        "pce-multi-output-surrogate",
        "pce",
        (2, 3),
        frozenset({"given-data", "multi-output", "surrogate"}),
        _run_pce,
    ),
    FeatureCase(
        "shapley-scalar-surrogate",
        "shapley",
        (3,),
        frozenset({"given-data", "scalar-output", "surrogate"}),
        _run_shapley,
    ),
    FeatureCase(
        "sobol-second-order-series-bootstrap",
        "sobol",
        (2, 2, 3),
        frozenset({"own-design", "time-series", "second-order", "bootstrap"}),
        _run_sobol,
    ),
    FeatureCase(
        "vkoga-correlated-multi-output-surrogate",
        "vkoga",
        (2, 3),
        frozenset({"given-data", "multi-output", "correlated", "surrogate"}),
        _run_vkoga,
    ),
)


def _methods_with_feature(feature: str) -> frozenset[str]:
    """Return methods whose matrix row exercises ``feature``."""
    return frozenset(case.method for case in MATRIX if feature in case.features)


def _registered_with(feature: str, value: object = True) -> frozenset[str]:
    """Return registry methods whose declaration field has ``value``."""
    return frozenset(name for name, spec in methods().items() if getattr(spec, feature) == value)


REGISTERED_FEATURES = {
    "own-design": _registered_with("is_design_based"),
    "given-data": _registered_with("is_design_based", False),
    "correlation-accepts": _registered_with("correlation", "accepts"),
    "correlation-refuses": _registered_with("correlation", "refuses"),
    "categorical-accepts": _registered_with("categorical", "accepts"),
    "categorical-refuses": _registered_with("categorical", "refuses"),
    "bootstrap-supported": frozenset(
        name for name, spec in methods().items() if spec.bootstrap is not None
    ),
    "pure-core": _registered_with("pure_core"),
}
"""Capability sets derived directly from the method registry."""

MATRIX_FEATURE_COVERAGE = {
    "own-design": _methods_with_feature("own-design"),
    "given-data": _methods_with_feature("given-data"),
    "correlated": _methods_with_feature("correlated"),
    "categorical": _methods_with_feature("categorical"),
    "bootstrap": _methods_with_feature("bootstrap"),
    "surrogate": _methods_with_feature("surrogate"),
    "output-shape": frozenset(case.method for case in MATRIX),
    "pure-core-row": frozenset(case.method for case in MATRIX if methods()[case.method].pure_core),
}
"""Actual feature coverage, kept separate from registry capability support."""

EXPECTED_CAPABILITY_COVERAGE = {
    # These subsets are intentional: the registry tests cover every gate, but
    # the fast end-to-end rows avoid repeating expensive fits for every option.
    "correlated": ("correlation-accepts", frozenset({"hsic", "vkoga"})),
    "categorical": (
        "categorical-accepts",
        frozenset({"borgonovo", "optimal_transport", "pawn"}),
    ),
    "bootstrap": (
        "bootstrap-supported",
        frozenset({"kucherenko", "morris", "optimal_transport", "sobol"}),
    ),
}
"""The deliberately small positive capability routes in this fast matrix."""

SURROGATE_METHODS = frozenset({"hdmr", "pce", "shapley", "vkoga"})
"""Methods whose matrix row exercises a fitted surrogate path."""


MATRIX_METHOD_MODES = {
    "optimal_transport": frozenset(
        feature.removeprefix("ot-")
        for case in MATRIX
        if case.method == "optimal_transport"
        for feature in case.features
        if feature.startswith("ot-")
    )
}
"""Method-specific modes exercised without rerunning a method unnecessarily."""


def test_matrix_routes_and_shapes_match_the_registry() -> None:
    """Every registry method has exactly one route and a checked shape."""
    registered = frozenset(methods())
    assert MATRIX_FEATURE_COVERAGE["own-design"] == REGISTERED_FEATURES["own-design"]
    assert MATRIX_FEATURE_COVERAGE["given-data"] == REGISTERED_FEATURES["given-data"]
    assert MATRIX_FEATURE_COVERAGE["output-shape"] == registered
    assert MATRIX_FEATURE_COVERAGE["pure-core-row"] == REGISTERED_FEATURES["pure-core"]


def test_matrix_capability_coverage_is_explicit_and_supported() -> None:
    """Capability subsets never claim support the registry does not declare."""
    for feature, (registry_feature, expected) in EXPECTED_CAPABILITY_COVERAGE.items():
        actual = MATRIX_FEATURE_COVERAGE[feature]
        assert actual == expected
        assert actual <= REGISTERED_FEATURES[registry_feature]

    assert MATRIX_FEATURE_COVERAGE["surrogate"] == SURROGATE_METHODS

    registered = frozenset(methods())
    for capability in ("correlation", "categorical"):
        accepts = REGISTERED_FEATURES[f"{capability}-accepts"]
        refuses = REGISTERED_FEATURES[f"{capability}-refuses"]
        assert accepts | refuses == registered
        assert accepts.isdisjoint(refuses)

    assert MATRIX_METHOD_MODES["optimal_transport"] == frozenset(
        {"scalar", "multivariate", "trajectory"}
    )


@pytest.mark.parametrize("case", MATRIX, ids=[case.name for case in MATRIX])
def test_every_matrix_case_completes_with_a_finite_primary_result(case: FeatureCase) -> None:
    """Every row reaches the public result object on deterministic data."""
    spec = methods()[case.method]
    with warnings.catch_warnings():
        # The matrix is not the warning-policy suite. These are expected for
        # deliberately tiny fixtures (single precision, a small conditional
        # design, or a last-iterate Sinkhorn solve) and would otherwise bury a
        # real failure in CI's warning summary.
        warnings.simplefilter("ignore", jaxgsa.JaxgsaWarning)
        warnings.filterwarnings("ignore", message="Explicitly requested dtype int64")
        result = case.run()

    assert isinstance(result, spec.result)
    assert result.invalid.n_invalid == 0
    primary = getattr(result, type(result)._schema.primary)
    assert primary.shape == case.expected_shape
    assert bool(jnp.all(jnp.isfinite(primary)))

    if "bootstrap" in case.features:
        assert result.ci is not None
        assert result.ci.n_bootstrap == 2
        assert result.ci.replicates is None
    else:
        assert getattr(result, "ci", None) is None

    if "dummy" in case.features:
        assert result.ot_dummy is not None
        assert result.above_dummy is not None
        assert result.ot_dummy.shape == result.ot.shape

    if "second-order" in case.features:
        assert result.S2 is not None
        assert result.S2.shape == (2, 2, 3, 3)
        diagonal = result.S2[..., jnp.arange(3), jnp.arange(3)]
        assert bool(jnp.all(jnp.isnan(diagonal)))


def test_matrix_covers_every_registered_method() -> None:
    """Adding a method without an end-to-end row fails the CI gate."""
    assert {case.method for case in MATRIX} == set(methods())


def test_matrix_declares_all_required_feature_dimensions() -> None:
    """The table stays honest about the dimensions it is meant to cover."""
    covered = frozenset().union(*(case.features for case in MATRIX))
    assert REQUIRED_FEATURES <= covered
