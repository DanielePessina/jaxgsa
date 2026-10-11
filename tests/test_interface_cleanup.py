"""Interface rules from the pre-announcement cleanup.

Covers the renames and input checks that touch every method: ``on_invalid``
values, ``key`` coercion, ``seed`` rejection, ``Problem`` argument rules, the
private registry names and ``jaxgsa.sampling.transform``.
"""

import importlib

import jax
import numpy as np
import pytest
from scipy.stats import qmc

import jaxgsa
from jaxgsa import Problem
from jaxgsa._core.validation import as_key
from jaxgsa.sampling import monte_carlo, transform

METHODS = (
    "borgonovo",
    "dgsm",
    "efast",
    "hdmr",
    "hsic",
    "kucherenko",
    "morris",
    "optimal_transport",
    "pawn",
    "pce",
    "rank",
    "shapley",
    "sobol",
    "vkoga",
)


def test_on_invalid_none_is_rejected_and_skip_is_listed():
    from jaxgsa._core.invalid import resolve_policy

    unit = jaxgsa.InvalidUnit.ROW
    with pytest.raises(ValueError, match="'skip'") as exc:
        resolve_policy("none", method="m", unit=unit, allow_drop=True)
    assert "'none'" in str(exc.value)
    assert resolve_policy("skip", method="m", unit=unit, allow_drop=True) == "skip"


def test_as_key_accepts_int_and_keys_and_rejects_the_rest():
    assert as_key(None, "pawn") is None
    assert (
        jax.random.key_data(as_key(3, "pawn")).tolist()
        == jax.random.key_data(jax.random.key(3)).tolist()
    )
    typed = jax.random.key(1)
    assert as_key(typed, "pawn") is typed
    legacy = jax.random.PRNGKey(1)
    assert as_key(legacy, "pawn") is legacy
    for bad in (1.5, True, "7", np.random.default_rng(0), np.zeros(3)):
        with pytest.raises(TypeError, match="jaxgsa.pawn: key must be a jax.random.key"):
            as_key(bad, "pawn")


def test_analyze_takes_an_int_key():
    problem = Problem(("a", "b"), ((0.0, 1.0), (0.0, 1.0)))
    X = monte_carlo(problem, 256, seed=0)
    Y = X[:, 0] + 2.0 * X[:, 1]
    by_int = jaxgsa.pawn.analyze(problem, X, Y, n_bootstrap=3, key=5, verbose=False)
    by_key = jaxgsa.pawn.analyze(
        problem, X, Y, n_bootstrap=3, key=jax.random.key(5), verbose=False
    )
    np.testing.assert_array_equal(by_int.pawn_conf, by_key.pawn_conf)
    with pytest.raises(TypeError, match="jaxgsa.pawn: key must be"):
        jaxgsa.pawn.analyze(problem, X, Y, key=1.5, verbose=False)


def test_samplers_reject_a_jax_key_as_seed():
    problem = Problem(("a", "b"), ((0.0, 1.0), (0.0, 1.0)))
    key = jax.random.key(0)
    with pytest.raises(TypeError, match="jax keys go to analyze"):
        monte_carlo(problem, 8, seed=key)
    with pytest.raises(TypeError, match="jax keys go to analyze"):
        jaxgsa.sobol.sample(problem, 64, seed=key, verbose=False)
    with pytest.raises(TypeError, match="jax keys go to analyze"):
        jaxgsa.morris.sample(problem, 4, seed=key, verbose=False)
    with pytest.raises(TypeError, match="jax keys go to analyze"):
        jaxgsa.efast.sample(problem, 65, seed=key, verbose=False)


def test_problem_rejects_bare_strings_and_keeps_keyword_only_arguments():
    with pytest.raises(TypeError, match="output_names"):
        Problem(("a",), ((0.0, 1.0),), output_names="temp")
    with pytest.raises(TypeError, match="names"):
        Problem("ab", ((0.0, 1.0), (0.0, 1.0)))
    with pytest.raises(TypeError, match="output_names"):
        Problem.from_dict({"a": (0.0, 1.0)}, output_names="temp")
    with pytest.raises(TypeError, match="output_names"):
        Problem.from_dict({"a": (0.0, 1.0)}).with_output_names("temp")
    with pytest.raises(TypeError):
        Problem(("a",), ((0.0, 1.0),), ("y",))  # type: ignore[misc]
    with pytest.raises(TypeError):
        Problem.from_dict({"a": (0.0, 1.0)}, ("y",))  # type: ignore[misc]
    assert Problem(("a",), ((0.0, 1.0),), output_names=["y"]).output_names == ("y",)


def test_from_dict_accepts_list_bounds():
    listed = Problem.from_dict({"a": [0.0, 2.0], "b": (1.0, 3.0)})
    assert listed.bounds == ((0.0, 2.0), (1.0, 3.0))
    with pytest.raises(ValueError, match="exactly two"):
        Problem.from_dict({"a": [0.0, 1.0, 2.0]})


@pytest.mark.parametrize("method", METHODS)
def test_method_namespaces_keep_registry_names_private(method):
    module = importlib.import_module(f"jaxgsa.{method}")
    public = [name for name in dir(module) if not name.startswith("_")]
    for leaked in ("SPEC", "MethodSpec", "register", "InvalidUnit"):
        assert leaked not in public
    assert module._SPEC.name == method


def test_sampling_and_config_namespaces_are_clean():
    for module in (jaxgsa.sampling, jaxgsa.config):
        public = {name for name in dir(module) if not name.startswith("_")}
        assert public <= set(module.__all__)


def test_transform_matches_the_marginals_and_monte_carlo():
    problem = Problem.from_dict(
        {"a": (1.0, 3.0), "b": {"dist": "gaussian", "mean": 5.0, "variance": 4.0}}
    )
    U = qmc.LatinHypercube(d=2, seed=0).random(64)
    X = transform(problem, U)
    assert X.shape == (64, 2)
    assert X[:, 0].min() >= 1.0 and X[:, 0].max() <= 3.0
    np.testing.assert_allclose(X[:, 0], 1.0 + 2.0 * U[:, 0])
    # monte_carlo is transform of the generator's own uniforms.
    rng = np.random.default_rng(9)
    np.testing.assert_array_equal(
        monte_carlo(problem, 32, seed=9), transform(problem, rng.random((32, 2)))
    )


def test_transform_applies_the_copula_and_the_sampler():
    R = [[1.0, 0.9], [0.9, 1.0]]
    correlated = Problem(("a", "b"), ((0.0, 1.0), (0.0, 1.0)), correlation=R)
    U = np.random.default_rng(1).random((4000, 2))
    X = transform(correlated, U)
    assert np.corrcoef(X, rowvar=False)[0, 1] > 0.5
    assert X.min() >= 0.0 and X.max() <= 1.0
    edge = transform(correlated, np.array([[0.0, 1.0]]))
    assert np.all(np.isfinite(edge))

    def doubled(u, **_):
        return np.column_stack((u[:, 0], u[:, 0]))

    custom = Problem(("a", "b"), ((0.0, 1.0), (0.0, 1.0))).with_sampler(
        doubled, sampler_dimension=1
    )
    X = transform(custom, np.array([[0.25], [0.5]]))
    np.testing.assert_array_equal(X, [[0.25, 0.25], [0.5, 0.5]])


def test_transform_validates_its_input():
    problem = Problem(("a", "b"), ((0.0, 1.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match=r"U must be \(n, 2\)"):
        transform(problem, np.zeros((3, 3)))
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        transform(problem, np.array([[0.5, 1.5]]))
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        transform(problem, np.array([[0.5, np.nan]]))
