"""``Problem`` as a JAX pytree: flattening, tracing, and differentiation.

The point of the registration is ``d(index)/d(marginal parameter)``. These
tests therefore check the split between leaves and static metadata, that the
hashability existing code relies on survived, and that a gradient taken
through a real Sobol analysis matches central finite differences instead of
being silently zero.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa.sobol as sobol
from jaxgsa._core.validation import _warn_zero_variance_slices
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.problem import CategoricalSpec, GaussianSpec, Problem, UniformSpec

UNIT_BOUNDS: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
UNIT_PROBLEM = Problem(("x1", "x2", "x3"), UNIT_BOUNDS)


def mixed_problem() -> Problem:
    """Return a problem with one marginal of every family."""
    return Problem.from_dict(
        {
            "a": (0.0, 2.0),
            "b": {"dist": "gaussian", "mean": 1.0, "variance": 4.0, "low": -1.0},
            "c": {"dist": "categorical", "probs": [0.25, 0.75], "labels": ["off", "on"]},
        },
        output_names=("y",),
    )


class TestFlattening:
    """What is a leaf and what is structure."""

    def test_leaves_are_exactly_the_marginal_numbers(self):
        leaves = jax.tree_util.tree_leaves(mixed_problem())
        # low, high | mean, variance, low (high is absent) | two probabilities
        assert leaves == [0.0, 2.0, 1.0, 4.0, -1.0, 0.25, 0.75]

    def test_structural_metadata_is_not_a_leaf(self):
        problem = mixed_problem().with_correlation(np.eye(3))
        leaves = jax.tree_util.tree_leaves(problem)

        # Names, output names, categorical labels and the correlation matrix
        # are all absent from the leaves; only floats are there.
        assert all(isinstance(leaf, float) for leaf in leaves)
        assert not any(isinstance(leaf, str) for leaf in leaves)
        for structural in ("a", "b", "c", "y", "off", "on"):
            assert structural not in leaves
        assert 1.0 not in leaves[:2]  # no correlation diagonal smuggled in

    def test_leaf_paths_name_the_attribute_they_came_from(self):
        paths = [
            jax.tree_util.keystr(path)
            for path, _ in jax.tree_util.tree_flatten_with_path(mixed_problem())[0]
        ]
        assert paths[0] == ".input_specs[0].low"
        assert paths[3] == ".input_specs[1].variance"
        assert paths[-1] == ".input_specs[2].probs[1]"

    def test_gaussian_truncation_state_is_structure_not_a_leaf(self):
        unbounded = Problem.from_dict({"g": {"dist": "gaussian", "mean": 0.0, "variance": 1.0}})
        bounded = Problem.from_dict(
            {"g": {"dist": "gaussian", "mean": 0.0, "variance": 1.0, "low": -2.0, "high": 2.0}}
        )
        # An absent bound is an empty subtree, so the two differ in structure,
        # not in a leaf value. Which sampler runs is therefore never traced.
        assert len(jax.tree_util.tree_leaves(unbounded)) == 2
        assert len(jax.tree_util.tree_leaves(bounded)) == 4
        assert jax.tree_util.tree_structure(unbounded) != jax.tree_util.tree_structure(bounded)

    def test_specs_flatten_on_their_own(self):
        assert jax.tree_util.tree_leaves(UniformSpec(0.0, 3.0)) == [0.0, 3.0]
        assert jax.tree_util.tree_leaves(GaussianSpec(1.0, 2.0)) == [1.0, 2.0]
        assert jax.tree_util.tree_leaves(CategoricalSpec((0.5, 0.5))) == [0.5, 0.5]


class TestRoundTrip:
    """``tree_flatten`` then ``tree_unflatten`` must give the same problem."""

    def test_round_trip_is_exact_for_a_mixed_problem(self):
        problem = mixed_problem()
        leaves, treedef = jax.tree_util.tree_flatten(problem)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

        assert rebuilt == problem
        assert rebuilt.names == problem.names
        assert rebuilt.output_names == problem.output_names
        assert rebuilt.bounds == problem.bounds
        assert rebuilt.input_specs == problem.input_specs
        assert rebuilt.categorical_labels == {"c": ("off", "on")}
        assert rebuilt.has_categorical_inputs is True

    def test_round_trip_preserves_the_correlation_matrix(self):
        R = np.array([[1.0, 0.4, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]])
        problem = mixed_problem().with_correlation(R)
        rebuilt = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(problem)))

        assert rebuilt.correlation is not None
        np.testing.assert_array_equal(rebuilt.correlation, problem.correlation)
        assert rebuilt.has_correlated_inputs is True

    def test_bounds_follow_the_leaves_they_cache(self):
        widened = jax.tree_util.tree_map(lambda leaf: leaf * 2.0, UNIT_PROBLEM)
        assert widened.bounds == ((0.0, 2.0), (0.0, 2.0), (0.0, 2.0))


class TestHashability:
    """Registration must not cost the hashability existing code relies on."""

    def test_hash_and_equality_still_behave(self):
        one = mixed_problem()
        same = mixed_problem()
        other = Problem(("x",), ((0.0, 1.0),))

        assert one == same
        assert hash(one) == hash(same)
        assert one != other
        assert len({one, same, other}) == 2
        assert {one: "kept"}[same] == "kept"

    def test_problem_still_works_as_a_static_jit_argument(self):
        @jax.jit
        def scale(y, problem):
            return y * problem.num_vars

        static = jax.jit(scale, static_argnums=(1,))
        assert float(static(2.0, UNIT_PROBLEM)) == 6.0
        # Second call with the same problem hits the cache rather than raising.
        assert float(static(3.0, UNIT_PROBLEM)) == 9.0


def uniform_transform(problem: Problem, unit: jnp.ndarray) -> jnp.ndarray:
    """Map a unit-hypercube design onto a uniform problem's bounds."""
    specs = [spec for spec in problem.input_specs if isinstance(spec, UniformSpec)]
    assert len(specs) == problem.num_vars, "this helper handles uniform marginals only"
    lows = jnp.array([spec.low for spec in specs])
    highs = jnp.array([spec.high for spec in specs])
    return lows + (highs - lows) * unit


def model(X: jnp.ndarray) -> jnp.ndarray:
    """A smooth, interacting test model."""
    return jnp.sin(X[:, 0]) + 2.0 * X[:, 1] * X[:, 2] + 0.5 * X[:, 0] ** 2


class TestTracing:
    """A ``Problem`` argument survives ``jit``, ``vmap`` and ``grad``."""

    def test_jit_over_a_function_taking_a_problem(self):
        @jax.jit
        def uniform_variance(problem):
            spec = problem.input_specs[0]
            return (spec.high - spec.low) ** 2 / 12.0

        assert float(uniform_variance(Problem(("x",), ((0.0, 2.0),)))) == pytest.approx(1.0 / 3.0)
        # A different bound reuses the compiled function: it is a leaf, not a
        # cache key, so nothing recompiles and the answer still moves.
        assert float(uniform_variance(Problem(("x",), ((0.0, 4.0),)))) == pytest.approx(4.0 / 3.0)

    def test_vmap_over_a_batch_of_marginal_parameters(self):
        # The model output over the transformed design is a genuine function
        # of every bound, so a batch of bounds must give a batch of answers.
        # This stops short of `sobol.analyze`: that path still reads a value
        # on the host inside `_core/invalid.py`, which no `Problem` change can
        # reach. The gradient test below does go through `analyze`, because
        # `grad` never abstracts values.
        unit = jnp.asarray(np.asarray(sobol.sample(UNIT_PROBLEM, 256, seed=0).transform()))

        def output_variance(problem):
            return jnp.var(model(uniform_transform(problem, unit)))

        leaves, treedef = jax.tree_util.tree_flatten(UNIT_PROBLEM)
        highs = jnp.array([1.0, 2.0, 3.0])
        batched_leaves = [
            highs if index == 1 else jnp.full(3, leaf) for index, leaf in enumerate(leaves)
        ]
        batched = jax.tree_util.tree_unflatten(treedef, batched_leaves)

        out = jax.vmap(output_variance)(batched)
        assert out.shape == (3,)
        assert len(set(np.asarray(out).tolist())) == 3  # the batch really varies
        # Each batch element must agree with the same problem run on its own.
        for index, high in enumerate([1.0, 2.0, 3.0]):
            one = Problem(UNIT_PROBLEM.names, ((0.0, high), (0.0, 1.0), (0.0, 1.0)))
            assert float(out[index]) == pytest.approx(float(output_variance(one)), abs=1e-5)

    def test_grad_through_marginal_bounds_matches_central_differences(self):
        samples = sobol.sample(UNIT_PROBLEM, 256, seed=0)
        unit = jnp.asarray(np.asarray(samples.transform()))

        def first_order(problem):
            return sobol.analyze(samples, model(uniform_transform(problem, unit))).S1[0]

        gradient = jax.tree_util.tree_leaves(jax.grad(first_order)(UNIT_PROBLEM))
        assert len(gradient) == 6  # one per bound of the three uniform marginals

        step = 1e-3
        for index in range(6):
            param, side = divmod(index, 2)

            def shifted(delta, param=param, side=side):
                bounds = list(UNIT_BOUNDS)
                low, high = bounds[param]
                bounds[param] = (low + delta, high) if side == 0 else (low, high + delta)
                return float(first_order(Problem(UNIT_PROBLEM.names, tuple(bounds))))

            central = (shifted(step) - shifted(-step)) / (2.0 * step)
            derivative = float(gradient[index])
            # A silently zero chain would pass a shape-only check, so the
            # derivative has to be both finite and clearly non-zero.
            assert np.isfinite(derivative)
            assert abs(derivative) > 1e-3
            assert derivative == pytest.approx(central, abs=2e-3, rel=1e-2)


class TestZeroVarianceUnderTrace:
    """The zero-variance warning must reach the user in every mode."""

    @staticmethod
    def constant_second_output() -> jnp.ndarray:
        """Return a ``(N, 2)`` output whose second column never moves."""
        return jnp.stack([jnp.arange(8.0), jnp.zeros(8)], axis=1)

    @staticmethod
    def check(Y):
        """Run the zero-variance check and return something traceable."""
        _warn_zero_variance_slices(Y, output_names=("a", "b"))
        return Y.sum()

    def test_eager_path_still_warns(self):
        with pytest.warns(JaxgsaWarning, match=r"1/2 output\(s\) have zero variance"):
            self.check(self.constant_second_output())

    def test_jit_path_warns_from_the_host_callback(self):
        with pytest.warns(JaxgsaWarning, match=r"1/2 output\(s\) have zero variance"):
            jax.block_until_ready(jax.jit(self.check)(self.constant_second_output()))

    def test_vmap_path_warns_once_per_batch_element(self):
        batch = jnp.stack([self.constant_second_output()] * 2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jax.block_until_ready(jax.vmap(self.check)(batch))
        messages = [str(record.message) for record in caught]
        assert len(messages) == 2
        assert all("zero variance" in message for message in messages)

    def test_a_traced_healthy_output_stays_quiet(self):
        healthy = jnp.stack([jnp.arange(8.0), jnp.arange(8.0) ** 2], axis=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jax.block_until_ready(jax.jit(self.check)(healthy))
        assert [record for record in caught if issubclass(record.category, JaxgsaWarning)] == []
