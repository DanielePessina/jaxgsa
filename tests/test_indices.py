import jax
import jax.numpy as jnp
import numpy as np

from gsax.sobol._indices import first_order, second_order, total_order


def test_first_order_jit():
    A = jnp.array([1.0, 2.0, 3.0, 4.0])
    AB_j = jnp.array([1.5, 2.5, 3.5, 4.5])
    B = jnp.array([2.0, 3.0, 4.0, 5.0])
    result = jax.jit(first_order)(A, AB_j, B)
    assert jnp.isfinite(result)


def test_total_order_jit():
    A = jnp.array([1.0, 2.0, 3.0, 4.0])
    AB_j = jnp.array([1.5, 2.5, 3.5, 4.5])
    B = jnp.array([2.0, 3.0, 4.0, 5.0])
    result = jax.jit(total_order)(A, AB_j, B)
    assert jnp.isfinite(result)
    assert result >= 0  # total order should be non-negative for reasonable inputs


def test_second_order_jit():
    A = jnp.array([1.0, 2.0, 3.0, 4.0])
    AB_j = jnp.array([1.5, 2.5, 3.5, 4.5])
    AB_k = jnp.array([1.2, 2.2, 3.2, 4.2])
    BA_j = jnp.array([1.8, 2.8, 3.8, 4.8])
    B = jnp.array([2.0, 3.0, 4.0, 5.0])
    result = jax.jit(second_order)(A, AB_j, AB_k, BA_j, B)
    assert jnp.isfinite(result)


class TestKnownValues:
    """Tests that verify actual computed values from the Sobol index estimators
    using large random samples for statistical robustness."""

    def _random_data(self, n: int = 10_000):
        """Generate reproducible random A and B arrays."""
        rng = np.random.default_rng(42)
        A = jnp.array(rng.standard_normal(n))
        B = jnp.array(rng.standard_normal(n))
        return A, B

    def test_first_order_no_sensitivity(self):
        """When AB_j == A (replacing param j has no effect), S1_j should be ~0."""
        A, B = self._random_data()
        result = first_order(A, A, B)
        assert jnp.abs(result) < 0.01

    def test_first_order_full_sensitivity(self):
        """When AB_j == B (full replacement), S1_j should be high (>0.8)."""
        A, B = self._random_data()
        result = first_order(A, B, B)
        assert result > 0.8

    def test_total_order_no_sensitivity(self):
        """When AB_j == A (no change from replacing param j), ST_j should be ~0."""
        A, B = self._random_data()
        result = total_order(A, A, B)
        assert jnp.abs(result) < 0.01

    def test_total_order_full_sensitivity(self):
        """When AB_j == B (full replacement), ST_j should be high."""
        A, B = self._random_data()
        result = total_order(A, B, B)
        assert result > 0.8

    def test_second_order_independent(self):
        """When no parameter has any effect, S2 should be ~0.

        If param j has no sensitivity, AB_j = A (replacing col j of A with
        col j of B has no effect on output) and BA_j = B (replacing col j
        of B with col j of A has no effect on B-based output).
        """
        A, B = self._random_data()
        # AB_j = A, AB_k = A (no first-order effects), BA_j = B
        result = second_order(A, A, A, B, B)
        assert jnp.abs(result) < 0.01
