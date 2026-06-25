import jax
import jax.numpy as jnp

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


def test_first_order_known_value():
    # If AB_j == A, then S1_j should be ~0 (no sensitivity to param j)
    A = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    B = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])
    AB_j = A  # no change → no sensitivity
    result = first_order(A, AB_j, B)
    assert jnp.abs(result) < 0.01


def test_total_order_known_value():
    # If AB_j == A, then ST_j should be ~0
    A = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    B = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])
    AB_j = A
    result = total_order(A, AB_j, B)
    assert jnp.abs(result) < 0.01
