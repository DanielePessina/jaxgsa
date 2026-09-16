"""``cdf_to_unit_interval`` stays on device and preserves CDF invariants.

Every method's pure ``indices()`` core runs its inputs through this transform,
so a single host read here breaks the ``jit``/``vmap``/``jacrev`` contract for
all of them at once. Two separate defects did exactly that, and both are
pinned below. The scalar CDF checks use only monotonicity, normalization, and
the declared truncation interval, so this module does not depend on a second
implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa._core.transforms import _truncnorm_cdf, cdf_to_unit_interval

TRUNCATED = jaxgsa.Problem.from_dict(
    {
        "x1": {"dist": "gaussian", "mean": 0.0, "variance": 4.0, "low": -3.0, "high": 3.0},
        "x2": (0.0, 1.0),
    }
)
PLAIN_GAUSSIAN = jaxgsa.Problem.from_dict(
    {"x1": {"dist": "gaussian", "mean": 1.0, "variance": 4.0}, "x2": (0.0, 1.0)}
)


def _sample(problem):
    """Return a small in-range sample for a two-parameter problem."""
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.uniform(-1.0, 1.0, size=(16, problem.num_vars)))


@pytest.mark.parametrize("problem", [TRUNCATED, PLAIN_GAUSSIAN], ids=["truncated", "untruncated"])
def test_the_transform_survives_jit_vmap_and_jacrev(problem):
    """Tier T4 (behavioural contract): no host read of an array value.

    Both parametrisations bite, and they caught different defects.

    ``truncated`` previously failed because the transform converted a tracer
    to a host array while evaluating the truncated CDF.

    ``untruncated`` failed under ``jit`` alone because of
    ``std = float(jnp.sqrt(spec.variance))``. The variance is a plain Python
    float, but ``jnp.sqrt`` of it *inside a trace* returns a tracer, so
    ``float()`` on the result raises. That one broke every Gaussian input, not
    only the truncated ones, and it passed eagerly, which is why it survived.
    """
    X = _sample(problem)
    jax.jit(lambda x: cdf_to_unit_interval(x, problem))(X)
    jax.vmap(lambda row: cdf_to_unit_interval(row[None, :], problem)[0])(X)
    jax.jacrev(lambda x: cdf_to_unit_interval(x, problem).sum())(X)


@pytest.mark.parametrize(
    ("a", "b"),
    [(-2.0, 2.0), (-5.0, 5.0), (-np.inf, 1.5), (-1.5, np.inf), (0.5, 3.0), (3.0, 4.0), (5.0, 6.0)],
)
def test_truncnorm_cdf_is_monotone_and_normalized(a, b):
    """The on-device CDF is ordered and spans the declared interval.

    ``(5.0, 6.0)`` is the case that justifies the survival-function branch.
    The endpoint checks ensure that the branch does not collapse a far-tail
    interval to a constant value.
    """
    with jax.enable_x64():
        lo = a if np.isfinite(a) else -8.0
        hi = b if np.isfinite(b) else 8.0
        z = np.linspace(lo, hi, 201)
        got = np.asarray(_truncnorm_cdf(jnp.asarray(z), a, b))
    assert np.all(np.isfinite(got))
    assert np.all((got >= 0.0) & (got <= 1.0))
    assert np.all(np.diff(got) >= 0.0)
    assert got[0] == pytest.approx(0.0, abs=1e-12)
    assert got[-1] == pytest.approx(1.0, abs=1e-12)
