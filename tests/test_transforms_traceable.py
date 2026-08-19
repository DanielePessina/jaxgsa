"""``cdf_to_unit_interval`` stays on device.

Tier T2 (permissive library, recorded): the truncated-normal CDF is checked
against ``scipy.stats.truncnorm``, which is the reference this transform used
to call directly.

Every method's pure ``indices()`` core runs its inputs through this transform,
so a single host read here breaks the ``jit``/``vmap``/``jacrev`` contract for
all of them at once. Two separate defects did exactly that, and both are
pinned below.
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

    ``truncated`` failed because the transform called
    ``scipy.stats.truncnorm.cdf(np.asarray(X[:, d]), ...)``, which converts a
    tracer to a host array.

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
def test_truncnorm_cdf_matches_scipy(a, b):
    """Tier T2: the on-device CDF equals ``scipy.stats.truncnorm``.

    ``(5.0, 6.0)`` is the case that justifies the survival-function branch.
    Written as the single form ``(Phi(z) - Phi(a)) / (Phi(b) - Phi(a))`` the
    error there is 3.9e-10, because both CDF values are within rounding of 1
    and the difference is all round-off. The branch taken here holds it at
    1.5e-15.
    """
    truncnorm = pytest.importorskip("scipy.stats").truncnorm
    with jax.enable_x64():
        lo = a if np.isfinite(a) else -8.0
        hi = b if np.isfinite(b) else 8.0
        z = np.linspace(lo, hi, 201)
        got = np.asarray(_truncnorm_cdf(jnp.asarray(z), a, b))
    np.testing.assert_allclose(got, truncnorm.cdf(z, a=a, b=b), atol=1e-12)
