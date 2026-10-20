"""The Sobol bootstrap uses count-weighted moment forms of the estimators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa._core.bootstrap import draw_rows, resample_keys
from jaxgsa.problem import Problem
from jaxgsa.sobol._bootstrap import _bootstrap_indices
from jaxgsa.sobol._estimators import (
    ESTIMATORS,
    first_total_kernel,
    moment_form,
    requires_second_order_design,
    second_order_kernel,
)
from jaxgsa.sobol._result import SobolResult

N, D = 512, 3


def _data(seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=N)
    B = rng.normal(size=N)
    AB = 0.6 * A[:, None] + 0.5 * rng.normal(size=(N, D)) + 0.3 * B[:, None]
    BA = 0.6 * B[:, None] + 0.5 * rng.normal(size=(N, D)) + 0.3 * A[:, None]
    out = []
    for x in (A, B, AB, BA):
        out.append(jnp.asarray(x, dtype=jnp.result_type(float)))
    A, B, AB, BA = out
    # The bootstrap assumes standardized A and B, as analyze() provides.
    pooled = jnp.concatenate([A, B])
    mu, sd = pooled.mean(), pooled.std()
    return (A - mu) / sd, (AB - mu) / sd, (BA - mu) / sd, (B - mu) / sd


def _moment(estimator, second, counts, A, AB, BA, B):
    features, finalize = moment_form(estimator, second, D)
    F = features(A, AB, B, BA if second else None)
    m = (counts @ F) / N
    cross = (jnp.einsum("rn,nj,nk->rjk", counts, BA, AB) / N) if second else None
    return finalize(m, cross)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("second", [False, True])
def test_moment_form_matches_point_kernel_and_gather(estimator, second):
    if requires_second_order_design(estimator) and not second:
        pytest.skip("needs BA")
    A, AB, BA, B = _data()
    tol = 1e-8 if jax.config.jax_enable_x64 else 2e-4

    def point(a, ab, ba, b):
        if second:
            return second_order_kernel(estimator)(a, ab, ba, b)
        s1, st = first_total_kernel(estimator)(a, ab, b)
        return s1, st, None

    ones = jnp.ones((1, N), dtype=A.dtype)
    got = _moment(estimator, second, ones, A, AB, BA, B)
    want = point(A, AB, BA, B)
    for g, w in zip(got, want, strict=True):
        if w is None:
            assert g is None
        else:
            np.testing.assert_allclose(np.asarray(g[0]), np.asarray(w), rtol=tol, atol=tol)

    keys = resample_keys(jax.random.key(3), 4)
    rows = jax.vmap(lambda k: draw_rows(k, N))(keys)
    counts = jax.vmap(lambda r: jnp.bincount(r, length=N))(rows).astype(A.dtype)
    got = _moment(estimator, second, counts, A, AB, BA, B)
    for r in range(4):
        i = rows[r]
        want = point(A[i], AB[i], BA[i], B[i])
        for g, w in zip(got, want, strict=True):
            if w is not None:
                np.testing.assert_allclose(np.asarray(g[r]), np.asarray(w), rtol=tol, atol=tol)


def test_draws_do_not_depend_on_resample_chunk_width():
    A, AB, BA, B = _data()
    args = (A[None], AB[None], BA[None], B[None], 1, "saltelli-jansen")
    keys = resample_keys(jax.random.key(1), 10)
    whole = _bootstrap_indices(keys, *args)
    parts = _bootstrap_indices(keys, *args, resample_chunk_size=3)
    for w, p in zip(whole, parts, strict=True):
        np.testing.assert_allclose(np.asarray(w), np.asarray(p), rtol=1e-5, atol=1e-6)


def test_inert_input_first_order_replicates_are_exactly_zero():
    problem = Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0), "inert": (0.0, 1.0)})
    s = jaxgsa.sobol.sample(problem, base_n=256, calc_second_order=False, seed=0, verbose=False)
    Y = np.asarray(s.samples[:, 0] + s.samples[:, 1] ** 2, dtype=np.float32)
    r = jaxgsa.sobol.analyze(s, Y, n_bootstrap=20, key=0, keep_replicates=True, verbose=False)
    assert isinstance(r, SobolResult)
    assert r.ci is not None and r.ci.replicates is not None
    np.testing.assert_array_equal(np.asarray(r.ci.replicates["S1"])[..., 2], 0.0)
