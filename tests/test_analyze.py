import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import truncnorm

import jaxgsa
from jaxgsa.benchmarks import sobol_g
from jaxgsa.benchmarks.ishigami import ANALYTICAL_S1, ANALYTICAL_ST, PROBLEM, evaluate
from jaxgsa.problem import GaussianInputSpec
from jaxgsa.sobol import SobolSamples


@pytest.fixture(scope="module")
def ishigami_bootstrap_result():
    """Run Ishigami analysis with bootstrap CIs once for all tests."""
    sr = jaxgsa.sobol.sample(PROBLEM, n_samples=2**14 * 8, seed=42, verbose=False)
    Y = evaluate(jnp.asarray(sr.samples))
    return jaxgsa.sobol.analyze(sr, Y, n_bootstrap=200, key=jax.random.key(0))


@pytest.fixture(scope="module")
def ishigami_bootstrap_result_gaussian():
    """Run Ishigami analysis with gaussian bootstrap CIs once for all tests."""
    sr = jaxgsa.sobol.sample(PROBLEM, n_samples=2**14 * 8, seed=42, verbose=False)
    Y = evaluate(jnp.asarray(sr.samples))
    return jaxgsa.sobol.analyze(
        sr,
        Y,
        n_bootstrap=200,
        ci_method="gaussian",
        key=jax.random.key(0),
    )


def _assert_bootstrap_ci_contains_point_estimate(result):
    """Point estimates should lie within lower/upper CI endpoint arrays."""
    assert np.all(np.array(result.S1_conf[0]) <= np.array(result.S1))
    assert np.all(np.array(result.S1) <= np.array(result.S1_conf[1]))
    assert np.all(np.array(result.ST_conf[0]) <= np.array(result.ST))
    assert np.all(np.array(result.ST) <= np.array(result.ST_conf[1]))

    S2 = np.array(result.S2)
    S2_lo = np.array(result.S2_conf[0])
    S2_hi = np.array(result.S2_conf[1])
    upper = np.triu_indices_from(S2, k=1)
    lower = (upper[1], upper[0])

    assert np.all(np.isnan(np.diag(S2_lo))), (
        f"S2_conf lower diagonal should be NaN, got {np.diag(S2_lo)}"
    )
    assert np.all(np.isnan(np.diag(S2_hi))), (
        f"S2_conf upper diagonal should be NaN, got {np.diag(S2_hi)}"
    )
    assert np.allclose(S2_lo[upper], S2_lo[lower]), "Lower S2_conf bound should be symmetric"
    assert np.allclose(S2_hi[upper], S2_hi[lower]), "Upper S2_conf bound should be symmetric"
    assert np.all(S2_lo[upper] <= S2[upper])
    assert np.all(S2[upper] <= S2_hi[upper])


def test_bootstrap_ci_contains_point_estimate(ishigami_bootstrap_result):
    """Point estimates should lie within their bootstrap CIs."""
    _assert_bootstrap_ci_contains_point_estimate(ishigami_bootstrap_result)


def test_gaussian_bootstrap_ci_contains_point_estimate(ishigami_bootstrap_result_gaussian):
    """Gaussian mode should still return endpoint arrays around the estimates."""
    _assert_bootstrap_ci_contains_point_estimate(ishigami_bootstrap_result_gaussian)


def test_bootstrap_ci_contains_analytical(ishigami_bootstrap_result):
    """95% bootstrap CIs should contain the known analytical Ishigami values."""
    r = ishigami_bootstrap_result
    S1_lo, S1_hi = np.array(r.S1_conf[0]), np.array(r.S1_conf[1])
    ST_lo, ST_hi = np.array(r.ST_conf[0]), np.array(r.ST_conf[1])
    for i, expected in enumerate(ANALYTICAL_S1):
        assert S1_lo[i] <= expected <= S1_hi[i], (
            f"S1[{i}]: analytical {expected} not in CI [{S1_lo[i]}, {S1_hi[i]}]"
        )
    for i, expected in enumerate(ANALYTICAL_ST):
        assert ST_lo[i] <= expected <= ST_hi[i], (
            f"ST[{i}]: analytical {expected} not in CI [{ST_lo[i]}, {ST_hi[i]}]"
        )


def test_point_estimates_are_affine_invariant():
    """``Y -> a*Y + b`` must not move a Sobol point estimate.

    Tier T0 (closed form): every index is a variance ratio, so the shift
    drops out and the scale cancels. The estimators do not get that for free
    — the first-order and second-order formulas are uncentred products — so
    ``analyze`` standardizes the outputs before they run. This test is the
    guard on that: before it was unconditional, an offset of 1e4 turned
    Ishigami's S1 into ``[6.26, 0.434, 1.71]``.
    """
    sr = jaxgsa.sobol.sample(PROBLEM, n_samples=2**10, seed=19, verbose=False)
    Y = evaluate(jnp.asarray(sr.samples))
    Y_affine = 3.0 * Y + 1e4

    base = jaxgsa.sobol.analyze(sr, Y)
    shifted = jaxgsa.sobol.analyze(sr, Y_affine)

    # The tolerance is float32 cancellation in the standardization itself:
    # subtracting a mean of 1e4 from a float32 array costs about four digits,
    # which lands near 1e-5 on an index. The bias this replaces was 1e0.
    np.testing.assert_allclose(np.asarray(base.S1), np.asarray(shifted.S1), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.asarray(base.ST), np.asarray(shifted.ST), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(base.S2),
        np.asarray(shifted.S2),
        rtol=1e-3,
        atol=1e-4,
        equal_nan=True,
    )


def test_repeated_bootstrap_calls_identical():
    """Repeated identical bootstrap calls should preserve point estimates and CI shapes."""
    sr = jaxgsa.sobol.sample(PROBLEM, n_samples=2**10, seed=11)
    Y = evaluate(jnp.asarray(sr.samples))
    key = jax.random.key(123)

    first = jaxgsa.sobol.analyze(sr, Y, n_bootstrap=20, key=key)
    second = jaxgsa.sobol.analyze(sr, Y, n_bootstrap=20, key=key)

    np.testing.assert_allclose(np.asarray(first.S1), np.asarray(second.S1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(first.ST), np.asarray(second.ST), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(first.S2),
        np.asarray(second.S2),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(first.S1_conf),
        np.asarray(second.S1_conf),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(first.ST_conf),
        np.asarray(second.ST_conf),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(first.S2_conf),
        np.asarray(second.S2_conf),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_mixed_uniform_and_gaussian_linear_model_matches_analytical_indices():
    problem = jaxgsa.Problem.from_dict(
        {
            "uniform": (0.0, 2.0),
            "gaussian": GaussianInputSpec(dist="gaussian", mean=1.0, variance=2.25),
            "truncated": GaussianInputSpec(
                dist="gaussian",
                mean=0.5,
                variance=1.44,
                low=-0.5,
                high=1.0,
            ),
        },
        output_names=("response",),
    )
    sr = jaxgsa.sobol.sample(
        problem,
        n_samples=8192,
        calc_second_order=False,
        seed=101,
        verbose=False,
    )

    coeffs = jnp.array([1.5, -0.75, 2.0])
    X = jnp.asarray(sr.samples)
    Y = (X @ coeffs)[:, None, None]
    result = jaxgsa.sobol.analyze(sr, Y)

    std = np.sqrt(1.44)
    a = (-0.5 - 0.5) / std
    b = (1.0 - 0.5) / std
    variances = np.array(
        [
            (2.0 - 0.0) ** 2 / 12.0,
            2.25,
            truncnorm.var(a, b, loc=0.5, scale=std),
        ]
    )
    coeff_sq = np.square(np.array([1.5, -0.75, 2.0]))
    expected = coeff_sq * variances
    expected = expected / expected.sum()

    np.testing.assert_allclose(np.asarray(result.S1[0, 0]), expected, atol=0.03, rtol=0.03)
    np.testing.assert_allclose(np.asarray(result.ST[0, 0]), expected, atol=0.03, rtol=0.03)
    assert result.S2 is None


def test_bootstrap_is_affine_invariant():
    """The bootstrap path standardizes too, so its endpoints move with nothing.

    Tier T0 (closed form). The standardization happens once, before the
    resampling, so every replicate reads the same standardized array and an
    interval stays centred on the point estimate.
    """
    sr = jaxgsa.sobol.sample(PROBLEM, n_samples=2**10, seed=23, verbose=False)
    Y = evaluate(jnp.asarray(sr.samples))
    Y_affine = 3.0 * Y + 1e4
    key = jax.random.key(321)

    base = jaxgsa.sobol.analyze(sr, Y, n_bootstrap=20, key=key)
    shifted = jaxgsa.sobol.analyze(sr, Y_affine, n_bootstrap=20, key=key)

    np.testing.assert_allclose(np.asarray(base.S1), np.asarray(shifted.S1), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.asarray(base.ST), np.asarray(shifted.ST), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(base.S2),
        np.asarray(shifted.S2),
        rtol=1e-3,
        atol=1e-4,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(base.S1_conf),
        np.asarray(shifted.S1_conf),
        rtol=1e-3,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(base.ST_conf),
        np.asarray(shifted.ST_conf),
        rtol=1e-3,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(base.S2_conf),
        np.asarray(shifted.S2_conf),
        rtol=1e-3,
        atol=1e-4,
        equal_nan=True,
    )


def _legacy_sampling_result(sr: SobolSamples) -> SobolSamples:
    n_expanded = sr.n_expanded
    expanded_samples = sr.samples[sr.expanded_to_unique]
    return SobolSamples(
        samples=expanded_samples,
        n_expanded=n_expanded,
        expanded_to_unique=np.arange(n_expanded, dtype=np.int64),
        base_n=sr.base_n,
        n_params=sr.n_params,
        calc_second_order=sr.calc_second_order,
        problem=sr.problem,
    )


def test_unique_analysis_matches_expanded_layout():
    sr = jaxgsa.sobol.sample(PROBLEM, n_samples=1024, seed=7, verbose=False)
    Y_unique = evaluate(jnp.asarray(sr.samples))
    result_unique = jaxgsa.sobol.analyze(sr, Y_unique)

    legacy_sr = _legacy_sampling_result(sr)
    Y_expanded = Y_unique[sr.expanded_to_unique]
    result_expanded = jaxgsa.sobol.analyze(legacy_sr, Y_expanded)

    assert np.allclose(np.asarray(result_unique.S1), np.asarray(result_expanded.S1))
    assert np.allclose(np.asarray(result_unique.ST), np.asarray(result_expanded.ST))
    assert np.allclose(
        np.asarray(result_unique.S2),
        np.asarray(result_expanded.S2),
        equal_nan=True,
    )


def test_unique_bootstrap_matches_expanded_layout():
    sr = jaxgsa.sobol.sample(PROBLEM, n_samples=1024, seed=7, verbose=False)
    Y_unique = evaluate(jnp.asarray(sr.samples))
    key = jax.random.key(123)
    result_unique = jaxgsa.sobol.analyze(sr, Y_unique, n_bootstrap=50, key=key)

    legacy_sr = _legacy_sampling_result(sr)
    Y_expanded = Y_unique[sr.expanded_to_unique]
    result_expanded = jaxgsa.sobol.analyze(legacy_sr, Y_expanded, n_bootstrap=50, key=key)

    assert np.allclose(np.asarray(result_unique.S1), np.asarray(result_expanded.S1))
    assert np.allclose(np.asarray(result_unique.ST), np.asarray(result_expanded.ST))
    assert np.allclose(
        np.asarray(result_unique.S2),
        np.asarray(result_expanded.S2),
        equal_nan=True,
    )
    assert np.allclose(np.asarray(result_unique.S1_conf), np.asarray(result_expanded.S1_conf))
    assert np.allclose(np.asarray(result_unique.ST_conf), np.asarray(result_expanded.ST_conf))
    assert np.allclose(
        np.asarray(result_unique.S2_conf),
        np.asarray(result_expanded.S2_conf),
        equal_nan=True,
    )


def test_gaussian_and_quantile_bootstrap_endpoints_differ():
    """The ci_method switch must exercise distinct endpoint calculations.

    This is the only observable difference between the two endpoint rules.
    ``CIInfo.method`` records the name the caller asked for, so a gaussian
    branch that quietly returned percentile endpoints would still record
    ``"gaussian"`` and pass every other test in the suite.
    """
    sr = jaxgsa.sobol.sample(PROBLEM, n_samples=2**10, seed=61, verbose=False)
    Y = evaluate(jnp.asarray(sr.samples))
    key = jax.random.key(222)

    quantile = jaxgsa.sobol.analyze(sr, Y, n_bootstrap=50, ci_method="quantile", key=key)
    gaussian = jaxgsa.sobol.analyze(sr, Y, n_bootstrap=50, ci_method="gaussian", key=key)

    assert not np.allclose(np.asarray(gaussian.S1_conf), np.asarray(quantile.S1_conf))


def _assert_sobol_fields_match(chunked, full, fields):
    """Compare the named fields of two Sobol results element by element.

    Args:
        chunked: The result computed with a small ``slice_chunk_size``.
        full: The result computed with the default ``slice_chunk_size``.
        fields: Names of the ``SobolResult`` fields to compare.
    """
    for field in fields:
        expected = getattr(full, field)
        actual = getattr(chunked, field)
        assert (expected is None) == (actual is None), field
        if expected is None:
            continue
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=1e-5,
            atol=1e-7,
            err_msg=field,
        )


def test_slice_chunk_size_invariance():
    """Tier T4 (internal consistency): chunking changes no index, on both paths.

    ``sobol.analyze`` dispatches on ``n_bootstrap``, and ``slice_chunk_size``
    means a different thing on each side, so the test runs both halves.

    * ``n_bootstrap=32`` takes ``_analyze_bootstrap``. There
      ``slice_chunk_size`` caps the *output slices* per device call, and the
      resamples of a slice ride along inside that call. So it chunks the
      point estimates and the draws alike, and all six fields are sensitive
      to this half.
    * ``n_bootstrap=0`` takes ``_analyze_no_bootstrap``. That is the path
      whose loop chunks the ``T*K`` output columns and reassembles them with
      ``jnp.concatenate`` plus ``_normalize_s2_matrix``. This half is the only
      coverage of that loop in the suite, so it uses three outputs and
      compares ``S1``, ``ST`` and ``S2``.

    Both halves share one design, one key and one seed, so every compared
    field must match to floating-point noise.
    """
    sr = jaxgsa.sobol.sample(PROBLEM, n_samples=2**10, seed=7, verbose=False)
    Y = evaluate(jnp.asarray(sr.samples))
    Y_multi = jnp.stack([Y, 2.0 * Y, jnp.sin(Y)], axis=-1)

    # Bootstrap path: the *_conf fields are what slice_chunk_size touches.
    full = jaxgsa.sobol.analyze(sr, Y_multi, n_bootstrap=32, key=jax.random.key(3))
    chunked = jaxgsa.sobol.analyze(
        sr, Y_multi, n_bootstrap=32, key=jax.random.key(3), slice_chunk_size=1
    )
    _assert_sobol_fields_match(chunked, full, ("S1", "ST", "S2", "S1_conf", "ST_conf", "S2_conf"))

    # Plain path: this is the half that exercises the output-column loop.
    plain_full = jaxgsa.sobol.analyze(sr, Y_multi, n_bootstrap=0)
    plain_chunked = jaxgsa.sobol.analyze(sr, Y_multi, n_bootstrap=0, slice_chunk_size=1)
    assert plain_full.S1_conf is None
    _assert_sobol_fields_match(plain_chunked, plain_full, ("S1", "ST", "S2"))


def test_bootstrap_point_estimates_equal_the_plain_path_exactly():
    """Tier T4 (internal consistency): asking for an interval moves no index.

    ``S1``, ``ST`` and ``S2`` must not depend on whether a bootstrap was
    requested. Both paths now run the same chunked ``vmap`` over output
    slices, so the agreement is bit-for-bit and not merely to a tolerance.

    This used not to hold. The bootstrap path called the scalar fused kernel
    once per slice while the plain path vmapped the same kernel over a chunk
    of slices, and XLA schedules the two reductions differently: the two
    ``S2`` matrices for one design disagreed in the last bits. An interval
    centred on a number the plain analysis does not report is the defect this
    pins shut.

    The model and the shape here are the ones that expose it. The g-function
    with two outputs is the ``sobol_g_multi`` case from
    ``scripts/baseline_dump.py``, and against the previous code this test
    fails with ``S2`` differing by 1.4e-7. A wider output stack does not
    discriminate -- both paths already agreed there -- so the narrow shape is
    the point and not an arbitrary choice.
    """
    sr = jaxgsa.sobol.sample(sobol_g.PROBLEM, n_samples=2**9, seed=11, verbose=False)
    g = sobol_g.evaluate(jnp.asarray(sr.samples))
    Y_multi = jnp.stack([g, 2.0 * g + 0.37], axis=-1)

    boot = jaxgsa.sobol.analyze(sr, Y_multi, n_bootstrap=8, key=jax.random.key(5))
    plain = jaxgsa.sobol.analyze(sr, Y_multi, n_bootstrap=0)

    for field in ("S1", "ST", "S2"):
        np.testing.assert_array_equal(
            np.asarray(getattr(boot, field)),
            np.asarray(getattr(plain, field)),
            err_msg=field,
        )


def test_the_bootstrap_batches_slices_instead_of_looping_over_them(monkeypatch):
    """Tier T4 (internal consistency): one device call carries many slices.

    This test patches an internal and records shapes, which the suite avoids
    elsewhere. Keep it anyway: batching is *defined* to return the same
    numbers, so it has no other observable effect, and
    :func:`test_slice_chunk_size_invariance` passes just as well against a
    per-slice Python loop. This is the only place the batching is asserted.

    Three things are pinned. The resampler must see a chunk of slices, not
    one slice at a time. Every call must carry the whole ``(R, N)`` index
    array, because the resamples belong inside the batch rather than in a
    loop around it. And every call must be the *same* width: six slices at a
    chunk of four is two calls of four, not ``[4, 2]``, because the short
    trailing chunk is padded back and the answer sliced out. A jitted kernel
    is traced per input shape, so a ragged tail would compile it twice.
    """
    from jaxgsa.sobol import _bootstrap as sobol_bootstrap

    sr = jaxgsa.sobol.sample(PROBLEM, n_samples=2**8, seed=13, verbose=False)
    Y = evaluate(jnp.asarray(sr.samples))
    Y_multi = jnp.stack([(i + 1.0) * Y for i in range(6)], axis=-1).reshape(-1, 3, 2)

    widths: list[int] = []
    n_bootstrap: list[int] = []
    real_get = sobol_bootstrap._resample_kernel

    def recording_get(estimator: str, calc_second_order: bool, single: bool):
        kernel = real_get(estimator, calc_second_order, single)

        def recorder(idx, A, AB, BA, B):
            widths.append(int(A.shape[0]))
            n_bootstrap.append(int(idx.shape[0]))
            return kernel(idx, A, AB, BA, B)

        return recorder

    monkeypatch.setattr(sobol_bootstrap, "_resample_kernel", recording_get)
    jaxgsa.sobol.analyze(sr, Y_multi, n_bootstrap=8, key=jax.random.key(2), slice_chunk_size=4)

    assert widths == [4, 4], f"expected two equal-width chunks, the tail padded back, got {widths}"
    assert n_bootstrap == [8, 8], "every call must carry the whole resample batch"
