"""Tests for PAWN sensitivity analysis."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks import ishigami
from jaxgsa.pawn import analyze, indices
from jaxgsa.problem import InputSpecValue, Problem
from jaxgsa.sampling import monte_carlo


@pytest.fixture(scope="module")
def ishigami_data():
    """Generate Ishigami test data."""
    N = 5000
    X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=N, seed=42))
    Y = ishigami.evaluate(X)
    return X, Y


class TestPAWNBasic:
    def test_values_in_unit_interval(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y)
        pawn = np.asarray(result.pawn)
        assert np.all(pawn >= 0.0)
        assert np.all(pawn <= 1.0)

    def test_x1_x2_more_important_than_x3(self, ishigami_data):
        """x1 and x2 should have higher PAWN than x3 (weak first-order)."""
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y)
        pawn = np.asarray(result.pawn)
        assert pawn[0] > pawn[2], "x1 should be more important than x3"
        assert pawn[1] > pawn[2], "x2 should be more important than x3"

    def test_slice_chunk_size_invariance(self, ishigami_data):
        """Tier T4 (internal consistency): chunking changes no index.

        ``slice_chunk_size`` splits the flattened ``T*K`` output columns into
        separate kernel calls. Every column is independent of every other, so
        the chunked result must equal the unchunked one exactly, not merely
        to a tolerance. The bootstrap intervals are compared too: they run
        through the same loop, once per resample.

        The test is not vacuous. The output has ``T*K == 6`` columns and the
        chunk size is 4, so the loop runs twice and the second chunk is a
        short one. Both asserts below state that, and they fail if a future
        edit shrinks the output or raises the chunk size past it.
        """
        X, Y = ishigami_data
        X = X[:500]
        Y = Y[:500]
        # (N, T, K) with T = 3, K = 2, so six flattened output columns.
        Y_3d = jnp.stack(
            [
                jnp.stack([Y, 2.0 * Y], axis=-1),
                jnp.stack([jnp.sin(Y), Y**2], axis=-1),
                jnp.stack([-Y, jnp.cos(Y)], axis=-1),
            ],
            axis=1,
        )
        total = Y_3d.shape[1] * Y_3d.shape[2]
        chunk = 4
        assert total > chunk, "chunk must be smaller than T*K or nothing is split"
        assert total % chunk != 0, "an uneven split exercises the short trailing chunk"

        full = analyze(
            ishigami.PROBLEM, X, Y_3d, n_bootstrap=8, conf_level=0.9, key=jax.random.key(3)
        )
        chunked = analyze(
            ishigami.PROBLEM,
            X,
            Y_3d,
            n_bootstrap=8,
            conf_level=0.9,
            key=jax.random.key(3),
            slice_chunk_size=chunk,
        )

        assert full.pawn.shape == (3, 2, 3)
        assert full.pawn_conf is not None
        assert chunked.pawn_conf is not None
        np.testing.assert_array_equal(np.asarray(chunked.pawn), np.asarray(full.pawn))
        np.testing.assert_array_equal(np.asarray(chunked.pawn_conf), np.asarray(full.pawn_conf))
        assert chunked.problem is full.problem

    def test_slice_chunk_size_splits_the_columns(self, ishigami_data, monkeypatch):
        """Tier T4 (internal consistency): the chunk loop really splits.

        This test patches an internal and counts calls, which this suite
        avoids elsewhere. Keep it anyway. Chunking is *defined* to return
        the identical numbers, so the loop has no other observable effect:
        :meth:`test_slice_chunk_size_invariance` above passes unchanged if
        ``slice_chunk_size`` is dropped on the floor, because an ignored
        keyword trivially gives an identical result. Without the check
        below, the fix that chunks the outer vmap over output columns is
        indistinguishable from a no-op. This is not a mirror of the
        implementation; it is the only place the memory bound is asserted.

        The recorded per-call column widths are the contract, not merely
        the number of calls: ``T*K == 6`` columns at a chunk size of 4 must
        arrive as two calls, and both must be 4 columns wide. A loop that
        passed all six columns in one call and sliced afterwards would give
        the same indices and fail here.

        Both calls are 4 wide because the short trailing chunk is padded
        back to the chunk width and the answer sliced out again. The kernel
        is jitted and traced per input shape, so a 2-column tail would
        compile it a second time. The padded columns are separate ``vmap``
        lanes, so the indices are unchanged, which is why the width and not
        the values is what this test can see.
        """
        from jaxgsa.pawn import _analyze as pawn_analyze

        X, Y = ishigami_data
        X = X[:500]
        Y = Y[:500]
        # (N, T, K) with T = 3, K = 2, so six flattened output columns.
        Y_3d = jnp.stack(
            [
                jnp.stack([Y, 2.0 * Y], axis=-1),
                jnp.stack([jnp.sin(Y), Y**2], axis=-1),
                jnp.stack([-Y, jnp.cos(Y)], axis=-1),
            ],
            axis=1,
        )
        assert Y_3d.shape[1] * Y_3d.shape[2] == 6

        widths: list[int] = []
        real_get = pawn_analyze._get_pawn_ks

        def recording_get(n_bins: int):
            kernel = real_get(n_bins)

            def recorder(bin_idx, Y_cols):
                widths.append(int(Y_cols.shape[1]))
                return kernel(bin_idx, Y_cols)

            return recorder

        monkeypatch.setattr(pawn_analyze, "_get_pawn_ks", recording_get)
        analyze(ishigami.PROBLEM, X, Y_3d, slice_chunk_size=4, n_bootstrap=0)

        assert widths == [4, 4], f"expected two calls of the full chunk width, got {widths}"

    def test_slice_chunk_size_must_be_positive(self, ishigami_data):
        """Tier T4 (internal consistency): an unusable chunk size is refused."""
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="slice_chunk_size must be >= 1"):
            analyze(ishigami.PROBLEM, X[:200], Y[:200], slice_chunk_size=0)


class TestPAWNSALibComparison:
    def test_matches_salib_median(self, ishigami_data):
        """Compare against SALib's PAWN implementation."""
        from SALib.analyze import pawn as salib_pawn

        X_np = np.asarray(ishigami_data[0])
        Y_np = np.asarray(ishigami_data[1])

        salib_problem = {
            "num_vars": 3,
            "names": ["x1", "x2", "x3"],
            "bounds": [[-np.pi, np.pi]] * 3,
        }
        salib_result = salib_pawn.analyze(salib_problem, X_np, Y_np, S=10, seed=0)
        salib_median = salib_result["median"]

        jaxgsa_result = analyze(
            ishigami.PROBLEM,
            ishigami_data[0],
            ishigami_data[1],
            n_bins=10,
            statistic="median",
        )
        jaxgsa_pawn = np.asarray(jaxgsa_result.pawn)

        np.testing.assert_allclose(jaxgsa_pawn, salib_median, atol=0.01)


class TestPAWNStatistics:
    def test_max_geq_median(self, ishigami_data):
        X, Y = ishigami_data
        r_median = analyze(ishigami.PROBLEM, X, Y, statistic="median")
        r_max = analyze(ishigami.PROBLEM, X, Y, statistic="max")
        assert np.all(np.asarray(r_max.pawn) >= np.asarray(r_median.pawn) - 1e-6)


class TestPAWNBootstrap:
    def test_bootstrap_lower_leq_upper(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(
            ishigami.PROBLEM, X, Y, n_bootstrap=20, conf_level=0.95, key=jax.random.key(0)
        )
        assert result.pawn_conf is not None
        lower = np.asarray(result.pawn_conf[0])
        upper = np.asarray(result.pawn_conf[1])
        assert np.all(lower <= upper + 1e-6)

    def test_bootstrap_without_a_key_raises(self, ishigami_data):
        """Tier T4. A bootstrap needs a key, and an int seed is not one."""
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="key is required"):
            analyze(ishigami.PROBLEM, X, Y, n_bootstrap=4)

    def test_same_key_same_interval(self, ishigami_data):
        """Tier T4. Determinism is a property of the key, not of a seed."""
        X, Y = ishigami_data
        r1 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=8, key=jax.random.key(11))
        r2 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=8, key=jax.random.key(11))
        assert r1.pawn_conf is not None and r2.pawn_conf is not None
        np.testing.assert_array_equal(np.asarray(r1.pawn_conf), np.asarray(r2.pawn_conf))

    def test_different_keys_differ(self, ishigami_data):
        """Tier T4. A different key draws a different resample."""
        X, Y = ishigami_data
        r1 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=8, key=jax.random.key(11))
        r2 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=8, key=jax.random.key(12))
        assert r1.pawn_conf is not None and r2.pawn_conf is not None
        assert not np.array_equal(np.asarray(r1.pawn_conf), np.asarray(r2.pawn_conf))
        # The point estimate reads no key at all.
        np.testing.assert_array_equal(np.asarray(r1.pawn), np.asarray(r2.pawn))

    def test_gaussian_ci_is_centred_on_the_estimate(self, ishigami_data):
        """Tier T0 (closed form): the Gaussian interval is symmetric.

        ``estimate +/- z * sd`` puts the point estimate exactly at the
        midpoint of the two endpoints, which the percentile interval does
        not do. That is what tells the two branches apart.
        """
        X, Y = ishigami_data
        result = analyze(
            ishigami.PROBLEM, X, Y, n_bootstrap=20, ci_method="gaussian", key=jax.random.key(0)
        )
        assert result.ci is not None
        assert result.ci.method == "gaussian"
        assert result.pawn_conf is not None
        midpoint = 0.5 * (np.asarray(result.pawn_conf[0]) + np.asarray(result.pawn_conf[1]))
        np.testing.assert_allclose(midpoint, np.asarray(result.pawn), atol=1e-6)

    def test_unknown_ci_method_rejected(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="ci_method"):
            analyze(
                ishigami.PROBLEM,
                X,
                Y,
                n_bootstrap=4,
                ci_method=cast(Any, "bca"),
                key=jax.random.key(0),
            )


class TestPAWNTiedOutputs:
    """The KS statistic must be tie-aware, matching ``scipy.stats.ks_2samp``.

    For discrete, quantized, or otherwise tied outputs, equal output
    values must be treated as a single value (not strictly ordered), so
    the per-bin KS equals the two-sample KS of ``ks_2samp``.
    """

    def test_discrete_output_matches_ks_2samp(self):
        from scipy.stats import ks_2samp

        rng = np.random.default_rng(0)
        D, N, n_bins = 3, 400, 8
        problem = Problem(
            names=("a", "b", "c"),
            bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )
        X = rng.uniform(0.0, 1.0, size=(N, D))
        # Heavily quantized output with many tied values.
        y = np.round((2 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2]) * 4) / 4

        got = np.asarray(analyze(problem, jnp.asarray(X), jnp.asarray(y), n_bins=n_bins).pawn)

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ref = np.full(D, np.nan)
        for d in range(D):
            ks = []
            for b in range(n_bins):
                lo, hi = edges[b], edges[b + 1]
                if b == n_bins - 1:
                    mask = (X[:, d] >= lo) & (X[:, d] <= hi)
                else:
                    mask = (X[:, d] >= lo) & (X[:, d] < hi)
                y_bin = y[mask]
                if y_bin.shape[0] >= 2:
                    ks.append(ks_2samp(y, y_bin).statistic)
            ref[d] = np.median(ks)

        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_constant_output_zero_ks(self):
        """All-equal Y => conditional == unconditional => KS is 0."""
        problem = Problem(names=("a",), bounds=((0.0, 1.0),))
        X = np.linspace(0.0, 1.0, 100).reshape(-1, 1)
        y = np.full(100, 3.0)
        result = analyze(problem, jnp.asarray(X), jnp.asarray(y), n_bins=5)
        np.testing.assert_allclose(np.asarray(result.pawn), 0.0, atol=1e-6)


class TestPAWNOutOfBounds:
    def test_out_of_range_gets_sentinel(self):
        """Samples outside [0, 1] are excluded (sentinel -1), not clamped."""
        from jaxgsa.pawn._analyze import _equal_width_bins

        X = jnp.asarray([[-0.5], [0.0], [0.4], [1.0], [1.5], [jnp.nan]])
        idx = np.asarray(_equal_width_bins(X, 4)).ravel()
        assert idx[0] == -1  # below 0 -> excluded
        assert idx[4] == -1  # above 1 -> excluded
        assert idx[1] == 0  # exactly 0.0 -> first bin
        assert idx[3] == 3  # exactly 1.0 -> last bin
        assert idx[5] == -1  # NaN -> excluded, not folded into a bin


class TestPAWNEmptyBinWarning:
    def test_warns_once_not_per_bootstrap(self):
        """The empty-bin warning fires once, not once per bootstrap resample."""
        import warnings as _warnings

        problem = Problem(names=("a",), bounds=((0.0, 1.0),))
        # N < n_bins with spread inputs => every bin has < 2 samples.
        X = np.linspace(0.02, 0.98, 8).reshape(-1, 1)
        y = np.arange(8, dtype=float)
        with _warnings.catch_warnings(record=True) as rec:
            _warnings.simplefilter("always")
            analyze(
                problem,
                jnp.asarray(X),
                jnp.asarray(y),
                n_bins=20,
                n_bootstrap=5,
                key=jax.random.key(0),
            )
        msgs = [r for r in rec if "all bins empty" in str(r.message)]
        assert len(msgs) == 1

    def test_warns_once_across_a_chunk_boundary(self):
        """Tier T4 (internal consistency): the empty-bin flag reduces over chunks.

        Whether every bin of a parameter is empty depends on the binning
        only, so the per-chunk flags must be combined with an AND across
        chunks and reported once per parameter. The other empty-bin test
        uses a scalar output, where ``T*K == 1`` gives a single chunk and
        the cross-chunk reduce never runs. Here the output has six
        flattened columns at a chunk size of 4, so the flags come from two
        chunks of different widths.

        Parameter ``a`` spreads over 20 bins with one sample each, so all
        of its bins are empty. Parameter ``b`` is constant, so all eight
        samples land in one bin and it is never empty. Exactly one warning
        must name parameter 0, which fails both if the reduce drops the
        per-parameter granularity and if it warns once per chunk.
        """
        import warnings as _warnings

        problem = Problem(names=("a", "b"), bounds=((0.0, 1.0), (0.0, 1.0)))
        X = np.column_stack([np.linspace(0.02, 0.98, 8), np.full(8, 0.5)])
        y = np.arange(8, dtype=float)
        # (N, T, K) = (8, 3, 2), so six flattened columns and chunks of 4 + 2.
        Y_3d = jnp.stack(
            [
                jnp.stack([y, 2.0 * y], axis=-1),
                jnp.stack([np.sin(y), y**2], axis=-1),
                jnp.stack([-y, np.cos(y)], axis=-1),
            ],
            axis=1,
        )
        assert Y_3d.shape == (8, 3, 2)

        with _warnings.catch_warnings(record=True) as rec:
            _warnings.simplefilter("always")
            analyze(problem, jnp.asarray(X), Y_3d, n_bins=20, slice_chunk_size=4)
        msgs = [str(r.message) for r in rec if "all bins empty" in str(r.message)]
        assert len(msgs) == 1
        assert "parameter 0" in msgs[0]


def _invalid_sample(n: int = 200, seed: int = 0):
    """Build a clean two-parameter PAWN sample for the on_invalid tests."""
    problem = Problem(names=("a", "b"), bounds=((0.0, 1.0), (0.0, 1.0)))
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 2))
    Y = X[:, 0] ** 2 + 0.5 * X[:, 1]
    return problem, jnp.asarray(X), jnp.asarray(Y)


class TestPAWNInvalidPolicy:
    """T4 (behaviour): jaxgsa.pawn.analyze honours the shared on_invalid policy."""

    def test_raise_is_the_default_and_names_the_rows(self):
        """T4: a non-finite Y row refuses the analysis and says which row it is.

        The row index is the whole point of the message: it names the model
        evaluation the user has to investigate.
        """
        problem, X, Y = _invalid_sample()
        Y = Y.at[7].set(jnp.nan)
        with pytest.raises(ValueError) as exc:
            analyze(problem, X, Y)
        message = str(exc.value)
        assert "jaxgsa.pawn.analyze" in message
        assert "1 of 200 rows" in message
        assert "[7]" in message

    def test_propagate_warns_and_keeps_every_row(self):
        """T4: 'propagate' computes anyway, loudly, and drops nothing.

        The point of the policy is that the user is told and that the bad row
        still reaches the estimator. PAWN reads the output through an ECDF,
        so a NaN there shifts the index rather than turning it into a NaN;
        the test therefore proves the row was kept by comparing against the
        result of removing it, which must differ.
        """
        problem, X, Y = _invalid_sample()
        Y = Y.at[7].set(jnp.nan)
        with pytest.warns(JaxgsaWarning, match="reaches the indices"):
            result = analyze(problem, X, Y, on_invalid="propagate")
        assert result.invalid.policy == "propagate"
        assert result.invalid.n_invalid == 1
        assert result.invalid.unit_indices == (7,)

        keep = np.ones(200, dtype=bool)
        keep[7] = False
        without = analyze(problem, X[keep], Y[keep])
        assert not np.array_equal(np.asarray(result.pawn), np.asarray(without.pawn))

    def test_drop_removes_the_row_and_returns_finite_indices(self):
        """T4: 'drop' analyzes the remainder and the indices come back usable."""
        problem, X, Y = _invalid_sample()
        Y = Y.at[7].set(jnp.nan)
        with pytest.warns(JaxgsaWarning, match="dropped 1 of 200 rows"):
            result = analyze(problem, X, Y, on_invalid="drop")
        assert result.invalid.n_kept == 199
        assert result.invalid.unit_indices == (7,)
        assert np.all(np.isfinite(np.asarray(result.pawn)))

    def test_a_non_finite_x_is_no_longer_swallowed_by_the_bin_sentinel(self):
        """T4: the silent -1 path is unreachable from analyze().

        ``_equal_width_bins`` still maps a non-finite value to the
        out-of-range sentinel, and out-of-range inputs are a separate,
        legitimate concern. What changed is that ``analyze`` never lets a
        non-finite value get that far: every policy reports it first, so no
        sample leaves the conditional sets in silence. The old behaviour was
        to return an index with no warning and a smaller effective sample.
        """
        import warnings as _warnings

        problem, X, Y = _invalid_sample()
        X = X.at[5, 0].set(jnp.nan)

        # The default refuses outright.
        with pytest.raises(ValueError, match="non-finite"):
            analyze(problem, X, Y)

        # Every other policy at least says something.
        for policy in ("propagate", "drop"):
            with _warnings.catch_warnings(record=True) as rec:
                _warnings.simplefilter("always")
                result = analyze(problem, X, Y, on_invalid=policy)
            assert any(isinstance(r.message, JaxgsaWarning) for r in rec), policy
            assert result.invalid.unit_indices == (5,), policy


class TestPureCore:
    """The transformable core ``pawn.indices``.

    Tier T4 throughout (internal consistency and transformability). The KS
    statistic itself is checked against ``scipy.stats.ks_2samp`` elsewhere in
    this file, and ``indices`` runs the same kernel ``analyze`` runs.
    """

    def test_matches_analyze_scalar(self, ishigami_data):
        """T4: ``indices`` returns exactly what ``analyze`` reports."""
        X, Y = ishigami_data

        result = analyze(ishigami.PROBLEM, X, Y, n_bins=8)
        (pawn,) = indices(ishigami.PROBLEM, X, Y, n_bins=8)

        np.testing.assert_array_equal(np.asarray(pawn), np.asarray(result.pawn))

    @pytest.mark.parametrize("statistic", ["median", "max", "mean"])
    def test_matches_analyze_for_every_statistic(self, ishigami_data, statistic):
        """T4: the aggregation choice is a Python branch, shared by both paths."""
        X, Y = ishigami_data
        kwargs: dict[str, Any] = {"n_bins": 8, "statistic": statistic}

        result = analyze(ishigami.PROBLEM, X, Y, **kwargs)
        (pawn,) = indices(ishigami.PROBLEM, X, Y, **kwargs)

        np.testing.assert_array_equal(np.asarray(pawn), np.asarray(result.pawn))

    def test_matches_analyze_time_series(self, ishigami_data):
        """T4: the chunked ``(N, T, K)`` path agrees with ``analyze``."""
        X, base = ishigami_data
        Y = jnp.stack(
            [
                jnp.stack([base, 2.0 * base + X[:, 0]], axis=-1),
                jnp.stack([base**2, X[:, 1]], axis=-1),
            ],
            axis=1,
        )
        assert Y.shape == (X.shape[0], 2, 2)

        result = analyze(ishigami.PROBLEM, X, Y, n_bins=8)
        (pawn,) = indices(ishigami.PROBLEM, X, Y, n_bins=8)

        assert pawn.shape == (2, 2, 3)
        np.testing.assert_array_equal(np.asarray(pawn), np.asarray(result.pawn))

    def test_matches_analyze_with_a_categorical_parameter(self):
        """T4: the traced level-code path bins a categorical column as ``analyze`` does."""
        problem = Problem.from_dict(
            {
                "a": {"dist": "categorical", "probs": [0.4, 0.6]},
                "b": (0.0, 1.0),
            }
        )
        X = jnp.asarray(monte_carlo(problem, n=1000, seed=4))
        Y = 2.0 * X[:, 0] + X[:, 1]

        result = analyze(problem, X, Y, n_bins=5)
        (pawn,) = indices(problem, X, Y, n_bins=5)

        np.testing.assert_array_equal(np.asarray(pawn), np.asarray(result.pawn))

    def test_returns_a_bare_tuple_of_one_array(self, ishigami_data):
        """T4: the core hands back arrays, with no result object attached."""
        X, Y = ishigami_data

        out = indices(ishigami.PROBLEM, X, Y, n_bins=8)

        assert isinstance(out, tuple)
        assert len(out) == 1
        assert isinstance(out[0], jax.Array)

    def test_is_jittable(self, ishigami_data):
        """T4: ``jit`` traces ``indices`` and returns the eager values."""
        X, Y = ishigami_data

        jitted = jax.jit(lambda outputs: indices(ishigami.PROBLEM, X, outputs, n_bins=8))
        (pawn_jit,) = jitted(Y)
        (pawn,) = indices(ishigami.PROBLEM, X, Y, n_bins=8)

        np.testing.assert_allclose(np.asarray(pawn_jit), np.asarray(pawn), rtol=1e-6)

    def test_is_vmappable(self, ishigami_data):
        """T4: ``vmap`` maps ``indices`` over a batch of output vectors."""
        X, base = ishigami_data
        batch = jnp.stack([base, X[:, 0], X[:, 1] ** 2])

        (pawn_batch,) = jax.vmap(lambda outputs: indices(ishigami.PROBLEM, X, outputs, n_bins=8))(
            batch
        )

        assert pawn_batch.shape == (3, 3)
        for row, outputs in enumerate(batch):
            (pawn,) = indices(ishigami.PROBLEM, X, outputs, n_bins=8)
            np.testing.assert_allclose(
                np.asarray(pawn_batch[row]), np.asarray(pawn), rtol=1e-5, atol=1e-6
            )

    def test_jit_of_jacrev_compiles_and_the_derivative_is_zero(self, ishigami_data):
        """T4: ``jit(jacrev(...))`` compiles, and the answer is a structural zero.

        A PAWN index reads the *ordering* of the outputs and the bin an input
        falls in, never the values themselves. Both are piecewise constant, so
        the derivative is exactly 0 almost everywhere. This test therefore
        checks two different things: that the core traces under reverse-mode
        differentiation at all, and that what comes back is the honest zero of
        a rank statistic rather than a NaN.
        """
        X, Y = ishigami_data

        def total(scale):
            return indices(ishigami.PROBLEM, X, Y * scale, n_bins=8)[0].sum()

        jac = float(jax.jit(jax.jacrev(total))(1.0))

        assert np.isfinite(jac)
        assert jac == 0.0

    def test_returns_the_point_estimate_only(self, ishigami_data):
        """T4: a bootstrap is policy, so the core has no ``n_bootstrap``.

        ``analyze`` keeps the interval, and its point estimate is the number
        the core returns, so an interval is always centred on it.
        """
        import inspect

        X, Y = ishigami_data
        assert "n_bootstrap" not in inspect.signature(indices).parameters

        result = analyze(ishigami.PROBLEM, X, Y, n_bins=8, n_bootstrap=5, key=jax.random.key(0))
        (pawn,) = indices(ishigami.PROBLEM, X, Y, n_bins=8)

        assert result.pawn_conf is not None
        np.testing.assert_array_equal(np.asarray(pawn), np.asarray(result.pawn))

    def test_an_all_empty_parameter_is_nan_without_a_warning(self, ishigami_data, recwarn):
        """T4: the core drops the host sync that the empty-bin warning needs.

        ``analyze`` reads the emptiness back to the host and warns. The core
        cannot, because a warning needs a concrete value, so it returns NaN
        and says nothing.
        """
        X, Y = ishigami_data
        # Far outside the declared bounds, so every sample of that column
        # takes the out-of-range sentinel and no bin is filled.
        X = X.at[:, 0].set(1e6)

        with pytest.warns(JaxgsaWarning, match="all bins empty"):
            result = analyze(ishigami.PROBLEM, X, Y, n_bins=8)
        assert np.isnan(np.asarray(result.pawn)[0])

        recwarn.clear()
        (pawn,) = indices(ishigami.PROBLEM, X, Y, n_bins=8)

        assert np.isnan(np.asarray(pawn)[0])
        assert [w for w in recwarn if issubclass(w.category, JaxgsaWarning)] == []

    def test_rejects_mismatched_shapes(self, ishigami_data):
        """T4: a shape check still raises; only data-driven policy is dropped."""
        X, Y = ishigami_data

        with pytest.raises(ValueError, match="rows"):
            indices(ishigami.PROBLEM, X, Y[:-1], n_bins=8)
        with pytest.raises(ValueError, match="columns"):
            indices(ishigami.PROBLEM, X[:, :2], Y, n_bins=8)

    # Each marginal family reaches ``cdf_to_unit_interval`` by a different
    # branch, and two of those branches used to break tracing in different
    # ways: a truncated Gaussian read ``X`` on the host through SciPy, and
    # *every* Gaussian, truncated or not, took ``float()`` of a tracer while
    # standardising. They were separate defects, so they get separate cases.
    _MARGINALS: dict[str, InputSpecValue] = {
        "uniform": (0.0, 1.0),
        "gaussian": {"dist": "gaussian", "mean": 0.3, "variance": 2.0},
        "truncated": {
            "dist": "gaussian",
            "mean": 0.0,
            "variance": 1.0,
            "low": -1.5,
            "high": 2.0,
        },
        # Both bounds far out in the upper tail. The direct
        # ``(Phi(z) - Phi(a)) / (Phi(b) - Phi(a))`` form cancels to noise here,
        # so this case pins the survival-function branch of the transform.
        "upper_tail": {
            "dist": "gaussian",
            "mean": 0.0,
            "variance": 1.0,
            "low": 5.0,
            "high": 6.0,
        },
    }

    @pytest.mark.parametrize("marginal", sorted(_MARGINALS))
    def test_every_marginal_traces_and_matches_analyze(self, marginal):
        """T4: ``jit`` holds for every marginal family, not only uniform.

        A uniform-only contract test passes while the Gaussian branch of the
        unit-interval transform cannot be traced at all, which is exactly the
        state this package shipped in. The eager comparison against ``analyze``
        and the jitted comparison against the eager core are both needed: the
        first is the value contract, the second is the traceability one, and a
        host read inside the transform breaks only the second.
        """
        problem = Problem.from_dict({"a": self._MARGINALS[marginal], "b": (0.0, 1.0)})
        X = jnp.asarray(monte_carlo(problem, n=1200, seed=17))
        Y = jnp.sin(X[:, 0]) + 0.5 * X[:, 1]

        result = analyze(problem, X, Y, n_bins=6)
        (pawn,) = indices(problem, X, Y, n_bins=6)
        np.testing.assert_array_equal(np.asarray(pawn), np.asarray(result.pawn))

        (pawn_jit,) = jax.jit(lambda outputs: indices(problem, X, outputs, n_bins=6))(Y)
        np.testing.assert_allclose(np.asarray(pawn_jit), np.asarray(pawn), rtol=1e-6)


class TestBinCountDiagnostic:
    """Tier T4 (internal consistency): the contributing-bin diagnostic."""

    def test_full_bins_report_n_bins_and_match_pawn_shape(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, n_bins=10)
        n_valid = np.asarray(result.n_valid_bins)
        assert n_valid.shape == np.asarray(result.pawn).shape
        # 5000 samples over 10 equal-probability bins: every bin holds
        # hundreds of samples, so every bin contributes.
        np.testing.assert_array_equal(n_valid, np.full_like(n_valid, 10))

    def test_diagnostic_broadcasts_over_output_axes(self, ishigami_data):
        """Bin occupancy never reads Y, so the count repeats across (T, K)."""
        X, Y = ishigami_data
        Y3 = jnp.stack([jnp.stack([Y, Y**2], axis=1)] * 2, axis=1)  # (N, 2, 2)
        result = analyze(ishigami.PROBLEM, X, Y3, n_bins=8)
        n_valid = np.asarray(result.n_valid_bins)
        assert n_valid.shape == (2, 2, 3)
        assert (n_valid == n_valid[0, 0]).all()

    def test_to_dataset_carries_the_diagnostic(self, ishigami_data):
        X, Y = ishigami_data
        ds = analyze(ishigami.PROBLEM, X, Y).to_dataset()
        assert "n_valid_bins" in ds
        assert ds["n_valid_bins"].dims == ("param",)

    def test_sparse_bins_warn_and_name_the_fix(self):
        """A parameter whose samples pile into few bins gets one warning."""
        problem = Problem(names=("narrow", "wide"), bounds=((0.0, 1.0), (0.0, 1.0)))
        rng = np.random.default_rng(0)
        X = np.column_stack(
            [
                # All mass in the first of 10 equal-width bins: 1/10 bins
                # contribute, which is below half.
                rng.uniform(0.0, 0.05, size=64),
                rng.uniform(0.0, 1.0, size=64),
            ]
        )
        Y = X[:, 0] + X[:, 1]
        with pytest.warns(JaxgsaWarning, match="fewer bins .* or more samples"):
            result = analyze(problem, jnp.asarray(X), jnp.asarray(Y), n_bins=10)
        n_valid = np.asarray(result.n_valid_bins)
        assert n_valid[0] == 1
        assert n_valid[1] > 5

    def test_healthy_sample_does_not_warn(self, ishigami_data):
        import warnings as _warnings

        X, Y = ishigami_data
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", JaxgsaWarning)
            analyze(ishigami.PROBLEM, X, Y, n_bins=10)

    def test_all_empty_warning_is_not_doubled(self):
        """A fully out-of-range parameter keeps its own warning, once."""
        import warnings as _warnings

        problem = Problem(names=("out", "ok"), bounds=((0.0, 1.0), (0.0, 1.0)))
        X = np.column_stack(
            [
                np.full(32, 2.0),  # outside the declared bounds: every bin empty
                np.linspace(0.0, 1.0, 32),
            ]
        )
        Y = X[:, 1]
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            analyze(problem, jnp.asarray(X), jnp.asarray(Y), n_bins=4)
        empty = [w for w in caught if "all bins empty" in str(w.message)]
        sparse = [w for w in caught if "fewer bins" in str(w.message)]
        assert len(empty) == 1
        assert not sparse
