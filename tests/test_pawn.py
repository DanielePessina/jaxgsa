"""Tests for PAWN sensitivity analysis."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _fixtures_givendata import BOOT, invalid_sample

from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks import ishigami
from jaxgsa.pawn import analyze, indices
from jaxgsa.problem import InputSpecValue, Problem
from jaxgsa.sampling import monte_carlo


class TestPAWNBasic:
    def test_plain_result_properties(self, pawn_ishigami):
        """Indices lie in [0, 1], and x1 and x2 outrank the weak x3."""
        pawn = np.asarray(pawn_ishigami.pawn)
        assert np.all(pawn >= 0.0) and np.all(pawn <= 1.0)
        assert pawn[0] > pawn[2], "x1 should be more important than x3"
        assert pawn[1] > pawn[2], "x2 should be more important than x3"

    @pytest.mark.parametrize("statistic", ["median", "max"])
    def test_closed_form_for_a_single_active_input(self, statistic):
        """Tier T0 (closed form): ``Y = x1`` has a known KS distance per bin.

        With ``n_bins = b`` equal-width bins of a uniform ``x1``, the
        conditional law of ``Y`` in bin ``i`` is uniform on ``[i/b, (i+1)/b]``.
        Its KS distance from the unconditional uniform law is
        ``max(i/b, 1 - (i+1)/b)``. The ``"max"`` statistic is the largest of
        these, ``1 - 1/b``, and ``"median"`` is their median. The unused
        ``x2`` has conditional laws equal to the unconditional one, so its
        index is small, and only finite-sample noise lifts it above 0.

        At ``N = 2048`` and ``b = 8`` each bin holds about 256 samples. Five
        seeds measured a spread of 0.01 on ``x1`` (the tolerance below is
        three times that) and at most 0.07 on ``x2`` (the bound is twice that).
        """
        b = 8
        X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=2048, seed=0))
        result = analyze(
            ishigami.PROBLEM, X, X[:, 0], n_bins=b, statistic=statistic, verbose=False
        )
        bin_ks = np.maximum(np.arange(b) / b, 1.0 - (np.arange(b) + 1) / b)
        target = 1.0 - 1.0 / b if statistic == "max" else float(np.median(bin_ks))
        pawn = np.asarray(result.pawn)
        assert pawn[0] == pytest.approx(target, abs=0.03)
        assert pawn[1] < 0.15
        assert pawn[2] < 0.15

    def test_slice_chunk_size_invariance(self, given_ishigami_xy):
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
        X, Y = given_ishigami_xy
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

    def test_slice_chunk_size_splits_the_columns(self, given_ishigami_xy, monkeypatch):
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

        X, Y = given_ishigami_xy
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

    def test_slice_chunk_size_must_be_positive(self, given_ishigami_xy):
        """Tier T4 (internal consistency): an unusable chunk size is refused."""
        X, Y = given_ishigami_xy
        with pytest.raises(ValueError, match="slice_chunk_size must be >= 1"):
            analyze(ishigami.PROBLEM, X[:200], Y[:200], slice_chunk_size=0)


class TestPAWNStatistics:
    def test_max_geq_median(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        r_median = analyze(ishigami.PROBLEM, X, Y, statistic="median")
        r_max = analyze(ishigami.PROBLEM, X, Y, statistic="max")
        assert np.all(np.asarray(r_max.pawn) >= np.asarray(r_median.pawn) - 1e-6)


class TestPAWNBootstrap:
    def test_bootstrap_lower_leq_upper(self, pawn_ishigami_boot):
        assert pawn_ishigami_boot.pawn_conf is not None
        lower = np.asarray(pawn_ishigami_boot.pawn_conf[0])
        upper = np.asarray(pawn_ishigami_boot.pawn_conf[1])
        assert np.all(lower <= upper + 1e-6)

    def test_bootstrap_without_a_key_raises(self, given_ishigami_xy):
        """Tier T4. A bootstrap needs a key, and an int seed is not one."""
        X, Y = given_ishigami_xy
        with pytest.raises(ValueError, match="key is required"):
            analyze(ishigami.PROBLEM, X, Y, n_bootstrap=BOOT)

    def test_the_key_decides_the_interval(self, given_ishigami_xy, pawn_ishigami_boot):
        """Tier T4. Determinism is a property of the key, not of a seed.

        The same key gives the same interval and a different key draws a
        different resample. The point estimate reads no key at all.
        """
        X, Y = given_ishigami_xy
        same = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, key=jax.random.key(0))
        other = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=BOOT, key=jax.random.key(12))
        assert pawn_ishigami_boot.pawn_conf is not None
        assert same.pawn_conf is not None and other.pawn_conf is not None
        np.testing.assert_array_equal(
            np.asarray(same.pawn_conf), np.asarray(pawn_ishigami_boot.pawn_conf)
        )
        assert not np.array_equal(np.asarray(other.pawn_conf), np.asarray(same.pawn_conf))
        np.testing.assert_array_equal(np.asarray(other.pawn), np.asarray(same.pawn))

    def test_gaussian_ci_is_centred_on_the_estimate(self, given_ishigami_xy):
        """Tier T0 (closed form): the Gaussian interval is symmetric.

        ``estimate +/- z * sd`` puts the point estimate exactly at the
        midpoint of the two endpoints, which the percentile interval does
        not do. That is what tells the two branches apart.
        """
        X, Y = given_ishigami_xy
        result = analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_bootstrap=BOOT,
            ci_method="gaussian",
            key=jax.random.key(0),
        )
        assert result.ci is not None
        assert result.ci.method == "gaussian"
        assert result.pawn_conf is not None
        midpoint = 0.5 * (np.asarray(result.pawn_conf[0]) + np.asarray(result.pawn_conf[1]))
        np.testing.assert_allclose(midpoint, np.asarray(result.pawn), atol=1e-6)

    def test_unknown_ci_method_rejected(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        with pytest.raises(ValueError, match="ci_method"):
            analyze(
                ishigami.PROBLEM,
                X,
                Y,
                n_bootstrap=BOOT,
                ci_method=cast(Any, "bca"),
                key=jax.random.key(0),
            )


class TestPAWNTiedOutputs:
    """Tied outputs still produce a bounded PAWN index."""

    def test_discrete_output_is_finite_and_bounded(self):
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
        assert np.all(np.isfinite(got))
        assert np.all(got >= 0.0) and np.all(got <= 1.0)

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


class TestPAWNInvalidInput:
    """T4 (behaviour): the policy-specific part of ``on_invalid`` for PAWN.

    The raise, propagate and drop cases shared by all given-data methods live
    in ``test_given_data_on_invalid.py``.
    """

    def test_a_non_finite_x_is_no_longer_swallowed_by_the_bin_sentinel(self):
        """T4: the silent -1 path is unreachable from analyze().

        ``_equal_width_bins`` still maps a non-finite value to the
        out-of-range sentinel, and out-of-range inputs are a separate,
        legitimate concern. What changed is that ``analyze`` never lets a
        non-finite value get that far: every policy reports it first, so no
        sample leaves the conditional sets in silence. The old behaviour was
        to return an index with no warning and a smaller effective sample.
        """
        problem, X, Y = invalid_sample()
        X = X.at[5, 0].set(jnp.nan)

        with pytest.raises(ValueError, match="non-finite"):
            analyze(problem, X, Y)

        with pytest.warns(JaxgsaWarning):
            result = analyze(problem, X, Y, on_invalid="propagate")
        assert result.invalid.unit_indices == (5,)


class TestPureCore:
    """The transformable core ``pawn.indices``.

    Tier T4 throughout (internal consistency and transformability). The
    statistic's bounded behavior is checked on package-owned data above, and
    ``indices`` runs the same kernel ``analyze`` runs.
    """

    @pytest.mark.parametrize("statistic", ["median", "max", "mean"])
    def test_matches_analyze_for_every_statistic(self, given_ishigami_xy, statistic):
        """T4: the aggregation choice is a Python branch, shared by both paths."""
        X, Y = given_ishigami_xy
        kwargs: dict[str, Any] = {"n_bins": 8, "statistic": statistic}

        result = analyze(ishigami.PROBLEM, X, Y, **kwargs)
        (pawn,) = indices(ishigami.PROBLEM, X, Y, **kwargs)

        np.testing.assert_array_equal(np.asarray(pawn), np.asarray(result.pawn))

    def test_matches_analyze_time_series(self, given_ishigami_xy):
        """T4: the chunked ``(N, T, K)`` path agrees with ``analyze``."""
        X, base = given_ishigami_xy
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

    def test_is_jittable(self, given_ishigami_xy):
        """T4: ``jit`` traces ``indices`` and returns the eager values."""
        X, Y = given_ishigami_xy

        jitted = jax.jit(lambda outputs: indices(ishigami.PROBLEM, X, outputs, n_bins=8))
        (pawn_jit,) = jitted(Y)
        (pawn,) = indices(ishigami.PROBLEM, X, Y, n_bins=8)

        np.testing.assert_allclose(np.asarray(pawn_jit), np.asarray(pawn), rtol=1e-6)

    def test_is_vmappable(self, given_ishigami_xy):
        """T4: ``vmap`` maps ``indices`` over a batch of output vectors."""
        X, base = given_ishigami_xy
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

    def test_jit_of_jacrev_compiles_and_the_derivative_is_zero(self, given_ishigami_xy):
        """T4: ``jit(jacrev(...))`` compiles, and the answer is a structural zero.

        A PAWN index reads the *ordering* of the outputs and the bin an input
        falls in, never the values themselves. Both are piecewise constant, so
        the derivative is exactly 0 almost everywhere. This test therefore
        checks two different things: that the core traces under reverse-mode
        differentiation at all, and that what comes back is the honest zero of
        a rank statistic rather than a NaN.
        """
        X, Y = given_ishigami_xy

        def total(scale):
            return indices(ishigami.PROBLEM, X, Y * scale, n_bins=8)[0].sum()

        jac = float(jax.jit(jax.jacrev(total))(1.0))

        assert np.isfinite(jac)
        assert jac == 0.0

    def test_returns_the_point_estimate_only(self, given_ishigami_xy, pawn_ishigami_boot):
        """T4: a bootstrap is policy, so the core has no ``n_bootstrap``.

        ``analyze`` keeps the interval, and its point estimate is the number
        the core returns, so an interval is always centred on it. The core
        returns a bare tuple holding one array.
        """
        import inspect

        X, Y = given_ishigami_xy
        assert "n_bootstrap" not in inspect.signature(indices).parameters

        out = indices(ishigami.PROBLEM, X, Y)

        assert isinstance(out, tuple) and len(out) == 1
        assert pawn_ishigami_boot.pawn_conf is not None
        np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(pawn_ishigami_boot.pawn))

    def test_an_all_empty_parameter_is_nan_without_a_warning(self, given_ishigami_xy, recwarn):
        """T4: the core drops the host sync that the empty-bin warning needs.

        ``analyze`` reads the emptiness back to the host and warns. The core
        cannot, because a warning needs a concrete value, so it returns NaN
        and says nothing.
        """
        X, Y = given_ishigami_xy
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

    def test_rejects_mismatched_shapes(self, given_ishigami_xy):
        """T4: a shape check still raises; only data-driven policy is dropped."""
        X, Y = given_ishigami_xy

        with pytest.raises(ValueError, match="rows"):
            indices(ishigami.PROBLEM, X, Y[:-1], n_bins=8)
        with pytest.raises(ValueError, match="columns"):
            indices(ishigami.PROBLEM, X[:, :2], Y, n_bins=8)

    # Each marginal family reaches ``cdf_to_unit_interval`` by a different
    # branch, and two of those branches used to break tracing in different
    # ways: a truncated Gaussian read ``X`` on the host through an external
    # CDF implementation, and *every* Gaussian, truncated or not, took
    # ``float()`` of a tracer while
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

    def test_full_bins_report_n_bins_and_match_pawn_shape(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
        result = analyze(ishigami.PROBLEM, X, Y, n_bins=10)
        n_valid = np.asarray(result.n_valid_bins)
        assert n_valid.shape == np.asarray(result.pawn).shape
        # 2048 samples over 10 equal-probability bins: every bin holds
        # about two hundred samples, so every bin contributes.
        np.testing.assert_array_equal(n_valid, np.full_like(n_valid, 10))

    def test_diagnostic_broadcasts_over_output_axes(self, given_ishigami_xy):
        """Bin occupancy never reads Y, so the count repeats across (T, K)."""
        X, Y = given_ishigami_xy
        Y3 = jnp.stack([jnp.stack([Y, Y**2], axis=1)] * 2, axis=1)  # (N, 2, 2)
        result = analyze(ishigami.PROBLEM, X, Y3, n_bins=8)
        n_valid = np.asarray(result.n_valid_bins)
        assert n_valid.shape == (2, 2, 3)
        assert (n_valid == n_valid[0, 0]).all()

    def test_to_dataset_carries_the_diagnostic(self, given_ishigami_xy):
        X, Y = given_ishigami_xy
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

    def test_healthy_sample_does_not_warn(self, given_ishigami_xy):
        import warnings as _warnings

        X, Y = given_ishigami_xy
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
