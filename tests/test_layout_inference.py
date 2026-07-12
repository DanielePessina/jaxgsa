"""Tests for the smart output-layout inference (`_infer_output_layout`).

The ladder under test: exact canonical shapes pass silently, unambiguously
recoverable layouts are fixed with a UserWarning, ambiguous ones raise.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import gsax
from gsax._normalization import (
    LayoutOps,
    _infer_output_layout,
    _infer_output_layout_ops,
)
from gsax.problem import Problem

UNLABELED = Problem(names=("x0", "x1", "x2"), bounds=((0.0, 1.0),) * 3)
ONE_LABEL = Problem(names=("x0", "x1", "x2"), bounds=((0.0, 1.0),) * 3, output_names=("pressure",))
TWO_LABELS = Problem(names=("x0", "x1", "x2"), bounds=((0.0, 1.0),) * 3, output_names=("a", "b"))

N = 40


class TestSilentRung:
    """Exact canonical shapes pass through untouched, with no warning."""

    @pytest.mark.parametrize(
        "shape,problem",
        [
            ((N,), UNLABELED),
            ((N, 5), UNLABELED),
            ((N, 3, 5), UNLABELED),
            ((N, 2), TWO_LABELS),
            ((N, 7, 2), TWO_LABELS),
        ],
    )
    def test_passthrough(self, shape, problem):
        Y = jnp.ones(shape)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = _infer_output_layout(Y, problem, N)
        assert out.shape == shape

    def test_square_positional_trust(self):
        """When the leading axis matches n, position wins even if another
        axis coincidentally matches too."""
        Y = jnp.arange(N * N, dtype=jnp.float32).reshape(N, N)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = _infer_output_layout(Y, UNLABELED, N)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(Y))


class TestWarnRung:
    """Unambiguously recoverable layouts are fixed with a UserWarning."""

    def test_transposed_2d_moved(self):
        Y = jnp.arange(5 * N, dtype=jnp.float32).reshape(5, N)
        with pytest.warns(UserWarning, match="sample axis"):
            out = _infer_output_layout(Y, UNLABELED, N)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(Y).T)

    def test_sample_axis_in_middle_3d(self):
        Y = jnp.arange(3 * N * 5, dtype=jnp.float32).reshape(3, N, 5)
        with pytest.warns(UserWarning, match="sample axis"):
            out = _infer_output_layout(Y, UNLABELED, N)
        assert out.shape == (N, 3, 5)
        np.testing.assert_array_equal(np.asarray(out), np.moveaxis(np.asarray(Y), 1, 0))

    def test_3d_swapped_output_axis(self):
        """(n, K, T) with labels identifying the middle axis as K is swapped."""
        Y = jnp.arange(N * 2 * 7, dtype=jnp.float32).reshape(N, 2, 7)
        with pytest.warns(UserWarning, match="swapping"):
            out = _infer_output_layout(Y, TWO_LABELS, N)
        assert out.shape == (N, 7, 2)
        np.testing.assert_array_equal(np.asarray(out), np.swapaxes(np.asarray(Y), 1, 2))


class TestLabelRules:
    def test_2d_single_label_is_time(self):
        """(n, T) with exactly one named output flows as (n, T, 1)."""
        Y = jnp.arange(N * 6, dtype=jnp.float32).reshape(N, 6)
        out = _infer_output_layout(Y, ONE_LABEL, N)
        assert out.shape == (N, 6, 1)
        np.testing.assert_array_equal(np.asarray(out)[:, :, 0], np.asarray(Y))

    def test_2d_label_count_mismatch_raises(self):
        Y = jnp.ones((N, 5))
        with pytest.raises(ValueError, match="output_names"):
            _infer_output_layout(Y, TWO_LABELS, N)

    def test_3d_no_axis_matches_labels_raises(self):
        Y = jnp.ones((N, 5, 7))
        with pytest.raises(ValueError, match="output_names"):
            _infer_output_layout(Y, TWO_LABELS, N)


class TestLayoutOpsRecord:
    """`_infer_output_layout_ops` reports the transforms it applied."""

    def test_canonical_is_identity(self):
        _, ops = _infer_output_layout_ops(jnp.ones((N, 5)), UNLABELED, N)
        assert ops == LayoutOps()

    def test_records_sample_axis_move(self):
        Y = jnp.arange(5 * N, dtype=jnp.float32).reshape(5, N)
        with pytest.warns(UserWarning, match="sample axis"):
            _, ops = _infer_output_layout_ops(Y, UNLABELED, N)
        assert ops == LayoutOps(sample_axis=1)

    def test_records_inserted_output_axis(self):
        Y = jnp.arange(N * 6, dtype=jnp.float32).reshape(N, 6)
        _, ops = _infer_output_layout_ops(Y, ONE_LABEL, N)
        assert ops == LayoutOps(inserted_output_axis=True)

    def test_records_swapped_tk(self):
        Y = jnp.arange(N * 2 * 7, dtype=jnp.float32).reshape(N, 2, 7)
        with pytest.warns(UserWarning, match="swapping"):
            _, ops = _infer_output_layout_ops(Y, TWO_LABELS, N)
        assert ops == LayoutOps(swapped_tk=True)


class TestWarningAttribution:
    """Layout warnings point at the user's call site, not gsax internals."""

    def test_direct_caller_points_at_user(self):
        sr = gsax.sample_morris(UNLABELED, n_trajectories=8, seed=1, verbose=False)
        base = jnp.sin(jnp.asarray(sr.samples) @ jnp.array([1.0, 2.0, 0.5]))
        Y2 = jnp.stack([base, 2.0 * base], axis=-1)  # (n_total, 2)
        with pytest.warns(UserWarning, match="sample axis") as rec:
            gsax.analyze_morris(sr, Y2.T)
        assert rec[0].filename == __file__

    def test_wrapped_caller_points_at_user(self):
        X = jnp.asarray(gsax.sample_mc(UNLABELED, 200, seed=3))
        base = jnp.sin(X[:, 0]) + 2.0 * X[:, 1]
        Y2 = jnp.stack([base, 2.0 * base], axis=-1)  # (N, 2)
        with pytest.warns(UserWarning, match="sample axis") as rec:
            gsax.analyze_pawn(UNLABELED, X, Y2.T)
        assert rec[0].filename == __file__


class TestRaiseRung:
    def test_no_axis_matches_sample_count(self):
        with pytest.raises(ValueError, match="sample rows"):
            _infer_output_layout(jnp.ones(N + 3), UNLABELED, N)

    def test_ambiguous_sample_axis_raises(self):
        """Two non-leading axes matching n cannot be resolved."""
        Y = jnp.ones((5, N, N))
        with pytest.raises(ValueError, match="sample rows"):
            _infer_output_layout(Y, UNLABELED, N)

    def test_4d_raises(self):
        with pytest.raises(ValueError, match="Y must be 1-D"):
            _infer_output_layout(jnp.ones((N, 2, 2, 2)), UNLABELED, N)


# ---------------------------------------------------------------------------
# Integration through public entry points
# ---------------------------------------------------------------------------


class TestIntegration:
    def _xy(self, problem, K=None):
        X = jnp.asarray(gsax.sample_mc(problem, 600, seed=11))
        base = jnp.sin(X[:, 0]) + 2.0 * X[:, 1]
        if K is None:
            return X, base
        return X, jnp.stack([(k + 1.0) * base for k in range(K)], axis=-1)

    def test_pawn_transposed_matches_canonical(self):
        X, Y2 = self._xy(UNLABELED, K=2)
        canonical = gsax.analyze_pawn(UNLABELED, X, Y2)
        with pytest.warns(UserWarning, match="sample axis"):
            fixed = gsax.analyze_pawn(UNLABELED, X, Y2.T)
        np.testing.assert_allclose(np.asarray(fixed.pawn), np.asarray(canonical.pawn))

    def test_hdmr_single_label_2d_is_time_series(self):
        X, base = self._xy(ONE_LABEL)
        Y2 = jnp.stack([base, 2.0 * base], axis=-1)  # (N, 2) = two timepoints
        result = gsax.analyze_hdmr(ONE_LABEL, X, Y2)
        ds = result.to_dataset(time_coords=[0.0, 1.0])
        assert ds["ST"].dims == ("time", "output", "param")
        assert list(ds.coords["output"].values) == ["pressure"]
        # Same data as explicit (N, T, 1) — results must match exactly.
        explicit = gsax.analyze_hdmr(ONE_LABEL, X, Y2[:, :, None])
        np.testing.assert_allclose(np.asarray(result.ST), np.asarray(explicit.ST))

    def test_morris_bootstrap_conf_matches_canonical(self):
        """Inference happens before resampling: an inferred layout yields
        bit-identical bootstrap CIs to the canonical run."""
        import jax

        problem = Problem(
            names=("x0", "x1", "x2"),
            bounds=((0.0, 1.0),) * 3,
            output_names=("pressure",),
        )
        sr = gsax.sample_morris(problem, n_trajectories=12, seed=5, verbose=False)
        base = jnp.sin(jnp.asarray(sr.samples) @ jnp.array([1.0, 2.0, 0.5]))
        Y2 = jnp.stack([base, 2.0 * base], axis=-1)  # (n, T=2) single label
        key = jax.random.PRNGKey(0)
        res_2d = gsax.analyze_morris(sr, Y2, num_resamples=64, key=key)
        res_3d = gsax.analyze_morris(sr, Y2[:, :, None], num_resamples=64, key=key)
        assert res_2d.mu_star.shape == (2, 1, 3)
        assert res_2d.mu_star_conf is not None and res_3d.mu_star_conf is not None
        np.testing.assert_array_equal(
            np.asarray(res_2d.mu_star_conf), np.asarray(res_3d.mu_star_conf)
        )
