"""Live checks against outside libraries: SALib, POT and SciPy.

These tests import the libraries at run time and compare jaxgsa with them.
They are not part of the default test suite. The file name does not match
pytest's ``test_*.py`` pattern, so ``uv run pytest`` never collects it, and
CI never runs it. Run it by hand::

    uv run --extra dev pytest scripts/oracles/external_checks.py

The saved-literal versions of some of these checks (OpenTURNS Sobol
estimators, SALib Plischke class counts) live in ``tests/`` and do run in
the default suite.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa.benchmarks import ishigami, sobol_g
from jaxgsa.borgonovo import analyze as _borgonovo_analyze
from jaxgsa.borgonovo._analyze import _plischke_n_classes
from jaxgsa.morris import analyze as _morris_analyze
from jaxgsa.morris import sample as morris_sample
from jaxgsa.optimal_transport import analyze as _ot_analyze
from jaxgsa.pawn import analyze as _pawn_analyze
from jaxgsa.problem import Problem
from jaxgsa.sampling import monte_carlo
from jaxgsa.sobol._estimators import DEFAULT_ESTIMATOR

# These return a result-or-irregular union; the checks use regular inputs only.
borgonovo_analyze: Any = _borgonovo_analyze
morris_analyze: Any = _morris_analyze
ot_analyze: Any = _ot_analyze
pawn_analyze: Any = _pawn_analyze


def _ishigami_data(n: int, seed: int = 42):
    X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n_samples=n, seed=seed))
    return X, ishigami.evaluate(X)


@pytest.fixture(scope="module")
def ishigami_data_5k():
    """Ishigami sample, N = 5000 (PAWN and Borgonovo checks)."""
    return _ishigami_data(5000)


@pytest.fixture(scope="module")
def ishigami_data_8k():
    """Ishigami sample, N = 8192 (optimal transport check)."""
    return _ishigami_data(8192)


class TestSobolSALibParity:
    """T2. SALib implements the default scheme, so it pins today's numbers."""

    def test_default_matches_salib_sobol(self):
        """``saltelli-jansen`` reproduces ``SALib.analyze.sobol`` exactly.

        SALib is fed its own design, and jaxgsa's estimator is then run on
        the same outputs, de-interleaved by hand. SALib standardises the
        output before estimating, so the comparison does the same; without
        it the two differ by about 2e-5, which is the standardisation and
        not the formula.
        """
        pytest.importorskip("SALib")
        from SALib.analyze import sobol as salib_analyze
        from SALib.sample import sobol as salib_sample

        from jaxgsa.sobol._estimators import first_total_kernel

        problem = {
            "num_vars": 3,
            "names": ["x1", "x2", "x3"],
            "bounds": [[-np.pi, np.pi]] * 3,
        }
        base_n = 2**12
        X = salib_sample.sample(problem, base_n, calc_second_order=False)
        Y = np.asarray(ishigami.evaluate(jnp.asarray(X)), dtype=np.float64)
        expected = salib_analyze.analyze(problem, Y, calc_second_order=False)

        standardised = (Y - Y.mean()) / Y.std()
        grouped = standardised.reshape(base_n, 5)
        S1, ST = first_total_kernel(DEFAULT_ESTIMATOR)(
            jnp.asarray(grouped[:, 0]),
            jnp.asarray(grouped[:, 1:4]),
            jnp.asarray(grouped[:, 4]),
        )
        np.testing.assert_allclose(np.asarray(S1), expected["S1"], atol=1e-6)
        np.testing.assert_allclose(np.asarray(ST), expected["ST"], atol=1e-6)


class TestMorrisSALibParity:
    def test_trajectory_matches_salib(self):
        """SALib analyzing the identical expanded design must agree exactly."""
        from SALib.analyze import morris as salib_morris

        num_levels = 4
        sr = morris_sample(
            sobol_g.PROBLEM, n_trajectories=25, n_levels=num_levels, seed=42, verbose=False
        )
        Y_unique = sobol_g.evaluate(jnp.asarray(sr.samples))
        res = morris_analyze(sr, Y_unique)

        # Re-expand to the full design; unit-cube bounds make jaxgsa's samples
        # identical to unit-cube coordinates, sidestepping SALib's rescaling.
        X_expanded = sr.samples[sr.expanded_to_unique]
        Y_expanded = np.asarray(Y_unique)[sr.expanded_to_unique]
        problem_dict = {
            "num_vars": sr.n_params,
            "names": list(sobol_g.PROBLEM.names),
            "bounds": [[0.0, 1.0]] * sr.n_params,
        }
        Si = salib_morris.analyze(
            problem_dict,
            X_expanded,
            Y_expanded.astype(np.float64),
            num_levels=num_levels,
            num_resamples=2,
            seed=0,
        )

        np.testing.assert_allclose(np.asarray(res.mu), Si["mu"], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(res.mu_star), Si["mu_star"], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(res.sigma), Si["sigma"], rtol=1e-4, atol=1e-4)


class TestDeltaSALibComparison:
    def test_plugin_matches_salib_calc_delta(self, ishigami_data_5k):
        """Unit-level parity: plug-in estimator vs SALib's calc_delta."""
        from SALib.analyze import delta as salib_delta

        X_np = np.asarray(ishigami_data_5k[0])
        Y_np = np.asarray(ishigami_data_5k[1])
        N = len(Y_np)

        M = _plischke_n_classes(N)
        m = np.linspace(0, N, M + 1)
        Ygrid = np.linspace(Y_np.min(), Y_np.max(), 100)
        salib_plugin = np.array(
            [salib_delta.calc_delta(Y_np, Ygrid, X_np[:, i], m) for i in range(3)]
        )

        result = borgonovo_analyze(
            ishigami.PROBLEM, ishigami_data_5k[0], ishigami_data_5k[1], n_bootstrap=0
        )
        np.testing.assert_allclose(np.asarray(result.delta), salib_plugin, atol=1e-5)

    def test_matches_salib_end_to_end(self, ishigami_data_5k):
        """Bias-corrected delta and S1 vs SALib.analyze.delta.

        SALib's central estimates are computed on a random resample of the
        data (jaxgsa deliberately uses the original sample), so the tolerance
        covers that resampling noise on top of bootstrap RNG differences.
        """
        from SALib.analyze import delta as salib_delta

        X_np = np.asarray(ishigami_data_5k[0])
        Y_np = np.asarray(ishigami_data_5k[1])

        salib_problem = {
            "num_vars": 3,
            "names": ["x1", "x2", "x3"],
            "bounds": [[-np.pi, np.pi]] * 3,
        }
        salib_result = salib_delta.analyze(salib_problem, X_np, Y_np, num_resamples=100, seed=7)

        result = borgonovo_analyze(
            ishigami.PROBLEM, ishigami_data_5k[0], ishigami_data_5k[1], key=jax.random.key(0)
        )
        np.testing.assert_allclose(
            np.asarray(result.delta), salib_result["delta_balanced"], atol=0.03
        )
        np.testing.assert_allclose(np.asarray(result.S1), salib_result["S1"], atol=0.02)


class TestPAWNSALibComparison:
    def test_matches_salib_median(self, ishigami_data_5k):
        """Compare against SALib's PAWN implementation."""
        from SALib.analyze import pawn as salib_pawn

        X_np = np.asarray(ishigami_data_5k[0])
        Y_np = np.asarray(ishigami_data_5k[1])

        salib_problem = {
            "num_vars": 3,
            "names": ["x1", "x2", "x3"],
            "bounds": [[-np.pi, np.pi]] * 3,
        }
        salib_result = salib_pawn.analyze(salib_problem, X_np, Y_np, S=10, seed=0)
        salib_median = salib_result["median"]

        jaxgsa_result = pawn_analyze(
            ishigami.PROBLEM,
            ishigami_data_5k[0],
            ishigami_data_5k[1],
            n_classes=10,
            statistic="median",
        )
        jaxgsa_pawn = np.asarray(jaxgsa_result.pawn)

        np.testing.assert_allclose(jaxgsa_pawn, salib_median, atol=0.01)


class TestPAWNTiedOutputsKS:
    """The KS statistic must be tie-aware, matching ``scipy.stats.ks_2samp``."""

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

        got = np.asarray(
            pawn_analyze(problem, jnp.asarray(X), jnp.asarray(y), n_classes=n_bins).pawn
        )

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


class TestOTPOTComparison:
    """Numerical parity against POT (MIT-licensed reference)."""

    def test_pot_wasserstein_1d_semantics_pin(self):
        """Pin that ot.wasserstein_1d(p=2) returns W_2^2, not W_2.

        Two point masses at 0 vs two at 1: W_2^2 = 1.0 (and W_2 = 1.0 too),
        so add an asymmetric case where they differ: {0} vs {2} gives
        W_2^2 = 4, W_2 = 2.
        """
        ot = pytest.importorskip("ot")
        val = ot.wasserstein_1d(np.array([0.0]), np.array([2.0]), p=2)
        assert val == pytest.approx(4.0), "POT changed wasserstein_1d semantics"

    @pytest.mark.parametrize(
        ("N", "M", "label"),
        [
            (1000, 25, "divisible: n_m = 40 divides N"),
            (1013, 25, "not divisible: n_m does not divide N (Section 5)"),
        ],
    )
    def test_per_bin_w2_matches_pot(self, N, M, label):
        """Per-bin 1-D W2^2 vs POT, divisible and not.

        The exact quantile coupling (:func:`_quantile_rank_split`) couples
        every mass point to at most two class members with a split weight,
        so it is exact for any class size, not only one that divides N. The
        prior version of this test covered the divisible case only.
        """
        ot = pytest.importorskip("ot")
        from jaxgsa._core.partition import _build_class_indices, _class_layout
        from jaxgsa.optimal_transport._analyze import _ot_1d_kernel

        key = jax.random.PRNGKey(1)
        X = jax.random.uniform(key, (N, 2))
        Y = jnp.sin(3 * X[:, 0]) + 0.3 * jax.random.normal(jax.random.PRNGKey(2), (N,))

        take, sizes = _class_layout(N, M)
        all_idx = jnp.arange(N, dtype=jnp.int32)[None, :]
        cls_idx = _build_class_indices(X, all_idx, jnp.asarray(take))
        # Canonical partition-group layout: (cls_idx, counts) with
        # length-1 leading axes on the shared counts.
        group = (cls_idx, jnp.asarray(sizes)[None, None, :])
        ot_idx, _, _, _ = _ot_1d_kernel(Y[:, None], all_idx, (group,), (None,))

        y_np = np.asarray(Y, dtype=np.float64)
        V = 2 * np.var(y_np, ddof=1)
        cls = np.asarray(cls_idx[0])
        ref = np.zeros(2)
        for d in range(2):
            acc = 0.0
            for m in range(M):
                cond = y_np[cls[d, m][: sizes[m]]]
                acc += sizes[m] / N * ot.wasserstein_1d(y_np, cond, p=2)
            ref[d] = acc / V
        np.testing.assert_allclose(np.asarray(ot_idx[0, 0]), ref, atol=1e-4)

    def test_sinkhorn_matches_pot_and_approaches_emd(self):
        """Solver-level parity vs ot.sinkhorn2 and monotone approach to ot.emd2."""
        ot = pytest.importorskip("ot")
        from jaxgsa.optimal_transport._solver import _sinkhorn_w2

        rng = np.random.default_rng(0)
        N, P, E = 200, 40, 2
        Zu = rng.normal(size=(N, E))
        Zc = rng.normal(loc=0.5, size=(P, E))
        C = ((Zu[:, None, :] - Zc[None, :, :]) ** 2).sum(-1)
        a = np.full(N, 1.0 / N)
        b = np.full(P, 1.0 / P)
        log_b = jnp.log(jnp.full(P, 1.0 / P))

        emd = ot.emd2(a, b, C)
        prev = np.inf
        for eps in (0.05, 0.02, 0.008):
            # Pass the cost matrix's own max as the scale, so POT's
            # convention (dividing by C.max()) and ours mean the same eps.
            ref = ot.sinkhorn2(a, b, C / C.max(), reg=eps, numItermax=20000, stopThr=1e-12)
            ref *= C.max()
            ours, err = _sinkhorn_w2(
                jnp.asarray(C),
                log_b,
                jnp.asarray(eps),
                jnp.asarray(20000),
                jnp.asarray(1e-6),
                jnp.asarray(C.max()),
            )
            assert float(err) <= 1e-6
            assert float(ours) == pytest.approx(ref, rel=1e-4)
            assert float(ours) < prev  # entropic cost decreases toward exact
            prev = float(ours)
        assert prev > emd  # entropic plan cost upper-bounds the exact cost
        assert prev == pytest.approx(emd, rel=0.25)  # loose: eps=0.008 is still biased

    def test_sinkhorn_pad_invariance(self):
        """Padded target columns (zero mass, zeroed cost) change nothing.

        Pads are clamped duplicates of a real sample, so the kernel zeroes
        their cost columns before solving. This mimics that contract with
        deliberately extreme pad points, holding the scale fixed across
        both calls -- the caller's own normalizer (M3), not each cost
        matrix's own max, so an outlier pad cannot even reach the scale.
        """
        from jaxgsa.optimal_transport._solver import _sinkhorn_w2

        rng = np.random.default_rng(3)
        N, P, P_pad, E = 100, 20, 27, 3
        Zu = rng.normal(size=(N, E))
        Zc = rng.normal(size=(P, E))
        Zc_pad = np.vstack([Zc, np.full((P_pad - P, E), 50.0)])  # outlier pads
        C = ((Zu[:, None, :] - Zc[None, :, :]) ** 2).sum(-1)
        C_pad = ((Zu[:, None, :] - Zc_pad[None, :, :]) ** 2).sum(-1)
        C_pad[:, P:] = 0.0  # the kernel's mask-zeroing of pad columns
        log_b = np.full(P_pad, -np.inf)
        log_b[:P] = -np.log(P)

        args = (jnp.asarray(0.02), jnp.asarray(5000), jnp.asarray(1e-6), jnp.asarray(C.max()))
        ref, _ = _sinkhorn_w2(jnp.asarray(C), jnp.log(jnp.full(P, 1.0 / P)), *args)
        padded, _ = _sinkhorn_w2(jnp.asarray(C_pad), jnp.asarray(log_b), *args)
        assert float(padded) == pytest.approx(float(ref), abs=1e-6)

    def test_multivariate_mode_matches_pot_exact_emd(self):
        """T2: the whole 'multivariate' pipeline against POT's exact emd2.

        Nothing in the suite checked the point-cloud modes against an
        external oracle end to end (only the solver alone, above), which is
        exactly the gap M3 slipped through: the Sinkhorn regularization was
        scaled by each class's own max cost instead of the index's own
        normalizer, biasing 'ot' by tens of percent, and no test caught it.
        This drives epsilon small enough that the entropic bias is under
        the tolerance, and cross-checks jaxgsa's whole per-class weighting
        and normalization against POT's exact transport, not just the
        solver kernel.
        """
        ot = pytest.importorskip("ot")

        N, M = 300, 5
        rng = np.random.default_rng(7)
        X = rng.uniform(size=(N, 2))
        Y = np.stack(
            [
                np.sin(3 * X[:, 0]) + 0.2 * rng.normal(size=N),
                0.5 * X[:, 0] + X[:, 1] ** 2 + 0.2 * rng.normal(size=N),
            ],
            axis=1,
        )
        problem = jaxgsa.Problem(("x1", "x2"), ((0.0, 1.0), (0.0, 1.0)))
        # epsilon is deliberately small (tight entropic bias) so 20000
        # iterations do not always reach tol; that non-convergence warning
        # is expected here and not what this test checks.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=jaxgsa.JaxgsaWarning)
            result = ot_analyze(
                problem,
                jnp.asarray(X),
                jnp.asarray(Y),
                mode="multivariate",
                n_classes=M,
                epsilon=0.002,
                max_iter=20000,
                tol=1e-6,
                verbose=False,
            )

        V = 2.0 * np.var(Y, axis=0, ddof=1).sum()
        edges = np.floor(np.linspace(0.0, N, M + 1)).astype(np.int64)
        ref = np.zeros(2)
        for d in range(2):
            order = np.argsort(X[:, d])
            for m in range(M):
                cls = order[edges[m] : edges[m + 1]]
                w = cls.shape[0] / N
                a = np.full(N, 1.0 / N)
                b = np.full(cls.shape[0], 1.0 / cls.shape[0])
                C = ((Y[:, None, :] - Y[cls][None, :, :]) ** 2).sum(-1)
                ref[d] += w * ot.emd2(a, b, C)
            ref[d] /= V

        np.testing.assert_allclose(np.asarray(result.ot), ref, atol=0.01)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
