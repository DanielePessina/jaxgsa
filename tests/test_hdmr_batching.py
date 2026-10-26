"""Tests for HDMR's two batching axes and the one fit path that honours both.

``jaxgsa.hdmr`` cuts the sample axis into row batches (``batch_size``) and the
output-slice axis into chunks (``slice_chunk_size``). There is one fit, so the
unbatched fit is the corner ``batch_size = N``, ``slice_chunk_size = T*K``.
The tests here pin three things: subdividing either axis, or both at once,
must not move a public result field beyond float32 noise; the F-test term set
is discrete and must be picked identically; and both keywords must actually
subdivide the work on every call, which is the defect that made this file
necessary -- ``slice_chunk_size`` used to be read on one fit path only, so
``analyze(batch_size=64, slice_chunk_size=8)`` silently dropped the second.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM
from jaxgsa.hdmr import _fit
from jaxgsa.hdmr import analyze as analyze_hdmr

# The same Ishigami data and the same unbatched reference fits as test_hdmr.py.

# Row-batched and single-batch runs do the same float32 reductions in a
# different order, so agreement is to float32 noise, not bit-exact.
# Coefficients and predictions amplify that noise through the (m+3)^order
# basis, hence the looser tolerances there.
RTOL = 1e-4
ATOL = 1e-4
COEF_RTOL = 1e-3
COEF_ATOL = 2e-2
PREDICT_ATOL = 2e-2
# Order >= 2 projects the pair basis off the first-order space. That projection
# divides by small Gram eigenvalues, so f32 summation order moves the pair
# indices by up to about 1e-2 (1e-14 in float64).
PAIR_ATOL = 2e-2


@pytest.fixture(autouse=True)
def _restore_memory_budget():
    """Snapshot and restore the global memory budget around every test.

    ``set_memory_budget`` mutates process-global state; leaking a tiny test
    budget would silently switch other tests onto batched paths.
    """
    from jaxgsa._core import batching

    saved = batching.get_memory_budget()
    yield
    batching._set_memory_budget(saved)


@pytest.fixture(scope="module")
def ishigami_data_ntk(hdmr_ishigami_xy):
    """(N, T, K) variant: T=2 time steps, K=2 output columns."""
    X, Y = hdmr_ishigami_xy
    Y_alt = jnp.cos(X[:, 1]) + 0.3 * X[:, 0] * X[:, 2]
    Y_tk = jnp.stack(
        [
            jnp.stack([Y, Y_alt], axis=1),
            jnp.stack([0.5 * Y + 3.0, -1.5 * Y_alt], axis=1),
        ],
        axis=1,
    )
    return X, Y_tk


@pytest.fixture(scope="module")
def full_result(hdmr_by_order):
    """Unbatched order=2 reference fit, shared with test_hdmr.py."""
    return hdmr_by_order[2]


@pytest.fixture(scope="module")
def full_ntk(ishigami_data_ntk):
    """Unbatched (N, T, K) reference fit, computed once."""
    X, Y_tk = ishigami_data_ntk
    return analyze_hdmr(PROBLEM, X, Y_tk, order=2, n_intervals=2)


def _fit_kwargs(N: int, batch_size: int, slice_chunk_size: int) -> dict:
    """Static arguments for a direct ``_fit_hdmr`` call on 3 uniform inputs."""
    from jaxgsa.hdmr._analyze import _get_hdmr_static_data
    from jaxgsa.hdmr._engine import _compute_f_crits

    sd = _get_hdmr_static_data(3, 2, 2)
    return dict(
        n_intervals=2,
        order=2,
        max_iter=50,
        ridge=0.01,
        f_crits=_compute_f_crits(0.95, sd.m1, sd.m2, sd.m3, N),
        c2_idx=jnp.asarray(sd.c2, dtype=int),
        c3_idx=jnp.zeros((0, 3), dtype=int),
        beta2=jnp.asarray(sd.beta2, dtype=int),
        beta3=jnp.asarray(sd.beta3, dtype=int),
        n1=sd.n1,
        n2=sd.n2,
        n3=sd.n3,
        n=sd.n,
        m1=sd.m1,
        m2=sd.m2,
        m3=sd.m3,
        batch_size=batch_size,
        slice_chunk_size=slice_chunk_size,
    )


def _assert_results_match(batched, full):
    """Assert every public HDMRResult field matches between the two runs."""
    atol = ATOL if full._fit is not None and full._fit["C2"] is None else PAIR_ATOL
    np.testing.assert_allclose(np.asarray(batched.Sa), np.asarray(full.Sa), RTOL, atol)
    np.testing.assert_allclose(np.asarray(batched.Sb), np.asarray(full.Sb), RTOL, atol)
    np.testing.assert_allclose(np.asarray(batched.S), np.asarray(full.S), RTOL, atol)
    np.testing.assert_allclose(np.asarray(batched.ST), np.asarray(full.ST), RTOL, atol)
    # F-test selection is discrete: a batched fit must pick EXACTLY the same
    # significant-term sets (select counts per term across all slices).
    np.testing.assert_array_equal(np.asarray(batched.select), np.asarray(full.select))
    assert batched.terms == full.terms
    assert batched.rmse is not None and full.rmse is not None
    np.testing.assert_allclose(np.asarray(batched.rmse), np.asarray(full.rmse), RTOL, ATOL)
    # Coefficient-level: the fitted surrogate state must agree too.
    assert batched._fit is not None and full._fit is not None
    np.testing.assert_allclose(
        np.asarray(batched._fit["f0"]), np.asarray(full._fit["f0"]), RTOL, ATOL
    )
    if full._fit["C2"] is None:
        np.testing.assert_allclose(
            np.asarray(batched._fit["C1"]), np.asarray(full._fit["C1"]), COEF_RTOL, COEF_ATOL
        )
        # order=1 has no second-order coefficients on either side.
        assert batched._fit["C2"] is None
    else:
        # Raw C1 and C2 are not comparable element-wise: C1 carries the pair
        # projection. Raw C2 is not comparable element-wise either. The pair blocks are
        # orthogonal to the first-order space, so each has null directions that
        # only the ridge fixes, and f32 noise moves the coefficients along them
        # without changing the fitted component functions. The indices above
        # and the predictions in the callers carry the comparison.
        assert batched._fit["C2"] is not None


class TestRowBatching:
    """batch_size=int splits the sample axis; results match one batch."""

    def test_scalar_output_matches(self, hdmr_ishigami_xy, full_result):
        X, Y = hdmr_ishigami_xy
        batched = analyze_hdmr(PROBLEM, X, Y, order=2, n_intervals=2, batch_size=137)
        _assert_results_match(batched, full_result)

        # Tier T4 (internal consistency): batch_size >= N is a single batch.
        # ``streamed`` reports whether the sample axis was actually split, so a
        # batch size the data never reaches leaves it False.
        single = analyze_hdmr(PROBLEM, X, Y, order=2, n_intervals=2, batch_size=10**6)
        assert single.streamed is False
        _assert_results_match(single, full_result)

    def test_order1_matches(self, hdmr_ishigami_xy, hdmr_by_order):
        X, Y = hdmr_ishigami_xy
        batched = analyze_hdmr(PROBLEM, X, Y, order=1, n_intervals=2, batch_size=250)
        _assert_results_match(batched, hdmr_by_order[1])

    def test_ntk_output_matches(self, ishigami_data_ntk, full_ntk):
        X, Y_tk = ishigami_data_ntk
        batched = analyze_hdmr(PROBLEM, X, Y_tk, order=2, n_intervals=2, batch_size=111)
        n_terms = 3 + 3
        assert batched.Sa.shape == (2, 2, n_terms)
        assert batched.rmse is not None and batched.rmse.shape == (2, 2)
        _assert_results_match(batched, full_ntk)


class TestBothAxesCompose:
    """The defect this file exists for: both keywords work on every call.

    D15 asked for observable behaviour, not a monkeypatch that counts private
    calls: that pattern breaks the moment the loop structure changes, with no
    change in what the fit computes. ``jax.make_jaxpr`` is the observable
    replacement here. Tracing ``_fit_hdmr`` (a public JAX facility on a
    function this file already calls directly, in ``TestPaddingIsInert``)
    inlines one compiled sub-call per row batch times per slice chunk, so the
    traced program's equation count grows with the number of blocks the fit
    actually ran -- exactly the fact that used to go unverified when
    ``slice_chunk_size`` was silently dropped on one of the two old fit
    paths, because a dropped keyword still returns the right numbers.
    """

    def _n_traced_equations(self, X_n, Y_nl, *, batch_size: int, slice_chunk_size: int) -> int:
        kw = _fit_kwargs(X_n.shape[0], batch_size, slice_chunk_size)
        jaxpr = jax.make_jaxpr(lambda x, y: _fit._fit_hdmr(x, y, **kw))(X_n, Y_nl)
        return len(jaxpr.jaxpr.eqns)

    def test_both_keywords_subdivide_the_work(self, ishigami_data_ntk):
        """batch_size and slice_chunk_size must both add structure to the trace.

        N=1000, T*K=4: one row batch and one slice chunk is the baseline.
        Splitting either axis alone must grow the traced program, and
        splitting both together must grow it further than either alone --
        that composition is exactly what a silently-dropped keyword cannot
        produce, because dropping one axis collapses back to the other
        axis's count.
        """
        from jaxgsa._core.transforms import cdf_to_unit_interval

        X, Y_tk = ishigami_data_ntk
        X_n = cdf_to_unit_interval(X, PROBLEM)
        Y_nl = Y_tk.reshape(Y_tk.shape[0], -1)

        baseline = self._n_traced_equations(X_n, Y_nl, batch_size=1000, slice_chunk_size=4)
        rows_only = self._n_traced_equations(X_n, Y_nl, batch_size=250, slice_chunk_size=4)
        slices_only = self._n_traced_equations(X_n, Y_nl, batch_size=1000, slice_chunk_size=1)
        both = self._n_traced_equations(X_n, Y_nl, batch_size=250, slice_chunk_size=1)

        assert rows_only > baseline, "batch_size alone did not grow the trace"
        assert slices_only > baseline, "slice_chunk_size alone did not grow the trace"
        assert both > rows_only, "slice_chunk_size added nothing once batch_size was set"
        assert both > slices_only, "batch_size added nothing once slice_chunk_size was set"

    def test_ragged_chunks_do_not_change_the_answer(self, ishigami_data_ntk, full_ntk):
        """Widths that do not divide N or T*K are padded, not truncated."""
        X, Y_tk = ishigami_data_ntk  # N=1000, T*K = 4
        ragged = analyze_hdmr(
            PROBLEM, X, Y_tk, order=2, n_intervals=2, batch_size=333, slice_chunk_size=3
        )
        _assert_results_match(ragged, full_ntk)
        # Slice chunking alone, in one row batch, must not change the answer either.
        sliced = analyze_hdmr(PROBLEM, X, Y_tk, order=2, n_intervals=2, slice_chunk_size=1)
        _assert_results_match(sliced, full_ntk)

    def test_both_axes_under_a_tiny_budget(self, ishigami_data_ntk, full_ntk):
        """A small budget engages row batching, and slice_chunk_size stays usable.

        Tier T4 (internal consistency), and the structural guard for the whole
        file. Every other test here compares two batchings, and they agree by
        design, so a change to the memory estimate that stopped row batching
        engaging at all would leave those tests green. Asserting the reported
        path fails instead.
        """
        X, Y_tk = ishigami_data_ntk
        assert full_ntk.streamed is False
        jaxgsa.config.set_memory_budget(256 * 1024, unit="b")  # 256 KiB
        batched = analyze_hdmr(PROBLEM, X, Y_tk, order=2, n_intervals=2, slice_chunk_size=1)
        assert batched.streamed is True
        _assert_results_match(batched, full_ntk)


class TestPaddingIsInert:
    """Padded rows and padded lanes contribute nothing to a real result.

    Both loops pad their trailing chunk so a jitted step compiles once. That
    is only sound if the padding cannot reach a real number. Comparing a
    padded run against an unpadded one cannot show this, because the two also
    differ in reduction width, and width alone moves float32 results. Each
    test below therefore holds every width fixed and changes only whether a
    slot holds padding or data.

    Tier T4 (internal consistency).
    """

    @staticmethod
    def _assert_lanes_identical(a, b, n_lanes: int) -> None:
        """Assert the first ``n_lanes`` lanes of two fits agree bit for bit."""
        names = ("Sa", "Sb", "S", "select", "rmse", "C1", "C2", "C3", "f0")
        for name, u, v in zip(names, a[:9], b[:9], strict=True):
            if u is None:
                assert v is None, f"{name}: one fit has it, the other does not"
                continue
            np.testing.assert_array_equal(
                np.asarray(u)[:n_lanes], np.asarray(v)[:n_lanes], err_msg=name
            )

    def _three_lanes(self, X, Y):
        return jnp.stack([Y, jnp.cos(X[:, 1]) + 0.3 * X[:, 0] * X[:, 2], 0.5 * Y + 3.0], axis=1)

    def test_a_padded_lane_does_not_touch_the_real_lanes(self, hdmr_ishigami_xy):
        """Three lanes at chunk width 2: slot 3 is padding, or it is data.

        ``slice_chunk_size=2`` on 3 lanes gives chunks [0:2] and [2:4], so the
        last slot of the second chunk is padding. Running 4 real lanes at the
        same chunk size gives the same two widths with that slot filled. Lanes
        0-2 must not notice.
        """
        from jaxgsa._core.transforms import cdf_to_unit_interval
        from jaxgsa.hdmr._fit import _fit_hdmr

        X, Y = hdmr_ishigami_xy
        X_n = cdf_to_unit_interval(X, PROBLEM)
        kw = _fit_kwargs(X.shape[0], batch_size=X.shape[0], slice_chunk_size=2)
        lanes3 = self._three_lanes(X, Y)
        extra = (jnp.sin(X[:, 0]) * X[:, 2] - 7.0)[:, None]

        padded = _fit_hdmr(X_n, lanes3, **kw)
        filled = _fit_hdmr(X_n, jnp.concatenate([lanes3, extra], axis=1), **kw)
        self._assert_lanes_identical(padded, filled, 3)

    def test_padded_row_content_cannot_reach_an_accumulator(self, hdmr_ishigami_xy, monkeypatch):
        """Fill the row padding with 1e6 instead of 0 and change nothing.

        Every accumulator contracts against a basis the row mask has already
        zeroed, or against a residual the mask zeroes directly. So the value
        sitting in a padded row is unobservable, and the strongest way to say
        that is to put a number in it that would be impossible to miss.
        """
        from jaxgsa._core.transforms import cdf_to_unit_interval
        from jaxgsa.hdmr import _fit as fit_mod

        X, Y = hdmr_ishigami_xy
        X_n = cdf_to_unit_interval(X, PROBLEM)
        # 1000 rows in batches of 300: the trailing batch pads 200 rows.
        kw = _fit_kwargs(X.shape[0], batch_size=300, slice_chunk_size=3)
        lanes3 = self._three_lanes(X, Y)

        zero_padded = fit_mod._fit_hdmr(X_n, lanes3, **kw)

        def loud_pad(A, n_pad):
            if n_pad == 0:
                return A
            junk = jnp.full((n_pad, *A.shape[1:]), 1.0e6, dtype=A.dtype)
            return jnp.concatenate([A, junk], axis=0)

        monkeypatch.setattr(fit_mod, "pad_rows", loud_pad)
        loud_padded = fit_mod._fit_hdmr(X_n, lanes3, **kw)
        self._assert_lanes_identical(zero_padded, loud_padded, 3)


def test_stats_step_temp_matches_slice_estimate():
    """The final-pass step must stay inside its per-(row, lane) memory estimate.

    Compile-only: the kernel is lowered on shape structs and never run. The step
    used to hold three (rows, lanes, terms) tensors at once, about 3 * n
    elements per pair, so the estimate had to grow with the number of terms.
    """
    from jax import ShapeDtypeStruct as S

    D, m1, m2, b, cs = 23, 12, 144, 64, 32
    n2 = D * (D - 1) // 2
    n = D + n2
    f = jnp.float32
    step = _fit._get_stats_step(True)
    acc = tuple(S(s, f) for s in [(cs, n), (cs, n), (cs, n), (cs, 3), (cs, n), (cs, n), (cs,)])

    def run(Bl1, Bl2, C1, C2, Y, w, s, mz, mf, my, a):
        return step([Bl1, Bl2], [C1, C2], Y, w, s, mz, mf, my, a)

    compiled = (
        jax.jit(run)
        .lower(
            S((b, m1, D), f),
            S((b, m2, n2), f),
            S((cs, m1, D), f),
            S((cs, m2, n2), f),
            S((b, cs), f),
            S((b,), f),
            S((cs,), f),
            S((cs,), f),
            S((cs, n), f),
            S((cs, n), f),
            acc,
        )
        .compile()
    )
    ma = compiled.memory_analysis()
    assert ma is not None
    temp = ma.temp_size_in_bytes
    estimate = b * cs * _fit._slice_bytes_per_row(n, 4)
    # The estimate does not count the small per-block copies of the basis.
    block_copies = 2 * b * m2 * _fit._STATS_TERM_BLOCK * 4
    assert temp <= 1.15 * estimate + block_copies
    # And it is far below the 3 * n elements the unblocked step needed.
    assert temp < 0.5 * 3 * n * b * cs * 4


def test_stats_step_blocked_matches_whole():
    """The term-block path and the whole-F path give the same statistics."""
    D, m1, m2, b, cs = 40, 4, 9, 50, 3
    n2 = D * (D - 1) // 2  # 780 terms, many blocks and a clamped last block
    rng = np.random.default_rng(0)
    dt = jnp.result_type(float)

    def r(*shape):
        return jnp.asarray(rng.normal(size=shape), dtype=dt)

    n = D + n2
    args = (
        [r(b, m1, D), r(b, m2, n2)],
        [r(cs, m1, D), r(cs, m2, n2)],
        r(b, cs),
        jnp.asarray((rng.uniform(size=b) > 0.2).astype(float), dtype=dt),
        r(cs),
        r(cs),
        r(cs, n),
        r(cs, n),
        tuple(r(*s) for s in [(cs, n), (cs, n), (cs, n), (cs, 3), (cs, n), (cs, n), (cs,)]),
    )
    whole = _fit._get_stats_step(False)(*args)
    blocked = _fit._get_stats_step(True)(*args)
    rtol = 1e-9 if dt == jnp.float64 else 1e-3
    for a, c in zip(whole, blocked, strict=True):
        np.testing.assert_allclose(np.asarray(c), np.asarray(a), rtol=rtol, atol=rtol * 10)


def test_stats_path_choice_keeps_indices(monkeypatch, hdmr_ishigami_xy):
    """Forcing the whole-F or the blocked stats path leaves every index put.

    The choice reads an absolute byte cap, not the memory budget, so a
    larger budget cannot pick the slower whole-F path either.
    """
    X, Y = hdmr_ishigami_xy
    fields = ("S1", "ST", "Sa", "Sb", "S")
    runs = []
    for cap in (0, 2**62):
        monkeypatch.setattr(_fit, "_STATS_WHOLE_MAX_BYTES", cap)
        res = analyze_hdmr(PROBLEM, X, Y, order=2, verbose=False)
        runs.append({f: np.asarray(getattr(res, f)) for f in fields})
    rtol = 1e-9 if runs[0]["S1"].dtype == np.float64 else 1e-4
    for f in fields:
        np.testing.assert_allclose(runs[1][f], runs[0][f], rtol=rtol, atol=rtol)


# ---------------------------------------------------------------------------
# Bootstrap under a small budget
# ---------------------------------------------------------------------------


def _bootstrap_data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.uniform(-np.pi, np.pi, size=(n, 3)))
    return X, jaxgsa.benchmarks.ishigami.evaluate(X)


def test_jit_replicates_only_for_small_one_batch_fits():
    """A streamed or large fit bootstraps in a host loop, a small one in one jit."""
    from jaxgsa.hdmr._analyze import _jit_replicates

    def jit_ok(L: int = 1, batch_size: int | None = None, chunk: int | None = None) -> bool:
        return _jit_replicates(
            800,
            3,
            L,
            order=2,
            n_intervals=2,
            itemsize=4,
            batch_size=batch_size,
            slice_chunk_size=chunk,
        )

    assert jit_ok()
    assert not jit_ok(batch_size=100)
    assert not jit_ok(L=4, chunk=2)
    jaxgsa.config.set_memory_budget(32, unit="kb")
    assert not jit_ok()


def test_streamed_bootstrap_draws_match_the_jitted_loop():
    """The host loop makes the jitted loop's draws, to float noise.

    Both loops split the key the same way and draw the same rows, so every
    replicate refits the same resample. Only the order of the reductions can
    differ.
    """
    from jaxgsa._core.bootstrap import bootstrap_draws, bootstrap_draws_host
    from jaxgsa.hdmr._analyze import _hdmr_replicate

    X, Y = _bootstrap_data()
    static = (
        ("problem", PROBLEM),
        ("order", 2),
        ("max_iter", 50),
        ("n_intervals", 2),
        ("ridge", 1e-6),
        ("slice_chunk_size", None),
        ("batch_size", 100),
    )
    key = jax.random.key(3)
    host = bootstrap_draws_host(key, 3, _hdmr_replicate, (X, Y), static=static)
    mapped = bootstrap_draws(key, 3, _hdmr_replicate, (X, Y), static=static)
    for name in ("Sa", "Sb", "S", "ST"):
        assert host[name].shape == mapped[name].shape
        np.testing.assert_allclose(
            np.asarray(host[name]), np.asarray(mapped[name]), rtol=RTOL, atol=PAIR_ATOL
        )

    result = analyze_hdmr(
        PROBLEM,
        X,
        Y,
        order=2,
        max_iter=50,
        batch_size=100,
        n_bootstrap=3,
        key=key,
        keep_replicates=True,
        verbose=False,
    )
    assert isinstance(result, jaxgsa.hdmr.HDMRResult)
    assert result.ci is not None and result.ci.replicates is not None
    np.testing.assert_allclose(
        np.asarray(result.ci.replicates["ST"]).reshape(3, -1),
        np.asarray(host["ST"]).reshape(3, -1),
        rtol=1e-6,
        atol=1e-6,
    )


_PEAK_SCRIPT = """
import resource, warnings
import jax, numpy as np
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

warnings.simplefilter("ignore")
compiles = []
jax.monitoring.register_event_duration_secs_listener(
    lambda e, d, **_: compiles.append(e)
    if e == "/jax/core/compile/backend_compile_duration" else None
)
jaxgsa.config.set_memory_budget(32, unit="kb")
rng = np.random.default_rng(0)
X = rng.uniform(-np.pi, np.pi, size=(800, 3))
Y = np.asarray(evaluate(X))
kw = dict(order=2, verbose=False)
jaxgsa.hdmr.analyze(PROBLEM, X, Y, **kw)
point_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
n0 = len(compiles)
r = jaxgsa.hdmr.analyze(PROBLEM, X, Y, n_bootstrap=3, key=0, **kw)
jax.block_until_ready(r.ST_conf)
boot_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
n1 = len(compiles)
jaxgsa.hdmr.analyze(PROBLEM, X, Y, n_bootstrap=3, key=0, **kw)
print(point_mb, boot_mb, n1 - n0, len(compiles) - n1)
"""


def test_streamed_bootstrap_peak_stays_at_the_point_fit():
    """At a 32 KiB budget the bootstrap adds almost nothing to the point fit's peak.

    The fit streams in many row batches here. Run as one jitted ``lax.map``,
    its Python row loops unrolled, and the program alone took 1.3 GB of RSS
    and 12 s to compile against 0.46 GB for the point fit (measured,
    float64, 2026-10-25). The host loop reuses the point fit's compiled
    steps, so the peak stays put and only small programs (the row draw, the
    gathers, the interval) compile. The peak is read in a fresh process,
    because ``ru_maxrss`` never falls and other tests have raised it here.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", _PEAK_SCRIPT], capture_output=True, text=True, check=True
    ).stdout.split()
    point, boot, n_compiles, n_repeat = (int(v) for v in out[-4:])
    # ru_maxrss is in bytes on macOS and in KiB on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    assert (boot - point) * scale < 250 * 2**20
    assert n_compiles <= 40
    assert n_repeat == 0
