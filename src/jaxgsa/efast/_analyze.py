"""eFAST analysis: compute S1 and ST from Fourier amplitude decomposition.

For each parameter i the model output along its search curve is
Fourier-transformed. First-order indices are estimated from the power
at harmonics of the focal frequency omega_0, and total-order indices
from the complementary low-frequency content.

eFAST reports no confidence interval, and that is a property of the design
rather than a missing feature. Its resampling unit is the search curve, one
per parameter, and a curve is an ordered sweep read by a discrete Fourier
transform: removing a point does not shrink the sample, it changes what the
estimator computes. With exactly one curve per parameter there is nothing to
resample, so :func:`analyze` takes no ``n_bootstrap``. An eFAST interval needs
replicated designs drawn with different random phase shifts, which is a change
to :func:`jaxgsa.efast.sample`, not a keyword on :func:`analyze`.

Array shape conventions used throughout:
    N: number of samples per search curve (``n_per_curve``)
    D: number of input parameters
    T: number of time steps (singleton-squeezed when absent)
    K: number of output variables (singleton-squeezed when absent)

References:
    Saltelli, Tarantola & Chan (1999). Technometrics 41(1):39-56.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.batching import resolve_batch_size
from jaxgsa._core.entry import at_least, check_scalars, prepare, require
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.validation import YLayout, _prepare_Y
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.efast._result import EFASTResult
from jaxgsa.efast._sampling import EFASTSamples, _frequency_plan

# Live length-N arrays the per-slice kernel holds at once: the padded curve
# batch, the complex spectrum (two reals per element), its magnitude, and the
# one-sided power spectrum, plus headroom for the reduction temporaries. The
# automatic ``slice_chunk_size`` is sized against this, so it tracks the
# memory budget instead of a hard-coded slice count.
_EFAST_LIVE_ARRAYS = 8


def _compute_indices(
    Y_curve: Array, N: int, M: int, omega_0: int, analysis_max: int
) -> tuple[Array, Array]:
    """Compute S1 and ST for a single search curve.

    Args:
        Y_curve: Model outputs along one search curve, shape ``(N,)``.
        N: Number of samples in the curve.
        M: Interference factor.
        omega_0: Primary frequency, from the design's frequency plan.
        analysis_max: Top of the analysis band, from the same plan. It is
            wider than the band the sampler assigned carriers in; see
            :class:`jaxgsa.efast._sampling._FrequencyPlan`.

    Returns:
        A tuple ``(S1, ST)`` of scalar arrays.
    """
    # Discrete Fourier spectrum of the model output along one search curve.
    f = jnp.fft.fft(Y_curve)
    # One-sided power spectrum |F_k/N|^2, positive frequencies only. The DC
    # term at k=0 is skipped. Dividing by N before squaring keeps N**2 out of
    # the jitted kernel: as a Python int it would trace as int32, which
    # overflows for N >= 46341 (SALib divides the same way, for the same
    # reason). S1 and ST are both ratios of these values, so the normaliser
    # cancels either way.
    Sp = (jnp.abs(f[1 : (N + 1) // 2]) / N) ** 2
    V = 2.0 * jnp.sum(Sp)
    if N % 2 == 0:
        V = V + (jnp.abs(f[N // 2]) / N) ** 2

    # First-order partial variance: sum the power at the harmonics p*omega_0
    # for p = 1..M. Those frequencies carry variance attributable to the focal
    # parameter alone.
    harmonics = jnp.arange(1, M + 1) * omega_0
    D1 = 2.0 * jnp.sum(Sp[harmonics - 1])

    # Complementary variance: power at frequencies 1..analysis_max, DC
    # excluded, where Sp[k] holds frequency k+1. Everything in that band comes
    # from the non-focal parameters — the carriers the sampler assigned, and
    # the interference between them, which lands on frequencies nobody was
    # assigned. The band is therefore wider than the assigned band on purpose.
    compl_range = jnp.arange(analysis_max)
    Dt = 2.0 * jnp.sum(Sp[compl_range])

    # A constant curve has no variance to attribute, so S1 and ST are
    # 0/0. max == min is the exact test (jaxgsa's _is_constant_slice
    # semantics): a constant float32 curve leaves FFT round-off behind, so
    # V is tiny-but-nonzero and a V == 0 guard would miss it.
    constant_curve = jnp.max(Y_curve) == jnp.min(Y_curve)
    S1 = jnp.where(constant_curve, jnp.nan, D1 / V)
    ST = jnp.where(constant_curve, jnp.nan, 1.0 - Dt / V)
    return S1, ST


# ---------------------------------------------------------------------------
# Cached JIT kernels
# ---------------------------------------------------------------------------


def _row_count_message(n_rows: int, N: int, D: int, n_runs: int) -> str:
    """Build the "wrong number of Y rows" message shared by ``indices`` and ``analyze``.

    Args:
        n_rows: The number of rows ``Y`` actually has.
        N: ``n_per_curve`` from the design.
        D: Number of parameters, one search curve each.
        n_runs: The row count the design expects.

    Returns:
        The error text, naming both the mismatch and how to fix it.
    """
    return (
        f"Y has {n_rows} rows but this eFAST design requires "
        f"n_runs = n_per_curve * D = {N} * {D} = {n_runs}; "
        "evaluate the model on every row of sampling_result.samples, in order"
    )


# The closure captures concrete (N, M, omega_0), so jnp.arange() sees Python
# ints rather than JAX tracers. vmap traces the Y_curve argument only.


@lru_cache(maxsize=32)
def _get_efast_kernel(N: int, M: int, omega_0: int, analysis_max: int):
    """Build and cache a JIT-compiled eFAST kernel for one design.

    Every argument is a plain Python int, so it can serve as a cache key. The
    frequency plan itself holds a NumPy array and must not be passed here;
    read its fields instead.

    Args:
        N: Number of samples per search curve.
        M: Interference factor.
        omega_0: Primary frequency, from the design's frequency plan.
        analysis_max: Top of the analysis band, from the same plan.

    Returns:
        A compiled callable mapping a batch of curve outputs, vmapped over a
        leading curve axis, to ``(S1, ST)``.
    """

    def kernel(Y_curve: Array) -> tuple[Array, Array]:
        return _compute_indices(Y_curve, N, M, omega_0, analysis_max)

    return jax.jit(jax.vmap(kernel))


def _indices_from_3d(
    Y3: Array,
    layout: YLayout,
    D: int,
    N: int,
    M: int,
    omega_0: int,
    analysis_max: int,
    slice_chunk_size: int | None,
) -> tuple[Array, Array]:
    """Run the eFAST kernel over every output slice of a promoted ``Y``.

    This is the shared estimator body of :func:`analyze` and :func:`indices`,
    so the two can never report different numbers. Everything it branches on
    is a shape or a Python scalar, and it reads no array value on the host.

    Args:
        Y3: Model outputs promoted to ``(n_runs, T, K)``.
        layout: Rank the caller passed, used to squeeze the promoted axes back
            off the result.
        D: Number of parameters, which is also the number of search curves.
        N: Number of samples per search curve.
        M: Interference factor.
        omega_0: Primary frequency, from the design's frequency plan.
        analysis_max: Top of the analysis band, from the same plan.
        slice_chunk_size: Output slices per vmapped batch, or ``None`` to
            derive one from the memory budget.

    Returns:
        ``(S1, ST)`` at the caller's own rank: ``(D,)``, ``(K, D)`` or
        ``(T, K, D)``.
    """
    _, T, K = Y3.shape

    # Split the contiguous search curves into (D, N, T, K).
    Y_reshaped = Y3.reshape(D, N, T, K)

    if layout is YLayout.SCALAR:
        # Scalar path: squeeze the trailing singletons and vmap over the D
        # curves only.
        Y_curves = Y_reshaped[:, :, 0, 0]  # (D, N)
        kernel = _get_efast_kernel(N, M, omega_0, analysis_max)
        return kernel(Y_curves)  # each (D,)

    # Batched path: flatten (D, T, K) into a single vmap axis.
    Y_batched = Y_reshaped.transpose(0, 2, 3, 1).reshape(D * T * K, N)

    total = D * T * K
    # One vmapped slice costs a handful of length-N arrays, so the chunk
    # follows the transient-memory budget when the caller gives no size.
    itemsize = jnp.dtype(jnp.result_type(Y_batched.dtype, jnp.float32)).itemsize
    cs = resolve_batch_size(_EFAST_LIVE_ARRAYS * N * itemsize, total, slice_chunk_size)
    batched = _get_efast_kernel(N, M, omega_0, analysis_max)

    s1_parts: list[Array] = []
    st_parts: list[Array] = []
    for start in range(0, total, cs):
        end = min(start + cs, total)
        actual = end - start
        batch = Y_batched[start:end]
        if actual < cs:
            # dtype= keeps the pad in the batch's own dtype: an unannotated
            # jnp.zeros is float64 under x64, which would promote the whole
            # trailing chunk and re-trace the kernel at a second dtype.
            pad = jnp.zeros((cs - actual, N), dtype=Y_batched.dtype)
            batch = jnp.concatenate([batch, pad], axis=0)
        s1_chunk, st_chunk = batched(batch)
        s1_parts.append(s1_chunk[:actual])
        st_parts.append(st_chunk[:actual])

    S1_flat = jnp.concatenate(s1_parts)  # (D*T*K,)
    ST_flat = jnp.concatenate(st_parts)

    # Reshape to (D, T, K), then transpose to the (T, K, D) convention.
    S1 = S1_flat.reshape(D, T, K).transpose(1, 2, 0)
    ST = ST_flat.reshape(D, T, K).transpose(1, 2, 0)

    # Drop the singleton dims that _prepare_Y inserted.
    return layout.squeeze(S1), layout.squeeze(ST)


def indices(
    sampling_result: EFASTSamples,
    Y: Array,
    *,
    slice_chunk_size: int | None = None,
) -> tuple[Array, Array]:
    """Compute eFAST indices as plain arrays, with no diagnostics.

    This is the transformable core of :func:`analyze`. It runs the same
    Fourier decomposition on the same data and returns the same numbers, but
    it does nothing else: no non-finite check, no zero-variance warning, no
    out-of-range warning, no :class:`jaxgsa.efast.EFASTResult`, and no read of
    any array value on the host. So it composes with ``jax.jit``, ``jax.vmap``,
    ``jax.grad`` and ``jax.jacrev``, which :func:`analyze` cannot, because a
    policy decision needs a concrete value and a tracer has none.

    Use :func:`analyze` for ordinary analysis. Nothing here checks the outputs,
    so a single NaN silently turns every index on that curve into NaN. A
    constant curve (``max == min``) yields NaN from inside the kernel rather
    than a warning.

    Like :func:`analyze`, this reports no confidence interval, and for a
    reason that no keyword can fix: see the module docstring and
    :func:`analyze`.

    Tier T4 (behavioural contract): the returned arrays must equal the ``S1``
    and ``ST`` fields of ``analyze``'s result on clean outputs, and the
    function must survive ``jit``, ``vmap`` and ``jit(jacrev(...))``. Checked
    in ``tests/test_efast.py``.

    Args:
        sampling_result: The design from :func:`jaxgsa.efast.sample`, used
            only for its curve layout: ``n_per_curve``, ``M`` and the
            parameter count.
        Y: Model outputs evaluated at each row of
            ``sampling_result.samples``, in the same row order. Shapes are
            those :func:`analyze` accepts: ``(n_runs,)``, ``(n_runs, K)`` or
            ``(n_runs, T, K)``.
        slice_chunk_size: Number of output slices per vmapped batch. ``None``
            (default) derives one from the active memory budget. Lower it if
            you hit device out-of-memory errors. It changes no index.

    Returns:
        ``(S1, ST)``, each shaped ``(D,)``, ``(K, D)`` or ``(T, K, D)`` to
        mirror the layout of ``Y``. The shapes are those ``analyze`` reports.

    Raises:
        ValueError: If ``Y`` has an invalid rank, if its leading dimension
            does not equal ``sampling_result.n_runs``, or if
            ``slice_chunk_size`` is given and is below 1.
    """
    problem = sampling_result.problem
    D = problem.num_vars
    N = sampling_result.n_per_curve
    M = sampling_result.M

    Y_arr = jnp.asarray(Y)
    rank_ok = Y_arr.ndim in (1, 2, 3)
    check_scalars(
        (
            at_least("slice_chunk_size", slice_chunk_size, 1),
            require(rank_ok, f"Y must have 1, 2 or 3 dimensions, got {Y_arr.ndim}"),
            require(
                not rank_ok or Y_arr.shape[0] == sampling_result.n_runs,
                _row_count_message(
                    Y_arr.shape[0] if Y_arr.ndim else 0, N, D, sampling_result.n_runs
                ),
            ),
        )
    )

    # Rebuild the frequency plan sample() used, from the design metadata that
    # travels inside EFASTSamples.
    plan = _frequency_plan(D, N, M)
    Y3, layout = _prepare_Y(Y_arr)
    return _indices_from_3d(Y3, layout, D, N, M, plan.omega_0, plan.analysis_max, slice_chunk_size)


def analyze(
    sampling_result: EFASTSamples,
    Y: Array,
    *,
    slice_chunk_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
) -> EFASTResult:
    """Compute eFAST first- and total-order sensitivity indices.

    eFAST attributes output variance to each parameter from the Fourier
    spectrum of the model output along that parameter's search curve. It is a
    structured, deterministic alternative to Monte Carlo Sobol estimation. It
    yields S1 and ST, but no second-order indices, from ``n_per_curve * D``
    model runs.

    ``Y`` must be the model evaluated row by row on
    ``sampling_result.samples``, with the rows in the same order. The design
    metadata ``n_per_curve``, ``M``, and ``problem`` is read from
    ``sampling_result``, so it can never be mismatched with the sampling step.

    **There is no confidence interval, and there is no ``n_bootstrap``
    keyword.** This is a property of the eFAST design, not an omission. The
    resampling unit is the search curve, and a curve is an ordered sweep read
    by a discrete Fourier transform: removing a point does not shrink the
    sample, it changes what the estimator computes. The design holds exactly
    one curve per parameter, so there is nothing to resample. To put an
    interval on an eFAST index, draw several designs with different random
    phase shifts and compare the indices across those designs. That is a
    change to :func:`jaxgsa.efast.sample`, not a keyword here.

    Use :func:`indices` when you need the same numbers inside ``jit``,
    ``vmap`` or ``jacrev``.

    Args:
        sampling_result: Design returned by ``jaxgsa.efast.sample()``,
            carrying the sample matrix plus ``n_per_curve``, ``M``, and the
            problem.
        Y: Model outputs evaluated at each row of
            ``sampling_result.samples``, in the same row order. Accepted
            shapes, where ``n_runs`` is
            ``sampling_result.n_runs = n_per_curve * D``:
            - ``(n_runs,)`` for scalar output
            - ``(n_runs, K)`` for K output variables
            - ``(n_runs, T, K)`` for K outputs over T time steps
        slice_chunk_size: Maximum number of output slices to process in one
            vmapped batch. It caps peak device memory for a large ``T * K``. A
            smaller value trades speed for memory. ``None`` (default) derives
            one from the active memory budget
            (:func:`jaxgsa.config.set_memory_budget`). An explicit value must
            be at least 1.
        on_invalid: What to do about non-finite model outputs. Only
            ``"raise"`` (the default) and ``"propagate"`` are available.
            ``"drop"`` raises, because a search curve is an ordered sweep read
            by a discrete Fourier transform: removing a point changes what the
            estimator computes instead of shrinking the sample. See
            :mod:`jaxgsa._core.invalid`.
        verbose: If ``True`` (default), print a short summary to stdout: the
            problem and the data, the wall-clock timing, and the top
            parameters by ``ST``. Pass ``False`` for a silent run.

    Returns:
        An ``EFASTResult`` with ``S1`` and ``ST``, shape ``(D,)`` /
        ``(K, D)`` / ``(T, K, D)``, mirroring the layout of ``Y``, plus the
        ``invalid`` report.

    Raises:
        ValueError: If ``Y``'s leading dimension does not equal
            ``sampling_result.n_runs``, or ``Y`` has an invalid rank; if
            ``on_invalid`` is not one of the three policies or is ``"drop"``;
            if ``slice_chunk_size`` is given and is not a positive integer; or
            if the sample holds a non-finite value under
            ``on_invalid="raise"``.
    """
    from jaxgsa.efast import SPEC

    problem = sampling_result.problem
    M = sampling_result.M
    D = problem.num_vars
    N = sampling_result.n_per_curve

    Y_arr = jnp.asarray(Y)
    # A row-count complaint reads better in the design's own terms than in the
    # generic one, so it is phrased here and handed to prepare as a check.
    rows_match = Y_arr.ndim not in (1, 2, 3) or Y_arr.shape[0] == sampling_result.n_runs

    ctx = prepare(
        SPEC,
        problem,
        Y_arr,
        on_invalid=on_invalid,
        checks=(
            at_least("slice_chunk_size", slice_chunk_size, 1),
            require(
                rows_match,
                _row_count_message(
                    Y_arr.shape[0] if Y_arr.ndim else 0, N, D, sampling_result.n_runs
                ),
            ),
        ),
        n_expected=sampling_result.n_runs,
        # The design lays the D search curves out contiguously, N rows each.
        # A curve cannot be dropped, so `keep` is always all-True; the report
        # still names the curve to investigate.
        n_units=D,
        unit_of_row=np.repeat(np.arange(D), N),
        # One search curve is one unit of N contiguous rows.
        unit_stride=N,
    )
    invalid = ctx.invalid
    Y = ctx.Y3

    # Rebuild the frequency plan sample() used, from the design metadata that
    # travels inside EFASTSamples. Both sides read the same numbers, so the
    # harmonics line up by construction rather than by agreement.
    plan = _frequency_plan(D, N, M)
    omega_0 = plan.omega_0

    _, T, K = Y.shape
    Y_reshaped = Y.reshape(D, N, T, K)

    # Per-curve zero-variance check. The global zero-variance warning that
    # prepare() ran (see _core/entry.py) can miss a curve where a single
    # parameter has no effect, giving V=0 on that curve alone and a silent
    # NaN index.
    per_curve_var = jnp.var(Y_reshaped, axis=1)  # (D, T, K)
    n_zero_curves = int(jnp.sum(per_curve_var == 0))
    if n_zero_curves > 0:
        warnings.warn(
            f"jaxgsa.efast: {n_zero_curves} search-curve/output slice(s) have zero "
            "variance — corresponding indices will be NaN",
            stacklevel=2,
            category=JaxgsaWarning,
        )

    t0 = _verbose.tic()
    S1, ST = _indices_from_3d(Y, ctx.layout, D, N, M, omega_0, plan.analysis_max, slice_chunk_size)

    result = EFASTResult(
        S1=S1,
        ST=ST,
        problem=problem,
        invalid=invalid,
        omega_0=omega_0,
        M=M,
    )

    if verbose:
        elapsed = _verbose.stop(t0, result.S1, result.ST)
        # Same resolver the kernel loop ran, on the same shapes: cheap
        # arithmetic, reported rather than re-derived by hand.
        total = D * T * K
        itemsize = jnp.dtype(jnp.result_type(Y.dtype, jnp.float32)).itemsize
        cs = resolve_batch_size(_EFAST_LIVE_ARRAYS * N * itemsize, total, slice_chunk_size)
        origin = "user-set" if slice_chunk_size is not None else "resolved from the memory budget"
        chunk_note = f"slice_chunk_size: {cs} ({origin})"
        _verbose.analysis_summary(
            method="jaxgsa.efast.analyze",
            problem=problem,
            n_runs=int(Y.shape[0]),
            T=T,
            K=K,
            invalid=invalid,
            timings=[("estimator (includes compile on the first call)", elapsed)],
            notes=[chunk_note, f"omega_0: {omega_0}, M: {M}"],
            index_name="ST",
            values=result.ST,
        )
    return result
