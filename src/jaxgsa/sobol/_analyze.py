"""Main Sobol sensitivity analysis computation using JAX.

This module implements the Saltelli sampling-based Sobol variance
decomposition. It splits the model outputs Y into the base matrices A and B
and their cross-matrices AB (plus BA for second order). It then computes
first-order (S1), total-order (ST), and optionally second-order (S2) Sobol
indices.

The estimator formulas live in :mod:`jaxgsa.sobol._estimators`, one function
per named ``estimator=``. This module reaches them through exactly one seam,
``_indices_from_expanded``. ``analyze`` runs the host-side diagnostics and
wraps that seam in a result object; ``indices`` calls it with nothing around
it, so it stays traceable by ``jit``, ``vmap`` and ``jacrev``. The bootstrap
path resamples through the same named estimator, so an interval always
describes the quantity at its centre.

Array shape conventions used throughout:
    N: number of base Sobol samples (base_n after cleaning)
    D: number of input parameters
    T: number of time steps (singleton-squeezed when absent)
    K: number of output variables (singleton-squeezed when absent)
    R: number of bootstrap resamples
    step: rows per Saltelli group, 2D+2 (second order) or D+2 (first only)
"""

import dataclasses
from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.bootstrap import _bootstrap_ci_endpoints, interval
from jaxgsa._core.entry import (
    at_least,
    check_scalars,
    in_open_interval,
    one_of,
    prepare,
    require,
)
from jaxgsa._core.invalid import InvalidReport, OnInvalid
from jaxgsa._core.result import CIInfo
from jaxgsa._core.validation import (
    YLayout,
    _prepare_Y,
    _standardize_outputs,
)
from jaxgsa.sobol._bootstrap import _bootstrap_indices
from jaxgsa.sobol._chunking import (
    pad_slice_axis,
    resolve_point_chunk_size,
    resolve_resample_chunk_size,
)
from jaxgsa.sobol._estimators import (
    DEFAULT_ESTIMATOR,
    ESTIMATORS,
    Estimator,
    first_total_kernel,
    requires_second_order_design,
    second_order_kernel,
)
from jaxgsa.sobol._result import SobolResult
from jaxgsa.sobol._sampling import SobolSamples, _saltelli_step

# ---------------------------------------------------------------------------
# Cached JIT kernels
# ---------------------------------------------------------------------------


# lru_cache ensures each (calc_second_order, estimator) variant is JIT-compiled
# exactly once across the process lifetime, avoiding repeated tracing overhead.


@lru_cache(maxsize=None)
def _get_scalar_kernel(calc_second_order: bool, estimator: str):
    """Cache JIT-compiled fused kernels for the scalar (T*K=1) path."""
    if calc_second_order:
        return jax.jit(second_order_kernel(estimator))
    return jax.jit(first_total_kernel(estimator))


@lru_cache(maxsize=None)
def _get_batched_kernel(calc_second_order: bool, estimator: str):
    """Cache JIT-compiled batched kernels for the multi-output path."""
    # vmap over axis 0 maps fused kernels across T*K output slices in parallel
    if calc_second_order:
        return jax.jit(jax.vmap(second_order_kernel(estimator), in_axes=(0, 0, 0, 0)))
    return jax.jit(jax.vmap(first_total_kernel(estimator), in_axes=(0, 0, 0)))


def _separate_output_values(
    Y: Array, D: int, calc_second_order: bool
) -> tuple[Array, Array, Array | None, Array]:
    """Standardize the outputs, then de-interleave them into A, AB, BA, B.

    The standardization is ``(Y - mean) / std`` over the sample axis, one mean
    and one standard deviation per output slice, and it is not optional. The
    Sobol'-Mauntz first-order estimator and every second-order estimator are
    uncentred products, so a non-zero output mean adds an error term
    proportional to that mean: on Ishigami at ``N = 4096`` an offset of 1e4
    turns S1 into ``[6.26, 0.434, 1.71]`` against the analytic
    ``[0.314, 0.442, 0.000]``, in float64 as well as float32. This is
    estimator bias, not rounding, and SALib removes it the same way
    (``SALib/analyze/sobol.py``: ``Y = (Y - Y.mean()) / Y.std()``).

    Scaling by the standard deviation on top of centring moves no index
    (every estimator is a ratio of two quantities of the same degree), but
    doing both makes the arithmetic identical to SALib's rather than merely
    equivalent to it.

    This function is the one seam both the point-estimate path and the
    bootstrap path pass through, which is why the standardization lives here:
    :func:`indices` and :func:`analyze` cannot drift apart, and the bootstrap
    resamples an already-standardized array, so an interval and its centre
    describe the same quantity. It is plain arithmetic with no host read, so
    the ``jit``/``vmap``/``jacrev`` guarantee on :func:`indices` survives.

    Args:
        Y: Expanded outputs, shape ``(N * step, ...)``, with rows in Saltelli
            group order.
        D: Number of input parameters.
        calc_second_order: Whether the layout includes BA blocks.

    Returns:
        Tuple ``(A, AB, BA, B)`` with shapes ``(N, ...)``, ``(N, D, ...)``,
        ``(N, D, ...)`` and ``(N, ...)``. This is the order every estimator
        kernel in :mod:`jaxgsa.sobol._estimators` consumes, so a caller that
        unpacks this tuple straight into a kernel call cannot swap two
        blocks by accident. BA is None when second order is off.

    Raises:
        ValueError: If ``Y``'s row count is not a multiple of the Saltelli
            group size ``step``, which a hand-built design or a misaligned
            ``Y`` can produce. Designs built by :func:`jaxgsa.sobol.sample`
            always satisfy it.
    """
    # Per output slice: axis 0 is the expanded sample axis, every trailing
    # axis keeps its own mean and standard deviation.
    Y, _, _, _ = _standardize_outputs(Y)

    step = 2 * D + 2 if calc_second_order else D + 2
    n_rows = Y.shape[0]
    if n_rows % step != 0:
        raise ValueError(
            f"Y has n_rows={n_rows}, which is not a multiple of the Saltelli "
            f"group size step={step}; the design cannot be split into whole "
            "groups. This design was not built by jaxgsa.sobol.sample, or Y "
            "does not match its expanded layout."
        )
    base_n = n_rows // step
    trailing = Y.shape[1:]

    # Reshape-based extraction: one reshape converts the flat output vector
    # into (base_n, step, ...) groups, then simple slicing pulls A/B/AB/BA.
    # This is much faster than D separate stride-slices + a stack.
    #
    # Saltelli row layout within each group of `step` rows:
    #   [0]=A, [1..D]=AB_j (A with col j from B), [D+1..2D]=BA_j (B with
    #   col j from A), [2D+1]=B.  For first-order only, BA is omitted and
    #   step = D+2 with B at position [D+1].
    grouped = Y.reshape(base_n, step, *trailing)

    A = grouped[:, 0]  # (N, ...)
    B = grouped[:, -1]  # (N, ...)
    AB = grouped[:, 1 : D + 1]  # (N, D, ...)

    BA = None
    if calc_second_order:
        BA = grouped[:, D + 1 : 2 * D + 1]  # (N, D, ...)

    return A, AB, BA, B


def _symmetrize_s2(S2: Array) -> Array:
    """Average the upper and lower triangles of a raw S2 matrix.

    The kernel computes ``S2_jk`` and ``S2_kj`` independently, from the same
    formula with ``j`` and ``k`` swapped. They estimate the same second-order
    index, and the two triangles differ only by Monte Carlo noise, not
    floating-point drift: on Ishigami at ``base_n = 256`` the mean absolute
    gap is 0.026. Averaging the two is a deliberate departure from SALib,
    which reports the upper triangle alone. The cost is that S2 matches
    SALib only up to that noise, not bit for bit.

    Averaging usually lowers the error, but it is not a guarantee. Measured
    on Ishigami over 200 seeds, against the analytic S2, over all three
    parameter pairs: RMSE falls 25 per cent at ``base_n = 64`` and 13 per
    cent at ``base_n = 1024``, and rises 9 per cent at ``base_n = 256``.
    The rise comes from one pair whose lower-triangle estimate happened to
    be three times noisier than its upper one at that design size, which
    the scrambled Sobol' sequence can do. Averaging two estimates of
    unequal spread can beat neither of them. It is still the better default:
    the caller cannot know which triangle is the lucky one, and picking the
    upper one always, as SALib does, is a coin flip.

    Args:
        S2: The kernel's raw ``(D, D)``-trailing output, both triangles
            populated, diagonal undefined.

    Returns:
        The symmetric average, same shape. The diagonal is meaningless here
        too; :func:`_normalize_s2_matrix` masks it.
    """
    return 0.5 * (S2 + jnp.swapaxes(S2, -1, -2))


def _mask_s2_diagonal(S2: Array) -> Array:
    """Set the diagonal of an already-symmetric S2 matrix to NaN.

    ``S2_jj`` (a parameter's interaction with itself) is undefined in Sobol
    ANOVA, so it is never a value to read; NaN says so.
    """
    D = S2.shape[-1]
    return jnp.where(jnp.eye(D, dtype=bool), jnp.nan, S2)


def _normalize_s2_matrix(S2: Array) -> Array:
    """Symmetrise the S2 matrix (see :func:`_symmetrize_s2`) and NaN the diagonal."""
    return _mask_s2_diagonal(_symmetrize_s2(S2))


def _estimator_checks(estimator: str, calc_second_order: bool) -> tuple:
    """Build the scalar checks that settle ``estimator`` before any array work.

    Both entry points run these, so a bad name or an estimator the design
    cannot feed is refused in the same words either way.

    Args:
        estimator: The requested estimator name, unvalidated.
        calc_second_order: Whether the design carries the BA blocks.

    Returns:
        The checks, in the order they should be reported: the name first,
        then whether the design can feed it.
    """
    return (
        one_of("estimator", estimator, ESTIMATORS),
        require(
            not (requires_second_order_design(estimator) and not calc_second_order),
            f"estimator={estimator!r} reads the BA blocks of the Saltelli design, "
            "so it needs a design drawn with calc_second_order=True. Either "
            "re-draw the design, or pick an estimator that runs on the N(D+2) "
            "layout.",
        ),
    )


def _flatten_slices(
    A: Array, AB: Array, BA: Array | None, B: Array
) -> tuple[Array, Array, Array | None, Array]:
    """Fold the (T, K) output axes of the separated matrices into one slice axis.

    Every estimator kernel reads one output slice at a time, so the (T, K)
    grid becomes a single leading batch axis of ``S = T * K`` slices. Both the
    point-estimate path and the bootstrap path map over that axis, so they
    flatten the same way and index the same slices in the same order.

    Args and return both use ``(A, AB, BA, B)``, the order every estimator
    kernel consumes, so unpacking straight into a kernel call cannot swap two
    blocks by accident.

    Args:
        A: Outputs from sample matrix A, shape ``(N, T, K)``.
        AB: Outputs from the AB cross-matrices, shape ``(N, D, T, K)``.
        BA: Outputs from the BA cross-matrices, shape ``(N, D, T, K)``, or
            ``None`` when second order is off.
        B: Outputs from sample matrix B, shape ``(N, T, K)``.

    Returns:
        ``(A_flat, AB_flat, BA_flat, B_flat)`` with shapes ``(S, N)``,
        ``(S, N, D)``, ``(S, N, D)`` and ``(S, N)``. ``BA_flat`` is ``None``
        when ``BA`` is.
    """
    base_n, T, K = A.shape
    D = AB.shape[1]
    total = T * K
    # (N,T,K) -> transpose -> (T,K,N) -> reshape -> (T*K, N)
    # AB is (N,D,T,K) so its transpose puts (T,K) first then (N,D).
    A_flat = A.transpose(1, 2, 0).reshape(total, base_n)
    B_flat = B.transpose(1, 2, 0).reshape(total, base_n)
    AB_flat = AB.transpose(2, 3, 0, 1).reshape(total, base_n, D)
    BA_flat = None if BA is None else BA.transpose(2, 3, 0, 1).reshape(total, base_n, D)
    return A_flat, AB_flat, BA_flat, B_flat


def _point_indices_3d(
    A_flat: Array,
    AB_flat: Array,
    BA_flat: Array | None,
    B_flat: Array,
    *,
    T: int,
    K: int,
    D: int,
    is_scalar: bool,
    slice_chunk_size: int | None,
    estimator: str,
) -> tuple[Array, Array, Array | None]:
    """Run the estimator over every output slice, in the promoted 3-D layout.

    This is the one place the point-estimate kernels are called. Both
    :func:`_indices_from_expanded` and the bootstrap path go through it, so
    an index and the centre of its confidence interval come from the same
    arithmetic by construction.

    The shape is the library's usual one: an atomic kernel for a single
    slice, ``vmap`` over a chunk of slices, a host loop over the chunks. A
    scalar output skips the ``vmap`` and calls the fused kernel directly,
    which avoids the dispatch and traces a simpler graph.

    Args:
        A_flat: Outputs from sample matrix A, shape ``(S, N)``.
        AB_flat: Outputs from the AB cross-matrices, shape ``(S, N, D)``.
        BA_flat: Outputs from the BA cross-matrices, shape ``(S, N, D)``, or
            ``None`` when second order is off.
        B_flat: Outputs from sample matrix B, shape ``(S, N)``.
        T: Number of time steps in the promoted layout.
        K: Number of output variables in the promoted layout.
        D: Number of input parameters.
        is_scalar: Whether the caller passed a rank-1 ``Y``, which enables
            the direct fused-kernel path.
        slice_chunk_size: Number of output slices per vmap batch, or ``None``
            to derive one from the active memory budget.
        estimator: Which named estimator pair to use.

    Returns:
        ``(S1, ST, S2)`` shaped ``(T, K, D)``, ``(T, K, D)`` and
        ``(T, K, D, D)``, with ``S2`` ``None`` when second order is off. The
        S2 matrix is the kernel's raw output, not yet symmetrised: see
        :func:`_symmetrize_s2`. The bootstrap path symmetrises the point
        estimate and every draw the same way before either feeds a
        confidence interval, so an interval always describes the symmetrised
        quantity the point estimate reports. The inserted T and K axes are
        still in place; the caller squeezes them.

    Raises:
        ValueError: If ``slice_chunk_size`` is below 1.
    """
    calc_second_order = BA_flat is not None
    total = T * K
    # Sizes the automatic width, and refuses an explicit width below 1. Both
    # happen before the scalar shortcut so every path validates the argument.
    cs = resolve_point_chunk_size(
        slice_chunk_size,
        total,
        A_flat.shape[1],
        D,
        calc_second_order,
        A_flat.dtype.itemsize,
    )

    if is_scalar:
        # Scalar path (T*K=1): call the fused kernel directly on 1-D arrays.
        # This avoids vmap dispatch overhead and produces a simpler XLA graph.
        a, b, ab = A_flat[0], B_flat[0], AB_flat[0]
        if calc_second_order:
            assert BA_flat is not None
            kernel = _get_scalar_kernel(True, estimator)
            s1, st, s2_raw = kernel(a, ab, BA_flat[0], b)
            S2_out = s2_raw.reshape(1, 1, D, D)
        else:
            kernel = _get_scalar_kernel(False, estimator)
            s1, st = kernel(a, ab, b)
            S2_out = None
        return s1.reshape(1, 1, D), st.reshape(1, 1, D), S2_out

    # Every chunk is padded back to `cs` slices, so the kernel traces once
    # instead of once more for a ragged tail. The padded lanes are separate
    # vmap lanes and are dropped before anything reads them.
    if calc_second_order:
        assert BA_flat is not None
        batched = _get_batched_kernel(True, estimator)
        s1_parts, st_parts, s2_parts = [], [], []
        for start in range(0, total, cs):
            end = min(start + cs, total)
            actual = end - start
            s1, st, s2 = batched(
                pad_slice_axis(A_flat[start:end], cs),
                pad_slice_axis(AB_flat[start:end], cs),
                pad_slice_axis(BA_flat[start:end], cs),
                pad_slice_axis(B_flat[start:end], cs),
            )
            s1_parts.append(s1[:actual])
            st_parts.append(st[:actual])
            s2_parts.append(s2[:actual])

        S1_out = jnp.concatenate(s1_parts).reshape(T, K, D)
        ST_out = jnp.concatenate(st_parts).reshape(T, K, D)
        S2_out = jnp.concatenate(s2_parts).reshape(T, K, D, D)
    else:
        batched = _get_batched_kernel(False, estimator)
        s1_parts, st_parts = [], []
        for start in range(0, total, cs):
            end = min(start + cs, total)
            actual = end - start
            s1, st = batched(
                pad_slice_axis(A_flat[start:end], cs),
                pad_slice_axis(AB_flat[start:end], cs),
                pad_slice_axis(B_flat[start:end], cs),
            )
            s1_parts.append(s1[:actual])
            st_parts.append(st[:actual])

        S1_out = jnp.concatenate(s1_parts).reshape(T, K, D)
        ST_out = jnp.concatenate(st_parts).reshape(T, K, D)
        S2_out = None

    return S1_out, ST_out, S2_out


def _indices_from_expanded(
    Y: Array, D: int, calc_second_order: bool, slice_chunk_size: int | None, estimator: str
) -> tuple[Array, Array, Array | None]:
    """Compute Sobol indices from expanded-layout outputs, picking the faster kernel.

    This is the single implementation of the estimator maths. Both the public
    :func:`indices` and the ``analyze`` no-bootstrap path go through it, so
    the two can never drift apart.

    It is deliberately free of policy: no diagnostics, no host-side reads of
    array values, no result object. Every branch here is on a shape or a
    Python flag, all of which stay concrete under tracing, so the function is
    ``jit``-, ``vmap``- and ``jacrev``-able.

    For a scalar output (T*K=1) it calls a direct fused kernel that computes
    the variance once. For a multi-output analysis it vmaps the fused kernel
    over a chunk of the T*K slices at a time.

    Args:
        Y: Model outputs in the expanded Saltelli layout, shape
            ``(base_n * step, ...)``.
        D: Number of input parameters.
        calc_second_order: Whether the layout includes the BA blocks.
        slice_chunk_size: Number of (T, K) output slices per vmap batch, or
            ``None`` to derive one from the active memory budget.
        estimator: Which named estimator pair to use. See
            :mod:`jaxgsa.sobol._estimators`.

    Returns:
        ``(S1, ST, S2)``, with ``S2`` ``None`` when second order is off. The
        output axes match ``Y``'s trailing axes, as ``analyze`` documents.

    Raises:
        ValueError: If ``slice_chunk_size`` is below 1.
    """
    # Promote to uniform 3-D shape (N, T, K) so downstream code is shape-agnostic.
    Y, layout = _prepare_Y(Y)
    _, T, K = Y.shape

    A, AB, BA, B = _separate_output_values(Y, D, calc_second_order)
    A_flat, AB_flat, BA_flat, B_flat = _flatten_slices(A, AB, BA, B)

    S1_out, ST_out, S2_out = _point_indices_3d(
        A_flat,
        AB_flat,
        BA_flat,
        B_flat,
        T=T,
        K=K,
        D=D,
        is_scalar=layout is YLayout.SCALAR,
        slice_chunk_size=slice_chunk_size,
        estimator=estimator,
    )

    S1_out = layout.squeeze(S1_out)
    ST_out = layout.squeeze(ST_out)
    if S2_out is not None:
        S2_out = layout.squeeze(_normalize_s2_matrix(S2_out), n_trailing=2)
    return S1_out, ST_out, S2_out


def _analyze_no_bootstrap(
    sampling_result: SobolSamples,
    Y: Array,
    *,
    slice_chunk_size: int | None,
    estimator: str,
    invalid: InvalidReport,
) -> SobolResult:
    """Wrap the estimator core in a ``SobolResult``, without a bootstrap."""
    S1_out, ST_out, S2_out = _indices_from_expanded(
        Y,
        sampling_result.n_params,
        sampling_result.calc_second_order,
        slice_chunk_size,
        estimator,
    )
    return SobolResult(
        S1=S1_out,
        ST=ST_out,
        S2=S2_out,
        problem=sampling_result.problem,
        invalid=invalid,
        estimator=estimator,
    )


def indices(
    sampling_result: SobolSamples,
    Y: Array,
    *,
    estimator: Estimator = DEFAULT_ESTIMATOR,
    slice_chunk_size: int | None = None,
) -> tuple[Array, ...]:
    """Compute Sobol indices as plain arrays, with no diagnostics.

    This is the transformable core of :func:`analyze`. It runs the same
    estimators on the same data and returns the same numbers, but it does
    nothing else: no non-finite check, no zero-variance warning, no
    :class:`jaxgsa.sobol.SobolResult`, and no read of any array value on the
    host. So it composes with ``jax.jit``, ``jax.vmap``, ``jax.grad`` and
    ``jax.jacrev``, which :func:`analyze` cannot, because a policy decision
    needs a concrete value and a tracer has none.

    Pair it with :meth:`jaxgsa.sobol.SobolSamples.transform` to differentiate
    an index with respect to the input distribution parameters::

        def s1(theta):
            Y = model(sampling_result.transform(theta))
            return jaxgsa.sobol.indices(sampling_result, Y)[0]

        dS1_dtheta = jax.jacrev(s1)(theta)

    Use :func:`analyze` for ordinary analysis. Nothing here checks the outputs,
    so a single NaN silently turns every index into NaN.

    Like :func:`analyze`, this standardizes every output slice to mean 0 and
    unit standard deviation before the estimators run. That is arithmetic, not
    policy: the first- and second-order estimators are uncentred products, so
    the standardization removes a bias term proportional to the output mean.
    It reduces over the sample axis only, so it stays traceable.

    Tier T4 (behavioural contract): the returned arrays must equal the
    corresponding fields of ``analyze``'s result on clean outputs, and the
    function must survive ``jit``, ``vmap`` and ``jit(jacrev(...))``. Checked
    in ``tests/test_sobol_gradients.py``.

    Args:
        sampling_result: The design from :func:`jaxgsa.sobol.sample`, used
            only for its expansion map and its Saltelli layout flags.
        Y: Model outputs evaluated at each unique row of
            ``sampling_result.samples``, in the same row order. Shapes are
            those :func:`analyze` accepts: ``(n_runs,)``, ``(n_runs, K)`` or
            ``(n_runs, T, K)``.
        estimator: Which estimator pair to use, as in :func:`analyze`. Every
            one of them is plain arithmetic on the output vectors, so the
            choice does not affect what ``jit``, ``vmap`` or ``jacrev`` can
            do with this function.
        slice_chunk_size: Number of (T, K) output slices per vmap batch, or
            ``None`` (the default) to derive one from
            :func:`jaxgsa.config.set_memory_budget`. Lower it if you hit
            device out-of-memory errors.

    Returns:
        ``(S1, ST)``, or ``(S1, ST, S2)`` when the design was drawn with
        ``calc_second_order=True``. The shapes are those ``analyze`` reports.

    Raises:
        ValueError: If ``estimator`` is not a known name; if it needs the BA
            blocks and the design has none; if ``Y``'s first axis does not
            match ``sampling_result.n_runs``; or if ``slice_chunk_size`` is
            below 1.
    """
    check_scalars(_estimator_checks(estimator, sampling_result.calc_second_order))
    S1, ST, S2 = _indices_from_expanded(
        sampling_result.expand_outputs(Y),
        sampling_result.n_params,
        sampling_result.calc_second_order,
        slice_chunk_size,
        estimator,
    )
    if S2 is None:
        return S1, ST
    return S1, ST, S2


def _analyze_bootstrap(
    sampling_result: SobolSamples,
    Y: Array,
    *,
    estimator: str,
    n_bootstrap: int,
    conf_level: float,
    ci_method: Literal["quantile", "gaussian"],
    key: Array,
    slice_chunk_size: int | None,
    invalid: InvalidReport,
    keep_replicates: bool,
) -> SobolResult:
    """Compute Sobol indices with bootstrap confidence intervals.

    The point estimates come from :func:`_point_indices_3d`, the same chunked
    ``vmap`` the no-bootstrap path uses, so an interval is always centred on
    the number the plain analysis reports.

    The draws follow the same shape one level deeper. The atomic unit is one
    estimator call on one slice and one resample; the resamples of a slice are
    vmapped in a bounded batch, that pair is vmapped over a chunk of slices,
    and both loops run on the host. The pre-generated random index matrix is
    sliced contiguously, so changing either width does not change any draw.
    When ``keep_replicates=False``, each output chunk is reduced to CI
    endpoints before the next chunk is processed. Keeping replicates opts in
    to assembling the full public draw arrays.
    """
    Y, layout = _prepare_Y(Y)
    D = sampling_result.n_params
    calc_second_order = sampling_result.calc_second_order

    _, T, K = Y.shape
    A, AB, BA, B = _separate_output_values(Y, D, calc_second_order)
    base_n = A.shape[0]
    A_flat, AB_flat, BA_flat, B_flat = _flatten_slices(A, AB, BA, B)
    total = T * K

    # Pre-generate all R bootstrap index sets (sampling with replacement).
    # Shared across (T, K) slices so every output sees the same resamples.
    # Named resample_idx, not indices: `indices` is this module's public
    # transformable entry point.
    resample_idx = jax.random.randint(key, shape=(n_bootstrap, base_n), minval=0, maxval=base_n)

    S1_out, ST_out, S2_raw = _point_indices_3d(
        A_flat,
        AB_flat,
        BA_flat,
        B_flat,
        T=T,
        K=K,
        D=D,
        is_scalar=layout is YLayout.SCALAR,
        slice_chunk_size=slice_chunk_size,
        estimator=estimator,
    )

    cs = resolve_point_chunk_size(
        slice_chunk_size,
        total,
        base_n,
        D,
        calc_second_order,
        A_flat.dtype.itemsize,
        n_bootstrap=n_bootstrap,
    )

    # A slice chunk can be large enough for the point path but still leave an
    # unbounded R axis in the bootstrap gather. Resolve the second width from
    # the same transient working-set model. The input/resample arrays and the
    # final result remain resident, so this bounds the estimator transient.
    rs = resolve_resample_chunk_size(
        n_bootstrap,
        base_n,
        D,
        calc_second_order,
        A_flat.dtype.itemsize,
        cs,
    )

    S1_flat = S1_out.reshape(total, D)
    ST_flat = ST_out.reshape(total, D)
    S2_sym = None
    if calc_second_order:
        assert S2_raw is not None
        S2_sym = _symmetrize_s2(S2_raw)
        S2_flat = S2_sym.reshape(total, D, D)
    else:
        S2_flat = None

    s1_draw_parts, st_draw_parts, s2_draw_parts = [], [], []
    s1_conf_parts, st_conf_parts, s2_conf_parts = [], [], []
    for start in range(0, total, cs):
        end = min(start + cs, total)
        actual = end - start
        A_chunk = pad_slice_axis(A_flat[start:end], cs)
        AB_chunk = pad_slice_axis(AB_flat[start:end], cs)
        B_chunk = pad_slice_axis(B_flat[start:end], cs)
        BA_chunk = None if BA_flat is None else pad_slice_axis(BA_flat[start:end], cs)
        s1_chunk, st_chunk, s2_chunk = _bootstrap_indices(
            resample_idx,
            A_chunk,
            AB_chunk,
            BA_chunk,
            B_chunk,
            cs,
            estimator,
            resample_chunk_size=rs,
        )
        s1_draws = jnp.moveaxis(s1_chunk[:actual], 1, 0)
        st_draws = jnp.moveaxis(st_chunk[:actual], 1, 0)
        if keep_replicates:
            s1_draw_parts.append(s1_draws)
            st_draw_parts.append(st_draws)

        if not keep_replicates:
            s1_lo, s1_hi = _bootstrap_ci_endpoints(
                S1_flat[start:end],
                s1_draws,
                conf_level=conf_level,
                ci_method=ci_method,
            )
            st_lo, st_hi = _bootstrap_ci_endpoints(
                ST_flat[start:end],
                st_draws,
                conf_level=conf_level,
                ci_method=ci_method,
            )
            s1_conf_parts.append(jnp.stack([s1_lo, s1_hi]))
            st_conf_parts.append(jnp.stack([st_lo, st_hi]))

        if calc_second_order:
            assert s2_chunk is not None and S2_flat is not None
            s2_draws = _symmetrize_s2(jnp.moveaxis(s2_chunk[:actual], 1, 0))
            if keep_replicates:
                s2_draw_parts.append(s2_draws)
            if not keep_replicates:
                s2_lo, s2_hi = _bootstrap_ci_endpoints(
                    S2_flat[start:end],
                    s2_draws,
                    conf_level=conf_level,
                    ci_method=ci_method,
                )
                s2_conf_parts.append(jnp.stack([s2_lo, s2_hi]))

    if keep_replicates:
        S1_boot = jnp.concatenate(s1_draw_parts, axis=1).reshape(n_bootstrap, T, K, D)
        ST_boot = jnp.concatenate(st_draw_parts, axis=1).reshape(n_bootstrap, T, K, D)
        points = {"S1": S1_out, "ST": ST_out}
        draws = {"S1": S1_boot, "ST": ST_boot}
        if calc_second_order:
            assert S2_sym is not None
            S2_boot = jnp.concatenate(s2_draw_parts, axis=1).reshape(n_bootstrap, T, K, D, D)
            points["S2"] = S2_sym
            draws["S2"] = S2_boot
        conf, ci = interval(
            points,
            draws,
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            keep_replicates=True,
        )
    else:
        conf = {
            "S1": jnp.concatenate(s1_conf_parts, axis=1).reshape(2, T, K, D),
            "ST": jnp.concatenate(st_conf_parts, axis=1).reshape(2, T, K, D),
        }
        if calc_second_order:
            conf["S2"] = jnp.concatenate(s2_conf_parts, axis=1).reshape(2, T, K, D, D)
        ci = CIInfo(
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            replicates=None,
        )
    S1_conf = conf["S1"]
    ST_conf = conf["ST"]

    S2_out = None
    S2_conf = None
    if calc_second_order:
        assert S2_sym is not None
        S2_out = _mask_s2_diagonal(S2_sym)
        S2_conf = _mask_s2_diagonal(conf["S2"])
        if ci.replicates is not None:
            # dataclasses are frozen, so build the replacement dict first and
            # swap it in below with dataclasses.replace. Every other key
            # (S1, ST) is already the reported quantity; only S2 needs its
            # diagonal masked after the fact.
            replicates = dict(ci.replicates)
            replicates["S2"] = _mask_s2_diagonal(replicates["S2"])
            ci = dataclasses.replace(ci, replicates=replicates)

    S1_out = layout.squeeze(S1_out)
    ST_out = layout.squeeze(ST_out)
    S1_conf = layout.squeeze(S1_conf)
    ST_conf = layout.squeeze(ST_conf)
    if S2_out is not None:
        S2_out = layout.squeeze(S2_out, n_trailing=2)
    if S2_conf is not None:
        S2_conf = layout.squeeze(S2_conf, n_trailing=2)
    if ci.replicates is not None:
        # The draws already lead with the resample axis, which the squeeze
        # leaves alone because it addresses T and K from the end.
        squeezed = {
            "S1": layout.squeeze(ci.replicates["S1"], n_trailing=1),
            "ST": layout.squeeze(ci.replicates["ST"], n_trailing=1),
        }
        if "S2" in ci.replicates:
            squeezed["S2"] = layout.squeeze(ci.replicates["S2"], n_trailing=2)
        ci = dataclasses.replace(ci, replicates=squeezed)

    return SobolResult(
        S1=S1_out,
        ST=ST_out,
        S2=S2_out,
        problem=sampling_result.problem,
        invalid=invalid,
        estimator=estimator,
        S1_conf=S1_conf,
        ST_conf=ST_conf,
        S2_conf=S2_conf,
        ci=ci,
    )


def analyze(
    sampling_result: SobolSamples,
    Y: Array,
    *,
    estimator: Estimator = DEFAULT_ESTIMATOR,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    slice_chunk_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> SobolResult:
    """Compute Sobol sensitivity indices from model outputs using JAX.

    This is the main entry point of the package. Sobol indices apportion the
    variance of a model output among its input parameters. S1 (first-order) is
    the fraction of output variance explained by each parameter alone. ST
    (total-order) also includes all of that parameter's interactions with the
    other parameters. S2 (second-order) isolates pairwise interactions.

    Every output slice is standardized to mean 0 and unit standard deviation
    over the sample axis before the estimators run, exactly as SALib does. The
    first-order and second-order estimators are uncentred products, so a
    non-zero output mean would otherwise add an error term proportional to
    that mean. The indices are ratios, so the standardization itself moves
    nothing else.

    The function takes the model outputs Y evaluated at the unique rows that
    ``jaxgsa.sobol.sample()`` returned. It rebuilds the expanded Saltelli
    ordering internally and checks it for non-finite values under the
    ``on_invalid`` policy. It then dispatches on ``n_bootstrap``: to the
    fast no-bootstrap path, or to the bootstrap confidence-interval path.

    Those checks read array values on the host, so ``analyze`` cannot be
    traced by ``jax.jit`` or differentiated by ``jax.grad``. Use
    :func:`jaxgsa.sobol.indices` when you need that; it runs the same
    estimators and skips every check.

    Args:
        sampling_result: Result from ``jaxgsa.sobol.sample()`` with the unique
            sample matrix plus expansion metadata.
        Y: Model outputs evaluated at each unique row of
            ``sampling_result.samples``, in the same row order, where
            ``n_runs`` is the unique row count. Accepted shapes:
                ``(n_runs,)``: scalar output, single time step
                ``(n_runs, K)``: K outputs, single time step
                ``(n_runs, T, K)``: K outputs over T time steps
            Indices are computed independently for every (t, k) output slice.
        estimator: Which pair of estimator formulas to use. Every one of
            them converges to the same indices; they differ in how much
            sampling noise they carry at a small ``N``, and in whether they
            can return a value outside ``[0, 1]``.

                ``"saltelli-jansen"`` (default): Sobol'-Mauntz first order,
                Jansen (1999) total order. It is the default because Jansen's
                total-order estimator is a mean of squares, so ``ST`` can
                never go negative, and because it is the same pairing SALib
                uses by default, so the two libraries agree out of the box.
                ``"jansen"``: Jansen (1999) for both orders. ``ST`` cannot go
                negative; ``S1`` can, and is unbounded below (see
                :mod:`jaxgsa.sobol._estimators`).
                ``"janon-monod"``: one self-consistent normaliser shared
                between numerator and denominator. Its asymptotic variance
                is never worse than the classical one.
                ``"martinez"``: the same pairings read as empirical
                correlations.
                ``"mauntz-kucherenko"``: Sobol' et al. (2007) for both
                orders. Same first order as the default.
                ``"azzini-rosati"``: Azzini, Mara and Rosati (2021). It is
                the only scheme that holds ``S1 <= ST`` on every sample, and
                the only one that reads the BA blocks, so it needs a design
                drawn with ``calc_second_order=True``.

            See :mod:`jaxgsa.sobol._estimators` for the formulas and the
            references, and the methods guide for the measured errors.
        n_bootstrap: R, the number of bootstrap resamples used to estimate
            confidence intervals. Set to 0 (default) to skip the bootstrap
            entirely; a few hundred resamples is typically enough for stable
            intervals.
        conf_level: Two-sided confidence level for bootstrap CIs
            (default 0.95).
        ci_method: Bootstrap CI endpoint method. ``"quantile"`` returns
            percentile lower/upper endpoints from the bootstrap draws.
            ``"gaussian"`` returns symmetric gaussian lower/upper endpoints
            from the bootstrap standard deviation around the point estimate.
            Both methods still return lower/upper bounds, not half-widths.
        key: JAX PRNG key for bootstrap randomness. Required when
            ``n_bootstrap > 0``.
        slice_chunk_size: Memory/speed trade-off for batched computation.
            It is the number of (T, K) output slices per vmap batch on both
            paths. On the bootstrap path the resample axis is batched too;
            its width is derived from the transient memory budget after this
            slice width is chosen. ``None`` (the default) derives the slice
            width from that budget, on both paths: a slice costs about
            ``2 * N * (D + 2)`` elements first-order-only, and
            ``2 * N * (2D + 2)`` with second order. Both orders are now
            linear in ``D``: the second-order cross-moment is written as a
            matrix product (``BA.T @ AB``), so the ``(N, D, D)`` outer
            product earlier versions materialised no longer exists. The
            bootstrap transient is sized for the product of slice width and
            resample width. Resident inputs, the pre-generated index matrix,
            and final results are outside this estimate. Give an integer to
            cap the slice width yourself if you hit device out-of-memory
            errors; the resample width remains budget-bounded.

            It changes no index beyond floating-point noise. The estimator
            sums over the sample axis, and XLA schedules that reduction
            differently for a different batch width, so two chunk sizes can
            disagree in the last bits of a float32 result.
        on_invalid: What to do about non-finite model outputs. The unit here
            is one Saltelli group, so a single bad value removes the whole
            group of ``D + 2`` (or ``2D + 2``) rows. ``"raise"`` (the default)
            refuses the sample, ``"propagate"`` lets the value reach the
            indices, and ``"drop"`` analyzes the surviving groups. See
            :mod:`jaxgsa._core.invalid`.
        keep_replicates: Keep the per-resample index draws on
            ``result.ci.replicates``, keyed ``"S1"``, ``"ST"`` and (with a
            second-order design) ``"S2"``, each leading with the resample
            axis. The stored ``S2`` draws follow the reported convention:
            symmetrised, with a NaN diagonal, exactly as the ``S2`` point
            estimate is. Off by default, because the draws are ``n_bootstrap``
            copies of every index array.
        verbose: If ``True`` (default), print a short summary to stdout: the
            problem and the data, the wall-clock timing, and the top
            parameters by ``ST``. Pass ``False`` for a silent run.

    Returns:
        SobolResult holding:
            S1: first-order indices, shape ``(D,)`` / ``(K, D)`` /
                ``(T, K, D)`` for Y of shape (n,) / (n, K) / (n, T, K)
                respectively
            ST: total-order indices, same shape as S1
            S2: second-order indices with shape ``(..., D, D)``, or None when
                the design was drawn with ``calc_second_order=False``
            S1_conf, ST_conf, S2_conf: ``(2, ...)`` [lower, upper] CI bounds,
                or None when ``n_bootstrap == 0``
            invalid: What the non-finite check found, and what it did

    Raises:
        ValueError: If ``estimator`` is not a known name, or needs the BA
            blocks and the design has none; if ``on_invalid`` is not one of
            the three policies; if
            ``ci_method`` is not ``"quantile"`` or ``"gaussian"``; if
            ``n_bootstrap`` is negative; if ``slice_chunk_size`` is below 1;
            if ``conf_level`` is not in ``(0, 1)``; if the sample holds a
            non-finite value under ``on_invalid="raise"``; or if fewer than 2
            Saltelli groups survive a drop.

    Warns:
        JaxgsaWarning: If an output slice has zero variance, which makes its
            indices NaN.
    """
    from jaxgsa.sobol import SPEC

    D = sampling_result.n_params
    # step = rows per Saltelli group: D+2 (first-order only) or 2D+2 (with S2)
    step = _saltelli_step(D, sampling_result.calc_second_order)
    # A Saltelli group [A_i, AB_{i,0}, …, B_i] is one indivisible sampling
    # unit, so the check runs at group granularity: a single bad row condemns
    # the whole block of `step` rows, and a partial group would corrupt the
    # A/B/AB/BA split. The group count comes from the design, not from Y, so
    # it is known before Y is looked at.
    base_n = sampling_result.n_expanded // step

    ctx = prepare(
        SPEC,
        sampling_result.problem,
        Y,
        on_invalid=on_invalid,
        checks=(
            *_estimator_checks(estimator, sampling_result.calc_second_order),
            one_of("ci_method", ci_method, ("quantile", "gaussian")),
            at_least("n_bootstrap", n_bootstrap, 0),
            at_least("slice_chunk_size", slice_chunk_size, 1),
            in_open_interval("conf_level", conf_level, 0.0, 1.0),
            # Checked here, with the other cheap argument checks, so a
            # missing key is reported before the point estimate is paid for.
            require(
                not (n_bootstrap > 0 and key is None),
                "key is required when n_bootstrap > 0",
            ),
        ),
        n_expected=int(sampling_result.samples.shape[0]),
        expand=sampling_result.expand_outputs,
        n_units=base_n,
        unit_of_row=np.repeat(np.arange(base_n), step),
        # Y is checked expanded, but the caller passed one output per unique
        # run. Report the rows they hold, not the expanded ones.
        row_labels=sampling_result.expanded_to_unique,
        min_kept=2,
    )
    # The estimator reads the expanded layout, and its scalar fast path
    # branches on Y's own rank, so this is ctx.Y and not ctx.Y3.
    Y = ctx.Y
    invalid = ctx.invalid
    if not ctx.keep.all():
        trailing = Y.shape[1:]
        # Boolean indexing gives a variable-length result, which JAX cannot
        # trace, so the compaction round-trips through NumPy.
        grouped = np.asarray(Y).reshape(base_n, step, *trailing)[ctx.keep]
        Y = jnp.asarray(grouped.reshape(-1, *trailing))

    # The outputs are standardized inside _separate_output_values, which both
    # paths below reach, so there is nothing to do to Y here.
    t0 = _verbose.tic()
    if n_bootstrap > 0:
        assert key is not None  # checked above, in checks=
        result = _analyze_bootstrap(
            sampling_result,
            Y,
            estimator=estimator,
            n_bootstrap=n_bootstrap,
            conf_level=conf_level,
            ci_method=ci_method,
            key=key,
            slice_chunk_size=slice_chunk_size,
            invalid=invalid,
            keep_replicates=keep_replicates,
        )
    else:
        result = _analyze_no_bootstrap(
            sampling_result,
            Y,
            slice_chunk_size=slice_chunk_size,
            estimator=estimator,
            invalid=invalid,
        )

    if verbose:
        elapsed = _verbose.stop(t0, result.S1, result.ST)
        _, T, K = ctx.Y3.shape
        # Same resolver the point kernel ran, on the same shapes: cheap
        # arithmetic, reported rather than re-derived by hand.
        cs = resolve_point_chunk_size(
            slice_chunk_size,
            T * K,
            int(Y.shape[0]) // step,
            D,
            sampling_result.calc_second_order,
            Y.dtype.itemsize,
        )
        origin = "user-set" if slice_chunk_size is not None else "resolved from the memory budget"
        notes = [f"slice_chunk_size: {cs} ({origin})", f"estimator: {estimator}"]
        if n_bootstrap > 0:
            # The bootstrap path resolves both axes from the transient model.
            # ``cs`` is the slice width; ``boot_rs`` is the resample width
            # after accounting for that chosen slice width.
            boot_rs = resolve_resample_chunk_size(
                n_bootstrap,
                int(Y.shape[0]) // step,
                D,
                sampling_result.calc_second_order,
                Y.dtype.itemsize,
                cs,
            )
            notes.insert(1, f"bootstrap resample_chunk_size: {boot_rs} ({origin})")
        _verbose.analysis_summary(
            method="jaxgsa.sobol.analyze",
            problem=sampling_result.problem,
            n_runs=int(Y.shape[0]),
            T=T,
            K=K,
            invalid=invalid,
            timings=[("estimators (includes compile on the first call)", elapsed)],
            notes=notes,
            index_name="ST",
            values=result.ST,
            conf=result.ST_conf,
        )
    return result
